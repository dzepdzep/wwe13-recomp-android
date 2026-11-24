// wwe2k14 patch (Session 51, STATE LOG): the state a frame builder needs, from the game's own D3D library calls
// (SESSION51_PLAN.md step 1, docs\native\STATE_LOG.md in the game repository).
//
// gpu_native_half2 3: each library call the game's hooks bracket (the draw writers, BeginVertices, the clear and the
// resolve core) has its own packet bytes copied into a log - the dirty state the call flushed, as the library encoded
// it (type-0 register runs, SET_CONSTANT, LOAD_ALU_CONSTANT literals, IM_LOAD / the patched inline vertex shader, the
// draw packet). The library writes these packets whether or not it talks to the GPU (listing 0x82CB19E0: no read of
// its no-GPU byte), so the log stays valid in its no-GPU mode. A call made inside another call shares its record.
//
// The command processor thread rebuilds the register state from the log alone: at each draw packet it matched to a
// library call it applies the log up to that packet (seeded once from the register file at the first match, and again
// after any loss), and with gpu_native_half2_log_verify compares the rebuilt state with the register file: every
// register 0x2000-0x23FF, the constants the bound shaders read, any register below 0x2000 the log wrote, the bound
// shaders. [native][half2log] lines every gpu_native_half2_log_frames frames.

#include <algorithm>
#include <bit>
#include <chrono>
#include <functional>
#include <thread>
#include <cstring>
#include <string>
#include <vector>

#include <fmt/format.h>

#include <rex/cvar.h>
#include <rex/det.h>
#include <rex/graphics/command_processor.h>
#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/xenos.h>
#include <rex/logging.h>
#include <rex/memory.h>

#include "half2_state.h"
#include "vulkan/native/shader_swap.h"

REXCVAR_DEFINE_BOOL(gpu_native_half2_log_verify, false, "GPU/Native",
                    "Session 51 (gpu_native_half2 3): compare the register state rebuilt from the library calls' "
                    "state log with the register file at every matched draw packet ([native][half2log] lines)");
REXCVAR_DEFINE_INT32(gpu_native_detached, 0, "GPU/Native",
                     "Session 51 (SESSION51_PLAN.md steps 2-3, docs\\native\\STATE_LOG.md): frames drawn from the library "
                     "calls' state log (needs gpu_native_half2 3). 0 off; 1 log-fed - each library segment the ring "
                     "submits is executed from the log's copies of the calls' packets, words no record holds (the "
                     "segment's prologue, the end-of-segment fence) from guest memory ([native][detached] census); 2 "
                     "detached - as 1, and the game's present hook puts the D3D library in its no-GPU mode: nothing "
                     "reaches the ring, the command processor executes the log by itself; any loss turns it off from "
                     "the next present (300 frames)");
REXCVAR_DEFINE_INT32(gpu_native_detached_wait_us, 100, "GPU/Native",
                     "gpu_native_detached: how long a segment waits for a record not yet published (a kick inside a "
                     "call publishes its piece just after the submit) before the rest runs from guest memory");
REXCVAR_DEFINE_INT32(gpu_native_detached_pace_ms, 2000, "GPU/Native",
                     "gpu_native_detached 2: how long a present waits for the command thread to finish the previous "
                     "frame from the log before the no-GPU mode is turned off (a hang guard)");
REXCVAR_DECLARE(int32_t, gpu_native_half2_log_frames);
REXCVAR_DECLARE(bool, gpu_native_half2_skip_gpr_draws);

namespace rex::graphics {

namespace {

// The game's writers run on its render thread; a call made inside another (the clear's own draws, say) belongs to the
// outer call's record. A call's packets are one range unless the library kicked the segment inside it (cmdbuf_kickoff
// 0x82CB6EA8: the write pointer goes on in another buffer): then each finished piece is published at once as its own
// record (flag kFlagContinues: the call goes on in the next record of the same sequence), so a segment the library
// submitted has its words in the log right after the kick.
constexpr uint32_t kMaxPieceWords = 1u << 16;  // longer = no plausible range
thread_local uint32_t t_depth = 0;
thread_local uint64_t t_seq = 0;
thread_local uint32_t t_kind = 0;
thread_local uint32_t t_begin = 0;  // the open piece's first word (guest physical)
thread_local uint32_t t_pieces = 0;  // pieces published so far
thread_local bool t_split = false;
// Where the last logged packets ended (guest physical; 0 = unknown). Packets in a segment are contiguous, so words
// between it and the next call's first word were written outside every hooked call: logged as a gap record (kind 0).
thread_local uint32_t t_last_end = 0;

uint64_t NowNs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}

// A piece's length in words: 0 and false when [begin, end) is no plausible range.
bool PieceWords(uint32_t begin, uint32_t end, uint32_t& words) {
  words = 0;
  if (begin == UINT32_MAX || end == UINT32_MAX || end < begin ||
      end - begin > 4 * kMaxPieceWords) {
    return false;
  }
  words = (end - begin) / 4;
  return true;
}

}  // namespace

// Writes one single-piece record (the caller holds no lock). False when the ring is full.
bool CommandProcessor::Half2LogWriteRecord(Half2Log& log, uint32_t kind, uint64_t seq, uint32_t begin, const uint32_t* src,
                 uint32_t words, uint32_t flags) {
  std::lock_guard lock(log.mutex);
  const uint64_t w = log.write.load(std::memory_order_relaxed);
  const uint32_t pieces = words ? 1 : 0;
  const uint32_t head = Half2Log::kHeaderWords + 2 * pieces;
  const uint32_t total = head + words;
  if (w + total - log.read_published.load(std::memory_order_acquire) > Half2Log::kWords &&
      log.detached.load(std::memory_order_relaxed)) {
    // (Step 3) Detached, the command processor drains the log at its pace: wait for room (up to 100 ms) instead of
    // losing the record - the library no longer waits for the GPU itself.
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    while (w + total - log.read_published.load(std::memory_order_acquire) > Half2Log::kWords &&
           std::chrono::steady_clock::now() < until) {
      std::this_thread::yield();
    }
  }
  if (w + total - log.read_published.load(std::memory_order_acquire) > Half2Log::kWords) {
    log.ring_full.fetch_add(1, std::memory_order_relaxed);
    if (log.detached.load(std::memory_order_relaxed)) log.ring_waits_full.fetch_add(1, std::memory_order_relaxed);
    log.lost.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  constexpr uint64_t kMask = Half2Log::kWords - 1;
  uint32_t* ring = log.ring.get();
  const uint32_t header[Half2Log::kHeaderWords] = {kind, uint32_t(seq), uint32_t(seq >> 32), pieces,
                                                                    words, flags};
  for (uint32_t i = 0; i < Half2Log::kHeaderWords; ++i) ring[(w + i) & kMask] = header[i];
  if (pieces) {
    ring[(w + Half2Log::kHeaderWords) & kMask] = begin;
    ring[(w + Half2Log::kHeaderWords + 1) & kMask] = words;
    const uint64_t at = (w + head) & kMask;
    const uint64_t first = std::min<uint64_t>(words, Half2Log::kWords - at);
    std::memcpy(ring + at, src, first * 4);
    if (first < words) std::memcpy(ring, src + first, (words - first) * 4);
  }
  log.write.store(w + total, std::memory_order_release);
  return true;
}

// The words between the last logged packets and `upto` (guest physical), as a gap record. At a call's start and at a
// kick outside every call.
void CommandProcessor::Half2LogGap(Half2Log& log, uint32_t upto) {
  if (!t_last_end || upto == t_last_end) {
    return;
  }
  uint32_t words;
  if (!PieceWords(t_last_end, upto, words)) {
    // The write pointer moved to another buffer without a kick the hooks saw: what was written there is unknown. The
    // jump takes a sequence number with no record (review 1: the consumer sees the hole where it happened).
    log.next_seq.fetch_add(1, std::memory_order_relaxed);
    log.jumps.fetch_add(1, std::memory_order_relaxed);
    if (log.jump_logged.fetch_add(1, std::memory_order_relaxed) < 8) {
      std::string words;
      if (t_last_end < 0x1FFFFFF0) {
        const uint32_t* w = memory_->TranslatePhysical<const uint32_t*>(t_last_end);
        for (int i = 0; i < 4; ++i) words += fmt::format(" {:08X}", std::byteswap(w[i]));
      }
      REXLOG_INFO("[native][half2log] write pointer jump: the last packets ended at {:08X}, the next begin at {:08X} "
                  "({}detached); words there:{}",
                  t_last_end, upto, log.detached.load(std::memory_order_relaxed) ? "" : "not ", words);
    }
    log.lost.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  const uint64_t seq = log.next_seq.fetch_add(1, std::memory_order_relaxed);
  if (Half2LogWriteRecord(log, 0, seq, t_last_end, memory_->TranslatePhysical<const uint32_t*>(t_last_end), words, 0)) {
    log.gap_records.fetch_add(1, std::memory_order_relaxed);
    log.gap_words.fetch_add(words, std::memory_order_relaxed);
  }
}

uint64_t CommandProcessor::Half2LogReserve(uint32_t kind, uint32_t packets_begin_physical) {
  Half2Log* const log_ptr = half2_log_.load(std::memory_order_acquire);
  if (!log_ptr) {
    return 0;  // made at the frame boundary (Half2LogFrame), on the command processor thread
  }
  Half2Log& log = *log_ptr;
  // (Review 2) gpu_native_half2 switched (away and back): what was written meanwhile is not in the log.
  thread_local uint32_t t_generation = 0;
  if (const uint32_t generation = half2_generation(); generation != t_generation) {
    t_generation = generation;
    t_last_end = 0;
  }
  if (t_depth++ == 0) {
    Half2LogGap(log, packets_begin_physical);
    t_seq = log.next_seq.fetch_add(1, std::memory_order_relaxed);
    t_kind = kind;
    t_begin = packets_begin_physical;
    t_pieces = 0;
    t_split = false;
  } else {
    log.nested.fetch_add(1, std::memory_order_relaxed);
  }
  return t_seq;
}

void CommandProcessor::Half2LogRefused() {
  if (Half2Log* const log = half2_log_.load(std::memory_order_acquire); log && half2_logging()) {
    log->lost.fetch_add(1, std::memory_order_relaxed);
  }
}

void CommandProcessor::Half2LogCall(uint32_t kind, uint32_t packets_begin_virtual) {
  if (!half2_logging()) {
    return;
  }
  if (!Half2LogReserve(kind, memory_->GetPhysicalAddress(packets_begin_virtual))) {
    return;
  }
}

void CommandProcessor::Half2LogCallEnd(uint32_t packets_end_virtual) {
  if (t_depth) Half2LogAppend(memory_->GetPhysicalAddress(packets_end_virtual));
  // (Step 3) Detached, the present's record completes a frame for the command thread: wake it.
  if (Half2Log* const log = half2_log_.load(std::memory_order_acquire);
      log && log->detached.load(std::memory_order_relaxed)) {
    write_ptr_index_event_->Set();
  }
}

// (Review) Before the kickoff submits the segment: the call's piece (or the gap) up to the old write pointer is published
// now, so the segment never runs before its records exist; Half2LogKick after it only moves on to the new buffer.
thread_local bool t_kick_prepared = false;
void CommandProcessor::Half2LogKickPrepare(uint32_t old_end_virtual) {
  Half2Log* const log_ptr = half2_log_.load(std::memory_order_acquire);
  t_kick_prepared = false;
  if (!log_ptr || !half2_logging()) {
    return;
  }
  Half2Log& log = *log_ptr;
  const uint32_t old_end = memory_->GetPhysicalAddress(old_end_virtual);
  if (!t_depth) {
    Half2LogGap(log, old_end);
    t_last_end = old_end;
  } else {
    uint32_t words;
    if (t_split || !PieceWords(t_begin, old_end, words)) {
      t_split = true;
    } else if (words && Half2LogWriteRecord(log, t_kind, t_seq, t_begin,
                                            memory_->TranslatePhysical<const uint32_t*>(t_begin), words,
                                            Half2Log::kFlagContinues)) {
      ++t_pieces;
      log.records.fetch_add(1, std::memory_order_relaxed);
      log.payload_words.fetch_add(words, std::memory_order_relaxed);
    }
    t_begin = old_end;  // (the piece is published: nothing more until the new buffer)
  }
  t_kick_prepared = true;
}

void CommandProcessor::Half2LogKick(uint32_t old_end_virtual, uint32_t new_begin_virtual) {
  if (t_kick_prepared) {  // published before the submit: only the new buffer
    t_kick_prepared = false;
    Half2Log* const log_ptr = half2_log_.load(std::memory_order_acquire);
    if (!log_ptr || !half2_logging()) return;
    const uint32_t new_begin = memory_->GetPhysicalAddress(new_begin_virtual);
    if (!t_depth) {
      t_last_end = new_begin;
      log_ptr->kicks_outside.fetch_add(1, std::memory_order_relaxed);
    } else {
      t_begin = new_begin;
      log_ptr->kicks.fetch_add(1, std::memory_order_relaxed);
    }
    return;
  }
  Half2Log* const log_ptr = half2_log_.load(std::memory_order_acquire);
  if (!log_ptr || !half2_logging()) {
    return;
  }
  Half2Log& log = *log_ptr;
  if (!t_depth) {
    // Outside every call: what was written before the kick is a gap; the packets go on in the new buffer.
    Half2LogGap(log, memory_->GetPhysicalAddress(old_end_virtual));
    t_last_end = memory_->GetPhysicalAddress(new_begin_virtual);
    log.kicks_outside.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  log.kicks.fetch_add(1, std::memory_order_relaxed);
  uint32_t words;
  const uint32_t old_end = memory_->GetPhysicalAddress(old_end_virtual);
  if (t_split || !PieceWords(t_begin, old_end, words)) {
    if (log.split_logged.fetch_add(1, std::memory_order_relaxed) < 6) {
      REXLOG_INFO("[native][half2log] split at a kick: call kind {} piece {} began {:08X}, old write pointer end {:08X} "
                  "(virtual {:08X}), new begin virtual {:08X}",
                  t_kind, t_pieces, t_begin, old_end, old_end_virtual, new_begin_virtual);
    }
    t_split = true;
  } else if (words) {
    if (Half2LogWriteRecord(log, t_kind, t_seq, t_begin, memory_->TranslatePhysical<const uint32_t*>(t_begin), words,
                            Half2Log::kFlagContinues)) {
      ++t_pieces;
      log.records.fetch_add(1, std::memory_order_relaxed);
      log.payload_words.fetch_add(words, std::memory_order_relaxed);
    }
  }
  t_begin = memory_->GetPhysicalAddress(new_begin_virtual);
}

void CommandProcessor::Half2LogAppend(uint32_t packets_end_physical) {
  Half2Log* const log_ptr = half2_log_.load(std::memory_order_acquire);
  if (!log_ptr || !t_depth || --t_depth) {
    return;  // the outer call appends the record
  }
  Half2Log& log = *log_ptr;
  // The copy's time is sampled (1 record in 16, scaled): two clock reads a record cost about as much as the copy.
  thread_local uint32_t t_sample = 0;
  const bool timed = (t_sample++ & 15) == 0;
  const uint64_t start_ns = timed ? NowNs() : 0;
  uint32_t last = 0;
  t_last_end = packets_end_physical;
  if (!PieceWords(t_begin, packets_end_physical, last)) {
    if (log.split_logged.fetch_add(1, std::memory_order_relaxed) < 6) {
      REXLOG_INFO("[native][half2log] split at the end: call kind {} piece {} began {:08X}, ended {:08X}", t_kind,
                  t_pieces, t_begin, packets_end_physical);
    }
    t_split = true;
  }
  uint32_t flags = 0;
  if (t_split) {
    flags |= Half2Log::kFlagSplit;
    last = 0;
    log.splits.fetch_add(1, std::memory_order_relaxed);
    log.lost.fetch_add(1, std::memory_order_relaxed);
    t_last_end = 0;  // unknown: no gap is measured from it
  }
  if (Half2LogWriteRecord(log, t_kind, t_seq, t_begin, last ? memory_->TranslatePhysical<const uint32_t*>(t_begin) : nullptr,
                          last, flags)) {
    log.records.fetch_add(1, std::memory_order_relaxed);
    log.payload_words.fetch_add(last, std::memory_order_relaxed);
    if (t_pieces) log.multi_piece.fetch_add(1, std::memory_order_relaxed);
  }
  if (timed) log.copy_ns.fetch_add(16 * (NowNs() - start_ns), std::memory_order_relaxed);
}

void CommandProcessor::Half2LogAt(uint64_t seq, uint32_t at, uint32_t kind, bool via_ib) {
  Half2Log* const log_ptr = half2_log_.load(std::memory_order_relaxed);
  if (!log_ptr || !half2_logging() || REXCVAR_GET(gpu_native_detached) >= 1) {
    return;  // (gpu_native_detached: the segment executor consumes the log)
  }
  Half2Log& log = *log_ptr;
  Half2Log::Census& c = log.window;
  ++c.at_calls;
  const bool timed = (c.at_calls & 15) == 0;  // sampled as the copy is (1 in 16, scaled)
  const uint64_t start_ns = timed ? NowNs() : 0;
  auto add_time = [&]() {
    if (timed) c.apply_ns += 16 * (NowNs() - start_ns);
  };
  constexpr uint64_t kMask = Half2Log::kWords - 1;
  const uint32_t* ring = log.ring.get();
  auto word = [&](uint32_t i) { return std::byteswap(ring[(log.cur_payload + i) & kMask]); };
  // Guest physical address <-> payload word, through the open record's pieces.
  auto to_payload = [&](uint32_t address) -> uint32_t {
    uint32_t base = 0;
    for (uint32_t i = 0; i < log.cur_pieces; ++i) {
      const uint32_t b = log.cur_piece_begin[i], n = log.cur_piece_words[i];
      if (address >= b && (address - b) / 4 < n) return base + (address - b) / 4;
      base += n;
    }
    return UINT32_MAX;
  };
  auto to_physical = [&](uint32_t pos) -> uint32_t {
    for (uint32_t i = 0; i < log.cur_pieces; ++i) {
      if (pos < log.cur_piece_words[i]) return log.cur_piece_begin[i] + 4 * pos;
      pos -= log.cur_piece_words[i];
    }
    return 0;
  };
  // Diagnostic: the open record's packet headers (offset: header), up to 48.
  auto dump = [&](uint32_t at_pos, const std::string& why) {
    std::string list;
    uint32_t pos = 0, shown = 0;
    while (pos < log.cur_words && shown < 64 && pos <= at_pos + 40) {
      const uint32_t h = word(pos);
      const uint32_t t = h >> 30;
      const uint32_t n = t == 0 || t == 3 ? ((h >> 16) & 0x3FFF) + 2 : t == 1 ? 3 : 1;
      if (pos + 400 >= at_pos) {
        list += fmt::format(" {}:{:08X}", pos, h);
        ++shown;
      }
      pos += n;
    }
    list += "; raw from the spot:";
    for (uint32_t i = 0; i < 12 && at_pos + i < log.cur_words; ++i) list += fmt::format(" {:08X}", word(at_pos + i));
    std::string pieces;
    for (uint32_t i = 0; i < log.cur_pieces; ++i)
      pieces += fmt::format(" {:08X}+{}", log.cur_piece_begin[i], log.cur_piece_words[i]);
    REXLOG_INFO("[native][half2log] dump ({} at word {}): record {} kind {}, pieces{}, {} words:{}", why, at_pos,
                log.cur_seq, log.cur_kind, pieces, log.cur_words, list);
  };
  // One packet at payload word `pos`: its length in words (0: past the record); applied to the log's state when
  // `apply`. Mirrors the command processor's packet executor for what the library writes (ExecutePacketType0..3).
  // Where packets are read from: the open record (the ring), or guest memory (an indirect buffer a packet called).
  struct Src {
    const uint32_t* guest = nullptr;  // null: the open record
    uint32_t guest_phys = 0;
    uint32_t words = 0;
  };
  auto src_word = [&](const Src& src, uint32_t i) {
    return src.guest ? std::byteswap(src.guest[i]) : word(i);
  };
  auto src_phys = [&](const Src& src, uint32_t pos) {
    return src.guest ? src.guest_phys + 4 * pos : to_physical(pos);
  };
  std::function<uint32_t(const Src&, uint32_t, bool, uint32_t)> packet;
  packet = [&](const Src& src, uint32_t pos, bool apply, uint32_t depth) -> uint32_t {
    auto word = [&](uint32_t i) { return src_word(src, i); };
    auto to_physical = [&](uint32_t i) { return src_phys(src, i); };
    if (pos >= src.words) return 0;
    const uint32_t p = word(pos);
    const uint32_t type = p >> 30;
    uint32_t* v = log.values.get();
    auto set = [&](uint32_t reg, uint32_t value) {
      if (reg < RegisterFile::kRegisterCount) {
        v[reg] = value;
        log.touched[reg >> 6] |= uint64_t(1) << (reg & 63);
      }
    };
    if (type == 0) {
      const uint32_t count = ((p >> 16) & 0x3FFF) + 1, base = p & 0x7FFF;
      const bool one = (p >> 15) & 1;
      if (pos + 1 + count > src.words) return 0;
      if (apply) {
        for (uint32_t i = 0; i < count; ++i) set(base + (one ? 0 : i), word(pos + 1 + i));
      }
      return 1 + count;
    }
    if (type == 1) {
      if (pos + 3 > src.words) return 0;
      if (apply && !src.guest && log.type1_dumps < 4) {
        ++log.type1_dumps;
        dump(pos, "a type-1 packet");
      }
      if (apply) set(p & 0x7FF, word(pos + 1)), set((p >> 11) & 0x7FF, word(pos + 2));
      return 3;
    }
    if (type == 2) return 1;
    const uint32_t opcode = (p >> 8) & 0x7F, count = ((p >> 16) & 0x3FFF) + 1;
    if (pos + 1 + count > src.words) return 0;
    if (!apply) return 1 + count;
    ++c.applied_packets;
    ++log.opcodes[opcode];  // (a fixed array)
    if ((p & 1) && !(log.bin_select & log.bin_mask)) {
      ++c.predicated_skipped;
      return 1 + count;
    }
    const uint32_t d = pos + 1;
    static constexpr uint32_t kBase[5] = {0x4000, 0x4800, 0x4900, 0x4908, 0x2000};
    switch (opcode) {
      case xenos::PM4_SET_CONSTANT: {
        const uint32_t offset_type = word(d), index = offset_type & 0x7FF, t = (offset_type >> 16) & 0xFF;
        if (t <= 4) {
          for (uint32_t i = 0; i + 1 < count; ++i) set(kBase[t] + index + i, word(d + 1 + i));
        }
      } break;
      case xenos::PM4_SET_CONSTANT2:
      case xenos::PM4_SET_SHADER_CONSTANTS: {
        const uint32_t index = word(d) & 0xFFFF;
        for (uint32_t i = 0; i + 1 < count; ++i) set(index + i, word(d + 1 + i));
      } break;
      case xenos::PM4_LOAD_ALU_CONSTANT: {
        const uint32_t address = word(d) & 0x3FFFFFFF, offset_type = word(d + 1);
        const uint32_t index = offset_type & 0x7FF, t = (offset_type >> 16) & 0xFF, n = word(d + 2) & 0xFFF;
        if (t <= 4 && uint64_t(address) + 4 * uint64_t(n) <= 0x20000000) {
          const uint32_t* src = memory_->TranslatePhysical<const uint32_t*>(address);
          for (uint32_t i = 0; i < n; ++i) set(kBase[t] + index + i, std::byteswap(src[i]));
        }
      } break;
      case xenos::PM4_IM_LOAD: {
        const uint32_t addr_type = word(d);
        const uint32_t t = addr_type & 3;
        // (Session 90: a D3D-layer microcode address tagged with bit 31 = guest virtual is masked away here; IM_LOAD packets in a
        // ring never carry the tag today - only records / tokens do - so this log line cannot see one.)
        log.ucode[t == uint32_t(xenos::ShaderType::kVertex) ? 0 : 1] = (addr_type & ~3u) & 0x1FFFFFFF;
        log.shader_at[t == uint32_t(xenos::ShaderType::kVertex) ? 0 : 1] = to_physical(pos);
      } break;
      case xenos::PM4_IM_LOAD_IMMEDIATE: {
        const bool vertex = word(d) == uint32_t(xenos::ShaderType::kVertex);
        log.ucode[vertex ? 0 : 1] = 0;
        log.shader_at[vertex ? 0 : 1] = to_physical(pos);
        if (vertex) log.inline_vs_at = to_physical(pos);
      } break;
      case xenos::PM4_SET_BIN_MASK_LO:
        log.bin_mask = (log.bin_mask & 0xFFFFFFFF00000000ull) | word(d);
        break;
      case xenos::PM4_SET_BIN_MASK_HI:
        log.bin_mask = (log.bin_mask & 0xFFFFFFFFull) | (uint64_t(word(d)) << 32);
        break;
      case xenos::PM4_SET_BIN_SELECT_LO:
        log.bin_select = (log.bin_select & 0xFFFFFFFF00000000ull) | word(d);
        break;
      case xenos::PM4_SET_BIN_SELECT_HI:
        log.bin_select = (log.bin_select & 0xFFFFFFFFull) | (uint64_t(word(d)) << 32);
        break;
      case xenos::PM4_SET_BIN_MASK:
        log.bin_mask = (uint64_t(word(d)) << 32) | word(d + 1);
        break;
      case xenos::PM4_SET_BIN_SELECT:
        log.bin_select = (uint64_t(word(d)) << 32) | word(d + 1);
        break;
      case xenos::PM4_DRAW_INDX:
      case xenos::PM4_DRAW_INDX_2: {
        // As ExecutePacketType3Draw: the initiator, and for a DMA source the index base and size.
        const uint32_t first = opcode == xenos::PM4_DRAW_INDX ? d + 1 : d;
        reg::VGT_DRAW_INITIATOR initiator;
        initiator.value = word(first);
        set(XE_GPU_REG_VGT_DRAW_INITIATOR, initiator.value);
        if (initiator.source_select == xenos::SourceSelect::kDMA && first + 2 < d + count) {
          set(XE_GPU_REG_VGT_DMA_BASE, word(first + 1));
          set(XE_GPU_REG_VGT_DMA_SIZE, word(first + 2));
        }
      } break;
      case xenos::PM4_REG_RMW: {
        // As ExecutePacketType3_REG_RMW: register = (register & (and register or immediate)) | (or register or
        // immediate).
        const uint32_t info = word(d), and_mask = word(d + 1), or_mask = word(d + 2);
        uint32_t* vv = log.values.get();
        uint32_t value = vv[info & 0x1FFF];
        value &= (info >> 31) & 1 ? vv[and_mask & 0x1FFF] : and_mask;
        value |= (info >> 30) & 1 ? vv[or_mask & 0x1FFF] : or_mask;
        set(info & 0x1FFF, value);
      } break;
      case xenos::PM4_INDIRECT_BUFFER:
      case xenos::PM4_INDIRECT_BUFFER_PFD: {
        // As ExecutePacketType3_INDIRECT_BUFFER: the called buffer's packets, read from guest memory now (the library's
        // prebuilt buffers at 0x10FExxxx: the GPR flush, the resolve / clear helpers' shaders; run 3465). The GPR-flush
        // buffers of 24 point draws are skipped as the command processor skips them (gpu_native_half2_skip_gpr_draws).
        ++c.ib_packets;
        const uint32_t list = xenos::GpuToCpu(xenos::CpuToGpu(word(d))) & 0x1FFFFFFF, length = word(d + 1) & 0xFFFFF;
        if (depth >= 2 || length > 1024 || uint64_t(list) + 4 * uint64_t(length) > 0x20000000) {
          ++c.ib_not_followed;
          log.seeded = false;  // (review 4) its writes are not applied: reseed at the next match
          break;
        }
        Src ib{memory_->TranslatePhysical<const uint32_t*>(list), list, length};
        if (REXCVAR_GET(gpu_native_half2_skip_gpr_draws) && length == 48) {
          bool gpr = true;
          for (uint32_t i = 0; i < 48 && gpr; i += 2) {
            gpr = std::byteswap(ib.guest[i]) == 0xC0003600u && std::byteswap(ib.guest[i + 1]) == 0x00010081u;
          }
          if (gpr) break;
        }
        ++c.ib_followed;
        if (log.ib_logged < 6 && log.ib_seen.insert(list).second) {
          ++log.ib_logged;
          REXLOG_INFO("[native][half2log] following an indirect buffer at {:08X} ({} words) called from {:08X}", list,
                      length, to_physical(pos));
        }
        for (uint32_t at = 0; at < length;) {
          const uint32_t n = packet(ib, at, true, depth + 1);
          if (!n) break;
          at += n;
        }
      } break;
      case xenos::PM4_EVENT_WRITE:
      case xenos::PM4_EVENT_WRITE_SHD:
      case xenos::PM4_EVENT_WRITE_EXT:
        set(XE_GPU_REG_VGT_EVENT_INITIATOR, word(d) & 0x3F);  // as ExecutePacketType3_EVENT_WRITE*
        break;
      default:
        break;
    }
    return 1 + count;
  };
  // Applies the open record's packets from cur_pos while their header is at or before `until` (payload words).
  auto advance = [&](uint32_t until, bool apply) {
    const Src record{nullptr, 0, log.cur_words};
    while (log.cur_pos < log.cur_words && log.cur_pos <= until) {
      const uint32_t n = packet(record, log.cur_pos, apply, 0);
      if (!n) {
        log.cur_pos = log.cur_words;
        break;
      }
      if (apply) c.applied_words += n;
      log.cur_pos += n;
    }
  };
  auto close = [&]() {
    if (!log.open) return;
    advance(UINT32_MAX, log.seeded);
    log.read = log.cur_payload + log.cur_words;
    log.read_published.store(log.read, std::memory_order_release);
    log.open = false;
  };
  // Up to the record of call `seq`: every earlier record applied in full.
  while (!log.open || log.cur_seq != seq) {
    if (log.open && log.cur_seq > seq) {
      ++c.behind;  // a later record is already open (this draw's record was passed): nothing to do
      add_time();
      return;
    }
    close();
    if (log.read == log.write.load(std::memory_order_acquire)) {
      ++c.not_ready;
      add_time();
      return;
    }
    const uint32_t* h = ring;
    const uint64_t r = log.read;
    log.cur_kind = h[r & kMask];
    log.cur_seq = uint64_t(h[(r + 1) & kMask]) | (uint64_t(h[(r + 2) & kMask]) << 32);
    // (Review 1) A loss leaves a hole in the sequence where it happened (a jump takes a number, a refused record keeps
    // its own): reseed there. The next piece of the same call has the same number.
    if (log.last_seq && log.cur_seq != log.last_seq && log.cur_seq != log.last_seq + 1) {
      log.seeded = false;
      ++c.seq_holes;
    }
    log.last_seq = log.cur_seq;
    log.cur_pieces = std::min<uint32_t>(h[(r + 3) & kMask], Half2Log::kMaxPieces);
    log.cur_words = h[(r + 4) & kMask];
    const uint32_t flags = h[(r + 5) & kMask];
    for (uint32_t i = 0; i < log.cur_pieces; ++i) {
      log.cur_piece_begin[i] = h[(r + Half2Log::kHeaderWords + 2 * i) & kMask];
      log.cur_piece_words[i] = h[(r + Half2Log::kHeaderWords + 2 * i + 1) & kMask];
    }
    log.cur_begin = log.cur_pieces ? log.cur_piece_begin[0] : 0;
    log.cur_payload = r + Half2Log::kHeaderWords + 2 * uint64_t(h[(r + 3) & kMask]);
    log.cur_flags = flags;
    log.cur_pos = 0;
    log.open = true;
    if (log.cur_kind == 0 && log.kind2_dumps < 8) {
      ++log.kind2_dumps;
      dump(0, "a gap record");
    }
    if (flags & Half2Log::kFlagSplit) log.seeded = false;
    if (log.cur_seq > seq) {
      ++c.seq_gaps;  // the call's record is missing (refused): reseed at the next match
      log.seeded = false;
      add_time();
      return;
    }
  }
  uint32_t target = to_payload(at);
  // The call went on in its next piece (a kick inside it): that record is next, with the same sequence.
  while (target == UINT32_MAX && (log.cur_flags & Half2Log::kFlagContinues) &&
         log.read != log.write.load(std::memory_order_acquire)) {
    close();
    const uint64_t r = log.read;
    if ((uint64_t(ring[(r + 1) & kMask]) | (uint64_t(ring[(r + 2) & kMask]) << 32)) != seq) break;
    log.cur_kind = ring[r & kMask];
    log.cur_seq = seq;
    log.cur_pieces = std::min<uint32_t>(ring[(r + 3) & kMask], Half2Log::kMaxPieces);
    log.cur_words = ring[(r + 4) & kMask];
    log.cur_flags = ring[(r + 5) & kMask];
    for (uint32_t i = 0; i < log.cur_pieces; ++i) {
      log.cur_piece_begin[i] = ring[(r + Half2Log::kHeaderWords + 2 * i) & kMask];
      log.cur_piece_words[i] = ring[(r + Half2Log::kHeaderWords + 2 * i + 1) & kMask];
    }
    log.cur_begin = log.cur_pieces ? log.cur_piece_begin[0] : 0;
    log.cur_payload = r + Half2Log::kHeaderWords + 2 * uint64_t(ring[(r + 3) & kMask]);
    log.cur_pos = 0;
    log.open = true;
    if (log.cur_flags & Half2Log::kFlagSplit) log.seeded = false;
    target = to_payload(at);
  }
  if (target == UINT32_MAX) {
    ++c.out_of_range;
    add_time();
    return;
  }
  if (!log.seeded && via_ib) {
    ++c.via_ib;  // (review 3) never seed inside a called buffer: its later writes would be missed
    add_time();
    return;
  }
  if (!log.seeded) {
    // The register file at this packet is the state the log continues from (the command processor has executed
    // exactly the packets before it, and this draw packet's own writes).
    std::memcpy(log.values.get(), register_file_->values, sizeof(uint32_t) * RegisterFile::kRegisterCount);
    std::memset(log.touched.get(), 0, sizeof(uint64_t) * ((RegisterFile::kRegisterCount + 63) / 64));
    log.bin_select = bin_select_;
    log.bin_mask = bin_mask_;
    log.ucode[0] = half2_ucode_[0];
    log.ucode[1] = half2_ucode_[1];
    log.inline_vs_at = half2_inline_vs_at_;
    log.shader_at[0] = half2_shader_at_[0];
    log.shader_at[1] = half2_shader_at_[1];
    advance(target, false);
    log.seeded = true;
    ++c.seeds;
    add_time();
    return;
  }
  advance(target, true);  // (already past it - a tiling replay of the packets: the state already holds it)
  add_time();
  if (via_ib) {
    ++c.via_ib;  // the draw is inside a buffer the call's packets called: its writes are not in the log
    return;
  }
  if (!REXCVAR_GET(gpu_native_half2_log_verify)) {
    return;
  }
  // The compare: registers 0x2000-0x23FF, the constants the bound shaders read, the log's registers below 0x2000.
  ++c.compared;
  const RegisterFile& rf = *register_file_;
  const uint32_t* v = log.values.get();
  bool analysed = false;
  uint64_t used_groups[2] = {};
  uint32_t used_fetch[6] = {};
  uint64_t used_bool_loop = 0;
  const Shader* shaders[2] = {active_vertex_shader_, active_pixel_shader_};
  if (shaders[0] && shaders[1] && shaders[0]->is_ucode_analyzed() && shaders[1]->is_ucode_analyzed()) {
    analysed = true;
    for (uint32_t k = 0; k < 2; ++k) {
      const Shader::ConstantRegisterMap& map = shaders[k]->constant_register_map();
      for (uint32_t cst = 0; cst < 256; ++cst) {
        if (map.float_dynamic_addressing || (map.float_bitmap[cst >> 6] >> (cst & 63) & 1)) {
          const uint32_t group = 64 * k + cst / 4;
          used_groups[group >> 6] |= uint64_t(1) << (group & 63);
        }
      }
      for (uint32_t b = 0; b < 8; ++b) {
        if (map.bool_bitmap[b]) used_bool_loop |= uint64_t(1) << b;
      }
      for (uint32_t l = 0; l < 32; ++l) {
        if (map.loop_bitmap >> l & 1) used_bool_loop |= uint64_t(1) << (8 + l);
      }
      for (uint32_t vf = 0; vf < 96; ++vf) {
        if (map.vertex_fetch_bitmap[vf >> 5] >> (vf & 31) & 1) {
          for (uint32_t w = 2 * vf; w < 2 * vf + 2; ++w) used_fetch[w >> 5] |= 1u << (w & 31);
        }
      }
      for (const Shader::TextureBinding& t : shaders[k]->texture_bindings()) {
        for (uint32_t w = 6 * t.fetch_constant; w < 6 * t.fetch_constant + 6 && w < 192; ++w) {
          used_fetch[w >> 5] |= 1u << (w & 31);
        }
      }
    }
  }
  uint32_t differ = 0;
  auto check = [&](uint32_t reg) {
    // COHER_SIZE / BASE / STATUS_PM4 (0x0A2F-0x0A31): cache-coherency requests the command processor acknowledges
    // itself (it clears the status); VGT_EVENT_INITIATOR (0x21F9): the last event triggered (EVENT_WRITE packets
    // between the library calls write it too); DC_LUT_RW_INDEX .. DC_LUT_30_COLOR (0x1922-): the gamma ramp's access
    // registers, whose index the command processor increments as ramp data is written (run 3466). Not draw state.
    if ((reg >= 0x0A2F && reg <= 0x0A31) || reg == XE_GPU_REG_VGT_EVENT_INITIATOR ||
        (reg >= XE_GPU_REG_DC_LUT_RW_INDEX && reg <= XE_GPU_REG_DC_LUT_30_COLOR)) {
      return;
    }
    ++c.compared_values;
    if (rf.values[reg] == v[reg]) return;
    ++differ;
    ++log.diff_registers[reg];
    if (log.lines_logged < 40 && log.diff_logged[reg]++ < 2) {
      ++log.lines_logged;
      REXLOG_INFO("[native][half2log] differs: register {:04X} register file {:08X}, state log {:08X} (call kind {}, "
                  "record {} at {:08X}+{}, draw packet {:08X}{})",
                  reg, rf.values[reg], v[reg], kind, log.cur_seq, log.cur_begin, 4 * target, at,
                  (log.touched[reg >> 6] >> (reg & 63) & 1) ? "" : "; never written by the log since the seed");
    }
  };
  for (uint32_t w = 0; w < 0x2000 / 64; ++w) {
    for (uint64_t bits = log.touched[w]; bits; bits &= bits - 1) check(64 * w + uint32_t(std::countr_zero(bits)));
  }
  for (uint32_t reg = 0x2000; reg < 0x2400; ++reg) check(reg);
  for (uint32_t reg = 0x4000; reg < 0x4800; ++reg) {
    const uint32_t group = (reg - 0x4000) / 16;
    if (!analysed || (used_groups[group >> 6] >> (group & 63) & 1)) check(reg);
  }
  for (uint32_t reg = 0x4800; reg < 0x48C0; ++reg) {
    if (!analysed || (used_fetch[(reg - 0x4800) >> 5] >> ((reg - 0x4800) & 31) & 1)) check(reg);
  }
  for (uint32_t reg = 0x4900; reg < 0x4928; ++reg) {
    if (!analysed || (used_bool_loop >> (reg - 0x4900) & 1)) check(reg);
  }
  // The bound shaders: IM_LOAD's microcode address, or the inline vertex shader's packet.
  const bool vs_same = log.ucode[0] == half2_ucode_[0] && (log.ucode[0] || log.inline_vs_at == half2_inline_vs_at_);
  const bool ps_same = log.ucode[1] == half2_ucode_[1];
  if (!vs_same || !ps_same) {
    ++c.shader_differs;
    if (log.lines_logged < 40) {
      ++log.lines_logged;
      REXLOG_INFO("[native][half2log] shader differs: VS register file {:08X} (inline at {:08X}), log {:08X} (inline at "
                  "{:08X}); PS {:08X} / {:08X} (call kind {}, record {}); bin select / mask: register file "
                  "{:016X} / {:016X}, log {:016X} / {:016X}; last loads VS / PS: register file {:08X} / {:08X}, log "
                  "{:08X} / {:08X}",
                  half2_ucode_[0], half2_inline_vs_at_, log.ucode[0], log.inline_vs_at, half2_ucode_[1], log.ucode[1],
                  kind, log.cur_seq, bin_select_, bin_mask_, log.bin_select, log.bin_mask, half2_shader_at_[0],
                  half2_shader_at_[1], log.shader_at[0], log.shader_at[1]);
    }

  }
  if (differ) {
    ++c.differing_draws;
    c.differing_values += differ;
  }
}

// Session 51 step 2 (gpu_native_detached 1): a library segment the ring submitted, executed from the log. The records
// whose piece lies in [start, start + 4 count) are executed from the log's copies, in order; the words no record holds
// (before the first, between two, after the last: the segment's prologue, the end-of-segment fence the kickoff writes)
// from guest memory. Every word runs once either way, through the same packet executor, with the packets' guest
// addresses as before (the attribution, lean draws and Half2EndSegment see no difference). False: not handled (the
// caller executes the segment from memory as before).
// Executes `words` words at host `data`, whose first word is guest physical `physical`, through the packet executor.
bool CommandProcessor::Half2LogRun(const uint32_t* data, uint32_t physical, uint32_t words) {
  memory::RingBuffer reader(reinterpret_cast<uint8_t*>(const_cast<uint32_t*>(data)), words * sizeof(uint32_t));
  reader.set_write_offset(words * sizeof(uint32_t));
  // ExecutePacket takes a packet's guest address from where it is read: shifted to the copy's original place.
  half2_exec_bias_ = int64_t(physical) - int64_t(reinterpret_cast<const uint8_t*>(data) -
                                                 reinterpret_cast<const uint8_t*>(memory_->TranslatePhysical(0)));
  bool ok = true;
  do {
    if (!ExecutePacket(&reader)) {
      ok = false;
      break;
    }
  } while (reader.read_count());
  half2_exec_bias_ = 0;
  return ok;
}

bool CommandProcessor::Half2LogExecuteSegment(uint32_t start, uint32_t count) {
  Half2Log* const log_ptr = half2_log_.load(std::memory_order_relaxed);
  if (!log_ptr || !half2_logging() || REXCVAR_GET(gpu_native_detached) < 1 || !count) {
    return false;
  }
  Half2Log& log = *log_ptr;
  log.open = false;  // (review: the verify consumer's record is not this executor's)
  Half2Log::Exec& x = log.exec;
  const uint64_t start_ns = NowNs();
  constexpr uint64_t kMask = Half2Log::kWords - 1;
  const uint32_t* ring = log.ring.get();
  const uint32_t end = start + 4 * count;
  uint32_t pos = start;
  bool any = false;
  bool ok = true;
  auto run = [&](const uint32_t* data, uint32_t physical, uint32_t words) {
    if (!Half2LogRun(data, physical, words)) ok = false;
  };
  // Step 3: records written while detached run first (a switch back); the words before the switch's write position
  // already ran from the log.
  if (!log.detached.load(std::memory_order_acquire) && log.drain_until_seq.load(std::memory_order_relaxed)) {
    Half2LogDrive();  // (only the drain: while detached the ring's segments still own their records)
  }
  // (Only the first ring segment after the switch: a skip left for later would hit a reuse of the same memory.)
  if (const uint32_t skip = log.skip_until.exchange(0, std::memory_order_relaxed); skip > start && skip <= end) {
    pos = skip;
    x.skipped_words += (skip - start) / 4;
  } else if (skip) {
    ++x.skip_missed;
  }
  auto from_memory = [&](uint32_t upto, uint64_t& counter) {
    if (upto > pos && ok) {
      run(memory_->TranslatePhysical<const uint32_t*>(pos), pos, (upto - pos) / 4);
      counter += (upto - pos) / 4;
      pos = upto;
    }
  };
  ++x.segments;
  const auto wait_limit = std::chrono::microseconds(std::max(0, REXCVAR_GET(gpu_native_detached_wait_us)));
  while (ok && pos < end) {
    // The next record (a kick inside a call publishes its piece just after the submit: wait a little).
    if (log.read == log.write.load(std::memory_order_acquire)) {
      ++x.waits;
      const auto until = std::chrono::steady_clock::now() + wait_limit;
      while (log.read == log.write.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < until) {
        std::this_thread::yield();
      }
      if (log.read == log.write.load(std::memory_order_acquire)) {
        ++x.wait_timeouts;
        break;
      }
    }
    const uint64_t r = log.read;
    const uint32_t kind = ring[r & kMask], pieces = ring[(r + 3) & kMask], words = ring[(r + 4) & kMask];
    const uint32_t begin = pieces ? ring[(r + Half2Log::kHeaderWords) & kMask] : 0;
    const uint64_t payload = r + Half2Log::kHeaderWords + 2 * uint64_t(pieces);
    const uint64_t record_seq = uint64_t(ring[(r + 1) & kMask]) | (uint64_t(ring[(r + 2) & kMask]) << 32);
    if (!pieces || !words) {
      log.self_last_seq = record_seq;  // (the last record taken by either executor: Half2LogDrive's hole check)
      ++x.split_records;  // no payload (a split): its words run from memory with the rest
      log.read = payload + words;
      log.read_published.store(log.read, std::memory_order_release);
      continue;
    }
    if (begin < pos || begin + 4 * words > end) {
      // Not in what is left of this segment: a later segment's (normal end), or a record of a segment the command
      // processor never executed (dropped once it has blocked 8 segments in a row).
      if (!any && ++log.blocked_segments >= 8) {
        log.self_last_seq = record_seq;
        ++x.stale_records;
        log.read = payload + words;
        log.read_published.store(log.read, std::memory_order_release);
        log.blocked_segments = 0;
        continue;
      }
      break;
    }
    from_memory(begin, any ? x.memory_between : x.memory_head);
    if (!ok) break;
    log.scratch.resize(words);
    const uint64_t at = payload & kMask;
    const uint64_t first = std::min<uint64_t>(words, Half2Log::kWords - at);
    std::memcpy(log.scratch.data(), ring + at, first * 4);
    if (first < words) std::memcpy(log.scratch.data() + first, ring, (words - first) * 4);
    log.read = payload + words;
    log.read_published.store(log.read, std::memory_order_release);
    log.self_last_seq = record_seq;
    run(log.scratch.data(), begin, words);
    pos = begin + 4 * words;
    x.log_words += words;
    ++x.records;
    ++x.records_by_kind[std::min<uint32_t>(kind, 5)];
    any = true;
    log.blocked_segments = 0;
  }
  if (!any) ++x.not_logged_segments;
  if (!ok) {
    ++x.failed;
    REXLOG_ERROR("[native][detached] a packet failed in the segment {:08X}+{} at {:08X}", start, count, pos);
  } else {
    from_memory(end, x.memory_tail);
  }
  x.exec_ns += NowNs() - start_ns;
  return true;
}

// Session 51 step 3 (gpu_native_detached 2): the library's no-GPU mode. The game's present hook sets the device's
// no-GPU bit when half2_detached_wanted() and reports the switch here; from then on the library hands nothing to the
// ring (listing 0x82CB5EF0) and the command processor thread executes the log by itself (Half2LogDrive, when its ring
// is idle). Switching back: the records written while detached are executed first, and the first ring segment after
// the switch skips the words before the switch's write position (they ran from the log).
// (Step 3) The pacing the library's no-GPU mode drops: its present waited for the GPU to finish the previous frame;
// in the no-GPU mode it marks every segment finished at once (listing 0x82CB64E4-0x82CB64FC: the fence written by the
// CPU) and reuses their memory. At the end of each present, while detached, the game waits here until the command
// thread has executed the log up to the present before the previous one (at most gpu_native_detached_pace_ms, then the fallback: a
// hang guard - the library's own wait had no limit; first-use pipeline compiles can take a frame well past 100 ms).
void CommandProcessor::Half2DetachedPace() {
  Half2Log* const log_ptr = half2_log_.load(std::memory_order_acquire);
  if (!log_ptr || !log_ptr->detached.load(std::memory_order_acquire)) {
    return;
  }
  Half2Log& log = *log_ptr;
  // Two presents in flight (the log's packets wait for values the game's CPU writes in the next frame -
  // WAIT_REG_MEM on 0x137FE006 once a frame: one frame in flight deadlocked at the first entrance, runs 3486 / 3488).
  const uint64_t previous = log.pace_mark_older;
  log.pace_mark_older = log.pace_mark;
  log.pace_mark = log.write.load(std::memory_order_acquire);  // this present's record is the last one written
  write_ptr_index_event_->Set();
  if (!previous) {
    return;
  }
  const auto start = std::chrono::steady_clock::now();
  const uint32_t limit_ms = uint32_t(std::max(1, REXCVAR_GET(gpu_native_detached_pace_ms)));
  if (det::SerialEnabled() && det::SerialIsGuestThread()) {
    // The deterministic scheduler runs one guest thread at a time: a host-side spin here would keep the baton from
    // the guest's interrupt callback that clears the word the log's WAIT_REG_MEM waits for (0x137FE006, run 3490:
    // a deadlock). Block as a guest wait does (the deadline in det vblanks).
    det::SerialBlock(
        [&log, previous] { return log.read_published.load(std::memory_order_acquire) >= previous; },
        det::Vblanks() + det::VblanksForMs(limit_ms));
  } else {
    const auto until = start + std::chrono::milliseconds(limit_ms);
    while (log.read_published.load(std::memory_order_acquire) < previous && std::chrono::steady_clock::now() < until) {
      std::this_thread::yield();
    }
  }
  const uint64_t waited =
      uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
  log.pace_us.fetch_add(waited, std::memory_order_relaxed);
  if (log.read_published.load(std::memory_order_acquire) < previous) {
    if (log.pace_timeouts.load(std::memory_order_relaxed) < 1 && log.fallback_logged < 8) {
      REXLOG_WARN("[native][detached] the command thread is behind after {} ms: waiting kind {} (1 idle, 2 memory, 3 "
                  "register) at {:08X} for {:08X} (mask {:08X}, function {})",
                  waited / 1000, det_wait_kind_.load(), det_wait_addr_.load(), det_wait_ref_.load(),
                  det_wait_mask_.load(), det_wait_func_.load());
    }
    log.pace_timeouts.fetch_add(1, std::memory_order_relaxed);
    Half2DetachedFallback(log, 0);
  }
}

// wwe2k14 patch (Session 72): no NATIVE renderer here (VulkanCommandProcessor overrides it).
bool CommandProcessor::RegisterNativeShaderSwap(uint64_t, const uint32_t*, size_t, const uint8_t*, size_t,
                                                const char*, uint32_t) {
  return false;
}
bool CommandProcessor::RegisterNativeVertexSwap(uint64_t, const uint32_t*, size_t, const uint8_t*, size_t, const char*,
                                                uint32_t, uint32_t, uint32_t) {
  return false;
}
bool CommandProcessor::RegisterNativeRestoreShader(uint64_t, uint32_t, uint32_t, const char*) { return false; }
bool CommandProcessor::RegisterNativeOwnCrowd(uint64_t, const uint32_t*, size_t, const uint32_t*, size_t, uint32_t) {
  return false;
}

bool CommandProcessor::half2_detached_wanted() const {
  const Half2Log* log = half2_log_.load(std::memory_order_acquire);
  return log && half2_logging() && REXCVAR_GET(gpu_native_detached) >= 2 &&
         log->fallback_frames.load(std::memory_order_relaxed) == 0;
}

void CommandProcessor::Half2DetachedSwitch(bool on, uint32_t write_position_virtual) {
  Half2Log* const log_ptr = half2_log_.load(std::memory_order_acquire);
  if (!log_ptr) {
    return;
  }
  Half2Log& log = *log_ptr;
  if (on) {
    log.pace_mark = log.pace_mark_older = 0;
    log.detached.store(true, std::memory_order_release);
    log.switches_on.fetch_add(1, std::memory_order_relaxed);
  } else {
    log.drain_until_seq.store(log.next_seq.load(std::memory_order_relaxed), std::memory_order_relaxed);
    log.skip_until.store(memory_->GetPhysicalAddress(write_position_virtual), std::memory_order_relaxed);
    log.detached.store(false, std::memory_order_release);
    log.switches_off.fetch_add(1, std::memory_order_relaxed);
  }
  write_ptr_index_event_->Set();
}

// Executes the records ahead of the ring: while detached, every published record; after a switch back, the records
// written before it. True when anything ran.
bool CommandProcessor::Half2LogDrive() {
  Half2Log* const log_ptr = half2_log_.load(std::memory_order_relaxed);
  if (!log_ptr) {
    return false;
  }
  Half2Log& log = *log_ptr;
  const bool detached = log.detached.load(std::memory_order_acquire);
  const uint64_t drain = log.drain_until_seq.load(std::memory_order_relaxed);
  if (!detached && !drain) {
    return false;
  }
  Half2Log::Exec& x = log.exec;
  constexpr uint64_t kMask = Half2Log::kWords - 1;
  const uint32_t* ring = log.ring.get();
  bool any = false;
  const uint64_t start_ns = NowNs();
  while (log.read != log.write.load(std::memory_order_acquire)) {
    const uint64_t r = log.read;
    const uint32_t kind = ring[r & kMask], pieces = ring[(r + 3) & kMask], words = ring[(r + 4) & kMask];
    const uint64_t seq = uint64_t(ring[(r + 1) & kMask]) | (uint64_t(ring[(r + 2) & kMask]) << 32);
    if (!detached && seq >= drain) {
      break;  // written after the switch back: the ring brings its segment
    }
    const uint32_t begin = pieces ? ring[(r + Half2Log::kHeaderWords) & kMask] : 0;
    const uint64_t payload = r + Half2Log::kHeaderWords + 2 * uint64_t(pieces);
    if (log.self_last_seq && seq != log.self_last_seq && seq != log.self_last_seq + 1) {
      Half2DetachedFallback(log, 1);  // a hole: a loss while detached
    }
    log.self_last_seq = seq;
    if (!pieces || !words) {
      ++x.split_records;
      Half2DetachedFallback(log, 2);  // no payload: the frame cannot be drawn from the log
    } else {
      log.scratch.resize(words);
      const uint64_t at = payload & kMask;
      const uint64_t first = std::min<uint64_t>(words, Half2Log::kWords - at);
      std::memcpy(log.scratch.data(), ring + at, first * 4);
      if (first < words) std::memcpy(log.scratch.data() + first, ring, (words - first) * 4);
    }
    log.read = payload + words;
    log.read_published.store(log.read, std::memory_order_release);
    if (pieces && words) {
      // (Review: as inside a ring segment - buffers the record calls run at depth 2, never through the segment
      // executor.)
      ++native_ib_depth_;
      const bool ran = Half2LogRun(log.scratch.data(), begin, words);
      --native_ib_depth_;
      if (!ran) {
        if (log.fallback_logged < 8) {
          std::string head;
          for (uint32_t i = 0; i < words && i < 8; ++i) head += fmt::format(" {:08X}", std::byteswap(log.scratch[i]));
          REXLOG_WARN("[native][detached] a packet failed in record {} (kind {}, {:08X}+{} words, first words{}; "
                      "last {:08X})",
                      seq, kind, begin, words, head, std::byteswap(log.scratch[words - 1]));
        }
        ++x.failed;
        Half2DetachedFallback(log, 3);
      }
      x.self_words += words;
      ++x.self_records;
      ++x.records_by_kind[std::min<uint32_t>(kind, 5)];
    }
    any = true;
  }
  // wwe2k14 patch (Session 77, review of 7f8e1f9): also when nothing ran and later records wait (was "read == write ||
  // any"): the loop then stopped at a record written after the switch, so the drain is over (next >= drain below).
  // Before, the drain stayed set until the next ring segment: the command processor's idle loop spun on
  // Half2LogHasSelfWork, and under det_serial its drain check (DetDrainedFor 1) never passed - a hang if the render
  // thread blocked before its next kick.
  if (!detached && drain) {
    // Drained up to the switch (the loop stops at the first later record).
    const uint64_t r = log.read;
    const bool more = r != log.write.load(std::memory_order_acquire);
    const uint64_t next = more ? (uint64_t(ring[(r + 1) & kMask]) | (uint64_t(ring[(r + 2) & kMask]) << 32)) : drain;
    if (next >= drain) {
      log.drain_until_seq.store(0, std::memory_order_relaxed);
      log.self_last_seq = 0;
    }
  }
  if (any) x.self_ns += NowNs() - start_ns;
  return any;
}

bool CommandProcessor::Half2LogHasSelfWork() const {
  const Half2Log* log = half2_log_.load(std::memory_order_relaxed);
  return log && (log->detached.load(std::memory_order_acquire) || log->drain_until_seq.load(std::memory_order_relaxed)) &&
         log->read != log->write.load(std::memory_order_acquire);
}

// wwe2k14 patch (Session 77): the same test from another thread (det_serial's drain check): read_published is the
// consumer's position as an atomic (read itself belongs to the command processor thread).
bool CommandProcessor::Half2LogHasPublishedSelfWork() const {
  const Half2Log* log = half2_log_.load(std::memory_order_acquire);
  return log && (log->detached.load(std::memory_order_acquire) || log->drain_until_seq.load(std::memory_order_relaxed)) &&
         log->read_published.load(std::memory_order_acquire) != log->write.load(std::memory_order_acquire);
}

void CommandProcessor::Half2DetachedFallback(Half2Log& log, uint32_t reason) {
  if (reason < 4) ++log.exec.fallbacks[reason];
  // From the next present the game turns the no-GPU mode off; tried again after 300 frames (hysteresis).
  if (log.fallback_frames.exchange(300, std::memory_order_relaxed) == 0 && log.fallback_logged < 8) {
    ++log.fallback_logged;
    REXLOG_WARN("[native][detached] falling back to the ring (reason {}: {})", reason,
                reason == 0   ? "the command thread is 2 presents behind"
                : reason == 1 ? "a sequence hole"
                : reason == 2 ? "a record without payload"
                              : "a packet failed");
  }
}

void CommandProcessor::Half2LogFrame() {
  if (half2_logging() && !half2_log_storage_) {
    half2_log_storage_ = std::make_unique<Half2Log>();
    half2_log_.store(half2_log_storage_.get(), std::memory_order_release);
    REXLOG_INFO("[native][half2log] on: library calls' packets logged, register state rebuilt from the log{}",
                REXCVAR_GET(gpu_native_half2_log_verify) ? " and compared with the register file" : "");
  }
  if (!half2_log_storage_) {
    return;
  }
  Half2Log& log = *half2_log_storage_;
  // (Review 2) Logging switched back on: the records left from before belong to frames already executed.
  const bool logging = half2_logging();
  if (logging && !log.was_logging) {
    log.open = false;
    log.read = log.write.load(std::memory_order_acquire);
    log.read_published.store(log.read, std::memory_order_release);
    log.seeded = false;
    log.last_seq = 0;
  }
  log.was_logging = logging;
  // (Review) gpu_native_detached changed: the verify consumer (Half2LogAt) and the executors do not share an open
  // record - start both from the current position.
  if (const int32_t detached_mode = REXCVAR_GET(gpu_native_detached); detached_mode != log.last_detached_mode) {
    log.last_detached_mode = detached_mode;
    if (!log.detached.load(std::memory_order_relaxed) && !log.drain_until_seq.load(std::memory_order_relaxed)) {
      log.open = false;
      log.read = log.write.load(std::memory_order_acquire);
      log.read_published.store(log.read, std::memory_order_release);
      log.seeded = false;
      log.last_seq = 0;
      log.blocked_segments = 0;
    }
  }
  if (const uint32_t f = log.fallback_frames.load(std::memory_order_relaxed)) {
    log.fallback_frames.store(f - 1, std::memory_order_relaxed);
  }
  log.exec.detached_frames += log.detached.load(std::memory_order_relaxed) ? 1 : 0;
  log.exec.swaps += half2_swaps_ - log.swaps_seen;
  log.swaps_seen = half2_swaps_;
  const uint32_t every = uint32_t(std::max(1, REXCVAR_GET(gpu_native_half2_log_frames)));
  if (++log.window_frames < every) {
    return;
  }
  const double n = double(log.window_frames);
  const Half2Log::Census& c = log.window;
  std::vector<std::pair<uint32_t, uint32_t>> regs(log.diff_registers.begin(), log.diff_registers.end());
  std::sort(regs.begin(), regs.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  std::string top;
  for (size_t i = 0; i < regs.size() && i < 12; ++i) top += fmt::format(" {:04X}x{}", regs[i].first, regs[i].second);
  std::vector<std::pair<uint32_t, uint32_t>> ops;
  for (uint32_t op = 0; op < 128; ++op) {
    if (log.opcodes[op]) ops.push_back({op, log.opcodes[op]});
  }
  std::sort(ops.begin(), ops.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  std::string op_list;
  for (const auto& [op, count] : ops) op_list += fmt::format(" {:02X}x{:.1f}", op, count / n);
  const uint64_t records = log.records.exchange(0), words = log.payload_words.exchange(0);
  REXLOG_INFO("[native][half2log] {} frames: records {:.1f} a frame ({:.1f} KB), nested calls {:.1f}, gaps {:.1f} "
              "({:.1f} words), kicks inside calls {:.1f} (records in pieces {:.1f}) / outside {:.1f}, write pointer jumps {}, "
              "split {}, ring full {}, copy {:.4f} ms a frame (render thread); matched draws {:.1f} (via an indirect buffer {:.1f}, not "
              "ready {}, record gaps {}, out of range {}, passed {}), seeds {}, apply {:.4f} ms a frame (command thread, "
              "{:.1f} packets, {:.1f} predicated off, {:.1f} indirect buffers: followed {:.1f}, too long {:.1f}); compared {:.1f} draws ({:.0f} values), "
              "with a difference {:.1f} ({:.1f} values), shaders differ {}, sequence holes {}; registers that differed:{}; type-3 opcodes "
              "a frame:{}",
              log.window_frames, records / n, words * 4 / 1024.0 / n, log.nested.exchange(0) / n,
              log.gap_records.exchange(0) / n, log.gap_words.exchange(0) / n, log.kicks.exchange(0) / n,
              log.multi_piece.exchange(0) / n, log.kicks_outside.exchange(0) / n, log.jumps.exchange(0),
              log.splits.exchange(0),
              log.ring_full.exchange(0), log.copy_ns.exchange(0) / 1e6 / n, c.at_calls / n, c.via_ib / n, c.not_ready,
              c.seq_gaps, c.out_of_range, c.behind, c.seeds, c.apply_ns / 1e6 / n, c.applied_packets / n,
              c.predicated_skipped / n, c.ib_packets / n, c.ib_followed / n, c.ib_not_followed / n, c.compared / n, c.compared_values / n, c.differing_draws / n,
              double(c.differing_values) / n, c.shader_differs, c.seq_holes, top.empty() ? " none" : top,
              op_list.empty() ? " none" : op_list);
  if (REXCVAR_GET(gpu_native_detached) >= 1 || log.exec.segments || log.exec.self_records) {
    const Half2Log::Exec& x = log.exec;
    const double total = double(x.log_words + x.memory_head + x.memory_between + x.memory_tail);
    REXLOG_INFO("[native][detached] {} frames: segments {:.1f} a frame ({:.1f} with no record), words {:.0f} a frame - "
                "from the log {:.2f}%, from guest memory: before the first record {:.1f}, between records {:.1f}, after "
                "the last {:.1f} a frame; records {:.1f} a frame (gaps {:.1f}, indexed draws {:.1f}, inline vertices "
                "{:.1f}, clears {:.1f}, resolves {:.1f}, presents {:.1f}), without payload {}, stale {}; waits {} "
                "(timeouts {}), failed {}; {:.3f} ms a frame; detached (no-GPU mode) {:.0f}% of the frames, records run "
                "by the command thread itself {:.1f} a frame ({:.0f} words, {:.3f} ms), words skipped after a switch "
                "back {} (not in the first segment {}), switches on {} / off {}, fallbacks: sequence hole {}, no payload {}, failed packet {}, ring "
                "full {}, behind {} (pacing waits {:.3f} ms a frame); swaps {:.2f} a frame",
                log.window_frames, x.segments / n, x.not_logged_segments / n, total / n,
                total ? 100.0 * x.log_words / total : 0.0, x.memory_head / n, x.memory_between / n, x.memory_tail / n,
                x.records / n, x.records_by_kind[0] / n, x.records_by_kind[1] / n, x.records_by_kind[2] / n,
                x.records_by_kind[3] / n, x.records_by_kind[4] / n, x.records_by_kind[5] / n, x.split_records,
                x.stale_records, x.waits, x.wait_timeouts, x.failed, x.exec_ns / 1e6 / n, 100.0 * x.detached_frames / n,
                x.self_records / n, x.self_words / n, x.self_ns / 1e6 / n, x.skipped_words, x.skip_missed,
                log.switches_on.exchange(0), log.switches_off.exchange(0), x.fallbacks[1], x.fallbacks[2],
                x.fallbacks[3], log.ring_waits_full.exchange(0), log.pace_timeouts.exchange(0),
                log.pace_us.exchange(0) / 1e3 / n, x.swaps / n);
    log.exec = {};
  }
  log.window = {};
  log.window_frames = 0;
  log.opcodes.fill(0);
  log.diff_registers.clear();
  log.diff_logged.clear();
  log.lines_logged = 0;
}

}  // namespace rex::graphics
