// wwe2k14 patch (Session 48, HALF2): the game's own D3D library draws, matched to the command processor's draw
// packets and compared with its register file (N5b / N6a; docs\native\HALF2_MAP.md in the game repository).
//
// gpu_native_half2 1: the game's hooks on its draw writers queue each library draw (the guest device's shadow
// registers and float-constant hashes, and where its packets went); at each draw packet the command processor takes
// the library draw whose packets hold the packet's address and compares every queued register with its register
// file. One [native][half2] line every gpu_native_half2_log_frames frames: draws matched, packets no library draw
// wrote (the clear and resolve paths, recorded buffers), library draws passed over, mismatching registers.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#include <fmt/format.h>

#include <rex/cvar.h>
#include <rex/graphics/d3d_noring.h>  // wwe2k14 patch (Session 87, D3D layer stage 2c)
#include <rex/graphics/command_processor.h>
#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/xenos.h>
#include <rex/logging.h>
#include <rex/memory.h>

#include "half2_state.h"

REXCVAR_DEFINE_INT32(gpu_native_half2, 0, "GPU/Native",
                     "Session 48 (Half 2, docs\\native\\HALF2_MAP.md): the game's own D3D library draws. 0 off; 1 "
                     "compare - every draw packet matched to the library draw that wrote it and the guest device's "
                     "shadow registers compared with the register file ([native][half2] lines). Diagnostic, changes "
                     "nothing drawn. 2 attribute (Session 50) - every draw packet matched to its library draw only (the "
                     "hooks copy no state), for gpu_native_lean_draws. 3 log (Session 51) - as 2, plus each library "
                     "call's own packets copied into a state log the command processor thread rebuilds the register "
                     "state from ([native][half2log] lines; gpu_native_half2_log_verify compares it). Takes effect at "
                     "the next frame");
REXCVAR_DEFINE_BOOL(gpu_native_half2_skip_gpr_draws, false, "GPU/Native",
                    "Session 48 (Half 2): skip the D3D library's GPR-allocation flush draws - an indirect buffer of "
                    "exactly 24 one-vertex point draws (DRAW_INDX_2 0x00010081), which 0x82CB0F58 calls after every "
                    "shader GPR split change (~40 a match frame, ~17% of the command thread's busy time). Off by default");
REXCVAR_DEFINE_INT32(gpu_native_half2_log_frames, 60, "GPU/Native",
                     "gpu_native_half2: frames per [native][half2] census line");
REXCVAR_DEFINE_INT32(gpu_native_half2_log_mismatches, 40, "GPU/Native",
                     "gpu_native_half2: the first N mismatching registers logged one by one (register, both values)");

namespace rex::graphics {

namespace {

void CopyDraw(Half2LibraryDraw& to, const Half2LibraryDraw& from) {
  to.kind = from.kind;
  to.packets_begin = from.packets_begin;
  to.packets_end = from.packets_end;
  to.device = from.device;
  to.run_count = std::min(from.run_count, Half2LibraryDraw::kMaxRuns);
  to.value_count = std::min(from.value_count, Half2LibraryDraw::kMaxValues);
  std::memcpy(to.runs, from.runs, sizeof(Half2Run) * to.run_count);
  std::memcpy(to.values, from.values, sizeof(uint32_t) * to.value_count);
  std::memcpy(to.float_hash, from.float_hash, sizeof(from.float_hash));
  to.prim = from.prim;
  to.index_count = from.index_count;
  to.index_address = from.index_address;
  to.index32 = from.index32;
  to.vs_ucode[0] = from.vs_ucode[0];
  to.vs_ucode[1] = from.vs_ucode[1];
  to.ps_ucode = from.ps_ucode;
  to.vs_inline_ucode = from.vs_inline_ucode;
  to.log_seq = from.log_seq;
}

bool Holds(const Half2LibraryDraw& d, uint32_t address) {
  return address >= d.packets_begin && d.packets_end > d.packets_begin && address < d.packets_end;
}

}  // namespace

Half2LibraryDraw* CommandProcessor::Half2BeginLibraryDraw(uint32_t kind, uint32_t packets_begin_virtual) {
  if (!half2_on() || !half2_) {
    return nullptr;
  }
  Half2State& s = *half2_;
  std::lock_guard lock(s.producer_mutex);
  const uint64_t tail = s.tail.load(std::memory_order_relaxed);
  // The command processor's head is read without its lock: a stale (smaller) value only refuses early.
  if (tail - s.head_published.load(std::memory_order_acquire) >= Half2State::kSlots) {
    s.dropped.fetch_add(1, std::memory_order_relaxed);
    Half2LogRefused();  // wwe2k14 patch (Session 51, STATE LOG): this call's packets are not in the log
    return nullptr;
  }
  Half2LibraryDraw& d = s.slots[tail % Half2State::kSlots];
  d.kind = kind;
  d.queued_frame = s.frames_seen.load(std::memory_order_relaxed);
  d.packets_begin = memory_->GetPhysicalAddress(packets_begin_virtual);
  d.packets_end = 0;
  d.run_count = 0;
  d.value_count = 0;
  d.prim = d.index_count = d.index_address = d.ps_ucode = 0;
  d.vs_ucode[0] = d.vs_ucode[1] = 0;
  d.vs_inline_ucode = 0;
  d.index32 = false;
  d.has_ps_c32 = false;  // wwe2k14 patch (Session 79)
  d.log_seq = 0;
  if (half2_logging()) d.log_seq = Half2LogReserve(kind, d.packets_begin);  // wwe2k14 patch (Session 51, STATE LOG)
  d.state.store(1, std::memory_order_release);
  s.tail.store(tail + 1, std::memory_order_release);
  return &d;
}

void CommandProcessor::Half2EndLibraryDraw(Half2LibraryDraw* draw, uint32_t packets_end_virtual) {
  if (!draw) {
    return;
  }
  draw->packets_end = memory_->GetPhysicalAddress(packets_end_virtual);
  if (draw->log_seq) Half2LogAppend(draw->packets_end);  // wwe2k14 patch (Session 51, STATE LOG)
  auto physical = [&](uint32_t& a) {
    const uint32_t p = a ? memory_->GetPhysicalAddress(a) : UINT32_MAX;
    a = p != UINT32_MAX ? (p & 0x1FFFFFFF) : 0;  // GetPhysicalAddress: UINT32_MAX outside the physical heaps
  };
  physical(draw->index_address);
  physical(draw->vs_ucode[0]);
  physical(draw->vs_ucode[1]);
  physical(draw->ps_ucode);
  physical(draw->vs_inline_ucode);
  // wwe2k14 patch (Session 79, NF-D look-ahead): the hook copied c32 for every draw; kept for the watched shader.
  const uint32_t watch = half2_watch_ps_ucode_.load(std::memory_order_relaxed);
  draw->has_ps_c32 = watch && draw->ps_ucode == watch;
  draw->state.store(2, std::memory_order_release);
}

// wwe2k14 patch (Session 79, NF-D look-ahead; see command_processor.h).
void CommandProcessor::Half2WatchLastMatchedPixelShader() {
  if (half2_ && half2_->last_valid && half2_->last.ps_ucode) {
    half2_watch_ps_ucode_.store(half2_->last.ps_ucode, std::memory_order_relaxed);
    // The look-ahead's exactness check: the c32 the library call carried = the register the draw reads.
    if (half2_->last.has_ps_c32) {
      const uint32_t reg = register_file_->values[XE_GPU_REG_SHADER_CONSTANT_000_X + 4 * (256 + 32)];
      ++(reg == half2_->last.ps_c32[0] ? nf_census_.glare_check_ok : nf_census_.glare_check_bad);
    }
  }
}

bool CommandProcessor::Half2FindWatchedAhead(uint32_t& c32x) {
  if (!half2_ || !nf_seg_ptr_) return false;
  Half2State& s = *half2_;
  const uint32_t seg_begin = nf_seg_ptr_ & 0x1FFFFFFF, seg_end = seg_begin + nf_seg_words_ * 4;
  const uint64_t tail = s.tail.load(std::memory_order_acquire);
  const uint64_t last = std::min(tail, s.head + Half2State::kSlots);
  for (uint64_t i = s.head; i < last; ++i) {
    const Half2LibraryDraw& d = s.slots[i % Half2State::kSlots];
    if (d.state.load(std::memory_order_acquire) != 2) break;  // not published yet: nothing further is either
    const uint32_t at = d.packets_begin & 0x1FFFFFFF;
    if (at < seg_begin || at >= seg_end) continue;  // another segment's
    if (d.has_ps_c32) {
      c32x = d.ps_c32[0];
      return true;
    }
  }
  return false;
}

void CommandProcessor::Half2Compare(const Half2LibraryDraw& d, const Half2PacketDraw& packet, bool first_chunk) {
  // The library's clears and resolves (kinds 3, 4) are attributed only: their own inline shaders and immediate
  // register writes are not in the device's shadow.
  if (d.kind >= 3) {
    return;
  }
  Half2State& s = *half2_;
  const RegisterFile& rf = *register_file_;
  // What the bound shaders read (both analysed): float constants by group of 4 (VS 0x4000 + 4 x i, PS 0x4400 +
  // 4 x i: SQ_VS_CONST / SQ_PS_CONST as 2K14 sets them), fetch constants, bool and loop constants. Registers outside
  // the constants are always compared. Without both shaders analysed everything counts as read.
  bool analysed = false;
  uint64_t used_groups[2] = {};      // 128 float groups
  uint32_t used_fetch[6] = {};       // 192 fetch dwords 0x4800-0x48BF
  uint64_t used_bool_loop = 0;       // 40 dwords 0x4900-0x4927
  const Shader* shaders[2] = {active_vertex_shader_, active_pixel_shader_};
  if (shaders[0] && shaders[1] && shaders[0]->is_ucode_analyzed() && shaders[1]->is_ucode_analyzed()) {
    analysed = true;
    for (uint32_t k = 0; k < 2; ++k) {
      const Shader::ConstantRegisterMap& map = shaders[k]->constant_register_map();
      for (uint32_t c = 0; c < 256; ++c) {
        if (map.float_dynamic_addressing || (map.float_bitmap[c >> 6] >> (c & 63) & 1)) {
          const uint32_t group = 64 * k + c / 4;
          used_groups[group >> 6] |= uint64_t(1) << (group & 63);
        }
      }
      for (uint32_t b = 0; b < 8; ++b) {
        if (map.bool_bitmap[b]) used_bool_loop |= uint64_t(1) << b;
      }
      for (uint32_t l = 0; l < 32; ++l) {
        if (map.loop_bitmap >> l & 1) used_bool_loop |= uint64_t(1) << (8 + l);
      }
      for (uint32_t v = 0; v < 96; ++v) {
        if (map.vertex_fetch_bitmap[v >> 5] >> (v & 31) & 1) {
          for (uint32_t w = 2 * v; w < 2 * v + 2; ++w) used_fetch[w >> 5] |= 1u << (w & 31);
        }
      }
      for (const Shader::TextureBinding& t : shaders[k]->texture_bindings()) {
        for (uint32_t w = 6 * t.fetch_constant; w < 6 * t.fetch_constant + 6 && w < 192; ++w) {
          used_fetch[w >> 5] |= 1u << (w & 31);
        }
      }
    }
  }
  auto is_used = [&](uint32_t reg) -> bool {
    if (!analysed) return true;
    if (reg >= 0x4800 && reg < 0x48C0) return used_fetch[(reg - 0x4800) >> 5] >> ((reg - 0x4800) & 31) & 1;
    if (reg >= 0x4900 && reg < 0x4928) return used_bool_loop >> (reg - 0x4900) & 1;
    if (reg >= 0x4000 && reg < 0x4800) {
      const uint32_t group = (reg - 0x4000) / 16;
      return used_groups[group >> 6] >> (group & 63) & 1;
    }
    return true;
  };
  uint32_t differ = 0, unused = 0;
  auto note = [&](uint32_t reg, uint32_t cp_value, uint32_t shadow_value, bool hash) {
    if (!is_used(reg)) {
      ++unused;
      ++s.unused_registers[reg];
      return;
    }
    ++differ;
    ++s.mismatch_registers[reg];
    if (s.mismatches_logged < uint32_t(std::max(0, REXCVAR_GET(gpu_native_half2_log_mismatches))) &&
        s.logged_per_register[reg]++ < 3) {
      ++s.mismatches_logged;
      REXLOG_INFO("[native][half2] mismatch frame {} draw {} (kind {}, packets {:08X}-{:08X}, VS {:016X} PS "
                  "{:016X}): {} {:04X} command processor {:08X}, device shadow {:08X}",
                  s.frames_seen.load(), s.frame.cp_draws, d.kind, d.packets_begin, d.packets_end,
                  shaders[0] ? shaders[0]->ucode_data_hash() : 0, shaders[1] ? shaders[1]->ucode_data_hash() : 0,
                  hash ? "float group at" : "register", reg, cp_value, shadow_value);
      if (hash) {
        std::string cp_words;
        for (uint32_t i = 0; i < 16; ++i) cp_words += fmt::format(" {:08X}", rf.values[reg + i]);
        REXLOG_INFO("[native][half2]   command processor's {:04X}..{:04X}:{}", reg, reg + 15, cp_words);
      }
    }
  };
  uint32_t v = 0;
  for (uint32_t r = 0; r < d.run_count; ++r) {
    const Half2Run run = d.runs[r];
    for (uint32_t i = 0; i < run.count && v < d.value_count; ++i, ++v) {
      const uint32_t reg = uint32_t(run.first_register) + i;
      if (reg < RegisterFile::kRegisterCount && rf.values[reg] != d.values[v]) {
        note(reg, rf.values[reg], d.values[v], false);
      }
    }
  }
  for (uint32_t g = 0; g < Half2LibraryDraw::kFloatGroups; ++g) {
    const uint32_t reg = 0x4000 + 16 * g;
    const uint64_t h = Half2HashGroup(&rf.values[reg]);
    if (h != d.float_hash[g]) {
      note(reg, uint32_t(h), uint32_t(d.float_hash[g]), true);
    }
  }
  // The draw description (kind 1, the indexed writer): primitive, the first chunk's index count and base, the
  // shaders' microcode. Reported as pseudo-registers: 0x10000 prim, 0x10001 count, 0x10002 index base, 0x10003 index
  // size, 0x10004 vertex shader, 0x10005 pixel shader.
  if (d.kind == 1) {
    auto field = [&](uint32_t id, uint32_t cp_value, uint32_t lib_value) {
      ++s.frame.fields_compared;
      if (cp_value != lib_value) note(id, cp_value, lib_value, false);
    };
    field(0x10000, packet.prim, d.prim);
    if (first_chunk) {
      field(0x10001, packet.count, std::min<uint32_t>(d.index_count, 0xFFFF));
      field(0x10002, packet.index_base, d.index_address);
    }
    field(0x10003, packet.index32, d.index32);
    bool vs_known =
        half2_ucode_[0] && (half2_ucode_[0] == d.vs_ucode[0] || half2_ucode_[0] == d.vs_ucode[1]);
    // An inline load (IM_LOAD_IMMEDIATE, ~200 draws a match frame, run 3294) has no address: compare the microcode
    // with the object's variants in guest memory (the shader's ucode_data is in host byte order).
    // The library's own patched copy (0x82CBE210): the inline load's microcode starts 12 bytes after its header.
    if (!half2_ucode_[0] && d.vs_inline_ucode && half2_inline_vs_at_ + 12 == d.vs_inline_ucode) {
      vs_known = true;
      ++s.frame.inline_vs_same;
    }
    if (!vs_known && !half2_ucode_[0] && shaders[0]) {
      size_t n = shaders[0]->ucode_dword_count();
      for (uint32_t k = 0; k < 2 && !vs_known; ++k) {
        if (!d.vs_ucode[k] || d.vs_ucode[k] + 4 * uint64_t(n) > 0x20000000) continue;
        const uint32_t* guest = memory_->TranslatePhysical<const uint32_t*>(d.vs_ucode[k]);
        bool same = n != 0;
        for (size_t i = 0; i < n && same; ++i) same = std::byteswap(guest[i]) == shaders[0]->ucode_dwords()[i];
        if (same) {
          vs_known = true;
          ++s.frame.inline_vs_same;
        }
      }
      if (!vs_known && s.inline_logged < 4) {
        ++s.inline_logged;
        auto words = [&](uint32_t phys) {
          std::string w;
          if (!phys || phys + 24 > 0x20000000) return w;
          const uint32_t* g = memory_->TranslatePhysical<const uint32_t*>(phys);
          for (int i = 0; i < 6; ++i) w += fmt::format(" {:08X}", std::byteswap(g[i]));
          return w;
        };
        std::string cp_words;
        for (size_t i = 0; i < n && i < 6; ++i) cp_words += fmt::format(" {:08X}", shaders[0]->ucode_dwords()[i]);
        REXLOG_INFO("[native][half2] inline vertex shader {:016X} ({} dwords):{}; the device's variants {:08X}:{} / "
                    "{:08X}:{}; loaded by the packet at {:08X} (called from {:08X}), this library draw's packets "
                    "{:08X}-{:08X}",
                    shaders[0]->ucode_data_hash(), n, cp_words, d.vs_ucode[0], words(d.vs_ucode[0]), d.vs_ucode[1],
                    words(d.vs_ucode[1]), half2_inline_vs_at_, half2_inline_vs_caller_, d.packets_begin,
                    d.packets_end);
      }
    }
    field(0x10004, half2_ucode_[0], vs_known ? half2_ucode_[0] : d.vs_ucode[0]);
    field(0x10005, half2_ucode_[1], d.ps_ucode);
  }
  s.frame.unanalysed += analysed ? 0 : 1;
  s.frame.unused_values += unused;
  if (differ) {
    ++s.frame.mismatched_draws;
    s.frame.mismatched_values += differ;
  }
}

void CommandProcessor::Half2OnCpDraw(uint32_t packet_physical, bool skipped, const Half2PacketDraw& packet) {
  if (!half2_ || !half2_on()) {
    return;
  }
  Half2State& s = *half2_;
  ++s.frame.cp_draws;
  // wwe2k14 patch (Session 87, D3D layer stage 2c): a draw of a record token run from the game's host queue knows its library
  // kind (a 0x7C token IS a library indexed draw, a 0x7E token an inline draw): no matching by packet address. The library
  // draws whose packets went into records have an empty packet range and are dropped from the head of the queue here.
  if (const uint32_t record_kind = D3dNoringRecordKind()) {
    Half2DropDeadLibraryDraws();
    if (skipped) {
      ++s.frame.skipped_packets;
      return;
    }
    ++s.frame.matched;
    if (record_kind < 5) ++s.frame.kind_matched[record_kind];
    half2_matched_kind_ = record_kind < 5 ? record_kind : 0;
    return;
  }
  Half2DropDeadLibraryDraws();
  // A draw inside a nested indirect buffer (depth 2: a buffer called from a segment) is attributed through the
  // INDIRECT_BUFFER packet that called it, when no library draw wrote the draw packet itself.
  const uint32_t caller = native_ib_depth_ >= 2 ? half2_ib_caller_ : 0;
  // Session 50 (attribution mode): the rest of a nested buffer whose earlier draw no library draw held is not looked
  // up again (the GPR-flush buffers' 24 point draws each scanned the queue: ~0.05 ms a frame, profile 3403). A miss
  // only hands the draw to IssueDraw.
  if (!half2_full() && caller && caller == half2_unmatched_caller_) {
    ++s.frame.unmatched_ib;
    return;
  }
  const uint64_t queued_tail = s.tail.load(std::memory_order_acquire);
  const Half2LibraryDraw* front = s.head < queued_tail ? &s.slots[s.head % Half2State::kSlots] : nullptr;
  const bool front_holds = front && front->state.load(std::memory_order_acquire) == 2 &&
                           (Holds(*front, packet_physical) || (caller && Holds(*front, caller)));
  if (!front_holds && s.last_valid && (Holds(s.last, packet_physical) || (caller && Holds(s.last, caller)))) {
    if (skipped) {
      ++s.frame.skipped_packets;
      return;
    }
    ++s.frame.matched;
    ++(Holds(s.last, packet_physical) ? s.frame.replays : s.frame.via_ib);
    half2_matched_kind_ = s.last.kind;  // wwe2k14 patch (Session 50, LEAN DRAWS)
    if (s.last.log_seq) {  // wwe2k14 patch (Session 51, STATE LOG)
      const bool in_ib = !Holds(s.last, packet_physical);
      Half2LogAt(s.last.log_seq, in_ib ? caller : packet_physical, s.last.kind, in_ib);
    }
    if (half2_full()) Half2Compare(s.last, packet, false);
    return;
  }
  // The library draw whose packets hold this one: the queued draws in order, a few ahead (draws the command
  // processor never executed are passed over). Every draw looked at must be published; the game is at most inside
  // one writer call, so a reserved entry is waited for briefly.
  constexpr uint32_t kLookAhead = 16;
  const uint64_t tail = s.tail.load(std::memory_order_acquire);
  int64_t found = -1;
  bool found_via_ib = false;
  for (uint64_t i = s.head; i < tail && i < s.head + kLookAhead; ++i) {
    Half2LibraryDraw& d = s.slots[i % Half2State::kSlots];
    if (d.state.load(std::memory_order_acquire) != 2) {
      ++s.frame.waits;
      // At most a few microseconds: the writer may be blocked on ring space the command processor frees.
      const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(50);
      while (d.state.load(std::memory_order_acquire) != 2 && std::chrono::steady_clock::now() < until) {
        std::this_thread::yield();
      }
      if (d.state.load(std::memory_order_acquire) != 2) {
        ++s.frame.wait_timeouts;
        break;
      }
    }
    if (Holds(d, packet_physical) || (caller && Holds(d, caller))) {
      found = int64_t(i);
      found_via_ib = !Holds(d, packet_physical);
      break;
    }
  }
  if (found < 0) {
    ++(native_ib_depth_ ? s.frame.unmatched_ib : s.frame.unmatched_ring);
    half2_unmatched_caller_ = caller;  // (Session 50)
    if (!half2_full()) return;         // (Session 50: the region census and the log lines are the compare mode's)
    ++s.unmatched_regions[packet_physical >> 16];
    // A few per census window, nested buffers (the frame's point draws) left out.
    if (native_ib_depth_ < 2 && s.unmatched_logged < 4) {
      ++s.unmatched_logged;
      const Half2LibraryDraw* next = s.head < tail ? &s.slots[s.head % Half2State::kSlots] : nullptr;
      const uint8_t* words = memory_->TranslatePhysical(packet_physical);
      uint32_t w0, w1;
      std::memcpy(&w0, words, 4);
      std::memcpy(&w1, words + 4, 4);
      REXLOG_INFO("[native][half2] frame {} draw {}: packet {:08X} ({:08X} {:08X}, indirect buffer depth {}, called "
                  "from {:08X}) from no library draw; last matched {:08X}-{:08X}, next queued {:08X}-{:08X}",
                  s.frames_seen.load(), s.frame.cp_draws, packet_physical, std::byteswap(w0), std::byteswap(w1),
                  native_ib_depth_, half2_ib_caller_, s.last_valid ? s.last.packets_begin : 0,
                  s.last_valid ? s.last.packets_end : 0, next ? next->packets_begin : 0, next ? next->packets_end : 0);
    }
    return;
  }
  for (uint64_t i = s.head; i < uint64_t(found); ++i) {
    s.slots[i % Half2State::kSlots].state.store(0, std::memory_order_relaxed);
    ++s.frame.library_skipped;
  }
  Half2LibraryDraw& d = s.slots[uint64_t(found) % Half2State::kSlots];
  half2_unmatched_caller_ = 0;  // (Session 50)
  if (half2_full()) {
    CopyDraw(s.last, d);
  } else {  // wwe2k14 patch (Session 50, LEAN DRAWS): attribution only (gpu_native_half2 2): where its packets are
    s.last.kind = d.kind;
    s.last.packets_begin = d.packets_begin;
    s.last.packets_end = d.packets_end;
    s.last.device = d.device;
    s.last.log_seq = d.log_seq;  // (Session 51, STATE LOG)
    s.last.ps_ucode = d.ps_ucode;  // (Session 79: 0 unless nf_lookahead - the look-ahead's watch is learned from it)
    s.last.has_ps_c32 = d.has_ps_c32;  // (and checked against the registers at the combine's draw)
    std::copy(std::begin(d.ps_c32), std::end(d.ps_c32), std::begin(s.last.ps_c32));
  }
  s.last_valid = true;
  d.state.store(0, std::memory_order_relaxed);
  // The producer reads only head_published (to refuse when full): no lock here (Session 50: the mutex a matched draw
  // was ~0.06 ms a frame of the command thread, profile 3397).
  s.head = uint64_t(found) + 1;
  s.head_published.store(s.head, std::memory_order_release);
  if (skipped) {
    ++s.frame.skipped_packets;
    return;
  }
  ++s.frame.matched;
  s.frame.via_ib += found_via_ib ? 1 : 0;
  if (s.last.kind < 5) {
    ++s.frame.kind_matched[s.last.kind];
  }
  half2_matched_kind_ = s.last.kind;  // wwe2k14 patch (Session 50, LEAN DRAWS)
  if (s.last.log_seq) {  // wwe2k14 patch (Session 51, STATE LOG)
    Half2LogAt(s.last.log_seq, found_via_ib ? caller : packet_physical, s.last.kind, found_via_ib);
  }
  if (half2_full()) Half2Compare(s.last, packet, true);
}

void CommandProcessor::Half2DropDeadLibraryDraws() {
  if (!half2_) return;
  Half2State& s = *half2_;
  for (;;) {
    if (s.head >= s.tail.load(std::memory_order_acquire)) return;
    Half2LibraryDraw& d = s.slots[s.head % Half2State::kSlots];
    if (d.state.load(std::memory_order_acquire) != 2) return;  // (not published yet: the producer is inside the writer)
    if (d.packets_end > d.packets_begin) return;               // packets exist: a stray packet may still match it
    d.state.store(0, std::memory_order_relaxed);
    ++s.frame.library_skipped;
    ++s.head;
    s.head_published.store(s.head, std::memory_order_release);
  }
}

void CommandProcessor::Half2NoteWrite(uint32_t first, uint32_t count, uint32_t opcode, uint32_t packet_physical,
                                      uint32_t source) {
  if (!half2_full()) {  // (Session 50: the compare mode's diagnostic only)
    return;
  }
  // Diagnostic (runs 3278-3285): the packets writing VS c240-c243 (0x43C0-0x43CF), a group the device's float shadow
  // holds as 0 - shader literal constants (LOAD_ALU_CONSTANT) and the shadow flush; the game's hooks now follow both.
  auto hits = [&](uint32_t lo, uint32_t hi) { return first < hi && first + count > lo; };
  if (!hits(0x43C0, 0x43D0)) {
    return;
  }
  Half2State& s = *half2_;
  ++s.frame.owned_writes;
  const bool in_last = s.last_valid && Holds(s.last, packet_physical);
  s.frame.owned_writes_in_draws += in_last ? 1 : 0;
  if (s.owned_logged < 6) {
    ++s.owned_logged;
    REXLOG_INFO("[native][half2] frame {}: {} writes {:04X}..{:04X} ({} registers) at {:08X} (indirect buffer depth "
                "{}, called from {:08X}){}; last matched library draw {:08X}-{:08X}{}",
                s.frames_seen.load(), opcode == 0 ? "type 0" : opcode == xenos::PM4_SET_CONSTANT ? "SET_CONSTANT"
                                                                                    : "LOAD_ALU_CONSTANT",
                first, first + count - 1, count, packet_physical, native_ib_depth_, half2_ib_caller_,
                source ? fmt::format(" from guest memory {:08X}", source) : std::string(),
                s.last_valid ? s.last.packets_begin : 0, s.last_valid ? s.last.packets_end : 0,
                in_last ? " (inside it)" : "");
  }
}

void CommandProcessor::Half2EndSegment(uint32_t start, uint32_t end) {
  if (!half2_on()) {
    return;
  }
  Half2State& s = *half2_;
  const uint64_t tail = s.tail.load(std::memory_order_acquire);
  while (s.head < tail) {
    Half2LibraryDraw& d = s.slots[s.head % Half2State::kSlots];
    if (d.state.load(std::memory_order_acquire) != 2 || d.packets_begin < start || d.packets_begin >= end) {
      break;
    }
    d.state.store(0, std::memory_order_relaxed);
    ++s.frame.stale;
    std::lock_guard lock(s.producer_mutex);
    ++s.head;
    s.head_published.store(s.head, std::memory_order_release);
  }
}

bool CommandProcessor::Half2SkipGprBuffer(uint32_t list_ptr, uint32_t list_length) {
  if (list_length != 48 || !REXCVAR_GET(gpu_native_half2_skip_gpr_draws)) {
    return false;
  }
  const uint32_t* words = memory_->TranslatePhysical<const uint32_t*>(xenos::GpuToCpu(list_ptr));
  for (uint32_t i = 0; i < 48; i += 2) {
    if (std::byteswap(words[i]) != 0xC0003600u || std::byteswap(words[i + 1]) != 0x00010081u) {
      return false;
    }
  }
  if (!(half2_gpr_buffers_skipped_++ % 4096)) {
    REXLOG_INFO("[native][half2] GPR-flush point-draw buffers skipped: {} so far", half2_gpr_buffers_skipped_);
  }
  return true;
}

void CommandProcessor::Half2FrameBoundary() {
  const int32_t mode = REXCVAR_GET(gpu_native_half2);
  if (mode && !half2_) {
    half2_ = std::make_unique<Half2State>();
    REXLOG_INFO("[native][half2] on (mode {}): library draws matched to draw packets and compared", mode);
  }
  // (Review H1: any change of mode - also 2 -> 1, whose compare needs what the attribution mode never filled.)
  if (mode && mode != half2_mode_.load(std::memory_order_relaxed) && half2_) {
    // Switched on (again): whatever was queued before belongs to frames the command processor did not match.
    Half2State& s = *half2_;
    std::lock_guard lock(s.producer_mutex);
    const uint64_t tail = s.tail.load(std::memory_order_relaxed);
    for (uint64_t i = s.head; i < tail; ++i) s.slots[i % Half2State::kSlots].state.store(0, std::memory_order_relaxed);
    s.head = tail;
    s.head_published.store(s.head, std::memory_order_release);
    s.last_valid = false;
  }
  // (Session 50: also from the attribute mode to the compare mode - the literal constants were not followed.)
  if (mode && mode != half2_mode_.load(std::memory_order_relaxed)) {
    half2_generation_.fetch_add(1, std::memory_order_release);
  }
  half2_mode_.store(mode, std::memory_order_release);
  Half2LogFrame();  // wwe2k14 patch (Session 51, STATE LOG)
  if (!half2_) {
    return;
  }
  Half2State& s = *half2_;
  ++s.frames_seen;
  half2_unmatched_caller_ = 0;  // (Session 50)
  {
    // Entries the command processor never reached in 30 of its frames (a buffer it never executed - the segment
    // drop above only covers ring-level segments): dropped so the queue cannot stay stuck. The game runs only a few
    // frames ahead, so live entries are never this old.
    std::lock_guard lock(s.producer_mutex);
    const uint64_t tail = s.tail.load(std::memory_order_acquire);
    while (s.head < tail) {
      Half2LibraryDraw& d = s.slots[s.head % Half2State::kSlots];
      if (d.state.load(std::memory_order_acquire) != 2 || d.queued_frame + 30 > s.frames_seen.load()) break;
      d.state.store(0, std::memory_order_relaxed);
      ++s.frame.stale;
      ++s.head;
    }
    s.head_published.store(s.head, std::memory_order_release);
  }
  s.frame.library_pending = uint32_t(s.tail.load(std::memory_order_acquire) - s.head);
  Half2State::Census& w = s.window;
  const Half2State::Census& f = s.frame;
  w.cp_draws += f.cp_draws;
  w.matched += f.matched;
  w.replays += f.replays;
  w.via_ib += f.via_ib;
  w.unmatched_ib += f.unmatched_ib;
  w.unmatched_ring += f.unmatched_ring;
  w.library_skipped += f.library_skipped;
  w.library_pending += f.library_pending;
  w.skipped_packets += f.skipped_packets;
  w.stale += f.stale;
  w.waits += f.waits;
  w.wait_timeouts += f.wait_timeouts;
  w.mismatched_draws += f.mismatched_draws;
  w.mismatched_values += f.mismatched_values;
  w.unanalysed += f.unanalysed;
  w.fields_compared += f.fields_compared;
  w.inline_vs_same += f.inline_vs_same;
  w.unused_values += f.unused_values;
  w.owned_writes += f.owned_writes;
  w.owned_writes_in_draws += f.owned_writes_in_draws;
  for (int k = 0; k < 5; ++k) w.kind_matched[k] += f.kind_matched[k];
  s.frame = {};
  const uint32_t every = uint32_t(std::max(1, REXCVAR_GET(gpu_native_half2_log_frames)));
  if (++s.window_frames < every) {
    return;
  }
  std::vector<std::pair<uint32_t, uint32_t>> top(s.mismatch_registers.begin(), s.mismatch_registers.end());
  std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  std::string regs;
  for (size_t i = 0; i < top.size() && i < 12; ++i) {
    regs += fmt::format(" {:04X}x{}", top[i].first, top[i].second);
  }
  std::vector<std::pair<uint32_t, uint32_t>> where(s.unmatched_regions.begin(), s.unmatched_regions.end());
  std::sort(where.begin(), where.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  std::string regions;
  for (size_t i = 0; i < where.size() && i < 8; ++i) {
    regions += fmt::format(" {:04X}xxxx x{}", where[i].first, where[i].second);
  }
  s.unmatched_regions.clear();
  const double n = double(s.window_frames);
  REXLOG_INFO("[native][half2] {} frames: draw packets {:.1f} a frame, matched {:.1f} (indexed {:.1f}, inline {:.1f}, clears "
              "{:.1f}, resolves {:.1f}, "
              "replays {:.1f}, through an indirect buffer call {:.1f}), not from a library draw {:.1f} (in indirect buffers {:.1f}, ring {:.1f}); library "
              "draws passed over {:.1f}, queued at the boundary {:.1f}, never executed in their segment {:.1f}, dropped {}; bin-skipped packets "
              "{:.1f}; waits {} (timeouts {}); matched "
              "draws with a difference the shaders read {:.1f} ({:.1f} values); registers that differed:{}; packets from no library "
              "draw by region:{}",
              s.window_frames, w.cp_draws / n, w.matched / n, w.kind_matched[1] / n, w.kind_matched[2] / n, w.kind_matched[3] / n,
              w.kind_matched[4] / n,
              w.replays / n, w.via_ib / n, (w.unmatched_ib + w.unmatched_ring) / n, w.unmatched_ib / n, w.unmatched_ring / n,
              w.library_skipped / n, w.library_pending / n, w.stale / n, s.dropped.exchange(0), w.skipped_packets / n, w.waits, w.wait_timeouts,
              w.mismatched_draws / n, double(w.mismatched_values) / n, regs.empty() ? " none" : regs,
              regions.empty() ? " none" : regions);
  std::vector<std::pair<uint32_t, uint32_t>> top_unused(s.unused_registers.begin(), s.unused_registers.end());
  std::sort(top_unused.begin(), top_unused.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  std::string unused_regs;
  for (size_t i = 0; i < top_unused.size() && i < 8; ++i) {
    unused_regs += fmt::format(" {:04X}x{}", top_unused[i].first, top_unused[i].second);
  }
  REXLOG_INFO("[native][half2]   matched draws without both shaders analysed {:.1f} a frame (compared in full); "
              "draw-description fields compared {:.1f} a frame (inline vertex shaders equal by content {:.1f}); differences the shaders do not read {:.1f} a frame:{}; "
              "packets writing VS c240-c243 {:.1f} a frame "
              "({:.1f} inside a library draw's packets)",
              w.unanalysed / n, w.fields_compared / n, w.inline_vs_same / n, double(w.unused_values) / n, unused_regs.empty() ? " none" : unused_regs,
              w.owned_writes / n, w.owned_writes_in_draws / n);
  s.window = {};
  s.window_frames = 0;
  s.unmatched_logged = 0;
  s.owned_logged = 0;
  s.mismatch_registers.clear();
  s.unused_registers.clear();
}

}  // namespace rex::graphics
