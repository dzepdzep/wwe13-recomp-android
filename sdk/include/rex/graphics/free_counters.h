#pragma once
// wwe2k14 patch (Session 92, M0): the "free from the 360" counters. LOG ONLY: what of the 360 machinery still runs
// under TRUE NATIVE, printed as one `[d3d][free]` line per owned-frame window (OwnedFrameSummary), then reset.
// Cheap by design: a relaxed load + store (no locked RMW) on the hot counters (a lost increment between the two CP sides
// is accepted: these are census numbers); the translator's counter is a real atomic add (workers); the first-time shader
// set takes a mutex only when a shader is translated (never per call). Inline variables: one copy in the SDK library.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <set>
#include <string>

#include <rex/graphics/d3d_noring.h>  // (M5b) D3dNoringTakeCoherDropped
#include <rex/graphics/cut_probe.h>
#include <rex/logging.h>

namespace rex::graphics::free_counters {

struct Counters {
  std::atomic<uint64_t> truth_entries{0};    // NativeRenderer::TruthRecordDraw entries
  std::atomic<uint64_t> translator_runs{0};  // ShaderTranslator::TranslateAnalyzedShader runs of the translator itself (any thread; M6: not the store's hits)
  std::atomic<uint64_t> store_hits{0};       // (M6) translations served from the translation store instead of the translator
  std::atomic<uint64_t> packets{0};          // CommandProcessor::ExecutePacket(RingBuffer*) calls
  std::atomic<uint64_t> image_stores{0};     // registers stored into the D3D feed's register image (D3dFeedAccess::Apply)
  std::atomic<uint64_t> write_registers{0};  // CommandProcessor::WriteRegister calls
  std::atomic<uint64_t> wait_reg_mem{0};     // ExecutePacketType3_WAIT_REG_MEM calls
  std::atomic<uint64_t> answers_walks{0};    // d3d_noring's answers-side NfRunRingPacketForDraw walks
  std::atomic<uint64_t> clear_stock{0};      // D3dCopyAccess::ClearStock (clears drawn the library's way)
  std::atomic<uint64_t> direct_waits{0};     // (M5a) CommandProcessor::WaitGuestWord calls
  std::atomic<uint64_t> direct_wait_us{0};   // (M5a) microseconds spent in them
  std::atomic<uint64_t> resolves_regs{0};    // (M3) record resolves that wrote the register file and ran IssueCopy from registers
  std::atomic<uint64_t> cache_resolves{0};   // (M3) VulkanRenderTargetCache::Resolve calls
  std::atomic<uint64_t> presents_packet{0};  // (M3) the XE_SWAP packet handler entered from a packet
  std::atomic<uint64_t> swap_fetch_regs{0};  // (M3) the swap texture's fetch constant read from the register file / image
  std::atomic<uint64_t> fetch0_stale{0};     // (M3) draws reading fetch slot 0 with the value it had before the last present (unwritten since)
  std::atomic<uint64_t> vblank_wakes{0};     // (M5b) WaitGuestWord waits ended by the vblank event
  std::atomic<uint64_t> idle_wakes{0};       // (M5b) command-loop idle waits ended by the producer's wake
  std::atomic<uint64_t> idle_guard_timeouts{0};  // (M5b) idle guard timeouts with work queued (expected ~0)
};

inline Counters g_c;

// (Session 92, M4f-C) The cut's counters (gpu_d3d_owned_front bit 128): the record decoder as the executor. Command thread only (plain words: a lost
// increment between the two CP sides is not possible, the executor runs on the answers side's thread only); printed as `[d3d][cut]` by PrintAndReset
// when a queue attached with the cut on (`active`), then reset. Inline variable: one copy in the SDK library.
enum CutOldReason : uint32_t {
  kCutOldDraw = 0,    // a token with a DRAW item while the draw entry point is not wired
  kCutOldClear,       // a 0x7D clear record, entry not wired
  kCutOldResolve,     // a 0x7D resolve record, entry not wired
  kCutOldInline,      // a 0x7E inline draw, entry not wired
  kCutOldOther,       // a packet that is not a 0x7C / 0x7D / 0x7E record (or a 0x7D of another kind)
  kCutOldFraming,     // the payload's packet framing is wrong
  kCutOldCount
};
struct CutCounters {
  bool active = false;           // a queue attached with bit 128 on (d3d_noring.cpp Shared())
  bool arena_allocated = false;  // d3d_noring's record arena was allocated (any mode; under the cut it stays false)
  // executor, per item kind
  uint64_t items_packets = 0, items_regs = 0;
  uint64_t tokens = 0, runs = 0, run_words = 0, shader_items = 0, shader_load_failed = 0, draw_items = 0;
  uint64_t draws_before_owned = 0;  // DRAW items dropped: the native renderer is not enabled (boot)
  uint64_t draws_no_entry = 0;      // DRAW items dropped: the draw entry point is not wired (cannot happen: the item goes the old path)
  uint64_t records_before_owned = 0;  // clear / resolve / inline records dropped: the native renderer is not enabled (boot)
  uint64_t records_executed = 0;      // clear / resolve / inline records handed to their entry point
  uint64_t records_noop = 0;          // 0x7D GPR-split records (census only: a no-op)
  // the notifications (D3dFeedAccess::Notify's, made from the decoder's blocks): calls, and the constant buffers actually marked out of date
  uint64_t float_runs = 0, float_marks_vs = 0, float_marks_ps = 0;
  uint64_t fetch_runs = 0, fetch_marks = 0, fetch_texture_notes = 0, fetch_vertex_ranges = 0;
  uint64_t bool_runs = 0, bool_marks = 0;
  // register ops
  uint64_t reg_writes = 0, reg_rmw = 0, reg_tex = 0, reg_bin = 0;
  uint64_t reg_dropped = 0;       // register writes / rmw with no field and no bank (no reader): dropped
  uint64_t reg_out_of_range = 0;  // register numbers past the register file's size
  uint64_t run_words_no_field = 0;  // RUN / inline words stored with no typed field and no bank (incl. 0x5000-0x5002): dropped
  uint64_t entries_failed = 0;      // entry point answers of false
  uint64_t poison_rewritten = 0;    // gpu_d3d_cut_poison: poisoned registers a writer changed (bar 0)
  uint64_t owned_flips_ignored = 0;  // gpu_native_owned_frame switched off by a harness: ignored under the cut
  // draws / records dropped under the cut, by reason (pointer-keyed string literals)
  const char* drop_reason[24] = {};
  uint64_t drop_count[24] = {};
  uint64_t drop_other = 0;  // reasons past the 24 slots
  // draws skipped by a setting / drawn in a counted special way (NOT drops): hide HUD, shadow chain, half-rate shadows, crowd density, inline rect lists drawn, FSR jitter drawn
  const char* skip_reason[16] = {};
  uint64_t skip_count[16] = {};
  uint64_t skip_other = 0;
  uint32_t dropped_by_index[0x5003] = {};
  // items the executor did not take (the old path ran them), by reason
  uint64_t old_path[kCutOldCount] = {};
  // the old packet machinery under the cut (bar 0 after the latch frame): `after` = free_counters::g_owned_only_frame was set
  uint64_t packets_records_boot = 0, packets_records_after = 0;  // kNrPackets items run by D3dNoringExecute
  uint64_t segments_boot = 0, segments_after = 0;                // kNrSegment pieces run
  uint64_t segments_skipped = 0, skipped_reg[3] = {};              // kNrSegment pieces consumed without running: only the library's misc flush (0x05C8 / 0x0E00 / 0x0E40) type-0 writes + nops
  uint64_t segment_opcode[128] = {};                             // type-3 opcodes of the pieces after the latch
  uint64_t segment_type0 = 0, segment_other = 0;                 // type-0 packets / anything else in those pieces
  uint32_t segment_t0_base[0x8000] = {};                          // the base register of each type-0 packet in those pieces
  uint64_t ns = 0;
};
inline CutCounters g_cut;
// A draw skipped by a player setting (or drawn in a counted special way) under the cut: counted on its own, never as a drop.
inline void NoteCutSkip(const char* reason) {
  for (uint32_t i = 0; i < 16; ++i) {
    if (g_cut.skip_reason[i] == reason || !g_cut.skip_reason[i]) {
      g_cut.skip_reason[i] = reason;
      ++g_cut.skip_count[i];
      return;
    }
  }
  ++g_cut.skip_other;
}
// A draw / record dropped under the cut (counted by reason; pointer-keyed string literals, at most 24 reasons a window).
inline void NoteCutDrop(const char* reason) {
  for (uint32_t i = 0; i < 24; ++i) {
    if (g_cut.drop_reason[i] == reason || !g_cut.drop_reason[i]) {
      g_cut.drop_reason[i] = reason;
      ++g_cut.drop_count[i];
      return;
    }
  }
  ++g_cut.drop_other;
}
// The 8 most-dropped register indices of `table` as "0xIDX xN, ..." (plain words), then reset it.
inline std::string TakeTop8Plain(uint32_t* table, uint32_t size) {
  uint32_t top_index[8] = {}, top_count[8] = {};
  for (uint32_t i = 0; i < size; ++i) {
    const uint32_t c = table[i];
    table[i] = 0;
    if (!c) continue;
    for (int k = 0; k < 8; ++k) {
      if (c > top_count[k]) {
        for (int m = 7; m > k; --m) { top_index[m] = top_index[m - 1]; top_count[m] = top_count[m - 1]; }
        top_index[k] = i;
        top_count[k] = c;
        break;
      }
    }
  }
  std::string out;
  char buf[40];
  for (int k = 0; k < 8 && top_count[k]; ++k) {
    std::snprintf(buf, sizeof buf, "%s0x%04X x%u", k ? ", " : "", top_index[k], top_count[k]);
    out += buf;
  }
  return out.empty() ? std::string("none") : out;
}
// The `[d3d][cut]` line of a window, then the window's counters reset (`active` and `arena_allocated` stay).
inline std::string TakeCutLine(int64_t first, int64_t last) {
  CutCounters& c = g_cut;
  const double frames = double(last >= first ? last - first + 1 : 1);
  std::string ops_seg;
  {
    uint32_t order[128];
    for (uint32_t i = 0; i < 128; ++i) order[i] = i;
    std::sort(order, order + 128, [&](uint32_t a, uint32_t b) { return c.segment_opcode[a] > c.segment_opcode[b]; });
    char buf[48];
    for (uint32_t k = 0; k < 8 && c.segment_opcode[order[k]]; ++k) {
      std::snprintf(buf, sizeof buf, "%sop 0x%02X x%llu", k ? ", " : "", order[k], static_cast<unsigned long long>(c.segment_opcode[order[k]]));
      ops_seg += buf;
    }
    if (c.segment_type0) {
      std::snprintf(buf, sizeof buf, "%stype-0 x%llu", ops_seg.empty() ? "" : ", ", static_cast<unsigned long long>(c.segment_type0));
      ops_seg += buf;
    }
    {
      const std::string bases = TakeTop8Plain(c.segment_t0_base, 0x8000);
      if (bases != "none") ops_seg += std::string(ops_seg.empty() ? "" : "; ") + "type-0 base registers (top 8) " + bases;
    }
    if (c.segment_other) {
      std::snprintf(buf, sizeof buf, "%sother x%llu", ops_seg.empty() ? "" : ", ", static_cast<unsigned long long>(c.segment_other));
      ops_seg += buf;
    }
    if (ops_seg.empty()) ops_seg = "none";
  }
  const std::string dropped_top = TakeTop8Plain(c.dropped_by_index, 0x5003);
  std::string head_extra;
  char head[3072];
  std::snprintf(
      head, sizeof head,
      "[d3d][cut] frames %lld-%lld: executor items %llu packet + %llu register (tokens %llu, runs %llu of %llu words, shader items %llu (loads failed %llu), "
      "draw items %llu (dropped before the owned frame %llu, dropped with no draw entry %llu); clear / resolve / inline records handed on %llu (dropped before the owned frame %llu; GPR-split no-ops %llu)); "
      "notifications: float runs %llu (vertex float buffer marked out of date %llu, pixel %llu), fetch runs %llu (fetch buffer marked %llu, texture fetch notes %llu, vertex residency ranges %llu), "
      "bool/loop runs %llu (marked %llu); register ops: writes %llu, read-modify-write %llu, texture-table ops %llu, bin ops %llu; register ops with no reader dropped %llu "
      "(out of range %llu), by index, top 8: %s; packet items refused and DROPPED (never the old path), by reason: draw (no entry) %llu, clear %llu, resolve %llu, inline %llu, other packet record %llu, framing %llu; "
      "records executed as packets after the latch %llu (before it %llu); segment pieces after the latch %llu (before it %llu), opcodes after the latch: %s; "
      "arena allocated: %s; run words with no typed field %llu; entry calls failed %llu; poisoned registers rewritten %llu; executor %.1f us a frame",
      static_cast<long long>(first), static_cast<long long>(last), static_cast<unsigned long long>(c.items_packets),
      static_cast<unsigned long long>(c.items_regs), static_cast<unsigned long long>(c.tokens), static_cast<unsigned long long>(c.runs),
      static_cast<unsigned long long>(c.run_words), static_cast<unsigned long long>(c.shader_items), static_cast<unsigned long long>(c.shader_load_failed),
      static_cast<unsigned long long>(c.draw_items), static_cast<unsigned long long>(c.draws_before_owned), static_cast<unsigned long long>(c.draws_no_entry),
      static_cast<unsigned long long>(c.records_executed), static_cast<unsigned long long>(c.records_before_owned), static_cast<unsigned long long>(c.records_noop),
      static_cast<unsigned long long>(c.float_runs), static_cast<unsigned long long>(c.float_marks_vs), static_cast<unsigned long long>(c.float_marks_ps),
      static_cast<unsigned long long>(c.fetch_runs), static_cast<unsigned long long>(c.fetch_marks), static_cast<unsigned long long>(c.fetch_texture_notes),
      static_cast<unsigned long long>(c.fetch_vertex_ranges), static_cast<unsigned long long>(c.bool_runs), static_cast<unsigned long long>(c.bool_marks),
      static_cast<unsigned long long>(c.reg_writes), static_cast<unsigned long long>(c.reg_rmw), static_cast<unsigned long long>(c.reg_tex),
      static_cast<unsigned long long>(c.reg_bin), static_cast<unsigned long long>(c.reg_dropped), static_cast<unsigned long long>(c.reg_out_of_range),
      dropped_top.c_str(), static_cast<unsigned long long>(c.old_path[kCutOldDraw]), static_cast<unsigned long long>(c.old_path[kCutOldClear]),
      static_cast<unsigned long long>(c.old_path[kCutOldResolve]), static_cast<unsigned long long>(c.old_path[kCutOldInline]),
      static_cast<unsigned long long>(c.old_path[kCutOldOther]), static_cast<unsigned long long>(c.old_path[kCutOldFraming]),
      static_cast<unsigned long long>(c.packets_records_after), static_cast<unsigned long long>(c.packets_records_boot),
      static_cast<unsigned long long>(c.segments_after), static_cast<unsigned long long>(c.segments_boot), ops_seg.c_str(),
      c.arena_allocated ? "yes" : "no", static_cast<unsigned long long>(c.run_words_no_field), static_cast<unsigned long long>(c.entries_failed),
      static_cast<unsigned long long>(c.poison_rewritten), double(c.ns) / 1000.0 / frames);
  {
    std::string extra = "; segment pieces skipped (misc flush, no reader) " + std::to_string(c.segments_skipped) + " (0x05C8 x" + std::to_string(c.skipped_reg[0]) + ", 0x0E00 x" + std::to_string(c.skipped_reg[1]) + ", 0x0E40 x" + std::to_string(c.skipped_reg[2]) + ")";
    extra += "; owned-frame switch-offs ignored " + std::to_string(c.owned_flips_ignored) + "; draws / records dropped by reason: ";
    bool any = false;
    for (uint32_t i = 0; i < 24 && c.drop_reason[i]; ++i) {
      extra += std::string(any ? ", " : "") + c.drop_reason[i] + " x" + std::to_string(c.drop_count[i]);
      any = true;
    }
    if (c.drop_other) {
      extra += std::string(any ? ", " : "") + "other reasons x" + std::to_string(c.drop_other);
      any = true;
    }
    if (!any) extra += "none";
    if (cut_probe::g_poison_n) {
      extra += "; first poisoned registers rewritten:";
      for (uint32_t i = 0; i < cut_probe::g_poison_n && i < 4; ++i) {
        char pb[24];
        std::snprintf(pb, sizeof pb, " 0x%04X", cut_probe::g_poison_regs[i] - 1);
        extra += pb;
      }
      cut_probe::g_poison_n = 0;
    }
    extra += "; draws skipped by a setting or drawn in a counted special way (not drops): ";
    bool any_skip = false;
    for (uint32_t i = 0; i < 16 && c.skip_reason[i]; ++i) {
      extra += std::string(any_skip ? ", " : "") + c.skip_reason[i] + " x" + std::to_string(c.skip_count[i]);
      any_skip = true;
    }
    if (c.skip_other) extra += std::string(any_skip ? ", " : "") + "other x" + std::to_string(c.skip_other);
    else if (!any_skip) extra += "none";
    extra += std::string("; detector 2 armed ") + (cut_probe::g_armed_window ? "yes" : "no") + ", register readers called after the latch:";
    char buf[96];
    for (uint32_t r = 0; r < cut_probe::kReaderCount; ++r) {
      std::snprintf(buf, sizeof buf, " %s %llu", cut_probe::kReaderNames[r], static_cast<unsigned long long>(cut_probe::g_count[r]));
      extra += buf;
      for (uint32_t k = 0; k < 3 && k < cut_probe::g_count[r]; ++k) {
        std::snprintf(buf, sizeof buf, "%s%+lld", k ? "," : " (module-relative sites ", static_cast<long long>(cut_probe::g_first[r][k]));
        extra += buf;
      }
      if (cut_probe::g_count[r]) extra += ")";
      cut_probe::g_count[r] = 0;
      cut_probe::g_first[r][0] = cut_probe::g_first[r][1] = cut_probe::g_first[r][2] = 0;
    }
    c.owned_flips_ignored = 0;
    c.drop_other = c.skip_other = 0;
    cut_probe::g_armed_window = cut_probe::g_on;
    for (uint32_t i = 0; i < 16; ++i) {
      c.skip_reason[i] = nullptr;
      c.skip_count[i] = 0;
    }
    for (uint32_t i = 0; i < 24; ++i) {
      c.drop_reason[i] = nullptr;
      c.drop_count[i] = 0;
    }
    head_extra = extra;
  }
  const bool active = c.active, arena = c.arena_allocated;
  c.items_packets = c.items_regs = c.tokens = c.runs = c.run_words = c.shader_items = c.shader_load_failed = c.draw_items = 0;
  c.draws_before_owned = c.draws_no_entry = c.records_before_owned = c.records_executed = c.records_noop = 0;
  c.float_runs = c.float_marks_vs = c.float_marks_ps = c.fetch_runs = c.fetch_marks = c.fetch_texture_notes = c.fetch_vertex_ranges = 0;
  c.bool_runs = c.bool_marks = c.reg_writes = c.reg_rmw = c.reg_tex = c.reg_bin = c.reg_dropped = c.reg_out_of_range = 0;
  for (uint64_t& v : c.old_path) v = 0;
  c.packets_records_boot = c.packets_records_after = c.segments_boot = c.segments_after = 0;
  c.segments_skipped = c.skipped_reg[0] = c.skipped_reg[1] = c.skipped_reg[2] = 0;
  for (uint64_t& v : c.segment_opcode) v = 0;
  c.segment_type0 = c.segment_other = c.ns = 0;
  c.run_words_no_field = c.entries_failed = c.poison_rewritten = 0;
  c.active = active;
  c.arena_allocated = arena;
  return std::string(head) + head_extra;
}

// (M3a) The owned frame's record resolves (native/owned_resolve.cpp), command thread only; printed as `[d3d][owned-resolve]`.
struct OwnedResolveCounters {
  uint64_t calls = 0, served = 0, written = 0, filled_written = 0, wb_failed = 0, not_taken = 0, nothing = 0, clears_skipped = 0, rename_skipped = 0;
  uint64_t wb_no_shader = 0, range_failed = 0, read_back = 0;
  uint64_t chain_by_src[5] = {};  // (review P2) shadow-chain copy skips by copy source
};
inline OwnedResolveCounters g_or;
// (M3b) The present item's swap texture fetch (six dwords, host order values as the register holds them), set for the one
// PresentFrame call it belongs to; null = RequestSwapTexture reads fetch constant 0 of the register source as before.
inline const uint32_t* g_swap_fetch_override = nullptr;
// (M3b diagnostic, command thread, log only) Does any draw read fetch slot 0 after a present before the game wrote it again? With the
// present as data fetch 0 of the register image is not written by it: `last` is the value the last draw using slot 0 saw before the
// present; after a present, a draw whose slot 0 still equals it counts (unwritten since); a different value ends the check.
inline bool g_fetch0_after_present = false, g_fetch0_track = false;  // (track: the first present item under the latch)
// Writes-based (review P5): any write of fetch constant 0 (register write, bulk range, image store) ends the check; a draw sampling slot 0 while it is
// still set counts.
inline void NoteFetch0Write() { g_fetch0_after_present = false; }
inline void NoteFetch0Draw() {
  if (g_fetch0_after_present) g_c.fetch0_stale.store(g_c.fetch0_stale.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}
// The frame's owned-only mode for the base class (review P6): set at the frame end by the Vulkan command processor from the native renderer's
// frame-stable mode; false on any other backend.
inline bool g_owned_only_frame = false;
inline void NotePresentForFetch0() { g_fetch0_track = g_fetch0_after_present = true; }
// (M5a diagnostics, log only) per register index: guest MMIO reads (GraphicsSystem::ReadRegister) and CommandProcessor::WriteRegister calls.
inline std::atomic<uint32_t> g_mmio_reads[0x2000];
inline std::atomic<uint32_t> g_register_writes[0x2000];
inline void BumpIndex(std::atomic<uint32_t>* table, uint32_t index) {
  if (index < 0x2000) table[index].fetch_add(1, std::memory_order_relaxed);
}
// The top 8 of a table as "0xIDX xN, ...", then reset it.
inline std::string TakeTop8(std::atomic<uint32_t>* table) {
  uint32_t top_index[8] = {}, top_count[8] = {};
  for (uint32_t i = 0; i < 0x2000; ++i) {
    const uint32_t c = table[i].exchange(0, std::memory_order_relaxed);
    if (!c) continue;
    for (int k = 0; k < 8; ++k) {
      if (c > top_count[k]) {
        for (int m = 7; m > k; --m) { top_index[m] = top_index[m - 1]; top_count[m] = top_count[m - 1]; }
        top_index[k] = i;
        top_count[k] = c;
        break;
      }
    }
  }
  std::string out;
  char buf[40];
  for (int k = 0; k < 8 && top_count[k]; ++k) {
    std::snprintf(buf, sizeof buf, "%s0x%04X x%u", k ? ", " : "", top_index[k], top_count[k]);
    out += buf;
  }
  return out.empty() ? std::string("none") : out;
}
inline std::mutex g_shader_mutex;
inline std::set<uint64_t> g_seen_shaders;
inline std::string g_new_shaders;  // "VS 0x..., PS 0x..." translated for the first time since the last line (<= 16)
inline uint32_t g_new_shader_count = 0;

inline void Bump(std::atomic<uint64_t>& a, uint64_t n = 1) {
  a.store(a.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

// (M6) A translation was served from the translation store; the translator did not run.
inline void NoteStoreHit() { g_c.store_hits.fetch_add(1, std::memory_order_relaxed); }

// A guest shader went through the translator. is_vertex: VS, else PS.
inline void NoteTranslation(bool is_vertex, uint64_t hash) {
  g_c.translator_runs.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(g_shader_mutex);
  if (!g_seen_shaders.insert(hash ^ (is_vertex ? 0 : 0x8000000000000001ull)).second) return;
  ++g_new_shader_count;
  if (g_new_shader_count <= 16) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%s0x%016llX", g_new_shaders.empty() ? (is_vertex ? "VS " : "PS ") : (is_vertex ? ", VS " : ", PS "),
                  static_cast<unsigned long long>(hash));
    g_new_shaders += buf;
  }
}

// Print (when `print`) and reset. frames A-B = the window.
inline void PrintAndReset(bool print, int64_t first, int64_t last) {
  const auto take = [](std::atomic<uint64_t>& a) { return a.exchange(0, std::memory_order_relaxed); };
  const uint64_t te = take(g_c.truth_entries), tr = take(g_c.translator_runs), sh = take(g_c.store_hits), pk = take(g_c.packets),
                 is = take(g_c.image_stores), wr = take(g_c.write_registers), wm = take(g_c.wait_reg_mem),
                 aw = take(g_c.answers_walks), cs = take(g_c.clear_stock), dw = take(g_c.direct_waits),
                 dwu = take(g_c.direct_wait_us), rr = take(g_c.resolves_regs), cr = take(g_c.cache_resolves),
                 pp = take(g_c.presents_packet), sf = take(g_c.swap_fetch_regs), f0 = take(g_c.fetch0_stale),
                 vw = take(g_c.vblank_wakes), iw = take(g_c.idle_wakes), igt = take(g_c.idle_guard_timeouts);
  const uint64_t cd = D3dNoringTakeCoherDropped();
  std::string fresh;
  uint32_t fresh_count;
  {
    std::lock_guard<std::mutex> lock(g_shader_mutex);
    fresh = std::move(g_new_shaders);
    fresh_count = g_new_shader_count;
    g_new_shaders.clear();
    g_new_shader_count = 0;
  }
  const std::string cut_line = g_cut.active ? TakeCutLine(first, last) : std::string();  // (M4f-C) read and reset with the window
  if (!print) return;
  if (!cut_line.empty()) REXGPU_INFO("{}", cut_line);
  REXGPU_INFO("[d3d][free] frames {}-{}: truth entries {}, translator runs {} (new shaders: {}{}), store hits {}, packets executed {}, register writes {} "
              "(feed image stores {}, WriteRegister calls {}), wait_reg_mem {}, answers-side walks {}, record clears drawn the library's way {}, direct guest-word waits {} ({:.1f} ms), resolves via IssueCopy registers {}, cache resolves {}, presents as packets {}, "
              "swap fetch from registers {}, draws reading fetch 0 unwritten since the present {}, vblank wakes {}, idle wakes {}, idle guard timeouts with work queued {}, coherency handshakes dropped {}",
              first, last, te, tr, fresh.empty() ? "none" : fresh,
              fresh_count > 16 ? ", ... (" + std::to_string(fresh_count) + " in all)" : std::string(), sh, pk, is + wr, is, wr, wm, aw, cs, dw, double(dwu) / 1000.0, rr, cr, pp, sf, f0, vw, iw, igt, cd);
}

}  // namespace rex::graphics::free_counters
