// wwe2k14 patch (Session 87, D3D layer STAGE 2 "no ring"): the consumer of the typed host queue (include\rex\graphics\
// d3d_noring.h). New file; the edits elsewhere are the four member declarations in command_processor.h, three call points in
// command_processor.cpp (the loop's drain, the idle wait's condition, DetDrainedFor) and this file's CMake line.
//
// The command thread pops the game's queue in FRONT of the ring branch of the worker loop, strictly in order, and executes each
// item with the code the ring packet it replaces would have reached:
//   kNrSegment        -> an INDIRECT_BUFFER packet handed to the ordinary packet loop (the segment's own packets: the F1 / F2 /
//                        F3 tokens, any fallback packet, the library's remaining small writes) - the nf segment machinery,
//                        half2 attribution and the det bookkeeping see exactly what a ring segment gives them.
//   kNrPackets/Present-> packets by value, run through the same loop as a ring-level packet (the drawing side first, then the
//                        answers side: ExecutePrimaryBuffer's loop on host memory).
//   kNrFence          -> the two EVENT_WRITE_SHD stores of the segment end (heap marker, then fence number), after everything
//                        before the item has been executed. The fence is written by THIS thread in stream order and only ever
//                        by it: the library's own CPU-store mode (device byte 11069 bit 0x02, Session 51) stays off.
//   kNrFlipHandshake  -> flip word = 1, scratch registers, the guest interrupt callback (the flip scheduler 0x82CB2A28) and the
//                        one pacing wait: the flip word (scratch+4) == 0 (WAIT_REG_MEM with the ring's own wait policy).
//   kNrSceneTiming / kNrInterrupt -> scratch registers + the guest callback (0x82C16978 / 0x82C16980 for the scene-timing slots).
//   kNrBarrier        -> publishes the ticket.
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rex/cvar.h>
#include <rex/det.h>
#include <rex/graphics/command_processor.h>
#include <rex/graphics/d3d_feed.h>
#include <rex/graphics/d3d_noring.h>
#include <rex/graphics/d3d_owned_record.h>  // wwe2k14 patch (Session 92, M4a)
#include <rex/graphics/graphics_system.h>
#include <rex/graphics/d3d_tex.h>
#include <rex/graphics/xenos.h>
#include <rex/graphics/free_counters.h>
#include <rex/logging.h>
#include <rex/memory.h>
#include <rex/system/xmemory.h>
#include <rex/memory/ring_buffer.h>
#include <rex/memory/utils.h>
#include <rex/perf/timeline.h>
#include <rex/thread.h>
#include <rex/ui/presenter.h>  // wwe2k14 patch (Session 92, M5c): the held publish
#include <rex/platform.h>
#if REX_PLATFORM_WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// Capability marker (the exe looks it up by name before it produces anything), the SDK-side switch, and the hand-over of the
// exe's shared block. All read-only for the user except gpu_d3d_noring.
REXCVAR_DEFINE_INT32(gpu_d3d_noring_abi, rex::graphics::kD3dNoringAbi, "GPU/Native",
                     "Session 87 (D3D layer stage 2): the host queue ABI this build consumes (include\\rex\\graphics\\d3d_noring.h). "
                     "Read-only marker, do not set")
    .range(1, 2);
// (Session 92, M5b) capability marker: the highest block ABI this build understands (3 = the wake fields). An exe that sees it sets the
// block's abi to 3; an ABI-2 exe (or one that does not look) keeps abi 2 and the SDK polls as before.
REXCVAR_DEFINE_INT32(gpu_d3d_noring_abi_max, rex::graphics::kD3dNoringAbiWake, "GPU/Native",
                     "Session 92 (M5b): the highest host queue ABI this build understands (3 = consumer_idle / wake fields). "
                     "Read-only marker, do not set")
    .range(2, 3);
REXCVAR_DEFINE_INT32(gpu_d3d_noring, 0, "GPU/Native",
                     "Session 87 (D3D layer stage 2, experimental): 0 = off (the command loop never looks at the host queue), "
                     "1 = consume the game's queue ahead of the ring (the game's --wwe2k14_d3d_noring sets it)")
    .range(0, 1);
REXCVAR_DEFINE_INT32(gpu_d3d_noring_ops, 1, "GPU/Native",
                     "Session 87 (D3D layer stage 2): capability marker, 1 = the Registers item also carries the tiling bin state "
                     "(ops 3-6). Read-only marker, do not set")
    .range(1, 1);
REXCVAR_DEFINE_UINT64(gpu_d3d_noring_shared, 0, "GPU/Native",
                      "Session 87 (D3D layer stage 2): the address of the game exe's D3dNoringShared block (set by the exe; "
                      "do not set by hand)");
// wwe2k14 patch (Session 92, M4a): the owned front end's latch, a bit mask. 0 = off (this file behaves as before; nothing is decoded).
REXCVAR_DEFINE_UINT32(gpu_d3d_owned_front, 0, "GPU/Native",
                      "Session 92 (M4a): bit 0 = the record decoder runs as a SHADOW (the queue's items are decoded into typed state and compared with the "
                      "feed image; the existing path still draws). 0 = off, byte-for-byte the old path. Allowed bits 0-7 (0xFF): bit 0 = M4a, bit 1 = M4b, bit 2 = M4c, bit 3 = M4d, bit 4 (16) = M4e (clears, resolves and inline draws built from the typed state and compared, shadow); bit 5 (32) = M4f-A: the typed state DRAWS the token draw items (needs bits 1 and 2, i.e. 6; a draw it cannot take goes the old way, counted by reason); bit 6 (64) is taken by M4f-B; bit 7 (128) = M4f-C THE CUT, read when the queue attaches: the decoder EXECUTES the queue's packet and register items in order on its own typed state (the D3D feed is off, no record arena, no register-file writes for them); an item whose draw / clear / resolve / inline entry point is not wired goes the old way, counted; a bit nothing reads yet does nothing")
    .range(0, 255);

// wwe2k14 patch (Session 92, M5a): the sync items as plain ordered stores + a direct wait. Default false = byte-for-byte the old path.
REXCVAR_DEFINE_BOOL(gpu_d3d_sync_direct, false, "GPU/Native",
                    "Session 92 (M5a): the no-ring queue's fence / handshake / scene-timing / interrupt items store their guest-visible words "
                    "directly (no register writes, no WAIT_REG_MEM packet) and the pacing wait is CommandProcessor::WaitGuestWord. "
                    "Same order, same thread, same guest stores. Effective only with the host queue (gpu_d3d_noring)");

// wwe2k14 patch (Session 92, M3b): the present as data. Effective only with the owned-only latch (gpu_native_owned_only); default false =
// byte-for-byte the packet path (type-0 fetch constant 0 + XE_SWAP through ExecutePacket).
REXCVAR_DEFINE_BOOL(gpu_d3d_owned_present, false, "GPU/Native",
                    "Session 92 (M3b): the no-ring queue's present item is decoded as data - the swap texture's fetch and the front buffer words "
                    "go to CommandProcessor::PresentFrame, no fetch constant 0 written, no XE_SWAP packet executed. A payload that is not the "
                    "expected words takes the packet path (counted). Needs gpu_native_owned_only");

// wwe2k14 patch (Session 92, M5b): events instead of polls. Default false = byte-for-byte the old path.
REXCVAR_DEFINE_BOOL(gpu_d3d_sync_events, false, "GPU/Native",
                    "Session 92 (M5b): the pacing wait of the no-ring handshake waits on the vblank event (MarkVblank sets it; 2 ms guard) instead "
                    "of 1 ms sleeps, and the command loop's idle wait is woken by the producer (block ABI 3: consumer_idle / wake; 5 ms guard) "
                    "instead of polling every 1 ms. Effective only with the host queue AND gpu_d3d_sync_direct");

// wwe2k14 patch (Session 92, M5c): the pacing wait off the record thread. Default false = byte-for-byte M5a / M5b.
REXCVAR_DEFINE_BOOL(gpu_d3d_pace_gate, false, "GPU/Native",
                    "Session 92 (M5c): the flip handshake does not wait for the flip word: it opens a gate, the command thread keeps decoding, fences "
                    "are queued (written in order at the release, never earlier), the present's host publish is held, and the first check that finds "
                    "the flip word at 0 (between items, or on the vblank event while idle) publishes the held frame, writes the fences and closes the "
                    "gate. Effective only with the host queue, gpu_d3d_sync_direct AND gpu_d3d_sync_events");

// wwe2k14 patch (Session 92, M4f-C): the proof switch of the cut. Default false; only active with the cut.
REXCVAR_DEFINE_BOOL(gpu_d3d_cut_poison, false, "GPU/Native",
                    "Session 92 (M4f-C proof): with the cut latched, at the first item after the owned frame starts the register file's render-state, constant, "
                    "fetch and bool/loop ranges (and every typed field's register) are filled with a pattern; every window re-checks them and "
                    "'[d3d][cut] poisoned registers rewritten' counts any that a writer changed (bar 0). Registers below 0x2000 are left alone");

// wwe2k14 patch (Session 95, the wall): the command thread's own CPU time per frame in the log. Default 0 = off, nothing read.
REXCVAR_DEFINE_INT32(gpu_d3d_wall_stats, 0, "GPU/Native",
                     "Session 95 (the wall): every N presents, one [d3d][wall] line with the command thread's own CPU per frame (user, kernel, cycles). 0 = off")
    .range(0, 100000);

// wwe2k14 patch (Session 95, the wall): the command thread's per-draw cuts. 0 = off, every path as before.
REXCVAR_DEFINE_UINT32(gpu_d3d_wall, 0, "GPU/Native",
                      "Session 95 (the wall): the command thread's per-draw work, a bit mask. Bit 1 = the dead-page list is asked through its mask; bit 2 = a draw no crowd run matches skips the crowd "
                      "call; bit 4 = the decoder notifies only what really changed (constant buffers, texture slots, vertex residency: a re-write of the same words marks nothing); bit 8 = the vertex "
                      "and index range stamps are kept by range, not by fetch slot; bit 16 = the texture key of a fetch constant is kept per (object, words); bit 32 = small trims (owned pages searched from the end, the crowd mesh box kept by its bytes); bit 64 = the "
                      "interpreter's memo and the bounds key by the constants the shader reads (gpu_interp_memo); bit 128 = a crowd run's members after the first are repeats of it. 0 = off")
    .range(0, 255);
REXCVAR_DECLARE(int32_t, gpu_tex_bind_memo);  // (Session 95, r6tex: texture\cache.cpp)
REXCVAR_DECLARE(int32_t, gpu_tex_bind_memo_ab);
REXCVAR_DECLARE(int32_t, gpu_d3d_draw_record);  // (Session 95, r3cmd: owned_front_own.cpp)
REXCVAR_DEFINE_INT32(gpu_d3d_draw_record_ab, 0, "GPU/Native",
                     "Session 95 (r3cmd, measuring only): with gpu_d3d_wall_stats on, every stats window flips gpu_d3d_draw_record between 0 and this value, so both arms share one run's load. 0 = off");
REXCVAR_DEFINE_UINT32(gpu_d3d_wall_ab, 0, "GPU/Native",
                      "Session 95 (the wall, measuring only): with gpu_d3d_wall_stats on, every stats window flips gpu_d3d_wall between 0 and this mask, so both arms share one run's load. 0 = off")
    .range(0, 255);

// (defined in command_processor.cpp)
REXCVAR_DECLARE(bool, gpu_native_frame_draw);
REXCVAR_DECLARE(bool, gpu_native_owned_only);  // (owned_steps.cpp)
REXCVAR_DECLARE(bool, gpu_dlaa);
REXCVAR_DECLARE(bool, gpu_dlaa_jitter);
REXCVAR_DECLARE(bool, gpu_dlss);
REXCVAR_DECLARE(bool, gpu_edram_rename);
REXCVAR_DECLARE(int32_t, gpu_native_verify_every);
REXCVAR_DECLARE(bool, gpu_native_lean_draws);
REXCVAR_DECLARE(bool, gpu_native_lean_draws_verify);
REXCVAR_DECLARE(bool, gpu_native_lean_dirty_census);
REXCVAR_DECLARE(int32_t, gpu_debug_draw_census_count);
REXCVAR_DECLARE(bool, gpu_d3d_owned_resolve);  // (d3d_copy_records.cpp)


namespace rex::graphics {

namespace {

std::atomic<D3dNoringShared*> g_shared_checked{nullptr};  // the block that passed the checks (written by the command thread; read by others: review D4)
std::atomic<const void*> g_shared_rejected{nullptr};
// (M5b) the vblank event (file static: no new member in graphics_system.h) and the census counters ([d3d][free], via free_counters)
std::unique_ptr<rex::thread::Event> g_vblank_event;
std::once_flag g_vblank_once;
rex::thread::Event& VblankEvent() {
  std::call_once(g_vblank_once, [] { g_vblank_event = rex::thread::Event::CreateAutoResetEvent(false); });
  return *g_vblank_event;
}
std::atomic<bool> g_wake_active{false};  // an ABI-3 block is attached and the latch is on: the idle wait is a 5 ms guard, not a 1 ms poll
uint64_t g_coher_seen = 0;               // coher_dropped already reported
void WakeThunk(void* ctx) { static_cast<rex::thread::Event*>(ctx)->Set(); }
// (M5b review fix) the wake target is an event that is NEVER freed (leaked on purpose): a producer that read consumer_idle == 1 may call
// wake() at any time, including during the CP's teardown; the CP's own write_ptr_index_event_ dies with it. The idle loop waits on both.
rex::thread::Event& WakeEvent() {
  static rex::thread::Event* const e = rex::thread::Event::CreateAutoResetEvent(false).release();
  return *e;
}  // a block that failed them (HasWork must not report work for it)
uint64_t g_present_census = 0;
uint64_t g_sync_before_setup = 0;  // (M5a S1) sync items that took the old path because the scratch block was not set up yet
uint64_t g_present_data = 0, g_present_unexpected = 0, g_present_packet_path = 0;  // (M3b) presents run as data / payload not as expected / latch off or not applicable

// (Session 92, M5c) the pacing gate. Command thread only, except the two atomics (HasWork reads them from the det thread).
struct GateFence {
  uint32_t marker_at, marker, value_at, value;
};
struct Gate {
  std::atomic<bool> open{false};
  std::atomic<uint32_t> flip_at{0};  // the flip word's guest address with the endian bits, as the wait takes it
  std::chrono::steady_clock::time_point t_open;
  uint32_t items_since = 0;  // items run since the gate opened
  bool slept = false;        // the command loop was idle (waiting on the vblank event too) while it was open
  bool held = false;         // the present of this gate held its publish
  std::vector<GateFence> q;
} g_gate;
struct GateStats {
  uint64_t opened = 0, rel_immediate = 0, rel_poll = 0, rel_vblank_idle = 0, rel_second_hs = 0, rel_ring = 0;
  uint64_t fences_deferred = 0, max_queue = 0, fences_written = 0;
  uint64_t second_hs_waits = 0, second_hs_wait_us = 0, ring_waits = 0;
  uint64_t skipped_wait_us = 0;  // sum of (release - open): the wait the old path would have made at the handshake
  uint64_t pub_deferred_scene = 0, pub_at_scene_end = 0;
  uint64_t presents_gated = 0, presents_held = 0, presents_not_held = 0, published_at_release = 0, presents_run = 0;
} g_gs;
bool g_scene_open = false;  // (M5c, G1) between a scene-timing begin and end item (0x82C16978 / 0x82C16980); command thread only
constexpr uint32_t kSceneBeginCb = 0x82C16978u, kSceneEndCb = 0x82C16980u;
bool GateOn() { return REXCVAR_GET(gpu_d3d_pace_gate) && D3dNoringSyncEventsOn(); }

// Stage 2c: packets by value (record tokens, the present's words) are copied into an arena of guest physical memory of ours and
// read from there, so that every packet has a real, unique guest address (the scene crowd pass compares packet addresses; a
// host buffer reused per item would look like the same packet executed twice). Command thread only.
struct Arena {
  uint32_t virtual_base = 0;
  uint32_t size = 0;
  uint32_t offset = 0;
  uint32_t wraps = 0;
  bool failed = false;
};
Arena g_arena;
constexpr uint32_t kArenaBytes = 4u << 20;  // (review B1: measured ~260 KB of records a frame in the bench; was 16 MB)
bool g_arena_item = false;        // the item being run reads its packets from the arena
uint32_t g_record_kind = 0;       // see D3dNoringRecordKind()
// (Session 92, M4f-C) THE CUT: gpu_d3d_owned_front bit 128, read ONCE where the queue attaches (Shared()), not at a frame boundary. Under it the arena is
// never allocated (packets that still reach D3dNoringRunPackets run from a host buffer), kNrPackets / kNrRegs go to the decoder's executor, and
// the packets that run anyway are counted (free_counters::g_cut).
bool g_cut = false;
bool g_poison_armed = false;  // gpu_d3d_cut_poison: the register file's executor-owned ranges were filled with the pattern

// (Session 92, M4f-C) Settings the cut does not support are forced while it is latched (they read registers captured by the old draws, or run the old path's
// own diagnostics): DLSS / DLAA (a future step), the EDRAM rename, the native verify, and the lean-draw switches the typed draw needs on / off. Done at the
// latch and before every drain (the in-game GRAPHICS menu can change them); each is logged once with what it was. gpu_dlaa_jitter (FSR) is drawn.
void CutEnforceSettings(bool first) {
  (void)first;
  static uint32_t logged = 0;
  const auto note = [&](const char* name) {
    if (logged++ < 16) REXGPU_WARN("[d3d][cut] setting {} is not supported under the cut: forced to its supported value while the cut is latched", name);
  };
  if (REXCVAR_GET(gpu_dlaa)) { REXCVAR_SET(gpu_dlaa, false); note("gpu_dlaa (off)"); }
  if (REXCVAR_GET(gpu_dlaa_jitter)) { REXCVAR_SET(gpu_dlaa_jitter, false); note("gpu_dlaa_jitter (off: the per-frame half-pixel jitter reads registers captured by the old draws)"); }
  if (REXCVAR_GET(gpu_dlss)) { REXCVAR_SET(gpu_dlss, false); note("gpu_dlss (off)"); }
  if (REXCVAR_GET(gpu_edram_rename)) { REXCVAR_SET(gpu_edram_rename, false); note("gpu_edram_rename (off)"); }
  if (REXCVAR_GET(gpu_native_verify_every) != 0) { REXCVAR_SET(gpu_native_verify_every, 0); note("gpu_native_verify_every (0)"); }
  if (!REXCVAR_GET(gpu_native_lean_draws)) { REXCVAR_SET(gpu_native_lean_draws, true); note("gpu_native_lean_draws (on)"); }
  if (REXCVAR_GET(gpu_native_lean_draws_verify)) { REXCVAR_SET(gpu_native_lean_draws_verify, false); note("gpu_native_lean_draws_verify (off)"); }
  if (REXCVAR_GET(gpu_native_lean_dirty_census)) { REXCVAR_SET(gpu_native_lean_dirty_census, false); note("gpu_native_lean_dirty_census (off)"); }
  if (REXCVAR_GET(gpu_debug_draw_census_count) > 0) { REXCVAR_SET(gpu_debug_draw_census_count, 0); note("gpu_debug_draw_census_count (0)"); }
}

// The shared block, validated once. Command thread only.
D3dNoringShared* Shared() {
  if (!REXCVAR_GET(gpu_d3d_noring)) return nullptr;
  auto* s = reinterpret_cast<D3dNoringShared*>(static_cast<uintptr_t>(REXCVAR_GET(gpu_d3d_noring_shared)));
  if (!s) return nullptr;
  if (s == g_shared_checked.load(std::memory_order_acquire)) return s;
  if (s->magic != kD3dNoringMagic || (s->abi != kD3dNoringAbi && s->abi != kD3dNoringAbiWake) || !s->storage || s->capacity < 1024 ||
      (s->capacity & (s->capacity - 1)) != 0) {
    if (g_shared_rejected.exchange(s) != s) {
      REXGPU_ERROR("[d3d][noring] the shared block at {:p} is not a valid D3dNoringShared (magic {:08X}, abi {}, capacity {}): "
                   "the host queue stays unused",
                   static_cast<void*>(s), s->magic, s->abi, s->capacity);
    }
    return nullptr;
  }
  g_shared_checked.store(s, std::memory_order_release);
  REXGPU_INFO("[d3d][noring] host queue attached: {} dwords of storage at {:p}", s->capacity, static_cast<void*>(s->storage));
  return s;
}

}  // namespace

int D3dNoringIdleWaitMs() {
  if (g_wake_active.load(std::memory_order_relaxed)) return 5;  // (M5b) the producer wakes the loop: the wait is only a guard
  return REXCVAR_GET(gpu_d3d_noring) ? 1 : 5;
}

bool D3dNoringSyncEventsOn() {
  return REXCVAR_GET(gpu_d3d_sync_events) && REXCVAR_GET(gpu_d3d_noring) && REXCVAR_GET(gpu_d3d_sync_direct);
}

void D3dNoringVblankSignal() {
  if (D3dNoringSyncEventsOn()) VblankEvent().Set();
}

bool D3dNoringVblankWait(int timeout_ms) {
  const bool signalled =
      rex::thread::Wait(&VblankEvent(), false, std::chrono::milliseconds(timeout_ms)) == rex::thread::WaitResult::kSuccess;
  if (signalled) rex::graphics::free_counters::Bump(rex::graphics::free_counters::g_c.vblank_wakes);
  return signalled;
}

int D3dNoringIdleArm(rex::thread::Event* loop_event) {
  if (!D3dNoringSyncEventsOn()) {
    g_wake_active.store(false, std::memory_order_relaxed);
    return 0;
  }
  D3dNoringShared* s = g_shared_checked.load(std::memory_order_acquire);
  if (!s || s->abi < kD3dNoringAbiWake) {  // an ABI-2 exe: no wake, the old poll (counted as a guard timeout when it finds work)
    g_wake_active.store(false, std::memory_order_relaxed);
    return 0;
  }
  (void)loop_event;
  if (s->wake_ctx != &WakeEvent() || s->wake != &WakeThunk) {  // attach: fill the wake before the first idle store (it publishes them)
    s->wake_ctx = &WakeEvent();
    s->wake = &WakeThunk;
  }
  g_wake_active.store(true, std::memory_order_relaxed);
  s->consumer_idle.store(1, std::memory_order_seq_cst);
  if (s->head.load(std::memory_order_relaxed) != s->tail.load(std::memory_order_seq_cst)) {  // the Dekker re-check
    s->consumer_idle.store(0, std::memory_order_seq_cst);
    return 2;
  }
  return 1;
}

rex::thread::WaitResult D3dNoringIdleWait(rex::thread::Event* loop_event, int timeout_ms) {
  if (g_gate.open.load(std::memory_order_relaxed)) {
    // (M5c) an open pacing gate: the vblank event (the flip word's usual clearer) ends the idle wait too, so the held frame goes out at the vblank
    rex::thread::WaitHandle* handles[3] = {loop_event, &WakeEvent(), &VblankEvent()};
    const auto r = rex::thread::WaitAny(handles, 3, true, std::chrono::milliseconds(timeout_ms)).first;
    g_gate.slept = true;
    return r;
  }
  rex::thread::WaitHandle* handles[2] = {loop_event, &WakeEvent()};
  return rex::thread::WaitAny(handles, 2, true, std::chrono::milliseconds(timeout_ms)).first;
}

void D3dNoringIdleDisarm(bool timed_out) {
  D3dNoringShared* s = g_shared_checked.load(std::memory_order_acquire);
  if (!s) return;
  s->consumer_idle.store(0, std::memory_order_seq_cst);
  if (!timed_out) {
    rex::graphics::free_counters::Bump(rex::graphics::free_counters::g_c.idle_wakes);
  } else if (s->head.load(std::memory_order_relaxed) != s->tail.load(std::memory_order_seq_cst)) {
    rex::graphics::free_counters::Bump(rex::graphics::free_counters::g_c.idle_guard_timeouts);  // work was queued and nobody woke us
  }
}

uint64_t D3dNoringTakeCoherDropped() {
  D3dNoringShared* s = g_shared_checked.load(std::memory_order_acquire);
  if (!s || s->abi < kD3dNoringAbiWake) return 0;
  const uint64_t now = s->coher_dropped.load(std::memory_order_relaxed);
  const uint64_t d = now - g_coher_seen;
  g_coher_seen = now;
  return d;
}

uint32_t D3dNoringRecordKind() { return g_record_kind; }

namespace {
// (Session 95, the wall) Called on the command thread once per present.
void WallStatsPresent(uint64_t frame) {
  static bool told = false;
  if (!told && REXCVAR_GET(gpu_d3d_wall)) {
    told = true;
    REXGPU_INFO("[d3d][wall] on: gpu_d3d_wall {} (frame {})", REXCVAR_GET(gpu_d3d_wall), frame);
  }
  const int32_t every = REXCVAR_GET(gpu_d3d_wall_stats);
  if (every <= 0) return;
#if REX_PLATFORM_WIN32
  static uint32_t seen = 0;
  static uint64_t last_user = 0, last_kernel = 0, last_cycles = 0;
  static auto last_t = std::chrono::steady_clock::now();
  if (++seen < uint32_t(every)) return;
  FILETIME c, e, k, u;
  ULONG64 cycles = 0;
  if (!GetThreadTimes(GetCurrentThread(), &c, &e, &k, &u)) return;
  QueryThreadCycleTime(GetCurrentThread(), &cycles);
  const uint64_t user = (uint64_t(u.dwHighDateTime) << 32) | u.dwLowDateTime, kernel = (uint64_t(k.dwHighDateTime) << 32) | k.dwLowDateTime;
  const auto now = std::chrono::steady_clock::now();
  const double wall_ms = std::chrono::duration<double, std::milli>(now - last_t).count();
  const double n = double(seen);
  REXGPU_INFO("[d3d][wall] frame {}: arm {}: record {}: command thread per frame over {}: cpu {:.3f} ms (user {:.3f}, kernel {:.3f}), {:.3f} Mcycles, wall {:.2f} ms, memo {}", frame, REXCVAR_GET(gpu_d3d_wall), REXCVAR_GET(gpu_d3d_draw_record), seen,
              double(user - last_user + kernel - last_kernel) / 1e4 / n, double(user - last_user) / 1e4 / n, double(kernel - last_kernel) / 1e4 / n,
              double(cycles - last_cycles) / 1e6 / n, wall_ms / n, REXCVAR_GET(gpu_tex_bind_memo));
  if (const uint32_t ab = REXCVAR_GET(gpu_d3d_wall_ab)) REXCVAR_SET(gpu_d3d_wall, REXCVAR_GET(gpu_d3d_wall) ? 0u : ab);
  if (const int32_t ab = REXCVAR_GET(gpu_d3d_draw_record_ab)) REXCVAR_SET(gpu_d3d_draw_record, REXCVAR_GET(gpu_d3d_draw_record) ? 0 : ab);  // (Session 95, r3cmd)
  if (const int32_t ab = REXCVAR_GET(gpu_tex_bind_memo_ab)) REXCVAR_SET(gpu_tex_bind_memo, REXCVAR_GET(gpu_tex_bind_memo) ^ ab);  // (Session 95, r6tex)
  last_user = user;
  last_kernel = kernel;
  last_cycles = cycles;
  last_t = now;
  seen = 0;
#else
  (void)frame;
#endif
}
}  // namespace

bool CommandProcessor::D3dNoringHasWork() const {
  if (!REXCVAR_GET(gpu_d3d_noring)) return false;
  if (g_gate.open.load(std::memory_order_relaxed)) {  // (M5c) an open gate whose flip word has cleared is work: the release
    const uint32_t at = g_gate.flip_at.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    const uint32_t v = *reinterpret_cast<const volatile uint32_t*>(memory_->TranslatePhysical(at & ~uint32_t(0x3)));
    if (xenos::GpuSwap(v, static_cast<xenos::Endian>(at & 0x3)) == 0) return true;
  }
  const D3dNoringShared* s =
      reinterpret_cast<const D3dNoringShared*>(static_cast<uintptr_t>(REXCVAR_GET(gpu_d3d_noring_shared)));
  if (!s) return false;
  if (s != g_shared_checked.load(std::memory_order_acquire)) return s != g_shared_rejected.load(std::memory_order_relaxed);  // (not validated yet: Drain looks once)
  return s->head.load(std::memory_order_relaxed) != s->tail.load(std::memory_order_acquire);
}

// ExecutePrimaryBuffer's inner loop on host memory: the words are packets in HOST order (the reader byte-swaps, so they are
// swapped to guest order first). The drawing side runs a ring-level packet first (not a segment call), then the answers side.
bool CommandProcessor::D3dNoringRunPackets(const uint32_t* words, uint32_t count) {
  if (!count) return true;
  static thread_local std::vector<uint32_t> be;
  uint32_t* dst = nullptr;
  if (g_arena_item && !g_cut) {  // (M4f-C: under the cut the arena is never allocated or used)
    Arena& a = g_arena;
    if (!a.virtual_base && !a.failed) {
      // (Session 90: the arena is in the guest VIRTUAL heap, not the physical pool: a packet address is the reader pointer less the
      // physical base, which for a virtual address V is V again as a uint32 - unique, and nothing reads it as physical.)
      memory::BaseHeap* heap = memory_->LookupHeapByType(false, 4096);
      uint32_t address = 0;
      if (heap && heap->Alloc(kArenaBytes, 4096, memory::kMemoryAllocationReserve | memory::kMemoryAllocationCommit,
                              memory::kMemoryProtectRead | memory::kMemoryProtectWrite, true /* top down */, &address)) {
        a.virtual_base = address;
        a.size = kArenaBytes;
        free_counters::g_cut.arena_allocated = true;  // (M4f-C: the proof line "arena allocated: no")
        REXGPU_INFO("[d3d][noring] record arena: {} bytes at guest virtual {:08X}", kArenaBytes, address);
        memory::D3dLayerNoteGuestMemory("no-ring record arena", kArenaBytes, true, address);
      } else {
        a.failed = true;
        REXGPU_ERROR("[d3d][noring] the record arena could not be allocated: packets by value run from host memory (their guest "
                     "addresses are not unique; the scene crowd pass may stand down)");
      }
    }
    const uint32_t bytes = count * 4u;
    if (a.virtual_base && bytes <= a.size) {
      uint32_t use = (a.offset + 15u) & ~15u;
      if (use + bytes > a.size) {
        use = 0;  // wrap: the older items ran to completion already (synchronous)
        if (++a.wraps == 1 || a.wraps % 1000 == 0) REXGPU_INFO("[d3d][noring] record arena wrapped ({} times so far)", a.wraps);
      }
      a.offset = use + bytes;
      dst = memory_->TranslateVirtual<uint32_t*>(a.virtual_base + use);
    }
  }
  if (!dst) {
    be.resize(count);
    dst = be.data();
  }
  for (uint32_t i = 0; i < count; ++i) dst[i] = std::byteswap(words[i]);
  memory::RingBuffer reader(reinterpret_cast<uint8_t*>(dst), size_t(count) * sizeof(uint32_t));
  reader.set_write_offset(size_t(count) * sizeof(uint32_t));
  do {
    if (nf_ && nf_role_ == NfRole::kAnswers) {
      memory::RingBuffer peek = reader;
      const uint32_t header = peek.ReadAndSwap<uint32_t>();
      const uint32_t opcode = (header >> 8) & 0x7F;
      if (header && !((header >> 30) == 3 && (opcode == xenos::PM4_INDIRECT_BUFFER || opcode == xenos::PM4_INDIRECT_BUFFER_PFD))) {
        rex::graphics::free_counters::Bump(rex::graphics::free_counters::g_c.answers_walks);  // (Session 92 M0: log-only counter)
        NfRunRingPacketForDraw(reader);
      }
    }
    if (!ExecutePacket(&reader)) {
      REXGPU_ERROR("[d3d][noring] failed to execute a packet of a host queue item ({} dwords)", count);
      return false;
    }
  } while (reader.read_count());
  return true;
}

// One item. fields = the entry's dwords 1-6, payload = the dwords after them.
bool CommandProcessor::D3dNoringExecute(uint32_t kind, const uint32_t* f, const uint32_t* payload, uint32_t payload_dwords) {
  namespace fld = d3d_noring_field;
  D3dNoringShared& shared = *g_shared_checked.load(std::memory_order_acquire);
  // wwe2k14 patch (Session 87, stage 3 review fix 1): these items write registers outside ExecutePacket - the window scissor
  // 0x2080-0x2082 (kNrRegs, ~160 a frame), VGT_EVENT_INITIATOR and 0x5C8 (kNrFence and the handshakes) - so the record's register
  // image (gpu_d3d_own_input) would stay "valid" with the old values and a later flush would overwrite them. With
  // gpu_d3d_own_copy on, kNrRegs and kNrFence run in an image scope: their writes go into the image (notified, pending for the
  // register file), the register file is not brought up to date. The handshakes / interrupts / scene timing write only scratch
  // registers and 0x5C8 (no derivation reads it) and wait or call back for long, so they leave the image alone. With the mode
  // off, every one of them first brings the register file up to date and ends the image's validity (D3dFeedForeign).
  // (Session 87, textures: a Registers item made only of texture-table ops (Define / Destroy / Changed / Bind, ops 16-22) writes no
  // register: it has nothing to bring up to date and cannot make the record's image stale, so it takes neither the image scope nor the
  // foreign path. The texture ops are pushed from the game's create / destroy hooks at any time, e.g. a scene's loads.)
  const bool texture_ops_only = [&] {
    if (kind != kNrRegs || !payload_dwords || (payload_dwords & 3u)) return false;
    for (uint32_t i = 0; i < payload_dwords; i += 4) {
      if (payload[i] < d3d_tex::kOpDefine0 || payload[i] > d3d_tex::kOpBind) return false;
    }
    return true;
  }();
  if (kind == kNrRegs && payload_dwords >= 4) {  // (diagnostic: what a foreign flush by this item prints)
    D3dFeedItemInfo() = (payload[0] << 24) | (std::min<uint32_t>(payload_dwords / 4, 255u) << 16) | (payload[1] & 0xFFFFu);
  }
  struct FeedScopeGuard {
    CommandProcessor* cp;
    bool open = false;
    ~FeedScopeGuard() {
      if (open) D3dFeedScopeEnd(*cp);
    }
  } feed_scope{this};
  if (nf_role_ != NfRole::kAnswers &&  // (under the native frame the answers side's register file is the real one: not the image's)
      !texture_ops_only &&
      (kind == kNrFence || kind == kNrFlipHandshake || kind == kNrSceneTiming || kind == kNrInterrupt || kind == kNrRegs)) {
    D3dFeedLastPacket() = (2u << 8) | kind;  // (census key: "type 2" = a no-ring item, opcode = its kind)
    if (D3dFeedScopeWanted(*this)) {
      if (kind == kNrFence || kind == kNrRegs) feed_scope.open = D3dFeedScopeBegin(*this);
    } else {
      D3dFeedForeign(*this);
    }
  }
  auto word_at = [&](uint32_t gpu_address) {  // a memory poll's value, as WAIT_REG_MEM reads it
    uint32_t v = *reinterpret_cast<uint32_t*>(memory_->TranslatePhysical(gpu_address & ~uint32_t(0x3)));
    return xenos::GpuSwap(v, static_cast<xenos::Endian>(gpu_address & 0x3));
  };
  // The wait the library's writers put after a handshake: WAIT_REG_MEM, memory, function "equal", value, mask all ones, wait 256.
  auto wait_memory_equal = [&](uint32_t gpu_address_with_endian, uint32_t value) {
    const uint32_t wait[6] = {0xC0043C00u, 0x13u, gpu_address_with_endian, value, 0xFFFFFFFFu, 256u};
    return D3dNoringRunPackets(wait, 6);
  };
  // EVENT_WRITE_SHD's body for the segment-end pair (initiator 3: a value, not the counter).
  auto event_write_value = [&](uint32_t address, uint32_t value) {
    WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, 3u & 0x3F);
    perf::timeline::Record("evw", address, value, 3u);
    OnGuestVisibleWrite(GuestVisibleEvent::kEventWriteShd, address & ~uint32_t(0x3));
    const auto endianness = static_cast<xenos::Endian>(address & 0x3);
    const uint32_t swapped = xenos::GpuSwap(value, endianness);
    memory::store(memory_->TranslatePhysical(address & ~uint32_t(0x3)), swapped);
  };

  const bool direct_latch = REXCVAR_GET(gpu_d3d_sync_direct);  // (Session 92, M5a)
  const bool direct = direct_latch;  // (the segment piece log and the fence; the handshake cases re-decide per item, S1)
  // The fence's two stores without the register write (M5a): the readbacks land first, then the word.
  auto event_write_direct = [&](uint32_t address, uint32_t value) {
    perf::timeline::Record("evw", address, value, 3u);
    if (nf_role_ != NfRole::kAnswers) RepeatNotePacket(xenos::PM4_EVENT_WRITE_SHD);  // (S2: the old path's register write noted it)
    OnGuestVisibleWrite(GuestVisibleEvent::kEventWriteShd, address & ~uint32_t(0x3));
    const auto endianness = static_cast<xenos::Endian>(address & 0x3);
    memory::store(memory_->TranslatePhysical(address & ~uint32_t(0x3)), xenos::GpuSwap(value, endianness));
  };
  // A scratch register's write-back (SCRATCH_REG0 + n stored at the scratch block + 4n, byte-swapped) without the register file.
  auto scratch_store = [&](uint32_t scratch_block, uint32_t n, uint32_t value) {
    if (nf_role_ != NfRole::kAnswers || (nf_ && REXCVAR_GET(gpu_native_frame_draw))) RepeatNotePacket(xenos::PM4_WAIT_REG_MEM);  // (S2: a sync item ends a repeat)
    OnGuestVisibleWrite(GuestVisibleEvent::kScratchWriteback);
    // (H1) the register file keeps the scratch value as WriteRegister left it (the guest can read it through MMIO); a plain store.
    const_cast<volatile uint32_t&>(register_file_->values[XE_GPU_REG_SCRATCH_REG0 + n]) = value;
    memory::store_and_swap<uint32_t>(memory_->TranslatePhysical(scratch_block + n * 4u), value);
  };

  switch (kind) {
    case kNrSegment: {
      const uint32_t length = f[fld::kSegLength];
      if (!length) return true;  // (an empty piece: nothing to run; an empty indirect buffer would read a packet that is not there)
      const uint32_t opcode = (f[fld::kSegFlags] & 1) ? xenos::PM4_INDIRECT_BUFFER_PFD : xenos::PM4_INDIRECT_BUFFER;
      const uint32_t packet[3] = {0xC0010000u | (opcode << 8), f[fld::kSegAddress], length & 0xFFFFFu};
      shared.segment_dwords.fetch_add(length, std::memory_order_relaxed);
      if (g_cut) {
        // (M4f-C) segment pieces under the cut: after the latch (the owned frame's mode is set) the bar is 0, and the packets in any that run are named
        free_counters::CutCounters& cc = free_counters::g_cut;
        if (!free_counters::g_owned_only_frame) {
          ++cc.segments_boot;
        } else {
          ++cc.segments_after;
          if (length <= 65536) {
            const uint32_t* words = reinterpret_cast<const uint32_t*>(memory_->TranslatePhysical(xenos::GpuToCpu(f[fld::kSegAddress]) & 0x1FFFFFFF));
            for (uint32_t i = 0; i < length;) {
              const uint32_t h = std::byteswap(words[i]);
              uint32_t n = 1;
              if ((h >> 30) == 3) {
                n = ((h >> 16) & 0x3FFF) + 2;
                ++cc.segment_opcode[(h >> 8) & 0x7F];
              } else if ((h >> 30) == 0) {
                n = ((h >> 16) & 0x3FFF) + 2;
                ++cc.segment_type0;
                ++cc.segment_t0_base[h & 0x7FFF];
              } else {
                n = (h >> 30) == 1 ? 3 : 1;  // a type-1 packet is 3 dwords, a type-2 nop 1
                ++cc.segment_other;
              }
              i += n;
            }
          }
        }
      }
      if (direct) {  // (M5a, log only: the first four pieces' packets - confirms or refutes the coherency-handshake inference)
        static uint32_t logged = 0;
        static bool seen_11 = false, seen_late = false;
        const bool log_11 = length == 11 && !seen_11;
        const bool log_late = !log_11 && g_present_census >= 600 && !seen_late;
        if ((log_11 || log_late) && length <= 64) {  // (H3b: by content - the first 11-dword piece, and the first after the 600th present)
          if (log_11) seen_11 = true; else seen_late = true;
          ++logged;
          const uint32_t* words = reinterpret_cast<const uint32_t*>(memory_->TranslatePhysical(xenos::GpuToCpu(f[fld::kSegAddress]) & 0x1FFFFFFF));
          std::string text;
          for (uint32_t i = 0; i < length;) {
            const uint32_t h = std::byteswap(words[i]);
            char buf[96];
            uint32_t n = 1;
            if ((h >> 30) == 3) {
              n = ((h >> 16) & 0x3FFF) + 2;
              std::snprintf(buf, sizeof buf, " [T3 op %02X cnt %u:", (h >> 8) & 0x7F, n - 1);
            } else if ((h >> 30) == 0) {
              n = ((h >> 16) & 0x3FFF) + 2;
              std::snprintf(buf, sizeof buf, " [T0 reg %04X cnt %u:", h & 0x7FFF, n - 1);
            } else {
              std::snprintf(buf, sizeof buf, " [T%u %08X:", h >> 30, h);
            }
            text += buf;
            for (uint32_t j = 1; j < n && i + j < length; ++j) {
              std::snprintf(buf, sizeof buf, " %08X", std::byteswap(words[i + j]));
              text += buf;
            }
            text += "]";
            i += n;
          }
          REXGPU_INFO("[d3d][sync] segment piece {} at {:08X}, {} dwords, flags {}:{}", logged, f[fld::kSegAddress], length, f[fld::kSegFlags], text);
        }
      }
      return D3dNoringRunPackets(packet, 3);
    }
    case kNrPackets: {
      // A record token by value (stage 2c). Its draws know their library kind: the tag is the token's opcode (0x7C an indexed
      // draw = kind 1, 0x7E an inline draw = kind 2, anything else a record that draws nothing by itself).
      shared.packet_dwords.fetch_add(payload_dwords, std::memory_order_relaxed);
      if (g_cut) ++(free_counters::g_owned_only_frame ? free_counters::g_cut.packets_records_after : free_counters::g_cut.packets_records_boot);  // (M4f-C: the bar is 0 after the latch)
      const uint32_t opcode = f[fld::kPktTag];
      g_record_kind = opcode == 0x7C ? 1u : opcode == 0x7E ? 2u : 0xFFu;
      g_arena_item = true;
      const bool ok = D3dNoringRunPackets(payload, payload_dwords);
      g_arena_item = false;
      g_record_kind = 0;
      return ok;
    }
    case kNrPresent: {
      shared.packet_dwords.fetch_add(payload_dwords, std::memory_order_relaxed);
      bool ok;
      ++g_gs.presents_run;
      WallStatsPresent(g_gs.presents_run);
      // (Session 92, M5c) behind an open pacing gate the present runs its frame end and submission, but its host publish is held
      const bool hold_publish = g_gate.open.load(std::memory_order_relaxed);
      if (hold_publish) {
        ui::Presenter::HoldGuestOutputPublish(true);
        ++g_gs.presents_gated;
      }
      // (Session 92, M3b) The present as data: VdSwap's words are a type-0 packet writing fetch constant 0 (6 dwords: the swap texture) and the
      // XE_SWAP packet (signature, front buffer, width, height). Under the latch the fetch goes to the present directly (a file-static
      // override read by RequestSwapTexture) - no register / image write, no packet. Not under the native frame (its two sides run packets
      // differently) or an active tiling log; a payload that is not exactly those words takes the packet path.
      bool as_data = false;
      if (REXCVAR_GET(gpu_d3d_owned_present) && free_counters::g_owned_only_frame && !nf_ && !tiling_log_active_) {
        if (payload_dwords == 12 && payload[0] == xenos::MakePacketType0(XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0, 6) &&
            payload[7] == xenos::MakePacketType3(xenos::PM4_XE_SWAP, 4) && payload[8] == xenos::kSwapSignature) {
          as_data = true;
        } else if (++g_present_unexpected <= 4) {
          REXGPU_WARN("[d3d][owned-present] present item of {} dwords is not the expected words (word 0 {:08X}, 7 {:08X}, 8 {:08X}): the packet path runs it",
                      payload_dwords, payload_dwords > 0 ? payload[0] : 0, payload_dwords > 7 ? payload[7] : 0, payload_dwords > 8 ? payload[8] : 0);
        }
      }
      if (as_data) {
        ++g_present_data;
        uint32_t fetch[6];
        std::memcpy(fetch, payload + 1, sizeof(fetch));
        free_counters::g_swap_fetch_override = fetch;
        free_counters::NotePresentForFetch0();
        if (nf_role_ != NfRole::kAnswers) RepeatNotePacket(xenos::PM4_XE_SWAP);  // (the packet path's: a swap ends a repeat)
        ++frame_counters_.type3_packets;
        ok = PresentFrame(payload[9], payload[10], payload[11]);
        free_counters::g_swap_fetch_override = nullptr;
      } else {
        ++g_present_packet_path;
        g_arena_item = true;
        ok = D3dNoringRunPackets(payload, payload_dwords);
        g_arena_item = false;
      }
      if (hold_publish) {
        ui::Presenter::HoldGuestOutputPublish(false);
        if (ui::Presenter::GuestOutputPublishHeld()) {
          g_gate.held = true;
          ++g_gs.presents_held;
        } else {
          ++g_gs.presents_not_held;  // (no refresh reached the presenter: an early return of IssueSwap, or the output inactive)
        }
      }
      if (g_poison_armed && g_present_census % 600 == 599) {  // (detector 1: a writer that survived rewrote a poisoned register)
        free_counters::g_cut.poison_rewritten += OwnedRecordHookSlots().poison(register_file_->values, true);
      }
      if (++g_present_census % 600 == 0) {
        uint64_t n[kNrKindCount];
        for (uint32_t k = 0; k < kNrKindCount; ++k) n[k] = shared.items[k].load(std::memory_order_relaxed);
        REXGPU_INFO("[d3d][noring] consumer after {} presents: segments {} ({} dwords executed from them), packet items {}, fences {} "
                    "({} written), flip handshakes {} ({} waited, {:.1f} ms in the waits), scene timings {}, other interrupts {}, "
                    "barriers {}, errors {}, sync items before the scratch setup {}",
                    g_present_census, n[kNrSegment], shared.segment_dwords.load(std::memory_order_relaxed), n[kNrPackets],
                    n[kNrFence], shared.fences.load(std::memory_order_relaxed), n[kNrFlipHandshake],
                    shared.flip_waits.load(std::memory_order_relaxed),
                    double(shared.flip_wait_us.load(std::memory_order_relaxed)) / 1000.0, n[kNrSceneTiming], n[kNrInterrupt],
                    n[kNrBarrier], shared.bad.load(std::memory_order_relaxed), g_sync_before_setup);
        if (REXCVAR_GET(gpu_d3d_owned_present)) {  // (M3b)
          REXGPU_INFO("[d3d][owned-present] after {} presents: as data {}, packet path {}, payload not as expected {}", g_present_census, g_present_data,
                      g_present_packet_path, g_present_unexpected);
        }
        if (REXCVAR_GET(gpu_d3d_pace_gate)) {  // (M5c, log only)
          const auto hs = ui::Presenter::GetHoldStats();
          REXGPU_INFO("[d3d][pace] after {} presents (run {}): gates opened {} (released: immediately {}, between items {}, vblank while idle {}, at the "
                      "next handshake {}, before the ring {}), fences deferred {} (max queued {}, written {}), CP waits at a second handshake {} ({:.1f} ms), "
                      "before the ring {}, wait skipped at the handshake {:.1f} ms, presents under a gate {} (held {}, no frame to hold {}), publishes deferred past an open scene {} (published at its end {}), presenter: "
                      "published now {}, held {}, published from hold {}, forced {}, still held {}",
                      g_present_census, g_gs.presents_run, g_gs.opened, g_gs.rel_immediate, g_gs.rel_poll, g_gs.rel_vblank_idle, g_gs.rel_second_hs,
                      g_gs.rel_ring, g_gs.fences_deferred, g_gs.max_queue, g_gs.fences_written, g_gs.second_hs_waits,
                      double(g_gs.second_hs_wait_us) / 1000.0, g_gs.ring_waits, double(g_gs.skipped_wait_us) / 1000.0, g_gs.presents_gated,
                      g_gs.presents_held, g_gs.presents_not_held, g_gs.pub_deferred_scene, g_gs.pub_at_scene_end, hs.published_now, hs.held, hs.published_held, hs.forced,
                      ui::Presenter::GuestOutputPublishHeld() ? 1 : 0);
        }
        {  // (Session 93, log only; the Adreno 720 stock-driver freeze) presents still arrive but the painting thread is inside a host call
          const char* call = nullptr;
          const uint64_t call_ms = ui::Presenter::HostPresentCallMs(&call);
          if (call_ms >= 250) {
            REXGPU_WARN("[pace][present][stuck] after {} presents: the painting thread has been inside {} for {} ms (the guest frames are made, "
                        "none reaches the screen)", g_present_census, call ? call : "?", call_ms);
          }
        }
        if (REXCVAR_GET(gpu_d3d_sync_direct)) {  // (M5a diagnostics, log only)
          REXGPU_INFO("[d3d][sync] per {} presents: guest MMIO reads by register, top 8: {}; WriteRegister calls by register, top 8: {}",
                      600, free_counters::TakeTop8(free_counters::g_mmio_reads), free_counters::TakeTop8(free_counters::g_register_writes));
        }
      }
      return ok;
    }
    case kNrFence: {
      if (direct && g_gate.open.load(std::memory_order_relaxed)) {
        // (M5c) behind an open pacing gate: queued in order, written at the release (never earlier: the game reuses memory on a completed fence)
        if (nf_role_ != NfRole::kAnswers) RepeatNotePacket(xenos::PM4_EVENT_WRITE_SHD);  // (S2: as the immediate path's, at the item's place)
        g_gate.q.push_back({f[fld::kFenceMarkerAt], f[fld::kFenceMarker], f[fld::kFenceValueAt], f[fld::kFenceValue]});
        ++g_gs.fences_deferred;
        g_gs.max_queue = std::max<uint64_t>(g_gs.max_queue, g_gate.q.size());
        return true;
      }
      if (direct) {  // (M5a: no 0x5C8 / VGT_EVENT_INITIATOR writes - no reader)
        event_write_direct(f[fld::kFenceMarkerAt], f[fld::kFenceMarker]);
        event_write_direct(f[fld::kFenceValueAt], f[fld::kFenceValue]);
        shared.fences.fetch_add(1, std::memory_order_relaxed);
        return true;
      }
      if (f[fld::kFenceFlags] & 1) WriteRegister(0x5C8, 0x20000u);  // the segment end's coherency write (as the packet pair's)
      event_write_value(f[fld::kFenceMarkerAt], f[fld::kFenceMarker]);  // the heap marker first, then the fence number
      event_write_value(f[fld::kFenceValueAt], f[fld::kFenceValue]);
      shared.fences.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
    case kNrFlipHandshake:
    case kNrSceneTiming:
    case kNrInterrupt: {
      // (M5c) a gate still open from the previous frame's handshake: its flip word is waited for (as M5a / M5b) and released first
      if (kind == kNrFlipHandshake && !D3dNoringGateWaitRelease(true)) return false;  // (also flushes a publish deferred past a scene)
      if (kind == kNrFlipHandshake) g_scene_open = false;  // (the present closed any open scene before the handshake)
      const uint32_t raw_flags = f[fld::kIntFlags];
      const uint32_t options = f[fld::kIntOptions];
      const uint32_t scratch = f[fld::kIntScratch];
      // (S1) the direct stores assume the library's scratch block is set up as the item says; before that, the old path.
      bool direct = direct_latch;
      if (direct) {
        const uint32_t sa = register_file_->values[XE_GPU_REG_SCRATCH_ADDR], um = register_file_->values[XE_GPU_REG_SCRATCH_UMSK];
        if (sa != scratch || (um & 0x33) != 0x33) {
          direct = false;
          if (g_sync_before_setup++ == 0) {
            REXGPU_INFO("[d3d][sync] first sync item before the scratch setup: SCRATCH_ADDR {:08X}, UMSK {:08X}, item scratch {:08X}", sa, um, scratch);
          }
        }
      }
      uint32_t cpu_mask = (raw_flags >> 24) & 0x3F;
      if (!cpu_mask) cpu_mask = 4;
      uint32_t flags = raw_flags;
      if (!(flags & 6)) flags |= 6;  // (the emitter's default: callback registers + interrupt + the wait for the CPU)
      if (kind == kNrFlipHandshake && (options & 1)) {  // the flip word := 1
        if (direct) scratch_store(scratch, 1, 1);
        else WriteRegister(XE_GPU_REG_SCRATCH_REG1, 1);
      }
      if (flags & 2) {
        if (direct) {  // (M5a: scratch+16 callback, +20 argument, +0 cpu mask; no 0x5C8)
          scratch_store(scratch, 4, f[fld::kIntCallback]);
          if (!(flags & 0x80)) scratch_store(scratch, 5, f[fld::kIntArgument]);
          scratch_store(scratch, 0, cpu_mask);
        } else {
          if (options & 4) WriteRegister(0x5C8, 0x20000u);
          WriteRegister((XE_GPU_REG_SCRATCH_REG0 + 4), f[fld::kIntCallback]);
          if (!(flags & 0x80)) WriteRegister((XE_GPU_REG_SCRATCH_REG0 + 5), f[fld::kIntArgument]);
          WriteRegister(XE_GPU_REG_SCRATCH_REG0, cpu_mask);
        }
        // (the library's waits for these three write-backs to land are self-satisfied: WriteRegister stores synchronously)
        OnGuestVisibleWrite(GuestVisibleEvent::kInterrupt);
        for (int n = 0; n < 6; n++) {
          if (!(cpu_mask & (1u << n)) || !graphics_system_) continue;
          if (det::SerialEnabled()) {  // as ExecutePacketType3_INTERRUPT: the guest routine runs between guest turns
            SetDetWaitKind(4);
            det::SerialRunExternal([&] { graphics_system_->DispatchInterruptCallback(1, n); }, true);
            SetDetWaitKind(0);
          } else {
            graphics_system_->DispatchInterruptCallback(1, n);
          }
        }
        if (!direct) WriteRegister(XE_GPU_REG_CALLBACK_ACK, cpu_mask);
      }
      if (flags & 4) {  // every targeted CPU has run it: the dispatcher cleared its bit of scratch+0
        if (direct) {
          if (word_at(scratch | 2u) != 0 && !WaitGuestWord(scratch | 2u, 0)) return false;
          scratch_store(scratch, 4, 0x0BADF00Du);
        } else {
          if (word_at(scratch | 2u) != 0 && !wait_memory_equal(scratch | 2u, 0)) return false;
          WriteRegister((XE_GPU_REG_SCRATCH_REG0 + 4), 0x0BADF00Du);
        }
      }
      if (kind == kNrFlipHandshake && (options & 2)) {
        // THE pacing wait: the flip word is cleared by the flip scheduler at once (a plain present) or by the vblank handler
        // at the due vblank. The ring's wait policy (1 ms sleeps with vsync, yields without, the det wait reporting) is the
        // WAIT_REG_MEM handler's own.
        const uint32_t flip_at = (scratch + 4u) | 2u;
        const bool will_wait = word_at(flip_at) != 0;
        if (will_wait && direct && GateOn()) {
          // (M5c) no wait: the gate opens; the first check that finds the flip word at 0 releases it. The flip word was just set by this item
          // (options & 1) and the callback has run, so it is still set only when the scheduler queued the flip for a later vblank.
          g_gate.flip_at.store(flip_at, std::memory_order_relaxed);
          g_gate.t_open = std::chrono::steady_clock::now();
          g_gate.items_since = 0;
          g_gate.slept = false;
          g_gate.held = false;
          g_gate.q.clear();
          g_gate.open.store(true, std::memory_order_release);
          ++g_gs.opened;
          perf::timeline::Record("gate_o", flip_at, uint32_t(g_gs.opened), 0u);
          return true;
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (!(direct ? WaitGuestWord(flip_at, 0) : wait_memory_equal(flip_at, 0))) return false;
        if (will_wait) {
          shared.flip_waits.fetch_add(1, std::memory_order_relaxed);
          shared.flip_wait_us.fetch_add(uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                                                     std::chrono::steady_clock::now() - t0).count()),
                                        std::memory_order_relaxed);
        }
      }
      if (kind == kNrSceneTiming) {  // (M5c, G1) the scene's open / closed state; a deferred publish goes out when the scene ends
        const uint32_t cb = f[fld::kIntCallback];
        if (cb == kSceneBeginCb) {
          g_scene_open = true;
        } else if (cb == kSceneEndCb) {
          g_scene_open = false;
          if (ui::Presenter::GuestOutputPublishHeld() && !g_gate.open.load(std::memory_order_relaxed)) {
            D3dNoringFlushHeldPublish();
            ++g_gs.pub_at_scene_end;
          }
        }
      }
      return true;
    }
    case kNrRegs: {
      // Plain register writes, with the packet loop's own split between the two sides of the native frame: a ring-level packet runs on
      // the DRAWING side first (its own register file: render state and constants are its alone - the window scissor 0x2080-0x2082 the
      // owned frame reads), then on the answers side, which in the lean mode stores only registers below 0x2000
      // (ExecutePacketType0Body). Writing through WriteRegister on the answers side only left the drawing side's scissor stale.
      struct Op {
        uint32_t op, reg, a, b;
      };
      auto apply = [&](bool answers_side) {
        d3d_tex::WriterScope tex_writer(d3d_tex::kWRegsItem);  // wwe2k14 patch (Session 87, stage 4)
        for (uint32_t i = 0; i + 4 <= payload_dwords; i += 4) {
          const Op o{payload[i], payload[i + 1], payload[i + 2], payload[i + 3]};
          if (o.op >= d3d_tex::kOpDefine0 && o.op <= d3d_tex::kOpBind) {
            // wwe2k14 patch (Session 87, D3D layer stage 4): the texture object table's ops, applied once (the drawing side).
            if (!answers_side && d3d_tex::Enabled()) {
              d3d_tex::Table& table = d3d_tex::Table::Get();
              if (o.op <= d3d_tex::kOpDefine3) {
                table.Define(o.op - d3d_tex::kOpDefine0, o.reg, o.a, o.b, 0);
              } else if (o.op == d3d_tex::kOpDestroy) {
                table.Destroy(o.reg);
              } else if (o.op == d3d_tex::kOpChanged) {
                table.Changed(o.reg);
              } else {
                table.Bind(o.reg, o.a);
              }
            }
            continue;
          }
          if (answers_side && nf_answers_lean_ && o.reg >= 0x2000) continue;
          if (o.op >= d3d_noring_field::kRegsOpBinMaskLo && o.op <= d3d_noring_field::kRegsOpBinSelectHi) {
            // The tiling predication state, as the four SET_BIN_* packet cases of ExecutePacketType3 set it (each side has its own).
            if (o.op == d3d_noring_field::kRegsOpBinMaskLo) {
              bin_mask_ = (bin_mask_ & 0xFFFFFFFF00000000ull) | o.a;
              BinMaskWritten("mask_lo");
            } else if (o.op == d3d_noring_field::kRegsOpBinMaskHi) {
              bin_mask_ = (bin_mask_ & 0xFFFFFFFFull) | (static_cast<uint64_t>(o.a) << 32);
              BinMaskWritten("mask_hi");
            } else {
              const uint64_t old_select = bin_select_;
              if (o.op == d3d_noring_field::kRegsOpBinSelectLo) {
                bin_select_ = (bin_select_ & 0xFFFFFFFF00000000ull) | o.a;
                BinSelectWritten("sel_lo", old_select);
              } else {
                bin_select_ = (bin_select_ & 0xFFFFFFFFull) | (static_cast<uint64_t>(o.a) << 32);
                BinSelectWritten("sel_hi", old_select);
              }
            }
          } else if (o.op == d3d_noring_field::kRegsOpRmw) {
            const uint32_t v = register_file_->values[o.reg & 0x1FFF];  // (as ExecutePacketType3_REG_RMW)
            WriteRegister(o.reg & 0x1FFF, (v & o.a) | o.b);
          } else {
            WriteRegister(o.reg, o.a);
          }
        }
      };
      if (nf_ && nf_role_ == NfRole::kAnswers) {
        if (REXCVAR_GET(gpu_native_frame_draw)) {
          ++nf_census_.ring_packets;
          NfEnterDraw();
          // (Session 87 stage 3, found by diagnosis run A/B on the picture failure: this item runs on the ANSWERS side, so the
          // top-of-function scope was skipped, but the writes below are the DRAWING side's own - the window scissor the owned frame
          // reads - and went straight into the drawing side's register file while the record's image stayed "valid" with the old
          // scissor. With gpu_d3d_own_copy the image is not re-copied after every copy token any more, so the stale scissor
          // showed as a wrong picture. They are applied in an image scope; with the mode off the register file is brought up to
          // date and the image ends stale.)
          bool image_scope = false;
          if (!texture_ops_only) {  // (texture-table ops write no register)
            D3dFeedLastPacket() = (2u << 8) | kNrRegs;
            if (D3dFeedScopeWanted(*this)) {
              image_scope = D3dFeedScopeBegin(*this);
            } else {
              D3dFeedForeign(*this);
            }
          }
          apply(false);
          if (image_scope) D3dFeedScopeEnd(*this);
          NfLeaveDraw();
        }
        apply(true);
      } else {
        apply(false);
      }
      return true;
    }
    case kNrBarrier:
      // (M5c) the hand-over drains everything before it: an open pacing gate is waited for and released (queued fences written) first
      if (!D3dNoringGateWaitRelease(false)) return false;
      shared.consumed_ticket.store(f[fld::kBarrierTicket], std::memory_order_release);
      return true;
    default:
      shared.bad.fetch_add(1, std::memory_order_relaxed);
      REXGPU_ERROR("[d3d][noring] unknown item kind {}: skipped", kind);
      return true;
  }
}

// (Session 92, M5c) The pacing gate. All on the command thread.
// One load of the flip word (the value as the wait reads it); at 0 releases the gate. True = released.
// (Session 93, log only; the Adreno 720 stock-driver freeze) a gate still open after 500 ms: reported once per gate, then every 5 s while it stays
// open (64 lines at most), with what it waits for. The clock is read every 32nd poll only (polls run between items while the gate is open).
static void GateLongOpenNote(uint32_t flip_value, bool idle) {
  static uint32_t polls = 0;
  if ((++polls & 31) != 0) return;
  static uint64_t logged_gate = 0;
  static std::chrono::steady_clock::time_point last_log;
  static uint32_t lines = 0;
  if (lines >= 64) return;
  const auto now = std::chrono::steady_clock::now();
  const int64_t open_ms = int64_t(std::chrono::duration_cast<std::chrono::milliseconds>(now - g_gate.t_open).count());
  if (open_ms < 500) return;
  if (logged_gate == g_gs.opened && now - last_log < std::chrono::seconds(5)) return;
  logged_gate = g_gs.opened;
  last_log = now;
  ++lines;
  const char* call = nullptr;
  const uint64_t call_ms = ui::Presenter::HostPresentCallMs(&call);
  REXGPU_WARN("[d3d][pace][stuck] gate {} open for {} ms: flip word {:08X} at {:08X} (released at 0), polled {}, items since it opened {}, the loop "
              "slept {}, scene open {}, fences queued {}, frame held by this gate {} (presenter holds one {}); host present call in progress: {} {} ms",
              g_gs.opened, open_ms, flip_value, g_gate.flip_at.load(std::memory_order_relaxed), idle ? "idle" : "between items", g_gate.items_since,
              g_gate.slept ? 1 : 0, g_scene_open ? 1 : 0, g_gate.q.size(), g_gate.held ? 1 : 0, ui::Presenter::GuestOutputPublishHeld() ? 1 : 0,
              call_ms ? call : "none", call_ms);
}

bool CommandProcessor::D3dNoringGatePoll(bool idle) {
  if (!g_gate.open.load(std::memory_order_relaxed)) return false;
  const uint32_t at = g_gate.flip_at.load(std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_acquire);
  uint32_t v = *reinterpret_cast<volatile uint32_t*>(memory_->TranslatePhysical(at & ~uint32_t(0x3)));
  v = xenos::GpuSwap(v, static_cast<xenos::Endian>(at & 0x3));
  if (v != 0) {
    GateLongOpenNote(v, idle);  // (Session 93, log only)
    return false;
  }
  // (G1) between items inside an open scene the fences are written but the host publish waits for the scene's end (the game's scene timer
  // would otherwise measure the paint / a present block); idle callers and waits publish at once
  const bool publish = idle || !g_scene_open || g_gate.slept;
  D3dNoringGateRelease(g_gate.slept ? 2 : (g_gate.items_since == 0 ? 0 : 1), publish);
  return true;
}

// Waits for the flip word (the M5a / M5b wait, det kind 2 included) and releases. second_handshake: the next handshake reached with the gate
// open (counted with its wait); otherwise the hand-over to the ring. False when the worker stopped.
bool CommandProcessor::D3dNoringGateWaitRelease(bool second_handshake) {
  if (!g_gate.open.load(std::memory_order_relaxed)) {
    if (ui::Presenter::GuestOutputPublishHeld()) D3dNoringFlushHeldPublish();  // (G1) a publish deferred past a scene
    return true;
  }
  const uint32_t at = g_gate.flip_at.load(std::memory_order_relaxed);
  uint32_t v = *reinterpret_cast<volatile uint32_t*>(memory_->TranslatePhysical(at & ~uint32_t(0x3)));
  v = xenos::GpuSwap(v, static_cast<xenos::Endian>(at & 0x3));
  const bool will_wait = v != 0;
  const auto t0 = std::chrono::steady_clock::now();
  if (!WaitGuestWord(at, 0)) return false;
  const uint64_t us = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
  if (will_wait) {
    if (second_handshake) {
      ++g_gs.second_hs_waits;
      g_gs.second_hs_wait_us += us;
    } else {
      ++g_gs.ring_waits;
    }
  }
  D3dNoringGateRelease(second_handshake ? 3 : 4);
  return true;
}

// how: 0 = immediately (nothing ran since it opened), 1 = between items, 2 = on a vblank-event idle wait, 3 = at the next handshake, 4 = before the ring.
// Order: the held frame is published, then the queued fences are written in the order they were decoded, then the gate closes.
void CommandProcessor::D3dNoringFlushHeldPublish() {
  if (ui::Presenter::PublishHeldGuestOutput()) ++g_gs.published_at_release;
}

void CommandProcessor::D3dNoringGateRelease(int how, bool publish) {
  if (!g_gate.open.load(std::memory_order_relaxed)) return;
  g_gs.skipped_wait_us += uint64_t(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - g_gate.t_open).count());
  if (publish) {
    if (ui::Presenter::PublishHeldGuestOutput()) ++g_gs.published_at_release;
  } else if (ui::Presenter::GuestOutputPublishHeld()) {
    ++g_gs.pub_deferred_scene;  // (G1) stays held; published at the scene's end, the idle poll, the next handshake, the ring or the barrier
  }
  D3dNoringShared* shared = g_shared_checked.load(std::memory_order_acquire);
  for (const GateFence& q : g_gate.q) {
    for (int part = 0; part < 2; ++part) {
      const uint32_t address = part == 0 ? q.marker_at : q.value_at, value = part == 0 ? q.marker : q.value;
      perf::timeline::Record("evw", address, value, 3u);
      OnGuestVisibleWrite(GuestVisibleEvent::kEventWriteShd, address & ~uint32_t(0x3));
      memory::store(memory_->TranslatePhysical(address & ~uint32_t(0x3)),
                    xenos::GpuSwap(value, static_cast<xenos::Endian>(address & 0x3)));
    }
    if (shared) shared->fences.fetch_add(1, std::memory_order_relaxed);
    ++g_gs.fences_written;
  }
  perf::timeline::Record("gate_r", uint32_t(how), uint32_t(g_gate.q.size()), 0u);
  g_gate.q.clear();
  g_gate.held = false;
  g_gate.open.store(false, std::memory_order_release);
  switch (how) {
    case 0: ++g_gs.rel_immediate; break;
    case 1: ++g_gs.rel_poll; break;
    case 2: ++g_gs.rel_vblank_idle; break;
    case 3: ++g_gs.rel_second_hs; break;
    default: ++g_gs.rel_ring; break;
  }
}

// Pops and runs everything queued. True if anything ran.
bool CpSafePointDue();  // command_processor.cpp (WWE '13 port)
bool CommandProcessor::D3dNoringDrain() {
  D3dNoringShared* s = Shared();
  if (!s) return false;
  // (Session 92, M4f-C) THE CUT is latched ONCE, at the first attach (never re-read): bit 128, the executor wired (run slot and all four entry points),
  // and not the native frame (nf_). Before any item runs: the feed is flushed and ended (no stale image), the record arena is never allocated.
  static bool cut_latched = false;
  if (!cut_latched) {
    cut_latched = true;
    const OwnedRecordHooks& h = OwnedRecordHookSlots();
    const uint32_t front_bits = REXCVAR_GET(gpu_d3d_owned_front);
    const bool wired = h.run != nullptr && h.entries_wired != nullptr && h.entries_wired();
    // (the entries depend on: B's owned clears and the typed draw (bits 32 + 64), the owned-only latch, the owned resolve and the present as data)
    const bool deps = (front_bits & 96u) == 96u && REXCVAR_GET(gpu_native_owned_only) && REXCVAR_GET(gpu_d3d_owned_resolve) && REXCVAR_GET(gpu_d3d_owned_present);
    g_cut = (front_bits & 128u) != 0 && wired && deps && !nf_;
    free_counters::g_cut.active = g_cut;
    if (g_cut) {
      D3dFeedFlush(*this);
      D3dFeedState() = 0;
      CutEnforceSettings(true);
      REXGPU_INFO("[d3d][noring] THE CUT is on (gpu_d3d_owned_front bit 128): the decoder executes the queue's packet and register items in order; "
                  "the record arena is never allocated, the D3D feed is off, an item it cannot take is dropped and counted");
    } else if (front_bits & 128u) {
      REXGPU_INFO("[d3d][noring] the cut (bit 128) is NOT latched: run slot {}, entries wired {}, native frame {} (must be 0), bits 32+64 {}, gpu_native_owned_only {}, "
                  "gpu_d3d_owned_resolve {}, gpu_d3d_owned_present {}", h.run != nullptr ? 1 : 0, wired ? 1 : 0, nf_ ? 1 : 0, (front_bits & 96u) == 96u ? 1 : 0,
                  REXCVAR_GET(gpu_native_owned_only) ? 1 : 0, REXCVAR_GET(gpu_d3d_owned_resolve) ? 1 : 0, REXCVAR_GET(gpu_d3d_owned_present) ? 1 : 0);
    }
  }
  if (g_cut) CutEnforceSettings(false);  // (the in-game GRAPHICS menu can change them at run time)
  bool released = false;
  uint32_t head = s->head.load(std::memory_order_relaxed);
  uint32_t tail = s->tail.load(std::memory_order_acquire);
  // (M5c) the check at the entry (also with nothing queued; nothing queued = idle: the held frame is published at once, even in a scene)
  if (g_gate.open.load(std::memory_order_relaxed)) released = D3dNoringGatePoll(head == tail);
  if (head == tail) {
    if (ui::Presenter::GuestOutputPublishHeld() && !g_gate.open.load(std::memory_order_relaxed)) D3dNoringFlushHeldPublish();  // (G1) idle poll
    return released;
  }
  RefreshGpuDebugLogCache();  // (as the start of ExecutePrimaryBuffer)
  const uint32_t mask = s->capacity - 1;
  const OwnedRecordHooks& scout_hooks = OwnedRecordHookSlots();  // (Session 95, gpu_jobs: the scout reads ahead of this loop; each hook returns at once when it is off)
  if (g_cut && scout_hooks.scout_enter) scout_hooks.scout_enter(s, memory_);
  while (head != tail) {
    const uint32_t pos = head & mask;
    const uint32_t w0 = s->storage[pos];
    const uint32_t kind = w0 & 0xFF, total = w0 >> 8;
    if (total == 0 || total > s->capacity - pos || (kind != kNrPad && total < kD3dNoringHeaderDwords)) {
      s->bad.fetch_add(1, std::memory_order_relaxed);
      REXGPU_ERROR("[d3d][noring] queue framing error at {} (header {:08X}): the rest of the queue ({} dwords) is dropped", pos, w0,
                   tail - head);
      head = tail;
      s->head.store(head, std::memory_order_release);
      break;
    }
    if (kind != kNrPad) {
      // WWE '13 port (gpu_submit_safe_point_ms): between two items is where the queue runs dry when this thread keeps
      // up, and where the recorded work then goes to the GPU (PrepareForWait); do the same here once it has built up.
      if (CpSafePointDue()) PrepareForWait();
      if (g_gate.open.load(std::memory_order_relaxed)) D3dNoringGatePoll();  // (M5c) one load of the flip word between items
      const bool gate_was_open = g_gate.open.load(std::memory_order_relaxed);  // (G2) the item that opens the gate is not counted
      s->items[kind < kNrKindCount ? kind : 0].fetch_add(1, std::memory_order_relaxed);
      // (Session 92, M4a: the shadow decoder reads the item in host order before the existing path runs it, compares after)
      const OwnedRecordHooks& front = OwnedRecordHookSlots();
      // (Session 92, M4f-C: THE CUT) the decoder EXECUTES the packet and register items in order; handled = nothing else runs for the item (no `before`,
      // no D3dNoringExecute, no `after`). An item it refuses (counted by reason) goes the existing way below, decoded first as the shadow does.
      bool handled = false;
      if (g_cut && !g_poison_armed && free_counters::g_owned_only_frame && REXCVAR_GET(gpu_d3d_cut_poison) && front.poison) {
        g_poison_armed = true;  // the first item after the latch frame: the register file's executor-owned ranges are filled with a pattern
        front.poison(register_file_->values, false);
        REXGPU_INFO("[d3d][cut] register file poisoned (gpu_d3d_cut_poison): the typed fields' registers, 0x2000-0x23FF, 0x4000-0x48BF, 0x4900-0x4927, 0x5000-0x5002");
      }
      if (g_cut && kind != kNrPackets && kind != kNrRegs && front.inline_flush) front.inline_flush(*this);  // (Session 95 r5)
      if (g_cut && kind == kNrPresent && scout_hooks.scout_present) scout_hooks.scout_present(s, memory_);
      if (g_cut && (kind == kNrPackets || kind == kNrRegs)) {
        if (scout_hooks.scout_item) scout_hooks.scout_item(head);
        // under the cut an item the executor cannot take is DROPPED and counted by reason (Run does the counting); never the old path
        front.run(*this, memory_, kind, s->storage + pos + 1, s->storage + pos + kD3dNoringHeaderDwords, total - kD3dNoringHeaderDwords);
        handled = true;
        if (kind == kNrPackets) s->packet_dwords.fetch_add(total - kD3dNoringHeaderDwords, std::memory_order_relaxed);
      }
      if (!handled) {
        const bool shadow = front.before && ((REXCVAR_GET(gpu_d3d_owned_front) & 1u) || g_cut);  // (the cut implies the decoder: presents, segments)
        if (shadow) front.before(*this, memory_, kind, s->storage + pos + 1, s->storage + pos + kD3dNoringHeaderDwords, total - kD3dNoringHeaderDwords);
        // (Session 92, M4f-C) After the latch a segment piece made only of type-0 writes to the library's misc flush registers (0x05C8 / 0x0E00 / 0x0E40 - the
        // same three the executor drops as "no reader") and type-2 NOPs is consumed without running: nothing reads those registers (WriteRegister's only
        // side-effect registers are scratch, COHER and DC_LUT). Any other piece executes as before.
        bool skipped_flush = false;
        if (g_cut && kind == kNrSegment && free_counters::g_owned_only_frame) {
          const uint32_t length = s->storage[pos + 1 + d3d_noring_field::kSegLength];
          if (length && length <= 64) {
            const uint32_t* words = reinterpret_cast<const uint32_t*>(
                memory_->TranslatePhysical(xenos::GpuToCpu(s->storage[pos + 1 + d3d_noring_field::kSegAddress]) & 0x1FFFFFFF));
            uint32_t counts[3] = {};
            bool only_flush = true;
            for (uint32_t i = 0; i < length && only_flush;) {
              const uint32_t h = std::byteswap(words[i]);
              if ((h >> 30) == 0) {
                const uint32_t n = ((h >> 16) & 0x3FFF) + 1, base = h & 0x7FFF;
                if (base != 0x05C8 && base != 0x0E00 && base != 0x0E40) only_flush = false;
                else if (n != 1) only_flush = false;  // (one register a packet, as the library writes them)
                else ++counts[base == 0x05C8 ? 0 : base == 0x0E00 ? 1 : 2];
                i += 1 + n;
              } else if ((h >> 30) == 2) {
                ++i;
              } else {
                only_flush = false;
              }
            }
            if (only_flush) {
              skipped_flush = true;
              free_counters::CutCounters& cc = free_counters::g_cut;
              ++cc.segments_skipped;
              for (uint32_t k = 0; k < 3; ++k) cc.skipped_reg[k] += counts[k];
              s->segment_dwords.fetch_add(length, std::memory_order_relaxed);
            }
          }
        }
        if (!skipped_flush && !D3dNoringExecute(kind, s->storage + pos + 1, s->storage + pos + kD3dNoringHeaderDwords,
                                                total - kD3dNoringHeaderDwords)) {
          s->bad.fetch_add(1, std::memory_order_relaxed);
        }
        if (shadow) front.after(*this, kind);
      }
      if (g_cut && kind == kNrInterrupt && scout_hooks.scout_guest_write) scout_hooks.scout_guest_write();  // (Session 95: the guest callback ran)
      if (gate_was_open) ++g_gate.items_since;
    }
    head += total;
    s->head.store(head, std::memory_order_release);  // (the producer may reuse the entry's storage from here)
    if (!worker_running_) break;
    if (head == tail) tail = s->tail.load(std::memory_order_acquire);
  }
  if (g_cut && scout_hooks.scout_leave) scout_hooks.scout_leave();
  Half2DropDeadLibraryDraws();  // (stage 2c: library draws whose packets went as records never match a packet)
  OnPrimaryBufferEnd();  // (as the end of ExecutePrimaryBuffer: the Vulkan processor may hand its work to the host GPU)
  return true;
}

}  // namespace rex::graphics
