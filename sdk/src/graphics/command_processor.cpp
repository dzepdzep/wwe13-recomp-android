/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <rex/graphics/cut_probe.h>  // wwe2k14 patch (Session 92, M4f-C)
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <unordered_map>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <mutex>
#include <string_view>
#ifdef _WIN32
#include <share.h>
#endif

#include <fmt/format.h>

#include <rex/det.h>  // wwe2k14 patch (Session 15): det_frames
#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/hash.h>  // wwe2k14 patch (Session 19, DRAWRUNS): XXH3 for the anatomy state hashes
#include <rex/perf/counter.h>
#include <rex/perf/hang_beats.h>  // wwe2k14 patch (Session 95, HANG)
#include <rex/perf/processor.h>
#include <rex/perf/timeline.h>
#include <rex/chrono/clock.h>
#include <rex/graphics/command_processor.h>
#include <rex/graphics/d3d_copy.h>  // wwe2k14 patch (Session 87, D3D layer F2)
#include <rex/graphics/d3d_noring.h>  // wwe2k14 patch (Session 87, D3D layer stage 2)
#include <rex/graphics/d3d_direct.h>  // wwe2k14 patch (Session 87, D3D layer F4)
#include <rex/graphics/d3d_feed.h>  // wwe2k14 patch (Session 87, D3D layer stage 3)
#include "shadow_chain.h"  // wwe2k14 patch (Session 85, gpu_dynamic_shadows)
#include "vulkan/native/owned_shadow_skip.h"  // wwe2k14 patch (Session 95, item K)
#include "half2_state.h"  // wwe2k14 patch (Session 48, HALF2): complete for the destructor
#include "crowd_member_hook.h"  // wwe2k14 patch (Session 92, M4f-B)
#include "d3d_shader_tag.h"  // wwe2k14 patch (Session 90): tagged (guest virtual) shader microcode addresses
#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/graphics_system.h>
#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/sampler_info.h>
#include <rex/graphics/util/draw_extent_estimator.h>
#include <rex/graphics/xenos.h>
#include <rex/graphics/d3d_tex.h>
#include <rex/graphics/free_counters.h>
#include <rex/ui/presenter.h>  // wwe2k14 patch (Session 93, log only): HostPresentCallMs in WaitGuestWord's stuck line
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/memory.h>
#include <rex/memory/ring_buffer.h>
#include <rex/stream.h>
#include <rex/system/kernel_state.h>
#include <rex/system/user_module.h>

REXCVAR_DEFINE_BOOL(vsync, true, "GPU", "Enable vertical sync");
// wwe2k14 patch (Session 87, D3D LAYER F1; docs\native\d3d\F1_CONTRACT.md in the game repository): the private token packet
// of the game's D3D layer hook (wwe2k14_d3d_layer). This build handles it always; the cvar below is the capability
// marker the game exe looks up by name (rex::cvar::GetFlagInfo) before it writes any token: a build without it never gets one.
REXCVAR_DEFINE_INT32(gpu_d3d_layer_abi, 1, "GPU/Native",
                     "Session 87 (D3D layer F1): the token packet ABI this build executes (opcode 0x7C, payload magic "
                     "0xD3D10002, sequence, then the record: register runs, shader loads, draws). Read-only marker, do not set")
    .range(1, 1);
REXCVAR_DEFINE_INT32(gpu_d3d_layer_log_frames, 600, "GPU/Native",
                     "Session 87 (D3D layer F1): [d3d] census line every N frames (0 = none)");

namespace {
// wwe2k14 patch (Session 95, RACE R1): pending_fns_ (a std::queue) is shared by every CallInThread caller and the command thread. On ARM the
// command thread could see "not empty" before the element was built. Push and pop take this lock (they are rare: boot and tools); the
// command thread's hot loops read only the count. File-static, not members: command_processor.h keeps its layout, and there is one
// command processor per process.
std::mutex g_pending_fns_mutex;
std::atomic<uint32_t> g_pending_fns_count{0};
inline bool PendingFnsWaiting() { return g_pending_fns_count.load(std::memory_order_acquire) != 0; }

// wwe2k14 patch (Session 87, D3D LAYER F1): the private token packet. Opcode 0x7C is used by no packet of the SDK's
// tables nor by the game's D3D library (header scan of 0x82C9F000-0x82CD0000, map B section 8).
constexpr uint32_t kPm4D3dToken = 0x7C;
constexpr uint32_t kD3dTokenMagic = 0xD3D10002u;
// The census (the command thread's; relaxed atomics so the frame-end line can read them).
struct D3dCensus {
  std::atomic<uint64_t> tokens{0};
  std::atomic<uint64_t> runs{0};            // type-0 runs applied directly
  std::atomic<uint64_t> run_registers{0};
  std::atomic<uint64_t> shaders{0};         // shader loads (the IM_LOAD handler's work, from the record's identity)
  std::atomic<uint64_t> draws{0};           // draw chunks issued (DRAW_INDX's work after its parse)
  std::atomic<uint64_t> bytes{0};           // payload bytes carried
  std::atomic<uint64_t> replays{0};         // a token whose sequence is not above the last one's (the tiling replays)
  std::atomic<uint64_t> bad{0};             // a framing / magic error (loud)
  std::atomic<uint32_t> last_seq{0};
  uint32_t window_start = 0;
};
D3dCensus g_d3d_census;

// Once per guest frame (the swap): the census line every gpu_d3d_layer_log_frames frames, only while tokens arrive.
void D3dCensusFrame(uint32_t frame) {
  D3dCensus& c = g_d3d_census;
  const int32_t every = REXCVAR_GET(gpu_d3d_layer_log_frames);
  if (every <= 0 || frame - c.window_start < uint32_t(every)) return;
  const uint64_t tokens = c.tokens.exchange(0, std::memory_order_relaxed);
  const uint64_t runs = c.runs.exchange(0, std::memory_order_relaxed);
  const uint64_t regs = c.run_registers.exchange(0, std::memory_order_relaxed);
  const uint64_t shaders = c.shaders.exchange(0, std::memory_order_relaxed);
  const uint64_t draws = c.draws.exchange(0, std::memory_order_relaxed);
  const uint64_t bytes = c.bytes.exchange(0, std::memory_order_relaxed);
  const uint64_t replays = c.replays.exchange(0, std::memory_order_relaxed);
  const uint64_t bad = c.bad.exchange(0, std::memory_order_relaxed);
  if (tokens || bad) {
    REXLOG_INFO("[d3d] frames {}-{}: tokens {} ({} sequence repeats = tiling replays), records applied {}: {} register "
                "runs ({} registers), {} shader loads, {} draw chunks; {} bytes of record carried; framing errors {}",
                c.window_start, frame, tokens, replays, tokens - bad, runs, regs, shaders, draws, bytes, bad);
  }
  c.window_start = frame;
}
// wwe2k14 patch (Session 87, D3D LAYER F3): the private inline-draw token (src\graphics\d3d_inline_draw.cpp). Opcode 0x7E is
// used by no packet of the SDK's tables nor by the game's D3D library (F1 checked 0x7C by the header scan; 0x7D is
// the next one in PM4's unused tail: confirm with the same scan before merging).
constexpr uint32_t kPm4D3dInline = 0x7E;  // (F2's PM4_D3D_COPY is 0x7D, F1's token 0x7C)
}  // namespace

// WWE '13 port: the no-ring drain (d3d_noring.cpp) hands the thread back to the loop when a CallInThread job waits.
namespace rex::graphics { bool CpPendingFnsWaiting() { return PendingFnsWaiting(); } }
// wwe2k14 patch (Session 77, cross-device determinism runs; Jarrod: "no drawing could be huge"): the command processor
// still executes every packet (registers, interrupts, fences, waits, the frame swap) but records no host draw or copy
// (resolve): no pipelines, textures or render passes. Under det_serial the command thread's work sits on the critical
// path (the scheduler drains it at every hand-off; ~90% of a core on the phone). Not render-only: copies the game reads
// back (readback windows) and memexport draws do not write guest memory, so the comparer treats it as a game setting.
REXCVAR_DEFINE_BOOL(gpu_det_skip_host_draws, false, "GPU",
                    "Determinism runs: execute GPU packets but record no host draws or copies (no picture; resolves the "
                    "game reads back are not written)");
// wwe2k14 patch (Session 78, NATIVE FRAME; docs\native\NATIVE_FRAME_DESIGN.md in the game repository): the command
// processor as two sides (see nf_role_ in command_processor.h). Read once at start.
REXCVAR_DEFINE_BOOL(gpu_native_frame, false, "GPU/Native",
                    "Session 78 (HD 520 arc): the native frame - the command processor answers the guest without "
                    "drawing; a drawing side with its own register state runs the same commands behind it and catches "
                    "up before every guest-visible point. Read at start");
REXCVAR_DEFINE_BOOL(gpu_native_frame_draw, true, "GPU/Native",
                    "Session 78: with gpu_native_frame, run the drawing side (false: nothing drawn, the answers alone "
                    "- the rollback / no-draw case)");
REXCVAR_DEFINE_BOOL(gpu_native_frame_answers_lean, true, "GPU/Native",
                    "Session 79 (NF-C): with gpu_native_frame, the answers side stores only registers below 0x2000 "
                    "(render state and constants are the drawing side's; nothing reads them back). Read at start");
REXCVAR_DEFINE_INT32(gpu_native_frame_dead_glare, 1, "GPU/Native",
                     "Session 79 (NF-D look-ahead): with gpu_native_frame, the entrance's star glare draws when the "
                     "combine queued ahead in the segment has glare strength 0: 1 = not drawn (default; acts only with "
                     "gpu_native_frame), 2 = counted only, 0 = off. Pictures byte-identical (Session 79, runs "
                     "5008 / 5009). Read at start");
REXCVAR_DEFINE_INT32(gpu_native_frame_batch_log, 0, "GPU/Native",
                     "Session 79 (diagnostic): log the native-frame catch-up batch of the first N star-glare / entrance "
                     "combine draws (NativeRenderer::TruthRecordDraw)");
REXCVAR_DEFINE_INT32(gpu_native_frame_log_frames, 600, "GPU/Native",
                     "Session 78: [native][frame] census line every N frames (0 = none)");
REXCVAR_DECLARE(uint32_t, gpu_d3d_wall);  // (Session 95: d3d_noring.cpp)
REXCVAR_DECLARE(int32_t, gpu_d3d_wall_stats);
REXCVAR_DECLARE(int32_t, gpu_native_detached);
REXCVAR_DECLARE(bool, gpu_d3d_direct);  // wwe2k14 patch (Session 87, F4): defined in vulkan/native/d3d_direct.cpp
REXCVAR_DEFINE_BOOL(gpu_scene_crowd_batch, true, "GPU",
                    "wwe2k14 (Session 60): the crowd members the game queued (wwe2k14_scene_crowd 1) are added as "
                    "instances of their run's first draw in one go (their constants uploaded, nothing else derived "
                    "again) instead of one draw dispatch each; false = Session 50's one dispatch per member");
REXCVAR_DEFINE_INT32(gpu_scene_crowd_cull, 0, "GPU",
                     "wwe2k14 (Session 70): the crowd members the game queued (wwe2k14_scene_crowd 1) that lie "
                     "entirely outside the view of the draw that takes their run (its own camera, so every pass and "
                     "camera: cuts, CineCam, photo mode) - 1 counted only ([scene][crowd] cull lines), 2 not drawn "
                     "(exact: only the two known crowd vertex shaders, clipping on, a 1% margin)")
    .range(0, 2)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_scene_crowd_thin, 0, "GPU",
                     "wwe2k14 (Session 79, HD 520 arc look lever): with gpu_scene_crowd_cull 2, of the crowd members "
                     "farther than N/100 x their run's median view depth, every other one is not drawn (a thinner "
                     "distant crowd; the same members each frame). 0 = off (default); e.g. 100 = the farther half");

REXCVAR_DEFINE_INT32(gpu_scene_crowd_thin_keep, 2, "GPU",
                     "wwe2k14 (Session 82, look lever): with gpu_scene_crowd_thin, of the distant crowd members one in this "
                     "many is kept (2 = every other one, as before; 4 = a quarter of them; the same members each frame). "
                     "Not exact: a sparser far crowd - judged by eye")
    .range(1, 16);
REXCVAR_DEFINE_INT32(gpu_scene_crowd_density, 100, "GPU",
                     "wwe2k14 (Session 85, Jarrod 2026-11-03: a percentage, 10% steps, 0% allowed, evenly spread): the "
                     "share of the crowd drawn, in percent. Each member has a fixed rank from a hash of its world "
                     "position and is drawn when the rank is below this, so the kept members are spread evenly over the "
                     "arena and the same ones stay from frame to frame. 100 = the normal look (off); 0 = no crowd. "
                     "Applies to the queued crowd runs (wwe2k14_scene_crowd 1), after the cull and the thinning")
    .range(0, 100);

// wwe2k14 patch (Session 90, builder C): cull diagnostics - per shader kind (0 cards, 1 entrance, 2 match) members judged,
// culled; and with gpu_scene_crowd_cull_diag the culled members whose whole mesh box projects (all 8 corners w > 0) onto
// the screen rectangle (which a correct cull never produces).
static uint64_t g_cull_kind_members[3] = {}, g_cull_kind_outside[3] = {}, g_cull_diag_overlap = 0, g_cull_diag_behind = 0;
static uint64_t g_cull_diag_lines = 0;
REXCVAR_DEFINE_INT32(gpu_scene_crowd_cull_diag, 0, "GPU",
                     "wwe2k14 (Session 90, diagnostic): 1 = log the culled crowd members whose mesh box projects onto the screen")
    .range(0, 2)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_scene_crowd_thin_stable, true, "GPU",
                    "wwe2k14 (Session 90, Jarrod: \"make it stable\"): gpu_scene_crowd_thin's depth limit is the median over "
                    "the run's whole member list (before the view cull) and every member keeps its state inside a +-8% band "
                    "(hysteresis), so a member does not flip with the camera or with which other members are in view; "
                    "false = the Session 79 rule (median of this frame's survivors, no memory)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// Session 90: thin state per member (key: position hash, mesh, viewport scale), flip counters.
// Two generations (current + previous, swapped every ~600 guest frames in SceneCrowdFrameEnd); never freed (no static
// destructor at shutdown, Android). A lookup falls back to the previous generation and carries the state forward.
static auto& g_thin_state = *new std::unordered_map<uint64_t, uint8_t>();
static auto& g_thin_state_prev = *new std::unordered_map<uint64_t, uint8_t>();
static uint64_t g_thin_flips = 0, g_thin_new_keys = 0;
// The member's previous state (0 drawn, 1 thinned) or -1 when the key is new.
static int ThinStateGet(uint64_t key) {
  if (const auto it = g_thin_state.find(key); it != g_thin_state.end()) return it->second;
  if (const auto it = g_thin_state_prev.find(key); it != g_thin_state_prev.end()) return it->second;
  ++g_thin_new_keys;
  return -1;
}
// Notes the new state and counts a flip against the previous one.
static void ThinStateSet(uint64_t key, int prev, bool thinned) {
  if (prev >= 0 && bool(prev) != thinned) ++g_thin_flips;
  g_thin_state[key] = thinned ? 1 : 0;
}
static uint64_t g_crowd_density_dropped = 0;  // Session 85: members left out by gpu_scene_crowd_density (command thread)

// wwe2k14 patch (Session 95, the flickering crowd signs): diagnostic. A density rank hashes the member's exact position, so a member that moves gets a new rank every
// frame. Per site and shader: rank tests, and those whose position word was not tested the frame before.
REXCVAR_DEFINE_BOOL(gpu_scene_crowd_rank_diag, false, "GPU",
                    "Session 95, diagnostic: [crowd] rank lines every 600 frames: per site and vertex shader the density rank tests and how many were at a position not seen the frame before");
namespace {
struct RankDiagCell {
  uint64_t tests = 0, moved = 0, rejected = 0;
};
auto& g_rank_cells = *new std::unordered_map<uint64_t, RankDiagCell>();
auto& g_rank_seen = *new std::unordered_map<uint64_t, uint64_t>();  // position word -> last frame
uint64_t g_rank_frame = 0;
}  // namespace
void CrowdRankDiagNote(uint32_t site, uint64_t shader_or_mesh, uint32_t h, bool rejected) {
  if (!REXCVAR_GET(gpu_scene_crowd_rank_diag)) return;
  RankDiagCell& c = g_rank_cells[(shader_or_mesh & 0x0FFFFFFFFFFFFFFFull) | (uint64_t(site) << 60)];
  ++c.tests;
  c.rejected += rejected;
  uint64_t& last = g_rank_seen[(uint64_t(h) << 8) ^ site];
  if (last + 1 < g_rank_frame) ++c.moved;
  last = g_rank_frame;
}
// wwe2k14 patch (Session 95, the flickering crowd signs): gpu_scene_crowd_density's rank is a hash of the member's exact position, so a member that moves (a spectator
// waving a sign: 1-2 of the ~70 lone draws a frame in WrestleMania V, run rank70) drew a new rank every frame and, below 100%, vanished in that share of the frames
// (run sign70b: Take Your Vitamins gone from about a third of the shots at 70%). With the tracker a position it has not seen takes the rank of the nearest member of the
// last two frames within kTrackRadius, so a member keeps one rank while it moves. A new position inherits from the nearest member of the last two frames that was not tested again this frame, which can be a stationary neighbour not yet tested this frame: ranks can depend on camera history, and a member untested for 3+ frames gets a new rank. Positions already known keep their rank. Command thread only.
REXCVAR_DEFINE_BOOL(gpu_scene_crowd_density_track, false, "GPU",
                    "Session 95: gpu_scene_crowd_density's rank follows a member that moves (a waved sign) instead of changing with its position every frame. Off: the rank of the exact position");
namespace {
struct RankTrackEntry {
  float x, y, z;
  uint8_t rank;
  uint32_t key;  // the position hash it was tested under
};
struct RankKnown {
  uint8_t rank;
  uint64_t last;  // frame last tested
};
constexpr float kTrackRadius = 24.0f;
auto& g_track_known = *new std::unordered_map<uint32_t, RankKnown>();
auto& g_track_known_prev = *new std::unordered_map<uint32_t, RankKnown>();
auto& g_track_now = *new std::vector<RankTrackEntry>();
auto& g_track_last = *new std::vector<RankTrackEntry>();
auto& g_track_last2 = *new std::vector<RankTrackEntry>();
uint64_t g_track_frame = 1, g_track_inherited = 0, g_track_new = 0;
inline float RankF(uint32_t bits) {
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}
}  // namespace
// The rank (0-99) of the member at the position words w3 / w7 / w11 (the w components of c76 / c77 / c78).
int32_t CrowdRankOf(uint32_t w3, uint32_t w7, uint32_t w11) {
  const uint32_t key = w3 * 0x9E3779B1u ^ w7 * 0x85EBCA77u ^ w11 * 0xC2B2AE3Du;
  uint32_t h = key;
  h ^= h >> 16, h *= 0x7FEB352Du, h ^= h >> 15, h *= 0x846CA68Bu, h ^= h >> 16;  // (a finalizer: uniform ranks)
  const int32_t hashed = int32_t(h % 100u);
  if (!REXCVAR_GET(gpu_scene_crowd_density_track)) return hashed;
  const float x = RankF(w3), y = RankF(w7), z = RankF(w11);
  RankKnown* known = nullptr;
  if (const auto it = g_track_known.find(key); it != g_track_known.end()) {
    known = &it->second;
  } else if (const auto it2 = g_track_known_prev.find(key); it2 != g_track_known_prev.end()) {
    known = &(g_track_known[key] = it2->second);
  }
  if (known) {
    if (known->last != g_track_frame) {
      known->last = g_track_frame;
      g_track_now.push_back({x, y, z, known->rank, key});
    }
    return known->rank;
  }
  // A position not seen before: a member that moved, or one that came into view.
  float best = kTrackRadius * kTrackRadius;
  int32_t rank = hashed;
  bool found = false;
  for (const auto* list : {&g_track_last, &g_track_last2}) {
    for (const RankTrackEntry& e : *list) {
      // (a member already tested again this frame at its old position has not moved: not a source)
      if (const auto kt = g_track_known.find(e.key); kt != g_track_known.end() && kt->second.last == g_track_frame) continue;
      const float dx = e.x - x, dy = e.y - y, dz = e.z - z, d = dx * dx + dy * dy + dz * dz;
      if (d > 0.0f && d < best) best = d, rank = e.rank, found = true;
    }
    if (found) break;
  }
  ++(found ? g_track_inherited : g_track_new);
  g_track_known[key] = {uint8_t(rank), g_track_frame};
  g_track_now.push_back({x, y, z, uint8_t(rank), key});
  return rank;
}
static void CrowdRankTrackFrame() {
  if (!REXCVAR_GET(gpu_scene_crowd_density_track)) return;
  g_track_last2.swap(g_track_last);
  g_track_last.swap(g_track_now);
  g_track_now.clear();
  if (++g_track_frame % 600) return;
  REXGPU_INFO("[crowd] density track: per frame {:.2f} members followed, {:.2f} new; {} positions known", double(g_track_inherited) / 600.0, double(g_track_new) / 600.0, g_track_known.size());
  g_track_inherited = g_track_new = 0;
  g_track_known_prev.swap(g_track_known);
  g_track_known.clear();
}

static void CrowdRankDiagFrame() {
  if (!REXCVAR_GET(gpu_scene_crowd_rank_diag)) return;
  ++g_rank_frame;
  if (g_rank_frame % 600) return;
  for (const auto& [key, c] : g_rank_cells) {
    REXGPU_INFO("[crowd] rank site {} key {:016X}: per frame tests {:.1f}, at a new position {:.1f}, rejected {:.1f}", key >> 60, key & 0x0FFFFFFFFFFFFFFFull, double(c.tests) / 600.0,
                double(c.moved) / 600.0, double(c.rejected) / 600.0);
  }
  g_rank_cells.clear();
  if (g_rank_seen.size() > 200000) g_rank_seen.clear();
}

// wwe2k14 patch (Session 87, crowd2): every draw of the three instanced crowd vertex shaders goes to our crowd program.
REXCVAR_DEFINE_BOOL(gpu_native_own_crowd_all, false, "GPU/Native",
                    "Session 87 (crowd2): with gpu_native_own_crowd, EVERY draw of the three instanced crowd vertex shaders "
                    "(the cards D5E2FCC9, the entrance C1028A64, the match 183CC64A) is drawn by our crowd program - a run's "
                    "first member and a lone draw outside the queued runs as a run of one (the per-member c76-c78 from the "
                    "draw's register image), so none of them reaches the game's translated instanced draw. false = only "
                    "the queued runs' members after the first (default). Live")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// wwe2k14 patch (Session 87, crowd2): the crowd census - per 600 guest frames, for each of the three instanced crowd vertex
// shaders, the draws seen, the members drawn by our crowd program, and the draws that reached the game's own draw by
// reason: F = a held run's first member (the run had no others, or the switch is off), S = a lone draw outside the queued
// runs, H = handed back (our crowd program was tried and declined), O = a run's other members the game drew without a
// try of ours. (Command thread only.)
namespace {
struct CrowdCensus {
  uint64_t draws = 0, ours = 0, ours_calls = 0, first = 0, stray = 0, handed_back = 0, other = 0;
  bool any() const { return draws || ours || first || stray || handed_back || other; }
};
CrowdCensus g_crowd_census[3];
uint64_t g_crowd_census_window = UINT64_MAX;
uint32_t g_scene_crowd_item_at = 0;   // (B3) the D3D layer's direct path: this item's guest address (0 = the stock packet path)
uint64_t g_scene_crowd_reg_writes = 0;  // (B3) times SceneCrowdAfterDraw wrote c76-c78 itself (a run the game drew)
bool g_crowd_census_lone = false;      // this draw was taken as a lone run of one (gpu_native_own_crowd_all)
int g_crowd_census_unheld = -1;        // this draw is one of the three shaders, left to the game (index), until classified
// 0 cards D5E2FCC9F76BC40C, 1 entrance C1028A64EFCF8229, 2 match 183CC64A4C4E96AC, -1 another shader.
int CrowdCensusIndex(const rex::graphics::Shader* vs) {
  if (!vs) return -1;
  switch (vs->ucode_data_hash()) {
    case 0xD5E2FCC9F76BC40Cull: return 0;
    case 0xC1028A64EFCF8229ull: return 1;
    case 0x183CC64A4C4E96ACull: return 2;
    default: return -1;
  }
}
}  // namespace

REXCVAR_DECLARE(double, gpu_wide_factor);  // Session 73: the crowd cull's view (vulkan/command_processor.cpp)
REXCVAR_DECLARE(double, gpu_tall_factor);
REXCVAR_DEFINE_BOOL(gpu_wait_reg_mem_spin, false, "GPU",
                    "WAIT_REG_MEM: yield-spin while the condition is unmet instead of sleeping "
                    "wait/256 ms per poll (sleeping can overshoot by up to a timer tick)");
// wwe2k14 patch (Session 37): the spin above keeps the GPU command thread at ~95% of a core in a match (run 2622);
// on a laptop that core's power is the integrated GPU's too.
REXCVAR_DEFINE_INT32(gpu_wait_reg_mem_sleep_us, 0, "GPU",
                     "With gpu_wait_reg_mem_spin (and vsync on): sleep this many microseconds per poll on a "
                     "high-resolution timer instead of yielding (0 = yield-spin as before)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_wait_reg_mem_stats, false, "GPU",
                    "Log WAIT_REG_MEM statistics (packets, waits, time slept, hottest address) every 5 s");

// wwe2k14 patch (Session 6, CP1): type-0 packets carry most register writes in 2K14 (including shader
// constants; SET_CONSTANT isn't used), and each register went through a virtual WriteRegister call.
REXCVAR_DEFINE_BOOL(gpu_type0_bulk_writes, false, "GPU",
                    "Type-0 packets that write consecutive registers go through the register-range path "
                    "(WriteRegisterRangeFromRing, as SET_CONSTANT does): shader constant ranges become one "
                    "copy plus one range invalidation, other ranges still use per-register writes. Same "
                    "results; read per packet, so it can be toggled mid-run");
// wwe2k14 patch (Session 6, debug): drop every draw whose vertex shader has this ucode hash (hex, as in the frame
// anatomy CSV), to see on screen what a shader draws. Read once per frame. Empty = off. Debug only.
REXCVAR_DEFINE_STRING(gpu_debug_skip_vertex_shader, "", "GPU",
                      "Debug: skip draws whose vertex shader ucode hash equals this hex value, or any of a "
                      "comma-separated list of up to 64 (Session 16) (empty = off)");
// wwe2k14 patch (Session 27.5, DRAWPROBE, debug): which mesh is which (Cena's title). See DrawProbe in
// command_processor.h. All off by default; read at each swap.
REXCVAR_DEFINE_STRING(gpu_debug_draw_probe, "", "GPU",
                      "Debug (Session 27.5): log the draws whose vertex shader ucode hash is in this comma-separated "
                      "hex list (* = every draw) to gpu_debug_draw_probe_path, for gpu_debug_draw_probe_frames frames");
REXCVAR_DEFINE_INT32(gpu_debug_draw_probe_frames, 0, "GPU",
                     "Debug (Session 27.5): N > 0 = probe the next N frames (set back to 0 when they start)");
REXCVAR_DEFINE_STRING(gpu_debug_draw_probe_path, "", "GPU",
                      "Debug (Session 27.5): the draw probe CSV (empty = logs/drawprobe.csv in the working folder)");
REXCVAR_DEFINE_STRING(gpu_debug_draw_probe_consts, "", "GPU",
                      "Debug (Session 27.5): float constants A-B (float4 registers, inclusive; 256-511 = the pixel shader bank, Session 80) written per "
                      "probed draw (empty = none)");
REXCVAR_DEFINE_STRING(gpu_debug_skip_pixel_shader, "", "GPU",
                      "Debug (Session 72): skip draws whose guest pixel shader ucode hash is in this comma-separated "
                      "hex list (up to 64; empty = off), as gpu_debug_skip_vertex_shader");
REXCVAR_DEFINE_STRING(gpu_debug_skip_index_base, "", "GPU",
                      "Debug (Session 27.5): skip draws whose index buffer guest address is in this comma-separated "
                      "hex list (up to 64; empty = off)");
// wwe2k14 patch (Session 15, sub-second loads): see skip_frame_draws_ in command_processor.h.
REXCVAR_DEFINE_INT32(gpu_skip_frame_draws, 0, "GPU",
                     "wwe2k14 (Session 15): N > 1 = skip the draws (not memexport ones) and resolves of N-1 of every "
                     "N frames (for loading screens running far above the display rate); 0 = off. Latched per frame");
REXCVAR_DEFINE_BOOL(gpu_type0_bulk_writes_verify, false, "GPU",
                    "With gpu_type0_bulk_writes: replay every bulk-written shader constant range through "
                    "per-register writes and compare register values, invalidation bits and ring offset; "
                    "logs [s6][CP1] mismatches and a summary every 300 frames (slow, for checking only)");
// wwe2k14 patch (Session 6, CP1B): with gpu_type0_bulk_writes, type-0 ranges not within one shader constant
// block (about 2.4-2.5M packets per 300 frames of the CH103 entrance, by gpu_type0_bulk_writes_verify) still
// went through one virtual WriteRegister per register. After Canary's D3D12
// WriteRegisterRangeFromMem_WithKnownBound.
REXCVAR_DEFINE_BOOL(gpu_type0_block_split, false, "GPU",
                    "With gpu_type0_bulk_writes: register ranges not within one shader constant block are "
                    "written block by block in register order: plain registers (no side effects) copied in one "
                    "go, shader constant sub-ranges through the backend's bulk constant write, only registers "
                    "with side effects (scratch, COHER_STATUS_HOST, gamma ramp, past the register file) one by "
                    "one; write_one_reg type-0 writes to plain registers skip the virtual call. Same results; "
                    "read per packet/range, so it can be toggled mid-run. No effect while GPU debug logging is "
                    "on. Logs a [s6][CP1B] path summary every 300 frames");
REXCVAR_DEFINE_BOOL(gpu_type0_block_split_verify, false, "GPU",
                    "With gpu_type0_block_split: after each split range, replay its plain registers and shader "
                    "constants through per-register writes and compare register values, invalidation bits and "
                    "which registers took the per-register path; logs [s6][CP1B] mismatches (slow, for checking "
                    "only)");

REXCVAR_DEFINE_BOOL(clear_memory_page_state, true, "GPU",
                    "Refresh page-valid state from GPU-written memory at frame end. "
                    "Disable for minor CPU overhead reduction, but may break memory coherency.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(occlusion_query_enable, true, "GPU", "Enable host occlusion query handling")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_STRING(readback_resolve, "none", "GPU",
                      "Controls CPU readback of render-to-texture resolve results.\n"
                      " none: Disable readback (default)\n"
                      " fast: Read previous frame (delayed, copy every frame)\n"
                      " some: Read previous frame (delayed, copy on cache miss)\n"
                      " full: Immediate sync readback (accurate but stalls)")
    .allowed({"none", "fast", "some", "full"})
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// wwe2k14 patch (Session 57, tester bug 4): a readback mode the app switches at run time, read at every resolve (so
// it applies from the next resolve on, not the next swap): -1 = none set (readback_resolve applies), 0 none, 1 fast,
// 2 some, 3 full. The app sets 3 while the game is in a mode whose rendered pictures its CPU reads back (a Custom
// Superstar's render, an arena's menu screenshot): with readback off those came out as noise.
REXCVAR_DEFINE_INT32(readback_resolve_override, -1, "GPU",
                     "Readback mode set at run time by the app, ahead of readback_resolve: -1 none set, 0 none, "
                     "1 fast, 2 some, 3 full")
    .range(-1, 3)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(readback_resolve_half_pixel_offset, false, "GPU",
                    "When draw resolution scaling is active, sample from the center of each "
                    "scaled block during resolve readback downscale")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(readback_memexport, true, "GPU",
                    "Enable CPU readback of shader memexport writes for guest memory "
                    "coherency (can reduce correctness issues, but may add GPU/CPU sync cost)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(readback_memexport_fast, true, "GPU",
                    "Use fast double-buffered memexport readback when possible, with "
                    "automatic fallback to full synchronous readback")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(query_occlusion_fake_sample_count, 1000, "GPU",
                     "Fake sample count for occlusion queries")
    .range(1, 100000)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(async_shader_compilation, true, "GPU",
                    "Compile shaders and create pipelines asynchronously in background "
                    "threads. This reduces stutter but may cause brief visual artifacts while "
                    "pipelines are being prepared.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// wwe2k14 patch (Session 6, COUNTERS): frame-anatomy capture, measurement only (no effect on emulation).
REXCVAR_DEFINE_INT32(gpu_anatomy_capture, 0, "GPU",
                     "Frame anatomy: when set to N > 0, record one CSV row per draw and resolve for the next "
                     "N command processor frames (starting at the next swap), then reset to 0");
REXCVAR_DEFINE_STRING(gpu_anatomy_path, "", "GPU",
                      "Frame anatomy output file; %d = capture index in this run (default "
                      "<current dir>/logs/anatomy_%d.csv). Per-frame totals go to <name>_frames.csv");
// wwe2k14 patch (Session 7, TILING-STAGE0): predicated tiling log, measurement only (no effect on emulation).
REXCVAR_DEFINE_INT32(gpu_tiling_log_frames, 0, "GPU",
                     "Tiling log: when set to N > 0, write every executed SET_BIN_SELECT*/SET_BIN_MASK* value, "
                     "EVENT_WRITE_EXT, INTERRUPT and unpredicated SCREEN_EXT_INIT, per-bin-select segment counts "
                     "and the frame boundaries of the next N command processor frames (starting at the next "
                     "swap) to tiling_*.csv in gpu_anatomy_path's directory (anatomy_NNN_%d.csv -> "
                     "tiling_NNN_%d.csv), then reset to 0");

REXCVAR_DECLARE(bool, gpu_dynamic_shadows);  // wwe2k14 patch (Session 85): defined in vulkan/command_processor.cpp

// wwe2k14 patch (Session 85, gpu_dynamic_shadows false): the shadow chain's frame end (state: shadow_chain.h). The
// setting is latched here, once per frame; a frame with no caster and no chain draw ends the chain (menus, loads);
// kPrime consecutive active frames prime it (draws run, casters skipped, copy destinations noted), then it is skipped
// whole. A skipped copy to an unknown destination re-primes from the next frame.
void ShadowChainFrameEnd(uint64_t frame) {
  using namespace wwe_shadow_chain;
  State& s = g;
  const bool want_off = !REXCVAR_GET(gpu_dynamic_shadows);
  if (!want_off && !s.off) return;  // (the default: nothing but this test)
  const auto reset = [&s]() {
    s.skip = s.disabled = s.reprime_pending = false;
    s.primed = 0;
    s.keys.clear();
    s.excluded.clear();
    s.dests.clear();
  };
  if (want_off != s.off) {
    reset();
    s.off = want_off;
    s.seen = s.casters = 0;
    return;
  }
  const bool active = s.casters || s.seen;
  if (!active) {
    reset();
  } else if (s.reprime_pending) {
    s.reprime_pending = false;
    s.skip = false;
    s.primed = 0;
    ++s.reprimes;
  } else if (s.skip) {
    ++s.frames_skipped;
  } else {
    ++s.frames_primed;
    if (s.primed < kPrime) ++s.primed;
    s.skip = s.primed >= kPrime && !s.disabled && !s.keys.empty() && !s.dests.empty();
  }
  s.seen = s.casters = 0;
  if (frame % 600 == 0 && (s.draws_skipped || s.copies_skipped || s.frames_primed)) {
    REXGPU_INFO("[shadow] dynamic shadows OFF, chain to frame {}: frames primed {}, skipped {}; draws skipped {}, copies "
                "skipped {}, re-primes {}, surfaces {}, destinations {}{}",
                frame, s.frames_primed, s.frames_skipped, s.draws_skipped, s.copies_skipped, s.reprimes, s.keys.size(),
                s.dests.size(), s.disabled ? " (disabled: too many)" : "");
    s.draws_skipped = s.copies_skipped = s.reprimes = s.frames_skipped = s.frames_primed = 0;
  }
}

REXCVAR_DEFINE_INT32(gpu_submit_safe_point_ms, 0, "GPU",
                     "Hand the recorded work to the GPU between two commands once it has been building up this many ms "
                     "(0 = only when the command thread runs out of commands)")
    .range(0, 100)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

namespace rex::graphics {
bool CpSafePointDue();


using namespace rex::graphics::xenos;

namespace {

ReadbackResolveMode ParseReadbackResolveMode(std::string_view value) {
  if (value == "fast") {
    return ReadbackResolveMode::kFast;
  }
  if (value == "some") {
    return ReadbackResolveMode::kSome;
  }
  if (value == "full") {
    return ReadbackResolveMode::kFull;
  }
  return ReadbackResolveMode::kDisabled;
}

// The readback_resolve part of GetReadbackResolveMode. The HasNonDefaultValue lookup takes the cvar
// registry lock, allocates a std::string and does a map find.
ReadbackResolveMode GetSharedReadbackResolveMode(bool& shared_mode_overrides_legacy_out) {
  ReadbackResolveMode shared_mode = ParseReadbackResolveMode(REXCVAR_GET(readback_resolve));
  shared_mode_overrides_legacy_out = shared_mode != ReadbackResolveMode::kDisabled ||
                                     rex::cvar::HasNonDefaultValue("readback_resolve");
  return shared_mode;
}

// wwe2k14 patch (Session 6, CP1B): register classes for gpu_type0_block_split. A register write is "plain" when
// CommandProcessor::WriteRegister and the backends' overrides do nothing for it but store the value (the Vulkan
// and D3D12 overrides only add handling for the shader constant blocks below), apart from the unknown-register
// debug message, which is why the split is only used while gpu_debug_log_may_be_enabled_ is false. Every register
// that isn't plain is in one of these half-open intervals, listed in register order:
struct RegisterSplitInterval {
  uint32_t first;
  uint32_t end;
  // A shader constant block: written through the backend's WriteRegistersFromMem, whose single-block bulk code
  // does for the whole sub-range what its per-register WriteRegister does for each register (copy, plus the
  // constant buffer, texture binding and vertex buffer invalidation; CP1). Otherwise each register goes through
  // the virtual WriteRegister.
  bool constant_block;
};
constexpr RegisterSplitInterval kRegisterSplitIntervals[] = {
    // SCRATCH_REG0-7: written back to memory when enabled in SCRATCH_UMSK.
    {XE_GPU_REG_SCRATCH_REG0, XE_GPU_REG_SCRATCH_REG7 + 1, false},
    // COHER_STATUS_HOST: bit 31 is set after the store.
    {XE_GPU_REG_COHER_STATUS_HOST, XE_GPU_REG_COHER_STATUS_HOST + 1, false},
    // DC_LUT_RW_INDEX, DC_LUT_SEQ_COLOR, DC_LUT_PWL_DATA, DC_LUT_30_COLOR: gamma ramp access with sequential
    // component and index increments (not idempotent).
    {XE_GPU_REG_DC_LUT_RW_INDEX, XE_GPU_REG_DC_LUT_30_COLOR + 1, false},
    // Shader constants: backend invalidation (and the COUNTERS fetch constant notifications on Vulkan). The
    // unused 0x48C0-0x48FF gap between the fetch and bool constants is plain.
    {XE_GPU_REG_SHADER_CONSTANT_000_X, XE_GPU_REG_SHADER_CONSTANT_511_W + 1, true},
    {XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0, XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5 + 1, true},
    {XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031, XE_GPU_REG_SHADER_CONSTANT_LOOP_31 + 1, true},
    // Past the register file: stored in extended_register_values_.
    {uint32_t(RegisterFile::kRegisterCount), UINT32_MAX, false},
};

constexpr bool IsShaderConstantBlockRegister(uint32_t index) {
  return (index >= XE_GPU_REG_SHADER_CONSTANT_000_X && index <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) ||
         (index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 && index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31);
}

// The same classification as a short test, for single register writes.
constexpr bool IsPlainRegisterWrite(uint32_t index) {
  return index < RegisterFile::kRegisterCount &&
         uint32_t(index - XE_GPU_REG_SCRATCH_REG0) >
             uint32_t(XE_GPU_REG_SCRATCH_REG7 - XE_GPU_REG_SCRATCH_REG0) &&
         index != XE_GPU_REG_COHER_STATUS_HOST &&
         uint32_t(index - XE_GPU_REG_DC_LUT_RW_INDEX) >
             uint32_t(XE_GPU_REG_DC_LUT_30_COLOR - XE_GPU_REG_DC_LUT_RW_INDEX) &&
         !IsShaderConstantBlockRegister(index);
}

// The interval list is sorted and describes exactly the registers IsPlainRegisterWrite rejects (checked for every
// register index and some past the register file), and its constant blocks are exactly
// IsShaderConstantBlockRegister.
constexpr bool RegisterSplitClassesAgree() {
  for (size_t i = 0; i + 1 < std::size(kRegisterSplitIntervals); ++i) {
    if (kRegisterSplitIntervals[i].first >= kRegisterSplitIntervals[i].end ||
        kRegisterSplitIntervals[i].end > kRegisterSplitIntervals[i + 1].first) {
      return false;
    }
  }
  size_t interval = 0;
  for (uint32_t index = 0; index < uint32_t(RegisterFile::kRegisterCount) + 64; ++index) {
    while (index >= kRegisterSplitIntervals[interval].end) {
      ++interval;
    }
    bool listed = index >= kRegisterSplitIntervals[interval].first;
    if (listed == IsPlainRegisterWrite(index) ||
        (listed && kRegisterSplitIntervals[interval].constant_block) !=
            IsShaderConstantBlockRegister(index)) {
      return false;
    }
  }
  return true;
}
static_assert(RegisterSplitClassesAgree(),
              "kRegisterSplitIntervals, IsPlainRegisterWrite and IsShaderConstantBlockRegister disagree");

// Plain runs shorter than this are copied inline rather than through the out-of-line copy_and_swap.
constexpr uint32_t kPlainRunCopyAndSwapThreshold = 8;

}  // namespace

// wwe2k14 patch (Session 6, COUNTERS): frame-anatomy capture state (defined before the destructor, which
// destroys it). See AnatomyFrameBoundary.
struct CommandProcessor::AnatomyCapture {
  struct Row {
    int64_t start_ns;  // relative to the frame's first command
    int64_t cp_ns;     // command processor time in IssueDraw
    uint64_t vs_hash;
    uint64_t ps_hash;
    uint32_t prim;
    uint32_t count;
    uint32_t indexed;
    uint32_t index_format;
    uint32_t rb_surface_info;
    uint32_t rb_color_info[4];
    uint32_t rb_depth_info;
    uint32_t rb_modecontrol;
    uint32_t rb_depthcontrol;
    uint32_t rb_colorcontrol;
    uint32_t rb_color_mask;
    uint32_t vs_textures;  // texture fetch constants used, bit per fetch constant
    uint32_t ps_textures;
    uint32_t vs_float_constants;  // float constants used
    uint32_t ps_float_constants;
    uint32_t pipeline_changed;
    uint32_t skipped;  // AnatomySkipReason
    // wwe2k14 patch (Session 19, DRAWRUNS): what a draw has in common with the previous one, to count runs of draws
    // that differ only in shader constants. state_hash = registers 0x2000-0x3FFF (render state, render targets,
    // index buffer, draw initiator, viewport) + the bound shader objects; fetch_hash = the fetch constants
    // 0x4800-0x48BF (textures, vertex buffers). Float / bool / loop constants are left out.
    uint64_t state_hash;
    uint64_t fetch_hash;
    // wwe2k14 patch (Session 7, EXTENTS-SHADOW): the draw's computed screen extents (gpu_extents_mode).
    bool ext_attempted;  // an estimate ran for this draw (ext_* columns empty otherwise)
    bool ext_ok;
    uint16_t ext_min_x, ext_max_x, ext_min_y, ext_max_y;  // 8-pixel cells, as EVENT_WRITE_EXT would report
    uint32_t ext_fallback;                                // ScreenExtentsFallback when !ext_ok
    float ext_us;                                         // estimate time
    uint32_t copy_dest;
    uint32_t copy_info;
    bool is_copy;
    // wwe2k14 patch (Session 7, TILING-STAGE0): predication and the tile pass's window, scissor and viewport
    // (raw register values at the draw).
    bool predicated;  // the draw packet had the predicate bit
    uint32_t bin_select_lo;
    uint32_t bin_mask_lo;
    uint32_t pa_sc_window_offset;
    uint32_t pa_sc_window_scissor_tl;
    uint32_t pa_sc_window_scissor_br;
    uint32_t pa_sc_screen_scissor_tl;
    uint32_t pa_sc_screen_scissor_br;
    uint32_t pa_cl_vport_xscale;
    uint32_t pa_cl_vport_xoffset;
    uint32_t pa_cl_vport_yscale;
    uint32_t pa_cl_vport_yoffset;
    // wwe2k14 patch (Session 7c, CAUSES): what the backend recorded for the row (AnatomyRowCauses).
    bool causes_valid;
    AnatomyCauses causes;
    // wwe2k14 patch (Session 42, POSTMAP): the copy's control and pitch, and the texture fetch constants the draw's
    // shaders use (slot, base address, format, size, dimension, tiled), to map which resolve feeds which pass.
    uint32_t copy_control;
    uint32_t copy_dest_pitch;
    uint32_t tex_count;
    struct Tex {
      uint8_t slot, format, dimension, tiled;
      uint32_t base;
      uint16_t width, height;
    } tex[8];
  };

  // A guard against unbounded memory if swaps stop coming.
  static constexpr size_t kMaxRowsPerFrame = size_t(1) << 18;

  ~AnatomyCapture() { CloseFiles(); }
  void CloseFiles() {
    for (std::FILE** file : {&rows_file, &frames_file}) {
      if (*file) {
        std::fclose(*file);
        *file = nullptr;
      }
    }
  }

  // Captures started in this run (the %d in the file name).
  uint32_t captures_started = 0;

  // The running capture.
  std::FILE* rows_file = nullptr;
  std::FILE* frames_file = nullptr;
  std::string rows_path;
  int32_t requested = 0;  // gpu_anatomy_capture when it started (frames)
  uint32_t frame = 0;
  uint64_t rows_written = 0;
  uint64_t rows_dropped = 0;
  std::vector<Row> rows;  // of the current frame
  fmt::memory_buffer text;
  int64_t frame_start_ns = 0;
  int64_t frame_end_ns = 0;
  bool frame_started = false;

  // The IssueDraw in progress.
  int64_t draw_start_ns = 0;
  uint32_t draws_issued_before = 0;
  uint32_t copies_before = 0;
  uint32_t pipeline_binds_before = 0;
};

// wwe2k14 patch (Session 7, TILING-STAGE0): gpu_tiling_log_frames state (defined before the destructor, which
// destroys it). See TilingLogFrameBoundary for the file layout.
struct CommandProcessor::TilingLogCounts {
  uint32_t pred_exec = 0;    // predicated type-3 packets executed
  uint32_t pred_skip = 0;    // predicated type-3 packets skipped by the bin test
  uint32_t draw_exec = 0;    // predicated DRAW_INDX / DRAW_INDX_2 executed
  uint32_t draw_skip = 0;    // predicated DRAW_INDX / DRAW_INDX_2 skipped by the bin test
  uint32_t draw_unpred = 0;  // DRAW_INDX / DRAW_INDX_2 without the predicate bit
  uint32_t ext_exec = 0;     // EVENT_WRITE_EXT executed
  uint32_t ext_skip = 0;     // EVENT_WRITE_EXT skipped by the bin test
  bool any() const {
    return pred_exec || pred_skip || draw_exec || draw_skip || draw_unpred || ext_exec || ext_skip;
  }
  void Add(const TilingLogCounts& other) {
    pred_exec += other.pred_exec;
    pred_skip += other.pred_skip;
    draw_exec += other.draw_exec;
    draw_skip += other.draw_skip;
    draw_unpred += other.draw_unpred;
    ext_exec += other.ext_exec;
    ext_skip += other.ext_skip;
  }
};

struct CommandProcessor::TilingLog {
  // Rows are buffered and written at frame boundaries, or earlier past this size.
  static constexpr size_t kFlushBytes = size_t(8) << 20;

  ~TilingLog() { CloseFile(); }
  void CloseFile() {
    if (file) {
      std::fclose(file);
      file = nullptr;
    }
  }

  // Logs started in this run (the %d in the file name).
  uint32_t logs_started = 0;

  // The running log.
  std::FILE* file = nullptr;
  std::string path;
  int32_t requested = 0;  // gpu_tiling_log_frames when it started (frames)
  uint32_t frame = 0;
  uint32_t seq = 0;  // rows written in this frame
  uint64_t rows_written = 0;
  bool write_failed = false;
  fmt::memory_buffer text;
  int64_t frame_start_ns = 0;
  TilingLogCounts segment;  // since the last bin select change (or the frame start)
  TilingLogCounts frame_totals;

  // The type-3 packet being executed (TilingLogPacket).
  uint32_t packet_address = 0;  // guest physical address of its header
  uint32_t packet_t3 = 0;       // its number among the frame's type-3 packets (1 = first)
  uint32_t packet_draw = 0;     // IssueDraw calls before it in the frame (= the frame anatomy's next idx)
  bool packet_predicated = false;
};

CommandProcessor::CommandProcessor(GraphicsSystem* graphics_system,
                                   system::KernelState* kernel_state)
    : memory_(graphics_system->memory()),
      kernel_state_(kernel_state),
      graphics_system_(graphics_system),
      register_file_(graphics_system_->register_file()),
      worker_running_(true),
      write_ptr_index_event_(rex::thread::Event::CreateAutoResetEvent(false)),
      write_ptr_index_(0) {
  assert_not_null(write_ptr_index_event_);
}

CommandProcessor::~CommandProcessor() = default;

bool CommandProcessor::Initialize() {
  // Initialize the gamma ramps to their default (linear) values - taken from
  // what games set when starting with the sRGB (return value 1)
  // VdGetCurrentDisplayGamma.
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t value = i * 0x3FF / 0xFF;
    reg::DC_LUT_30_COLOR& gamma_ramp_entry = gamma_ramp_256_entry_table_[i];
    gamma_ramp_entry.color_10_blue = value;
    gamma_ramp_entry.color_10_green = value;
    gamma_ramp_entry.color_10_red = value;
  }
  for (uint32_t i = 0; i < 128; ++i) {
    reg::DC_LUT_PWL_DATA gamma_ramp_entry = {};
    gamma_ramp_entry.base = (i * 0xFFFF / 0x7F) & ~UINT32_C(0x3F);
    gamma_ramp_entry.delta = i < 0x7F ? 0x200 : 0;
    for (uint32_t j = 0; j < 3; ++j) {
      gamma_ramp_pwl_rgb_[i][j] = gamma_ramp_entry;
    }
  }

  // wwe2k14 patch (Session 78, NATIVE FRAME): the split latched before the worker starts (the backend builds its
  // drawing caches on DrawRegisterFile in SetupContext). Not with Session 51's detached / log-fed modes (their own
  // segment executor).
  nf_ = REXCVAR_GET(gpu_native_frame);
  if (nf_ && REXCVAR_GET(gpu_native_detached) != 0) {
    REXLOG_WARN("[native][frame] gpu_native_frame refused: gpu_native_detached {} is on",
                REXCVAR_GET(gpu_native_detached));
    nf_ = false;
  }
  if (nf_) {
    nf_register_file_ = std::make_unique<RegisterFile>(*register_file_);
    nf_real_register_file_ = register_file_;
    nf_role_ = NfRole::kAnswers;
    nf_answers_lean_ = REXCVAR_GET(gpu_native_frame_answers_lean);
    nf_lookahead_ = REXCVAR_GET(gpu_native_frame_dead_glare) != 0;  // (Session 79)
    REXLOG_INFO("[native][frame] on: answers from the command processor (no draws), drawing side behind it (draw {})",
                REXCVAR_GET(gpu_native_frame_draw));
  }
  // wwe2k14 patch (Session 77): det_serial drain check registered before the worker starts (kind 0 = not drained
  // until its first wait), so no decision passes while the CP has not started yet; removed again at Shutdown.
  if (det::SerialEnabled()) det_drain_id_ = det::AddDrainCheck([this] { return DetDrained(); });
  worker_running_ = true;
  worker_thread_ = system::object_ref<system::XHostThread>(
      new system::XHostThread(kernel_state_, 128 * 1024, 0, [this]() {
        WorkerThreadMain();
        return 0;
      }));
  worker_thread_->set_name("GPU Commands");
  worker_thread_->Create();

  return true;
}

void CommandProcessor::Shutdown() {
  // wwe2k14 patch (Session 77): the CP is never drained again once its loop ends: unregister (the check would call
  // into this object after it is gone), and once the loop has ended mark the CP waiting for good (it leaves its last
  // wait with the flag cleared: a scheduler decision, the vblank thread's, would wait for it forever).
  if (det_drain_id_ >= 0) {
    det::RemoveDrainCheck(det_drain_id_);
    det_drain_id_ = -1;
  }
  worker_running_ = false;
  write_ptr_index_event_->Set();
  worker_thread_->Wait(0, 0, 0, nullptr);
  worker_thread_.reset();
  if (det::SerialEnabled()) det::SetCpWaiting(true);
}

void CommandProcessor::InitializeShaderStorage(const std::filesystem::path& cache_root,
                                               uint32_t title_id, bool blocking) {}

void CommandProcessor::CallInThread(std::function<void()> fn) {
  if (!PendingFnsWaiting() && system::XThread::IsInThread(worker_thread_.get())) {
    fn();
  } else {
    {
      std::lock_guard<std::mutex> lock(g_pending_fns_mutex);
      pending_fns_.push(std::move(fn));
    }
    g_pending_fns_count.fetch_add(1, std::memory_order_release);
    write_ptr_index_event_->Set();  // (Session 95, RACE R1: an idle command thread sees it now, not at its next timed wake-up)
  }
}

void CommandProcessor::ClearCaches() {}

void CommandProcessor::InvalidateGpuMemory() {}

ReadbackResolveMode CommandProcessor::GetReadbackResolveMode(
    bool legacy_readback_resolve_enabled) const {
  // wwe2k14 patch (Session 6, CP10): IssueCopy calls this for every resolve; use the readback_resolve parse
  // cached at context setup and at every guest swap (RefreshReadbackResolveModeCache), so a change of
  // readback_resolve takes effect at the next swap. The legacy backend flag is still read on every call.
  // wwe2k14 patch (Session 57): the app's run-time mode first (one int load).
  switch (REXCVAR_GET(readback_resolve_override)) {
    case 0:
      return ReadbackResolveMode::kDisabled;
    case 1:
      return ReadbackResolveMode::kFast;
    case 2:
      return ReadbackResolveMode::kSome;
    case 3:
      return ReadbackResolveMode::kFull;
    default:
      break;
  }
  ReadbackResolveMode shared_mode;
  bool shared_mode_overrides_legacy;
  if (readback_resolve_cache_valid_) {
    shared_mode = readback_resolve_cached_shared_mode_;
    shared_mode_overrides_legacy = readback_resolve_cached_overrides_legacy_;
  } else {
    shared_mode = GetSharedReadbackResolveMode(shared_mode_overrides_legacy);
  }
  if (shared_mode_overrides_legacy) {
    return shared_mode;
  }
  return legacy_readback_resolve_enabled ? ReadbackResolveMode::kFast
                                         : ReadbackResolveMode::kDisabled;
}

void CommandProcessor::RefreshReadbackResolveModeCache() {
  readback_resolve_cached_shared_mode_ =
      GetSharedReadbackResolveMode(readback_resolve_cached_overrides_legacy_);
  readback_resolve_cache_valid_ = true;
}

void CommandProcessor::RefreshGpuDebugLogCache() {
  auto* gpu_log = ::rex::GetLoggerRaw(::rex::log::gpu());
  gpu_debug_log_may_be_enabled_ = gpu_log && gpu_log->should_log(spdlog::level::debug);
}

bool CommandProcessor::IsReadbackMemexportEnabled(bool legacy_backend_flag) const {
  if (legacy_readback_memexport_cvar_name_ &&
      rex::cvar::HasNonDefaultValue(legacy_readback_memexport_cvar_name_)) {
    return legacy_backend_flag;
  }
  return REXCVAR_GET(readback_memexport);
}

void CommandProcessor::SetDesiredSwapPostEffect(SwapPostEffect swap_post_effect) {
  if (swap_post_effect_desired_ == swap_post_effect) {
    return;
  }
  swap_post_effect_desired_ = swap_post_effect;
  CallInThread([this, swap_post_effect]() { swap_post_effect_actual_ = swap_post_effect; });
}

void ApplyCommandThreadOwnCore();  // wwe2k14 patch (Session 95, spike): graphics_system.cpp, gpu_cmd_own_core

void CommandProcessor::WorkerThreadMain() {
  ApplyCommandThreadOwnCore();  // wwe2k14 patch (Session 95, spike)
  if (!SetupContext()) {
    rex::FatalError("Unable to setup command processor internal state");
    return;
  }
  // wwe2k14 patch (Session 15): det_serial: drained = waiting with nothing to do (see DetDrained; registered in
  // Initialize since Session 77).
  // wwe2k14 patch (Session 77): det_serial_trace_detail: what the CP waits on, before each scheduler pick.
  if (det::SerialEnabled() && det::TraceDetail()) {
    det::SetCpDescribe([this] {
      char b[64];
      const int kind = det_wait_kind_.load();
      if (kind == 0) {
        std::snprintf(b, sizeof(b), "run:%X", read_ptr_index_.load(std::memory_order_relaxed));
      } else if (kind == 1) {
        std::snprintf(b, sizeof(b), "w1:%X/%X", det_wait_addr_.load(), write_ptr_index_.load());
      } else {
        std::snprintf(b, sizeof(b), "w%d:%X&%X=%X?%d", kind, det_wait_addr_.load(), det_wait_mask_.load(),
                      det_wait_ref_.load(), det_wait_func_.load());
      }
      return std::string(b);
    });
  }
  // wwe2k14 patch (Session 6, CP10): initial values of the cached setting lookups.
  RefreshReadbackResolveModeCache();
  RefreshGpuDebugLogCache();

  // wwe2k14 patch (Session 95, HANG): this thread's sign of life, busy or idle.
  auto& command_beat = rex::perf::hang_beats::Get().command;
  while (worker_running_) {
    rex::perf::hang_beats::Beat(command_beat);
    // WWE '13 Android port (entrance slowdowns, gpu_submit_safe_point_ms): see CpSafePointDue.
    if (CpSafePointDue()) {
      PrepareForWait();
    }
    while (PendingFnsWaiting()) {
      std::function<void()> fn;
      {
        std::lock_guard<std::mutex> lock(g_pending_fns_mutex);
        fn = std::move(pending_fns_.front());
        pending_fns_.pop();
      }
      g_pending_fns_count.fetch_sub(1, std::memory_order_release);
      fn();
    }

    // wwe2k14 patch (Session 87, D3D layer STAGE 2 "no ring"; src\graphics\d3d_noring.cpp): the game's typed host queue first,
    // in order (the library writes nothing to the ring while the game's no-ring mode is on). One cvar load when gpu_d3d_noring is 0.
    if (D3dNoringDrain()) {
      det::NoteCpProgress();
      continue;
    }

    uint32_t write_ptr_index = write_ptr_index_.load();
    // wwe2k14 patch (Session 51 step 3, gpu_native_detached 2): with the ring idle, the library calls' log (the library
    // is in its no-GPU mode: nothing reaches the ring).
    if ((write_ptr_index == 0xBAADF00D || read_ptr_index_.load(std::memory_order_relaxed) == write_ptr_index) && half2_log_.load() && Half2LogDrive()) {
      det::NoteCpProgress();  // (review: GPU wait loops re-check after progress, as after the ring)
      continue;
    }
    if (write_ptr_index == 0xBAADF00D || read_ptr_index_.load(std::memory_order_relaxed) == write_ptr_index) {
      SCOPE_profile_cpu_i("gpu", "rex::graphics::CommandProcessor::Stall");
      perf::timeline::Scope idle_scope("cp_idle_b", "cp_idle_e", read_ptr_index_.load(std::memory_order_relaxed));
      // We've run out of commands to execute.
      // We spin here waiting for new ones, as the overhead of waiting on our
      // event is too high.
      PrepareForWait();
      det_wait_addr_.store(read_ptr_index_.load(std::memory_order_relaxed));
      SetDetWaitKind(1);
      det::SetCpWaiting(true);  // wwe2k14 patch (Session 15): det_frames
      uint32_t loop_count = 0;
      do {
        // If we spin around too much, revert to a "low-power" state.
        if (loop_count > 500) {
          // wwe2k14 patch (Session 92, M5b): with gpu_d3d_sync_events and an ABI-3 block the producer wakes this wait (consumer_idle /
          // wake, the Dekker pair in d3d_noring.cpp) and the 5 ms is only a guard; otherwise as before (Session 87: 5 ms, 1 with the queue).
          const int idle_arm = D3dNoringIdleArm(write_ptr_index_event_.get());
          if (idle_arm != 2) {
            const int wait_time_ms = D3dNoringIdleWaitMs();
            const auto idle_result = idle_arm == 1
                                         ? D3dNoringIdleWait(write_ptr_index_event_.get(), wait_time_ms)
                                         : rex::thread::Wait(write_ptr_index_event_.get(), true,
                                                             std::chrono::milliseconds(wait_time_ms));
            if (idle_arm == 1) D3dNoringIdleDisarm(idle_result == rex::thread::WaitResult::kTimeout);
          }
        }

        rex::thread::MaybeYield();
        loop_count++;
        rex::perf::hang_beats::Beat(command_beat);  // wwe2k14 patch (Session 95, HANG)
        write_ptr_index = write_ptr_index_.load();
      } while (worker_running_ && !PendingFnsWaiting() &&
               (write_ptr_index == 0xBAADF00D || read_ptr_index_.load(std::memory_order_relaxed) == write_ptr_index) &&
               !(half2_log_.load(std::memory_order_relaxed) && Half2LogHasSelfWork()) &&
               !D3dNoringHasWork());  // (wwe2k14 patch, Session 87: the host queue is not empty)
      ReturnFromWait();
      det::SetCpWaiting(false);  // wwe2k14 patch (Session 15): det_frames
      SetDetWaitKind(0);
      // (Session 51 review: a wake-up for the log leaves the ring idle - never run the primary buffer then, it would
      // execute a whole stale lap.)
      if (!worker_running_ || PendingFnsWaiting() || write_ptr_index == 0xBAADF00D ||
          read_ptr_index_.load(std::memory_order_relaxed) == write_ptr_index) {
        continue;
      }
    }
    assert_true(read_ptr_index_.load(std::memory_order_relaxed) != write_ptr_index);

    // wwe2k14 patch (Session 92, M5c): ring packets run in order after everything the host queue held back (a fence queued behind an open
    // pacing gate must not be overtaken by a ring packet's stores): wait for the flip word and release the gate first. No-op with the latch off.
    D3dNoringGateWaitRelease(false);

    // Execute. Note that we handle wraparound transparently.
    read_ptr_index_.store(ExecutePrimaryBuffer(read_ptr_index_.load(std::memory_order_relaxed), write_ptr_index), std::memory_order_relaxed);
    det::NoteCpProgress();  // wwe2k14 patch (Session 15): det_serial (GPU wait loops re-check after progress)

    // TODO(benvanik): use reader->Read_update_freq_ and only issue after moving
    //     that many indices.
    if (read_ptr_writeback_ptr_) {
      OnGuestVisibleWrite(GuestVisibleEvent::kRptrWriteback);  // wwe2k14 patch (Session 66, H1.1)
      memory::store_and_swap<uint32_t>(memory_->TranslatePhysical(read_ptr_writeback_ptr_),
                                       read_ptr_index_.load(std::memory_order_relaxed));
    }

    // FIXME: We're supposed to process the WAIT_UNTIL register at this point,
    // but no games seem to actually use it.
  }

  ShutdownContext();
}

void CommandProcessor::Pause() {
  if (paused_) {
    return;
  }
  paused_ = true;

  thread::Fence fence;
  CallInThread([&fence]() {
    fence.Signal();
    thread::Thread::GetCurrentThread()->Suspend();
  });

  fence.Wait();
}

void CommandProcessor::Resume() {
  if (!paused_) {
    return;
  }
  paused_ = false;

  worker_thread_->thread()->Resume();
}

bool CommandProcessor::Save(::rex::stream::ByteStream* stream) {
  assert_true(paused_);

  stream->Write<uint32_t>(primary_buffer_ptr_);
  stream->Write<uint32_t>(primary_buffer_size_);
  stream->Write<uint32_t>(read_ptr_index_.load(std::memory_order_relaxed));
  stream->Write<uint32_t>(read_ptr_update_freq_);
  stream->Write<uint32_t>(read_ptr_writeback_ptr_);
  stream->Write<uint32_t>(write_ptr_index_.load());

  return true;
}

bool CommandProcessor::Restore(::rex::stream::ByteStream* stream) {
  assert_true(paused_);

  primary_buffer_ptr_ = stream->Read<uint32_t>();
  primary_buffer_size_ = stream->Read<uint32_t>();
  read_ptr_index_.store(stream->Read<uint32_t>(), std::memory_order_relaxed);
  read_ptr_update_freq_ = stream->Read<uint32_t>();
  read_ptr_writeback_ptr_ = stream->Read<uint32_t>();
  write_ptr_index_.store(stream->Read<uint32_t>());

  return true;
}

bool CommandProcessor::SetupContext() {
  return true;
}

void CommandProcessor::ShutdownContext() {}

void CommandProcessor::InitializeRingBuffer(uint32_t ptr, uint32_t size_log2) {
  read_ptr_index_.store(0, std::memory_order_relaxed);
  primary_buffer_ptr_ = ptr;
  primary_buffer_size_ = uint32_t(1) << (size_log2 + 3);
}

void CommandProcessor::EnableReadPointerWriteBack(uint32_t ptr, uint32_t block_size_log2) {
  // CP_RB_RPTR_ADDR Ring Buffer Read Pointer Address 0x70C
  // ptr = RB_RPTR_ADDR, pointer to write back the address to.
  read_ptr_writeback_ptr_ = ptr;
  // CP_RB_CNTL Ring Buffer Control 0x704
  // block_size = RB_BLKSZ, log2 of number of quadwords read between updates of
  //              the read pointer.
  read_ptr_update_freq_ = uint32_t(1) << block_size_log2 >> 2;
}

// wwe2k14 patch (Session 15): det_serial drain check (scheduler thread): true when the wait the command processor
// is in would not end now (nothing new to execute / the polled value still does not match).
// wwe2k14 patch (Session 77): the wait kind alone decides; 0 (executing) is NOT drained. Before, the scheduler read
// the separate "waiting" flag (still set) and then the kind (already 0 = "default: drained") while the CP was leaving a
// wait to run new commands: the next guest turn started before those commands' GPU interrupts were queued, so the
// interrupts ran one turn later in some runs (0.85% of det vblanks in runs 4841 / 4842; found with
// det_serial_trace_detail in 4843 / 4844: "[run:89B]7" = a pick while the CP ran). Kind 4 = blocked on its own
// interrupt request, which the scheduler must run (drained).
bool CommandProcessor::DetDrained() {
  // Session 77: one read of the kind (after the sequence; a wait's fields are stored before its SetDetWaitKind), and
  // drained only if the CP is still in that same wait after the check: waiting, same kind, no kind change in
  // between (it may leave the wait, or move on to another wait of the same kind, while it is evaluated).
  const uint32_t seq = det_wait_seq_.load();
  const int kind = det_wait_kind_.load();
  if (!det::CpWaiting()) return false;
  const bool drained = DetDrainedFor(kind);
  return drained && det::CpWaiting() && det_wait_kind_.load() == kind && det_wait_seq_.load() == seq;
}

bool CommandProcessor::DetDrainedFor(int kind) {
  switch (kind) {
    case 0:
      return false;
    case 4:
      return true;
    case 1: {
      // Session 77 (review): the idle loop also ends for the library calls' log (gpu_native_detached 2).
      if (Half2LogHasPublishedSelfWork()) return false;
      if (D3dNoringHasWork()) return false;  // wwe2k14 patch (Session 87): items queued by the game are work too
      const uint32_t w = write_ptr_index_.load();
      return w == 0xBAADF00D || w == det_wait_addr_.load();
    }
    case 2:
    case 3: {
      const uint32_t addr = det_wait_addr_.load();
      uint32_t value;
      if (kind == 2) {
        value = *reinterpret_cast<uint32_t*>(memory_->TranslatePhysical(addr & ~uint32_t(0x3)));
        value = xenos::GpuSwap(value, static_cast<xenos::Endian>(addr & 0x3));
      } else {
        if (addr == XE_GPU_REG_COHER_STATUS_HOST) return false;  // the command processor makes it coherent itself
        value = ReadRegisterValue(addr);
      }
      const uint32_t v = value & det_wait_mask_.load(), ref = det_wait_ref_.load();
      bool matched = false;
      switch (det_wait_func_.load()) {
        case 0x1: matched = v < ref; break;
        case 0x2: matched = v <= ref; break;
        case 0x3: matched = v == ref; break;
        case 0x4: matched = v != ref; break;
        case 0x5: matched = v >= ref; break;
        case 0x6: matched = v > ref; break;
        case 0x7: matched = true; break;
        default: matched = false; break;
      }
      return !matched;
    }
    default:
      return true;
  }
}

void CommandProcessor::UpdateWritePointer(uint32_t value) {
  perf::timeline::Record("wptr", value);
  write_ptr_index_ = value;
  write_ptr_index_event_->Set();
}

uint32_t CommandProcessor::ReadRegisterValue(uint32_t index) const {
  if (index >= 0x2000) NfNoteHighRead(index, "an answer packet");  // wwe2k14 patch (Session 79): census
  if (index < RegisterFile::kRegisterCount) {
    return register_file_->values[index];
  }
  auto it = extended_register_values_.find(index);
  return it != extended_register_values_.end() ? it->second : 0;
}

void CommandProcessor::WriteRegister(uint32_t index, uint32_t value) {
  // wwe2k14 patch (Session 6, CP1B): gpu_type0_block_split skips this for "plain" registers; any register given
  // side effects here must be added to kRegisterSplitIntervals and IsPlainRegisterWrite.
  rex::graphics::free_counters::Bump(rex::graphics::free_counters::g_c.write_registers);  // (Session 92 M0: log-only counter)
  rex::graphics::free_counters::BumpIndex(rex::graphics::free_counters::g_register_writes, index);  // (M5a, log only)
  RegisterFile& regs = *register_file_;
  if (index >= RegisterFile::kRegisterCount) {
    auto [it, inserted] = extended_register_values_.insert_or_assign(index, value);
    (void)it;
    if (inserted) {
      REXGPU_WARN(
          "CommandProcessor::WriteRegister index out of bounds: {} (stored as extended register)",
          index);
    }
    return;
  }

  // wwe2k14 patch (Session 78, NATIVE FRAME): the drawing side writes its own file and nothing else (the scratch
  // write-backs, coherency and gamma-ramp side effects are the answers side's); the answers side notes no repeats.
  if (nf_role_ == NfRole::kDraw) {
    if (regs.values[index] != value || (index >= 0x4000 && index <= 0x47FF)) {
      RepeatNoteRegisters(index, 1);
    }
    regs.values[index] = value;
    return;
  }
  // wwe2k14 patch (Session 23, REPEAT): a write that changes the value (a float constant: any write).
  if (nf_role_ != NfRole::kAnswers && (regs.values[index] != value || (index >= 0x4000 && index <= 0x47FF))) {
    RepeatNoteRegisters(index, 1);
  }
  // wwe2k14 patch (Session 74): a scratch register's new value is the game's "the GPU is past this" signal, and the
  // guest reads it through MMIO the moment it is stored - so the queued resolve readbacks land BEFORE the store, as
  // every other flush point does before its write. (Session 66 flushed after it: a guest polling the register could
  // see the signal while the flush still waited for the GPU and read stale destinations. Not the cause of Undertaker's
  // white marks on the phone - those stayed with this fixed, phone run 094; see readback_resolve_batch_min_bytes.)
  if (index >= XE_GPU_REG_SCRATCH_REG0 && index <= XE_GPU_REG_SCRATCH_REG7) {
    OnGuestVisibleWrite(GuestVisibleEvent::kScratchWriteback);
  }
  // Volatile for the WAIT_REG_MEM loop.
  const_cast<volatile uint32_t&>(regs.values[index]) = value;
  // wwe2k14 patch: the register table lookup (a large switch) only feeds a debug message; skip it unless GPU
  // debug logging is on (this runs for every register write).
  // wwe2k14 patch (Session 6, CP10): the logger check itself (a guarded function-local static, a cross-DLL
  // GetLoggerRaw call and should_log) is cached in gpu_debug_log_may_be_enabled_, refreshed at the start
  // of every ExecutePrimaryBuffer. When it's set, the full original check still runs, so the output is the
  // same, except that enabling GPU debug logging mid-buffer takes effect at the next primary buffer.
  if (gpu_debug_log_may_be_enabled_) {
    if (auto* gpu_log = ::rex::GetLoggerRaw(::rex::log::gpu());
        gpu_log && gpu_log->should_log(spdlog::level::debug) && !regs.GetRegisterInfo(index)) {
      REXGPU_DEBUG("GPU: Write to unknown register ({:04X} = {:08X})", index, value);
    }
  }

  // Scratch register writeback.
  if (index >= XE_GPU_REG_SCRATCH_REG0 && index <= XE_GPU_REG_SCRATCH_REG7) {
    // (wwe2k14 patch, Session 66 / 74: the queued readbacks were flushed before the register store above.)
    uint32_t scratch_reg = index - XE_GPU_REG_SCRATCH_REG0;
    if ((1 << scratch_reg) & regs.values[XE_GPU_REG_SCRATCH_UMSK]) {
      // Enabled - write to address.
      uint32_t scratch_addr = regs.values[XE_GPU_REG_SCRATCH_ADDR];
      uint32_t mem_addr = scratch_addr + (scratch_reg * 4);
      memory::store_and_swap<uint32_t>(memory_->TranslatePhysical(mem_addr), value);
    }
  } else {
    switch (index) {
      // If this is a COHER register, set the dirty flag.
      // This will block the command processor the next time it WAIT_REG_MEMs
      // and allow us to synchronize the memory.
      case XE_GPU_REG_COHER_STATUS_HOST: {
        const_cast<volatile uint32_t&>(regs.values[index]) |= UINT32_C(0x80000000);
      } break;

      case XE_GPU_REG_DC_LUT_RW_INDEX: {
        // Reset the sequential read / write component index (see the M56
        // DC_LUT_SEQ_COLOR documentation).
        gamma_ramp_rw_component_ = 0;
      } break;

      case XE_GPU_REG_DC_LUT_SEQ_COLOR: {
        // Should be in the 256-entry table writing mode.
        assert_zero(regs[XE_GPU_REG_DC_LUT_RW_MODE] & 0b1);
        auto gamma_ramp_rw_index = regs.Get<reg::DC_LUT_RW_INDEX>();
        // DC_LUT_SEQ_COLOR is in the red, green, blue order, but the write
        // enable mask is blue, green, red.
        bool write_gamma_ramp_component = (regs[XE_GPU_REG_DC_LUT_WRITE_EN_MASK] &
                                           (UINT32_C(1) << (2 - gamma_ramp_rw_component_))) != 0;
        if (write_gamma_ramp_component) {
          reg::DC_LUT_30_COLOR& gamma_ramp_entry =
              gamma_ramp_256_entry_table_[gamma_ramp_rw_index.rw_index];
          // Bits 0:5 are hardwired to zero.
          uint32_t gamma_ramp_seq_color = regs.Get<reg::DC_LUT_SEQ_COLOR>().seq_color >> 6;
          switch (gamma_ramp_rw_component_) {
            case 0:
              gamma_ramp_entry.color_10_red = gamma_ramp_seq_color;
              break;
            case 1:
              gamma_ramp_entry.color_10_green = gamma_ramp_seq_color;
              break;
            case 2:
              gamma_ramp_entry.color_10_blue = gamma_ramp_seq_color;
              break;
          }
        }
        if (++gamma_ramp_rw_component_ >= 3) {
          gamma_ramp_rw_component_ = 0;
          reg::DC_LUT_RW_INDEX new_gamma_ramp_rw_index = gamma_ramp_rw_index;
          ++new_gamma_ramp_rw_index.rw_index;
          WriteRegister(XE_GPU_REG_DC_LUT_RW_INDEX,
                        rex::memory::Reinterpret<uint32_t>(new_gamma_ramp_rw_index));
        }
        if (write_gamma_ramp_component) {
          OnGammaRamp256EntryTableValueWritten();
        }
      } break;

      case XE_GPU_REG_DC_LUT_PWL_DATA: {
        // Should be in the PWL writing mode.
        assert_not_zero(regs[XE_GPU_REG_DC_LUT_RW_MODE] & 0b1);
        auto gamma_ramp_rw_index = regs.Get<reg::DC_LUT_RW_INDEX>();
        // Bit 7 of the index is ignored for PWL.
        uint32_t gamma_ramp_rw_index_pwl = gamma_ramp_rw_index.rw_index & 0x7F;
        // DC_LUT_PWL_DATA is likely in the red, green, blue order because
        // DC_LUT_SEQ_COLOR is, but the write enable mask is blue, green, red.
        bool write_gamma_ramp_component = (regs[XE_GPU_REG_DC_LUT_WRITE_EN_MASK] &
                                           (UINT32_C(1) << (2 - gamma_ramp_rw_component_))) != 0;
        if (write_gamma_ramp_component) {
          reg::DC_LUT_PWL_DATA& gamma_ramp_entry =
              gamma_ramp_pwl_rgb_[gamma_ramp_rw_index_pwl][gamma_ramp_rw_component_];
          auto gamma_ramp_value = regs.Get<reg::DC_LUT_PWL_DATA>();
          // Bits 0:5 are hardwired to zero.
          gamma_ramp_entry.base = gamma_ramp_value.base & ~UINT32_C(0x3F);
          gamma_ramp_entry.delta = gamma_ramp_value.delta & ~UINT32_C(0x3F);
        }
        if (++gamma_ramp_rw_component_ >= 3) {
          gamma_ramp_rw_component_ = 0;
          reg::DC_LUT_RW_INDEX new_gamma_ramp_rw_index = gamma_ramp_rw_index;
          // TODO(Triang3l): Should this increase beyond 7 bits for PWL?
          // Direct3D 9 explicitly sets rw_index to 0x80 after writing the last
          // PWL entry. However, the DC_LUT_RW_INDEX documentation says that for
          // PWL, the bit 7 is ignored.
          new_gamma_ramp_rw_index.rw_index = (gamma_ramp_rw_index.rw_index & ~UINT32_C(0x7F)) |
                                             ((gamma_ramp_rw_index_pwl + 1) & 0x7F);
          WriteRegister(XE_GPU_REG_DC_LUT_RW_INDEX,
                        rex::memory::Reinterpret<uint32_t>(new_gamma_ramp_rw_index));
        }
        if (write_gamma_ramp_component) {
          OnGammaRampPWLValueWritten();
        }
      } break;

      case XE_GPU_REG_DC_LUT_30_COLOR: {
        // Should be in the 256-entry table writing mode.
        assert_zero(regs[XE_GPU_REG_DC_LUT_RW_MODE] & 0b1);
        auto gamma_ramp_rw_index = regs.Get<reg::DC_LUT_RW_INDEX>();
        uint32_t gamma_ramp_write_enable_mask = regs[XE_GPU_REG_DC_LUT_WRITE_EN_MASK] & 0b111;
        if (gamma_ramp_write_enable_mask) {
          reg::DC_LUT_30_COLOR& gamma_ramp_entry =
              gamma_ramp_256_entry_table_[gamma_ramp_rw_index.rw_index];
          auto gamma_ramp_value = regs.Get<reg::DC_LUT_30_COLOR>();
          if (gamma_ramp_write_enable_mask & 0b001) {
            gamma_ramp_entry.color_10_blue = gamma_ramp_value.color_10_blue;
          }
          if (gamma_ramp_write_enable_mask & 0b010) {
            gamma_ramp_entry.color_10_green = gamma_ramp_value.color_10_green;
          }
          if (gamma_ramp_write_enable_mask & 0b100) {
            gamma_ramp_entry.color_10_red = gamma_ramp_value.color_10_red;
          }
        }
        // TODO(Triang3l): Should this reset the component write index? If this
        // increase is assumed to behave like a full DC_LUT_RW_INDEX write, it
        // probably should. Currently this also calls WriteRegister for
        // DC_LUT_RW_INDEX, which resets gamma_ramp_rw_component_ as well.
        gamma_ramp_rw_component_ = 0;
        reg::DC_LUT_RW_INDEX new_gamma_ramp_rw_index = gamma_ramp_rw_index;
        ++new_gamma_ramp_rw_index.rw_index;
        WriteRegister(XE_GPU_REG_DC_LUT_RW_INDEX,
                      rex::memory::Reinterpret<uint32_t>(new_gamma_ramp_rw_index));
        if (gamma_ramp_write_enable_mask) {
          OnGammaRamp256EntryTableValueWritten();
        }
      } break;
    }
  }
}

void CommandProcessor::WriteRegistersFromMem(uint32_t start_index, uint32_t* base,
                                             uint32_t num_registers) {
  // wwe2k14 patch (Session 6, CP1B): gpu_type0_block_split. The backends call this for ranges that aren't within
  // one shader constant block; split them by block instead of writing every register through WriteRegister.
  if (!type0_block_split_in_constant_block_ && Type0BlockSplitActive()) {
    WriteRegistersFromMemSplit(start_index, base, num_registers);
    return;
  }
  for (uint32_t i = 0; i < num_registers; ++i) {
    uint32_t data = memory::load_and_swap<uint32_t>(base + i);
    WriteRegister(start_index + i, data);
  }
}

bool CommandProcessor::Type0BlockSplitActive() const {
  // wwe2k14 patch (Session 6, CP1B): only together with gpu_type0_bulk_writes, and not while the GPU logger may
  // log at debug level (WriteRegister's unknown-register message; gpu_debug_log_may_be_enabled_ is refreshed at
  // the start of every primary buffer, as for CP10).
  return REXCVAR_GET(gpu_type0_block_split) && !gpu_debug_log_may_be_enabled_ &&
         REXCVAR_GET(gpu_type0_bulk_writes);
}

void CommandProcessor::WriteRegistersFromMemSplit(uint32_t start_index, uint32_t* base,
                                                  uint32_t num_registers) {
  // wwe2k14 patch (Session 6, CP1B): gpu_type0_block_split, after Canary's D3D12
  // WriteRegisterRangeFromMem_WithKnownBound. Walks the range in register order, one run of registers of the same
  // class at a time (kRegisterSplitIntervals):
  // - plain registers: stored with no further work, exactly what every WriteRegister does for them;
  // - a shader constant block: the virtual WriteRegistersFromMem for the sub-range within the block, which the
  //   backends write in bulk (copy plus the same invalidation as their per-register writes, for the whole
  //   sub-range at once; CP1);
  // - any other register: the virtual WriteRegister, as the per-register loop does.
  // Each run is written only after all earlier runs, and the source dwords of a run are read only when it's
  // written, so a register with side effects sees every earlier register of the range already written, and any
  // memory it changes is seen by later runs, as with the per-register loop.
  if (!num_registers) {
    return;
  }
  // Exclusive end. Indices come from 15- or 16-bit packet fields plus a block offset, far from overflowing.
  const uint32_t end_index = start_index + num_registers;
  uint32_t plain_count = 0, constant_count = 0, per_register_count = 0;
  Type0BlockSplitStats& stats = type0_block_split_stats_;
  const RegisterSplitInterval* interval = kRegisterSplitIntervals;
  uint32_t index = start_index;
  while (index < end_index) {
    // The last interval ends at UINT32_MAX, so this stays within the list.
    while (index >= interval->end) {
      ++interval;
    }
    uint32_t* source = base + (index - start_index);
    if (index < interval->first) {
      uint32_t run = std::min(end_index, interval->first) - index;
      uint32_t* dest = register_file_->values + index;
      // wwe2k14 patch (Session 23, REPEAT): plain registers count only if a value changes - noted as the whole run
      // (review fix: noting only the first changed register could hide later changes in the run when that first
      // one is a draw-packet register, which RepeatNoteRegisters ignores on its own; a run of several registers
      // is never ignored).
      for (uint32_t i = 0; i < run && nf_role_ != NfRole::kAnswers; ++i) {  // (Session 78: the drawing side's)
        if (dest[i] != memory::load_and_swap<uint32_t>(source + i)) {
          RepeatNoteRegisters(index, run);
          break;
        }
      }
      if (run >= kPlainRunCopyAndSwapThreshold) {
        memory::copy_and_swap(dest, source, run);
      } else {
        for (uint32_t i = 0; i < run; ++i) {
          dest[i] = memory::load_and_swap<uint32_t>(source + i);
        }
      }
      ++stats.plain_runs;
      plain_count += run;
      index += run;
      continue;
    }
    uint32_t run = std::min(end_index, interval->end) - index;
    if (interval->constant_block) {
      type0_block_split_in_constant_block_ = true;
      WriteRegistersFromMem(index, source, run);
      type0_block_split_in_constant_block_ = false;
      ++stats.constant_runs;
      constant_count += run;
    } else {
      for (uint32_t i = 0; i < run; ++i) {
        WriteRegister(index + i, memory::load_and_swap<uint32_t>(source + i));
      }
      per_register_count += run;
    }
    index += run;
  }
  ++stats.ranges;
  stats.plain_registers += plain_count;
  stats.constant_registers += constant_count;
  stats.per_register_registers += per_register_count;
  if (REXCVAR_GET(gpu_type0_block_split_verify)) {
    VerifyRegistersFromMemSplit(start_index, base, num_registers, plain_count, constant_count,
                                per_register_count);
  }
}

void CommandProcessor::VerifyRegistersFromMemSplit(uint32_t start_index, const uint32_t* base,
                                                   uint32_t num_registers, uint32_t plain_count,
                                                   uint32_t constant_count,
                                                   uint32_t per_register_count) {
  // wwe2k14 patch (Session 6, CP1B): gpu_type0_block_split_verify, the same approach as
  // gpu_type0_bulk_writes_verify (ExecutePacketType0BulkVerify). Right after the split wrote the range:
  // - recount which registers had to take which path with the short tests (IsShaderConstantBlockRegister,
  //   IsPlainRegisterWrite) and compare with what the split did, so every register with side effects is known
  //   to have gone through the per-register WriteRegister;
  // - replay the plain registers and shader constants, in register order, through the per-register
  //   WriteRegister, checking that each already holds its source value and that the replay changes none of the
  //   backend's invalidation bits (so the split invalidated at least what per-register writes would). Their
  //   per-register writes are idempotent (a store, and invalidation bits that are only cleared), so the replay
  //   leaves the state as the per-register path would even after a mismatch.
  // The registers with side effects aren't replayed (their handling isn't idempotent: gamma ramp index
  // increments, the scratch memory write-back; COHER_STATUS_HOST and DC_LUT_RW_INDEX don't keep the written
  // value), and the split wrote them through that very per-register path. Not detected: the split invalidating
  // more than needed (only a cost; the constant sub-ranges use the CP1 range checks, the exact union of the
  // per-register ones). The source is read again; a LOAD_ALU_CONSTANT range in guest memory that the guest
  // rewrites in between would show up as a false value mismatch.
  Type0BlockSplitStats& stats = type0_block_split_stats_;
  RegisterWriteInvalidationState state_split;
  bool have_state = GetRegisterWriteInvalidationState(state_split);
  uint32_t expected_plain = 0, expected_constant = 0, expected_per_register = 0;
  uint32_t value_mismatch_count = 0;
  uint32_t first_mismatch_index = 0, first_mismatch_split = 0, first_mismatch_expected = 0;
  for (uint32_t i = 0; i < num_registers; ++i) {
    uint32_t index = start_index + i;
    if (IsShaderConstantBlockRegister(index)) {
      ++expected_constant;
    } else if (IsPlainRegisterWrite(index)) {
      ++expected_plain;
    } else {
      ++expected_per_register;
      continue;
    }
    uint32_t expected = memory::load_and_swap<uint32_t>(base + i);
    uint32_t split_value = register_file_->values[index];
    if (split_value != expected) {
      if (!value_mismatch_count) {
        first_mismatch_index = index;
        first_mismatch_split = split_value;
        first_mismatch_expected = expected;
      }
      ++value_mismatch_count;
    }
    WriteRegister(index, expected);
  }
  RegisterWriteInvalidationState state_replayed;
  bool invalidation_mismatch =
      have_state && GetRegisterWriteInvalidationState(state_replayed) && state_replayed != state_split;
  bool path_mismatch = expected_plain != plain_count || expected_constant != constant_count ||
                       expected_per_register != per_register_count;

  ++stats.verify_ranges;
  stats.verify_registers += num_registers;
  stats.verify_not_replayed += expected_per_register;
  stats.verify_value_mismatches += value_mismatch_count;
  stats.verify_path_mismatches += path_mismatch ? 1 : 0;
  stats.verify_invalidation_mismatches += invalidation_mismatch ? 1 : 0;
  if ((value_mismatch_count || path_mismatch || invalidation_mismatch) &&
      stats.verify_mismatches_logged < 8) {
    ++stats.verify_mismatches_logged;
    REXGPU_WARN(
        "[s6][CP1B] verify mismatch, split write of {} registers at {:04X} (frame {}): {} value mismatches "
        "(first {:04X}: split {:08X}, source {:08X}); registers plain/constant/per-register: split {}/{}/{}, "
        "expected {}/{}/{}; invalidation bits after split {:08X} {:016X} {:016X} {:08X}, after per-register "
        "replay {:08X} {:016X} {:016X} {:08X}",
        num_registers, start_index, counter_, value_mismatch_count, first_mismatch_index,
        first_mismatch_split, first_mismatch_expected, plain_count, constant_count, per_register_count,
        expected_plain, expected_constant, expected_per_register, state_split.bits[0], state_split.bits[1],
        state_split.bits[2], state_split.bits[3], state_replayed.bits[0], state_replayed.bits[1],
        state_replayed.bits[2], state_replayed.bits[3]);
  }
}

void CommandProcessor::Type0BlockSplitFrameEnd() {
  // wwe2k14 patch (Session 6, CP1B): "[s6][CP1B]" summary every ~300 frames while gpu_type0_block_split did
  // anything. Per-frame figures are averages over the window.
  Type0BlockSplitStats& stats = type0_block_split_stats_;
  if (!stats.debug_log_note_logged && gpu_debug_log_may_be_enabled_ &&
      REXCVAR_GET(gpu_type0_block_split)) {
    stats.debug_log_note_logged = true;
    REXGPU_INFO("[s6][CP1B] gpu_type0_block_split has no effect while GPU debug logging is enabled");
  }
  if (!stats.summary_frame_valid) {
    if (stats.any_counted()) {
      stats.summary_frame_valid = true;
      stats.last_summary_frame = counter_;
    }
    return;
  }
  uint32_t frames = counter_ - stats.last_summary_frame;
  if (frames < 300) {
    return;
  }
  if (stats.any_counted()) {
    double per_frame = 1.0 / double(frames);
    REXGPU_INFO(
        "[s6][CP1B] frames {}-{}: type-0 range packets {:.0f}/frame ({:.0f} registers/frame, of which {:.0f} "
        "split and {:.0f} whole single-block bulk); split ranges (any packet) {:.0f}/frame: registers/frame "
        "{:.0f} plain copied ({:.0f} runs), {:.0f} shader constants in bulk ({:.0f} runs), {:.1f} per-register; "
        "write_one_reg registers/frame {:.1f} stored inline, {:.1f} through WriteRegister; verify: {} ranges "
        "({} registers, {} not replayed), mismatches: {} values, {} paths, {} invalidation",
        stats.last_summary_frame, counter_, stats.type0_range_packets * per_frame,
        stats.type0_range_registers * per_frame, stats.type0_split_registers * per_frame,
        (stats.type0_range_registers - stats.type0_split_registers) * per_frame, stats.ranges * per_frame,
        stats.plain_registers * per_frame, stats.plain_runs * per_frame,
        stats.constant_registers * per_frame, stats.constant_runs * per_frame,
        stats.per_register_registers * per_frame, stats.one_reg_inline_registers * per_frame,
        stats.one_reg_virtual_registers * per_frame, stats.verify_ranges, stats.verify_registers,
        stats.verify_not_replayed, stats.verify_value_mismatches, stats.verify_path_mismatches,
        stats.verify_invalidation_mismatches);
  }
  uint64_t mismatches_logged = stats.verify_mismatches_logged;
  bool debug_log_note_logged = stats.debug_log_note_logged;
  stats = Type0BlockSplitStats();
  stats.verify_mismatches_logged = mismatches_logged;
  stats.debug_log_note_logged = debug_log_note_logged;
  stats.summary_frame_valid = true;
  stats.last_summary_frame = counter_;
}

void CommandProcessor::WriteRegisterRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                                  uint32_t num_registers) {
  if (!num_registers) {
    return;
  }
  memory::RingBuffer::ReadRange range = ring->BeginRead(size_t(num_registers) * sizeof(uint32_t));
  if (range.first_length != 0) {
    uint32_t first_count = uint32_t(range.first_length / sizeof(uint32_t));
    WriteRegistersFromMem(base, reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(range.first)),
                          first_count);
    base += first_count;
  }
  if (range.second_length != 0) {
    WriteRegistersFromMem(base, reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(range.second)),
                          uint32_t(range.second_length / sizeof(uint32_t)));
  }
  ring->EndRead(range);
}

void CommandProcessor::WriteALURangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                             uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x4000, num_registers);
}

void CommandProcessor::WriteFetchRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                               uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x4800, num_registers);
}

void CommandProcessor::WriteBoolRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                              uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x4900, num_registers);
}

void CommandProcessor::WriteLoopRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                              uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x4908, num_registers);
}

void CommandProcessor::WriteREGISTERSRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                                   uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x2000, num_registers);
}

void CommandProcessor::WriteALURangeFromMem(uint32_t start_index, uint32_t* base,
                                            uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4000, base, num_registers);
}

void CommandProcessor::WriteFetchRangeFromMem(uint32_t start_index, uint32_t* base,
                                              uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4800, base, num_registers);
}

void CommandProcessor::WriteBoolRangeFromMem(uint32_t start_index, uint32_t* base,
                                             uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4900, base, num_registers);
}

void CommandProcessor::WriteLoopRangeFromMem(uint32_t start_index, uint32_t* base,
                                             uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4908, base, num_registers);
}

void CommandProcessor::WriteREGISTERSRangeFromMem(uint32_t start_index, uint32_t* base,
                                                  uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x2000, base, num_registers);
}

void CommandProcessor::MakeCoherent() {
  SCOPE_profile_cpu_f("gpu");

  // Status host often has 0x01000000 or 0x03000000.
  // This is likely toggling VC (vertex cache) or TC (texture cache).
  // Or, it also has a direction in here maybe - there is probably
  // some way to check for dest coherency (what all the COHER_DEST_BASE_*
  // registers are for).
  // Best docs I've found on this are here:
  // https://web.archive.org/web/20160711162346/https://amd-dev.wpengine.netdna-cdn.com/wordpress/media/2013/10/R6xx_R7xx_3D.pdf
  // https://cgit.freedesktop.org/xorg/driver/xf86-video-radeonhd/tree/src/r6xx_accel.c?id=3f8b6eccd9dba116cc4801e7f80ce21a879c67d2#n454

  // Volatile because this may be called from the WAIT_REG_MEM loop.
  volatile uint32_t* regs_volatile = register_file_->values;
  auto status_host = rex::memory::Reinterpret<reg::COHER_STATUS_HOST>(
      uint32_t(regs_volatile[XE_GPU_REG_COHER_STATUS_HOST]));
  uint32_t base_host = regs_volatile[XE_GPU_REG_COHER_BASE_HOST];
  uint32_t size_host = regs_volatile[XE_GPU_REG_COHER_SIZE_HOST];

  if (!status_host.status) {
    return;
  }

  const char* action = "N/A";
  if (status_host.vc_action_ena && status_host.tc_action_ena) {
    action = "VC | TC";
  } else if (status_host.tc_action_ena) {
    action = "TC";
  } else if (status_host.vc_action_ena) {
    action = "VC";
  }

  // TODO(benvanik): notify resource cache of base->size and type.
  REXGPU_TRACE("Make {:08X} -> {:08X} ({}b) coherent, action = {}", base_host,
               base_host + size_host, size_host, action);

  // Mark coherent.
  regs_volatile[XE_GPU_REG_COHER_STATUS_HOST] = 0;
}

// WWE '13 Android port (entrance slowdowns): when this thread runs behind the game it never runs out of commands, so the
// idle-time submission (PrepareForWait -> vulkan_submit_on_wait) never happens and a whole heavy frame reaches the GPU
// only with its swap - the GPU then finishes ~10 ms after the swap, the flip misses its vblank and the game drops to
// 30 fps until the load eases. The top of this loop, between two commands, is the same point where the idle-time
// submission happens; with gpu_submit_safe_point_ms > 0 it also happens there once the work has been building up that
// long since the last submission (CpNoteSubmissionEnded, called by the backend).

static std::atomic<int64_t> g_cp_last_submission_us{0};
static int64_t CpNowUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
void CpNoteSubmissionEnded() { g_cp_last_submission_us.store(CpNowUs(), std::memory_order_relaxed); }
bool CpSafePointDue() {
  const int32_t ms = REXCVAR_GET(gpu_submit_safe_point_ms);
  if (ms <= 0) return false;
  const int64_t now = CpNowUs();
  if (now - g_cp_last_submission_us.load(std::memory_order_relaxed) < int64_t(ms) * 1000) return false;
  g_cp_last_submission_us.store(now, std::memory_order_relaxed);  // (also when nothing was open: check again in ms)
  return true;
}
void CommandProcessor::PrepareForWait() {}

void CommandProcessor::ReturnFromWait() {}

uint32_t CommandProcessor::ExecutePrimaryBuffer(uint32_t read_index, uint32_t write_index) {
  SCOPE_profile_cpu_f("gpu");

  // wwe2k14 patch (Session 6, CP10): see WriteRegister.
  RefreshGpuDebugLogCache();

  // Execute commands!
  memory::RingBuffer reader(memory_->TranslatePhysical(primary_buffer_ptr_), primary_buffer_size_);
  reader.set_read_offset(read_index * sizeof(uint32_t));
  reader.set_write_offset(write_index * sizeof(uint32_t));
  do {
    // wwe2k14 patch (Session 78, NATIVE FRAME): a ring-level packet other than a segment call runs on the drawing
    // side first (the segments themselves: ExecutePacketType3_INDIRECT_BUFFER).
    if (nf_ && nf_role_ == NfRole::kAnswers) {
      memory::RingBuffer peek = reader;
      const uint32_t header = peek.ReadAndSwap<uint32_t>();
      const uint32_t opcode = (header >> 8) & 0x7F;
      if (header && !((header >> 30) == 3 && (opcode == PM4_INDIRECT_BUFFER || opcode == PM4_INDIRECT_BUFFER_PFD))) {
        NfRunRingPacketForDraw(reader);
      }
    }
    if (!ExecutePacket(&reader)) {
      // This probably should be fatal - but we're going to continue anyways.
      REXGPU_ERROR("**** PRIMARY RINGBUFFER: Failed to execute packet.");
      assert_always();
      break;
    }
  } while (reader.read_count());

  OnPrimaryBufferEnd();

  return write_index;
}

void CommandProcessor::ExecuteIndirectBuffer(uint32_t ptr, uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

  // Execute commands!
  memory::RingBuffer reader(memory_->TranslatePhysical(ptr), count * sizeof(uint32_t));
  reader.set_write_offset(count * sizeof(uint32_t));
  do {
    if (!ExecutePacket(&reader)) {
      // Return up a level if we encounter a bad packet.
      REXGPU_ERROR("**** INDIRECT RINGBUFFER: Failed to execute packet.");
      assert_always();
      break;
    }
  } while (reader.read_count());
}

void CommandProcessor::ExecutePacket(uint32_t ptr, uint32_t count) {
  // Execute commands!
  memory::RingBuffer reader(memory_->TranslatePhysical(ptr), count * sizeof(uint32_t));
  reader.set_write_offset(count * sizeof(uint32_t));
  do {
    if (!ExecutePacket(&reader)) {
      REXGPU_ERROR("**** ExecutePacket: Failed to execute packet.");
      assert_always();
      break;
    }
  } while (reader.read_count());
}

// wwe2k14 patch (Session 78, NATIVE FRAME; see nf_role_ in command_processor.h).
void CommandProcessor::NfEnterDraw() {
  nf_role_ = NfRole::kDraw;
  register_file_ = nf_register_file_.get();
  std::swap(bin_select_, nf_bin_select_);
  std::swap(bin_mask_, nf_bin_mask_);
}

void CommandProcessor::NfLeaveDraw() {
  std::swap(bin_select_, nf_bin_select_);
  std::swap(bin_mask_, nf_bin_mask_);
  register_file_ = nf_real_register_file_;
  nf_role_ = NfRole::kAnswers;
}

void CommandProcessor::NfCatchUp(uint32_t target_word) {
  if (nf_role_ != NfRole::kAnswers || !nf_seg_ptr_) {
    return;
  }
  target_word = std::min(target_word, nf_seg_words_);
  if (nf_seg_done_ >= target_word) {
    return;
  }
  if (!REXCVAR_GET(gpu_native_frame_draw)) {
    nf_census_.skipped_draw_side += target_word == nf_seg_words_;
    nf_seg_done_ = target_word;
    return;
  }
  ++nf_census_.catchups;
  ++nf_batch_seq_;
  nf_census_.words += target_word - nf_seg_done_;
  // The packet attribution state belongs to whichever side executes; the drawing side runs inside the segment (depth
  // 1, as the segment's own packets do when one side runs everything).
  const uint32_t packet_at_before = half2_packet_at_, ib_caller_before = half2_ib_caller_;
  const int64_t bias_before = half2_exec_bias_;
  const uint32_t depth_before = native_ib_depth_;
  half2_exec_bias_ = 0;
  native_ib_depth_ = 1;
  NfEnterDraw();
  memory::RingBuffer reader(memory_->TranslatePhysical(nf_seg_ptr_), nf_seg_words_ * sizeof(uint32_t));
  reader.set_read_offset(nf_seg_done_ * sizeof(uint32_t));
  reader.set_write_offset(target_word * sizeof(uint32_t));
  do {
    if (!ExecutePacket(&reader)) {
      ++nf_census_.overruns;
      break;
    }
  } while (reader.read_count());
  NfLeaveDraw();
  native_ib_depth_ = depth_before;
  half2_exec_bias_ = bias_before;
  half2_ib_caller_ = ib_caller_before;
  half2_packet_at_ = packet_at_before;
  nf_seg_done_ = target_word;
}

void CommandProcessor::NfRunRingPacketForDraw(const memory::RingBuffer& reader) {
  if (!REXCVAR_GET(gpu_native_frame_draw)) {
    return;
  }
  ++nf_census_.ring_packets;
  memory::RingBuffer copy = reader;
  const uint32_t packet_at_before = half2_packet_at_;
  NfEnterDraw();
  if (!ExecutePacket(&copy)) {
    ++nf_census_.overruns;
  }
  NfLeaveDraw();
  half2_packet_at_ = packet_at_before;
}

namespace {
std::atomic<uint64_t> g_nf_high_reads[2];  // [0] 0x2000-0x3FFF (render state), [1] >= 0x4000 (constants)
std::atomic<uint32_t> g_nf_high_logged{0};
}  // namespace

void CommandProcessor::NfNoteHighRead(uint32_t index, const char* who) {
  if (index < 0x2000) {
    return;
  }
  const int kind = index >= 0x4000;
  const uint64_t n = g_nf_high_reads[kind].fetch_add(1, std::memory_order_relaxed) + 1;
  if ((n & (n - 1)) == 0 && g_nf_high_logged.fetch_add(1, std::memory_order_relaxed) < 24) {
    REXLOG_INFO("[native][frame] census: {} read register {:04X} ({} read {} {})", who, index, kind ? "constants" :
                "render state", n, n == 1 ? "so far" : "in all");
  }
}

void CommandProcessor::NfFrameLog() {
  static uint32_t frames = 0;
  const int32_t every = REXCVAR_GET(gpu_native_frame_log_frames);
  if (every <= 0 || ++frames < uint32_t(every)) {
    return;
  }
  const NfCensus& c = nf_census_;
  REXLOG_INFO("[native][frame] {} frames: segments {}, catch-ups {} ({:.1f} a frame), words {:.0f} a frame, ring "
              "packets on the drawing side {}, nested buffers {} (waits inside them {}), overruns {}, segments not "
              "drawn {}; star glare draws {} (combine found ahead {}, dead {}); combine c32 carried = register {}, "
              "differs {}; dead glare copies skipped {} (kept: with a clear {})",
              frames, c.segments, c.catchups, double(c.catchups) / frames, double(c.words) / frames, c.ring_packets,
              c.nested, c.nested_waits, c.overruns, c.skipped_draw_side, c.glare_seen, c.glare_ahead, c.glare_dead,
              c.glare_check_ok, c.glare_check_bad, c.glare_copies_skipped, c.glare_copies_kept_clear);
  nf_census_ = {};
  frames = 0;
}

bool CommandProcessor::ExecutePacket(memory::RingBuffer* reader) {
  rex::graphics::free_counters::Bump(rex::graphics::free_counters::g_c.packets);  // (Session 92 M0: log-only counter)
  // wwe2k14 patch (Session 78, NATIVE FRAME): where the answers side is in the current segment (the drawing side
  // catches up to here before a guest-visible point); the measurement hooks belong to the drawing side.
  if (nf_role_ == NfRole::kAnswers) {
    if (native_ib_depth_ == 1) {
      nf_answer_pos_ = uint32_t(reader->read_offset() / sizeof(uint32_t));
    }
  } else if (anatomy_frame_start_pending_) {
    // wwe2k14 patch (Session 6, COUNTERS): a frame-anatomy capture times each frame from its first command.
    AnatomyStartFrame();
  }
  // wwe2k14 patch (Session 48, HALF2): the packet's header word in guest physical memory, before anything is read
  // (reading the last word of a buffer wraps the reader's offset to its start).
  if (half2_ || scene_crowd_seen_.load(std::memory_order_relaxed) ||
      d3d_tex::g_active.load(std::memory_order_relaxed)) {  // (Session 50: the crowd pass needs it too; Session 87: the texture table's census)
    half2_packet_at_ =
        uint32_t(int64_t(reader->read_ptr() - uintptr_t(memory_->TranslatePhysical(0))) + half2_exec_bias_);
    d3d_tex::g_packet_at = half2_packet_at_;
    d3d_tex::g_ib_caller = half2_ib_caller_;
  }
  const uint32_t packet = reader->ReadAndSwap<uint32_t>();
  const uint32_t packet_type = packet >> 30;
  if (packet == 0) {
    return true;
  }

  if (packet == 0xCDCDCDCD) {
    REXGPU_WARN("GPU packet is CDCDCDCD - probably read uninitialized memory!");
  }

  // wwe2k14 patch (Session 87, D3D layer stage 3, gpu_d3d_own_input): the record's register image may be ahead of the
  // register file. Anything that is not the token, a nop or an indirect buffer (which runs its packets through here)
  // sees the register file brought up to date first, and the image is stale after it (rex/graphics/d3d_feed.h).
  // Under the native frame the image belongs to the drawing side (its register file is the one it lags); the answers
  // side never touches it, and the answer packets the drawing side skips (below) leave the register file alone.
  if (D3dFeedState() && nf_role_ != NfRole::kAnswers) {
    const uint32_t feed_opcode = (packet >> 8) & 0x7F;
    bool feed_inert =
        packet_type == 2 || (packet_type == 3 && (feed_opcode == kPm4D3dToken || feed_opcode == PM4_NOP ||
                                                  feed_opcode == PM4_INDIRECT_BUFFER ||
                                                  feed_opcode == PM4_INDIRECT_BUFFER_PFD ||
                                                  // (gpu_d3d_own_copy census: these touch no register - the state-selection
                                                  // masks and INVALIDATE_STATE are not in the register file)
                                                  feed_opcode == PM4_INVALIDATE_STATE ||
                                                  (feed_opcode >= PM4_SET_BIN_MASK_LO && feed_opcode <= PM4_SET_BIN_SELECT_HI)));
    if (!feed_inert && D3dFeedScopeWanted(*this)) {
      // gpu_d3d_own_copy: a type-0 packet wholly below the image's ranges (scratch, coherency, ring registers) leaves the image
      // alone; one in the ranges is stored into the image by the type-0 body; the copy and inline tokens run in an image scope.
      if (packet_type == 0) {
        const uint32_t type0_base = packet & 0x7FFF, type0_count = ((packet >> 16) & 0x3FFF) + 1;
        if (type0_base + type0_count <= 0x2000 || !((packet >> 15) & 1)) feed_inert = true;
      } else if (packet_type == 3 && (feed_opcode == PM4_REG_RMW || feed_opcode == PM4_D3D_COPY || feed_opcode == kPm4D3dInline ||
                                      feed_opcode == PM4_XE_SWAP)) {
        feed_inert = true;
      }
    }
    if (!feed_inert && nf_role_ == NfRole::kDraw && packet_type == 3) {
      switch (feed_opcode) {  // (the list the drawing side skips: answers packets)
        case PM4_WAIT_REG_MEM:
        case PM4_ME_INIT:
        case PM4_INTERRUPT:
        case PM4_MEM_WRITE:
        case PM4_COND_WRITE:
        case PM4_REG_TO_MEM:
        case PM4_EVENT_WRITE_SHD:
        case PM4_EVENT_WRITE_EXT:
        case PM4_EVENT_WRITE_ZPD:
          feed_inert = true;
          break;
        default:
          break;
      }
    }
    if (!feed_inert) {
      D3dFeedLastPacket() = (packet_type << 8) | feed_opcode;
      D3dFeedForeign(*this);
    }
  }

  switch (packet_type) {
    case 0x00:
      return ExecutePacketType0(reader, packet);
    case 0x01:
      return ExecutePacketType1(reader, packet);
    case 0x02:
      return ExecutePacketType2(reader, packet);
    case 0x03:
      return ExecutePacketType3(reader, packet);
    default:
      assert_unhandled_case(packet_type);
      return false;
  }
}

bool CommandProcessor::ExecutePacketType0(memory::RingBuffer* reader, uint32_t packet) {
  // Type-0 packet.
  // Write count registers in sequence to the registers starting at
  // (base_index << 2).

  uint32_t count = ((packet >> 16) & 0x3FFF) + 1;
  if (reader->read_count() < count * sizeof(uint32_t)) {
    REXGPU_ERROR("ExecutePacketType0 overflow (read count {:08X}, packet count {:08X})",
                 reader->read_count(), count * sizeof(uint32_t));
    return false;
  }

  uint32_t base_index = (packet & 0x7FFF);
  uint32_t write_one_reg = (packet >> 15) & 0x1;
  return ExecutePacketType0Body(reader, base_index, count, write_one_reg);
}

// wwe2k14 patch (Session 87, D3D LAYER F1): the rest of ExecutePacketType0, unchanged, after the header is parsed.
bool CommandProcessor::ExecutePacketType0Body(memory::RingBuffer* reader, uint32_t base_index, uint32_t count,
                                              uint32_t write_one_reg) {
  // wwe2k14 patch (Session 79, NATIVE FRAME NF-C, gpu_native_frame_answers_lean): the answers side stores only the
  // registers below 0x2000 (control: scratch, coherency, the ring, events). Render state and constants are the drawing
  // side's, and nothing reads them back from the real file (census NfNoteHighRead: no guest MMIO or answer-packet read
  // at or above 0x2000 in the stage A and entrance replays).
  if (nf_role_ == NfRole::kAnswers && nf_answers_lean_ && base_index >= 0x2000) {
    reader->AdvanceRead(count * sizeof(uint32_t));
    return true;
  }
  // wwe2k14 patch (Session 87, D3D layer stage 3, gpu_d3d_own_copy): a ring type-0 range in the image's registers is stored
  // into the record's image (the foreign check left it alone); a range wholly below 0x2000 is the register file's (scratch,
  // coherency: not modelled by the image).
  if (!write_one_reg && base_index + count > 0x2000 && D3dFeedScopeWanted(*this) &&
      D3dFeedRun(*this, reader, base_index, count)) {
    return true;
  }
  // wwe2k14 patch (Session 48, HALF2): who writes the registers the library draws' shadow lacks (Session 79: the
  // drawing side's bookkeeping).
  if (half2_ && nf_role_ != NfRole::kAnswers) {
    Half2NoteWrite(base_index, write_one_reg ? 1 : count, 0, half2_packet_at_, 0);
  }
  // wwe2k14 patch (Session 6, COUNTERS): type-0 registers by block (measurement only).
  if (frame_counters_enabled_) {
    CountType0Packet(base_index, count, write_one_reg != 0);
  }
  // wwe2k14 patch (Session 6, CP1): consecutive registers go through the register-range path, as
  // SET_CONSTANT does. The backends' WriteRegistersFromMem copies a range that lies entirely within the float,
  // fetch or bool/loop constant block in one go and clears the same constant buffer, texture binding and
  // vertex buffer "in sync" bits that the per-register writes would (for the whole range at once). Any other
  // range (special registers, ranges spanning blocks or going past the register file) falls back to
  // per-register virtual WriteRegister. BeginRead/EndRead handle the ring wrap. Both paths leave the same
  // state after a packet, so the setting is read per packet and can be toggled between packets.
  if (!write_one_reg && REXCVAR_GET(gpu_type0_bulk_writes)) {
    // wwe2k14 patch (Session 6, CP1B): with gpu_type0_block_split, the ranges that aren't within one shader
    // constant block are split by block in CommandProcessor::WriteRegistersFromMem. Path counts for the
    // "[s6][CP1B]" summary.
    const bool block_split = Type0BlockSplitActive();
    const uint64_t split_registers_before =
        block_split ? type0_block_split_stats_.split_registers() : 0;
    if (REXCVAR_GET(gpu_type0_bulk_writes_verify)) {
      ExecutePacketType0BulkVerify(reader, base_index, count);
    } else {
      WriteRegisterRangeFromRing(reader, base_index, count);
    }
    if (block_split) {
      Type0BlockSplitStats& stats = type0_block_split_stats_;
      ++stats.type0_range_packets;
      stats.type0_range_registers += count;
      stats.type0_split_registers += stats.split_registers() - split_registers_before;
    }
    return true;
  }
  // wwe2k14 patch (Session 6, CP1B): gpu_type0_block_split (with gpu_type0_bulk_writes, only write_one_reg
  // packets get here). Every write of the packet goes to base_index; when that's a plain register (one whose
  // writes in CommandProcessor::WriteRegister and the backends' overrides only store the value, see
  // kRegisterSplitIntervals), store each value in order as WriteRegister would, without the two virtual calls
  // per write. Anything else keeps the per-register loop below.
  if (write_one_reg && Type0BlockSplitActive()) {
    if (IsPlainRegisterWrite(base_index)) {
      // Volatile as in WriteRegister.
      volatile uint32_t* reg = &register_file_->values[base_index];
      for (uint32_t m = 0; m < count; ++m) {
        uint32_t value = reader->ReadAndSwap<uint32_t>();
        // wwe2k14 patch (Session 23, REPEAT): counts only if the value changes.
        if (*reg != value) {
          RepeatNoteRegisters(base_index, 1);
        }
        *reg = value;
      }
      type0_block_split_stats_.one_reg_inline_registers += count;
      return true;
    }
    type0_block_split_stats_.one_reg_virtual_registers += count;
  }
  for (uint32_t m = 0; m < count; m++) {
    uint32_t reg_data = reader->ReadAndSwap<uint32_t>();
    uint32_t target_index = write_one_reg ? base_index : base_index + m;
    WriteRegister(target_index, reg_data);
  }

  return true;
}

void CommandProcessor::ExecutePacketType0BulkVerify(memory::RingBuffer* reader, uint32_t base_index,
                                                    uint32_t count) {
  // wwe2k14 patch (Session 6, CP1): gpu_type0_bulk_writes_verify. Does the bulk write, then replays the same
  // packet through per-register WriteRegister from a copy of the reader and checks that the replay finds every
  // register already holding its value, changes none of the backend's invalidation bits (so the bulk write
  // cleared at least everything the per-register writes clear) and ends at the same ring read offset. The
  // replay is only done for ranges within a single shader constant block (the only ranges the backends write
  // in bulk): their per-register side effects are idempotent, so replaying them leaves the state as the
  // per-register path alone would, even after a mismatch. Other ranges go through per-register writes in both
  // paths and may have non-idempotent side effects (gamma ramp index increments), so they aren't replayed.
  // Not detected: the bulk write clearing more bits than needed (only a cost; by code reading, the range
  // checks clear exactly the union of the per-register ones).
  Type0BulkVerifyStats& stats = type0_bulk_verify_stats_;
  if (!stats.summary_frame_valid) {
    stats.summary_frame_valid = true;
    stats.last_summary_frame = counter_;
  } else if (counter_ - stats.last_summary_frame >= 300) {
    REXGPU_INFO(
        "[s6][CP1] verify frames {}-{}: {} type-0 packets ({} registers) checked, {} not checked (not "
        "within one shader constant block); mismatches: {} values, {} invalidation, {} ring offset",
        stats.last_summary_frame, counter_, stats.packets_checked, stats.registers_checked,
        stats.packets_not_checked, stats.value_mismatches, stats.invalidation_mismatches,
        stats.read_offset_mismatches);
    stats.packets_checked = 0;
    stats.registers_checked = 0;
    stats.packets_not_checked = 0;
    stats.value_mismatches = 0;
    stats.invalidation_mismatches = 0;
    stats.read_offset_mismatches = 0;
    stats.last_summary_frame = counter_;
  }

  // count >= 1 and base_index <= 0x7FFF, so this doesn't overflow.
  uint32_t end_index = base_index + count - 1;
  bool single_constant_block = (base_index >= XE_GPU_REG_SHADER_CONSTANT_000_X &&
                                end_index <= XE_GPU_REG_SHADER_CONSTANT_511_W) ||
                               (base_index >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
                                end_index <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) ||
                               (base_index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
                                end_index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31);
  if (!single_constant_block) {
    ++stats.packets_not_checked;
    WriteRegisterRangeFromRing(reader, base_index, count);
    return;
  }

  memory::RingBuffer replay_reader = *reader;
  WriteRegisterRangeFromRing(reader, base_index, count);

  RegisterWriteInvalidationState state_bulk;
  bool have_state = GetRegisterWriteInvalidationState(state_bulk);
  uint32_t value_mismatch_count = 0;
  uint32_t first_mismatch_index = 0, first_mismatch_bulk = 0, first_mismatch_expected = 0;
  for (uint32_t m = 0; m < count; ++m) {
    uint32_t index = base_index + m;
    uint32_t expected = replay_reader.ReadAndSwap<uint32_t>();
    uint32_t bulk_value = register_file_->values[index];
    if (bulk_value != expected) {
      if (!value_mismatch_count) {
        first_mismatch_index = index;
        first_mismatch_bulk = bulk_value;
        first_mismatch_expected = expected;
      }
      ++value_mismatch_count;
    }
    WriteRegister(index, expected);
  }
  RegisterWriteInvalidationState state_replayed;
  bool invalidation_mismatch =
      have_state && GetRegisterWriteInvalidationState(state_replayed) && state_replayed != state_bulk;
  uint32_t bulk_read_offset = reader->read_offset();
  bool read_offset_mismatch = replay_reader.read_offset() != bulk_read_offset;
  if (read_offset_mismatch) {
    // Continue from where the per-register path would.
    reader->set_read_offset(replay_reader.read_offset());
  }

  ++stats.packets_checked;
  stats.registers_checked += count;
  stats.value_mismatches += value_mismatch_count;
  stats.invalidation_mismatches += invalidation_mismatch ? 1 : 0;
  stats.read_offset_mismatches += read_offset_mismatch ? 1 : 0;
  if ((value_mismatch_count || invalidation_mismatch || read_offset_mismatch) &&
      stats.mismatches_logged < 8) {
    ++stats.mismatches_logged;
    REXGPU_WARN(
        "[s6][CP1] verify mismatch, type-0 write of {} registers at {:04X} (frame {}): {} value "
        "mismatches (first {:04X}: bulk {:08X}, ring {:08X}); invalidation bits after bulk "
        "{:08X} {:016X} {:016X} {:08X}, after per-register replay {:08X} {:016X} {:016X} {:08X}; ring "
        "read offset bulk {:X}, per-register {:X}",
        count, base_index, counter_, value_mismatch_count, first_mismatch_index, first_mismatch_bulk,
        first_mismatch_expected, state_bulk.bits[0], state_bulk.bits[1], state_bulk.bits[2],
        state_bulk.bits[3], state_replayed.bits[0], state_replayed.bits[1], state_replayed.bits[2],
        state_replayed.bits[3], bulk_read_offset, replay_reader.read_offset());
  }
}

bool CommandProcessor::ExecutePacketType1(memory::RingBuffer* reader, uint32_t packet) {
  // Type-1 packet.
  // Contains two registers of data. Type-0 should be more common.
  uint32_t reg_index_1 = packet & 0x7FF;
  uint32_t reg_index_2 = (packet >> 11) & 0x7FF;
  uint32_t reg_data_1 = reader->ReadAndSwap<uint32_t>();
  uint32_t reg_data_2 = reader->ReadAndSwap<uint32_t>();
  WriteRegister(reg_index_1, reg_data_1);
  WriteRegister(reg_index_2, reg_data_2);
  return true;
}

bool CommandProcessor::ExecutePacketType2(memory::RingBuffer* reader, uint32_t packet) {
  // Type-2 packet.
  // No-op. Do nothing.
  return true;
}

bool CommandProcessor::ExecutePacketType3(memory::RingBuffer* reader, uint32_t packet) {
  // Type-3 packet.
  uint32_t opcode = (packet >> 8) & 0x7F;
  uint32_t count = ((packet >> 16) & 0x3FFF) + 1;
  auto data_start_offset = reader->read_offset();
  ++frame_counters_.type3_packets;  // wwe2k14 patch (Session 6, COUNTERS)

  if (reader->read_count() < count * sizeof(uint32_t)) {
    REXGPU_ERROR("ExecutePacketType3 overflow (read count {:08X}, packet count {:08X})",
                 reader->read_count(), count * sizeof(uint32_t));
    return false;
  }

  // & 1 == predicate - when set, we do bin check to see if we should execute
  // the packet. Only type 3 packets are affected.
  // We also skip predicated swaps, as they are never valid (probably?).
  if (packet & 1) {
    bool any_pass = (bin_select_ & bin_mask_) != 0;
    // wwe2k14 patch (Session 7, TILING-STAGE0): predication counters (cpf_tile, cpf_bin) and the tiling log.
    // Measurement only.
    if (!any_pass) {
      ++tiling_counters_.predicated_skipped;
    } else if (opcode != PM4_XE_SWAP) {
      if (opcode == PM4_DRAW_INDX || opcode == PM4_DRAW_INDX_2) {
        ++tiling_counters_.predicated_draws;
      }
      if (!tiling_select_counted_) {
        CountTilingSelect();
      }
    }
    if (tiling_log_active_) {
      TilingLogPacket(reader, uint32_t(data_start_offset), opcode, true, any_pass && opcode != PM4_XE_SWAP);
    }
    // wwe2k14 patch (Session 48, HALF2): a draw packet the bin test skips still consumes its library draw.
    // (Session 78: on the drawing side only - the answers side never matches library draws.)
    if (half2_ && nf_role_ != NfRole::kAnswers && !any_pass && (opcode == PM4_DRAW_INDX || opcode == PM4_DRAW_INDX_2)) {
      Half2OnCpDraw(half2_packet_at_, true, {});
    }
    if (!any_pass || opcode == PM4_XE_SWAP) {
      reader->AdvanceRead(count * sizeof(uint32_t));
      return true;
    }
  } else if (tiling_log_active_) {
    // wwe2k14 patch (Session 7, TILING-STAGE0): the tiling log.
    TilingLogPacket(reader, uint32_t(data_start_offset), opcode, false, true);
  }

  // wwe2k14 patch (Session 78, NATIVE FRAME): what each side runs (docs\native\NATIVE_FRAME_DESIGN.md section 2).
  if (nf_role_ == NfRole::kAnswers) {
    if (nf_answers_lean_) {
      switch (opcode) {
        // (Session 79, NF-C) constants and render state written by packet: the drawing side's (see ExecutePacketType0).
        case PM4_SET_CONSTANT:
        case PM4_SET_CONSTANT2:
        case PM4_LOAD_ALU_CONSTANT:
        case PM4_SET_SHADER_CONSTANTS:
          reader->AdvanceRead(count * sizeof(uint32_t));
          return true;
        default:
          break;
      }
    }
    switch (opcode) {
      case PM4_DRAW_INDX:
      case PM4_DRAW_INDX_2:
      case PM4_IM_LOAD:
      case PM4_IM_LOAD_IMMEDIATE:
      case PM4_INVALIDATE_STATE:
      case PM4_VIZ_QUERY:
      case PM4_D3D_COPY:  // wwe2k14 patch (Session 87, D3D layer F2): the drawing side's
      case kPm4D3dInline:  // wwe2k14 patch (Session 87, D3D LAYER F3): a draw, the drawing side's
        reader->AdvanceRead(count * sizeof(uint32_t));
        return true;
      case PM4_XE_SWAP:
        // The guest-visible part of the swap: the point itself and the swap counter (EVENT_WRITE_SHD's bit 31
        // value); the presentation is the drawing side's (it caught up to here first).
        ++half2_swaps_;
        OnGuestVisibleWrite(GuestVisibleEvent::kSwap);
        reader->AdvanceRead(count * sizeof(uint32_t));
        ++counter_;
        NfFrameLog();
        return true;
      default:
        break;
    }
  } else if (nf_role_ == NfRole::kDraw) {
    switch (opcode) {
      case PM4_WAIT_REG_MEM:
        // Passed by the answers side before the drawing side got here, except inside a buffer the drawing side runs
        // whole before the answers side enters it (counted: a CPU-gated wait there would be passed early).
        if (native_ib_depth_ >= 2) ++nf_census_.nested_waits;
        [[fallthrough]];
      case PM4_ME_INIT:
      case PM4_INTERRUPT:
      case PM4_MEM_WRITE:
      case PM4_COND_WRITE:
      case PM4_REG_TO_MEM:
      case PM4_EVENT_WRITE_SHD:
      case PM4_EVENT_WRITE_EXT:
      case PM4_EVENT_WRITE_ZPD:
        RepeatNotePacket(opcode);  // (as when it ran: it ends a repeat)
        reader->AdvanceRead(count * sizeof(uint32_t));
        return true;
      default:
        break;
    }
  }

  // wwe2k14 patch (Session 23, REPEAT): any other packet between two draws ends a repeat (register and constant
  // writes are noted by the register write paths themselves; the draw packets by the backend).
  switch (opcode) {
    case PM4_DRAW_INDX:
    case PM4_DRAW_INDX_2:
    case kPm4D3dInline:  // (Session 87, F3: its register writes and draw note themselves)
    case PM4_SET_CONSTANT:
    case PM4_SET_CONSTANT2:
    case PM4_LOAD_ALU_CONSTANT:
    case PM4_SET_SHADER_CONSTANTS:
    case PM4_NOP:
    case PM4_INDIRECT_BUFFER:
    case PM4_INDIRECT_BUFFER_PFD:
    case kPm4D3dToken:  // (Session 87: the token's own runs and sub-packets note themselves)
      break;
    default:
      if (nf_role_ != NfRole::kAnswers) RepeatNotePacket(opcode);  // (Session 78: the drawing side's state)
      break;
  }

  bool result = false;
  switch (opcode) {
    case PM4_ME_INIT:
      result = ExecutePacketType3_ME_INIT(reader, packet, count);
      break;
    case PM4_NOP:
      result = ExecutePacketType3_NOP(reader, packet, count);
      break;
    case PM4_INTERRUPT:
      result = ExecutePacketType3_INTERRUPT(reader, packet, count);
      break;
    case PM4_XE_SWAP: {
      // (Session 87 stage 3, gpu_d3d_own_copy) The present reads one thing from the register file: the swap texture's fetch
      // constant 0 (0x4800-0x4805, which VdSwap's words wrote), in VulkanTextureCache::RequestSwapTexture through the texture cache's
      // register source. In an image scope that is the record's image - the type-0 packets of the swap items stored it there - and
      // the register file is not brought up to date for the present.
      const bool feed_scope = D3dFeedScopeBegin(*this);
      result = ExecutePacketType3_XE_SWAP(reader, packet, count);
      if (feed_scope) D3dFeedScopeEnd(*this);
    } break;
    case PM4_INDIRECT_BUFFER:
    case PM4_INDIRECT_BUFFER_PFD:
      result = ExecutePacketType3_INDIRECT_BUFFER(reader, packet, count);
      break;
    case PM4_WAIT_REG_MEM:
      result = ExecutePacketType3_WAIT_REG_MEM(reader, packet, count);
      break;
    case PM4_REG_RMW:
      result = ExecutePacketType3_REG_RMW(reader, packet, count);
      break;
    case PM4_REG_TO_MEM:
      result = ExecutePacketType3_REG_TO_MEM(reader, packet, count);
      break;
    case PM4_MEM_WRITE:
      result = ExecutePacketType3_MEM_WRITE(reader, packet, count);
      break;
    case PM4_COND_WRITE:
      result = ExecutePacketType3_COND_WRITE(reader, packet, count);
      break;
    case PM4_EVENT_WRITE:
      result = ExecutePacketType3_EVENT_WRITE(reader, packet, count);
      break;
    case PM4_EVENT_WRITE_SHD:
      result = ExecutePacketType3_EVENT_WRITE_SHD(reader, packet, count);
      break;
    case PM4_EVENT_WRITE_EXT:
      result = ExecutePacketType3_EVENT_WRITE_EXT(reader, packet, count);
      break;
    case PM4_EVENT_WRITE_ZPD:
      result = ExecutePacketType3_EVENT_WRITE_ZPD(reader, packet, count);
      break;
    case PM4_D3D_COPY: {  // wwe2k14 patch (Session 87, D3D layer F2): rex/graphics/d3d_copy.h
      // (stage 3, gpu_d3d_own_copy: the clear / resolve executor runs in an image scope - its register writes go into the
      // record's image and its draw / copy derivations read it)
      const bool feed_scope = D3dFeedScopeBegin(*this);
      result = D3dCopyPacket(*this, reader, count);
      if (feed_scope) D3dFeedScopeEnd(*this);
    } break;
    case PM4_DRAW_INDX:
      result = ExecutePacketType3_DRAW_INDX(reader, packet, count);
      break;
    case PM4_DRAW_INDX_2:
      result = ExecutePacketType3_DRAW_INDX_2(reader, packet, count);
      break;
    case PM4_SET_CONSTANT:
      result = ExecutePacketType3_SET_CONSTANT(reader, packet, count);
      break;
    case PM4_SET_CONSTANT2:
      result = ExecutePacketType3_SET_CONSTANT2(reader, packet, count);
      break;
    case PM4_LOAD_ALU_CONSTANT:
      result = ExecutePacketType3_LOAD_ALU_CONSTANT(reader, packet, count);
      break;
    case PM4_SET_SHADER_CONSTANTS:
      result = ExecutePacketType3_SET_SHADER_CONSTANTS(reader, packet, count);
      break;
    case PM4_IM_LOAD:
      result = ExecutePacketType3_IM_LOAD(reader, packet, count);
      break;
    case PM4_IM_LOAD_IMMEDIATE:
      result = ExecutePacketType3_IM_LOAD_IMMEDIATE(reader, packet, count);
      break;
    case PM4_INVALIDATE_STATE:
      result = ExecutePacketType3_INVALIDATE_STATE(reader, packet, count);
      break;
    case PM4_VIZ_QUERY:
      result = ExecutePacketType3_VIZ_QUERY(reader, packet, count);
      break;
    case kPm4D3dToken:  // wwe2k14 patch (Session 87, D3D LAYER F1)
      result = ExecutePacketType3_D3D_TOKEN(reader, packet, count);
      break;
    case kPm4D3dInline: {  // wwe2k14 patch (Session 87, D3D LAYER F3): src\graphics\d3d_inline_draw.cpp
      const bool feed_scope = D3dFeedScopeBegin(*this);  // (stage 3, gpu_d3d_own_copy: see PM4_D3D_COPY above)
      result = ExecutePacketType3_D3D_INLINE(reader, packet, count);
      if (feed_scope) D3dFeedScopeEnd(*this);
    } break;

    // wwe2k14 patch (Session 7, TILING-STAGE0): Bin{Select,Mask}Written count the packets (cpf_bin) and log
    // the new values (gpu_tiling_log_frames).
    case PM4_SET_BIN_MASK_LO: {
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      bin_mask_ = (bin_mask_ & 0xFFFFFFFF00000000ull) | value;
      BinMaskWritten("mask_lo");
      result = true;
    } break;
    case PM4_SET_BIN_MASK_HI: {
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      bin_mask_ = (bin_mask_ & 0xFFFFFFFFull) | (static_cast<uint64_t>(value) << 32);
      BinMaskWritten("mask_hi");
      result = true;
    } break;
    case PM4_SET_BIN_SELECT_LO: {
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      const uint64_t old_select = bin_select_;
      bin_select_ = (bin_select_ & 0xFFFFFFFF00000000ull) | value;
      BinSelectWritten("sel_lo", old_select);
      result = true;
    } break;
    case PM4_SET_BIN_SELECT_HI: {
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      const uint64_t old_select = bin_select_;
      bin_select_ = (bin_select_ & 0xFFFFFFFFull) | (static_cast<uint64_t>(value) << 32);
      BinSelectWritten("sel_hi", old_select);
      result = true;
    } break;
    case PM4_SET_BIN_MASK: {
      assert_true(count == 2);
      uint64_t val_hi = reader->ReadAndSwap<uint32_t>();
      uint64_t val_lo = reader->ReadAndSwap<uint32_t>();
      bin_mask_ = (val_hi << 32) | val_lo;
      BinMaskWritten("mask64");
      result = true;
    } break;
    case PM4_SET_BIN_SELECT: {
      assert_true(count == 2);
      uint64_t val_hi = reader->ReadAndSwap<uint32_t>();
      uint64_t val_lo = reader->ReadAndSwap<uint32_t>();
      const uint64_t old_select = bin_select_;
      bin_select_ = (val_hi << 32) | val_lo;
      BinSelectWritten("sel64", old_select);
      result = true;
    } break;
    case PM4_CONTEXT_UPDATE: {
      assert_true(count == 1);
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      REXGPU_INFO("GPU context update = {:08X}", value);
      assert_true(value == 0);
      result = true;
      break;
    }
    case PM4_WAIT_FOR_IDLE: {
      // This opcode is used by 5454084E while going / being ingame.
      assert_true(count == 1);
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      REXGPU_INFO("GPU wait for idle = {:08X}", value);
      result = true;
      break;
    }

    default:
      REXGPU_INFO("Unimplemented GPU OPCODE: 0x{:02X}\t\tCOUNT: {}\n", opcode, count);
      assert_always();
      reader->AdvanceRead(count * sizeof(uint32_t));
      break;
  }

  assert_true(reader->read_offset() ==
              (data_start_offset + (count * sizeof(uint32_t))) % reader->capacity());
  return result;
}

bool CommandProcessor::ExecutePacketType3_ME_INIT(memory::RingBuffer* reader, uint32_t packet,
                                                  uint32_t count) {
  // initialize CP's micro-engine
  me_bin_.clear();
  for (uint32_t i = 0; i < count; i++) {
    me_bin_.push_back(reader->ReadAndSwap<uint32_t>());
  }

  return true;
}

bool CommandProcessor::ExecutePacketType3_NOP(memory::RingBuffer* reader, uint32_t packet,
                                              uint32_t count) {
  // skip N 32-bit words to get to the next packet
  // No-op, ignore some data.
  reader->AdvanceRead(count * sizeof(uint32_t));
  return true;
}

bool CommandProcessor::ExecutePacketType3_INTERRUPT(memory::RingBuffer* reader, uint32_t packet,
                                                    uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

  // generate interrupt from the command stream
  uint32_t cpu_mask = reader->ReadAndSwap<uint32_t>();
  OnGuestVisibleWrite(GuestVisibleEvent::kInterrupt);  // wwe2k14 patch (Session 66, H1.1): the guest callback runs
  // wwe2k14 patch (Session 7, TILING-STAGE0): the tiling log times the interrupt callbacks (with predicated
  // tiling, the tile-0 interrupt is where the game turns the screen extents into tile masks).
  const bool tiling_log = tiling_log_active_;
  if (tiling_log) {
    TilingLogEvent("interrupt", cpu_mask, -1, true);
  }
  for (int n = 0; n < 6; n++) {
    if (cpu_mask & (1 << n)) {
      if (graphics_system_) {
        // wwe2k14 patch (Session 15): det_serial: the guest interrupt routine runs between guest turns.
        if (det::SerialEnabled()) {
          if (det::TraceDetail()) {  // Session 77: which packet asked (ring position, CPU)
            char tag[48];
            std::snprintf(tag, sizeof(tag), "i%X.%d", uint32_t(reader->read_offset()), n);
            det::TraceTag(tag);
          }
          // Session 77: blocked on our own request (DetDrained: drained) until the scheduler has run it.
          SetDetWaitKind(4);
          det::SerialRunExternal([&] { graphics_system_->DispatchInterruptCallback(1, n); }, true);
          SetDetWaitKind(0);
        } else {
          graphics_system_->DispatchInterruptCallback(1, n);
        }
      }
    }
  }
  if (tiling_log && tiling_log_active_) {
    TilingLogEvent("interrupt_end", cpu_mask, -1, true);
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_XE_SWAP(memory::RingBuffer* reader, uint32_t packet,
                                                  uint32_t count) {
  // wwe2k14 patch (Session 92, M3b): the packet's own part is reading its words; the present is PresentFrame (the same body for
  // the host queue's present item, which carries the words as data).
  rex::graphics::free_counters::Bump(rex::graphics::free_counters::g_c.presents_packet);
  // Xenia-specific VdSwap hook.
  // VdSwap will post this to tell us we need to swap the screen/fire an
  // interrupt.
  // 63 words here, but only the first has any data.
  uint32_t magic = reader->ReadAndSwap<memory::fourcc_t>();
  assert_true(magic == kSwapSignature);

  // TODO(benvanik): only swap frontbuffer ptr.
  uint32_t frontbuffer_ptr = reader->ReadAndSwap<uint32_t>();
  uint32_t frontbuffer_width = reader->ReadAndSwap<uint32_t>();
  uint32_t frontbuffer_height = reader->ReadAndSwap<uint32_t>();
  reader->AdvanceRead((count - 4) * sizeof(uint32_t));
  return PresentFrame(frontbuffer_ptr, frontbuffer_width, frontbuffer_height);
}

// wwe2k14 patch (Session 92, M3b): the swap's work after its words are read (the body of the XE_SWAP packet handler, moved here
// unchanged): called by the handler, and by the host queue's present item (gpu_d3d_owned_present) with the words decoded.
bool CommandProcessor::PresentFrame(uint32_t frontbuffer_ptr, uint32_t frontbuffer_width, uint32_t frontbuffer_height) {
  // wwe2k14 patch (Session 78, NATIVE FRAME): on the drawing side the presentation only (the answers side counted
  // the swap and made it guest-visible: ExecutePacketType3).
  const bool nf_draw = nf_role_ == NfRole::kDraw;
  if (!nf_draw) {
    ++half2_swaps_;  // wwe2k14 patch (Session 51 step 3: the detached census)
    OnGuestVisibleWrite(GuestVisibleEvent::kSwap);  // wwe2k14 patch (Session 66, H1.1)
  }
  SCOPE_profile_cpu_f("gpu");

#ifdef REXGLUE_ENABLE_PERF_COUNTERS
  {
    static uint64_t last_frame_tick = 0;
    uint64_t now = rex::chrono::Clock::QueryHostTickCount();
    if (last_frame_tick) {
      uint64_t freq = rex::chrono::Clock::QueryHostTickFrequency();
      int64_t dt_us = static_cast<int64_t>((now - last_frame_tick) * 1000000 / freq);
      PROFILE_FRAME_TIME_US(dt_us);
      PROFILE_FPS(freq / (now - last_frame_tick));
    }
    last_frame_tick = now;
  }
#endif
  rex::perf::Profiler::Flip();

  // wwe2k14 patch (Session 48, HALF2): the library-draw census, and gpu_native_half2 taking effect.
  Half2FrameBoundary();
  SceneCrowdFrameEnd();  // wwe2k14 patch (Session 50, SCENE CROWD)
  // wwe2k14 patch (Session 6, COUNTERS): the frame's counters (cpf* events) right before cp_swap; the end of a
  // frame-anatomy capture's frame.
  RecordFrameCounters();
  if (anatomy_active_) {
    AnatomyEndFrame();
  }
  // wwe2k14 patch (Session 7, TILING-STAGE0 review): the vsync thread also increments counter_
  // (GraphicsSystem::MarkVblank), so keep the exact value cp_swap records for the tiling log's frame row.
  tiling_swap_counter_ = counter_;
  perf::timeline::Record("cp_swap", frontbuffer_ptr, tiling_swap_counter_);
  IssueSwap(frontbuffer_ptr, frontbuffer_width, frontbuffer_height);

  // wwe2k14 patch (Session 6, CP10): once per frame, see GetReadbackResolveMode.
  RefreshReadbackResolveModeCache();
  // wwe2k14 patch (Session 6, debug): parse gpu_debug_skip_vertex_shader once per frame (empty or invalid = off).
  {
    // Copied under the cvar registry lock (SetFlagByName assigns the string under it from other threads).
    const std::string skip = rex::cvar::Query<std::string>("gpu_debug_skip_vertex_shader");
    debug_skip_vertex_shader_hash_ = skip.empty() ? 0 : std::strtoull(skip.c_str(), nullptr, 16);
    // wwe2k14 patch (Session 16, MATTE): a comma-separated list.
    debug_skip_vertex_shader_count_ = 0;
    for (size_t pos = 0; pos < skip.size() && debug_skip_vertex_shader_count_ < 64;) {
      const size_t end = std::min(skip.find(',', pos), skip.size());
      const uint64_t h = std::strtoull(skip.substr(pos, end - pos).c_str(), nullptr, 16);
      if (h) debug_skip_vertex_shader_hashes_[debug_skip_vertex_shader_count_++] = h;
      pos = end + 1;
    }
    // wwe2k14 patch (Session 72, debug): gpu_debug_skip_pixel_shader, the same for the pixel shader.
    const std::string skip_ps = rex::cvar::Query<std::string>("gpu_debug_skip_pixel_shader");
    debug_skip_pixel_shader_count_ = 0;
    for (size_t pos = 0; pos < skip_ps.size() && debug_skip_pixel_shader_count_ < 64;) {
      const size_t end = std::min(skip_ps.find(',', pos), skip_ps.size());
      const uint64_t h = std::strtoull(skip_ps.substr(pos, end - pos).c_str(), nullptr, 16);
      if (h) debug_skip_pixel_shader_hashes_[debug_skip_pixel_shader_count_++] = h;
      pos = end + 1;
    }
  }
  DrawProbeLatch();  // wwe2k14 patch (Session 27.5, DRAWPROBE)
  // wwe2k14 patch (Session 6, CP1B): gpu_type0_block_split summary.
  Type0BlockSplitFrameEnd();
  // wwe2k14 patch (Session 7, EXTENTS-SHADOW): close the frame's tiling brackets, summary, latch the settings.
  ExtentsFrameBoundary();

  D3dCensusFrame(counter_);  // wwe2k14 patch (Session 87, D3D LAYER F1)
  d3d_tex::LogCensus(counter_);  // wwe2k14 patch (Session 87, D3D layer stage 4, textures)
  D3dInlineCensusFrame(counter_);  // wwe2k14 patch (Session 87, D3D LAYER F3)
  if (!nf_draw) ++counter_;
  nf_dead_targets_.clear();  // wwe2k14 patch (Session 79): the dead glare targets are per frame
  // wwe2k14 patch (Session 80, gpu_shadow_half_rate): the frame count; a skip mark left without its copy is dropped (a
  // census line every 600 frames while the lever acts).
  ++shadow_frame_;
  shadow_open_ = false;  // (a map left without its copy: the next frame starts afresh)
  shadow_maps_done_.clear();
  if (shadow_frame_maps_drawn_ || shadow_frame_maps_skipped_) {
    ++shadow_frames_by_skips_[!shadow_frame_maps_skipped_ ? 0 : !shadow_frame_maps_drawn_ ? 2 : 1];
  }
  shadow_frame_maps_drawn_ = shadow_frame_maps_skipped_ = 0;
  if (shadow_frame_ % 600 == 0 && (shadow_drawn_ || shadow_skipped_)) {
    REXGPU_INFO("[shadow] half rate, frames to {}: wrestler shadow draws drawn {} / skipped {}; copies skipped {}, kept "
                "with a clear {}; maps known {}; frames with no map skipped {}, some {}, all {}",
                shadow_frame_, shadow_drawn_, shadow_skipped_, shadow_copies_skipped_, shadow_copies_kept_clear_,
                shadow_maps_.size(), shadow_frames_by_skips_[0], shadow_frames_by_skips_[1], shadow_frames_by_skips_[2]);
    shadow_drawn_ = shadow_skipped_ = shadow_copies_skipped_ = shadow_copies_kept_clear_ = 0;
    shadow_frames_by_skips_[0] = shadow_frames_by_skips_[1] = shadow_frames_by_skips_[2] = 0;
  }
  // wwe2k14 patch (Session 85, gpu_dynamic_shadows false): the shadow chain's frame end - a frame without shadow casters
  // (state in src\graphics\shadow_chain.h, not in this class: the layout of its data is shared with the game exe).
  ShadowChainFrameEnd(shadow_frame_);
  vulkan::shadow_skip::FrameEnd(shadow_frame_);  // (Session 95, item K)
  SkipFrameDrawsLatch();  // wwe2k14 patch (Session 15): gpu_skip_frame_draws for the next frame
  // wwe2k14 patch (Session 6, COUNTERS): write a captured frame's rows (after the swap, outside the timed draws)
  // and start or end a frame-anatomy capture.
  if (anatomy_active_ || REXCVAR_GET(gpu_anatomy_capture) > 0) {
    AnatomyFrameBoundary();
  }
  // wwe2k14 patch (Session 7, TILING-STAGE0): end a tiling log frame, and start or end a tiling log.
  if (tiling_log_active_ || REXCVAR_GET(gpu_tiling_log_frames) > 0) {
    TilingLogFrameBoundary();
  }
  return true;
}

// wwe2k14 patch (Session 27.5, DRAWPROBE, debug): see draw_probe_* in command_processor.h.
void CommandProcessor::DrawProbeLatch() {
  ++draw_probe_swaps_;
  draw_probe_draw_ = 0;
  draw_probe_loads_ = 0;
  auto parse_hex_list = [](const std::string& text, auto* out, uint32_t capacity, bool* star) {
    uint32_t count = 0;
    if (star) *star = false;
    for (size_t pos = 0; pos < text.size() && count < capacity;) {
      const size_t end = std::min(text.find(',', pos), text.size());
      const std::string item = text.substr(pos, end - pos);
      if (star && item == "*") {
        *star = true;
      } else if (const uint64_t value = std::strtoull(item.c_str(), nullptr, 16)) {
        out[count++] = static_cast<std::remove_reference_t<decltype(*out)>>(value);
      }
      pos = end + 1;
    }
    return count;
  };
  skip_index_base_count_ = parse_hex_list(rex::cvar::Query<std::string>("gpu_debug_skip_index_base"),
                                          skip_index_bases_, 64, nullptr);
  if (draw_probe_frames_left_ > 0) {
    if (--draw_probe_frames_left_ == 0 && draw_probe_file_) {
      std::fflush(draw_probe_file_);
    }
  }
  const int32_t requested = REXCVAR_GET(gpu_debug_draw_probe_frames);
  if (requested <= 0) {
    return;
  }
  REXCVAR_SET(gpu_debug_draw_probe_frames, 0);
  draw_probe_hash_count_ = parse_hex_list(rex::cvar::Query<std::string>("gpu_debug_draw_probe"),
                                          draw_probe_hashes_, 64, &draw_probe_all_);
  draw_probe_const_first_ = draw_probe_const_last_ = -1;
  const std::string consts = rex::cvar::Query<std::string>("gpu_debug_draw_probe_consts");
  if (!consts.empty()) {
    int first = -1, last = -1;
    if (std::sscanf(consts.c_str(), "%d-%d", &first, &last) == 2 && first >= 0 && last >= first && last < 512) {  // (Session 80: 256-511 = the pixel shader bank)
      draw_probe_const_first_ = first;
      draw_probe_const_last_ = last;
    }
  }
  if (!draw_probe_file_) {
    std::string path = REXCVAR_GET(gpu_debug_draw_probe_path);
    if (path.empty()) {
      std::error_code error;
      path = (std::filesystem::current_path(error) / "logs" / "drawprobe.csv").string();
    }
    draw_probe_file_ = std::fopen(path.c_str(), "w");
    if (!draw_probe_file_) {
      REXGPU_WARN("[s27.5][DRAWPROBE] can't open {}", path);
      return;
    }
    std::fputs("swap,draw,vs_hash,prim,index_count,index_base,index_total,vfetch,vfetch_base,vfetch_bytes,"
               "float_count,float_first,float_last,float_dynamic,alu_loads,alu_load_address,alu_load_index,"
               "alu_load_dwords,consts\n",
               draw_probe_file_);
    REXGPU_INFO("[s27.5][DRAWPROBE] writing {}", path);
  }
  draw_probe_frames_left_ = requested;
  REXGPU_INFO("[s27.5][DRAWPROBE] probing {} frame(s) from swap {} ({} shader(s){}, consts {}-{})", requested,
              draw_probe_swaps_, draw_probe_hash_count_, draw_probe_all_ ? ", all draws" : "",
              draw_probe_const_first_, draw_probe_const_last_);
}

bool CommandProcessor::DrawProbe(const Shader& vertex_shader, uint32_t prim_type, uint32_t index_count,
                                 const IndexBufferInfo* index_buffer_info) {
  const uint32_t index_base = index_buffer_info ? index_buffer_info->guest_base : 0;
  bool skip = false;
  for (uint32_t i = 0; i < skip_index_base_count_; ++i) {
    if (index_base && skip_index_bases_[i] == index_base) skip = true;
  }
  if (draw_probe_frames_left_ <= 0 || !draw_probe_file_) {
    return skip;
  }
  const uint32_t draw = draw_probe_draw_++;
  const uint64_t hash = vertex_shader.ucode_data_hash();
  bool listed = draw_probe_all_;
  for (uint32_t i = 0; !listed && i < draw_probe_hash_count_; ++i) {
    listed = draw_probe_hashes_[i] == hash;
  }
  if (listed) {
    const RegisterFile& regs = *register_file_;
    const Shader::ConstantRegisterMap& map = vertex_shader.constant_register_map();
    uint32_t vfetch = UINT32_MAX, vfetch_base = 0, vfetch_bytes = 0;
    for (uint32_t i = 0; i < 96 && vfetch == UINT32_MAX; ++i) {
      if (map.vertex_fetch_bitmap[i >> 5] & (uint32_t(1) << (i & 31))) {
        const xenos::xe_gpu_vertex_fetch_t fetch = regs.GetVertexFetch(i);
        vfetch = i;
        vfetch_base = uint32_t(fetch.address) << 2;
        vfetch_bytes = uint32_t(fetch.size) << 2;
      }
    }
    int32_t float_first = -1, float_last = -1;
    for (uint32_t i = 0; i < 256; ++i) {
      if (map.float_bitmap[i >> 6] & (uint64_t(1) << (i & 63))) {
        if (float_first < 0) float_first = int32_t(i);
        float_last = int32_t(i);
      }
    }
    std::fprintf(draw_probe_file_, "%llu,%u,%016llX,%u,%u,%08X,%u,%d,%08X,%u,%u,%d,%d,%d,%u,%08X,%u,%u,",
                 static_cast<unsigned long long>(draw_probe_swaps_), draw,
                 static_cast<unsigned long long>(hash), prim_type, index_count, index_base,
                 index_buffer_info ? index_buffer_info->count : 0, vfetch == UINT32_MAX ? -1 : int32_t(vfetch),
                 vfetch_base, vfetch_bytes, map.float_count, float_first, float_last,
                 map.float_dynamic_addressing ? 1 : 0, draw_probe_loads_, draw_probe_load_address_,
                 draw_probe_load_index_, draw_probe_load_dwords_);
    for (int32_t c = draw_probe_const_first_; c >= 0 && c <= draw_probe_const_last_; ++c) {
      for (uint32_t k = 0; k < 4; ++k) {
        float value;
        const uint32_t bits = regs.values[XE_GPU_REG_SHADER_CONSTANT_000_X + uint32_t(c) * 4 + k];
        std::memcpy(&value, &bits, sizeof(value));
        std::fprintf(draw_probe_file_, "%s%.7g", (c == draw_probe_const_first_ && k == 0) ? "" : " ", value);
      }
    }
    std::fputc('\n', draw_probe_file_);
  }
  draw_probe_loads_ = 0;
  return skip;
}

// wwe2k14 patch (Session 15, sub-second loads): gpu_skip_frame_draws.
void CommandProcessor::SkipFrameDrawsLatch() {
  const int32_t n = REXCVAR_GET(gpu_skip_frame_draws);
  if (n > 1) {
    if (skip_frame_draws_n_ <= 1) {
      skip_frame_draws_index_ = 0;
      skip_frame_frames_ = skip_frame_skipped_frames_ = skip_frame_draws_skipped_ = 0;
      skip_frame_resolves_skipped_ = 0;
      skip_frame_drawn_resolves_.clear();
      skip_frame_skipped_resolves_.clear();
      REXGPU_INFO("[s15][SKIPDRAW] on: draw 1 of every {} frames", n);
    }
    skip_frame_draws_n_ = n;
    ++skip_frame_draws_index_;
    ++skip_frame_frames_;
    skip_frame_draws_ = (skip_frame_draws_index_ % uint64_t(n)) != 0;
    if (skip_frame_draws_) {
      ++skip_frame_skipped_frames_;
    }
    return;
  }
  if (skip_frame_draws_n_ > 1) {
    // Back to drawing every frame: the summary, and every skipped resolve destination no drawn frame wrote.
    uint32_t unique = 0;
    for (const auto& [dest, count] : skip_frame_skipped_resolves_) {
      if (!skip_frame_drawn_resolves_.count(dest)) {
        if (++unique <= 20) {
          REXGPU_WARN("[s15][SKIPDRAW] resolve to 0x{:08X} only in skipped frames ({} times): a one-off render "
                      "may be missing",
                      dest, count);
        }
      }
    }
    REXGPU_INFO("[s15][SKIPDRAW] off: {} frames, {} skipped, {} draws and {} resolves skipped; {} resolve "
                "destinations in drawn frames, {} only in skipped frames",
                skip_frame_frames_, skip_frame_skipped_frames_, skip_frame_draws_skipped_,
                skip_frame_resolves_skipped_, skip_frame_drawn_resolves_.size(), unique);
  }
  skip_frame_draws_n_ = 0;
  skip_frame_draws_ = false;
}

bool CommandProcessor::SkipFrameResolve(uint32_t dest) {
  if (skip_frame_draws_n_ <= 1) {
    return false;
  }
  if (!skip_frame_draws_) {
    ++skip_frame_drawn_resolves_[dest];
    return false;
  }
  ++skip_frame_resolves_skipped_;
  ++skip_frame_skipped_resolves_[dest];
  return true;
}

bool CommandProcessor::ExecutePacketType3_INDIRECT_BUFFER(memory::RingBuffer* reader,
                                                          uint32_t packet, uint32_t count) {
  // indirect buffer dispatch
  uint32_t list_ptr = CpuToGpu(reader->ReadAndSwap<uint32_t>());
  uint32_t list_length = reader->ReadAndSwap<uint32_t>();
  assert_zero(list_length & ~0xFFFFF);
  list_length &= 0xFFFFF;
  // wwe2k14 patch (Session 38, NATIVE): census of where the command buffers live.
  {
    NativeCensus& census = native_census_;
    ++census.indirect_buffers;
    const uint32_t region = (list_ptr & 0x1FFFFFFF) >> 20;
    bool known = false;
    for (uint32_t i = 0; i < census.ib_region_count; ++i) {
      known |= census.ib_regions[i] == region;
    }
    if (!known) {
      if (census.ib_region_count < 8) {
        census.ib_regions[census.ib_region_count++] = region;
      } else {
        ++census.ib_region_overflow;
      }
    }
  }
  // wwe2k14 patch (Session 48, HALF2): gpu_native_half2_skip_gpr_draws - the D3D library's GPR-allocation flush
  // (0x82CB0F58 calls a prebuilt buffer of 24 one-vertex point draws after every shader GPR split change, 40 a match
  // frame; HALF2_MAP.md section 3 in the game repository): a buffer of exactly those draws is not executed.
  if (Half2SkipGprBuffer(list_ptr, list_length)) {
    return true;
  }
  // wwe2k14 patch (Session 78, NATIVE FRAME), answers side: a segment the ring calls becomes the drawing side's
  // current one (it catches up inside it and completes it below); a buffer a segment calls is run whole by the drawing
  // side before the answers side enters it (its catch-up target is this packet's end).
  const bool nf_segment = nf_role_ == NfRole::kAnswers && native_ib_depth_ == 0;
  if (nf_segment) {
    nf_seg_ptr_ = GpuToCpu(list_ptr) & 0x1FFFFFFF;
    nf_seg_words_ = list_length;
    nf_seg_done_ = 0;
    ++nf_census_.segments;
  } else if (nf_role_ == NfRole::kAnswers && native_ib_depth_ == 1) {
    ++nf_census_.nested;
    NfCatchUp(nf_answer_pos_ + 1 + count);
  }
  // wwe2k14 patch (Session 48, HALF2): where the INDIRECT_BUFFER packet itself is (its header word), for the draws
  // inside the buffer.
  const uint32_t half2_caller_before = half2_ib_caller_;
  half2_ib_caller_ = half2_packet_at_;
  ++native_ib_depth_;
  // wwe2k14 patch (Session 51 step 2, gpu_native_detached 1): a library segment from the ring, executed from the
  // library calls' state log.
  // A buffer called from a copy is read from guest memory: no shift for its packets.
  const int64_t half2_bias_before = half2_exec_bias_;
  half2_exec_bias_ = 0;
  if (native_ib_depth_ != 1 || !Half2LogExecuteSegment(GpuToCpu(list_ptr) & 0x1FFFFFFF, list_length)) {
    ExecuteIndirectBuffer(GpuToCpu(list_ptr), list_length);
  }
  half2_exec_bias_ = half2_bias_before;
  --native_ib_depth_;
  half2_ib_caller_ = half2_caller_before;
  if (nf_segment) {  // (Session 78) the drawing side completes the segment
    NfCatchUp(nf_seg_words_);
    nf_seg_ptr_ = 0;
  }
  if (half2_ && !native_ib_depth_) {
    const uint32_t start = GpuToCpu(list_ptr) & 0x1FFFFFFF;
    Half2EndSegment(start, start + list_length * 4);
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_WAIT_REG_MEM(memory::RingBuffer* reader, uint32_t packet,
                                                       uint32_t count) {
  SCOPE_profile_cpu_f("gpu");
  rex::graphics::free_counters::Bump(rex::graphics::free_counters::g_c.wait_reg_mem);  // (Session 92 M0: log-only counter)

  // wait until a register or memory location is a specific value

  uint32_t wait_info = reader->ReadAndSwap<uint32_t>();
  uint32_t poll_reg_addr = reader->ReadAndSwap<uint32_t>();
  uint32_t ref = reader->ReadAndSwap<uint32_t>();
  uint32_t mask = reader->ReadAndSwap<uint32_t>();
  uint32_t wait = reader->ReadAndSwap<uint32_t>();

  bool is_memory = (wait_info & 0x10) != 0;
  if (is_memory) {
    // wwe2k14 patch (Session 66, H1.1): a poll of a pending readback destination sees its data first.
    OnGuestVisibleWrite(GuestVisibleEvent::kWaitRegMem, poll_reg_addr & ~uint32_t(0x3));
  }

  // Statistics (gpu_wait_reg_mem_stats); only touched on the command processor thread.
  static uint64_t stat_packets = 0, stat_waited = 0, stat_sleeps = 0, stat_wait_us = 0;
  static std::unordered_map<uint32_t, uint32_t> stat_addresses;
  static auto stat_start = std::chrono::steady_clock::now();
  const bool stats = REXCVAR_GET(gpu_wait_reg_mem_stats);
  const auto wait_start = std::chrono::steady_clock::now();
  bool waited = false;
  if (stats) {
    ++stat_packets;
  }

  bool matched = false;
  bool timeline_wait_recorded = false;
  uint32_t value = 0;
  do {
    if (is_memory) {
      value =
          *reinterpret_cast<uint32_t*>(memory_->TranslatePhysical(poll_reg_addr & ~uint32_t(0x3)));
      value = xenos::GpuSwap(value, static_cast<xenos::Endian>(poll_reg_addr & 0x3));
    } else {
      value = ReadRegisterValue(poll_reg_addr);
      if (poll_reg_addr == XE_GPU_REG_COHER_STATUS_HOST) {
        MakeCoherent();
        value = ReadRegisterValue(poll_reg_addr);
      }
    }
    switch (wait_info & 0x7) {
      case 0x0:  // Never.
        matched = false;
        break;
      case 0x1:  // Less than reference.
        matched = (value & mask) < ref;
        break;
      case 0x2:  // Less than or equal to reference.
        matched = (value & mask) <= ref;
        break;
      case 0x3:  // Equal to reference.
        matched = (value & mask) == ref;
        break;
      case 0x4:  // Not equal to reference.
        matched = (value & mask) != ref;
        break;
      case 0x5:  // Greater than or equal to reference.
        matched = (value & mask) >= ref;
        break;
      case 0x6:  // Greater than reference.
        matched = (value & mask) > ref;
        break;
      case 0x7:  // Always
        matched = true;
        break;
    }
    if (!matched) {
      // wwe2k14 patch (Session 77): the det wait fields once per wait (they do not change while it polls; a store
      // per poll would keep moving det_wait_seq_ and starve the scheduler's DetDrained).
      if (!waited) {
        det_wait_addr_.store(poll_reg_addr);
        det_wait_ref_.store(ref);
        det_wait_mask_.store(mask);
        det_wait_func_.store(wait_info & 0x7);
        SetDetWaitKind(is_memory ? 2 : 3);
        det::SetCpWaiting(true);  // wwe2k14 patch (Session 15): det_frames (cleared after the wait loop)
      }
      waited = true;
      if (!timeline_wait_recorded && perf::timeline::Enabled()) {
        perf::timeline::Record("wrm_b", poll_reg_addr, ref, value);
        timeline_wait_recorded = true;
      }
      // Wait.
      if (wait >= 0x100) {
        D3dNoringGatePoll(true);  // (M5c, G3) a wait inside an item still releases an open pacing gate (insurance; no-op with the latch off)
        PrepareForWait();
        if (stats) {
          ++stat_sleeps;
        }
        if (!REXCVAR_GET(vsync) || REXCVAR_GET(gpu_wait_reg_mem_spin)) {
          // User wants it fast and dangerous.
          // wwe2k14 patch (Session 37): gpu_wait_reg_mem_sleep_us > 0 = a short precise sleep per poll instead (laptops).
          int32_t sleep_us = REXCVAR_GET(gpu_wait_reg_mem_sleep_us);
          {  // WWE13: A/B within one run (WWE13_AB_PERIOD_S): phase B sleeps 100 us per poll, phase A as configured
            static const int ab_period = [] {
              const char* v = std::getenv("WWE13_AB_PERIOD_S");
              const char* what = std::getenv("WWE13_AB_WHAT");
              return v && what && std::strcmp(what, "regmem") == 0 ? std::atoi(v) : 0;
            }();
            if (ab_period > 0) {
              const int64_t s = std::chrono::duration_cast<std::chrono::seconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
              sleep_us = ((s / ab_period) & 1) ? 100 : 0;
            }
          }
          if (sleep_us > 0 && REXCVAR_GET(vsync)) {
            rex::thread::PreciseSleep(std::chrono::microseconds(sleep_us));
          } else {
            rex::thread::MaybeYield();
          }
        } else {
          rex::thread::Sleep(std::chrono::milliseconds(wait / 0x100));
        }
        rex::thread::SyncMemory();
        ReturnFromWait();

        if (!worker_running_) {
          // Short-circuited exit.
          return false;
        }
      } else {
        rex::thread::MaybeYield();
      }
    }
  } while (!matched);
  if (waited) {
    det::SetCpWaiting(false);  // wwe2k14 patch (Session 15): det_frames
    SetDetWaitKind(0);
    det::NoteCpProgress();
  }
  if (timeline_wait_recorded) {
    perf::timeline::Record("wrm_e", poll_reg_addr, ref, value);
  }

  if (stats) {
    const auto now = std::chrono::steady_clock::now();
    if (waited) {
      ++stat_waited;
      stat_wait_us += uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(now - wait_start).count());
      ++stat_addresses[(is_memory ? 0x80000000u : 0u) | (poll_reg_addr & 0x7FFFFFFFu)];
    }
    if (now - stat_start >= std::chrono::seconds(5)) {
      uint32_t top_address = 0, top_count = 0;
      for (const auto& [address, count] : stat_addresses) {
        if (count > top_count) {
          top_address = address;
          top_count = count;
        }
      }
      const double seconds = std::chrono::duration<double>(now - stat_start).count();
      REXGPU_INFO(
          "WAIT_REG_MEM stats over {:.1f} s: {} packets, {} had to wait, {} sleep polls, {:.1f} ms waiting "
          "({:.1f}%); most waited-on: {} 0x{:X} ({}x){}",
          seconds, stat_packets, stat_waited, stat_sleeps, stat_wait_us / 1000.0,
          stat_wait_us / 10000.0 / seconds, (top_address & 0x80000000u) ? "memory" : "register",
          top_address & 0x7FFFFFFFu, top_count, REXCVAR_GET(gpu_wait_reg_mem_spin) ? " [spin]" : " [sleep]");
      stat_packets = stat_waited = stat_sleeps = stat_wait_us = 0;
      stat_addresses.clear();
      stat_start = now;
    }
  }

  return true;
}

// wwe2k14 patch (Session 92, M5a): the pacing wait of the no-ring handshake without a packet. The WAIT_REG_MEM loop above for the
// one shape the queue needs (memory poll, function "equal", mask all ones, wait 256): same readback flush, same det wait fields and
// kind 2, same sleep / yield policy, SyncMemory, progress note and timeline records. The gpu_address carries the endian bits as the
// packet's poll address does. False when the worker stopped.
bool CommandProcessor::WaitGuestWord(uint32_t gpu_address, uint32_t ref) {
  constexpr uint32_t kMask = 0xFFFFFFFFu;
  constexpr uint32_t kWait = 256;
  constexpr uint32_t kWaitInfo = 0x13;  // memory, equal
  if (nf_role_ != NfRole::kAnswers || (nf_ && REXCVAR_GET(gpu_native_frame_draw))) RepeatNotePacket(PM4_WAIT_REG_MEM);  // (S2: as the packet did)
  OnGuestVisibleWrite(GuestVisibleEvent::kWaitRegMem, gpu_address & ~uint32_t(0x3));
  const auto wait_start = std::chrono::steady_clock::now();
  int64_t next_stuck_ms = 1000;  // (Session 93, log only)
  bool waited = false;
  bool timeline_wait_recorded = false;
  uint32_t value = 0;
  bool matched = false;
  do {
    value = *reinterpret_cast<uint32_t*>(memory_->TranslatePhysical(gpu_address & ~uint32_t(0x3)));
    value = xenos::GpuSwap(value, static_cast<xenos::Endian>(gpu_address & 0x3));
    matched = (value & kMask) == ref;
    if (!matched) {
      if (!waited) {
        det_wait_addr_.store(gpu_address);
        det_wait_ref_.store(ref);
        det_wait_mask_.store(kMask);
        det_wait_func_.store(kWaitInfo & 0x7);
        SetDetWaitKind(2);
        det::SetCpWaiting(true);
      }
      waited = true;
      if (!timeline_wait_recorded && perf::timeline::Enabled()) {
        perf::timeline::Record("wrm_b", gpu_address, ref, value);
        timeline_wait_recorded = true;
      }
      D3dNoringGatePoll(true);  // (M5c, G3) as in the WAIT_REG_MEM loop
      {  // wwe2k14 patch (Session 93, log only; the Adreno 720 stock-driver freeze): a wait that does not end - at 1 s, then every 5 s (64 lines)
        static std::atomic<uint32_t> stuck_lines{0};
        const int64_t waited_ms =
            int64_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - wait_start).count());
        if (waited_ms >= next_stuck_ms) {
          next_stuck_ms = waited_ms + 5000;
          if (stuck_lines.fetch_add(1, std::memory_order_relaxed) < 64) {
            const char* call = nullptr;
            const uint64_t call_ms = ui::Presenter::HostPresentCallMs(&call);
            REXGPU_WARN("[d3d][wait][stuck] the command thread has waited {} ms for guest word {:08X} == {:08X} (now {:08X}); host present "
                        "call in progress: {} {} ms", waited_ms, gpu_address, ref, value, call_ms ? call : "none", call_ms);
          }
        }
      }
      PrepareForWait();
      if (REXCVAR_GET(vsync) && D3dNoringSyncEventsOn()) {
        // wwe2k14 patch (Session 92, M5b): the flip word is cleared by the guest's vblank handler, which MarkVblank dispatches and then
        // signals: wait on that event (2 ms guard) instead of sleeping / spinning (also when gpu_wait_reg_mem_spin is on, as it is under
        // the low-power defaults: the check's first run showed that branch took every wait). The value is re-read at the loop top.
        D3dNoringVblankWait(2);
      } else if (!REXCVAR_GET(vsync) || REXCVAR_GET(gpu_wait_reg_mem_spin)) {
        const int32_t sleep_us = REXCVAR_GET(gpu_wait_reg_mem_sleep_us);
        if (sleep_us > 0 && REXCVAR_GET(vsync)) {
          rex::thread::PreciseSleep(std::chrono::microseconds(sleep_us));
        } else {
          rex::thread::MaybeYield();
        }
      } else {
        rex::thread::Sleep(std::chrono::milliseconds(kWait / 0x100));
      }
      rex::thread::SyncMemory();
      ReturnFromWait();
      if (!worker_running_) return false;
    }
  } while (!matched);
  if (waited) {
    det::SetCpWaiting(false);
    SetDetWaitKind(0);
    det::NoteCpProgress();
  }
  if (timeline_wait_recorded) perf::timeline::Record("wrm_e", gpu_address, ref, value);
  rex::graphics::free_counters::Bump(rex::graphics::free_counters::g_c.direct_waits);
  rex::graphics::free_counters::Bump(
      rex::graphics::free_counters::g_c.direct_wait_us,
      uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - wait_start).count()));
  return true;
}

bool CommandProcessor::ExecutePacketType3_REG_RMW(memory::RingBuffer* reader, uint32_t packet,
                                                  uint32_t count) {
  // register read/modify/write
  // ? (used during shader upload and edram setup)
  uint32_t rmw_info = reader->ReadAndSwap<uint32_t>();
  uint32_t and_mask = reader->ReadAndSwap<uint32_t>();
  uint32_t or_mask = reader->ReadAndSwap<uint32_t>();
  // wwe2k14 patch (Session 90, contract R): with the image mode on, the read-modify-write runs on the record's image (the foreign
  // check left this packet alone); declined (a register the image does not model) = the register file first, then the stock way.
  if (nf_role_ != NfRole::kAnswers && D3dFeedState()) {
    if (D3dFeedRegRmw(*this, rmw_info, and_mask, or_mask)) return true;
    D3dFeedLastPacket() = (3u << 8) | PM4_REG_RMW;
    D3dFeedForeign(*this);
  }
  uint32_t value = register_file_->values[rmw_info & 0x1FFF];
  if ((rmw_info >> 31) & 0x1) {
    // & reg
    value &= register_file_->values[and_mask & 0x1FFF];
  } else {
    // & imm
    value &= and_mask;
  }
  if ((rmw_info >> 30) & 0x1) {
    // | reg
    value |= register_file_->values[or_mask & 0x1FFF];
  } else {
    // | imm
    value |= or_mask;
  }
  WriteRegister(rmw_info & 0x1FFF, value);
  return true;
}

bool CommandProcessor::ExecutePacketType3_REG_TO_MEM(memory::RingBuffer* reader, uint32_t packet,
                                                     uint32_t count) {
  // Copy Register to Memory (?)
  // Count is 2, assuming a Register Addr and a Memory Addr.

  uint32_t reg_addr = reader->ReadAndSwap<uint32_t>();
  uint32_t mem_addr = reader->ReadAndSwap<uint32_t>();

  uint32_t reg_val = ReadRegisterValue(reg_addr);

  auto endianness = static_cast<xenos::Endian>(mem_addr & 0x3);
  mem_addr &= ~0x3;
  reg_val = GpuSwap(reg_val, endianness);
  OnGuestVisibleWrite(GuestVisibleEvent::kRegToMem, mem_addr);  // wwe2k14 patch (Session 66, H1.1)
  memory::store(memory_->TranslatePhysical(mem_addr), reg_val);

  return true;
}

bool CommandProcessor::ExecutePacketType3_MEM_WRITE(memory::RingBuffer* reader, uint32_t packet,
                                                    uint32_t count) {
  uint32_t write_addr = reader->ReadAndSwap<uint32_t>();
  OnGuestVisibleWrite(GuestVisibleEvent::kMemWrite, write_addr & ~uint32_t(0x3));  // wwe2k14 patch (Session 66, H1.1)
  for (uint32_t i = 0; i < count - 1; i++) {
    uint32_t write_data = reader->ReadAndSwap<uint32_t>();
    if (i == 0) {
      perf::timeline::Record("memw", write_addr, write_data, count - 1);
    }

    auto endianness = static_cast<xenos::Endian>(write_addr & 0x3);
    auto addr = write_addr & ~0x3;
    write_data = GpuSwap(write_data, endianness);
    memory::store(memory_->TranslatePhysical(addr), write_data);
    write_addr += 4;
  }

  return true;
}

bool CommandProcessor::ExecutePacketType3_COND_WRITE(memory::RingBuffer* reader, uint32_t packet,
                                                     uint32_t count) {
  // conditional write to memory or register
  uint32_t wait_info = reader->ReadAndSwap<uint32_t>();
  uint32_t poll_reg_addr = reader->ReadAndSwap<uint32_t>();
  uint32_t ref = reader->ReadAndSwap<uint32_t>();
  uint32_t mask = reader->ReadAndSwap<uint32_t>();
  uint32_t write_reg_addr = reader->ReadAndSwap<uint32_t>();
  uint32_t write_data = reader->ReadAndSwap<uint32_t>();
  uint32_t value;
  if (wait_info & 0x10) {
    // Memory.
    auto endianness = static_cast<xenos::Endian>(poll_reg_addr & 0x3);
    poll_reg_addr &= ~0x3;
    OnGuestVisibleWrite(GuestVisibleEvent::kWaitRegMem, poll_reg_addr);  // wwe2k14 patch (Session 66, H1.1)
    value = memory::load<uint32_t>(memory_->TranslatePhysical(poll_reg_addr));
    value = GpuSwap(value, endianness);
  } else {
    // Register.
    value = ReadRegisterValue(poll_reg_addr);
  }
  bool matched = false;
  switch (wait_info & 0x7) {
    case 0x0:  // Never.
      matched = false;
      break;
    case 0x1:  // Less than reference.
      matched = (value & mask) < ref;
      break;
    case 0x2:  // Less than or equal to reference.
      matched = (value & mask) <= ref;
      break;
    case 0x3:  // Equal to reference.
      matched = (value & mask) == ref;
      break;
    case 0x4:  // Not equal to reference.
      matched = (value & mask) != ref;
      break;
    case 0x5:  // Greater than or equal to reference.
      matched = (value & mask) >= ref;
      break;
    case 0x6:  // Greater than reference.
      matched = (value & mask) > ref;
      break;
    case 0x7:  // Always
      matched = true;
      break;
  }
  if (matched) {
    // Write.
    if (wait_info & 0x100) {
      // Memory.
      auto endianness = static_cast<xenos::Endian>(write_reg_addr & 0x3);
      write_reg_addr &= ~0x3;
      write_data = GpuSwap(write_data, endianness);
      OnGuestVisibleWrite(GuestVisibleEvent::kCondWrite, write_reg_addr);  // wwe2k14 patch (Session 66, H1.1)
      memory::store(memory_->TranslatePhysical(write_reg_addr), write_data);
    } else {
      // Register.
      WriteRegister(write_reg_addr, write_data);
    }
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_EVENT_WRITE(memory::RingBuffer* reader, uint32_t packet,
                                                      uint32_t count) {
  // generate an event that creates a write to memory when completed
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  // Writeback initiator.
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);
  // wwe2k14 patch (Session 7, TILING-STAGE0): the tiling log shows the unpredicated SCREEN_EXT_INIT (with
  // predicated tiling, the game emits one when a tiling bracket begins; the per-draw ones are predicated).
  if (tiling_log_active_ && !(packet & 1) && (initiator & 0x3F) == SCREEN_EXT_INIT) {
    TilingLogEvent("ext_init", -1, initiator, true);
  }
  if (count == 1) {
    // Just an event flag? Where does this write?
  } else {
    // Write to an address.
    assert_always();
    reader->AdvanceRead((count - 1) * sizeof(uint32_t));
  }
  // wwe2k14 patch (Session 7, EXTENTS-SHADOW): SCREEN_EXT_INIT starts a screen extents window (Session 78: the
  // drawing side's, which runs the draws the window measures).
  if (extents_active_ && nf_role_ != NfRole::kAnswers && (initiator & 0x3F) == SCREEN_EXT_INIT) {
    extents_shadow_->OnScreenExtentsInit(bin_select_);
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_EVENT_WRITE_SHD(memory::RingBuffer* reader,
                                                          uint32_t packet, uint32_t count) {
  // generate a VS|PS_done event
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  uint32_t address = reader->ReadAndSwap<uint32_t>();
  uint32_t value = reader->ReadAndSwap<uint32_t>();

  // Writeback initiator.
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);
  uint32_t data_value;
  if ((initiator >> 31) & 0x1) {
    // Write counter (GPU vblank counter?).
    data_value = counter_;
  } else {
    // Write value.
    data_value = value;
  }
  perf::timeline::Record("evw", address, data_value, initiator);
  OnGuestVisibleWrite(GuestVisibleEvent::kEventWriteShd, address & ~uint32_t(0x3));  // wwe2k14 patch (Session 66, H1.1)
  auto endianness = static_cast<xenos::Endian>(address & 0x3);
  address &= ~0x3;
  data_value = GpuSwap(data_value, endianness);
  memory::store(memory_->TranslatePhysical(address), data_value);
  return true;
}

bool CommandProcessor::ExecutePacketType3_EVENT_WRITE_EXT(memory::RingBuffer* reader,
                                                          uint32_t packet, uint32_t count) {
  // generate a screen extent event
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  uint32_t address = reader->ReadAndSwap<uint32_t>();
  // Writeback initiator.
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);
  auto endianness = static_cast<xenos::Endian>(address & 0x3);
  address &= ~0x3;
  OnGuestVisibleWrite(GuestVisibleEvent::kEventWriteExt, address);  // wwe2k14 patch (Session 66, H1.1)
  // wwe2k14 patch (Session 7, TILING-STAGE0): count it (cpf_tile) and log where the extents go.
  ++tiling_counters_.event_write_ext;
  if (tiling_log_active_) {
    TilingLogEvent("ext", address, initiator, false);
  }

  // Let us hope we can fake this.
  // This callback tells the driver the xy coordinates affected by a previous
  // drawcall.
  // https://www.google.com/patents/US20060055701
  uint16_t extents[] = {
      0 >> 3,                                    // min x
      xenos::kTexture2DCubeMaxWidthHeight >> 3,  // max x
      0 >> 3,                                    // min y
      xenos::kTexture2DCubeMaxWidthHeight >> 3,  // max y
      0,                                         // min z
      1,                                         // max z
  };
  // wwe2k14 patch (Session 7, EXTENTS-SHADOW): gpu_extents_mode 1 only counts (the values above are written);
  // mode 2 with gpu_extents_allow_write writes the computed extents when every draw of the window has them.
  if (extents_active_) {
    uint16_t computed[6];
    if (extents_shadow_->OnScreenExtentsReport(bin_select_, computed)) {
      std::memcpy(extents, computed, sizeof(extents));
    }
  }
  assert_true(endianness == xenos::Endian::k8in16);
  memory::copy_and_swap_16_unaligned(memory_->TranslatePhysical(address), extents,
                                     rex::countof(extents));
  return true;
}

bool CommandProcessor::ExecutePacketType3_EVENT_WRITE_ZPD(memory::RingBuffer* reader,
                                                          uint32_t packet, uint32_t count) {
  // Set by D3D as BE but struct ABI is LE
  const uint32_t kQueryFinished = rex::byte_swap(0xFFFFFEED);
  assert_true(count == 1);
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  ++native_census_.zpd_events;  // wwe2k14 patch (Session 38, NATIVE): census
  OnGuestVisibleWrite(GuestVisibleEvent::kEventWriteZpd);  // wwe2k14 patch (Session 66, H1.1)
  // Writeback initiator.
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);

  // Occlusion queries:
  // This command is send on query begin and end.
  // As a workaround report some fixed amount of passed samples.
  auto fake_sample_count = REXCVAR_GET(query_occlusion_fake_sample_count);
  if (fake_sample_count >= 0) {
    auto* pSampleCounts = memory_->TranslatePhysical<xe_gpu_depth_sample_counts*>(
        register_file_->values[XE_GPU_REG_RB_SAMPLE_COUNT_ADDR]);
    if (!pSampleCounts) {
      return true;
    }
    // 0xFFFFFEED is written to this two locations by D3D only on D3DISSUE_END
    // and used to detect a finished query.
    bool is_end_via_z_pass =
        pSampleCounts->ZPass_A == kQueryFinished || pSampleCounts->ZPass_B == kQueryFinished;
    // Older versions of D3D also checks for ZFail (4D5307D5).
    bool is_end_via_z_fail =
        pSampleCounts->ZFail_A == kQueryFinished || pSampleCounts->ZFail_B == kQueryFinished;
    std::memset(pSampleCounts, 0, sizeof(xe_gpu_depth_sample_counts));
    if (is_end_via_z_pass || is_end_via_z_fail) {
      pSampleCounts->ZPass_A = fake_sample_count;
      pSampleCounts->Total_A = fake_sample_count;
    }
  }

  return true;
}

bool CommandProcessor::ExecutePacketType3Draw(memory::RingBuffer* reader, uint32_t packet,
                                              const char* opcode_name, uint32_t viz_query_condition,
                                              uint32_t count_remaining) {
  // if viz_query_condition != 0, this is a conditional draw based on viz query.
  // This ID matches the one issued in PM4_VIZ_QUERY
  // uint32_t viz_id = viz_query_condition & 0x3F;
  // when true, render conditionally based on query result
  // uint32_t viz_use = viz_query_condition & 0x100;

  assert_not_zero(count_remaining);
  if (!count_remaining) {
    REXGPU_ERROR("{}: Packet too small, can't read VGT_DRAW_INITIATOR", opcode_name);
    return false;
  }
  // wwe2k14 patch (Session 48, HALF2): the packet's header word (already read) in guest physical memory.
  const uint32_t half2_packet_physical = half2_packet_at_;
  reg::VGT_DRAW_INITIATOR vgt_draw_initiator;
  vgt_draw_initiator.value = reader->ReadAndSwap<uint32_t>();
  --count_remaining;
  WriteRegister(XE_GPU_REG_VGT_DRAW_INITIATOR, vgt_draw_initiator.value);

  bool draw_succeeded = true;
  // wwe2k14 patch (Session 7, EXTENTS-SHADOW): what ExtentsOnDraw needs to know about the IssueDraw call.
  bool extents_issue_called = false;
  uint32_t extents_copies_before = 0;
  size_t extents_anatomy_rows_before = SIZE_MAX;
  // TODO(Triang3l): Remove IndexBufferInfo and replace handling of all this
  // with PrimitiveProcessor when the old Vulkan renderer is removed.
  bool is_indexed = false;
  IndexBufferInfo index_buffer_info;
  switch (vgt_draw_initiator.source_select) {
    case xenos::SourceSelect::kDMA: {
      // Indexed draw.
      is_indexed = true;

      // Two separate bounds checks so if there's only one missing register
      // value out of two, one uint32_t will be skipped in the command buffer,
      // not two.
      assert_not_zero(count_remaining);
      if (!count_remaining) {
        REXGPU_ERROR("{}: Packet too small, can't read VGT_DMA_BASE", opcode_name);
        return false;
      }
      uint32_t vgt_dma_base = reader->ReadAndSwap<uint32_t>();
      --count_remaining;
      WriteRegister(XE_GPU_REG_VGT_DMA_BASE, vgt_dma_base);
      reg::VGT_DMA_SIZE vgt_dma_size;
      assert_not_zero(count_remaining);
      if (!count_remaining) {
        REXGPU_ERROR("{}: Packet too small, can't read VGT_DMA_SIZE", opcode_name);
        return false;
      }
      vgt_dma_size.value = reader->ReadAndSwap<uint32_t>();
      --count_remaining;
      WriteRegister(XE_GPU_REG_VGT_DMA_SIZE, vgt_dma_size.value);

      uint32_t index_size_bytes = vgt_draw_initiator.index_size == xenos::IndexFormat::kInt16
                                      ? sizeof(uint16_t)
                                      : sizeof(uint32_t);
      // The base address must already be word-aligned according to the R6xx
      // documentation, but for safety.
      index_buffer_info.guest_base = vgt_dma_base & ~(index_size_bytes - 1);
      index_buffer_info.endianness = vgt_dma_size.swap_mode;
      index_buffer_info.format = vgt_draw_initiator.index_size;
      index_buffer_info.length = vgt_dma_size.num_words * index_size_bytes;
      index_buffer_info.count = vgt_draw_initiator.num_indices;
    } break;
    case xenos::SourceSelect::kImmediate: {
      // TODO(Triang3l): VGT_IMMED_DATA.
      REXGPU_ERROR(
          "{}: Using immediate vertex indices, which are not supported yet. "
          "Report the game to Xenia developers!",
          opcode_name, uint32_t(vgt_draw_initiator.source_select));
      draw_succeeded = false;
      assert_always();
    } break;
    case xenos::SourceSelect::kAutoIndex: {
      // Auto draw.
      index_buffer_info.guest_base = 0;
      index_buffer_info.length = 0;
    } break;
    default: {
      // Invalid source selection.
      draw_succeeded = false;
      assert_unhandled_case(vgt_draw_initiator.source_select);
    } break;
  }

  // wwe2k14 patch (Session 48, HALF2): match the packet (its guest physical address) to the game's library draw.
  half2_matched_kind_ = 0;  // wwe2k14 patch (Session 50, LEAN DRAWS): set by Half2OnCpDraw
  if (half2_) {
    Half2OnCpDraw(half2_packet_physical, false,
                  {uint32_t(vgt_draw_initiator.prim_type), uint32_t(vgt_draw_initiator.num_indices),
                   is_indexed ? index_buffer_info.guest_base & 0x1FFFFFFF : 0,
                   is_indexed && vgt_draw_initiator.index_size == xenos::IndexFormat::kInt32});
  }

  // Skip to the next command, for example, if there are immediate indexes that
  // we don't support yet.
  reader->AdvanceRead(count_remaining * sizeof(uint32_t));

  if (draw_succeeded) {
    auto viz_query = (D3dFeedDrawRegs() ? *D3dFeedDrawRegs() : *register_file_).Get<reg::PA_SC_VIZ_QUERY>();
    // wwe2k14 patch (Session 38, NATIVE): census.
    ++(native_ib_depth_ ? native_census_.draws_in_ib : native_census_.draws_in_ring);
    if (viz_query.viz_query_ena) {
      ++native_census_.viz_query_draws;
      native_census_.viz_query_dropped += viz_query.kill_pix_post_hi_z ? 1 : 0;
    }
    if (!(viz_query.viz_query_ena && viz_query.kill_pix_post_hi_z)) {
      // TODO(Triang3l): Don't drop the draw call completely if the vertex
      // shader has memexport.
      // TODO(Triang3l || JoelLinn): Handle this properly in the render
      // backends.

      bool major_mode_explicit =
          xenos::IsMajorModeExplicit(vgt_draw_initiator.major_mode, vgt_draw_initiator.prim_type);
      // wwe2k14 patch (Session 6, COUNTERS): count the call, and time and describe it during a frame-anatomy
      // capture (anatomy_active_ only changes at frame boundaries).
      ++frame_counters_.issue_draw;
      extents_issue_called = true;
      extents_copies_before = frame_counters_.issue_copy;
      if (anatomy_active_) {
        extents_anatomy_rows_before = anatomy_->rows.size();
      }
      const bool anatomy = anatomy_active_;
      if (anatomy) {
        AnatomyBeginDraw();
      }
      // wwe2k14 patch (Session 50, LEAN DRAWS): a library draw the backend records itself (gpu_native_lean_draws),
      // otherwise IssueDraw (never during a frame-anatomy capture).
      // wwe2k14 patch (Session 80, own crowd): a crowd run's first member left to our own crowd renderer.
      own_crowd_skip_draw_ = !anatomy && (!scene_crowd_local_.empty() ||
                                          scene_crowd_queued_.load(std::memory_order_relaxed)) &&
                             SceneCrowdPreTake(SceneCrowdC76());
      // wwe2k14 patch (Session 87, crowd2): the census of the three instanced crowd shaders' draws, and with
      // gpu_native_own_crowd_all a lone draw of one (outside the queued runs) is taken as a run of one: recorded
      // without its draw call, then given to OwnCrowdDraw in SceneCrowdAfterDraw (the game's draw if that declines).
      if (!anatomy && SceneCrowdCensusBegin(own_crowd_skip_draw_)) own_crowd_skip_draw_ = true;
      // wwe2k14 patch (Session 77): gpu_det_skip_host_draws - nothing recorded (counted as drawn).
      if (REXCVAR_GET(gpu_det_skip_host_draws)) {
        draw_succeeded = true;
      } else if (anatomy || !IssueDrawLean(vgt_draw_initiator.prim_type, vgt_draw_initiator.num_indices,
                                           is_indexed ? &index_buffer_info : nullptr, major_mode_explicit,
                                           half2_matched_kind_, draw_succeeded)) {
        draw_succeeded = IssueDraw(vgt_draw_initiator.prim_type, vgt_draw_initiator.num_indices,
                                   is_indexed ? &index_buffer_info : nullptr, major_mode_explicit);
      }
      // wwe2k14 patch (Session 50, SCENE CROWD): the crowd members the game queued instead of recording.
      if (!anatomy && (scene_crowd_verify_active_ || !scene_crowd_local_.empty() || own_crowd_skip_draw_ ||
                    scene_crowd_queued_.load(std::memory_order_relaxed))) {
        SceneCrowdAfterDraw(vgt_draw_initiator.prim_type, vgt_draw_initiator.num_indices,
                            is_indexed ? &index_buffer_info : nullptr, major_mode_explicit, half2_matched_kind_, SceneCrowdC76(),
                            half2_packet_at_);
      }
      SceneCrowdCensusEnd();
      if (anatomy) {
        AnatomyEndDraw(vgt_draw_initiator.value, is_indexed, draw_succeeded, (packet & 1) != 0);
      }
      if (!draw_succeeded) {
        auto vgt_output_path_cntl = (D3dFeedDrawRegs() ? *D3dFeedDrawRegs() : *register_file_).Get<reg::VGT_OUTPUT_PATH_CNTL>();
        auto vgt_hos_cntl = (D3dFeedDrawRegs() ? *D3dFeedDrawRegs() : *register_file_).Get<reg::VGT_HOS_CNTL>();
        auto rb_modecontrol = (D3dFeedDrawRegs() ? *D3dFeedDrawRegs() : *register_file_).Get<reg::RB_MODECONTROL>();
        REXGPU_ERROR(
            "{}({}, {}, {}): Failed in backend "
            "(major_mode={}, explicit_major={}, path_select={}, tess_mode={}, edram_mode={})",
            opcode_name, static_cast<uint32_t>(vgt_draw_initiator.num_indices),
            uint32_t(vgt_draw_initiator.prim_type), uint32_t(vgt_draw_initiator.source_select),
            uint32_t(vgt_draw_initiator.major_mode), uint32_t(major_mode_explicit),
            uint32_t(vgt_output_path_cntl.path_select), uint32_t(vgt_hos_cntl.tess_mode),
            uint32_t(rb_modecontrol.edram_mode));
      }
    }
  }

  // wwe2k14 patch (Session 7, EXTENTS-SHADOW): every executed draw packet goes through the extents shadow (a
  // draw the backend didn't draw makes its window full-screen).
  if (extents_active_) {
    ExtentsOnDraw(extents_issue_called && draw_succeeded,
                  extents_issue_called && frame_counters_.issue_copy != extents_copies_before,
                  extents_anatomy_rows_before);
  }

  // If read the packed correctly, but merely couldn't execute it (because of,
  // for instance, features not supported by the host), don't terminate command
  // buffer processing as that would leave rendering in a way more inconsistent
  // state than just a single dropped draw command.
  return true;
}

// wwe2k14 patch (Session 50, SCENE CROWD): see the header.
static constexpr uint32_t kSceneCrowdRegister = 0x4000 + 4 * 76;  // VS float constant c76 (SQ_VS_CONST base 0)

void CommandProcessor::SceneCrowdPush(const uint32_t* first, const uint32_t* others, uint32_t count, bool verify) {
  SceneCrowdRun run;
  std::copy(first, first + 12, run.first);
  run.others.assign(others, others + 12 * size_t(count));
  run.verify = verify;
  scene_crowd_seen_.store(true, std::memory_order_relaxed);
  std::lock_guard lock(scene_crowd_mutex_);
  run.seq = scene_crowd_seq_++;
  // Runs never consumed (frames the command processor did not draw) do not pile up: at most ~16 frames' worth.
  if (scene_crowd_queue_.size() >= 1024) {
    scene_crowd_queue_.erase(scene_crowd_queue_.begin(), scene_crowd_queue_.begin() + 256);
    scene_crowd_stats_.dropped += 256;
  }
  scene_crowd_queue_.push_back(std::move(run));
  ++scene_crowd_stats_.pushed;
  scene_crowd_queued_.store(uint32_t(scene_crowd_queue_.size()), std::memory_order_relaxed);
}

// wwe2k14 patch (Session 87, crowd2): called right before every draw (the stock packet path, and the D3D layer's direct path)
// with whether PreTake held it. Counts a draw of the three instanced crowd shaders; with gpu_native_own_crowd_all, one that
// is not held (outside every queued run) is taken as a run of one: true = the caller records it without its draw call, and
// SceneCrowdAfterDraw gives it to OwnCrowdDraw (the game's draw if that declines).
bool CommandProcessor::SceneCrowdCensusBegin(bool held, const uint32_t* c76_typed) {
  g_crowd_census_lone = false;
  g_crowd_census_unheld = -1;
  const int ci = CrowdCensusIndex(active_vertex_shader());
  if (ci < 0) return false;
  ++g_crowd_census[ci].draws;
  if (held) return false;
  // (Crowd2 review B1: the density rank test, as the backend's CrowdDensityRejects and the queued members' rank apply: a lone
  // draw above the density is not drawn - not taken here, and the backend skips it as before.)
  if (const int32_t density = REXCVAR_GET(gpu_scene_crowd_density); density < 100) {
    const RegisterFile& regs = (D3dFeedDrawRegs() ? *D3dFeedDrawRegs() : *register_file_);
    constexpr uint32_t kC76 = 0x4000 + 4 * 76;
    uint32_t h = c76_typed ? (c76_typed[3] * 0x9E3779B1u ^ c76_typed[7] * 0x85EBCA77u ^ c76_typed[11] * 0xC2B2AE3Du) : (regs[kC76 + 3] * 0x9E3779B1u ^ regs[kC76 + 7] * 0x85EBCA77u ^ regs[kC76 + 11] * 0xC2B2AE3Du);
    const int32_t lone_rank = c76_typed ? CrowdRankOf(c76_typed[3], c76_typed[7], c76_typed[11]) : CrowdRankOf(regs[kC76 + 3], regs[kC76 + 7], regs[kC76 + 11]);
    CrowdRankDiagNote(1, active_vertex_shader() ? active_vertex_shader()->ucode_data_hash() : 0, h, lone_rank >= density);
    if (lone_rank >= density) return false;
  }
  if (REXCVAR_GET(gpu_native_own_crowd_all) && OwnCrowdActive() && !scene_crowd_verify_active_ &&
      !scene_crowd_blocked_.load(std::memory_order_relaxed) && !REXCVAR_GET(gpu_det_skip_host_draws)) {
    g_crowd_census_lone = true;
    return true;
  }
  g_crowd_census_unheld = ci;  // the game draws it: a run's first member (AfterDraw matches a run) or a stray
  return false;
}

// Crowd2 review B3: the direct path tells the crowd code this draw item's own guest address (0 after it), and reads how often
// SceneCrowdAfterDraw wrote the crowd registers itself (a run the game drew: the record-fed image is stale after it).
void CommandProcessor::SceneCrowdItemAt(uint32_t at) { g_scene_crowd_item_at = at; }

// wwe2k14 patch (Session 95, round 7 contract 4): per-thread scratch vectors for the crowd paths, so a draw item allocates less (run.others is still per run). The
// buffer keeps its capacity between uses (cleared on take); a nested user (none expected) gets a plain local one. Output is unchanged.
template <class T, int Slot>
class CrowdScratch {
 public:
  CrowdScratch() : p_(Busy() ? &local_ : &Pool()) {
    if (p_ != &local_) Busy() = true;
    p_->clear();
  }
  ~CrowdScratch() {
    if (p_ != &local_) {
      Busy() = false;
      if (p_->capacity() > (size_t(1) << 20)) std::vector<T>().swap(*p_);  // (a one-off huge run does not stay allocated)
    }
  }
  CrowdScratch(const CrowdScratch&) = delete;
  CrowdScratch& operator=(const CrowdScratch&) = delete;
  std::vector<T>& get() { return *p_; }

 private:
  static std::vector<T>& Pool() { static thread_local std::vector<T> v; return v; }
  static bool& Busy() { static thread_local bool b = false; return b; }
  std::vector<T> local_;
  std::vector<T>* p_;
};


// wwe2k14 patch (Session 92, M4f-B): see crowd_member_hook.h.
namespace {
SceneCrowdMemberDraw g_crowd_member_draw = nullptr;
void* g_crowd_member_ctx = nullptr;
const OwnedCrowdSource* g_crowd_source = nullptr;
}  // namespace
void SceneCrowdSetMemberDraw(SceneCrowdMemberDraw fn, void* ctx) {
  g_crowd_member_draw = fn;
  g_crowd_member_ctx = fn ? ctx : nullptr;
}
bool SceneCrowdMemberDrawInstalled() { return g_crowd_member_draw != nullptr; }
void OwnedCrowdSourceSet(const OwnedCrowdSource* source) { g_crowd_source = source; }
const OwnedCrowdSource* OwnedCrowdSourceGet() { return g_crowd_source; }
uint64_t CommandProcessor::SceneCrowdRegisterWrites() const { return g_scene_crowd_reg_writes; }

// After the draw (and SceneCrowdAfterDraw): a draw of the three shaders the game drew that no run claimed is a stray.
void CommandProcessor::SceneCrowdCensusEnd() {
  if (g_crowd_census_unheld >= 0) {
    ++g_crowd_census[g_crowd_census_unheld].stray;
    g_crowd_census_unheld = -1;
  }
}

// wwe2k14 patch (Session 80, own crowd): see the declaration. The same queue and match as SceneCrowdAfterDraw.
// (Session 92, M4d) The old path's source of a draw's member constants c76-c78: the record's image in a fed draw, else the register file
// (the expression SceneCrowdPreTake / SceneCrowdAfterDraw read before). The caller hands the pointer in; they do not touch registers.
const uint32_t* CommandProcessor::SceneCrowdC76() const {
  REX_CUT_PROBE(kCrowdC76);
  const RegisterFile& regs = (D3dFeedDrawRegs() ? *D3dFeedDrawRegs() : *register_file_);
  return &regs.values[kSceneCrowdRegister];
}

bool CommandProcessor::SceneCrowdMatchLocal(const uint32_t* c76) const {
  if (!OwnCrowdActive() || scene_crowd_verify_active_ || scene_crowd_blocked_.load(std::memory_order_relaxed)) return false;
  const size_t look = std::min<size_t>(scene_crowd_local_.size(), 32);
  for (size_t i = 0; i < look; ++i) {
    const SceneCrowdRun& r = scene_crowd_local_[i];
    if (!r.verify && r.first[0] == c76[0] && std::equal(c76, c76 + 12, r.first)) return true;
  }
  return false;
}

// (Session 95, gpu_d3d_wall bit 2) Set by SceneCrowdPreTake when its search of the runs found nothing, so that the draw's SceneCrowdAfterDraw would find nothing either. Command thread only.
bool g_crowd_pretake_nothing = false;
bool SceneCrowdPreTakeFoundNothing() { return g_crowd_pretake_nothing; }

bool CommandProcessor::SceneCrowdPreTake(const uint32_t* c76) {
  g_crowd_pretake_nothing = false;
  if (!OwnCrowdActive() || scene_crowd_verify_active_ || scene_crowd_blocked_.load(std::memory_order_relaxed)) {
    return false;
  }
  if (scene_crowd_queued_.load(std::memory_order_relaxed)) {
    std::lock_guard lock(scene_crowd_mutex_);
    for (SceneCrowdRun& r : scene_crowd_queue_) scene_crowd_local_.push_back(std::move(r));
    scene_crowd_queue_.clear();
    scene_crowd_queued_.store(0, std::memory_order_relaxed);
  }
  uint32_t now[12];  // (Session 92, M4d: the caller's c76-c78; the old path hands in the record's image / register file, SceneCrowdC76)
  for (uint32_t i = 0; i < 12; ++i) now[i] = c76[i];
  const size_t look = std::min<size_t>(scene_crowd_local_.size(), 32);
  bool any = false;  // (Session 95: any run, verify runs too - what SceneCrowdAfterDraw's search looks for)
  for (size_t i = 0; i < look; ++i) {
    const SceneCrowdRun& r = scene_crowd_local_[i];
    if (r.first[0] == now[0] && std::equal(now, now + 12, r.first)) {
      if (!r.verify) return true;
      any = true;
    }
  }
  g_crowd_pretake_nothing = !any;
  return false;
}

void CommandProcessor::SceneCrowdAfterDraw(xenos::PrimitiveType prim_type, uint32_t index_count,
                                           IndexBufferInfo* index_buffer_info, bool major_mode_explicit,
                                           uint32_t library_kind, const uint32_t* c76, uint32_t packet_id) {
  // wwe2k14 patch (Session 80, own crowd): this draw (a run's first member) was left to our renderer. Unless our draw
  // takes it below, it is drawn as the game's after all - before the other members (they become its instances), or
  // at any return on the way (the guard).
  const bool first_ours = own_crowd_skip_draw_;
  own_crowd_skip_draw_ = false;
  // wwe2k14 patch (Session 87, crowd2): a lone draw taken as a run of one (gpu_native_own_crowd_all), and the census.
  const bool lone = g_crowd_census_lone;
  g_crowd_census_lone = false;
  const int census = CrowdCensusIndex(active_vertex_shader());
  const bool feed_live = D3dFeedLive();  // (Session 90, K2: the restore and the decline census run only while the D3D feed is live)
  struct FirstGuard {
    CommandProcessor* cp;
    xenos::PrimitiveType prim_type;
    uint32_t index_count;
    IndexBufferInfo* index_buffer_info;
    bool major_mode_explicit;
    uint32_t library_kind;
    bool armed;
    int census;
    bool lone;
    bool tried = false;  // OwnCrowdDraw was tried in this call and declined
    uint32_t first_c76[12] = {};  // (Session 92, M4f-B: the first member's c76-c78, for the typed draw)
    void Draw() {
      if (!armed) return;
      armed = false;
      if (census >= 0) {
        CrowdCensus& cc = g_crowd_census[census];
        ++(tried ? cc.handed_back : (lone ? cc.stray : cc.first));
      }
      bool ok = false;
      if (g_crowd_member_draw && g_crowd_member_draw(g_crowd_member_ctx, first_c76, ok)) return;  // (typed: the first member through the typed draw)
      if (!cp->IssueDrawLean(prim_type, index_count, index_buffer_info, major_mode_explicit, library_kind, ok)) {
        cp->IssueDraw(prim_type, index_count, index_buffer_info, major_mode_explicit);
      }
    }
    ~FirstGuard() { Draw(); }
  } first_guard{this, prim_type, index_count, index_buffer_info, major_mode_explicit, library_kind, first_ours, census, lone};
  std::memcpy(first_guard.first_c76, c76, sizeof(first_guard.first_c76));
  uint32_t now[12];  // (Session 92, M4d: the caller's c76-c78, read now, before any member is written)
  for (uint32_t i = 0; i < 12; ++i) now[i] = c76[i];
  SceneCrowdStats& st = scene_crowd_stats_;
  // wwe2k14 patch (Session 90, K2): why our crowd program declined a run (OwnCrowdDraw's reason), counted, the first 16 logged.
  const auto note_decline = [&](uint32_t member_count) {
    if (REXCVAR_GET(gpu_d3d_wall_stats) > 0) {  // (Session 95, log only: under the cut the feed is not live and a declined run was silent)
      static uint32_t wall_declines = 0;
      if (wall_declines++ < 24 || wall_declines % 4096 == 0) {
        REXGPU_INFO("[scene][crowd] wall: declined run #{}: frame {}, {} member(s), reason {}, vs {:016X}", wall_declines, counter_, member_count, D3dFeedCrowd().last_decline,
                    active_vertex_shader() ? active_vertex_shader()->ucode_data_hash() : 0);
      }
    }
    if (!feed_live) return;
    D3dFeedCrowdState& cs = D3dFeedCrowd();
    const uint32_t reason = std::min<uint32_t>(cs.last_decline, 15);
    ++cs.declined_runs;
    ++cs.decline_by_reason[reason];
    if (cs.decline_logged < 16) {
      ++cs.decline_logged;
      REXGPU_INFO("[scene][crowd] our crowd program declined a run: frame {}, {} member(s), reason {} (1 no program, 2 mesh stride, "
                  "3 mesh endian / size, 4 position flags, 5 colour texture, 6 lighting map, 7 vertex lighting map, 8 pixel shader "
                  "differs, 10 switches / renderer not replacing, 11 no shaders, 12 RecordOwnCrowd refused, 0 none set); the game "
                  "draws it",
                  counter_, member_count, cs.last_decline);
    }
  };
  // wwe2k14 patch (Session 87, crowd2, gpu_native_own_crowd_all): the draw held for our crowd program that no run took
  // along (a lone draw, or a first member whose run has no other member to draw) is a run of one for it. The game's
  // draw (the guard) if it declines.
  const auto own_alone = [&]() {
    if (!first_guard.armed || census < 0 || !REXCVAR_GET(gpu_native_own_crowd_all)) return;
    if (OwnCrowdDraw(now, 1)) {
      first_guard.armed = false;
      ++g_crowd_census[census].ours;
      ++g_crowd_census[census].ours_calls;
    } else {
      first_guard.tried = true;
      note_decline(1);
    }
  };
  // Verify: the draws after a run's first must carry the queued matrices, in order.
  if (scene_crowd_verify_active_) {
    const uint32_t* expect = scene_crowd_verify_.others.data() + 12 * size_t(scene_crowd_verify_next_);
    if (std::equal(now, now + 12, expect)) {
      ++st.verify_ok;
      if (++scene_crowd_verify_next_ * 12 >= scene_crowd_verify_.others.size()) scene_crowd_verify_active_ = false;
      return;
    }
    ++st.verify_bad;
    if (scene_crowd_logged_ < 8) {
      ++scene_crowd_logged_;
      REXGPU_WARN("[scene][crowd] verify: member {} of {} differs (c76.x {:08X}, queued {:08X})",
                  scene_crowd_verify_next_ + 1, scene_crowd_verify_.others.size() / 12, now[0], expect[0]);
    }
    scene_crowd_verify_active_ = false;
  }
  // (Profile 3430: a lock and a vector erase per draw cost more than the pass saved - the game's runs are taken into
  // the command processor's own deque in one lock, searched without it.)
  if (scene_crowd_queued_.load(std::memory_order_relaxed)) {
    std::lock_guard lock(scene_crowd_mutex_);
    for (SceneCrowdRun& r : scene_crowd_queue_) scene_crowd_local_.push_back(std::move(r));
    scene_crowd_queue_.clear();
    scene_crowd_queued_.store(0, std::memory_order_relaxed);
  }
  if (scene_crowd_local_.empty()) {
    own_alone();
    return;
  }
  // (Review: predicated tiling replays the scene buffers per tile - the same first-member packet again in this
  // frame: the crowd pass stands down for the session.)
  // (Session 87, crowd2 review B3: in the D3D layer's direct path half2_packet_at_ is the TOKEN's address, equal for every draw
  // item of a token - the item's own address stands in: SceneCrowdItemAt.)
  // (Session 92, M4d: the packet id is the caller's - the item's arena address in the direct path, the packet's address in the stock one.)
  if (packet_id && packet_id == scene_crowd_matched_packet_ && counter_ == scene_crowd_matched_frame_) {
    if (!scene_crowd_blocked_.exchange(true)) {
      REXGPU_WARN("[scene][crowd] frame {}: a crowd draw packet executed twice (predicated tiling): the pass stands "
                  "down for this session", counter_);
    }
    st.dropped_local += scene_crowd_local_.size();
    scene_crowd_local_.clear();
    own_alone();
    return;
  }
  if (scene_crowd_local_.size() > 1024) {
    const size_t drop = scene_crowd_local_.size() - 768;
    scene_crowd_local_.erase(scene_crowd_local_.begin(), scene_crowd_local_.begin() + drop);
    st.dropped_local += drop;
  }
  // The run whose first member this draw is: among the oldest few (the recording threads' order may differ from the
  // replay's a little); the ones before it were never drawn (dropped).
  size_t found = SIZE_MAX;
  const size_t look = std::min<size_t>(scene_crowd_local_.size(), 32);
  for (size_t i = 0; i < look; ++i) {
    const uint32_t* f = scene_crowd_local_[i].first;
    if (f[0] == now[0] && std::equal(now, now + 12, f)) {
      found = i;
      break;
    }
  }
  if (found == SIZE_MAX) {
    own_alone();
    return;
  }
  // (Review: only the matched run - another recording thread's runs in front of it may still come; ageing drops the
  // ones that never do.)
  SceneCrowdRun run = std::move(scene_crowd_local_[found]);
  scene_crowd_local_.erase(scene_crowd_local_.begin() + found);
  scene_crowd_matched_packet_ = packet_id;
  scene_crowd_matched_frame_ = counter_;
  ++st.matched;
  if (g_crowd_census_unheld >= 0) {  // (the game drew this first member itself: nothing of ours held it)
    ++g_crowd_census[g_crowd_census_unheld].first;
    g_crowd_census_unheld = -1;
  }
  if (run.others.empty()) {
    own_alone();
    return;
  }  if (run.verify) {
    scene_crowd_verify_ = std::move(run);
    scene_crowd_verify_next_ = 0;
    scene_crowd_verify_active_ = true;
    return;
  }
  // wwe2k14 patch (Session 70, CROWD CULL): the members entirely outside this draw's view left out (mode 2) or only
  // counted (mode 1). (The registers then end on the last member drawn; the game's closing op-21 record, next in the
  // stream, writes the last member's matrix as before.)
  const std::vector<uint32_t>* list = &run.others;
  CrowdScratch<uint32_t, 0> kept_scratch;  // (Session 95, r7c4)
  std::vector<uint32_t>& kept = kept_scratch.get();
  if (const int32_t cull = REXCVAR_GET(gpu_scene_crowd_cull)) {
    if (SceneCrowdCull(run.others, kept) && cull == 2) list = &kept;
  }
  // wwe2k14 patch (Session 85, gpu_scene_crowd_density): of the members left, only those whose fixed rank (a hash of
  // the world position c76.w c77.w c78.w) is below the percentage; the run's first member too when our crowd renderer
  // holds it (else the game has drawn it already), so 0% is an empty arena.
  CrowdScratch<uint32_t, 1> dense_scratch;
  std::vector<uint32_t>& dense = dense_scratch.get();
  if (const int32_t density = REXCVAR_GET(gpu_scene_crowd_density); density < 100) {
    const auto rank = [](const uint32_t* w) { return CrowdRankOf(w[3], w[7], w[11]); };
    dense.reserve(list->size());
    const uint64_t rank_shader = active_vertex_shader() ? active_vertex_shader()->ucode_data_hash() : 0;
    for (size_t k = 0; k + 12 <= list->size(); k += 12) {
      const uint32_t* rank_w = list->data() + k;
      CrowdRankDiagNote(2, rank_shader, rank_w[3] * 0x9E3779B1u ^ rank_w[7] * 0x85EBCA77u ^ rank_w[11] * 0xC2B2AE3Du, rank(rank_w) >= density);
      if (rank(list->data() + k) < density) dense.insert(dense.end(), list->begin() + k, list->begin() + k + 12);
      else ++g_crowd_density_dropped;
    }
    list = &dense;
    if (first_guard.armed && rank(now) >= density) {
      first_guard.armed = false;
      ++g_crowd_density_dropped;
    }
  }
  // The omitted members: each one's c76-c78 written as its op-21 record would have left them (the library's flush
  // writes the whole group c76-c79; c79 is unchanged), then its draw as the packet's would be issued.
  const uint32_t members = uint32_t(list->size() / 12);
  if (!members) {
    own_alone();
    return;
  }
  // wwe2k14 patch (Session 80, OWN_RENDERER_PLAN.md, gpu_native_own_crowd): the other members drawn by our own crowd
  // renderer in one draw (the game's closing op-21 record, next in the stream, leaves c76-c78 as the loop would).
  if (first_guard.armed) {
    // The first member (its c76-c78 are the registers' now) and the others in one draw of ours.
    CrowdScratch<uint32_t, 2> all_scratch;
    std::vector<uint32_t>& all = all_scratch.get();
    all.assign(now, now + 12);
    all.insert(all.end(), list->begin(), list->end());
    if (OwnCrowdDraw(all.data(), members + 1)) {
      first_guard.armed = false;
      st.members += members;
      if (census >= 0) g_crowd_census[census].ours += members + 1, ++g_crowd_census[census].ours_calls;
      return;
    }
    first_guard.tried = true;
    note_decline(members + 1);
    first_guard.Draw();  // (before the others: they are appended as its instances)
  } else if (OwnCrowdDraw(list->data(), members)) {
    st.members += members;
    if (census >= 0) g_crowd_census[census].ours += members, ++g_crowd_census[census].ours_calls;
    return;
  } else {
    first_guard.tried = true;
    note_decline(members);
  }
  // wwe2k14 patch (Session 60, CROWD BATCH): one member through the draw dispatch (it starts a new instanced run: right
  // after a run's first draw its other constant uploads lie between, so no member can follow it as an instance), then
  // as many as fit straight as instances of that run, and again.
  const bool batch = REXCVAR_GET(gpu_scene_crowd_batch);
  // (Session 90, K2: the loop below writes each member's c76-c78 - into the image in a record-fed draw, else the register file - and
  // leaves the LAST member's matrix there. After the loop they are put back to `now` while the D3D feed is live: that makes the image
  // and the register file agree again at the end of this draw item, so the direct path can skip its register-file catch-up safely.
  // It is NOT what sets c76+ for later draws: scene_crowd.h's Flush records a closing op-21 with the LAST member's matrix right after
  // the run, and the state after the next item is that one.)
  bool wrote_registers = false;  // (Session 92, M4f-B: a member the old way drew wrote c76-c78 into the register file / the image)
  for (uint32_t m = 0; m < members;) {
    if (batch && m) {
      const uint32_t b = AppendCrowdInstances(list->data() + 12 * size_t(m), members - m);
      if (census >= 0) (first_guard.tried ? g_crowd_census[census].handed_back : g_crowd_census[census].other) += b;
      m += b;
      st.members += b;
      st.batched += b;
      if (m >= members) break;
    }
    bool ok = false;
    // (Session 92, M4f-B) A typed token draw's run: the member goes through the typed draw, with its 12 words in a copy of the typed state;
    // no register is written and nothing is put back.
    if (g_crowd_member_draw && !REXCVAR_GET(gpu_det_skip_host_draws) &&
        g_crowd_member_draw(g_crowd_member_ctx, list->data() + 12 * size_t(m), ok)) {
      ++st.members;
      st.failed += ok ? 0 : 1;
      if (census >= 0) ++(first_guard.tried ? g_crowd_census[census].handed_back : g_crowd_census[census].other);
      ++m;
      continue;
    }
    wrote_registers = true;
    for (uint32_t i = 0; i < 12; ++i) WriteRegister(kSceneCrowdRegister + i, (*list)[12 * size_t(m) + i]);
    D3dFeedCrowd().member_c76 = list->data() + 12 * size_t(m);  // (Session 92, M4d: for the typed shadow only; nothing here reads it)
    if (REXCVAR_GET(gpu_det_skip_host_draws)) {  // wwe2k14 patch (Session 77): see its definition
      ok = true;
    } else if (!IssueDrawLean(prim_type, index_count, index_buffer_info, major_mode_explicit, library_kind, ok)) {
      ok = IssueDraw(prim_type, index_count, index_buffer_info, major_mode_explicit);
    }
    D3dFeedCrowd().member_c76 = nullptr;
    ++st.members;
    st.failed += ok ? 0 : 1;
    if (census >= 0) ++(first_guard.tried ? g_crowd_census[census].handed_back : g_crowd_census[census].other);
    ++m;
  }
  // Put c76-c78 back to what the game set (same place the members were written). Without the feed they stay as the loop left them,
  // counted in g_scene_crowd_reg_writes as before (the direct path then still catches the register file up).
  if (g_crowd_member_draw && !wrote_registers) {
    // (Session 92, M4f-B: every member went through the typed draw: nothing was written, nothing to put back)
  } else if (feed_live) {
    D3dFeedCrowdState& crowd_state = D3dFeedCrowd();
    const bool on_image = D3dFeedDrawRegs() != nullptr;
    for (uint32_t i = 0; i < 12; ++i) WriteRegister(kSceneCrowdRegister + i, now[i]);
    ++(on_image ? crowd_state.restored_image : crowd_state.restored_regfile);
  } else {
    ++g_scene_crowd_reg_writes;  // (B3: the loop above wrote c76-c78 into the registers and nothing put them back)
    ++D3dFeedCrowd().unrestored_runs;
  }
}

// wwe2k14 patch (Session 70, CROWD CULL). The crowd vertex shaders (T_Audience / _LightMap in the entrances, ucode
// out\s60\shader_dump; the match's 0x183CC64A4C4E96AC, out\s70\shader_dump) place a vertex p of the member mesh (vf
// position, 32_32_32 float, offset 0) at world (c76 . p, c77 . p, c78 . p) and clip position c56..c59 x (world, 1) -
// nothing else moves it (no time, no skinning: the waving / signs are texture-side).
// So clip_i = M_i . p with M_i = sum_j c(56+i)[j] * c(76+j) + (0,0,0,c(56+i).w),
// and a member whose whole mesh box is on the outer side of one of the four side planes (w +- x, w +- y; with 1%
// more w: a margin for the host's pixel offsets) is clipped away entirely: drawing it changes no pixel. Judged only
// with clipping on and the usual viewport transform; the lit shaders only while c255 is (0, 1) (their cndeq passes the
// world position with w 1 then). Every draw judges with its own c56-c59, so every pass and camera is covered.
bool CommandProcessor::SceneCrowdCull(const std::vector<uint32_t>& members, std::vector<uint32_t>& kept) {
  SceneCrowdCullStats& cs = scene_crowd_cull_stats_;
  ++cs.runs;
  const RegisterFile& regs = (D3dFeedDrawRegs() ? *D3dFeedDrawRegs() : *register_file_);  // (Session 87, stage 3: the record's image in a fed draw)
  // (Session 92, M4f-C: a typed crowd run (the cut) reads the typed state's constants, clip / VTE words, vertex fetch and viewport scale - never the register file)
  const OwnedCrowdSource* const ts = OwnedCrowdSourceGet();
  const auto creg = [&](uint32_t index) -> uint32_t {
    if (ts) {
      if (index >= 0x4000 && index < 0x4400) return ts->vs_float[index - 0x4000];
      if (index == XE_GPU_REG_PA_CL_VPORT_XSCALE) return ts->vport_xscale;
    }
    return regs[index];
  };
  const Shader* vs = active_vertex_shader();
  // (The lit ones - entrances 0xC1028A64EFCF8229, the match 0x183CC64A4C4E96AC - pass the world position through
  // `cndeq r, c255.xxxy, world.yxzz, c255.yyyy`: the world position with w = c255.y while c255.x is 0; judged only
  // while c255 is (0, 1).)
  constexpr uint64_t kCrowdVs = 0xD5E2FCC9F76BC40Cull, kCrowdLitVs = 0xC1028A64EFCF8229ull,
                     kCrowdMatchVs = 0x183CC64A4C4E96ACull;
  const auto fbits = [](uint32_t v) {
    float f;
    std::memcpy(&f, &v, 4);
    return double(f);
  };
  constexpr uint32_t kVsConst = 0x4000;  // VS float constant c0 (SQ_VS_CONST base 0)
  const uint64_t vs_hash = vs ? vs->ucode_data_hash() : 0;
  const bool lit = vs_hash == kCrowdLitVs || vs_hash == kCrowdMatchVs;
  if (!vs || (vs_hash != kCrowdVs && !lit) ||
      (lit && (fbits(creg(kVsConst + 4 * 255)) != 0.0 || fbits(creg(kVsConst + 4 * 255 + 1)) != 1.0))) {
    ++cs.not_judged_shader;
    // (The other crowd shaders, once each: the next ones to map.)
    if (vs && scene_crowd_cull_unknown_.size() < 8 &&
        std::find(scene_crowd_cull_unknown_.begin(), scene_crowd_cull_unknown_.end(), vs->ucode_data_hash()) ==
            scene_crowd_cull_unknown_.end()) {
      scene_crowd_cull_unknown_.push_back(vs->ucode_data_hash());
      REXGPU_INFO("[scene][crowd] cull: vertex shader {:016X} not judged (not a mapped crowd shader)",
                  vs->ucode_data_hash());
    }
    return false;
  }
  reg::PA_CL_CLIP_CNTL clip;
  reg::PA_CL_VTE_CNTL vte;
  clip.value = ts ? ts->clip_cntl : regs[XE_GPU_REG_PA_CL_CLIP_CNTL];
  vte.value = ts ? ts->vte_cntl : regs[XE_GPU_REG_PA_CL_VTE_CNTL];
  if (clip.clip_disable || vte.vtx_xy_fmt || !vte.vtx_w0_fmt) {
    ++cs.not_judged_state;
    return false;
  }
  // The mesh's box: the position binding's whole buffer (every vertex an index can reach), read this frame.
  const Shader::VertexBinding* pos = nullptr;
  for (const Shader::VertexBinding& b : vs->vertex_bindings()) {
    for (const Shader::VertexBinding::Attribute& a : b.attributes) {
      if (a.fetch_instr.attributes.offset == 0 && a.fetch_instr.attributes.data_format == xenos::VertexFormat::k_32_32_32_FLOAT) {
        pos = &b;
      }
    }
  }
  if (!pos || pos->stride_words < 3) {
    ++cs.not_judged_mesh;
    return false;
  }
  xenos::xe_gpu_vertex_fetch_t fetch;
  if (ts) {
    std::memcpy(&fetch, ts->fetch + 2 * size_t(pos->fetch_constant), sizeof(fetch));
  } else {
    fetch = regs.GetVertexFetch(pos->fetch_constant);
  }
  const uint32_t base = uint32_t(fetch.address) << 2, bytes = uint32_t(fetch.size) << 2;
  const uint32_t stride = pos->stride_words * 4, endian = uint32_t(fetch.endian);
  const SceneCrowdMesh* mesh = nullptr;
  for (const SceneCrowdMesh& m : scene_crowd_meshes_) {
    if (m.base == base && m.bytes == bytes && m.stride == stride && m.endian == endian) mesh = &m;
  }
  if (!mesh) {
    SceneCrowdMesh m;
    m.base = base, m.bytes = bytes, m.stride = stride, m.endian = endian;
    const uint32_t count = stride ? bytes / stride : 0;
    // (Session 95, gpu_d3d_wall bit 32) The box of a mesh is its bytes': kept across frames under a hash of the bytes (the hash runs ~10x faster than the scan, and the list stays "this frame's").
    static std::unordered_map<uint64_t, SceneCrowdMesh> kept_boxes;  // (by the bytes' hash mixed with where they are; a miss only scans)
    const bool keep = (REXCVAR_GET(gpu_d3d_wall) & 32u) != 0 && count && bytes <= (64u << 20) && uint64_t(base) + bytes <= 0x20000000ull;
    uint64_t bytes_hash = 0;
    const SceneCrowdMesh* kept_mesh = nullptr;
    if (keep) {
      bytes_hash = XXH3_64bits(memory_->TranslatePhysical(base), bytes) ^ (uint64_t(base) << 32) ^ (uint64_t(stride) << 8) ^ endian;
      const auto kept_it = kept_boxes.find(bytes_hash);
      if (kept_it != kept_boxes.end() && kept_it->second.base == base && kept_it->second.bytes == bytes && kept_it->second.stride == stride && kept_it->second.endian == endian) {
        kept_mesh = &kept_it->second;
      }
    }
    if (kept_mesh) {
      m = *kept_mesh;
    } else if (count && bytes <= (64u << 20) && uint64_t(base) + bytes <= 0x20000000ull) {
      const uint8_t* p = memory_->TranslatePhysical(base);
      m.ok = true;
      for (int k = 0; k < 3; ++k) m.min[k] = INFINITY, m.max[k] = -INFINITY;
      for (uint32_t v = 0; v < count && m.ok; ++v) {
        for (int k = 0; k < 3; ++k) {
          uint32_t word;
          std::memcpy(&word, p + size_t(v) * stride + 4 * k, 4);
          word = xenos::GpuSwap(word, xenos::Endian(endian));
          float f;
          std::memcpy(&f, &word, 4);
          if (!std::isfinite(f)) {
            m.ok = false;
            break;
          }
          m.min[k] = std::min(m.min[k], f), m.max[k] = std::max(m.max[k], f);
        }
      }
    }
    if (keep && !kept_mesh) {
      if (kept_boxes.size() >= 8192) kept_boxes.clear();
      kept_boxes[bytes_hash] = m;
    }
    scene_crowd_meshes_.push_back(m);
    mesh = &scene_crowd_meshes_.back();
  }
  if (!mesh->ok) {
    ++cs.not_judged_mesh;
    return false;
  }
  ++cs.judged;
  double vp[4][4];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) vp[i][j] = fbits(creg(kVsConst + 4 * (56 + i) + j));
  const size_t n = members.size() / 12;
  kept.clear();
  kept.reserve(members.size());
  // (Session 79) gpu_scene_crowd_thin: the kept members' view depths.
  // wwe2k14 patch (Session 87, step 1 piece 2): the crowd's flat cards (kCrowdVs) are never thinned - a look change that
  // was approved for the 3D crowd, not for the cards (the exact outside-the-view drop and the density still apply).
  const int32_t thin = vs_hash == kCrowdVs ? 0 : REXCVAR_GET(gpu_scene_crowd_thin);
  CrowdScratch<double, 3> kept_w_scratch;
  std::vector<double>& kept_w = kept_w_scratch.get();
  const bool stable = thin && REXCVAR_GET(gpu_scene_crowd_thin_stable);
  if (thin && !stable) kept_w.reserve(n);
  constexpr double kMargin = 1.01;
  // Session 73 (Jarrod: in fill-the-screen "crowd members outside of the 16:9 box getting culled"): the scene draws see
  // clip x / gpu_wide_factor (y / gpu_tall_factor), so the visible area reaches that much further than the game's own
  // clip planes.
  const double wide = REXCVAR_GET(gpu_wide_factor), tall = REXCVAR_GET(gpu_tall_factor);
  const double margin[2] = {kMargin * (wide > 1.001 && wide < 8.0 ? wide : 1.0),
                            kMargin * (tall > 1.001 && tall < 8.0 ? tall : 1.0)};
  // Session 90 (stable thinning): the decision for every member of the run, before the cull - it depends on the member's own
  // view depth, the run's whole-list median (not on which other members survive the cull) and the member's own state.
  CrowdScratch<uint8_t, 8> thin_flag_scratch;
  std::vector<uint8_t>& thin_flag = thin_flag_scratch.get();
  if (stable) {
    constexpr double kBand = 0.08;
    const uint32_t keep = uint32_t(REXCVAR_GET(gpu_scene_crowd_thin_keep));
    CrowdScratch<double, 4> all_w_scratch;
    std::vector<double>& all_w = all_w_scratch.get();
    all_w.resize(n);
    CrowdScratch<double, 5> front_w_scratch;
    std::vector<double>& front_w = front_w_scratch.get();  // (only members in front of the camera: w <= 0 would pull the median to nothing)
    front_w.reserve(n);
    for (size_t m = 0; m < n; ++m) {
      const uint32_t* w = members.data() + 12 * m;
      double s = vp[3][3];
      for (int j = 0; j < 3; ++j) s += vp[3][j] * fbits(w[4 * j + 3]);
      all_w[m] = s;
      if (s > 0.0) front_w.push_back(s);
    }
    thin_flag.assign(n, 0);
    if (front_w.size() >= 8) {
      CrowdScratch<double, 6> sorted_scratch;
      std::vector<double>& sorted = sorted_scratch.get();
      sorted.assign(front_w.begin(), front_w.end());
      std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
      const double limit = sorted[sorted.size() / 2] * double(thin) / 100.0;
      if (limit > 0.0) {
      const uint64_t pass_key = (uint64_t(mesh->base) * 0x9E3779B97F4A7C15ull) ^
                                (uint64_t(creg(XE_GPU_REG_PA_CL_VPORT_XSCALE)) * 0xC2B2AE3D27D4EB4Full);
      for (size_t m = 0; m < n; ++m) {
        const uint32_t* w = members.data() + 12 * m;
        const uint32_t h = (w[3] * 0x9E3779B1u) ^ (w[7] * 0x85EBCA77u) ^ (w[11] * 0xC2B2AE3Du);
        if (!(keep > 1 && ((h >> 7) % keep) != 0)) continue;  // (not a thinning candidate: always drawn)
        const uint64_t key = pass_key ^ (uint64_t(h) * 0x165667B19E3779F9ull);
        const int prev = ThinStateGet(key);
        bool thinned;
        if (prev < 0) {
          thinned = all_w[m] > limit;
        } else if (prev) {
          thinned = !(all_w[m] < limit * (1.0 - kBand));
        } else {
          thinned = all_w[m] > limit * (1.0 + kBand);
        }
        ThinStateSet(key, prev, thinned);
        thin_flag[m] = thinned ? 1 : 0;
      }
      }
    }
  }
  for (size_t m = 0; m < n; ++m) {
    const uint32_t* w = members.data() + 12 * m;
    double M[4][4];
    bool finite = true;
    for (int i = 0; i < 4; ++i) {
      for (int k = 0; k < 4; ++k) {
        double s = k == 3 ? vp[i][3] : 0.0;
        for (int j = 0; j < 3; ++j) s += vp[i][j] * fbits(w[4 * j + k]);
        M[i][k] = s;
        finite = finite && std::isfinite(s);
      }
    }
    bool outside = false;
    for (int plane = 0; plane < 4 && finite && !outside; ++plane) {
      const int axis = plane >> 1;
      const double sign = (plane & 1) ? -1.0 : 1.0;
      double pmax = margin[axis] * M[3][3] + sign * M[axis][3];
      for (int k = 0; k < 3; ++k) {
        const double c = margin[axis] * M[3][k] + sign * M[axis][k];
        pmax += c * (c > 0 ? mesh->max[k] : mesh->min[k]);
      }
      outside = pmax < 0.0;
    }
    ++cs.members;
    const int kind = vs_hash == kCrowdVs ? 0 : (vs_hash == kCrowdLitVs ? 1 : 2);
    ++g_cull_kind_members[kind];
    if (outside) {
      ++g_cull_kind_outside[kind];
      if (REXCVAR_GET(gpu_scene_crowd_cull_diag) && finite) {
        double nx0 = 1e30, nx1 = -1e30, ny0 = 1e30, ny1 = -1e30;
        bool front = true;
        for (int c = 0; c < 8 && front; ++c) {
          double cc[4];
          for (int i = 0; i < 4; ++i)
            cc[i] = M[i][3] + M[i][0] * ((c & 1) ? mesh->max[0] : mesh->min[0]) +
                    M[i][1] * ((c & 2) ? mesh->max[1] : mesh->min[1]) + M[i][2] * ((c & 4) ? mesh->max[2] : mesh->min[2]);
          if (cc[3] <= 1e-6) {
            front = false;
            break;
          }
          nx0 = std::min(nx0, cc[0] / cc[3]), nx1 = std::max(nx1, cc[0] / cc[3]);
          ny0 = std::min(ny0, cc[1] / cc[3]), ny1 = std::max(ny1, cc[1] / cc[3]);
        }
        if (!front) {
          ++g_cull_diag_behind;
        } else if (nx1 >= -1.0 && nx0 <= 1.0 && ny1 >= -1.0 && ny0 <= 1.0) {
          ++g_cull_diag_overlap;
          if (g_cull_diag_lines++ < 60) {
            REXGPU_INFO("[scene][crowd] cull-diag: kind {} culled member {}/{} box ndc x {:.3f}..{:.3f} y {:.3f}..{:.3f}; "
                        "mesh min ({:.2f} {:.2f} {:.2f}) max ({:.2f} {:.2f} {:.2f}) bytes {} stride {} margin {:.3f}/{:.3f} "
                        "origin w {:.3f} ndc ({:.3f} {:.3f}) vp {}x{} c255 {} {} {}",
                        kind, m, n, nx0, nx1, ny0, ny1, mesh->min[0], mesh->min[1], mesh->min[2], mesh->max[0],
                        mesh->max[1], mesh->max[2], mesh->bytes, mesh->stride, margin[0], margin[1], M[3][3],
                        M[0][3] / M[3][3], M[1][3] / M[3][3], vp[0][0], vp[1][1], fbits(creg(kVsConst + 4 * 255)),
                        fbits(creg(kVsConst + 4 * 255 + 1)), fbits(creg(kVsConst + 4 * 255 + 2)));
          }
        }
      }
      ++cs.outside;
    } else if (stable && thin_flag[m]) {
      ++cs.thinned;
    } else {
      kept.insert(kept.end(), w, w + 12);
      if (thin && !stable) kept_w.push_back(M[3][3]);
    }
  }
  // wwe2k14 patch (Session 79, HD 520 arc look lever, gpu_scene_crowd_thin N): of the members farther than N/100 x the
  // run's median view depth (clip w at the member's origin), every other one left out - picked by a hash of its
  // position, so the same members stay from frame to frame. Not exact (a thinner distant crowd): off by default.
  if (thin && !stable && kept_w.size() >= 8) {
    CrowdScratch<double, 6> sorted_scratch;
    std::vector<double>& sorted = sorted_scratch.get();
    sorted.assign(kept_w.begin(), kept_w.end());
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    const double limit = sorted[sorted.size() / 2] * double(thin) / 100.0;
    const uint32_t keep = uint32_t(REXCVAR_GET(gpu_scene_crowd_thin_keep));  // (Session 82: 1 in `keep` of the far ones stays)
    CrowdScratch<uint32_t, 7> thinned_scratch;
    std::vector<uint32_t>& thinned = thinned_scratch.get();
    thinned.reserve(kept.size());
    for (size_t k = 0; k < kept_w.size(); ++k) {
      const uint32_t* w = kept.data() + 12 * k;
      const uint32_t h = (w[3] * 0x9E3779B1u) ^ (w[7] * 0x85EBCA77u) ^ (w[11] * 0xC2B2AE3Du);
      const bool cand = keep > 1 && ((h >> 7) % keep) != 0;
      const bool is_thin = cand && kept_w[k] > limit;
      if (cand) {  // (flip counting for the old rule: the same key as the stable one, members that stayed in view)
        const uint64_t key = (uint64_t(mesh->base) * 0x9E3779B97F4A7C15ull) ^
                             (uint64_t(creg(XE_GPU_REG_PA_CL_VPORT_XSCALE)) * 0xC2B2AE3D27D4EB4Full) ^
                             (uint64_t(h) * 0x165667B19E3779F9ull);
        ThinStateSet(key, ThinStateGet(key), is_thin);
      }
      if (is_thin) {
        ++cs.thinned;
        continue;
      }
      thinned.insert(thinned.end(), w, w + 12);
    }
    kept.swap(thinned);
  }
  return true;
}

// wwe2k14 patch (Session 93): the own crowd's window lines are counted from here - this runs every guest frame, also under TRUE NATIVE / the cut,
// where NativeRenderer::OnFrameEnd (the old caller) did not print them. Set by the Vulkan command processor.
void (*g_own_crowd_frame_hook)(CommandProcessor*) = nullptr;

void CommandProcessor::SceneCrowdFrameEnd() {
  CrowdRankDiagFrame();  // (Session 95)
  CrowdRankTrackFrame();
  {  // wwe2k14 patch (Session 90): the thin state's generations - the previous one dropped, the current one becomes it.
    static uint64_t last_swap = 0;
    if (counter_ - last_swap >= 600) {
      last_swap = counter_;
      g_thin_state_prev.swap(g_thin_state);
      g_thin_state.clear();
    }
  }
  scene_crowd_meshes_.clear();  // wwe2k14 patch (Session 70, CROWD CULL): the mesh boxes are this frame's
  SceneCrowdStats& st = scene_crowd_stats_;
  // (Review: ageing.) Runs pushed before the frame end 3 frame ends ago were for frames already drawn: dropped. The
  // game records up to ~2 frames ahead of the command processor.
  {
    std::lock_guard lock(scene_crowd_mutex_);
    for (SceneCrowdRun& r : scene_crowd_queue_) scene_crowd_local_.push_back(std::move(r));
    scene_crowd_queue_.clear();
    scene_crowd_queued_.store(0, std::memory_order_relaxed);
    const uint64_t oldest = scene_crowd_marks_[(scene_crowd_mark_index_ + 1) & 3];
    scene_crowd_marks_[scene_crowd_mark_index_ & 3] = scene_crowd_seq_;
    ++scene_crowd_mark_index_;
    while (!scene_crowd_local_.empty() && scene_crowd_local_.front().seq < oldest) {
      scene_crowd_local_.pop_front();
      ++st.dropped_local;
    }
  }
  if (g_own_crowd_frame_hook) g_own_crowd_frame_hook(this);
  const uint64_t frame = counter_;
  // wwe2k14 patch (Session 87, crowd2): the crowd census, one line per window of 600 guest frames with any crowd draw.
  if (const uint64_t w = frame / 600; w != g_crowd_census_window) {
    static const char* const kNames[3] = {"cards D5E2FCC9F76BC40C", "entrance C1028A64EFCF8229",
                                          "match 183CC64A4C4E96AC"};
    if (g_crowd_census_window != UINT64_MAX) {
      uint64_t game = 0;
      for (const CrowdCensus& c : g_crowd_census) game += c.first + c.stray + c.handed_back + c.other;
      bool any = false;
      for (const CrowdCensus& c : g_crowd_census) any = any || c.any();
      if (any) {
        std::string text;
        for (int i = 0; i < 3; ++i) {
          const CrowdCensus& c = g_crowd_census[i];
          if (!c.any()) continue;
          text += fmt::format(" | {}: draws {}, drawn by us {} members in {} calls, reached the game's draw {} (first "
                              "member {}, stray {}, handed back {}, other members {})",
                              kNames[i], c.draws, c.ours, c.ours_calls,
                              c.first + c.stray + c.handed_back + c.other, c.first, c.stray, c.handed_back, c.other);
        }
        REXGPU_INFO("[scene][crowd] census (gpu_native_own_crowd_all {}), frames {}-{}: reached the game's draw in all: "
                    "{}{}",
                    REXCVAR_GET(gpu_native_own_crowd_all) ? "on" : "off", g_crowd_census_window * 600,
                    g_crowd_census_window * 600 + 599, game, text);
      }
      for (CrowdCensus& c : g_crowd_census) c = CrowdCensus();
    }
    g_crowd_census_window = w;
  }
  if (st.first_frame == UINT64_MAX) st.first_frame = frame;
  if (frame - st.first_frame < 300) return;
  // (pushed / dropped change under the queue's mutex: the game's threads push - the whole reset under it, review.)
  std::lock_guard lock(scene_crowd_mutex_);
  st.pushed_window = st.pushed;
  st.dropped_window = st.dropped;
  if (st.pushed_window) {
    const double n = double(frame - st.first_frame);
    REXGPU_INFO("[scene][crowd] frames {}-{}, per frame: runs queued {:.1f}, matched {:.1f}, dropped {:.1f}; members "
                "drawn from the queue {:.1f} (batched as instances {:.1f}, failed {}); verify: {} members equal, "
                "{} differ",
                st.first_frame, frame, st.pushed_window / n, st.matched / n,
                (st.dropped_window + st.dropped_local) / n, st.members / n, st.batched / n,
                st.failed,
                st.verify_ok, st.verify_bad);
  }
  // wwe2k14 patch (Session 70, CROWD CULL): what the cull judged in this window.
  if (SceneCrowdCullStats& cs = scene_crowd_cull_stats_; cs.runs) {
    const double n = double(frame - st.first_frame);
    REXGPU_INFO("[scene][crowd] cull ({}): frames {}-{}, per frame: runs {:.1f} (judged {:.1f}; not judged: shader "
                "{:.1f}, clipping / viewport {:.1f}, mesh {:.1f}), members {:.1f}, outside the view {:.1f} ({:.1f}%), "
                "thinned (gpu_scene_crowd_thin) {:.1f}, queued members left out by gpu_scene_crowd_density {:.1f}",
                REXCVAR_GET(gpu_scene_crowd_cull) == 2 ? "not drawn" : "counted", st.first_frame, frame, cs.runs / n,
                cs.judged / n, cs.not_judged_shader / n, cs.not_judged_state / n, cs.not_judged_mesh / n,
                cs.members / n, cs.outside / n, cs.members ? 100.0 * double(cs.outside) / double(cs.members) : 0.0,
                cs.thinned / n, g_crowd_density_dropped / n);
    g_crowd_density_dropped = 0;
    REXGPU_INFO("[scene][crowd] cull per kind (per frame; members / outside): cards {:.1f}/{:.1f}, entrance {:.1f}/{:.1f}, "
                "match {:.1f}/{:.1f}; diag: culled yet box on screen {}, culled with a corner behind the camera {}",
                g_cull_kind_members[0] / n, g_cull_kind_outside[0] / n, g_cull_kind_members[1] / n,
                g_cull_kind_outside[1] / n, g_cull_kind_members[2] / n, g_cull_kind_outside[2] / n,
                g_cull_diag_overlap, g_cull_diag_behind);
    REXGPU_INFO("[scene][crowd] thin ({}): per frame: thinned {:.1f}, flips {:.2f}, saving {:.1f}% of the queued members",
                REXCVAR_GET(gpu_scene_crowd_thin_stable) ? "stable" : "old rule", cs.thinned / n, g_thin_flips / n,
                cs.members ? 100.0 * double(cs.thinned) / double(cs.members) : 0.0);
    REXGPU_INFO("[scene][crowd] thin state: {} keys (+{} in the previous generation), new keys per frame {:.2f}",
                g_thin_state.size(), g_thin_state_prev.size(), g_thin_new_keys / n);
    g_thin_flips = g_thin_new_keys = 0;
    for (int k = 0; k < 3; ++k) g_cull_kind_members[k] = g_cull_kind_outside[k] = 0;
    g_cull_diag_overlap = g_cull_diag_behind = 0;
    cs = SceneCrowdCullStats();
  }
  g_crowd_density_dropped = 0;  // (Session 85: every window, also with the cull off)
  st = SceneCrowdStats();
  st.first_frame = frame;
}

bool CommandProcessor::ExecutePacketType3_DRAW_INDX(memory::RingBuffer* reader, uint32_t packet,
                                                    uint32_t count) {
  // "initiate fetch of index buffer and draw"
  // Generally used by Xbox 360 Direct3D 9 for kDMA and kAutoIndex sources.
  // With a viz query token as the first one.
  uint32_t count_remaining = count;
  assert_not_zero(count_remaining);
  if (!count_remaining) {
    REXGPU_ERROR("PM4_DRAW_INDX: Packet too small, can't read the viz query token");
    return false;
  }
  uint32_t viz_query_condition = reader->ReadAndSwap<uint32_t>();
  --count_remaining;
  return ExecutePacketType3Draw(reader, packet, "PM4_DRAW_INDX", viz_query_condition,
                                count_remaining);
}

bool CommandProcessor::ExecutePacketType3_DRAW_INDX_2(memory::RingBuffer* reader, uint32_t packet,
                                                      uint32_t count) {
  // "draw using supplied indices in packet"
  // Generally used by Xbox 360 Direct3D 9 for kAutoIndex source.
  // No viz query token.
  return ExecutePacketType3Draw(reader, packet, "PM4_DRAW_INDX_2", 0, count);
}

bool CommandProcessor::ExecutePacketType3_SET_CONSTANT(memory::RingBuffer* reader, uint32_t packet,
                                                       uint32_t count) {
  // load constant into chip and to memory
  // PM4_REG(reg) ((0x4 << 16) | (GSL_HAL_SUBBLOCK_OFFSET(reg)))
  //                                     reg - 0x2000
  uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0x7FF;
  uint32_t type = (offset_type >> 16) & 0xFF;
  uint32_t count_registers = count - 1;
  // wwe2k14 patch (Session 48, HALF2): who writes the registers the library draws' shadow lacks.
  if (half2_ && type <= 4) {
    static constexpr uint32_t kBase[5] = {0x4000, 0x4800, 0x4900, 0x4908, 0x2000};
    Half2NoteWrite(kBase[type] + index, count_registers, PM4_SET_CONSTANT,
                   half2_packet_at_, 0);
  }
  switch (type) {
    case 0:  // ALU
      WriteALURangeFromRing(reader, index, count_registers);
      break;
    case 1:  // FETCH
      WriteFetchRangeFromRing(reader, index, count_registers);
      break;
    case 2:  // BOOL
      WriteBoolRangeFromRing(reader, index, count_registers);
      break;
    case 3:  // LOOP
      WriteLoopRangeFromRing(reader, index, count_registers);
      break;
    case 4:  // REGISTERS
      WriteREGISTERSRangeFromRing(reader, index, count_registers);
      break;
    default:
      assert_always();
      reader->AdvanceRead((count - 1) * sizeof(uint32_t));
      return true;
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_SET_CONSTANT2(memory::RingBuffer* reader, uint32_t packet,
                                                        uint32_t count) {
  uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0xFFFF;
  WriteRegisterRangeFromRing(reader, index, count - 1);
  return true;
}

bool CommandProcessor::ExecutePacketType3_LOAD_ALU_CONSTANT(memory::RingBuffer* reader,
                                                            uint32_t packet, uint32_t count) {
  // load constants from memory
  uint32_t address = reader->ReadAndSwap<uint32_t>();
  address &= 0x3FFFFFFF;
  uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0x7FF;
  uint32_t size_dwords = reader->ReadAndSwap<uint32_t>();
  size_dwords &= 0xFFF;
  uint32_t type = (offset_type >> 16) & 0xFF;
  uint32_t* xlat_address = memory_->TranslatePhysical<uint32_t*>(address);
  // wwe2k14 patch (Session 48, HALF2): who writes the registers the library draws' shadow lacks.
  if (half2_ && type <= 4) {
    static constexpr uint32_t kBase[5] = {0x4000, 0x4800, 0x4900, 0x4908, 0x2000};
    Half2NoteWrite(kBase[type] + index, size_dwords, PM4_LOAD_ALU_CONSTANT,
                   half2_packet_at_, address);
  }
  // wwe2k14 patch (Session 27.5, DRAWPROBE): constants loaded from guest memory since the last draw.
  if (draw_probe_frames_left_ > 0 && type == 0) {
    if (!draw_probe_loads_) {
      draw_probe_load_address_ = address;
      draw_probe_load_index_ = index;
      draw_probe_load_dwords_ = size_dwords;
    }
    ++draw_probe_loads_;
  }
  switch (type) {
    case 0:  // ALU
      WriteALURangeFromMem(index, xlat_address, size_dwords);
      break;
    case 1:  // FETCH
      WriteFetchRangeFromMem(index, xlat_address, size_dwords);
      break;
    case 2:  // BOOL
      WriteBoolRangeFromMem(index, xlat_address, size_dwords);
      break;
    case 3:  // LOOP
      WriteLoopRangeFromMem(index, xlat_address, size_dwords);
      break;
    case 4:  // REGISTERS
      WriteREGISTERSRangeFromMem(index, xlat_address, size_dwords);
      break;
    default:
      assert_always();
      return true;
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_SET_SHADER_CONSTANTS(memory::RingBuffer* reader,
                                                               uint32_t packet, uint32_t count) {
  uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0xFFFF;
  WriteRegisterRangeFromRing(reader, index, count - 1);
  return true;
}

bool CommandProcessor::ExecutePacketType3_IM_LOAD(memory::RingBuffer* reader, uint32_t packet,
                                                  uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

  // load sequencer instruction memory (pointer-based)
  uint32_t addr_type = reader->ReadAndSwap<uint32_t>();
  auto shader_type = static_cast<xenos::ShaderType>(addr_type & 0x3);
  uint32_t addr = addr_type & ~0x3;
  uint32_t start_size = reader->ReadAndSwap<uint32_t>();
  uint32_t start = start_size >> 16;
  uint32_t size_dwords = start_size & 0xFFFF;  // dwords
  assert_true(start == 0);

  auto shader =
      LoadShader(shader_type, addr, D3dShaderMicrocodePtrOf(memory_, addr), size_dwords);
  // wwe2k14 patch (Session 48, HALF2): the microcode's guest address, for the library-draw compare.
  half2_ucode_[shader_type == xenos::ShaderType::kVertex ? 0 : 1] = D3dShaderHalf2Key(addr);
  half2_shader_at_[shader_type == xenos::ShaderType::kVertex ? 0 : 1] = half2_packet_at_;  // (Session 51)
  switch (shader_type) {
    case xenos::ShaderType::kVertex:
      active_vertex_shader_ = shader;
      break;
    case xenos::ShaderType::kPixel:
      active_pixel_shader_ = shader;
      break;
    default:
      assert_unhandled_case(shader_type);
      return false;
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_IM_LOAD_IMMEDIATE(memory::RingBuffer* reader,
                                                            uint32_t packet, uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

  // load sequencer instruction memory (code embedded in packet)
  uint32_t dword0 = reader->ReadAndSwap<uint32_t>();
  uint32_t dword1 = reader->ReadAndSwap<uint32_t>();
  auto shader_type = static_cast<xenos::ShaderType>(dword0);
  uint32_t start_size = dword1;
  uint32_t start = start_size >> 16;
  uint32_t size_dwords = start_size & 0xFFFF;  // dwords
  assert_true(start == 0);
  assert_true(reader->read_count() >= size_dwords * 4);
  assert_true(count - 2 >= size_dwords);
  auto shader = LoadShader(shader_type, uint32_t(reader->read_ptr()),
                           reinterpret_cast<uint32_t*>(reader->read_ptr()), size_dwords);
  // wwe2k14 patch (Session 48, HALF2): an inline shader (the library's own clear / resolve shaders): no address.
  half2_ucode_[shader_type == xenos::ShaderType::kVertex ? 0 : 1] = 0;
  half2_shader_at_[shader_type == xenos::ShaderType::kVertex ? 0 : 1] = half2_packet_at_;  // (Session 51)
  if (shader_type == xenos::ShaderType::kVertex) {
    half2_inline_vs_at_ = half2_packet_at_;
    half2_inline_vs_caller_ = native_ib_depth_ >= 2 ? half2_ib_caller_ : 0;
  }
  switch (shader_type) {
    case xenos::ShaderType::kVertex:
      active_vertex_shader_ = shader;
      break;
    case xenos::ShaderType::kPixel:
      active_pixel_shader_ = shader;
      break;
    default:
      assert_unhandled_case(shader_type);
      return false;
  }
  reader->AdvanceRead(size_dwords * sizeof(uint32_t));
  return true;
}

// wwe2k14 patch (Session 87, D3D LAYER F1; docs/native/d3d/F1_CONTRACT.md): the record the game's indexed-draw hook
// (src/d3d_layer.h) writes in place of the library's state flush AND its draw packets. It is not PM4. Payload (count
// dwords, big-endian words as every packet): [0] magic 0xD3D10002, [1] the hook's sequence number, then items in the
// order the library's flush applies them:
//   RUN    [(1<<28)|n] [first register] [n values]  applied as a type-0 packet's registers are (ExecutePacketType0Body)
//   SHADER [(2<<28)|type] [physical address] [size in dwords]  type 0 vertex / 1 pixel: what IM_LOAD does with its words
//   DRAW   [(3<<28)] [base vertex] [viz query] [VGT_DRAW_INITIATOR] [DMA base] [DMA size]  the base vertex register
//          (0x2102) written as the library's chunk loop writes it, then the draw as DRAW_INDX does it after its header
// The record lives in the guest command memory, so the tiling worker's per-strip replays of a recorded segment execute
// it again (the sequence repeats: counted, not an error). Unpredicated. No sub-packet is forwarded or executed.
bool CommandProcessor::ExecutePacketType3_D3D_TOKEN(memory::RingBuffer* reader, uint32_t packet, uint32_t count) {
#if REX_TRUE_NATIVE_ONLY
  // (the TRUE NATIVE proof build: the token handler is out - the record is skipped, counted; under the cut the decoder executes tokens and none reaches here)
  {
    static uint64_t skipped = 0;
    if (++skipped == 1) REXGPU_ERROR("[d3d] a D3D token reached the packet path in the TRUE NATIVE proof build: skipped");
    reader->AdvanceRead(size_t(count) * sizeof(uint32_t));
    return true;
  }
#endif
  D3dCensus& c = g_d3d_census;
  d3d_tex::WriterScope tex_writer(d3d_tex::kWToken, nf_role_ != NfRole::kAnswers);  // wwe2k14 patch (Session 87, stage 4): who writes fetch constants
  const bool answers = nf_role_ == NfRole::kAnswers;
  const bool counted = !answers;  // (both sides run it; the drawing side's count is the census)
  const size_t payload_end =
      (reader->read_offset() + size_t(count) * sizeof(uint32_t)) % size_t(reader->capacity());
  auto fail = [&](const char* what, uint32_t a, uint32_t b) {
    REXGPU_ERROR("[d3d] record framing error: {} ({:08X}, {:08X}); the rest of the record ({} dwords) is skipped", what, a,
                 b, count);
    c.bad.fetch_add(1, std::memory_order_relaxed);
    reader->set_read_offset(payload_end);
    return true;
  };
  if (count < 2) return fail("short record", count, 0);
  const uint32_t magic = reader->ReadAndSwap<uint32_t>();
  const uint32_t seq = reader->ReadAndSwap<uint32_t>();
  if (magic != kD3dTokenMagic) return fail("bad magic", magic, seq);
  // (Session 87 review item 11: F4's benign-write latch from the record's first item on - its runs and shader loads are noted
  // under the same rule its draw item uses, not under the previous draw's.)
  if (counted && REXCVAR_GET(gpu_d3d_direct)) repeat_benign_ = true;
  if (counted) {
    c.tokens.fetch_add(1, std::memory_order_relaxed);
    c.bytes.fetch_add(uint64_t(count) * sizeof(uint32_t), std::memory_order_relaxed);
    const uint32_t last = c.last_seq.exchange(seq, std::memory_order_relaxed);
    if (last && int32_t(seq - last) <= 0) c.replays.fetch_add(1, std::memory_order_relaxed);
  }
  uint32_t left = count - 2;
  while (left) {
    const uint32_t header = reader->ReadAndSwap<uint32_t>();
    const uint32_t kind = header >> 28;
    uint32_t used = 0;  // dwords of the item, header included
    if (kind == 1) {
      const uint32_t n = header & 0x0FFFFFFF;
      used = 2 + n;
      if (n == 0 || used > left) return fail("run past the end", header, left);
      const uint32_t first = reader->ReadAndSwap<uint32_t>();
      // wwe2k14 patch (Session 87, D3D layer stage 3): gpu_d3d_own_input - the run goes into the record's register image
      // (no register-file write); false = the stock type-0 path.
      if (!D3dFeedRun(*this, reader, first, n) && !ExecutePacketType0Body(reader, first, n, 0)) return false;
      if (counted) {
        c.runs.fetch_add(1, std::memory_order_relaxed);
        c.run_registers.fetch_add(n, std::memory_order_relaxed);
      }
    } else if (kind == 2) {
      used = 3;
      if (used > left) return fail("shader item past the end", header, left);
      const uint32_t type = header & 0xF;
      const uint32_t addr = reader->ReadAndSwap<uint32_t>() & ~0x3u;
      const uint32_t size_dwords = reader->ReadAndSwap<uint32_t>() & 0xFFFF;
      if (type > 1) return fail("shader type", header, addr);
      if (!answers) {  // (the answers side skips IM_LOAD, as in ExecutePacketType3)
        SCOPE_profile_cpu_f("gpu");
        const auto shader_type = type == 0 ? xenos::ShaderType::kVertex : xenos::ShaderType::kPixel;
        auto shader = LoadShader(shader_type, addr, D3dShaderMicrocodePtrOf(memory_, addr), size_dwords);
        half2_ucode_[type] = D3dShaderHalf2Key(addr);  // (as ExecutePacketType3_IM_LOAD)
        half2_shader_at_[type] = half2_packet_at_;
        if (type == 0) {
          active_vertex_shader_ = shader;
        } else {
          active_pixel_shader_ = shader;
        }
        RepeatNotePacket(PM4_IM_LOAD);
        if (counted) c.shaders.fetch_add(1, std::memory_order_relaxed);
      }
    } else if (kind == 3) {
      used = 6;
      if (used > left) return fail("draw item past the end", header, left);
      // (the chunk's base vertex; Session 87 stage 3: into the record's image like any run)
      if (!D3dFeedRun(*this, reader, XE_GPU_REG_VGT_INDX_OFFSET, 1) &&
          !ExecutePacketType0Body(reader, XE_GPU_REG_VGT_INDX_OFFSET, 1, 0)) {
        return false;
      }
      if (answers) {
        reader->AdvanceRead(4 * sizeof(uint32_t));  // (the answers side skips DRAW_INDX)
      } else {
        // wwe2k14 patch (Session 87, D3D LAYER F4): gpu_d3d_direct - the backend's handler takes the item straight (the
        // library-draw kind and the dirty classes come from the record); false = the stock draw packet path.
        if (!D3dDirectDrawItem(*this, reader) && !ExecutePacketType3_DRAW_INDX(reader, 0xC0032200u, 4)) return false;
        if (counted) c.draws.fetch_add(1, std::memory_order_relaxed);
      }
    } else {
      return fail("unknown item", header, left);
    }
    left -= used;
  }
  if (reader->read_offset() != payload_end) {
    return fail("items did not end on the record's end", uint32_t(reader->read_offset()), uint32_t(payload_end));
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_INVALIDATE_STATE(memory::RingBuffer* reader,
                                                           uint32_t packet, uint32_t count) {
  // selective invalidation of state pointers
  /*uint32_t mask =*/reader->ReadAndSwap<uint32_t>();
  // driver_->InvalidateState(mask);
  return true;
}

bool CommandProcessor::ExecutePacketType3_VIZ_QUERY(memory::RingBuffer* reader, uint32_t packet,
                                                    uint32_t count) {
  // begin/end initiator for viz query extent processing
  // https://www.google.com/patents/US20050195186
  assert_true(count == 1);

  uint32_t dword0 = reader->ReadAndSwap<uint32_t>();

  uint32_t id = dword0 & 0x3F;
  uint32_t end = dword0 & 0x100;
  if (!end) {
    // begin a new viz query @ id
    // On hardware this clears the internal state of the scan converter (which
    // is different to the register)
    WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, VIZQUERY_START);
    REXGPU_INFO("Begin viz query ID {:02X}", id);
  } else {
    // end the viz query
    WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, VIZQUERY_END);
    REXGPU_INFO("End viz query ID {:02X}", id);
    // The scan converter writes the internal result back to the register here.
    // We just fake it and say it was visible in case it is read back.
    if (id < 32) {
      register_file_->values[XE_GPU_REG_PA_SC_VIZ_QUERY_STATUS_0] |= uint32_t(1) << id;
    } else {
      register_file_->values[XE_GPU_REG_PA_SC_VIZ_QUERY_STATUS_1] |= uint32_t(1) << (id - 32);
    }
  }

  return true;
}

// wwe2k14 patch (Session 6, COUNTERS): per-frame counters and the frame-anatomy capture. Measurement only:
// nothing here changes what is emulated or drawn.

void CommandProcessor::RecordFrameCounters() {
  FrameCounters& counters = frame_counters_;
  CollectFrameCounters(counters);
  if (perf::timeline::Enabled()) {
    counters.processor_number = perf::CurrentProcessorNumber();
    // Fixed layout (a contract with the report tool; don't rename or reorder):
    //   cpf      a = IssueDraw calls (including the ones handled as copies), b = draws that reached a host draw
    //            call, c = IssueCopy (resolve) calls
    //   cpf_t0   a = type-0 packets, b = registers written by them, c = of those, float constants 0x4000-0x47FF
    //   cpf_t0x  a = type-0 registers in fetch constants 0x4800-0x48FF, b = in bool/loop constants 0x4900-0x4927,
    //            c = type-3 packets
    //   cpf_rr   a = SharedMemory::RequestRange(s) calls, b = of those, calls that uploaded nothing, c = KiB
    //            uploaded by UploadRanges
    //   cpf_bind a = texture descriptor sets written (vertex + pixel), b = constants descriptor sets written,
    //            c = GetSamplerParameters calls
    //   cpf_fc   a = TextureFetchConstantWritten notifications (per fetch constant), b = of those, where the 6
    //            dwords differ from the last value seen for that fetch constant, c = graphics pipeline binds that
    //            changed the bound pipeline
    //   cpf_cpu  a = processor number of this thread now, b = next crash-checkpoint record id (0 if crash
    //            diagnostics are off), c = submissions ended
    //   cpf_tile a = EVENT_WRITE_EXT packets executed (not skipped by predication), b = predicated type-3
    //            packets skipped by the bin test ((BIN_SELECT & BIN_MASK) == 0), c = predicated DRAW_INDX /
    //            DRAW_INDX_2 packets executed (Session 7, TILING-STAGE0)
    //   cpf_bin  a = SET_BIN_SELECT_LO/_HI/SET_BIN_SELECT packets executed, b = SET_BIN_MASK_LO/_HI/SET_BIN_MASK
    //            packets executed, c = distinct 64-bit BIN_SELECT values that executed predicated packets ran
    //            under (17 = more than 16; a count, not a sum: per frame only) (Session 7, TILING-STAGE0)
    //   cpf_gm   a = GPU timestamp marks recorded, b = marks not recorded (a frame's query slot full), c = marked
    //            frames whose results were written to the _gpu CSVs (Session 7b, GPUMARKS; only recorded when
    //            one of them is not 0, that is around a frame-anatomy capture with gpu_anatomy_gpu_marks)
    // Counts cover the command processor's work between two cp_swap events (the swap's own submission is
    // counted in the next frame). The type-0 split and the fetch constant comparison only count while the
    // timeline is on (latched at the previous swap).
    perf::timeline::Record("cpf", counters.issue_draw, counters.draws_issued, counters.issue_copy);
    perf::timeline::Record("cpf_t0", counters.type0_packets, counters.type0_registers,
                           counters.type0_float_constants);
    perf::timeline::Record("cpf_t0x", counters.type0_fetch_constants, counters.type0_bool_loop_constants,
                           counters.type3_packets);
    perf::timeline::Record("cpf_rr", counters.request_ranges, counters.request_ranges_all_valid,
                           counters.upload_kib);
    perf::timeline::Record("cpf_bind", counters.texture_sets_written, counters.constants_sets_written,
                           counters.sampler_parameters);
    perf::timeline::Record("cpf_fc", counters.fetch_constant_notifications, counters.fetch_constant_changed,
                           counters.pipeline_binds_changed);
    perf::timeline::Record("cpf_cpu", counters.processor_number, counters.checkpoint_next_id,
                           counters.submissions_ended);
    // wwe2k14 patch (Session 7, TILING-STAGE0).
    const TilingFrameCounters& tiling = tiling_counters_;
    perf::timeline::Record("cpf_tile", tiling.event_write_ext, tiling.predicated_skipped,
                           tiling.predicated_draws);
    perf::timeline::Record("cpf_bin", tiling.bin_select_packets, tiling.bin_mask_packets,
                           tiling.bin_select_distinct);
    // wwe2k14 patch (Session 7b, GPUMARKS).
    if (counters.gpu_marks || counters.gpu_marks_dropped || counters.gpu_marks_frames_written) {
      perf::timeline::Record("cpf_gm", counters.gpu_marks, counters.gpu_marks_dropped,
                             counters.gpu_marks_frames_written);
    }
  }
  counters = FrameCounters();
  frame_counters_enabled_ = perf::timeline::Enabled();
  // wwe2k14 patch (Session 7, TILING-STAGE0).
  tiling_counters_ = TilingFrameCounters();
  tiling_select_counted_ = false;
}

// wwe2k14 patch (Session 7, TILING-STAGE0): cpf_bin c. Called for the first executed predicated packet after
// bin_select_ may have changed (a SET_BIN_SELECT* packet or a frame boundary), so at most a few times per tile.
void CommandProcessor::CountTilingSelect() {
  tiling_select_counted_ = true;
  TilingFrameCounters& tiling = tiling_counters_;
  const uint32_t stored = std::min(tiling.bin_select_distinct, kTilingSelectValuesMax);
  for (uint32_t i = 0; i < stored; ++i) {
    if (tiling.bin_select_values[i] == bin_select_) {
      return;
    }
  }
  if (tiling.bin_select_distinct < kTilingSelectValuesMax) {
    tiling.bin_select_values[tiling.bin_select_distinct++] = bin_select_;
  } else {
    // Full: any value not stored makes it "more than kTilingSelectValuesMax".
    tiling.bin_select_distinct = kTilingSelectValuesMax + 1;
  }
}

namespace {

int64_t AnatomyNowNs() {
  return int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now().time_since_epoch())
                     .count());
}

std::FILE* AnatomyOpenFile(const std::filesystem::path& path) {
#ifdef _WIN32
  // Shared, so the file can be read while the capture runs.
  return _wfsopen(path.c_str(), L"wb", _SH_DENYNO);
#else
  return std::fopen(path.c_str(), "wb");
#endif
}

void AnatomyShaderInfo(const Shader* shader, uint64_t& hash, uint32_t& textures, uint32_t& float_constants) {
  hash = 0;
  textures = 0;
  float_constants = 0;
  if (!shader) {
    return;
  }
  hash = shader->ucode_data_hash();
  // Analysis happens in IssueDraw; a shader the backend didn't need (for instance the pixel shader of a
  // depth-only draw) may not be analyzed.
  if (!shader->is_ucode_analyzed()) {
    return;
  }
  for (const Shader::TextureBinding& binding : shader->texture_bindings()) {
    textures |= uint32_t(1) << (binding.fetch_constant & 31);
  }
  float_constants = shader->constant_register_map().float_count;
}

void AnatomyAppendHash(fmt::memory_buffer& text, uint64_t hash) {
  if (hash) {
    fmt::format_to(std::back_inserter(text), "{:016X}", hash);
  } else {
    text.push_back('0');
  }
}

}  // namespace

void CommandProcessor::AnatomyStartFrame() {
  anatomy_frame_start_pending_ = false;
  if (anatomy_) {
    anatomy_->frame_start_ns = AnatomyNowNs();
    anatomy_->frame_started = true;
  }
}

void CommandProcessor::AnatomyBeginDraw() {
  AnatomyCapture& capture = *anatomy_;
  capture.draws_issued_before = frame_counters_.draws_issued;
  capture.copies_before = frame_counters_.issue_copy;
  capture.pipeline_binds_before = frame_counters_.pipeline_binds_changed;
  anatomy_skip_reason_ = kAnatomySkipOther;
  // wwe2k14 patch (Session 7b, GPUMARKS): the row index of this call's GPU marks.
  anatomy_gpu_row_ = uint32_t(capture.rows.size());
  anatomy_in_draw_ = true;  // wwe2k14 patch (Session 7c, CAUSES)
  capture.draw_start_ns = AnatomyNowNs();
}

void CommandProcessor::AnatomyEndDraw(uint32_t vgt_draw_initiator_value, bool is_indexed,
                                      bool succeeded, bool predicated) {
  const int64_t end_ns = AnatomyNowNs();
  AnatomyCapture& capture = *anatomy_;
  anatomy_in_draw_ = false;  // wwe2k14 patch (Session 7c, CAUSES)
  if (capture.rows.size() >= AnatomyCapture::kMaxRowsPerFrame) {
    ++capture.rows_dropped;
    AnatomyRowCauses(nullptr);  // wwe2k14 patch (Session 7c, CAUSES): not carried over to a later row
    return;
  }
  AnatomyCapture::Row& row = capture.rows.emplace_back();
  anatomy_gpu_row_ = uint32_t(capture.rows.size());  // wwe2k14 patch (Session 7b, GPUMARKS): the next row
  // wwe2k14 patch (Session 7c, CAUSES): what the backend recorded for this row (also the row's end for the
  // sparse GPU marks).
  row.causes_valid = AnatomyRowCauses(&row.causes);
  row.start_ns = capture.frame_started ? capture.draw_start_ns - capture.frame_start_ns : 0;
  row.cp_ns = end_ns - capture.draw_start_ns;
  // What happened is read from the counters the backend increments (draws_issued at its host draw call,
  // issue_copy in IssueCopy, pipeline_binds_changed when it binds a different pipeline).
  row.is_copy = frame_counters_.issue_copy != capture.copies_before;
  const bool drew = frame_counters_.draws_issued != capture.draws_issued_before;
  row.pipeline_changed = frame_counters_.pipeline_binds_changed != capture.pipeline_binds_before;
  if (drew || (row.is_copy && succeeded)) {
    row.skipped = kAnatomyDrew;
  } else if (!succeeded) {
    row.skipped = kAnatomyFailed;
  } else {
    row.skipped = anatomy_skip_reason_;
  }
  reg::VGT_DRAW_INITIATOR vgt_draw_initiator;
  vgt_draw_initiator.value = vgt_draw_initiator_value;
  row.prim = uint32_t(vgt_draw_initiator.prim_type);
  row.count = vgt_draw_initiator.num_indices;
  row.indexed = is_indexed ? 1 : 0;
  row.index_format =
      is_indexed ? (vgt_draw_initiator.index_size == xenos::IndexFormat::kInt32 ? 32 : 16) : 0;
  const RegisterFile& regs = *register_file_;
  row.rb_surface_info = regs[XE_GPU_REG_RB_SURFACE_INFO];
  row.rb_color_info[0] = regs[XE_GPU_REG_RB_COLOR_INFO];
  row.rb_color_info[1] = regs[XE_GPU_REG_RB_COLOR1_INFO];
  row.rb_color_info[2] = regs[XE_GPU_REG_RB_COLOR2_INFO];
  row.rb_color_info[3] = regs[XE_GPU_REG_RB_COLOR3_INFO];
  row.rb_depth_info = regs[XE_GPU_REG_RB_DEPTH_INFO];
  row.rb_modecontrol = regs[XE_GPU_REG_RB_MODECONTROL];
  row.rb_depthcontrol = regs[XE_GPU_REG_RB_DEPTHCONTROL];
  row.rb_colorcontrol = regs[XE_GPU_REG_RB_COLORCONTROL];
  row.rb_color_mask = regs[XE_GPU_REG_RB_COLOR_MASK];
  // wwe2k14 patch (Session 7, TILING-STAGE0): predication and the tile pass's window, scissor and viewport.
  row.predicated = predicated;
  row.bin_select_lo = uint32_t(bin_select_);
  row.bin_mask_lo = uint32_t(bin_mask_);
  row.pa_sc_window_offset = regs[XE_GPU_REG_PA_SC_WINDOW_OFFSET];
  row.pa_sc_window_scissor_tl = regs[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL];
  row.pa_sc_window_scissor_br = regs[XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR];
  row.pa_sc_screen_scissor_tl = regs[XE_GPU_REG_PA_SC_SCREEN_SCISSOR_TL];
  row.pa_sc_screen_scissor_br = regs[XE_GPU_REG_PA_SC_SCREEN_SCISSOR_BR];
  row.pa_cl_vport_xscale = regs[XE_GPU_REG_PA_CL_VPORT_XSCALE];
  row.pa_cl_vport_xoffset = regs[XE_GPU_REG_PA_CL_VPORT_XOFFSET];
  row.pa_cl_vport_yscale = regs[XE_GPU_REG_PA_CL_VPORT_YSCALE];
  row.pa_cl_vport_yoffset = regs[XE_GPU_REG_PA_CL_VPORT_YOFFSET];
  if (row.is_copy) {
    // A resolve runs no guest shaders.
    row.copy_dest = regs[XE_GPU_REG_RB_COPY_DEST_BASE];
    row.copy_info = regs[XE_GPU_REG_RB_COPY_DEST_INFO];
    // wwe2k14 patch (Session 42, POSTMAP).
    row.copy_control = regs[XE_GPU_REG_RB_COPY_CONTROL];
    row.copy_dest_pitch = regs[XE_GPU_REG_RB_COPY_DEST_PITCH];
  } else {
    // The shaders the guest bound for the draw.
    AnatomyShaderInfo(active_vertex_shader_, row.vs_hash, row.vs_textures, row.vs_float_constants);
    AnatomyShaderInfo(active_pixel_shader_, row.ps_hash, row.ps_textures, row.ps_float_constants);
    // wwe2k14 patch (Session 42, POSTMAP): the fetch constants behind the used texture slots.
    uint32_t slots = row.vs_textures | row.ps_textures;
    row.tex_count = 0;
    while (slots && row.tex_count < 8) {
      const uint32_t slot = uint32_t(std::countr_zero(slots));
      slots &= slots - 1;
      const xenos::xe_gpu_texture_fetch_t fetch = regs.GetTextureFetch(slot);
      AnatomyCapture::Row::Tex& t = row.tex[row.tex_count++];
      t.slot = uint8_t(slot);
      t.format = uint8_t(fetch.format);
      t.dimension = uint8_t(fetch.dimension);
      t.tiled = uint8_t(fetch.tiled);
      t.base = fetch.base_address << 12;
      t.width = uint16_t(fetch.size_2d.width + 1);
      t.height = uint16_t(fetch.size_2d.height + 1);
    }
  }
  // wwe2k14 patch (Session 19, DRAWRUNS).
  {
    const void* shaders[2] = {active_vertex_shader_, active_pixel_shader_};
    XXH3_state_t* state = XXH3_createState();
    XXH3_64bits_reset(state);
    XXH3_64bits_update(state, &regs.values[0x2000], sizeof(uint32_t) * 0x2000);
    XXH3_64bits_update(state, shaders, sizeof(shaders));
    row.state_hash = XXH3_64bits_digest(state);
    XXH3_freeState(state);
    row.fetch_hash = XXH3_64bits(&regs.values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0], sizeof(uint32_t) * 0xC0);
  }
}

void CommandProcessor::AnatomyEndFrame() { anatomy_->frame_end_ns = AnatomyNowNs(); }

void CommandProcessor::AnatomyFrameBoundary() {
  if (anatomy_active_) {
    // Write the frame that just ended.
    AnatomyCapture& capture = *anatomy_;
    fmt::memory_buffer& text = capture.text;
    text.clear();
    uint32_t draws = 0, copies = 0;
    int64_t cp_ns_sum = 0;
    for (size_t i = 0; i < capture.rows.size(); ++i) {
      const AnatomyCapture::Row& row = capture.rows[i];
      ++(row.is_copy ? copies : draws);
      cp_ns_sum += row.cp_ns;
      fmt::format_to(std::back_inserter(text), "{},{},{},{:.1f},{:.1f},", capture.frame, i,
                     row.is_copy ? "copy" : "draw", double(row.start_ns) / 1000.0,
                     double(row.cp_ns) / 1000.0);
      AnatomyAppendHash(text, row.vs_hash);
      text.push_back(',');
      AnatomyAppendHash(text, row.ps_hash);
      fmt::format_to(std::back_inserter(text),
                     ",{},{},{},{},0x{:08X},0x{:08X},0x{:08X},0x{:08X},0x{:08X},0x{:08X},0x{:08X},0x{:08X},"
                     "0x{:08X},0x{:08X},0x{:08X},0x{:08X},{},{},{},{},",
                     row.prim, row.count, row.indexed, row.index_format, row.rb_surface_info,
                     row.rb_color_info[0], row.rb_color_info[1], row.rb_color_info[2], row.rb_color_info[3],
                     row.rb_depth_info, row.rb_modecontrol, row.rb_depthcontrol, row.rb_colorcontrol,
                     row.rb_color_mask, row.vs_textures, row.ps_textures, row.vs_float_constants,
                     row.ps_float_constants, row.pipeline_changed, row.skipped);
      // wwe2k14 patch (Session 7, TILING-STAGE0): the predication and window columns follow copy_info.
      if (row.is_copy) {
        fmt::format_to(std::back_inserter(text), "0x{:08X},0x{:08X},", row.copy_dest, row.copy_info);
      } else {
        fmt::format_to(std::back_inserter(text), ",,");
      }
      fmt::format_to(std::back_inserter(text),
                     "0x{:08X},0x{:08X},{},0x{:08X},0x{:08X},0x{:08X},0x{:08X},0x{:08X},0x{:08X},0x{:08X},"
                     "0x{:08X},0x{:08X},",
                     row.bin_select_lo, row.bin_mask_lo, row.predicated ? 1 : 0, row.pa_sc_window_offset,
                     row.pa_sc_window_scissor_tl, row.pa_sc_window_scissor_br, row.pa_sc_screen_scissor_tl,
                     row.pa_sc_screen_scissor_br, row.pa_cl_vport_xscale, row.pa_cl_vport_xoffset,
                     row.pa_cl_vport_yscale, row.pa_cl_vport_yoffset);
      // wwe2k14 patch (Session 7, EXTENTS-SHADOW): ext_minx,ext_maxx,ext_miny,ext_maxy,ext_ok,ext_why,ext_us, after
      // the TILING-STAGE0 columns.
      if (row.ext_attempted) {
        fmt::format_to(std::back_inserter(text), "{},{},{},{},{},{},{:.1f}", row.ext_min_x, row.ext_max_x,
                       row.ext_min_y, row.ext_max_y, row.ext_ok ? 1 : 0,
                       row.ext_ok ? "" : GetScreenExtentsFallbackName(ScreenExtentsFallback(row.ext_fallback)),
                       row.ext_us);
      } else {
        fmt::format_to(std::back_inserter(text), ",,,,,,");
      }
      // wwe2k14 patch (Session 7c, CAUSES): the cause columns (AnatomyCauses, in order) end the row.
      if (row.causes_valid) {
        const AnatomyCauses& causes = row.causes;
        fmt::format_to(std::back_inserter(text), ",{},{},{},{},{},{},{},0x{:08X},0x{:08X},{},{},{},{},{}",
                       causes.up_calls, causes.up_pages, causes.tex_loads, causes.tex_loads_scaled,
                       causes.tex_load_kib, causes.dispatches, causes.barrier_batches, causes.barrier_src_stages,
                       causes.barrier_dst_stages, causes.rendering_restarts, causes.fsi_edram_barriers,
                       causes.desc_writes, causes.pipeline_created, causes.pipeline_pending);
      } else {
        fmt::format_to(std::back_inserter(text), ",,,,,,,,,,,,,,");
      }
      // wwe2k14 patch (Session 19, DRAWRUNS): state_hash,fetch_hash end the row.
      fmt::format_to(std::back_inserter(text), ",{:016X},{:016X}", row.state_hash, row.fetch_hash);
      // wwe2k14 patch (Session 42, POSTMAP): copy_control,copy_pitch,tex end the row; tex = space-separated
      // slot:base:format:WxH:dimension:tiled.
      if (row.is_copy) {
        fmt::format_to(std::back_inserter(text), ",0x{:08X},0x{:08X},", row.copy_control, row.copy_dest_pitch);
      } else {
        fmt::format_to(std::back_inserter(text), ",,,");
        for (uint32_t t = 0; t < row.tex_count; ++t) {
          const AnatomyCapture::Row::Tex& tex = row.tex[t];
          fmt::format_to(std::back_inserter(text), "{}{}:0x{:08X}:{}:{}x{}:{}:{}", t ? " " : "", tex.slot, tex.base,
                         tex.format, tex.width, tex.height, tex.dimension, tex.tiled);
        }
      }
      text.push_back('\n');
    }
    bool write_ok = true;
    if (text.size()) {
      write_ok = std::fwrite(text.data(), 1, text.size(), capture.rows_file) == text.size();
    }
    const int64_t span_ns = capture.frame_started ? capture.frame_end_ns - capture.frame_start_ns : 0;
    write_ok = std::fprintf(capture.frames_file, "%u,%u,%u,%.1f,%.1f\n", capture.frame, draws, copies,
                            double(cp_ns_sum) / 1000.0, double(span_ns) / 1000.0) > 0 &&
               write_ok;
    // Ties the capture's frames to the timeline's cp_swap events.
    perf::timeline::Record("anatomy", capture.captures_started - 1, capture.frame,
                           uint32_t(capture.rows.size()));
    capture.rows_written += capture.rows.size();
    capture.rows.clear();
    capture.frame_started = false;
    ++capture.frame;
    if (!write_ok) {
      AnatomyFinish("write error");
      return;
    }
    if (capture.frame >= uint32_t(capture.requested)) {
      AnatomyFinish("done");
      return;
    }
    anatomy_frame_start_pending_ = true;
    // wwe2k14 patch (Session 7b, GPUMARKS): the next capture frame's GPU marks (anatomy_gpu_row_ is still the
    // ended frame's row count during the call).
    AnatomyGpuFrameBoundary(capture.rows_path, int64_t(capture.frame));
    anatomy_gpu_row_ = 0;
    // wwe2k14 patch (Session 7c, CAUSES): the next frame's first row counts from here (the swap's own commands
    // belong to no row: they are its submission's tail).
    AnatomyRowCauses(nullptr);
    return;
  }

  // Start a capture at this frame boundary.
  const int32_t requested = REXCVAR_GET(gpu_anatomy_capture);
  if (requested <= 0) {
    return;
  }
  if (!anatomy_) {
    anatomy_ = std::make_unique<AnatomyCapture>();
  }
  AnatomyCapture& capture = *anatomy_;
  const std::string index_string = std::to_string(capture.captures_started++);
  std::filesystem::path rows_path;
  std::string path_pattern = REXCVAR_GET(gpu_anatomy_path);
  if (path_pattern.empty()) {
    std::error_code current_path_error;
    rows_path = std::filesystem::current_path(current_path_error) / "logs" /
                ("anatomy_" + index_string + ".csv");
  } else if (path_pattern.find("%d") != std::string::npos) {
    for (size_t position = path_pattern.find("%d"); position != std::string::npos;
         position = path_pattern.find("%d", position + index_string.size())) {
      path_pattern.replace(position, 2, index_string);
    }
    rows_path = path_pattern;
  } else {
    rows_path = path_pattern;
    if (index_string != "0") {
      // No %d: don't overwrite the first capture's file.
      rows_path.replace_filename(rows_path.stem().string() + "_" + index_string +
                                 rows_path.extension().string());
    }
  }
  std::filesystem::path frames_path = rows_path;
  frames_path.replace_filename(rows_path.stem().string() + "_frames" + rows_path.extension().string());
  if (rows_path.has_parent_path()) {
    std::error_code create_error;
    std::filesystem::create_directories(rows_path.parent_path(), create_error);
  }
  capture.CloseFiles();
  capture.rows_file = AnatomyOpenFile(rows_path);
  capture.frames_file = AnatomyOpenFile(frames_path);
  capture.rows_path = rows_path.string();
  if (!capture.rows_file || !capture.frames_file) {
    REXGPU_WARN("[s6][COUNTERS] Frame anatomy: can't open {} / {}", capture.rows_path, frames_path.string());
    capture.CloseFiles();
    REXCVAR_SET(gpu_anatomy_capture, 0);
    return;
  }
  std::fputs(
      "frame,idx,kind,t_us,cp_us,vs_hash,ps_hash,prim,count,indexed,index_fmt,rb_surface_info,"
      "rb_color_info0,rb_color_info1,rb_color_info2,rb_color_info3,rb_depth_info,rb_modecontrol,"
      "rb_depthcontrol,rb_colorcontrol,rb_color_mask,vs_tex,ps_tex,vs_consts,ps_consts,pipeline_changed,"
      "skipped,copy_dest,copy_info,"
      // wwe2k14 patch (Session 7, TILING-STAGE0): appended at the end, so existing tools keep working.
      "bin_select_lo,bin_mask_lo,predicated,pa_sc_window_offset,pa_sc_window_scissor_tl,"
      "pa_sc_window_scissor_br,pa_sc_screen_scissor_tl,pa_sc_screen_scissor_br,pa_cl_vport_xscale,"
      "pa_cl_vport_xoffset,pa_cl_vport_yscale,pa_cl_vport_yoffset,"
      // wwe2k14 patch (Session 7, EXTENTS-SHADOW): after the TILING-STAGE0 columns.
      "ext_minx,ext_maxx,ext_miny,ext_maxy,ext_ok,ext_why,ext_us,"
      // wwe2k14 patch (Session 7c, CAUSES): after the EXTENTS-SHADOW columns (AnatomyCauses).
      "up_calls,up_pages,tex_loads,tex_loads_scaled,tex_load_kib,dispatches,barrier_batches,barrier_src_stages,"
      "barrier_dst_stages,rendering_restarts,fsi_edram_barriers,desc_writes,pipeline_created,pipeline_pending,"
      // wwe2k14 patch (Session 19, DRAWRUNS): after the CAUSES columns.
      "state_hash,fetch_hash,"
      // wwe2k14 patch (Session 42, POSTMAP): after the DRAWRUNS columns.
      "copy_control,copy_pitch,tex\n",
      capture.rows_file);
  std::fputs("frame,draws,copies,cp_us_sum,span_us\n", capture.frames_file);
  capture.requested = requested;
  capture.frame = 0;
  capture.rows_written = 0;
  capture.rows_dropped = 0;
  capture.rows.clear();
  capture.rows.reserve(8192);
  capture.frame_started = false;
  anatomy_active_ = true;
  anatomy_frame_start_pending_ = true;
  REXGPU_INFO("[s6][COUNTERS] Frame anatomy capture {} started: {} frames to {}", index_string, requested,
              capture.rows_path);
  // wwe2k14 patch (Session 7b, GPUMARKS): the first capture frame's GPU marks.
  AnatomyGpuFrameBoundary(capture.rows_path, 0);
  anatomy_gpu_row_ = 0;
  AnatomyRowCauses(nullptr);  // wwe2k14 patch (Session 7c, CAUSES): the first row counts from here
}

void CommandProcessor::AnatomyFinish(const char* why) {
  AnatomyCapture& capture = *anatomy_;
  capture.CloseFiles();
  anatomy_active_ = false;
  anatomy_frame_start_pending_ = false;
  // Back to 0 so the harness can see that the capture is over. A different value set meanwhile is a new request
  // and is kept (it starts at the next frame boundary).
  if (REXCVAR_GET(gpu_anatomy_capture) == capture.requested) {
    REXCVAR_SET(gpu_anatomy_capture, 0);
  }
  REXGPU_INFO("[s6][COUNTERS] Frame anatomy capture {}: {} frame(s), {} row(s), {} dropped -> {}", why,
              capture.frame, capture.rows_written, capture.rows_dropped, capture.rows_path);
  capture.rows = std::vector<AnatomyCapture::Row>();
  // wwe2k14 patch (Session 7b, GPUMARKS): no more GPU marks for this capture (the backend still writes the
  // marked frames whose GPU work hasn't completed yet when it has).
  AnatomyGpuFrameBoundary(capture.rows_path, -1);
}

// wwe2k14 patch (Session 7, EXTENTS-SHADOW): gpu_extents_mode (util/draw_extent_estimator.h). Off (mode 0 and no
// tracker created yet): one setting read per frame, one bool test per draw packet and per extents event.

void CommandProcessor::ExtentsFrameBoundary() {
  if (!extents_shadow_) {
    if (!ScreenExtentsShadow::Requested()) {
      return;
    }
    extents_shadow_ = std::make_unique<ScreenExtentsShadow>(*register_file_, *memory_);
  }
  extents_active_ = extents_shadow_->EndFrame(counter_);
}

void CommandProcessor::ExtentsOnDraw(bool issued, bool is_copy, size_t anatomy_rows_before) {
  ScreenExtentsShadow::DrawRow row;
  extents_shadow_->OnDraw(active_vertex_shader_, bin_select_, issued, is_copy, GetSharedMemoryForExtents(), row);
  if (row.attempted && anatomy_active_ && anatomy_ && anatomy_rows_before != SIZE_MAX &&
      anatomy_->rows.size() == anatomy_rows_before + 1) {
    AnatomyCapture::Row& anatomy_row = anatomy_->rows.back();
    anatomy_row.ext_attempted = true;
    anatomy_row.ext_ok = row.ok;
    anatomy_row.ext_min_x = row.min_x;
    anatomy_row.ext_max_x = row.max_x;
    anatomy_row.ext_min_y = row.min_y;
    anatomy_row.ext_max_y = row.max_y;
    anatomy_row.ext_fallback = uint32_t(row.fallback);
    anatomy_row.ext_us = row.us;
  }
}

// wwe2k14 patch (Session 7, TILING-STAGE0): the predicated tiling log (gpu_tiling_log_frames). Measurement
// only: nothing here changes what is emulated or drawn. One CSV row per event, in command processor order:
//   frame       log frame (0 = the frame after the swap where the log started)
//   seq         row number in the frame
//   t3          the packet's number among the frame's type-3 packets (1 = first; cpf_t0x c counts them)
//   draw        IssueDraw calls (draws and resolves) before the packet in the frame = the frame anatomy's idx of
//               the next draw/copy row when a frame-anatomy capture covers the same frames
//   pkt         guest physical address of the packet's header (the game's extents records hold "the address of
//               the next packet")
//   kind        sel_lo / sel_hi / sel64: SET_BIN_SELECT_LO / _HI / SET_BIN_SELECT (select = the new value)
//               mask_lo / mask_hi / mask64: SET_BIN_MASK_LO / _HI / SET_BIN_MASK (mask = the new value)
//               ext: EVENT_WRITE_EXT executed (value = the address the extents are written to, endian bits
//                    cleared; info = the event initiator)
//               ext_init: EVENT_WRITE SCREEN_EXT_INIT without the predicate bit (info = the initiator)
//               interrupt / interrupt_end: INTERRUPT, before and after its callbacks (value = the CPU mask)
//               seg: packet counts since the last bin select change (or the frame start), written before the
//                    next sel_* row and at the frame end when not all zero (select = the select they ran under)
//               frame: the frame boundary (value = cp_swap's b, the CP frame number; counts = the frame's totals)
//   pred        the packet had the predicate bit
//   select, mask  64-bit BIN_SELECT / BIN_MASK after the packet
//   value, info   see kind (hex; empty when not used)
//   t_us        microseconds since the frame started (the previous swap), for sel_*, ext_init, interrupt*, seg
//               and frame rows only
//   pred_exec, pred_skip    predicated type-3 packets executed / skipped by the bin test (seg and frame rows)
//   draw_exec, draw_skip    predicated DRAW_INDX / DRAW_INDX_2 executed / skipped
//   draw_unpred             DRAW_INDX / DRAW_INDX_2 without the predicate bit
//   ext_exec, ext_skip      EVENT_WRITE_EXT executed / skipped
// Rows are buffered and written at each frame boundary (or every 8 MiB).

namespace {

// gpu_anatomy_path's directory: anatomy_NNN_%d.csv -> tiling_NNN_<index>.csv; any other file name gets a
// "tiling_" prefix (and, without %d, an "_<index>" suffix from the second log on, as the anatomy capture does).
std::filesystem::path TilingLogPath(const std::string& anatomy_path, const std::string& index_string) {
  if (anatomy_path.empty()) {
    std::error_code current_path_error;
    return std::filesystem::current_path(current_path_error) / "logs" / ("tiling_" + index_string + ".csv");
  }
  const std::filesystem::path anatomy(anatomy_path);
  std::string name = anatomy.filename().string();
  if (name.compare(0, 7, "anatomy") == 0) {
    name.replace(0, 7, "tiling");
  } else {
    name.insert(0, "tiling_");
  }
  if (name.find("%d") != std::string::npos) {
    for (size_t position = name.find("%d"); position != std::string::npos;
         position = name.find("%d", position + index_string.size())) {
      name.replace(position, 2, index_string);
    }
  } else if (index_string != "0") {
    const std::filesystem::path name_path(name);
    name = name_path.stem().string() + "_" + index_string + name_path.extension().string();
  }
  return anatomy.parent_path() / name;
}

}  // namespace

void CommandProcessor::TilingLogPacket(const memory::RingBuffer* reader, uint32_t data_start_offset,
                                       uint32_t opcode, bool predicated, bool executed) {
  TilingLog& log = *tiling_log_;
  // The header is the dword before the packet's data (the ring may wrap there). Command buffers are in guest
  // physical memory (TranslatePhysical).
  const uint32_t capacity = uint32_t(reader->capacity());
  const uint32_t header_offset = (data_start_offset + capacity - uint32_t(sizeof(uint32_t))) % capacity;
  log.packet_address = uint32_t((reader->buffer() + header_offset) - memory_->physical_membase());
  log.packet_t3 = frame_counters_.type3_packets;
  log.packet_draw = frame_counters_.issue_draw;
  log.packet_predicated = predicated;
  TilingLogCounts& segment = log.segment;
  if (predicated) {
    ++(executed ? segment.pred_exec : segment.pred_skip);
  }
  if (opcode == PM4_DRAW_INDX || opcode == PM4_DRAW_INDX_2) {
    ++(!predicated ? segment.draw_unpred : executed ? segment.draw_exec : segment.draw_skip);
  } else if (opcode == PM4_EVENT_WRITE_EXT) {
    ++(executed ? segment.ext_exec : segment.ext_skip);
  }
}

void CommandProcessor::TilingLogRow(const char* kind, uint64_t select, int64_t value, int64_t info,
                                    bool timed, const TilingLogCounts* counts) {
  TilingLog& log = *tiling_log_;
  fmt::memory_buffer& text = log.text;
  fmt::format_to(std::back_inserter(text), "{},{},{},{},0x{:08X},{},{},0x{:016X},0x{:016X},", log.frame,
                 log.seq++, log.packet_t3, log.packet_draw, log.packet_address, kind,
                 log.packet_predicated ? 1 : 0, select, bin_mask_);
  if (value >= 0) {
    fmt::format_to(std::back_inserter(text), "0x{:08X}", uint32_t(value));
  }
  text.push_back(',');
  if (info >= 0) {
    fmt::format_to(std::back_inserter(text), "0x{:08X}", uint32_t(info));
  }
  text.push_back(',');
  if (timed) {
    fmt::format_to(std::back_inserter(text), "{:.1f}", double(AnatomyNowNs() - log.frame_start_ns) / 1000.0);
  }
  if (counts) {
    fmt::format_to(std::back_inserter(text), ",{},{},{},{},{},{},{}\n", counts->pred_exec, counts->pred_skip,
                   counts->draw_exec, counts->draw_skip, counts->draw_unpred, counts->ext_exec,
                   counts->ext_skip);
  } else {
    fmt::format_to(std::back_inserter(text), ",,,,,,,\n");
  }
  if (text.size() >= TilingLog::kFlushBytes) {
    TilingLogFlush();
  }
}

void CommandProcessor::TilingLogFlush() {
  TilingLog& log = *tiling_log_;
  if (log.text.size() && log.file &&
      std::fwrite(log.text.data(), 1, log.text.size(), log.file) != log.text.size()) {
    log.write_failed = true;
  }
  log.text.clear();
}

void CommandProcessor::TilingLogEndSegment(uint64_t select) {
  TilingLog& log = *tiling_log_;
  if (!log.segment.any()) {
    return;
  }
  TilingLogRow("seg", select, -1, -1, true, &log.segment);
  log.frame_totals.Add(log.segment);
  log.segment = TilingLogCounts();
}

void CommandProcessor::TilingLogBinSelect(const char* kind, uint64_t old_select) {
  TilingLogEndSegment(old_select);
  TilingLogRow(kind, bin_select_, -1, -1, true, nullptr);
}

void CommandProcessor::TilingLogBinMask(const char* kind) {
  TilingLogRow(kind, bin_select_, -1, -1, false, nullptr);
}

void CommandProcessor::TilingLogEvent(const char* kind, int64_t value, int64_t info, bool timed) {
  TilingLogRow(kind, bin_select_, value, info, timed, nullptr);
}

void CommandProcessor::TilingLogFrameBoundary() {
  if (tiling_log_active_) {
    // Close the frame that just ended (called after the swap; the swap packet is the last one TilingLogPacket
    // noted). value = the counter_ value cp_swap recorded, not counter_ - 1: vblanks also increment counter_.
    TilingLog& log = *tiling_log_;
    TilingLogEndSegment(bin_select_);
    TilingLogRow("frame", bin_select_, int64_t(tiling_swap_counter_), -1, true, &log.frame_totals);
    TilingLogFlush();
    // Ties the log's frames to the timeline's cp_swap events.
    perf::timeline::Record("tiling_log", log.logs_started - 1, log.frame, log.seq);
    log.rows_written += log.seq;
    log.seq = 0;
    log.frame_totals = TilingLogCounts();
    ++log.frame;
    if (log.write_failed) {
      TilingLogFinish("write error");
      return;
    }
    if (log.frame >= uint32_t(log.requested)) {
      TilingLogFinish("done");
      return;
    }
    log.frame_start_ns = AnatomyNowNs();
    return;
  }

  // Start a log at this frame boundary.
  const int32_t requested = REXCVAR_GET(gpu_tiling_log_frames);
  if (requested <= 0) {
    return;
  }
  if (!tiling_log_) {
    tiling_log_ = std::make_unique<TilingLog>();
  }
  TilingLog& log = *tiling_log_;
  const std::string index_string = std::to_string(log.logs_started++);
  const std::filesystem::path path = TilingLogPath(REXCVAR_GET(gpu_anatomy_path), index_string);
  if (path.has_parent_path()) {
    std::error_code create_error;
    std::filesystem::create_directories(path.parent_path(), create_error);
  }
  log.CloseFile();
  log.file = AnatomyOpenFile(path);
  log.path = path.string();
  if (!log.file) {
    REXGPU_WARN("[s7][TILING] Tiling log: can't open {}", log.path);
    REXCVAR_SET(gpu_tiling_log_frames, 0);
    return;
  }
  std::fputs(
      "frame,seq,t3,draw,pkt,kind,pred,select,mask,value,info,t_us,pred_exec,pred_skip,draw_exec,draw_skip,"
      "draw_unpred,ext_exec,ext_skip\n",
      log.file);
  log.requested = requested;
  log.frame = 0;
  log.seq = 0;
  log.rows_written = 0;
  log.write_failed = false;
  log.text.clear();
  log.segment = TilingLogCounts();
  log.frame_totals = TilingLogCounts();
  log.packet_address = 0;
  log.packet_t3 = 0;
  log.packet_draw = 0;
  log.packet_predicated = false;
  log.frame_start_ns = AnatomyNowNs();
  tiling_log_active_ = true;
  REXGPU_INFO("[s7][TILING] Tiling log {} started: {} frames to {}", index_string, requested, log.path);
}

void CommandProcessor::TilingLogFinish(const char* why) {
  TilingLog& log = *tiling_log_;
  TilingLogFlush();
  log.CloseFile();
  tiling_log_active_ = false;
  // Back to 0 so the harness can see that the log is over. A different value set meanwhile is a new request and
  // is kept (it starts at the next frame boundary).
  if (REXCVAR_GET(gpu_tiling_log_frames) == log.requested) {
    REXCVAR_SET(gpu_tiling_log_frames, 0);
  }
  REXGPU_INFO("[s7][TILING] Tiling log {}: {} frame(s), {} row(s) -> {}", why, log.frame, log.rows_written,
              log.path);
  log.text = fmt::memory_buffer();
}

}  // namespace rex::graphics
