// wwe2k14 patch (Session 93): the phase 0 census of the memory mirror (ideas 11 + 13: docs\native\d3d\S93_MIRROR_DESIGN.md section 4.1) and of the
// decoupled rendering stretch (docs\native\d3d\S93_DECOUPLE_DESIGN.md section 6, phase 0). LOG ONLY: gpu_d3d_phase0_census (default false); off = no call
// does anything. Everything here reads the typed state (OwnedRecordState: the draw's own words, the float banks, the fetch constants) and guest memory
// (index and vertex bytes, sampled); nothing is written to either, nothing is drawn differently.
//
// Where it runs: Phase0Draw is called by D3dDirectAccess::TypedDraw (owned_front_own.cpp) for every draw item the typed draw is given (the cut's DMA draws, inline
// draws, crowd members); Phase0FrameEnd by the base class' SceneCrowdFrameEnd hook (vulkan/command_processor.cpp), which runs every guest frame also under the cut.
//
// What is computed and how (the full list is in the window lines below; every line starts with "[d3d][phase0]"):
//  - per pass (a pass = a run of draws with the same colour0 / depth / surface RB words; its key and a mean ordinal are printed) the top vertex shaders by draws;
//    "ours" = shader_swap::VertexTakes(hash) (needs gpu_native_vs_swap on), else "translated".
//  - mirror: per used vertex fetch constant (the shader's vertex_bindings, the typed fetch words) the range [address, address + size) keyed by its END (stream
//    offsets move the start); static = seen in 3+ consecutive frames with an unchanged sample hash (3 x 512 bytes at the range's end - the cheap stand-in for an
//    invalidation callback), transient = otherwise; bytes = union of the ranges per frame and per window. The index scan reads the guest indices once per
//    distinct mesh (dma base, count, base vertex, vertex buffer, sample) for min / max index (the exact vertex range), the index identity checks of the design
//    (3.4 point 5), and whether the game's own VGT min / max registers are tight enough to use instead (reported).
//  - off-screen (our arena family = one binding of stride 8 words): the mesh's object-space box from the position dwords 0-3 of vertices [min, max] + base, 8 corners
//    through c76-c78 and c56-c59 (clip.k = dot((dot(p,c76), dot(p,c77), dot(p,c78), 1), c5k)), culled when all 8 corners are outside the same x / y plane (or all
//    behind the eye) with a 1% margin; z > w is reported on its own, never counted.
//  - merge: consecutive draws with the same shaders and no typed state change (all decoder epochs equal) whose index ranges are back to back (same base vertex,
//    format, triangle list); the run lengths, and the draws that differ only in the vertex float constants (and only in c76-c78).
//  - decouple: draws matched with the previous guest frame's by (vs, ps, vertex fetch addresses, index dma base + count, surface key, ordinal among equal keys);
//    per matched draw which of the 256 vertex constant rows changed (a 32-bit hash per row), per vertex shader and in total; the camera block (c56-c59) coverage;
//    draws whose vertex data sample changed between matched frames (CPU-written: skinned, particles); the match + lerp cost.
// Not computed: camera cuts per minute (the game's camera hand-off is not visible from here), draws overlapping the native-skin output (the sample-hash change is the
// stand-in, a lower bound), the pass ordinal of the owned frame (the render-target key is the stand-in), the registry trace of the game's buffer creates (game side).

#include "phase0_census.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <rex/cvar.h>
#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/vulkan/shader.h>
#include <rex/graphics/xenos.h>
#include <rex/hash.h>
#include <rex/logging.h>
#include <rex/system/xmemory.h>

#include "draw_bounds.h"
#include "owned_record.h"
#include "shader_swap.h"

REXCVAR_DEFINE_BOOL(gpu_d3d_phase0_census, false, "GPU/Native",
                    "Session 93 (LOG ONLY, docs\\native\\d3d\\S93_MIRROR_DESIGN.md + S93_DECOUPLE_DESIGN.md phase 0): census of the typed draws for the memory mirror (vertex / index "
                    "buffer sizes, static vs transient, off-screen draws, exact merge candidates) and for decoupled rendering (draw match rate between guest frames, which "
                    "vertex constants change, camera block coverage, CPU-written vertex data, blend cost). One set of [d3d][phase0] lines per 600 guest frames. Off = nothing runs.");

namespace rex::graphics::vulkan {

namespace {

using Clock = std::chrono::steady_clock;
inline uint64_t NowNs() { return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count()); }

constexpr uint32_t kRows = 256;
constexpr uint32_t kWindowFrames = 600;
constexpr uint32_t kCamRow = 56;    // c56..c59: the view-projection rows
constexpr uint32_t kModelRow = 76;  // c76..c78: the model matrix rows
constexpr uint32_t kScanBudget = 400000;  // indices + vertices read by the scans, per guest frame

inline uint64_t Mix(uint64_t a, uint64_t b) {
  a ^= b + 0x9E3779B97F4A7C15ull + (a << 6) + (a >> 2);
  return a;
}
inline uint32_t RowHash(const uint32_t* w) {
  uint32_t h = w[0] * 0x9E3779B1u;
  h = (h ^ (h >> 15)) + w[1] * 0x85EBCA77u;
  h = (h ^ (h >> 13)) + w[2] * 0xC2B2AE3Du;
  h = (h ^ (h >> 16)) + w[3] * 0x27D4EB2Fu;
  return h ^ (h >> 15);
}
inline float F(uint32_t bits) {
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}
inline double Pct(uint64_t a, uint64_t b) { return b ? 100.0 * double(a) / double(b) : 0.0; }

struct Rec {  // one draw of a guest frame (the previous frame's list is what the next frame matches against)
  uint64_t key = 0;
  uint32_t ord = 0;
  uint64_t vs = 0, ps = 0;
  uint64_t cam_hash = 0;
  uint64_t vb_sample = 0;  // 0 = none
  uint32_t pool_off = 0;   // the 256 row hashes of the draw's vertex constants, in the frame's pool
  int32_t matched = -1;    // the index of the match in the previous frame's list
  bool ours = false, inl = false, cam_zero = true;
  float mat[28] = {};      // c56..c59 (16) then c76..c78 (12)
};

struct VsStat {
  uint64_t draws = 0, matched = 0, cam_changed = 0, model_changed = 0, vb_checked = 0, vb_changed = 0, boxable = 0, offscreen = 0, changed_rows = 0;
  uint32_t bindings = 0;
  bool ours = false;
  std::array<uint32_t, kRows> row_changed{};
};

struct PassStat {
  uint64_t draws = 0, entries = 0, ord_sum = 0, matched = 0, boxable = 0, offscreen = 0;
  uint32_t color = 0, depth = 0, surface = 0;
  std::unordered_map<uint64_t, uint32_t> vs_draws;
};

struct VbEntry {  // a vertex buffer range as the fetch constants name it, keyed by its end
  uint32_t start = 0, end = 0;
  uint64_t frame = UINT64_MAX;  // the last frame it was touched
  uint64_t sample = 0;
  uint32_t streak = 0;  // consecutive frames touched with an unchanged sample
  uint32_t changes = 0;
  uint64_t window = UINT64_MAX;
  bool ours = false;
};

struct Mesh {  // one distinct (index range, base vertex, vertex buffer) - scanned once
  uint64_t window = UINT64_MAX;
  uint32_t min_i = 0, max_i = 0;
  bool ok = false, resets = false;
  uint32_t used_bytes = 0;
  bool box_done = false, box_ok = false;
  draw_bounds::Box box;
};

struct Last {  // the previous draw, for the merge census
  bool valid = false;
  uint64_t vs = 0, ps = 0;
  uint64_t e_render = 0, e_vs = 0, e_ps = 0, e_bl = 0, e_fetch = 0;
  uint32_t dma_end = 0, base = 0, fmt = 0, swap = 0, prim = 0, pool_off = 0, run = 0;
  bool dma = false;
};

struct Window {
  uint64_t first_frame = 0;
  uint64_t frames_with_draws = 0;
  uint64_t draws = 0, dma_draws = 0, inline_draws = 0, crowd_members = 0, ours_draws = 0, no_vs = 0;
  uint64_t frame_draw_max = 0;
  uint64_t pass_entries = 0;
  uint64_t fetches = 0, fetch_not_vertex = 0;
  // index scans / identity
  uint64_t scans = 0, scan_indices = 0, scan_skipped_budget = 0, scan_unsupported = 0, id_checked = 0, id_fail_min = 0, id_fail_max = 0, id_fail_mask = 0, id_fail_any = 0;
  uint64_t tight_regs = 0, tight_checked = 0;
  uint64_t used_bytes_meshes = 0;
  // mirror
  uint64_t frame_union_sum = 0, frame_union_max = 0;
  // off-screen
  uint64_t ours_not_boxable_shape = 0, boxable = 0, no_bounds = 0, w_not_one = 0, tested = 0, off_xy = 0, off_behind = 0, off_z_only = 0, box_scan_vertices = 0;
  // merge
  uint64_t consecutive = 0, state_identical = 0, contiguous = 0, contiguous_tri = 0, same_state_gap = 0, vsfloat_only = 0, model_only = 0, runs = 0, saved = 0;
  uint64_t run_hist[4] = {};  // 2, 3-4, 5-8, 9+
  // match
  uint64_t match_draws = 0, matched = 0, matched_ours = 0, ours_match_draws = 0, matched_inline = 0, inline_match_draws = 0;
  uint64_t pairs_cam_changed = 0, pairs_model_changed = 0, pairs_other_changed_draws = 0, pairs_bone_rows = 0, pairs_bone_draws4 = 0, pairs_rows_total = 0;
  uint64_t vb_checked = 0, vb_changed = 0, vb_changed_ours = 0;
  std::array<uint64_t, kRows> row_changed{};
  // camera
  uint64_t cam_distinct_sum = 0, cam_distinct_max = 0, cam_mapped_sum = 0;
  uint64_t cam_ours = 0, cam_ours_covered = 0, cam_tr = 0, cam_tr_covered = 0, cam_zero_draws = 0;
  // cost (ns)
  uint64_t ns_match = 0, ns_lerp = 0, lerps = 0, ns_total = 0, ns_frame_end = 0;
  std::unordered_map<uint64_t, VsStat> vs;
  std::unordered_map<uint64_t, PassStat> passes;
};

struct State {
  uint64_t frame_no = 0;  // guest frames seen by Phase0FrameEnd
  Window w;
  // the current and previous frame
  std::vector<Rec> cur, prev;
  std::vector<uint32_t> cur_pool, prev_pool;
  std::unordered_map<uint64_t, uint32_t> cur_ord, cur_map, prev_map;
  uint64_t frame_draws = 0;
  bool pool_valid = false;
  uint64_t last_vs_epoch = 0;
  uint32_t last_pool_off = 0;
  uint64_t cur_pass_key = 0;
  bool pass_valid = false;
  uint32_t pass_ord = 0;
  Last last;
  uint32_t run = 0;
  uint32_t scan_budget = kScanBudget;
  std::unordered_map<uint32_t, VbEntry> vb;  // by end
  std::vector<std::pair<uint32_t, uint32_t>> frame_ranges;
  std::unordered_map<uint64_t, Mesh> meshes;
  bool started = false;
};
State g;

// ---- guest memory sampling ---------------------------------------------------------------------------------------
uint64_t SampleRange(rex::memory::Memory* mem, uint32_t start, uint32_t end) {
  if (!mem || end <= start || end - start > (256u << 20)) return 0;
  uint64_t h = 0x1234567;
  const uint32_t offsets[3] = {512, 4096, 16384};
  for (uint32_t back : offsets) {
    uint32_t a = end > back ? end - back : 0;
    if (a < start) a = start;
    const uint32_t n = std::min<uint32_t>(512, end - a);
    if (!n) continue;
    h = Mix(h, XXH3_64bits(mem->TranslatePhysical<const uint8_t*>(a), n));
  }
  return h | 1;  // never 0
}

VbEntry& TouchVb(rex::memory::Memory* mem, uint32_t start, uint32_t end, bool ours) {
  VbEntry& e = g.vb[end];
  if (e.window == UINT64_MAX) {
    e.start = start;
    e.end = end;
  } else if (start < e.start) {
    e.start = start;
  }
  e.window = g.frame_no / kWindowFrames;
  e.ours |= ours;
  if (e.frame != g.frame_no) {  // the first touch this frame
    const uint64_t sample = SampleRange(mem, e.start, e.end);
    if (e.frame != UINT64_MAX && e.frame + 1 == g.frame_no && sample == e.sample) {
      ++e.streak;
    } else {
      if (e.frame != UINT64_MAX && sample != e.sample) ++e.changes;
      e.streak = 1;
    }
    e.sample = sample;
    e.frame = g.frame_no;
    g.frame_ranges.emplace_back(e.start, e.end);
  }
  return e;
}

uint64_t UnionBytes(std::vector<std::pair<uint32_t, uint32_t>>& r) {
  std::sort(r.begin(), r.end());
  uint64_t total = 0;
  uint32_t cs = 0, ce = 0;
  bool open = false;
  for (const auto& p : r) {
    if (!open) {
      cs = p.first;
      ce = p.second;
      open = true;
    } else if (p.first <= ce) {
      ce = std::max(ce, p.second);
    } else {
      total += ce - cs;
      cs = p.first;
      ce = p.second;
    }
  }
  if (open) total += ce - cs;
  return total;
}

// ---- the index scan + the box ------------------------------------------------------------------------------------
// Returns the mesh (scanned now or cached); ok = false when the indices could not be read (budget, size, unsupported endian).
Mesh& ScanMesh(rex::memory::Memory* mem, const OwnedRecordState& s, uint64_t key, uint32_t* stride_words_unused) {
  (void)stride_words_unused;
  Mesh& m = g.meshes[key];
  if (m.window == UINT64_MAX || (!m.ok && m.window != g.frame_no / kWindowFrames)) {
    m = Mesh();
  }
  m.window = g.frame_no / kWindowFrames;
  if (m.ok || m.min_i == 0xFFFFFFFFu) return m;
  draw_bounds::IndexScan sc;
  switch (draw_bounds::ScanIndices(mem, s, g.scan_budget, sc)) {
    case draw_bounds::ScanStatus::kUnsupported:
      ++g.w.scan_unsupported;
      m.min_i = 0xFFFFFFFFu;  // do not retry
      return m;
    case draw_bounds::ScanStatus::kBudget:
      ++g.w.scan_skipped_budget;
      return m;  // not ok, retried next frame
    case draw_bounds::ScanStatus::kEmpty:
      ++g.w.scans;
      g.w.scan_indices += sc.n;
      m.min_i = 0xFFFFFFFFu;
      return m;
    case draw_bounds::ScanStatus::kOk:
      break;
  }
  ++g.w.scans;
  g.w.scan_indices += sc.n;
  m.min_i = sc.min_i;
  m.max_i = sc.max_i;
  m.resets = sc.resets;
  m.ok = true;
  return m;
}

// The object-space box of vertices [min, max] + base of a 32-byte-vertex stream (draw_bounds::BuildBox; the census' flags and counters).
void BuildBox(rex::memory::Memory* mem, Mesh& m, uint32_t vb_start_byte, uint32_t vb_size_bytes, uint32_t stride_words, uint32_t base, xenos::Endian endian) {
  m.box_done = true;
  m.box_ok = false;
  uint32_t read = 0;
  const draw_bounds::BoxStatus st = draw_bounds::BuildBox(mem, m.min_i, m.max_i, vb_start_byte, vb_size_bytes, stride_words, base, endian, g.scan_budget, m.box, read);
  g.w.box_scan_vertices += read;
  m.box_ok = st == draw_bounds::BoxStatus::kOk;  // a vertex with w != 1: not cullable (the design's rule)
  if (st == draw_bounds::BoxStatus::kWNotOne) ++g.w.w_not_one;
}

std::string RowRanges(const std::array<uint32_t, kRows>& rows, uint32_t threshold, uint32_t max_items) {
  std::string out;
  uint32_t items = 0;
  for (uint32_t r = 0; r < kRows && items < max_items;) {
    if (rows[r] < threshold) {
      ++r;
      continue;
    }
    uint32_t e = r;
    while (e + 1 < kRows && rows[e + 1] >= threshold) ++e;
    out += (out.empty() ? "c" : " c") + std::to_string(r);
    if (e != r) out += "-" + std::to_string(e);
    ++items;
    r = e + 1;
  }
  return out.empty() ? "none" : out;
}

const char* Kind(bool ours) { return ours ? "ours" : "translated"; }

// ---- the window lines --------------------------------------------------------------------------------------------
void Flush() {
  Window& w = g.w;
  const uint64_t a = w.first_frame, b = g.frame_no;
  const double fr = double(std::max<uint64_t>(1, w.frames_with_draws));
  const uint64_t dma_like = w.draws - w.crowd_members;
  REXGPU_INFO("[d3d][phase0] frames {}-{}: {} frames with draws, draws {} ({:.1f} a frame, max {}; dma {}, inline {}, crowd members {}), ours {} ({:.1f}%), no shader known {}, "
              "vertex shaders {}, pass entries {:.1f} a frame, census cost {:.1f} us a frame (match + lerp {:.1f})",
              a, b, w.frames_with_draws, w.draws, double(w.draws) / fr, w.frame_draw_max, w.dma_draws, w.inline_draws, w.crowd_members, w.ours_draws,
              Pct(w.ours_draws, dma_like), w.no_vs, w.vs.size(), double(w.pass_entries) / fr, double(w.ns_total) / fr / 1000.0, double(w.ns_match + w.ns_lerp) / fr / 1000.0);

  // 1. per pass, top vertex shaders by draws
  {
    std::vector<std::pair<uint64_t, const PassStat*>> ps;
    for (const auto& kv : w.passes) ps.emplace_back(kv.first, &kv.second);
    std::sort(ps.begin(), ps.end(), [](const auto& x, const auto& y) { return x.second->draws > y.second->draws; });
    const size_t pn = std::min<size_t>(ps.size(), 8);
    for (size_t i = 0; i < pn; ++i) {
      const PassStat& p = *ps[i].second;
      REXGPU_INFO("[d3d][phase0][pass] #{} rt c0 {:08X} depth {:08X} surf {:08X}: draws {:.1f} a frame ({} shaders), entered {:.2f} a frame at mean ordinal {:.1f}, matched {:.1f}%, boxable {}, off-screen {}",
                  i, p.color, p.depth, p.surface, double(p.draws) / fr, p.vs_draws.size(), double(p.entries) / fr, p.entries ? double(p.ord_sum) / double(p.entries) : 0.0,
                  Pct(p.matched, p.draws), p.boxable, p.offscreen);
      std::vector<std::pair<uint64_t, uint32_t>> top(p.vs_draws.begin(), p.vs_draws.end());
      std::sort(top.begin(), top.end(), [](const auto& x, const auto& y) { return x.second > y.second; });
      for (size_t k = 0; k < std::min<size_t>(top.size(), 8); ++k) {
        const VsStat& v = w.vs[top[k].first];
        REXGPU_INFO("[d3d][phase0][pass]   #{} vs {:016X} {}: {:.1f} draws a frame, {} bindings", i, top[k].first, Kind(v.ours), double(top[k].second) / fr, v.bindings);
      }
    }
    // overall top 24
    std::vector<std::pair<uint64_t, const VsStat*>> vs;
    for (const auto& kv : w.vs) vs.emplace_back(kv.first, &kv.second);
    std::sort(vs.begin(), vs.end(), [](const auto& x, const auto& y) { return x.second->draws > y.second->draws; });
    for (size_t k = 0; k < std::min<size_t>(vs.size(), 24); ++k) {
      const VsStat& v = *vs[k].second;
      REXGPU_INFO("[d3d][phase0][top] vs {:016X} {}: {:.1f} draws a frame ({:.1f}% of all), {} bindings, matched {:.1f}%", vs[k].first, Kind(v.ours), double(v.draws) / fr,
                  Pct(v.draws, w.draws), v.bindings, Pct(v.matched, v.draws));
    }
  }

  // 2. mirror
  {
    std::vector<std::pair<uint32_t, uint32_t>> all, stat, trans, ours_r;
    uint64_t n_static = 0, n_trans = 0, n_new = 0, changes = 0;
    for (const auto& kv : g.vb) {
      const VbEntry& e = kv.second;
      if (e.window != g.frame_no / kWindowFrames && e.window != (g.frame_no - 1) / kWindowFrames) continue;  // (the window just ended: touched in it)
      all.emplace_back(e.start, e.end);
      if (e.ours) ours_r.emplace_back(e.start, e.end);
      changes += e.changes;
      if (e.streak >= 3 && e.changes == 0) {
        ++n_static;
        stat.emplace_back(e.start, e.end);
      } else if (e.streak >= 3) {
        ++n_static;  // (changed once, then stable again: counted static; the change count is printed)
        stat.emplace_back(e.start, e.end);
      } else {
        ++n_trans;
        trans.emplace_back(e.start, e.end);
      }
    }
    (void)n_new;
    const uint64_t ub = UnionBytes(all), sb = UnionBytes(stat), tb = UnionBytes(trans), ob = UnionBytes(ours_r);
    REXGPU_INFO("[d3d][phase0][mirror] frames {}-{}: distinct vertex ranges {} (static {}: stable 3+ frames, transient {}), bytes union {:.2f} MiB (static {:.2f}, transient {:.2f}, our shaders' ranges {:.2f}), "
                "per frame touched union avg {:.2f} MiB max {:.2f} MiB, sample changes seen {}, fetches {} ({:.1f} a frame, not vertex type {}), the ranges' size field is the buffer's rest: an upper bound",
                a, b, all.size(), n_static, n_trans, double(ub) / 1048576.0, double(sb) / 1048576.0, double(tb) / 1048576.0, double(ob) / 1048576.0,
                double(w.frame_union_sum) / fr / 1048576.0, double(w.frame_union_max) / 1048576.0, changes, w.fetches, double(w.fetches) / fr, w.fetch_not_vertex);
    uint64_t used = 0;
    uint64_t meshes = 0;
    for (const auto& kv : g.meshes) {
      if (kv.second.window == g.frame_no / kWindowFrames || kv.second.window == (g.frame_no - 1) / kWindowFrames) {
        ++meshes;
        used += kv.second.used_bytes;
      }
    }
    w.used_bytes_meshes = used;
    REXGPU_INFO("[d3d][phase0][index] index scans {} ({} indices read, budget skips {}, unsupported {}), distinct meshes {} (vertex bytes used by their [min,max] ranges {:.2f} MiB), identity checks {}: fails {} (min {}, max {}, 24-bit {}), "
                "VGT min/max registers tight (range <= scanned range + 16) {}/{} ({:.1f}%)",
                w.scans, w.scan_indices, w.scan_skipped_budget, w.scan_unsupported, meshes, double(used) / 1048576.0, w.id_checked, w.id_fail_any, w.id_fail_min, w.id_fail_max,
                w.id_fail_mask, w.tight_regs, w.tight_checked, Pct(w.tight_regs, w.tight_checked));
  }

  // 3. off-screen
  {
    uint64_t ours_total = w.ours_draws;
    REXGPU_INFO("[d3d][phase0][cull] our draws {}, of them boxable (one stream, stride 8 words, a bound known, all w == 1) {} ({:.1f}%), not boxable by shape {} / no bounds yet {} / w != 1 meshes {}; "
                "tested {}: off-screen x/y {} + behind the eye {} = {} ({:.1f}% of tested), z > w only (not counted) {}; box vertices read {}",
                ours_total, w.boxable, Pct(w.boxable, ours_total), w.ours_not_boxable_shape, w.no_bounds, w.w_not_one, w.tested, w.off_xy, w.off_behind, w.off_xy + w.off_behind,
                Pct(w.off_xy + w.off_behind, w.tested), w.off_z_only, w.box_scan_vertices);
    std::vector<std::pair<uint64_t, const VsStat*>> vs;
    for (const auto& kv : w.vs) if (kv.second.boxable) vs.emplace_back(kv.first, &kv.second);
    std::sort(vs.begin(), vs.end(), [](const auto& x, const auto& y) { return x.second->boxable > y.second->boxable; });
    for (size_t k = 0; k < std::min<size_t>(vs.size(), 12); ++k) {
      REXGPU_INFO("[d3d][phase0][cull]   vs {:016X}: boxable {} ({:.1f} a frame), off-screen {} ({:.1f}%)", vs[k].first, vs[k].second->boxable, double(vs[k].second->boxable) / fr,
                  vs[k].second->offscreen, Pct(vs[k].second->offscreen, vs[k].second->boxable));
    }
  }

  // 4. merge candidates
  REXGPU_INFO("[d3d][phase0][merge] consecutive draw pairs {}: state identical (same shaders, no typed state change) {} ({:.1f}%); of them contiguous index ranges (back to back, same base vertex / format / prim) {} "
              "(triangle lists {}), same state but not contiguous {}; differing only in the vertex float constants {} (only in c76-c78: {}); runs of contiguous draws {} saving {} draws ({:.1f}% of dma draws), "
              "run lengths 2: {}, 3-4: {}, 5-8: {}, 9+: {}",
              w.consecutive, w.state_identical, Pct(w.state_identical, w.consecutive), w.contiguous, w.contiguous_tri, w.same_state_gap, w.vsfloat_only, w.model_only, w.runs, w.saved,
              Pct(w.saved, w.dma_draws), w.run_hist[0], w.run_hist[1], w.run_hist[2], w.run_hist[3]);

  // 5. decouple: the match
  {
    const double mp = Pct(w.matched, w.match_draws);
    REXGPU_INFO("[d3d][phase0][match] draws with a previous frame {}: matched {} ({:.1f}%) - ours {}/{} ({:.1f}%), inline {}/{} ({:.1f}%); matched pairs whose c56-c59 changed {} ({:.1f}%), c76-c78 changed {} ({:.1f}%), "
                "other rows changed in {} pairs, rows 79-255 changed {} (pairs with 4+ of them: {}), changed rows per matched draw {:.1f}",
                w.match_draws, w.matched, mp, w.matched_ours, w.ours_match_draws, Pct(w.matched_ours, w.ours_match_draws), w.matched_inline, w.inline_match_draws,
                Pct(w.matched_inline, w.inline_match_draws), w.pairs_cam_changed, Pct(w.pairs_cam_changed, w.matched), w.pairs_model_changed, Pct(w.pairs_model_changed, w.matched),
                w.pairs_other_changed_draws, w.pairs_bone_rows, w.pairs_bone_draws4, w.matched ? double(w.pairs_rows_total) / double(w.matched) : 0.0);
    std::vector<std::pair<uint32_t, uint64_t>> rows;
    for (uint32_t r = 0; r < kRows; ++r) if (w.row_changed[r]) rows.emplace_back(r, w.row_changed[r]);
    std::sort(rows.begin(), rows.end(), [](const auto& x, const auto& y) { return x.second > y.second; });
    std::string line;
    for (size_t k = 0; k < std::min<size_t>(rows.size(), 16); ++k) line += fmt::format(" c{}:{}", rows[k].first, rows[k].second);
    REXGPU_INFO("[d3d][phase0][match] constant rows changed between matched draws (all {} matched pairs), top 16 rows:{}", w.matched, line.empty() ? " none" : line);
    std::vector<std::pair<uint64_t, const VsStat*>> vs;
    for (const auto& kv : w.vs) if (kv.second.matched) vs.emplace_back(kv.first, &kv.second);
    std::sort(vs.begin(), vs.end(), [](const auto& x, const auto& y) { return x.second->matched > y.second->matched; });
    for (size_t k = 0; k < std::min<size_t>(vs.size(), 16); ++k) {
      const VsStat& v = *vs[k].second;
      std::array<uint32_t, kRows> half{};
      for (uint32_t r = 0; r < kRows; ++r) half[r] = (v.row_changed[r] * 2 >= v.matched) ? 1 : 0;
      REXGPU_INFO("[d3d][phase0][match]   vs {:016X} {}: matched {}/{} ({:.1f}%), c56-c59 changed {:.1f}%, c76-c78 changed {:.1f}%, mean changed rows {:.1f}, rows changed in 50%+ of its pairs: {}; vertex data sample changed {}/{}",
                  vs[k].first, Kind(v.ours), v.matched, v.draws, Pct(v.matched, v.draws), Pct(v.cam_changed, v.matched), Pct(v.model_changed, v.matched),
                  v.matched ? double(v.changed_rows) / double(v.matched) : 0.0, RowRanges(half, 1, 10), v.vb_changed, v.vb_checked);
    }
    REXGPU_INFO("[d3d][phase0][camera] distinct non-zero c56-c59 blocks a frame avg {:.1f} (max {}), blocks carried by a matched pair avg {:.1f}; our draws carrying a matched block {}/{} ({:.1f}%), translated draws carrying one by value "
                "{}/{} ({:.1f}%; the registers persist, so this is an upper bound for draws that do not use the camera), draws with an all-zero block {}; camera cuts a minute: not computed",
                double(w.cam_distinct_sum) / fr, w.cam_distinct_max, double(w.cam_mapped_sum) / fr, w.cam_ours_covered, w.cam_ours, Pct(w.cam_ours_covered, w.cam_ours), w.cam_tr_covered, w.cam_tr,
                Pct(w.cam_tr_covered, w.cam_tr), w.cam_zero_draws);
    REXGPU_INFO("[d3d][phase0][vbdata] matched draws with a vertex-data sample compared {}: changed {} ({:.1f}%, ours {}); a lower bound of CPU-written vertex data (skinned meshes, particles; 3 x 512 bytes at the range's end)",
                w.vb_checked, w.vb_changed, Pct(w.vb_changed, w.vb_checked), w.vb_changed_ours);
    const double per_lerp_ns = w.lerps ? double(w.ns_lerp) / double(w.lerps) : 0.0;
    const double changed_rows = w.matched ? double(w.pairs_rows_total) / double(w.matched) : 0.0;
    REXGPU_INFO("[d3d][phase0][cost] per frame: match lookup + row compare {:.1f} us over {:.0f} draws, lerp of 7 vec4 {:.1f} us over {:.0f} matched draws ({:.0f} ns each, i.e. {:.0f} ns per row); estimate of a full blend (every changed row, "
                "{:.1f} per matched draw) {:.1f} us a frame; census total {:.1f} us a frame",
                double(w.ns_match) / fr / 1000.0, double(w.match_draws) / fr, double(w.ns_lerp) / fr / 1000.0, double(w.lerps) / fr, per_lerp_ns, per_lerp_ns / 7.0, changed_rows,
                double(w.matched) / fr * changed_rows * (per_lerp_ns / 7.0) / 1000.0, double(w.ns_total) / fr / 1000.0);
    const double cam_pct = Pct(w.cam_ours_covered, w.cam_ours);
    REXGPU_INFO("[d3d][phase0][decouple] go / no-go rule (S93_DECOUPLE_DESIGN.md section 6, applies to an ENTRANCE window): matched {:.1f}% of all draws (rule: >= 90%) {}, camera block found for {:.1f}% of our world-space draws (rule: >= 95%) {} -> {}",
                mp, mp >= 90.0 ? "pass" : "FAIL", cam_pct, cam_pct >= 95.0 ? "pass" : "FAIL", (mp >= 90.0 && cam_pct >= 95.0) ? "GO" : "NO-GO");
  }
}

void ResetWindow() {
  g.w = Window();
  g.w.first_frame = g.frame_no;
  // prune the vertex ranges and meshes not touched in the window that just ended
  const uint64_t cur_window = g.frame_no / kWindowFrames;
  for (auto it = g.vb.begin(); it != g.vb.end();) it = (it->second.window + 1 < cur_window) ? g.vb.erase(it) : std::next(it);
  for (auto it = g.meshes.begin(); it != g.meshes.end();) it = (it->second.window + 1 < cur_window) ? g.meshes.erase(it) : std::next(it);
}

}  // namespace

// =====================================================================================================================================================
void Phase0Draw(rex::memory::Memory* mem, const OwnedRecordState& s, const VulkanShader* vs, bool crowd_member) {
  const uint64_t t_start = NowNs();
  Window& w = g.w;
  if (!g.started) {
    g.started = true;
    w.first_frame = g.frame_no;
  }
  ++w.draws;
  ++g.frame_draws;
  if (crowd_member) {
    ++w.crowd_members;
    w.ns_total += NowNs() - t_start;
    return;
  }
  const bool inl = s.inline_draw != 0;
  const uint64_t vs_hash = s.shader[0].hash, ps_hash = s.shader[1].hash;
  if (vs && vs->ucode_data_hash() != vs_hash) vs = nullptr;
  if (!vs) ++w.no_vs;
  (inl ? w.inline_draws : w.dma_draws) += 1;
  const bool ours = shader_swap::VertexTakes(vs_hash);
  if (ours) ++w.ours_draws;

  // ---- the pass
  const uint64_t pass_key = Mix(Mix(s.color_info[0].value, s.depth_info.value), s.surface_info.value);
  PassStat& pass = w.passes[pass_key];
  if (!g.pass_valid || pass_key != g.cur_pass_key) {
    g.cur_pass_key = pass_key;
    g.pass_valid = true;
    ++g.pass_ord;
    ++w.pass_entries;
    ++pass.entries;
    pass.ord_sum += g.pass_ord;
    pass.color = s.color_info[0].value;
    pass.depth = s.depth_info.value;
    pass.surface = s.surface_info.value;
  }
  ++pass.draws;
  ++pass.vs_draws[vs_hash];
  VsStat& vstat = w.vs[vs_hash];
  ++vstat.draws;
  vstat.ours = ours;
  vstat.bindings = vs ? uint32_t(vs->vertex_bindings().size()) : 0;

  // ---- the vertex constants' row hashes (recomputed only when the decoder's epoch moved)
  const OwnedRecordEpochs& ep = OwnedRecordEpochsNow();
  if (!g.pool_valid || ep.vs_float != g.last_vs_epoch) {
    g.last_pool_off = uint32_t(g.cur_pool.size());
    g.cur_pool.resize(g.cur_pool.size() + kRows);
    uint32_t* out = g.cur_pool.data() + g.last_pool_off;
    for (uint32_t r = 0; r < kRows; ++r) out[r] = RowHash(s.vs_float + 4 * r);
    g.last_vs_epoch = ep.vs_float;
    g.pool_valid = true;
  }
  const uint32_t pool_off = g.last_pool_off;

  // ---- the vertex ranges (mirror) + the index scan / box
  uint64_t key = Mix(Mix(vs_hash, ps_hash), Mix(pass_key, inl ? 1u : 0u));
  uint64_t vb_sample = 0;
  uint32_t first_start = 0, first_size = 0, first_stride = 0;
  xenos::Endian first_endian = xenos::Endian::kNone;
  bool have_vb = false;
  uint32_t first_addr_dword = 0;
  if (vs && !inl) {
    for (const auto& binding : vs->vertex_bindings()) {
      if (binding.fetch_constant >= 96) continue;
      xenos::xe_gpu_vertex_fetch_t f;
      std::memcpy(&f, &s.fetch[binding.fetch_constant * 2], sizeof(f));
      ++w.fetches;
      if (f.type != xenos::FetchConstantType::kVertex) {
        ++w.fetch_not_vertex;
        continue;
      }
      const uint32_t start = f.address * 4, bytes = f.size * 4;
      key = Mix(key, f.dword_0);
      if (!bytes) continue;
      VbEntry& e = TouchVb(mem, start, start + bytes, ours);
      if (!have_vb) {
        have_vb = true;
        vb_sample = e.sample;
        first_start = start;
        first_size = bytes;
        first_stride = binding.stride_words;
        first_endian = f.endian;
        first_addr_dword = f.dword_0;
      }
    }
  }
  if (!inl) key = Mix(Mix(key, s.dma_base), s.draw_initiator.num_indices);

  Mesh* mesh = nullptr;
  uint32_t base_vertex = s.indx_offset.indx_offset;
  if (!inl && s.draw_initiator.source_select == xenos::SourceSelect::kDMA) {
    uint64_t mkey = Mix(Mix(Mix(s.dma_base, s.draw_initiator.num_indices), uint64_t(s.dma_size.swap_mode) << 8 | uint64_t(s.draw_initiator.index_size)),
                        Mix(Mix(base_vertex, first_addr_dword), vb_sample));
    Mesh& m = ScanMesh(mem, s, mkey, nullptr);
    if (m.ok) {
      mesh = &m;
      if (!m.used_bytes && have_vb) {
        m.used_bytes = uint32_t(std::min<uint64_t>(uint64_t(m.max_i - m.min_i + 1) * first_stride * 4, 0xFFFFFFFFull));
      }
      // the design's index identity checks (3.4 point 5)
      ++w.id_checked;
      const uint32_t vmin = s.min_vtx_indx.min_indx, vmax = s.max_vtx_indx.max_indx;
      const bool f_min = m.min_i + base_vertex < vmin, f_max = m.max_i + base_vertex > vmax, f_mask = uint64_t(m.max_i) + base_vertex >= (1u << 24);
      w.id_fail_min += f_min;
      w.id_fail_max += f_max;
      w.id_fail_mask += f_mask;
      w.id_fail_any += (f_min || f_max || f_mask);
      ++w.tight_checked;
      if (vmax >= vmin && (vmax - vmin) <= (m.max_i - m.min_i) + 16) ++w.tight_regs;
    }
  }

  // ---- off-screen (our arena family)
  if (ours) {
    const bool shape = vs && vs->vertex_bindings().size() == 1 && vs->vertex_bindings()[0].stride_words == 8 && have_vb && !inl;
    if (!shape) {
      ++w.ours_not_boxable_shape;
    } else if (!mesh) {
      ++w.no_bounds;
    } else {
      if (!mesh->box_done) BuildBox(mem, *mesh, first_start, first_size, first_stride, base_vertex, first_endian);
      if (!mesh->box_ok) {
        ++w.no_bounds;
      } else {
        ++w.boxable;
        ++vstat.boxable;
        ++pass.boxable;
        ++w.tested;
        const int r = draw_bounds::BoxTest(s.vs_float, mesh->box);
        if (r == 1) {
          ++w.off_xy;
        } else if (r == 2) {
          ++w.off_behind;
        } else if (r == 3) {
          ++w.off_z_only;
        }
        if (r == 1 || r == 2) {
          ++vstat.offscreen;
          ++pass.offscreen;
        }
      }
    }
  }

  // ---- the merge census (consecutive draws)
  {
    Last& L = g.last;
    const bool dma = !inl;
    const uint32_t isz = s.draw_initiator.index_size == xenos::IndexFormat::kInt16 ? 2 : 4;
    const uint32_t dma_end = s.dma_base + s.draw_initiator.num_indices * isz;
    if (L.valid) {
      ++w.consecutive;
      const bool same_shaders = L.vs == vs_hash && L.ps == ps_hash;
      const bool same_nonvs = L.e_render == ep.render && L.e_ps == ep.ps_float && L.e_bl == ep.bool_loop && L.e_fetch == ep.fetch;
      const bool same_vs_float = L.e_vs == ep.vs_float;
      bool continued = false;
      if (same_shaders && same_nonvs && same_vs_float && dma && L.dma) {
        ++w.state_identical;
        const bool contiguous = L.dma_end == s.dma_base && L.base == base_vertex && L.fmt == uint32_t(s.draw_initiator.index_size) && L.swap == uint32_t(s.dma_size.swap_mode) &&
                                L.prim == uint32_t(s.draw_initiator.prim_type);
        if (contiguous) {
          ++w.contiguous;
          const bool tri = s.draw_initiator.prim_type == xenos::PrimitiveType::kTriangleList;
          w.contiguous_tri += tri;
          if (tri) {
            ++L.run;
            ++w.saved;
            continued = true;
          }
        } else {
          ++w.same_state_gap;
        }
      } else if (same_shaders && same_nonvs && !same_vs_float) {
        ++w.vsfloat_only;
        bool only_model = true;
        const uint32_t* a = g.cur_pool.data() + L.pool_off;
        const uint32_t* b = g.cur_pool.data() + pool_off;
        for (uint32_t r = 0; r < kRows && only_model; ++r) {
          if (a[r] != b[r] && (r < kModelRow || r > kModelRow + 2)) only_model = false;
        }
        w.model_only += only_model;
      }
      if (!continued && L.run) {  // a run of contiguous draws ends: L.run follower draws were merged into its first
        ++w.runs;
        const uint32_t len = L.run + 1;
        ++w.run_hist[len <= 2 ? 0 : len <= 4 ? 1 : len <= 8 ? 2 : 3];
        L.run = 0;
      }
    }
    const uint32_t keep_run = L.run;
    L.valid = true;
    L.vs = vs_hash;
    L.ps = ps_hash;
    L.e_render = ep.render;
    L.e_vs = ep.vs_float;
    L.e_ps = ep.ps_float;
    L.e_bl = ep.bool_loop;
    L.e_fetch = ep.fetch;
    L.dma_end = dma_end;
    L.base = base_vertex;
    L.fmt = uint32_t(s.draw_initiator.index_size);
    L.swap = uint32_t(s.dma_size.swap_mode);
    L.prim = uint32_t(s.draw_initiator.prim_type);
    L.pool_off = pool_off;
    L.dma = dma;
    L.run = keep_run;
  }

  // ---- the draw record + the match with the previous frame
  const uint64_t t_match = NowNs();
  Rec rec;
  rec.key = key;
  rec.vs = vs_hash;
  rec.ps = ps_hash;
  rec.ours = ours;
  rec.inl = inl;
  rec.pool_off = pool_off;
  rec.vb_sample = vb_sample;
  std::memcpy(rec.mat, s.vs_float + 4 * kCamRow, 16 * sizeof(uint32_t));
  std::memcpy(rec.mat + 16, s.vs_float + 4 * kModelRow, 12 * sizeof(uint32_t));
  uint64_t ch = 0x51ED;
  bool zero = true;
  for (int i = 0; i < 16; ++i) {
    uint32_t bits;
    std::memcpy(&bits, &rec.mat[i], 4);
    ch = Mix(ch, bits);
    zero &= bits == 0;
  }
  rec.cam_hash = ch;
  rec.cam_zero = zero;
  rec.ord = g.cur_ord[key]++;
  const uint64_t mk = Mix(key, rec.ord);
  g.cur_map[mk] = uint32_t(g.cur.size());
  if (!g.prev.empty()) {
    ++w.match_draws;
    if (ours) ++w.ours_match_draws;
    if (inl) ++w.inline_match_draws;
    const auto it = g.prev_map.find(mk);
    if (it != g.prev_map.end()) {
      const Rec& pr = g.prev[it->second];
      rec.matched = int32_t(it->second);
      ++w.matched;
      ++vstat.matched;
      ++pass.matched;
      if (ours) ++w.matched_ours;
      if (inl) ++w.matched_inline;
      const uint32_t* a = g.prev_pool.data() + pr.pool_off;
      const uint32_t* b = g.cur_pool.data() + pool_off;
      bool cam = false, model = false;
      uint32_t other = 0, bone = 0, total = 0;
      for (uint32_t r = 0; r < kRows; ++r) {
        if (a[r] == b[r]) continue;
        ++total;
        ++w.row_changed[r];
        ++vstat.row_changed[r];
        if (r >= kCamRow && r < kCamRow + 4) {
          cam = true;
        } else if (r >= kModelRow && r < kModelRow + 3) {
          model = true;
        } else {
          ++other;
          if (r >= kModelRow + 3) ++bone;
        }
      }
      w.pairs_cam_changed += cam;
      vstat.cam_changed += cam;
      w.pairs_model_changed += model;
      vstat.model_changed += model;
      w.pairs_other_changed_draws += other != 0;
      w.pairs_bone_rows += bone;
      w.pairs_bone_draws4 += bone >= 4;
      w.pairs_rows_total += total;
      vstat.changed_rows += total;
      if (pr.vb_sample && rec.vb_sample) {
        ++w.vb_checked;
        ++vstat.vb_checked;
        if (pr.vb_sample != rec.vb_sample) {
          ++w.vb_changed;
          ++vstat.vb_changed;
          w.vb_changed_ours += ours;
        }
      }
    }
  }
  w.ns_match += NowNs() - t_match;
  g.cur.push_back(rec);
  w.ns_total += NowNs() - t_start;
}

void Phase0FrameEnd() {
  if (!REXCVAR_GET(gpu_d3d_phase0_census)) return;
  const uint64_t t0 = NowNs();
  Window& w = g.w;
  if (!g.started) {
    g.started = true;
    w.first_frame = g.frame_no;
  }
  if (g.frame_draws) {
    ++w.frames_with_draws;
    w.frame_draw_max = std::max(w.frame_draw_max, g.frame_draws);
    // the mirror's per-frame union
    const uint64_t u = UnionBytes(g.frame_ranges);
    w.frame_union_sum += u;
    w.frame_union_max = std::max(w.frame_union_max, u);

    // the camera block coverage + the lerp timing (on the matched pairs; the result goes to a sink, nothing is drawn)
    std::unordered_set<uint64_t> blocks, mapped;
    for (const Rec& r : g.cur) {
      if (r.cam_zero) continue;
      blocks.insert(r.cam_hash);
      if (r.matched >= 0) mapped.insert(r.cam_hash);
    }
    w.cam_distinct_sum += blocks.size();
    w.cam_distinct_max = std::max<uint64_t>(w.cam_distinct_max, blocks.size());
    w.cam_mapped_sum += mapped.size();
    for (const Rec& r : g.cur) {
      if (r.inl) continue;
      if (r.cam_zero) {
        ++w.cam_zero_draws;
        continue;
      }
      const bool covered = mapped.count(r.cam_hash) != 0;
      if (r.ours) {
        ++w.cam_ours;
        w.cam_ours_covered += covered;
      } else {
        ++w.cam_tr;
        w.cam_tr_covered += covered;
      }
    }
    const uint64_t t_lerp = NowNs();
    volatile float sink = 0;
    float acc = 0;
    uint64_t n = 0;
    for (const Rec& r : g.cur) {
      if (r.matched < 0) continue;
      const Rec& p = g.prev[size_t(r.matched)];
      float out[28];
      for (int i = 0; i < 28; ++i) out[i] = p.mat[i] + (r.mat[i] - p.mat[i]) * 0.5f;
      acc += out[0] + out[27];
      ++n;
    }
    sink = acc;
    (void)sink;
    w.ns_lerp += NowNs() - t_lerp;
    w.lerps += n;
  }
  // the frame becomes the previous one
  g.prev.swap(g.cur);
  g.prev_pool.swap(g.cur_pool);
  g.prev_map.swap(g.cur_map);
  g.cur.clear();
  g.cur_pool.clear();
  g.cur_map.clear();
  g.cur_ord.clear();
  g.frame_ranges.clear();
  g.frame_draws = 0;
  g.pool_valid = false;
  g.pass_valid = false;
  g.pass_ord = 0;
  g.scan_budget = kScanBudget;
  g.last.valid = false;
  if (g.last.run) {  // (a run open at the frame's end)
    ++w.runs;
    const uint32_t len = g.last.run + 1;
    ++w.run_hist[len <= 2 ? 0 : len <= 4 ? 1 : len <= 8 ? 2 : 3];
    g.last.run = 0;
  }
  w.ns_frame_end += NowNs() - t0;
  w.ns_total += NowNs() - t0;
  ++g.frame_no;
  if (g.frame_no % kWindowFrames == 0) {
    Flush();
    ResetWindow();
  }
}

}  // namespace rex::graphics::vulkan
