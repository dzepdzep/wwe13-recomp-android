// wwe2k14 patch (Session 93, idea 13): see cull_ours.h.
//
// A draw is skipped only when ALL of these hold (each is a counted refusal, [d3d][cull] window line):
//  - gpu_d3d_cull_ours on; a draw item of the cut (the caller), not a crowd member, not an inline draw (a DMA index buffer);
//  - its vertex shader is one of ours AND position-exact (shader_swap::VertexBoxCullable: the plain arena materials; not skinned, not bump, not particles);
//  - the vertex transform is the plain one (VTE: xy not pre-divided, w not reciprocal), the primitive a triangle list / strip / fan / quad list (a rectangle list's 4th
//    vertex is not in the box, points and lines expand);
//  - no occlusion query and no viz query on the draw; neither shader exports to memory (both analysed, the caller);
//  - the renderer's pass is already open and the previous drawn draw had the same render-target words, with no break between (a refused / old-path / inline draw, a clear or
//    resolve, the end of the renderer's pass: NoteBreak); a skipped first draw never decides a pass's structure, and a frame's first draw is never skipped;
//  - the vertex shader has one binding, in the registered fetch constant, 8 dwords a vertex (the caller); all bytes read lie in the 512 MiB physical space;
//  - the mesh's bounds are known: the box of the vertices [min, max] + base the indices name (scanned once per mesh identity = index address + count + format + base vertex +
//    the vertex fetch constant's two words), all w == 1; a mesh seen for the first time is drawn that time; re-checked once a frame against a sample of the index and vertex
//    bytes (a change = a new scan, the draw is drawn), and re-scanned in full every 600 frames (staggered);
//  - the test proves invisibility: all 8 corners outside the same x or y clip plane (1% margin) or all behind the eye; z > w alone never culls;
//  - the host clip space of the draw is the guest's (the caller's viewport callback, required: the test is repeated in the host viewport's clip space, ndc scale / offset on x / y; no wide / tall factor).
// The caller keeps the typed / bound state as the other cut skips do (repeat base and lean base invalidated).
#include "cull_ours.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>

#include <rex/cvar.h>
#include <rex/graphics/xenos.h>
#include <rex/hash.h>
#include <rex/logging.h>
#include <rex/system/xmemory.h>

#include "draw_bounds.h"
#include "owned_record.h"
#include "shader_swap.h"

REXCVAR_DEFINE_BOOL(gpu_d3d_cull_ours, false, "GPU/Native",
                    "Session 93 (idea 13, docs\\native\\d3d\\S93_MIRROR_DESIGN.md section 4, needs the cut and gpu_native_vs_swap): a draw of OUR arena vertex shaders (the plain "
                    "materials) whose mesh box a conservative test proves outside the view (all corners beyond the same x / y clip plane, or behind the eye) is not drawn. "
                    "Never a draw with an occlusion / viz query, memexport, a crowd member, an inline draw, the first draw of a pass, or a mesh whose bounds are not known yet. "
                    "[d3d][cull] lines per 600 guest frames. Off = nothing runs.");

namespace rex::graphics::vulkan::cull_ours {

namespace {

constexpr uint32_t kWindowFrames = 600;
constexpr uint32_t kRescanFrames = 600;
constexpr uint32_t kBudgetPerFrame = 400000;  // indices + vertices read per guest frame

inline uint64_t Mix(uint64_t a, uint64_t b) {
  a ^= b + 0x9E3779B97F4A7C15ull + (a << 6) + (a >> 2);
  return a;
}
inline double Pct(uint64_t a, uint64_t b) { return b ? 100.0 * double(a) / double(b) : 0.0; }

struct Entry {
  bool scanned = false;  // the indices were read (min_i / max_i valid)
  bool no_box = false;   // not boxable (range, non-finite, unsupported indices): not retried until the next full rescan
  bool w_not_one = false;
  bool has_box = false;
  uint32_t min_i = 0, max_i = 0;
  uint64_t born = 0;              // the frame of the last full scan
  uint64_t checked = UINT64_MAX;  // the last frame the sample was compared
  uint64_t sample = 0;
  draw_bounds::Box box;
};

struct Window {
  uint64_t first_frame = 0, frames_with_draws = 0;
  uint64_t ours = 0, cullable_shader = 0, not_cullable_shader = 0;
  uint64_t r_crowd = 0, r_inline = 0, r_vte = 0, r_prim = 0, r_occlusion = 0, r_shaders = 0, r_pass = 0, r_fetch = 0, r_no_bounds = 0, r_w_not_one = 0, r_viewport = 0;
  uint64_t tested = 0, visible = 0, z_only = 0, culled_xy = 0, culled_behind = 0;
  uint64_t scans = 0, rescans = 0, sample_changes = 0, budget_skips = 0, scan_indices = 0, box_vertices = 0;
};

struct ShaderInfo {
  bool ours = false, cullable = false;
  uint32_t fetch = 95;
};

struct State {
  uint64_t frame_no = 0;
  Window w;
  bool started = false;
  uint32_t budget = kBudgetPerFrame;
  bool last_valid = false;
  uint64_t last_pass_key = 0;
  std::unordered_map<uint64_t, Entry> meshes;
  std::unordered_map<uint64_t, ShaderInfo> shaders;
  uint64_t shaders_generation = 0;
  uint64_t frame_draws = 0;
};
State g;

uint64_t PassKey(const OwnedRecordState& s) { return Mix(Mix(s.color_info[0].value, s.depth_info.value), s.surface_info.value); }

// A sample of what the box was made from: 128 bytes of the index buffer at 3 places and up to 4 used vertices at 3 places (position dwords).
uint64_t Sample(rex::memory::Memory* mem, const OwnedRecordState& s, uint32_t vb_start, uint32_t vb_size, uint32_t stride_bytes, uint32_t first, uint32_t last) {
  uint64_t h = 0x9876543;
  const uint32_t isz = s.draw_initiator.index_size == xenos::IndexFormat::kInt16 ? 2 : 4;
  uint32_t n = s.draw_initiator.num_indices;
  n = std::min(n, s.dma_size.num_words * (4 / isz));
  const uint64_t ibytes = uint64_t(n) * isz;
  const uint32_t ibase = s.dma_base & ~(isz - 1);
  for (int k = 0; k < 3; ++k) {
    const uint64_t off = ibytes > 128 ? (ibytes - 128) * uint64_t(k) / 2 : 0;
    const uint32_t len = uint32_t(std::min<uint64_t>(128, ibytes - off));
    if (len) h = Mix(h, XXH3_64bits(mem->TranslatePhysical<const uint8_t*>(ibase + uint32_t(off)), len));
  }
  const uint64_t vcount = uint64_t(last) - first + 1;
  for (int k = 0; k < 3; ++k) {  // up to 4 used vertices at 3 places; of each the position dwords 0-3 (what the box is made from)
    const uint64_t v0 = uint64_t(first) + (vcount > 4 ? (vcount - 4) * uint64_t(k) / 2 : 0);
    const uint64_t cnt = std::min<uint64_t>(4, uint64_t(first) + vcount - v0);
    for (uint64_t j = 0; j < cnt; ++j) {
      const uint64_t off = (v0 + j) * stride_bytes;
      if (off + 16 <= vb_size) h = Mix(h, XXH3_64bits(mem->TranslatePhysical<const uint8_t*>(vb_start + uint32_t(off)), 16));
    }
  }
  return h | 1;
}

void Print() {
  const Window& w = g.w;
  const double fr = double(std::max<uint64_t>(1, w.frames_with_draws));
  const uint64_t culled = w.culled_xy + w.culled_behind;
  REXGPU_INFO(
      "[d3d][cull] frames {}-{}: our draws {} ({:.1f} a frame), shader cullable {} / not {}; refused: crowd member {}, inline {}, vte format {}, primitive {}, occlusion / viz {}, shaders unknown / memexport {}, "
      "pass not open / first draw {}, fetch not a vertex stream {}, bounds not known yet {}, w != 1 {}, viewport {}; tested {}: culled {} ({:.1f}% of tested, {:.1f} a frame; off-screen x/y {}, behind the eye {}), "
      "visible {} (z > w only {}); mesh scans {} ({} indices, {} vertices; full rescans {}, sample changes {}, budget skips {}), meshes cached {}",
      w.first_frame, g.frame_no, w.ours, double(w.ours) / fr, w.cullable_shader, w.not_cullable_shader, w.r_crowd, w.r_inline, w.r_vte, w.r_prim, w.r_occlusion, w.r_shaders, w.r_pass, w.r_fetch,
      w.r_no_bounds, w.r_w_not_one, w.r_viewport, w.tested, culled, Pct(culled, w.tested), double(culled) / fr, w.culled_xy, w.culled_behind, w.visible, w.z_only, w.scans, w.scan_indices,
      w.box_vertices, w.rescans, w.sample_changes, w.budget_skips, g.meshes.size());
}

}  // namespace

void NoteDrawn(const OwnedRecordState& s) {
  g.last_valid = true;
  g.last_pass_key = PassKey(s);
}

namespace {
const ShaderInfo* g_shader_last = nullptr;  // (Session 95)
uint64_t g_shader_last_hash = 0;
}  // namespace
const ShaderInfo& ShaderOf(uint64_t vs_hash) {
  if (g.shaders_generation != shader_swap::TranslationGeneration()) {
    g.shaders.clear();
    g.shaders_generation = shader_swap::TranslationGeneration();
    g_shader_last = nullptr;
  }
  if (g_shader_last && g_shader_last_hash == vs_hash) return *g_shader_last;  // (Session 95: draws of one shader come in runs; unordered_map nodes stay put)
  auto sit = g.shaders.find(vs_hash);
  if (sit == g.shaders.end()) {
    ShaderInfo info;
    info.ours = shader_swap::VertexTakes(vs_hash);
    std::string why;
    info.cullable = shader_swap::VertexBoxCullable(vs_hash, &info.fetch, &why);
    REXGPU_INFO("[d3d][cull] shader {:016X} {}: cullable {} (reason: {})", vs_hash, info.ours ? "ours" : "not ours", info.cullable ? "yes" : "no", why.empty() ? "not ours" : why);
    sit = g.shaders.emplace(vs_hash, info).first;
  }
  g_shader_last = &sit->second;
  g_shader_last_hash = vs_hash;
  return sit->second;
}

bool CullableShader(uint64_t vs_hash) { return ShaderOf(vs_hash).cullable; }
uint32_t CullableFetch(uint64_t vs_hash) { return ShaderOf(vs_hash).fetch; }

void NoteBreak() { g.last_valid = false; }

bool Evaluate(const Request& r) {
  const OwnedRecordState& s = *r.s;
  Window& w = g.w;
  if (!g.started) {
    g.started = true;
    w.first_frame = g.frame_no;
  }
  ++g.frame_draws;
  const uint64_t vs_hash = s.shader[0].hash;
  const ShaderInfo& si = ShaderOf(vs_hash);
  if (!si.ours) return false;
  ++w.ours;
  if (!si.cullable) return ++w.not_cullable_shader, false;
  ++w.cullable_shader;
  if (r.crowd_member) return ++w.r_crowd, false;
  if (s.inline_draw || s.draw_initiator.source_select != xenos::SourceSelect::kDMA) return ++w.r_inline, false;
  if (s.vte_cntl.vtx_xy_fmt || !s.vte_cntl.vtx_w0_fmt) return ++w.r_vte, false;
  switch (s.draw_initiator.prim_type) {
    case xenos::PrimitiveType::kTriangleList:
    case xenos::PrimitiveType::kTriangleStrip:
    case xenos::PrimitiveType::kTriangleFan:
    case xenos::PrimitiveType::kQuadList:
      break;
    default:
      return ++w.r_prim, false;
  }
  if (r.occlusion_query || s.viz_query_cntl.viz_query_ena || s.draw_viz_query) return ++w.r_occlusion, false;
  if (!r.shaders_known) return ++w.r_shaders, false;
  if (!r.truth_pass_open || !g.last_valid || g.last_pass_key != PassKey(s)) return ++w.r_pass, false;

  // ---- the vertex stream (our shaders read one stream, 8 dwords a vertex, in the registered fetch constant)
  if (si.fetch >= 96) return ++w.r_fetch, false;
  xenos::xe_gpu_vertex_fetch_t f;
  std::memcpy(&f, &s.fetch[si.fetch * 2], sizeof(f));
  if (f.type != xenos::FetchConstantType::kVertex || !f.size) return ++w.r_fetch, false;
  const uint32_t vb_start = f.address * 4, vb_size = f.size * 4;
  const uint32_t base = s.indx_offset.indx_offset;

  // ---- the mesh's bounds
  const uint32_t isz = s.draw_initiator.index_size == xenos::IndexFormat::kInt16 ? 2 : 4;
  // (everything read below must lie in the 512 MiB physical space)
  const uint32_t n_avail = std::min(s.draw_initiator.num_indices, s.dma_size.num_words * (4 / isz));
  if (uint64_t(f.address) * 4 + uint64_t(f.size) * 4 > 0x20000000ull || uint64_t(s.dma_base & ~(isz - 1)) + uint64_t(n_avail) * isz > 0x20000000ull) return ++w.r_fetch, false;
  const uint64_t key = Mix(Mix(Mix(s.dma_base, s.draw_initiator.num_indices), uint64_t(s.dma_size.swap_mode) << 8 | uint64_t(isz)),
                           Mix(Mix(base, s.fetch[si.fetch * 2]), s.fetch[si.fetch * 2 + 1]));
  Entry& e = g.meshes[key];
  bool fresh = false;
  if ((e.scanned || e.no_box) && g.frame_no - e.born >= kRescanFrames + (key & 63)) {  // the periodic full rescan (staggered)
    e = Entry();
    ++w.rescans;
  }
  if (e.has_box && e.checked != g.frame_no) {
    // the once-a-frame check of the data the box was made from
    const uint64_t sample = Sample(r.mem, s, vb_start, vb_size, 32, e.min_i + base, e.max_i + base);
    e.checked = g.frame_no;
    if (sample != e.sample) {
      ++w.sample_changes;
      e = Entry();
    }
  }
  if (!e.scanned && !e.no_box && n_avail > kBudgetPerFrame) e.no_box = true;  // (can never fit the per-frame allowance)
  if (!e.scanned && !e.no_box) {
    draw_bounds::IndexScan sc;
    const draw_bounds::ScanStatus st = draw_bounds::ScanIndices(r.mem, s, g.budget, sc);
    if (st == draw_bounds::ScanStatus::kBudget) return ++w.budget_skips, ++w.r_no_bounds, false;
    ++w.scans;
    w.scan_indices += sc.n;
    e.born = g.frame_no;
    e.checked = g.frame_no;
    if (st != draw_bounds::ScanStatus::kOk) {
      e.no_box = true;
    } else {
      e.scanned = true;  // (kept when only the box waits for the budget: then only BuildBox is retried)
      e.min_i = sc.min_i;
      e.max_i = sc.max_i;
    }
    fresh = true;  // (the first time a mesh is seen: its bounds are only built now - drawn this time)
  }
  if (e.scanned && !e.has_box && !e.no_box) {
    if (uint64_t(e.max_i) - e.min_i + 1 > kBudgetPerFrame) {
      e.no_box = true;
    } else {
      uint32_t read = 0;
      const draw_bounds::BoxStatus bs = draw_bounds::BuildBox(r.mem, e.min_i, e.max_i, vb_start, vb_size, 8, base, f.endian, g.budget, e.box, read);
      w.box_vertices += read;
      if (bs == draw_bounds::BoxStatus::kBudget) return ++w.budget_skips, ++w.r_no_bounds, false;
      e.w_not_one = bs == draw_bounds::BoxStatus::kWNotOne;
      e.no_box = bs != draw_bounds::BoxStatus::kOk;
      e.has_box = bs == draw_bounds::BoxStatus::kOk;
      if (e.has_box) e.sample = Sample(r.mem, s, vb_start, vb_size, 32, e.min_i + base, e.max_i + base);
      fresh = true;
    }
  }
  if (e.w_not_one) return ++w.r_w_not_one, false;
  if (!e.has_box) return ++w.r_no_bounds, false;
  if (fresh) return ++w.r_no_bounds, false;

  // ---- the test
  ++w.tested;
  const int t = draw_bounds::BoxTest(s.vs_float, e.box);
  if (t == 3) ++w.z_only;
  if (t != 1 && t != 2) return ++w.visible, false;
  float host[4];
  if (!r.viewport || !r.viewport(r.viewport_ctx, host)) return ++w.r_viewport, false;
  if (t == 1) {  // (behind the eye needs no transform: x' and y' scale with w, which the viewport transform leaves alone)
    // the test again in the clip space the host really clips in (scale / offset of the draw's host viewport); culled only when it still holds
    static uint32_t logged = 0;
    const int h = draw_bounds::BoxTest(s.vs_float, e.box, host);
    if (logged < 8) {
      ++logged;
      REXGPU_INFO("[d3d][cull] viewport sample {}: scale {:.5f} {:.5f} offset {:.5f} {:.5f}: plain test {}, host-space test {}", logged, host[0], host[1], host[2], host[3], t, h);
    }
    if (h != 1 && h != 2) return ++w.r_viewport, false;
  }
  (t == 1 ? w.culled_xy : w.culled_behind) += 1;
  return true;
}

void FrameEnd() {
  ++g.frame_no;
  g.budget = kBudgetPerFrame;
  g.last_valid = false;  // (a new frame's first draw opens its pass: never skipped)
  if (g.frame_draws) ++g.w.frames_with_draws;
  g.frame_draws = 0;
  if (g.started && g.frame_no - g.w.first_frame >= kWindowFrames) {
    Print();
    // entries not touched for two windows go (a map that only grows is a leak across arenas)
    for (auto it = g.meshes.begin(); it != g.meshes.end();) {
      const uint64_t last = it->second.checked == UINT64_MAX ? it->second.born : it->second.checked;
      if (g.frame_no - last > 2 * kWindowFrames) {
        it = g.meshes.erase(it);
      } else {
        ++it;
      }
    }
    g.w = Window();
    g.started = false;
  }
}

}  // namespace rex::graphics::vulkan::cull_ours
