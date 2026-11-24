// wwe2k14 patch (Session 93): the object-space bounds of a draw's mesh and the conservative off-screen test, shared by the phase 0 census
// (phase0_census.cpp, log only) and the draw cut of our arena draws (cull_ours.cpp, gpu_d3d_cull_ours). Pure functions over guest memory and the typed
// vertex float bank; no state of their own (the caller owns the caches and the scan budget). Moved out of the census unchanged: the census' numbers do not move.
#pragma once

#include <cstdint>

#include <rex/graphics/xenos.h>

namespace rex::memory {
class Memory;
}

namespace rex::graphics::vulkan {

struct OwnedRecordState;

namespace draw_bounds {

enum class ScanStatus { kOk, kUnsupported /* never retry */, kBudget /* retry next frame */, kEmpty /* no valid index: never retry */ };
struct IndexScan {
  uint32_t n = 0;  // the indices read (counted by the census)
  uint32_t min_i = 0, max_i = 0;
  bool resets = false;
};
// Reads the draw's DMA indices once (the draw's own words in `s`); `budget` is the caller's per-frame allowance of indices + vertices read (decremented).
ScanStatus ScanIndices(rex::memory::Memory* mem, const OwnedRecordState& s, uint32_t& budget, IndexScan& out);

struct Box {
  float lo[3] = {}, hi[3] = {};
};
enum class BoxStatus { kOk, kWNotOne /* a vertex with w != 1: not cullable */, kRejected /* out of range, too many vertices, non-finite: never retry */, kBudget /* retry */ };
// The box of the position dwords 0-3 (big-endian floats after `endian`) of vertices [min_i + base, max_i + base] of a stream with `stride_words` dwords a vertex that starts
// at guest physical `vb_start_byte` and is `vb_size_bytes` long. `vertices_read` = the vertices read (0 when none).
BoxStatus BuildBox(rex::memory::Memory* mem, uint32_t min_i, uint32_t max_i, uint32_t vb_start_byte, uint32_t vb_size_bytes, uint32_t stride_words, uint32_t base,
                   xenos::Endian endian, uint32_t& budget, Box& out, uint32_t& vertices_read);

// 0 = visible / unknown, 1 = outside an x / y plane, 2 = behind the eye, 3 = only z > w (not counted as culled). `vs_float` = the typed vertex float bank (vec4 constants):
// the 8 corners go through c76-c78 (model) and c56-c59 (view-projection); a 1% margin.
// `host` (optional) = the draw's host viewport transform {scale x, scale y, offset x, offset y} (draw_util::ViewportInfo ndc_scale / ndc_offset): the planes are then tested
// on x' = scale x * x + offset x * w, y' likewise - the clip space the host really clips in (a sign flip or a tiling strip's scale changes nothing about the guest's x / y planes
// being symmetric, but a scale != 1 or an offset moves them).
int BoxTest(const uint32_t* vs_float, const Box& box, const float* host = nullptr);

}  // namespace draw_bounds
}  // namespace rex::graphics::vulkan
