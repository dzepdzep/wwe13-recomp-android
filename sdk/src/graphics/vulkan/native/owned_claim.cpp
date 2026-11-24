// wwe2k14 patch (Session 92, M2d, Part 2): what the owned entry (owned_entry.cpp) and the command processor still take from the truth claim code - moved out of
// native_truth.cpp UNCHANGED so that the truth path can be compiled out (REX_TRUE_NATIVE_ONLY): the claim's reasons (a draw our pipelines cannot take), the repeat / lean
// claim shortcuts (their latch branch calls the owned claims), the draw's rectangle, and the cover / bounds wants.

#include <rex/graphics/vulkan/native/native_renderer.h>
#include "native_truth_reasons.h"

#include <algorithm>
#include <cmath>
#include <string>

#include <rex/cvar.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/logging.h>

REXCVAR_DECLARE(bool, gpu_native_truth_carry_copies);
REXCVAR_DECLARE(std::string, gpu_native_truth_bounds_vs);
REXCVAR_DECLARE(int32_t, gpu_native_truth_debug_draws);
REXCVAR_DECLARE(bool, gpu_d3d_typed_lean_bind);  // (Session 94, B7: defined in owned_front_draw.cpp)

namespace rex::graphics::vulkan::native {

const char* NativeRenderer::TruthReasonName(uint32_t reason) {
  static const char* const names[] = {"ok",           "emulated draw", "clearing resolve", "memexport",
                                      "primitive",    "msaa",          "other targets",    "colour format",
                                      "stencil",      "occlusion",     "no shader",        "no pipeline",
                                      "16-bit blend", "images",        "staging",          "debug"};
  return reason < sizeof(names) / sizeof(names[0]) ? names[reason] : "?";
}

uint32_t NativeRenderer::TruthReason(const Draw& draw, bool check_shader) const {
  if (draw.memexport) return kTruthMemexport;
  if (!draw.raster.supported && !draw.raster.expanded) return kTruthPrimitive;
  if (((draw.surface_info >> 16) & 3) && !draw.fast_clear) return kTruthMsaa;
  if (draw.color_mask_all & ~uint32_t(0xF)) return kTruthOtherTargets;
  const bool writes_color = (draw.color_write_mask & 0xF) != 0;
  if (writes_color && !TruthClass(draw)) return kTruthColorFormat;
  // (A stencil draw: ClaimTruth checks the stencil under it.)
  if (draw.occlusion_query) return kTruthOcclusion;
  if (writes_color && TruthClass(draw) == 2 && PostNeedsDst(draw) && !truth_class2_blend_) return kTruthBlend16;
  if (!draw.vertex_count || draw.vertex_shader == VK_NULL_HANDLE || draw.layout == VK_NULL_HANDLE) {
    return kTruthNoShader;
  }
  if (check_shader) {
    if (draw.pixel_shader_hash && draw.pixel_shader == VK_NULL_HANDLE) return kTruthNoShader;
    if (draw.float24_depth) return kTruthNoShader;  // translated for an exact frame (WantsFloat24Depth)
    if (writes_color && draw.pixel_shader_hash &&
        draw.color_quantize != (PostNeedsDst(draw) ? 0u : TruthClass(draw))) {
      return kTruthNoShader;
    }
  }
  return kTruthOk;
}

bool NativeRenderer::ClaimTruthRepeat(Draw& draw) {
  if (owned_surf_mode_) return OwnedClaimRepeat(draw);  // Session 92, M2b
  if (!draw.truth || !draw.owned || !truth_ || !replacing_ || draw.truth_epoch != truth_claim_epoch_ ||
      draw.occlusion_query || draw.pipeline == VK_NULL_HANDLE || REXCVAR_GET(gpu_native_truth_debug_draws) > 0 ||
      (draw.raster.stencil && !draw.fast_clear)) {  // (review: the stencil check is over the repeat's own rectangle)
    return Claim(draw);
  }
  declined_pending_ = false;
  claim_retry_ = false;
  draw.declined = false;
  ++truth_frame_draws_;
  return true;
}

bool NativeRenderer::ClaimTruthLean(Draw& draw) {
  if (owned_surf_mode_) return OwnedClaimLean(draw);  // Session 92, M2b
  if (!draw.truth || !draw.owned || !truth_ || !replacing_ || draw.truth_epoch != truth_claim_epoch_ ||
      draw.occlusion_query || draw.pipeline == VK_NULL_HANDLE || REXCVAR_GET(gpu_native_truth_debug_draws) > 0 ||
      (draw.raster.stencil && !draw.fast_clear) || draw.fast_clear ||
      draw.xfb_buffer != VK_NULL_HANDLE) {  // Session 61: a captured draw's copy would capture into its range again
    return false;
  }
  declined_pending_ = false;
  claim_retry_ = false;
  draw.declined = false;
  ++truth_frame_draws_;
  return true;
}

void NativeRenderer::NoteTruthInstance() {
  FrameSlot& census = slots_[SlotOf(frame_)];
  ++census.frame_draws;
  ++census.frame_native_post;
  ++truth_counts_.draws;  // (review: the claim counted it in truth_frame_draws_ already)
}

void NativeRenderer::TruthDrawRect(const Draw& d, int32_t rect[4]) const {
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const int32_t sx = d.truth_sx, sy = d.truth_sy;
  int32_t x0 = d.scissor.offset.x * sx, y0 = d.scissor.offset.y * sy;
  int32_t x1 = (d.scissor.offset.x + int32_t(d.scissor.extent.width)) * sx;
  int32_t y1 = (d.scissor.offset.y + int32_t(d.scissor.extent.height)) * sy;
  if (d.fast_clear && d.fill_rect[2] > d.fill_rect[0] && d.fill_rect[3] > d.fill_rect[1]) {
    x0 = std::max(x0, int32_t(d.fill_rect[0] * scale_x));
    y0 = std::max(y0, int32_t(d.fill_rect[1] * scale_y));
    x1 = std::min(x1, int32_t(d.fill_rect[2] * scale_x));
    y1 = std::min(y1, int32_t(d.fill_rect[3] * scale_y));
  } else if (d.screen_space && d.cover[2] > d.cover[0] && d.cover[3] > d.cover[1]) {
    x0 = std::max(x0, d.cover[0] * sx * int32_t(scale_x));
    y0 = std::max(y0, d.cover[1] * sy * int32_t(scale_y));
    x1 = std::min(x1, d.cover[2] * sx * int32_t(scale_x));
    y1 = std::min(y1, d.cover[3] * sy * int32_t(scale_y));
  }
  // Session 80 (gpu_native_truth_bounds_vs): the box the draw's vertices span - nothing outside it is rasterized.
  if (d.bounds[2] > d.bounds[0] && d.bounds[3] > d.bounds[1]) {
    x0 = std::max(x0, d.bounds[0] * sx * int32_t(scale_x));
    y0 = std::max(y0, d.bounds[1] * sy * int32_t(scale_y));
    x1 = std::min(x1, d.bounds[2] * sx * int32_t(scale_x));
    y1 = std::min(y1, d.bounds[3] * sy * int32_t(scale_y));
  }
  if (!d.screen_space) {
    x0 = std::max(x0, int32_t(std::floor(d.viewport.x * sx)));
    y0 = std::max(y0, int32_t(std::floor(d.viewport.y * sy)));
    x1 = std::min(x1, int32_t(std::ceil((d.viewport.x + d.viewport.width) * sx)));
    y1 = std::min(y1, int32_t(std::ceil((d.viewport.y + d.viewport.height) * sy)));
  }
  rect[0] = std::max(0, x0), rect[1] = std::max(0, y0), rect[2] = x1, rect[3] = y1;
}

void NativeRenderer::TruthServeDropped() {
  truth_counts_.serve_lost += uint32_t(serve_pending_.size());
  serve_pending_.clear();
  // (A rename for it stays as made: its empty rectangle has the flush copy every pixel into the fresh image.)
  serve_rename_.valid = false;
}

bool NativeRenderer::TruthWantsCover() const {
  return REXCVAR_GET(gpu_native_truth_carry_copies) && !truth_deferred_.empty();
}

bool NativeRenderer::TruthWantsBounds(uint64_t vertex_shader_hash) {
  if (!truth_ || !replacing_) return false;
  // Session 94 (gpu_d3d_typed_lean_bind): the cvar string is compared with the parsed one (a string copy + compare, 2-5 calls a draw) only on the first call of a frame; a list
  // changed mid-frame takes effect on the next frame. File-static state: there is one NativeRenderer.
  static uint64_t checked_frame = UINT64_MAX;
  static const NativeRenderer* checked_for = nullptr;
  const bool latched = REXCVAR_GET(gpu_d3d_typed_lean_bind) && checked_for == this && checked_frame == frame_;
  const std::string& list = latched ? truth_bounds_list_ : REXCVAR_GET(gpu_native_truth_bounds_vs);
  checked_for = this;
  checked_frame = frame_;
  if (!latched && list != truth_bounds_list_) {
    truth_bounds_list_ = list;
    truth_bounds_vs_.clear();
    size_t i = 0;
    while (i < list.size()) {
      const size_t end = std::min(list.find(',', i), list.size());
      const std::string item = list.substr(i, end - i);
      char* stop = nullptr;
      const unsigned long long hash = std::strtoull(item.c_str(), &stop, 16);
      if (hash && stop && stop != item.c_str()) truth_bounds_vs_.push_back(uint64_t(hash));
      i = end + 1;
    }
  }
  return std::find(truth_bounds_vs_.begin(), truth_bounds_vs_.end(), vertex_shader_hash) != truth_bounds_vs_.end();
}

void NativeRenderer::TruthNoteBounds(uint32_t outcome, double ms) {
  if (outcome == 0) ++truth_counts_.bounds_made;
  if (outcome == 1) ++truth_counts_.bounds_cached;
  if (outcome == 2) ++truth_counts_.bounds_refused;
  truth_counts_.bounds_ms += ms;
}

}  // namespace rex::graphics::vulkan::native
