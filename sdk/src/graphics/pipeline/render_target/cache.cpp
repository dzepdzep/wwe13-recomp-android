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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <rex/assert.h>
#include <rex/cvar.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/pipeline/render_target/cache.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/util/bit_inline.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/xenos.h>
#include <rex/logging.h>
#include <rex/math.h>

REXCVAR_DEFINE_BOOL(mrt_edram_used_range_clamp_to_min, true, "GPU",
                    "Clamp MRT EDRAM used range to minimum");

REXCVAR_DEFINE_BOOL(execute_unclipped_draw_vs_on_cpu_for_psi_render_backend, true, "GPU",
                    "Execute unclipped draw VS on CPU for PSI render backend");

REXCVAR_DEFINE_BOOL(snorm16_render_target_full_range, true, "GPU",
                    "Use full range for SNORM16 render targets");

REXCVAR_DEFINE_BOOL(direct_host_resolve, true, "GPU",
                    "Resolve from host render targets directly to shared memory when possible")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// wwe2k14 patch (Session 6, CP8): with pixel shader interlock (render_target_path_vulkan=fsi), Update still
// sorted the EDRAM bases, computed the range lengths and ran WouldOwnershipChangeRequireTransfers and
// ChangeOwnership for every draw (about 2% of the command processor thread in heavy entrance shots,
// stacks_067), although consecutive draws mostly use render targets that already own their ranges. That part
// depends only on UpdateSkipKey (derived from the registers, the shaders' depth / color usage, the estimated
// height and mrt_edram_used_range_clamp_to_min, all still computed as before) and on ownership_ranges_, and its
// only effects are the interlock barrier request and changes to ownership_ranges_. So once a full run for a key
// has been seen to have no effect (no barrier requested, ownership_ranges_ not modified), a later update with
// the same key while ownership_ranges_ is still unmodified (ownership_generation_) would have no effect either,
// and is skipped. Host render targets keep the original path. Read per update; the entries stay correct while
// the settings are off (ownership_generation_ is always maintained), so both can be flipped at any time.
REXCVAR_DEFINE_BOOL(gpu_rt_cache_skip_unchanged, false, "GPU",
                    "Pixel shader interlock path: skip the EDRAM ownership part of the render target cache "
                    "update when the render targets, their estimated extent and the ownership state are those "
                    "of an earlier update that changed nothing");
REXCVAR_DEFINE_BOOL(gpu_rt_cache_skip_unchanged_verify, false, "GPU",
                    "Shadow-check gpu_rt_cache_skip_unchanged (works with it on or off): on every would-be "
                    "skip, run the full update anyway, check that it changed nothing, and count and log "
                    "mismatches ([s6][CP8])");

// wwe2k14 patch (Session 37): log what the next N render target updates claim of the EDRAM (the fbo path's black
// frames: a depth-only draw's estimated height let its render target take the frame's colour tiles). Counts down.
REXCVAR_DEFINE_INT32(gpu_rt_update_log, 0, "GPU",
                     "Log the next N render target cache updates: pitch, MSAA, estimated height, and each bound "
                     "render target's EDRAM base and claimed length ([rt_update] lines)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

namespace rex::graphics {

// wwe2k14 patch (Session 6, CP8): gpu_rt_cache_skip_unchanged statistics and logs.
namespace {

constexpr uint64_t kUpdateSkipStatsFrames = 300;
constexpr uint32_t kUpdateSkipMismatchLogLimit = 8;

enum : uint32_t {
  // ownership_ranges_ differs from when the entries were confirmed, without an ownership_generation_ change.
  kUpdateSkipMismatchStaleOwnership = 1 << 0,
  // The full run made different ownership calls than the run that confirmed the key.
  kUpdateSkipMismatchRanges = 1 << 1,
  // The full run requested an interlock barrier.
  kUpdateSkipMismatchBarrier = 1 << 2,
  // The full run modified ownership_ranges_.
  kUpdateSkipMismatchOwnershipChanged = 1 << 3,
};

std::string DescribeUpdateSkipMismatch(uint32_t flags) {
  std::string description;
  auto append = [&description](const char* text) {
    if (!description.empty()) {
      description += ", ";
    }
    description += text;
  };
  if (flags & kUpdateSkipMismatchStaleOwnership) {
    append("ownership map modified without a generation change");
  }
  if (flags & kUpdateSkipMismatchRanges) {
    append("different ownership calls");
  }
  if (flags & kUpdateSkipMismatchBarrier) {
    append("interlock barrier requested");
  }
  if (flags & kUpdateSkipMismatchOwnershipChanged) {
    append("ownership map modified");
  }
  return description;
}

std::string DescribeUpdateSkipValues(const uint32_t* values, uint32_t count) {
  std::string description = "[";
  for (uint32_t i = 0; i < count; ++i) {
    if (i) {
      description += " ";
    }
    description += std::to_string(values[i]);
  }
  description += "]";
  return description;
}

double GetUpdateSkipPercentage(uint64_t part, uint64_t total) {
  return total ? double(part) * 100.0 / double(total) : 0.0;
}

}  // namespace

void RenderTargetCache::GetPSIColorFormatInfo(xenos::ColorRenderTargetFormat format,
                                              uint32_t write_mask, float& clamp_rgb_low,
                                              float& clamp_alpha_low, float& clamp_rgb_high,
                                              float& clamp_alpha_high, uint32_t& keep_mask_low,
                                              uint32_t& keep_mask_high) {
  keep_mask_low = keep_mask_high = 0;
  switch (format) {
    case xenos::ColorRenderTargetFormat::k_8_8_8_8:
    case xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA: {
      clamp_rgb_low = clamp_alpha_low = 0.0f;
      clamp_rgb_high = clamp_alpha_high = 1.0f;
      for (uint32_t i = 0; i < 4; ++i) {
        if (!(write_mask & (1 << i))) {
          keep_mask_low |= uint32_t(0xFF) << (i * 8);
        }
      }
    } break;
    case xenos::ColorRenderTargetFormat::k_2_10_10_10:
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_AS_10_10_10_10: {
      clamp_rgb_low = clamp_alpha_low = 0.0f;
      clamp_rgb_high = clamp_alpha_high = 1.0f;
      for (uint32_t i = 0; i < 3; ++i) {
        if (!(write_mask & (1 << i))) {
          keep_mask_low |= uint32_t(0x3FF) << (i * 10);
        }
      }
      if (!(write_mask & 0b1000)) {
        keep_mask_low |= uint32_t(3) << 30;
      }
    } break;
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT:
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT_AS_16_16_16_16: {
      clamp_rgb_low = clamp_alpha_low = 0.0f;
      clamp_rgb_high = 31.875f;
      clamp_alpha_high = 1.0f;
      for (uint32_t i = 0; i < 3; ++i) {
        if (!(write_mask & (1 << i))) {
          keep_mask_low |= uint32_t(0x3FF) << (i * 10);
        }
      }
      if (!(write_mask & 0b1000)) {
        keep_mask_low |= uint32_t(3) << 30;
      }
    } break;
    case xenos::ColorRenderTargetFormat::k_16_16:
    case xenos::ColorRenderTargetFormat::k_16_16_16_16:
      // Alpha clamping affects blending source, so it's non-zero for alpha for
      // k_16_16 (the render target is fixed-point). There's one deviation from
      // how Direct3D 11.3 functional specification defines SNorm conversion
      // (NaN should be 0, not the lowest negative number), and that needs to be
      // handled separately.
      clamp_rgb_low = clamp_alpha_low = -32.0f;
      clamp_rgb_high = clamp_alpha_high = 32.0f;
      if (!(write_mask & 0b0001)) {
        keep_mask_low |= 0xFFFFu;
      }
      if (!(write_mask & 0b0010)) {
        keep_mask_low |= 0xFFFF0000u;
      }
      if (format == xenos::ColorRenderTargetFormat::k_16_16_16_16) {
        if (!(write_mask & 0b0100)) {
          keep_mask_high |= 0xFFFFu;
        }
        if (!(write_mask & 0b1000)) {
          keep_mask_high |= 0xFFFF0000u;
        }
      } else {
        write_mask &= 0b0011;
      }
      break;
    case xenos::ColorRenderTargetFormat::k_16_16_FLOAT:
    case xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT:
      // No NaNs on the Xbox 360 GPU, though can't use the extended range with
      // Direct3D and Vulkan conversions.
      // TODO(Triang3l): Use the extended-range encoding in all implementations.
      clamp_rgb_low = clamp_alpha_low = -65504.0f;
      clamp_rgb_high = clamp_alpha_high = 65504.0f;
      if (!(write_mask & 0b0001)) {
        keep_mask_low |= 0xFFFFu;
      }
      if (!(write_mask & 0b0010)) {
        keep_mask_low |= 0xFFFF0000u;
      }
      if (format == xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT) {
        if (!(write_mask & 0b0100)) {
          keep_mask_high |= 0xFFFFu;
        }
        if (!(write_mask & 0b1000)) {
          keep_mask_high |= 0xFFFF0000u;
        }
      } else {
        write_mask &= 0b0011;
      }
      break;
    case xenos::ColorRenderTargetFormat::k_32_FLOAT:
      // No clamping - let min/max always pick the original value.
      clamp_rgb_low = clamp_alpha_low = clamp_rgb_high = clamp_alpha_high = std::nanf("");
      write_mask &= 0b0001;
      if (!(write_mask & 0b0001)) {
        keep_mask_low = ~uint32_t(0);
      }
      break;
    case xenos::ColorRenderTargetFormat::k_32_32_FLOAT:
      // No clamping - let min/max always pick the original value.
      clamp_rgb_low = clamp_alpha_low = clamp_rgb_high = clamp_alpha_high = std::nanf("");
      write_mask &= 0b0011;
      if (!(write_mask & 0b0001)) {
        keep_mask_low = ~uint32_t(0);
      }
      if (!(write_mask & 0b0010)) {
        keep_mask_high = ~uint32_t(0);
      }
      break;
    default:
      assert_unhandled_case(format);
      // Disable invalid render targets.
      write_mask = 0;
      break;
  }
  // Special case handled in the shaders for empty write mask to completely skip
  // a disabled render target: all keep bits are set.
  if (!write_mask) {
    keep_mask_low = keep_mask_high = ~uint32_t(0);
  }
}

uint32_t RenderTargetCache::Transfer::GetRangeRectangles(uint32_t start_tiles, uint32_t end_tiles,
                                                         uint32_t base_tiles, uint32_t pitch_tiles,
                                                         xenos::MsaaSamples msaa_samples,
                                                         bool is_64bpp, Rectangle* rectangles_out,
                                                         const Rectangle* cutout) {
  // EDRAM addressing wrapping must be handled by doing GetRangeRectangles for
  // two clamped ranges, in this case start_tiles == end_tiles will also
  // unambiguously mean an empty range rather than the entire EDRAM.
  assert_true(start_tiles < xenos::kEdramTileCount);
  assert_true(end_tiles <= xenos::kEdramTileCount);
  assert_true(start_tiles <= end_tiles);
  // If start_tiles < base_tiles, this is the tail after EDRAM addressing
  // wrapping.
  assert_true(start_tiles >= base_tiles || end_tiles <= base_tiles);
  assert_not_zero(pitch_tiles);
  if (start_tiles == end_tiles) {
    return 0;
  }
  uint32_t tile_width = xenos::kEdramTileWidthSamples >>
                        (uint32_t(msaa_samples >= xenos::MsaaSamples::k4X) + uint32_t(is_64bpp));
  uint32_t tile_height =
      xenos::kEdramTileHeightSamples >> uint32_t(msaa_samples >= xenos::MsaaSamples::k2X);
  // If the first and / or the last rows have the same X spans as the middle
  // part, merge them with it.
  uint32_t rectangle_count = 0;
  // If start_tiles < base_tiles, this is the tail after EDRAM addressing
  // wrapping.
  uint32_t local_offset = start_tiles < base_tiles ? xenos::kEdramTileCount : 0;
  uint32_t local_start = local_offset + start_tiles - base_tiles;
  uint32_t local_end = local_offset + end_tiles - base_tiles;
  // Inclusive.
  uint32_t rows_start = local_start / pitch_tiles;
  // Exclusive.
  uint32_t rows_end = (local_end + (pitch_tiles - 1)) / pitch_tiles;
  uint32_t row_first_start = local_start - rows_start * pitch_tiles;
  uint32_t row_last_end = pitch_tiles - (rows_end * pitch_tiles - local_end);
  uint32_t rows = rows_end - rows_start;
  if (rows == 1 || row_first_start) {
    Rectangle rectangle_first;
    rectangle_first.x_pixels = row_first_start * tile_width;
    rectangle_first.y_pixels = rows_start * tile_height;
    rectangle_first.width_pixels =
        ((rows == 1 ? row_last_end : pitch_tiles) - row_first_start) * tile_width;
    rectangle_first.height_pixels = tile_height;
    rectangle_count += AddRectangle(
        rectangle_first, rectangles_out ? rectangles_out + rectangle_count : nullptr, cutout);
    if (rows == 1) {
      return rectangle_count;
    }
  }
  uint32_t mid_rows_start = rows_start + 1;
  uint32_t mid_rows = rows - 2;
  if (!row_first_start) {
    --mid_rows_start;
    ++mid_rows;
  }
  if (row_last_end == pitch_tiles) {
    ++mid_rows;
  }
  if (mid_rows) {
    Rectangle rectangle_mid;
    rectangle_mid.x_pixels = 0;
    rectangle_mid.y_pixels = mid_rows_start * tile_height;
    rectangle_mid.width_pixels = pitch_tiles * tile_width;
    rectangle_mid.height_pixels = mid_rows * tile_height;
    rectangle_count += AddRectangle(
        rectangle_mid, rectangles_out ? rectangles_out + rectangle_count : nullptr, cutout);
  }
  if (row_last_end != pitch_tiles) {
    Rectangle rectangle_last;
    rectangle_last.x_pixels = 0;
    rectangle_last.y_pixels = (rows_end - 1) * tile_height;
    rectangle_last.width_pixels = row_last_end * tile_width;
    rectangle_last.height_pixels = tile_height;
    rectangle_count += AddRectangle(
        rectangle_last, rectangles_out ? rectangles_out + rectangle_count : nullptr, cutout);
  }
  assert_true(rectangle_count <= (cutout ? kMaxRectanglesWithCutout : kMaxRectanglesWithoutCutout));
  return rectangle_count;
}

uint32_t RenderTargetCache::Transfer::AddRectangle(const Rectangle& rectangle,
                                                   Rectangle* rectangles_out,
                                                   const Rectangle* cutout) {
  uint32_t rectangle_right = rectangle.x_pixels + rectangle.width_pixels;
  uint32_t rectangle_bottom = rectangle.y_pixels + rectangle.height_pixels;
  // If nothing to cut out (no region specified, or no intersection - if the
  // cutout region is in the middle on Y, but completely to the left / right on
  // X, don't split), add the whole rectangle.
  if (!cutout || !cutout->width_pixels || !cutout->height_pixels ||
      cutout->x_pixels >= rectangle_right ||
      cutout->x_pixels + cutout->width_pixels <= rectangle.x_pixels ||
      cutout->y_pixels >= rectangle_bottom ||
      cutout->y_pixels + cutout->height_pixels <= rectangle.y_pixels) {
    if (rectangles_out) {
      rectangles_out[0] = rectangle;
    }
    return 1;
  }
  uint32_t rectangle_count = 0;
  uint32_t cutout_right = cutout->x_pixels + cutout->width_pixels;
  uint32_t cutout_bottom = cutout->y_pixels + cutout->height_pixels;
  // Upper part after cutout.
  if (cutout->y_pixels > rectangle.y_pixels) {
    // The completely outside case has already been checked.
    assert_true(cutout->y_pixels < rectangle_bottom);
    if (rectangles_out) {
      Rectangle& rectangle_upper = rectangles_out[rectangle_count];
      rectangle_upper.x_pixels = rectangle.x_pixels;
      rectangle_upper.y_pixels = rectangle.y_pixels;
      rectangle_upper.width_pixels = rectangle.width_pixels;
      // cutout->y_pixels is already known to be < rectangle_bottom, no need for
      // min(cutout->y_pixels - rectangle.y_pixels, rectangle.height_pixels).
      rectangle_upper.height_pixels = cutout->y_pixels - rectangle.y_pixels;
    }
    ++rectangle_count;
  }
  // Middle part after cutout.
  uint32_t middle_top = std::max(cutout->y_pixels, rectangle.y_pixels);
  uint32_t middle_height = std::min(cutout_bottom, rectangle_bottom) - middle_top;
  // Middle left.
  if (cutout->x_pixels > rectangle.x_pixels) {
    assert_true(cutout->x_pixels < rectangle_right);
    if (rectangles_out) {
      Rectangle& rectangle_middle_left = rectangles_out[rectangle_count];
      rectangle_middle_left.x_pixels = rectangle.x_pixels;
      rectangle_middle_left.y_pixels = middle_top;
      rectangle_middle_left.width_pixels = cutout->x_pixels - rectangle.x_pixels;
      rectangle_middle_left.height_pixels = middle_height;
    }
    ++rectangle_count;
  }
  // Middle right.
  if (cutout_right < rectangle_right) {
    assert_true(cutout_right > rectangle.x_pixels);
    if (rectangles_out) {
      Rectangle& rectangle_middle_right = rectangles_out[rectangle_count];
      rectangle_middle_right.x_pixels = cutout_right;
      rectangle_middle_right.y_pixels = middle_top;
      rectangle_middle_right.width_pixels = rectangle_right - cutout_right;
      rectangle_middle_right.height_pixels = middle_height;
    }
    ++rectangle_count;
  }
  // Lower part after cutout.
  if (cutout_bottom < rectangle_bottom) {
    assert_true(cutout_bottom > rectangle.y_pixels);
    if (rectangles_out) {
      Rectangle& rectangle_upper = rectangles_out[rectangle_count];
      rectangle_upper.x_pixels = rectangle.x_pixels;
      rectangle_upper.y_pixels = cutout_bottom;
      rectangle_upper.width_pixels = rectangle.width_pixels;
      rectangle_upper.height_pixels = rectangle_bottom - cutout_bottom;
    }
    ++rectangle_count;
  }
  assert_true(rectangle_count <= kMaxCutoutBorderRectangles);
  return rectangle_count;
}

RenderTargetCache::~RenderTargetCache() {
  ShutdownCommon();
}

void RenderTargetCache::InitializeCommon() {
  assert_true(ownership_ranges_.empty());
  ++ownership_generation_;  // wwe2k14 patch (Session 6, CP8)
  ownership_ranges_.emplace(std::piecewise_construct, std::forward_as_tuple(uint32_t(0)),
                            std::forward_as_tuple(xenos::kEdramTileCount, RenderTargetKey(),
                                                  RenderTargetKey(), RenderTargetKey()));
}

void RenderTargetCache::DestroyAllRenderTargets(bool shutting_down) {
  ++ownership_generation_;  // wwe2k14 patch (Session 6, CP8)
  ownership_ranges_.clear();
  if (!shutting_down) {
    ownership_ranges_.emplace(std::piecewise_construct, std::forward_as_tuple(uint32_t(0)),
                              std::forward_as_tuple(xenos::kEdramTileCount, RenderTargetKey(),
                                                    RenderTargetKey(), RenderTargetKey()));
  }

  for (const auto& render_target_pair : render_targets_) {
    if (render_target_pair.second) {
      delete render_target_pair.second;
    }
  }
  render_targets_.clear();
}

void RenderTargetCache::ShutdownCommon() {
  DestroyAllRenderTargets(true);
}

void RenderTargetCache::ClearCache() {
  // wwe2k14 patch (Session 6, CP8): ownership_ranges_ isn't modified here, but cache clears are rare, so drop
  // the gpu_rt_cache_skip_unchanged entries anyway (conservative).
  ++ownership_generation_;
  // Keep only render targets currently owning any EDRAM data.
  if (!render_targets_.empty()) {
    std::unordered_set<RenderTargetKey, RenderTargetKey::Hasher> used_render_targets;
    for (const auto& ownership_range_pair : ownership_ranges_) {
      const OwnershipRange& ownership_range = ownership_range_pair.second;
      if (!ownership_range.render_target.IsEmpty()) {
        used_render_targets.emplace(ownership_range.render_target);
      }
      if (!ownership_range.host_depth_render_target_unorm24.IsEmpty()) {
        used_render_targets.emplace(ownership_range.host_depth_render_target_unorm24);
      }
      if (!ownership_range.host_depth_render_target_float24.IsEmpty()) {
        used_render_targets.emplace(ownership_range.host_depth_render_target_float24);
      }
    }
    if (render_targets_.size() != used_render_targets.size()) {
      typename decltype(render_targets_)::iterator it_next;
      for (auto it = render_targets_.begin(); it != render_targets_.end(); it = it_next) {
        it_next = std::next(it);
        if (!it->second) {
          render_targets_.erase(it);
          continue;
        }
        if (used_render_targets.find(it->second->key()) == used_render_targets.end()) {
          delete it->second;
          render_targets_.erase(it);
        }
      }
    }
  }
}

void RenderTargetCache::BeginFrame() {
  ResetAccumulatedRenderTargets();
}

bool RenderTargetCache::Update(bool is_rasterization_done,
                               reg::RB_DEPTHCONTROL normalized_depth_control,
                               uint32_t normalized_color_mask, const Shader& vertex_shader) {
  const RegisterFile& regs = register_file();
  bool interlock_barrier_only = GetPath() == Path::kPixelShaderInterlock;

  auto rb_surface_info = regs.Get<reg::RB_SURFACE_INFO>();
  xenos::MsaaSamples msaa_samples = rb_surface_info.msaa_samples;
  assert_true(msaa_samples <= xenos::MsaaSamples::k4X);
  if (msaa_samples > xenos::MsaaSamples::k4X) {
    // Safety check because a lot of code assumes up to 4x.
    assert_always();
    REXGPU_ERROR("{}x MSAA requested by the guest, Xenos only supports up to 4x",
                 uint32_t(1) << uint32_t(msaa_samples));
    return false;
  }
  uint32_t msaa_samples_x_log2 = uint32_t(msaa_samples >= xenos::MsaaSamples::k4X);
  uint32_t pitch_pixels = rb_surface_info.surface_pitch;
  // surface_pitch 0 should be handled in disabling rasterization (hopefully
  // it's safe to assume that).
  assert_true(pitch_pixels || !is_rasterization_done);
  if (!pitch_pixels) {
    is_rasterization_done = false;
  } else if (pitch_pixels > xenos::kTexture2DCubeMaxWidthHeight) {
    REXGPU_ERROR(
        "Surface pitch {} larger than the maximum texture width {} specified "
        "by the guest",
        pitch_pixels, xenos::kTexture2DCubeMaxWidthHeight);
    return false;
  }
  uint32_t pitch_tiles_at_32bpp =
      ((pitch_pixels << msaa_samples_x_log2) + (xenos::kEdramTileWidthSamples - 1)) /
      xenos::kEdramTileWidthSamples;
  if (!interlock_barrier_only) {
    uint32_t pitch_pixels_tile_aligned_scaled =
        pitch_tiles_at_32bpp * (xenos::kEdramTileWidthSamples >> msaa_samples_x_log2) *
        draw_resolution_scale_x();
    uint32_t max_render_target_width = GetMaxRenderTargetWidth();
    if (pitch_pixels_tile_aligned_scaled > max_render_target_width) {
      // TODO(Triang3l): If really needed for some game on some device, clamp
      // the pitch and generate multiple ranges (each for every row of tiles)
      // with gaps for padding. Very few PowerVR GPUs have 4096, not 8192, as
      // the limit, though with 8192 (on Mali) the actual limit for Xenia is
      // 8160 because tile padding is stored - but 8192 should be extremely rare
      // anyway.
      REXGPU_ERROR(
          "Surface pitch aligned to EDRAM tiles and resolution-scaled {} "
          "larger than the maximum host render target width {}",
          pitch_pixels_tile_aligned_scaled, max_render_target_width);
      return false;
    }
  }

  // Get used render targets.
  // [0] is depth / stencil where relevant, [1...4] is color.
  // Depth / stencil testing / writing is before color in the pipeline.
  uint32_t depth_and_color_rts_used_bits = 0;
  // depth_and_color_rts_used_bits -> EDRAM base.
  uint32_t edram_bases[1 + xenos::kMaxColorRenderTargets];
  uint32_t resource_formats[1 + xenos::kMaxColorRenderTargets];
  uint32_t rts_are_64bpp = 0;
  if (is_rasterization_done) {
    if (normalized_depth_control.z_enable || normalized_depth_control.stencil_enable) {
      depth_and_color_rts_used_bits |= 1;
      auto rb_depth_info = regs.Get<reg::RB_DEPTH_INFO>();
      edram_bases[0] = rb_depth_info.depth_base;
      // With pixel shader interlock, always the same addressing disregarding
      // the format.
      resource_formats[0] = interlock_barrier_only ? 0 : uint32_t(rb_depth_info.depth_format);
    }
    for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
      if (!(normalized_color_mask & (uint32_t(0b1111) << (4 * i)))) {
        continue;
      }
      auto color_info = regs.Get<reg::RB_COLOR_INFO>(reg::RB_COLOR_INFO::rt_register_indices[i]);
      uint32_t rt_bit_index = 1 + i;
      depth_and_color_rts_used_bits |= uint32_t(1) << rt_bit_index;
      edram_bases[rt_bit_index] = color_info.color_base;
      xenos::ColorRenderTargetFormat color_format =
          regs.Get<reg::RB_COLOR_INFO>(reg::RB_COLOR_INFO::rt_register_indices[i]).color_format;
      bool is_64bpp = xenos::IsColorRenderTargetFormat64bpp(color_format);
      if (is_64bpp) {
        rts_are_64bpp |= uint32_t(1) << rt_bit_index;
      }
      xenos::ColorRenderTargetFormat color_resource_format;
      if (interlock_barrier_only) {
        // Only changes in mapping between coordinates and addresses are
        // interesting (along with access overlap between draw calls), thus only
        // pixel size is relevant.
        color_resource_format = is_64bpp ? xenos::ColorRenderTargetFormat::k_16_16_16_16
                                         : xenos::ColorRenderTargetFormat::k_8_8_8_8;
      } else {
        color_resource_format = GetColorResourceFormat(xenos::GetStorageColorFormat(color_format));
      }
      resource_formats[rt_bit_index] = uint32_t(color_resource_format);
    }
  }

  uint32_t rts_remaining;
  uint32_t rt_index;

  // Eliminate other bound render targets if their EDRAM base conflicts with
  // another render target - it's an error in most host implementations to bind
  // the same render target into multiple slots, also the behavior would be
  // unpredictable if that happens.
  // Depth is considered the least important as it's earlier in the pipeline
  // (issues caused by color and depth render target collisions haven't been
  // found yet), but render targets with smaller index are considered more
  // important - specifically, because of the usage in the lighting pass of
  // 4D5307E6, which can be checked in the vertical look calibration sequence in
  // the beginning of the game: if render target 0 is removed in favor of 1, the
  // characters and the world will be too dark, like fully in shadow -
  // especially prominent on the helmet. This happens because the shader picks
  // between two render targets to write dynamically (though with a static, bool
  // constant condition), but all other state is set up in a way that implies
  // the same render target being bound twice. On Direct3D 9, if you don't write
  // to a color pixel shader output on the control flow that was taken, the
  // render target will not be written to. However, this has been relaxed in
  // Direct3D 10, where if the shader declares an output, it's assumed to be
  // always written (or with an undefined value otherwise).
  rts_remaining = depth_and_color_rts_used_bits & ~(uint32_t(1));
  while (bit_inline::bit_scan_forward(rts_remaining, &rt_index)) {
    rts_remaining &= ~(uint32_t(1) << rt_index);
    uint32_t edram_base = edram_bases[rt_index];
    uint32_t rts_other_remaining =
        depth_and_color_rts_used_bits & (~((uint32_t(1) << (rt_index + 1)) - 1) | uint32_t(1));
    uint32_t rt_other_index;
    while (bit_inline::bit_scan_forward(rts_other_remaining, &rt_other_index)) {
      rts_other_remaining &= ~(uint32_t(1) << rt_other_index);
      if (edram_bases[rt_other_index] == edram_base) {
        depth_and_color_rts_used_bits &= ~(uint32_t(1) << rt_other_index);
      }
    }
  }

  // Clear ownership transfers before adding any.
  if (!interlock_barrier_only) {
    for (size_t i = 0; i < rex::countof(last_update_transfers_); ++i) {
      last_update_transfers_[i].clear();
    }
  }

  if (!depth_and_color_rts_used_bits) {
    // Nothing to bind, don't waste time on things like memexport-only draws -
    // just check if old bindings can still be used.
    std::memset(last_update_used_render_targets_, 0, sizeof(last_update_used_render_targets_));
    if (are_accumulated_render_targets_valid_) {
      for (size_t i = 0; i < rex::countof(last_update_accumulated_render_targets_); ++i) {
        const RenderTarget* render_target = last_update_accumulated_render_targets_[i];
        if (!render_target) {
          continue;
        }
        RenderTargetKey rt_key = render_target->key();
        if (rt_key.pitch_tiles_at_32bpp != pitch_tiles_at_32bpp ||
            rt_key.msaa_samples != msaa_samples) {
          are_accumulated_render_targets_valid_ = false;
          break;
        }
      }
    }
    if (!are_accumulated_render_targets_valid_) {
      std::memset(last_update_accumulated_render_targets_, 0,
                  sizeof(last_update_accumulated_render_targets_));
    }
    return true;
  }

  // Estimate height used by render targets (for color for writes, for depth /
  // stencil for both reads and writes) from various sources.
  uint32_t height_used =
      std::min(GetRenderTargetHeight(pitch_tiles_at_32bpp, msaa_samples),
               draw_extent_estimator_.EstimateMaxY(
                   interlock_barrier_only
                       ? REXCVAR_GET(execute_unclipped_draw_vs_on_cpu_for_psi_render_backend)
                       : true,
                   vertex_shader));

  // wwe2k14 patch (Session 6, CP8): gpu_rt_cache_skip_unchanged, see the comment at the setting. Everything
  // above still runs as before (including EstimateMaxY, whose result may depend on guest memory with
  // execute_unclipped_draw_vs_on_cpu); skip_unchanged_key holds everything the code below depends on, apart
  // from ownership_ranges_ (the sort, the lengths and the keys are computed from these values only). Pixel
  // shader interlock only: with host render targets, the rest also creates render targets, gathers transfers
  // and updates the bindings.
  bool skip_unchanged_verify = false;
  bool skip_unchanged_active = false;
  // While a would-be skip is being verified, the matching entry, otherwise UINT32_MAX.
  uint32_t skip_unchanged_hit_entry = UINT32_MAX;
  UpdateSkipKey skip_unchanged_key;
  uint64_t skip_unchanged_generation_before = 0;
  uint64_t skip_unchanged_map_hash_before = 0;
  if (interlock_barrier_only) {
    skip_unchanged_verify = REXCVAR_GET(gpu_rt_cache_skip_unchanged_verify);
    skip_unchanged_active = skip_unchanged_verify || REXCVAR_GET(gpu_rt_cache_skip_unchanged);
  }
  if (skip_unchanged_active) {
    std::memset(&skip_unchanged_key, 0, sizeof(skip_unchanged_key));
    skip_unchanged_key.msaa_samples = uint32_t(msaa_samples);
    skip_unchanged_key.pitch_tiles_at_32bpp = pitch_tiles_at_32bpp;
    skip_unchanged_key.depth_and_color_rts_used_bits = depth_and_color_rts_used_bits;
    // Only the bits of the used render targets are read below.
    skip_unchanged_key.rts_are_64bpp = rts_are_64bpp & depth_and_color_rts_used_bits;
    skip_unchanged_key.height_used = height_used;
    skip_unchanged_key.mrt_edram_used_range_clamp_to_min =
        uint32_t(REXCVAR_GET(mrt_edram_used_range_clamp_to_min));
    rts_remaining = depth_and_color_rts_used_bits;
    while (bit_inline::bit_scan_forward(rts_remaining, &rt_index)) {
      rts_remaining &= ~(uint32_t(1) << rt_index);
      skip_unchanged_key.edram_bases[rt_index] = edram_bases[rt_index];
      skip_unchanged_key.resource_formats[rt_index] = resource_formats[rt_index];
    }
    UpdateSkipState& skip_unchanged = update_skip_;
    ++skip_unchanged.stats.updates;
    skip_unchanged_generation_before = ownership_generation_;
    if (!skip_unchanged.entry_count || skip_unchanged.generation != ownership_generation_) {
      // Nothing confirmed for the current ownership state.
      ++skip_unchanged.stats.miss_state;
    } else {
      for (uint32_t i = 0; i < skip_unchanged.entry_count; ++i) {
        if (!std::memcmp(&skip_unchanged.entries[i].key, &skip_unchanged_key,
                         sizeof(skip_unchanged_key))) {
          skip_unchanged_hit_entry = i;
          break;
        }
      }
      if (skip_unchanged_hit_entry == UINT32_MAX) {
        ++skip_unchanged.stats.miss_key;
      } else if (!skip_unchanged_verify) {
        // The full run would request no barrier and leave ownership_ranges_ as it is - nothing else below
        // has effects with pixel shader interlock.
        ++skip_unchanged.stats.skipped;
        ReportUpdateSkipStats();
        return true;
      } else {
        skip_unchanged_map_hash_before = HashOwnershipRanges();
      }
    }
  }

  // Sorted by EDRAM base and then by index in the pipeline - for simplicity,
  // treat render targets placed closer to the end of the EDRAM as truncating
  // the previous one (and in case multiple render targets are placed at the
  // same EDRAM base, though normally this shouldn't happen, treat the color
  // ones as more important than the depth one, which may be not needed and just
  // a leftover if the draw, for instance, has depth / stencil happening to be
  // always passing and never writing with the current state, and also because
  // depth testing has to happen before the color is written). Overall it's
  // normal for estimated EDRAM ranges of render targets to intersect if drawing
  // without a viewport (as there's nothing to clamp the estimated height) and
  // multiple render targets are bound.
  std::pair<uint32_t, uint32_t> edram_bases_sorted[1 + xenos::kMaxColorRenderTargets];
  uint32_t edram_bases_sorted_count = 0;
  rts_remaining = depth_and_color_rts_used_bits;
  while (bit_inline::bit_scan_forward(rts_remaining, &rt_index)) {
    rts_remaining &= ~(uint32_t(1) << rt_index);
    edram_bases_sorted[edram_bases_sorted_count++] =
        std::make_pair(edram_bases[rt_index], rt_index);
  }
  std::sort(edram_bases_sorted, edram_bases_sorted + edram_bases_sorted_count);
  // "As if it was 64bpp" (contribution of 32bpp render targets multiplied by 2,
  // and clamping for 32bpp render targets divides this by 2) because 32bpp
  // render targets can be combined with twice as long 64bpp render targets. An
  // example is the 4541099D menu background (1-sample 1152x720, or 1200x720
  // after rounding to tiles, with a 32bpp depth buffer at 0 requiring 675
  // tiles, and a 64bpp color buffer at 675 requiring 1350 tiles, but the
  // smallest distance between two render target bases is 675 tiles).
  uint32_t rt_max_distance_tiles_at_64bpp = xenos::kEdramTileCount * 2;
  if (REXCVAR_GET(mrt_edram_used_range_clamp_to_min) && edram_bases_sorted_count >= 2) {
    for (uint32_t i = 1; i < edram_bases_sorted_count; ++i) {
      const std::pair<uint32_t, uint32_t>& rt_base_prev = edram_bases_sorted[i - 1];
      rt_max_distance_tiles_at_64bpp =
          std::min(rt_max_distance_tiles_at_64bpp,
                   (edram_bases_sorted[i].first - rt_base_prev.first)
                       << (((rts_are_64bpp >> rt_base_prev.second) & 1) ^ 1));
    }
    // Clamp to the distance from the last render target to the first with
    // EDRAM addressing wrapping.
    const std::pair<uint32_t, uint32_t>& rt_base_last =
        edram_bases_sorted[edram_bases_sorted_count - 1];
    rt_max_distance_tiles_at_64bpp =
        std::min(rt_max_distance_tiles_at_64bpp,
                 (xenos::kEdramTileCount + edram_bases_sorted[0].first - rt_base_last.first)
                     << (((rts_are_64bpp >> rt_base_last.second) & 1) ^ 1));
  }

  // Make sure all the needed render targets are created, and gather lengths of
  // ranges used by each render target.
  RenderTargetKey rt_keys[1 + xenos::kMaxColorRenderTargets];
  RenderTarget* rts[1 + xenos::kMaxColorRenderTargets];
  uint32_t rt_lengths_tiles[1 + xenos::kMaxColorRenderTargets];
  uint32_t length_used_tiles_at_32bpp =
      ((height_used << uint32_t(msaa_samples >= xenos::MsaaSamples::k2X)) +
       (xenos::kEdramTileHeightSamples - 1)) /
      xenos::kEdramTileHeightSamples * pitch_tiles_at_32bpp;
  for (uint32_t i = 0; i < edram_bases_sorted_count; ++i) {
    const std::pair<uint32_t, uint32_t>& rt_base_index = edram_bases_sorted[i];
    uint32_t rt_base = rt_base_index.first;
    uint32_t rt_bit_index = rt_base_index.second;
    RenderTargetKey& rt_key = rt_keys[rt_bit_index];
    rt_key.base_tiles = rt_base;
    rt_key.pitch_tiles_at_32bpp = pitch_tiles_at_32bpp;
    rt_key.msaa_samples = msaa_samples;
    rt_key.is_depth = rt_bit_index == 0;
    rt_key.resource_format = resource_formats[rt_bit_index];
    if (!interlock_barrier_only) {
      RenderTarget* render_target = GetOrCreateRenderTarget(rt_key);
      if (!render_target) {
        return false;
      }
      rts[rt_bit_index] = render_target;
    }
    uint32_t rt_is_64bpp = (rts_are_64bpp >> rt_bit_index) & 1;
    // The last render target can occupy the EDRAM until the base of the first
    // render target (itself in case of 1 render target) with EDRAM addressing
    // wrapping.
    rt_lengths_tiles[i] = std::min(std::min(length_used_tiles_at_32bpp << rt_is_64bpp,
                                            rt_max_distance_tiles_at_64bpp >> (rt_is_64bpp ^ 1)),
                                   ((i + 1 < edram_bases_sorted_count)
                                        ? edram_bases_sorted[i + 1].first
                                        : (xenos::kEdramTileCount + edram_bases_sorted[0].first)) -
                                       rt_base);
  }

  // wwe2k14 patch (Session 37): gpu_rt_update_log.
  if (const int32_t n = REXCVAR_GET(gpu_rt_update_log); n > 0) {
    rex::cvar::SetFlagByName("gpu_rt_update_log", std::to_string(n - 1));
    std::string rts_text;
    for (uint32_t i = 0; i < edram_bases_sorted_count; ++i) {
      const uint32_t bit = edram_bases_sorted[i].second;
      rts_text += fmt::format(" {}{}@{}+{}", bit ? "c" : "d", bit ? bit - 1 : 0, edram_bases_sorted[i].first,
                              rt_lengths_tiles[i]);
    }
    REXGPU_INFO("[rt_update] {} pitch {} tiles, msaa {}, height used {} (max {}), rts{}",
                interlock_barrier_only ? "fsi" : "fbo", pitch_tiles_at_32bpp, uint32_t(msaa_samples), height_used,
                GetRenderTargetHeight(pitch_tiles_at_32bpp, msaa_samples), rts_text);
  }

  // wwe2k14 patch (Session 6, CP8): declared outside the block for gpu_rt_cache_skip_unchanged.
  bool interlock_barrier_needed = false;
  if (interlock_barrier_only) {
    // Because a full pixel shader interlock barrier may clear the ownership map
    // (since it flushes all previous writes, and there's no need for another
    // barrier if an overlap is encountered later between pre-barrier and
    // post-barrier usages), check if any overlap requiring a barrier happens,
    // and then insert the barrier if needed.
    for (uint32_t i = 0; i < edram_bases_sorted_count; ++i) {
      const std::pair<uint32_t, uint32_t>& rt_base_index = edram_bases_sorted[i];
      if (WouldOwnershipChangeRequireTransfers(rt_keys[rt_base_index.second], 0,
                                               rt_lengths_tiles[i])) {
        interlock_barrier_needed = true;
        break;
      }
    }
    if (interlock_barrier_needed) {
      RequestPixelShaderInterlockBarrier();
    }
  }

  // From now on ownership transfers should succeed for simplicity and
  // consistency, even if they fail in the implementation (just ignore that and
  // draw with whatever contents currently are in the render target in this
  // case).

  for (uint32_t i = 0; i < edram_bases_sorted_count; ++i) {
    const std::pair<uint32_t, uint32_t>& rt_base_index = edram_bases_sorted[i];
    uint32_t rt_bit_index = rt_base_index.second;
    ChangeOwnership(rt_keys[rt_bit_index], 0, rt_lengths_tiles[i],
                    interlock_barrier_only ? nullptr : &last_update_transfers_[rt_bit_index]);
  }

  if (interlock_barrier_only) {
    // wwe2k14 patch (Session 6, CP8): gpu_rt_cache_skip_unchanged - remember the key if this run had no
    // effect, and check a would-be skip.
    if (skip_unchanged_active) {
      UpdateSkipRange skip_unchanged_ranges[1 + xenos::kMaxColorRenderTargets];
      for (uint32_t i = 0; i < edram_bases_sorted_count; ++i) {
        UpdateSkipRange& skip_unchanged_range = skip_unchanged_ranges[i];
        skip_unchanged_range.rt_bit_index = edram_bases_sorted[i].second;
        skip_unchanged_range.rt_key = rt_keys[edram_bases_sorted[i].second].key;
        skip_unchanged_range.length_tiles = rt_lengths_tiles[i];
      }
      UpdateSkipAfterFullRun(skip_unchanged_key, skip_unchanged_ranges, edram_bases_sorted_count,
                             interlock_barrier_needed, skip_unchanged_generation_before,
                             skip_unchanged_hit_entry, skip_unchanged_map_hash_before,
                             skip_unchanged_verify);
      ReportUpdateSkipStats();
    }
    // No copying transfers or render target bindings - only needed the barrier.
    return true;
  }

  // If everything succeeded, update the used render targets.
  for (uint32_t i = 0; i < 1 + xenos::kMaxColorRenderTargets; ++i) {
    last_update_used_render_targets_[i] =
        (depth_and_color_rts_used_bits & (uint32_t(1) << i)) ? rts[i] : nullptr;
  }
  if (are_accumulated_render_targets_valid_) {
    // Check if the only re-enabling a previously bound render target.
    for (uint32_t i = 0; i < 1 + xenos::kMaxColorRenderTargets; ++i) {
      RenderTarget* current_rt =
          (depth_and_color_rts_used_bits & (uint32_t(1) << i)) ? rts[i] : nullptr;
      const RenderTarget* accumulated_rt = last_update_accumulated_render_targets_[i];
      if (!accumulated_rt) {
        if (current_rt) {
          // Binding a totally new render target - won't keep the existing
          // render pass anyway, no much need to try to re-enable previously
          // disabled render targets in other slots as well, even though that
          // would be valid.
          are_accumulated_render_targets_valid_ = false;
          break;
        }
        // Append the new render target.
        last_update_accumulated_render_targets_[i] = current_rt;
        continue;
      }
      if (current_rt) {
        if (current_rt != accumulated_rt) {
          // Changing a render target in a slot.
          are_accumulated_render_targets_valid_ = false;
          break;
        }
      } else {
        RenderTargetKey accumulated_rt_key = accumulated_rt->key();
        if (accumulated_rt_key.pitch_tiles_at_32bpp != pitch_tiles_at_32bpp ||
            accumulated_rt_key.msaa_samples != msaa_samples) {
          // The previously bound render target is incompatible with the
          // current surface info.
          are_accumulated_render_targets_valid_ = false;
          break;
        }
      }
    }
    // Make sure the same render target isn't bound into two different slots
    // over time.
    for (uint32_t i = 1;
         are_accumulated_render_targets_valid_ && i < 1 + xenos::kMaxColorRenderTargets; ++i) {
      const RenderTarget* render_target = last_update_accumulated_render_targets_[i];
      if (!render_target) {
        continue;
      }
      for (uint32_t j = 0; j < i; ++j) {
        if (last_update_accumulated_render_targets_[j] == render_target) {
          are_accumulated_render_targets_valid_ = false;
          break;
        }
      }
    }
  }
  if (!are_accumulated_render_targets_valid_) {
    std::memcpy(last_update_accumulated_render_targets_, last_update_used_render_targets_,
                sizeof(last_update_accumulated_render_targets_));
    are_accumulated_render_targets_valid_ = true;
  }

  return true;
}

void RenderTargetCache::UpdateSkipAfterFullRun(const UpdateSkipKey& key,
                                               const UpdateSkipRange* ranges, uint32_t range_count,
                                               bool barrier_requested, uint64_t generation_before,
                                               uint32_t hit_entry, uint64_t map_hash_before,
                                               bool verify) {
  // wwe2k14 patch (Session 6, CP8): see the comment at gpu_rt_cache_skip_unchanged.
  UpdateSkipState& skip = update_skip_;
  UpdateSkipStats& stats = skip.stats;
  // Every modification of ownership_ranges_ increments ownership_generation_ (including the ones done by the
  // barrier request through PixelShaderInterlockFullEdramBarrierPlaced).
  bool ownership_changed = ownership_generation_ != generation_before;
  if (barrier_requested) {
    ++stats.full_barrier;
  } else if (ownership_changed) {
    ++stats.full_changed;
  } else {
    ++stats.full_noop;
  }

  if (hit_entry != UINT32_MAX) {
    // Verification of a would-be skip. The entries were for generation_before (checked by the caller).
    ++stats.verified;
    const UpdateSkipEntry& entry = skip.entries[hit_entry];
    uint64_t map_hash_after = HashOwnershipRanges();
    uint32_t mismatch_flags = 0;
    if (skip.map_hash_valid && skip.map_hash != map_hash_before) {
      mismatch_flags |= kUpdateSkipMismatchStaleOwnership;
    }
    if (entry.range_count != range_count ||
        std::memcmp(entry.ranges, ranges, sizeof(UpdateSkipRange) * range_count)) {
      mismatch_flags |= kUpdateSkipMismatchRanges;
    }
    if (barrier_requested) {
      mismatch_flags |= kUpdateSkipMismatchBarrier;
    }
    if (ownership_changed || map_hash_after != map_hash_before) {
      mismatch_flags |= kUpdateSkipMismatchOwnershipChanged;
    }
    if (mismatch_flags) {
      ++stats.mismatches;
      ++stats.mismatches_total;
      if (stats.mismatches_logged < kUpdateSkipMismatchLogLimit) {
        ++stats.mismatches_logged;
        auto describe_ranges = [](const UpdateSkipRange* described_ranges, uint32_t count) {
          std::string description = "[";
          for (uint32_t i = 0; i < count; ++i) {
            const UpdateSkipRange& range = described_ranges[i];
            RenderTargetKey range_key;
            range_key.key = range.rt_key;
            description += fmt::format("{}{}: {} +{}t", i ? "; " : "", range.rt_bit_index,
                                       range_key.GetDebugName(), range.length_tiles);
          }
          description += "]";
          return description;
        };
        REXGPU_WARN(
            "[s6][CP8] Update skip mismatch {}: {} - msaa {} pitch {}t render targets 0x{:X} 64bpp 0x{:X} "
            "height {} clamp {} bases {} formats {}; ownership calls confirmed {}, now {}; barrier {}; "
            "ownership generation {} -> {}; ownership map hash confirmed {} {:016X}, before {:016X}, "
            "after {:016X} ({} ranges)",
            stats.mismatches_total, DescribeUpdateSkipMismatch(mismatch_flags), key.msaa_samples,
            key.pitch_tiles_at_32bpp, key.depth_and_color_rts_used_bits, key.rts_are_64bpp,
            key.height_used, key.mrt_edram_used_range_clamp_to_min,
            DescribeUpdateSkipValues(key.edram_bases, 1 + xenos::kMaxColorRenderTargets),
            DescribeUpdateSkipValues(key.resource_formats, 1 + xenos::kMaxColorRenderTargets),
            describe_ranges(entry.ranges, entry.range_count), describe_ranges(ranges, range_count),
            barrier_requested, generation_before, ownership_generation_,
            skip.map_hash_valid ? "valid" : "not recorded", skip.map_hash, map_hash_before,
            map_hash_after, ownership_ranges_.size());
      }
      // The original results have been used. Start over.
      skip.entry_count = 0;
      skip.entry_next = 0;
      return;
    }
  }

  if (barrier_requested || ownership_changed) {
    // Not a confirmation. If ownership_ranges_ was modified, the entries are for an older state and won't
    // match (generation); if not, they're still valid.
    return;
  }
  // Not if the setting in the key was changed (from another thread) during the update - the full run might
  // have used the other value.
  if (key.mrt_edram_used_range_clamp_to_min != uint32_t(REXCVAR_GET(mrt_edram_used_range_clamp_to_min))) {
    return;
  }
  if (!skip.entry_count || skip.generation != ownership_generation_) {
    // The first confirmation for this ownership state - the older entries were for an earlier one.
    skip.entry_count = 0;
    skip.entry_next = 0;
    skip.generation = ownership_generation_;
    skip.map_hash_valid = verify;
    skip.map_hash = verify ? HashOwnershipRanges() : 0;
  } else {
    for (uint32_t i = 0; i < skip.entry_count; ++i) {
      if (!std::memcmp(&skip.entries[i].key, &key, sizeof(key))) {
        // Already there (a verified would-be skip).
        return;
      }
    }
  }
  uint32_t entry_index;
  if (skip.entry_count < kUpdateSkipEntryCount) {
    entry_index = skip.entry_count++;
  } else {
    entry_index = skip.entry_next;
    skip.entry_next = (skip.entry_next + 1) % kUpdateSkipEntryCount;
  }
  UpdateSkipEntry& entry = skip.entries[entry_index];
  entry.key = key;
  entry.range_count = range_count;
  std::memcpy(entry.ranges, ranges, sizeof(UpdateSkipRange) * range_count);
}

uint64_t RenderTargetCache::HashOwnershipRanges() const {
  // wwe2k14 patch (Session 6, CP8): gpu_rt_cache_skip_unchanged_verify only.
  uint64_t hash = UINT64_C(0xCBF29CE484222325) ^ uint64_t(ownership_ranges_.size());
  auto mix = [&hash](uint64_t value) {
    hash = (hash ^ value) * UINT64_C(0x9E3779B97F4A7C15);
    hash ^= hash >> 31;
  };
  for (const auto& ownership_range_pair : ownership_ranges_) {
    const OwnershipRange& ownership_range = ownership_range_pair.second;
    mix(uint64_t(ownership_range_pair.first) | (uint64_t(ownership_range.end_tiles) << 32));
    mix(uint64_t(ownership_range.render_target.key) |
        (uint64_t(ownership_range.host_depth_render_target_unorm24.key) << 32));
    mix(uint64_t(ownership_range.host_depth_render_target_float24.key));
  }
  return hash;
}

void RenderTargetCache::ReportUpdateSkipStats() {
  // wwe2k14 patch (Session 6, CP8): gpu_rt_cache_skip_unchanged summary every kUpdateSkipStatsFrames frames.
  UpdateSkipStats& stats = update_skip_.stats;
  uint64_t frame = GetUpdateSkipStatsFrame();
  if (!frame) {
    return;
  }
  if (!stats.frame_start || frame < stats.frame_start) {
    stats.frame_start = frame;
    return;
  }
  if (frame - stats.frame_start < kUpdateSkipStatsFrames) {
    return;
  }
  REXGPU_INFO(
      "[s6][CP8] Render target update skip (skip {}, verify {}), frames {}-{}: {} interlock updates with "
      "render targets, {} skipped ({:.1f}%), {} would-be skips verified, {} mismatches; misses: {} nothing "
      "confirmed for the ownership state, {} different key; full runs: {} no effect, {} ownership changed, "
      "{} barrier; ownership generation {}; mismatches since start {}",
      REXCVAR_GET(gpu_rt_cache_skip_unchanged), REXCVAR_GET(gpu_rt_cache_skip_unchanged_verify),
      stats.frame_start, frame - 1, stats.updates, stats.skipped,
      GetUpdateSkipPercentage(stats.skipped, stats.updates), stats.verified, stats.mismatches,
      stats.miss_state, stats.miss_key, stats.full_noop, stats.full_changed, stats.full_barrier,
      ownership_generation_, stats.mismatches_total);
  stats.updates = 0;
  stats.skipped = 0;
  stats.verified = 0;
  stats.miss_state = 0;
  stats.miss_key = 0;
  stats.full_noop = 0;
  stats.full_changed = 0;
  stats.full_barrier = 0;
  stats.mismatches = 0;
  stats.frame_start = frame;
}

uint32_t RenderTargetCache::GetLastUpdateBoundRenderTargets(
    uint32_t* depth_and_color_formats_out) const {
  if (GetPath() != Path::kHostRenderTargets) {
    if (depth_and_color_formats_out) {
      std::memset(depth_and_color_formats_out, 0,
                  sizeof(uint32_t) * (1 + xenos::kMaxColorRenderTargets));
    }
    return 0;
  }
  uint32_t rts_used = 0;
  for (uint32_t i = 0; i < 1 + xenos::kMaxColorRenderTargets; ++i) {
    const RenderTarget* render_target = last_update_accumulated_render_targets_[i];
    if (!render_target) {
      if (depth_and_color_formats_out) {
        depth_and_color_formats_out[i] = 0;
      }
      continue;
    }
    rts_used |= uint32_t(1) << i;
    if (depth_and_color_formats_out) {
      depth_and_color_formats_out[i] = render_target->key().resource_format;
    }
  }
  return rts_used;
}

uint32_t RenderTargetCache::GetRenderTargetHeight(uint32_t pitch_tiles_at_32bpp,
                                                  xenos::MsaaSamples msaa_samples) const {
  if (!pitch_tiles_at_32bpp) {
    return 0;
  }
  // Down to the beginning of the render target in the next 11-bit EDRAM
  // addressing period.
  uint32_t tile_rows = (xenos::kEdramTileCount + (pitch_tiles_at_32bpp - 1)) / pitch_tiles_at_32bpp;
  // Clamp to the guest limit (tile padding should exceed it) and to the host
  // limit (tile padding mustn't exceed it).
  static_assert(!(xenos::kTexture2DCubeMaxWidthHeight % xenos::kEdramTileHeightSamples),
                "Maximum guest render target height is assumed to always be a multiple "
                "of an EDRAM tile height");
  uint32_t max_height_scaled = std::min(
      xenos::kTexture2DCubeMaxWidthHeight * draw_resolution_scale_y(), GetMaxRenderTargetHeight());
  uint32_t msaa_samples_y_log2 = uint32_t(msaa_samples >= xenos::MsaaSamples::k2X);
  uint32_t tile_height_samples_scaled = xenos::kEdramTileHeightSamples * draw_resolution_scale_y();
  tile_rows =
      std::min(tile_rows, (max_height_scaled << msaa_samples_y_log2) / tile_height_samples_scaled);
  assert_not_zero(tile_rows);
  return tile_rows * (xenos::kEdramTileHeightSamples >> msaa_samples_y_log2);
}

void RenderTargetCache::GetHostDepthStoreRectangleInfo(
    const Transfer::Rectangle& transfer_rectangle, xenos::MsaaSamples msaa_samples,
    HostDepthStoreRectangleConstant& rectangle_constant_out, uint32_t& group_count_x_out,
    uint32_t& group_count_y_out) const {
  // Initialize to all bits zeroed.
  HostDepthStoreRectangleConstant rectangle_constant;
  // 8 pixels is the resolve granularity, both clearing and tile size are
  // aligned to 8.
  assert_zero(transfer_rectangle.x_pixels & 7);
  assert_zero(transfer_rectangle.y_pixels & 7);
  assert_zero(transfer_rectangle.width_pixels & 7);
  assert_zero(transfer_rectangle.height_pixels & 7);
  assert_not_zero(transfer_rectangle.width_pixels);
  rectangle_constant.x_pixels_div_8 = transfer_rectangle.x_pixels >> 3;
  rectangle_constant.y_pixels_div_8 = transfer_rectangle.y_pixels >> 3;
  rectangle_constant.width_pixels_div_8_minus_1 = (transfer_rectangle.width_pixels >> 3) - 1;
  rectangle_constant_out = rectangle_constant;
  // 1 thread group = 64x8 host samples.
  uint32_t pixel_size_x = draw_resolution_scale_x()
                          << uint32_t(msaa_samples >= xenos::MsaaSamples::k4X);
  uint32_t pixel_size_y = draw_resolution_scale_y()
                          << uint32_t(msaa_samples >= xenos::MsaaSamples::k2X);
  group_count_x_out = (transfer_rectangle.width_pixels * pixel_size_x + 63) >> 6;
  group_count_y_out = (transfer_rectangle.height_pixels * pixel_size_y) >> 3;
}

void RenderTargetCache::GetResolveCopyRectanglesToDump(
    uint32_t base, uint32_t row_length, uint32_t rows, uint32_t pitch,
    std::vector<ResolveCopyDumpRectangle>& rectangles_out) const {
  rectangles_out.clear();
  assert_true(row_length <= pitch);
  row_length = std::min(row_length, pitch);
  if (!row_length || !rows) {
    return;
  }
  auto get_rectangles_in_extent = [&](uint32_t extent_start, uint32_t extent_end,
                                      uint32_t range_local_offset) {
    // Collect render targets owning ranges within the specified rectangle. The
    // first render target in the range may be before the lower_bound, only
    // being in the range with its tail.
    auto it = ownership_ranges_.lower_bound(extent_start);
    if (it != ownership_ranges_.cbegin()) {
      auto it_pre = std::prev(it);
      if (it_pre->second.end_tiles > extent_start) {
        it = it_pre;
      }
    }
    for (; it != ownership_ranges_.cend(); ++it) {
      uint32_t range_global_start = std::max(it->first, extent_start);
      if (range_global_start >= extent_end) {
        break;
      }
      RenderTargetKey rt_key = it->second.render_target;
      if (rt_key.IsEmpty()) {
        continue;
      }
      // Merge with other render target ranges with the same current ownership,
      // but different depth ownership, since it's not relevant to resolving.
      while (it != ownership_ranges_.cend()) {
        auto it_next = std::next(it);
        if (it_next == ownership_ranges_.cend() || it_next->first >= extent_end ||
            it_next->second.render_target != rt_key) {
          break;
        }
        it = it_next;
      }

      uint32_t range_local_start =
          range_local_offset + std::max(range_global_start, extent_start) - base;
      uint32_t range_local_end =
          range_local_offset + std::min(it->second.end_tiles, extent_end) - base;
      assert_true(range_local_start < range_local_end);

      uint32_t rows_start = range_local_start / pitch;
      uint32_t rows_end = (range_local_end + (pitch - 1)) / pitch;
      uint32_t row_first_start = range_local_start - rows_start * pitch;
      if (row_first_start >= row_length) {
        // The first row starts within the pitch padding.
        if (rows_start + 1 < rows_end) {
          // Multiple rows - start at the second.
          ++rows_start;
          row_first_start = 0;
        } else {
          // Single row - nothing to dump.
          continue;
        }
      }

      auto it_rt = render_targets_.find(rt_key);
      assert_true(it_rt != render_targets_.cend());
      assert_not_null(it_rt->second);
      // Don't include pitch padding in the last row.
      rectangles_out.emplace_back(
          it_rt->second, rows_start, rows_end - rows_start, row_first_start,
          std::min(pitch - (rows_end * pitch - range_local_end), row_length));
    }
  };
  uint32_t resolve_area_end = base + (rows - 1) * pitch + row_length;
  get_rectangles_in_extent(base, std::min(resolve_area_end, xenos::kEdramTileCount), 0);
  if (resolve_area_end > xenos::kEdramTileCount) {
    // The resolve area goes to the next EDRAM addressing period.
    get_rectangles_in_extent(0, std::min(resolve_area_end & (xenos::kEdramTileCount - 1), base),
                             xenos::kEdramTileCount);
  }
}

void RenderTargetCache::GetResolveCopyDispatchesToDump(
    uint32_t base, uint32_t row_length, uint32_t rows, uint32_t pitch,
    std::vector<ResolveCopyDumpRectangle>& rectangles_out,
    std::vector<ResolveCopyDispatch>& dispatches_out) const {
  GetResolveCopyRectanglesToDump(base, row_length, rows, pitch, rectangles_out);
  dispatches_out.clear();
  for (uint32_t rectangle_index = 0; rectangle_index < uint32_t(rectangles_out.size());
       ++rectangle_index) {
    const ResolveCopyDumpRectangle& rectangle = rectangles_out[rectangle_index];
    ResolveCopyDumpRectangle::Dispatch
        rectangle_dispatches[ResolveCopyDumpRectangle::kMaxDispatches];
    uint32_t dispatch_count = rectangle.GetDispatches(pitch, row_length, rectangle_dispatches);
    for (uint32_t i = 0; i < dispatch_count; ++i) {
      dispatches_out.emplace_back(rectangle_index, rectangle_dispatches[i]);
    }
  }
}

bool RenderTargetCache::PrepareHostRenderTargetsResolveClear(
    const draw_util::ResolveInfo& resolve_info, Transfer::Rectangle& clear_rectangle_out,
    RenderTarget*& depth_render_target_out, std::vector<Transfer>& depth_transfers_out,
    RenderTarget*& color_render_target_out, std::vector<Transfer>& color_transfers_out) {
  assert_true(GetPath() == Path::kHostRenderTargets);

  uint32_t pitch_tiles_at_32bpp;
  uint32_t base_offset_tiles_at_32bpp;
  xenos::MsaaSamples msaa_samples;
  if (resolve_info.IsClearingDepth()) {
    pitch_tiles_at_32bpp = resolve_info.depth_edram_info.pitch_tiles;
    base_offset_tiles_at_32bpp =
        resolve_info.depth_edram_info.base_tiles - resolve_info.depth_original_base;
    msaa_samples = resolve_info.depth_edram_info.msaa_samples;
  } else if (resolve_info.IsClearingColor()) {
    pitch_tiles_at_32bpp = resolve_info.color_edram_info.pitch_tiles;
    base_offset_tiles_at_32bpp =
        resolve_info.color_edram_info.base_tiles - resolve_info.color_original_base;
    if (resolve_info.color_edram_info.format_is_64bpp) {
      assert_zero(pitch_tiles_at_32bpp & 1);
      pitch_tiles_at_32bpp >>= 1;
      assert_zero(base_offset_tiles_at_32bpp & 1);
      base_offset_tiles_at_32bpp >>= 1;
    }
    msaa_samples = resolve_info.color_edram_info.msaa_samples;
  } else {
    return false;
  }
  assert_true(msaa_samples <= xenos::MsaaSamples::k4X);
  if (!pitch_tiles_at_32bpp) {
    return false;
  }
  uint32_t msaa_samples_x_log2 = uint32_t(msaa_samples >= xenos::MsaaSamples::k4X);
  uint32_t msaa_samples_y_log2 = uint32_t(msaa_samples >= xenos::MsaaSamples::k2X);
  if (pitch_tiles_at_32bpp > ((xenos::kTexture2DCubeMaxWidthHeight << msaa_samples_x_log2) +
                              (xenos::kEdramTileWidthSamples - 1)) /
                                 xenos::kEdramTileWidthSamples) {
    REXGPU_ERROR(
        "Surface pitch in 80-sample groups {} at {}x MSAA larger than the "
        "maximum texture width {} specified by the guest in a resolve",
        pitch_tiles_at_32bpp, uint32_t(1) << uint32_t(msaa_samples),
        xenos::kTexture2DCubeMaxWidthHeight);
    return false;
  }
  uint32_t pitch_pixels =
      pitch_tiles_at_32bpp * (xenos::kEdramTileWidthSamples >> msaa_samples_x_log2);
  uint32_t pitch_pixels_scaled = pitch_pixels * draw_resolution_scale_x();
  uint32_t max_render_target_width = GetMaxRenderTargetWidth();
  if (pitch_pixels_scaled > max_render_target_width) {
    // TODO(Triang3l): If really needed for some game on some device, clamp the
    // pitch the same way as explained in the comment in Update.
    REXGPU_ERROR(
        "Surface pitch aligned to EDRAM tiles and resolution-scaled {} larger "
        "than the maximum host render target width {} in a resolve",
        pitch_pixels_scaled, max_render_target_width);
    return false;
  }

  uint32_t render_target_height_pixels = GetRenderTargetHeight(pitch_tiles_at_32bpp, msaa_samples);
  uint32_t base_offset_rows_at_32bpp = base_offset_tiles_at_32bpp / pitch_tiles_at_32bpp;
  Transfer::Rectangle clear_rectangle;
  clear_rectangle.x_pixels =
      std::min((base_offset_tiles_at_32bpp - base_offset_rows_at_32bpp * pitch_tiles_at_32bpp) *
                       (xenos::kEdramTileWidthSamples >> msaa_samples_x_log2) +
                   (uint32_t(resolve_info.coordinate_info.edram_offset_x_div_8) << 3),
               pitch_pixels);
  clear_rectangle.y_pixels =
      std::min(base_offset_rows_at_32bpp * (xenos::kEdramTileHeightSamples >> msaa_samples_y_log2) +
                   (uint32_t(resolve_info.coordinate_info.edram_offset_y_div_8) << 3),
               render_target_height_pixels);
  clear_rectangle.width_pixels = std::min(uint32_t(resolve_info.coordinate_info.width_div_8) << 3,
                                          pitch_pixels - clear_rectangle.x_pixels);
  clear_rectangle.height_pixels = std::min(uint32_t(resolve_info.height_div_8) << 3,
                                           render_target_height_pixels - clear_rectangle.y_pixels);
  if (!clear_rectangle.width_pixels || !clear_rectangle.height_pixels) {
    // Outside the pitch / height (or initially specified as 0).
    return false;
  }

  // Change ownership of the tiles containing the area to be cleared, so the
  // up-to-date host render target for the cleared range will be the cleared
  // one.
  uint32_t clear_start_tiles_at_32bpp =
      ((clear_rectangle.y_pixels << msaa_samples_y_log2) / xenos::kEdramTileHeightSamples) *
          pitch_tiles_at_32bpp +
      (clear_rectangle.x_pixels << msaa_samples_x_log2) / xenos::kEdramTileWidthSamples;
  uint32_t clear_length_tiles_at_32bpp =
      (((clear_rectangle.y_pixels + clear_rectangle.height_pixels - 1) << msaa_samples_y_log2) /
       xenos::kEdramTileHeightSamples) *
          pitch_tiles_at_32bpp +
      ((clear_rectangle.x_pixels + clear_rectangle.width_pixels - 1) << msaa_samples_x_log2) /
          xenos::kEdramTileWidthSamples +
      1 - clear_start_tiles_at_32bpp;
  // Up to the range from the base in the current 11 tile index bits to the base
  // in the next 11 tile index bits after wrapping can be cleared.
  uint32_t depth_clear_start_tiles_base_relative = 0;
  uint32_t depth_clear_length_tiles = 0;
  if (resolve_info.IsClearingDepth()) {
    depth_clear_start_tiles_base_relative =
        std::min(clear_start_tiles_at_32bpp, xenos::kEdramTileCount);
    depth_clear_length_tiles =
        std::min(clear_start_tiles_at_32bpp + clear_length_tiles_at_32bpp, xenos::kEdramTileCount) -
        depth_clear_start_tiles_base_relative;
  }
  uint32_t color_clear_start_tiles_base_relative = 0;
  uint32_t color_clear_length_tiles = 0;
  if (resolve_info.IsClearingColor()) {
    color_clear_start_tiles_base_relative =
        std::min(clear_start_tiles_at_32bpp << resolve_info.color_edram_info.format_is_64bpp,
                 xenos::kEdramTileCount);
    color_clear_length_tiles = std::min((clear_start_tiles_at_32bpp + clear_length_tiles_at_32bpp)
                                            << resolve_info.color_edram_info.format_is_64bpp,
                                        xenos::kEdramTileCount) -
                               color_clear_start_tiles_base_relative;
  }
  if (depth_clear_length_tiles && color_clear_length_tiles) {
    // Prevent overlap - clear the depth only until the color, the color only
    // until the depth, in the current or the next 11 bits of the tile index.
    uint32_t depth_clear_start_tiles_wrapped =
        (resolve_info.depth_original_base + depth_clear_start_tiles_base_relative) &
        (xenos::kEdramTileCount - 1);
    uint32_t color_clear_start_tiles_wrapped =
        (resolve_info.color_original_base + color_clear_start_tiles_base_relative) &
        (xenos::kEdramTileCount - 1);
    depth_clear_length_tiles =
        std::min(depth_clear_length_tiles,
                 ((color_clear_start_tiles_wrapped < depth_clear_start_tiles_wrapped)
                      ? xenos::kEdramTileCount
                      : 0) +
                     color_clear_start_tiles_wrapped - depth_clear_start_tiles_wrapped);
    color_clear_length_tiles =
        std::min(color_clear_length_tiles,
                 ((depth_clear_start_tiles_wrapped < color_clear_start_tiles_wrapped)
                      ? xenos::kEdramTileCount
                      : 0) +
                     depth_clear_start_tiles_wrapped - color_clear_start_tiles_wrapped);
  }

  RenderTargetKey depth_render_target_key;
  RenderTarget* depth_render_target = nullptr;
  if (depth_clear_length_tiles) {
    depth_render_target_key.base_tiles = resolve_info.depth_original_base;
    depth_render_target_key.pitch_tiles_at_32bpp = pitch_tiles_at_32bpp;
    depth_render_target_key.msaa_samples = msaa_samples;
    depth_render_target_key.is_depth = 1;
    depth_render_target_key.resource_format = resolve_info.depth_edram_info.format;
    depth_render_target = GetOrCreateRenderTarget(depth_render_target_key);
    if (!depth_render_target) {
      // Failed to create the depth render target, don't clear it.
      depth_render_target_key = RenderTargetKey();
      depth_clear_length_tiles = 0;
    }
  }
  RenderTargetKey color_render_target_key;
  RenderTarget* color_render_target = nullptr;
  if (color_clear_length_tiles) {
    color_render_target_key.base_tiles = resolve_info.color_original_base;
    color_render_target_key.pitch_tiles_at_32bpp = pitch_tiles_at_32bpp;
    color_render_target_key.msaa_samples = msaa_samples;
    color_render_target_key.is_depth = 0;
    color_render_target_key.resource_format = uint32_t(GetColorResourceFormat(
        xenos::ColorRenderTargetFormat(resolve_info.color_edram_info.format)));
    color_render_target = GetOrCreateRenderTarget(color_render_target_key);
    if (!color_render_target) {
      // Failed to create the color render target, don't clear it.
      color_render_target_key = RenderTargetKey();
      color_clear_length_tiles = 0;
    }
  }
  if (!depth_clear_length_tiles && !color_clear_length_tiles) {
    // Complete overlap, or failed to create all the render targets.
    return false;
  }

  clear_rectangle_out = clear_rectangle;
  depth_render_target_out = depth_render_target;
  depth_transfers_out.clear();
  if (depth_render_target) {
    ChangeOwnership(depth_render_target_key, depth_clear_start_tiles_base_relative,
                    depth_clear_length_tiles, &depth_transfers_out, &clear_rectangle);
  }
  color_render_target_out = color_render_target;
  color_transfers_out.clear();
  if (color_render_target) {
    ChangeOwnership(color_render_target_key, color_clear_start_tiles_base_relative,
                    color_clear_length_tiles, &color_transfers_out, &clear_rectangle);
  }
  return true;
}

void RenderTargetCache::PixelShaderInterlockFullEdramBarrierPlaced() {
  assert_true(GetPath() == Path::kPixelShaderInterlock);
  ++ownership_generation_;  // wwe2k14 patch (Session 6, CP8)
  // Clear ownership - any overlap of data written before the barrier is safe.
  OwnershipRange empty_range(xenos::kEdramTileCount, RenderTargetKey(), RenderTargetKey(),
                             RenderTargetKey());
  if (ownership_ranges_.size() == 1) {
    // Do not reallocate map elements if not needed (either nothing drawn since
    // the last barrier, or all of the EDRAM is owned by one render target).
    // The ownership map contains no gaps - the first element should always be
    // at 0.
    assert_true(!ownership_ranges_.begin()->first);
    OwnershipRange& all_edram_range = ownership_ranges_.begin()->second;
    assert_true(all_edram_range.end_tiles == xenos::kEdramTileCount);
    all_edram_range = empty_range;
    return;
  }
  ownership_ranges_.clear();
  ownership_ranges_.emplace(0, empty_range);
}

RenderTargetCache::RenderTarget* RenderTargetCache::GetOrCreateRenderTarget(RenderTargetKey key) {
  assert_true(GetPath() == Path::kHostRenderTargets);
  auto it_rt = render_targets_.find(key);
  RenderTarget* render_target;
  if (it_rt != render_targets_.end()) {
    render_target = it_rt->second;
  } else {
    render_target = CreateRenderTarget(key);
    uint32_t width = key.GetWidth();
    uint32_t height = GetRenderTargetHeight(key.pitch_tiles_at_32bpp, key.msaa_samples);
    if (render_target) {
      REXGPU_DEBUG(
          "Created a {}x{} {}xMSAA {} render target with guest format {} at "
          "EDRAM base {}",
          width, height, uint32_t(1) << uint32_t(key.msaa_samples),
          key.is_depth ? "depth" : "color", static_cast<uint32_t>(key.resource_format),
          static_cast<uint32_t>(key.base_tiles));
    } else {
      REXGPU_ERROR(
          "Failed to create a {}x{} {}xMSAA {} render target with guest format "
          "{} at EDRAM base {}",
          width, height, uint32_t(1) << uint32_t(key.msaa_samples),
          key.is_depth ? "depth" : "color", static_cast<uint32_t>(key.resource_format),
          static_cast<uint32_t>(key.base_tiles));
    }
    // Insert even if failed to create, not to try to create again.
    render_targets_.emplace(key, render_target);
  }
  return render_target;
}

bool RenderTargetCache::WouldOwnershipChangeRequireTransfers(RenderTargetKey dest,
                                                             uint32_t start_tiles_base_relative,
                                                             uint32_t length_tiles) const {
  // xenos::kEdramTileCount with length 0 is fine if both the start and the end
  // are clamped to xenos::kEdramTileCount.
  assert_true(start_tiles_base_relative <= (xenos::kEdramTileCount - uint32_t(length_tiles != 0)));
  assert_true(length_tiles <= xenos::kEdramTileCount);
  if (length_tiles == 0) {
    return false;
  }
  bool host_depth_encoding_different = dest.is_depth && GetPath() == Path::kHostRenderTargets &&
                                       IsHostDepthEncodingDifferent(dest.GetDepthFormat());
  auto would_require_transfers_in_extent = [&](uint32_t extent_start, uint32_t extent_end) -> bool {
    // The map contains consecutive ranges, merged if the adjacent ones are the
    // same. Find the range starting at >= the start. A portion of the range
    // preceding it may be intersecting the render target's range (or even fully
    // contain it).
    auto it = ownership_ranges_.lower_bound(extent_start);
    if (it != ownership_ranges_.begin()) {
      auto it_pre = std::prev(it);
      if (it_pre->second.end_tiles > extent_start) {
        it = it_pre;
      }
    }
    for (; it != ownership_ranges_.end(); ++it) {
      if (it->first >= extent_end) {
        // Outside the touched extent already.
        break;
      }
      if (it->second.IsOwnedBy(dest, host_depth_encoding_different)) {
        // Already owned by the needed render target - no need to transfer
        // anything.
        continue;
      }
      RenderTargetKey transfer_source = it->second.render_target;
      // Only perform the transfer when actually changing the latest owner, not
      // just the latest host depth owner - the transfer source is expected to
      // be different than the destination.
      if (!transfer_source.IsEmpty() && transfer_source != dest) {
        return true;
      }
    }
    return false;
  };
  // start_tiles_base_relative may already be in the next 11 bits - wrap the
  // start tile index to use the same code as if that was not the case.
  uint32_t start_tiles =
      (dest.base_tiles + start_tiles_base_relative) & (xenos::kEdramTileCount - 1);
  uint32_t end_tiles = start_tiles + length_tiles;
  if (would_require_transfers_in_extent(start_tiles, std::min(end_tiles, xenos::kEdramTileCount))) {
    return true;
  }
  if (end_tiles > xenos::kEdramTileCount) {
    // The check extent goes to the next EDRAM addressing period.
    if (would_require_transfers_in_extent(
            0, std::min(end_tiles & (xenos::kEdramTileCount - 1), start_tiles))) {
      return true;
    }
  }
  return false;
}

void RenderTargetCache::ChangeOwnership(RenderTargetKey dest, uint32_t start_tiles_base_relative,
                                        uint32_t length_tiles,
                                        std::vector<Transfer>* transfers_append_out,
                                        const Transfer::Rectangle* resolve_clear_cutout) {
  // xenos::kEdramTileCount with length 0 is fine if both the start and the end
  // are clamped to xenos::kEdramTileCount.
  assert_true(start_tiles_base_relative <= (xenos::kEdramTileCount - uint32_t(length_tiles != 0)));
  assert_true(length_tiles <= xenos::kEdramTileCount);
  if (length_tiles == 0) {
    return;
  }
  uint32_t dest_pitch_tiles = dest.GetPitchTiles();
  bool dest_is_64bpp = dest.Is64bpp();
  bool host_depth_encoding_different = dest.is_depth && GetPath() == Path::kHostRenderTargets &&
                                       IsHostDepthEncodingDifferent(dest.GetDepthFormat());
  auto change_ownership_in_extent = [&](uint32_t extent_start, uint32_t extent_end) {
    // The map contains consecutive ranges, merged if the adjacent ones are the
    // same. Find the range starting at >= the start. A portion of the range
    // preceding it may be intersecting the render target's range (or even fully
    // contain it) - split it into the untouched head and the claimed tail if
    // needed.
    auto it = ownership_ranges_.lower_bound(extent_start);
    if (it != ownership_ranges_.begin()) {
      auto it_pre = std::prev(it);
      if (it_pre->second.end_tiles > extent_start &&
          !it_pre->second.IsOwnedBy(dest, host_depth_encoding_different)) {
        // Different render target overlapping the range - split the head.
        ++ownership_generation_;  // wwe2k14 patch (Session 6, CP8)
        ownership_ranges_.emplace(extent_start, it_pre->second);
        it_pre->second.end_tiles = extent_start;
        // Let the next loop do the transfer and needed merging and splitting
        // starting from the added tail.
        it = std::next(it_pre);
      }
    }
    while (it != ownership_ranges_.end()) {
      if (it->first >= extent_end) {
        // Outside the touched extent already.
        break;
      }
      if (it->second.IsOwnedBy(dest, host_depth_encoding_different)) {
        // Already owned by the needed render target - no need to transfer
        // anything.
        ++it;
        continue;
      }
      // wwe2k14 patch (Session 6, CP8): everything from here to the end of the
      // iteration modifies ownership_ranges_ (the claim always assigns).
      ++ownership_generation_;
      // Take over the current range. Handle the tail - may be outside the range
      // (split in this case) or within it.
      if (it->second.end_tiles > extent_end) {
        // Split the tail.
        ownership_ranges_.emplace(extent_end, it->second);
        it->second.end_tiles = extent_end;
      }
      if (transfers_append_out) {
        RenderTargetKey transfer_source = it->second.render_target;
        // Only perform the copying when actually changing the latest owner, not
        // just the latest host depth owner - the transfer source is expected to
        // be different than the destination.
        if (!transfer_source.IsEmpty() && transfer_source != dest) {
          uint32_t transfer_end_tiles = std::min(it->second.end_tiles, extent_end);
          if (!resolve_clear_cutout ||
              Transfer::GetRangeRectangles(it->first, transfer_end_tiles, dest.base_tiles,
                                           dest_pitch_tiles, dest.msaa_samples, dest_is_64bpp,
                                           nullptr, resolve_clear_cutout)) {
            RenderTargetKey transfer_host_depth_source =
                host_depth_encoding_different
                    ? it->second.GetHostDepthRenderTarget(dest.GetDepthFormat())
                    : RenderTargetKey();
            if (transfer_host_depth_source == transfer_source) {
              // Same render target, don't provide a separate host depth source.
              transfer_host_depth_source = RenderTargetKey();
            }
            if (!transfers_append_out->empty() &&
                transfers_append_out->back().end_tiles == it->first &&
                transfers_append_out->back().source->key() == transfer_source &&
                ((transfers_append_out->back().host_depth_source == nullptr) ==
                 transfer_host_depth_source.IsEmpty()) &&
                (transfer_host_depth_source.IsEmpty() ||
                 transfers_append_out->back().host_depth_source->key() ==
                     transfer_host_depth_source)) {
              // Extend the last transfer if, for example, transferring color,
              // but host depth is different.
              transfers_append_out->back().end_tiles = transfer_end_tiles;
            } else {
              auto transfer_source_rt_it = render_targets_.find(transfer_source);
              if (transfer_source_rt_it != render_targets_.end()) {
                assert_not_null(transfer_source_rt_it->second);
                auto transfer_host_depth_source_rt_it =
                    !transfer_host_depth_source.IsEmpty()
                        ? render_targets_.find(transfer_host_depth_source)
                        : render_targets_.end();
                if (transfer_host_depth_source.IsEmpty() ||
                    transfer_host_depth_source_rt_it != render_targets_.end()) {
                  assert_false(transfer_host_depth_source_rt_it != render_targets_.end() &&
                               !transfer_host_depth_source_rt_it->second);
                  transfers_append_out->emplace_back(
                      it->first, transfer_end_tiles, transfer_source_rt_it->second,
                      transfer_host_depth_source_rt_it != render_targets_.end()
                          ? transfer_host_depth_source_rt_it->second
                          : nullptr);
                }
              }
            }
          }
        }
      }
      // Claim the current range.
      it->second.render_target = dest;
      if (host_depth_encoding_different) {
        it->second.GetHostDepthRenderTarget(dest.GetDepthFormat()) = dest;
      }
      // Check if can merge with the next range after claiming.
      std::map<uint32_t, OwnershipRange>::iterator it_next;
      if (it != ownership_ranges_.end()) {
        it_next = std::next(it);
        if (it_next != ownership_ranges_.end() && it_next->second.AreOwnersSame(it->second)) {
          // Merge with the next range.
          it->second.end_tiles = it_next->second.end_tiles;
          auto it_after = std::next(it_next);
          ownership_ranges_.erase(it_next);
          it_next = it_after;
        }
      } else {
        it_next = ownership_ranges_.end();
      }
      // Check if can merge with the previous range after claiming and merging
      // with the next (thus obtaining the correct end pointer).
      if (it != ownership_ranges_.begin()) {
        auto it_prev = std::prev(it);
        if (it_prev->second.AreOwnersSame(it->second)) {
          it_prev->second.end_tiles = it->second.end_tiles;
          ownership_ranges_.erase(it);
        }
      }
      it = it_next;
    }
  };
  // start_tiles_base_relative may already be in the next 11 bits - wrap the
  // start tile index to use the same code as if that was not the case.
  uint32_t start_tiles =
      (dest.base_tiles + start_tiles_base_relative) & (xenos::kEdramTileCount - 1);
  uint32_t end_tiles = start_tiles + length_tiles;
  change_ownership_in_extent(start_tiles, std::min(end_tiles, xenos::kEdramTileCount));
  if (end_tiles > xenos::kEdramTileCount) {
    // The ownership change extent goes to the next EDRAM addressing period.
    change_ownership_in_extent(0, std::min(end_tiles & (xenos::kEdramTileCount - 1), start_tiles));
  }
}

}  // namespace rex::graphics
