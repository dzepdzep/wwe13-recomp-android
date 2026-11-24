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
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <rex/assert.h>
#include <rex/chrono/clock.h>
#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/pipeline/texture/cache.h>
#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/util/bit_inline.h>
#include <rex/graphics/xenos.h>
#include <rex/logging.h>

// wwe2k14 patch (Session 43, S0b-lite): defined in vulkan/texture_cache.cpp.
REXCVAR_DECLARE(bool, vulkan_texture_load_direct_unscaled);
REXCVAR_DECLARE(bool, gpu_tex_planes_unwatched);  // (Session 95, r6tex: shared_memory.cpp)
#include <rex/math.h>

REXCVAR_DEFINE_INT32(texture_cache_memory_limit_render_to_texture, 24, "GPU",
                     "Texture cache memory limit for render-to-texture (MB)")
    .range(1, 256)
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_INT32(texture_cache_memory_limit_soft, 384, "GPU",
                     "Soft texture cache memory limit (MB)")
    .range(64, 4096)
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_INT32(texture_cache_memory_limit_hard, 768, "GPU",
                     "Hard texture cache memory limit (MB)")
    .range(128, 8192)
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_INT32(texture_cache_memory_limit_soft_lifetime, 30, "GPU",
                     "Soft texture cache memory limit lifetime (seconds)")
    .range(1, 3600);

REXCVAR_DEFINE_BOOL(gpu_3d_to_2d_texture, true, "GPU",
                    "Sample problematic 3D textures through 2D-compatible wrappers")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(anisotropic_override, 3, "GPU",
                     "Forces anisotropic filtering for eligible textures.\n"
                     "Higher values keep textures sharper at oblique angles, but increase texture "
                     "sampling cost.\n"
                     " -1 = No override\n"
                     "  0 = Disable anisotropic filtering\n"
                     "  1 = Force 1x anisotropic filtering\n"
                     "  2 = Force 2x anisotropic filtering\n"
                     "  3 = Force 4x anisotropic filtering\n"
                     "  4 = Force 8x anisotropic filtering\n"
                     "  5 = Force 16x anisotropic filtering")
    .range(-1, 5)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(draw_resolution_scale_x, 1, "GPU", "Draw resolution scale X (1 = no scaling)")
    .range(1, 8)
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_INT32(draw_resolution_scale_y, 1, "GPU", "Draw resolution scale Y (1 = no scaling)")
    .range(1, 8)
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_INT32(resolution_scale, 1, "GPU",
                     "Draw resolution scale for both X and Y axes (same as setting "
                     "draw_resolution_scale_x and draw_resolution_scale_y)")
    .range(1, 8)
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_BOOL(pre_mask_resolve_l2_block, true, "GPU",
                    "Pre-mask scaled resolve L2 blocks to the write range before iterating");

// wwe2k14 patch (Session 7, TEXREQ): RequestTextures runs for every draw. In the heavy CH103 entrance window
// (Session 6 run 111, all Session 6 settings on) it was ~5.5% of the command processor thread, 4.1% self, and
// ~63% of its time was sampled on the instructions right after the texture_became_outdated_ exchange - a
// locked read-modify-write done on every draw, although the flag is almost never set. Draws also re-derive
// texture bindings from fetch constants rewritten with the same value, and the Vulkan part re-marks and
// re-checks the same textures for every draw of a run (the crowd: thousands of draws with the same textures).
REXCVAR_DEFINE_BOOL(gpu_request_textures_skip_unchanged, false, "GPU",
                    "Per-draw texture requests skip what would change nothing: read the 'a texture became "
                    "outdated' flag before exchanging it, return early when no used binding is out of sync, "
                    "keep a binding whose fetch constant was rewritten with the value it was derived from, and "
                    "(Vulkan) skip the LRU marks and usage checks when the used bindings, the submission and "
                    "every texture usage are unchanged since they last ran. Same results; read per draw");
REXCVAR_DEFINE_BOOL(gpu_request_textures_skip_unchanged_verify, false, "GPU",
                    "Shadow-check gpu_request_textures_skip_unchanged (works with it on or off): find what would "
                    "be skipped, but run the full path and use its results; count and log ([s7][TEXREQ]) every "
                    "would-be skip the full path changed something in, summary every 300 frames");

// DEFINE_int32(
//     draw_resolution_scale_x, 1,
//     "Integer pixel width scale used for scaling the rendering resolution "
//     "opaquely to the game.\n"
//     "1, 2 and 3 may be supported, but support of anything above 1 depends on "
//     "the device properties, such as whether it supports sparse binding / tiled "
//     "resources, the number of virtual address bits per resource, and other "
//     "factors.\n"
//     "Various effects and parts of game rendering pipelines may work "
//     "incorrectly as pixels become ambiguous from the game's perspective and "
//     "because half-pixel offset (which normally doesn't affect coverage when "
//     "MSAA isn't used) becomes full-pixel.",
//     "GPU");
// DEFINE_int32(
//     draw_resolution_scale_y, 1,
//     "Integer pixel width scale used for scaling the rendering resolution "
//     "opaquely to the game.\n"
//     "See draw_resolution_scale_x for more information.",
//     "GPU");
// DEFINE_uint32(
//     texture_cache_memory_limit_soft, 384,
//     "Maximum host texture memory usage (in megabytes) above which old textures "
//     "will be destroyed.",
//     "GPU");
// DEFINE_uint32(
//     texture_cache_memory_limit_soft_lifetime, 30,
//     "Seconds a texture should be unused to be considered old enough to be "
//     "deleted if texture memory usage exceeds texture_cache_memory_limit_soft.",
//     "GPU");
// DEFINE_uint32(
//     texture_cache_memory_limit_hard, 768,
//     "Maximum host texture memory usage (in megabytes) above which textures "
//     "will be destroyed as soon as possible.",
//     "GPU");
// DEFINE_uint32(
//     texture_cache_memory_limit_render_to_texture, 24,
//     "Part of the host texture memory budget (in megabytes) that will be scaled "
//     "by the current drawing resolution scale.\n"
//     "If texture_cache_memory_limit_soft, for instance, is 384, and this is 24, "
//     "it will be assumed that the game will be using roughly 24 MB of "
//     "render-to-texture (resolve) targets and 384 - 24 = 360 MB of regular "
//     "textures - so with 2x2 resolution scaling, the soft limit will be 360 + "
//     "96 MB, and with 3x3, it will be 360 + 216 MB.",
//     "GPU");

REXCVAR_DEFINE_INT32(gpu_tex_bind_memo, 0, "GPU",
                     "Session 95 (r6tex): the texture binding memo, bits: 1 a slot's key, signs, swizzle and texture from its fetch words seen before; 2 a slot whose page is in no owned page skips the view maths; "
                     "4 an owned page's view from a memo; 8 the record of the draw that followed this one last time is fetched ahead into the cache. 0 = off, 15 = all. Logs [tex] once and the [d3d][typed] textures split line")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_tex_load_census, false, "GPU",
                    "Session 95 (r6tex, test): on the sampled frames, which textures are loaded and what each costs: the [tex][loads] lines beside the textures split line")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_tex_bind_memo_ab, 0, "GPU",
                     "Session 95 (r6tex, test): with gpu_d3d_wall_stats, these bits of gpu_tex_bind_memo flip at every stats line: an in-run A/B of the command thread's cycles")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_tex_bind_memo_verify, false, "GPU",
                    "Session 95 (r6tex, test): every gpu_tex_bind_memo hit is compared with the full way, whose answer is used; differences are counted in the textures split line and the first logged")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

namespace rex::graphics {

const TextureCache::LoadShaderInfo TextureCache::load_shader_info_[kLoadShaderCount] = {
    // k8bpb
    {3, 4, 1, 4},
    // k16bpb
    {4, 4, 2, 4},
    // k32bpb
    {4, 4, 4, 3},
    // k64bpb
    {4, 4, 8, 2},
    // k128bpb
    {4, 4, 16, 1},
    // kR5G5B5A1ToB5G5R5A1
    {4, 4, 2, 4},
    // kR5G6B5ToB5G6R5
    {4, 4, 2, 4},
    // kR5G6B5ToRGBA8
    {4, 4, 4, 4},
    // kR5G5B6ToB5G6R5WithRBGASwizzle
    {4, 4, 2, 4},
    // kRGBA4ToBGRA4
    {4, 4, 2, 4},
    // kRGBA4ToARGB4
    {4, 4, 2, 4},
    // kRGBA4ToRGBA8
    {4, 4, 4, 4},
    // kGBGR8ToGRGB8
    {4, 4, 4, 3},
    // kGBGR8ToRGB8
    {4, 4, 8, 3},
    // kBGRG8ToRGBG8
    {4, 4, 4, 3},
    // kBGRG8ToRGB8
    {4, 4, 8, 3},
    // kR10G11B11ToRGBA16
    {4, 4, 8, 3},
    // kR10G11B11ToRGBA16SNorm
    {4, 4, 8, 3},
    // kR11G11B10ToRGBA16
    {4, 4, 8, 3},
    // kR11G11B10ToRGBA16SNorm
    {4, 4, 8, 3},
    // kR16UNormToFloat
    {4, 4, 2, 4},
    // kR16SNormToFloat
    {4, 4, 2, 4},
    // kRG16UNormToFloat
    {4, 4, 4, 3},
    // kRG16SNormToFloat
    {4, 4, 4, 3},
    // kRGBA16UNormToFloat
    {4, 4, 8, 2},
    // kRGBA16SNormToFloat
    {4, 4, 8, 2},
    // kDXT1ToRGBA8
    {4, 4, 4, 2},
    // kDXT3ToRGBA8
    {4, 4, 4, 1},
    // kDXT5ToRGBA8
    {4, 4, 4, 1},
    // kDXNToRG8
    {4, 4, 2, 1},
    // kDXT3A
    {4, 4, 1, 2},
    // kDXT3AAs1111ToBGRA4
    {4, 4, 2, 2},
    // kDXT3AAs1111ToARGB4
    {4, 4, 2, 2},
    // kDXT5AToR8
    {4, 4, 1, 2},
    // kCTX1
    {4, 4, 2, 2},
    // kDepthUnorm
    {4, 4, 4, 3},
    // kDepthFloat
    {4, 4, 4, 3},
    // kDXT1ToASTC4x4 (Session 93: 4 BC1 blocks of 8 bytes -> 4 ASTC blocks of 16 bytes)
    {4, 4, 16, 2},
    // kDXT5AToEACR11 (4 BC4 blocks -> 4 EAC blocks of 8 bytes)
    {4, 4, 8, 2},
    // kDXNToEACRG11 (2 BC5 blocks -> 2 EAC RG11 blocks of 16 bytes)
    {4, 4, 16, 1},
};

TextureCache::TextureCache(const RegisterFile& register_file, SharedMemory& shared_memory,
                           uint32_t draw_resolution_scale_x, uint32_t draw_resolution_scale_y)
    : register_file_(register_file),
      shared_memory_(shared_memory),
      draw_resolution_scale_x_(draw_resolution_scale_x),
      draw_resolution_scale_y_(draw_resolution_scale_y) {
  assert_true(draw_resolution_scale_x >= 1);
  assert_true(draw_resolution_scale_x <= kMaxDrawResolutionScaleAlongAxis);
  assert_true(draw_resolution_scale_y >= 1);
  assert_true(draw_resolution_scale_y <= kMaxDrawResolutionScaleAlongAxis);

  if (draw_resolution_scale_x > 1 || draw_resolution_scale_y > 1) {
    constexpr uint32_t kScaledResolvePageDwordCount = SharedMemory::kBufferSize / 4096 / 32;
    scaled_resolve_pages_ = std::unique_ptr<uint32_t[]>(new uint32_t[kScaledResolvePageDwordCount]);
    std::memset(scaled_resolve_pages_.get(), 0, kScaledResolvePageDwordCount * sizeof(uint32_t));
    std::memset(scaled_resolve_pages_l2_, 0, sizeof(scaled_resolve_pages_l2_));
    scaled_resolve_global_watch_handle_ =
        shared_memory.RegisterGlobalWatch(ScaledResolveGlobalWatchCallbackThunk, this);
  }
}

TextureCache::~TextureCache() {
  DestroyAllTextures(true);

  if (scaled_resolve_global_watch_handle_) {
    shared_memory().UnregisterGlobalWatch(scaled_resolve_global_watch_handle_);
  }
}

bool TextureCache::GetConfigDrawResolutionScale(uint32_t& x_out, uint32_t& y_out) {
  uint32_t shared_scale = uint32_t(std::max(INT32_C(1), REXCVAR_GET(resolution_scale)));
  bool use_shared_scale = rex::cvar::HasNonDefaultValue("resolution_scale");
  uint32_t config_x = use_shared_scale && !rex::cvar::HasNonDefaultValue("draw_resolution_scale_x")
                          ? shared_scale
                          : uint32_t(std::max(INT32_C(1), REXCVAR_GET(draw_resolution_scale_x)));
  uint32_t config_y = use_shared_scale && !rex::cvar::HasNonDefaultValue("draw_resolution_scale_y")
                          ? shared_scale
                          : uint32_t(std::max(INT32_C(1), REXCVAR_GET(draw_resolution_scale_y)));
  uint32_t clamped_x = std::min(kMaxDrawResolutionScaleAlongAxis, config_x);
  uint32_t clamped_y = std::min(kMaxDrawResolutionScaleAlongAxis, config_y);
  x_out = clamped_x;
  y_out = clamped_y;
  return clamped_x == config_x && clamped_y == config_y;
}

void TextureCache::ClearCache() {
  DestroyAllTextures();
}

void TextureCache::CompletedSubmissionUpdated(uint64_t completed_submission_index) {
  // If memory usage is too high, destroy unused textures.
  uint64_t current_time = rex::chrono::Clock::QueryHostUptimeMillis();
  // texture_cache_memory_limit_render_to_texture is assumed to be included in
  // texture_cache_memory_limit_soft and texture_cache_memory_limit_hard, at 1x,
  // so subtracting 1 from the scale.
  uint32_t limit_scaled_resolve_add_mb =
      REXCVAR_GET(texture_cache_memory_limit_render_to_texture) *
      (draw_resolution_scale_x() * draw_resolution_scale_y() - 1);
  uint32_t limit_soft_mb =
      REXCVAR_GET(texture_cache_memory_limit_soft) + limit_scaled_resolve_add_mb;
  uint32_t limit_hard_mb =
      REXCVAR_GET(texture_cache_memory_limit_hard) + limit_scaled_resolve_add_mb;
  uint32_t limit_soft_lifetime = REXCVAR_GET(texture_cache_memory_limit_soft_lifetime) * 1000;
  bool destroyed_any = false;
  while (texture_used_first_ != nullptr) {
    uint64_t total_host_memory_usage_mb =
        (textures_total_host_memory_usage_ + ((UINT32_C(1) << 20) - 1)) >> 20;
    bool limit_hard_exceeded = total_host_memory_usage_mb > limit_hard_mb;
    if (total_host_memory_usage_mb <= limit_soft_mb && !limit_hard_exceeded) {
      break;
    }
    Texture* texture = texture_used_first_;
    if (texture->last_usage_submission_index() > completed_submission_index) {
      break;
    }
    // wwe2k14 patch (Session 49): a texture holding resolve output guest memory does not have (skipped write-back).
    if (texture->native_pins()) {
      break;
    }
    if (!limit_hard_exceeded && (texture->last_usage_time() + limit_soft_lifetime) > current_time) {
      break;
    }
    if (!destroyed_any) {
      destroyed_any = true;
      // The texture being destroyed might have been bound in the previous
      // submissions, and nothing has overwritten the binding yet, so completion
      // of the submission where the texture was last actually used on the GPU
      // doesn't imply that it's not bound currently. Reset bindings if
      // any texture has been destroyed.
      ResetTextureBindings();
    }
    // Remove the texture from the map and destroy it via its unique_ptr.
    auto found_texture_it = textures_.find(texture->key());
    assert_true(found_texture_it != textures_.end());
    if (found_texture_it != textures_.end()) {
      assert_true(found_texture_it->second.get() == texture);
      textures_.erase(found_texture_it);
      // `texture` is invalid now.
    }
  }
  if (destroyed_any) {
    COUNT_profile_set("gpu/texture_cache/textures", textures_.size());
  }
}

void TextureCache::BeginSubmission(uint64_t new_submission_index) {
  assert_true(new_submission_index > current_submission_index_);
  current_submission_index_ = new_submission_index;
  current_submission_time_ = rex::chrono::Clock::QueryHostUptimeMillis();
}

void TextureCache::BeginFrame() {
  // In case there was a failure to create something in the previous frame, make
  // sure bindings are reset so a new attempt will surely be made if the texture
  // is requested again.
  ResetTextureBindings();

  ReportRequestTexturesSkipStats();  // wwe2k14 patch (Session 7, TEXREQ)
}

// wwe2k14 patch (Session 7, TEXREQ): gpu_request_textures_skip_unchanged statistics and _verify results.

void TextureCache::ReportRequestTexturesSkipStats() {
  RequestTexturesSkipStats& stats = request_textures_skip_stats_;
  if (++stats.frames < kRequestTexturesSkipSummaryFrames) {
    return;
  }
  if (stats.calls) {
    auto percentage = [](uint64_t part, uint64_t whole) {
      return whole ? 100.0 * double(part) / double(whole) : 0.0;
    };
    uint64_t bindings = stats.bindings_updated + stats.bindings_kept;
    uint64_t backend_parts = stats.backend_parts_run + stats.backend_parts_skipped;
    REXGPU_INFO(
        "[s7][TEXREQ] RequestTextures (skip {}, verify {}), last {} frames: {} calls; outdated-flag "
        "exchanges skipped {} ({:.1f}%); early returns {} ({:.1f}%); out-of-sync bindings {}, kept {} "
        "({:.1f}%); backend parts {}, skipped {} ({:.1f}%); verify: {} checks, {} mismatches, {} flag races; "
        "mismatches since start: {}",
        REXCVAR_GET(gpu_request_textures_skip_unchanged),
        REXCVAR_GET(gpu_request_textures_skip_unchanged_verify), stats.frames, stats.calls,
        stats.flag_exchanges_skipped, percentage(stats.flag_exchanges_skipped, stats.calls),
        stats.early_returns, percentage(stats.early_returns, stats.calls), bindings, stats.bindings_kept,
        percentage(stats.bindings_kept, bindings), backend_parts, stats.backend_parts_skipped,
        percentage(stats.backend_parts_skipped, backend_parts), stats.verify_checks,
        stats.verify_mismatches, stats.verify_flag_races, stats.verify_mismatches_total);
  }
  uint64_t verify_mismatches_total = stats.verify_mismatches_total;
  uint32_t verify_mismatches_logged = stats.verify_mismatches_logged;
  stats = RequestTexturesSkipStats();
  stats.verify_mismatches_total = verify_mismatches_total;
  stats.verify_mismatches_logged = verify_mismatches_logged;
}

void TextureCache::VerifyRequestTexturesSkip(bool mismatch, const char* part,
                                             uint32_t fetch_constant_index, const char* detail) {
  RequestTexturesSkipStats& stats = request_textures_skip_stats_;
  ++stats.verify_checks;
  if (!mismatch) {
    return;
  }
  ++stats.verify_mismatches;
  ++stats.verify_mismatches_total;
  if (stats.verify_mismatches_logged < kRequestTexturesSkipMismatchesLogged) {
    ++stats.verify_mismatches_logged;
    REXGPU_WARN(
        "[s7][TEXREQ] verify mismatch #{}: {} (fetch constant {}): the full path {} (submission {})",
        stats.verify_mismatches_total, part, fetch_constant_index, detail, current_submission_index_);
  }
}

void TextureCache::MarkRangeAsResolved(uint32_t start_unscaled, uint32_t length_unscaled) {
  if (length_unscaled == 0) {
    return;
  }
  start_unscaled &= 0x1FFFFFFF;
  length_unscaled = std::min(length_unscaled, 0x20000000 - start_unscaled);

  MarkRangeScaledResolved(start_unscaled, length_unscaled, false);

  // wwe2k14 patch (Session 43, S0b-lite): pages resolves wrote, at resolution scale 1 (textures created over them
  // may load straight into their image: vulkan_texture_load_direct_unscaled).
  if (!IsDrawResolutionScaled() && REXCVAR_GET(vulkan_texture_load_direct_unscaled)) {
    for (uint32_t page = start_unscaled >> 12, last = (start_unscaled + length_unscaled - 1) >> 12; page <= last;
         ++page) {
      resolved_pages_unscaled_[page >> 5] |= UINT32_C(1) << (page & 31);
    }
  }

  // Invalidate textures. Toggling individual textures between scaled and
  // unscaled also relies on invalidation through shared memory.
  shared_memory().RangeWrittenByGpu(start_unscaled, length_unscaled);
}

bool TextureCache::MarkRangeScaledResolved(uint32_t start_unscaled, uint32_t length_unscaled,
                                           bool invalidate_if_new) {
  // (Split out of MarkRangeAsResolved, Session 80: the left-out copies of owned steps mark their pages too.)
  if (length_unscaled == 0 || !IsDrawResolutionScaled()) {
    return false;
  }
  start_unscaled &= 0x1FFFFFFF;
  length_unscaled = std::min(length_unscaled, 0x20000000 - start_unscaled);
  uint32_t page_first = start_unscaled >> 12;
  uint32_t page_last = (start_unscaled + length_unscaled - 1) >> 12;
  uint32_t block_first = page_first >> 5;
  uint32_t block_last = page_last >> 5;
  bool added = false;
  {
    auto global_lock = global_critical_region_.Acquire();
    for (uint32_t i = block_first; i <= block_last; ++i) {
      uint32_t add_bits = UINT32_MAX;
      if (i == block_first) {
        add_bits &= ~((UINT32_C(1) << (page_first & 31)) - 1);
      }
      if (i == block_last && (page_last & 31) != 31) {
        add_bits &= (UINT32_C(1) << ((page_last & 31) + 1)) - 1;
      }
      added |= (scaled_resolve_pages_[i] & add_bits) != add_bits;
      scaled_resolve_pages_[i] |= add_bits;
      scaled_resolve_pages_l2_[i >> 6] |= UINT64_C(1) << (i & 63);
    }
  }
  // (As MarkRangeAsResolved: the textures over a newly scaled range invalidated - outside the lock.)
  if (added) d3d_tex::Table::Get().marks_version.fetch_add(1, std::memory_order_relaxed);  // wwe2k14 patch (Session 87, stage 4)
  if (added && invalidate_if_new) shared_memory().RangeWrittenByGpu(start_unscaled, length_unscaled);
  return added;
}

uint32_t TextureCache::GuestToHostSwizzle(uint32_t guest_swizzle, uint32_t host_format_swizzle) {
  uint32_t host_swizzle = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t guest_swizzle_component = (guest_swizzle >> (3 * i)) & 0b111;
    uint32_t host_swizzle_component;
    if (guest_swizzle_component >= xenos::XE_GPU_TEXTURE_SWIZZLE_0) {
      // Get rid of 6 and 7 values (to prevent host GPU errors if the game has
      // something broken) the simple way - by changing them to 4 (0) and 5 (1).
      host_swizzle_component = guest_swizzle_component & 0b101;
    } else {
      host_swizzle_component = (host_format_swizzle >> (3 * guest_swizzle_component)) & 0b111;
    }
    host_swizzle |= host_swizzle_component << (3 * i);
  }
  return host_swizzle;
}

bool TextureCache::PrepareTextureLoad(Texture& texture, PendingTextureLoad& pending_load_out,
                                      PendingSharedMemoryRange* pending_ranges_out,
                                      size_t& pending_range_count_out) {
  uint32_t outdated_mask = texture.outdated_mask();
  if (!outdated_mask) {
    return false;
  }

  bool base_outdated = false;
  bool mips_outdated = false;
  {
    auto global_lock = global_critical_region_.Acquire();
    if (outdated_mask & Texture::kOutdatedBitBase) {
      base_outdated = texture.base_outdated(global_lock);
    }
    if (outdated_mask & Texture::kOutdatedBitMips) {
      mips_outdated = texture.mips_outdated(global_lock);
    }
  }
  if (!base_outdated && !mips_outdated) {
    return false;
  }

  pending_load_out.texture = &texture;
  pending_load_out.load_base = base_outdated;
  pending_load_out.load_mips = mips_outdated;
  pending_range_count_out = 0;

  TextureKey texture_key = texture.key();
  if (base_outdated) {
    PendingSharedMemoryRange pending_range;
    pending_range.start = texture_key.base_page << 12;
    pending_range.length =
        static_cast<uint32_t>(rex::align(texture.GetGuestBaseSize(), UINT32_C(16)));
    pending_ranges_out[pending_range_count_out++] = pending_range;
  }
  if (mips_outdated) {
    PendingSharedMemoryRange pending_range;
    pending_range.start = texture_key.mip_page << 12;
    pending_range.length =
        static_cast<uint32_t>(rex::align(texture.GetGuestMipsSize(), UINT32_C(16)));
    pending_ranges_out[pending_range_count_out++] = pending_range;
  }

  return true;
}

bool TextureCache::CommitPreparedTextureLoad(const PendingTextureLoad& pending_load) {
  if (!pending_load.texture || (!pending_load.load_base && !pending_load.load_mips)) {
    return true;
  }

  Texture& texture = *pending_load.texture;
  TextureKey texture_key = texture.key();
  if (texture_key.scaled_resolve) {
    // Make sure all the scaled resolve memory is resident and accessible from
    // the shader, including any possible padding that hasn't yet been touched
    // by an actual resolve, but is still included in the texture size, so the
    // GPU won't be trying to access unmapped memory.
    if (pending_load.load_base && !EnsureScaledResolveMemoryCommitted(
                                      texture_key.base_page << 12, texture.GetGuestBaseSize(), 4)) {
      return false;
    }
    if (pending_load.load_mips && !EnsureScaledResolveMemoryCommitted(
                                      texture_key.mip_page << 12, texture.GetGuestMipsSize(), 4)) {
      return false;
    }
  }

  // wwe2k14 patch (Session 56): records still to be written into the scaled resolve memory go in first.
  if (texture_key.scaled_resolve) {
    if (pending_load.load_base) BeforeScaledResolveRead(texture_key.base_page << 12, texture.GetGuestBaseSize());
    if (pending_load.load_mips) BeforeScaledResolveRead(texture_key.mip_page << 12, texture.GetGuestMipsSize());
  }
  const uint64_t stat_impl_t = g_tex_stat_on ? TexStatTicks() : 0;
  if (!LoadTextureDataFromResidentMemoryImpl(texture, pending_load.load_base,
                                             pending_load.load_mips)) {
    return false;
  }
  const uint64_t stat_watch_t = g_tex_stat_on ? TexStatTicks() : 0;
  if (g_tex_stat_on) g_tex_stat[kTsLoadImplTicks] += stat_watch_t - stat_impl_t;

  // Mark the ranges as uploaded and watch them. This is needed for scaled
  // resolves as well to detect when the CPU wants to reuse the memory for a
  // regular texture or a vertex buffer, and thus the scaled resolve version is
  // not up to date anymore.
  texture.MakeUpToDateAndWatch(global_critical_region_.Acquire());
  if (g_tex_stat_on) g_tex_stat[kTsWatchTicks] += TexStatTicks() - stat_watch_t;
  texture.LogAction("Loaded");

  return true;
}

// wwe2k14 patch (Session 58, a diagnostic): the textures of the draws that bind a watched texture. For the next
// gpu_texture_watch_draws draws that bind a texture whose base or mip address is in gpu_texture_watch (hex guest
// addresses, comma-separated), one [s58][texwatch] line per texture the draw uses: fetch index, base / mip address,
// format, size, swizzle, endianness, tiling, signs, mip levels and dimension. Then it stops until the list changes.
REXCVAR_DEFINE_STRING(gpu_texture_watch, "", "GPU",
                      "Session 58 (diagnostic): hex guest addresses (comma-separated); the draws that bind a texture "
                      "with one of them as base or mip address log all their textures ([s58][texwatch]); * = every "
                      "draw")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_texture_watch_draws, 60, "GPU",
                     "Session 58: how many draws gpu_texture_watch logs after its list is set")
    .range(1, 100000)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

namespace {
void LogWatchedTextures(const RegisterFile& regs, uint32_t used_texture_mask) {
  static std::string list_seen;
  static std::vector<uint32_t> watched;
  static int32_t remaining = 0;
  static uint32_t draw_index = 0;
  const std::string list = REXCVAR_GET(gpu_texture_watch);
  if (list != list_seen) {
    list_seen = list;
    watched.clear();
    for (size_t at = 0; at < list.size();) {
      size_t end = list.find(',', at);
      if (end == std::string::npos) end = list.size();
      const std::string item = list.substr(at, end - at);
      if (!item.empty()) watched.push_back(uint32_t(std::strtoul(item.c_str(), nullptr, 16)));
      at = end + 1;
    }
    remaining = REXCVAR_GET(gpu_texture_watch_draws);
    draw_index = 0;
  }
  const bool all = list == "*";  // every draw
  if (remaining <= 0 || (watched.empty() && !all)) return;
  ++draw_index;
  bool hit = all;
  uint32_t mask = used_texture_mask, index = 0;
  while (!hit && bit_inline::bit_scan_forward(mask, &index)) {
    mask &= ~(UINT32_C(1) << index);
    const xenos::xe_gpu_texture_fetch_t fetch = regs.GetTextureFetch(index);
    for (uint32_t w : watched) {
      hit |= (uint32_t(fetch.base_address) << 12) == w || (uint32_t(fetch.mip_address) << 12) == w;
    }
  }
  if (!hit) return;
  --remaining;
  mask = used_texture_mask;
  while (bit_inline::bit_scan_forward(mask, &index)) {
    mask &= ~(UINT32_C(1) << index);
    const xenos::xe_gpu_texture_fetch_t f = regs.GetTextureFetch(index);
    REXGPU_INFO("[s58][texwatch] draw {} fetch {}: type {} base 0x{:08X} mip 0x{:08X} format {} {}x{} dim {} "
                "swizzle 0x{:03X} endian {} tiled {} pitch {} signs {}{}{}{} num_format {} mips {}-{} packed {} "
                "exp_adjust {}",
                draw_index, index, uint32_t(f.type), uint32_t(f.base_address) << 12, uint32_t(f.mip_address) << 12,
                uint32_t(f.format), uint32_t(f.size_2d.width) + 1, uint32_t(f.size_2d.height) + 1,
                uint32_t(f.dimension), uint32_t(f.swizzle), uint32_t(f.endianness), uint32_t(f.tiled),
                uint32_t(f.pitch) << 5, uint32_t(f.sign_x), uint32_t(f.sign_y), uint32_t(f.sign_z),
                uint32_t(f.sign_w), uint32_t(f.num_format), uint32_t(f.mip_min_level), uint32_t(f.mip_max_level),
                uint32_t(f.packed_mips), int32_t(f.exp_adjust));
  }
}
}  // namespace

// wwe2k14 patch (Session 95, r6tex, gpu_tex_bind_memo): the command thread's texture cost, measured (Bryan's entrance, PC, gpu_d3d_wall_stats): a slot whose fetch words moved costs about 3000 ticks
// (two table locks and hash lookups for the object, the key, the swizzle, the texture map), and half the draws move a slot: the same few hundred textures, in the same order, every frame. All of that
// is a function of the six words, the slot's object and what can move the object or the texture map. So:
// bit 1: a direct-mapped memo keyed by those: a slot that moved to words seen before takes the key, signs, swizzle and texture from it. What the words do not decide stays per draw: the sync bits, the
//        change masks, the loads (a texture is still checked for being outdated), the object census. Valid while no object was defined or destroyed (d3d_tex::ObjectsGeneration), no texture was destroyed
//        (g_bind_memo_destroyed), and for a tiled texture at a scaled resolution under the same scaled-resolve marks (as the object's remembered image: find_texture's comment).
// bit 2: VulkanTextureCache::RequestTextures: a slot whose page no owned page can hold skips the view maths (NativeOwnedPageFree).
// bit 4: NativeRenderer::OwnedView: a page's answer from a memo checked against the page itself, every read's side effect kept.
// gpu_tex_bind_memo_verify computes everything the full way as well, uses the full answer and counts differences. Command thread only; nothing here is part of any picture.
namespace {
struct BindMemo {
  uint32_t words[6] = {};
  uint32_t id = 0;
  uint32_t host_swizzle = 0;
  uint32_t kind = 0;  // (the object's, for the census)
  bool valid = false;
  uint8_t signs = 0;
  alignas(8) uint8_t key[16] = {};  // a TextureKey (memcpy'd: the type is protected; d3d_tex::Entry does the same)
  uint64_t objects = 0;
  void* texture = nullptr;  // the image found for the key, while texture_gen and (a tiled texture at a scaled resolution) texture_marks hold
  uint64_t texture_gen = 0, texture_marks = 0;
};
constexpr uint32_t kBindMemoSize = 4096;
BindMemo g_bind_memo[kBindMemoSize];
uint64_t g_bind_memo_destroyed = 1;  // (textures destroyed: every Texture destructor; a pointer in the memo is valid only under the count it was stored with)
inline uint32_t BindMemoSlot(const uint32_t* w, uint32_t id) {
  uint32_t h = w[1] * 0x9E3779B1u;  // (word 1: the base address; word 2: the size; word 4: the view bits)
  h ^= (w[2] + id) * 0x85EBCA77u;
  h ^= (w[4] ^ w[0]) * 0xC2B2AE3Du;
  h ^= (w[3] ^ w[5]) * 0x27D4EB2Fu;
  return (h ^ (h >> 15)) & (kBindMemoSize - 1);
}
}  // namespace
uint64_t g_tex_bind_mismatches = 0;
namespace {
struct TexLoadRow {
  uint64_t loads = 0, ticks = 0, base = 0, mips = 0;
  std::string name;
};
std::unordered_map<uint64_t, TexLoadRow> g_tex_load_census;
}  // namespace
void TexLoadCensusLog() {
  if (g_tex_load_census.empty()) return;
  std::vector<const TexLoadRow*> rows;
  uint64_t total = 0, loads = 0;
  for (const auto& kv : g_tex_load_census) {
    rows.push_back(&kv.second);
    total += kv.second.ticks;
    loads += kv.second.loads;
  }
  std::sort(rows.begin(), rows.end(), [](const TexLoadRow* a, const TexLoadRow* b) { return a->ticks > b->ticks; });
  REXGPU_INFO("[tex][loads] {} textures, {} loads, {} Mticks on the sampled frames", rows.size(), loads, total / 1000000);
  for (size_t i = 0; i < rows.size() && i < 14; ++i) {
    REXGPU_INFO("[tex][loads] {}: {} loads (base {} mips {}) {} kticks", rows[i]->name, rows[i]->loads, rows[i]->base, rows[i]->mips, rows[i]->ticks / 1000);
  }
  g_tex_load_census.clear();
}

// wwe2k14 patch (Session 95, r6tex): the split of RequestTextures on the typed draw's sampled frames (1 in 8; tick reads only then). Log only: the [d3d][typed] line's neighbour.
uint64_t g_tex_stat[kTexStatCount] = {};
bool g_tex_stat_on = false;

void TextureCache::RequestTextures(uint32_t used_texture_mask) {
  // (M4f-C: the register source is taken only on the branch that reads it - detector 2)
  if (!REXCVAR_GET(gpu_texture_watch).empty()) LogWatchedTextures(register_file(), used_texture_mask);  // Session 58

  // wwe2k14 patch (Session 87, D3D layer stage 4, textures in our own form; include/rex/graphics/d3d_tex.h): a slot written with a
  // Bind id is bound by its OBJECT's identity key (the object's words, Define'd by the game's registry, plus the slot's own view
  // word), not by a key derived from the fetch constant; the object remembers its host image. gpu_d3d_textures 1 = count only.
  const bool d3d_tex_on = d3d_tex::g_active.load(std::memory_order_relaxed) && d3d_tex::Enabled();
  const bool d3d_tex_ids = d3d_tex_on && d3d_tex::UseIds();
  if (d3d_tex_on && d3d_tex::Table::Get().changed_any.load(std::memory_order_relaxed)) D3dTexProcessChanged();
  d3d_tex::Table& d3d_table = d3d_tex::Table::Get();
  static_assert(sizeof(TextureKey) <= 16, "d3d_tex::Entry stores a TextureKey in 16 bytes");
  // The object's key for a slot: false = fall back to the fetch constant (counted by reason), key_out / signs_out then hold the
  // fetch constant's. entry_id is set when the key is the object's.
  auto derive_key = [&](uint32_t index, const xenos::xe_gpu_texture_fetch_t& fetch, TextureKey& key_out, uint8_t& signs_out,
                        uint32_t& entry_id) -> bool {
    entry_id = 0;
    auto fall_back = [&](d3d_tex::Fallback why) {
      if (fetch.type == xenos::FetchConstantType::kTexture || why == d3d_tex::kFbNotTexture) {
        d3d_table.census.fallback[why].fetch_add(1, std::memory_order_relaxed);
        d3d_table.census.fallback_slot[index & 31].fetch_add(1, std::memory_order_relaxed);
        if (why == d3d_tex::kFbNoId) d3d_table.census.no_id_by_writer[d3d_table.LastWriter(index)].fetch_add(1, std::memory_order_relaxed);
        if (d3d_table.census.fallback_logged.fetch_add(1, std::memory_order_relaxed) < 60) {  // name them: the first few, with the fetch constant
          REXGPU_INFO("[d3d][tex] fetch fallback ({}) slot {} (last written by {}, with a Bind: {}): bound id {}, fetch words {:08X} {:08X} {:08X} {:08X} {:08X} {:08X}",
                      d3d_tex::kFallbackNames[why], index, d3d_tex::kWriterNames[d3d_table.LastWriter(index)], d3d_table.LastHadBind(index),
                      d3d_table.BoundId(index), fetch.dword_0, fetch.dword_1, fetch.dword_2,
                      fetch.dword_3, fetch.dword_4, fetch.dword_5);
        }
      }
      BindingInfoFromFetchConstant(fetch, key_out, &signs_out);
      return false;
    };
    uint32_t fetch_words[6];
    static_assert(sizeof(fetch) == sizeof(fetch_words));
    std::memcpy(fetch_words, &fetch, sizeof(fetch_words));
    uint32_t id = d3d_table.BoundId(index);
    if (!id) {
      // A slot written without a Bind (a replay or a re-write by a writer that sent none): the same six dwords as the slot's last by-id
      // bind are the same texture state, so the id is kept (still verified against the fetch constant below, and the object must still exist).
      bool never = true;
      id = d3d_table.Recall(index, fetch_words, never);
      if (id) {
        d3d_table.census.recalled.fetch_add(1, std::memory_order_relaxed);
      } else {
        (never ? d3d_table.census.no_id_never : d3d_table.census.no_id_new_content).fetch_add(1, std::memory_order_relaxed);
        return fall_back(d3d_tex::kFbNoId);
      }
    }
    d3d_tex::Entry e;
    if (!d3d_table.Find(id, e)) return fall_back(d3d_tex::kFbUnknownId);
    if (fetch.type != xenos::FetchConstantType::kTexture) return fall_back(d3d_tex::kFbNotTexture);
    TextureKey object_key;
    bool valid;
    if (e.key_known && e.key_w4 == fetch.dword_4) {
      std::memcpy(&object_key, e.key, sizeof(object_key));
      valid = e.key_valid;
    } else {
      xenos::xe_gpu_texture_fetch_t object_fetch;
      static_assert(sizeof(object_fetch) == sizeof(e.words));
      std::memcpy(&object_fetch, e.words, sizeof(e.words));
      object_fetch.dword_4 = fetch.dword_4;  // the view bits (mip range, lod) are the slot's
      BindingInfoFromFetchConstant(object_fetch, object_key, nullptr);
      valid = object_key.is_valid != 0;
      uint8_t bytes[16] = {};
      std::memcpy(bytes, &object_key, sizeof(object_key));
      d3d_table.StoreKey(id, bytes, fetch.dword_4, valid);
    }
    if (!valid) return fall_back(d3d_tex::kFbBadObject);
    const uint8_t object_signs = texture_util::SwizzleSigns(fetch);
    if (d3d_tex::VerifyKeys()) {
      TextureKey fetch_key;
      uint8_t fetch_signs;
      BindingInfoFromFetchConstant(fetch, fetch_key, &fetch_signs);
      if (fetch_key != object_key || fetch_signs != object_signs) {
        d3d_table.census.fallback[d3d_tex::kFbKeyMismatch].fetch_add(1, std::memory_order_relaxed);
        if (d3d_table.census.mismatch_logged.fetch_add(1, std::memory_order_relaxed) < 40) {
          REXGPU_WARN("[d3d][tex] key mismatch, slot {} object {} ({}): object key base {:X} mip {:X} {}x{}x{} dim {} pitch {} mips {} tiled {} "
                      "packed {} fmt {} endian {}; fetch key base {:X} mip {:X} {}x{}x{} dim {} pitch {} mips {} tiled {} packed {} fmt {} endian {}; "
                      "fetch words {:08X} {:08X} {:08X} {:08X} {:08X} {:08X}; object words {:08X} {:08X} {:08X} {:08X} {:08X} {:08X}",
                      index, id, d3d_tex::kKindNames[e.kind], uint32_t(object_key.base_page), uint32_t(object_key.mip_page),
                      object_key.GetWidth(), object_key.GetHeight(), object_key.GetDepthOrArraySize(), uint32_t(object_key.dimension),
                      uint32_t(object_key.pitch), uint32_t(object_key.mip_max_level), uint32_t(object_key.tiled),
                      uint32_t(object_key.packed_mips), uint32_t(object_key.format), uint32_t(object_key.endianness),
                      uint32_t(fetch_key.base_page), uint32_t(fetch_key.mip_page), fetch_key.GetWidth(), fetch_key.GetHeight(),
                      fetch_key.GetDepthOrArraySize(), uint32_t(fetch_key.dimension), uint32_t(fetch_key.pitch),
                      uint32_t(fetch_key.mip_max_level), uint32_t(fetch_key.tiled), uint32_t(fetch_key.packed_mips),
                      uint32_t(fetch_key.format), uint32_t(fetch_key.endianness), fetch.dword_0, fetch.dword_1, fetch.dword_2,
                      fetch.dword_3, fetch.dword_4, fetch.dword_5, e.words[0], e.words[1], e.words[2], e.words[3], e.words[4],
                      e.words[5]);
        }
        key_out = fetch_key;  // the fetch constant wins
        signs_out = fetch_signs;
        return false;
      }
    }
    key_out = object_key;
    signs_out = object_signs;
    entry_id = id;
    d3d_table.Remember(index, id, fetch_words);
    d3d_table.census.by_id.fetch_add(1, std::memory_order_relaxed);
    d3d_table.census.by_id_kind[e.kind].fetch_add(1, std::memory_order_relaxed);
    return true;
  };
  // The host image of a binding: the object's remembered one while it is valid for the same key and no texture was destroyed since.
  auto find_texture = [&](const TextureKey& key, uint32_t entry_id) -> Texture* {
    if (!entry_id) return FindOrCreateTexture(key);
    uint8_t bytes[16] = {};
    std::memcpy(bytes, &key, sizeof(key));
    d3d_tex::Entry e;
    // FindOrCreateTexture's answer is not a function of the key alone: at a scaled draw resolution a tiled texture whose range a resolve
    // writes later gets another key (scaled_resolve) and another image (check 1: 1440p 15 of 18 passes owned with a shortcut that forgot this,
    // the 720p arm and the switch-off control 17 of 17). The object's remembered image is therefore valid for such a texture only under the
    // scaled-resolve marks it was found under (d3d_tex marks_version, bumped where the marks change); linear textures (every static) never scale.
    const bool answer_can_change = IsDrawResolutionScaled() && key.tiled;
    const uint64_t marks_now = d3d_table.marks_version.load(std::memory_order_relaxed);
    const bool found = d3d_table.Find(entry_id, e) && e.texture && e.texture_epoch == d3d_table.epoch() &&
                       !std::memcmp(e.key_requested, bytes, sizeof(key));
    if (found && answer_can_change && e.texture_marks != marks_now) {
      d3d_table.census.shortcut_marks_miss.fetch_add(1, std::memory_order_relaxed);
    }
    if (found && (!answer_can_change || e.texture_marks == marks_now)) {
      Texture* remembered = static_cast<Texture*>(e.texture);
      if (d3d_tex::VerifyKeys()) {
        // Verify also covers the shortcut: the remembered image must be what the full lookup returns now; if not, the full lookup's answer is
        // used (and remembered), and the difference is counted.
        Texture* fresh = FindOrCreateTexture(key);
        if (fresh != remembered) {
          static std::atomic<uint32_t> logged{0};
          d3d_table.census.shortcut_mismatch.fetch_add(1, std::memory_order_relaxed);
          if (logged.fetch_add(1, std::memory_order_relaxed) < 12) {
            REXGPU_WARN("[d3d][tex] verify: object {} remembered image {:p}, FindOrCreateTexture(key) now returns {:p}", entry_id,
                        static_cast<const void*>(remembered), static_cast<const void*>(fresh));
          }
          if (fresh) d3d_table.StoreTexture(entry_id, fresh, bytes, d3d_table.epoch(), marks_now);
          return fresh;
        }
      }
      d3d_table.census.texture_hits.fetch_add(1, std::memory_order_relaxed);
      return remembered;
    }
    Texture* texture = FindOrCreateTexture(key);
    d3d_table.census.texture_finds.fetch_add(1, std::memory_order_relaxed);
    if (texture) d3d_table.StoreTexture(entry_id, texture, bytes, d3d_table.epoch(), marks_now);
    return texture;
  };

  // wwe2k14 patch (Session 95, r6tex): see gpu_tex_bind_memo. gpu_d3d_textures_verify (on by default) is a function of the same words: a key it turned down comes back with no object and is not stored.
  const int32_t bind_memo_bits = REXCVAR_GET(gpu_tex_bind_memo);
  const bool bind_memo_on = (bind_memo_bits & 1) != 0;
  const bool bind_memo_verify = bind_memo_on && REXCVAR_GET(gpu_tex_bind_memo_verify);
  const uint64_t bind_memo_objects = bind_memo_on ? d3d_tex::ObjectsGeneration() : 0;
  if (bind_memo_bits) {
    static bool bind_memo_logged = false;
    if (!bind_memo_logged) {
      bind_memo_logged = true;
      REXGPU_INFO("[tex] gpu_tex_bind_memo {} on (verify {}): {} entries", bind_memo_bits, REXCVAR_GET(gpu_tex_bind_memo_verify), kBindMemoSize);
    }
  }
  BindMemo* slot_memo = nullptr;  // (this slot's entry: filled below)
  bool slot_memo_hit = false;
  // find_texture through the memo: the entry holds this key.
  auto find_bound = [&](const TextureKey& key, uint32_t entry_id) -> Texture* {
    if (!slot_memo) return find_texture(key, entry_id);
    const bool answer_can_change = IsDrawResolutionScaled() && key.tiled;
    const uint64_t marks_now = d3d_table.marks_version.load(std::memory_order_relaxed);
    if (slot_memo_hit && slot_memo->texture && slot_memo->texture_gen == g_bind_memo_destroyed && (!answer_can_change || slot_memo->texture_marks == marks_now)) {
      if (bind_memo_verify) {
        Texture* fresh = find_texture(key, entry_id);
        if (fresh != slot_memo->texture) {
          if (++g_tex_bind_mismatches <= 16) REXGPU_WARN("[tex] gpu_tex_bind_memo verify: the memo's image {:p}, the full way's {:p}", slot_memo->texture, static_cast<const void*>(fresh));
          return fresh;
        }
      }
      if (entry_id) d3d_table.census.texture_hits.fetch_add(1, std::memory_order_relaxed);
      if (g_tex_stat_on) ++g_tex_stat[kTsTextureHits];
      return static_cast<Texture*>(slot_memo->texture);
    }
    Texture* texture = find_texture(key, entry_id);
    slot_memo->texture = texture;
    slot_memo->texture_gen = g_bind_memo_destroyed;
    slot_memo->texture_marks = marks_now;
    return texture;
  };

  // wwe2k14 patch (Session 7, TEXREQ): the settings are latched for this call, including the backend's part
  // of it (its override runs right after this), so they can be flipped at any time. _verify runs the full
  // path, checking what would have been skipped.
  request_textures_skip_verify_ = REXCVAR_GET(gpu_request_textures_skip_unchanged_verify);
  request_textures_skip_unchanged_ =
      !request_textures_skip_verify_ && REXCVAR_GET(gpu_request_textures_skip_unchanged);
  const bool skip_unchanged = request_textures_skip_unchanged_;
  const bool skip_verify = request_textures_skip_verify_;
  const bool skip_or_verify = skip_unchanged || skip_verify;
  if (skip_or_verify) {
    ++request_textures_skip_stats_.calls;
  }

  bool texture_became_outdated;
  if (skip_or_verify) {
    // wwe2k14 patch (Session 7, TEXREQ): a plain read before the locked exchange. Other threads (memory
    // watches) only ever set the flag, and only this function clears it, so a load reading false is the
    // exchange reading false (and writing back the false already there) at the moment of the load; a set flag
    // is still exchanged, so it is acted on in the same call as before. A flag set by another thread after
    // the read is acted on in the next call - as it was when set right after the exchange.
    bool flag_read = texture_became_outdated_.load(std::memory_order_acquire);
    if (!flag_read) {
      ++request_textures_skip_stats_.flag_exchanges_skipped;
    }
    if (skip_unchanged) {
      texture_became_outdated =
          flag_read && texture_became_outdated_.exchange(false, std::memory_order_acquire);
    } else {
      texture_became_outdated = texture_became_outdated_.exchange(false, std::memory_order_acquire);
      if (!flag_read && texture_became_outdated) {
        ++request_textures_skip_stats_.verify_flag_races;
      }
    }
  } else {
    texture_became_outdated = texture_became_outdated_.exchange(false, std::memory_order_acquire);
  }
  if (texture_became_outdated) {
    if (g_tex_stat_on) ++g_tex_stat[kTsOutdatedResets];
    // A texture has become outdated - make sure whether textures are outdated
    // is rechecked in this draw and in subsequent ones to reload the new data
    // if needed.
    ResetTextureBindings();
  }

  // wwe2k14 patch (Session 7, TEXREQ): with every used binding in sync, the rest of this function does nothing
  // (the loop below runs for no binding, so nothing is loaded and no binding changes).
  bool would_return_early = false;
  if (skip_or_verify && !(used_texture_mask & ~texture_bindings_in_sync_)) {
    ++request_textures_skip_stats_.early_returns;
    if (g_tex_stat_on) ++g_tex_stat[kTsBaseEarly];
    if (skip_unchanged) {
      COUNT_profile_set("gpu/texture_cache/request_textures_pending_load_count", 0);
      COUNT_profile_set("gpu/texture_cache/request_textures_pending_range_count", 0);
      return;
    }
    would_return_early = true;
  }
  // wwe2k14 patch (Session 7, TEXREQ): bindings kept (or, with _verify, that would have been kept) because
  // their fetch constant holds the value they were derived from, and of the re-derived bindings, the ones that
  // requested a texture load.
  uint32_t bindings_kept = 0;
  uint32_t bindings_load_requested = 0;

  // Update the texture keys and the textures.
  uint32_t bindings_changed = 0;
  // wwe2k14 patch (Session 6, CP9): valid bindings whose swizzled signs changed, possibly with
  // nothing in bindings_changed changing (for active_texture_bindings_version_).
  uint32_t bindings_swizzled_signs_changed = 0;
  std::vector<PendingTextureLoad> pending_texture_loads;
  std::vector<PendingSharedMemoryRange> pending_shared_memory_ranges;
  auto queue_pending_texture_load = [&](Texture* texture) {
    if (texture == nullptr) {
      return;
    }
    for (const PendingTextureLoad& pending_load : pending_texture_loads) {
      if (pending_load.texture == texture) {
        return;
      }
    }
    PendingTextureLoad pending_load;
    PendingSharedMemoryRange pending_ranges[2];
    size_t pending_range_count = 0;
    if (!PrepareTextureLoad(*texture, pending_load, pending_ranges, pending_range_count)) {
      return;
    }
    if (d3d_tex_on) d3d_table.NoteLoad(texture);  // wwe2k14 patch (Session 87, stage 4): a registered object's first upload / re-upload
    pending_texture_loads.push_back(pending_load);
    for (size_t i = 0; i < pending_range_count; ++i) {
      pending_shared_memory_ranges.push_back(pending_ranges[i]);
    }
  };
  uint32_t textures_remaining = used_texture_mask & ~texture_bindings_in_sync_;
  uint32_t index = 0;
  const uint64_t stat_slots_t = g_tex_stat_on ? TexStatTicks() : 0;
  while (bit_inline::bit_scan_forward(textures_remaining, &index)) {
    uint32_t index_bit = UINT32_C(1) << index;
    textures_remaining &= ~index_bit;
    TextureBinding& binding = texture_bindings_[index];
    xenos::xe_gpu_texture_fetch_t fetch = TextureFetch(index);  // (Session 92, M4f-A: the fetch source or the register source)
    // wwe2k14 patch (Session 7, TEXREQ): a valid binding whose fetch constant was only rewritten with the value
    // it was derived from. The update below is a pure function of the fetch constant (through
    // BindingInfoFromFetchConstant - for kTexture fetch constants independent of
    // gpu_allow_invalid_fetch_constants - and the host format tables, fixed after initialization) and of the
    // binding's previous state, and applying it again to its own result changes nothing: the key, host
    // swizzle and signedness come out the same, so no binding is marked changed, the textures are kept (with
    // the same key and signedness, FindOrCreateTexture isn't called again and the textures aren't reset), and
    // no load is queued. A valid binding was last written by that update (with the fetch constant value
    // stored below), because the only other writer, TextureBinding::Reset, makes it invalid. Only the in-sync
    // bit remains to be set.
    uint32_t(&fetch_dwords)[sizeof(fetch) / sizeof(uint32_t)] = texture_binding_fetch_dwords_[index];
    static_assert(sizeof(fetch_dwords) == sizeof(fetch));
    if (skip_or_verify && binding.key.is_valid &&
        fetch.type == xenos::FetchConstantType::kTexture &&
        !std::memcmp(fetch_dwords, &fetch, sizeof(fetch))) {
      ++request_textures_skip_stats_.bindings_kept;
      if (g_tex_stat_on) ++g_tex_stat[kTsSlotsKept];
      bindings_kept |= index_bit;
      if (skip_unchanged) {
        texture_bindings_in_sync_ |= index_bit;
        continue;
      }
      request_textures_verify_bindings_[index] = binding;
    } else if (skip_or_verify) {
      ++request_textures_skip_stats_.bindings_updated;
    }
    if (g_tex_stat_on) ++g_tex_stat[kTsSlotsDerived];
    std::memcpy(fetch_dwords, &fetch, sizeof(fetch));
    TextureKey old_key = binding.key;
    uint8_t old_swizzled_signs = binding.swizzled_signs;
    uint32_t object_id = 0;  // wwe2k14 patch (Session 87, stage 4): the object the key came from, 0 = the fetch constant
    // wwe2k14 patch (Session 95, r6tex, gpu_tex_bind_memo bit 1): the object is taken as derive_key takes it, with no lock; a slot with no object goes the full way (its fallback census).
    slot_memo = nullptr;
    slot_memo_hit = false;
    uint32_t memo_id = 0;
    if (bind_memo_on && fetch.type == xenos::FetchConstantType::kTexture) {
      bool memo_recalled = false, memo_usable = true;
      if (d3d_tex_ids) {
        memo_id = d3d_table.BoundId(index);
        if (!memo_id) {
          bool never = true;
          memo_id = d3d_table.Recall(index, fetch_dwords, never);
          memo_recalled = memo_id != 0;
        }
        memo_usable = memo_id != 0;
      }
      if (memo_usable) {
        slot_memo = &g_bind_memo[BindMemoSlot(fetch_dwords, memo_id)];
        slot_memo_hit = slot_memo->valid && slot_memo->id == memo_id && slot_memo->objects == bind_memo_objects && !std::memcmp(slot_memo->words, fetch_dwords, sizeof(slot_memo->words));
        if (slot_memo_hit) {
          std::memcpy(&binding.key, slot_memo->key, sizeof(binding.key));
          binding.swizzled_signs = slot_memo->signs;
          object_id = memo_id;
          if (memo_id) {  // (what derive_key leaves behind)
            if (memo_recalled) d3d_table.census.recalled.fetch_add(1, std::memory_order_relaxed);
            d3d_table.Remember(index, memo_id, fetch_dwords);
            d3d_table.census.by_id.fetch_add(1, std::memory_order_relaxed);
            d3d_table.census.by_id_kind[slot_memo->kind].fetch_add(1, std::memory_order_relaxed);
          }
          if (g_tex_stat_on) ++g_tex_stat[kTsMemoHits];
          if (bind_memo_verify) {
            TextureKey fresh_key;
            uint8_t fresh_signs = 0;
            uint32_t fresh_id = 0;
            if (d3d_tex_ids) {
              derive_key(index, fetch, fresh_key, fresh_signs, fresh_id);
            } else {
              BindingInfoFromFetchConstant(fetch, fresh_key, &fresh_signs);
            }
            if (fresh_key != binding.key || fresh_signs != binding.swizzled_signs || fresh_id != object_id) {
              if (++g_tex_bind_mismatches <= 16) REXGPU_WARN("[tex] gpu_tex_bind_memo verify: slot {} object {}: the memo's key or signs are not the full way's (object {})", index, object_id, fresh_id);
              binding.key = fresh_key;
              binding.swizzled_signs = fresh_signs;
              object_id = fresh_id;
              slot_memo = nullptr;
              slot_memo_hit = false;
            }
          }
        }
      }
    }
    if (!slot_memo_hit) {
    if (d3d_tex_ids) {
      derive_key(index, fetch, binding.key, binding.swizzled_signs, object_id);
    } else {
      BindingInfoFromFetchConstant(fetch, binding.key, &binding.swizzled_signs);
    }
    }
    texture_bindings_in_sync_ |= index_bit;
    if (!binding.key.is_valid) {
      if (old_key.is_valid) {
        bindings_changed |= index_bit;
      }
      binding.Reset();
      continue;
    }
    uint32_t old_host_swizzle = binding.host_swizzle;
    if (slot_memo_hit) {
      binding.host_swizzle = slot_memo->host_swizzle;  // (Session 95, r6tex)
    } else {
      binding.host_swizzle = GuestToHostSwizzle(fetch.swizzle, GetHostFormatSwizzle(binding.key));
      if (slot_memo) {
        // (Stored only for a key that came the way the entry says: an object's, or the fetch constant's with no objects in use. A fallback's stays out.)
        if (object_id == memo_id) {
          std::memcpy(slot_memo->words, fetch_dwords, sizeof(slot_memo->words));
          slot_memo->id = memo_id;
          slot_memo->objects = bind_memo_objects;
          slot_memo->host_swizzle = binding.host_swizzle;
          slot_memo->signs = binding.swizzled_signs;
          std::memset(slot_memo->key, 0, sizeof(slot_memo->key));
          std::memcpy(slot_memo->key, &binding.key, sizeof(binding.key));
          slot_memo->kind = 0;
          if (memo_id) {
            d3d_tex::Entry memo_entry;
            if (d3d_table.Find(memo_id, memo_entry)) slot_memo->kind = memo_entry.kind < d3d_tex::kKindCount ? memo_entry.kind : 0;
          }
          slot_memo->texture = nullptr;
          slot_memo->valid = true;
          if (g_tex_stat_on) ++g_tex_stat[kTsMemoStores];
        } else {
          slot_memo = nullptr;
        }
      }
    }

    // Check if need to load the unsigned and the signed versions of the texture
    // (if the format is emulated with different host bit representations for
    // signed and unsigned - otherwise only the unsigned one is loaded).
    bool key_changed = binding.key != old_key;
    bool any_sign_was_not_signed = texture_util::IsAnySignNotSigned(old_swizzled_signs);
    bool any_sign_was_signed = texture_util::IsAnySignSigned(old_swizzled_signs);
    bool any_sign_is_not_signed = texture_util::IsAnySignNotSigned(binding.swizzled_signs);
    bool any_sign_is_signed = texture_util::IsAnySignSigned(binding.swizzled_signs);
    if (key_changed || binding.host_swizzle != old_host_swizzle ||
        any_sign_is_not_signed != any_sign_was_not_signed ||
        any_sign_is_signed != any_sign_was_signed) {
      bindings_changed |= index_bit;
    }
    if (binding.swizzled_signs != old_swizzled_signs) {
      bindings_swizzled_signs_changed |= index_bit;  // wwe2k14 patch (Session 6, CP9)
    }
    bool load_unsigned_data = false, load_signed_data = false;
    if (IsSignedVersionSeparateForFormat(binding.key)) {
      // Can reuse previously loaded unsigned/signed versions if the key is the
      // same and the texture was previously bound as unsigned/signed
      // respectively (checking the previous values of signedness rather than
      // binding.texture != nullptr and binding.texture_signed != nullptr also
      // prevents repeated attempts to load the texture if it has failed to
      // load).
      if (any_sign_is_not_signed) {
        if (key_changed || !any_sign_was_not_signed) {
          binding.texture = find_bound(binding.key, object_id);
          load_unsigned_data = true;
        }
      } else {
        binding.texture = nullptr;
      }
      if (any_sign_is_signed) {
        if (key_changed || !any_sign_was_signed) {
          TextureKey signed_key = binding.key;
          signed_key.signed_separate = 1;
          binding.texture_signed = FindOrCreateTexture(signed_key);
          load_signed_data = true;
        }
      } else {
        binding.texture_signed = nullptr;
      }
    } else {
      // Same resource for both unsigned and signed, but descriptor formats may
      // be different.
      if (key_changed) {
        binding.texture = find_bound(binding.key, object_id);
        load_unsigned_data = true;
      }
      binding.texture_signed = nullptr;
    }
    if (g_tex_stat_on && (load_unsigned_data || load_signed_data)) ++g_tex_stat[kTsSlotsFound];
    if (load_unsigned_data) {
      queue_pending_texture_load(binding.texture);
    }
    if (load_signed_data) {
      queue_pending_texture_load(binding.texture_signed);
    }
    if (load_unsigned_data || load_signed_data) {
      bindings_load_requested |= index_bit;  // wwe2k14 patch (Session 7, TEXREQ)
    }
  }

  if (g_tex_stat_on) g_tex_stat[kTsSlotLoopTicks] += TexStatTicks() - stat_slots_t;
  // wwe2k14 patch (Session 7, TEXREQ): _verify - what the full path did where a skip would have done nothing.
  if (skip_verify) {
    if (would_return_early) {
      bool changed = (bindings_changed | bindings_swizzled_signs_changed) != 0 ||
                     !pending_texture_loads.empty();
      VerifyRequestTexturesSkip(changed, "early return", 0,
                                "changed a binding or queued a load with no used binding out of sync");
    }
    uint32_t verify_remaining = bindings_kept;
    uint32_t verify_index;
    while (bit_inline::bit_scan_forward(verify_remaining, &verify_index)) {
      uint32_t verify_index_bit = UINT32_C(1) << verify_index;
      verify_remaining &= ~verify_index_bit;
      const TextureBinding& kept = request_textures_verify_bindings_[verify_index];
      const TextureBinding& updated = texture_bindings_[verify_index];
      const char* detail = nullptr;
      if (updated.key != kept.key) {
        detail = "changed the texture key";
      } else if (updated.host_swizzle != kept.host_swizzle) {
        detail = "changed the host swizzle";
      } else if (updated.swizzled_signs != kept.swizzled_signs) {
        detail = "changed the swizzled signs";
      } else if (updated.texture != kept.texture || updated.texture_signed != kept.texture_signed) {
        detail = "changed the bound textures";
      } else if ((bindings_changed | bindings_swizzled_signs_changed) & verify_index_bit) {
        detail = "marked the binding as changed";
      } else if (bindings_load_requested & verify_index_bit) {
        detail = "requested a texture load";
      }
      VerifyRequestTexturesSkip(detail != nullptr, "binding kept on a same-value fetch constant write",
                                verify_index, detail ? detail : "");
    }
  }

  COUNT_profile_set("gpu/texture_cache/request_textures_pending_load_count",
                    uint32_t(pending_texture_loads.size()));
  COUNT_profile_set("gpu/texture_cache/request_textures_pending_range_count",
                    uint32_t(pending_shared_memory_ranges.size()));

  const uint64_t stat_load_t = g_tex_stat_on ? TexStatTicks() : 0;
  if (g_tex_stat_on) g_tex_stat[kTsLoads] += pending_texture_loads.size();
  bool batched_shared_memory_request_succeeded = true;
  std::vector<std::pair<uint32_t, uint32_t>> pending_shared_memory_range_pairs;
  if (!pending_shared_memory_ranges.empty()) {
    pending_shared_memory_range_pairs.reserve(pending_shared_memory_ranges.size());
    for (const PendingSharedMemoryRange& pending_range : pending_shared_memory_ranges) {
      pending_shared_memory_range_pairs.emplace_back(pending_range.start, pending_range.length);
    }
    SharedMemory::RamReaderScope ram_reader(shared_memory(), SharedMemory::RamReader::kTexture);  // wwe2k14 (S49)
    const uint64_t stat_range_t = g_tex_stat_on ? TexStatTicks() : 0;
    batched_shared_memory_request_succeeded = shared_memory().RequestRanges(
        pending_shared_memory_range_pairs.data(), pending_shared_memory_range_pairs.size());
    if (g_tex_stat_on) g_tex_stat[kTsRangeTicks] += TexStatTicks() - stat_range_t;
  }

  if (batched_shared_memory_request_succeeded) {
    for (const PendingTextureLoad& pending_load : pending_texture_loads) {
      const uint64_t stat_commit_t = g_tex_stat_on ? TexStatTicks() : 0;
      CommitPreparedTextureLoad(pending_load);
      if (g_tex_stat_on && REXCVAR_GET(gpu_tex_load_census) && pending_load.texture) {
        const TextureKey& k = pending_load.texture->key();
        TexLoadRow& row = g_tex_load_census[uint64_t(k.base_page) | (uint64_t(k.GetWidth()) << 20) | (uint64_t(k.GetHeight()) << 34) | (uint64_t(k.format) << 48) | (uint64_t(k.tiled) << 56)];
        if (row.name.empty()) {
          row.name = fmt::format("page 0x{:05X} {}x{} format {} mips {} tiled {} scaled {}", uint32_t(k.base_page), k.GetWidth(), k.GetHeight(), uint32_t(k.format), uint32_t(k.mip_max_level), uint32_t(k.tiled), uint32_t(k.scaled_resolve));
        }
        ++row.loads;
        row.base += pending_load.load_base;
        row.mips += pending_load.load_mips;
        row.ticks += TexStatTicks() - stat_commit_t;
      }
    }
  } else {
    for (const PendingTextureLoad& pending_load : pending_texture_loads) {
      if (pending_load.texture != nullptr) {
        LoadTextureData(*pending_load.texture);
      }
    }
  }
  if (g_tex_stat_on) g_tex_stat[kTsLoadTicks] += TexStatTicks() - stat_load_t;
  if (d3d_tex_on && d3d_table.drop_any.load(std::memory_order_relaxed)) D3dTexDropWatches();  // wwe2k14 patch (Session 87, stage 4)
  if (bindings_changed | bindings_swizzled_signs_changed) {
    ++active_texture_bindings_version_;  // wwe2k14 patch (Session 6, CP9)
  }
  if (bindings_changed) {
    const uint64_t stat_t = g_tex_stat_on ? TexStatTicks() : 0;
    UpdateTextureBindingsImpl(bindings_changed);
    if (g_tex_stat_on) {
      g_tex_stat[kTsImplTicks] += TexStatTicks() - stat_t;
      ++g_tex_stat[kTsImplCalls];
    }
  }
}

// wwe2k14 patch (Session 87, D3D layer stage 4): Changed ops (the engine's unlock flush of a registered object): the object's host
// image is outdated and reloads at its next bind, as after a watch fire.
void TextureCache::D3dTexProcessChanged() {
  d3d_tex::Table& table = d3d_tex::Table::Get();
  if (!table.changed_any.exchange(false, std::memory_order_relaxed)) return;
  std::vector<std::pair<const void*, uint32_t>> list;
  table.TakeChanged(list);
  if (list.empty()) return;
  std::vector<const void*> newly_flushed;  // (Session 95, r6tex: see below)
  {
    auto global_lock = global_critical_region_.Acquire();
    for (const auto& item : list) {
      Texture* const texture = static_cast<Texture*>(const_cast<void*>(item.first));
      texture->MarkOutdatedByEngine(global_lock);
      // wwe2k14 patch (Session 95, r6tex, gpu_tex_planes_unwatched): a movie plane: the CPU writes it whole and the plane flush says when it is done, so this Changed is the whole truth about it. Its
      // watches stay gone and its pages go unwatched (shared_memory.cpp: what that means and how it ends). Review: engine-dynamic alone does not say plane (the unlock flush sends Changed for any
      // registered object, the plane flush's match has a page of slack at each end, and only a watch sees a resolve into a texture): the planes are the linear k_8 ones without mips (gpu_tex_load_census:
      // format 2, tiled 0, mips 0 on every plane of the census runs); any other keeps its watch. Review: the texture outlives its object, and the same key can come back on reused memory with no flush
      // to announce it: so the object's end has to reach it. A texture listed as watch-dropped is outdated by RetireTexture, which arrives here as kind unknown: its next load takes a watch again.
      const TextureKey& key = texture->key();
      if (item.second == d3d_tex::kEngineDynamic && REXCVAR_GET(gpu_tex_planes_unwatched) && key.format == xenos::TextureFormat::k_8 && !key.tiled && !key.mip_max_level) {
        if (texture->SetEngineFlushed()) newly_flushed.push_back(item.first);
        if (texture->GetGuestBaseSize()) shared_memory().DeclareEnginePages(key.base_page << 12, texture->GetGuestBaseSize());
      } else {
        texture->ClearEngineFlushed();
      }
    }
  }
  for (const void* flushed : newly_flushed) table.MarkWatchDropped(flushed);  // (the table's calls stay outside the global critical region)
  texture_became_outdated_.store(true, std::memory_order_release);
  table.census.changed_marked.fetch_add(list.size(), std::memory_order_relaxed);
}

// wwe2k14 patch (Session 87, D3D layer stage 4): the memory watches of registered statics go once their first upload is done.
void TextureCache::D3dTexDropWatches() {
  d3d_tex::Table& table = d3d_tex::Table::Get();
  std::vector<std::pair<const void*, uint32_t>> list;
  table.TakeDrops(list);
  if (list.empty()) return;
  std::vector<std::pair<const void*, uint32_t>> again;
  {
    auto global_lock = global_critical_region_.Acquire();
    for (const auto& item : list) {
      Texture* texture = static_cast<Texture*>(const_cast<void*>(item.first));
      if (texture->DropWatches(global_lock)) {
        table.MarkWatchDropped(item.first);
      } else {
        table.census.drop_failed.fetch_add(1, std::memory_order_relaxed);  // (not retried: the watch stays, the image stays safe)
      }
    }
  }
  for (const auto& item : again) table.RequeueDrop(item.first, item.second);
}

const char* TextureCache::TextureKey::GetLogDimensionName(xenos::DataDimension dimension) {
  switch (dimension) {
    case xenos::DataDimension::k1D:
      return "1D";
    case xenos::DataDimension::k2DOrStacked:
      return "2D";
    case xenos::DataDimension::k3D:
      return "3D";
    case xenos::DataDimension::kCube:
      return "cube";
    default:
      assert_unhandled_case(dimension);
      return "unknown";
  }
}

void TextureCache::TextureKey::LogAction(const char* action) const {
  REXGPU_TRACE(
      "{} {} {}{}x{}x{} {} {} texture with {} {}packed mip level{}, "
      "base at 0x{:08X} (pitch {}), mips at 0x{:08X}",
      action, tiled ? "tiled" : "linear", scaled_resolve ? "scaled " : "", GetWidth(), GetHeight(),
      GetDepthOrArraySize(), GetLogDimensionName(), FormatInfo::Get(format)->name,
      mip_max_level + 1, packed_mips ? "" : "un", mip_max_level != 0 ? "s" : "", base_page << 12,
      pitch << 5, mip_page << 12);
}

void TextureCache::Texture::LogAction(const char* action) const {
  REXGPU_TRACE(
      "{} {} {}{}x{}x{} {} {} texture with {} {}packed mip level{}, "
      "base at 0x{:08X} (pitch {}, size 0x{:08X}), mips at 0x{:08X} (size "
      "0x{:08X})",
      action, key_.tiled ? "tiled" : "linear", key_.scaled_resolve ? "scaled " : "",
      key_.GetWidth(), key_.GetHeight(), key_.GetDepthOrArraySize(), key_.GetLogDimensionName(),
      FormatInfo::Get(key_.format)->name, key_.mip_max_level + 1, key_.packed_mips ? "" : "un",
      key_.mip_max_level != 0 ? "s" : "", key_.base_page << 12, key_.pitch << 5, GetGuestBaseSize(),
      key_.mip_page << 12, GetGuestMipsSize());
}

// The texture must be in the recent usage list. Place it in front now because
// after creation, the texture will likely be used immediately, and it should
// not be destroyed immediately after creation if dropping of old textures is
// performed somehow. The list is maintained by the Texture, not the
// TextureCache itself (unlike the `textures_` container).
TextureCache::Texture::Texture(TextureCache& texture_cache, const TextureKey& key, bool track_usage)
    : texture_cache_(texture_cache),
      key_(key),
      guest_layout_(key.GetGuestLayout()),
      last_usage_submission_index_(texture_cache.current_submission_index_),
      last_usage_time_(texture_cache.current_submission_time_),
      used_previous_(track_usage ? texture_cache.texture_used_last_ : nullptr),
      used_next_(nullptr),
      in_usage_list_(track_usage) {
  if (track_usage) {
    if (texture_cache.texture_used_last_) {
      texture_cache.texture_used_last_->used_next_ = this;
    } else {
      texture_cache.texture_used_first_ = this;
    }
    texture_cache.texture_used_last_ = this;
  }

  // Never try to upload data that doesn't exist.
  base_outdated_ = guest_layout().base.level_data_extent_bytes != 0;
  mips_outdated_ = guest_layout().mips_total_extent_bytes != 0;
  outdated_mask_.store(
      (base_outdated_ ? kOutdatedBitBase : 0) | (mips_outdated_ ? kOutdatedBitMips : 0),
      std::memory_order_relaxed);
}

TextureCache::Texture::~Texture() {
  ++g_bind_memo_destroyed;  // wwe2k14 patch (Session 95, r6tex)
  if (d3d_tex::g_active.load(std::memory_order_relaxed)) d3d_tex::Table::Get().ForgetTexture(this);  // wwe2k14 patch (Session 87, stage 4)
  if (mips_watch_handle_) {
    texture_cache().shared_memory().UnwatchMemoryRange(mips_watch_handle_);
  }
  if (base_watch_handle_) {
    texture_cache().shared_memory().UnwatchMemoryRange(base_watch_handle_);
  }

  if (in_usage_list_) {
    if (used_previous_) {
      used_previous_->used_next_ = used_next_;
    } else {
      texture_cache_.texture_used_first_ = used_next_;
    }
    if (used_next_) {
      used_next_->used_previous_ = used_previous_;
    } else {
      texture_cache_.texture_used_last_ = used_previous_;
    }
  }

  texture_cache_.UpdateTexturesTotalHostMemoryUsage(0, host_memory_usage_);
}

void TextureCache::Texture::MakeUpToDateAndWatch(
    const std::unique_lock<std::recursive_mutex>& global_lock) {
  SharedMemory& shared_memory = texture_cache().shared_memory();
  invalidated_since_load_.store(0, std::memory_order_relaxed);  // wwe2k14 patch (Session 7c, TEXLOAD)
  if (base_outdated_) {
    assert_not_zero(GetGuestBaseSize());
    base_outdated_ = false;
    if (!engine_flushed_) {  // wwe2k14 patch (Session 95, r6tex)
    base_watch_handle_ = shared_memory.WatchMemoryRange(
        key().base_page << 12, GetGuestBaseSize(), TextureCache::WatchCallback, this, nullptr, 0);
    }
    outdated_mask_.fetch_and(~kOutdatedBitBase, std::memory_order_release);
  }
  if (mips_outdated_) {
    assert_not_zero(GetGuestMipsSize());
    mips_outdated_ = false;
    if (!engine_flushed_) {  // wwe2k14 patch (Session 95, r6tex)
    mips_watch_handle_ = shared_memory.WatchMemoryRange(
        key().mip_page << 12, GetGuestMipsSize(), TextureCache::WatchCallback, this, nullptr, 1);
    }
    outdated_mask_.fetch_and(~kOutdatedBitMips, std::memory_order_release);
  }
}

void TextureCache::Texture::MarkAsUsed() {
  if (!in_usage_list_) {
    return;
  }
  assert_true(last_usage_submission_index_ <= texture_cache_.current_submission_index_);
  // This is called very frequently, don't relink unless needed for caching.
  if (last_usage_submission_index_ >= texture_cache_.current_submission_index_) {
    return;
  }
  last_usage_submission_index_ = texture_cache_.current_submission_index_;
  last_usage_time_ = texture_cache_.current_submission_time_;
  if (used_next_ == nullptr) {
    // Already the most recently used.
    return;
  }
  if (used_previous_ != nullptr) {
    used_previous_->used_next_ = used_next_;
  } else {
    texture_cache_.texture_used_first_ = used_next_;
  }
  used_next_->used_previous_ = used_previous_;
  used_previous_ = texture_cache_.texture_used_last_;
  used_next_ = nullptr;
  texture_cache_.texture_used_last_->used_next_ = this;
  texture_cache_.texture_used_last_ = this;
}

void TextureCache::Texture::MarkOutdatedByEngine([[maybe_unused]] const std::unique_lock<std::recursive_mutex>& global_lock) {
  // wwe2k14 patch (Session 87, D3D layer stage 4): like a watch fire, but the live watch handles are released first (the reload
  // makes new ones).
  if (GetGuestBaseSize()) {
    if (base_watch_handle_) {
      texture_cache().shared_memory().UnwatchMemoryRange(base_watch_handle_);
      base_watch_handle_ = nullptr;
    }
    base_outdated_ = true;
    outdated_mask_.fetch_or(kOutdatedBitBase, std::memory_order_release);
  }
  if (GetGuestMipsSize()) {
    if (mips_watch_handle_) {
      texture_cache().shared_memory().UnwatchMemoryRange(mips_watch_handle_);
      mips_watch_handle_ = nullptr;
    }
    mips_outdated_ = true;
    outdated_mask_.fetch_or(kOutdatedBitMips, std::memory_order_release);
  }
}

bool TextureCache::Texture::DropWatches([[maybe_unused]] const std::unique_lock<std::recursive_mutex>& global_lock) {
  if (base_outdated_ || mips_outdated_) return false;
  if (base_watch_handle_) {
    texture_cache().shared_memory().UnwatchMemoryRange(base_watch_handle_);
    base_watch_handle_ = nullptr;
  }
  if (mips_watch_handle_) {
    texture_cache().shared_memory().UnwatchMemoryRange(mips_watch_handle_);
    mips_watch_handle_ = nullptr;
  }
  return true;
}

void TextureCache::Texture::WatchCallback(
    [[maybe_unused]] const std::unique_lock<std::recursive_mutex>& global_lock, bool is_mip) {
  if (is_mip) {
    assert_not_zero(GetGuestMipsSize());
    mips_outdated_ = true;
    mips_watch_handle_ = nullptr;
    outdated_mask_.fetch_or(kOutdatedBitMips, std::memory_order_release);
  } else {
    assert_not_zero(GetGuestBaseSize());
    base_outdated_ = true;
    base_watch_handle_ = nullptr;
    outdated_mask_.fetch_or(kOutdatedBitBase, std::memory_order_release);
  }
}

void TextureCache::WatchCallback(const std::unique_lock<std::recursive_mutex>& global_lock,
                                 void* context, void* data, uint64_t argument,
                                 bool invalidated_by_gpu) {
  Texture& texture = *static_cast<Texture*>(context);
  texture.NoteInvalidated(invalidated_by_gpu);  // wwe2k14 patch (Session 7c, TEXLOAD): measurement only
  if (d3d_tex::g_active.load(std::memory_order_relaxed)) d3d_tex::Table::Get().NoteWatchFire(&texture);  // wwe2k14 patch (Session 87, stage 4)
  texture.WatchCallback(global_lock, argument != 0);
  texture.texture_cache().texture_became_outdated_.store(true, std::memory_order_release);
}

void TextureCache::DestroyAllTextures(bool from_destructor) {
  ResetTextureBindings(from_destructor);
  textures_.clear();
  COUNT_profile_set("gpu/texture_cache/textures", 0);
}

TextureCache::Texture* TextureCache::FindOrCreateTexture(TextureKey key) { return FindOrCreateTextureImpl(key, true); }
TextureCache::Texture* TextureCache::FindTextureOnly(TextureKey key) { return FindOrCreateTextureImpl(key, false); }

TextureCache::Texture* TextureCache::FindOrCreateTextureImpl(TextureKey key, bool create) {
  // Check if the texture is a scaled resolve texture.
  if (IsDrawResolutionScaled() && key.tiled && IsScaledResolveSupportedForFormat(key)) {
    texture_util::TextureGuestLayout scaled_resolve_guest_layout = key.GetGuestLayout();
    if ((scaled_resolve_guest_layout.base.level_data_extent_bytes &&
         IsRangeScaledResolved(key.base_page << 12,
                               scaled_resolve_guest_layout.base.level_data_extent_bytes)) ||
        (scaled_resolve_guest_layout.mips_total_extent_bytes &&
         IsRangeScaledResolved(key.mip_page << 12,
                               scaled_resolve_guest_layout.mips_total_extent_bytes))) {
      key.scaled_resolve = 1;
    }
  }

  auto get_host_extent = [this, &key](uint32_t& host_width_out, uint32_t& host_height_out,
                                      uint32_t& host_depth_or_array_size_out) {
    host_width_out = key.GetWidth();
    host_height_out = key.GetHeight();
    if (key.scaled_resolve) {
      host_width_out *= draw_resolution_scale_x();
      host_height_out *= draw_resolution_scale_y();
    }
    host_depth_or_array_size_out = key.GetDepthOrArraySize();
  };
  auto try_drop_top_level = [&key]() {
    if (!key.mip_page || !key.mip_max_level) {
      return false;
    }
    uint32_t old_width = key.GetWidth();
    uint32_t old_height = key.GetHeight();
    uint32_t old_depth_or_array_size = key.GetDepthOrArraySize();
    uint32_t new_width = std::max(old_width >> 1, UINT32_C(1));
    uint32_t new_height = std::max(old_height >> 1, UINT32_C(1));
    uint32_t new_depth_or_array_size = key.dimension == xenos::DataDimension::k3D
                                           ? std::max(old_depth_or_array_size >> 1, UINT32_C(1))
                                           : old_depth_or_array_size;
    if (new_width == old_width && new_height == old_height &&
        new_depth_or_array_size == old_depth_or_array_size) {
      return false;
    }
    const FormatInfo* format_info = FormatInfo::Get(key.format);
    uint32_t rebased_pitch_texels = std::max(rex::next_pow2(old_width) >> 1, UINT32_C(1));
    rebased_pitch_texels = rex::align(rebased_pitch_texels, format_info->block_width);
    rebased_pitch_texels = rex::align(rebased_pitch_texels, xenos::kTextureTileWidthHeight);
    if (!key.tiled) {
      uint32_t rebased_pitch_blocks = rebased_pitch_texels / format_info->block_width;
      uint32_t rebased_row_pitch_bytes = rebased_pitch_blocks * format_info->bytes_per_block();
      rebased_row_pitch_bytes =
          rex::align(rebased_row_pitch_bytes, xenos::kTextureLinearRowAlignmentBytes);
      rebased_pitch_blocks = rebased_row_pitch_bytes / format_info->bytes_per_block();
      rebased_pitch_texels = rebased_pitch_blocks * format_info->block_width;
    }
    uint32_t rebased_pitch = std::max(rebased_pitch_texels >> 5, UINT32_C(1));
    if (rebased_pitch > ((UINT32_C(1) << 9) - 1)) {
      return false;
    }
    key.width_minus_1 = new_width - 1;
    key.height_minus_1 = new_height - 1;
    key.depth_or_array_size_minus_1 = new_depth_or_array_size - 1;
    key.base_page = key.mip_page;
    key.mip_page = 0;
    key.pitch = rebased_pitch;
    key.mip_max_level = 0;
    key.packed_mips = 0;
    return true;
  };
  uint32_t host_width, host_height, host_depth_or_array_size;
  get_host_extent(host_width, host_height, host_depth_or_array_size);
  // If the host can't support the full texture extent, first try using the
  // unscaled version of a scaled resolve texture, then fall back to the first
  // stored mip level only.
  uint32_t max_host_width_height = GetMaxHostTextureWidthHeight(key.dimension);
  uint32_t max_host_depth_or_array_size = GetMaxHostTextureDepthOrArraySize(key.dimension);
  while (true) {
    bool width_height_too_large =
        host_width > max_host_width_height || host_height > max_host_width_height;
    bool depth_or_array_too_large = host_depth_or_array_size > max_host_depth_or_array_size;
    if (!width_height_too_large && !depth_or_array_too_large) {
      break;
    }
    bool size_adjusted = false;
    if (key.scaled_resolve) {
      key.scaled_resolve = 0;
      size_adjusted = true;
    } else if ((width_height_too_large ||
                (depth_or_array_too_large && key.dimension == xenos::DataDimension::k3D)) &&
               try_drop_top_level()) {
      size_adjusted = true;
    }
    if (!size_adjusted) {
      return nullptr;
    }
    get_host_extent(host_width, host_height, host_depth_or_array_size);
  }

  // Try to find an existing texture.
  // TODO(Triang3l): Reuse a texture with mip_page unchanged, but base_page
  // previously 0, now not 0, to save memory - common case in streaming.
  auto found_texture_it = textures_.find(key);
  if (found_texture_it != textures_.end()) {
    return found_texture_it->second.get();
  }
  if (!create) return nullptr;  // wwe2k14 patch (Session 92, M4c): find-only

  // Create the texture and add it to the map.
  Texture* texture;
  {
    std::unique_ptr<Texture> new_texture = CreateTexture(key);
    if (!new_texture) {
      key.LogAction("Failed to create");
      return nullptr;
    }
    assert_true(new_texture->key() == key);
    texture = textures_.emplace(key, std::move(new_texture)).first->second.get();
  }
  COUNT_profile_set("gpu/texture_cache/textures", textures_.size());
  texture->LogAction("Created");
  return texture;
}

// wwe2k14 patch (Session 18, DLAA): see the header.
TextureCache::Texture* TextureCache::DlaaFindResolveTexture(uint32_t base_address, uint32_t pitch_texels,
                                                            uint32_t width, uint32_t height,
                                                            xenos::TextureFormat format, xenos::Endian endian) {
  const uint32_t page = (base_address & 0x1FFFFFFFu) >> 12;
  Texture* found = nullptr;
  for (auto& entry : textures_) {
    const TextureKey& key = entry.first;
    if (key.base_page == page && key.dimension == xenos::DataDimension::k2DOrStacked && key.tiled &&
        key.GetWidth() == width && key.GetHeight() == height && key.format == format && !key.signed_separate &&
        key.GetDepthOrArraySize() == 1) {
      found = entry.second.get();
      if (key.scaled_resolve) break;
    }
  }
  if (!found) {
    TextureKey key;
    key.base_page = page;
    key.dimension = xenos::DataDimension::k2DOrStacked;
    key.width_minus_1 = width - 1;
    key.height_minus_1 = height - 1;
    key.tiled = 1;
    key.pitch = pitch_texels >> 5;
    key.format = format;
    key.endianness = endian;
    key.is_valid = 1;
    found = FindOrCreateTexture(key);
  }
  if (found && !LoadTextureData(*found)) return nullptr;
  return found;
}

bool TextureCache::LoadTextureData(Texture& texture) {
  PendingTextureLoad pending_load;
  PendingSharedMemoryRange pending_ranges[2];
  size_t pending_range_count = 0;
  if (!PrepareTextureLoad(texture, pending_load, pending_ranges, pending_range_count)) {
    return true;
  }

  std::pair<uint32_t, uint32_t> pending_range_pairs[2];
  for (size_t i = 0; i < pending_range_count; ++i) {
    pending_range_pairs[i] = std::make_pair(pending_ranges[i].start, pending_ranges[i].length);
  }
  SharedMemory::RamReaderScope ram_reader(shared_memory(), SharedMemory::RamReader::kTexture);  // wwe2k14 (S49)
  if (!shared_memory().RequestRanges(pending_range_pairs, pending_range_count)) {
    return false;
  }

  return CommitPreparedTextureLoad(pending_load);
}

// WWE13 port: some titles set the "invalid texture" type on fetch constants that are otherwise
// complete textures; accept those (from the WWE13 recomp SDK).
bool IsSelfConsistentInvalidTextureFetch(const xenos::xe_gpu_texture_fetch_t& fetch) {
  // The base address must point somewhere.
  if (fetch.base_address == 0) {
    return false;
  }
  // The size fields are stored with 1 subtracted from each component. Reject
  // zero dimensions and dimensions beyond the hardware limits of the
  // dimension. The 1D width limit mirrors the check in
  // BindingInfoFromFetchConstant.
  uint32_t width;
  switch (fetch.dimension) {
    case xenos::DataDimension::k1D:
      width = fetch.size_1d.width + 1;
      if (width == 0 || width > xenos::kTexture1DMaxWidth) {
        return false;
      }
      break;
    case xenos::DataDimension::k2DOrStacked: {
      width = fetch.size_2d.width + 1;
      uint32_t height = fetch.size_2d.height + 1;
      uint32_t depth_or_array_size = fetch.stacked ? fetch.size_2d.stack_depth + 1 : 1;
      if (width == 0 || height == 0 || width > xenos::kTexture2DCubeMaxWidthHeight ||
          height > xenos::kTexture2DCubeMaxWidthHeight ||
          depth_or_array_size > xenos::kTexture2DMaxStackDepth) {
        return false;
      }
    } break;
    case xenos::DataDimension::k3D: {
      width = fetch.size_3d.width + 1;
      uint32_t height = fetch.size_3d.height + 1;
      uint32_t depth = fetch.size_3d.depth + 1;
      if (width == 0 || height == 0 || depth == 0 ||
          width > xenos::kTexture3DMaxWidthHeight || height > xenos::kTexture3DMaxWidthHeight ||
          depth > xenos::kTexture3DMaxDepth) {
        return false;
      }
    } break;
    case xenos::DataDimension::kCube: {
      width = fetch.size_2d.width + 1;
      uint32_t height = fetch.size_2d.height + 1;
      if (width == 0 || height == 0 || width > xenos::kTexture2DCubeMaxWidthHeight ||
          height > xenos::kTexture2DCubeMaxWidthHeight) {
        return false;
      }
    } break;
    default:
      return false;
  }
  // The format must be one that the texture system knows about (the 6-bit
  // field always has a table entry, but keep the guard).
  const FormatInfo* format_info = FormatInfo::Get(fetch.format);
  if (format_info == nullptr) {
    return false;
  }
  // The pitch is in pixels >> 5, and may legitimately exceed the aligned
  // width, but it must be able to cover at least the block-aligned width, and
  // the field is 9 bits wide.
  uint32_t pitch = fetch.pitch;
  if (pitch > ((UINT32_C(1) << 9) - 1) ||
      (pitch << 5) < rex::align(width, format_info->block_width)) {
    return false;
  }
  return true;
}


void TextureCache::BindingInfoFromFetchConstant(const xenos::xe_gpu_texture_fetch_t& fetch,
                                                TextureKey& key_out, uint8_t* swizzled_signs_out) {
  // Reset the key and the signedness.
  key_out.MakeInvalid();
  if (swizzled_signs_out != nullptr) {
    *swizzled_signs_out = uint8_t(xenos::TextureSign::kUnsigned) * uint8_t(0b01010101);
  }

  switch (fetch.type) {
    case xenos::FetchConstantType::kTexture:
      break;
    case xenos::FetchConstantType::kInvalidTexture:
      if (REXCVAR_GET(gpu_allow_invalid_fetch_constants) || IsSelfConsistentInvalidTextureFetch(fetch)) {
        break;
      }
      REXGPU_WARN(
          "Texture fetch constant ({:08X} {:08X} {:08X} {:08X} {:08X} {:08X}) "
          "has \"invalid\" type! This is incorrect behavior, but you can try "
          "bypassing this by launching Xenia with "
          "--gpu_allow_invalid_fetch_constants=true.",
          fetch.dword_0, fetch.dword_1, fetch.dword_2, fetch.dword_3, fetch.dword_4, fetch.dword_5);
      return;
    default:
      REXGPU_WARN(
          "Texture fetch constant ({:08X} {:08X} {:08X} {:08X} {:08X} {:08X}) "
          "is completely invalid!",
          fetch.dword_0, fetch.dword_1, fetch.dword_2, fetch.dword_3, fetch.dword_4, fetch.dword_5);
      return;
  }

  uint32_t width_minus_1, height_minus_1, depth_or_array_size_minus_1;
  uint32_t base_page, mip_page, mip_max_level;
  texture_util::GetSubresourcesFromFetchConstant(fetch, &width_minus_1, &height_minus_1,
                                                 &depth_or_array_size_minus_1, &base_page,
                                                 &mip_page, nullptr, &mip_max_level);
  if (base_page == 0 && mip_page == 0) {
    // No texture data at all.
    return;
  }
  if (fetch.dimension == xenos::DataDimension::k1D) {
    bool is_invalid_1d = false;
    // TODO(Triang3l): Support long 1D textures.
    if (width_minus_1 >= xenos::kTexture2DCubeMaxWidthHeight) {
      REXGPU_ERROR(
          "1D texture is too wide ({}) - ignoring! Report the game to Xenia "
          "developers",
          width_minus_1 + 1);
      is_invalid_1d = true;
    }
    assert_false(fetch.tiled);
    if (fetch.tiled) {
      REXGPU_ERROR(
          "1D texture has tiling enabled in the fetch constant, but this "
          "appears to be completely wrong - ignoring! Report the game to Xenia "
          "developers");
      is_invalid_1d = true;
    }
    assert_false(fetch.packed_mips);
    if (fetch.packed_mips) {
      REXGPU_ERROR(
          "1D texture has packed mips enabled in the fetch constant, but this "
          "appears to be completely wrong - ignoring! Report the game to Xenia "
          "developers");
      is_invalid_1d = true;
    }
    if (is_invalid_1d) {
      return;
    }
  }

  xenos::TextureFormat format = GetBaseFormat(fetch.format);

  key_out.base_page = base_page;
  key_out.mip_page = mip_page;
  key_out.dimension = fetch.dimension;
  key_out.width_minus_1 = width_minus_1;
  key_out.height_minus_1 = height_minus_1;
  key_out.depth_or_array_size_minus_1 = depth_or_array_size_minus_1;
  key_out.pitch = fetch.pitch;
  key_out.mip_max_level = mip_max_level;
  key_out.tiled = fetch.tiled;
  key_out.packed_mips = fetch.packed_mips;
  key_out.format = format;
  key_out.endianness = fetch.endianness;

  key_out.is_valid = 1;

  if (swizzled_signs_out != nullptr) {
    *swizzled_signs_out = texture_util::SwizzleSigns(fetch);
  }
}

void TextureCache::ResetTextureBindings(bool from_destructor) {
  uint32_t bindings_reset = 0;
  for (size_t i = 0; i < texture_bindings_.size(); ++i) {
    TextureBinding& binding = texture_bindings_[i];
    if (!binding.key.is_valid) {
      continue;
    }
    binding.Reset();
    bindings_reset |= UINT32_C(1) << i;
  }
  texture_bindings_in_sync_ &= ~bindings_reset;
  if (bindings_reset) {
    ++active_texture_bindings_version_;  // wwe2k14 patch (Session 6, CP9)
  }
  if (!from_destructor && bindings_reset) {
    UpdateTextureBindingsImpl(bindings_reset);
  }
}

void TextureCache::UpdateTexturesTotalHostMemoryUsage(uint64_t add, uint64_t subtract) {
  textures_total_host_memory_usage_ = textures_total_host_memory_usage_ - subtract + add;
  COUNT_profile_set(
      "gpu/texture_cache/total_host_memory_usage_mb",
      uint32_t((textures_total_host_memory_usage_ + ((UINT32_C(1) << 20) - 1)) >> 20));
}

bool TextureCache::IsRangeScaledResolved(uint32_t start_unscaled, uint32_t length_unscaled) {
  if (!IsDrawResolutionScaled()) {
    return false;
  }

  start_unscaled = std::min(start_unscaled, SharedMemory::kBufferSize);
  length_unscaled = std::min(length_unscaled, SharedMemory::kBufferSize - start_unscaled);
  if (!length_unscaled) {
    return false;
  }

  // Two-level check for faster rejection since resolve targets are usually
  // placed in relatively small and localized memory portions (confirmed by
  // testing - pretty much all times the deeper level was entered, the texture
  // was a resolve target).
  uint32_t page_first = start_unscaled >> 12;
  uint32_t page_last = (start_unscaled + length_unscaled - 1) >> 12;
  uint32_t block_first = page_first >> 5;
  uint32_t block_last = page_last >> 5;
  uint32_t l2_block_first = block_first >> 6;
  uint32_t l2_block_last = block_last >> 6;
  auto global_lock = global_critical_region_.Acquire();
  for (uint32_t i = l2_block_first; i <= l2_block_last; ++i) {
    uint64_t l2_block = scaled_resolve_pages_l2_[i];
    if (i == l2_block_first) {
      l2_block &= ~((UINT64_C(1) << (block_first & 63)) - 1);
    }
    if (i == l2_block_last && (block_last & 63) != 63) {
      l2_block &= (UINT64_C(1) << ((block_last & 63) + 1)) - 1;
    }
    uint32_t block_relative_index;
    while (rex::bit_scan_forward(l2_block, &block_relative_index)) {
      l2_block &= ~(UINT64_C(1) << block_relative_index);
      uint32_t block_index = (i << 6) + block_relative_index;
      uint32_t check_bits = UINT32_MAX;
      if (block_index == block_first) {
        check_bits &= ~((UINT32_C(1) << (page_first & 31)) - 1);
      }
      if (block_index == block_last && (page_last & 31) != 31) {
        check_bits &= (UINT32_C(1) << ((page_last & 31) + 1)) - 1;
      }
      if (scaled_resolve_pages_[block_index] & check_bits) {
        return true;
      }
    }
  }
  return false;
}

void TextureCache::ScaledResolveGlobalWatchCallbackThunk(
    const std::unique_lock<std::recursive_mutex>& global_lock, void* context,
    uint32_t address_first, uint32_t address_last, bool invalidated_by_gpu) {
  TextureCache* texture_cache = reinterpret_cast<TextureCache*>(context);
  texture_cache->ScaledResolveGlobalWatchCallback(global_lock, address_first, address_last,
                                                  invalidated_by_gpu);
}

void TextureCache::ScaledResolveGlobalWatchCallback(
    const std::unique_lock<std::recursive_mutex>& global_lock, uint32_t address_first,
    uint32_t address_last, bool invalidated_by_gpu) {
  assert_true(IsDrawResolutionScaled());
  if (invalidated_by_gpu) {
    // Resolves themselves do exactly the opposite of what this should do.
    return;
  }
  // Mark scaled resolve ranges as non-scaled. Textures themselves will be
  // invalidated by their shared memory watches.
  uint32_t resolve_page_first = address_first >> 12;
  uint32_t resolve_page_last = address_last >> 12;
  uint32_t resolve_block_first = resolve_page_first >> 5;
  uint32_t resolve_block_last = resolve_page_last >> 5;
  uint32_t resolve_l2_block_first = resolve_block_first >> 6;
  uint32_t resolve_l2_block_last = resolve_block_last >> 6;
  for (uint32_t i = resolve_l2_block_first; i <= resolve_l2_block_last; ++i) {
    uint64_t resolve_l2_block = scaled_resolve_pages_l2_[i];
    if (REXCVAR_GET(pre_mask_resolve_l2_block)) {
      // Pre-mask to only process blocks within the write range.
      if (i == resolve_l2_block_first) {
        resolve_l2_block &= ~((UINT64_C(1) << (resolve_block_first & 63)) - 1);
      }
      if (i == resolve_l2_block_last && (resolve_block_last & 63) != 63) {
        resolve_l2_block &= (UINT64_C(1) << ((resolve_block_last & 63) + 1)) - 1;
      }
    }
    uint32_t resolve_block_relative_index;
    while (rex::bit_scan_forward(resolve_l2_block, &resolve_block_relative_index)) {
      resolve_l2_block &= ~(UINT64_C(1) << resolve_block_relative_index);
      uint32_t resolve_block_index = (i << 6) + resolve_block_relative_index;
      uint32_t resolve_keep_bits = 0;
      if (resolve_block_index == resolve_block_first) {
        resolve_keep_bits |= (UINT32_C(1) << (resolve_page_first & 31)) - 1;
      }
      if (resolve_block_index == resolve_block_last && (resolve_page_last & 31) != 31) {
        resolve_keep_bits |= ~((UINT32_C(1) << ((resolve_page_last & 31) + 1)) - 1);
      }
      if (scaled_resolve_pages_[resolve_block_index] & ~resolve_keep_bits) {
        d3d_tex::Table::Get().marks_version.fetch_add(1, std::memory_order_relaxed);  // wwe2k14 patch (Session 87, stage 4)
      }
      scaled_resolve_pages_[resolve_block_index] &= resolve_keep_bits;
      if (scaled_resolve_pages_[resolve_block_index] == 0) {
        scaled_resolve_pages_l2_[i] &= ~(UINT64_C(1) << resolve_block_relative_index);
      }
    }
  }
}

}  // namespace rex::graphics
