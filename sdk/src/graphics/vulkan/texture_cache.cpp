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
#include <array>
#include <atomic>  // Session 95
#include <cctype>  // Session 94
#include <rex/perf/frame_events.h>  // Session 73 (texture loads per frame)
#include <cstddef>
#include <cstring>
#include <filesystem>  // wwe2k14 patch (Session 7c, TEXLOAD): the texture load CSV
#include <iterator>
#include <string>
#include <utility>
#ifdef _WIN32
#include <share.h>  // wwe2k14 patch (Session 7c, TEXLOAD): _wfsopen sharing mode
#endif

#include <rex/assert.h>
#include <rex/bit.h>
#include <rex/chrono/clock.h>  // wwe2k14 patch (Session 7c, TEXDIRECT)
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/free_counters.h>  // wwe2k14 patch (Session 92, M3b)
#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/graphics/util/bit_inline.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/deferred_command_buffer.h>
#include <rex/graphics/vulkan/texture_cache.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/perf/timeline.h>  // wwe2k14 patch (Session 7c, TEXDIRECT / TEXLOAD): cpf_texdirect, cpf_texload
#include <rex/ui/vulkan/mem_alloc.h>
#include <rex/ui/vulkan/ui_samplers.h>
#include <rex/ui/vulkan/util.h>

REXCVAR_DEFINE_BOOL(non_seamless_cube_map, false, "GPU", "Use non-seamless cube map sampling");
// wwe2k14 patch (Session 55, the red blocks on Intel laptops): other GPUs may sample other mip levels than this one
// (their own LOD from the same gradients). Diagnostic: every mipmapped sampler starts this many levels further down, so
// the smaller levels - the ones a resolve fills or a load writes that this PC hardly samples - are what the pictures
// show (compare runs with a setting on / off at the same value).
REXCVAR_DEFINE_DOUBLE(gpu_debug_texture_min_lod, 0.0, "GPU",
                      "Diagnostic (Session 55): add this to every mipmapped sampler's minimum LOD (1 = the first "
                      "level below the base is the largest sampled); 0 = off. Read when a sampler is made: set it at "
                      "start");
// wwe2k14 patch (Session 45, NATIVE TEXWRITE): the native renderer's settings (native_renderer.cpp).
REXCVAR_DECLARE(int32_t, gpu_tex_bind_memo);  // (Session 95, r6tex: pipeline\texture\cache.cpp)
REXCVAR_DECLARE(bool, gpu_tex_bind_memo_verify);
REXCVAR_DECLARE(bool, gpu_native_all);
REXCVAR_DECLARE(bool, gpu_native_resolve_texture);
REXCVAR_DECLARE(bool, gpu_native_truth);  // wwe2k14 patch (Session 47, NATIVE TRUTH M6)

// wwe2k14 patch (Session 7c, TEXDIRECT): at resolution_scale 3 every reload of a render target resolved at scale
// costs ~0.6 ms of GPU time in three serialized steps: the load shader untiles into one shared scratch buffer,
// vkCmdCopyBufferToImage copies that into the image, then a barrier (docs/research/texture_loads_scaled.md, lever
// L1). With vulkan_texture_load_direct, loads of resolution-scaled single-level single-layer textures that use the
// 32bpb or 64bpb load shader write the texels from the compute shader directly into the image (see
// shaders/texture_load_scaled_direct.cs.glsl). The image needs the storage usage, so this applies to textures
// created while it's on (loads of others keep the old path, counted as fallbacks). Read at texture creation and per
// load, so it can be toggled at any time: images created while it was on keep the storage and transfer source
// usage and the unsigned integer view format in their mutable format list, and take the old path while it's off.
// wwe2k14 patch (Session 93): transcode counters (file-static: no header layout change).
static uint64_t g_transcode_blocks[3] = {};
static uint64_t g_transcode_loads[3] = {};
static uint32_t g_transcode_frames = 0;

// wwe2k14 patch (Session 93, ASTC / EAC transcode): on a GPU that cannot sample BC1 / BC4 / BC5 (most Mali, some Adreno,
// PowerVR) the old fallback decompresses to RGBA8 / R8 / RG8 (4-8x the bytes). With this on, DXT1 is loaded as ASTC 4x4
// LDR (BC1 is transcoded bit by bit, error <= 1.33 of 255), DXT5A as EAC R11 and DXN as EAC RG11 (lossy: EAC has 16
// fixed modifier tables, so BC4's evenly spaced levels are fitted, Session 94: max error ~13 of 255, p99 ~9, mean ~1.3 on random worst-case blocks), whenever the target
// format is supported for sampling with linear filtering. DXT3 / DXT5 keep RGBA8. Signed and non-multiple-of-4 textures
// keep the old path. Read at startup. Off = the old behaviour byte for byte.
REXCVAR_DEFINE_BOOL(vulkan_texture_transcode, false, "GPU/Vulkan",
                    "Load DXT1 as ASTC 4x4 instead of decompressing to RGBA8 when the GPU lacks BC1 support "
                    "(DXT5A / DXN: see vulkan_texture_transcode_eac; read at startup). Transparent texels keep the colour of the block's first endpoint "
                    "(BC1 gives black): see vulkan_texture_transcode_bc1_alpha_fix");
// Session 93: the EAC half of the transcode is lossy (EAC has 16 fixed modifier tables, BC4's evenly spaced levels are
// fitted per block), so it has its own switch; vulkan_texture_transcode alone only transcodes DXT1 -> ASTC.
REXCVAR_DEFINE_BOOL(vulkan_texture_transcode_eac, false, "GPU/Vulkan",
                    "With vulkan_texture_transcode: also load DXT5A / DXN as EAC R11 / RG11 (per-block fit; Session 94: "
                    "wider table / multiplier / base search with a worst-case term, max error ~13 (was 26), mean 1.27 at "
                    "8 bits on random worst-case blocks, smooth alpha is far lower; EAC cannot reach 1 LSB) - normal maps "
                    "may show it; off keeps R8 / RG8 (read at startup)");
// Session 95 (Lane E): a storage buffer binding must stay within maxStorageBufferRange. Ranges are texture-sized today, so this
// should never fire; if it does the work is refused and logged (the first few times) instead of a binding the driver may not honour.
namespace {
bool VulkanTextureStorageRangeOk(uint64_t range, uint64_t limit, const char* what) {
  if (range <= limit) return true;
  static std::atomic<uint32_t> refused{0};
  const uint32_t n = refused.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n <= 8 || (n & (n - 1)) == 0) {
    REXGPU_ERROR("[tex][storage-range] {} of {} bytes is over maxStorageBufferRange {}: refused (the {}. time)", what, range, limit, n);
  }
  return false;
}
}  // namespace

// Session 94: the transparent texels of a 3-colour + transparent BC1 block (c0 <= c1, index 3) kept the colour of ASTC
// weight 0 (the darker endpoint), which can darken or tint bilinear filtering of straight-alpha edges; with this switch they take the
// colour level nearest the mean of the block's opaque texels (alpha stays exactly 0).
REXCVAR_DEFINE_BOOL(vulkan_texture_transcode_bc1_alpha_fix, false, "GPU/Vulkan",
                    "Session 94: with vulkan_texture_transcode, transparent texels of 3-colour BC1 blocks take the colour "
                    "nearest the block's mean opaque colour instead of the first endpoint's (alpha stays 0). For alpha-test / "
                    "straight-alpha content; premultiplied or additive content wants black, not this (read at startup)");
// Session 95 (the purple Savage): a PC cannot sample ASTC, so what the transcode does to 3-colour BC1 blocks could not be seen on one. This loads DXT1 through
// a variant of the transcode shader into BC3: the texels as the transcode leaves them (RGB of a transparent texel not black), on any GPU with BC. Test switch only.
// 1 = as vulkan_texture_transcode gives them, 2 = with vulkan_texture_transcode_bc1_alpha_fix. Read at startup.
REXCVAR_DEFINE_INT32(vulkan_emulate_bc1_transcode, 0, "GPU/Vulkan",
                     "Session 95, test only: DXT1 loads with the transparent texels' RGB as the ASTC transcode gives it, through BC3 (1 the first endpoint, "
                     "2 the alpha-fix level). 0 = off (read at startup)")
    .range(0, 2);
// Session 94: k_16 / k_16_16 / k_16_16_16_16 UNORM / SNORM are optional in Vulkan and not linear-filterable on e.g. Mali
// r49 (Bifrost); the SFLOAT substitute (converted by the load shader) is always filterable. auto = the substitute only
// when the format query says the UNORM / SNORM format cannot be sampled with linear filtering (the PC never takes it);
// on = always (tests the substitute on a PC); off = never (wrong filtering where the query says not filterable).
REXCVAR_DEFINE_STRING(vulkan_texture_16bit_filter_substitute, "auto", "GPU/Vulkan",
                      "Session 94: auto / on / off. 16-bit UNORM / SNORM textures use an SFLOAT host format (converted in "
                      "the load shader) when the GPU cannot linear-filter the UNORM / SNORM one (auto), always (on) or "
                      "never (off); logs [tex] 16-bit substitute (read at startup)");
// Session 93: treat every BC format as unsupported, so a PC with BC support exercises the fallback / transcode selection
// (the target format may be unsupported on the PC too: then the log says so and the RGBA8 path runs).
REXCVAR_DEFINE_BOOL(vulkan_emulate_no_bc, false, "GPU/Vulkan",
                    "Treat BC1-BC5 as unsupported by the GPU (tests the decompress / transcode fallbacks on a PC; read at "
                    "startup)");

REXCVAR_DEFINE_BOOL(vulkan_texture_load_direct, false, "GPU/Vulkan",
                    "Write resolution-scaled single-level 32bpb / 64bpb texture loads directly into the image from "
                    "the load compute shader instead of through a scratch buffer and a buffer-to-image copy "
                    "(applies to textures created while on)");
// Review fix: diagnostic only - the comparisons run on the command processor thread at the frame boundary (two
// copies of every verified image, each ~33 MB for a 32bpb / ~66 MB for a 64bpb 3840x2160 load, up to
// vulkan_texture_load_direct_verify_per_frame loads), and the reference dispatches and readback copies add GPU work,
// so keep it off for timing runs ([s7][TEXDIRECT] reports the compare time). Its readback buffers are released at
// the frame boundaries after it's turned off.
// wwe2k14 patch (Session 43, S0b-lite): see the comment above vulkan_texture_load_direct.
REXCVAR_DEFINE_BOOL(vulkan_texture_load_direct_unscaled, false, "GPU/Vulkan",
                    "Session 43 (S0b-lite): at resolution scale 1, textures over pages a resolve has written (the "
                    "outputs of render passes the next pass samples, reloaded every frame) are loaded straight into "
                    "their image as vulkan_texture_load_direct does for scaled ones (no scratch buffer, no copy). "
                    "vulkan_texture_load_direct_verify then compares with the regular 1x load shader");
// wwe2k14 patch (Session 52): which guest formats vulkan_texture_load_direct_unscaled makes direct images for.
REXCVAR_DEFINE_INT32(vulkan_texture_load_direct_unscaled_kinds, 127, "GPU/Vulkan",
                     "Session 52: with vulkan_texture_load_direct_unscaled, the guest formats whose 1x textures get a "
                     "direct-load image (the storage usage), as a bit mask: 1 k_24_8 / k_24_8_FLOAT, 2 k_8_8_8_8, 8 "
                     "k_16_16_16_16 (and _FLOAT), 16 k_16_16 (and _FLOAT), 32 k_8 / k_4_4_4_4, 64 any other; the "
                     "others are loaded as without the setting. Applies to textures created while set")
    .range(0, 127);
REXCVAR_DEFINE_BOOL(vulkan_texture_native_storage_small, false, "GPU/Vulkan",
                    "Session 59: new 1x k_8 textures without mips (R8 host formats) get an R8_UINT storage view (with "
                    "shaderStorageImageExtendedFormats), so native fills and resolves can write them directly "
                    "(k_4_4_4_4: no - B4G4R4A4 has no storage image support; review)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_fill_cube_faces, false, "GPU/Vulkan",
                    "Session 79 (HD 520 arc): new 1x tiled single-level k_8_8_8_8 cube textures get an R32_UINT storage "
                    "view (one 2D view a face), and a truth frame's direct resolve into a face also fills that face of "
                    "the cube when the cube is up to date (the memory is still written; the cube is not reloaded)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(vulkan_texture_load_direct_unscaled_any_page, false, "GPU/Vulkan",
                    "Session 52: with vulkan_texture_load_direct_unscaled, a 1x tiled texture of those kinds gets a "
                    "direct-load image even if no resolve has written its page yet when it is made (textures loaded "
                    "before their first resolve - the native truth fills and texture-only resolves need the image). "
                    "Applies to textures created while on");
// wwe2k14 patch (Session 44, S0b ceiling): measurement only - the most full S0b (the resolve writing the texture's
// image, so no load follows) could save is the GPU time of the direct 1x reloads it would remove.
REXCVAR_DEFINE_BOOL(vulkan_texture_load_skip_debug, false, "GPU/Vulkan",
                    "Measurement only, the picture is WRONG with it on: a direct 1x load "
                    "(vulkan_texture_load_direct_unscaled) into an image it has filled before is skipped, so the "
                    "texture keeps old contents. In-run A/B of this gives the GPU time the reloads cost (the ceiling "
                    "of S0b)");
REXCVAR_DEFINE_BOOL(vulkan_texture_load_direct_verify, false, "GPU/Vulkan",
                    "For a few direct texture loads per frame, also run the original load shader into a readback "
                    "buffer, read the directly written image back, and compare them byte for byte after the "
                    "submission completes ([s7][TEXDIRECT] verify mismatch). Diagnostic only: costs CPU time at "
                    "every frame boundary and GPU time, keep it off for timing runs");
REXCVAR_DEFINE_INT32(vulkan_texture_load_direct_verify_per_frame, 4, "GPU/Vulkan",
                     "Maximum number of direct texture loads verified per frame with "
                     "vulkan_texture_load_direct_verify");
REXCVAR_DEFINE_BOOL(vulkan_texture_load_direct_verify_control, false, "GPU/Vulkan",
                    "Positive control for vulkan_texture_load_direct_verify: the reference load runs with a different "
                    "endianness, so verified loads of non-uniform data must be reported as mismatches");

// wwe2k14 patch (Session 6, CP6): GetSamplerParameters runs for every sampler binding of every draw
// (VulkanCommandProcessor::IssueDraw) and recomputes everything from the fetch constant (2.5% of the command
// processor thread in heavy entrance shots, stacks_067), then UseSampler does a hash map lookup (0.4%). The
// inputs repeat from draw to draw and from frame to frame, so both results are cached by value.
REXCVAR_DEFINE_BOOL(gpu_sampler_mip_point, false, "GPU/Vulkan",
                    "Session 95 (r3gpu, default off; a look lever for LOW, judged by eye): every sampler takes the nearer mip level instead of blending two (trilinear to bilinear): half the "
                    "sampler work per fetch on Intel Gen9. Visible as mip lines on floors and far textures. Read when a sampler is made");
REXCVAR_DEFINE_BOOL(vulkan_sampler_param_cache, false, "GPU/Vulkan",
                    "Cache sampler parameters keyed on the texture fetch constant value and the shader's filter "
                    "overrides, and skip sampler lookups already done in the current submission");
REXCVAR_DEFINE_BOOL(vulkan_sampler_param_cache_verify, false, "GPU/Vulkan",
                    "Shadow-check vulkan_sampler_param_cache (works with it on or off): run the cache lookups, "
                    "but also compute the original results, use those, and count and log mismatches");

// wwe2k14 patch (Session 7c, TEXLOAD): texture loads (LoadTextureDataFromResidentMemoryImpl). At 4K a reload of a
// full-screen render target resolved at scale takes ~0.6 ms of GPU time (docs/research/texture_loads_scaled.md).
// Both read per load, so they can be flipped at any time.
REXCVAR_DEFINE_BOOL(vulkan_texture_load_discard_old, false, "GPU/Vulkan",
                    "Texture loads that rewrite every mip level and array layer of the image transition it from "
                    "the undefined layout before the copy (discarding contents the copy overwrites anyway) instead "
                    "of preserving them. Same texture contents");
REXCVAR_DEFINE_BOOL(vulkan_debug_texture_load_names, false, "GPU/Vulkan",
                    "Session 95, validation diagnosis: names the texture load scratch buffers and the shared memory buffer for the validation layer, and counts the loads of more than one dispatch "
                    "([tex] load dispatches lines). Nothing else changes.");
REXCVAR_DEFINE_BOOL(vulkan_debug_texture_load_dispatch_barrier, false, "GPU/Vulkan",
                    "Session 95, validation diagnosis: a compute-to-compute barrier on the scratch buffer between the dispatches of one texture load. The levels and slices of a load write separate "
                    "ranges, so this only tells validation's descriptor-wide reading apart from a true overlap.");
REXCVAR_DEFINE_BOOL(vulkan_texture_load_scratch_ring, false, "GPU/Vulkan",
                    "Texture loads use a ring of 4 scratch buffers (each grown as needed) instead of one shared "
                    "scratch buffer, so the untiling dispatch of one load can overlap the copy of the previous one. "
                    "Same texture contents (except texels the load never writes, see AcquireScratchGpuBufferRing); "
                    "up to 4x the scratch memory (freed when turned off)");
// wwe2k14 patch (Session 22, RESWAW): see UseScaledResolveBufferForWrite. Read per resolve (can be flipped any time).
REXCVAR_DEFINE_BOOL(vulkan_resolve_skip_disjoint_waw, false, "GPU/Vulkan",
                    "No write-after-write barrier between resolves that write non-overlapping ranges of the "
                    "resolution-scaled resolve buffer (a resolve right after another one then needs no pipeline "
                    "barrier). Readers still get the whole-buffer barrier");

// wwe2k14 patch (Session 22, EAGER): see VulkanTextureCache::RequestTextures. Read per draw (can be flipped any time).
// wwe2k14 patch (Session 56): read by the native renderer (native_truth.cpp).
REXCVAR_DEFINE_BOOL(vulkan_texture_sample_written_early, false, "GPU/Vulkan",
                    "Session 56 (NATIVE truth frames): before a truth pass opens, every texture written by compute or a "
                    "copy since (resolves, fills, loads) is moved to the sampled layout in one barrier - not at its "
                    "first sampling, which in the middle of a pass ends the pass (Bryan's entrance and the match: ~33 "
                    "pass splits in two frames). Layouts only: the same pictures")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(vulkan_texture_load_eager, false, "GPU/Vulkan",
                    "Load a resolution-scaled texture right after the resolve that completed it, when in the "
                    "previous frame it was loaded only after more draws (so its barrier waits for one resolve "
                    "instead of the draws in flight). Same texture contents (a texture written again before its "
                    "use is reloaded at its use)");
REXCVAR_DEFINE_BOOL(vulkan_texture_load_early_cpu, false, "GPU/Vulkan",
                    "Session 87 (step 3): a texture the CPU rewrote and the previous frame loaded at its use (the entrance tron "
                    "video's planes: three loads a frame, each ending the scene's render pass) is loaded at the frame's first "
                    "draw instead. Same contents: the load re-arms the texture's watch, so a CPU write after it reloads the "
                    "texture at its use, as before")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 79, diagnostic): which textures each draw samples, in the segment log's order (is a copy read
// before the next copy over it?).
REXCVAR_DEFINE_BOOL(vulkan_texture_sampled_seg_log, false, "GPU/Vulkan",
                    "Diagnostic (Session 79): while the native segment log runs, a '[native][seg] texture sampled' line "
                    "per texture a draw samples (page, format, size, fetch constant)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// wwe2k14 patch (Session 22, TEXDEPTH): see GetDirectLoadViewFormat. Read when a texture is created (whether its
// image gets the storage usage) and per load.
REXCVAR_DEFINE_BOOL(vulkan_texture_load_direct_depth, false, "GPU/Vulkan",
                    "With vulkan_texture_load_direct: also load resolution-scaled 24-bit float depth textures "
                    "(k_24_8_FLOAT -> R32_SFLOAT) directly into the image (one dispatch instead of a dispatch into a "
                    "scratch buffer + a copy). Bit-exact (vulkan_texture_load_direct_verify compares)");

REXCVAR_DEFINE_INT32(vulkan_texture_load_direct_verify_shader, -1, "GPU/Vulkan",
                     "vulkan_texture_load_direct_verify only for the loads of one direct load shader (0 32bpb, 1 "
                     "64bpb, 2 24-bit float depth); -1 = any");

namespace rex::graphics::vulkan {

// wwe2k14 patch (Session 52): the vulkan_texture_load_direct_unscaled_kinds bit of a guest format.
static uint32_t DirectUnscaledKindBit(xenos::TextureFormat format) {
  switch (format) {
    case xenos::TextureFormat::k_24_8:
    case xenos::TextureFormat::k_24_8_FLOAT:
      return 1;
    case xenos::TextureFormat::k_8_8_8_8:
      return 2;
    case xenos::TextureFormat::k_16_16_16_16:
    case xenos::TextureFormat::k_16_16_16_16_FLOAT:
      return 8;
    case xenos::TextureFormat::k_16_16:
    case xenos::TextureFormat::k_16_16_FLOAT:
      return 16;
    case xenos::TextureFormat::k_8:
    case xenos::TextureFormat::k_4_4_4_4:
      return 32;
    default:
      return 64;
  }
}

// Generated with `xb buildshaders`.
namespace shaders {
#include "../shaders/vulkan_spirv/texture_load_128bpb_cs.h"
#include "../shaders/vulkan_spirv/texture_load_128bpb_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_16bpb_cs.h"
#include "../shaders/vulkan_spirv/texture_load_16bpb_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_32bpb_cs.h"
#include "../shaders/vulkan_spirv/texture_load_32bpb_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_64bpb_cs.h"
#include "../shaders/vulkan_spirv/texture_load_64bpb_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_8bpb_cs.h"
#include "../shaders/vulkan_spirv/texture_load_8bpb_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_bgrg8_rgb8_cs.h"
#include "../shaders/vulkan_spirv/texture_load_ctx1_cs.h"
#include "../shaders/vulkan_spirv/texture_load_depth_float_cs.h"
#include "../shaders/vulkan_spirv/texture_load_depth_float_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_depth_unorm_cs.h"
#include "../shaders/vulkan_spirv/texture_load_depth_unorm_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_dxn_rg8_cs.h"
#include "../shaders/vulkan_spirv/texture_load_dxt1_rgba8_cs.h"
#include "../shaders/vulkan_spirv/texture_load_dxt3_rgba8_cs.h"
#include "../shaders/vulkan_spirv/texture_load_dxt3a_cs.h"
#include "../shaders/vulkan_spirv/texture_load_dxt3aas1111_argb4_cs.h"
#include "../shaders/vulkan_spirv/texture_load_dxt5_rgba8_cs.h"
#include "../shaders/vulkan_spirv/texture_load_dxt5a_r8_cs.h"
// wwe2k14 patch (Session 93): BC -> ASTC / EAC transcoding loads (vulkan_glsl/texture_transcode_cs.glsl).
#include "../shaders/vulkan_spirv/texture_load_dxn_eac_rg11_cs.h"
#include "../shaders/vulkan_spirv/texture_load_dxt1_astc4x4_alphafix_cs.h"  // Session 94
#include "../shaders/vulkan_spirv/texture_load_dxt1_astc4x4_cs.h"
#include "../shaders/vulkan_spirv/texture_load_dxt1_emu_bc3_alphafix_cs.h"  // Session 95 (test only)
#include "../shaders/vulkan_spirv/texture_load_dxt1_emu_bc3_cs.h"
#include "../shaders/vulkan_spirv/texture_load_dxt5a_eac_r11_cs.h"
#include "../shaders/vulkan_spirv/texture_load_gbgr8_rgb8_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r10g11b11_rgba16_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r10g11b11_rgba16_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r10g11b11_rgba16_snorm_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r10g11b11_rgba16_snorm_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r11g11b10_rgba16_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r11g11b10_rgba16_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r11g11b10_rgba16_snorm_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r11g11b10_rgba16_snorm_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r16_snorm_float_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r16_snorm_float_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r16_unorm_float_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r16_unorm_float_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r4g4b4a4_a4r4g4b4_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r4g4b4a4_a4r4g4b4_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r4g4b4a4_rgba8_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r4g4b4a4_rgba8_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r5g5b5a1_b5g5r5a1_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r5g5b5a1_b5g5r5a1_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r5g5b6_b5g6r5_swizzle_rbga_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r5g5b6_b5g6r5_swizzle_rbga_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r5g6b5_b5g6r5_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r5g6b5_b5g6r5_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r5g6b5_rgba8_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r5g6b5_rgba8_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_rg16_snorm_float_cs.h"
#include "../shaders/vulkan_spirv/texture_load_rg16_snorm_float_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_rg16_unorm_float_cs.h"
#include "../shaders/vulkan_spirv/texture_load_rg16_unorm_float_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_rgba16_snorm_float_cs.h"
#include "../shaders/vulkan_spirv/texture_load_rgba16_snorm_float_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_rgba16_unorm_float_cs.h"
#include "../shaders/vulkan_spirv/texture_load_rgba16_unorm_float_scaled_cs.h"
// wwe2k14 patch (Session 7c, TEXDIRECT): from texture_load_scaled_direct.cs.glsl (see the header comments).
#include "../shaders/vulkan_spirv/texture_load_32bpb_scaled_direct_cs.h"
#include "../shaders/vulkan_spirv/texture_load_64bpb_scaled_direct_cs.h"
#include "../shaders/vulkan_spirv/texture_load_depth_float_scaled_direct_cs.h"  // Session 22, TEXDEPTH
}  // namespace shaders

namespace {

// Keep invalid fetch constant texture reads aligned with D3D12 null SRV
// descriptors, which force all sampled channels to 0.
constexpr VkFormat kInvalidTextureFetchFallbackFormat = VK_FORMAT_R8G8B8A8_UNORM;
const VkClearColorValue kInvalidTextureFetchFallbackColor = {{0.0f, 0.0f, 0.0f, 0.0f}};

}  // namespace

static_assert(VK_FORMAT_UNDEFINED == VkFormat(0),
              "Assuming that skipping a VkFormat in an initializer results in "
              "VK_FORMAT_UNDEFINED");
const VulkanTextureCache::HostFormatPair VulkanTextureCache::kBestHostFormats[64] = {
    // k_1_REVERSE
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_1
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_8
    {{kLoadShaderIndex8bpb, VK_FORMAT_R8_UNORM},
     {kLoadShaderIndex8bpb, VK_FORMAT_R8_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR,
     true},
    // k_1_5_5_5
    // Red and blue swapped in the load shader for simplicity.
    {{kLoadShaderIndexR5G5B5A1ToB5G5R5A1, VK_FORMAT_A1R5G5B5_UNORM_PACK16},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_5_6_5
    // Red and blue swapped in the load shader for simplicity.
    {{kLoadShaderIndexR5G6B5ToB5G6R5, VK_FORMAT_R5G6B5_UNORM_PACK16},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_6_5_5
    // On the host, green bits in blue, blue bits in green.
    {{kLoadShaderIndexR5G5B6ToB5G6R5WithRBGASwizzle, VK_FORMAT_R5G6B5_UNORM_PACK16},
     {kLoadShaderIndexUnknown},
     XE_GPU_MAKE_TEXTURE_SWIZZLE(R, B, G, G)},
    // k_8_8_8_8
    {{kLoadShaderIndex32bpb, VK_FORMAT_R8G8B8A8_UNORM},
     {kLoadShaderIndex32bpb, VK_FORMAT_R8G8B8A8_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
     true},
    // k_2_10_10_10
    // Signed 2_10_10_10 is unsupported on D3D12, keep Vulkan behavior aligned.
    {{kLoadShaderIndex32bpb, VK_FORMAT_A2B10G10R10_UNORM_PACK32},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_8_A
    {{kLoadShaderIndex8bpb, VK_FORMAT_R8_UNORM},
     {kLoadShaderIndex8bpb, VK_FORMAT_R8_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR,
     true},
    // k_8_B
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_8_8
    {{kLoadShaderIndex16bpb, VK_FORMAT_R8G8_UNORM},
     {kLoadShaderIndex16bpb, VK_FORMAT_R8G8_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG,
     true},
    // k_Cr_Y1_Cb_Y0_REP
    // VK_FORMAT_G8B8G8R8_422_UNORM (added in
    // VK_KHR_sampler_ycbcr_conversion and promoted to Vulkan 1.1) is
    // optional.
    // Signed variant is unsupported on D3D12.
    {{kLoadShaderIndex32bpb, VK_FORMAT_G8B8G8R8_422_UNORM, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_Y1_Cr_Y0_Cb_REP
    // VK_FORMAT_B8G8R8G8_422_UNORM (added in
    // VK_KHR_sampler_ycbcr_conversion and promoted to Vulkan 1.1) is
    // optional.
    // Signed variant is unsupported on D3D12.
    {{kLoadShaderIndex32bpb, VK_FORMAT_B8G8R8G8_422_UNORM, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_16_16_EDRAM
    // Not usable as a texture, also has -32...32 range.
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_8_8_8_8_A
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_4_4_4_4
    // Components swapped in the load shader for simplicity.
    {{kLoadShaderIndexRGBA4ToARGB4, VK_FORMAT_B4G4R4A4_UNORM_PACK16},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_10_11_11
    // Runtime fallback to float16 may be enabled if normalized RGBA16 host
    // formats are not linearly filterable.
    {{kLoadShaderIndexR11G11B10ToRGBA16, VK_FORMAT_R16G16B16A16_UNORM},
     {kLoadShaderIndexR11G11B10ToRGBA16SNorm, VK_FORMAT_R16G16B16A16_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_11_11_10
    // Runtime fallback to float16 may be enabled if normalized RGBA16 host
    // formats are not linearly filterable.
    {{kLoadShaderIndexR10G11B11ToRGBA16, VK_FORMAT_R16G16B16A16_UNORM},
     {kLoadShaderIndexR10G11B11ToRGBA16SNorm, VK_FORMAT_R16G16B16A16_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_DXT1
    // VK_FORMAT_BC1_RGBA_UNORM_BLOCK is optional.
    {{kLoadShaderIndex64bpb, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_DXT2_3
    // VK_FORMAT_BC2_UNORM_BLOCK is optional.
    {{kLoadShaderIndex128bpb, VK_FORMAT_BC2_UNORM_BLOCK, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_DXT4_5
    // VK_FORMAT_BC3_UNORM_BLOCK is optional.
    {{kLoadShaderIndex128bpb, VK_FORMAT_BC3_UNORM_BLOCK, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_16_16_16_16_EDRAM
    // Not usable as a texture, also has -32...32 range.
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_24_8
    // Keep signed and unsigned sampling on the same R32 float descriptor path
    // for parity with D3D12.
    {{kLoadShaderIndexDepthUnorm, VK_FORMAT_R32_SFLOAT},
     {kLoadShaderIndexDepthUnorm, VK_FORMAT_R32_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR,
     true},
    // k_24_8_FLOAT
    // Keep signed and unsigned sampling on the same R32 float descriptor path
    // for parity with D3D12.
    {{kLoadShaderIndexDepthFloat, VK_FORMAT_R32_SFLOAT},
     {kLoadShaderIndexDepthFloat, VK_FORMAT_R32_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR,
     true},
    // k_16
    // VK_FORMAT_R16_UNORM and VK_FORMAT_R16_SNORM are optional.
    {{kLoadShaderIndex16bpb, VK_FORMAT_R16_UNORM},
     {kLoadShaderIndex16bpb, VK_FORMAT_R16_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR,
     true},
    // k_16_16
    // VK_FORMAT_R16G16_UNORM and VK_FORMAT_R16G16_SNORM are optional.
    {{kLoadShaderIndex32bpb, VK_FORMAT_R16G16_UNORM},
     {kLoadShaderIndex32bpb, VK_FORMAT_R16G16_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG,
     true},
    // k_16_16_16_16
    // VK_FORMAT_R16G16B16A16_UNORM and VK_FORMAT_R16G16B16A16_SNORM are
    // optional.
    {{kLoadShaderIndex64bpb, VK_FORMAT_R16G16B16A16_UNORM},
     {kLoadShaderIndex64bpb, VK_FORMAT_R16G16B16A16_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
     true},
    // k_16_EXPAND
    {{kLoadShaderIndex16bpb, VK_FORMAT_R16_SFLOAT},
     {kLoadShaderIndex16bpb, VK_FORMAT_R16_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR,
     true},
    // k_16_16_EXPAND
    {{kLoadShaderIndex32bpb, VK_FORMAT_R16G16_SFLOAT},
     {kLoadShaderIndex32bpb, VK_FORMAT_R16G16_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG,
     true},
    // k_16_16_16_16_EXPAND
    {{kLoadShaderIndex64bpb, VK_FORMAT_R16G16B16A16_SFLOAT},
     {kLoadShaderIndex64bpb, VK_FORMAT_R16G16B16A16_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
     true},
    // k_16_FLOAT
    {{kLoadShaderIndex16bpb, VK_FORMAT_R16_SFLOAT},
     {kLoadShaderIndex16bpb, VK_FORMAT_R16_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR,
     true},
    // k_16_16_FLOAT
    {{kLoadShaderIndex32bpb, VK_FORMAT_R16G16_SFLOAT},
     {kLoadShaderIndex32bpb, VK_FORMAT_R16G16_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG,
     true},
    // k_16_16_16_16_FLOAT
    {{kLoadShaderIndex64bpb, VK_FORMAT_R16G16B16A16_SFLOAT},
     {kLoadShaderIndex64bpb, VK_FORMAT_R16G16B16A16_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
     true},
    // k_32
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_32_32
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_32_32_32_32
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_32_FLOAT
    {{kLoadShaderIndex32bpb, VK_FORMAT_R32_SFLOAT},
     {kLoadShaderIndex32bpb, VK_FORMAT_R32_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR,
     true},
    // k_32_32_FLOAT
    {{kLoadShaderIndex64bpb, VK_FORMAT_R32G32_SFLOAT},
     {kLoadShaderIndex64bpb, VK_FORMAT_R32G32_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG,
     true},
    // k_32_32_32_32_FLOAT
    {{kLoadShaderIndex128bpb, VK_FORMAT_R32G32B32A32_SFLOAT},
     {kLoadShaderIndex128bpb, VK_FORMAT_R32G32B32A32_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
     true},
    // k_32_AS_8
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_32_AS_8_8
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_16_MPEG
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_16_16_MPEG
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_8_INTERLACED
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_32_AS_8_INTERLACED
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_32_AS_8_8_INTERLACED
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_16_INTERLACED
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_16_MPEG_INTERLACED
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_16_16_MPEG_INTERLACED
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_DXN
    // VK_FORMAT_BC5_UNORM_BLOCK is optional.
    {{kLoadShaderIndex128bpb, VK_FORMAT_BC5_UNORM_BLOCK, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_8_8_8_8_AS_16_16_16_16
    {{kLoadShaderIndex32bpb, VK_FORMAT_R8G8B8A8_UNORM},
     {kLoadShaderIndex32bpb, VK_FORMAT_R8G8B8A8_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
     true},
    // k_DXT1_AS_16_16_16_16
    // VK_FORMAT_BC1_RGBA_UNORM_BLOCK is optional.
    {{kLoadShaderIndex64bpb, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_DXT2_3_AS_16_16_16_16
    // VK_FORMAT_BC2_UNORM_BLOCK is optional.
    {{kLoadShaderIndex128bpb, VK_FORMAT_BC2_UNORM_BLOCK, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_DXT4_5_AS_16_16_16_16
    // VK_FORMAT_BC3_UNORM_BLOCK is optional.
    {{kLoadShaderIndex128bpb, VK_FORMAT_BC3_UNORM_BLOCK, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_2_10_10_10_AS_16_16_16_16
    // Signed 2_10_10_10 is unsupported on D3D12, keep Vulkan behavior aligned.
    {{kLoadShaderIndex32bpb, VK_FORMAT_A2B10G10R10_UNORM_PACK32},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_10_11_11_AS_16_16_16_16
    // Runtime fallback to float16 may be enabled if normalized RGBA16 host
    // formats are not linearly filterable.
    {{kLoadShaderIndexR11G11B10ToRGBA16, VK_FORMAT_R16G16B16A16_UNORM},
     {kLoadShaderIndexR11G11B10ToRGBA16SNorm, VK_FORMAT_R16G16B16A16_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_11_11_10_AS_16_16_16_16
    // Runtime fallback to float16 may be enabled if normalized RGBA16 host
    // formats are not linearly filterable.
    {{kLoadShaderIndexR10G11B11ToRGBA16, VK_FORMAT_R16G16B16A16_UNORM},
     {kLoadShaderIndexR10G11B11ToRGBA16SNorm, VK_FORMAT_R16G16B16A16_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_32_32_32_FLOAT
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_DXT3A
    {{kLoadShaderIndexDXT3A, VK_FORMAT_R8_UNORM},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_DXT5A
    // VK_FORMAT_BC4_UNORM_BLOCK is optional.
    {{kLoadShaderIndex64bpb, VK_FORMAT_BC4_UNORM_BLOCK, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_CTX1
    {{kLoadShaderIndexCTX1, VK_FORMAT_R8G8_UNORM},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_DXT3A_AS_1_1_1_1
    {{kLoadShaderIndexDXT3AAs1111ToARGB4, VK_FORMAT_B4G4R4A4_UNORM_PACK16},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_8_8_8_8_GAMMA_EDRAM
    // Not usable as a texture.
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_2_10_10_10_FLOAT_EDRAM
    // Not usable as a texture.
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
};

// Vulkan requires 2x1 (4:2:2) subsampled images to have an even width.
// Always decompressing them to RGBA8 UNORM for D3D12 parity.

const VulkanTextureCache::HostFormatPair VulkanTextureCache::kHostFormatGBGRUnaligned = {
    {kLoadShaderIndexGBGR8ToRGB8, VK_FORMAT_R8G8B8A8_UNORM, false, true},
    {kLoadShaderIndexUnknown},
    xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB,
    false};

const VulkanTextureCache::HostFormatPair VulkanTextureCache::kHostFormatBGRGUnaligned = {
    {kLoadShaderIndexBGRG8ToRGB8, VK_FORMAT_R8G8B8A8_UNORM, false, true},
    {kLoadShaderIndexUnknown},
    xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB,
    false};

// Keep BC unaligned fallback mapping compatible with the D3D12 texture cache
// decompression mapping.
const VulkanTextureCache::HostFormatPair VulkanTextureCache::kHostFormatDXT1Unaligned = {
    {kLoadShaderIndexDXT1ToRGBA8, VK_FORMAT_R8G8B8A8_UNORM, false, true},
    {kLoadShaderIndexUnknown},
    xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
    false};

const VulkanTextureCache::HostFormatPair VulkanTextureCache::kHostFormatDXT2_3Unaligned = {
    {kLoadShaderIndexDXT3ToRGBA8, VK_FORMAT_R8G8B8A8_UNORM, false, true},
    {kLoadShaderIndexUnknown},
    xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
    false};

const VulkanTextureCache::HostFormatPair VulkanTextureCache::kHostFormatDXT4_5Unaligned = {
    {kLoadShaderIndexDXT5ToRGBA8, VK_FORMAT_R8G8B8A8_UNORM, false, true},
    {kLoadShaderIndexUnknown},
    xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
    false};

const VulkanTextureCache::HostFormatPair VulkanTextureCache::kHostFormatDXNUnaligned = {
    {kLoadShaderIndexDXNToRG8, VK_FORMAT_R8G8_UNORM, false, true},
    {kLoadShaderIndexUnknown},
    xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG,
    false};

const VulkanTextureCache::HostFormatPair VulkanTextureCache::kHostFormatDXT5AUnaligned = {
    {kLoadShaderIndexDXT5AToR8, VK_FORMAT_R8_UNORM, false, true},
    {kLoadShaderIndexUnknown},
    xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR,
    false};

VulkanTextureCache::~VulkanTextureCache() {
  // wwe2k14 patch (Session 7c, TEXLOAD): a capture still running.
  if (texload_csv_) {
    TexLoadCsvFlush();
    std::fclose(texload_csv_);
    texload_csv_ = nullptr;
  }

  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  for (const std::pair<const SamplerParameters, Sampler>& sampler_pair : samplers_) {
    dfn.vkDestroySampler(device, sampler_pair.second.sampler, nullptr);
  }
  samplers_.clear();
  custom_border_color_sampler_count_ = 0;
  COUNT_profile_set("gpu/texture_cache/vulkan/samplers", 0);
  sampler_used_last_ = nullptr;
  sampler_used_first_ = nullptr;

  ShutdownScaledResolveBuffer();

  if (null_image_view_3d_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImageView(device, null_image_view_3d_, nullptr);
  }
  if (null_image_view_cube_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImageView(device, null_image_view_cube_, nullptr);
  }
  if (null_image_view_2d_array_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImageView(device, null_image_view_2d_array_, nullptr);
  }
  if (null_image_3d_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImage(device, null_image_3d_, nullptr);
  }
  if (null_image_2d_array_cube_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImage(device, null_image_2d_array_cube_, nullptr);
  }
  for (VkDeviceMemory null_images_memory : null_images_memory_) {
    if (null_images_memory != VK_NULL_HANDLE) {
      dfn.vkFreeMemory(device, null_images_memory, nullptr);
    }
  }
  for (VkPipeline load_pipeline : load_pipelines_scaled_) {
    if (load_pipeline != VK_NULL_HANDLE) {
      dfn.vkDestroyPipeline(device, load_pipeline, nullptr);
    }
  }
  for (VkPipeline load_pipeline : load_pipelines_) {
    if (load_pipeline != VK_NULL_HANDLE) {
      dfn.vkDestroyPipeline(device, load_pipeline, nullptr);
    }
  }
  if (load_pipeline_layout_ != VK_NULL_HANDLE) {
    dfn.vkDestroyPipelineLayout(device, load_pipeline_layout_, nullptr);
  }

  // Textures memory is allocated using the Vulkan Memory Allocator, destroy all
  // textures before destroying VMA.
  DestroyAllTextures(true);

  // wwe2k14 patch (Session 7c, TEXDIRECT): after the textures, which return their storage descriptors to the pool.
  ShutdownDirectLoad();

  if (vma_allocator_ != VK_NULL_HANDLE) {
    vmaDestroyAllocator(vma_allocator_);
  }
}

void VulkanTextureCache::BeginSubmission(uint64_t new_submission_index) {
  TextureCache::BeginSubmission(new_submission_index);

  if (!null_images_cleared_ && null_image_2d_array_cube_ != VK_NULL_HANDLE &&
      null_image_3d_ != VK_NULL_HANDLE) {
    VkImage null_images[] = {null_image_2d_array_cube_, null_image_3d_};
    VkImageSubresourceRange null_image_subresource_range(
        ui::vulkan::util::InitializeSubresourceRange());
    for (size_t i = 0; i < rex::countof(null_images); ++i) {
      command_processor_.PushImageMemoryBarrier(
          null_images[i], null_image_subresource_range, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
          VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
          false);
    }
    command_processor_.SubmitBarriers(true);
    DeferredCommandBuffer& command_buffer = command_processor_.deferred_command_buffer();
    VkClearColorValue null_image_clear_color = kInvalidTextureFetchFallbackColor;
    for (size_t i = 0; i < rex::countof(null_images); ++i) {
      command_buffer.CmdVkClearColorImage(null_images[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                          &null_image_clear_color, 1,
                                          &null_image_subresource_range);
    }
    for (size_t i = 0; i < rex::countof(null_images); ++i) {
      command_processor_.PushImageMemoryBarrier(
          null_images[i], null_image_subresource_range, VK_PIPELINE_STAGE_TRANSFER_BIT,
          guest_shader_pipeline_stages_, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
          VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
    }
    null_images_cleared_ = true;
  }
}

void VulkanTextureCache::BeginFrame() {
  // wwe2k14 patch (Session 56): the written-textures list only while vulkan_texture_sample_written_early is on.
  track_written_ = REXCVAR_GET(vulkan_texture_sample_written_early);
  if (!track_written_ && !written_textures_.empty()) {
    for (VulkanTexture* texture : written_textures_) texture->in_written_list = false;
    written_textures_.clear();
  }
  TextureCache::BeginFrame();

  std::memset(unsupported_format_features_used_, 0, sizeof(unsupported_format_features_used_));

  // wwe2k14 patch (Session 7c, TEXLOAD): the scratch ring's memory goes once vulkan_texture_load_scratch_ring is off
  // (each buffer after its last submission has completed). Nothing to do when it was never on.
  if (!REXCVAR_GET(vulkan_texture_load_scratch_ring)) {
    command_processor_.ReleaseScratchRing();
  }
}

void VulkanTextureCache::EndFrame() {
  // wwe2k14 patch (Session 93): transcoded block counters, every 600 frames while anything was transcoded.
  if (++g_transcode_frames >= 600) {
    g_transcode_frames = 0;
    if (g_transcode_blocks[0] || g_transcode_blocks[1] || g_transcode_blocks[2]) {
      REXGPU_INFO(
          "[tex][transcode] last 600 frames: blocks DXT1->ASTC {} ({} loads), DXT5A->EAC_R11 {} ({} loads), "
          "DXN->EAC_RG11 {} ({} loads)",
          g_transcode_blocks[0], g_transcode_loads[0], g_transcode_blocks[1], g_transcode_loads[1],
          g_transcode_blocks[2], g_transcode_loads[2]);
      for (int i = 0; i < 3; ++i) {
        g_transcode_blocks[i] = 0;
        g_transcode_loads[i] = 0;
      }
    }
  }
  // wwe2k14 patch (Session 6, CP6): periodic sampler cache statistics.
  if (REXCVAR_GET(vulkan_sampler_param_cache) || REXCVAR_GET(vulkan_sampler_param_cache_verify)) {
    ReportSamplerCacheStats();
  }

  bool unsupported_header_written = false;
  for (uint32_t i = 0; i < 64; ++i) {
    uint32_t unsupported_features = unsupported_format_features_used_[i];
    if (!unsupported_features) {
      continue;
    }
    if (!unsupported_header_written) {
      REXGPU_ERROR("Unsupported texture formats used in the frame:");
      unsupported_header_written = true;
    }
    REXGPU_ERROR("* {}{}{}{}", FormatInfo::Get(xenos::TextureFormat(i))->name,
                 unsupported_features & kUnsupportedResourceBit ? " resource" : "",
                 unsupported_features & kUnsupportedUnormBit ? " unsigned" : "",
                 unsupported_features & kUnsupportedSnormBit ? " signed" : "");
    unsupported_format_features_used_[i] = 0;
  }
}

namespace {
// wwe2k14 patch (Session 95, r6tex, gpu_tex_bind_memo bit 4): the OwnedView memo entry of each slot's owned binding, under the binding's generation. Command thread only.
struct SlotRead {
  uint64_t generation = UINT64_MAX;
  uint32_t memo = UINT32_MAX;
};
SlotRead g_slot_read[32];
}  // namespace

void VulkanTextureCache::RequestTextures(uint32_t used_texture_mask) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES
  native_last_request_served_ = false;  // wwe2k14 patch (Session 80; set after the binding loop)
  const bool stat_on = g_tex_stat_on;  // wwe2k14 patch (Session 95, r6tex)
  uint64_t stat_t = stat_on ? TexStatTicks() : 0;
  const auto stat_lap = [&](uint32_t slot) {
    if (!stat_on) return;
    const uint64_t now = TexStatTicks();
    g_tex_stat[slot] += now - stat_t;
    stat_t = now;
  };
  if (stat_on) ++g_tex_stat[kTsCalls];

  // wwe2k14 patch (Session 22, EAGER): vulkan_texture_load_eager. A texture made from resolved data is loaded at
  // its first use after the resolve - often after a long run of draws (Bryan's pan: 22 loads per frame each after
  // ~68 crowd draws), so the load's barrier waits for all those draws, and a second barrier makes the image
  // readable right before the draw. Here, a texture that the previous frame loaded late (2+ draws after the
  // frame's k-th resolve, with no resolve in between) is loaded at the first draw after this frame's k-th resolve
  // instead, where the draw needs a barrier anyway (EDRAM) and only the resolve is in flight; its image is made
  // readable in that draw's barrier. Why the contents stay the same:
  // - It is the same load (LoadTextureData: the source ranges requested, the same load code with its barriers:
  //   resolve writes -> read, earlier sampling of the image -> write). It only runs if the texture is outdated,
  //   i.e. its memory was written since its last load, and it reads what the memory holds at this point.
  // - The load re-arms the texture's memory watch (MakeUpToDateAndWatch). If anything writes the memory again
  //   before the texture's use (another resolve, the CPU), the watch marks it outdated and the draw that samples it
  //   loads it again, as without this setting. So every draw samples the latest memory, as before.
  // - The image is transitioned to the sampled usage right after (compute write -> shader read, the barrier
  //   RequestTextures records for a loaded texture), recorded before any draw that can sample it.
  // Learning a late load changes no command; off = no early load.
  {
    const uint64_t frame = command_processor_.GetCurrentFrame();
    if (frame != eager_frame_) {
      EagerRotate(frame);
    }
    if (!eager_pending_.empty()) {
      if (REXCVAR_GET(vulkan_texture_load_eager)) {
        VkPipelineStageFlags sampled_stage_mask;
        VkAccessFlags sampled_access_mask;
        VkImageLayout sampled_layout;
        GetTextureUsageMasks(VulkanTexture::Usage::kGuestShaderSampled, sampled_stage_mask,
                             sampled_access_mask, sampled_layout);
        for (VulkanTexture* texture : eager_pending_) {
          if (!texture->outdated_mask()) {
            ++eager_stats_.already_valid;
            continue;
          }
          eager_loading_ = true;
          const bool loaded = LoadTextureData(*texture);
          eager_loading_ = false;
          if (!loaded || texture->outdated_mask()) {
            ++eager_stats_.failed;
            continue;
          }
          texture->SetEagerLoadedFrame(frame);
          ++eager_stats_.loads;
          texture->native_used = true;  // wwe2k14 patch (Session 80)
          VulkanTexture::Usage old_usage = texture->SetUsage(VulkanTexture::Usage::kGuestShaderSampled);
          // wwe2k14 patch (Session 52): a deferred native resolve may still have to write it.
          if (old_usage == VulkanTexture::Usage::kComputeStorageWrite) command_processor_.NativeFlushDeferredResolves(texture);
          if (old_usage != VulkanTexture::Usage::kGuestShaderSampled) {
            VkPipelineStageFlags src_stage_mask;
            VkAccessFlags src_access_mask;
            VkImageLayout old_layout;
            GetTextureUsageMasks(old_usage, src_stage_mask, src_access_mask, old_layout);
            command_processor_.PushImageMemoryBarrier(texture->image(),
                                                      ui::vulkan::util::InitializeSubresourceRange(),
                                                      src_stage_mask, sampled_stage_mask, src_access_mask,
                                                      sampled_access_mask, old_layout, sampled_layout);
          }
        }
      }
      eager_pending_.clear();
    }
    ++eager_draws_since_resolve_;
    // wwe2k14 patch (Session 87, step 3): vulkan_texture_load_early_cpu - the planned textures, at the frame's first draw.
    if (!cpu_pending_.empty()) {
      if (REXCVAR_GET(vulkan_texture_load_early_cpu)) {
        VkPipelineStageFlags sampled_stage_mask;
        VkAccessFlags sampled_access_mask;
        VkImageLayout sampled_layout;
        GetTextureUsageMasks(VulkanTexture::Usage::kGuestShaderSampled, sampled_stage_mask, sampled_access_mask,
                             sampled_layout);
        for (VulkanTexture* texture : cpu_pending_) {
          if (!texture->outdated_mask()) {
            ++cpu_early_stats_.already_valid;
            continue;
          }
          cpu_loading_ = true;
          const bool loaded = LoadTextureData(*texture);
          cpu_loading_ = false;
          if (!loaded || texture->outdated_mask()) {
            ++cpu_early_stats_.failed;
            continue;
          }
          texture->SetEagerLoadedFrame(frame);
          ++cpu_early_stats_.loads;
          texture->native_used = true;
          VulkanTexture::Usage old_usage = texture->SetUsage(VulkanTexture::Usage::kGuestShaderSampled);
          if (old_usage == VulkanTexture::Usage::kComputeStorageWrite) command_processor_.NativeFlushDeferredResolves(texture);
          if (old_usage != VulkanTexture::Usage::kGuestShaderSampled) {
            VkPipelineStageFlags src_stage_mask;
            VkAccessFlags src_access_mask;
            VkImageLayout old_layout;
            GetTextureUsageMasks(old_usage, src_stage_mask, src_access_mask, old_layout);
            command_processor_.PushImageMemoryBarrier(texture->image(), ui::vulkan::util::InitializeSubresourceRange(),
                                                      src_stage_mask, sampled_stage_mask, src_access_mask,
                                                      sampled_access_mask, old_layout, sampled_layout);
          }
        }
      }
      cpu_pending_.clear();
    }
  }

  stat_lap(kTsPreTicks);
  TextureCache::RequestTextures(used_texture_mask);
  stat_lap(kTsBaseTicks);

  // wwe2k14 patch (Session 87, step 3): vulkan_texture_load_early_cpu - which planned textures this draw samples. Run 010 loaded 21
  // textures a frame in Brock's entrance against 12 without the setting: a planned texture stayed planned for good, loaded every
  // time the CPU touched it, sampled or not. Only a planned texture a draw sampled is planned again for the next frame.
  if (!cpu_plan_next_.empty() && REXCVAR_GET(vulkan_texture_load_early_cpu)) {
    uint32_t used_mask = used_texture_mask;
    uint32_t used_index;
    while (bit_inline::bit_scan_forward(used_mask, &used_index)) {
      used_mask &= ~(uint32_t(1) << used_index);
      const TextureBinding* used_binding = GetValidTextureBinding(used_index);
      if (!used_binding || !used_binding->texture) continue;
      VulkanTexture* used_texture = static_cast<VulkanTexture*>(used_binding->texture);
      if (std::find(cpu_plan_next_.begin(), cpu_plan_next_.end(), used_texture) != cpu_plan_next_.end() &&
          std::find(cpu_used_.begin(), cpu_used_.end(), used_texture) == cpu_used_.end()) {
        cpu_used_.push_back(used_texture);
      }
    }
  }

  // wwe2k14 patch (Session 79, diagnostic): vulkan_texture_sampled_seg_log.
  if (REXCVAR_GET(vulkan_texture_sampled_seg_log) && command_processor_.NativeSegmentLogActive()) {
    uint32_t log_mask = used_texture_mask;
    uint32_t log_index;
    while (bit_inline::bit_scan_forward(log_mask, &log_index)) {
      log_mask &= ~(uint32_t(1) << log_index);
      const TextureBinding* log_binding = GetValidTextureBinding(log_index);
      if (!log_binding) continue;
      REXGPU_INFO("[native][seg] texture sampled: page 0x{:05X} format {} {}x{} fetch {}",
                  uint32_t(log_binding->key.base_page), uint32_t(log_binding->key.format),
                  log_binding->key.GetWidth(), log_binding->key.GetHeight(), log_index);
    }
  }

  // wwe2k14 patch (Session 66, G1.0 resolve census, gpu_native_alias_census): the textures this draw samples (counter
  // only; before the skip below, which would leave a repeated binding uncounted).
  if (command_processor_.NativeAliasCensusActive()) {
    uint32_t census_mask = used_texture_mask;
    uint32_t census_index;
    while (bit_inline::bit_scan_forward(census_mask, &census_index)) {
      census_mask &= ~(uint32_t(1) << census_index);
      const TextureBinding* census_binding = GetValidTextureBinding(census_index);
      if (!census_binding) {
        continue;
      }
      if (census_binding->texture) {
        command_processor_.NativeAliasSampled(static_cast<VulkanTexture*>(census_binding->texture));
      }
      if (census_binding->texture_signed) {
        command_processor_.NativeAliasSampled(static_cast<VulkanTexture*>(census_binding->texture_signed));
      }
    }
  }

  // wwe2k14 patch (Session 7, TEXREQ): the loops below create 3D-as-2D views for 3D bindings, mark the used
  // textures as used in the current submission (LRU) and transition them to the guest shader sampled usage.
  // They would do nothing when, since they last ran in full: the used fetch constant mask is the same and
  // texture_bindings_ hasn't changed (active_texture_bindings_version - so the same bindings with the same
  // textures; textures are only destroyed after ResetTextureBindings bumps it), none of the used bindings is
  // 3D, the submission is the same (MarkAsUsed returns early for a texture already marked in the current
  // submission), and no texture's usage has changed (texture_usage_changes_: every used texture is still in
  // the sampled usage, so no barrier). A new submission or frame, a binding change, a texture load
  // (kTransferDestination) or the swap texture (kSwapSampled) makes them run again.
  bool loops_would_be_skipped = false;
  if (request_textures_skip_unchanged() || request_textures_skip_verify()) {
    const RequestTexturesLoopState& loop_state = request_textures_loop_state_;
    loops_would_be_skipped = loop_state.valid && loop_state.used_texture_mask == used_texture_mask &&
                             loop_state.bindings_version == active_texture_bindings_version() &&
                             loop_state.submission == current_submission_index() &&
                             loop_state.usage_changes == texture_usage_changes_ &&
                             !(used_texture_mask & texture_bindings_3d_mask_) &&
                             !(used_texture_mask & native_served_mask_) &&  // (Session 80: served bindings re-checked)
                             !(used_texture_mask & native_owned_mask_);     // (and owned ones: gpu_native_own_steps)
    CountRequestTexturesBackendPart(loops_would_be_skipped);
    if (loops_would_be_skipped && request_textures_skip_unchanged()) {
      native_last_request_served_ = false;  // (Session 80: a skip needs no served used binding)
      if (stat_on) ++g_tex_stat[kTsLoopSkips];
      stat_lap(kTsLoopTicks);
      return;
    }
  }
  // _verify: whether running the loops in full changed anything where they would have been skipped.
  const char* loops_verify_detail = nullptr;
  uint32_t loops_verify_fetch_constant = 0;

  // Pre-create 3D-as-2D wrappers while still in texture request phase.
  if (REXCVAR_GET(gpu_3d_to_2d_texture)) {
    // (Session 95, r6tex, gpu_tex_bind_memo bit 2: UpdateTextureBindingsImpl keeps the mask of the 3D bindings; the loop acts on those only.)
    uint32_t textures_3d = (REXCVAR_GET(gpu_tex_bind_memo) & 2) ? (used_texture_mask & texture_bindings_3d_mask_) : used_texture_mask;
    uint32_t index_3d;
    while (bit_inline::bit_scan_forward(textures_3d, &index_3d)) {
      textures_3d &= ~(uint32_t(1) << index_3d);
      const TextureBinding* binding = GetValidTextureBinding(index_3d);
      if (!binding || binding->key.dimension != xenos::DataDimension::k3D) {
        continue;
      }
      VulkanTexture* texture = static_cast<VulkanTexture*>(binding->texture);
      if (texture) {
        texture->GetOrCreate3DAs2DImageView(false, binding->host_swizzle);
      }
      VulkanTexture* texture_signed = static_cast<VulkanTexture*>(binding->texture_signed);
      if (texture_signed) {
        texture_signed->GetOrCreate3DAs2DImageView(true, binding->host_swizzle);
      }
      // wwe2k14 patch (Session 7, TEXREQ): _verify - a skip requires no used 3D binding.
      if (loops_would_be_skipped && !loops_verify_detail) {
        loops_verify_detail = "found a used 3D binding";
        loops_verify_fetch_constant = index_3d;
      }
    }
  }

  // Transition the textures into the needed usage.
  VkPipelineStageFlags dst_stage_mask;
  VkAccessFlags dst_access_mask;
  VkImageLayout new_layout;
  GetTextureUsageMasks(VulkanTexture::Usage::kGuestShaderSampled, dst_stage_mask, dst_access_mask,
                       new_layout);
  // wwe2k14 patch (Session 95, r6tex): not with the skip's own verify, which watches what these loops change.
  const bool slot_fast = (REXCVAR_GET(gpu_tex_bind_memo) & 2) != 0 && !request_textures_skip_verify();
  const bool slot_verify = slot_fast && REXCVAR_GET(gpu_tex_bind_memo_verify);
  const bool slot_again = (REXCVAR_GET(gpu_tex_bind_memo) & 4) != 0 && !request_textures_skip_verify() && !REXCVAR_GET(gpu_tex_bind_memo_verify);
  uint32_t textures_remaining = used_texture_mask;
  uint32_t index;
  while (bit_inline::bit_scan_forward(textures_remaining, &index)) {
    textures_remaining &= ~(uint32_t(1) << index);
    const TextureBinding* binding = GetValidTextureBinding(index);
    if (!binding) {
      continue;
    }
    if (stat_on) ++g_tex_stat[kTsLoopSlots];
    VulkanTexture* binding_texture = static_cast<VulkanTexture*>(binding->texture);
    // wwe2k14 patch (Session 95, r6tex, gpu_tex_bind_memo bit 2): the common slot, a texture of its own in the sampled usage on a page no owned page can hold: both blocks below give nothing for it
    // (OwnedView's answer is no view with its side effects run by NativeOwnedPageFree, a served view needs a texture in the compute-write usage, neither mask has it), so what is left is the texture's usage.
    // (Bit 4: a slot that is ours and whose binding has not moved since (its generation counts every change of key, textures, swizzle and views): the same page, read the same way, so the same view if the
    // memo still says so. One view only: a signed one takes the way below.)
    if (slot_again && (native_owned_mask_ & (uint32_t(1) << index)) && g_slot_read[index].generation == texture_binding_generations_[index] &&
        vulkan_texture_bindings_[index].image_view_signed == VK_NULL_HANDLE &&
        command_processor_.NativeOwnedReadAgain(g_slot_read[index].memo, binding->key.base_page, vulkan_texture_bindings_[index].image_view_unsigned)) {
      if (stat_on) ++g_tex_stat[kTsSlotAgain];
      continue;
    }
    bool slot_page_free = false;
    if (slot_fast && binding_texture && !binding->texture_signed && !((native_owned_mask_ | native_served_mask_) & (uint32_t(1) << index)) &&
        binding_texture->usage() == VulkanTexture::Usage::kGuestShaderSampled &&
        command_processor_.NativeOwnedPageFree(binding->key.base_page, binding->key.GetWidth(), binding->key.GetHeight(), uint32_t(binding->key.format))) {
      slot_page_free = true;
      if (!slot_verify) {
        if (stat_on) ++g_tex_stat[kTsFreeSlots];
        binding_texture->MarkAsUsed();
        binding_texture->native_used = true;
        continue;
      }
    }
    // wwe2k14 patch (Session 80, gpu_native_truth_serve_copies): a texture a pending native copy is still to write,
    // sampled from the image that copy reads instead (the same texels): it stays in its compute-write usage (nothing
    // flushed, no barrier) and the binding takes that image's view while it is served.
    // wwe2k14 patch (Session 80, gpu_native_own_steps): a page an owned step's image holds (its copy was left out:
    // the texture's memory and image are stale) - that image's view, whatever the texture's usage; nothing of the
    // texture's own is touched (no usage change, no barrier: our image is readable since its pass ended).
    // (Session 80, owned 16-bit maps: its unsigned and signed views both ours, as the fetch's signs use them - the
    // signed one too when it is a separate texture; a binding of an owned page that cannot be served - 3D, an array,
    // mips - is refused there: its producers are not owned any more.)
    {
      const uint32_t bit = uint32_t(1) << index;
      VulkanTextureBinding& vb = vulkan_texture_bindings_[index];
      VkImageView owned = VK_NULL_HANDLE, owned_signed = VK_NULL_HANDLE;
      // (Session 82, P2, G4: mip levels are the substitution's to judge - a feedback texture of ours has them.)
      const bool is_cube = binding->key.dimension == xenos::DataDimension::kCube;
      const bool servable = (binding->key.dimension == xenos::DataDimension::k2DOrStacked &&
                             !binding->key.depth_or_array_size_minus_1) ||
                            is_cube;
      const bool has_mips = binding->key.mip_max_level != 0;
      const bool uses_unsigned = texture_util::IsAnySignNotSigned(binding->swizzled_signs);
      const bool uses_signed = texture_util::IsAnySignSigned(binding->swizzled_signs);
      uint32_t swizzle = binding->host_swizzle;
      if (!command_processor_.GetVulkanDevice()->properties().imageViewFormatSwizzle) {
        swizzle = xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA;
      }
      const VkComponentMapping components = {
          VulkanTexture::NativeComponentSwizzle(swizzle, 0), VulkanTexture::NativeComponentSwizzle(swizzle, 1),
          VulkanTexture::NativeComponentSwizzle(swizzle, 2), VulkanTexture::NativeComponentSwizzle(swizzle, 3)};
      // (Session 80, 1440p: the scaled-resolve flag is the texture's - FindOrCreateTexture sets it on the texture's key
      // from the range's marks; the binding's key, from the fetch constant, never has it: at a resolution scale no page
      // was ever given our image - run 5688. As the shader's textures_resolution_scaled constant takes it: the unsigned
      // or the separate signed texture - review: a signed-only binding has no unsigned one.)
      const bool texture_scaled = IsActiveTextureResolutionScaled(index);
      const auto view = [&](bool is_signed) {
        return command_processor_.NativeOwnedView(
            binding->key.base_page, binding->key.GetWidth(), binding->key.GetHeight(), uint32_t(binding->key.format),
            uint32_t(binding->key.endianness), texture_scaled == IsDrawResolutionScaled(), components, is_signed,
            servable, has_mips, is_cube);
      };
      const uint64_t stat_v = stat_on ? TexStatTicks() : 0;
      if (uses_unsigned || !uses_signed) owned = view(false);
      if (uses_signed && (owned != VK_NULL_HANDLE || !uses_unsigned)) owned_signed = view(true);
      if (slot_page_free && (owned != VK_NULL_HANDLE || owned_signed != VK_NULL_HANDLE) && ++g_tex_bind_mismatches <= 16) {
        REXGPU_WARN("[tex] gpu_tex_bind_memo verify: page 0x{:05X} was said to be free and has a view", uint32_t(binding->key.base_page));
      }
      if (stat_on) {
        g_tex_stat[kTsViewTicks] += TexStatTicks() - stat_v;
        ++g_tex_stat[kTsViewCalls];
        if (owned != VK_NULL_HANDLE || owned_signed != VK_NULL_HANDLE) ++g_tex_stat[kTsViewServed];
      }
      if (uses_signed && (owned != VK_NULL_HANDLE || owned_signed != VK_NULL_HANDLE)) {  // (Session 90, Q diagnosis: a mixed / signed fetch of an owned page)
        static uint32_t signed_logged = 0;
        if (signed_logged < 60) {
          ++signed_logged;
          REXGPU_INFO("[native][ownedsign] page 0x{:05X} slot {} signs 0x{:08X} unsigned-use {} signed-use {} views u {} s {} swizzle 0x{:X} format {}",
                      binding->key.base_page, index, uint32_t(binding->swizzled_signs), uses_unsigned, uses_signed, owned != VK_NULL_HANDLE,
                      owned_signed != VK_NULL_HANDLE, binding->host_swizzle, uint32_t(binding->key.format));
        }
      }
      if ((owned != VK_NULL_HANDLE || !uses_unsigned) && (owned_signed != VK_NULL_HANDLE || !uses_signed) &&
          (owned != VK_NULL_HANDLE || owned_signed != VK_NULL_HANDLE)) {
        if (vb.image_view_unsigned != owned || vb.image_view_signed != owned_signed) {
          vb.image_view_unsigned = owned;
          vb.image_view_signed = owned_signed;
          ++texture_binding_generations_[index];
        }
        native_owned_mask_ |= bit;
        g_slot_read[index] = {texture_binding_generations_[index], command_processor_.NativeOwnedViewMemoLast()};  // (Session 95, r6tex)
        continue;
      }
      if (native_owned_mask_ & bit) {  // (ours before, not now: the texture's own views again)
        native_owned_mask_ &= ~bit;
        // (As UpdateTextureBindingsImpl: only views of formats the host has - review.)
        const HostFormatPair& host_format_pair = GetHostFormatPair(binding->key);
        vb.image_view_unsigned =
            binding_texture && uses_unsigned && host_format_pair.format_unsigned.format != VK_FORMAT_UNDEFINED
                ? binding_texture->GetView(false, binding->host_swizzle)
                : VK_NULL_HANDLE;
        VulkanTexture* signed_texture = IsSignedVersionSeparateForFormat(binding->key)
                                            ? static_cast<VulkanTexture*>(binding->texture_signed)
                                            : binding_texture;
        vb.image_view_signed =
            signed_texture && uses_signed && host_format_pair.format_signed.format != VK_FORMAT_UNDEFINED
                ? signed_texture->GetView(true, binding->host_swizzle)
                : VK_NULL_HANDLE;
        ++texture_binding_generations_[index];
      }
    }
    {
      const uint32_t bit = uint32_t(1) << index;
      VulkanTextureBinding& vb = vulkan_texture_bindings_[index];
      VkImageView served = VK_NULL_HANDLE;
      if (binding_texture && !binding->texture_signed &&
          binding_texture->usage() == VulkanTexture::Usage::kComputeStorageWrite &&
          binding->key.dimension == xenos::DataDimension::k2DOrStacked && !binding->key.depth_or_array_size_minus_1 &&
          !binding->key.mip_max_level && !texture_util::IsAnySignSigned(binding->swizzled_signs)) {
        uint32_t swizzle = binding->host_swizzle;
        if (!command_processor_.GetVulkanDevice()->properties().imageViewFormatSwizzle) {
          swizzle = xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA;
        }
        const VkComponentMapping components = {
            VulkanTexture::NativeComponentSwizzle(swizzle, 0), VulkanTexture::NativeComponentSwizzle(swizzle, 1),
            VulkanTexture::NativeComponentSwizzle(swizzle, 2), VulkanTexture::NativeComponentSwizzle(swizzle, 3)};
        served = command_processor_.NativeServeView(binding_texture, binding->key.GetWidth(),
                                                    binding->key.GetHeight(),
                                                    GetHostFormatPair(binding->key).format_unsigned.format, components);
      }
      if (served != VK_NULL_HANDLE) {
        if (vb.image_view_unsigned != served) {
          vb.image_view_unsigned = served;
          ++texture_binding_generations_[index];
        }
        native_served_mask_ |= bit;
        continue;
      }
      if (native_served_mask_ & bit) {  // (served before, not now: the texture's own view again)
        native_served_mask_ &= ~bit;
        vb.image_view_unsigned = binding_texture ? binding_texture->GetView(false, binding->host_swizzle) : VK_NULL_HANDLE;
        ++texture_binding_generations_[index];
      }
    }
    if (binding_texture != nullptr) {
      // wwe2k14 patch (Session 7, TEXREQ): _verify.
      uint64_t last_usage_submission =
          loops_would_be_skipped ? binding_texture->last_usage_submission_index() : 0;
      // Will be referenced by the command buffer, so mark as used.
      binding_texture->MarkAsUsed();
      binding_texture->native_used = true;  // wwe2k14 patch (Session 80)
      VulkanTexture::Usage old_usage =
          binding_texture->SetUsage(VulkanTexture::Usage::kGuestShaderSampled);
      // wwe2k14 patch (Session 52): a deferred native resolve may still have to write it.
      if (old_usage == VulkanTexture::Usage::kComputeStorageWrite) command_processor_.NativeFlushDeferredResolves(binding_texture);
      if (loops_would_be_skipped && !loops_verify_detail) {
        if (binding_texture->last_usage_submission_index() != last_usage_submission) {
          loops_verify_detail = "marked a texture as used in the submission";
          loops_verify_fetch_constant = index;
        } else if (old_usage != VulkanTexture::Usage::kGuestShaderSampled) {
          loops_verify_detail = "transitioned a texture to the sampled usage";
          loops_verify_fetch_constant = index;
        }
      }
      if (old_usage != VulkanTexture::Usage::kGuestShaderSampled) {
        VkPipelineStageFlags src_stage_mask;
        VkAccessFlags src_access_mask;
        VkImageLayout old_layout;
        GetTextureUsageMasks(old_usage, src_stage_mask, src_access_mask, old_layout);
        command_processor_.PushImageMemoryBarrier(binding_texture->image(),
                                                  ui::vulkan::util::InitializeSubresourceRange(),
                                                  src_stage_mask, dst_stage_mask, src_access_mask,
                                                  dst_access_mask, old_layout, new_layout);
      }
    }
    VulkanTexture* binding_texture_signed = static_cast<VulkanTexture*>(binding->texture_signed);
    if (binding_texture_signed != nullptr) {
      // wwe2k14 patch (Session 7, TEXREQ): _verify.
      uint64_t last_usage_submission =
          loops_would_be_skipped ? binding_texture_signed->last_usage_submission_index() : 0;
      binding_texture_signed->MarkAsUsed();
      binding_texture_signed->native_used = true;  // wwe2k14 patch (Session 80)
      VulkanTexture::Usage old_usage =
          binding_texture_signed->SetUsage(VulkanTexture::Usage::kGuestShaderSampled);
      if (old_usage == VulkanTexture::Usage::kComputeStorageWrite) command_processor_.NativeFlushDeferredResolves(binding_texture_signed);
      if (loops_would_be_skipped && !loops_verify_detail) {
        if (binding_texture_signed->last_usage_submission_index() != last_usage_submission) {
          loops_verify_detail = "marked a signed texture as used in the submission";
          loops_verify_fetch_constant = index;
        } else if (old_usage != VulkanTexture::Usage::kGuestShaderSampled) {
          loops_verify_detail = "transitioned a signed texture to the sampled usage";
          loops_verify_fetch_constant = index;
        }
      }
      if (old_usage != VulkanTexture::Usage::kGuestShaderSampled) {
        VkPipelineStageFlags src_stage_mask;
        VkAccessFlags src_access_mask;
        VkImageLayout old_layout;
        GetTextureUsageMasks(old_usage, src_stage_mask, src_access_mask, old_layout);
        command_processor_.PushImageMemoryBarrier(binding_texture_signed->image(),
                                                  ui::vulkan::util::InitializeSubresourceRange(),
                                                  src_stage_mask, dst_stage_mask, src_access_mask,
                                                  dst_access_mask, old_layout, new_layout);
      }
    }
  }
  native_last_request_served_ = (native_served_mask_ & used_texture_mask) != 0;  // wwe2k14 patch (Session 80)
  stat_lap(kTsLoopTicks);

  // wwe2k14 patch (Session 7, TEXREQ): the state these loops ran in (whatever the settings, so they can be
  // flipped at any time), and the _verify result.
  RequestTexturesLoopState& loop_state = request_textures_loop_state_;
  loop_state.valid = true;
  loop_state.used_texture_mask = used_texture_mask;
  loop_state.bindings_version = active_texture_bindings_version();
  loop_state.submission = current_submission_index();
  loop_state.usage_changes = texture_usage_changes_;
  if (loops_would_be_skipped) {
    VerifyRequestTexturesSkip(loops_verify_detail != nullptr, "Vulkan texture usage loops",
                              loops_verify_fetch_constant,
                              loops_verify_detail ? loops_verify_detail : "");
  }
}

VkImageView VulkanTextureCache::GetActiveBindingOrNullImageView(uint32_t fetch_constant_index,
                                                                xenos::FetchOpDimension dimension,
                                                                bool is_signed) {
  VkImageView image_view = VK_NULL_HANDLE;
  const TextureBinding* binding = GetValidTextureBinding(fetch_constant_index);
  if (binding && AreDimensionsCompatible(dimension, binding->key.dimension)) {
    bool force_special_view =
        binding->key.dimension == xenos::DataDimension::k3D &&
        (dimension == xenos::FetchOpDimension::k1D || dimension == xenos::FetchOpDimension::k2D);
    if (force_special_view) {
      Texture* texture = nullptr;
      bool use_signed = is_signed && texture_util::IsAnySignSigned(binding->swizzled_signs);
      if (use_signed && IsSignedVersionSeparateForFormat(binding->key)) {
        texture = binding->texture_signed;
      } else {
        texture = binding->texture;
      }
      if (texture) {
        image_view = static_cast<VulkanTexture*>(texture)->GetOrCreate3DAs2DImageView(
            use_signed, binding->host_swizzle);
      }
    } else {
      const VulkanTextureBinding& vulkan_binding = vulkan_texture_bindings_[fetch_constant_index];
      image_view =
          is_signed ? vulkan_binding.image_view_signed : vulkan_binding.image_view_unsigned;
    }
  }
  if (image_view != VK_NULL_HANDLE) {
    return image_view;
  }
  if (command_processor_.GetVulkanDevice()->properties().nullDescriptor) {
    return VK_NULL_HANDLE;
  }
  switch (dimension) {
    case xenos::FetchOpDimension::k3DOrStacked:
      return null_image_view_3d_;
    case xenos::FetchOpDimension::kCube:
      return null_image_view_cube_;
    default:
      return null_image_view_2d_array_;
  }
}

VulkanTextureCache::SamplerParameters VulkanTextureCache::GetSamplerParameters(
    const VulkanShader::SamplerBinding& binding) const {
  xenos::xe_gpu_texture_fetch_t fetch = TextureFetch(binding.fetch_constant);  // (Session 92, M4f-A: the fetch source or the register source)

  // wwe2k14 patch (Session 6, CP6): value-keyed cache. Each call is independent, so the settings can be
  // flipped at any time.
  bool cache_verify = REXCVAR_GET(vulkan_sampler_param_cache_verify);
  if (cache_verify || REXCVAR_GET(vulkan_sampler_param_cache)) {
    return GetSamplerParametersCached(fetch, binding, cache_verify);
  }
  return GetSamplerParametersForFetch(fetch, binding);
}

VulkanTextureCache::SamplerParameters VulkanTextureCache::GetSamplerParametersForFetch(
    const xenos::xe_gpu_texture_fetch_t& fetch, const VulkanShader::SamplerBinding& binding) const {
  SamplerParameters parameters;

  xenos::ClampMode fetch_clamp_x, fetch_clamp_y, fetch_clamp_z;
  texture_util::GetClampModesForDimension(fetch, fetch_clamp_x, fetch_clamp_y, fetch_clamp_z);
  parameters.clamp_x = NormalizeClampMode(fetch_clamp_x);
  parameters.clamp_y = NormalizeClampMode(fetch_clamp_y);
  parameters.clamp_z = NormalizeClampMode(fetch_clamp_z);
  if (xenos::ClampModeUsesBorder(parameters.clamp_x) ||
      xenos::ClampModeUsesBorder(parameters.clamp_y) ||
      xenos::ClampModeUsesBorder(parameters.clamp_z)) {
    parameters.border_color = fetch.border_color;
  } else {
    parameters.border_color = xenos::BorderColor::k_ABGR_Black;
  }

  xenos::TextureFilter mag_filter = binding.mag_filter == xenos::TextureFilter::kUseFetchConst
                                        ? fetch.mag_filter
                                        : binding.mag_filter;
  parameters.mag_linear = mag_filter == xenos::TextureFilter::kLinear;
  xenos::TextureFilter min_filter = binding.min_filter == xenos::TextureFilter::kUseFetchConst
                                        ? fetch.min_filter
                                        : binding.min_filter;
  parameters.min_linear = min_filter == xenos::TextureFilter::kLinear;
  xenos::TextureFilter mip_filter = binding.mip_filter == xenos::TextureFilter::kUseFetchConst
                                        ? fetch.mip_filter
                                        : binding.mip_filter;
  // wwe2k14 patch (Session 95, r3gpu, gpu_sampler_mip_point): LOW lever, picture-changing: trilinear costs the Gen9 sampler two bilinear lookups; the nearer mip only.
  parameters.mip_linear = mip_filter == xenos::TextureFilter::kLinear && !REXCVAR_GET(gpu_sampler_mip_point);
  if (REXCVAR_GET(gpu_sampler_mip_point) && mip_filter == xenos::TextureFilter::kLinear) {
    static bool first = false;
    if (!first) {
      first = true;
      REXGPU_INFO("[tex] gpu_sampler_mip_point on: the first trilinear sampler takes the nearer mip level");
    }
  }
  xenos::AnisoFilter aniso_filter = binding.aniso_filter == xenos::AnisoFilter::kUseFetchConst
                                        ? fetch.aniso_filter
                                        : binding.aniso_filter;
  aniso_filter = std::min(aniso_filter, max_anisotropy_);
  if (parameters.mag_linear || parameters.min_linear || parameters.mip_linear ||
      aniso_filter != xenos::AnisoFilter::kDisabled) {
    // Check if the texture is actually filterable on the host.
    bool linear_filterable = true;
    TextureKey texture_key;
    uint8_t texture_swizzled_signs;
    BindingInfoFromFetchConstant(fetch, texture_key, &texture_swizzled_signs);
    if (texture_key.is_valid) {
      const HostFormatPair& host_format_pair = GetHostFormatPair(texture_key);
      if ((texture_util::IsAnySignNotSigned(texture_swizzled_signs) &&
           !host_format_pair.format_unsigned.linear_filterable) ||
          (texture_util::IsAnySignSigned(texture_swizzled_signs) &&
           !host_format_pair.format_signed.linear_filterable)) {
        linear_filterable = false;
      }
    } else {
      linear_filterable = false;
    }
    if (!linear_filterable) {
      parameters.mag_linear = 0;
      parameters.min_linear = 0;
      parameters.mip_linear = 0;
      aniso_filter = xenos::AnisoFilter::kDisabled;
    }
  }
  bool min_mag_linear = parameters.mag_linear && parameters.min_linear;
  bool mip_filter_bilinear_or_trilinear =
      mip_filter == xenos::TextureFilter::kPoint || mip_filter == xenos::TextureFilter::kLinear;
  parameters.mip_base_map = mip_filter == xenos::TextureFilter::kBaseMap;
  uint32_t mip_min_level, mip_max_level;
  texture_util::GetSubresourcesFromFetchConstant(fetch, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                 &mip_min_level, &mip_max_level);
  parameters.mip_min_level = mip_min_level;
  bool has_mips = mip_max_level > mip_min_level;
  int32_t anisotropic_override = REXCVAR_GET(anisotropic_override);
  if (anisotropic_override > -1 && anisotropic_override < 6 && has_mips &&
      !parameters.mip_base_map && min_mag_linear && mip_filter_bilinear_or_trilinear) {
    aniso_filter = xenos::AnisoFilter(anisotropic_override);
  }
  aniso_filter = std::min(aniso_filter, max_anisotropy_);
  if (aniso_filter != xenos::AnisoFilter::kDisabled) {
    // Match D3D12 behavior: anisotropic filtering forces linear min/mag/mip.
    parameters.mag_linear = 1;
    parameters.min_linear = 1;
    parameters.mip_linear = 1;
  }
  parameters.aniso_filter = aniso_filter;

  return parameters;
}

// wwe2k14 patch (Session 6, CP6): sampler parameter and sampler lookup caches.
namespace {

constexpr uint32_t kSamplerCacheMismatchLogLimit = 8;
constexpr uint32_t kSamplerCacheSummaryFrames = 300;

std::string DescribeSamplerParameters(VulkanTextureCache::SamplerParameters parameters) {
  return "clamp " + std::to_string(uint32_t(parameters.clamp_x)) + "/" +
         std::to_string(uint32_t(parameters.clamp_y)) + "/" +
         std::to_string(uint32_t(parameters.clamp_z)) + " border " +
         std::to_string(uint32_t(parameters.border_color)) + " linear mag/min/mip " +
         std::to_string(uint32_t(parameters.mag_linear)) + "/" +
         std::to_string(uint32_t(parameters.min_linear)) + "/" +
         std::to_string(uint32_t(parameters.mip_linear)) + " aniso " +
         std::to_string(uint32_t(parameters.aniso_filter)) + " mip_min_level " +
         std::to_string(uint32_t(parameters.mip_min_level)) + " mip_base_map " +
         std::to_string(uint32_t(parameters.mip_base_map));
}

// Field-by-field comparison, returns the names of the fields that differ.
std::string GetSamplerParametersDifference(VulkanTextureCache::SamplerParameters a,
                                           VulkanTextureCache::SamplerParameters b) {
  std::string difference;
  auto check = [&difference](bool differs, const char* name) {
    if (differs) {
      if (!difference.empty()) {
        difference += ',';
      }
      difference += name;
    }
  };
  check(a.clamp_x != b.clamp_x, "clamp_x");
  check(a.clamp_y != b.clamp_y, "clamp_y");
  check(a.clamp_z != b.clamp_z, "clamp_z");
  check(a.border_color != b.border_color, "border_color");
  check(a.mag_linear != b.mag_linear, "mag_linear");
  check(a.min_linear != b.min_linear, "min_linear");
  check(a.mip_linear != b.mip_linear, "mip_linear");
  check(a.aniso_filter != b.aniso_filter, "aniso_filter");
  check(a.mip_min_level != b.mip_min_level, "mip_min_level");
  check(a.mip_base_map != b.mip_base_map, "mip_base_map");
  // The unused bits are always zero (the constructor zeroes `value`), but compare them too.
  check(difference.empty() && a.value != b.value, "unused_bits");
  return difference;
}

uint64_t GetSamplerHandleValue(VkSampler sampler) {
  uint64_t value = 0;
  std::memcpy(&value, &sampler, std::min(sizeof(value), sizeof(sampler)));
  return value;
}

double GetSamplerCachePercentage(uint64_t part, uint64_t total) {
  return total ? 100.0 * double(part) / double(total) : 0.0;
}

}  // namespace

uint32_t VulkanTextureCache::GetSamplerParameterCacheExtraKey(
    const VulkanShader::SamplerBinding& binding) {
  // Everything GetSamplerParametersForFetch reads, besides the 6 fetch constant dwords (the key's other part):
  // - The shader's filter overrides: binding.mag_filter, min_filter, mip_filter, aniso_filter (not
  //   binding.fetch_constant: only the value of the fetch constant matters, so entries are shared between
  //   fetch constant slots).
  // - anisotropic_override: only whether it's within [0, 6), and its value then.
  // - gpu_allow_invalid_fetch_constants (in BindingInfoFromFetchConstant, for "invalid"-type fetch constants).
  // Not in the key because they're fixed for the lifetime of the texture cache: max_anisotropy_ and
  // host_formats_ (set only in Initialize), the device's samplerMirrorClampToEdge (NormalizeClampMode) and the
  // static format tables (GetHostFormatPair's unaligned fallbacks, FormatInfo, GetBaseFormat).
  int32_t anisotropic_override = REXCVAR_GET(anisotropic_override);
  uint32_t anisotropic_override_key =
      (anisotropic_override > -1 && anisotropic_override < 6) ? uint32_t(anisotropic_override) + 1 : 0;
  return uint32_t(binding.mag_filter) | (uint32_t(binding.min_filter) << 2) |
         (uint32_t(binding.mip_filter) << 4) | (uint32_t(binding.aniso_filter) << 6) |
         (anisotropic_override_key << 9) |
         (uint32_t(REXCVAR_GET(gpu_allow_invalid_fetch_constants) ? 1 : 0) << 12) |
         // Valid entry marker - zero-initialized entries never match.
         (UINT32_C(1) << 31);
}

VulkanTextureCache::SamplerParameters VulkanTextureCache::GetSamplerParametersCached(
    const xenos::xe_gpu_texture_fetch_t& fetch, const VulkanShader::SamplerBinding& binding,
    bool verify) const {
  if (!sampler_parameter_cache_) {
    sampler_parameter_cache_.reset(
        new SamplerParameterCacheSet[size_t(1) << kSamplerParameterCacheSetCountLog2]());
    REXGPU_INFO("[s6][CP6] Sampler parameter cache allocated ({} sets of 2 entries)",
                uint32_t(1) << kSamplerParameterCacheSetCountLog2);
  }

  // Keyed on the value: a fetch constant re-sent with the same value (or the same texture bound again later)
  // still hits, unlike invalidation through TextureFetchConstantWritten.
  uint64_t key_fetch[3];
  static_assert(sizeof(key_fetch) == sizeof(fetch));
  std::memcpy(key_fetch, &fetch, sizeof(key_fetch));
  uint32_t key_extra = GetSamplerParameterCacheExtraKey(binding);
  uint64_t hash = key_fetch[0] * UINT64_C(0x9E3779B97F4A7C15);
  hash = (hash ^ (hash >> 32) ^ key_fetch[1]) * UINT64_C(0xBF58476D1CE4E5B9);
  hash = (hash ^ (hash >> 32) ^ key_fetch[2]) * UINT64_C(0x94D049BB133111EB);
  hash = (hash ^ (hash >> 32) ^ key_extra) * UINT64_C(0x9E3779B97F4A7C15);
  SamplerParameterCacheSet& set =
      sampler_parameter_cache_[size_t(hash >> (64 - kSamplerParameterCacheSetCountLog2))];
  const SamplerParameterCacheEntry* hit_entry = nullptr;
  for (const SamplerParameterCacheEntry& entry : set.entries) {
    if (!((entry.key_fetch[0] ^ key_fetch[0]) | (entry.key_fetch[1] ^ key_fetch[1]) |
          (entry.key_fetch[2] ^ key_fetch[2]) | uint64_t(entry.key_extra ^ key_extra))) {
      hit_entry = &entry;
      break;
    }
  }

  SamplerCacheStats& stats = sampler_cache_stats_;
  ++stats.parameter_lookups;
  if (hit_entry) {
    ++stats.parameter_hits;
    if (!verify) {
      return hit_entry->parameters;
    }
  }

  SamplerParameters parameters = GetSamplerParametersForFetch(fetch, binding);

  if (hit_entry) {
    // Verification of a hit. The original result is returned, and the entry is left as is, so a wrong entry
    // keeps being counted.
    ++stats.parameter_verified;
    if (hit_entry->parameters != parameters) {
      ++stats.parameter_mismatches;
      ++stats.parameter_mismatches_total;
      if (stats.parameter_mismatches_logged < kSamplerCacheMismatchLogLimit) {
        ++stats.parameter_mismatches_logged;
        REXGPU_WARN(
            "[s6][CP6] Sampler parameter cache mismatch {}: fetch constant {} ({:08X} {:08X} {:08X} {:08X} "
            "{:08X} {:08X}), shader filters mag {} min {} mip {} aniso {}, key extra 0x{:08X}; cached "
            "0x{:06X} ({}), computed 0x{:06X} ({}); differing fields: {}",
            stats.parameter_mismatches_total, uint32_t(binding.fetch_constant), fetch.dword_0,
            fetch.dword_1, fetch.dword_2, fetch.dword_3, fetch.dword_4, fetch.dword_5,
            uint32_t(binding.mag_filter), uint32_t(binding.min_filter), uint32_t(binding.mip_filter),
            uint32_t(binding.aniso_filter), key_extra, hit_entry->parameters.value,
            DescribeSamplerParameters(hit_entry->parameters), parameters.value,
            DescribeSamplerParameters(parameters),
            GetSamplerParametersDifference(hit_entry->parameters, parameters));
      }
    }
    return parameters;
  }

  // Miss - store, unless a setting in the key was changed (from another thread) during the computation. A
  // 2-entry FIFO per set: the new entry goes first, the older first entry replaces the second.
  if (GetSamplerParameterCacheExtraKey(binding) == key_extra) {
    set.entries[1] = set.entries[0];
    SamplerParameterCacheEntry& new_entry = set.entries[0];
    std::memcpy(new_entry.key_fetch, key_fetch, sizeof(key_fetch));
    new_entry.key_extra = key_extra;
    new_entry.parameters = parameters;
  }
  return parameters;
}

void VulkanTextureCache::ReportSamplerCacheStats() {
  SamplerCacheStats& stats = sampler_cache_stats_;
  if (++stats.frames < kSamplerCacheSummaryFrames) {
    return;
  }
  REXGPU_INFO(
      "[s6][CP6] Sampler caches (cache {}, verify {}), last {} frames: parameters {} lookups, {} hits "
      "({:.1f}%), {} hits verified, {} mismatches; samplers {} lookups, {} skipped ({:.1f}%), {} skips "
      "verified, {} mismatches; mismatches since start: {} parameters, {} samplers",
      REXCVAR_GET(vulkan_sampler_param_cache), REXCVAR_GET(vulkan_sampler_param_cache_verify),
      stats.frames, stats.parameter_lookups, stats.parameter_hits,
      GetSamplerCachePercentage(stats.parameter_hits, stats.parameter_lookups),
      stats.parameter_verified, stats.parameter_mismatches, stats.use_calls, stats.use_hits,
      GetSamplerCachePercentage(stats.use_hits, stats.use_calls), stats.use_verified,
      stats.use_mismatches, stats.parameter_mismatches_total, stats.use_mismatches_total);
  stats.parameter_lookups = 0;
  stats.parameter_hits = 0;
  stats.parameter_verified = 0;
  stats.parameter_mismatches = 0;
  stats.use_calls = 0;
  stats.use_hits = 0;
  stats.use_verified = 0;
  stats.use_mismatches = 0;
  stats.frames = 0;
}

VkSampler VulkanTextureCache::UseSampler(SamplerParameters parameters, bool& has_overflown_out) {
  // wwe2k14 patch (Session 6, CP6): skip the samplers_ lookup for parameters already looked up in the current
  // submission. Within one submission, UseSamplerUncached's result for given parameters can't change: it only
  // destroys samplers whose last usage submission has completed (the current one is still open, so a sampler
  // returned in it stays alive until the submission index changes), and for a sampler already used in the
  // current submission it only returns it (the LRU position is updated once per submission, on the first
  // call). The submission index changes only when a submission is ended, so entries from earlier submissions
  // never match. Settings read per call, safe to flip at any time.
  bool cache_verify = REXCVAR_GET(vulkan_sampler_param_cache_verify);
  if (!cache_verify && !REXCVAR_GET(vulkan_sampler_param_cache)) {
    return UseSamplerUncached(parameters, has_overflown_out);
  }

  assert_true(command_processor_.submission_open());
  uint64_t submission_current = command_processor_.GetCurrentSubmission();
  if (!sampler_use_cache_) {
    sampler_use_cache_.reset(new SamplerUseCacheEntry[size_t(1) << kSamplerUseCacheSizeLog2]());
  }
  SamplerUseCacheEntry& entry =
      sampler_use_cache_[(parameters.value * UINT32_C(0x9E3779B1)) >> (32 - kSamplerUseCacheSizeLog2)];
  SamplerCacheStats& stats = sampler_cache_stats_;
  ++stats.use_calls;
  bool hit = entry.submission == submission_current && entry.parameters == parameters;
  if (hit) {
    ++stats.use_hits;
    if (!cache_verify) {
      has_overflown_out = false;
      return entry.sampler;
    }
  }

  VkSampler sampler = UseSamplerUncached(parameters, has_overflown_out);

  if (hit) {
    // Verification of a skip - the looked up result is returned.
    ++stats.use_verified;
    if (sampler != entry.sampler || has_overflown_out) {
      ++stats.use_mismatches;
      ++stats.use_mismatches_total;
      if (stats.use_mismatches_logged < kSamplerCacheMismatchLogLimit) {
        ++stats.use_mismatches_logged;
        REXGPU_WARN(
            "[s6][CP6] Sampler lookup cache mismatch {}: parameters 0x{:06X} ({}), submission {}: cached "
            "sampler 0x{:X}, looked up 0x{:X} (overflow {})",
            stats.use_mismatches_total, parameters.value, DescribeSamplerParameters(parameters),
            submission_current, GetSamplerHandleValue(entry.sampler), GetSamplerHandleValue(sampler),
            has_overflown_out);
      }
    }
    return sampler;
  }

  if (sampler != VK_NULL_HANDLE) {
    entry.submission = submission_current;
    entry.sampler = sampler;
    entry.parameters = parameters;
  }
  return sampler;
}

VkSampler VulkanTextureCache::UseSamplerUncached(SamplerParameters parameters,
                                                 bool& has_overflown_out) {
  assert_true(command_processor_.submission_open());
  uint64_t submission_current = command_processor_.GetCurrentSubmission();

  // Try to find an existing sampler.
  auto it_existing = samplers_.find(parameters);
  if (it_existing != samplers_.end()) {
    std::pair<const SamplerParameters, Sampler>& sampler = *it_existing;
    assert_true(sampler.second.last_usage_submission <= submission_current);
    // This is called very frequently, don't relink unless needed for caching.
    if (sampler.second.last_usage_submission < submission_current) {
      // Move to the front of the LRU queue.
      sampler.second.last_usage_submission = submission_current;
      if (sampler.second.used_next) {
        if (sampler.second.used_previous) {
          sampler.second.used_previous->second.used_next = sampler.second.used_next;
        } else {
          sampler_used_first_ = sampler.second.used_next;
        }
        sampler.second.used_next->second.used_previous = sampler.second.used_previous;
        sampler.second.used_previous = sampler_used_last_;
        sampler.second.used_next = nullptr;
        sampler_used_last_->second.used_next = &sampler;
        sampler_used_last_ = &sampler;
      }
    }
    has_overflown_out = false;
    return sampler.second.sampler;
  }

  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  // See if an existing sampler can be destroyed to create space for the new
  // one.
  if (samplers_.size() >= sampler_max_count_) {
    assert_not_null(sampler_used_first_);
    if (!sampler_used_first_) {
      has_overflown_out = false;
      return VK_NULL_HANDLE;
    }
    if (sampler_used_first_->second.last_usage_submission >
        command_processor_.GetCompletedSubmission()) {
      has_overflown_out = true;
      return VK_NULL_HANDLE;
    }
    auto it_reuse = samplers_.find(sampler_used_first_->first);
    if (sampler_used_first_->second.uses_custom_border_color) {
      assert_true(custom_border_color_sampler_count_ > 0);
      if (custom_border_color_sampler_count_) {
        --custom_border_color_sampler_count_;
      }
    }
    dfn.vkDestroySampler(device, sampler_used_first_->second.sampler, nullptr);
    if (sampler_used_first_->second.used_next) {
      sampler_used_first_->second.used_next->second.used_previous =
          sampler_used_first_->second.used_previous;
    } else {
      sampler_used_last_ = sampler_used_first_->second.used_previous;
    }
    sampler_used_first_ = sampler_used_first_->second.used_next;
    assert_true(it_reuse != samplers_.end());
    if (it_reuse != samplers_.end()) {
      // This destroys the Sampler object.
      samplers_.erase(it_reuse);
      COUNT_profile_set("gpu/texture_cache/vulkan/samplers", samplers_.size());
    } else {
      has_overflown_out = false;
      return VK_NULL_HANDLE;
    }
  }

  // Create a new sampler and make it the least recently used.
  // The values are normalized, and unsupported ones are excluded, in
  // GetSamplerParameters.
  VkSamplerCreateInfo sampler_create_info = {};
  sampler_create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  if (vulkan_device->properties().nonSeamlessCubeMap && REXCVAR_GET(non_seamless_cube_map)) {
    sampler_create_info.flags |= VK_SAMPLER_CREATE_NON_SEAMLESS_CUBE_MAP_BIT_EXT;
  }
  sampler_create_info.magFilter = parameters.mag_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
  sampler_create_info.minFilter = parameters.min_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
  sampler_create_info.mipmapMode =
      parameters.mip_linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
  static const VkSamplerAddressMode kAddressModeMap[] = {
      // kRepeat
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      // kMirroredRepeat
      VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
      // kClampToEdge
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      // kMirrorClampToEdge
      VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
      // kClampToHalfway
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      // kMirrorClampToHalfway
      VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
      // kClampToBorder
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
      // kMirrorClampToBorder
      VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
  };
  sampler_create_info.addressModeU = kAddressModeMap[uint32_t(parameters.clamp_x)];
  sampler_create_info.addressModeV = kAddressModeMap[uint32_t(parameters.clamp_y)];
  sampler_create_info.addressModeW = kAddressModeMap[uint32_t(parameters.clamp_z)];
  // LOD biasing is performed in shaders.
  if (parameters.aniso_filter != xenos::AnisoFilter::kDisabled) {
    sampler_create_info.anisotropyEnable = VK_TRUE;
    sampler_create_info.maxAnisotropy =
        float(UINT32_C(1) << (uint32_t(parameters.aniso_filter) -
                              uint32_t(xenos::AnisoFilter::kMax_1_1)));
  }
  sampler_create_info.minLod = float(parameters.mip_min_level);
  // wwe2k14 patch (Session 55): gpu_debug_texture_min_lod (not for base-map-only samplers).
  if (const double extra = REXCVAR_GET(gpu_debug_texture_min_lod); extra > 0 && !parameters.mip_base_map) {
    sampler_create_info.minLod += float(extra);
  }
  if (parameters.mip_base_map) {
    assert_false(parameters.mip_linear);
    sampler_create_info.maxLod = sampler_create_info.minLod + 0.25f;
  } else {
    sampler_create_info.maxLod = VK_LOD_CLAMP_NONE;
  }
  bool uses_custom_border_color = false;
  VkSamplerCustomBorderColorCreateInfoEXT sampler_custom_border_color_create_info = {};
  const bool custom_border_colors_supported = vulkan_device->properties().customBorderColors;
  const bool custom_border_color_slots_available =
      custom_border_color_sampler_count_ < vulkan_device->properties().maxCustomBorderColorSamplers;
  switch (parameters.border_color) {
    case xenos::BorderColor::k_ABGR_White:
      sampler_create_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
      break;
    case xenos::BorderColor::k_ACBYCR_Black:
    case xenos::BorderColor::k_ACBCRY_Black:
      if (custom_border_colors_supported && custom_border_color_slots_available) {
        sampler_create_info.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
        sampler_custom_border_color_create_info.sType =
            VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT;
        sampler_custom_border_color_create_info.pNext = sampler_create_info.pNext;
        sampler_custom_border_color_create_info.customBorderColor.float32[0] =
            parameters.border_color == xenos::BorderColor::k_ACBYCR_Black ? 0.5f : 0.0f;
        sampler_custom_border_color_create_info.customBorderColor.float32[1] =
            parameters.border_color == xenos::BorderColor::k_ACBYCR_Black ? 0.0f : 0.5f;
        sampler_custom_border_color_create_info.customBorderColor.float32[2] = 0.5f;
        sampler_custom_border_color_create_info.customBorderColor.float32[3] = 0.0f;
        sampler_custom_border_color_create_info.format =
            vulkan_device->properties().customBorderColorWithoutFormat
                ? VK_FORMAT_UNDEFINED
                : VK_FORMAT_R32G32B32A32_SFLOAT;
        sampler_create_info.pNext = &sampler_custom_border_color_create_info;
        uses_custom_border_color = true;
      } else {
        static bool custom_border_color_fallback_logged = false;
        if (!custom_border_color_fallback_logged) {
          custom_border_color_fallback_logged = true;
          if (!custom_border_colors_supported) {
            REXGPU_WARN(
                "VulkanTextureCache: VK_EXT_custom_border_color support is unavailable; "
                "falling back to transparent black for YCbCr border color sampling");
          } else {
            REXGPU_WARN(
                "VulkanTextureCache: maxCustomBorderColorSamplers limit ({}) reached; "
                "falling back to transparent black for additional YCbCr border samplers",
                vulkan_device->properties().maxCustomBorderColorSamplers);
          }
        }
        sampler_create_info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
      }
      break;
    default:
      sampler_create_info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
      break;
  }
  VkSampler vulkan_sampler;
  if (dfn.vkCreateSampler(device, &sampler_create_info, nullptr, &vulkan_sampler) != VK_SUCCESS) {
    REXGPU_ERROR(
        "VulkanTextureCache: Failed to create the sampler for parameters "
        "0x{:08X}",
        parameters.value);
    has_overflown_out = false;
    return VK_NULL_HANDLE;
  }
  std::pair<const SamplerParameters, Sampler>& new_sampler =
      *(samplers_
            .emplace(std::piecewise_construct, std::forward_as_tuple(parameters),
                     std::forward_as_tuple())
            .first);
  COUNT_profile_set("gpu/texture_cache/vulkan/samplers", samplers_.size());
  new_sampler.second.sampler = vulkan_sampler;
  new_sampler.second.uses_custom_border_color = uses_custom_border_color;
  new_sampler.second.last_usage_submission = submission_current;
  new_sampler.second.used_previous = sampler_used_last_;
  new_sampler.second.used_next = nullptr;
  if (uses_custom_border_color) {
    ++custom_border_color_sampler_count_;
  }
  if (sampler_used_last_) {
    sampler_used_last_->second.used_next = &new_sampler;
  } else {
    sampler_used_first_ = &new_sampler;
  }
  sampler_used_last_ = &new_sampler;
  return vulkan_sampler;
}

uint64_t VulkanTextureCache::GetSubmissionToAwaitOnSamplerOverflow(
    uint32_t overflowed_sampler_count) const {
  if (!overflowed_sampler_count) {
    return 0;
  }
  std::pair<const SamplerParameters, Sampler>* sampler_used = sampler_used_first_;
  if (!sampler_used_first_) {
    return 0;
  }
  for (uint32_t samplers_remaining = overflowed_sampler_count - 1; samplers_remaining;
       --samplers_remaining) {
    std::pair<const SamplerParameters, Sampler>* sampler_used_next = sampler_used->second.used_next;
    if (!sampler_used_next) {
      break;
    }
    sampler_used = sampler_used_next;
  }
  return sampler_used->second.last_usage_submission;
}

VkImageView VulkanTextureCache::RequestSwapTexture(uint32_t& width_scaled_out,
                                                   uint32_t& height_scaled_out,
                                                   xenos::TextureFormat& format_out,
                                                   uint32_t* width_unscaled_out,
                                                   uint32_t* height_unscaled_out,
                                                   bool* swap_source_needs_rb_swap_out) {
  // wwe2k14 patch (Session 92, M3b, gpu_d3d_owned_present): the present item carries the swap texture's fetch as data - no fetch
  // constant 0 written into the register file / image; else (the packet path) fetch constant 0 of the register source, as before.
  xenos::xe_gpu_texture_fetch_t fetch;
  if (const uint32_t* const swap_fetch = rex::graphics::free_counters::g_swap_fetch_override) {
    std::memcpy(&fetch, swap_fetch, sizeof(fetch));
  } else {
    rex::graphics::free_counters::Bump(rex::graphics::free_counters::g_c.swap_fetch_regs);
    fetch = register_file().GetTextureFetch(0);
  }
  TextureKey key;
  BindingInfoFromFetchConstant(fetch, key, nullptr);
  if (!key.is_valid || key.base_page == 0 || key.dimension != xenos::DataDimension::k2DOrStacked) {
    return nullptr;
  }
  VulkanTexture* texture = static_cast<VulkanTexture*>(FindOrCreateTexture(key));
  if (!texture) {
    return VK_NULL_HANDLE;
  }
  uint32_t host_swizzle = GuestToHostSwizzle(fetch.swizzle, GetHostFormatSwizzle(key));
  // wwe2k14 patch (Session 82, the owned match frame, P2): the swap noted for the owned layer (its page, size, format,
  // byte order, scaled flag) - and when our image stands for the page (the copy that feeds the front buffer was left
  // out: the texture is stale) the presenter samples it instead. The texture itself is not touched then (no load, no
  // usage change, no barrier: our image is readable since the copy's barrier).
  {
    const TextureKey& tk = texture->key();
    const bool owned_servable = key.dimension == xenos::DataDimension::k2DOrStacked &&
                                !key.depth_or_array_size_minus_1 && !key.mip_max_level;
    uint32_t owned_swizzle = host_swizzle;
    if (!command_processor_.GetVulkanDevice()->properties().imageViewFormatSwizzle) {
      owned_swizzle = xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA;
    }
    const VkComponentMapping owned_components = {
        VulkanTexture::NativeComponentSwizzle(owned_swizzle, 0), VulkanTexture::NativeComponentSwizzle(owned_swizzle, 1),
        VulkanTexture::NativeComponentSwizzle(owned_swizzle, 2), VulkanTexture::NativeComponentSwizzle(owned_swizzle, 3)};
    const VkImageView owned_view = command_processor_.NativeOwnedSwapView(
        key.base_page, key.GetWidth(), key.GetHeight(), uint32_t(key.format), uint32_t(key.endianness),
        bool(tk.scaled_resolve), owned_components, owned_servable);
    if (owned_view != VK_NULL_HANDLE) {
      if (swap_source_needs_rb_swap_out) {
        auto swizzle_component = [](uint32_t swizzle,
                                    uint32_t component_index) -> xenos::XE_GPU_TEXTURE_SWIZZLE {
          return xenos::XE_GPU_TEXTURE_SWIZZLE((swizzle >> (3 * component_index)) & 0b111);
        };
        *swap_source_needs_rb_swap_out =
            swizzle_component(host_swizzle, 0) == xenos::XE_GPU_TEXTURE_SWIZZLE_B &&
            swizzle_component(host_swizzle, 1) == xenos::XE_GPU_TEXTURE_SWIZZLE_G &&
            swizzle_component(host_swizzle, 2) == xenos::XE_GPU_TEXTURE_SWIZZLE_R;
      }
      if (width_unscaled_out) *width_unscaled_out = tk.GetWidth();
      if (height_unscaled_out) *height_unscaled_out = tk.GetHeight();
      width_scaled_out = tk.GetWidth() * (tk.scaled_resolve ? draw_resolution_scale_x() : 1);
      height_scaled_out = tk.GetHeight() * (tk.scaled_resolve ? draw_resolution_scale_y() : 1);
      format_out = tk.format;
      return owned_view;
    }
  }
  if (swap_source_needs_rb_swap_out) {
    auto swizzle_component = [](uint32_t swizzle,
                                uint32_t component_index) -> xenos::XE_GPU_TEXTURE_SWIZZLE {
      return xenos::XE_GPU_TEXTURE_SWIZZLE((swizzle >> (3 * component_index)) & 0b111);
    };
    *swap_source_needs_rb_swap_out =
        swizzle_component(host_swizzle, 0) == xenos::XE_GPU_TEXTURE_SWIZZLE_B &&
        swizzle_component(host_swizzle, 1) == xenos::XE_GPU_TEXTURE_SWIZZLE_G &&
        swizzle_component(host_swizzle, 2) == xenos::XE_GPU_TEXTURE_SWIZZLE_R;
  }
  VkImageView texture_view = texture->GetView(false, host_swizzle, false);
  if (texture_view == VK_NULL_HANDLE) {
    return VK_NULL_HANDLE;
  }
  if (!LoadTextureData(*texture)) {
    return VK_NULL_HANDLE;
  }
  texture->MarkAsUsed();
  texture->native_used = true;  // wwe2k14 patch (Session 80)
  VulkanTexture::Usage old_usage = texture->SetUsage(VulkanTexture::Usage::kSwapSampled);
  // wwe2k14 patch (Session 59 review): as for a guest sample.
  if (old_usage == VulkanTexture::Usage::kComputeStorageWrite) command_processor_.NativeFlushDeferredResolves(texture);
  if (old_usage != VulkanTexture::Usage::kSwapSampled) {
    VkPipelineStageFlags src_stage_mask, dst_stage_mask;
    VkAccessFlags src_access_mask, dst_access_mask;
    VkImageLayout old_layout, new_layout;
    GetTextureUsageMasks(old_usage, src_stage_mask, src_access_mask, old_layout);
    GetTextureUsageMasks(VulkanTexture::Usage::kSwapSampled, dst_stage_mask, dst_access_mask,
                         new_layout);
    command_processor_.PushImageMemoryBarrier(
        texture->image(), ui::vulkan::util::InitializeSubresourceRange(), src_stage_mask,
        dst_stage_mask, src_access_mask, dst_access_mask, old_layout, new_layout);
  }
  // Only texture->key, not the result of BindingInfoFromFetchConstant, contains
  // whether the texture is scaled.
  key = texture->key();
  if (width_unscaled_out) {
    *width_unscaled_out = key.GetWidth();
  }
  if (height_unscaled_out) {
    *height_unscaled_out = key.GetHeight();
  }
  width_scaled_out = key.GetWidth() * (key.scaled_resolve ? draw_resolution_scale_x() : 1);
  height_scaled_out = key.GetHeight() * (key.scaled_resolve ? draw_resolution_scale_y() : 1);
  format_out = key.format;
  return texture_view;
}

bool VulkanTextureCache::IsSignedVersionSeparateForFormat(TextureKey key) const {
  const HostFormatPair& host_format_pair = GetHostFormatPair(key);
  if (host_format_pair.format_unsigned.format == VK_FORMAT_UNDEFINED ||
      host_format_pair.format_signed.format == VK_FORMAT_UNDEFINED) {
    // Just one signedness.
    return false;
  }
  return !host_format_pair.unsigned_signed_compatible;
}

bool VulkanTextureCache::IsScaledResolveSupportedForFormat(TextureKey key) const {
  if (!IsDrawResolutionScaled()) {
    return false;
  }
  const HostFormatPair& host_format_pair = GetHostFormatPair(key);
  bool host_format_is_signed;
  if (IsSignedVersionSeparateForFormat(key)) {
    host_format_is_signed = bool(key.signed_separate);
  } else {
    host_format_is_signed = host_format_pair.format_unsigned.load_shader == kLoadShaderIndexUnknown;
  }
  LoadShaderIndex load_shader =
      (host_format_is_signed ? host_format_pair.format_signed : host_format_pair.format_unsigned)
          .load_shader;
  return load_shader != kLoadShaderIndexUnknown &&
         load_pipelines_scaled_[load_shader] != VK_NULL_HANDLE;
}

uint32_t VulkanTextureCache::GetHostFormatSwizzle(TextureKey key) const {
  return GetHostFormatPair(key).swizzle;
}

uint32_t VulkanTextureCache::GetMaxHostTextureWidthHeight(xenos::DataDimension dimension) const {
  const ui::vulkan::VulkanDevice::Properties& device_properties =
      command_processor_.GetVulkanDevice()->properties();
  switch (dimension) {
    case xenos::DataDimension::k1D:
    case xenos::DataDimension::k2DOrStacked:
      // 1D and 2D are emulated as 2D arrays.
      return device_properties.maxImageDimension2D;
    case xenos::DataDimension::k3D:
      return device_properties.maxImageDimension3D;
    case xenos::DataDimension::kCube:
      return device_properties.maxImageDimensionCube;
    default:
      assert_unhandled_case(dimension);
      return 0;
  }
}

uint32_t VulkanTextureCache::GetMaxHostTextureDepthOrArraySize(
    xenos::DataDimension dimension) const {
  const ui::vulkan::VulkanDevice::Properties& device_properties =
      command_processor_.GetVulkanDevice()->properties();
  switch (dimension) {
    case xenos::DataDimension::k1D:
    case xenos::DataDimension::k2DOrStacked:
      // 1D and 2D are emulated as 2D arrays.
      return device_properties.maxImageArrayLayers;
    case xenos::DataDimension::k3D:
      return device_properties.maxImageDimension3D;
    case xenos::DataDimension::kCube:
      // Not requesting the imageCubeArray feature, and the Xenos doesn't
      // support cube map arrays.
      return 6;
    default:
      assert_unhandled_case(dimension);
      return 0;
  }
}

std::unique_ptr<TextureCache::Texture> VulkanTextureCache::CreateTexture(TextureKey key) {
  VkFormat formats[] = {VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};
  const HostFormatPair& host_format = GetHostFormatPair(key);
  if (host_format.format_signed.format == VK_FORMAT_UNDEFINED) {
    // Only the unsigned format may be available, if at all.
    formats[0] = host_format.format_unsigned.format;
  } else if (host_format.format_unsigned.format == VK_FORMAT_UNDEFINED) {
    // Only the signed format may be available, if at all.
    formats[0] = host_format.format_signed.format;
  } else {
    // Both unsigned and signed formats are available.
    if (IsSignedVersionSeparateForFormat(key)) {
      formats[0] = key.signed_separate ? host_format.format_signed.format
                                       : host_format.format_unsigned.format;
    } else {
      // Same format for unsigned and signed, or compatible formats.
      formats[0] = host_format.format_unsigned.format;
      if (host_format.format_signed.format != host_format.format_unsigned.format) {
        assert_not_zero(host_format.unsigned_signed_compatible);
        formats[1] = host_format.format_signed.format;
      }
    }
  }
  if (formats[0] == VK_FORMAT_UNDEFINED) {
    unsupported_format_features_used_[uint32_t(key.format)] |= kUnsupportedResourceBit;
    return nullptr;
  }

  // wwe2k14 patch (Session 7c, TEXDIRECT): vulkan_texture_load_direct is read here - a texture that may be loaded
  // directly needs the storage usage from creation. Review fix: direct loads are initialized here, on first use.
  VulkanTexture::DirectLoadCreation direct_load_creation =
      VulkanTexture::DirectLoadCreation::kNotRequested;
  VkFormat direct_load_format = VK_FORMAT_UNDEFINED;
  // wwe2k14 patch (Session 43, S0b-lite): or a 1x texture over a page a resolve wrote.
  // Review fix: tiled only (the direct shaders always address tiled memory; scaled textures are always tiled).
  const bool direct_unscaled = !key.scaled_resolve && key.tiled &&
                               REXCVAR_GET(vulkan_texture_load_direct_unscaled) &&
                               (REXCVAR_GET(vulkan_texture_load_direct_unscaled_any_page) ||
                                IsPageResolvedUnscaled(key.base_page << 12)) &&
                               (uint32_t(REXCVAR_GET(vulkan_texture_load_direct_unscaled_kinds)) &
                                DirectUnscaledKindBit(key.format)) != 0;
  // wwe2k14 patch (Session 59, vulkan_texture_native_storage_small): 1x k_8 / k_4_4_4_4 textures without mips get an
  // R8_UINT / R16_UINT storage view too (no direct load shader for them: they still load through the buffer path), so
  // a native resolve can write them (gpu_native_truth_direct_resolve_full_1x_texture).
  // (Review: k_8 only, on an R8 host format - the R8_UINT view is of its size class, and R8 storage images exist with
  // the extended formats; B4G4R4A4 has no storage image support, so k_4_4_4_4 would need extended usage.)
  const bool native_storage_small =
      REXCVAR_GET(vulkan_texture_native_storage_small) && !key.scaled_resolve && key.tiled && !key.mip_max_level &&
      key.dimension == xenos::DataDimension::k2DOrStacked && !key.depth_or_array_size_minus_1 &&
      key.format == xenos::TextureFormat::k_8 &&
      (formats[0] == VK_FORMAT_R8_UNORM || formats[0] == VK_FORMAT_R8_SNORM) &&
      (formats[1] == VK_FORMAT_UNDEFINED || formats[1] == VK_FORMAT_R8_UNORM || formats[1] == VK_FORMAT_R8_SNORM) &&
      key.GetGuestLayout().packed_level != 0 &&
      command_processor_.GetVulkanDevice()->properties().shaderStorageImageExtendedFormats;
  // wwe2k14 patch (Session 79, gpu_native_truth_fill_cube_faces): 1x tiled single-level k_8_8_8_8 cubes (RGBA8 host
  // formats) get the R32_UINT storage view too, for the faces' fills (they still load through the buffer path: the
  // direct load takes single-layer textures only).
  const bool native_storage_cube =
      REXCVAR_GET(gpu_native_truth_fill_cube_faces) && !key.scaled_resolve && key.tiled && !key.mip_max_level &&
      key.dimension == xenos::DataDimension::kCube && key.GetDepthOrArraySize() == VulkanTexture::kNativeLayerViews &&
      key.format == xenos::TextureFormat::k_8_8_8_8 &&
      (formats[0] == VK_FORMAT_R8G8B8A8_UNORM || formats[0] == VK_FORMAT_R8G8B8A8_SNORM) &&
      (formats[1] == VK_FORMAT_UNDEFINED || formats[1] == VK_FORMAT_R8G8B8A8_UNORM ||
       formats[1] == VK_FORMAT_R8G8B8A8_SNORM) &&
      key.GetGuestLayout().packed_level != 0;
  if (native_storage_small && EnsureDirectLoadInitialized()) {
    direct_load_format = VK_FORMAT_R8_UINT;
    direct_load_creation = VulkanTexture::DirectLoadCreation::kDeviceRejected;
  } else if (native_storage_cube && EnsureDirectLoadInitialized()) {
    direct_load_format = VK_FORMAT_R32_UINT;
    direct_load_creation = VulkanTexture::DirectLoadCreation::kDeviceRejected;
  } else if ((key.scaled_resolve && REXCVAR_GET(vulkan_texture_load_direct)) || direct_unscaled) {
    if (!EnsureDirectLoadInitialized()) {
      direct_load_creation = VulkanTexture::DirectLoadCreation::kUnavailable;
    } else {
      direct_load_format = GetDirectLoadFormatForNewTexture(key, formats, 2);
      // kDeviceRejected until the image is created with the storage usage.
      direct_load_creation = direct_load_format != VK_FORMAT_UNDEFINED
                                 ? VulkanTexture::DirectLoadCreation::kDeviceRejected
                                 : VulkanTexture::DirectLoadCreation::kUnsupported;
    }
  }

  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  bool is_3d = key.dimension == xenos::DataDimension::k3D;
  uint32_t depth_or_array_size = key.GetDepthOrArraySize();

  VkImageCreateInfo image_create_info;
  VkImageCreateInfo* image_create_info_last = &image_create_info;
  image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_create_info.pNext = nullptr;
  image_create_info.flags = 0;
  if (formats[1] != VK_FORMAT_UNDEFINED) {
    image_create_info.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
  }
  if (key.dimension == xenos::DataDimension::kCube) {
    image_create_info.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  }
  image_create_info.imageType = is_3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
  image_create_info.format = formats[0];
  image_create_info.extent.width = key.GetWidth();
  image_create_info.extent.height = key.GetHeight();
  if (key.scaled_resolve) {
    image_create_info.extent.width *= draw_resolution_scale_x();
    image_create_info.extent.height *= draw_resolution_scale_y();
  }
  image_create_info.extent.depth = is_3d ? depth_or_array_size : 1;
  image_create_info.mipLevels = key.mip_max_level + 1;
  image_create_info.arrayLayers = is_3d ? 1 : depth_or_array_size;
  image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_create_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_create_info.queueFamilyIndexCount = 0;
  image_create_info.pQueueFamilyIndices = nullptr;
  image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  // wwe2k14 patch (Session 7c, TEXDIRECT): the view formats of the image - plus, for direct loads, the unsigned
  // integer storage view format (with the storage usage, and the transfer source usage for the
  // vulkan_texture_load_direct_verify readback, so verified images are created exactly like unverified ones). Used
  // only if the device supports the image with them (kDeviceRejected otherwise); without them, the flags, usage
  // and format list are exactly the ones without the setting.
  const VkImageCreateFlags image_flags_base = image_create_info.flags;
  const VkImageUsageFlags image_usage_base = image_create_info.usage;
  bool image_direct_load = false;
  if (direct_load_format != VK_FORMAT_UNDEFINED) {
    VkImageFormatProperties direct_load_image_format_properties;
    image_direct_load =
        direct_load_get_image_format_properties_ &&
        direct_load_get_image_format_properties_(
            vulkan_device->physical_device(), image_create_info.format, image_create_info.imageType,
            image_create_info.tiling,
            image_usage_base | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            image_flags_base | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
            &direct_load_image_format_properties) == VK_SUCCESS &&
        image_create_info.extent.width <= direct_load_image_format_properties.maxExtent.width &&
        image_create_info.extent.height <= direct_load_image_format_properties.maxExtent.height &&
        image_create_info.mipLevels <= direct_load_image_format_properties.maxMipLevels &&
        image_create_info.arrayLayers <= direct_load_image_format_properties.maxArrayLayers;
  }
  VkFormat image_view_formats[3] = {formats[0], formats[1], VK_FORMAT_UNDEFINED};
  VkImageFormatListCreateInfo image_format_list_create_info;
  image_format_list_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO;
  image_format_list_create_info.pNext = nullptr;
  image_format_list_create_info.viewFormatCount = 0;
  image_format_list_create_info.pViewFormats = image_view_formats;

  VmaAllocationCreateInfo allocation_create_info = {};
  allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  VkImage image;
  VmaAllocation allocation;
  // Review fix: if creating the image with the direct load usage fails, it's created again without it (the second
  // iteration, exactly as with the setting off) instead of failing the texture.
  for (;;) {
    image_create_info.flags = image_flags_base;
    image_create_info.usage = image_usage_base;
    uint32_t image_view_format_count = formats[1] != VK_FORMAT_UNDEFINED ? 2 : 1;
    if (image_direct_load) {
      image_create_info.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
      image_create_info.usage |= VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
      image_view_formats[image_view_format_count++] = direct_load_format;
    }
    // The format list is the only structure chained to the image create info.
    image_create_info.pNext = nullptr;
    image_create_info_last = &image_create_info;
    if (image_view_format_count > 1 && vulkan_device->extensions().ext_1_2_KHR_image_format_list) {
      image_create_info_last->pNext = &image_format_list_create_info;
      image_create_info_last = reinterpret_cast<VkImageCreateInfo*>(&image_format_list_create_info);
      image_format_list_create_info.viewFormatCount = image_view_format_count;
    }
    if (!vmaCreateImage(vma_allocator_, &image_create_info, &allocation_create_info, &image,
                        &allocation, nullptr)) {
      break;
    }
    if (!image_direct_load) {
      return nullptr;
    }
    image_direct_load = false;
    if (direct_load_create_failures_logged_ < 4) {
      ++direct_load_create_failures_logged_;
      REXGPU_WARN(
          "[s7][TEXDIRECT] creating a {}x{} image (Vulkan format {}) with the storage usage failed - created "
          "without it (loaded through the buffer + copy path)",
          image_create_info.extent.width, image_create_info.extent.height, uint32_t(formats[0]));
    }
  }
  if (image_direct_load) {
    direct_load_creation = VulkanTexture::DirectLoadCreation::kCapable;
  } else {
    direct_load_format = VK_FORMAT_UNDEFINED;
  }

  VulkanTexture* texture = new VulkanTexture(*this, key, image, allocation);
  // wwe2k14 patch (Session 7c, TEXDIRECT).
  texture->SetDirectLoad(direct_load_creation, direct_load_format);
  switch (direct_load_creation) {
    case VulkanTexture::DirectLoadCreation::kNotRequested:
      break;
    case VulkanTexture::DirectLoadCreation::kUnsupported:
      ++direct_load_frame_stats_.created_unsupported;
      break;
    case VulkanTexture::DirectLoadCreation::kUnavailable:
      ++direct_load_frame_stats_.created_unavailable;
      break;
    case VulkanTexture::DirectLoadCreation::kDeviceRejected:
      ++direct_load_frame_stats_.created_rejected;
      break;
    case VulkanTexture::DirectLoadCreation::kCapable:
      ++direct_load_frame_stats_.created_capable;
      break;
  }
  return std::unique_ptr<Texture>(texture);
}

bool VulkanTextureCache::EnsureScaledResolveMemoryCommitted(uint32_t start_unscaled,
                                                            uint32_t length_unscaled,
                                                            uint32_t length_scaled_alignment_log2) {
  uint64_t start_scaled, length_scaled;
  if (!GetScaledResolveRange(start_unscaled, length_unscaled, length_scaled_alignment_log2,
                             start_scaled, length_scaled)) {
    return false;
  }
  return EnsureScaledResolveBufferAllocated(start_scaled, length_scaled);
}

bool VulkanTextureCache::LoadTextureDataFromResidentMemoryImpl(Texture& texture, bool load_base,
                                                               bool load_mips) {
  VulkanTexture& vulkan_texture = static_cast<VulkanTexture&>(texture);
  TextureKey texture_key = vulkan_texture.key();
  EagerRecordLoad(vulkan_texture);  // wwe2k14 patch (Session 22, EAGER)
  {  // wwe2k14 patch (Session 73): a slow frame's log line counts the texture loads in it.
    auto& events = rex::perf::frame_events::Get();
    events.textures.fetch_add(1, std::memory_order_relaxed);
    events.texture_kb.fetch_add(((load_base ? vulkan_texture.GetGuestBaseSize() : 0) +
                                 (load_mips ? vulkan_texture.GetGuestMipsSize() : 0)) >> 10,
                                std::memory_order_relaxed);
  }
  // wwe2k14 patch (Session 66, G1.0 resolve census): a load reads these guest pages (counter only).
  if (command_processor_.NativeAliasCensusActive()) {
    if (load_base) {
      command_processor_.NativeAliasLoad(texture_key.base_page, (vulkan_texture.GetGuestBaseSize() + 0xFFFu) >> 12);
    }
    if (load_mips && texture_key.mip_page) {
      command_processor_.NativeAliasLoad(texture_key.mip_page, (vulkan_texture.GetGuestMipsSize() + 0xFFFu) >> 12);
    }
  }

  // Get the pipeline.
  const HostFormatPair& host_format_pair = GetHostFormatPair(texture_key);
  bool host_format_is_signed;
  if (IsSignedVersionSeparateForFormat(texture_key)) {
    host_format_is_signed = bool(texture_key.signed_separate);
  } else {
    host_format_is_signed = host_format_pair.format_unsigned.load_shader == kLoadShaderIndexUnknown;
  }
  const HostFormat& host_format =
      host_format_is_signed ? host_format_pair.format_signed : host_format_pair.format_unsigned;
  LoadShaderIndex load_shader = host_format.load_shader;
  if (load_shader == kLoadShaderIndexUnknown) {
    return false;
  }
  VkPipeline pipeline = texture_key.scaled_resolve ? load_pipelines_scaled_[load_shader]
                                                   : load_pipelines_[load_shader];
  if (pipeline == VK_NULL_HANDLE) {
    return false;
  }
  const LoadShaderInfo& load_shader_info = GetLoadShaderInfo(load_shader);
  LoadShaderIndex load_shader_float_convert = kLoadShaderIndexUnknown;
  if (host_format.format == VK_FORMAT_R16G16B16A16_SFLOAT) {
    switch (load_shader) {
      case kLoadShaderIndexR10G11B11ToRGBA16:
      case kLoadShaderIndexR11G11B10ToRGBA16:
        load_shader_float_convert = kLoadShaderIndexRGBA16UNormToFloat;
        break;
      case kLoadShaderIndexR10G11B11ToRGBA16SNorm:
      case kLoadShaderIndexR11G11B10ToRGBA16SNorm:
        load_shader_float_convert = kLoadShaderIndexRGBA16SNormToFloat;
        break;
      default:
        break;
    }
  }
  const LoadShaderInfo* load_shader_info_float_convert = nullptr;
  VkPipeline pipeline_float_convert = VK_NULL_HANDLE;
  if (load_shader_float_convert != kLoadShaderIndexUnknown) {
    pipeline_float_convert = texture_key.scaled_resolve
                                 ? load_pipelines_scaled_[load_shader_float_convert]
                                 : load_pipelines_[load_shader_float_convert];
    if (pipeline_float_convert == VK_NULL_HANDLE) {
      return false;
    }
    load_shader_info_float_convert = &GetLoadShaderInfo(load_shader_float_convert);
  }

  // wwe2k14 patch (Session 7c, TEXDIRECT): vulkan_texture_load_direct, read per load. A scaled load that can't be
  // done directly takes the path below, counted by reason (cpf_texdirect b, [s7][TEXDIRECT]).
  // Session 43 (S0b-lite): 1x textures made for direct loads (vulkan_texture_load_direct_unscaled at creation).
  if ((texture_key.scaled_resolve && REXCVAR_GET(vulkan_texture_load_direct)) ||
      (!texture_key.scaled_resolve && REXCVAR_GET(vulkan_texture_load_direct_unscaled) &&
       vulkan_texture.direct_load_creation() != VulkanTexture::DirectLoadCreation::kNotRequested)) {
    DirectLoadFallback direct_load_fallback = kDirectLoadFallbackCount;
    VkDescriptorSet direct_load_descriptor_set = VK_NULL_HANDLE;
    VkFormat direct_load_view_format = GetDirectLoadViewFormat(load_shader);
    if (texture_key.mip_max_level != 0 || !load_base) {
      direct_load_fallback = kDirectLoadFallbackLevels;
    } else if ((texture_key.dimension != xenos::DataDimension::k1D &&
                texture_key.dimension != xenos::DataDimension::k2DOrStacked) ||
               texture_key.GetDepthOrArraySize() != 1) {
      direct_load_fallback = kDirectLoadFallbackLayers;
    } else if (vulkan_texture.guest_layout().packed_level == 0) {
      direct_load_fallback = kDirectLoadFallbackPacked;
    } else if (!texture_key.tiled) {
      // Session 43 (review fix): the direct shaders address tiled memory only.
      direct_load_fallback = kDirectLoadFallbackFormat;
    } else if (direct_load_view_format == VK_FORMAT_UNDEFINED || host_format.block_compressed ||
               FormatInfo::Get(texture_key.format)->block_width != 1 ||
               FormatInfo::Get(texture_key.format)->block_height != 1 ||
               pipeline_float_convert != VK_NULL_HANDLE ||
               vulkan_texture.direct_load_creation() ==
                   VulkanTexture::DirectLoadCreation::kUnsupported) {
      direct_load_fallback = kDirectLoadFallbackFormat;
    } else if (vulkan_texture.direct_load_creation() ==
               VulkanTexture::DirectLoadCreation::kDeviceRejected) {
      // Review fix: counted separately from format.
      direct_load_fallback = kDirectLoadFallbackDevice;
    } else if (vulkan_texture.direct_load_creation() ==
               VulkanTexture::DirectLoadCreation::kUnavailable) {
      // Review fix: counted separately from format.
      direct_load_fallback = kDirectLoadFallbackUnavailable;
    } else if (vulkan_texture.direct_load_creation() !=
                   VulkanTexture::DirectLoadCreation::kCapable ||
               vulkan_texture.direct_load_format() != direct_load_view_format) {
      direct_load_fallback = kDirectLoadFallbackNoStorage;
    } else {
      direct_load_descriptor_set = vulkan_texture.GetDirectLoadDescriptorSet();
      if (direct_load_descriptor_set == VK_NULL_HANDLE) {
        direct_load_fallback = kDirectLoadFallbackDescriptor;
      }
    }
    if (direct_load_fallback == kDirectLoadFallbackCount) {
      // Session 44 (S0b ceiling, measurement only): skip a 1x reload.
      if (!texture_key.scaled_resolve && vulkan_texture.direct_loaded_once() &&
          REXCVAR_GET(vulkan_texture_load_skip_debug)) {
        ++direct_load_skipped_debug_;
        return true;
      }
      // is_3d is false here (single-layer 1D / 2D only).
      bool direct_loaded =
          LoadTextureDataDirect(vulkan_texture, GetDirectLoadShader(load_shader),  // Session 22, TEXDEPTH
                                load_shader, direct_load_descriptor_set, vulkan_texture.force_load_3d_tiling());
      if (direct_loaded) {
        vulkan_texture.SetDirectLoadedOnce();
      }
      return direct_loaded;
    }
    ++direct_load_frame_stats_.fallbacks[direct_load_fallback];
  }

  // Get the guest layout.
  const texture_util::TextureGuestLayout& guest_layout = vulkan_texture.guest_layout();
  xenos::DataDimension dimension = texture_key.dimension;
  bool is_3d = dimension == xenos::DataDimension::k3D;
  bool is_3d_tiling = is_3d || vulkan_texture.force_load_3d_tiling();
  uint32_t width = texture_key.GetWidth();
  uint32_t height = texture_key.GetHeight();
  uint32_t depth_or_array_size = texture_key.GetDepthOrArraySize();
  uint32_t depth = is_3d ? depth_or_array_size : 1;
  uint32_t array_size = is_3d ? 1 : depth_or_array_size;
  xenos::TextureFormat guest_format = texture_key.format;
  const FormatInfo* guest_format_info = FormatInfo::Get(guest_format);
  uint32_t block_width = guest_format_info->block_width;
  uint32_t block_height = guest_format_info->block_height;
  uint32_t bytes_per_block = guest_format_info->bytes_per_block();
  uint32_t level_first = load_base ? 0 : 1;
  uint32_t level_last = load_mips ? texture_key.mip_max_level : 0;
  assert_true(level_first <= level_last);
  uint32_t level_packed = guest_layout.packed_level;
  uint32_t level_stored_first = std::min(level_first, level_packed);
  uint32_t level_stored_last = std::min(level_last, level_packed);
  uint32_t texture_resolution_scale_x = texture_key.scaled_resolve ? draw_resolution_scale_x() : 1;
  uint32_t texture_resolution_scale_y = texture_key.scaled_resolve ? draw_resolution_scale_y() : 1;

  // The loop counter can mean two things depending on whether the packed mip
  // tail is stored as mip 0, because in this case, it would be ambiguous since
  // both the base and the mips would be on "level 0", but stored in separate
  // places.
  uint32_t loop_level_first, loop_level_last;
  if (level_packed == 0) {
    // Packed mip tail is the level 0 - may need to load mip tails for the base,
    // the mips, or both.
    // Loop iteration 0 - base packed mip tail.
    // Loop iteration 1 - mips packed mip tail.
    loop_level_first = uint32_t(level_first != 0);
    loop_level_last = uint32_t(level_last != 0);
  } else {
    // Packed mip tail is not the level 0.
    // Loop iteration is the actual level being loaded.
    loop_level_first = level_stored_first;
    loop_level_last = level_stored_last;
  }

  // Get the host layout and the buffer.
  uint32_t host_block_width = host_format.block_compressed ? block_width : 1;
  uint32_t host_block_height = host_format.block_compressed ? block_height : 1;
  uint32_t host_x_blocks_per_thread = UINT32_C(1)
                                      << load_shader_info.guest_x_blocks_per_thread_log2;
  if (!host_format.block_compressed) {
    // Decompressing guest blocks.
    host_x_blocks_per_thread *= block_width;
  }
  VkDeviceSize host_buffer_size = 0;
  struct HostLayout {
    VkDeviceSize offset_bytes;
    VkDeviceSize slice_size_bytes;
    uint32_t x_pitch_blocks;
    uint32_t y_pitch_blocks;
  };
  HostLayout host_layout_base;
  // Indexing is the same as for guest stored mips:
  // 1...min(level_last, level_packed) if level_packed is not 0, or only 0 if
  // level_packed == 0.
  HostLayout host_layout_mips[xenos::kTextureMaxMips];
  for (uint32_t loop_level = loop_level_first; loop_level <= loop_level_last; ++loop_level) {
    bool is_base = loop_level == 0;
    uint32_t level = (level_packed == 0) ? 0 : loop_level;
    HostLayout& level_host_layout = is_base ? host_layout_base : host_layout_mips[level];
    level_host_layout.offset_bytes = host_buffer_size;
    uint32_t level_guest_x_extent_texels_unscaled;
    uint32_t level_guest_y_extent_texels_unscaled;
    uint32_t level_guest_z_extent_texels;
    if (level == level_packed) {
      // Loading the packed tail for the base or the mips - load the whole tail
      // to copy regions out of it.
      const texture_util::TextureGuestLayout::Level& guest_layout_packed =
          is_base ? guest_layout.base : guest_layout.mips[level];
      level_guest_x_extent_texels_unscaled = guest_layout_packed.x_extent_blocks * block_width;
      level_guest_y_extent_texels_unscaled = guest_layout_packed.y_extent_blocks * block_height;
      level_guest_z_extent_texels = guest_layout_packed.z_extent;
    } else {
      level_guest_x_extent_texels_unscaled = std::max(width >> level, UINT32_C(1));
      level_guest_y_extent_texels_unscaled = std::max(height >> level, UINT32_C(1));
      level_guest_z_extent_texels = std::max(depth >> level, UINT32_C(1));
    }
    level_host_layout.x_pitch_blocks =
        rex::round_up((level_guest_x_extent_texels_unscaled * texture_resolution_scale_x +
                       (host_block_width - 1)) /
                          host_block_width,
                      host_x_blocks_per_thread);
    level_host_layout.y_pitch_blocks =
        (level_guest_y_extent_texels_unscaled * texture_resolution_scale_y +
         (host_block_height - 1)) /
        host_block_height;
    level_host_layout.slice_size_bytes =
        VkDeviceSize(load_shader_info.bytes_per_host_block) * level_host_layout.x_pitch_blocks *
        level_host_layout.y_pitch_blocks * level_guest_z_extent_texels;
    host_buffer_size += level_host_layout.slice_size_bytes * array_size;
  }
  const uint64_t storage_range_limit = command_processor_.GetVulkanDevice()->properties().maxStorageBufferRange;  // (Session 95, Lane E; review: before the scratch acquire)
  if (!VulkanTextureStorageRangeOk(host_buffer_size, storage_range_limit, "texture load destination")) {
    return false;
  }
  // wwe2k14 patch (Session 7c, TEXLOAD): vulkan_texture_load_scratch_ring, latched for this load (see
  // AcquireScratchGpuBufferRing for why the ring keeps every hazard ordered).
  const bool texload_ring = REXCVAR_GET(vulkan_texture_load_scratch_ring);
  const uint64_t texload_ring_barriers_before = command_processor_.scratch_ring_stats().barriers;
  VulkanCommandProcessor::ScratchBufferAcquisition scratch_buffer_acquisition(
      texload_ring
          ? command_processor_.AcquireScratchGpuBufferRing(
                host_buffer_size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT)
          : command_processor_.AcquireScratchGpuBuffer(
                host_buffer_size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT));
  VkBuffer scratch_buffer = scratch_buffer_acquisition.buffer();
  if (scratch_buffer == VK_NULL_HANDLE) {
    return false;
  }

  // Begin loading.
  // TODO(Triang3l): Going from one descriptor to another on per-array-layer
  // or even per-8-depth-slices level to stay within maxStorageBufferRange.
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  VulkanSharedMemory& vulkan_shared_memory = static_cast<VulkanSharedMemory&>(shared_memory());
  // wwe2k14 patch (Session 95, the validation hazard): see vulkan_debug_texture_load_names / _dispatch_barrier.
  const bool texload_names = REXCVAR_GET(vulkan_debug_texture_load_names);
  const bool texload_dispatch_barrier = REXCVAR_GET(vulkan_debug_texture_load_dispatch_barrier);
  uint32_t texload_dispatches = 0;
  const auto texload_before_dispatch = [&]() {
    if (texload_dispatches++ && texload_dispatch_barrier) {
      command_processor_.PushBufferMemoryBarrier(scratch_buffer, 0, VK_WHOLE_SIZE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                                 VK_ACCESS_SHADER_WRITE_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
    }
  };
  if (texload_names) {
    vulkan_device->SetObjectName(VK_OBJECT_TYPE_BUFFER, scratch_buffer, "texture load scratch");
    vulkan_device->SetObjectName(VK_OBJECT_TYPE_BUFFER, vulkan_shared_memory.buffer(), "shared memory");
  }
  std::array<VkWriteDescriptorSet, 3> write_descriptor_sets;
  uint32_t write_descriptor_set_count = 0;
  VkDescriptorSet descriptor_set_dest = command_processor_.AllocateSingleTransientDescriptor(
      VulkanCommandProcessor::SingleTransientDescriptorLayout ::kStorageBufferCompute);
  if (!descriptor_set_dest) {
    return false;
  }
  VkDescriptorBufferInfo write_descriptor_set_dest_buffer_info;
  {
    write_descriptor_set_dest_buffer_info.buffer = scratch_buffer;
    write_descriptor_set_dest_buffer_info.offset = 0;
    write_descriptor_set_dest_buffer_info.range = host_buffer_size;
    VkWriteDescriptorSet& write_descriptor_set_dest =
        write_descriptor_sets[write_descriptor_set_count++];
    write_descriptor_set_dest.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor_set_dest.pNext = nullptr;
    write_descriptor_set_dest.dstSet = descriptor_set_dest;
    write_descriptor_set_dest.dstBinding = 0;
    write_descriptor_set_dest.dstArrayElement = 0;
    write_descriptor_set_dest.descriptorCount = 1;
    write_descriptor_set_dest.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write_descriptor_set_dest.pImageInfo = nullptr;
    write_descriptor_set_dest.pBufferInfo = &write_descriptor_set_dest_buffer_info;
    write_descriptor_set_dest.pTexelBufferView = nullptr;
  }
  // TODO(Triang3l): Use a single 512 MB shared memory binding if possible.
  // TODO(Triang3l): Scaled resolve buffer bindings.
  // Aligning because if the data for a vector in a storage buffer is provided
  // partially, the value read may still be (0, 0, 0, 0), and small (especially
  // linear) textures won't be loaded correctly.
  uint32_t source_length_alignment = UINT32_C(1) << load_shader_info.source_bpe_log2;
  VkDescriptorSet descriptor_set_source_base = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_source_mips = VK_NULL_HANDLE;
  uint64_t source_bytes = 0;  // wwe2k14 patch (Session 7c, CAUSES): the source ranges the load reads
  VkDescriptorBufferInfo write_descriptor_set_source_base_buffer_info;
  VkDescriptorBufferInfo write_descriptor_set_source_mips_buffer_info;
  if (level_first == 0) {
    descriptor_set_source_base = command_processor_.AllocateSingleTransientDescriptor(
        VulkanCommandProcessor::SingleTransientDescriptorLayout ::kStorageBufferCompute);
    if (!descriptor_set_source_base) {
      return false;
    }
    uint32_t source_base_start_unscaled = texture_key.base_page << 12;
    uint64_t source_base_start;
    uint64_t source_base_range;
    if (texture_key.scaled_resolve) {
      if (!GetScaledResolveRange(source_base_start_unscaled, vulkan_texture.GetGuestBaseSize(),
                                 load_shader_info.source_bpe_log2, source_base_start,
                                 source_base_range)) {
        return false;
      }
      write_descriptor_set_source_base_buffer_info.buffer = scaled_resolve_buffer_;
    } else {
      source_base_start = source_base_start_unscaled;
      source_base_range = rex::align(uint64_t(vulkan_texture.GetGuestBaseSize()),
                                     uint64_t(source_length_alignment));
      if (source_base_range > uint64_t(SharedMemory::kBufferSize) - source_base_start) {
        return false;
      }
      write_descriptor_set_source_base_buffer_info.buffer = vulkan_shared_memory.buffer();
    }
    write_descriptor_set_source_base_buffer_info.offset = source_base_start;
    write_descriptor_set_source_base_buffer_info.range = source_base_range;
    if (!VulkanTextureStorageRangeOk(source_base_range, storage_range_limit, "texture load source (base)")) {
      return false;
    }
    source_bytes += source_base_range;  // wwe2k14 patch (Session 7c, CAUSES)
    VkWriteDescriptorSet& write_descriptor_set_source_base =
        write_descriptor_sets[write_descriptor_set_count++];
    write_descriptor_set_source_base.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor_set_source_base.pNext = nullptr;
    write_descriptor_set_source_base.dstSet = descriptor_set_source_base;
    write_descriptor_set_source_base.dstBinding = 0;
    write_descriptor_set_source_base.dstArrayElement = 0;
    write_descriptor_set_source_base.descriptorCount = 1;
    write_descriptor_set_source_base.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write_descriptor_set_source_base.pImageInfo = nullptr;
    write_descriptor_set_source_base.pBufferInfo = &write_descriptor_set_source_base_buffer_info;
    write_descriptor_set_source_base.pTexelBufferView = nullptr;
  }
  if (level_last != 0) {
    descriptor_set_source_mips = command_processor_.AllocateSingleTransientDescriptor(
        VulkanCommandProcessor::SingleTransientDescriptorLayout ::kStorageBufferCompute);
    if (!descriptor_set_source_mips) {
      return false;
    }
    uint32_t source_mips_start_unscaled = texture_key.mip_page << 12;
    uint64_t source_mips_start;
    uint64_t source_mips_range;
    if (texture_key.scaled_resolve) {
      if (!GetScaledResolveRange(source_mips_start_unscaled, vulkan_texture.GetGuestMipsSize(),
                                 load_shader_info.source_bpe_log2, source_mips_start,
                                 source_mips_range)) {
        return false;
      }
      write_descriptor_set_source_mips_buffer_info.buffer = scaled_resolve_buffer_;
    } else {
      source_mips_start = source_mips_start_unscaled;
      source_mips_range = rex::align(uint64_t(vulkan_texture.GetGuestMipsSize()),
                                     uint64_t(source_length_alignment));
      if (source_mips_range > uint64_t(SharedMemory::kBufferSize) - source_mips_start) {
        return false;
      }
      write_descriptor_set_source_mips_buffer_info.buffer = vulkan_shared_memory.buffer();
    }
    write_descriptor_set_source_mips_buffer_info.offset = source_mips_start;
    write_descriptor_set_source_mips_buffer_info.range = source_mips_range;
    if (!VulkanTextureStorageRangeOk(source_mips_range, storage_range_limit, "texture load source (mips)")) {
      return false;
    }
    source_bytes += source_mips_range;  // wwe2k14 patch (Session 7c, CAUSES)
    VkWriteDescriptorSet& write_descriptor_set_source_mips =
        write_descriptor_sets[write_descriptor_set_count++];
    write_descriptor_set_source_mips.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor_set_source_mips.pNext = nullptr;
    write_descriptor_set_source_mips.dstSet = descriptor_set_source_mips;
    write_descriptor_set_source_mips.dstBinding = 0;
    write_descriptor_set_source_mips.dstArrayElement = 0;
    write_descriptor_set_source_mips.descriptorCount = 1;
    write_descriptor_set_source_mips.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write_descriptor_set_source_mips.pImageInfo = nullptr;
    write_descriptor_set_source_mips.pBufferInfo = &write_descriptor_set_source_mips_buffer_info;
    write_descriptor_set_source_mips.pTexelBufferView = nullptr;
  }
  if (write_descriptor_set_count) {
    dfn.vkUpdateDescriptorSets(device, write_descriptor_set_count, write_descriptor_sets.data(), 0,
                               nullptr);
  }
  if (texture_key.scaled_resolve) {
    UseScaledResolveBufferForRead();
  } else {
    // wwe2k14 patch (Session 7b, BARNARROW): read by the load dispatches (compute stage).
    vulkan_shared_memory.Use(VulkanSharedMemory::Usage::kRead, {}, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  }

  // Submit the copy buffer population commands.

  DeferredCommandBuffer& command_buffer = command_processor_.deferred_command_buffer();

  // wwe2k14 patch (Session 7c, CAUSES): measurement only - the load records its commands from here on (no early
  // return after this point).
  {
    VulkanCommandProcessor::CauseCounts& causes = command_processor_.cause_counts();
    ++causes.tex_loads;
    if (texture_key.scaled_resolve) {
      ++causes.tex_loads_scaled;
    }
    causes.tex_load_bytes += source_bytes;
  }
  // wwe2k14 patch (Session 7c, TEXLOAD): GPU marks (gpu_anatomy_gpu_marks): the load records its commands from here
  // on (no early return after this point).
  const bool texload_marks = command_processor_.gpu_marks_active();
  if (texload_marks) {
    command_processor_.GpuMarkTextureLoad(VulkanCommandProcessor::TextureLoadMark::kBegin);
  }

  command_processor_.BindExternalComputePipeline(pipeline);

  command_buffer.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, load_pipeline_layout_,
                                         kLoadDescriptorSetIndexDestination, 1,
                                         &descriptor_set_dest, 0, nullptr);

  VkDescriptorSet descriptor_set_source_current = VK_NULL_HANDLE;

  LoadConstants load_constants;
  // 3 bits for each.
  assert_true(texture_resolution_scale_x <= 7);
  assert_true(texture_resolution_scale_y <= 7);
  load_constants.is_tiled_3d_endian_scale =
      uint32_t(texture_key.tiled) | (uint32_t(is_3d_tiling) << 1) |
      (uint32_t(texture_key.endianness) << 2) | (texture_resolution_scale_x << 4) |
      (texture_resolution_scale_y << 7);

  uint32_t guest_x_blocks_per_group_log2 = load_shader_info.GetGuestXBlocksPerGroupLog2();
  for (uint32_t loop_level = loop_level_first; loop_level <= loop_level_last; ++loop_level) {
    bool is_base = loop_level == 0;
    uint32_t level = (level_packed == 0) ? 0 : loop_level;

    VkDescriptorSet descriptor_set_source =
        is_base ? descriptor_set_source_base : descriptor_set_source_mips;
    if (descriptor_set_source_current != descriptor_set_source) {
      descriptor_set_source_current = descriptor_set_source;
      command_buffer.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, load_pipeline_layout_,
                                             kLoadDescriptorSetIndexSource, 1,
                                             &descriptor_set_source, 0, nullptr);
    }

    // TODO(Triang3l): guest_offset relative to the storage buffer origin.
    load_constants.guest_offset = 0;
    if (!is_base) {
      load_constants.guest_offset += guest_layout.mip_offsets_bytes[level] *
                                     (texture_resolution_scale_x * texture_resolution_scale_y);
    }
    const texture_util::TextureGuestLayout::Level& level_guest_layout =
        is_base ? guest_layout.base : guest_layout.mips[level];
    uint32_t level_guest_pitch = level_guest_layout.row_pitch_bytes;
    if (texture_key.tiled) {
      // Shaders expect pitch in blocks for tiled textures.
      level_guest_pitch /= bytes_per_block;
      assert_zero(level_guest_pitch & (xenos::kTextureTileWidthHeight - 1));
    }
    load_constants.guest_pitch_aligned = level_guest_pitch;
    load_constants.guest_z_stride_block_rows_aligned = level_guest_layout.z_slice_stride_block_rows;
    assert_true(!is_3d_tiling || !(load_constants.guest_z_stride_block_rows_aligned &
                                   (xenos::kTextureTileWidthHeight - 1)));

    uint32_t level_width, level_height, level_depth;
    if (level == level_packed) {
      // This is the packed mip tail, containing not only the specified level,
      // but also other levels at different offsets - load the entire needed
      // extents.
      level_width = level_guest_layout.x_extent_blocks * block_width;
      level_height = level_guest_layout.y_extent_blocks * block_height;
      level_depth = level_guest_layout.z_extent;
    } else {
      level_width = std::max(width >> level, UINT32_C(1));
      level_height = std::max(height >> level, UINT32_C(1));
      level_depth = std::max(depth >> level, UINT32_C(1));
    }
    load_constants.size_blocks[0] =
        (level_width + (block_width - 1)) / block_width * texture_resolution_scale_x;
    load_constants.size_blocks[1] =
        (level_height + (block_height - 1)) / block_height * texture_resolution_scale_y;
    load_constants.size_blocks[2] = level_depth;
    load_constants.height_texels = level_height;

    uint32_t group_count_x =
        (load_constants.size_blocks[0] + ((UINT32_C(1) << guest_x_blocks_per_group_log2) - 1)) >>
        guest_x_blocks_per_group_log2;
    uint32_t group_count_y =
        (load_constants.size_blocks[1] + ((UINT32_C(1) << kLoadGuestYBlocksPerGroupLog2) - 1)) >>
        kLoadGuestYBlocksPerGroupLog2;

    // TODO(Triang3l): host_offset relative to the storage buffer origin.
    const HostLayout& level_host_layout = is_base ? host_layout_base : host_layout_mips[level];
    load_constants.host_offset = uint32_t(level_host_layout.offset_bytes);
    load_constants.host_pitch =
        load_shader_info.bytes_per_host_block * level_host_layout.x_pitch_blocks;

    command_buffer.CmdVkPushConstants(load_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                      sizeof(load_constants), &load_constants);

    uint32_t level_array_slice_stride_bytes_scaled =
        level_guest_layout.array_slice_stride_bytes *
        (texture_resolution_scale_x * texture_resolution_scale_y);
    for (uint32_t slice = 0; slice < array_size; ++slice) {
      if (slice != 0) {
        command_buffer.CmdVkPushConstants(load_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                                          offsetof(LoadConstants, guest_offset),
                                          sizeof(load_constants.guest_offset),
                                          &load_constants.guest_offset);
        command_buffer.CmdVkPushConstants(load_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                                          offsetof(LoadConstants, host_offset),
                                          sizeof(load_constants.host_offset),
                                          &load_constants.host_offset);
      }
      texload_before_dispatch();
      command_processor_.SubmitBarriers(true);
      command_buffer.CmdVkDispatch(group_count_x, group_count_y, load_constants.size_blocks[2]);
      load_constants.guest_offset += level_array_slice_stride_bytes_scaled;
      load_constants.host_offset += uint32_t(level_host_layout.slice_size_bytes);
    }
  }
  if (texload_names) {
    static uint64_t loads = 0, multi = 0, extra = 0;
    ++loads;
    multi += texload_dispatches > 1;
    extra += texload_dispatches > 1 ? texload_dispatches - 1 : 0;
    if (texload_dispatches > 1 && !(multi & 63)) {
      REXGPU_INFO("[tex] load dispatches: {} loads, {} of more than one dispatch, {} dispatches after a load's first (dispatch barrier {})", loads, multi, extra, texload_dispatch_barrier ? "on" : "off");
    }
  }
  if (texload_marks) {
    // wwe2k14 patch (Session 7c, TEXLOAD).
    command_processor_.GpuMarkTextureLoad(VulkanCommandProcessor::TextureLoadMark::kDispatch);
  }

  if (pipeline_float_convert != VK_NULL_HANDLE) {
    assert_not_null(load_shader_info_float_convert);
    static std::array<uint8_t, 64> float_fallback_log_mask = {};
    uint8_t float_fallback_log_bit = host_format_is_signed ? uint8_t(1 << 1) : uint8_t(1 << 0);
    uint8_t& float_fallback_log_state = float_fallback_log_mask[uint32_t(texture_key.format)];
    if ((float_fallback_log_state & float_fallback_log_bit) == 0) {
      float_fallback_log_state |= float_fallback_log_bit;
      REXGPU_INFO("VulkanTextureCache: Using two-pass float fallback for format {} ({})",
                  FormatInfo::Get(texture_key.format)->name,
                  host_format_is_signed ? "signed" : "unsigned");
    }
    // Make the normalized intermediate data visible for conversion and keep it
    // writable for in-place conversion to float16.
    command_processor_.PushBufferMemoryBarrier(
        scratch_buffer, 0, VK_WHOLE_SIZE,
        scratch_buffer_acquisition.SetStageMask(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        scratch_buffer_acquisition.SetAccessMask(VK_ACCESS_SHADER_READ_BIT |
                                                 VK_ACCESS_SHADER_WRITE_BIT),
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    command_processor_.BindExternalComputePipeline(pipeline_float_convert);
    command_buffer.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, load_pipeline_layout_,
                                           kLoadDescriptorSetIndexDestination, 1,
                                           &descriptor_set_dest, 0, nullptr);
    command_buffer.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, load_pipeline_layout_,
                                           kLoadDescriptorSetIndexSource, 1, &descriptor_set_dest,
                                           0, nullptr);

    LoadConstants load_constants_float_convert;
    // Linear source and destination in the same scratch buffer, 1x1 scaling.
    load_constants_float_convert.is_tiled_3d_endian_scale =
        (uint32_t(is_3d) << 1) | (UINT32_C(1) << 4) | (UINT32_C(1) << 7);
    uint32_t guest_x_blocks_per_group_log2_float_convert =
        load_shader_info_float_convert->GetGuestXBlocksPerGroupLog2();

    for (uint32_t loop_level = loop_level_first; loop_level <= loop_level_last; ++loop_level) {
      bool is_base = loop_level == 0;
      uint32_t level = (level_packed == 0) ? 0 : loop_level;

      const texture_util::TextureGuestLayout::Level& level_guest_layout =
          is_base ? guest_layout.base : guest_layout.mips[level];
      uint32_t level_width, level_height, level_depth;
      if (level == level_packed) {
        level_width = level_guest_layout.x_extent_blocks * block_width;
        level_height = level_guest_layout.y_extent_blocks * block_height;
        level_depth = level_guest_layout.z_extent;
      } else {
        level_width = std::max(width >> level, UINT32_C(1));
        level_height = std::max(height >> level, UINT32_C(1));
        level_depth = std::max(depth >> level, UINT32_C(1));
      }
      load_constants_float_convert.size_blocks[0] =
          (level_width + (block_width - 1)) / block_width * texture_resolution_scale_x;
      load_constants_float_convert.size_blocks[1] =
          (level_height + (block_height - 1)) / block_height * texture_resolution_scale_y;
      load_constants_float_convert.size_blocks[2] = level_depth;
      load_constants_float_convert.height_texels = level_height;

      uint32_t group_count_x =
          (load_constants_float_convert.size_blocks[0] +
           ((UINT32_C(1) << guest_x_blocks_per_group_log2_float_convert) - 1)) >>
          guest_x_blocks_per_group_log2_float_convert;
      uint32_t group_count_y = (load_constants_float_convert.size_blocks[1] +
                                ((UINT32_C(1) << kLoadGuestYBlocksPerGroupLog2) - 1)) >>
                               kLoadGuestYBlocksPerGroupLog2;

      const HostLayout& level_host_layout = is_base ? host_layout_base : host_layout_mips[level];
      load_constants_float_convert.guest_offset = uint32_t(level_host_layout.offset_bytes);
      load_constants_float_convert.guest_pitch_aligned =
          load_shader_info_float_convert->bytes_per_host_block * level_host_layout.x_pitch_blocks;
      load_constants_float_convert.guest_z_stride_block_rows_aligned =
          level_host_layout.y_pitch_blocks;
      load_constants_float_convert.host_offset = uint32_t(level_host_layout.offset_bytes);
      load_constants_float_convert.host_pitch =
          load_shader_info_float_convert->bytes_per_host_block * level_host_layout.x_pitch_blocks;

      command_buffer.CmdVkPushConstants(load_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                        sizeof(load_constants_float_convert),
                                        &load_constants_float_convert);

      uint32_t level_array_slice_stride_bytes = uint32_t(level_host_layout.slice_size_bytes);
      for (uint32_t slice = 0; slice < array_size; ++slice) {
        if (slice != 0) {
          command_buffer.CmdVkPushConstants(load_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                                            offsetof(LoadConstants, guest_offset),
                                            sizeof(load_constants_float_convert.guest_offset),
                                            &load_constants_float_convert.guest_offset);
          command_buffer.CmdVkPushConstants(load_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                                            offsetof(LoadConstants, host_offset),
                                            sizeof(load_constants_float_convert.host_offset),
                                            &load_constants_float_convert.host_offset);
        }
        command_processor_.SubmitBarriers(true);
        command_buffer.CmdVkDispatch(group_count_x, group_count_y,
                                     load_constants_float_convert.size_blocks[2]);
        load_constants_float_convert.guest_offset += level_array_slice_stride_bytes;
        load_constants_float_convert.host_offset += level_array_slice_stride_bytes;
      }
    }
    if (texload_marks) {
      // wwe2k14 patch (Session 7c, TEXLOAD).
      command_processor_.GpuMarkTextureLoad(VulkanCommandProcessor::TextureLoadMark::kConvert);
    }
  }

  // Submit copying from the copy buffer to the host texture.
  command_processor_.PushBufferMemoryBarrier(
      scratch_buffer, 0, VK_WHOLE_SIZE,
      scratch_buffer_acquisition.SetStageMask(VK_PIPELINE_STAGE_TRANSFER_BIT),
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      scratch_buffer_acquisition.SetAccessMask(VK_ACCESS_TRANSFER_READ_BIT),
      VK_ACCESS_TRANSFER_READ_BIT);
  vulkan_texture.MarkAsUsed();
  vulkan_texture.native_used = true;  // wwe2k14 patch (Session 80: loaded from memory - conservatively a use)
  VulkanTexture::Usage texture_old_usage =
      vulkan_texture.SetUsage(VulkanTexture::Usage::kTransferDestination);
  // wwe2k14 patch (Session 59 review): a deferred native write into it goes first (a batched fill, then a reload).
  if (texture_old_usage == VulkanTexture::Usage::kComputeStorageWrite) command_processor_.NativeFlushDeferredResolves(&vulkan_texture);
  // wwe2k14 patch (Session 7c, TEXLOAD): what the pre-copy image barrier does (texture load CSV): none (the image is
  // still a transfer destination), undefined (never used: no contents), preserve, discard, or waw (see below).
  const char* texload_transition = "none";
  if (texture_old_usage != VulkanTexture::Usage::kTransferDestination) {
    VkPipelineStageFlags texture_src_stage_mask, texture_dst_stage_mask;
    VkAccessFlags texture_src_access_mask, texture_dst_access_mask;
    VkImageLayout texture_old_layout, texture_new_layout;
    GetTextureUsageMasks(texture_old_usage, texture_src_stage_mask, texture_src_access_mask,
                         texture_old_layout);
    GetTextureUsageMasks(VulkanTexture::Usage::kTransferDestination, texture_dst_stage_mask,
                         texture_dst_access_mask, texture_new_layout);
    // wwe2k14 patch (Session 7c, TEXLOAD): vulkan_texture_load_discard_old. The barrier covers the whole image (all
    // mip levels and array layers: the usage, and with it the layout, is tracked per image). The copy below writes
    // levels level_first...level_last, each over its full extent ((width * scale) >> level etc., the extents the
    // image was created with in CreateTexture) and all array_size layers (= the image's array layers), right
    // after this barrier and before anything else touches the image. So when level_first is 0 and level_last is
    // the last level of the image, every texel of every subresource the barrier covers is overwritten by the copy,
    // and the old contents can be discarded (UNDEFINED old layout) without changing any texel the image holds
    // after the copy. The stage and access masks stay the same, so the copy still waits for the earlier reads of
    // the image. Not for the 3D-as-2D wrappers (force_load_3d_tiling), whose image isn't created by
    // CreateTexture (and whose first load is from the undefined layout anyway).
    if (texture_old_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
      texload_transition = "undefined";
    } else {
      ++texload_stats_.transitions;
      texload_transition = "preserve";
      if (level_first == 0 && level_last == texture_key.mip_max_level &&
          !vulkan_texture.force_load_3d_tiling()) {
        ++texload_stats_.rewrite_all;
        if (REXCVAR_GET(vulkan_texture_load_discard_old)) {
          texture_old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
          ++texload_stats_.discards;
          texload_transition = "discard";
        }
      }
    }
    command_processor_.PushImageMemoryBarrier(
        vulkan_texture.image(), ui::vulkan::util::InitializeSubresourceRange(),
        texture_src_stage_mask, texture_dst_stage_mask, texture_src_access_mask,
        texture_dst_access_mask, texture_old_layout, texture_new_layout);
  } else if (texload_ring || vulkan_texture.last_load_used_scratch_ring()) {
    // wwe2k14 patch (Session 7c, TEXLOAD): the image is still a transfer destination (loaded again before being
    // sampled; only loads make it one, so the last command on it is the previous load's copy), so nothing orders
    // the previous copy into it before this one. When both loads use the single scratch buffer, its reuse barrier
    // (transfer -> compute before this load's dispatches, then compute -> transfer before this copy) does, as
    // before. When either load used the ring, that barrier may be missing (this load's ring entry may need none;
    // after vulkan_texture_load_scratch_ring is turned off, the single buffer may be new or grown and push none, or
    // its last use may not have been a transfer), so an explicit write-after-write barrier keeps the order. With the
    // ring never used, this changes nothing.
    command_processor_.PushImageMemoryBarrier(
        vulkan_texture.image(), ui::vulkan::util::InitializeSubresourceRange(),
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        false);
    ++texload_stats_.ring_waw_barriers;
    texload_transition = "waw";
  }
  // wwe2k14 patch (Session 7c, TEXLOAD): for the barrier above at the image's next load (nothing returns between
  // here and the copy).
  vulkan_texture.SetLastLoadUsedScratchRing(texload_ring);
  command_processor_.SubmitBarriers(true);
  VkBufferImageCopy* copy_regions = command_buffer.CmdCopyBufferToImageEmplace(
      scratch_buffer, vulkan_texture.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      level_last - level_first + 1);
  for (uint32_t level = level_first; level <= level_last; ++level) {
    VkBufferImageCopy& copy_region = copy_regions[level - level_first];
    const HostLayout& level_host_layout =
        level != 0 ? host_layout_mips[std::min(level, level_packed)] : host_layout_base;
    copy_region.bufferOffset = level_host_layout.offset_bytes;
    if (level >= level_packed) {
      uint32_t level_offset_blocks_x, level_offset_blocks_y, level_offset_z;
      texture_util::GetPackedMipOffset(width, height, depth, guest_format, level,
                                       level_offset_blocks_x, level_offset_blocks_y,
                                       level_offset_z);
      uint32_t level_offset_host_blocks_x = texture_resolution_scale_x * level_offset_blocks_x;
      uint32_t level_offset_host_blocks_y = texture_resolution_scale_y * level_offset_blocks_y;
      if (!host_format.block_compressed) {
        level_offset_host_blocks_x *= block_width;
        level_offset_host_blocks_y *= block_height;
      }
      copy_region.bufferOffset +=
          load_shader_info.bytes_per_host_block *
          (level_offset_host_blocks_x +
           level_host_layout.x_pitch_blocks *
               (level_offset_host_blocks_y +
                level_host_layout.y_pitch_blocks * VkDeviceSize(level_offset_z)));
    }
    copy_region.bufferRowLength = level_host_layout.x_pitch_blocks * host_block_width;
    copy_region.bufferImageHeight = level_host_layout.y_pitch_blocks * host_block_height;
    copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.imageSubresource.mipLevel = level;
    copy_region.imageSubresource.baseArrayLayer = 0;
    copy_region.imageSubresource.layerCount = array_size;
    copy_region.imageOffset.x = 0;
    copy_region.imageOffset.y = 0;
    copy_region.imageOffset.z = 0;
    copy_region.imageExtent.width =
        std::max((width * texture_resolution_scale_x) >> level, UINT32_C(1));
    copy_region.imageExtent.height =
        std::max((height * texture_resolution_scale_y) >> level, UINT32_C(1));
    copy_region.imageExtent.depth = std::max(depth >> level, UINT32_C(1));
  }

  // wwe2k14 patch (Session 7c, TEXLOAD): measurement only. The bytes the copy writes into the image (whole host
  // blocks of each region, all layers), read from the regions before any other command is recorded (the pointer
  // is into the deferred command buffer's storage).
  uint64_t texload_bytes = 0;
  for (uint32_t i = 0; i <= level_last - level_first; ++i) {
    const VkBufferImageCopy& copy_region = copy_regions[i];
    texload_bytes += uint64_t(load_shader_info.bytes_per_host_block) *
                     ((copy_region.imageExtent.width + (host_block_width - 1)) / host_block_width) *
                     ((copy_region.imageExtent.height + (host_block_height - 1)) / host_block_height) *
                     copy_region.imageExtent.depth * copy_region.imageSubresource.layerCount;
    // wwe2k14 patch (Session 93): blocks transcoded per format (BC1 -> ASTC, BC4 -> EAC R11, BC5 -> EAC RG11), counted
    // here with the other reads of copy_regions (before GpuMarkTextureLoad may reallocate the command stream); reported
    // every 600 frames by EndFrame. Punch-through BC1 blocks are decided on the GPU: only the offline test counts them.
    if (load_shader == kLoadShaderIndexDXT1ToASTC4x4 || load_shader == kLoadShaderIndexDXT5AToEACR11 ||
        load_shader == kLoadShaderIndexDXNToEACRG11) {
      g_transcode_blocks[load_shader == kLoadShaderIndexDXT1ToASTC4x4
                             ? 0
                             : (load_shader == kLoadShaderIndexDXT5AToEACR11 ? 1 : 2)] +=
          uint64_t((copy_region.imageExtent.width + 3) / 4) * ((copy_region.imageExtent.height + 3) / 4) *
          copy_region.imageExtent.depth * copy_region.imageSubresource.layerCount;
    }
  }
  if (load_shader == kLoadShaderIndexDXT1ToASTC4x4 || load_shader == kLoadShaderIndexDXT5AToEACR11 ||
      load_shader == kLoadShaderIndexDXNToEACRG11) {
    ++g_transcode_loads[load_shader == kLoadShaderIndexDXT1ToASTC4x4
                            ? 0
                            : (load_shader == kLoadShaderIndexDXT5AToEACR11 ? 1 : 2)];
  }
  if (texload_marks) {
    command_processor_.GpuMarkTextureLoad(VulkanCommandProcessor::TextureLoadMark::kCopy);
  }
  // wwe2k14 patch (Session 45, NATIVE FRAME): the native renderer's frame census.
  ++native_load_counters_.loads;
  if (texture_key.scaled_resolve || IsPageResolvedUnscaled(texture_key.base_page << 12)) {
    ++native_load_counters_.resolved_loads;
    native_load_counters_.resolved_bytes += texload_bytes;
    // wwe2k14 patch (Session 47, NATIVE TRUTH M6): the texture a later truth resolve to this page may fill by a copy
    // (only with gpu_native_truth: learned from copy loads too).
    // (Review: only while truth frames run - else N3a's direct-load keys stay.)
    if (REXCVAR_GET(gpu_native_truth) && command_processor_.NativeTruthActive()) {
      NativeLearnTextureKey(vulkan_texture);  // (Session 47: scaled ones too - the 4x fills)
    }
    if (command_processor_.NativeSegmentLogActive()) {  // Session 45: the segment log
      REXGPU_INFO("[native][seg] texture load (copy): page 0x{:05X} format {} {}x{} pitch {} endian {} tiled {} mips {} "
                  "array {} scaled {}, {} bytes",
                  uint32_t(texture_key.base_page), uint32_t(texture_key.format), texture_key.GetWidth(),
                  texture_key.GetHeight(), uint32_t(texture_key.pitch), uint32_t(texture_key.endianness),
                  uint32_t(texture_key.tiled), uint32_t(texture_key.mip_max_level),
                  uint32_t(texture_key.depth_or_array_size_minus_1), uint32_t(texture_key.scaled_resolve),
                  texload_bytes);
    }
  }
  TexLoadStats& texload_stats = texload_stats_;
  ++texload_stats.frame_loads;
  ++texload_stats.loads;
  if (texture_key.scaled_resolve) {
    ++texload_stats.frame_loads_scaled;
    ++texload_stats.loads_scaled;
  }
  texload_stats.frame_bytes += texload_bytes;
  texload_stats.bytes += texload_bytes;
  texload_stats.float_converts += pipeline_float_convert != VK_NULL_HANDLE ? 1 : 0;
  // What invalidated the texture since its last load (cleared once this load is committed).
  const uint32_t texload_invalidated = vulkan_texture.invalidated_since_load();
  texload_stats.invalidated_gpu += (texload_invalidated & Texture::kInvalidatedByGpu) ? 1 : 0;
  texload_stats.invalidated_cpu += (texload_invalidated & Texture::kInvalidatedByCpu) ? 1 : 0;
  texload_stats.first_loads += texload_invalidated ? 0 : 1;
  const uint32_t texload_ring_index = scratch_buffer_acquisition.ring_index();
  texload_stats.ring_loads += texload_ring_index != UINT32_MAX ? 1 : 0;
  if (texload_csv_) {
    // One row per load of a frame-anatomy capture (see TexLoadAnatomyFrameBoundary for the columns).
    static const char* const kInvalidatedNames[] = {"new", "cpu", "gpu", "gpu+cpu"};
    fmt::format_to(std::back_inserter(texload_csv_text_),
                   "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},0x{:08X},0x{:08X},{},{},{},{},{},{},{}\n",
                   texload_csv_frame_, command_processor_.anatomy_row(),
                   command_processor_.GetCurrentSubmission(), width, height, depth_or_array_size,
                   texture_key.GetLogDimensionName(), texture_key.mip_max_level + 1, level_first, level_last,
                   guest_format_info->name, uint32_t(host_format.format), texture_key.scaled_resolve ? 1 : 0,
                   texture_resolution_scale_x, texture_resolution_scale_y, texture_key.tiled ? 1 : 0,
                   uint32_t(texture_key.base_page) << 12, uint32_t(texture_key.mip_page) << 12, texload_bytes,
                   host_buffer_size,
                   kInvalidatedNames[texload_invalidated & 3], texload_transition,
                   pipeline_float_convert != VK_NULL_HANDLE ? 1 : 0,
                   texload_ring_index != UINT32_MAX ? int32_t(texload_ring_index) : -1,
                   (texload_ring_index != UINT32_MAX &&
                    command_processor_.scratch_ring_stats().barriers != texload_ring_barriers_before)
                       ? 1
                       : 0);
  }

  return true;
}

// wwe2k14 patch (Session 7c, TEXLOAD): measurement only (nothing here changes what is loaded or drawn).

// wwe2k14 patch (Session 45, NATIVE TEXWRITE): see the header.
VkDescriptorSet VulkanTextureCache::NativeTextureForResolve(uint32_t base_page, uint32_t pitch_div_32,
                                                            uint32_t endianness, uint32_t& width_out,
                                                            uint32_t& height_out, void*& handle_out,
                                                            bool allow_scaled, xenos::TextureFormat format,
                                                            bool allow_layer, int32_t* layer_out) {
  const std::array<uint32_t, 4> cover = NativeTakeCover();  // wwe2k14 patch (Session 52)
  handle_out = nullptr;
  native_last_copy_untouched_ = false;  // wwe2k14 patch (Session 80: set only where it is prepared for the write)
  if (layer_out) *layer_out = -1;
  // Session 45: gpu_native_segment_log lists why a served resolve does not fill its texture.
  const auto refuse = [&](const char* why, const TextureKey* key) -> VkDescriptorSet {
    if (command_processor_.NativeSegmentLogActive()) {
      REXGPU_INFO("[native][seg] texture fill refused: page 0x{:05X} pitch {} endian {}: {}{}", base_page,
                  pitch_div_32, endianness, why,
                  key ? fmt::format(" (texture format {} {}x{} pitch {} endian {} tiled {} mips {} array {})",
                                    uint32_t(key->format), key->GetWidth(), key->GetHeight(), uint32_t(key->pitch),
                                    uint32_t(key->endianness), uint32_t(key->tiled), uint32_t(key->mip_max_level),
                                    uint32_t(key->depth_or_array_size_minus_1))
                      : std::string());
    }
    return VK_NULL_HANDLE;
  };
  // Session 79 (gpu_native_truth_fill_cube_faces): one face of a cube. Its other faces keep their texels, so the cube
  // must be up to date before (the write covers one face: no cover shortcut); afterwards it holds what memory holds.
  if (allow_layer && REXCVAR_GET(gpu_native_truth_fill_cube_faces)) {
    auto face = native_resolve_layer_keys_.find(base_page);
    // (Review fix: a face that cannot be filled falls through to the 2D lookup below - a 2D texture may be over the
    // page by now - and the entry of a cube that is gone is dropped.)
    const char* face_why = nullptr;
    VulkanTexture* face_texture = nullptr;
    VkDescriptorSet face_set = VK_NULL_HANDLE;
    TextureKey key;
    uint32_t layer = 0;
    if (face != native_resolve_layer_keys_.end()) {
      key = face->second.key;
      layer = face->second.layer;
      if (key.format != format || format != xenos::TextureFormat::k_8_8_8_8 || key.scaled_resolve ||
          uint32_t(key.endianness) != endianness || key.pitch != pitch_div_32) {
        face_why = "cube face key";
      } else if (Texture* found = NativeFindTexture(key); !found) {
        face_why = "cube gone";
        native_resolve_layer_keys_.erase(face);
      } else {
        face_texture = static_cast<VulkanTexture*>(found);
        if (face_texture->direct_load_creation() != VulkanTexture::DirectLoadCreation::kCapable ||
            face_texture->direct_load_format() != VK_FORMAT_R32_UINT) {
          face_why = "cube without the R32_UINT storage view";
        } else if (!NativeTextureBaseUpToDate(*face_texture)) {
          face_why = "cube not up to date";
        } else if ((face_set = face_texture->GetLayerStorageDescriptorSet(layer)) == VK_NULL_HANDLE) {
          face_why = "no storage set for the face";
        }
      }
      if (face_why && command_processor_.NativeSegmentLogActive()) {
        REXGPU_INFO("[native][seg] texture fill refused: cube face page 0x{:05X}: {} (texture format {} {}x{} pitch {} "
                    "array {}) - trying a 2D texture", base_page, face_why, uint32_t(key.format), key.GetWidth(),
                    key.GetHeight(), uint32_t(key.pitch), uint32_t(key.depth_or_array_size_minus_1));
      }
    }
    if (face_texture && !face_why) {
      VulkanTexture& texture = *face_texture;
      const VkDescriptorSet set = face_set;
      texture.MarkAsUsed();
      VulkanTexture::Usage old_usage = texture.SetUsage(VulkanTexture::Usage::kComputeStorageWrite);
      native_last_copy_untouched_ = !texture.native_used;  // (Session 80)
      texture.native_used = false;
      VkPipelineStageFlags src_stage_mask;
      VkAccessFlags src_access_mask;
      VkImageLayout old_layout;
      GetTextureUsageMasks(old_usage, src_stage_mask, src_access_mask, old_layout);
      if (native_hold_barrier_) {
        native_held_barrier_ = {true, texture.image(), src_stage_mask, src_access_mask, old_layout};
      } else {
        command_processor_.PushImageMemoryBarrier(
            texture.image(), ui::vulkan::util::InitializeSubresourceRange(), src_stage_mask,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, src_access_mask, VK_ACCESS_SHADER_WRITE_BIT, old_layout,
            VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
      }
      if (command_processor_.NativeSegmentLogActive()) {
        REXGPU_INFO("[native][seg] texture fill: cube face {} of page 0x{:05X} (face page 0x{:05X})", layer,
                    uint32_t(key.base_page), base_page);
      }
      width_out = key.GetWidth();
      height_out = key.GetHeight();
      handle_out = &texture;
      if (layer_out) *layer_out = int32_t(layer);
      return set;
    }
  }
  auto known = native_resolve_keys_.find(base_page);
  if (known == native_resolve_keys_.end()) return refuse("no direct load seen on this page", nullptr);
  const TextureKey key = known->second;
  // (Session 52: scaled textures at a resolution scale - gpu_native_truth_direct_resolve_scaled_texture.)
  if (key.format != format || key.dimension != xenos::DataDimension::k2DOrStacked ||
      key.depth_or_array_size_minus_1 || !key.tiled || key.mip_max_level || key.packed_mips ||
      bool(key.scaled_resolve) != (allow_scaled && IsDrawResolutionScaled()) ||
      uint32_t(key.endianness) != endianness || key.pitch != pitch_div_32) {
    return refuse("key", &key);
  }
  Texture* found = NativeFindTexture(key);
  if (!found) return refuse("texture gone", &key);
  VulkanTexture& texture = static_cast<VulkanTexture&>(*found);
  // (Session 59: k_16_16_16_16 - 64 bits a texel - through its R16G16B16A16_UINT view.)
  const VkFormat storage_format = format == xenos::TextureFormat::k_16_16_16_16 ? VK_FORMAT_R16G16B16A16_UINT
                                  : format == xenos::TextureFormat::k_8       ? VK_FORMAT_R8_UINT  // (Session 59)
                                  : format == xenos::TextureFormat::k_4_4_4_4 ? VK_FORMAT_R16_UINT
                                                                              : VK_FORMAT_R32_UINT;
  if (texture.direct_load_creation() != VulkanTexture::DirectLoadCreation::kCapable ||
      texture.direct_load_format() != storage_format) {
    return refuse("no R32_UINT / R16G16B16A16_UINT storage view", &key);
  }
  // (Session 52: or the write covers it whole - gpu_native_truth_cover_outdated, NativeSetCover.)
  if (!NativeCoversWhole(key, cover) && !NativeTextureBaseUpToDate(texture)) {
    return refuse("texture not up to date", &key);
  }
  VkDescriptorSet set = texture.GetDirectLoadDescriptorSet();
  if (set == VK_NULL_HANDLE) return VK_NULL_HANDLE;
  // A compute write that keeps the other texels (the old layout, not UNDEFINED); the texture's next use as a sampled
  // image switches it back (its usage tracking, as after a direct load).
  texture.MarkAsUsed();
  VulkanTexture::Usage old_usage = texture.SetUsage(VulkanTexture::Usage::kComputeStorageWrite);
  // wwe2k14 patch (Session 80, gpu_native_truth_drop_dead_copies): nothing used it since the last native write.
  native_last_copy_untouched_ = !texture.native_used;
  texture.native_used = false;
  VkPipelineStageFlags src_stage_mask;
  VkAccessFlags src_access_mask;
  VkImageLayout old_layout;
  GetTextureUsageMasks(old_usage, src_stage_mask, src_access_mask, old_layout);
  if (native_hold_barrier_) {  // wwe2k14 patch (Session 52): recorded by the caller
    native_held_barrier_ = {true, texture.image(), src_stage_mask, src_access_mask, old_layout};
  } else {
    command_processor_.PushImageMemoryBarrier(
        texture.image(), ui::vulkan::util::InitializeSubresourceRange(), src_stage_mask,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, src_access_mask, VK_ACCESS_SHADER_WRITE_BIT, old_layout,
        VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        false);  // (review fix: never skipped - a second native resolve into it must wait for the first)
  }
  width_out = key.GetWidth() * (key.scaled_resolve ? draw_resolution_scale_x() : 1);
  height_out = key.GetHeight() * (key.scaled_resolve ? draw_resolution_scale_y() : 1);
  handle_out = &texture;
  return set;
}

void VulkanTextureCache::NativeTextureResolved(void* handle) {
  if (handle) NativeMakeTextureUpToDate(*static_cast<VulkanTexture*>(handle));
}

// wwe2k14 patch (Session 49, NATIVE TRUTH skip write-back): see the header.
uint32_t VulkanTextureCache::NativeTextureReadable(void* handle, bool allow_scaled, int32_t layer,
                                                  bool allow_depth) const {
  if (!handle) return 0;
  const VulkanTexture& texture = *static_cast<const VulkanTexture*>(handle);
  const TextureKey& key = texture.key();
  // Session 79: a face of a 1x single-level tiled k_8_8_8_8 cube with the R32_UINT view (gpu_native_truth_fill_cube_faces).
  if (layer >= 0) {
    return key.dimension == xenos::DataDimension::kCube && !key.mip_max_level && key.tiled && !key.scaled_resolve &&
                   key.format == xenos::TextureFormat::k_8_8_8_8 &&
                   uint32_t(layer) < std::min(key.GetDepthOrArraySize(), VulkanTexture::kNativeLayerViews) &&
                   texture.direct_load_creation() == VulkanTexture::DirectLoadCreation::kCapable &&
                   texture.direct_load_format() == VK_FORMAT_R32_UINT
               ? 4u
               : 0u;
  }
  if (key.mip_max_level || key.depth_or_array_size_minus_1 || !key.tiled || (key.scaled_resolve && !allow_scaled) ||
      key.dimension != xenos::DataDimension::k2DOrStacked ||
      texture.direct_load_creation() != VulkanTexture::DirectLoadCreation::kCapable) {
    return 0;
  }
  // (Depth: the direct load converts 20e4 to 32-bit floats - not raw bits. Session 80: for a reader that converts them
  // back, allow_depth, the 20e4 texture's floats through its R32_UINT view.)
  if (key.format == xenos::TextureFormat::k_24_8_FLOAT && allow_depth) {
    return texture.direct_load_format() == VK_FORMAT_R32_UINT ? 4u : 0u;
  }
  if (key.format == xenos::TextureFormat::k_24_8_FLOAT || key.format == xenos::TextureFormat::k_24_8) return 0;
  return texture.direct_load_format() == VK_FORMAT_R32_UINT           ? 4u
         : texture.direct_load_format() == VK_FORMAT_R16G16B16A16_UINT ? 8u
                                                                       : 0u;
}

VkDescriptorSet VulkanTextureCache::NativeTextureForRead(void* handle, uint32_t& old_usage_out, bool allow_scaled,
                                                         int32_t layer, bool allow_depth) {
  old_usage_out = uint32_t(VulkanTexture::Usage::kComputeStorageWrite);
  if (!NativeTextureReadable(handle, allow_scaled, layer, allow_depth)) return VK_NULL_HANDLE;
  VulkanTexture& texture = *static_cast<VulkanTexture*>(handle);
  VkDescriptorSet set = layer >= 0 ? texture.GetLayerStorageDescriptorSet(uint32_t(layer))
                                   : texture.GetDirectLoadDescriptorSet();
  if (set == VK_NULL_HANDLE) return VK_NULL_HANDLE;
  texture.MarkAsUsed();
  texture.native_used = true;  // wwe2k14 patch (Session 80: read into guest memory)
  VulkanTexture::Usage old_usage = texture.SetUsage(VulkanTexture::Usage::kComputeStorageWrite);
  old_usage_out = uint32_t(old_usage);
  VkPipelineStageFlags src_stage_mask;
  VkAccessFlags src_access_mask;
  VkImageLayout old_layout;
  GetTextureUsageMasks(old_usage, src_stage_mask, src_access_mask, old_layout);
  command_processor_.PushImageMemoryBarrier(
      texture.image(), ui::vulkan::util::InitializeSubresourceRange(), src_stage_mask,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, src_access_mask, VK_ACCESS_SHADER_READ_BIT, old_layout,
      VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
  return set;
}

void VulkanTextureCache::BeforeScaledResolveRead(uint32_t start, uint32_t length) {
  command_processor_.NativeTruthMaterializeScaled(start, length);
}

uint32_t VulkanTextureCache::SampleWrittenTextures(void* const* keep, size_t keep_count) {
  if (written_textures_.empty()) return 0;
  VkPipelineStageFlags dst_stage_mask;
  VkAccessFlags dst_access_mask;
  VkImageLayout new_layout;
  GetTextureUsageMasks(VulkanTexture::Usage::kGuestShaderSampled, dst_stage_mask, dst_access_mask, new_layout);
  uint32_t moved = 0;
  size_t kept = 0;
  for (VulkanTexture* texture : written_textures_) {
    if (std::find(keep, keep + keep_count, static_cast<void*>(texture)) != keep + keep_count) {
      written_textures_[kept++] = texture;
      continue;
    }
    texture->in_written_list = false;
    const VulkanTexture::Usage usage = texture->usage();
    if (usage != VulkanTexture::Usage::kComputeStorageWrite && usage != VulkanTexture::Usage::kTransferDestination) {
      continue;
    }
    texture->SetUsage(VulkanTexture::Usage::kGuestShaderSampled);
    VkPipelineStageFlags src_stage_mask;
    VkAccessFlags src_access_mask;
    VkImageLayout old_layout;
    GetTextureUsageMasks(usage, src_stage_mask, src_access_mask, old_layout);
    command_processor_.PushImageMemoryBarrier(texture->image(), ui::vulkan::util::InitializeSubresourceRange(),
                                              src_stage_mask, dst_stage_mask, src_access_mask, dst_access_mask,
                                              old_layout, new_layout);
    ++moved;
  }
  written_textures_.resize(kept);
  return moved;
}

void VulkanTextureCache::NativeTextureEndRead(void* handle, uint32_t old_usage) {
  const auto usage = VulkanTexture::Usage(old_usage);
  if (!handle || usage == VulkanTexture::Usage::kComputeStorageWrite || usage == VulkanTexture::Usage::kUndefined) {
    return;
  }
  VulkanTexture& texture = *static_cast<VulkanTexture*>(handle);
  texture.SetUsage(usage);
  VkPipelineStageFlags dst_stage_mask;
  VkAccessFlags dst_access_mask;
  VkImageLayout new_layout;
  GetTextureUsageMasks(usage, dst_stage_mask, dst_access_mask, new_layout);
  command_processor_.PushImageMemoryBarrier(
      texture.image(), ui::vulkan::util::InitializeSubresourceRange(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, dst_stage_mask,
      VK_ACCESS_SHADER_READ_BIT, dst_access_mask, VK_IMAGE_LAYOUT_GENERAL, new_layout, VK_QUEUE_FAMILY_IGNORED,
      VK_QUEUE_FAMILY_IGNORED, false);
}

void VulkanTextureCache::NativeTexturePin(void* handle, bool pin) {
  if (handle) static_cast<VulkanTexture*>(handle)->NativePin(pin);
}

// wwe2k14 patch (Session 80, gpu_native_truth_serve_copies): see the header.
bool VulkanTextureCache::NativeUnserve(uint32_t used_texture_mask) {
  uint32_t served = native_served_mask_ & used_texture_mask;
  if (!served) return false;
  VkPipelineStageFlags dst_stage_mask;
  VkAccessFlags dst_access_mask;
  VkImageLayout new_layout;
  GetTextureUsageMasks(VulkanTexture::Usage::kGuestShaderSampled, dst_stage_mask, dst_access_mask, new_layout);
  uint32_t index;
  while (bit_inline::bit_scan_forward(served, &index)) {
    served &= ~(uint32_t(1) << index);
    native_served_mask_ &= ~(uint32_t(1) << index);
    const TextureBinding* binding = GetValidTextureBinding(index);
    VulkanTexture* texture = binding ? static_cast<VulkanTexture*>(binding->texture) : nullptr;
    vulkan_texture_bindings_[index].image_view_unsigned =
        texture ? texture->GetView(false, binding->host_swizzle) : VK_NULL_HANDLE;
    ++texture_binding_generations_[index];
    if (!texture) continue;
    texture->MarkAsUsed();
    texture->native_used = true;
    const VulkanTexture::Usage old_usage = texture->SetUsage(VulkanTexture::Usage::kGuestShaderSampled);
    if (old_usage == VulkanTexture::Usage::kComputeStorageWrite) {
      command_processor_.NativeFlushDeferredResolves(texture);
    }
    if (old_usage != VulkanTexture::Usage::kGuestShaderSampled) {
      VkPipelineStageFlags src_stage_mask;
      VkAccessFlags src_access_mask;
      VkImageLayout old_layout;
      GetTextureUsageMasks(old_usage, src_stage_mask, src_access_mask, old_layout);
      command_processor_.PushImageMemoryBarrier(texture->image(), ui::vulkan::util::InitializeSubresourceRange(),
                                                src_stage_mask, dst_stage_mask, src_access_mask, dst_access_mask,
                                                old_layout, new_layout);
    }
  }
  return true;
}

// wwe2k14 patch (Session 47, NATIVE TRUTH M6): the key of a texture loaded over resolved pages, by its base page and
// by the start page of each mip level outside the packed tail.
void VulkanTextureCache::NativeLearnTextureKey(const VulkanTexture& texture) {
  const TextureKey& key = texture.key();
  native_resolve_keys_[key.base_page] = key;
  // Session 82 (P2): a cube the owned frame may have made of its own face copies before this first load - it checks its guess.
  if (key.dimension == xenos::DataDimension::kCube && !key.mip_max_level) {
    command_processor_.NativeOwnedCubeLearned(key.base_page, key.GetWidth(), key.GetHeight(),
                                              std::min(key.GetDepthOrArraySize(), 6u),
                                              texture.guest_layout().base.array_slice_stride_bytes);
  }
  // Session 79 (gpu_native_truth_fill_cube_faces): each face of a cube with the R32_UINT storage view, by its start page
  // (faces lie array_slice_stride_bytes apart; only page-aligned strides).
  if (key.dimension == xenos::DataDimension::kCube && !key.mip_max_level &&
      texture.direct_load_creation() == VulkanTexture::DirectLoadCreation::kCapable &&
      texture.direct_load_format() == VK_FORMAT_R32_UINT && REXCVAR_GET(gpu_native_truth_fill_cube_faces)) {
    const uint32_t stride = texture.guest_layout().base.array_slice_stride_bytes;
    const uint32_t faces = std::min(key.GetDepthOrArraySize(), VulkanTexture::kNativeLayerViews);
    if (stride && !(stride & 0xFFF)) {
      for (uint32_t layer = 0; layer < faces; ++layer) {
        native_resolve_layer_keys_[key.base_page + layer * (stride >> 12)] = {key, layer};
      }
    }
  }
  if (!key.mip_max_level || !key.mip_page) return;
  const texture_util::TextureGuestLayout& layout = texture.guest_layout();
  const FormatInfo* info = FormatInfo::Get(key.format);
  const uint32_t bytes_per_block = info->bytes_per_block();
  for (uint32_t level = 1; level <= key.mip_max_level && level < xenos::kTextureMaxMips; ++level) {
    if (level >= layout.packed_level) break;  // (the packed tail: not filled)
    const uint32_t offset = layout.mip_offsets_bytes[level];
    if (offset & 0xFFF) continue;
    const uint32_t row_pitch = layout.mips[level].row_pitch_bytes;
    if (!bytes_per_block || row_pitch % (bytes_per_block * 32)) continue;
    native_resolve_mip_keys_[key.mip_page + (offset >> 12)] = {key, level, row_pitch / bytes_per_block / 32};
  }
}

// wwe2k14 patch (Session 82, P2, G7): see the header.
bool VulkanTextureCache::NativeResolveFace(uint32_t face_page, uint32_t endianness, xenos::TextureFormat format,
                                           uint32_t& texture_page_out, uint32_t& width_out, uint32_t& height_out,
                                           uint32_t& layers_out, uint32_t& layer_out, bool& scaled_out,
                                           uint32_t& extent_bytes_out) {
  auto face = native_resolve_layer_keys_.find(face_page);
  if (face == native_resolve_layer_keys_.end()) return false;
  const TextureKey key = face->second.key;
  if (key.format != format || key.dimension != xenos::DataDimension::kCube || !key.tiled || key.mip_max_level ||
      uint32_t(key.endianness) != endianness || bool(key.scaled_resolve) != IsDrawResolutionScaled()) {
    return false;
  }
  Texture* found = NativeFindTexture(key);
  if (!found) return false;
  const uint32_t faces = std::min(key.GetDepthOrArraySize(), 6u);
  if (faces != 6 || face->second.layer >= faces) return false;
  texture_page_out = key.base_page;
  width_out = key.GetWidth();
  height_out = key.GetHeight();
  layers_out = faces;
  layer_out = face->second.layer;
  scaled_out = bool(key.scaled_resolve);
  extent_bytes_out = found->guest_layout().base.array_slice_stride_bytes * faces;
  return true;
}

// wwe2k14 patch (Session 82, P2, G4): see the header.
bool VulkanTextureCache::NativeResolveTarget(uint32_t base_page, uint32_t pitch_div_32, uint32_t endianness,
                                             xenos::TextureFormat format, uint32_t& texture_page_out,
                                             uint32_t& width_out, uint32_t& height_out, uint32_t& levels_out,
                                             uint32_t& level_out, bool& scaled_out, uint32_t& extent_bytes_out) {
  TextureKey key;
  uint32_t level_pitch = 0;
  level_out = 0;
  auto known = native_resolve_keys_.find(base_page);
  if (known != native_resolve_keys_.end() && known->second.format == format) {
    key = known->second;
    level_pitch = key.pitch;
  } else {
    auto mip = native_resolve_mip_keys_.find(base_page);
    if (mip == native_resolve_mip_keys_.end()) return false;
    key = mip->second.key;
    level_out = mip->second.level;
    level_pitch = mip->second.pitch_div_32;
  }
  if (key.format != format || key.dimension != xenos::DataDimension::k2DOrStacked || key.depth_or_array_size_minus_1 ||
      !key.tiled || uint32_t(key.endianness) != endianness || level_pitch != pitch_div_32 ||
      bool(key.scaled_resolve) != IsDrawResolutionScaled()) {
    return false;
  }
  Texture* found = NativeFindTexture(key);
  if (!found) return false;
  const texture_util::TextureGuestLayout& layout = found->guest_layout();
  if (level_out >= layout.packed_level) return false;  // (the packed tail)
  texture_page_out = key.base_page;
  width_out = key.GetWidth();
  height_out = key.GetHeight();
  uint32_t full = 1;
  for (uint32_t m = std::max(width_out, height_out); m > 1; m >>= 1) ++full;
  levels_out = std::min(uint32_t(key.mip_max_level) + 1u, full);
  scaled_out = bool(key.scaled_resolve);
  extent_bytes_out = layout.base.level_data_extent_bytes;
  return true;
}

// wwe2k14 patch (Session 47, NATIVE TRUTH M6): see the header.
bool VulkanTextureCache::NativeTextureForCopy(uint32_t base_page, uint32_t pitch_div_32, uint32_t endianness,
                                              xenos::TextureFormat format, VkImage& image_out, uint32_t& width_out,
                                              uint32_t& height_out, uint32_t& level_out, bool& has_mips_out,
                                              void*& handle_out, const char*& why, uint32_t storage_bytes,
                                              VkDescriptorSet* storage_set_out) {
  const std::array<uint32_t, 4> cover = NativeTakeCover();  // wwe2k14 patch (Session 52)
  handle_out = nullptr;
  native_last_copy_untouched_ = false;  // wwe2k14 patch (Session 80: set only where it is prepared for the write)
  image_out = VK_NULL_HANDLE;
  if (storage_set_out) *storage_set_out = VK_NULL_HANDLE;
  level_out = 0;
  has_mips_out = false;
  TextureKey key;
  uint32_t level_pitch = 0;
  auto known = native_resolve_keys_.find(base_page);
  if (known != native_resolve_keys_.end() && known->second.format == format) {
    key = known->second;
    level_pitch = key.pitch;
  } else {
    auto mip = native_resolve_mip_keys_.find(base_page);
    if (mip == native_resolve_mip_keys_.end()) {
      why = known != native_resolve_keys_.end() ? "key" : "no load seen on this page";
      return false;
    }
    key = mip->second.key;
    level_out = mip->second.level;
    level_pitch = mip->second.pitch_div_32;
  }
  if (key.format != format || key.dimension != xenos::DataDimension::k2DOrStacked ||
      key.depth_or_array_size_minus_1 || !key.tiled || bool(key.scaled_resolve) != IsDrawResolutionScaled() ||
      uint32_t(key.endianness) != endianness || level_pitch != pitch_div_32) {
    why = "key";
    return false;
  }
  // (Review: the host format the fill writes - not a fallback, e.g. 16-bit floats where 16-bit unorm can't filter.)
  {
    const HostFormatPair& pair = GetHostFormatPair(key);
    const VkFormat host = (key.signed_separate ? pair.format_signed : pair.format_unsigned).format;
    bool host_ok = false;
    switch (format) {
      case xenos::TextureFormat::k_8_8_8_8:
        host_ok = host == VK_FORMAT_R8G8B8A8_UNORM || host == VK_FORMAT_R8G8B8A8_SNORM;
        break;
      case xenos::TextureFormat::k_16_16:
        host_ok = host == VK_FORMAT_R16G16_UNORM || host == VK_FORMAT_R16G16_SNORM;
        break;
      case xenos::TextureFormat::k_16_16_16_16:
        host_ok = host == VK_FORMAT_R16G16B16A16_UNORM || host == VK_FORMAT_R16G16B16A16_SNORM;
        break;
      case xenos::TextureFormat::k_24_8_FLOAT:
        host_ok = host == VK_FORMAT_R32_SFLOAT;
        break;
      case xenos::TextureFormat::k_8:
        host_ok = host == VK_FORMAT_R8_UNORM || host == VK_FORMAT_R8_SNORM;
        break;
      case xenos::TextureFormat::k_4_4_4_4:
        host_ok = host == VK_FORMAT_B4G4R4A4_UNORM_PACK16;
        break;
      default:
        break;
    }
    if (!host_ok) {
      why = "host format";
      return false;
    }
  }
  Texture* found = NativeFindTexture(key);
  if (!found) {
    why = "texture gone";
    return false;
  }
  if (!NativeCoversWhole(key, cover) && !NativeTextureBaseUpToDate(*found)) {  // (Session 52: or covered whole)
    why = "texture not up to date";
    return false;
  }
  VulkanTexture& texture = static_cast<VulkanTexture&>(*found);
  if (level_out >= texture.guest_layout().packed_level) {
    why = "packed mip tail";
    return false;
  }
  texture.MarkAsUsed();
  // wwe2k14 patch (Session 52, gpu_native_truth_fill_direct): a compute write into its raw-bits storage view instead.
  VkDescriptorSet storage_set = VK_NULL_HANDLE;
  // (Scaled textures too, Session 52: their direct-load view - NativeTextureReadable is 1x only, for read-backs.)
  // (Session 59, gpu_native_truth_fill_direct_small: 1 / 2 bytes - k_8 / k_4_4_4_4's R8_UINT / R16_UINT views.)
  const VkFormat storage_format = storage_bytes == 4   ? VK_FORMAT_R32_UINT
                                  : storage_bytes == 8 ? VK_FORMAT_R16G16B16A16_UINT
                                  : storage_bytes == 1 ? VK_FORMAT_R8_UINT
                                  : storage_bytes == 2 ? VK_FORMAT_R16_UINT
                                                       : VK_FORMAT_UNDEFINED;
  if (storage_bytes && storage_set_out && level_out == 0 && !key.mip_max_level && key.tiled &&
      !key.depth_or_array_size_minus_1 && key.dimension == xenos::DataDimension::k2DOrStacked &&
      // (Session 59, gpu_native_truth_fill_direct_depth: k_24_8_FLOAT too - its direct load's R32_UINT view holds the
      // 32-bit float bits the depth fill writes; k_24_8's unorm depth is not a fill kind.)
      key.format != xenos::TextureFormat::k_24_8 &&
      texture.direct_load_creation() == VulkanTexture::DirectLoadCreation::kCapable &&
      texture.direct_load_format() == storage_format) {
    storage_set = texture.GetDirectLoadDescriptorSet();
  }
  if (storage_set != VK_NULL_HANDLE) {
    const VulkanTexture::Usage old_usage = texture.SetUsage(VulkanTexture::Usage::kComputeStorageWrite);
    // wwe2k14 patch (Session 59 review): an immediate write after a deferred one into it - the deferred one first (a
    // held barrier is the caller's batching: TruthFillTexture orders overlapping writes itself).
    if (old_usage == VulkanTexture::Usage::kComputeStorageWrite && !native_hold_barrier_) {
      command_processor_.NativeFlushDeferredResolves(&texture);
    }
    native_last_copy_untouched_ = !texture.native_used;  // (Session 80)
    texture.native_used = false;
    native_last_copy_sampled_ = old_usage == VulkanTexture::Usage::kGuestShaderSampled ||
                                old_usage == VulkanTexture::Usage::kSwapSampled;
    VkPipelineStageFlags src_stage;
    VkAccessFlags src_access;
    VkImageLayout old_layout;
    GetTextureUsageMasks(old_usage, src_stage, src_access, old_layout);
    if (native_hold_barrier_) {  // wwe2k14 patch (Session 59, gpu_native_truth_batch_fills): recorded by the caller
      native_held_barrier_ = {true, texture.image(), src_stage, src_access, old_layout};
    } else {
      command_processor_.PushImageMemoryBarrier(texture.image(), ui::vulkan::util::InitializeSubresourceRange(),
                                                src_stage, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, src_access,
                                                VK_ACCESS_SHADER_WRITE_BIT, old_layout, VK_IMAGE_LAYOUT_GENERAL,
                                                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
    }
    *storage_set_out = storage_set;
    image_out = texture.image();
    width_out = key.GetWidth() * (key.scaled_resolve ? draw_resolution_scale_x() : 1);
    height_out = key.GetHeight() * (key.scaled_resolve ? draw_resolution_scale_y() : 1);
    has_mips_out = false;
    handle_out = &texture;
    return true;
  }
  const VulkanTexture::Usage old_usage = texture.SetUsage(VulkanTexture::Usage::kTransferDestination);
  // wwe2k14 patch (Session 59 review): a deferred native write into it goes first.
  if (old_usage == VulkanTexture::Usage::kComputeStorageWrite) command_processor_.NativeFlushDeferredResolves(&texture);
  // wwe2k14 patch (Session 49): sampled since its last fill or load (its usage moved to sampled since then).
  native_last_copy_untouched_ = false;  // (Session 80: a transfer write - not a compute one)
  texture.native_used = false;
  native_last_copy_sampled_ = old_usage == VulkanTexture::Usage::kGuestShaderSampled ||
                              old_usage == VulkanTexture::Usage::kSwapSampled;
  VkPipelineStageFlags src_stage;
  VkAccessFlags src_access;
  VkImageLayout old_layout;
  GetTextureUsageMasks(old_usage, src_stage, src_access, old_layout);
  command_processor_.PushImageMemoryBarrier(texture.image(), ui::vulkan::util::InitializeSubresourceRange(), src_stage,
                                            VK_PIPELINE_STAGE_TRANSFER_BIT, src_access, VK_ACCESS_TRANSFER_WRITE_BIT,
                                            old_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
                                            VK_QUEUE_FAMILY_IGNORED, false);
  image_out = texture.image();
  // (Session 47: in host texels - a scaled texture is the resolution scale times as large.)
  width_out = std::max(key.GetWidth() >> level_out, 1u) * (key.scaled_resolve ? draw_resolution_scale_x() : 1);
  height_out = std::max(key.GetHeight() >> level_out, 1u) * (key.scaled_resolve ? draw_resolution_scale_y() : 1);
  has_mips_out = key.mip_max_level != 0;
  handle_out = &texture;
  return true;
}

void VulkanTextureCache::TexLoadFrameBoundary(uint64_t frame) {
  TexLoadStats& stats = texload_stats_;
  // Fixed layout (for the report tool):
  //   cpf_texload  a = texture loads recorded (LoadTextureDataFromResidentMemoryImpl, each = untiling dispatches +
  //                one copy into the image), b = of those, from the resolution-scaled resolve buffer, c = KiB the
  //                copies wrote into images
  if (perf::timeline::Enabled()) {
    perf::timeline::Record("cpf_texload", stats.frame_loads, stats.frame_loads_scaled,
                           uint32_t(std::min(stats.frame_bytes >> 10, uint64_t(UINT32_MAX))));
  }
  stats.frame_loads = 0;
  stats.frame_loads_scaled = 0;
  stats.frame_bytes = 0;

  const VulkanCommandProcessor::ScratchRingStats& ring = command_processor_.scratch_ring_stats();
  auto start_window = [&]() {
    stats.window_first_frame = frame;
    stats.loads = 0;
    stats.loads_scaled = 0;
    stats.bytes = 0;
    stats.float_converts = 0;
    stats.invalidated_gpu = 0;
    stats.invalidated_cpu = 0;
    stats.first_loads = 0;
    stats.transitions = 0;
    stats.rewrite_all = 0;
    stats.discards = 0;
    stats.ring_loads = 0;
    stats.ring_waw_barriers = 0;
    stats.ring_acquisitions_start = ring.acquisitions;
    stats.ring_barriers_start = ring.barriers;
    stats.ring_pre_transitions_start = ring.pre_transitions;
    stats.ring_grows_start = ring.grows;
    stats.ring_releases_start = ring.releases;
  };
  if (stats.window_first_frame == UINT64_MAX || frame < stats.window_first_frame) {
    start_window();
    return;
  }
  if (frame - stats.window_first_frame < 300) {
    return;
  }
  // Logged when the window had loads (or the ring freed its memory).
  if (stats.loads || ring.releases != stats.ring_releases_start) {
    const double frames = double(frame - stats.window_first_frame);
    REXGPU_INFO(
        "[s7][TEXLOAD] frames {}-{}: {:.1f} loads/frame ({:.1f} scaled), {:.1f} MB/frame written to images; of "
        "all {} loads: {} float-converted, {} first loads, invalidated by GPU writes (resolve / memexport) {} / "
        "by the CPU {}; {} with a layout transition, {} of them rewriting the whole image, discarded {} "
        "(vulkan_texture_load_discard_old {}); scratch ring (vulkan_texture_load_scratch_ring {}): {} loads, "
        "{} acquisitions with a barrier, {} pre-transitions, {} buffers created, {} image WAW barriers, {} "
        "releases, {} MB held",
        stats.window_first_frame, frame, double(stats.loads) / frames, double(stats.loads_scaled) / frames,
        double(stats.bytes) / (1024.0 * 1024.0) / frames, stats.loads, stats.float_converts, stats.first_loads,
        stats.invalidated_gpu, stats.invalidated_cpu, stats.transitions, stats.rewrite_all, stats.discards,
        REXCVAR_GET(vulkan_texture_load_discard_old) ? "on" : "off",
        REXCVAR_GET(vulkan_texture_load_scratch_ring) ? "on" : "off", stats.ring_loads,
        ring.barriers - stats.ring_barriers_start, ring.pre_transitions - stats.ring_pre_transitions_start,
        ring.grows - stats.ring_grows_start, stats.ring_waw_barriers,
        ring.releases - stats.ring_releases_start, command_processor_.GetScratchRingSize() >> 20);
  }
  // wwe2k14 patch (Session 22, RESWAW): resolves that followed unordered resolve writes, per window.
  if (resolve_waw_skipped_ || resolve_waw_kept_) {
    REXGPU_INFO("[s22][RESWAW] frames {}-{}: write-after-write barriers between resolves: {:.1f}/frame kept, "
                "{:.1f}/frame skipped (vulkan_resolve_skip_disjoint_waw {})",
                stats.window_first_frame, frame,
                double(resolve_waw_kept_) / double(frame - stats.window_first_frame),
                double(resolve_waw_skipped_) / double(frame - stats.window_first_frame),
                REXCVAR_GET(vulkan_resolve_skip_disjoint_waw) ? "on" : "off");
    resolve_waw_skipped_ = 0;
    resolve_waw_kept_ = 0;
  }
  // wwe2k14 patch (Session 22, EAGER): per window.
  if (eager_stats_.loads || eager_stats_.learned || eager_stats_.reloaded) {
    const double frames = double(frame - stats.window_first_frame);
    REXGPU_INFO("[s22][EAGER] frames {}-{}: per frame {:.1f} early loads, {:.1f} late loads learned, {:.1f} "
                "loaded again at their use after an early load, {:.1f} due but already valid, {:.1f} failed "
                "(vulkan_texture_load_eager {})",
                stats.window_first_frame, frame, double(eager_stats_.loads) / frames,
                double(eager_stats_.learned) / frames, double(eager_stats_.reloaded) / frames,
                double(eager_stats_.already_valid) / frames, double(eager_stats_.failed) / frames,
                REXCVAR_GET(vulkan_texture_load_eager) ? "on" : "off");
    eager_stats_ = EagerStats();
  }
  // wwe2k14 patch (Session 87, step 3): vulkan_texture_load_early_cpu, per window.
  if (cpu_early_stats_.loads || cpu_early_stats_.learned || cpu_early_stats_.reloaded) {
    const double frames = double(frame - stats.window_first_frame);
    REXGPU_INFO("[s87][EARLYCPU] frames {}-{}: per frame {:.1f} early loads, {:.1f} late loads learned, {:.1f} loaded again at "
                "their use after an early load, {:.1f} due but already valid, {:.1f} failed (vulkan_texture_load_early_cpu {})",
                stats.window_first_frame, frame, double(cpu_early_stats_.loads) / frames,
                double(cpu_early_stats_.learned) / frames, double(cpu_early_stats_.reloaded) / frames,
                double(cpu_early_stats_.already_valid) / frames, double(cpu_early_stats_.failed) / frames,
                REXCVAR_GET(vulkan_texture_load_early_cpu) ? "on" : "off");
    cpu_early_stats_ = CpuEarlyStats();
  }
  start_window();
}

void VulkanTextureCache::TexLoadCsvFlush() {
  if (!texload_csv_ || texload_csv_text_.empty()) {
    texload_csv_text_.clear();
    return;
  }
  if (std::fwrite(texload_csv_text_.data(), 1, texload_csv_text_.size(), texload_csv_) !=
      texload_csv_text_.size()) {
    REXGPU_WARN("[s7][TEXLOAD] Write error on {}; no more texture load rows for this capture",
                texload_csv_path_);
    std::fclose(texload_csv_);
    texload_csv_ = nullptr;
  }
  texload_csv_text_.clear();
}

// <anatomy>_texloads.csv (anatomy_NNN_K.csv -> anatomy_NNN_K_texloads.csv): one row per texture load recorded
// during the capture's frames, in recording order:
//   frame          capture frame (the anatomy CSV's frame column)
//   idx            the anatomy row whose IssueDraw recorded the load (between draws, e.g. the swap texture: the
//                  index the next row will get)
//   submission     host submission number
//   width, height, depth_layers, dimension, mips   the texture key (guest texels, unscaled; mips = levels of the
//                  image)
//   level_first, level_last   the levels loaded (0 = base)
//   guest_format   the guest format name; host_format: the VkFormat number of the image
//   scaled         loaded from the resolution-scaled resolve buffer; scale_x, scale_y: the scale applied
//   tiled, base, mips_base   the key's tiling and guest addresses
//   bytes          written into the image by the copy; scratch_bytes: the scratch buffer range the dispatches wrote
//   invalidated    new (first load) | gpu (a resolve or memory export wrote the texture's memory since its last
//                  load) | cpu | gpu+cpu
//   transition     the pre-copy image barrier: preserve (old layout kept, contents preserved) | discard (old
//                  layout undefined, vulkan_texture_load_discard_old) | undefined (never used image) | none (still
//                  a transfer destination, no barrier) | waw (still a transfer destination, write-after-write
//                  barrier: this or the image's last load used the ring)
//   float_convert  the two-pass float conversion was used
//   ring_entry     vulkan_texture_load_scratch_ring entry (-1: the single scratch buffer); ring_barrier: its
//                  acquisition pushed a barrier for its own entry
void VulkanTextureCache::TexLoadAnatomyFrameBoundary(const std::string& rows_path, int64_t frame) {
  // The rows of the frame that just ended.
  TexLoadCsvFlush();
  if (frame < 0 || rows_path != texload_csv_rows_path_) {
    if (texload_csv_) {
      std::fclose(texload_csv_);
      texload_csv_ = nullptr;
      REXGPU_INFO("[s7][TEXLOAD] Texture load rows for this capture -> {}", texload_csv_path_);
    }
    texload_csv_rows_path_.clear();
    texload_csv_path_.clear();
  }
  if (frame < 0) {
    return;
  }
  if (texload_csv_rows_path_ != rows_path) {
    // A new capture (tried once: no retry each frame if the file can't be opened).
    texload_csv_rows_path_ = rows_path;
    const std::filesystem::path path(rows_path);
    std::filesystem::path csv_path = path;
    csv_path.replace_filename(path.stem().string() + "_texloads" + path.extension().string());
    texload_csv_path_ = csv_path.string();
#ifdef _WIN32
    // Shared, so the file can be read while the capture runs.
    texload_csv_ = _wfsopen(csv_path.c_str(), L"wb", _SH_DENYNO);
#else
    texload_csv_ = std::fopen(csv_path.c_str(), "wb");
#endif
    if (!texload_csv_) {
      REXGPU_WARN("[s7][TEXLOAD] Can't open {}; this capture gets no texture load rows", texload_csv_path_);
    } else {
      std::fputs(
          "frame,idx,submission,width,height,depth_layers,dimension,mips,level_first,level_last,guest_format,"
          "host_format,scaled,scale_x,scale_y,tiled,base,mips_base,bytes,scratch_bytes,invalidated,transition,"
          "float_convert,ring_entry,ring_barrier\n",
          texload_csv_);
    }
  }
  texload_csv_frame_ = uint32_t(frame);
}

void VulkanTextureCache::UpdateTextureBindingsImpl(uint32_t fetch_constant_mask) {
  uint32_t bindings_remaining = fetch_constant_mask;
  uint32_t binding_index;
  while (bit_inline::bit_scan_forward(bindings_remaining, &binding_index)) {
    bindings_remaining &= ~(UINT32_C(1) << binding_index);
    VulkanTextureBinding& vulkan_binding = vulkan_texture_bindings_[binding_index];
    vulkan_binding.Reset();
    native_served_mask_ &= ~(UINT32_C(1) << binding_index);  // wwe2k14 patch (Session 80): the texture's view again
    native_owned_mask_ &= ~(UINT32_C(1) << binding_index);   // (and gpu_native_own_steps' bindings)
    // wwe2k14 patch (Session 6, CP5): descriptors written for this fetch constant are outdated now.
    ++texture_binding_generations_[binding_index];
    texture_bindings_3d_mask_ &= ~(UINT32_C(1) << binding_index);
    const TextureBinding* binding = GetValidTextureBinding(binding_index);
    if (!binding) {
      continue;
    }
    if (binding->key.dimension == xenos::DataDimension::k3D) {
      texture_bindings_3d_mask_ |= UINT32_C(1) << binding_index;
    }
    const HostFormatPair& host_format_pair = GetHostFormatPair(binding->key);
    bool uses_unsigned = texture_util::IsAnySignNotSigned(binding->swizzled_signs);
    bool uses_signed = texture_util::IsAnySignSigned(binding->swizzled_signs);
    xenos::TextureFormat format = binding->key.format;
    if (uses_unsigned && host_format_pair.format_unsigned.format == VK_FORMAT_UNDEFINED) {
      unsupported_format_features_used_[uint32_t(format)] |= kUnsupportedUnormBit;
    }
    if (uses_signed && host_format_pair.format_signed.format == VK_FORMAT_UNDEFINED) {
      unsupported_format_features_used_[uint32_t(format)] |= kUnsupportedSnormBit;
    }
    if (IsSignedVersionSeparateForFormat(binding->key)) {
      if (binding->texture && uses_unsigned &&
          host_format_pair.format_unsigned.format != VK_FORMAT_UNDEFINED) {
        vulkan_binding.image_view_unsigned =
            static_cast<VulkanTexture*>(binding->texture)->GetView(false, binding->host_swizzle);
      }
      if (binding->texture_signed && uses_signed &&
          host_format_pair.format_signed.format != VK_FORMAT_UNDEFINED) {
        vulkan_binding.image_view_signed = static_cast<VulkanTexture*>(binding->texture_signed)
                                               ->GetView(true, binding->host_swizzle);
      }
    } else {
      VulkanTexture* texture = static_cast<VulkanTexture*>(binding->texture);
      if (texture) {
        if (uses_unsigned && host_format_pair.format_unsigned.format != VK_FORMAT_UNDEFINED) {
          vulkan_binding.image_view_unsigned = texture->GetView(false, binding->host_swizzle);
        }
        if (uses_signed && host_format_pair.format_signed.format != VK_FORMAT_UNDEFINED) {
          vulkan_binding.image_view_signed = texture->GetView(true, binding->host_swizzle);
        }
      }
    }
  }
}

VulkanTextureCache::VulkanTexture::VulkanTexture(VulkanTextureCache& texture_cache,
                                                 const TextureKey& key, VkImage image,
                                                 VmaAllocation allocation, bool track_usage)
    : Texture(texture_cache, key, track_usage), image_(image), allocation_(allocation) {
  VmaAllocationInfo allocation_info;
  vmaGetAllocationInfo(texture_cache.vma_allocator_, allocation_, &allocation_info);
  SetHostMemoryUsage(uint64_t(allocation_info.size));
}

VulkanTextureCache::VulkanTexture::~VulkanTexture() {
  const VulkanTextureCache& vulkan_texture_cache =
      static_cast<const VulkanTextureCache&>(texture_cache());
  // wwe2k14 patch (Session 22, EAGER): no plan or pending entry may outlive the texture.
  const_cast<VulkanTextureCache&>(vulkan_texture_cache).EagerForget(this);
  if (in_written_list) {  // wwe2k14 patch (Session 56)
    std::vector<VulkanTexture*>& list = const_cast<VulkanTextureCache&>(vulkan_texture_cache).written_textures_;
    list.erase(std::remove(list.begin(), list.end(), this), list.end());
  }
  const ui::vulkan::VulkanDevice* const vulkan_device =
      vulkan_texture_cache.command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  for (const auto& view_pair : views_) {
    dfn.vkDestroyImageView(device, view_pair.second, nullptr);
  }
  if (image_view_3d_as_2d_unsigned_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImageView(device, image_view_3d_as_2d_unsigned_, nullptr);
  }
  if (image_view_3d_as_2d_signed_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImageView(device, image_view_3d_as_2d_signed_, nullptr);
  }
  // wwe2k14 patch (Session 7c, TEXDIRECT): the texture is destroyed only after the submissions that used it have
  // completed (or with the GPU idle), so its descriptor set can be rewritten for another texture.
  if (direct_load_descriptor_index_ != SIZE_MAX && vulkan_texture_cache.direct_load_descriptor_pool_) {
    vulkan_texture_cache.direct_load_descriptor_pool_->Free(direct_load_descriptor_index_);
  }
  if (direct_load_view_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImageView(device, direct_load_view_, nullptr);
  }
  // wwe2k14 patch (Session 79): the cube faces' storage views.
  for (uint32_t i = 0; i + 1 < kNativeLayerViews; ++i) {
    if (layer_storage_indices_[i] != SIZE_MAX && vulkan_texture_cache.direct_load_descriptor_pool_) {
      vulkan_texture_cache.direct_load_descriptor_pool_->Free(layer_storage_indices_[i]);
    }
    if (layer_storage_views_[i] != VK_NULL_HANDLE) dfn.vkDestroyImageView(device, layer_storage_views_[i], nullptr);
  }
  vmaDestroyImage(vulkan_texture_cache.vma_allocator_, image_, allocation_);
}

// wwe2k14 patch (Session 18, DLAA): see the header.
bool VulkanTextureCache::DlaaResolveStorage(uint32_t base_address, uint32_t pitch_texels, uint32_t width,
                                            uint32_t height, xenos::TextureFormat format, xenos::Endian endian,
                                            DlaaStorage& out) {
  out = DlaaStorage();
  Texture* t = DlaaFindResolveTexture(base_address, pitch_texels, width, height, format, endian);
  if (!t) {
    out.failure = "no texture (unsupported or load failed)";
    return false;
  }
  VulkanTexture& vt = static_cast<VulkanTexture&>(*t);
  if (vt.direct_load_creation() != VulkanTexture::DirectLoadCreation::kCapable ||
      vt.direct_load_format() != VK_FORMAT_R32_UINT) {
    out.failure = "texture without the R32_UINT storage view (vulkan_texture_load_direct off or not eligible)";
    return false;
  }
  VkDescriptorSet set = vt.GetDirectLoadDescriptorSet();
  if (set == VK_NULL_HANDLE) {
    out.failure = "no storage descriptor";
    return false;
  }
  vt.MarkAsUsed();
  const VulkanTexture::Usage old_usage = vt.SetUsage(VulkanTexture::Usage::kComputeStorageWrite);
  // wwe2k14 patch (Session 80 review): a deferred native copy still to write it first.
  if (old_usage == VulkanTexture::Usage::kComputeStorageWrite) command_processor_.NativeFlushDeferredResolves(&vt);
  VkPipelineStageFlags src_stage;
  VkAccessFlags src_access;
  VkImageLayout old_layout;
  GetTextureUsageMasks(old_usage, src_stage, src_access, old_layout);
  command_processor_.PushImageMemoryBarrier(vt.image(), ui::vulkan::util::InitializeSubresourceRange(), src_stage,
                                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, src_access,
                                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, old_layout,
                                            VK_IMAGE_LAYOUT_GENERAL);
  const TextureKey& key = t->key();
  out.image = vt.image();
  out.set = set;
  out.host_format = GetHostFormatPair(key).format_unsigned.format;
  out.width = key.GetWidth() * (key.scaled_resolve ? draw_resolution_scale_x() : 1);
  out.height = key.GetHeight() * (key.scaled_resolve ? draw_resolution_scale_y() : 1);
  return true;
}

bool VulkanTextureCache::DlaaResolveCopyTarget(uint32_t base_address, uint32_t pitch_texels, uint32_t width,
                                               uint32_t height, xenos::TextureFormat format, xenos::Endian endian,
                                               DlaaStorage& out, bool general) {
  out = DlaaStorage();
  Texture* t = DlaaFindResolveTexture(base_address, pitch_texels, width, height, format, endian);
  if (!t) {
    out.failure = "no texture (unsupported or load failed)";
    return false;
  }
  VulkanTexture& vt = static_cast<VulkanTexture&>(*t);
  vt.MarkAsUsed();
  const VulkanTexture::Usage old_usage = vt.SetUsage(general ? VulkanTexture::Usage::kComputeStorageWrite
                                                              : VulkanTexture::Usage::kTransferDestination);
  // wwe2k14 patch (Session 80 review): a deferred native copy still to write it first.
  if (old_usage == VulkanTexture::Usage::kComputeStorageWrite) command_processor_.NativeFlushDeferredResolves(&vt);
  VkPipelineStageFlags src_stage;
  VkAccessFlags src_access;
  VkImageLayout old_layout;
  GetTextureUsageMasks(old_usage, src_stage, src_access, old_layout);
  command_processor_.PushImageMemoryBarrier(
      vt.image(), ui::vulkan::util::InitializeSubresourceRange(), src_stage, VK_PIPELINE_STAGE_TRANSFER_BIT,
      src_access, general ? (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT) : VK_ACCESS_TRANSFER_WRITE_BIT,
      old_layout, general ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  const TextureKey& key = t->key();
  out.image = vt.image();
  out.host_format = GetHostFormatPair(key).format_unsigned.format;
  out.width = key.GetWidth() * (key.scaled_resolve ? draw_resolution_scale_x() : 1);
  out.height = key.GetHeight() * (key.scaled_resolve ? draw_resolution_scale_y() : 1);
  return true;
}

VkDescriptorSet VulkanTextureCache::DlaaStorageSetForView(VkImageView view) {
  // Session 24: at resolution_scale 1 no scaled texture has initialized the direct-load objects yet.
  if (!EnsureDirectLoadInitialized()) return VK_NULL_HANDLE;
  const size_t index = direct_load_descriptor_pool_->Allocate();
  if (index == SIZE_MAX) return VK_NULL_HANDLE;
  VkDescriptorSet set = direct_load_descriptor_pool_->Get(index);
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  VkDescriptorImageInfo image_info;
  image_info.sampler = VK_NULL_HANDLE;
  image_info.imageView = view;
  image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  VkWriteDescriptorSet write;
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.pNext = nullptr;
  write.dstSet = set;
  write.dstBinding = 0;
  write.dstArrayElement = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  write.pImageInfo = &image_info;
  write.pBufferInfo = nullptr;
  write.pTexelBufferView = nullptr;
  vulkan_device->functions().vkUpdateDescriptorSets(vulkan_device->device(), 1, &write, 0, nullptr);
  return set;
}

VkDescriptorSet VulkanTextureCache::VulkanTexture::GetDirectLoadDescriptorSet() {
  // wwe2k14 patch (Session 7c, TEXDIRECT).
  VulkanTextureCache& vulkan_texture_cache = static_cast<VulkanTextureCache&>(texture_cache());
  ui::vulkan::SingleLayoutDescriptorSetPool* descriptor_pool =
      vulkan_texture_cache.direct_load_descriptor_pool_.get();
  // Review fix: a failure is not retried (and logged) on every later load of the texture.
  if (!descriptor_pool || direct_load_format_ == VK_FORMAT_UNDEFINED || direct_load_descriptor_failed_) {
    return VK_NULL_HANDLE;
  }
  if (direct_load_descriptor_index_ != SIZE_MAX) {
    return descriptor_pool->Get(direct_load_descriptor_index_);
  }
  const ui::vulkan::VulkanDevice* const vulkan_device =
      vulkan_texture_cache.command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  if (direct_load_view_ == VK_NULL_HANDLE) {
    VkImageViewCreateInfo view_create_info;
    view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_create_info.pNext = nullptr;
    view_create_info.flags = 0;
    view_create_info.image = image();
    view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_create_info.format = direct_load_format_;
    // Storage image descriptors require the identity swizzle.
    view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_create_info.subresourceRange =
        ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1);
    if (dfn.vkCreateImageView(device, &view_create_info, nullptr, &direct_load_view_) !=
        VK_SUCCESS) {
      direct_load_view_ = VK_NULL_HANDLE;
      direct_load_descriptor_failed_ = true;
      REXGPU_ERROR(
          "[s7][TEXDIRECT] failed to create a storage view (Vulkan format {}) of a texture - loaded through the "
          "buffer + copy path from now on",
          uint32_t(direct_load_format_));
      return VK_NULL_HANDLE;
    }
  }
  size_t descriptor_index = descriptor_pool->Allocate();
  if (descriptor_index == SIZE_MAX) {
    direct_load_descriptor_failed_ = true;
    REXGPU_ERROR(
        "[s7][TEXDIRECT] failed to allocate a storage image descriptor set for a texture - loaded through the "
        "buffer + copy path from now on");
    return VK_NULL_HANDLE;
  }
  VkDescriptorSet descriptor_set = descriptor_pool->Get(descriptor_index);
  VkDescriptorImageInfo descriptor_image_info;
  descriptor_image_info.sampler = VK_NULL_HANDLE;
  descriptor_image_info.imageView = direct_load_view_;
  descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  VkWriteDescriptorSet write_descriptor_set;
  write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_descriptor_set.pNext = nullptr;
  write_descriptor_set.dstSet = descriptor_set;
  write_descriptor_set.dstBinding = 0;
  write_descriptor_set.dstArrayElement = 0;
  write_descriptor_set.descriptorCount = 1;
  write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  write_descriptor_set.pImageInfo = &descriptor_image_info;
  write_descriptor_set.pBufferInfo = nullptr;
  write_descriptor_set.pTexelBufferView = nullptr;
  dfn.vkUpdateDescriptorSets(device, 1, &write_descriptor_set, 0, nullptr);
  direct_load_descriptor_index_ = descriptor_index;
  // Session 7c, CAUSES: a descriptor set allocated and written (once per texture).
  ++vulkan_texture_cache.command_processor_.cause_counts().desc_writes;
  return descriptor_set;
}

// wwe2k14 patch (Session 79, gpu_native_truth_fill_cube_faces): see the header.
VkDescriptorSet VulkanTextureCache::VulkanTexture::GetLayerStorageDescriptorSet(uint32_t layer) {
  if (!layer) return GetDirectLoadDescriptorSet();
  VulkanTextureCache& vulkan_texture_cache = static_cast<VulkanTextureCache&>(texture_cache());
  ui::vulkan::SingleLayoutDescriptorSetPool* descriptor_pool = vulkan_texture_cache.direct_load_descriptor_pool_.get();
  if (layer >= kNativeLayerViews || layer >= key().GetDepthOrArraySize() || !descriptor_pool ||
      direct_load_format_ == VK_FORMAT_UNDEFINED || direct_load_descriptor_failed_) {
    return VK_NULL_HANDLE;
  }
  const uint32_t i = layer - 1;
  if (layer_storage_indices_[i] != SIZE_MAX) return descriptor_pool->Get(layer_storage_indices_[i]);
  const ui::vulkan::VulkanDevice* const vulkan_device = vulkan_texture_cache.command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  if (layer_storage_views_[i] == VK_NULL_HANDLE) {
    VkImageViewCreateInfo view_create_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_create_info.image = image();
    view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_create_info.format = direct_load_format_;
    // (Storage image descriptors require the identity swizzle: zero-initialized.)
    view_create_info.subresourceRange =
        ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, layer, 1);
    if (dfn.vkCreateImageView(device, &view_create_info, nullptr, &layer_storage_views_[i]) != VK_SUCCESS) {
      layer_storage_views_[i] = VK_NULL_HANDLE;
      direct_load_descriptor_failed_ = true;
      REXGPU_ERROR("[s79][CUBE] failed to create a storage view of a cube face - its faces are not filled");
      return VK_NULL_HANDLE;
    }
  }
  const size_t descriptor_index = descriptor_pool->Allocate();
  if (descriptor_index == SIZE_MAX) {
    direct_load_descriptor_failed_ = true;
    REXGPU_ERROR("[s79][CUBE] failed to allocate a storage image descriptor set for a cube face");
    return VK_NULL_HANDLE;
  }
  const VkDescriptorSet descriptor_set = descriptor_pool->Get(descriptor_index);
  VkDescriptorImageInfo descriptor_image_info = {VK_NULL_HANDLE, layer_storage_views_[i], VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = descriptor_set;
  write.dstBinding = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  write.pImageInfo = &descriptor_image_info;
  dfn.vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
  layer_storage_indices_[i] = descriptor_index;
  ++vulkan_texture_cache.command_processor_.cause_counts().desc_writes;
  return descriptor_set;
}

VkImageView VulkanTextureCache::VulkanTexture::GetView(bool is_signed, uint32_t host_swizzle,
                                                       bool is_array) {
  xenos::DataDimension dimension = key().dimension;
  if (dimension == xenos::DataDimension::k3D || dimension == xenos::DataDimension::kCube) {
    is_array = false;
  }

  const VulkanTextureCache& vulkan_texture_cache =
      static_cast<const VulkanTextureCache&>(texture_cache());

  ViewKey view_key;

  const HostFormatPair& host_format_pair = vulkan_texture_cache.GetHostFormatPair(key());
  VkFormat format =
      (is_signed ? host_format_pair.format_signed : host_format_pair.format_unsigned).format;
  if (format == VK_FORMAT_UNDEFINED) {
    return VK_NULL_HANDLE;
  }
  // If not distinguishing between unsigned and signed formats for the same
  // image, don't create two views. As this happens within an image, no need to
  // care about whether unsigned and signed images are separate - if they are
  // (or if there are only unsigned or only signed images), this image will have
  // either all views unsigned or all views signed.
  view_key.is_signed_separate_view = is_signed && (host_format_pair.format_signed.format !=
                                                   host_format_pair.format_unsigned.format);

  const ui::vulkan::VulkanDevice* const vulkan_device =
      vulkan_texture_cache.command_processor_.GetVulkanDevice();

  if (!vulkan_device->properties().imageViewFormatSwizzle) {
    host_swizzle = xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA;
  }
  view_key.host_swizzle = host_swizzle;

  view_key.is_array = uint32_t(is_array);

  // Try to find an existing view.
  auto it = views_.find(view_key);
  if (it != views_.end()) {
    return it->second;
  }

  // Create a new view.
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  VkImageViewCreateInfo view_create_info;
  view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_create_info.pNext = nullptr;
  view_create_info.flags = 0;
  view_create_info.image = image();
  view_create_info.format = format;
  view_create_info.components.r = GetComponentSwizzle(host_swizzle, 0);
  view_create_info.components.g = GetComponentSwizzle(host_swizzle, 1);
  view_create_info.components.b = GetComponentSwizzle(host_swizzle, 2);
  view_create_info.components.a = GetComponentSwizzle(host_swizzle, 3);
  view_create_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange();
  switch (dimension) {
    case xenos::DataDimension::k3D:
      view_create_info.viewType = VK_IMAGE_VIEW_TYPE_3D;
      break;
    case xenos::DataDimension::kCube:
      view_create_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
      break;
    default:
      if (is_array) {
        view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      } else {
        view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_create_info.subresourceRange.layerCount = 1;
      }
      break;
  }
  VkImageView view;
  if (dfn.vkCreateImageView(device, &view_create_info, nullptr, &view) != VK_SUCCESS) {
    REXGPU_ERROR(
        "VulkanTextureCache: Failed to create an image view for Vulkan format "
        "{} ({}signed) with swizzle 0x{:3X}",
        uint32_t(format), is_signed ? "" : "un", host_swizzle);
    return VK_NULL_HANDLE;
  }
  views_.emplace(view_key, view);
  return view;
}

VkImageView VulkanTextureCache::VulkanTexture::GetOrCreate3DAs2DImageView(bool is_signed,
                                                                          uint32_t host_swizzle) {
  if (!REXCVAR_GET(gpu_3d_to_2d_texture)) {
    return VK_NULL_HANDLE;
  }

  VkImageView& cached_view =
      is_signed ? image_view_3d_as_2d_signed_ : image_view_3d_as_2d_unsigned_;
  if (cached_view != VK_NULL_HANDLE) {
    return cached_view;
  }

  VulkanTextureCache& vulkan_texture_cache = static_cast<VulkanTextureCache&>(texture_cache());

  if (!texture_3d_as_2d_) {
    const HostFormatPair& host_format_pair = vulkan_texture_cache.GetHostFormatPair(key());
    VkFormat wrapper_format = host_format_pair.format_unsigned.format;
    if (wrapper_format == VK_FORMAT_UNDEFINED) {
      wrapper_format = host_format_pair.format_signed.format;
    }
    if (wrapper_format == VK_FORMAT_UNDEFINED) {
      return VK_NULL_HANDLE;
    }

    VkImageCreateInfo image_create_info = {};
    image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_create_info.pNext = nullptr;
    image_create_info.flags = 0;
    image_create_info.imageType = VK_IMAGE_TYPE_2D;
    image_create_info.format = wrapper_format;
    image_create_info.extent.width = key().GetWidth();
    image_create_info.extent.height = key().GetHeight();
    image_create_info.extent.depth = 1;
    image_create_info.mipLevels = 1;
    image_create_info.arrayLayers = 1;
    image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_create_info.queueFamilyIndexCount = 0;
    image_create_info.pQueueFamilyIndices = nullptr;
    image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocation_create_info = {};
    allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VkImage image_2d;
    VmaAllocation allocation_2d;
    if (vmaCreateImage(vulkan_texture_cache.vma_allocator_, &image_create_info,
                       &allocation_create_info, &image_2d, &allocation_2d, nullptr) != VK_SUCCESS) {
      REXGPU_ERROR("VulkanTextureCache: Failed to create 3D-as-2D wrapper image");
      return VK_NULL_HANDLE;
    }

    TextureKey key_2d = key();
    key_2d.depth_or_array_size_minus_1 = 0;
    key_2d.mip_max_level = 0;
    texture_3d_as_2d_.reset(
        new VulkanTexture(vulkan_texture_cache, key_2d, image_2d, allocation_2d, false));
    texture_3d_as_2d_->SetForceLoad3DTiling(true);

    if (!vulkan_texture_cache.LoadTextureData(*texture_3d_as_2d_)) {
      REXGPU_ERROR("VulkanTextureCache: Failed to load 3D-as-2D wrapper data");
      texture_3d_as_2d_.reset();
      return VK_NULL_HANDLE;
    }

    if (!vulkan_texture_cache.command_processor_.PushImageMemoryBarrier(
            texture_3d_as_2d_->image(), ui::vulkan::util::InitializeSubresourceRange(),
            VK_PIPELINE_STAGE_TRANSFER_BIT, vulkan_texture_cache.guest_shader_pipeline_stages_,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)) {
      texture_3d_as_2d_.reset();
      return VK_NULL_HANDLE;
    }
    texture_3d_as_2d_->SetUsage(Usage::kGuestShaderSampled);
  }

  const HostFormatPair& host_format_pair = vulkan_texture_cache.GetHostFormatPair(key());
  VkFormat view_format =
      (is_signed ? host_format_pair.format_signed : host_format_pair.format_unsigned).format;
  if (view_format == VK_FORMAT_UNDEFINED) {
    return VK_NULL_HANDLE;
  }

  const ui::vulkan::VulkanDevice* const vulkan_device =
      vulkan_texture_cache.command_processor_.GetVulkanDevice();
  if (!vulkan_device->properties().imageViewFormatSwizzle) {
    host_swizzle = xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA;
  }

  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  VkImageViewCreateInfo view_create_info = {};
  view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_create_info.pNext = nullptr;
  view_create_info.flags = 0;
  view_create_info.image = texture_3d_as_2d_->image();
  view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  view_create_info.format = view_format;
  view_create_info.components.r = GetComponentSwizzle(host_swizzle, 0);
  view_create_info.components.g = GetComponentSwizzle(host_swizzle, 1);
  view_create_info.components.b = GetComponentSwizzle(host_swizzle, 2);
  view_create_info.components.a = GetComponentSwizzle(host_swizzle, 3);
  view_create_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange();
  view_create_info.subresourceRange.layerCount = 1;

  if (dfn.vkCreateImageView(device, &view_create_info, nullptr, &cached_view) != VK_SUCCESS) {
    REXGPU_ERROR("VulkanTextureCache: Failed to create 3D-as-2D wrapper image view");
    return VK_NULL_HANDLE;
  }

  return cached_view;
}

VulkanTextureCache::VulkanTextureCache(const RegisterFile& register_file,
                                       VulkanSharedMemory& shared_memory,
                                       uint32_t draw_resolution_scale_x,
                                       uint32_t draw_resolution_scale_y,
                                       VulkanCommandProcessor& command_processor,
                                       VkPipelineStageFlags guest_shader_pipeline_stages)
    : TextureCache(register_file, shared_memory, draw_resolution_scale_x, draw_resolution_scale_y),
      command_processor_(command_processor),
      guest_shader_pipeline_stages_(guest_shader_pipeline_stages) {}

bool VulkanTextureCache::InitializeScaledResolveBuffer() {
  assert_true(IsDrawResolutionScaled());

  uint64_t scale_area = uint64_t(draw_resolution_scale_x()) * uint64_t(draw_resolution_scale_y());
  scaled_resolve_buffer_size_ = uint64_t(SharedMemory::kBufferSize) * scale_area;

  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  const VkBufferCreateFlags sparse_flags =
      VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;

  VkBufferCreateInfo buffer_create_info;
  buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.pNext = nullptr;
  buffer_create_info.flags = sparse_flags;
  buffer_create_info.size = scaled_resolve_buffer_size_;
  buffer_create_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  buffer_create_info.queueFamilyIndexCount = 0;
  buffer_create_info.pQueueFamilyIndices = nullptr;

  if (vulkan_device->properties().sparseResidencyBuffer &&
      dfn.vkCreateBuffer(device, &buffer_create_info, nullptr, &scaled_resolve_buffer_) ==
          VK_SUCCESS) {
    VkMemoryRequirements memory_requirements;
    dfn.vkGetBufferMemoryRequirements(device, scaled_resolve_buffer_, &memory_requirements);
    if (rex::bit_scan_forward(
            memory_requirements.memoryTypeBits & vulkan_device->memory_types().device_local,
            &scaled_resolve_buffer_memory_type_)) {
      uint32_t alignment_log2;
      rex::bit_scan_forward(std::max(memory_requirements.alignment, VkDeviceSize(1)),
                            &alignment_log2);
      uint32_t size_log2 = rex::log2_ceil(scaled_resolve_buffer_size_);
      uint32_t preferred_log2 = std::max(uint32_t(0), size_log2 - uint32_t(10));
      scaled_resolve_sparse_granularity_log2_ =
          std::max(alignment_log2, std::max(uint32_t(22), preferred_log2));
      scaled_resolve_sparse_granularity_log2_ =
          std::min(scaled_resolve_sparse_granularity_log2_, size_log2);
      uint32_t allocation_count_log2 =
          std::max(uint32_t(6), size_log2 - scaled_resolve_sparse_granularity_log2_);
      scaled_resolve_sparse_allocated_.resize(size_t(1) << (allocation_count_log2 - uint32_t(6)));
      scaled_resolve_buffer_sparse_ = true;
    } else {
      dfn.vkDestroyBuffer(device, scaled_resolve_buffer_, nullptr);
      scaled_resolve_buffer_ = VK_NULL_HANDLE;
    }
  }

  if (scaled_resolve_buffer_ == VK_NULL_HANDLE) {
    buffer_create_info.flags = 0;
    if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
            vulkan_device, scaled_resolve_buffer_size_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            ui::vulkan::util::MemoryPurpose::kDeviceLocal, scaled_resolve_buffer_,
            scaled_resolve_buffer_memory_.emplace_back(), &scaled_resolve_buffer_memory_type_)) {
      REXGPU_ERROR(
          "VulkanTextureCache: Failed to create a scaled resolve buffer with "
          "{} MB address space",
          scaled_resolve_buffer_size_ >> 20);
      return false;
    }
    scaled_resolve_buffer_sparse_ = false;
    scaled_resolve_sparse_granularity_log2_ = UINT32_MAX;
    scaled_resolve_sparse_allocated_.clear();
  }

  scaled_resolve_last_usage_write_ = false;
  scaled_resolve_last_written_range_ = std::make_pair(uint64_t(0), uint64_t(0));
  scaled_resolve_unordered_writes_.clear();  // wwe2k14 patch (Session 22, RESWAW)
  return true;
}

void VulkanTextureCache::ShutdownScaledResolveBuffer() {
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, scaled_resolve_buffer_);
  for (VkDeviceMemory memory : scaled_resolve_buffer_memory_) {
    dfn.vkFreeMemory(device, memory, nullptr);
  }
  scaled_resolve_buffer_memory_.clear();
  scaled_resolve_sparse_allocated_.clear();
  scaled_resolve_buffer_size_ = 0;
  scaled_resolve_buffer_sparse_ = false;
  scaled_resolve_buffer_memory_type_ = UINT32_MAX;
  scaled_resolve_sparse_granularity_log2_ = UINT32_MAX;
  scaled_resolve_last_usage_write_ = false;
  scaled_resolve_last_written_range_ = std::make_pair(uint64_t(0), uint64_t(0));
  scaled_resolve_unordered_writes_.clear();  // wwe2k14 patch (Session 22, RESWAW)
}

bool VulkanTextureCache::GetScaledResolveRange(uint32_t start_unscaled, uint32_t length_unscaled,
                                               uint32_t length_scaled_alignment_log2,
                                               uint64_t& start_scaled_out,
                                               uint64_t& length_scaled_out) const {
  if (!IsDrawResolutionScaled()) {
    return false;
  }
  if (length_unscaled == 0) {
    start_scaled_out = 0;
    length_scaled_out = 0;
    return true;
  }
  if (start_unscaled > SharedMemory::kBufferSize ||
      (SharedMemory::kBufferSize - start_unscaled) < length_unscaled) {
    return false;
  }

  uint64_t scale_area = uint64_t(draw_resolution_scale_x()) * uint64_t(draw_resolution_scale_y());
  uint64_t start_scaled = uint64_t(start_unscaled) * scale_area;
  uint64_t length_scaled = uint64_t(length_unscaled) * scale_area;
  uint64_t alignment = uint64_t(1) << length_scaled_alignment_log2;
  length_scaled = rex::align(length_scaled, alignment);

  if (start_scaled > scaled_resolve_buffer_size_ ||
      length_scaled > scaled_resolve_buffer_size_ - start_scaled) {
    return false;
  }

  start_scaled_out = start_scaled;
  length_scaled_out = length_scaled;
  return true;
}

bool VulkanTextureCache::EnsureScaledResolveBufferAllocated(uint64_t start_scaled,
                                                            uint64_t length_scaled) {
  if (!length_scaled || !scaled_resolve_buffer_sparse_) {
    return true;
  }
  uint32_t allocation_first = uint32_t(start_scaled >> scaled_resolve_sparse_granularity_log2_);
  uint32_t allocation_last =
      uint32_t((start_scaled + (length_scaled - 1)) >> scaled_resolve_sparse_granularity_log2_);
  while (true) {
    std::pair<size_t, size_t> allocation_range =
        rex::bit::GetNextRangeUnset(scaled_resolve_sparse_allocated_.data(), allocation_first,
                                    allocation_last - allocation_first + 1);
    if (!allocation_range.second) {
      break;
    }

    const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
    const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();

    VkMemoryAllocateInfo memory_allocate_info;
    memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memory_allocate_info.pNext = nullptr;
    memory_allocate_info.allocationSize = VkDeviceSize(allocation_range.second)
                                          << scaled_resolve_sparse_granularity_log2_;
    memory_allocate_info.memoryTypeIndex = scaled_resolve_buffer_memory_type_;
    VkDeviceMemory allocation_memory;
    if (dfn.vkAllocateMemory(device, &memory_allocate_info, nullptr, &allocation_memory) !=
        VK_SUCCESS) {
      REXGPU_ERROR(
          "VulkanTextureCache: Failed to allocate sparse memory for the "
          "scaled resolve buffer");
      return false;
    }
    scaled_resolve_buffer_memory_.push_back(allocation_memory);

    VkSparseMemoryBind bind;
    bind.resourceOffset = VkDeviceSize(allocation_range.first)
                          << scaled_resolve_sparse_granularity_log2_;
    bind.size = memory_allocate_info.allocationSize;
    bind.memory = allocation_memory;
    bind.memoryOffset = 0;
    bind.flags = 0;
    command_processor_.SparseBindBuffer(
        scaled_resolve_buffer_, 1, &bind,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);

    rex::bit::SetRange(scaled_resolve_sparse_allocated_.data(), allocation_range.first,
                       allocation_range.second);
    allocation_first = uint32_t(allocation_range.first + allocation_range.second);
  }
  return true;
}

void VulkanTextureCache::GetScaledResolveUsageMasks(VkPipelineStageFlags& stage_mask_out,
                                                    VkAccessFlags& access_mask_out,
                                                    bool write) const {
  stage_mask_out = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  access_mask_out = VK_ACCESS_SHADER_READ_BIT;
  if (write) {
    access_mask_out |= VK_ACCESS_SHADER_WRITE_BIT;
  }
}

void VulkanTextureCache::UseScaledResolveBufferForRead() {
  if (!scaled_resolve_buffer_) {
    return;
  }
  if (!scaled_resolve_last_usage_write_ && !scaled_resolve_last_written_range_.second) {
    return;
  }

  VkPipelineStageFlags src_stage_mask, dst_stage_mask;
  VkAccessFlags src_access_mask, dst_access_mask;
  GetScaledResolveUsageMasks(src_stage_mask, src_access_mask, scaled_resolve_last_usage_write_);
  GetScaledResolveUsageMasks(dst_stage_mask, dst_access_mask, false);
  VkDeviceSize offset;
  VkDeviceSize size;
  if (!scaled_resolve_last_usage_write_) {
    offset = VkDeviceSize(scaled_resolve_last_written_range_.first);
    size = VkDeviceSize(scaled_resolve_last_written_range_.second);
  } else {
    offset = 0;
    size = VK_WHOLE_SIZE;
    scaled_resolve_last_usage_write_ = false;
    // wwe2k14 patch (Session 22, RESWAW): the whole-buffer barrier orders every earlier write.
    scaled_resolve_unordered_writes_.clear();
  }
  command_processor_.PushBufferMemoryBarrier(
      scaled_resolve_buffer_, offset, size, src_stage_mask, dst_stage_mask, src_access_mask,
      dst_access_mask, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
  scaled_resolve_last_written_range_ = std::make_pair(uint64_t(0), uint64_t(0));
}

// wwe2k14 patch (Session 22, EAGER).
void VulkanTextureCache::EagerRotate(uint64_t frame) {
  // wwe2k14 patch (Session 87, step 3): the planned textures a draw sampled in the frame that ends are planned again.
  cpu_plan_prev_.clear();
  for (VulkanTexture* texture : cpu_plan_next_) {
    if (std::find(cpu_used_.begin(), cpu_used_.end(), texture) != cpu_used_.end()) cpu_plan_prev_.push_back(texture);
  }
  cpu_plan_next_.clear();
  cpu_used_.clear();
  cpu_pending_ = cpu_plan_prev_;
  eager_plan_prev_.swap(eager_plan_next_);
  eager_plan_next_.clear();
  eager_plan_cursor_ = 0;
  eager_resolves_ = 0;
  eager_draws_since_resolve_ = 0;
  eager_pending_.clear();
  eager_frame_ = frame;
}

void VulkanTextureCache::EagerNoteResolve() {
  const uint64_t frame = command_processor_.GetCurrentFrame();
  if (frame != eager_frame_) {
    EagerRotate(frame);
  }
  ++eager_resolves_;
  eager_draws_since_resolve_ = 0;
  const bool on = REXCVAR_GET(vulkan_texture_load_eager);
  while (eager_plan_cursor_ < eager_plan_prev_.size() &&
         eager_plan_prev_[eager_plan_cursor_].first <= eager_resolves_) {
    const std::pair<uint32_t, VulkanTexture*>& entry = eager_plan_prev_[eager_plan_cursor_++];
    if (on && entry.first == eager_resolves_ && entry.second) {
      eager_pending_.push_back(entry.second);
    }
  }
}

void VulkanTextureCache::EagerRecordLoad(VulkanTexture& texture) {
  const uint64_t frame = command_processor_.GetCurrentFrame();
  if (frame != eager_frame_) {
    EagerRotate(frame);
  }
  // wwe2k14 patch (Session 87, step 3): vulkan_texture_load_early_cpu. Planned for the next frame: a texture loaded at its use
  // because the CPU alone invalidated it (a first load, or one a resolve invalidated, is not) - and one that was loaded early
  // this frame stays in the plan (else it would be learned every other frame).
  if (REXCVAR_GET(vulkan_texture_load_early_cpu)) {
    const uint32_t invalidated = texture.invalidated_since_load();
    if (cpu_loading_ || invalidated == Texture::kInvalidatedByCpu) {
      if (cpu_plan_next_.size() < 32 && std::find(cpu_plan_next_.begin(), cpu_plan_next_.end(), &texture) == cpu_plan_next_.end()) {
        cpu_plan_next_.push_back(&texture);
        if (!cpu_loading_) ++cpu_early_stats_.learned;
      }
    }
    if (!cpu_loading_ && texture.eager_loaded_frame() == frame) ++cpu_early_stats_.reloaded;
    if (cpu_loading_) return;
  }
  if (eager_loading_) {
    // An early load stays in the plan (at the resolve it now follows), or the next frame would load it late
    // again (every other frame).
    if (eager_plan_next_.size() < 256) {
      eager_plan_next_.emplace_back(eager_resolves_, &texture);
    }
    return;
  }
  if (texture.eager_loaded_frame() == frame) {
    ++eager_stats_.reloaded;
  }
  // A late load: of resolved data, after this frame's eager_resolves_-th resolve, 2+ draws after it (the first
  // draw after a resolve gains nothing from loading earlier).
  if (!texture.key().scaled_resolve || !eager_resolves_ || eager_draws_since_resolve_ < 2 ||
      eager_plan_next_.size() >= 256) {
    return;
  }
  eager_plan_next_.emplace_back(eager_resolves_, &texture);
  ++eager_stats_.learned;
}

void VulkanTextureCache::EagerForget(const VulkanTexture* texture) {
  for (std::pair<uint32_t, VulkanTexture*>& entry : eager_plan_prev_) {
    if (entry.second == texture) {
      entry.second = nullptr;
    }
  }
  std::erase_if(eager_plan_next_,
                [texture](const std::pair<uint32_t, VulkanTexture*>& e) { return e.second == texture; });
  std::erase(eager_pending_, texture);
  // (Session 87, step 3)
  std::erase(cpu_plan_prev_, const_cast<VulkanTexture*>(texture));
  std::erase(cpu_plan_next_, const_cast<VulkanTexture*>(texture));
  std::erase(cpu_pending_, const_cast<VulkanTexture*>(texture));
  std::erase(cpu_used_, const_cast<VulkanTexture*>(texture));
}

void VulkanTextureCache::UseScaledResolveBufferForWrite(uint64_t written_start_scaled,
                                                        uint64_t written_length_scaled) {
  if (!scaled_resolve_buffer_) {
    return;
  }
  if (written_start_scaled > scaled_resolve_buffer_size_) {
    written_start_scaled = scaled_resolve_buffer_size_;
  }
  written_length_scaled =
      std::min(written_length_scaled, scaled_resolve_buffer_size_ - written_start_scaled);

  // wwe2k14 patch (Session 22, RESWAW): vulkan_resolve_skip_disjoint_waw. Upstream orders every resolve after the
  // previous one's write (a write-after-write barrier on the previous range), which costs a pipeline barrier between
  // two resolves in a row even when they write different bytes (Bryan's pan: 29 of ~100 resolves per frame follow
  // another resolve directly, and that barrier is their only one). Why skipping it for non-overlapping ranges is safe:
  // - The barrier only orders this resolve's writes after the earlier ones on the same bytes (WAW). Two dispatches
  //   that write disjoint bytes and read nothing of this buffer have no hazard between them; the resolve reads only
  //   the EDRAM buffer, which UseEdramBuffer orders separately (unchanged).
  // - Every write not yet ordered is kept in scaled_resolve_unordered_writes_ (upstream needs only the last one,
  //   because it orders each write before the next). A write that overlaps any of them gets the barrier, over the
  //   whole buffer when more than one write is unordered.
  // - Readers are unchanged: every read of the buffer (texture loads, the readback downscale) calls
  //   UseScaledResolveBufferForRead, which after writes records a whole-buffer write -> read barrier (all compute
  //   writes before it, including skipped ones, made visible), and clears the list. The read -> write barrier before
  //   the first write after reads stays as upstream (whole buffer), and clears the list.
  // - Off (default): one entry at most, identical barriers to upstream. Turning it off after skips: the barrier
  //   covers the whole buffer while more than one write is unordered.
  if (scaled_resolve_last_usage_write_ &&
      (scaled_resolve_last_written_range_.second || !scaled_resolve_unordered_writes_.empty())) {
    bool skip = false;
    if (REXCVAR_GET(vulkan_resolve_skip_disjoint_waw) && scaled_resolve_unordered_writes_.size() < 64) {
      skip = true;
      for (const std::pair<uint64_t, uint64_t>& w : scaled_resolve_unordered_writes_) {
        if (written_start_scaled < w.first + w.second && w.first < written_start_scaled + written_length_scaled) {
          skip = false;
          break;
        }
      }
    }
    if (skip) {
      ++resolve_waw_skipped_;
    } else {
      VkDeviceSize offset = VkDeviceSize(scaled_resolve_last_written_range_.first);
      VkDeviceSize size = VkDeviceSize(scaled_resolve_last_written_range_.second);
      if (scaled_resolve_unordered_writes_.size() > 1 || !size) {
        offset = 0;
        size = VK_WHOLE_SIZE;
      }
      VkPipelineStageFlags stage_mask;
      VkAccessFlags access_mask;
      GetScaledResolveUsageMasks(stage_mask, access_mask, true);
      command_processor_.PushBufferMemoryBarrier(
          scaled_resolve_buffer_, offset, size, stage_mask, stage_mask, access_mask, access_mask,
          VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
      scaled_resolve_unordered_writes_.clear();
      ++resolve_waw_kept_;
    }
  } else if (!scaled_resolve_last_usage_write_) {
    VkPipelineStageFlags src_stage_mask, dst_stage_mask;
    VkAccessFlags src_access_mask, dst_access_mask;
    GetScaledResolveUsageMasks(src_stage_mask, src_access_mask, false);
    GetScaledResolveUsageMasks(dst_stage_mask, dst_access_mask, true);
    command_processor_.PushBufferMemoryBarrier(
        scaled_resolve_buffer_, 0, VK_WHOLE_SIZE, src_stage_mask, dst_stage_mask, src_access_mask,
        dst_access_mask, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
    scaled_resolve_last_usage_write_ = true;
    scaled_resolve_unordered_writes_.clear();  // wwe2k14 patch (Session 22, RESWAW)
  }

  scaled_resolve_last_written_range_ = std::make_pair(written_start_scaled, written_length_scaled);
  if (written_length_scaled) {
    scaled_resolve_unordered_writes_.push_back(scaled_resolve_last_written_range_);  // Session 22, RESWAW
  }
}

namespace {
bool g_emulate_bc1_active = false;  // (Session 95) the BC1 test emulation took the DXT1 format
}
bool VulkanTextureCache::Initialize() {
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanInstance::Functions& ifn = vulkan_device->vulkan_instance()->functions();
  const VkPhysicalDevice physical_device = vulkan_device->physical_device();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  const ui::vulkan::VulkanDevice::Properties& device_properties = vulkan_device->properties();

  // Vulkan Memory Allocator.

  vma_allocator_ = ui::vulkan::CreateVmaAllocator(vulkan_device, true);
  if (vma_allocator_ == VK_NULL_HANDLE) {
    return false;
  }

  if (IsDrawResolutionScaled() && !InitializeScaledResolveBuffer()) {
    return false;
  }

  // Image formats.

  // Initialize to the best formats.
  for (size_t i = 0; i < rex::countof(host_formats_); ++i) {
    host_formats_[i] = kBestHostFormats[i];
  }

  // Check format support and switch to fallbacks if needed.
  constexpr VkFormatFeatureFlags kLinearFilterFeatures =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
  VkFormatProperties r16_unorm_properties;
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R16_UNORM,
                                          &r16_unorm_properties);
  VkFormatProperties r16_snorm_properties;
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R16_SNORM,
                                          &r16_snorm_properties);
  VkFormatProperties r16g16_unorm_properties;
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R16G16_UNORM,
                                          &r16g16_unorm_properties);
  VkFormatProperties r16g16_snorm_properties;
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R16G16_SNORM,
                                          &r16g16_snorm_properties);
  VkFormatProperties r16g16b16a16_unorm_properties;
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R16G16B16A16_UNORM,
                                          &r16g16b16a16_unorm_properties);
  VkFormatProperties r16g16b16a16_snorm_properties;
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R16G16B16A16_SNORM,
                                          &r16g16b16a16_snorm_properties);
  VkFormatProperties r16g16b16a16_sfloat_properties;
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R16G16B16A16_SFLOAT,
                                          &r16g16b16a16_sfloat_properties);
  VkFormatProperties format_properties;
  // k_Cr_Y1_Cb_Y0_REP, k_Y1_Cr_Y0_Cb_REP.
  HostFormatPair& host_format_gbgr =
      host_formats_[uint32_t(xenos::TextureFormat::k_Cr_Y1_Cb_Y0_REP)];
  assert_true(host_format_gbgr.format_unsigned.format == VK_FORMAT_G8B8G8R8_422_UNORM_KHR);
  assert_true(host_format_gbgr.format_signed.format == VK_FORMAT_UNDEFINED);
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_G8B8G8R8_422_UNORM_KHR,
                                          &format_properties);
  if ((format_properties.optimalTilingFeatures & kLinearFilterFeatures) != kLinearFilterFeatures) {
    host_format_gbgr.format_unsigned.load_shader = kLoadShaderIndexGBGR8ToRGB8;
    host_format_gbgr.format_unsigned.format = VK_FORMAT_R8G8B8A8_UNORM;
    host_format_gbgr.format_unsigned.block_compressed = false;
    host_format_gbgr.unsigned_signed_compatible = false;
  }
  HostFormatPair& host_format_bgrg =
      host_formats_[uint32_t(xenos::TextureFormat::k_Y1_Cr_Y0_Cb_REP)];
  assert_true(host_format_bgrg.format_unsigned.format == VK_FORMAT_B8G8R8G8_422_UNORM_KHR);
  assert_true(host_format_bgrg.format_signed.format == VK_FORMAT_UNDEFINED);
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_B8G8R8G8_422_UNORM_KHR,
                                          &format_properties);
  if ((format_properties.optimalTilingFeatures & kLinearFilterFeatures) != kLinearFilterFeatures) {
    host_format_bgrg.format_unsigned.load_shader = kLoadShaderIndexBGRG8ToRGB8;
    host_format_bgrg.format_unsigned.format = VK_FORMAT_R8G8B8A8_UNORM;
    host_format_bgrg.format_unsigned.block_compressed = false;
    host_format_bgrg.unsigned_signed_compatible = false;
  }
  // Expand R5G6B5 while loading if the native packed format can't be sampled.
  HostFormatPair& host_format_r5g6b5 = host_formats_[uint32_t(xenos::TextureFormat::k_5_6_5)];
  assert_true(host_format_r5g6b5.format_unsigned.format == VK_FORMAT_R5G6B5_UNORM_PACK16);
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R5G6B5_UNORM_PACK16,
                                          &format_properties);
  if ((format_properties.optimalTilingFeatures & kLinearFilterFeatures) != kLinearFilterFeatures) {
    host_format_r5g6b5.format_unsigned.load_shader = kLoadShaderIndexR5G6B5ToRGBA8;
    host_format_r5g6b5.format_unsigned.format = VK_FORMAT_R8G8B8A8_UNORM;
    host_format_r5g6b5.format_unsigned.block_compressed = false;
    host_format_r5g6b5.unsigned_signed_compatible = false;
  }
  // MoltenVK doesn't expose B4G4R4A4 sampling on all Metal devices. Expand
  // guest RGBA4444 texels to the universally supported RGBA8 format while
  // loading them instead of dropping the texture binding entirely.
  HostFormatPair& host_format_rgba4 = host_formats_[uint32_t(xenos::TextureFormat::k_4_4_4_4)];
  assert_true(host_format_rgba4.format_unsigned.format == VK_FORMAT_B4G4R4A4_UNORM_PACK16);
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_B4G4R4A4_UNORM_PACK16,
                                          &format_properties);
  if ((format_properties.optimalTilingFeatures & kLinearFilterFeatures) != kLinearFilterFeatures) {
    host_format_rgba4.format_unsigned.load_shader = kLoadShaderIndexRGBA4ToRGBA8;
    host_format_rgba4.format_unsigned.format = VK_FORMAT_R8G8B8A8_UNORM;
    host_format_rgba4.format_unsigned.block_compressed = false;
    host_format_rgba4.unsigned_signed_compatible = false;
  }
  // k_10_11_11, k_11_11_10 (+ _AS_16_16_16_16 variants) - if normalized
  // R16G16B16A16 is not linearly filterable, decode to normalized 16-bit in a
  // scratch buffer first, then convert it to float16 in-place in a second pass
  // before copying to the image.
  bool rgba16_norm_filterable =
      (r16g16b16a16_unorm_properties.optimalTilingFeatures & kLinearFilterFeatures) ==
          kLinearFilterFeatures &&
      (r16g16b16a16_snorm_properties.optimalTilingFeatures & kLinearFilterFeatures) ==
          kLinearFilterFeatures;
  bool rgba16_float_filterable = (r16g16b16a16_sfloat_properties.optimalTilingFeatures &
                                  kLinearFilterFeatures) == kLinearFilterFeatures;
  if (!rgba16_norm_filterable) {
    if (!rgba16_float_filterable) {
      REXGPU_ERROR(
          "VulkanTextureCache: Neither normalized nor float16 RGBA16 formats are linearly "
          "filterable for 10_11_11/11_11_10 texture fallback");
      return false;
    }
    REXGPU_WARN("VulkanTextureCache: Using RGBA16 float fallback for 10_11_11 / 11_11_10 textures");
    constexpr std::array<xenos::TextureFormat, 4> kR10G11B11Formats = {
        xenos::TextureFormat::k_10_11_11,
        xenos::TextureFormat::k_11_11_10,
        xenos::TextureFormat::k_10_11_11_AS_16_16_16_16,
        xenos::TextureFormat::k_11_11_10_AS_16_16_16_16,
    };
    for (xenos::TextureFormat format : kR10G11B11Formats) {
      HostFormatPair& host_format = host_formats_[uint32_t(format)];
      assert_true(host_format.format_unsigned.format == VK_FORMAT_R16G16B16A16_UNORM);
      assert_true(host_format.format_signed.format == VK_FORMAT_R16G16B16A16_SNORM);
      host_format.format_unsigned.format = VK_FORMAT_R16G16B16A16_SFLOAT;
      host_format.format_signed.format = VK_FORMAT_R16G16B16A16_SFLOAT;
      // Different source signednesses still need separate decoded data.
      host_format.unsigned_signed_compatible = false;
    }
  }
  // S3TC.
  // Not checking the textureCompressionBC feature because its availability
  // means that all BC formats are supported, however, the device may expose
  // some BC formats without this feature. Xenia doesn't use BC6H and BC7 at
  // all, and has fallbacks for each used format.
  // TODO(Triang3l): Raise the host texture memory usage limit if S3TC has to be
  // decompressed.
  // TODO(Triang3l): S3TC -> 5551 or 4444 as an option.
  // TODO(Triang3l): S3TC -> ETC2 / EAC (a huge research topic).
  // wwe2k14 patch (Session 93): vulkan_emulate_no_bc / vulkan_texture_transcode. The selection is logged once per format.
  const bool emulate_no_bc = REXCVAR_GET(vulkan_emulate_no_bc);
  const bool transcode_requested = REXCVAR_GET(vulkan_texture_transcode);
  const bool transcode_eac_requested = transcode_requested && REXCVAR_GET(vulkan_texture_transcode_eac);
  auto bc_unsupported = [&](VkFormat bc_format) {
    VkFormatProperties bc_properties;
    ifn.vkGetPhysicalDeviceFormatProperties(physical_device, bc_format, &bc_properties);
    return emulate_no_bc ||
           (bc_properties.optimalTilingFeatures & kLinearFilterFeatures) != kLinearFilterFeatures;
  };
  auto transcode_target_ok = [&](VkFormat target_format, bool switch_on) {
    if (!switch_on) {
      return false;
    }
    VkFormatProperties target_properties;
    ifn.vkGetPhysicalDeviceFormatProperties(physical_device, target_format, &target_properties);
    return (target_properties.optimalTilingFeatures & kLinearFilterFeatures) == kLinearFilterFeatures;
  };
  // Logs the path chosen for one BC format: "<source> -> <host format> (<why>)".
  auto log_choice = [&](const char* source, bool bc_missing, const char* target_name, bool target_ok,
                        const char* decompressed_name, bool switch_on = true,
                        const char* switch_name = "vulkan_texture_transcode") {
    if (!bc_missing) {
      REXGPU_INFO("[tex][transcode] {} -> BC (supported)", source);
    } else if (target_ok) {
      REXGPU_INFO("[tex][transcode] {} -> {} (BC missing{}, {} sampled+filter yes, chosen by {})", source, target_name,
                  emulate_no_bc ? " [emulated]" : "", target_name, switch_name);
    } else if (!switch_on || target_name == nullptr) {
      // (Session 95, Lane E: the callers pass the plain switch name; "off" is added here, so the "chosen by" above no longer reads "<switch> off".)
      REXGPU_INFO("[tex][transcode] {} -> {} (BC missing{}, {}{})", source, decompressed_name,
                  emulate_no_bc ? " [emulated]" : "",
                  target_name == nullptr ? "no transcode for this format" : switch_name,
                  target_name == nullptr ? "" : " off");
    } else {
      REXGPU_INFO("[tex][transcode] {} -> {} (BC missing{}, no target: {} sampled+filter no)", source,
                  decompressed_name, emulate_no_bc ? " [emulated]" : "", target_name);
    }
  };
  HostFormatPair& host_format_dxt1 = host_formats_[uint32_t(xenos::TextureFormat::k_DXT1)];
  assert_true(host_format_dxt1.format_unsigned.format == VK_FORMAT_BC1_RGBA_UNORM_BLOCK);
  g_emulate_bc1_active = REXCVAR_GET(vulkan_emulate_bc1_transcode) && !bc_unsupported(VK_FORMAT_BC1_RGBA_UNORM_BLOCK);
  if (g_emulate_bc1_active) {  // (Session 95, test only)
    host_format_dxt1.format_unsigned.load_shader = kLoadShaderIndexDXT1ToASTC4x4;
    host_format_dxt1.format_unsigned.format = VK_FORMAT_BC3_UNORM_BLOCK;
    host_formats_[uint32_t(xenos::TextureFormat::k_DXT1_AS_16_16_16_16)] = host_format_dxt1;
    REXGPU_INFO("[tex] vulkan_emulate_bc1_transcode {}: DXT1 through BC3 with the transcode's transparent texels", REXCVAR_GET(vulkan_emulate_bc1_transcode));
  }
  if (bc_unsupported(VK_FORMAT_BC1_RGBA_UNORM_BLOCK)) {
    bool astc_ok = transcode_target_ok(VK_FORMAT_ASTC_4x4_UNORM_BLOCK, transcode_requested);
    log_choice("DXT1", true, "ASTC_4x4", astc_ok, "RGBA8", transcode_requested, "vulkan_texture_transcode");
    if (astc_ok) {
      host_format_dxt1.format_unsigned.load_shader = kLoadShaderIndexDXT1ToASTC4x4;
      host_format_dxt1.format_unsigned.format = VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
      // block_compressed stays true: the host block is 4x4 like the guest one, copied block by block.
    } else {
      host_format_dxt1.format_unsigned.load_shader = kLoadShaderIndexDXT1ToRGBA8;
      host_format_dxt1.format_unsigned.format = VK_FORMAT_R8G8B8A8_UNORM;
      host_format_dxt1.format_unsigned.block_compressed = false;
    }
    host_formats_[uint32_t(xenos::TextureFormat::k_DXT1_AS_16_16_16_16)] = host_format_dxt1;
  } else {
    log_choice("DXT1", false, nullptr, false, nullptr);
  }
  HostFormatPair& host_format_dxt2_3 = host_formats_[uint32_t(xenos::TextureFormat::k_DXT2_3)];
  assert_true(host_format_dxt2_3.format_unsigned.format == VK_FORMAT_BC2_UNORM_BLOCK);
  if (bc_unsupported(VK_FORMAT_BC2_UNORM_BLOCK)) {
    log_choice("DXT3", true, nullptr, false, "RGBA8");
    host_format_dxt2_3.format_unsigned.load_shader = kLoadShaderIndexDXT3ToRGBA8;
    host_format_dxt2_3.format_unsigned.format = VK_FORMAT_R8G8B8A8_UNORM;
    host_format_dxt2_3.format_unsigned.block_compressed = false;
    host_formats_[uint32_t(xenos::TextureFormat::k_DXT2_3_AS_16_16_16_16)] = host_format_dxt2_3;
  }
  HostFormatPair& host_format_dxt4_5 = host_formats_[uint32_t(xenos::TextureFormat::k_DXT4_5)];
  assert_true(host_format_dxt4_5.format_unsigned.format == VK_FORMAT_BC3_UNORM_BLOCK);
  if (bc_unsupported(VK_FORMAT_BC3_UNORM_BLOCK)) {
    log_choice("DXT5", true, nullptr, false, "RGBA8");
    host_format_dxt4_5.format_unsigned.load_shader = kLoadShaderIndexDXT5ToRGBA8;
    host_format_dxt4_5.format_unsigned.format = VK_FORMAT_R8G8B8A8_UNORM;
    host_format_dxt4_5.format_unsigned.block_compressed = false;
    host_formats_[uint32_t(xenos::TextureFormat::k_DXT4_5_AS_16_16_16_16)] = host_format_dxt4_5;
  }
  HostFormatPair& host_format_dxn = host_formats_[uint32_t(xenos::TextureFormat::k_DXN)];
  assert_true(host_format_dxn.format_unsigned.format == VK_FORMAT_BC5_UNORM_BLOCK);
  if (bc_unsupported(VK_FORMAT_BC5_UNORM_BLOCK)) {
    bool eac_ok = transcode_target_ok(VK_FORMAT_EAC_R11G11_UNORM_BLOCK, transcode_eac_requested);
    log_choice("DXN", true, "EAC_RG11", eac_ok, "RG8", transcode_eac_requested, "vulkan_texture_transcode_eac");
    if (eac_ok) {
      host_format_dxn.format_unsigned.load_shader = kLoadShaderIndexDXNToEACRG11;
      host_format_dxn.format_unsigned.format = VK_FORMAT_EAC_R11G11_UNORM_BLOCK;
    } else {
      host_format_dxn.format_unsigned.load_shader = kLoadShaderIndexDXNToRG8;
      host_format_dxn.format_unsigned.format = VK_FORMAT_R8G8_UNORM;
      host_format_dxn.format_unsigned.block_compressed = false;
    }
  } else {
    log_choice("DXN", false, nullptr, false, nullptr);
  }
  HostFormatPair& host_format_dxt5a = host_formats_[uint32_t(xenos::TextureFormat::k_DXT5A)];
  assert_true(host_format_dxt5a.format_unsigned.format == VK_FORMAT_BC4_UNORM_BLOCK);
  if (bc_unsupported(VK_FORMAT_BC4_UNORM_BLOCK)) {
    bool eac_ok = transcode_target_ok(VK_FORMAT_EAC_R11_UNORM_BLOCK, transcode_eac_requested);
    log_choice("DXT5A", true, "EAC_R11", eac_ok, "R8", transcode_eac_requested, "vulkan_texture_transcode_eac");
    if (eac_ok) {
      host_format_dxt5a.format_unsigned.load_shader = kLoadShaderIndexDXT5AToEACR11;
      host_format_dxt5a.format_unsigned.format = VK_FORMAT_EAC_R11_UNORM_BLOCK;
    } else {
      host_format_dxt5a.format_unsigned.load_shader = kLoadShaderIndexDXT5AToR8;
      host_format_dxt5a.format_unsigned.format = VK_FORMAT_R8_UNORM;
      host_format_dxt5a.format_unsigned.block_compressed = false;
    }
  } else {
    log_choice("DXT5A", false, nullptr, false, nullptr);
  }
  // Keep signed format support aligned with D3D12 for texture formats where
  // D3D12 doesn't expose signed sampling paths.
  constexpr std::array<xenos::TextureFormat, 19> kD3D12SignedUnsupportedFormats = {
      xenos::TextureFormat::k_1_5_5_5,
      xenos::TextureFormat::k_5_6_5,
      xenos::TextureFormat::k_6_5_5,
      xenos::TextureFormat::k_2_10_10_10,
      xenos::TextureFormat::k_Cr_Y1_Cb_Y0_REP,
      xenos::TextureFormat::k_Y1_Cr_Y0_Cb_REP,
      xenos::TextureFormat::k_4_4_4_4,
      xenos::TextureFormat::k_DXT1,
      xenos::TextureFormat::k_DXT2_3,
      xenos::TextureFormat::k_DXT4_5,
      xenos::TextureFormat::k_DXN,
      xenos::TextureFormat::k_DXT5A,
      xenos::TextureFormat::k_CTX1,
      xenos::TextureFormat::k_DXT1_AS_16_16_16_16,
      xenos::TextureFormat::k_DXT2_3_AS_16_16_16_16,
      xenos::TextureFormat::k_DXT4_5_AS_16_16_16_16,
      xenos::TextureFormat::k_2_10_10_10_AS_16_16_16_16,
      xenos::TextureFormat::k_DXT3A,
      xenos::TextureFormat::k_DXT3A_AS_1_1_1_1,
  };
  for (xenos::TextureFormat format : kD3D12SignedUnsupportedFormats) {
    HostFormatPair& host_format = host_formats_[uint32_t(format)];
    host_format.format_signed.load_shader = kLoadShaderIndexUnknown;
    host_format.format_signed.format = VK_FORMAT_UNDEFINED;
    host_format.format_signed.linear_filterable = false;
    host_format.unsigned_signed_compatible = false;
  }
  // k_16, k_16_16, k_16_16_16_16 - UNORM / SNORM are optional, fall back to
  // SFLOAT, which is mandatory and is always filterable.
  // For D3D12 parity, prefer keeping normalized-format sampling filterable over
  // keeping fixed-point precision if the host lacks linear filtering support on
  // UNORM/SNORM variants.
  VkFormatFeatureFlags norm16_required_features = kLinearFilterFeatures;
  // Session 94 (vulkan_texture_16bit_filter_substitute): auto = by the format query (the previous behaviour), on /
  // off override it. One log line per substituted format.
  std::string substitute_mode = REXCVAR_GET(vulkan_texture_16bit_filter_substitute);
  std::transform(substitute_mode.begin(), substitute_mode.end(), substitute_mode.begin(),
                 [](unsigned char ch) { return char(std::tolower(ch)); });
  auto norm16_substitute = [&](const VkFormatProperties& properties, const char* from, const char* to) {
    const bool filterable = (properties.optimalTilingFeatures & norm16_required_features) == norm16_required_features;
    const bool use = substitute_mode == "on" || (substitute_mode != "off" && !filterable);
    if (use) {
      REXGPU_INFO("[tex] 16-bit substitute: {} -> {} (linear filter {}, vulkan_texture_16bit_filter_substitute={})",
                  from, to, filterable ? "yes, forced" : "no", substitute_mode);
    } else if (!filterable) {
      REXGPU_INFO("[tex] 16-bit substitute: {} stays (linear filter no, vulkan_texture_16bit_filter_substitute=off)",
                  from);
    }
    return use;
  };
  HostFormatPair& host_format_16 = host_formats_[uint32_t(xenos::TextureFormat::k_16)];
  assert_true(host_format_16.format_unsigned.format == VK_FORMAT_R16_UNORM);
  if (norm16_substitute(r16_unorm_properties, "R16_UNORM", "R16_SFLOAT")) {
    host_format_16.format_unsigned.load_shader = kLoadShaderIndexR16UNormToFloat;
    host_format_16.format_unsigned.format = VK_FORMAT_R16_SFLOAT;
  }
  assert_true(host_format_16.format_signed.format == VK_FORMAT_R16_SNORM);
  if (norm16_substitute(r16_snorm_properties, "R16_SNORM", "R16_SFLOAT")) {
    host_format_16.format_signed.load_shader = kLoadShaderIndexR16SNormToFloat;
    host_format_16.format_signed.format = VK_FORMAT_R16_SFLOAT;
  }
  host_format_16.unsigned_signed_compatible =
      (host_format_16.format_unsigned.format == VK_FORMAT_R16_UNORM &&
       host_format_16.format_signed.format == VK_FORMAT_R16_SNORM) ||
      (host_format_16.format_unsigned.format == VK_FORMAT_R16_SFLOAT &&
       host_format_16.format_signed.format == VK_FORMAT_R16_SFLOAT);
  HostFormatPair& host_format_16_16 = host_formats_[uint32_t(xenos::TextureFormat::k_16_16)];
  assert_true(host_format_16_16.format_unsigned.format == VK_FORMAT_R16G16_UNORM);
  if (norm16_substitute(r16g16_unorm_properties, "RG16_UNORM", "RG16_SFLOAT")) {
    host_format_16_16.format_unsigned.load_shader = kLoadShaderIndexRG16UNormToFloat;
    host_format_16_16.format_unsigned.format = VK_FORMAT_R16G16_SFLOAT;
  }
  assert_true(host_format_16_16.format_signed.format == VK_FORMAT_R16G16_SNORM);
  if (norm16_substitute(r16g16_snorm_properties, "RG16_SNORM", "RG16_SFLOAT")) {
    host_format_16_16.format_signed.load_shader = kLoadShaderIndexRG16SNormToFloat;
    host_format_16_16.format_signed.format = VK_FORMAT_R16G16_SFLOAT;
  }
  host_format_16_16.unsigned_signed_compatible =
      (host_format_16_16.format_unsigned.format == VK_FORMAT_R16G16_UNORM &&
       host_format_16_16.format_signed.format == VK_FORMAT_R16G16_SNORM) ||
      (host_format_16_16.format_unsigned.format == VK_FORMAT_R16G16_SFLOAT &&
       host_format_16_16.format_signed.format == VK_FORMAT_R16G16_SFLOAT);
  HostFormatPair& host_format_16_16_16_16 =
      host_formats_[uint32_t(xenos::TextureFormat::k_16_16_16_16)];
  assert_true(host_format_16_16_16_16.format_unsigned.format == VK_FORMAT_R16G16B16A16_UNORM);
  if (norm16_substitute(r16g16b16a16_unorm_properties, "RGBA16_UNORM", "RGBA16_SFLOAT")) {
    host_format_16_16_16_16.format_unsigned.load_shader = kLoadShaderIndexRGBA16UNormToFloat;
    host_format_16_16_16_16.format_unsigned.format = VK_FORMAT_R16G16B16A16_SFLOAT;
  }
  assert_true(host_format_16_16_16_16.format_signed.format == VK_FORMAT_R16G16B16A16_SNORM);
  if (norm16_substitute(r16g16b16a16_snorm_properties, "RGBA16_SNORM", "RGBA16_SFLOAT")) {
    host_format_16_16_16_16.format_signed.load_shader = kLoadShaderIndexRGBA16SNormToFloat;
    host_format_16_16_16_16.format_signed.format = VK_FORMAT_R16G16B16A16_SFLOAT;
  }
  host_format_16_16_16_16.unsigned_signed_compatible =
      (host_format_16_16_16_16.format_unsigned.format == VK_FORMAT_R16G16B16A16_UNORM &&
       host_format_16_16_16_16.format_signed.format == VK_FORMAT_R16G16B16A16_SNORM) ||
      (host_format_16_16_16_16.format_unsigned.format == VK_FORMAT_R16G16B16A16_SFLOAT &&
       host_format_16_16_16_16.format_signed.format == VK_FORMAT_R16G16B16A16_SFLOAT);

  // Normalize format information structures.
  for (size_t i = 0; i < rex::countof(host_formats_); ++i) {
    HostFormatPair& host_format = host_formats_[i];
    // load_shader_index is left uninitialized for the tail (non-existent
    // formats), kLoadShaderIndexUnknown may be non-zero, and format support may
    // be disabled by setting the format to VK_FORMAT_UNDEFINED.
    if (host_format.format_unsigned.format == VK_FORMAT_UNDEFINED) {
      host_format.format_unsigned.load_shader = kLoadShaderIndexUnknown;
    }
    assert_false(host_format.format_unsigned.load_shader == kLoadShaderIndexUnknown &&
                 host_format.format_unsigned.format != VK_FORMAT_UNDEFINED);
    if (host_format.format_unsigned.load_shader == kLoadShaderIndexUnknown) {
      host_format.format_unsigned.format = VK_FORMAT_UNDEFINED;
      // Surely known it's unsupported with these two conditions.
      host_format.format_unsigned.linear_filterable = false;
    }
    if (host_format.format_signed.format == VK_FORMAT_UNDEFINED) {
      host_format.format_signed.load_shader = kLoadShaderIndexUnknown;
    }
    assert_false(host_format.format_signed.load_shader == kLoadShaderIndexUnknown &&
                 host_format.format_signed.format != VK_FORMAT_UNDEFINED);
    if (host_format.format_signed.load_shader == kLoadShaderIndexUnknown) {
      host_format.format_signed.format = VK_FORMAT_UNDEFINED;
      // Surely known it's unsupported with these two conditions.
      host_format.format_signed.linear_filterable = false;
    }

    // Check if the formats are supported and are linear-filterable.
    if (host_format.format_unsigned.format != VK_FORMAT_UNDEFINED) {
      ifn.vkGetPhysicalDeviceFormatProperties(physical_device, host_format.format_unsigned.format,
                                              &format_properties);
      if (format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) {
        host_format.format_unsigned.linear_filterable =
            (format_properties.optimalTilingFeatures &
             VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
      } else {
        host_format.format_unsigned.format = VK_FORMAT_UNDEFINED;
        host_format.format_unsigned.load_shader = kLoadShaderIndexUnknown;
        host_format.format_unsigned.linear_filterable = false;
      }
    }
    if (host_format.format_signed.format != VK_FORMAT_UNDEFINED) {
      ifn.vkGetPhysicalDeviceFormatProperties(physical_device, host_format.format_signed.format,
                                              &format_properties);
      if (format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) {
        host_format.format_signed.linear_filterable =
            (format_properties.optimalTilingFeatures &
             VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
      } else {
        host_format.format_signed.format = VK_FORMAT_UNDEFINED;
        host_format.format_signed.load_shader = kLoadShaderIndexUnknown;
        host_format.format_signed.linear_filterable = false;
      }
    }

    // Log which formats are not supported or supported via fallbacks.
    const HostFormatPair& best_host_format = kBestHostFormats[i];
    const char* guest_format_name = FormatInfo::Get(xenos::TextureFormat(i))->name;
    if (best_host_format.format_unsigned.format != VK_FORMAT_UNDEFINED) {
      assert_not_null(guest_format_name);
      if (host_format.format_unsigned.format != VK_FORMAT_UNDEFINED) {
        if (host_format.format_unsigned.format != best_host_format.format_unsigned.format) {
          REXGPU_INFO(
              "VulkanTextureCache: Format {} (unsigned) is supported via a "
              "fallback format (using the Vulkan format {} instead of the "
              "preferred {})",
              guest_format_name, uint32_t(host_format.format_unsigned.format),
              uint32_t(best_host_format.format_unsigned.format));
        }
      } else {
        REXGPU_INFO(
            "VulkanTextureCache: Format {} (unsigned) is not supported by the "
            "device (preferred Vulkan format is {})",
            guest_format_name, uint32_t(best_host_format.format_unsigned.format));
      }
    }
    if (best_host_format.format_signed.format != VK_FORMAT_UNDEFINED) {
      assert_not_null(guest_format_name);
      if (host_format.format_signed.format != VK_FORMAT_UNDEFINED) {
        if (host_format.format_signed.format != best_host_format.format_signed.format) {
          REXGPU_INFO(
              "VulkanTextureCache: Format {} (signed) is supported via a "
              "fallback format (using the Vulkan format {} instead of the "
              "preferred {})",
              guest_format_name, uint32_t(host_format.format_signed.format),
              uint32_t(best_host_format.format_signed.format));
        }
      } else {
        REXGPU_INFO(
            "VulkanTextureCache: Format {} (signed) is not supported by the "
            "device (preferred Vulkan format is {})",
            guest_format_name, uint32_t(best_host_format.format_signed.format));
      }
    }

    // Signednesses with different load shaders must have the data loaded
    // differently, therefore can't share the image even if the format is the
    // same. Also, if there's only one version, simplify the logic - there can't
    // be compatibility between two formats when one of them is undefined.
    if (host_format.format_unsigned.format != VK_FORMAT_UNDEFINED &&
        host_format.format_signed.format != VK_FORMAT_UNDEFINED) {
      if (host_format.format_unsigned.load_shader == host_format.format_signed.load_shader) {
        if (host_format.format_unsigned.format == host_format.format_signed.format) {
          // Same format after all the fallbacks - force compatibilty.
          host_format.unsigned_signed_compatible = true;
        }
      } else {
        host_format.unsigned_signed_compatible = false;
      }
      // Formats within the same compatibility class must have the same block
      // size, though the fallbacks are configured incorrectly if that's not the
      // case (since such formats just can't be in one compatibility class).
      assert_false(host_format.unsigned_signed_compatible &&
                   host_format.format_unsigned.block_compressed !=
                       host_format.format_signed.block_compressed);
      if (host_format.unsigned_signed_compatible &&
          host_format.format_unsigned.block_compressed !=
              host_format.format_signed.block_compressed) {
        host_format.unsigned_signed_compatible = false;
      }
    } else {
      host_format.unsigned_signed_compatible = false;
    }
  }

  // Load pipeline layout.

  VkDescriptorSetLayout load_descriptor_set_layouts[kLoadDescriptorSetCount] = {};
  VkDescriptorSetLayout load_descriptor_set_layout_storage_buffer =
      command_processor_.GetSingleTransientDescriptorLayout(
          VulkanCommandProcessor::SingleTransientDescriptorLayout ::kStorageBufferCompute);
  assert_true(load_descriptor_set_layout_storage_buffer != VK_NULL_HANDLE);
  load_descriptor_set_layouts[kLoadDescriptorSetIndexDestination] =
      load_descriptor_set_layout_storage_buffer;
  load_descriptor_set_layouts[kLoadDescriptorSetIndexSource] =
      load_descriptor_set_layout_storage_buffer;
  VkPushConstantRange load_pipeline_layout_push_constant_range;
  load_pipeline_layout_push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  load_pipeline_layout_push_constant_range.offset = 0;
  load_pipeline_layout_push_constant_range.size = sizeof(LoadConstants);
  VkPipelineLayoutCreateInfo load_pipeline_layout_create_info;
  load_pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  load_pipeline_layout_create_info.pNext = nullptr;
  load_pipeline_layout_create_info.flags = 0;
  load_pipeline_layout_create_info.setLayoutCount = kLoadDescriptorSetCount;
  load_pipeline_layout_create_info.pSetLayouts = load_descriptor_set_layouts;
  load_pipeline_layout_create_info.pushConstantRangeCount = 1;
  load_pipeline_layout_create_info.pPushConstantRanges = &load_pipeline_layout_push_constant_range;
  if (dfn.vkCreatePipelineLayout(device, &load_pipeline_layout_create_info, nullptr,
                                 &load_pipeline_layout_)) {
    REXGPU_ERROR("VulkanTexture: Failed to create the texture load pipeline layout");
    return false;
  }

  // Load pipelines, only the ones needed for the formats that will be used.

  bool load_shaders_needed[kLoadShaderCount] = {};
  auto mark_host_format_pair_load_shaders_needed = [&](const HostFormatPair& host_format_pair) {
    if (host_format_pair.format_unsigned.load_shader != kLoadShaderIndexUnknown) {
      load_shaders_needed[host_format_pair.format_unsigned.load_shader] = true;
    }
    if (host_format_pair.format_signed.load_shader != kLoadShaderIndexUnknown) {
      load_shaders_needed[host_format_pair.format_signed.load_shader] = true;
    }
  };
  for (size_t i = 0; i < rex::countof(host_formats_); ++i) {
    mark_host_format_pair_load_shaders_needed(host_formats_[i]);
  }
  mark_host_format_pair_load_shaders_needed(kHostFormatGBGRUnaligned);
  mark_host_format_pair_load_shaders_needed(kHostFormatBGRGUnaligned);
  mark_host_format_pair_load_shaders_needed(kHostFormatDXT1Unaligned);
  mark_host_format_pair_load_shaders_needed(kHostFormatDXT2_3Unaligned);
  mark_host_format_pair_load_shaders_needed(kHostFormatDXT4_5Unaligned);
  mark_host_format_pair_load_shaders_needed(kHostFormatDXNUnaligned);
  mark_host_format_pair_load_shaders_needed(kHostFormatDXT5AUnaligned);

  std::pair<const uint32_t*, size_t> load_shader_code[kLoadShaderCount] = {};
  load_shader_code[kLoadShaderIndex8bpb] =
      std::make_pair(shaders::texture_load_8bpb_cs, sizeof(shaders::texture_load_8bpb_cs));
  load_shader_code[kLoadShaderIndex16bpb] =
      std::make_pair(shaders::texture_load_16bpb_cs, sizeof(shaders::texture_load_16bpb_cs));
  load_shader_code[kLoadShaderIndex32bpb] =
      std::make_pair(shaders::texture_load_32bpb_cs, sizeof(shaders::texture_load_32bpb_cs));
  load_shader_code[kLoadShaderIndex64bpb] =
      std::make_pair(shaders::texture_load_64bpb_cs, sizeof(shaders::texture_load_64bpb_cs));
  load_shader_code[kLoadShaderIndex128bpb] =
      std::make_pair(shaders::texture_load_128bpb_cs, sizeof(shaders::texture_load_128bpb_cs));
  load_shader_code[kLoadShaderIndexR5G5B5A1ToB5G5R5A1] =
      std::make_pair(shaders::texture_load_r5g5b5a1_b5g5r5a1_cs,
                     sizeof(shaders::texture_load_r5g5b5a1_b5g5r5a1_cs));
  load_shader_code[kLoadShaderIndexR5G6B5ToB5G6R5] = std::make_pair(
      shaders::texture_load_r5g6b5_b5g6r5_cs, sizeof(shaders::texture_load_r5g6b5_b5g6r5_cs));
  load_shader_code[kLoadShaderIndexR5G6B5ToRGBA8] = std::make_pair(
      shaders::texture_load_r5g6b5_rgba8_cs, sizeof(shaders::texture_load_r5g6b5_rgba8_cs));
  load_shader_code[kLoadShaderIndexR5G5B6ToB5G6R5WithRBGASwizzle] =
      std::make_pair(shaders::texture_load_r5g5b6_b5g6r5_swizzle_rbga_cs,
                     sizeof(shaders::texture_load_r5g5b6_b5g6r5_swizzle_rbga_cs));
  load_shader_code[kLoadShaderIndexRGBA4ToARGB4] =
      std::make_pair(shaders::texture_load_r4g4b4a4_a4r4g4b4_cs,
                     sizeof(shaders::texture_load_r4g4b4a4_a4r4g4b4_cs));
  load_shader_code[kLoadShaderIndexRGBA4ToRGBA8] = std::make_pair(
      shaders::texture_load_r4g4b4a4_rgba8_cs, sizeof(shaders::texture_load_r4g4b4a4_rgba8_cs));
  load_shader_code[kLoadShaderIndexGBGR8ToRGB8] = std::make_pair(
      shaders::texture_load_gbgr8_rgb8_cs, sizeof(shaders::texture_load_gbgr8_rgb8_cs));
  load_shader_code[kLoadShaderIndexBGRG8ToRGB8] = std::make_pair(
      shaders::texture_load_bgrg8_rgb8_cs, sizeof(shaders::texture_load_bgrg8_rgb8_cs));
  load_shader_code[kLoadShaderIndexR10G11B11ToRGBA16] = std::make_pair(
      shaders::texture_load_r10g11b11_rgba16_cs, sizeof(shaders::texture_load_r10g11b11_rgba16_cs));
  load_shader_code[kLoadShaderIndexR10G11B11ToRGBA16SNorm] =
      std::make_pair(shaders::texture_load_r10g11b11_rgba16_snorm_cs,
                     sizeof(shaders::texture_load_r10g11b11_rgba16_snorm_cs));
  load_shader_code[kLoadShaderIndexR11G11B10ToRGBA16] = std::make_pair(
      shaders::texture_load_r11g11b10_rgba16_cs, sizeof(shaders::texture_load_r11g11b10_rgba16_cs));
  load_shader_code[kLoadShaderIndexR11G11B10ToRGBA16SNorm] =
      std::make_pair(shaders::texture_load_r11g11b10_rgba16_snorm_cs,
                     sizeof(shaders::texture_load_r11g11b10_rgba16_snorm_cs));
  load_shader_code[kLoadShaderIndexR16UNormToFloat] = std::make_pair(
      shaders::texture_load_r16_unorm_float_cs, sizeof(shaders::texture_load_r16_unorm_float_cs));
  load_shader_code[kLoadShaderIndexR16SNormToFloat] = std::make_pair(
      shaders::texture_load_r16_snorm_float_cs, sizeof(shaders::texture_load_r16_snorm_float_cs));
  load_shader_code[kLoadShaderIndexRG16UNormToFloat] = std::make_pair(
      shaders::texture_load_rg16_unorm_float_cs, sizeof(shaders::texture_load_rg16_unorm_float_cs));
  load_shader_code[kLoadShaderIndexRG16SNormToFloat] = std::make_pair(
      shaders::texture_load_rg16_snorm_float_cs, sizeof(shaders::texture_load_rg16_snorm_float_cs));
  load_shader_code[kLoadShaderIndexRGBA16UNormToFloat] =
      std::make_pair(shaders::texture_load_rgba16_unorm_float_cs,
                     sizeof(shaders::texture_load_rgba16_unorm_float_cs));
  load_shader_code[kLoadShaderIndexRGBA16SNormToFloat] =
      std::make_pair(shaders::texture_load_rgba16_snorm_float_cs,
                     sizeof(shaders::texture_load_rgba16_snorm_float_cs));
  load_shader_code[kLoadShaderIndexDXT1ToRGBA8] = std::make_pair(
      shaders::texture_load_dxt1_rgba8_cs, sizeof(shaders::texture_load_dxt1_rgba8_cs));
  load_shader_code[kLoadShaderIndexDXT3ToRGBA8] = std::make_pair(
      shaders::texture_load_dxt3_rgba8_cs, sizeof(shaders::texture_load_dxt3_rgba8_cs));
  load_shader_code[kLoadShaderIndexDXT5ToRGBA8] = std::make_pair(
      shaders::texture_load_dxt5_rgba8_cs, sizeof(shaders::texture_load_dxt5_rgba8_cs));
  load_shader_code[kLoadShaderIndexDXNToRG8] =
      std::make_pair(shaders::texture_load_dxn_rg8_cs, sizeof(shaders::texture_load_dxn_rg8_cs));
  load_shader_code[kLoadShaderIndexDXT3A] =
      std::make_pair(shaders::texture_load_dxt3a_cs, sizeof(shaders::texture_load_dxt3a_cs));
  load_shader_code[kLoadShaderIndexDXT3AAs1111ToARGB4] =
      std::make_pair(shaders::texture_load_dxt3aas1111_argb4_cs,
                     sizeof(shaders::texture_load_dxt3aas1111_argb4_cs));
  load_shader_code[kLoadShaderIndexDXT5AToR8] =
      std::make_pair(shaders::texture_load_dxt5a_r8_cs, sizeof(shaders::texture_load_dxt5a_r8_cs));
  load_shader_code[kLoadShaderIndexCTX1] =
      std::make_pair(shaders::texture_load_ctx1_cs, sizeof(shaders::texture_load_ctx1_cs));
  // wwe2k14 patch (Session 93): the transcoding loads (only created when a host format uses them).
  load_shader_code[kLoadShaderIndexDXT1ToASTC4x4] =
      REXCVAR_GET(vulkan_texture_transcode_bc1_alpha_fix)
          ? std::make_pair(shaders::texture_load_dxt1_astc4x4_alphafix_cs,
                           sizeof(shaders::texture_load_dxt1_astc4x4_alphafix_cs))
          : std::make_pair(shaders::texture_load_dxt1_astc4x4_cs, sizeof(shaders::texture_load_dxt1_astc4x4_cs));
  if (const int32_t emu = g_emulate_bc1_active ? REXCVAR_GET(vulkan_emulate_bc1_transcode) : 0) {  // (Session 95, test only; same condition as the format switch)
    load_shader_code[kLoadShaderIndexDXT1ToASTC4x4] =
        emu == 2 ? std::make_pair(shaders::texture_load_dxt1_emu_bc3_alphafix_cs, sizeof(shaders::texture_load_dxt1_emu_bc3_alphafix_cs))
                 : std::make_pair(shaders::texture_load_dxt1_emu_bc3_cs, sizeof(shaders::texture_load_dxt1_emu_bc3_cs));
  }
  load_shader_code[kLoadShaderIndexDXT5AToEACR11] = std::make_pair(
      shaders::texture_load_dxt5a_eac_r11_cs, sizeof(shaders::texture_load_dxt5a_eac_r11_cs));
  load_shader_code[kLoadShaderIndexDXNToEACRG11] = std::make_pair(
      shaders::texture_load_dxn_eac_rg11_cs, sizeof(shaders::texture_load_dxn_eac_rg11_cs));
  load_shader_code[kLoadShaderIndexDepthUnorm] = std::make_pair(
      shaders::texture_load_depth_unorm_cs, sizeof(shaders::texture_load_depth_unorm_cs));
  load_shader_code[kLoadShaderIndexDepthFloat] = std::make_pair(
      shaders::texture_load_depth_float_cs, sizeof(shaders::texture_load_depth_float_cs));
  std::pair<const uint32_t*, size_t> load_shader_code_scaled[kLoadShaderCount] = {};
  if (IsDrawResolutionScaled()) {
    load_shader_code_scaled[kLoadShaderIndex8bpb] = std::make_pair(
        shaders::texture_load_8bpb_scaled_cs, sizeof(shaders::texture_load_8bpb_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndex16bpb] = std::make_pair(
        shaders::texture_load_16bpb_scaled_cs, sizeof(shaders::texture_load_16bpb_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndex32bpb] = std::make_pair(
        shaders::texture_load_32bpb_scaled_cs, sizeof(shaders::texture_load_32bpb_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndex64bpb] = std::make_pair(
        shaders::texture_load_64bpb_scaled_cs, sizeof(shaders::texture_load_64bpb_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndex128bpb] = std::make_pair(
        shaders::texture_load_128bpb_scaled_cs, sizeof(shaders::texture_load_128bpb_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR5G5B5A1ToB5G5R5A1] =
        std::make_pair(shaders::texture_load_r5g5b5a1_b5g5r5a1_scaled_cs,
                       sizeof(shaders::texture_load_r5g5b5a1_b5g5r5a1_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR5G6B5ToB5G6R5] =
        std::make_pair(shaders::texture_load_r5g6b5_b5g6r5_scaled_cs,
                       sizeof(shaders::texture_load_r5g6b5_b5g6r5_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR5G6B5ToRGBA8] =
        std::make_pair(shaders::texture_load_r5g6b5_rgba8_scaled_cs,
                       sizeof(shaders::texture_load_r5g6b5_rgba8_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR5G5B6ToB5G6R5WithRBGASwizzle] =
        std::make_pair(shaders::texture_load_r5g5b6_b5g6r5_swizzle_rbga_scaled_cs,
                       sizeof(shaders::texture_load_r5g5b6_b5g6r5_swizzle_rbga_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexRGBA4ToARGB4] =
        std::make_pair(shaders::texture_load_r4g4b4a4_a4r4g4b4_scaled_cs,
                       sizeof(shaders::texture_load_r4g4b4a4_a4r4g4b4_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexRGBA4ToRGBA8] =
        std::make_pair(shaders::texture_load_r4g4b4a4_rgba8_scaled_cs,
                       sizeof(shaders::texture_load_r4g4b4a4_rgba8_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR10G11B11ToRGBA16] =
        std::make_pair(shaders::texture_load_r10g11b11_rgba16_scaled_cs,
                       sizeof(shaders::texture_load_r10g11b11_rgba16_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR10G11B11ToRGBA16SNorm] =
        std::make_pair(shaders::texture_load_r10g11b11_rgba16_snorm_scaled_cs,
                       sizeof(shaders::texture_load_r10g11b11_rgba16_snorm_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR11G11B10ToRGBA16] =
        std::make_pair(shaders::texture_load_r11g11b10_rgba16_scaled_cs,
                       sizeof(shaders::texture_load_r11g11b10_rgba16_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR11G11B10ToRGBA16SNorm] =
        std::make_pair(shaders::texture_load_r11g11b10_rgba16_snorm_scaled_cs,
                       sizeof(shaders::texture_load_r11g11b10_rgba16_snorm_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR16UNormToFloat] =
        std::make_pair(shaders::texture_load_r16_unorm_float_scaled_cs,
                       sizeof(shaders::texture_load_r16_unorm_float_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR16SNormToFloat] =
        std::make_pair(shaders::texture_load_r16_snorm_float_scaled_cs,
                       sizeof(shaders::texture_load_r16_snorm_float_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexRG16UNormToFloat] =
        std::make_pair(shaders::texture_load_rg16_unorm_float_scaled_cs,
                       sizeof(shaders::texture_load_rg16_unorm_float_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexRG16SNormToFloat] =
        std::make_pair(shaders::texture_load_rg16_snorm_float_scaled_cs,
                       sizeof(shaders::texture_load_rg16_snorm_float_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexRGBA16UNormToFloat] =
        std::make_pair(shaders::texture_load_rgba16_unorm_float_scaled_cs,
                       sizeof(shaders::texture_load_rgba16_unorm_float_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexRGBA16SNormToFloat] =
        std::make_pair(shaders::texture_load_rgba16_snorm_float_scaled_cs,
                       sizeof(shaders::texture_load_rgba16_snorm_float_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexDepthUnorm] =
        std::make_pair(shaders::texture_load_depth_unorm_scaled_cs,
                       sizeof(shaders::texture_load_depth_unorm_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexDepthFloat] =
        std::make_pair(shaders::texture_load_depth_float_scaled_cs,
                       sizeof(shaders::texture_load_depth_float_scaled_cs));
  }

  for (size_t i = 0; i < kLoadShaderCount; ++i) {
    if (!load_shaders_needed[i]) {
      continue;
    }
    const std::pair<const uint32_t*, size_t>& current_load_shader_code = load_shader_code[i];
    assert_not_null(current_load_shader_code.first);
    load_pipelines_[i] = ui::vulkan::util::CreateComputePipeline(
        vulkan_device, load_pipeline_layout_, current_load_shader_code.first,
        current_load_shader_code.second);
    if (load_pipelines_[i] == VK_NULL_HANDLE) {
      REXGPU_ERROR(
          "VulkanTextureCache: Failed to create the texture loading pipeline "
          "for shader {}",
          i);
      return false;
    }
    if (IsDrawResolutionScaled()) {
      const std::pair<const uint32_t*, size_t>& current_load_shader_code_scaled =
          load_shader_code_scaled[i];
      if (current_load_shader_code_scaled.first) {
        load_pipelines_scaled_[i] = ui::vulkan::util::CreateComputePipeline(
            vulkan_device, load_pipeline_layout_, current_load_shader_code_scaled.first,
            current_load_shader_code_scaled.second);
        if (load_pipelines_scaled_[i] == VK_NULL_HANDLE) {
          REXGPU_ERROR(
              "VulkanTextureCache: Failed to create the resolution-scaled "
              "texture loading pipeline for shader {}",
              i);
          return false;
        }
      }
    }
  }

  // Use true null descriptors when supported; otherwise, keep compatibility
  // fallback null images.
  if (device_properties.nullDescriptor) {
    null_images_cleared_ = true;
  } else {
    // Null images as a replacement for unneeded bindings and for bindings for
    // which the real image hasn't been created.
    VkImageCreateInfo null_image_create_info;
    null_image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    null_image_create_info.pNext = nullptr;
    null_image_create_info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    null_image_create_info.imageType = VK_IMAGE_TYPE_2D;
    // Match D3D12 null SRV behavior: invalid fetch constants read (0, 0, 0, 0).
    null_image_create_info.format = kInvalidTextureFetchFallbackFormat;
    null_image_create_info.extent.width = 1;
    null_image_create_info.extent.height = 1;
    null_image_create_info.extent.depth = 1;
    null_image_create_info.mipLevels = 1;
    null_image_create_info.arrayLayers = 6;
    null_image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    null_image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    null_image_create_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    null_image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    null_image_create_info.queueFamilyIndexCount = 0;
    null_image_create_info.pQueueFamilyIndices = nullptr;
    null_image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (dfn.vkCreateImage(device, &null_image_create_info, nullptr, &null_image_2d_array_cube_) !=
        VK_SUCCESS) {
      REXGPU_ERROR(
          "VulkanTextureCache: Failed to create the null 2D array and cube "
          "image");
      return false;
    }

    null_image_create_info.flags &= ~VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    null_image_create_info.imageType = VK_IMAGE_TYPE_3D;
    null_image_create_info.arrayLayers = 1;
    if (dfn.vkCreateImage(device, &null_image_create_info, nullptr, &null_image_3d_) !=
        VK_SUCCESS) {
      REXGPU_ERROR("VulkanTextureCache: Failed to create the null 3D image");
      return false;
    }

    VkMemoryRequirements null_image_memory_requirements_2d_array_cube_;
    dfn.vkGetImageMemoryRequirements(device, null_image_2d_array_cube_,
                                     &null_image_memory_requirements_2d_array_cube_);
    VkMemoryRequirements null_image_memory_requirements_3d_;
    dfn.vkGetImageMemoryRequirements(device, null_image_3d_, &null_image_memory_requirements_3d_);
    uint32_t null_image_memory_type_common = ui::vulkan::util::ChooseMemoryType(
        vulkan_device->memory_types(),
        null_image_memory_requirements_2d_array_cube_.memoryTypeBits &
            null_image_memory_requirements_3d_.memoryTypeBits,
        ui::vulkan::util::MemoryPurpose::kDeviceLocal);
    if (null_image_memory_type_common != UINT32_MAX) {
      // Place both null images in one memory allocation because maximum total
      // memory allocation count is limited.
      VkDeviceSize null_image_memory_offset_3d_ =
          rex::align(null_image_memory_requirements_2d_array_cube_.size,
                     std::max(null_image_memory_requirements_3d_.alignment, VkDeviceSize(1)));
      VkMemoryAllocateInfo null_image_memory_allocate_info;
      null_image_memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      null_image_memory_allocate_info.pNext = nullptr;
      null_image_memory_allocate_info.allocationSize =
          null_image_memory_offset_3d_ + null_image_memory_requirements_3d_.size;
      null_image_memory_allocate_info.memoryTypeIndex = null_image_memory_type_common;
      if (dfn.vkAllocateMemory(device, &null_image_memory_allocate_info, nullptr,
                               &null_images_memory_[0]) != VK_SUCCESS) {
        REXGPU_ERROR(
            "VulkanTextureCache: Failed to allocate the memory for the null "
            "images");
        return false;
      }
      if (dfn.vkBindImageMemory(device, null_image_2d_array_cube_, null_images_memory_[0], 0) !=
          VK_SUCCESS) {
        REXGPU_ERROR(
            "VulkanTextureCache: Failed to bind the memory to the null 2D array "
            "and cube image");
        return false;
      }
      if (dfn.vkBindImageMemory(device, null_image_3d_, null_images_memory_[0],
                                null_image_memory_offset_3d_) != VK_SUCCESS) {
        REXGPU_ERROR("VulkanTextureCache: Failed to bind the memory to the null 3D image");
        return false;
      }
    } else {
      // Place each null image in separate allocations.
      const uint32_t null_image_memory_type_2d_array_cube = ui::vulkan::util::ChooseMemoryType(
          vulkan_device->memory_types(),
          null_image_memory_requirements_2d_array_cube_.memoryTypeBits,
          ui::vulkan::util::MemoryPurpose::kDeviceLocal);
      const uint32_t null_image_memory_type_3d = ui::vulkan::util::ChooseMemoryType(
          vulkan_device->memory_types(), null_image_memory_requirements_3d_.memoryTypeBits,
          ui::vulkan::util::MemoryPurpose::kDeviceLocal);
      if (null_image_memory_type_2d_array_cube == UINT32_MAX ||
          null_image_memory_type_3d == UINT32_MAX) {
        REXGPU_ERROR(
            "VulkanTextureCache: Failed to get the memory types for the null "
            "images");
        return false;
      }

      VkMemoryAllocateInfo null_image_memory_allocate_info;
      VkMemoryAllocateInfo* null_image_memory_allocate_info_last = &null_image_memory_allocate_info;
      null_image_memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      null_image_memory_allocate_info.pNext = nullptr;
      null_image_memory_allocate_info.allocationSize =
          null_image_memory_requirements_2d_array_cube_.size;
      null_image_memory_allocate_info.memoryTypeIndex = null_image_memory_type_2d_array_cube;
      VkMemoryDedicatedAllocateInfo null_image_memory_dedicated_allocate_info;
      if (vulkan_device->extensions().ext_1_1_KHR_dedicated_allocation) {
        null_image_memory_allocate_info_last->pNext = &null_image_memory_dedicated_allocate_info;
        null_image_memory_allocate_info_last =
            reinterpret_cast<VkMemoryAllocateInfo*>(&null_image_memory_dedicated_allocate_info);
        null_image_memory_dedicated_allocate_info.sType =
            VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        null_image_memory_dedicated_allocate_info.pNext = nullptr;
        null_image_memory_dedicated_allocate_info.image = null_image_2d_array_cube_;
        null_image_memory_dedicated_allocate_info.buffer = VK_NULL_HANDLE;
      }
      if (dfn.vkAllocateMemory(device, &null_image_memory_allocate_info, nullptr,
                               &null_images_memory_[0]) != VK_SUCCESS) {
        REXGPU_ERROR(
            "VulkanTextureCache: Failed to allocate the memory for the null 2D "
            "array and cube image");
        return false;
      }
      if (dfn.vkBindImageMemory(device, null_image_2d_array_cube_, null_images_memory_[0], 0) !=
          VK_SUCCESS) {
        REXGPU_ERROR(
            "VulkanTextureCache: Failed to bind the memory to the null 2D array "
            "and cube image");
        return false;
      }

      null_image_memory_allocate_info.allocationSize = null_image_memory_requirements_3d_.size;
      null_image_memory_allocate_info.memoryTypeIndex = null_image_memory_type_3d;
      null_image_memory_dedicated_allocate_info.image = null_image_3d_;
      if (dfn.vkAllocateMemory(device, &null_image_memory_allocate_info, nullptr,
                               &null_images_memory_[1]) != VK_SUCCESS) {
        REXGPU_ERROR(
            "VulkanTextureCache: Failed to allocate the memory for the null 3D "
            "image");
        return false;
      }
      if (dfn.vkBindImageMemory(device, null_image_3d_, null_images_memory_[1], 0) != VK_SUCCESS) {
        REXGPU_ERROR("VulkanTextureCache: Failed to bind the memory to the null 3D image");
        return false;
      }
    }

    VkImageViewCreateInfo null_image_view_create_info;
    null_image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    null_image_view_create_info.pNext = nullptr;
    null_image_view_create_info.flags = 0;
    null_image_view_create_info.image = null_image_2d_array_cube_;
    null_image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    null_image_view_create_info.format = null_image_create_info.format;
    // Micro-optimization if this has any effect on the host GPU at all, use only
    // constant components instead of the real texels.
    VkComponentSwizzle null_image_view_swizzle = device_properties.imageViewFormatSwizzle
                                                     ? VK_COMPONENT_SWIZZLE_ZERO
                                                     : VK_COMPONENT_SWIZZLE_IDENTITY;
    null_image_view_create_info.components.r = null_image_view_swizzle;
    null_image_view_create_info.components.g = null_image_view_swizzle;
    null_image_view_create_info.components.b = null_image_view_swizzle;
    null_image_view_create_info.components.a = null_image_view_swizzle;
    null_image_view_create_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(
        VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, 1);
    if (dfn.vkCreateImageView(device, &null_image_view_create_info, nullptr,
                              &null_image_view_2d_array_) != VK_SUCCESS) {
      REXGPU_ERROR("VulkanTextureCache: Failed to create the null 2D array image view");
      return false;
    }
    null_image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    null_image_view_create_info.subresourceRange.layerCount = 6;
    if (dfn.vkCreateImageView(device, &null_image_view_create_info, nullptr,
                              &null_image_view_cube_) != VK_SUCCESS) {
      REXGPU_ERROR("VulkanTextureCache: Failed to create the null cube image view");
      return false;
    }
    null_image_view_create_info.image = null_image_3d_;
    null_image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_3D;
    null_image_view_create_info.subresourceRange.layerCount = 1;
    if (dfn.vkCreateImageView(device, &null_image_view_create_info, nullptr,
                              &null_image_view_3d_) != VK_SUCCESS) {
      REXGPU_ERROR("VulkanTextureCache: Failed to create the null 3D image view");
      return false;
    }

    null_images_cleared_ = false;
  }

  // Samplers.

  // Some MoltenVK devices have a maximum of 2048, 1024, or even 96 samplers,
  // below Vulkan's minimum requirement of 4000.
  // Assuming that the current VulkanTextureCache is the only one on this
  // VkDevice (true in a regular emulation scenario), so taking over all the
  // allocation slots exclusively.
  // Also leaving a few slots for use by things like overlay applications.
  sampler_max_count_ =
      device_properties.maxSamplerAllocationCount - ui::vulkan::UISamplers::kSamplerCount - 16;

  if (device_properties.samplerAnisotropy) {
    max_anisotropy_ = xenos::AnisoFilter(
        uint32_t(xenos::AnisoFilter::kMax_1_1) +
        (31 - rex::lzcnt(uint32_t(
                  std::min(16.0f, std::max(1.0f, device_properties.maxSamplerAnisotropy))))));
  } else {
    max_anisotropy_ = xenos::AnisoFilter::kDisabled;
  }

  // wwe2k14 patch (Session 7c, TEXDIRECT): direct loads are initialized on first use (EnsureDirectLoadInitialized,
  // review fix), so with vulkan_texture_load_direct off nothing of them is created at startup.

  return true;
}

const VulkanTextureCache::HostFormatPair& VulkanTextureCache::GetHostFormatPair(
    TextureKey key) const {
  const HostFormatPair& host_format_pair = host_formats_[uint32_t(key.format)];
  if (!host_format_pair.format_unsigned.block_compressed &&
      !host_format_pair.format_signed.block_compressed) {
    return host_format_pair;
  }

  // Keep unaligned decompression selection aligned with D3D12 fallback behavior:
  // only select unaligned fallback mappings when block alignment is actually
  // violated for the current guest format.
  const FormatInfo* format_info = FormatInfo::Get(key.format);
  if (!(key.GetWidth() & (format_info->block_width - 1)) &&
      !(key.GetHeight() & (format_info->block_height - 1))) {
    return host_format_pair;
  }

  switch (key.format) {
    case xenos::TextureFormat::k_Cr_Y1_Cb_Y0_REP:
      return kHostFormatGBGRUnaligned;
    case xenos::TextureFormat::k_Y1_Cr_Y0_Cb_REP:
      return kHostFormatBGRGUnaligned;
    case xenos::TextureFormat::k_DXT1:
    case xenos::TextureFormat::k_DXT1_AS_16_16_16_16:
      return kHostFormatDXT1Unaligned;
    case xenos::TextureFormat::k_DXT2_3:
    case xenos::TextureFormat::k_DXT2_3_AS_16_16_16_16:
      return kHostFormatDXT2_3Unaligned;
    case xenos::TextureFormat::k_DXT4_5:
    case xenos::TextureFormat::k_DXT4_5_AS_16_16_16_16:
      return kHostFormatDXT4_5Unaligned;
    case xenos::TextureFormat::k_DXN:
      return kHostFormatDXNUnaligned;
    case xenos::TextureFormat::k_DXT5A:
      return kHostFormatDXT5AUnaligned;
    default:
      return host_format_pair;
  }
}

void VulkanTextureCache::GetTextureUsageMasks(VulkanTexture::Usage usage,
                                              VkPipelineStageFlags& stage_mask,
                                              VkAccessFlags& access_mask, VkImageLayout& layout) {
  stage_mask = 0;
  access_mask = 0;
  layout = VK_IMAGE_LAYOUT_UNDEFINED;
  switch (usage) {
    case VulkanTexture::Usage::kUndefined:
      break;
    case VulkanTexture::Usage::kTransferDestination:
      stage_mask = VK_PIPELINE_STAGE_TRANSFER_BIT;
      access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
      layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      break;
    case VulkanTexture::Usage::kGuestShaderSampled:
      stage_mask = guest_shader_pipeline_stages_;
      access_mask = VK_ACCESS_SHADER_READ_BIT;
      layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      break;
    case VulkanTexture::Usage::kSwapSampled:
      // The swap texture is likely to be used only for the presentation
      // post-processing passes, and not during emulation where it'd be used in
      // other stages.
      stage_mask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      access_mask = VK_ACCESS_SHADER_READ_BIT;
      layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      break;
    case VulkanTexture::Usage::kComputeStorageWrite:
      // wwe2k14 patch (Session 7c, TEXDIRECT): a direct load dispatch. The verify readback copy after it is
      // chained into the compute stage (RecordDirectLoadVerify), so this also covers it.
      stage_mask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      access_mask = VK_ACCESS_SHADER_WRITE_BIT;
      layout = VK_IMAGE_LAYOUT_GENERAL;
      break;
  }
}

xenos::ClampMode VulkanTextureCache::NormalizeClampMode(xenos::ClampMode clamp_mode) const {
  if (clamp_mode == xenos::ClampMode::kClampToHalfway) {
    // No GL_CLAMP (clamp to half edge, half border) equivalent in Vulkan, but
    // there's no Direct3D 9 equivalent anyway, and too weird to be suitable for
    // intentional real usage.
    return xenos::ClampMode::kClampToEdge;
  }
  if (clamp_mode == xenos::ClampMode::kMirrorClampToEdge ||
      clamp_mode == xenos::ClampMode::kMirrorClampToHalfway ||
      clamp_mode == xenos::ClampMode::kMirrorClampToBorder) {
    // No equivalents for anything other than kMirrorClampToEdge in Vulkan.
    return command_processor_.GetVulkanDevice()->properties().samplerMirrorClampToEdge
               ? xenos::ClampMode::kMirrorClampToEdge
               : xenos::ClampMode::kMirroredRepeat;
  }
  return clamp_mode;
}

// wwe2k14 patch (Session 7c, TEXDIRECT): direct-to-image scaled texture loads (vulkan_texture_load_direct) and
// their verification (vulkan_texture_load_direct_verify). See the comment above the cvars and
// shaders/texture_load_scaled_direct.cs.glsl.

VkFormat VulkanTextureCache::GetDirectLoadViewFormat(LoadShaderIndex load_shader) {
  switch (load_shader) {
    case kLoadShaderIndex32bpb:
      // One 32-bit word per texel.
      return VK_FORMAT_R32_UINT;
    case kLoadShaderIndex64bpb:
      // The two 32-bit words of a texel as four 16-bit halves, low half first. Not R32G32_UINT: its storage
      // image format needs the StorageImageExtendedFormats capability, R16G16B16A16_UINT's doesn't, and storage
      // image support for it is mandatory.
      return VK_FORMAT_R16G16B16A16_UINT;
    case kLoadShaderIndexDepthFloat:
      // wwe2k14 patch (Session 22, TEXDEPTH): the converted 32-bit float's bits (the word the buffer variant
      // stores and the copy writes into the R32_SFLOAT image), one per texel. Off: the buffer + copy path as before
      // (and new images without the storage usage).
      return REXCVAR_GET(vulkan_texture_load_direct_depth) ? VK_FORMAT_R32_UINT : VK_FORMAT_UNDEFINED;
    default:
      return VK_FORMAT_UNDEFINED;
  }
}

bool VulkanTextureCache::IsDirectLoadHostFormatSupported(VkFormat format,
                                                         LoadShaderIndex load_shader) const {
  // Uncompressed color formats with the texel size of the load shader's host block, in the size compatibility
  // class of the storage view format. vkCmdCopyBufferToImage copies texels as raw bytes, and a view of a
  // compatible format reinterprets the same bytes, so the bits stored through the unsigned integer view are the
  // bits the copy would have written. Not covered: block-compressed and 4:2:2 formats (different texel blocks)
  // and depth formats (other load shaders anyway).
  switch (load_shader) {
    case kLoadShaderIndex32bpb:
      switch (format) {
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SNORM:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        case VK_FORMAT_R16G16_UNORM:
        case VK_FORMAT_R16G16_SNORM:
        case VK_FORMAT_R16G16_SFLOAT:
        case VK_FORMAT_R32_SFLOAT:
          break;
        default:
          return false;
      }
      break;
    case kLoadShaderIndex64bpb:
      switch (format) {
        case VK_FORMAT_R16G16B16A16_UNORM:
        case VK_FORMAT_R16G16B16A16_SNORM:
        case VK_FORMAT_R16G16B16A16_SFLOAT:
        case VK_FORMAT_R32G32_SFLOAT:
          break;
        default:
          return false;
      }
      break;
    case kLoadShaderIndexDepthFloat:
      // wwe2k14 patch (Session 22, TEXDEPTH): k_24_8_FLOAT's host format (both signednesses).
      if (format != VK_FORMAT_R32_SFLOAT) {
        return false;
      }
      break;
    default:
      return false;
  }
  // The image gets the storage usage, which its views inherit, so each of its view formats must support storage
  // images.
  return std::find(direct_load_storage_formats_.cbegin(), direct_load_storage_formats_.cend(), format) !=
         direct_load_storage_formats_.cend();
}

VkFormat VulkanTextureCache::GetDirectLoadFormatForNewTexture(TextureKey key, const VkFormat* formats,
                                                              uint32_t format_count) const {
  // Session 43 (S0b-lite): 1x textures too (the caller decides which).
  if (key.mip_max_level != 0 ||
      (key.dimension != xenos::DataDimension::k1D &&
       key.dimension != xenos::DataDimension::k2DOrStacked) ||
      key.GetDepthOrArraySize() != 1 || !direct_load_descriptor_pool_) {
    return VK_FORMAT_UNDEFINED;
  }
  // Review fix: a base level in the packed mip tail is never loaded directly (the per-load check in
  // LoadTextureDataFromResidentMemoryImpl, on the same layout - Texture::guest_layout() is key.GetGuestLayout()),
  // so such a texture doesn't get the storage usage (which may cost sampling performance through lost compression).
  if (key.GetGuestLayout().packed_level == 0) {
    return VK_FORMAT_UNDEFINED;
  }
  // The load shader and the host format as chosen in LoadTextureDataFromResidentMemoryImpl.
  const HostFormatPair& host_format_pair = GetHostFormatPair(key);
  bool host_format_is_signed;
  if (IsSignedVersionSeparateForFormat(key)) {
    host_format_is_signed = bool(key.signed_separate);
  } else {
    host_format_is_signed = host_format_pair.format_unsigned.load_shader == kLoadShaderIndexUnknown;
  }
  const HostFormat& host_format =
      host_format_is_signed ? host_format_pair.format_signed : host_format_pair.format_unsigned;
  // One texel per guest block: the image extent is then the load's size in blocks.
  const FormatInfo* guest_format_info = FormatInfo::Get(key.format);
  if (host_format.block_compressed || guest_format_info->block_width != 1 ||
      guest_format_info->block_height != 1) {
    return VK_FORMAT_UNDEFINED;
  }
  VkFormat view_format = GetDirectLoadViewFormat(host_format.load_shader);
  if (view_format == VK_FORMAT_UNDEFINED ||
      direct_load_pipelines_[GetDirectLoadShader(host_format.load_shader)] == VK_NULL_HANDLE ||
      std::find(direct_load_storage_formats_.cbegin(), direct_load_storage_formats_.cend(),
                view_format) == direct_load_storage_formats_.cend()) {
    return VK_FORMAT_UNDEFINED;
  }
  for (uint32_t i = 0; i < format_count; ++i) {
    if (formats[i] != VK_FORMAT_UNDEFINED &&
        !IsDirectLoadHostFormatSupported(formats[i], host_format.load_shader)) {
      return VK_FORMAT_UNDEFINED;
    }
  }
  return view_format;
}

bool VulkanTextureCache::EnsureDirectLoadInitialized() {
  if (!direct_load_initialize_attempted_) {
    // One attempt (on failure, InitializeDirectLoad has logged why and released what it created). Only reached
    // for resolution-scaled textures, so with resolution scaling.
    direct_load_initialize_attempted_ = true;
    InitializeDirectLoad();
  }
  return direct_load_descriptor_pool_ != nullptr;
}

bool VulkanTextureCache::InitializeDirectLoad() {
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanInstance* const vulkan_instance = vulkan_device->vulkan_instance();
  const ui::vulkan::VulkanInstance::Functions& ifn = vulkan_instance->functions();
  const VkPhysicalDevice physical_device = vulkan_device->physical_device();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  direct_load_get_image_format_properties_ =
      reinterpret_cast<PFN_vkGetPhysicalDeviceImageFormatProperties>(ifn.vkGetInstanceProcAddr(
          vulkan_instance->instance(), "vkGetPhysicalDeviceImageFormatProperties"));
  if (!direct_load_get_image_format_properties_) {
    REXGPU_WARN("[s7][TEXDIRECT] vkGetPhysicalDeviceImageFormatProperties unavailable - direct texture loads off");
    return false;
  }

  static constexpr VkFormat kFormatsConsidered[] = {
      VK_FORMAT_R32_UINT,
      VK_FORMAT_R8G8B8A8_UNORM,
      VK_FORMAT_R8G8B8A8_SNORM,
      VK_FORMAT_A2B10G10R10_UNORM_PACK32,
      VK_FORMAT_R16G16_UNORM,
      VK_FORMAT_R16G16_SNORM,
      VK_FORMAT_R16G16_SFLOAT,
      VK_FORMAT_R32_SFLOAT,
      VK_FORMAT_R16G16B16A16_UINT,
      VK_FORMAT_R16G16B16A16_UNORM,
      VK_FORMAT_R16G16B16A16_SNORM,
      VK_FORMAT_R16G16B16A16_SFLOAT,
      VK_FORMAT_R32G32_SFLOAT,
  };
  direct_load_storage_formats_.clear();
  std::string formats_without_storage;
  for (VkFormat format : kFormatsConsidered) {
    VkFormatProperties format_properties;
    ifn.vkGetPhysicalDeviceFormatProperties(physical_device, format, &format_properties);
    if (format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) {
      direct_load_storage_formats_.push_back(format);
    } else {
      formats_without_storage += ' ';
      formats_without_storage += std::to_string(uint32_t(format));
    }
  }

  VkDescriptorSetLayoutBinding descriptor_set_layout_binding;
  descriptor_set_layout_binding.binding = 0;
  descriptor_set_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  descriptor_set_layout_binding.descriptorCount = 1;
  descriptor_set_layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  descriptor_set_layout_binding.pImmutableSamplers = nullptr;
  VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info;
  descriptor_set_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptor_set_layout_create_info.pNext = nullptr;
  descriptor_set_layout_create_info.flags = 0;
  descriptor_set_layout_create_info.bindingCount = 1;
  descriptor_set_layout_create_info.pBindings = &descriptor_set_layout_binding;
  if (dfn.vkCreateDescriptorSetLayout(device, &descriptor_set_layout_create_info, nullptr,
                                      &direct_load_descriptor_set_layout_) != VK_SUCCESS) {
    direct_load_descriptor_set_layout_ = VK_NULL_HANDLE;
    REXGPU_ERROR("[s7][TEXDIRECT] failed to create the storage image descriptor set layout - off");
    ShutdownDirectLoad();
    return false;
  }

  // Same set indices and push constants as load_pipeline_layout_, with a storage image as the destination.
  VkDescriptorSetLayout descriptor_set_layouts[kLoadDescriptorSetCount];
  descriptor_set_layouts[kLoadDescriptorSetIndexDestination] = direct_load_descriptor_set_layout_;
  descriptor_set_layouts[kLoadDescriptorSetIndexSource] =
      command_processor_.GetSingleTransientDescriptorLayout(
          VulkanCommandProcessor::SingleTransientDescriptorLayout::kStorageBufferCompute);
  VkPushConstantRange push_constant_range;
  push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_constant_range.offset = 0;
  push_constant_range.size = sizeof(LoadConstants);
  VkPipelineLayoutCreateInfo pipeline_layout_create_info;
  pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_create_info.pNext = nullptr;
  pipeline_layout_create_info.flags = 0;
  pipeline_layout_create_info.setLayoutCount = kLoadDescriptorSetCount;
  pipeline_layout_create_info.pSetLayouts = descriptor_set_layouts;
  pipeline_layout_create_info.pushConstantRangeCount = 1;
  pipeline_layout_create_info.pPushConstantRanges = &push_constant_range;
  if (dfn.vkCreatePipelineLayout(device, &pipeline_layout_create_info, nullptr,
                                 &direct_load_pipeline_layout_) != VK_SUCCESS) {
    direct_load_pipeline_layout_ = VK_NULL_HANDLE;
    REXGPU_ERROR("[s7][TEXDIRECT] failed to create the direct load pipeline layout - off");
    ShutdownDirectLoad();
    return false;
  }

  const std::pair<const uint32_t*, size_t> shader_code[kDirectLoadShaderCount] = {
      {shaders::texture_load_32bpb_scaled_direct_cs,
       sizeof(shaders::texture_load_32bpb_scaled_direct_cs)},
      {shaders::texture_load_64bpb_scaled_direct_cs,
       sizeof(shaders::texture_load_64bpb_scaled_direct_cs)},
      // wwe2k14 patch (Session 22, TEXDEPTH).
      {shaders::texture_load_depth_float_scaled_direct_cs,
       sizeof(shaders::texture_load_depth_float_scaled_direct_cs)},
  };
  for (uint32_t i = 0; i < kDirectLoadShaderCount; ++i) {
    direct_load_pipelines_[i] = ui::vulkan::util::CreateComputePipeline(
        vulkan_device, direct_load_pipeline_layout_, shader_code[i].first, shader_code[i].second);
    if (direct_load_pipelines_[i] == VK_NULL_HANDLE) {
      REXGPU_ERROR("[s7][TEXDIRECT] failed to create the direct load pipeline {} - off", i);
      ShutdownDirectLoad();
      return false;
    }
  }

  VkDescriptorPoolSize descriptor_pool_size;
  descriptor_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  descriptor_pool_size.descriptorCount = 1;
  direct_load_descriptor_pool_ = std::make_unique<ui::vulkan::SingleLayoutDescriptorSetPool>(
      vulkan_device, 64, 1, &descriptor_pool_size, direct_load_descriptor_set_layout_);

  REXGPU_INFO(
      "[s7][TEXDIRECT] direct scaled texture loads initialized (vulkan_texture_load_direct on, first scaled "
      "texture created with it); formats without storage image support (not loaded directly):{}",
      formats_without_storage.empty() ? " none" : formats_without_storage);
  return true;
}

void VulkanTextureCache::ShutdownDirectLoad() {
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  // With the GPU idle (texture cache destruction) or nothing recorded yet (failed initialization).
  for (DirectLoadVerifySlot& slot : direct_load_verify_slots_) {
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, slot.buffer);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, slot.memory);
    slot = DirectLoadVerifySlot();
  }
  for (VkPipeline& pipeline : direct_load_pipelines_) {
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device, pipeline);
  }
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device,
                                         direct_load_pipeline_layout_);
  // The pool must be destroyed before its set layout.
  direct_load_descriptor_pool_.reset();
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device,
                                         direct_load_descriptor_set_layout_);
}

void VulkanTextureCache::ReleaseDirectLoadVerifySlots() {
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  for (DirectLoadVerifySlot& slot : direct_load_verify_slots_) {
    // A slot stops being pending only in CheckDirectLoadVerifies, once its submission has completed - nothing
    // on the GPU uses the buffer of a slot that isn't pending.
    if (slot.pending || slot.buffer == VK_NULL_HANDLE) {
      continue;
    }
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, slot.buffer);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, slot.memory);
    slot = DirectLoadVerifySlot();
  }
}

bool VulkanTextureCache::LoadTextureDataDirect(VulkanTexture& texture, DirectLoadShader shader,
                                               LoadShaderIndex load_shader,
                                               VkDescriptorSet descriptor_set_dest, bool is_3d_tiling) {
  // The base level of a single-level, single-layer (depth 1) scaled texture, which is level 0 and not a packed
  // mip tail. The source binding and its barrier, the constants and the group counts are the ones
  // LoadTextureDataFromResidentMemoryImpl uses for the same load; only the destination differs.
  const TextureKey& texture_key = texture.key();
  const LoadShaderInfo& load_shader_info = GetLoadShaderInfo(load_shader);
  const FormatInfo* guest_format_info = FormatInfo::Get(texture_key.format);
  uint32_t block_width = guest_format_info->block_width;
  uint32_t block_height = guest_format_info->block_height;
  uint32_t bytes_per_block = guest_format_info->bytes_per_block();
  uint32_t width = texture_key.GetWidth();
  uint32_t height = texture_key.GetHeight();
  // wwe2k14 patch (Session 43, S0b-lite): a 1x texture (vulkan_texture_load_direct_unscaled) reads the shared memory
  // as the regular 1x load does, with the scaled shader at scale 1.
  const bool scaled = texture_key.scaled_resolve != 0;
  uint32_t texture_resolution_scale_x = scaled ? draw_resolution_scale_x() : 1;
  uint32_t texture_resolution_scale_y = scaled ? draw_resolution_scale_y() : 1;

  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  VulkanSharedMemory& vulkan_shared_memory = static_cast<VulkanSharedMemory&>(shared_memory());

  // Source: the scaled resolve buffer range of the base level (1x: the shared memory range).
  uint64_t source_base_start;
  uint64_t source_base_range;
  if (scaled) {
    if (!GetScaledResolveRange(texture_key.base_page << 12, texture.GetGuestBaseSize(),
                               load_shader_info.source_bpe_log2, source_base_start, source_base_range)) {
      return false;
    }
  } else {
    source_base_start = texture_key.base_page << 12;
    source_base_range = rex::align(uint64_t(texture.GetGuestBaseSize()),
                                   uint64_t(UINT32_C(1) << load_shader_info.source_bpe_log2));
    if (source_base_range > uint64_t(SharedMemory::kBufferSize) - source_base_start) {
      return false;
    }
  }
  VkDescriptorSet descriptor_set_source = command_processor_.AllocateSingleTransientDescriptor(
      VulkanCommandProcessor::SingleTransientDescriptorLayout::kStorageBufferCompute);
  if (!descriptor_set_source) {
    return false;
  }
  VkDescriptorBufferInfo write_descriptor_set_source_buffer_info;
  write_descriptor_set_source_buffer_info.buffer = scaled ? scaled_resolve_buffer_ : vulkan_shared_memory.buffer();
  write_descriptor_set_source_buffer_info.offset = source_base_start;
  write_descriptor_set_source_buffer_info.range = source_base_range;
  VkWriteDescriptorSet write_descriptor_set_source;
  write_descriptor_set_source.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_descriptor_set_source.pNext = nullptr;
  write_descriptor_set_source.dstSet = descriptor_set_source;
  write_descriptor_set_source.dstBinding = 0;
  write_descriptor_set_source.dstArrayElement = 0;
  write_descriptor_set_source.descriptorCount = 1;
  write_descriptor_set_source.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write_descriptor_set_source.pImageInfo = nullptr;
  write_descriptor_set_source.pBufferInfo = &write_descriptor_set_source_buffer_info;
  write_descriptor_set_source.pTexelBufferView = nullptr;
  dfn.vkUpdateDescriptorSets(device, 1, &write_descriptor_set_source, 0, nullptr);

  // Session 7c, CAUSES (review fix: direct loads were missing from the counts): measurement only, as in
  // LoadTextureDataFromResidentMemoryImpl - the load records its commands from here on (no early return after
  // this point).
  {
    VulkanCommandProcessor::CauseCounts& causes = command_processor_.cause_counts();
    ++causes.tex_loads;
    if (scaled) ++causes.tex_loads_scaled;
    causes.tex_load_bytes += source_base_range;
  }

  if (scaled) {
    UseScaledResolveBufferForRead();
  } else {
    vulkan_shared_memory.Use(VulkanSharedMemory::Usage::kRead, {}, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  }

  LoadConstants load_constants;
  load_constants.is_tiled_3d_endian_scale =
      uint32_t(texture_key.tiled) | (uint32_t(is_3d_tiling) << 1) |
      (uint32_t(texture_key.endianness) << 2) | (texture_resolution_scale_x << 4) |
      (texture_resolution_scale_y << 7);
  load_constants.guest_offset = 0;
  const texture_util::TextureGuestLayout::Level& level_guest_layout = texture.guest_layout().base;
  uint32_t level_guest_pitch = level_guest_layout.row_pitch_bytes;
  if (texture_key.tiled) {
    // Shaders expect pitch in blocks for tiled textures.
    level_guest_pitch /= bytes_per_block;
  }
  load_constants.guest_pitch_aligned = level_guest_pitch;
  load_constants.guest_z_stride_block_rows_aligned = level_guest_layout.z_slice_stride_block_rows;
  load_constants.size_blocks[0] =
      (width + (block_width - 1)) / block_width * texture_resolution_scale_x;
  load_constants.size_blocks[1] =
      (height + (block_height - 1)) / block_height * texture_resolution_scale_y;
  load_constants.size_blocks[2] = 1;
  // Not read by the direct shaders: the buffer path's values (used by the verify reference dispatch).
  uint32_t host_x_blocks_per_thread =
      (UINT32_C(1) << load_shader_info.guest_x_blocks_per_thread_log2) * block_width;
  uint32_t host_x_pitch_blocks =
      rex::round_up(width * texture_resolution_scale_x, host_x_blocks_per_thread);
  load_constants.host_offset = 0;
  load_constants.host_pitch = load_shader_info.bytes_per_host_block * host_x_pitch_blocks;
  load_constants.height_texels = height;

  uint32_t guest_x_blocks_per_group_log2 = load_shader_info.GetGuestXBlocksPerGroupLog2();
  uint32_t group_count_x =
      (load_constants.size_blocks[0] + ((UINT32_C(1) << guest_x_blocks_per_group_log2) - 1)) >>
      guest_x_blocks_per_group_log2;
  uint32_t group_count_y =
      (load_constants.size_blocks[1] + ((UINT32_C(1) << kLoadGuestYBlocksPerGroupLog2) - 1)) >>
      kLoadGuestYBlocksPerGroupLog2;

  // The dispatch writes every texel of the image's only subresource (the threads cover size_blocks, which is the
  // image extent), so the old contents are not needed (UNDEFINED, also skipping a layout transition that
  // preserves them). The source stages of the old usage still order the earlier reads and writes before it.
  texture.MarkAsUsed();
  VulkanTexture::Usage texture_old_usage =
      texture.SetUsage(VulkanTexture::Usage::kComputeStorageWrite);
  // wwe2k14 patch (Session 80 review): a deferred native copy still to write it runs first - this load of the newer
  // memory comes after it (else the copy, run later, would write older texels over it).
  if (texture_old_usage == VulkanTexture::Usage::kComputeStorageWrite) {
    command_processor_.NativeFlushDeferredResolves(&texture);
  }
  VkPipelineStageFlags texture_src_stage_mask;
  VkAccessFlags texture_src_access_mask;
  VkImageLayout texture_old_layout;
  GetTextureUsageMasks(texture_old_usage, texture_src_stage_mask, texture_src_access_mask,
                       texture_old_layout);
  command_processor_.PushImageMemoryBarrier(
      texture.image(), ui::vulkan::util::InitializeSubresourceRange(), texture_src_stage_mask,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, texture_src_access_mask, VK_ACCESS_SHADER_WRITE_BIT,
      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

  DeferredCommandBuffer& command_buffer = command_processor_.deferred_command_buffer();
  command_processor_.BindExternalComputePipeline(direct_load_pipelines_[shader]);
  VkDescriptorSet descriptor_sets[kLoadDescriptorSetCount];
  descriptor_sets[kLoadDescriptorSetIndexDestination] = descriptor_set_dest;
  descriptor_sets[kLoadDescriptorSetIndexSource] = descriptor_set_source;
  command_buffer.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE,
                                         direct_load_pipeline_layout_, 0, kLoadDescriptorSetCount,
                                         descriptor_sets, 0, nullptr);
  command_buffer.CmdVkPushConstants(direct_load_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                    sizeof(load_constants), &load_constants);
  command_processor_.SubmitBarriers(true);
  command_buffer.CmdVkDispatch(group_count_x, group_count_y, 1);

  DirectLoadStats& stats = direct_load_frame_stats_;
  uint32_t direct_load_index_in_frame = stats.direct_total();
  ++stats.direct[shader];
  stats.direct_bytes += uint64_t(load_constants.size_blocks[0]) * load_constants.size_blocks[1] *
                        load_shader_info.bytes_per_host_block;
  // wwe2k14 patch (Session 45, NATIVE TEXWRITE): the texture a later resolve to this page may fill.
  // (Review fix: recorded only when the native resolves fill textures.)
  if (!scaled && (REXCVAR_GET(gpu_native_resolve_texture) || REXCVAR_GET(gpu_native_all))) {
    native_resolve_keys_[texture_key.base_page] = texture_key;
  }
  // wwe2k14 patch (Session 47, NATIVE TRUTH M6): a scaled texture (resolution scale) a truth resolve may fill.
  if (scaled && REXCVAR_GET(gpu_native_truth) && command_processor_.NativeTruthActive()) {
    NativeLearnTextureKey(texture);
  }
  // wwe2k14 patch (Session 45, NATIVE FRAME): direct loads are of resolved ranges (scaled, or 1x over resolved pages).
  ++native_load_counters_.loads;
  ++native_load_counters_.resolved_loads;
  native_load_counters_.resolved_bytes += uint64_t(load_constants.size_blocks[0]) * load_constants.size_blocks[1] *
                                          load_shader_info.bytes_per_host_block;
  if (command_processor_.NativeSegmentLogActive()) {  // Session 45: the segment log
    REXGPU_INFO("[native][seg] texture load (direct): page 0x{:05X} format {} {}x{} pitch {} endian {} tiled {} mips {} "
                "array {} scaled {}, {} bytes",
                uint32_t(texture_key.base_page), uint32_t(texture_key.format), texture_key.GetWidth(),
                texture_key.GetHeight(), uint32_t(texture_key.pitch), uint32_t(texture_key.endianness),
                uint32_t(texture_key.tiled), uint32_t(texture_key.mip_max_level),
                uint32_t(texture_key.depth_or_array_size_minus_1), uint32_t(scaled),
                uint64_t(load_constants.size_blocks[0]) * load_constants.size_blocks[1] *
                    load_shader_info.bytes_per_host_block);
  }

  if (REXCVAR_GET(vulkan_texture_load_direct_verify) &&
      direct_load_index_in_frame >= direct_load_verify_skip_ &&
      int32_t(direct_load_frame_verifies_) < REXCVAR_GET(vulkan_texture_load_direct_verify_per_frame) &&
      // wwe2k14 patch (Session 22, TEXDEPTH): measurement only - verify only one direct shader's loads.
      (REXCVAR_GET(vulkan_texture_load_direct_verify_shader) < 0 ||
       REXCVAR_GET(vulkan_texture_load_direct_verify_shader) == int32_t(shader))) {
    RecordDirectLoadVerify(texture, load_shader, descriptor_set_source, load_constants, group_count_x,
                           group_count_y, load_shader_info.bytes_per_host_block);
  }
  return true;
}

void VulkanTextureCache::RecordDirectLoadVerify(VulkanTexture& texture, LoadShaderIndex load_shader,
                                                VkDescriptorSet descriptor_set_source,
                                                const LoadConstants& load_constants,
                                                uint32_t group_count_x, uint32_t group_count_y,
                                                uint32_t bytes_per_texel) {
  ++direct_load_frame_verifies_;
  DirectLoadStats& stats = direct_load_frame_stats_;
  // Session 43 (S0b-lite): a 1x texture's reference is the regular 1x load shader (an independent check of the
  // scaled direct shader at scale 1).
  VkPipeline reference_pipeline =
      texture.key().scaled_resolve ? load_pipelines_scaled_[load_shader] : load_pipelines_[load_shader];
  if (reference_pipeline == VK_NULL_HANDLE) {
    ++stats.verify_dropped;
    return;
  }
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  uint32_t width_scaled = load_constants.size_blocks[0];
  uint32_t height_scaled = load_constants.size_blocks[1];
  VkDeviceSize half_size = VkDeviceSize(load_constants.host_pitch) * height_scaled;
  VkDeviceSize slot_size_needed = half_size * 2;

  // A slot not waiting for its submission: one large enough, else an unallocated one, else any (recreated).
  DirectLoadVerifySlot* slot = nullptr;
  for (uint32_t attempt = 0; attempt < 2 && !slot; ++attempt) {
    if (attempt) {
      CheckDirectLoadVerifies();
    }
    DirectLoadVerifySlot* slot_unallocated = nullptr;
    DirectLoadVerifySlot* slot_small = nullptr;
    for (DirectLoadVerifySlot& slot_candidate : direct_load_verify_slots_) {
      if (slot_candidate.pending) {
        continue;
      }
      if (slot_candidate.buffer == VK_NULL_HANDLE) {
        if (!slot_unallocated) {
          slot_unallocated = &slot_candidate;
        }
      } else if (slot_candidate.size >= slot_size_needed) {
        slot = &slot_candidate;
        break;
      } else if (!slot_small) {
        slot_small = &slot_candidate;
      }
    }
    if (!slot) {
      slot = slot_unallocated ? slot_unallocated : slot_small;
    }
  }
  if (!slot) {
    ++stats.verify_dropped;
    return;
  }
  if (slot->size < slot_size_needed) {
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, slot->buffer);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, slot->memory);
    *slot = DirectLoadVerifySlot();
    uint32_t memory_type = UINT32_MAX;
    void* mapping = nullptr;
    if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
            vulkan_device, slot_size_needed,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            ui::vulkan::util::MemoryPurpose::kReadback, slot->buffer, slot->memory, &memory_type) ||
        dfn.vkMapMemory(device, slot->memory, 0, VK_WHOLE_SIZE, 0, &mapping) != VK_SUCCESS) {
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, slot->buffer);
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, slot->memory);
      *slot = DirectLoadVerifySlot();
      REXGPU_ERROR("[s7][TEXDIRECT] verify: failed to create a {} MB readback buffer",
                   slot_size_needed >> 20);
      ++stats.verify_dropped;
      return;
    }
    slot->size = slot_size_needed;
    slot->mapping = static_cast<const uint8_t*>(mapping);
    slot->coherent = (vulkan_device->memory_types().host_coherent & (UINT32_C(1) << memory_type)) != 0;
  }

  // The reference: the buffer + copy path's load shader with the same source and constants, into the first half
  // of the slot (the scratch buffer layout: row pitch host_pitch).
  VkDescriptorSet descriptor_set_reference = command_processor_.AllocateSingleTransientDescriptor(
      VulkanCommandProcessor::SingleTransientDescriptorLayout::kStorageBufferCompute);
  if (!descriptor_set_reference) {
    ++stats.verify_dropped;
    return;
  }
  VkDescriptorBufferInfo write_descriptor_set_reference_buffer_info;
  write_descriptor_set_reference_buffer_info.buffer = slot->buffer;
  write_descriptor_set_reference_buffer_info.offset = 0;
  write_descriptor_set_reference_buffer_info.range = half_size;
  VkWriteDescriptorSet write_descriptor_set_reference;
  write_descriptor_set_reference.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_descriptor_set_reference.pNext = nullptr;
  write_descriptor_set_reference.dstSet = descriptor_set_reference;
  write_descriptor_set_reference.dstBinding = 0;
  write_descriptor_set_reference.dstArrayElement = 0;
  write_descriptor_set_reference.descriptorCount = 1;
  write_descriptor_set_reference.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write_descriptor_set_reference.pImageInfo = nullptr;
  write_descriptor_set_reference.pBufferInfo = &write_descriptor_set_reference_buffer_info;
  write_descriptor_set_reference.pTexelBufferView = nullptr;
  dfn.vkUpdateDescriptorSets(device, 1, &write_descriptor_set_reference, 0, nullptr);
  LoadConstants reference_constants = load_constants;
  bool control = REXCVAR_GET(vulkan_texture_load_direct_verify_control);
  if (control) {
    // Positive control: none <-> 8in16, 8in32 <-> 16in32.
    reference_constants.is_tiled_3d_endian_scale ^= UINT32_C(1) << 2;
  }
  DeferredCommandBuffer& command_buffer = command_processor_.deferred_command_buffer();
  command_processor_.BindExternalComputePipeline(reference_pipeline);
  VkDescriptorSet descriptor_sets[kLoadDescriptorSetCount];
  descriptor_sets[kLoadDescriptorSetIndexDestination] = descriptor_set_reference;
  descriptor_sets[kLoadDescriptorSetIndexSource] = descriptor_set_source;
  command_buffer.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, load_pipeline_layout_, 0,
                                         kLoadDescriptorSetCount, descriptor_sets, 0, nullptr);
  command_buffer.CmdVkPushConstants(load_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                    sizeof(reference_constants), &reference_constants);
  // Nothing written between the direct dispatch and this one (only the source is shared, and both read it).
  command_processor_.SubmitBarriers(true);
  command_buffer.CmdVkDispatch(group_count_x, group_count_y, 1);

  // The directly written image into the second half, with the same row pitch.
  VkImageSubresourceRange texture_subresource_range = ui::vulkan::util::InitializeSubresourceRange();
  command_processor_.PushBufferMemoryBarrier(slot->buffer, 0, half_size,
                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                             VK_ACCESS_HOST_READ_BIT);
  command_processor_.PushImageMemoryBarrier(
      texture.image(), texture_subresource_range, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
  command_processor_.SubmitBarriers(true);
  VkBufferImageCopy copy_region;
  copy_region.bufferOffset = half_size;
  copy_region.bufferRowLength = load_constants.host_pitch / bytes_per_texel;
  copy_region.bufferImageHeight = height_scaled;
  copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy_region.imageSubresource.mipLevel = 0;
  copy_region.imageSubresource.baseArrayLayer = 0;
  copy_region.imageSubresource.layerCount = 1;
  copy_region.imageOffset.x = 0;
  copy_region.imageOffset.y = 0;
  copy_region.imageOffset.z = 0;
  copy_region.imageExtent.width = width_scaled;
  copy_region.imageExtent.height = height_scaled;
  copy_region.imageExtent.depth = 1;
  command_buffer.CmdVkCopyImageToBuffer(texture.image(), VK_IMAGE_LAYOUT_GENERAL, slot->buffer, 1,
                                        &copy_region);
  command_processor_.PushBufferMemoryBarrier(slot->buffer, half_size, half_size,
                                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                                             VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                                             VK_ACCESS_HOST_READ_BIT);
  // The next transition of the image waits for the compute stage (Usage::kComputeStorageWrite) - chain the copy's
  // read into it, so a later write or layout transition can't overtake the copy.
  command_processor_.PushImageMemoryBarrier(texture.image(), texture_subresource_range,
                                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                                            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);

  const TextureKey& texture_key = texture.key();
  slot->pending = true;
  slot->submission = command_processor_.GetCurrentSubmission();
  slot->lost_upload_count = command_processor_.GetAsyncSubmitCommandsDropped();
  slot->frame = command_processor_.GetCurrentFrame();
  slot->half_size = half_size;
  slot->row_pitch = load_constants.host_pitch;
  slot->bytes_per_texel = bytes_per_texel;
  slot->width_scaled = width_scaled;
  slot->height_scaled = height_scaled;
  slot->base_address = texture_key.base_page << 12;
  slot->width = texture_key.GetWidth();
  slot->height = texture_key.GetHeight();
  slot->guest_format = texture_key.format;
  slot->host_format = texture.direct_load_format();
  slot->load_flags = load_constants.is_tiled_3d_endian_scale;
  slot->control = control;
  ++stats.verify_recorded;
}

void VulkanTextureCache::CheckDirectLoadVerifies() {
  const uint64_t submission_completed = command_processor_.GetCompletedSubmission();
  const uint64_t lost_upload_count = command_processor_.GetAsyncSubmitCommandsDropped();
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  DirectLoadStats& stats = direct_load_frame_stats_;
  // Review fix: the compare time is reported (verify_compare_us), as it's command processor thread time that
  // only exists with verify on.
  uint64_t compare_start_ticks = 0;
  for (DirectLoadVerifySlot& slot : direct_load_verify_slots_) {
    if (!slot.pending || slot.submission > submission_completed) {
      continue;
    }
    if (!compare_start_ticks) {
      compare_start_ticks = std::max(rex::chrono::Clock::QueryHostTickCount(), uint64_t(1));
    }
    slot.pending = false;
    // An async submission's commands were dropped since this readback was recorded (counted before that
    // submission's fence can be signaled): it may never have run, so it isn't compared.
    if (slot.lost_upload_count != lost_upload_count) {
      ++stats.verify_dropped;
      continue;
    }
    if (!slot.coherent) {
      VkMappedMemoryRange readback_memory_range = {};
      readback_memory_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      readback_memory_range.memory = slot.memory;
      readback_memory_range.offset = 0;
      readback_memory_range.size = VK_WHOLE_SIZE;
      vulkan_device->functions().vkInvalidateMappedMemoryRanges(vulkan_device->device(), 1,
                                                                &readback_memory_range);
    }
    ++stats.verify_checked;
    const size_t row_bytes = size_t(slot.width_scaled) * slot.bytes_per_texel;
    uint64_t texels_differing = 0;
    uint32_t first_x = 0;
    uint32_t first_y = 0;
    for (uint32_t y = 0; y < slot.height_scaled; ++y) {
      const uint8_t* reference = slot.mapping + size_t(y) * slot.row_pitch;
      const uint8_t* direct = slot.mapping + size_t(slot.half_size) + size_t(y) * slot.row_pitch;
      if (!std::memcmp(reference, direct, row_bytes)) {
        continue;
      }
      for (uint32_t x = 0; x < slot.width_scaled; ++x) {
        if (std::memcmp(reference + size_t(x) * slot.bytes_per_texel,
                        direct + size_t(x) * slot.bytes_per_texel, slot.bytes_per_texel)) {
          if (!texels_differing) {
            first_x = x;
            first_y = y;
          }
          ++texels_differing;
        }
      }
    }
    if (!texels_differing) {
      continue;
    }
    ++stats.verify_mismatches;
    if (direct_load_verify_mismatches_logged_ < 16) {
      ++direct_load_verify_mismatches_logged_;
      // Addresses, sizes and formats only - never texel contents.
      REXGPU_WARN(
          "[s7][TEXDIRECT] verify mismatch {}: load in frame {} (compared at frame {}), texture 0x{:08X} {}x{} {} "
          "-> {}x{} texels, view format {}, load flags 0x{:03X} (3D tiling {}, endian {}, scale {}x{}): {} of {} "
          "texels differ, first at ({}, {}){}",
          direct_load_verify_mismatches_logged_, slot.frame, command_processor_.GetCurrentFrame(),
          slot.base_address, slot.width, slot.height, FormatInfo::Get(slot.guest_format)->name,
          slot.width_scaled, slot.height_scaled, uint32_t(slot.host_format), slot.load_flags,
          (slot.load_flags >> 1) & 1, (slot.load_flags >> 2) & 3, (slot.load_flags >> 4) & 7,
          (slot.load_flags >> 7) & 7, texels_differing,
          uint64_t(slot.width_scaled) * slot.height_scaled, first_x, first_y,
          slot.control ? " (vulkan_texture_load_direct_verify_control: expected)" : "");
    }
  }
  if (compare_start_ticks) {
    const uint64_t tick_frequency = std::max(rex::chrono::Clock::QueryHostTickFrequency(), uint64_t(1));
    const uint64_t ticks = rex::chrono::Clock::QueryHostTickCount() - compare_start_ticks;
    stats.verify_compare_us += ticks * 1000000 / tick_frequency;
  }
}

void VulkanTextureCache::RecordDirectLoadFrameCounters(uint64_t frame) {
  // Measurement only. cpf_texdirect a = direct loads, b = scaled loads that took the buffer + copy path with
  // vulkan_texture_load_direct on, c = verify mismatches found at this boundary (loads of earlier frames, whose
  // submissions have completed). Recorded while the setting is on or any of them is non-zero.
  CheckDirectLoadVerifies();
  // Review fix: the readback buffers (up to kDirectLoadVerifySlotsMax of 2 copies of the largest image each slot
  // has served) are not kept for the rest of the run after verify is turned off.
  if (!REXCVAR_GET(vulkan_texture_load_direct_verify)) {
    ReleaseDirectLoadVerifySlots();
  }
  DirectLoadStats& frame_stats = direct_load_frame_stats_;
  uint32_t direct = frame_stats.direct_total();
  uint32_t fallbacks = frame_stats.fallbacks_total();
  bool setting_on = REXCVAR_GET(vulkan_texture_load_direct);
  if (perf::timeline::Enabled() &&
      (setting_on || direct || fallbacks || frame_stats.verify_mismatches)) {
    perf::timeline::Record("cpf_texdirect", direct, fallbacks, frame_stats.verify_mismatches);
  }
  direct_load_window_stats_.Add(frame_stats);
  direct_load_run_stats_.Add(frame_stats);
  frame_stats = DirectLoadStats();
  direct_load_frame_verifies_ = 0;
  // Verify different loads of the frame over time.
  direct_load_verify_skip_ = uint32_t(frame) & 7;

  if (direct_load_window_first_frame_ == UINT64_MAX) {
    direct_load_window_first_frame_ = frame;
    return;
  }
  if (frame - direct_load_window_first_frame_ < 300) {
    return;
  }
  const DirectLoadStats& window = direct_load_window_stats_;
  uint32_t window_direct = window.direct_total();
  if (window_direct || window.fallbacks_total() || window.created_capable ||
      window.created_unsupported || window.created_rejected || window.created_unavailable ||
      window.verify_recorded || window.verify_checked || window.verify_dropped) {
    const DirectLoadStats& run = direct_load_run_stats_;
    REXGPU_INFO(
        "[s7][TEXDIRECT] frames {}-{} (vulkan_texture_load_direct {}, _unscaled {}, verify {}{}): {} direct loads ({} 32bpb, "
        "{} 64bpb, {:.1f} MB written), {} eligible loads on the buffer + copy path (levels {}, layers {}, packed "
        "{}, format {}, device rejected {}, unavailable {}, created without storage {}, descriptor {}); "
        "textures (scaled, or 1x with _unscaled; Session 43) created with the storage usage {}, not eligible {}, rejected by the device {}, while "
        "unavailable {}; verify: recorded {}, compared {}, mismatches {}, dropped {}, compare time {:.1f} ms; "
        "run: {} direct loads, compared {}, mismatches {}",
        direct_load_window_first_frame_, frame, setting_on ? "on" : "off",
        REXCVAR_GET(vulkan_texture_load_direct_unscaled) ? "on" : "off",
        REXCVAR_GET(vulkan_texture_load_direct_verify) ? "on" : "off",
        REXCVAR_GET(vulkan_texture_load_direct_verify_control) ? ", positive control" : "", window_direct,
        window.direct[kDirectLoadShader32bpb], window.direct[kDirectLoadShader64bpb],
        double(window.direct_bytes) / (1024.0 * 1024.0), window.fallbacks_total(),
        window.fallbacks[kDirectLoadFallbackLevels], window.fallbacks[kDirectLoadFallbackLayers],
        window.fallbacks[kDirectLoadFallbackPacked], window.fallbacks[kDirectLoadFallbackFormat],
        window.fallbacks[kDirectLoadFallbackDevice], window.fallbacks[kDirectLoadFallbackUnavailable],
        window.fallbacks[kDirectLoadFallbackNoStorage],
        window.fallbacks[kDirectLoadFallbackDescriptor], window.created_capable,
        window.created_unsupported, window.created_rejected, window.created_unavailable,
        window.verify_recorded, window.verify_checked, window.verify_mismatches, window.verify_dropped,
        double(window.verify_compare_us) / 1000.0, run.direct_total(), run.verify_checked,
        run.verify_mismatches);
    if (direct_load_skipped_debug_) {
      REXGPU_INFO("[s44][S0B] vulkan_texture_load_skip_debug {}: {} direct 1x reloads skipped (run total)",
                  REXCVAR_GET(vulkan_texture_load_skip_debug) ? "on" : "off", direct_load_skipped_debug_);
    }
  }
  direct_load_window_stats_ = DirectLoadStats();
  direct_load_window_first_frame_ = frame;
}


// wwe2k14 patch (Session 92, M4c): GetView's view key and lookup, nothing else (never creates a view).
VkImageView VulkanTextureCache::VulkanTexture::FindView(bool is_signed, uint32_t host_swizzle, bool is_array) const {
  xenos::DataDimension dimension = key().dimension;
  if (dimension == xenos::DataDimension::k3D || dimension == xenos::DataDimension::kCube) {
    is_array = false;
  }
  const VulkanTextureCache& vulkan_texture_cache = static_cast<const VulkanTextureCache&>(texture_cache());
  ViewKey view_key;
  const HostFormatPair& host_format_pair = vulkan_texture_cache.GetHostFormatPair(key());
  VkFormat format = (is_signed ? host_format_pair.format_signed : host_format_pair.format_unsigned).format;
  if (format == VK_FORMAT_UNDEFINED) return VK_NULL_HANDLE;
  view_key.is_signed_separate_view = is_signed && (host_format_pair.format_signed.format != host_format_pair.format_unsigned.format);
  if (!vulkan_texture_cache.command_processor_.GetVulkanDevice()->properties().imageViewFormatSwizzle) {
    host_swizzle = xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA;
  }
  view_key.host_swizzle = host_swizzle;
  view_key.is_array = uint32_t(is_array);
  auto it = views_.find(view_key);
  return it != views_.end() ? it->second : VK_NULL_HANDLE;
}
}  // namespace rex::graphics::vulkan
