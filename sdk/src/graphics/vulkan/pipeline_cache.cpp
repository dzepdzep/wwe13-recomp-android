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

#include "disk_pipeline_cache.h"  // wwe2k14 patch (Session 63)
#include "spirv_opt_store.h"     // wwe2k14 patch (Session 70)
#include "../pipeline/shader/translation_store.h"  // wwe2k14 patch (Session 92, M6)

#include <algorithm>
#include <array>
#include <bit>
#include <mutex>
#include <unordered_set>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <tuple>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <rex/assert.h>
#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/filesystem.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/thread.h>
#include <rex/memory.h>
#include <rex/graphics/util/bit_inline.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/pipeline/texture/util.h>  // wwe2k14 patch (Session 66, PS spec): SwizzleSigns
#include <rex/graphics/pipeline_util.h>
#include <rex/graphics/pipeline/shader/spirv_builder.h>
#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/pipeline_cache.h>
#include <rex/perf/frame_events.h>  // Session 73 (what a frame did)
#include "s58_stats.h"  // wwe2k14 patch (Session 58)
#include <rex/graphics/vulkan/shader.h>
#include <rex/graphics/xenos.h>
#include <rex/hash.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/types.h>
#include <rex/ui/vulkan/util.h>
#include "native/shader_swap.h"  // wwe2k14 patch (Session 72, gpu_native_shader_swap)

REXCVAR_DEFINE_INT32(
    vulkan_pipeline_creation_threads, -1, "GPU/Vulkan",
    "Number of pipeline creation threads for Vulkan async pipeline creation (-1 for auto)")
    .range(-1, 32)
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DECLARE(bool, gpu_dlss);  // wwe2k14 patch (Session 24, DLSS)
REXCVAR_DECLARE(bool, vulkan_extended_dynamic_state);  // wwe2k14 patch (Session 93, idea 7): vulkan_device.cpp
REXCVAR_DECLARE(int32_t, gpu_quirks_eds_allowed);      // wwe2k14 patch (Session 93, idea 6): gpu_quirks_apply.cpp
REXCVAR_DECLARE(bool, store_shaders);  // wwe2k14 patch (Session 92, M6): graphics_system.cpp
REXCVAR_DECLARE(bool, gpu_native_shader_swap);  // wwe2k14 patch (Session 72, native/shader_swap.cpp)
REXCVAR_DECLARE(bool, gpu_native_vs_swap);  // wwe2k14 patch (Session 87, native/shader_swap.cpp)
REXCVAR_DEFINE_BOOL(gpu_d3d_replbind_memo, false, "GPU/Native",
                    "Session 95: VulkanPipelineCache::UsesReplacementBindings (several calls a draw per stage) uses a static_cast to SpirvShader (every shader here is a VulkanShader) "
                    "instead of a dynamic_cast every call (about 1% of the command thread). The answer is the same. Off = a dynamic_cast every call.");
// wwe2k14 patch (Session 87, stage 4): a shader with a registered replacement skips the microcode analysis and the translation.
REXCVAR_DEFINE_BOOL(gpu_native_swap_skip_translation, false, "GPU/Native",
                    "Session 87 stage 4: VERTEX shaders with a registered replacement (gpu_native_vs_swap) take their analysis "
                    "(float constants, vertex stream, interpolators) from the registration, and NATIVE's translation of them is "
                    "the replacement itself: no microcode analysis and no translator run. A shader the registration cannot "
                    "describe (it samples textures, or its replacement reads the bool / loop constants) keeps the real "
                    "analysis and the translator, as do the emulated (non-NATIVE) pipelines, which then also run the real "
                    "analysis. Needs gpu_native_vs_swap.")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);
REXCVAR_DEFINE_BOOL(gpu_native_swap_skip_translation_pixel, false, "GPU/Native",
                    "Session 87 stage 4: the same for PIXEL shaders (gpu_native_shader_swap). NOT FIT FOR USE: the shader keeps "
                    "one set of bindings, but a pixel shader has two translations with different ones (docs/native/d3d/"
                    "SKIP_TRANSLATION_PIXEL.md); the first check of it failed validation. Default off, unused for now.")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);
REXCVAR_DEFINE_BOOL(gpu_native_swap_skip_verify, false, "GPU/Native",
                    "Session 87 stage 4: with gpu_native_swap_skip_translation, every shader whose analysis came from its "
                    "registration is also analysed and translated for real in a scratch shader, and the two compared field by "
                    "field (log lines '[native][swap_skip][verify]'). Costs the analysis and translation it saves.")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

namespace {
// The census of gpu_native_swap_skip_translation (v = vertex, p = pixel), in every log line of it.
struct SwapSkipCensus {
  std::atomic<uint32_t> analysis_v{0}, analysis_p{0};      // analysis taken from the registration
  std::atomic<uint32_t> native_skip_v{0}, native_skip_p{0};  // NATIVE translations made without the translator
  std::atomic<uint32_t> refused_v{0}, refused_p{0};        // the replacement refused: the translator ran
  std::atomic<uint32_t> deferred_v{0}, deferred_p{0};      // emulated translations left undone
  std::atomic<uint32_t> forced_v{0}, forced_p{0};          // registered analyses dropped for the real one
  std::atomic<uint32_t> refused_tex_v{0}, refused_bool_v{0}, guard_bad_v{0};  // vertex shaders the registration cannot describe; analysis changes found when a forced real analysis ran
  std::atomic<uint32_t> verify_shaders{0}, verify_bad_shaders{0}, verify_bad_fields{0}, verify_logged{0};
};
SwapSkipCensus g_skip;

bool SwapSkipOn() {  // (vertex shaders)
  return REXCVAR_GET(gpu_native_swap_skip_translation) && REXCVAR_GET(gpu_native_vs_swap);
}
bool SwapSkipOnPixel() {
  return REXCVAR_GET(gpu_native_swap_skip_translation_pixel) && REXCVAR_GET(gpu_native_shader_swap);
}

std::string SwapSkipTotals() {
  return fmt::format(
      "totals: analysis from registration {}v/{}p, NATIVE translation without the translator {}v/{}p, refused {}v/{}p, "
      "emulated deferred (calls) {}v/{}p, real analysis forced {}v/{}p, vertex refused (textures {}, bool/loop {}), forced-analysis changes {}, verified {} ({} differ, {} fields)",
      g_skip.analysis_v.load(), g_skip.analysis_p.load(), g_skip.native_skip_v.load(), g_skip.native_skip_p.load(),
      g_skip.refused_v.load(), g_skip.refused_p.load(), g_skip.deferred_v.load(), g_skip.deferred_p.load(),
      g_skip.forced_v.load(), g_skip.forced_p.load(), g_skip.refused_tex_v.load(), g_skip.refused_bool_v.load(), g_skip.guard_bad_v.load(), g_skip.verify_shaders.load(), g_skip.verify_bad_shaders.load(),
      g_skip.verify_bad_fields.load());
}
}  // namespace
// wwe2k14 patch (Session 66, B2.1): see VulkanPipelineCache::GetVertexFetchEndianKey.
REXCVAR_DEFINE_BOOL(gpu_native_vertex_endian_spec, false, "GPU/Native",
                    "Session 66 (B2.1): the vertex shaders get the endian mode of each of their (up to 8) vertex fetch "
                    "constants baked in (a fixed swap instead of reading the fetch constant and branching): one "
                    "translation per distinct set of modes. The picture is unchanged. Off by default")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 66, PS spec 1): see VulkanPipelineCache::GetNativePixelSpecKey.
REXCVAR_DEFINE_BOOL(gpu_native_ps_texture_spec, false, "GPU/Native",
                    "Session 66 (PS spec 1): NATIVE's pixel shaders get each of their first 4 texture bindings whose "
                    "swizzled signs are all unsigned (the usual case) translated without the signed / biased / gamma "
                    "code and the second (signed) sample: one translation per distinct set of such bindings. Exact "
                    "(the code of the path taken today). Off by default")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 66, PS spec 2): see VulkanPipelineCache::GetNativePixelSpecKey.
REXCVAR_DEFINE_INT32(gpu_native_ps_mul_mode, 0, "GPU/Native",
                     "Session 66 (PS spec 2): the Xenos multiply rule (0 x anything = +0) in NATIVE's pixel shaders. 0 = "
                     "as the game's GPU does (exact). 1 = left out where both operands are proven finite - exact ONLY if "
                     "the interpolators, float constants and texture results are finite (the proof treats them as such; "
                     "sign of zero may differ: -0 instead of +0). 2 = as 1, and on the other multiplies only NaN -> 0 "
                     "(0 x inf stays 0; NaN x non-zero gives 0). 3 = left out everywhere (0 x inf / 0 x NaN give NaN: "
                     "can show black or flashing pixels). Off (0) by default")
    .range(0, 3)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 66, PS spec 3): see VulkanPipelineCache::GetNativePixelSpecKey.
REXCVAR_DEFINE_BOOL(gpu_native_ps_implicit_lod, false, "GPU/Native",
                    "Session 66 (PS spec 3): NATIVE's pixel shaders sample 1D / 2D / cube textures with computed LOD "
                    "(not register gradients) using the hardware's implicit LOD with the LOD bias, instead of explicit "
                    "gradients from coarse derivatives. Looks right, not exact (fine derivatives, the hardware's "
                    "anisotropy). Off by default")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 66, native_ext): see VulkanCommandProcessor::NativePixelExt.
REXCVAR_DEFINE_INT32(gpu_native_fma, 0, "GPU/Native",
                     "Session 66 (native_ext): bit mask - 1 = NATIVE's pixel shaders, 2 = NATIVE's vertex shaders (the plain "
                     "vertex kind): a multiply and the add after it (mad, the sum of a dot product) become one fused "
                     "multiply-add where the Xenos multiply rule is already gone (gpu_native_ps_mul_mode / "
                     "gpu_native_vs_mul_mode, on the lanes it cleared). One rounding instead of two: positions and "
                     "colours can differ by 1 ulp (looks right, not exact; it must be on for the depth and the colour "
                     "pass alike). Off (0) by default")
    .range(0, 3)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_ps_epilogue_spec, false, "GPU/Native",
                    "Session 66 (native_ext): NATIVE's pixel shaders for a draw with no alpha test, no alpha to coverage, "
                    "colour exponent bias 1 and no gamma conversion on any colour target get a translation without the "
                    "epilogue code for them (exact: the code of the paths taken). Off by default")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 66, VS mul): see VulkanPipelineCache::GetNativeVertexMulKey.
REXCVAR_DEFINE_INT32(gpu_native_vs_mul_mode, 0, "GPU/Native",
                     "Session 66 (VS mul): the Xenos multiply rule (0 x anything = +0) in the vertex shaders NATIVE draws "
                     "(only with the native renderer on; the emulated path is unchanged). 0 = as the game's GPU does "
                     "(exact). 1 = left out where both operands are proven finite - exact ONLY if the float constants (and "
                     "the vertex data, see gpu_native_vs_mul_fetch_finite) are finite; sign of zero may differ. 2 = as 1, "
                     "and on the other multiplies only NaN -> 0. 3 = left out everywhere (0 x inf / 0 x NaN give NaN: "
                     "broken positions possible). Off (0) by default")
    .range(0, 3)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_vs_mul_fetch_finite, false, "GPU/Native",
                    "Session 66 (VS mul): with gpu_native_vs_mul_mode 1 / 2, float vertex fetch results (32-bit and 16-bit "
                    "float formats) and vertex texture fetch results count as finite (the usual mesh data; normalized, "
                    "packed and integer formats are always finite). Off: they are tainted, so a multiply of them keeps "
                    "the rule")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(vulkan_tessellation_wireframe, false, "GPU/Vulkan",
                    "Render tessellation as wireframe")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// wwe2k14 patch (Session 6, H1): first-encounter pipeline hitches (290-834 ms in WWE 2K14) are the command
// processor thread building a placeholder pipeline per new pipeline (16-73 ms each, serial), then waiting in
// EndSubmission for the real pipelines (FSI pixel shaders, hundreds of ms) although no recorded command
// references them: the deferred command buffer stores VkPipeline values and IssueDraw skips draws whose
// pipeline is a placeholder, so the wait (inherited from the D3D12 backend, whose deferred command list
// resolves pipeline handles at submission) only stalls the guest.
REXCVAR_DEFINE_BOOL(vulkan_pipeline_no_submit_wait, false, "GPU/Vulkan",
                    "With async_shader_compilation: the command processor never waits for background "
                    "pipeline creation (no placeholder pipeline build, no wait at submission end, "
                    "mid-frame submissions allowed while compiling, pipelines without a pixel shader "
                    "created in the background too). Draws are skipped until their pipeline is ready; "
                    "vulkan_async_skip_incomplete_frames decides whether such frames are presented")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(vulkan_pipeline_sync_depth_only, false, "GPU/Vulkan",
                    "With vulkan_pipeline_no_submit_wait: still create pipelines for draws without a "
                    "pixel shader (depth-only) on the command processor thread, as before")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 55, the red blocks / white wrestlers on slow laptops): a draw skipped while its pipeline is
// being made is lost for good when it builds a texture the game keeps (the wrestlers' while a match loads; with the
// native renderer's compiles slowed down to 100 ms, run 3729, trunks, waistbands and skin stayed washed out all
// match). Only draws into screen-wide surfaces (1280 pixels or more) are skipped; the others wait for their pipeline.
REXCVAR_DEFINE_INT32(vulkan_pipeline_debug_delay_ms, 0, "GPU/Vulkan",
                     "Diagnostic (Session 55): every pipeline creation (creation threads, the command processor's "
                     "own, placeholders) waits this many milliseconds first (a slow laptop's compile time); 0 = none")
    .range(0, 10000)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 80, HD 520 arc): the stored pipelines are made at boot, blocking, before the game's first frame.
// With the native renderer's truth frames the emulated path draws nothing (the Surface's UHD 620, run 001: "emulated
// 0.0" in every frame summary, no emulated pipeline made after boot), and there the 406 stored ones took 2.6 s each, ~3
// minutes with the window "not responding". false keeps their records (the storage file is not trimmed) but makes none
// at boot; one needed later is made on first use like any new pipeline. The game turns it off with truth frames.
REXCVAR_DEFINE_BOOL(vulkan_stored_pipelines_at_boot, true, "GPU/Vulkan",
                    "wwe2k14 (Session 80): make the stored pipelines (and translate their shaders) at boot, before the "
                    "first frame. false = their records are kept but none is made at boot (made on first use)");
REXCVAR_DEFINE_BOOL(vulkan_pipeline_sync_offscreen, true, "GPU/Vulkan",
                    "Session 55: with async_shader_compilation, a draw into a surface narrower than the screen (under "
                    "1280 pixels) whose pipeline is not ready waits for it instead of being skipped (it may build a "
                    "texture the game keeps). false = every such draw is skipped (Sessions 6-54)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 57, red patches still reported on a GTX 1060): the screen-wide width for
// vulkan_pipeline_sync_offscreen (a texture the game builds in a screen-wide surface would still lose draws).
REXCVAR_DEFINE_INT32(vulkan_pipeline_sync_offscreen_min_pitch, 1280, "GPU/Vulkan",
                     "Session 57: with vulkan_pipeline_sync_offscreen, a draw whose pipeline is not ready may be "
                     "skipped only on a surface at least this many guest pixels wide (1280 = the screen); narrower "
                     "ones wait. 16384 = never skipped")
    .range(0, 16384)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 58): and at most this wide - a wider surface is not a screen (a character's 2048x1024 attire
// is drawn on a 2080-wide one).
REXCVAR_DEFINE_INT32(vulkan_pipeline_sync_offscreen_max_pitch, 2048, "GPU/Vulkan",
                     "Session 58: with vulkan_pipeline_sync_offscreen, a draw on a surface wider than this many guest "
                     "pixels waits for its pipeline too (2048: the 2080-wide surface of a character's attire waits, "
                     "screens up to 1920 may skip). 16384 = no upper limit")
    .range(0, 16384)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// wwe2k14 patch (Session 92, M6 groundwork, log-only): every translation VARIANT a run makes, once per run - the inputs a translation is
// keyed by (emulated or NATIVE, shader type, guest hash, modification, native_ext, the native translator's pixel position scale) and how it
// was made (translator or replacement). The final census's logs then list every variant the game needs, for the launcher's precompile (M6:
// translate before play, from the player's own files; the list carries keys only, rule 1). TRUE NATIVE turns it on (src\d3d_owned.h).
REXCVAR_DEFINE_BOOL(gpu_m6_variant_log, false, "GPU/Native",
                    "Session 92 (M6 groundwork): log each shader translation variant once per run, [m6][variant] (log-only)");

namespace rex::graphics::vulkan {

namespace {

// (M6 groundwork) One line per new variant; the set is file-level (no header member). Any thread (a mutex, taken only when the switch is on).
void M6NoteVariant(bool native, bool vertex, uint64_t hash, uint64_t modification, uint32_t native_ext, float scale_x, float scale_y,
                   bool replacement) {
  if (!REXCVAR_GET(gpu_m6_variant_log)) return;
  static std::mutex mutex;
  static std::set<std::tuple<bool, bool, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t>> seen;
  uint32_t sx, sy;
  std::memcpy(&sx, &scale_x, 4);
  std::memcpy(&sy, &scale_y, 4);
  size_t count;
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!seen.emplace(native, vertex, hash, modification, native_ext, sx, sy).second) return;
    count = seen.size();
  }
  REXGPU_INFO("[m6][variant] {} {} {:016X} mod {:016X} ext {:08X} scale {:.4f}x{:.4f} by {} (#{})", native ? "nat" : "emu", vertex ? "VS" : "PS",
              hash, modification, native_ext, scale_x, scale_y, replacement ? "replacement" : "translator", count);
}

constexpr std::string_view kPlaceholderPixelShaderSource = R"(#version 460
layout(location = 0) out vec4 oC0;
void main() {
  // Avoid full-screen black flashes while async pipelines are warming up.
  discard;
}
)";

constexpr bool IsTriangleDomainHostVertexShaderType(
    Shader::HostVertexShaderType host_vertex_shader_type) {
  return host_vertex_shader_type == Shader::HostVertexShaderType::kTriangleDomainCPIndexed ||
         host_vertex_shader_type == Shader::HostVertexShaderType::kTriangleDomainPatchIndexed;
}

constexpr bool IsQuadDomainHostVertexShaderType(
    Shader::HostVertexShaderType host_vertex_shader_type) {
  return host_vertex_shader_type == Shader::HostVertexShaderType::kQuadDomainCPIndexed ||
         host_vertex_shader_type == Shader::HostVertexShaderType::kQuadDomainPatchIndexed;
}

constexpr bool IsPatchIndexedHostVertexShaderType(
    Shader::HostVertexShaderType host_vertex_shader_type) {
  return host_vertex_shader_type == Shader::HostVertexShaderType::kTriangleDomainPatchIndexed ||
         host_vertex_shader_type == Shader::HostVertexShaderType::kQuadDomainPatchIndexed;
}

std::string GetTessellationSystemConstantsBlockGlsl() {
  std::string source;
  source += "layout(set = 0, binding = 0, std140) uniform XeSystemConstants {\n";
  source += fmt::format("  layout(offset = {}) uint xe_vertex_index_endian;\n",
                        offsetof(SpirvShaderTranslator::SystemConstants, vertex_index_endian));
  source += fmt::format("  layout(offset = {}) int xe_vertex_base_index;\n",
                        offsetof(SpirvShaderTranslator::SystemConstants, vertex_base_index));
  source += fmt::format("  layout(offset = {}) uint xe_vertex_index_min;\n",
                        offsetof(SpirvShaderTranslator::SystemConstants, vertex_index_min));
  source += fmt::format("  layout(offset = {}) uint xe_vertex_index_max;\n",
                        offsetof(SpirvShaderTranslator::SystemConstants, vertex_index_max));
  source +=
      fmt::format("  layout(offset = {}) float xe_tessellation_factor_range_min;\n",
                  offsetof(SpirvShaderTranslator::SystemConstants, tessellation_factor_range_min));
  source +=
      fmt::format("  layout(offset = {}) float xe_tessellation_factor_range_max;\n",
                  offsetof(SpirvShaderTranslator::SystemConstants, tessellation_factor_range_max));
  source += "} xe_system_cbuffer;\n";
  return source;
}

std::string GetTessellationIndexedVertexShaderGlsl() {
  std::string source;
  source += "#version 450\n";
  source += GetTessellationSystemConstantsBlockGlsl();
  source += R"(
layout(location = 0) out float xe_out_value;

uint xe_swap_8_in_16(uint value) {
  return ((value << 8u) & 0xFF00FF00u) | ((value >> 8u) & 0x00FF00FFu);
}

uint xe_swap_16_in_32(uint value) {
  return ((value & 0x0000FFFFu) << 16u) | (value >> 16u);
}

void main() {
  uint value = uint(gl_VertexIndex);
  uint endian = xe_system_cbuffer.xe_vertex_index_endian;
  if (endian == 1u || endian == 2u || endian == 3u) {
    value = xe_swap_8_in_16(value);
  }
  if (endian == 2u || endian == 3u) {
    value = xe_swap_16_in_32(value);
  }
  value = uint(int(value) + xe_system_cbuffer.xe_vertex_base_index);
  value &= 0x00FFFFFFu;
  value = max(value, xe_system_cbuffer.xe_vertex_index_min);
  value = min(value, xe_system_cbuffer.xe_vertex_index_max);
  xe_out_value = float(value);
}
)";
  return source;
}

std::string GetTessellationAdaptiveVertexShaderGlsl() {
  std::string source;
  source += "#version 450\n";
  source += GetTessellationSystemConstantsBlockGlsl();
  source += R"(
layout(location = 0) out float xe_out_value;

uint xe_swap_8_in_16(uint value) {
  return ((value << 8u) & 0xFF00FF00u) | ((value >> 8u) & 0x00FF00FFu);
}

uint xe_swap_16_in_32(uint value) {
  return ((value & 0x0000FFFFu) << 16u) | (value >> 16u);
}

void main() {
  uint value = uint(gl_VertexIndex);
  uint endian = xe_system_cbuffer.xe_vertex_index_endian;
  if (endian == 1u || endian == 2u || endian == 3u) {
    value = xe_swap_8_in_16(value);
  }
  if (endian == 2u || endian == 3u) {
    value = xe_swap_16_in_32(value);
  }
  float tessellation_factor = float(value) + 1.0;
  tessellation_factor = max(
      tessellation_factor, xe_system_cbuffer.xe_tessellation_factor_range_min);
  tessellation_factor = min(
      tessellation_factor, xe_system_cbuffer.xe_tessellation_factor_range_max);
  xe_out_value = tessellation_factor;
}
)";
  return source;
}

std::string GetTessellationControlShaderGlsl(Shader::HostVertexShaderType host_vertex_shader_type,
                                             xenos::TessellationMode tessellation_mode,
                                             uint32_t input_control_points,
                                             uint32_t output_control_points) {
  std::string source;
  source += "#version 450\n";
  source += GetTessellationSystemConstantsBlockGlsl();
  source += "layout(location = 0) in float xe_input_value[];\n";
  source += "layout(location = 0) patch out vec4 xe_out_patch_control_point_indices;\n";
  // In GLSL, primitive mode / spacing / winding are TES layout qualifiers.
  // TCS only supports `layout(vertices = N) out`.
  source += fmt::format("layout(vertices = {}) out;\n", output_control_points);

  source += R"(
void main() {
  gl_out[gl_InvocationID].gl_Position = vec4(0.0);
  if (gl_InvocationID != 0u) {
    return;
  }

  vec4 patch_control_point_indices = vec4(0.0);
)";
  if (tessellation_mode == xenos::TessellationMode::kAdaptive) {
    source += R"(
  uint patch_index = uint(int(gl_PrimitiveID) + xe_system_cbuffer.xe_vertex_base_index);
  patch_index &= 0x00FFFFFFu;
  patch_index = max(patch_index, xe_system_cbuffer.xe_vertex_index_min);
  patch_index = min(patch_index, xe_system_cbuffer.xe_vertex_index_max);
  patch_control_point_indices.x = float(patch_index);
)";
    if (IsTriangleDomainHostVertexShaderType(host_vertex_shader_type)) {
      source += R"(
  gl_TessLevelOuter[0] = xe_input_value[1];
  gl_TessLevelOuter[1] = xe_input_value[2];
  gl_TessLevelOuter[2] = xe_input_value[0];
  gl_TessLevelInner[0] = min(min(xe_input_value[2], xe_input_value[1]),
                             xe_input_value[0]);
)";
    } else {
      source += R"(
  gl_TessLevelOuter[0] = xe_input_value[3];
  gl_TessLevelOuter[1] = xe_input_value[0];
  gl_TessLevelOuter[2] = xe_input_value[1];
  gl_TessLevelOuter[3] = xe_input_value[2];
  gl_TessLevelInner[0] = min(xe_input_value[2], xe_input_value[0]);
  gl_TessLevelInner[1] = min(xe_input_value[1], xe_input_value[3]);
)";
    }
  } else {
    if (IsPatchIndexedHostVertexShaderType(host_vertex_shader_type)) {
      source += "  patch_control_point_indices.x = xe_input_value[0];\n";
    } else {
      source += "  patch_control_point_indices.x = xe_input_value[0];\n";
      if (input_control_points >= 2) {
        source += "  patch_control_point_indices.y = xe_input_value[1];\n";
      }
      if (input_control_points >= 3) {
        source += "  patch_control_point_indices.z = xe_input_value[2];\n";
      }
      if (input_control_points >= 4) {
        source += "  patch_control_point_indices.w = xe_input_value[3];\n";
      }
    }
    source += R"(
  float tessellation_factor = xe_system_cbuffer.xe_tessellation_factor_range_max;
)";
    if (IsTriangleDomainHostVertexShaderType(host_vertex_shader_type)) {
      source += R"(
  gl_TessLevelOuter[0] = tessellation_factor;
  gl_TessLevelOuter[1] = tessellation_factor;
  gl_TessLevelOuter[2] = tessellation_factor;
  gl_TessLevelInner[0] = tessellation_factor;
)";
    } else {
      source += R"(
  gl_TessLevelOuter[0] = tessellation_factor;
  gl_TessLevelOuter[1] = tessellation_factor;
  gl_TessLevelOuter[2] = tessellation_factor;
  gl_TessLevelOuter[3] = tessellation_factor;
  gl_TessLevelInner[0] = tessellation_factor;
  gl_TessLevelInner[1] = tessellation_factor;
)";
    }
  }
  source += R"(
  xe_out_patch_control_point_indices = patch_control_point_indices;
}
)";
  return source;
}

}  // namespace

VulkanPipelineCache::VulkanPipelineCache(VulkanCommandProcessor& command_processor,
                                         const RegisterFile& register_file,
                                         VulkanRenderTargetCache& render_target_cache,
                                         VkShaderStageFlags guest_shader_vertex_stages)
    : command_processor_(command_processor),
      register_file_(register_file),
      render_target_cache_(render_target_cache),
      guest_shader_vertex_stages_(guest_shader_vertex_stages) {}

VulkanPipelineCache::~VulkanPipelineCache() {
  Shutdown();
}

bool VulkanPipelineCache::EnsureInternalFragmentShaders() {
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const bool edram_fragment_shader_interlock =
      render_target_cache_.GetPath() == RenderTargetCache::Path::kPixelShaderInterlock;
  if (edram_fragment_shader_interlock ? depth_only_fragment_shader_ != VK_NULL_HANDLE
                                      : (depth_float24_truncate_fragment_shader_ != VK_NULL_HANDLE &&
                                         depth_float24_round_fragment_shader_ != VK_NULL_HANDLE)) {
    return true;  // made already
  }
  // (A half-made pair - a creation failure - is made again from scratch: nothing leaks.)
  ui::vulkan::util::DestroyAndNullHandle(vulkan_device->functions().vkDestroyShaderModule, vulkan_device->device(),
                                         depth_float24_truncate_fragment_shader_);
  ui::vulkan::util::DestroyAndNullHandle(vulkan_device->functions().vkDestroyShaderModule, vulkan_device->device(),
                                         depth_float24_round_fragment_shader_);
  if (edram_fragment_shader_interlock) {
    std::vector<uint8_t> depth_only_fragment_shader_code =
        shader_translator_->CreateDepthOnlyFragmentShader();
    depth_only_fragment_shader_ = ui::vulkan::util::CreateShaderModule(
        vulkan_device, reinterpret_cast<const uint32_t*>(depth_only_fragment_shader_code.data()),
        depth_only_fragment_shader_code.size());
    if (depth_only_fragment_shader_ == VK_NULL_HANDLE) {
      REXGPU_ERROR(
          "VulkanPipelineCache: Failed to create the depth/stencil-only "
          "fragment shader for the fragment shader interlock render backend "
          "implementation");
      return false;
    }
  } else {
    std::vector<uint8_t> depth_float24_truncate_fragment_shader_code =
        shader_translator_->CreateDepthOnlyFragmentShader(
            SpirvShaderTranslator::Modification::DepthStencilMode::kFloat24Truncating);
    depth_float24_truncate_fragment_shader_ = ui::vulkan::util::CreateShaderModule(
        vulkan_device,
        reinterpret_cast<const uint32_t*>(depth_float24_truncate_fragment_shader_code.data()),
        depth_float24_truncate_fragment_shader_code.size());
    if (depth_float24_truncate_fragment_shader_ == VK_NULL_HANDLE) {
      REXGPU_ERROR(
          "VulkanPipelineCache: Failed to create the float24 truncating "
          "depth-only fragment shader");
      return false;
    }
    std::vector<uint8_t> depth_float24_round_fragment_shader_code =
        shader_translator_->CreateDepthOnlyFragmentShader(
            SpirvShaderTranslator::Modification::DepthStencilMode::kFloat24Rounding);
    depth_float24_round_fragment_shader_ = ui::vulkan::util::CreateShaderModule(
        vulkan_device,
        reinterpret_cast<const uint32_t*>(depth_float24_round_fragment_shader_code.data()),
        depth_float24_round_fragment_shader_code.size());
    if (depth_float24_round_fragment_shader_ == VK_NULL_HANDLE) {
      REXGPU_ERROR(
          "VulkanPipelineCache: Failed to create the float24 rounding "
          "depth-only fragment shader");
      return false;
    }
  }
  return true;
}

bool VulkanPipelineCache::Initialize() {
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();

  bool edram_fragment_shader_interlock =
      render_target_cache_.GetPath() == RenderTargetCache::Path::kPixelShaderInterlock;

  // wwe2k14 patch (Session 93, idea 7): extended dynamic state for the guest pipelines, decided once.
  {
    const bool setting = REXCVAR_GET(vulkan_extended_dynamic_state);
    const bool device_has = vulkan_device->extensions().ext_EXT_extended_dynamic_state &&
                            vulkan_device->properties().extendedDynamicState;
    const int32_t quirk_bits = REXCVAR_GET(gpu_quirks_eds_allowed);
    eds_active_ = setting && device_has && (quirk_bits & 1) && !edram_fragment_shader_interlock;
    REXGPU_INFO(
        "[eds] extended dynamic state: {} (setting {}, device ext {}, quirks bits {}, host render target path {})",
        eds_active_ ? "on" : "off", setting ? "on" : "off", device_has ? "yes" : "no", quirk_bits,
        edram_fragment_shader_interlock ? "no" : "yes");
  }

  // wwe2k14 patch (Session 93, the per-draw storage window): the command processor latched the mode before the pipeline cache.
  SpirvShaderTranslator::Features translator_features(vulkan_device);
  translator_features.shared_memory_window = command_processor_.SharedMemoryWindowOn();
  shader_translator_ = std::make_unique<SpirvShaderTranslator>(
      translator_features,
      render_target_cache_.msaa_2x_attachments_supported(),
      render_target_cache_.msaa_2x_no_attachments_supported(), edram_fragment_shader_interlock,
      render_target_cache_.draw_resolution_scale_x(),
      render_target_cache_.draw_resolution_scale_y());

  // wwe2k14 patch (Session 92, M6): the depth-only shaders go through the translator: with the translation store on they are made
  // when InitializeShaderStorage has opened it (a stored translation is served, the translator does not run for them).
  if (!(REXCVAR_GET(gpu_m6_store) && REXCVAR_GET(store_shaders)) && !EnsureInternalFragmentShaders()) {
    return false;
  }

  std::vector<uint32_t> placeholder_pixel_shader_spirv;
  std::string placeholder_pixel_shader_compile_error;
  if (command_processor_.CompileGlslToSpirv(
          VK_SHADER_STAGE_FRAGMENT_BIT, kPlaceholderPixelShaderSource,
          placeholder_pixel_shader_spirv, placeholder_pixel_shader_compile_error)) {
    placeholder_pixel_shader_ = ui::vulkan::util::CreateShaderModule(
        vulkan_device, placeholder_pixel_shader_spirv.data(),
        placeholder_pixel_shader_spirv.size() * sizeof(uint32_t));
  } else {
    REXGPU_WARN("VulkanPipelineCache: Failed to compile placeholder pixel shader: {}",
                placeholder_pixel_shader_compile_error);
  }
  if (placeholder_pixel_shader_ == VK_NULL_HANDLE) {
    REXGPU_WARN(
        "VulkanPipelineCache: Failed to create placeholder pixel shader - "
        "async placeholder hot-swap will be unavailable");
  }

  uint32_t logical_processor_count = rex::thread::logical_processor_count();
  if (!logical_processor_count) {
    logical_processor_count = 6;
  }
  creation_threads_busy_ = 0;
  startup_loading_ = false;
  creation_completion_event_ = rex::thread::Event::CreateManualResetEvent(true);
  assert_not_null(creation_completion_event_);
  creation_completion_set_event_ = false;
  creation_threads_shutdown_from_ = SIZE_MAX;
  if (REXCVAR_GET(vulkan_pipeline_creation_threads) != 0) {
    size_t creation_thread_count;
    if (REXCVAR_GET(vulkan_pipeline_creation_threads) < 0) {
      creation_thread_count = std::max(logical_processor_count * 3 / 4, uint32_t(1));
    } else {
      creation_thread_count = std::min(uint32_t(REXCVAR_GET(vulkan_pipeline_creation_threads)),
                                       logical_processor_count);
    }
    for (size_t i = 0; i < creation_thread_count; ++i) {
      std::unique_ptr<rex::thread::Thread> creation_thread =
          rex::thread::Thread::Create({}, [this, i]() { CreationThread(i); });
      assert_not_null(creation_thread);
      creation_thread->set_name("Vulkan Pipelines");
      creation_threads_.push_back(std::move(creation_thread));
    }
  }

  return true;
}

void VulkanPipelineCache::InitializeShaderStorage(const std::filesystem::path& cache_root,
                                                  uint32_t title_id, bool blocking) {
  ShutdownShaderStorage();
  {
    std::lock_guard<std::mutex> lock(creation_request_lock_);
    startup_loading_ = false;
  }

  auto shader_storage_root = cache_root / "shaders";
  auto shader_storage_shareable_root = shader_storage_root / "shareable";
  if (!std::filesystem::exists(shader_storage_shareable_root)) {
    if (!std::filesystem::create_directories(shader_storage_shareable_root)) {
      REXGPU_ERROR(
          "Failed to create the shareable shader storage directory, persistent "
          "shader storage will be disabled: {}",
          rex::path_to_utf8(shader_storage_shareable_root));
      EnsureInternalFragmentShaders();  // (Session 92, M6: Initialize left them for the store; the translator makes them)
      return;
    }
  }

  // wwe2k14 patch (Session 63): the on-disk VkPipelineCache (vulkan_disk_pipeline_cache; Android by default).
  disk_pipeline_cache::Open(command_processor_.GetVulkanDevice(),
                            shader_storage_shareable_root / fmt::format("{:08X}.vkpc", title_id));
  // wwe2k14 patch (Session 70): the precompile's optimised SPIR-V (before any module is made: the stored pipelines
  // below come from it too).
  spirv_opt_store::Open(shader_storage_shareable_root / fmt::format("{:08X}.spvopt", title_id));
  // wwe2k14 patch (Session 92, M6): the translation store (and the variant list the precompile translates), before anything is
  // translated - the depth-only shaders Initialize left for this point included.
  translation_store::Open(shader_storage_shareable_root / fmt::format("{:08X}.vk.xtc", title_id),
                          shader_storage_shareable_root / fmt::format("{:08X}.vk.xvar", title_id),
                          SpirvShaderTranslator::Modification::kVersion);
  if (!EnsureInternalFragmentShaders()) {
    REXGPU_ERROR("VulkanPipelineCache: the depth-only fragment shaders could not be made");
  }

  bool edram_fragment_shader_interlock =
      render_target_cache_.GetPath() == RenderTargetCache::Path::kPixelShaderInterlock;

  // Initialize the pipeline storage stream - read pipeline descriptions and
  // collect used shader modifications to translate.
  std::vector<PipelineStoredDescription> pipeline_stored_descriptions;
  // <Shader hash, modification bits>.
  std::set<std::pair<uint64_t, uint64_t>> shader_translations_needed;
  auto pipeline_storage_file_path =
      shader_storage_shareable_root /
      fmt::format("{:08X}.{}.vk.xpso", title_id, edram_fragment_shader_interlock ? "fsi" : "fbo");
  pipeline_storage_file_ = rex::filesystem::OpenFile(pipeline_storage_file_path, "a+b");
  if (!pipeline_storage_file_) {
    REXGPU_ERROR(
        "Failed to open the Vulkan pipeline description storage file for "
        "writing, persistent shader storage will be disabled: {}",
        rex::path_to_utf8(pipeline_storage_file_path));
    return;
  }
  pipeline_storage_file_flush_needed_ = false;
  // wwe2k14 patch (Session 73, phones): read from the start. "a+" leaves the read position at the end of the file on
  // Android (bionic's stdio, from OpenBSD, seeks to the end on opening for append; Windows' and glibc's start at 0), so
  // the header read failed and the storage was emptied at every boot - a phone never kept its shaders (Jarrod's logs:
  // "N stored already" of the shader list 7 / 121 / 23 / 17, ~820 stored native pipelines waiting for their shaders, the
  // entrance making them on first use).
  rex::filesystem::Seek(pipeline_storage_file_, 0, SEEK_SET);
  // 'XEPS'.
  const uint32_t pipeline_storage_magic = 0x53504558;
  const uint32_t pipeline_storage_magic_api = edram_fragment_shader_interlock ? 1u : 0u;
  const uint32_t pipeline_storage_version_swapped = rex::byte_swap(
      std::max(PipelineDescription::kVersion, SpirvShaderTranslator::Modification::kVersion));
  struct {
    uint32_t magic;
    uint32_t magic_api;
    uint32_t version_swapped;
  } pipeline_storage_file_header;
  if (fread(&pipeline_storage_file_header, sizeof(pipeline_storage_file_header), 1,
            pipeline_storage_file_) &&
      pipeline_storage_file_header.magic == pipeline_storage_magic &&
      pipeline_storage_file_header.magic_api == pipeline_storage_magic_api &&
      pipeline_storage_file_header.version_swapped == pipeline_storage_version_swapped) {
    rex::filesystem::Seek(pipeline_storage_file_, 0, SEEK_END);
    int64_t pipeline_storage_told_end = rex::filesystem::Tell(pipeline_storage_file_);
    size_t pipeline_storage_told_count =
        size_t(pipeline_storage_told_end >= int64_t(sizeof(pipeline_storage_file_header))
                   ? (uint64_t(pipeline_storage_told_end) - sizeof(pipeline_storage_file_header)) /
                         sizeof(PipelineStoredDescription)
                   : 0);
    if (pipeline_storage_told_count &&
        rex::filesystem::Seek(pipeline_storage_file_, int64_t(sizeof(pipeline_storage_file_header)),
                              SEEK_SET)) {
      pipeline_stored_descriptions.resize(pipeline_storage_told_count);
      pipeline_stored_descriptions.resize(
          fread(pipeline_stored_descriptions.data(), sizeof(PipelineStoredDescription),
                pipeline_storage_told_count, pipeline_storage_file_));
      size_t pipeline_storage_read_count = pipeline_stored_descriptions.size();
      size_t pipeline_storage_kept_count = 0;
      size_t pipeline_storage_skipped_unsupported_count = 0;
      size_t pipeline_storage_skipped_corrupted_count = 0;
      for (size_t i = 0; i < pipeline_storage_read_count; ++i) {
        PipelineStoredDescription pipeline_stored_description = pipeline_stored_descriptions[i];
        // Validate file integrity, stop and truncate the stream if data is
        // corrupted.
        if (XXH3_64bits(&pipeline_stored_description.description,
                        sizeof(pipeline_stored_description.description)) !=
            pipeline_stored_description.description_hash) {
          pipeline_storage_skipped_corrupted_count = pipeline_storage_read_count - i;
          pipeline_storage_read_count = i;
          break;
        }
        // wwe2k14 patch (Session 93, idea 7): with extended dynamic state on, a record from before it is collapsed to the new key
        // (the shipped list's records too): its pipeline is the one its look-alikes share. The records kept are as before otherwise.
        if (eds_active_ && !pipeline_stored_description.description.eds_dynamic) {
          CollapseForEds(pipeline_stored_description.description);
        }
        if (!ArePipelineRequirementsMet(pipeline_stored_description.description)) {
          ++pipeline_storage_skipped_unsupported_count;
          continue;
        }
        pipeline_stored_descriptions[pipeline_storage_kept_count++] = pipeline_stored_description;
        // Mark the shader modifications as needed for translation.
        shader_translations_needed.emplace(
            pipeline_stored_description.description.vertex_shader_hash,
            pipeline_stored_description.description.vertex_shader_modification);
        if (pipeline_stored_description.description.pixel_shader_hash) {
          shader_translations_needed.emplace(
              pipeline_stored_description.description.pixel_shader_hash,
              pipeline_stored_description.description.pixel_shader_modification);
        }
      }
      pipeline_stored_descriptions.resize(pipeline_storage_kept_count);
      if (pipeline_storage_skipped_unsupported_count || pipeline_storage_skipped_corrupted_count) {
        REXGPU_INFO(
            "VulkanPipelineCache: Pipeline storage filtering kept {} descriptions (skipped {} "
            "unsupported, {} corrupted)",
            pipeline_storage_kept_count, pipeline_storage_skipped_unsupported_count,
            pipeline_storage_skipped_corrupted_count);
      }
    }
  }

  // Initialize the Xenos shader storage stream.
  auto shader_storage_file_path =
      shader_storage_shareable_root / fmt::format("{:08X}.xsh", title_id);
  shader_storage_file_ = rex::filesystem::OpenFile(shader_storage_file_path, "a+b");
  if (!shader_storage_file_) {
    REXGPU_ERROR(
        "Failed to open the guest shader storage file for writing, persistent "
        "shader storage will be disabled: {}",
        rex::path_to_utf8(shader_storage_file_path));
    fclose(pipeline_storage_file_);
    pipeline_storage_file_ = nullptr;
    return;
  }
  ++shader_storage_index_;
  shader_storage_file_flush_needed_ = false;
  rex::filesystem::Seek(shader_storage_file_, 0, SEEK_SET);  // Session 73: from the start (see the pipeline storage)
  struct {
    uint32_t magic;
    uint32_t version_swapped;
  } shader_storage_file_header;
  // 'XESH'.
  const uint32_t shader_storage_magic = 0x48534558;
  if (fread(&shader_storage_file_header, sizeof(shader_storage_file_header), 1,
            shader_storage_file_) &&
      shader_storage_file_header.magic == shader_storage_magic &&
      rex::byte_swap(shader_storage_file_header.version_swapped) == ShaderStoredHeader::kVersion) {
    uint64_t shader_storage_valid_bytes = sizeof(shader_storage_file_header);
    // Load shaders written by previous runs until the end of the file or until
    // a corrupted one is detected.
    ShaderStoredHeader shader_header;
    std::vector<uint32_t> ucode_dwords;
    ucode_dwords.reserve(0xFFFF);
    while (true) {
      if (!fread(&shader_header, sizeof(shader_header), 1, shader_storage_file_)) {
        break;
      }
      size_t ucode_byte_count = shader_header.ucode_dword_count * sizeof(uint32_t);
      ucode_dwords.resize(shader_header.ucode_dword_count);
      if (shader_header.ucode_dword_count &&
          !fread(ucode_dwords.data(), ucode_byte_count, 1, shader_storage_file_)) {
        break;
      }
      uint64_t ucode_data_hash = XXH3_64bits(ucode_dwords.data(), ucode_byte_count);
      if (shader_header.ucode_data_hash != ucode_data_hash) {
        // Validation failed.
        break;
      }
      shader_storage_valid_bytes += sizeof(shader_header) + ucode_byte_count;
      VulkanShader* shader = LoadShader(shader_header.type, ucode_dwords.data(),
                                        shader_header.ucode_dword_count, ucode_data_hash);
      if (shader->ucode_storage_index() == shader_storage_index_) {
        // Appeared twice in this file for some reason - skip.
        continue;
      }
      // Loaded from the current storage - don't write again.
      shader->set_ucode_storage_index(shader_storage_index_);
    }
    rex::filesystem::TruncateStdioFile(shader_storage_file_, shader_storage_valid_bytes);
  } else {
    rex::filesystem::TruncateStdioFile(shader_storage_file_, 0);
    shader_storage_file_header.magic = shader_storage_magic;
    shader_storage_file_header.version_swapped = rex::byte_swap(ShaderStoredHeader::kVersion);
    fwrite(&shader_storage_file_header, sizeof(shader_storage_file_header), 1,
           shader_storage_file_);
  }

  // wwe2k14 patch (Session 80, vulkan_stored_pipelines_at_boot false): the records stay (the trim below keeps them),
  // nothing is translated or made for them now.
  const bool make_stored = REXCVAR_GET(vulkan_stored_pipelines_at_boot);
  if (!make_stored) {
    if (!pipeline_stored_descriptions.empty()) {
      REXGPU_INFO("VulkanPipelineCache: {} stored pipelines kept, not made at boot (vulkan_stored_pipelines_at_boot)",
                  pipeline_stored_descriptions.size());
    }
    shader_translations_needed.clear();
  }

  // wwe2k14 patch (Session 66, B2.1): the stored vertex modifications were made with the settings of the run that
  // stored them. Converted to the current gpu_native_vertex_endian_spec before translating and creating, so a shipped
  // or older storage still prewarms (a record whose converted description is already loaded is skipped below).
  if (make_stored) {
    size_t converted = 0;
    shader_translations_needed.clear();
    for (PipelineStoredDescription& stored : pipeline_stored_descriptions) {
      PipelineDescription& description = stored.description;
      auto vertex_it = shaders_.find(description.vertex_shader_hash);
      if (vertex_it != shaders_.end()) {
        bool changed = false;
        description.vertex_shader_modification =
            ConformStoredVertexModification(*vertex_it->second, description.vertex_shader_modification, &changed,
                                            /*native=*/false);
        converted += changed ? 1 : 0;
      }
      // wwe2k14 patch (Session 90, Adreno link): as the native renderer's KeyFromRecord (a pair with a pixel replacement: both
      // modifications carry the interpolators the replacement reads; nothing when no replacement is registered yet).
      if (description.pixel_shader_hash && command_processor_.NativeRendererEnabled()) {
        shader_swap::WidenForReplacement(description.pixel_shader_hash, description.vertex_shader_modification,
                                         description.pixel_shader_modification);
      }
      shader_translations_needed.emplace(description.vertex_shader_hash, description.vertex_shader_modification);
      if (description.pixel_shader_hash) {
        shader_translations_needed.emplace(description.pixel_shader_hash, description.pixel_shader_modification);
      }
    }
    if (converted) {
      REXGPU_INFO("VulkanPipelineCache: stored pipelines converted to the current vertex endian setting: {} of {}",
                  converted, pipeline_stored_descriptions.size());
    }
  }

  // Translate shader modifications needed for stored pipelines.
  for (const std::pair<uint64_t, uint64_t>& translation_needed : shader_translations_needed) {
    auto shader_it = shaders_.find(translation_needed.first);
    if (shader_it == shaders_.end()) {
      continue;
    }
    auto* shader = shader_it->second;
    if (!shader->is_ucode_analyzed() || shader->analysis_registered()) {
      PrepareFullAnalysis(*shader);  // (Session 87: a stored pipeline's emulated translation needs the real analysis)
    }
    bool translation_is_new = false;
    auto* translation = static_cast<VulkanShader::VulkanTranslation*>(
        shader->GetOrCreateTranslation(translation_needed.second, &translation_is_new));
    if (!translation->is_translated() &&
        !TranslateAnalyzedShader(*shader_translator_, *translation)) {
      if (translation_is_new) {
        shader->DestroyTranslation(translation_needed.second);
      }
    }
  }

  // Create the pipelines.
  std::vector<PipelineCreationArguments> pipeline_creations;
  pipeline_creations.reserve(pipeline_stored_descriptions.size());
  static const std::vector<PipelineStoredDescription> kNoStoredPipelines;
  for (const PipelineStoredDescription& pipeline_stored_description :
       make_stored ? pipeline_stored_descriptions : kNoStoredPipelines) {
    const PipelineDescription& pipeline_description = pipeline_stored_description.description;

    auto find_it = pipelines_.find(pipeline_description);
    if (find_it != pipelines_.end()) {
      continue;
    }
    auto& pipeline = *pipelines_.emplace(pipeline_description, Pipeline()).first;
    PipelineCreationArguments creation_arguments;
    if (!TryGetPipelineCreationArgumentsForDescription(pipeline_description, &pipeline,
                                                       creation_arguments)) {
      pipelines_.erase(pipeline_description);
      continue;
    }
    uint32_t bound_rts = 0;
    for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
      if (pipeline_description.render_targets[i].color_write_mask) {
        bound_rts |= uint32_t(1) << i;
      }
    }
    uint32_t shader_writes_color_targets =
        creation_arguments.pixel_shader
            ? creation_arguments.pixel_shader->shader().writes_color_targets()
            : 0;
    bool shader_writes_depth = creation_arguments.pixel_shader
                                   ? creation_arguments.pixel_shader->shader().writes_depth()
                                   : (pipeline_description.eds_dynamic ? pipeline_description.eds_depth_used != 0
                                                                  : pipeline_description.depth_write_enable != 0);
    creation_arguments.priority = pipeline_util::CalculatePipelinePriority(
        bound_rts, shader_writes_color_targets, shader_writes_depth);
    pipeline_creations.push_back(creation_arguments);
  }

  if (!pipeline_creations.empty()) {
    {
      std::lock_guard<std::mutex> lock(creation_request_lock_);
      startup_loading_ = true;
    }
    uint32_t logical_processor_count = rex::thread::logical_processor_count();
    if (!logical_processor_count) {
      // Pick some reasonable amount if couldn't determine the number of cores.
      logical_processor_count = 6;
    }
    size_t creation_thread_count;
    if (REXCVAR_GET(vulkan_pipeline_creation_threads) == 0) {
      creation_thread_count = 1;
    } else if (REXCVAR_GET(vulkan_pipeline_creation_threads) < 0) {
      creation_thread_count = std::max(logical_processor_count * 3 / 4, uint32_t(1));
    } else {
      creation_thread_count = std::min(uint32_t(REXCVAR_GET(vulkan_pipeline_creation_threads)),
                                       logical_processor_count);
    }
    if (!blocking) {
      // Vulkan currently doesn't support deferred pipeline handles at draw time,
      // so complete preload creation regardless of blocking mode.
      creation_thread_count = std::max(creation_thread_count, size_t(1));
    }
    creation_thread_count = std::min(creation_thread_count, pipeline_creations.size());

    std::atomic<size_t> next_creation_index(0);
    std::atomic<size_t> created_pipeline_count(0);
    auto creation_worker = [this, &pipeline_creations, &next_creation_index,
                            &created_pipeline_count]() {
      while (true) {
        size_t creation_index = next_creation_index.fetch_add(1);
        if (creation_index >= pipeline_creations.size()) {
          break;
        }
        if (EnsurePipelineCreated(pipeline_creations[creation_index])) {
          created_pipeline_count.fetch_add(1);
        }
      }
    };

    if (creation_thread_count <= 1) {
      creation_worker();
    } else {
      std::vector<std::unique_ptr<rex::thread::Thread>> creation_threads;
      creation_threads.reserve(creation_thread_count);
      for (size_t i = 0; i < creation_thread_count; ++i) {
        std::unique_ptr<rex::thread::Thread> creation_thread =
            rex::thread::Thread::Create({}, creation_worker);
        assert_not_null(creation_thread);
        creation_thread->set_name("Vulkan Pipelines");
        creation_threads.push_back(std::move(creation_thread));
      }
      for (const auto& creation_thread : creation_threads) {
        rex::thread::Wait(creation_thread.get(), false);
      }
    }

    REXGPU_INFO("Created {} graphics pipelines from Vulkan storage ({} requested)",
                created_pipeline_count.load(), pipeline_creations.size());
    {
      std::lock_guard<std::mutex> lock(creation_request_lock_);
      startup_loading_ = false;
    }
  }

  // If any pipeline descriptions were corrupted (or the whole file has excess
  // bytes in the end), truncate to the last valid pipeline description.
  rex::filesystem::TruncateStdioFile(
      pipeline_storage_file_,
      uint64_t(sizeof(pipeline_storage_file_header) +
               sizeof(PipelineStoredDescription) * pipeline_stored_descriptions.size()));
  if (pipeline_stored_descriptions.empty()) {
    rex::filesystem::TruncateStdioFile(pipeline_storage_file_, 0);
    pipeline_storage_file_header.magic = pipeline_storage_magic;
    pipeline_storage_file_header.magic_api = pipeline_storage_magic_api;
    pipeline_storage_file_header.version_swapped = pipeline_storage_version_swapped;
    fwrite(&pipeline_storage_file_header, sizeof(pipeline_storage_file_header), 1,
           pipeline_storage_file_);
  }

  shader_storage_cache_root_ = cache_root;
  shader_storage_title_id_ = title_id;

  // Start the storage writing thread.
  storage_write_flush_shaders_ = false;
  storage_write_flush_pipelines_ = false;
  storage_write_thread_shutdown_ = false;
  storage_write_thread_ = rex::thread::Thread::Create({}, [this]() { StorageWriteThread(); });
  assert_not_null(storage_write_thread_);
  storage_write_thread_->set_name("Vulkan Storage writer");
}

void VulkanPipelineCache::ShutdownShaderStorage() {
  disk_pipeline_cache::Close();  // wwe2k14 patch (Session 63)
  spirv_opt_store::Close();      // wwe2k14 patch (Session 70)
  translation_store::Close();    // wwe2k14 patch (Session 92, M6)
  if (storage_write_thread_) {
    {
      std::lock_guard<std::mutex> lock(storage_write_request_lock_);
      storage_write_thread_shutdown_ = true;
    }
    storage_write_request_cond_.notify_all();
    rex::thread::Wait(storage_write_thread_.get(), false);
    storage_write_thread_.reset();
  }
  storage_write_shader_queue_.clear();
  storage_write_pipeline_queue_.clear();

  if (pipeline_storage_file_) {
    fclose(pipeline_storage_file_);
    pipeline_storage_file_ = nullptr;
    pipeline_storage_file_flush_needed_ = false;
  }

  if (shader_storage_file_) {
    fclose(shader_storage_file_);
    shader_storage_file_ = nullptr;
    shader_storage_file_flush_needed_ = false;
  }

  shader_storage_cache_root_.clear();
  shader_storage_title_id_ = 0;
}

void VulkanPipelineCache::EndSubmission() {
  if (shader_storage_file_flush_needed_ || pipeline_storage_file_flush_needed_) {
    {
      std::lock_guard<std::mutex> lock(storage_write_request_lock_);
      if (shader_storage_file_flush_needed_) {
        storage_write_flush_shaders_ = true;
      }
      if (pipeline_storage_file_flush_needed_) {
        storage_write_flush_pipelines_ = true;
      }
    }
    storage_write_request_cond_.notify_one();
    shader_storage_file_flush_needed_ = false;
    pipeline_storage_file_flush_needed_ = false;
  }

  if (!creation_threads_.empty()) {
    bool startup_loading = false;
    {
      std::lock_guard<std::mutex> lock(creation_request_lock_);
      startup_loading = startup_loading_;
    }
    if (startup_loading) {
      creation_request_cond_.notify_one();
    } else if (REXCVAR_GET(vulkan_pipeline_no_submit_wait)) {
      // wwe2k14 patch (Session 6, H1): neither help nor wait. Draws whose
      // pipeline is still being created were skipped in IssueDraw, so the
      // commands of this submission never reference such a pipeline, and the
      // creation threads keep working across submissions and frames. Inputs of
      // queued creations (shader modules, pipeline layouts, render passes) live
      // until shutdown, which joins the creation threads first; the one other
      // destroyer, the render target cache clear, calls AwaitPipelineCreation.
    } else {
      AwaitPipelineCreation();
    }
  }

  ProcessDeferredPipelineDestructions(false);
}

void VulkanPipelineCache::AwaitPipelineCreation() {
  // wwe2k14 patch (Session 6, H1): moved out of EndSubmission unchanged, so the
  // render target cache clear can also wait when EndSubmission didn't.
  if (creation_threads_.empty()) {
    return;
  }
  // Help worker threads on the processor thread to reduce warm-up latency.
  CreateQueuedPipelinesOnProcessorThread();
  bool await_creation_completion_event;
  {
    std::lock_guard<std::mutex> lock(creation_request_lock_);
    // The queue is empty because of CreateQueuedPipelinesOnProcessorThread,
    // only check in-flight creation by worker threads.
    await_creation_completion_event = creation_threads_busy_ != 0;
    if (await_creation_completion_event) {
      creation_completion_event_->Reset();
      creation_completion_set_event_ = true;
    }
  }
  if (await_creation_completion_event) {
    creation_request_cond_.notify_one();
    rex::thread::Wait(creation_completion_event_.get(), false);
  }
}

void VulkanPipelineCache::Shutdown() {
  // Shut down creation threads before destroying any pipelines they may touch.
  if (!creation_threads_.empty()) {
    {
      std::lock_guard<std::mutex> lock(creation_request_lock_);
      creation_threads_shutdown_from_ = 0;
    }
    creation_request_cond_.notify_all();
    for (size_t i = 0; i < creation_threads_.size(); ++i) {
      rex::thread::Wait(creation_threads_[i].get(), false);
    }
    creation_threads_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(creation_request_lock_);
    while (!creation_queue_.empty()) {
      creation_queue_.pop();
    }
    creation_threads_busy_ = 0;
    startup_loading_ = false;
    creation_completion_set_event_ = false;
    creation_threads_shutdown_from_ = SIZE_MAX;
  }
  creation_completion_event_.reset();

  ShutdownShaderStorage();

  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  ProcessDeferredPipelineDestructions(true);

  // Destroy all pipelines.
  last_pipeline_ = nullptr;
  for (const auto& pipeline_pair : pipelines_) {
    VkPipeline pipeline = pipeline_pair.second.pipeline.load(std::memory_order_acquire);
    if (pipeline != VK_NULL_HANDLE) {
      dfn.vkDestroyPipeline(device, pipeline, nullptr);
    }
  }
  pipelines_.clear();

  // Destroy all internal shaders.
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device,
                                         depth_only_fragment_shader_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device,
                                         depth_float24_truncate_fragment_shader_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device,
                                         depth_float24_round_fragment_shader_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device,
                                         placeholder_pixel_shader_);
  if (tessellation_indexed_vertex_shader_ != VK_NULL_HANDLE) {
    dfn.vkDestroyShaderModule(device, tessellation_indexed_vertex_shader_, nullptr);
    tessellation_indexed_vertex_shader_ = VK_NULL_HANDLE;
  }
  if (tessellation_adaptive_vertex_shader_ != VK_NULL_HANDLE) {
    dfn.vkDestroyShaderModule(device, tessellation_adaptive_vertex_shader_, nullptr);
    tessellation_adaptive_vertex_shader_ = VK_NULL_HANDLE;
  }
  tessellation_indexed_vertex_shader_attempted_ = false;
  tessellation_adaptive_vertex_shader_attempted_ = false;
  for (const auto& tessellation_control_shader_pair : tessellation_control_shaders_) {
    if (tessellation_control_shader_pair.second != VK_NULL_HANDLE) {
      dfn.vkDestroyShaderModule(device, tessellation_control_shader_pair.second, nullptr);
    }
  }
  tessellation_control_shaders_.clear();
  for (const auto& geometry_shader_pair : geometry_shaders_) {
    if (geometry_shader_pair.second != VK_NULL_HANDLE) {
      dfn.vkDestroyShaderModule(device, geometry_shader_pair.second, nullptr);
    }
  }
  geometry_shaders_.clear();

  // Destroy all translated shaders.
  for (auto it : shaders_) {
    delete it.second;
  }
  shaders_.clear();
  shader_storage_index_ = 0;
  texture_binding_layout_map_.clear();
  texture_binding_layouts_.clear();

  // Shut down shader translation.
  shader_translator_.reset();
}

VulkanShader* VulkanPipelineCache::LoadShader(xenos::ShaderType shader_type,
                                              const uint32_t* host_address, uint32_t dword_count) {
  // Hash the input memory and lookup the shader.
  return LoadShader(shader_type, host_address, dword_count,
                    XXH3_64bits(host_address, dword_count * sizeof(uint32_t)));
}

VulkanShader* VulkanPipelineCache::LoadShader(xenos::ShaderType shader_type,
                                              const uint32_t* host_address, uint32_t dword_count,
                                              uint64_t data_hash) {
  auto it = shaders_.find(data_hash);
  if (it != shaders_.end()) {
    // Shader has been previously loaded.
    return it->second;
  }
  // Always create the shader and stash it away.
  // We need to track it even if it fails translation so we know not to try
  // again.
  VulkanShader* shader = new VulkanShader(command_processor_.GetVulkanDevice(), shader_type,
                                          data_hash, host_address, dword_count);
  shaders_.emplace(data_hash, shader);
  return shader;
}

SpirvShaderTranslator::Modification VulkanPipelineCache::GetCurrentVertexShaderModification(
    const Shader& shader, Shader::HostVertexShaderType host_vertex_shader_type,
    uint32_t interpolator_mask, bool ps_param_gen_used) const {
  assert_true(shader.type() == xenos::ShaderType::kVertex);
  assert_true(shader.is_ucode_analyzed());
  const auto& regs = KeyRegisters();
  const auto& device_properties = command_processor_.GetVulkanDevice()->properties();

  SpirvShaderTranslator::Modification modification(
      shader_translator_->GetDefaultVertexShaderModification(
          shader.GetDynamicAddressableRegisterCount(regs.Get<reg::SQ_PROGRAM_CNTL>().vs_num_reg),
          host_vertex_shader_type));

  modification.vertex.interpolator_mask = interpolator_mask;
  if (Shader::IsHostVertexShaderTypeDomain(host_vertex_shader_type)) {
    modification.vertex.tessellation_mode = uint32_t(regs.Get<reg::VGT_HOS_CNTL>().tess_mode);
  } else {
    modification.vertex.tessellation_mode = 0;
  }

  auto pa_cl_clip_cntl = regs.Get<reg::PA_CL_CLIP_CNTL>();
  uint32_t user_clip_planes = pa_cl_clip_cntl.clip_disable ? 0 : pa_cl_clip_cntl.ucp_ena;
  if (user_clip_planes) {
    if (pa_cl_clip_cntl.ucp_cull_only_ena && device_properties.shaderCullDistance) {
      modification.vertex.user_clip_plane_cull = 1;
    } else if (!device_properties.shaderClipDistance && device_properties.shaderCullDistance) {
      // Fallback if clip distances are unavailable.
      modification.vertex.user_clip_plane_cull = 1;
    } else if (!device_properties.shaderClipDistance) {
      // No supported clip/cull built-in for user clip planes.
      user_clip_planes = 0;
      modification.vertex.user_clip_plane_cull = 0;
    } else {
      modification.vertex.user_clip_plane_cull = 0;
    }
  } else {
    modification.vertex.user_clip_plane_cull = 0;
  }
  modification.vertex.user_clip_plane_count = rex::bit_count(user_clip_planes);
  modification.vertex.point_ps_ucp_mode = pa_cl_clip_cntl.ps_ucp_mode;

  if (host_vertex_shader_type == Shader::HostVertexShaderType::kPointListAsTriangleStrip) {
    modification.vertex.output_point_parameters = uint32_t(ps_param_gen_used);
  } else {
    modification.vertex.output_point_parameters =
        uint32_t((shader.writes_point_size_edge_flag_kill_vertex() & 0b001) &&
                 regs.Get<reg::VGT_DRAW_INITIATOR>().prim_type == xenos::PrimitiveType::kPointList);
  }
  modification.vertex.vertex_kill_and = uint32_t(
      device_properties.shaderCullDistance &&
      (shader.writes_point_size_edge_flag_kill_vertex() & 0b100) && !pa_cl_clip_cntl.vtx_kill_or);

  // wwe2k14 patch (Session 23, MERGE): vulkan_repeat_draw_merge (0 unless the setting and the shader list say).
  modification.vertex.instanced_float_constants_stride =
      host_vertex_shader_type == Shader::HostVertexShaderType::kVertex
          ? command_processor_.RepeatMergeStrideVec4(shader)
          : 0;
  // wwe2k14 patch (Session 24, OBJMV): gpu_dlss_objmv's character shaders capture their positions.
  // (Review fix: triangle lists / strips only - a rectangle or point list gets a geometry shader after the VS.)
  const xenos::PrimitiveType objmv_prim = regs.Get<reg::VGT_DRAW_INITIATOR>().prim_type;
  modification.vertex.xfb_position =
      host_vertex_shader_type == Shader::HostVertexShaderType::kVertex &&
      (objmv_prim == xenos::PrimitiveType::kTriangleList || objmv_prim == xenos::PrimitiveType::kTriangleStrip) &&
      command_processor_.ObjmvCaptureShader(shader);

  // wwe2k14 patch (Session 66, B2.1): the fetch constants' endian modes (0 / 0 with gpu_native_vertex_endian_spec off:
  // the modification, and so every stored record and key, is what it was without the field).
  const uint32_t endian_key = GetVertexFetchEndianKey(shader);
  modification.vertex.vertex_fetch_endian_spec = endian_key & 1;
  modification.vertex.vertex_fetch_endians = endian_key >> 1;

  return modification;
}

// wwe2k14 patch (Session 66, B2.1). Bit 0: the endian bits are valid (the cvar is on, and the shader has 1 to 8 vertex
// bindings); bits 1-16: the xenos::Endian of binding i's fetch constant (word 1 bits 0-1 - what the shader would read)
// at bits 2i+1. 0 if not specialised. The draw paths that reuse a recorded draw's pipeline compare this key.
void VulkanPipelineCache::RefreshNativeSpec() const {
  NativeSpecLatch& s = spec_latch_;
  s.native = command_processor_.NativeRendererEnabled();
  s.endian = REXCVAR_GET(gpu_native_vertex_endian_spec);
  s.ps_texture = REXCVAR_GET(gpu_native_ps_texture_spec);
  s.ps_implicit_lod = REXCVAR_GET(gpu_native_ps_implicit_lod);
  s.ps_epilogue = REXCVAR_GET(gpu_native_ps_epilogue_spec);
  s.vs_fetch_finite = REXCVAR_GET(gpu_native_vs_mul_fetch_finite);
  s.ps_mul = uint8_t(REXCVAR_GET(gpu_native_ps_mul_mode) & 3);
  s.vs_mul = uint8_t(REXCVAR_GET(gpu_native_vs_mul_mode) & 3);
  s.fma = uint8_t(REXCVAR_GET(gpu_native_fma) & 3);
  s.ps_wanted = s.ps_texture || s.ps_mul != 0 || s.ps_implicit_lod;
  s.valid = true;
}

uint32_t VulkanPipelineCache::GetVertexFetchEndianKey(const Shader& shader) const {
  if (!NativeSpec().endian) {
    return 0;
  }
  // (wwe2k14 patch, Session 66, keycost: the endian modes only change with a fetch constant write.)
  const VulkanShader::FetchKeyCache& cached = static_cast<const VulkanShader&>(shader).endian_key_cache_;
  const uint64_t generation = command_processor_.FetchConstantsGeneration();
  if (cached.generation == generation) {
    return cached.value;
  }
  uint32_t key = 0;
  const std::vector<Shader::VertexBinding>& bindings = shader.vertex_bindings();
  if (!bindings.empty() && bindings.size() <= 7) {
    uint32_t endians = 0;
    for (size_t i = 0; i < bindings.size(); ++i) {
      endians |= uint32_t(KeyRegisters().GetVertexFetch(bindings[i].fetch_constant).endian) << (2 * i);
    }
    key = 1 | (endians << 1);
  }
  VulkanShader::FetchKeyCache& cache = static_cast<const VulkanShader&>(shader).endian_key_cache_;
  cache.generation = generation;
  cache.value = key;
  return key;
}

SpirvShaderTranslator::Modification VulkanPipelineCache::GetCurrentPixelShaderModification(
    const Shader& shader, uint32_t interpolator_mask, uint32_t param_gen_pos,
    reg::RB_DEPTHCONTROL normalized_depth_control) const {
  assert_true(shader.type() == xenos::ShaderType::kPixel);
  assert_true(shader.is_ucode_analyzed());
  const auto& regs = KeyRegisters();

  SpirvShaderTranslator::Modification modification(
      shader_translator_->GetDefaultPixelShaderModification(
          shader.GetDynamicAddressableRegisterCount(regs.Get<reg::SQ_PROGRAM_CNTL>().ps_num_reg)));

  modification.pixel.interpolator_mask = interpolator_mask;
  modification.pixel.interpolators_centroid =
      interpolator_mask & ~xenos::GetInterpolatorSamplingPattern(
                              regs.Get<reg::RB_SURFACE_INFO>().msaa_samples,
                              regs.Get<reg::SQ_CONTEXT_MISC>().sc_sample_cntl,
                              regs.Get<reg::SQ_INTERPOLATOR_CNTL>().sampling_pattern);

  if (param_gen_pos < xenos::kMaxInterpolators) {
    modification.pixel.param_gen_enable = 1;
    modification.pixel.param_gen_interpolator = param_gen_pos;
    modification.pixel.param_gen_point =
        uint32_t(regs.Get<reg::VGT_DRAW_INITIATOR>().prim_type == xenos::PrimitiveType::kPointList);
  } else {
    modification.pixel.param_gen_enable = 0;
    modification.pixel.param_gen_interpolator = 0;
    modification.pixel.param_gen_point = 0;
  }

  if (render_target_cache_.GetPath() == RenderTargetCache::Path::kHostRenderTargets) {
    using DepthStencilMode = SpirvShaderTranslator::Modification::DepthStencilMode;
    if (render_target_cache_.depth_float24_convert_in_pixel_shader() &&
        normalized_depth_control.z_enable &&
        regs.Get<reg::RB_DEPTH_INFO>().depth_format == xenos::DepthRenderTargetFormat::kD24FS8) {
      // For host D32 depth, emulate guest D24FS8 float24 depth behavior.
      modification.pixel.depth_stencil_mode = render_target_cache_.depth_float24_round()
                                                  ? DepthStencilMode::kFloat24Rounding
                                                  : DepthStencilMode::kFloat24Truncating;
    } else {
      if (shader.implicit_early_z_write_allowed() &&
          (!shader.writes_color_target(0) ||
           !draw_util::DoesCoverageDependOnAlpha(regs.Get<reg::RB_COLORCONTROL>()))) {
        modification.pixel.depth_stencil_mode = DepthStencilMode::kEarlyHint;
      } else {
        modification.pixel.depth_stencil_mode = DepthStencilMode::kNoModifiers;
      }
    }
  }

  // wwe2k14 patch (Session 24, DLSS): gpu_dlss_lod_bias for the scene draws (0 unless the setting says).
  modification.pixel.dlss_lod_bias_quarters = command_processor_.DlssLodBiasQuarters();

  return modification;
}

// wwe2k14 patch (Session 92, M4b): the typed variants of the derivations above (the same logic, the register reads replaced by the
// record decoder's typed fields). Kept next to the originals so a change to one is seen at the other; the shadow in
// native\owned_front_draw.cpp compares their results per draw.
uint32_t VulkanPipelineCache::GetVertexFetchEndianKey(const Shader& shader, const TypedKeyState& state) const {
  if (!NativeSpec().endian) {
    return 0;
  }
  uint32_t key = 0;
  const std::vector<Shader::VertexBinding>& bindings = shader.vertex_bindings();
  if (!bindings.empty() && bindings.size() <= 7) {
    uint32_t endians = 0;
    for (size_t i = 0; i < bindings.size(); ++i) {
      // (a vertex fetch constant is 2 words: FETCH_00_0 + 2 * index, index 0-95)
      xenos::xe_gpu_vertex_fetch_t fetch = {};
      if (2 * size_t(bindings[i].fetch_constant) + 2 <= 192) {
        std::memcpy(&fetch, state.fetch + 2 * size_t(bindings[i].fetch_constant), sizeof(fetch));
      }
      endians |= uint32_t(fetch.endian) << (2 * i);
    }
    key = 1 | (endians << 1);
  }
  return key;
}

SpirvShaderTranslator::Modification VulkanPipelineCache::GetCurrentVertexShaderModification(
    const Shader& shader, Shader::HostVertexShaderType host_vertex_shader_type, uint32_t interpolator_mask,
    bool ps_param_gen_used, const TypedKeyState& state) const {
  const auto& device_properties = command_processor_.GetVulkanDevice()->properties();
  SpirvShaderTranslator::Modification modification(shader_translator_->GetDefaultVertexShaderModification(
      shader.GetDynamicAddressableRegisterCount(state.program_cntl.vs_num_reg), host_vertex_shader_type));
  modification.vertex.interpolator_mask = interpolator_mask;
  if (Shader::IsHostVertexShaderTypeDomain(host_vertex_shader_type)) {
    modification.vertex.tessellation_mode = uint32_t(state.hos_cntl.tess_mode);
  } else {
    modification.vertex.tessellation_mode = 0;
  }
  const reg::PA_CL_CLIP_CNTL pa_cl_clip_cntl = state.clip_cntl;
  uint32_t user_clip_planes = pa_cl_clip_cntl.clip_disable ? 0 : pa_cl_clip_cntl.ucp_ena;
  if (user_clip_planes) {
    if (pa_cl_clip_cntl.ucp_cull_only_ena && device_properties.shaderCullDistance) {
      modification.vertex.user_clip_plane_cull = 1;
    } else if (!device_properties.shaderClipDistance && device_properties.shaderCullDistance) {
      modification.vertex.user_clip_plane_cull = 1;
    } else if (!device_properties.shaderClipDistance) {
      user_clip_planes = 0;
      modification.vertex.user_clip_plane_cull = 0;
    } else {
      modification.vertex.user_clip_plane_cull = 0;
    }
  } else {
    modification.vertex.user_clip_plane_cull = 0;
  }
  modification.vertex.user_clip_plane_count = rex::bit_count(user_clip_planes);
  modification.vertex.point_ps_ucp_mode = pa_cl_clip_cntl.ps_ucp_mode;
  if (host_vertex_shader_type == Shader::HostVertexShaderType::kPointListAsTriangleStrip) {
    modification.vertex.output_point_parameters = uint32_t(ps_param_gen_used);
  } else {
    modification.vertex.output_point_parameters =
        uint32_t((shader.writes_point_size_edge_flag_kill_vertex() & 0b001) &&
                 state.draw_initiator.prim_type == xenos::PrimitiveType::kPointList);
  }
  modification.vertex.vertex_kill_and = uint32_t(device_properties.shaderCullDistance &&
                                                 (shader.writes_point_size_edge_flag_kill_vertex() & 0b100) &&
                                                 !pa_cl_clip_cntl.vtx_kill_or);
  modification.vertex.instanced_float_constants_stride =
      host_vertex_shader_type == Shader::HostVertexShaderType::kVertex ? command_processor_.RepeatMergeStrideVec4(shader) : 0;
  const xenos::PrimitiveType objmv_prim = state.draw_initiator.prim_type;
  modification.vertex.xfb_position =
      host_vertex_shader_type == Shader::HostVertexShaderType::kVertex &&
      (objmv_prim == xenos::PrimitiveType::kTriangleList || objmv_prim == xenos::PrimitiveType::kTriangleStrip) &&
      command_processor_.ObjmvCaptureShader(shader);
  const uint32_t endian_key = GetVertexFetchEndianKey(shader, state);
  modification.vertex.vertex_fetch_endian_spec = endian_key & 1;
  modification.vertex.vertex_fetch_endians = endian_key >> 1;
  return modification;
}

SpirvShaderTranslator::Modification VulkanPipelineCache::GetCurrentPixelShaderModification(
    const Shader& shader, uint32_t interpolator_mask, uint32_t param_gen_pos, reg::RB_DEPTHCONTROL normalized_depth_control,
    const TypedKeyState& state) const {
  SpirvShaderTranslator::Modification modification(shader_translator_->GetDefaultPixelShaderModification(
      shader.GetDynamicAddressableRegisterCount(state.program_cntl.ps_num_reg)));
  modification.pixel.interpolator_mask = interpolator_mask;
  modification.pixel.interpolators_centroid =
      interpolator_mask & ~xenos::GetInterpolatorSamplingPattern(state.surface_info.msaa_samples, state.context_misc.sc_sample_cntl,
                                                                 state.interpolator_cntl.sampling_pattern);
  if (param_gen_pos < xenos::kMaxInterpolators) {
    modification.pixel.param_gen_enable = 1;
    modification.pixel.param_gen_interpolator = param_gen_pos;
    modification.pixel.param_gen_point = uint32_t(state.draw_initiator.prim_type == xenos::PrimitiveType::kPointList);
  } else {
    modification.pixel.param_gen_enable = 0;
    modification.pixel.param_gen_interpolator = 0;
    modification.pixel.param_gen_point = 0;
  }
  if (render_target_cache_.GetPath() == RenderTargetCache::Path::kHostRenderTargets) {
    using DepthStencilMode = SpirvShaderTranslator::Modification::DepthStencilMode;
    if (render_target_cache_.depth_float24_convert_in_pixel_shader() && normalized_depth_control.z_enable &&
        state.depth_info.depth_format == xenos::DepthRenderTargetFormat::kD24FS8) {
      modification.pixel.depth_stencil_mode = render_target_cache_.depth_float24_round() ? DepthStencilMode::kFloat24Rounding
                                                                                          : DepthStencilMode::kFloat24Truncating;
    } else {
      if (shader.implicit_early_z_write_allowed() &&
          (!shader.writes_color_target(0) || !draw_util::DoesCoverageDependOnAlpha(state.color_control))) {
        modification.pixel.depth_stencil_mode = DepthStencilMode::kEarlyHint;
      } else {
        modification.pixel.depth_stencil_mode = DepthStencilMode::kNoModifiers;
      }
    }
  }
  modification.pixel.dlss_lod_bias_quarters = command_processor_.DlssLodBiasQuarters();
  return modification;
}

uint32_t VulkanPipelineCache::GetNativePixelSpecKey(const Shader* pixel_shader, const TypedKeyState& state) const {
  if (!pixel_shader) {
    return 0;
  }
  const NativeSpecLatch& spec = NativeSpec();
  if (!spec.ps_wanted || !spec.native) {
    return 0;
  }
  uint32_t key = 0;
  if (spec.ps_texture) {
    for (const Shader::TextureBinding& binding : pixel_shader->texture_bindings()) {
      if (binding.binding_index >= 4) {
        continue;
      }
      const uint32_t bit = UINT32_C(1) << binding.binding_index;
      if (!(key & bit)) {
        xenos::xe_gpu_texture_fetch_t fetch;
        std::memcpy(&fetch, state.fetch + 6 * size_t(binding.fetch_constant), sizeof(fetch));
        if (!texture_util::SwizzleSigns(fetch)) key |= bit;
      }
    }
  }
  key |= uint32_t(spec.ps_mul) << 4;
  key |= uint32_t(spec.ps_implicit_lod) << 6;
  return key;
}

// wwe2k14 patch (Session 87, stage 4, gpu_native_swap_skip_translation): see the header.
namespace {
std::mutex g_skip_mutex;
std::unordered_set<uint64_t> g_skip_dropped;  // vertex shaders whose registered analysis was dropped for the real one
std::unordered_set<uint64_t> g_skip_refused;  // vertex shaders that must never take a registered analysis again
}  // namespace

void VulkanPipelineCache::AnalyzeShaderUcode(Shader& shader) {
  if (!shader.is_ucode_analyzed() && shader.type() == xenos::ShaderType::kVertex && SwapSkipOn()) {
    const uint64_t hash = shader.ucode_data_hash();
    SpirvShader* spirv = dynamic_cast<SpirvShader*>(&shader);
    bool refused_before;
    {
      std::lock_guard<std::mutex> lock(g_skip_mutex);
      refused_before = g_skip_refused.count(hash) != 0;
    }
    Shader::RegisteredAnalysis analysis;
    if (spirv && !refused_before && shader_swap::VertexTakes(hash) && shader_swap::DescribeVertex(hash, analysis)) {
      // Its textures and samplers are the replacement translation's own (SetReplacementBindings): the draws that use that
      // translation bind them; the emulated translation's come from the translator when a draw needs it.
      shader.ApplyRegisteredAnalysis(analysis);
      spirv->SetReplacementBindings(analysis);
      g_skip.analysis_v.fetch_add(1);
      REXGPU_INFO("[native][swap_skip] vertex shader {:016X}: analysis from the registration ({} float constants{}, {} "
                  "interpolators, {} textures, {} samplers); {}",
                  hash, shader.constant_register_map().float_count, analysis.float_dynamic_addressing ? ", indexed" : "",
                  std::popcount(analysis.writes_interpolators), analysis.textures.size(), analysis.samplers.size(),
                  SwapSkipTotals());
      if (REXCVAR_GET(gpu_native_swap_skip_verify)) {
        VerifyRegisteredAnalysis(shader, analysis);
      }
      return;
    }
  } else if (!shader.is_ucode_analyzed() && shader.type() == xenos::ShaderType::kPixel && SwapSkipOnPixel()) {
    // The shader's draws use the replacement's bindings (SpirvShader::SetReplacementBindings) until an emulated translation is
    // needed (PrepareFullAnalysis), which puts the shader back to the translator's lists for good.
    const uint64_t hash = shader.ucode_data_hash();
    SpirvShader* spirv = dynamic_cast<SpirvShader*>(&shader);
    Shader::RegisteredAnalysis analysis;
    if (spirv && REXCVAR_GET(gpu_native_shader_swap) && shader_swap::Has(hash) && !shader_swap::Skipped(hash) &&
        shader_swap::DescribePixel(hash, analysis)) {
      shader.ApplyRegisteredAnalysis(analysis);
      spirv->SetReplacementBindings(analysis);
      g_skip.analysis_p.fetch_add(1);
      REXGPU_INFO("[native][swap_skip] pixel shader {:016X}: analysis and bindings from the registration ({} textures, {} samplers, "
                  "{} float constants, kills {}); {}",
                  hash, analysis.textures.size(), analysis.samplers.size(), shader.constant_register_map().float_count,
                  analysis.kills_pixels ? 1 : 0, SwapSkipTotals());
      if (REXCVAR_GET(gpu_native_swap_skip_verify)) {
        VerifyRegisteredAnalysis(shader, analysis);
      }
      return;
    }
  }
  shader.AnalyzeUcode(ucode_disasm_buffer_);
}

void VulkanPipelineCache::PrepareFullAnalysis(Shader& shader) {
  if (shader.analysis_registered()) {
    const bool vertex = shader.type() == xenos::ShaderType::kVertex;
    const uint64_t hash = shader.ucode_data_hash();
    // What the registration said, to compare with the real analysis below.
    const Shader::ConstantRegisterMap derived_map = shader.constant_register_map();
    const uint32_t derived_interpolators = shader.writes_interpolators();
    const size_t derived_streams = shader.vertex_bindings().size();
    const uint32_t derived_stream_fc = derived_streams ? shader.vertex_bindings()[0].fetch_constant : 0;
    const uint32_t derived_stream_stride = derived_streams ? shader.vertex_bindings()[0].stride_words : 0;
    shader_swap::NoteTranslationChange();  // (Session 93: the shader's binding lists change; cached typed draw preparations are stale)
    shader.DropRegisteredAnalysis();
    if (SpirvShader* spirv = dynamic_cast<SpirvShader*>(&shader)) {
      spirv->DropRegisteredBindings();
    }
    // (The texture / endian keys cached per fetch constants generation were computed from the registered analysis' bindings.)
    if (VulkanShader* vulkan_shader = dynamic_cast<VulkanShader*>(&shader)) {
      vulkan_shader->texture_mask_cache_.generation = 0;
      vulkan_shader->endian_key_cache_.generation = 0;
    }
    (vertex ? g_skip.forced_v : g_skip.forced_p).fetch_add(1);
    shader.AnalyzeUcode(ucode_disasm_buffer_);
    const Shader::ConstantRegisterMap& real_map = shader.constant_register_map();
    bool changed = real_map.float_count != derived_map.float_count ||
                   real_map.float_dynamic_addressing != derived_map.float_dynamic_addressing ||
                   shader.writes_interpolators() != derived_interpolators ||
                   shader.vertex_bindings().size() != derived_streams;
    for (int i = 0; i < 4 && !changed; ++i) changed = real_map.float_bitmap[i] != derived_map.float_bitmap[i];
    if (!changed && derived_streams) {
      changed = shader.vertex_bindings()[0].fetch_constant != derived_stream_fc ||
                shader.vertex_bindings()[0].stride_words != derived_stream_stride;
    }
    if (vertex) {
      std::lock_guard<std::mutex> lock(g_skip_mutex);
      g_skip_dropped.insert(hash);
      if (changed) g_skip_refused.insert(hash);
    }
    if (changed) {
      g_skip.guard_bad_v.fetch_add(1);
      REXGPU_ERROR("[native][swap_skip] {} shader {:016X}: the real analysis differs from the registration's (constants, "
                   "stream or interpolators) - the skip is refused for this shader from now on; {}",
                   vertex ? "vertex" : "pixel", hash, SwapSkipTotals());
    } else {
      REXGPU_INFO("[native][swap_skip] {} shader {:016X}: the real analysis was needed (refusal fallback or an emulated "
                  "translation) and equals the registration's; {}",
                  vertex ? "vertex" : "pixel", hash, SwapSkipTotals());
    }
    return;
  }
  shader.AnalyzeUcode(ucode_disasm_buffer_);
}

void VulkanPipelineCache::VerifyRegisteredAnalysis(Shader& shader, const Shader::RegisteredAnalysis& analysis) {
  (void)analysis;
  const bool vertex = shader.type() == xenos::ShaderType::kVertex;
  const uint64_t hash = shader.ucode_data_hash();
  SpirvShader* derived_spirv = dynamic_cast<SpirvShader*>(&shader);
  if (!derived_spirv || !shader_translator_) {
    return;
  }
  // The scratch shader: the same microcode (already in host byte order), the real analysis, and one emulated translation
  // for the bindings a translation sets up.
  SpirvShader scratch(shader.type(), hash, shader.ucode_dwords(), shader.ucode_dword_count(), std::endian::native);
  scratch.AnalyzeUcode(ucode_disasm_buffer_);
  const uint64_t modification =
      vertex ? shader_translator_->GetDefaultVertexShaderModification(scratch.GetDynamicAddressableRegisterCount(0),
                                                                      Shader::HostVertexShaderType::kVertex)
             : shader_translator_->GetDefaultPixelShaderModification(scratch.GetDynamicAddressableRegisterCount(0));
  Shader::Translation* translation = scratch.GetOrCreateTranslation(modification);
  const bool translated = translation && shader_translator_->TranslateAnalyzedShader(*translation);
  uint32_t bad = 0;
  std::string lines, benign, names;  // names: the differing fields, in the header line
  const auto diff = [&](const std::string& field, uint64_t derived, uint64_t real) {
    if (derived == real) return;
    ++bad;
    names += (names.empty() ? "" : "; ") + field;
    lines += fmt::format("\n    {}: registration {:#x}, real {:#x}", field, derived, real);
  };
  const Shader::ConstantRegisterMap& dm = shader.constant_register_map();
  const Shader::ConstantRegisterMap& rm = scratch.constant_register_map();
  for (int i = 0; i < 4; ++i) {
    diff(fmt::format("float_bitmap[{}]", i), dm.float_bitmap[i], rm.float_bitmap[i]);
  }
  diff("float_count", dm.float_count, rm.float_count);
  diff("float_dynamic_addressing", dm.float_dynamic_addressing, rm.float_dynamic_addressing);
  // The bool / loop bitmaps: the replacement's are read from the access chains on its block (whole words), so the guest's bits must
  // be within ours (a superset is what the analysis promises its readers: half2.cpp:190,193 and half2_log.cpp:709,712, a
  // diagnostic comparison of library-draw registers, are the only readers - more bits only compare more registers; the constants
  // themselves are uploaded wholesale). label_addresses is read only by the translator (spirv_translator.cpp:254,264,674,726,
  // 1369,1400; translator.cpp:782), which a skipped translation never runs: benign.
  const auto superset = [&](const std::string& field, uint32_t derived, uint32_t real) {
    if ((real & ~derived) != 0) diff(field + " (a guest bit the replacement does not read)", derived, real);
  };
  superset("loop_bitmap", dm.loop_bitmap, rm.loop_bitmap);
  for (int i = 0; i < 8; ++i) {
    superset(fmt::format("bool_bitmap[{}]", i), dm.bool_bitmap[i], rm.bool_bitmap[i]);
  }
  if (shader.label_addresses().size() != scratch.label_addresses().size()) {
    benign += fmt::format("\n    (ignored, benign) label_addresses: registration {}, real {}", shader.label_addresses().size(),
                          scratch.label_addresses().size());
  }
  for (int i = 0; i < 3; ++i) {
    diff(fmt::format("vertex_fetch_bitmap[{}]", i), dm.vertex_fetch_bitmap[i], rm.vertex_fetch_bitmap[i]);
  }
  diff("writes_interpolators", shader.writes_interpolators(), scratch.writes_interpolators());
  diff("writes_point_size_edge_flag_kill_vertex", shader.writes_point_size_edge_flag_kill_vertex(),
       scratch.writes_point_size_edge_flag_kill_vertex());
  diff("writes_color_targets", shader.writes_color_targets(), scratch.writes_color_targets());
  diff("writes_depth", shader.writes_depth(), scratch.writes_depth());
  diff("kills_pixels", shader.kills_pixels(), scratch.kills_pixels());
  diff("memexport_eM_written", shader.memexport_eM_written(), scratch.memexport_eM_written());
  diff("uses_register_dynamic_addressing", shader.uses_register_dynamic_addressing(),
       scratch.uses_register_dynamic_addressing());
  // A pixel shader's interpolator count is the replacement's own inputs by design (docs/native/d3d/SKIP_TRANSLATION_PIXEL.md
  // section 3), not the guest's temp-register bound.
  if (!vertex) {
    const uint32_t mine = shader.register_static_address_bound(), guest = scratch.register_static_address_bound();
    if (mine != guest) benign += fmt::format("\n    (by design) register_static_address_bound: replacement inputs {}, guest {}", mine, guest);
  }
  if (vertex) diff("vertex_bindings", shader.vertex_bindings().size(), scratch.vertex_bindings().size());
  for (size_t i = 0; i < std::min(shader.vertex_bindings().size(), scratch.vertex_bindings().size()); ++i) {
    diff("vertex_binding.fetch_constant", shader.vertex_bindings()[i].fetch_constant,
         scratch.vertex_bindings()[i].fetch_constant);
    diff("vertex_binding.stride_words", shader.vertex_bindings()[i].stride_words,
         scratch.vertex_bindings()[i].stride_words);
  }
  // The textures and samplers are the replacement translation's own (its SPIR-V's), not the translator's list: what must hold is
  // that the replacement samples every texture and sampler fetch constant the guest shader does (the guest's set within ours),
  // and the guest's sampler filter overrides - which the replacement's samplers do not carry (they follow the fetch constant)
  // - are listed.
  if (translated) {
    uint32_t guest_textures = 0, guest_samplers = 0, own_textures = 0, own_samplers = 0;
    for (const auto& t : scratch.GetTextureBindingsAfterTranslation()) guest_textures |= UINT32_C(1) << t.fetch_constant;
    for (const auto& t : derived_spirv->TextureBindingsFor(true)) own_textures |= UINT32_C(1) << t.fetch_constant;
    for (const auto& t : scratch.GetSamplerBindingsAfterTranslation()) guest_samplers |= UINT32_C(1) << t.fetch_constant;
    for (const auto& t : derived_spirv->SamplerBindingsFor(true)) own_samplers |= UINT32_C(1) << t.fetch_constant;
    if ((guest_textures & ~own_textures) != 0) diff("texture fetch constants the guest samples and the replacement does not", own_textures, guest_textures);
    if ((guest_samplers & ~own_samplers) != 0) diff("sampler fetch constants the guest uses and the replacement does not", own_samplers, guest_samplers);
    // The guest's sampler filter overrides (its tfetch instructions) are carried by the registration: the replacement's sampler of
    // the same fetch constant must have the same filters.
    for (const auto& rs : scratch.GetSamplerBindingsAfterTranslation()) {
      bool found = false;
      for (const auto& ds : derived_spirv->SamplerBindingsFor(true)) {
        if (ds.fetch_constant != rs.fetch_constant) continue;
        found = true;
        diff(fmt::format("sampler filters of fetch constant {}", uint32_t(rs.fetch_constant)),
             uint32_t(ds.mag_filter) | uint32_t(ds.min_filter) << 2 | uint32_t(ds.mip_filter) << 4 | uint32_t(ds.aniso_filter) << 6,
             uint32_t(rs.mag_filter) | uint32_t(rs.min_filter) << 2 | uint32_t(rs.mip_filter) << 4 | uint32_t(rs.aniso_filter) << 6);
        break;
      }
      (void)found;
    }
  } else {
    lines += "\n    (the scratch translation failed: bindings not compared)";
  }
  g_skip.verify_shaders.fetch_add(1);
  if (bad) {
    g_skip.verify_bad_shaders.fetch_add(1);
    g_skip.verify_bad_fields.fetch_add(bad);
  }
  REXGPU_INFO("[native][swap_skip][verify] {} shader {:016X}: {}{}\n  {}", vertex ? "vertex" : "pixel", hash,
              bad ? fmt::format("{} fields differ ({})", bad, names) : std::string("all fields match"), lines + benign, SwapSkipTotals());
}

bool VulkanPipelineCache::EnsureShadersTranslated(VulkanShader::VulkanTranslation* vertex_shader,
                                                  VulkanShader::VulkanTranslation* pixel_shader, uint32_t defer_mask) {
  // Edge flags are not supported yet (because polygon primitives are not).
  assert_true(KeyRegisters().Get<reg::SQ_PROGRAM_CNTL>().vs_export_mode !=
                  xenos::VertexShaderExportMode::kPosition2VectorsEdge &&
              KeyRegisters().Get<reg::SQ_PROGRAM_CNTL>().vs_export_mode !=
                  xenos::VertexShaderExportMode::kPosition2VectorsEdgeKill);
  assert_false(KeyRegisters().Get<reg::SQ_PROGRAM_CNTL>().gen_index_vtx);
  // wwe2k14 patch (Session 87, stage 4, gpu_native_swap_skip_translation): a shader with a registered analysis is
  // translated by the translator only when the translation is not the replacement itself: NATIVE's vertex translation with
  // our vertex shader (TranslateAnalyzedShader makes it without), the emulated translations when the caller asks for them
  // (they need the real analysis); with defer_mask the emulated ones are left undone.
  bool vertex_deferred = false;
  if (!vertex_shader->is_translated()) {
    VulkanShader& shader = static_cast<VulkanShader&>(vertex_shader->shader());
    if (shader.analysis_registered() && (defer_mask & 1) && !vertex_shader->native_ext() && SwapSkipOn()) {
      vertex_deferred = true;
      g_skip.deferred_v.fetch_add(1);
    } else {
      if (!UsesReplacementBindings(&shader, vertex_shader->native_ext())) {
        PrepareFullAnalysis(shader);
      }
      if (!TranslateAnalyzedShader(*shader_translator_, *vertex_shader)) {
        REXGPU_ERROR("Failed to translate the vertex shader!");
        return false;
      }
    }
  }
  if (!vertex_deferred && !vertex_shader->is_valid()) {
    // Translation attempted previously, but not valid.
    return false;
  }
  if (pixel_shader != nullptr) {
    if (!pixel_shader->is_translated()) {
      VulkanShader& shader = static_cast<VulkanShader&>(pixel_shader->shader());
      if (shader.analysis_registered() && (defer_mask & 2) && !pixel_shader->native_ext() && SwapSkipOnPixel()) {
        g_skip.deferred_p.fetch_add(1);
        return true;
      }
      PrepareFullAnalysis(shader);
      if (!TranslateAnalyzedShader(*shader_translator_, *pixel_shader)) {
        REXGPU_ERROR("Failed to translate the pixel shader!");
        return false;
      }
    }
    if (!pixel_shader->is_valid()) {
      // Translation attempted previously, but not valid.
      return false;
    }
  }
  return true;
}

// wwe2k14 patch (Session 66, PS spec): the pixel shader specialisation bits NATIVE's translation of this shader gets for
// the current draw, as one key (bits 0-3: Modification::pixel::tex_unsigned_mask); 0 with every setting off. Only
// NATIVE's own translation carries them (ApplyNativePixelSpec; the emulated path's modification is not touched). The
// lean / repeat / crowd paths compare the key with the one their base draw's pixel shader module was made with.
uint32_t VulkanPipelineCache::GetNativePixelSpecKey(const Shader* pixel_shader, bool census) const {
  if (!pixel_shader) {
    return 0;
  }
  const NativeSpecLatch& spec = NativeSpec();
  if (!census && (!spec.ps_wanted || !spec.native)) {
    return 0;
  }
  uint32_t key = 0;
  if (census || spec.ps_texture) {
    // The fetch constant's swizzled signs byte is what texture_swizzled_signs holds for a valid texture binding (the
    // binding is derived from the same fetch constant), and 0 for an invalid one: a byte of 0 here means 0 there.
    // (wwe2k14 patch, Session 66, keycost: computed once per fetch constant generation and shader.)
    const VulkanShader::FetchKeyCache& cached = static_cast<const VulkanShader*>(pixel_shader)->texture_mask_cache_;
    const uint64_t generation = command_processor_.FetchConstantsGeneration();
    if (cached.generation == generation) {
      key = cached.value;
    } else {
      for (const Shader::TextureBinding& binding : pixel_shader->texture_bindings()) {
        if (binding.binding_index >= 4) {
          continue;
        }
        const uint32_t bit = UINT32_C(1) << binding.binding_index;
        if (!(key & bit) &&
            !texture_util::SwizzleSigns(KeyRegisters().GetTextureFetch(binding.fetch_constant))) {
          key |= bit;
        }
      }
      VulkanShader::FetchKeyCache& cache = static_cast<const VulkanShader*>(pixel_shader)->texture_mask_cache_;
      cache.generation = generation;
      cache.value = key;
    }
  }
  // Bits 4-5: gpu_native_ps_mul_mode (the translation applies it to NATIVE pixel shaders only, which these are).
  key |= uint32_t(spec.ps_mul) << 4;
  // Bit 6: gpu_native_ps_implicit_lod.
  key |= uint32_t(spec.ps_implicit_lod) << 6;
  return key;
}

bool VulkanPipelineCache::NativePixelSpecWanted() const { return NativeSpec().ps_wanted; }

// Whether a draw reusing a recorded draw's NATIVE pixel shader needs another one: its key now is not the recorded one.
bool VulkanPipelineCache::PixelSpecKeyChanged(const Shader* pixel_shader, uint32_t recorded_key) const {
  return (recorded_key || NativePixelSpecWanted()) && GetNativePixelSpecKey(pixel_shader) != recorded_key;
}

bool VulkanPipelineCache::PixelSpecKeyChanged(const Shader* pixel_shader, uint32_t recorded_key, const TypedKeyState& state) const {
  return (recorded_key || NativePixelSpecWanted()) && GetNativePixelSpecKey(pixel_shader, state) != recorded_key;
}

uint32_t VulkanPipelineCache::GetNativeVertexMulKey() const {
  const NativeSpecLatch& spec = NativeSpec();
  if (!spec.vs_mul || !spec.native) {
    return 0;
  }
  return spec.vs_mul | (spec.vs_fetch_finite ? 4u : 0u);
}

bool VulkanPipelineCache::VertexMulKeyChanged(uint32_t recorded_key) const {
  return (recorded_key || NativeSpec().vs_mul != 0) && GetNativeVertexMulKey() != recorded_key;
}

void VulkanPipelineCache::ApplyNativeVertexSpec(SpirvShaderTranslator::Modification& modification, uint32_t key) {
  modification.vertex.native_mul_mode = key & 3;
  modification.vertex.native_mul_fetch_finite = (key >> 2) & 1;
}

uint64_t VulkanPipelineCache::ConformStoredVertexModification(VulkanShader& vertex_shader, uint64_t modification,
                                                              bool* converted, bool native) {
  SpirvShaderTranslator::Modification out(modification);
  AnalyzeShaderUcode(vertex_shader);
  // The multiply rule bits: NATIVE's kVertex translations, as the current setting says (no prediction needed).
  out.vertex.native_mul_mode = 0;
  out.vertex.native_mul_fetch_finite = 0;
  if (native && out.vertex.host_vertex_shader_type == Shader::HostVertexShaderType::kVertex) {
    const uint32_t mode = uint32_t(REXCVAR_GET(gpu_native_vs_mul_mode)) & 3;
    if (mode) {
      ApplyNativeVertexSpec(out, mode | (REXCVAR_GET(gpu_native_vs_mul_fetch_finite) ? 4u : 0u));
    }
  }
  if (!REXCVAR_GET(gpu_native_vertex_endian_spec)) {
    out.vertex.vertex_fetch_endian_spec = 0;
    out.vertex.vertex_fetch_endians = 0;
  } else if (!out.vertex.vertex_fetch_endian_spec) {
    const size_t bindings = vertex_shader.vertex_bindings().size();
    if (bindings >= 1 && bindings <= 7) {
      uint32_t endians = 0;
      for (size_t i = 0; i < bindings; ++i) {
        endians |= uint32_t(xenos::Endian::k8in32) << (2 * i);
      }
      out.vertex.vertex_fetch_endian_spec = 1;
      out.vertex.vertex_fetch_endians = endians;
    }
  }
  if (converted && out.value != modification) {
    *converted = true;
  }
  return out.value;
}

// The typical draw: no alpha test, alpha to coverage, exponent bias or gamma (the census, gpu_native_ps_census, says
// how many shaders see both states).
uint32_t VulkanPipelineCache::PredictNativePixelExt() const {
  return (REXCVAR_GET(gpu_native_ps_epilogue_spec) ? uint32_t(SpirvShaderTranslator::kNativeExtEpilogueIdle) : 0u) |
         ((REXCVAR_GET(gpu_native_fma) & 1) ? uint32_t(SpirvShaderTranslator::kNativeExtFma) : 0u);
}

uint32_t VulkanPipelineCache::PredictNativeVertexExt(uint64_t vertex_modification, uint64_t vertex_shader_hash) const {
  const bool plain =
      SpirvShaderTranslator::Modification(vertex_modification).vertex.host_vertex_shader_type ==
      Shader::HostVertexShaderType::kVertex;
  return ((plain && (REXCVAR_GET(gpu_native_fma) & 2)) ? uint32_t(SpirvShaderTranslator::kNativeExtFma) : 0u) |
         // wwe2k14 patch (Session 87, stage 4, gpu_native_vs_swap): our own vertex shader for the plain kind.
         ((plain && vertex_shader_hash && shader_swap::VertexTakes(vertex_shader_hash))
              ? uint32_t(SpirvShaderTranslator::kNativeExtVsSwap)
              : 0u);
}

bool VulkanPipelineCache::NativeExtStale(uint32_t recorded_pixel_ext, uint32_t recorded_vertex_ext,
                                          const Shader* pixel_shader, const Shader* vertex_shader) const {
  // Only a setting that was turned off while the base's module still has its specialisation: the state the epilogue
  // specialisation depends on is held equal to the base's by the callers (render state registers), so a missing bit
  // for a setting just turned on is still a correct module, refreshed at the next full draw.
  const NativeSpecLatch& spec = NativeSpec();
  if ((recorded_pixel_ext & SpirvShaderTranslator::kNativeExtEpilogueIdle) && !spec.ps_epilogue) {
    return true;
  }
  // wwe2k14 patch (Session 72): the shader swap turned off (turned on, the next full draw picks it up).
  if ((recorded_pixel_ext & SpirvShaderTranslator::kNativeExtSwap) && !REXCVAR_GET(gpu_native_shader_swap)) {
    return true;
  }
  // wwe2k14 patch (Session 80): one replacement switched off or on (gpu_native_shader_swap_skip, in-run A/Bs).
  if (pixel_shader && shader_swap::SwapStale(pixel_shader->ucode_data_hash(), recorded_pixel_ext)) {
    return true;
  }
  // wwe2k14 patch (Session 87, stage 4): our own vertex shader switched on / off / skipped since the draw was recorded.
  if (vertex_shader && shader_swap::VertexSwapStale(vertex_shader->ucode_data_hash(), recorded_vertex_ext)) {
    return true;
  }
  // FMA: a function of the settings only (and of the native renderer being on, as the ext words are set only for its
  // draws).
  const bool native = spec.native;
  const bool fma_pixel = native && pixel_shader && (spec.fma & 1);
  const bool fma_vertex = native && (spec.fma & 2);
  if (((recorded_pixel_ext & SpirvShaderTranslator::kNativeExtFma) != 0) != fma_pixel) {
    return true;
  }
  if (((recorded_vertex_ext & SpirvShaderTranslator::kNativeExtFma) != 0) != fma_vertex) {
    return true;
  }
  return false;
}

uint64_t VulkanPipelineCache::ConformStoredNativePixelModification(VulkanShader& pixel_shader, uint64_t modification,
                                                                   bool* converted, bool mask_known) {
  SpirvShaderTranslator::Modification out(modification);
  AnalyzeShaderUcode(pixel_shader);
  if (!REXCVAR_GET(gpu_native_ps_texture_spec)) {
    out.pixel.tex_unsigned_mask = 0;
  } else if (!out.pixel.tex_unsigned_mask && !mask_known) {  // (Session 69 review: a known 0 is the draw's own)
    size_t bindings = 0;
    for (const Shader::TextureBinding& binding : pixel_shader.texture_bindings()) {
      bindings = std::max(bindings, binding.binding_index + 1);
    }
    out.pixel.tex_unsigned_mask = (UINT32_C(1) << std::min<size_t>(bindings, 4)) - 1;
  }
  out.pixel.native_mul_mode = REXCVAR_GET(gpu_native_ps_mul_mode) & 3;
  out.pixel.native_implicit_lod = REXCVAR_GET(gpu_native_ps_implicit_lod) ? 1 : 0;
  if (converted && out.value != modification) {
    *converted = true;
  }
  return out.value;
}

void VulkanPipelineCache::ApplyNativePixelSpec(SpirvShaderTranslator::Modification& modification, uint32_t key) {
  modification.pixel.tex_unsigned_mask = key & 0xF;
  modification.pixel.native_mul_mode = (key >> 4) & 3;
  modification.pixel.native_implicit_lod = (key >> 6) & 1;
}

void VulkanPipelineCache::StripNativePixelSpec(SpirvShaderTranslator::Modification& modification) {
  modification.pixel.tex_unsigned_mask = 0;
  modification.pixel.native_mul_mode = 0;
  modification.pixel.native_implicit_lod = 0;
}

// wwe2k14 patch (Session 38, NATIVE).
// wwe2k14 patch (Session 87, stage 4, gpu_native_swap_skip_translation_pixel): whether a pixel draw with this native ext uses the
// REPLACEMENT translation's bindings (its SPIR-V's) rather than the emulated translation's. A function of the translation
// identity alone (the swap bit of the ext, which is part of every translation key and of every cached draw's key).
bool VulkanPipelineCache::UsesReplacementBindings(const Shader* shader, uint32_t native_ext) const {
  // (The ext names the translation: the replacement kind of the shader's type, with its own bindings.)
  if (!shader || !(native_ext & SpirvShaderTranslator::kNativeExtOwnBindings)) return false;
  const bool vertex = shader->type() == xenos::ShaderType::kVertex;
  if (!(native_ext & (vertex ? SpirvShaderTranslator::kNativeExtVsSwap : SpirvShaderTranslator::kNativeExtSwap))) return false;
  if (!(vertex ? SwapSkipOn() : SwapSkipOnPixel())) return false;
  // Session 95 (gpu_d3d_replbind_memo): this runs several times a draw per stage (~1% of the command thread in dynamic_cast); every shader reaching here is a VulkanShader (a SpirvShader), so a static_cast gives the same pointer. has_replacement_bindings() is read live.
  const SpirvShader* spirv = REXCVAR_GET(gpu_d3d_replbind_memo) ? static_cast<const SpirvShader*>(shader) : dynamic_cast<const SpirvShader*>(shader);
  return spirv && spirv->has_replacement_bindings();
}

// Census: why NATIVE got no pixel translation for a draw (once per shader and reason; the native post / pass code then
// leaves the draw emulated).
void VulkanPipelineCache::NoteNativePixelRefusal(const Shader& shader, const char* why) {
  if (!SwapSkipOn() && !SwapSkipOnPixel()) return;  // (the default path prints nothing of this)
  static std::mutex mutex;
  static std::unordered_set<uint64_t> seen;
  const uint64_t key = shader.ucode_data_hash() ^ (std::hash<std::string_view>()(why) * 0x9E3779B97F4A7C15ull);
  std::lock_guard<std::mutex> lock(mutex);
  if (seen.insert(key).second) {
    REXGPU_WARN("[native][swap_skip] pixel shader {:016X}: NATIVE got no pixel translation - {}", shader.ucode_data_hash(), why);
  }
}

SpirvShaderTranslator& VulkanPipelineCache::NativeShaderTranslator() {
  if (!native_shader_translator_) {
    SpirvShaderTranslator::Features translator_features(command_processor_.GetVulkanDevice());
    translator_features.shared_memory_window = command_processor_.SharedMemoryWindowOn();  // (Session 93)
    native_shader_translator_ = std::make_unique<SpirvShaderTranslator>(
        translator_features,
        render_target_cache_.msaa_2x_attachments_supported(), render_target_cache_.msaa_2x_no_attachments_supported(), false,
        render_target_cache_.draw_resolution_scale_x(), render_target_cache_.draw_resolution_scale_y());
  }
  return *native_shader_translator_;
}

VulkanShader::VulkanTranslation* VulkanPipelineCache::GetNativePixelShaderTranslation(
    VulkanShader& shader, SpirvShaderTranslator::Modification modification, bool float24_depth,
    uint32_t color_quantize, bool twin_position, uint32_t native_ext) {
  modification.pixel.native_host_render_targets = 1;
  modification.pixel.native_twin_position = twin_position ? 1 : 0;  // Session 54
  modification.pixel.native_color_quantize = color_quantize & 3;  // Session 43 (NATIVE QUANTIZE)
  // Depth stays as the vertex shader and the rasterizer make it (float32 targets, no float24 emulation), or (Session
  // 43, float24_depth) is rounded to 20e4 in guest units by the shader.
  modification.pixel.depth_stencil_mode =
      float24_depth ? SpirvShaderTranslator::Modification::DepthStencilMode::kFloat24Rounding
                    : SpirvShaderTranslator::Modification::DepthStencilMode::kNoModifiers;
  auto* translation =
      static_cast<VulkanShader::VulkanTranslation*>(shader.GetOrCreateNativeTranslation(modification.value, native_ext));
  if (!translation->is_translated()) {
    NativeShaderTranslator();
    // wwe2k14 patch (Session 54, NATIVE draw sizes): drawn into a draw twin (a fractional size) - the pixel position a
    // shader reads is in the twin's pixels.
    const float ratio = twin_position ? command_processor_.NativeTruthDrawRatio() : 1.0f;
    if (ratio > 0.0f && ratio < 1.0f) {
      native_shader_translator_->SetPixelPositionScale(float(render_target_cache_.draw_resolution_scale_x()) * ratio,
                                                       float(render_target_cache_.draw_resolution_scale_y()) * ratio);
    } else {
      native_shader_translator_->SetPixelPositionScale(0.0f, 0.0f);
    }
    // The emulated translation set up the shader's texture and sampler bindings; ours must not change them.
    const size_t textures = shader.GetTextureBindingsAfterTranslation().size();
    const size_t samplers = shader.GetSamplerBindingsAfterTranslation().size();
    // wwe2k14 patch (Session 87, stage 4, gpu_native_swap_skip_translation): a registered pixel shader's NATIVE translation
    // with its replacement is the replacement itself, built from the analysis the registration gave: no translator. When the
    // replacement does not fit, the real analysis is made and the translator runs as it always did.
    bool translated = false;
    bool translated_without_translator = false;
    if (UsesReplacementBindings(&shader, native_ext)) {
      std::vector<uint8_t> replacement;
      const bool probe_skip = (native_ext & SpirvShaderTranslator::kNativeExtAlphaProbe) != 0;
      if (shader_swap::Build(shader, replacement, modification.pixel.native_color_quantize, probe_skip, true,
                             (native_ext & SpirvShaderTranslator::kNativeExtEpilogueIdle) != 0,
                             (native_ext & SpirvShaderTranslator::kNativeExtAblate) != 0)) {
        translation->SetTranslatedExternally(std::move(replacement));
        translated = translated_without_translator = true;
        g_skip.native_skip_p.fetch_add(1);
        REXGPU_INFO("[native][swap_skip] pixel shader {:016X}: NATIVE translation {:#x} is the replacement, no translator; {}",
                    shader.ucode_data_hash(), modification.value, SwapSkipTotals());
      } else {
        // Under kNativeExtOwnBindings the draw's descriptors were made for the replacement's bindings: no translator code can
        // stand in (it would be remapped onto the emulated lists and cached under this key). No translation: the draw is
        // left unclaimed.
        g_skip.refused_p.fetch_add(1);
        NoteNativePixelRefusal(shader, "the replacement does not fit this draw (no fallback under its own bindings)");
        return nullptr;
      }
    } else if (native_ext & SpirvShaderTranslator::kNativeExtOwnBindings) {
      NoteNativePixelRefusal(shader, "own bindings asked for but the shader has none (or the switch is off)");
      return nullptr;
    }
    if (!translated_without_translator) {
      PrepareFullAnalysis(shader);
      translated = native_shader_translator_->TranslateAnalyzedShader(*translation);
    }
    // wwe2k14 patch (Session 72, gpu_native_shader_swap): the title's replacement as this translation's code, before
    // its module is made (the translator's code stays if the replacement does not fit - logged once). The translation
    // above still ran: it sets up the shader's bindings and constants, which the replacement uses as they are.
    if (translated && !translated_without_translator && (native_ext & SpirvShaderTranslator::kNativeExtSwap)) {
      std::vector<uint8_t> replacement;
      const bool probe = (native_ext & SpirvShaderTranslator::kNativeExtAlphaProbe) != 0;
      if (shader_swap::Build(shader, replacement, modification.pixel.native_color_quantize, probe, false,
                             (native_ext & SpirvShaderTranslator::kNativeExtEpilogueIdle) != 0,
                             (native_ext & SpirvShaderTranslator::kNativeExtAblate) != 0)) {
        translation->ReplaceTranslatedBinary(std::move(replacement));
      } else if (probe) {
        NoteNativePixelRefusal(shader, "no alpha probe patch for the replacement");
        return nullptr;  // (a probe module is the replacement with its patch or nothing - never the plain translation)
      }
    }
    if (translated) {  // (Session 92, M6 groundwork) the NATIVE pixel variant, once per run, with the pixel position scale it was translated at
      M6NoteVariant(true, false, shader.ucode_data_hash(), modification.value, native_ext,
                    ratio > 0.0f && ratio < 1.0f ? float(render_target_cache_.draw_resolution_scale_x()) * ratio : 0.0f,
                    ratio > 0.0f && ratio < 1.0f ? float(render_target_cache_.draw_resolution_scale_y()) * ratio : 0.0f,
                    translated_without_translator);
    }
    if (!translated || translation->GetOrCreateShaderModule() == VK_NULL_HANDLE) {
      REXGPU_WARN("[native] pixel shader {:016X}: no host render target translation", shader.ucode_data_hash());
      NoteNativePixelRefusal(shader, "no host render target translation");
      return nullptr;
    }
    if (shader.GetTextureBindingsAfterTranslation().size() != textures ||
        shader.GetSamplerBindingsAfterTranslation().size() != samplers) {
      REXGPU_ERROR("[native] pixel shader {:016X}: the host render target translation has other bindings",
                   shader.ucode_data_hash());
      NoteNativePixelRefusal(shader, "the translation has other bindings than the emulated one");
      return nullptr;
    }
    // wwe2k14 patch (Session 66, diagnostic): dump_shaders covers NATIVE's own pixel shaders too ("vk_native").
    if (!REXCVAR_GET(dump_shaders).empty()) {
      translation->Dump(REXCVAR_GET(dump_shaders), "vk_native");
    }
  }
  return translation->is_valid() ? translation : nullptr;
}

// wwe2k14 patch (Session 42, NATIVE POST): as GetCurrentStateDescription picks the geometry shader.
VkShaderModule VulkanPipelineCache::GetNativeGeometryShader(
    const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
    SpirvShaderTranslator::Modification vertex_modification, SpirvShaderTranslator::Modification pixel_modification,
    VkPrimitiveTopology& topology_out) {
  if (primitive_processing_result.host_vertex_shader_type != Shader::HostVertexShaderType::kVertex) {
    return VK_NULL_HANDLE;
  }
  PipelineGeometryShader type;
  switch (primitive_processing_result.host_primitive_type) {
    case xenos::PrimitiveType::kPointList:
      type = PipelineGeometryShader::kPointList;
      topology_out = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
      break;
    case xenos::PrimitiveType::kRectangleList:
      type = PipelineGeometryShader::kRectangleList;
      topology_out = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      break;
    case xenos::PrimitiveType::kQuadList:
      type = PipelineGeometryShader::kQuadList;
      topology_out = VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
      break;
    default:
      return VK_NULL_HANDLE;
  }
  GeometryShaderKey key;
  if (!GetGeometryShaderKey(type, vertex_modification, pixel_modification, key)) {
    return VK_NULL_HANDLE;
  }
  return GetGeometryShader(key);
}

// wwe2k14 patch (Session 61).
void VulkanPipelineCache::StoreShader(Shader& shader) {
  if (!shader_storage_file_ || shader.ucode_storage_index() == shader_storage_index_) return;
  shader.set_ucode_storage_index(shader_storage_index_);
  assert_not_null(storage_write_thread_);
  shader_storage_file_flush_needed_ = true;
  {
    std::lock_guard<std::mutex> lock(storage_write_request_lock_);
    storage_write_shader_queue_.push_back(&shader);
  }
  storage_write_request_cond_.notify_all();
}

bool VulkanPipelineCache::ConfigurePipeline(
    VulkanShader::VulkanTranslation* vertex_shader, VulkanShader::VulkanTranslation* pixel_shader,
    const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
    reg::RB_DEPTHCONTROL normalized_depth_control, uint32_t normalized_color_mask,
    VulkanRenderTargetCache::RenderPassKey render_pass_key, VkPipeline& pipeline_out,
    const PipelineLayoutProvider*& pipeline_layout_out, void** pipeline_handle_out) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  // wwe2k14 patch (Session 92, M6): the depth-only shaders Initialize left for the translation store: made here on the command
  // thread when the shader storage never opened (no cache root / title 0), before any pipeline asks for them.
  if (depth_only_fragment_shader_ == VK_NULL_HANDLE && depth_float24_round_fragment_shader_ == VK_NULL_HANDLE) {
    EnsureInternalFragmentShaders();
  }
  // Ensure shaders are translated - needed now for GetCurrentStateDescription.
  if (!EnsureShadersTranslated(vertex_shader, pixel_shader)) {
    return false;
  }
  if (shader_storage_file_ &&
      vertex_shader->shader().ucode_storage_index() != shader_storage_index_) {
    vertex_shader->shader().set_ucode_storage_index(shader_storage_index_);
    assert_not_null(storage_write_thread_);
    shader_storage_file_flush_needed_ = true;
    {
      std::lock_guard<std::mutex> lock(storage_write_request_lock_);
      storage_write_shader_queue_.push_back(&vertex_shader->shader());
    }
    storage_write_request_cond_.notify_all();
  }
  if (pixel_shader && shader_storage_file_ &&
      pixel_shader->shader().ucode_storage_index() != shader_storage_index_) {
    pixel_shader->shader().set_ucode_storage_index(shader_storage_index_);
    assert_not_null(storage_write_thread_);
    shader_storage_file_flush_needed_ = true;
    {
      std::lock_guard<std::mutex> lock(storage_write_request_lock_);
      storage_write_shader_queue_.push_back(&pixel_shader->shader());
    }
    storage_write_request_cond_.notify_all();
  }

  PipelineDescription description;
  if (!GetCurrentStateDescription(vertex_shader, pixel_shader, primitive_processing_result,
                                  normalized_depth_control, normalized_color_mask, render_pass_key,
                                  description)) {
    REXGPU_ERROR(
        "VulkanPipelineCache: GetCurrentStateDescription failed "
        "(guest_prim={}, host_prim={}, host_vs_type={}, tess_mode={}, host_reset={}, "
        "render_pass_key=0x{:08X})",
        uint32_t(primitive_processing_result.guest_primitive_type),
        uint32_t(primitive_processing_result.host_primitive_type),
        uint32_t(primitive_processing_result.host_vertex_shader_type),
        uint32_t(primitive_processing_result.tessellation_mode),
        uint32_t(primitive_processing_result.host_primitive_reset_enabled), render_pass_key.key);
    return false;
  }
  // wwe2k14 patch (Session 55): vulkan_pipeline_sync_offscreen - the surface the draw goes to (guest pixels).
  // wwe2k14 patch (Session 58): and a surface wider than any screen (the 2080-wide surface a character's 2048x1024
  // attire is drawn on: two of its draws were skipped in the 30 Years menu, runs 3887 / 3897) waits too.
  const uint32_t surface_pitch = KeyRegisters().Get<reg::RB_SURFACE_INFO>().surface_pitch;
  const bool sync_offscreen = REXCVAR_GET(vulkan_pipeline_sync_offscreen) && !creation_threads_.empty() &&
                              (surface_pitch < uint32_t(REXCVAR_GET(vulkan_pipeline_sync_offscreen_min_pitch)) ||
                               surface_pitch > uint32_t(REXCVAR_GET(vulkan_pipeline_sync_offscreen_max_pitch)));
  if (last_pipeline_ && last_pipeline_->first == description) {
    VkPipeline last_pipeline = last_pipeline_->second.pipeline.load(std::memory_order_acquire);
    const PipelineLayoutProvider* last_pipeline_layout =
        last_pipeline_->second.pipeline_layout.load(std::memory_order_acquire);
    // (Session 55, review: not a placeholder for a narrow draw - the lookup below waits for the real one.)
    if (last_pipeline != VK_NULL_HANDLE && last_pipeline_layout != nullptr &&
        !(sync_offscreen && last_pipeline_->second.is_placeholder.load(std::memory_order_acquire))) {
      pipeline_out = last_pipeline;
      pipeline_layout_out = last_pipeline_layout;
      if (pipeline_handle_out) {
        *pipeline_handle_out = const_cast<Pipeline*>(&last_pipeline_->second);
      }
      return true;
    }
  }

  bool use_async = REXCVAR_GET(async_shader_compilation) && !creation_threads_.empty() &&
                   pixel_shader && placeholder_pixel_shader_ != VK_NULL_HANDLE;
  uint8_t async_priority = pipeline_util::kPriorityLowest;
  if (use_async) {
    uint32_t bound_rts =
        pipeline_util::GetBoundRTMaskFromNormalizedColorMask(normalized_color_mask);
    uint32_t shader_writes_color_targets =
        pixel_shader ? pixel_shader->shader().writes_color_targets() : 0;
    bool shader_writes_depth = pixel_shader ? pixel_shader->shader().writes_depth()
                                            : normalized_depth_control.z_write_enable != 0;
    async_priority = pipeline_util::CalculatePipelinePriority(
        bound_rts, shader_writes_color_targets, shader_writes_depth);
  }

  auto it = pipelines_.find(description);
  if (it != pipelines_.end()) {
    VkPipeline found_pipeline = it->second.pipeline.load(std::memory_order_acquire);
    if (sync_offscreen && it->second.is_placeholder.load(std::memory_order_acquire)) {
      // wwe2k14 patch (Session 55): still being made (queued for a screen-wide draw first): this one only, out of
      // the queue onto this thread, or waited for on its creation thread (review: AwaitPipelineCreation makes the
      // whole queue - a long stall on a slow laptop while a match loads).
      AwaitPendingPipeline(&*it);
      NoteOffscreenWait("awaited");
      found_pipeline = it->second.pipeline.load(std::memory_order_acquire);
    }
    if (found_pipeline == VK_NULL_HANDLE &&
        it->second.is_placeholder.load(std::memory_order_acquire)) {
      // wwe2k14 patch (Session 6, H1): queued by vulkan_pipeline_no_submit_wait
      // without a placeholder, a creation thread owns it until it clears
      // is_placeholder - creating it here too would race with that thread.
      // Pending: the caller skips the draw (GetPipelineAndLayoutByHandle). Not
      // gated by the setting, so that turning it off with creations still
      // pending stays safe; without it, this state never outlives
      // ConfigurePipeline (a placeholder is non-null, and a failed creation
      // clears the flag).
      last_pipeline_ = &*it;
      pipeline_out = VK_NULL_HANDLE;
      pipeline_layout_out = nullptr;
      if (pipeline_handle_out) {
        *pipeline_handle_out = &it->second;
      }
      return true;
    }
    if (found_pipeline == VK_NULL_HANDLE) {
      PipelineCreationArguments creation_arguments;
      if (!TryGetPipelineCreationArgumentsForDescription(description, &*it, creation_arguments) ||
          !EnsurePipelineCreated(creation_arguments)) {
        return false;
      }
      // wwe2k14 patch (Session 7c, CAUSES): measurement only - created now, on this thread (review fix: only
      // once the creation has succeeded).
      ++command_processor_.cause_counts().pipelines_created;
      found_pipeline = it->second.pipeline.load(std::memory_order_acquire);
    }
    const PipelineLayoutProvider* found_pipeline_layout =
        it->second.pipeline_layout.load(std::memory_order_acquire);
    last_pipeline_ = &*it;
    pipeline_out = found_pipeline;
    pipeline_layout_out = found_pipeline_layout;
    if (pipeline_handle_out) {
      *pipeline_handle_out = &it->second;
    }
    return pipeline_out != VK_NULL_HANDLE && pipeline_layout_out != nullptr;
  }

  // Create the pipeline if not already existing.
  auto& pipeline = *pipelines_.emplace(description, Pipeline()).first;
  PipelineCreationArguments creation_arguments_real;
  if (!TryGetPipelineCreationArgumentsForDescription(description, &pipeline,
                                                     creation_arguments_real)) {
    pipelines_.erase(description);
    return false;
  }

  bool queued_async_creation = false;
  // wwe2k14 patch (Session 6, H1): with vulkan_pipeline_no_submit_wait, queue
  // the real pipeline right away without building a placeholder on this thread
  // - IssueDraw skips draws whose pipeline is a placeholder before binding
  // anything, so a placeholder is only ever bound in a race (and then draws
  // nothing). "Pending" is is_placeholder with no VkPipeline; the creation
  // thread publishes the pipeline and the layout before clearing the flag.
  // Draws without a pixel shader (depth-only) are created this way too unless
  // vulkan_pipeline_sync_depth_only. Pipelines with shaders writing memory
  // (memexport) are created synchronously instead, even with a pixel shader:
  // skipping their draws for several frames would leave guest-visible memory
  // unwritten (the original path skips them for one frame).
  bool queued_without_placeholder = false;
  if (REXCVAR_GET(vulkan_pipeline_no_submit_wait) && REXCVAR_GET(async_shader_compilation) &&
      !creation_threads_.empty()) {
    if (vertex_shader->shader().memexport_eM_written() ||
        (pixel_shader && pixel_shader->shader().memexport_eM_written())) {
      use_async = false;
    } else if (sync_offscreen) {  // wwe2k14 patch (Session 55): made now (below)
    } else if (pixel_shader || !REXCVAR_GET(vulkan_pipeline_sync_depth_only)) {
      queued_without_placeholder = true;
    }
  }
  // wwe2k14 patch (Session 55): vulkan_pipeline_sync_offscreen - a narrow draw's new pipeline is made now, on this
  // thread, with or without vulkan_pipeline_no_submit_wait (review: no placeholder for it either).
  if (sync_offscreen && use_async) {
    use_async = false;
    NoteOffscreenWait("made at once");
  }
  if (queued_without_placeholder) {
    if (use_async) {
      creation_arguments_real.priority = async_priority;
    } else {
      // Depth-only (or no placeholder shader) - same formula as above.
      creation_arguments_real.priority = pipeline_util::CalculatePipelinePriority(
          pipeline_util::GetBoundRTMaskFromNormalizedColorMask(normalized_color_mask),
          pixel_shader ? pixel_shader->shader().writes_color_targets() : 0,
          pixel_shader ? pixel_shader->shader().writes_depth()
                       : normalized_depth_control.z_write_enable != 0);
    }
    pipeline.second.is_placeholder.store(true, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(creation_request_lock_);
      creation_queue_.push(creation_arguments_real);
    }
    creation_request_cond_.notify_one();
    queued_async_creation = true;
  }
  if (use_async && !queued_async_creation) {
    creation_arguments_real.priority = async_priority;
    PipelineCreationArguments creation_arguments_placeholder;
    if (TryGetPipelineCreationArgumentsForDescription(description, &pipeline,
                                                      creation_arguments_placeholder, true)) {
      pipeline.second.is_placeholder.store(true, std::memory_order_release);
      if (EnsurePipelineCreated(creation_arguments_placeholder, placeholder_pixel_shader_)) {
        {
          std::lock_guard<std::mutex> lock(creation_request_lock_);
          creation_queue_.push(creation_arguments_real);
        }
        creation_request_cond_.notify_one();
        queued_async_creation = true;
      } else {
        pipeline.second.is_placeholder.store(false, std::memory_order_release);
      }
    }
  }

  if (!queued_async_creation && !EnsurePipelineCreated(creation_arguments_real)) {
    pipelines_.erase(description);
    return false;
  }
  // wwe2k14 patch (Session 7c, CAUSES): measurement only - a new pipeline, created or queued above (review fix:
  // not counted when its creation arguments or its creation failed).
  if (eds_active_) {
    ++eds_stats_.pipelines_created;  // wwe2k14 patch (Session 93, idea 7)
  }
  ++command_processor_.cause_counts().pipelines_created;
  if (!queued_async_creation) {
    pipeline.second.is_placeholder.store(false, std::memory_order_release);
  }

  if (pipeline_storage_file_) {
    assert_not_null(storage_write_thread_);
    pipeline_storage_file_flush_needed_ = true;
    {
      std::lock_guard<std::mutex> lock(storage_write_request_lock_);
      storage_write_pipeline_queue_.emplace_back();
      PipelineStoredDescription& stored_description = storage_write_pipeline_queue_.back();
      stored_description.description_hash = description.GetHash();
      std::memcpy(&stored_description.description, &description, sizeof(description));
    }
    storage_write_request_cond_.notify_all();
  }
  last_pipeline_ = &pipeline;
  pipeline_out = pipeline.second.pipeline.load(std::memory_order_acquire);
  pipeline_layout_out = pipeline.second.pipeline_layout.load(std::memory_order_acquire);
  if (pipeline_handle_out) {
    *pipeline_handle_out = &pipeline.second;
  }
  if (queued_without_placeholder) {
    // wwe2k14 patch (Session 6, H1): pending is not a failure, the caller skips
    // the draw until the creation thread has published the pipeline.
    return true;
  }
  return pipeline_out != VK_NULL_HANDLE && pipeline_layout_out != nullptr;
}

bool VulkanPipelineCache::IsCreatingPipelines() const {
  std::lock_guard<std::mutex> lock(creation_request_lock_);
  if (creation_threads_.empty()) {
    return startup_loading_;
  }
  return startup_loading_ || !creation_queue_.empty() || creation_threads_busy_ != 0;
}

void VulkanPipelineCache::GetPipelineAndLayoutByHandle(
    void* handle, VkPipeline& pipeline_out, const PipelineLayoutProvider*& pipeline_layout_out,
    bool* is_placeholder_out) const {
  if (!handle) {
    pipeline_out = VK_NULL_HANDLE;
    pipeline_layout_out = nullptr;
    if (is_placeholder_out) {
      *is_placeholder_out = false;
    }
    return;
  }
  const auto* pipeline = reinterpret_cast<const Pipeline*>(handle);
  if (is_placeholder_out && REXCVAR_GET(vulkan_pipeline_no_submit_wait)) {
    // wwe2k14 patch (Session 6, H1): read the flag first. A creation thread
    // stores the pipeline and the layout before clearing it (release), so a
    // cleared flag (acquire) guarantees both are the ones published with it -
    // not the pending null pipeline or the placeholder they replaced (the
    // original order can see the old pipeline together with the new flag).
    *is_placeholder_out = pipeline->is_placeholder.load(std::memory_order_acquire);
    pipeline_out = pipeline->pipeline.load(std::memory_order_acquire);
    pipeline_layout_out = pipeline->pipeline_layout.load(std::memory_order_acquire);
    return;
  }
  pipeline_out = pipeline->pipeline.load(std::memory_order_acquire);
  pipeline_layout_out = pipeline->pipeline_layout.load(std::memory_order_acquire);
  if (is_placeholder_out) {
    *is_placeholder_out = pipeline->is_placeholder.load(std::memory_order_acquire);
  }
}

bool VulkanPipelineCache::TranslateAnalyzedShader(SpirvShaderTranslator& translator,
                                                  VulkanShader::VulkanTranslation& translation) {
  VulkanShader& shader = static_cast<VulkanShader&>(translation.shader());

  // wwe2k14 patch (Session 87, stage 4, gpu_native_swap_skip_translation): a registered vertex shader's NATIVE translation
  // is the replacement itself, built from the analysis the registration gave: no translator. When the replacement does not
  // fit this translation (BuildVertex refuses), the real analysis is made and the translator runs as it always did.
  bool translated_without_translator = false;
  if (shader.type() == xenos::ShaderType::kVertex && UsesReplacementBindings(&shader, translation.native_ext())) {
    std::vector<uint8_t> replacement;
    if (shader_swap::BuildVertex(shader, translation.modification(), replacement, true, false,
                                 (translation.native_ext() & (SpirvShaderTranslator::kNativeExtVsInput | SpirvShaderTranslator::kNativeExtVsSkinOnce)) != 0,
                                 (translation.native_ext() & SpirvShaderTranslator::kNativeExtVsSkinOnce) != 0)) {
      translation.SetTranslatedExternally(std::move(replacement));
      translated_without_translator = true;
      g_skip.native_skip_v.fetch_add(1);
      REXGPU_INFO("[native][swap_skip] vertex shader {:016X}: NATIVE translation {:#x} is the replacement, no translator; {}",
                  shader.ucode_data_hash(), translation.modification(), SwapSkipTotals());
    } else {
      // (Under own bindings no translator code can stand in: the draw's descriptors were made for the replacement's.)
      g_skip.refused_v.fetch_add(1);
      REXGPU_ERROR("[native][swap_skip] vertex shader {:016X}: the replacement does not fit translation {:#x} under its own "
                   "bindings - no translation, the draw is left unclaimed", shader.ucode_data_hash(), translation.modification());
      return false;
    }
  } else if ((translation.native_ext() & SpirvShaderTranslator::kNativeExtOwnBindings) &&
             shader.type() == xenos::ShaderType::kVertex) {
    REXGPU_ERROR("[native][swap_skip] vertex shader {:016X}: own bindings asked for but the shader has none", shader.ucode_data_hash());
    return false;
  }
  // Perform translation.
  // If this fails the shader will be marked as invalid and ignored later.
  if (!translated_without_translator && !translator.TranslateAnalyzedShader(translation)) {
    REXGPU_ERROR("Shader {:016X} translation failed; marking as ignored", shader.ucode_data_hash());
    return false;
  }
  // (Session 92, M6 groundwork) The variant, once per run (the emulated translator; a NATIVE-ext vertex translation also comes here).
  M6NoteVariant(translation.native_ext() != 0, shader.type() == xenos::ShaderType::kVertex, shader.ucode_data_hash(), translation.modification(),
                translation.native_ext(), 0.0f, 0.0f, translated_without_translator);
  // wwe2k14 patch (Session 87, stage 4, gpu_native_vs_swap): the title's own vertex shader as this translation's code,
  // before its module is made (the translation above still ran: it sets up the shader's bindings and constants; the
  // translator's code stays if the replacement does not fit - logged once).
  if (!translated_without_translator && (translation.native_ext() & SpirvShaderTranslator::kNativeExtVsSwap) &&
      shader.type() == xenos::ShaderType::kVertex) {
    std::vector<uint8_t> replacement;
    if (shader_swap::BuildVertex(shader, translation.modification(), replacement)) {
      translation.ReplaceTranslatedBinary(std::move(replacement));
    }
  }
  shader_swap::NoteTranslationChange();  // (Session 93: a translation landed)
  if (translation.GetOrCreateShaderModule() == VK_NULL_HANDLE) {
    return false;
  }
  // wwe2k14 patch (Session 20, DUMPSPV): dump_shaders for Vulkan too (the SPIR-V binary, as the D3D12 path dumps
  // DXBC; the ucode is dumped by the translator). Pixel shaders are tagged with the render-target path.
  if (!REXCVAR_GET(dump_shaders).empty()) {
    translation.Dump(REXCVAR_GET(dump_shaders),
                     shader.type() == xenos::ShaderType::kPixel
                         ? (render_target_cache_.GetPath() == RenderTargetCache::Path::kPixelShaderInterlock ? "vk_fsi"
                                                                                                              : "vk_fbo")
                         : "vk");
  }

  // TODO(Triang3l): Log that the shader has been successfully translated in
  // common code.

  // Set up the texture binding layout. (Session 87: from the translator's lists only - a replacement translation's bindings are
  // its own and the draws carry them; the shader's layout UIDs are the emulated translation's, set when it is made.)
  if (!translated_without_translator && shader.EnterBindingLayoutUserUIDSetup()) {
    // Obtain the unique IDs of the binding layout if there are any texture
    // bindings, for invalidation in the command processor.
    size_t texture_binding_layout_uid = kLayoutUIDEmpty;
    const std::vector<VulkanShader::TextureBinding>& texture_bindings =
        shader.GetTextureBindingsAfterTranslation();
    size_t texture_binding_count = texture_bindings.size();
    if (texture_binding_count) {
      size_t texture_binding_layout_bytes =
          texture_binding_count * sizeof(*texture_bindings.data());
      uint64_t texture_binding_layout_hash =
          XXH3_64bits(texture_bindings.data(), texture_binding_layout_bytes);
      auto found_range = texture_binding_layout_map_.equal_range(texture_binding_layout_hash);
      for (auto it = found_range.first; it != found_range.second; ++it) {
        if (it->second.vector_span_length == texture_binding_count &&
            !std::memcmp(texture_binding_layouts_.data() + it->second.vector_span_offset,
                         texture_bindings.data(), texture_binding_layout_bytes)) {
          texture_binding_layout_uid = it->second.uid;
          break;
        }
      }
      if (texture_binding_layout_uid == kLayoutUIDEmpty) {
        static_assert(kLayoutUIDEmpty == 0,
                      "Layout UID is size + 1 because it's assumed that 0 is the UID for "
                      "an empty layout");
        texture_binding_layout_uid = texture_binding_layout_map_.size() + 1;
        LayoutUID new_uid;
        new_uid.uid = texture_binding_layout_uid;
        new_uid.vector_span_offset = texture_binding_layouts_.size();
        new_uid.vector_span_length = texture_binding_count;
        texture_binding_layouts_.resize(new_uid.vector_span_offset + texture_binding_count);
        std::memcpy(texture_binding_layouts_.data() + new_uid.vector_span_offset,
                    texture_bindings.data(), texture_binding_layout_bytes);
        texture_binding_layout_map_.emplace(texture_binding_layout_hash, new_uid);
      }
    }
    shader.SetTextureBindingLayoutUserUID(texture_binding_layout_uid);

    // Use the sampler count for samplers because it's the only thing that must
    // be the same for layouts to be compatible in this case
    // (instruction-specified parameters are used as overrides for creating
    // actual samplers).
    static_assert(kLayoutUIDEmpty == 0,
                  "Empty layout UID is assumed to be 0 because for bindful samplers, the "
                  "UID is their count");
    shader.SetSamplerBindingLayoutUserUID(shader.GetSamplerBindingsAfterTranslation().size());
  }

  return true;
}

void VulkanPipelineCache::WritePipelineRenderTargetDescription(
    reg::RB_BLENDCONTROL blend_control, uint32_t write_mask,
    PipelineRenderTarget& render_target_out) const {
  if (write_mask) {
    assert_zero(write_mask & ~uint32_t(0b1111));
    // 32 because of 0x1F mask, for safety (all unknown to zero).
    static const PipelineBlendFactor kBlendFactorMap[32] = {
        /*  0 */ PipelineBlendFactor::kZero,
        /*  1 */ PipelineBlendFactor::kOne,
        /*  2 */ PipelineBlendFactor::kZero,  // ?
        /*  3 */ PipelineBlendFactor::kZero,  // ?
        /*  4 */ PipelineBlendFactor::kSrcColor,
        /*  5 */ PipelineBlendFactor::kOneMinusSrcColor,
        /*  6 */ PipelineBlendFactor::kSrcAlpha,
        /*  7 */ PipelineBlendFactor::kOneMinusSrcAlpha,
        /*  8 */ PipelineBlendFactor::kDstColor,
        /*  9 */ PipelineBlendFactor::kOneMinusDstColor,
        /* 10 */ PipelineBlendFactor::kDstAlpha,
        /* 11 */ PipelineBlendFactor::kOneMinusDstAlpha,
        /* 12 */ PipelineBlendFactor::kConstantColor,
        /* 13 */ PipelineBlendFactor::kOneMinusConstantColor,
        /* 14 */ PipelineBlendFactor::kConstantAlpha,
        /* 15 */ PipelineBlendFactor::kOneMinusConstantAlpha,
        /* 16 */ PipelineBlendFactor::kSrcAlphaSaturate,
    };
    render_target_out.src_color_blend_factor =
        kBlendFactorMap[uint32_t(blend_control.color_srcblend)];
    render_target_out.dst_color_blend_factor =
        kBlendFactorMap[uint32_t(blend_control.color_destblend)];
    render_target_out.color_blend_op = blend_control.color_comb_fcn;
    render_target_out.src_alpha_blend_factor =
        kBlendFactorMap[uint32_t(blend_control.alpha_srcblend)];
    render_target_out.dst_alpha_blend_factor =
        kBlendFactorMap[uint32_t(blend_control.alpha_destblend)];
    render_target_out.alpha_blend_op = blend_control.alpha_comb_fcn;
    if (!command_processor_.GetVulkanDevice()->properties().constantAlphaColorBlendFactors) {
      if (blend_control.color_srcblend == xenos::BlendFactor::kConstantAlpha) {
        render_target_out.src_color_blend_factor = PipelineBlendFactor::kConstantColor;
      } else if (blend_control.color_srcblend == xenos::BlendFactor::kOneMinusConstantAlpha) {
        render_target_out.src_color_blend_factor = PipelineBlendFactor::kOneMinusConstantColor;
      }
      if (blend_control.color_destblend == xenos::BlendFactor::kConstantAlpha) {
        render_target_out.dst_color_blend_factor = PipelineBlendFactor::kConstantColor;
      } else if (blend_control.color_destblend == xenos::BlendFactor::kOneMinusConstantAlpha) {
        render_target_out.dst_color_blend_factor = PipelineBlendFactor::kOneMinusConstantColor;
      }
    }
  } else {
    render_target_out.src_color_blend_factor = PipelineBlendFactor::kOne;
    render_target_out.dst_color_blend_factor = PipelineBlendFactor::kZero;
    render_target_out.color_blend_op = xenos::BlendOp::kAdd;
    render_target_out.src_alpha_blend_factor = PipelineBlendFactor::kOne;
    render_target_out.dst_alpha_blend_factor = PipelineBlendFactor::kZero;
    render_target_out.alpha_blend_op = xenos::BlendOp::kAdd;
  }
  render_target_out.color_write_mask = write_mask;
}

bool VulkanPipelineCache::GetCurrentStateDescriptionFull(
    const VulkanShader::VulkanTranslation* vertex_shader,
    const VulkanShader::VulkanTranslation* pixel_shader,
    const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
    reg::RB_DEPTHCONTROL normalized_depth_control, uint32_t normalized_color_mask,
    VulkanRenderTargetCache::RenderPassKey render_pass_key,
    PipelineDescription& description_out) const {
  description_out.Reset();

  const ui::vulkan::VulkanDevice::Properties& device_properties =
      command_processor_.GetVulkanDevice()->properties();

  const RegisterFile& regs = KeyRegisters();
  auto pa_su_sc_mode_cntl = regs.Get<reg::PA_SU_SC_MODE_CNTL>();

  description_out.vertex_shader_hash = vertex_shader->shader().ucode_data_hash();
  description_out.vertex_shader_modification = vertex_shader->modification();
  SpirvShaderTranslator::Modification vertex_shader_modification(
      description_out.vertex_shader_modification);
  Shader::HostVertexShaderType host_vertex_shader_type =
      vertex_shader_modification.vertex.host_vertex_shader_type;
  if (pixel_shader) {
    description_out.pixel_shader_hash = pixel_shader->shader().ucode_data_hash();
    description_out.pixel_shader_modification = pixel_shader->modification();
    SpirvShaderTranslator::Modification pixel_shader_modification(
        description_out.pixel_shader_modification);
    using DepthStencilMode = SpirvShaderTranslator::Modification::DepthStencilMode;
    description_out.sample_rate_shading =
        !pixel_shader->shader().writes_depth() &&
        (pixel_shader_modification.pixel.depth_stencil_mode ==
             DepthStencilMode::kFloat24Truncating ||
         pixel_shader_modification.pixel.depth_stencil_mode == DepthStencilMode::kFloat24Rounding);
  }
  description_out.render_pass_key = render_pass_key;

  bool tessellated = primitive_processing_result.IsTessellated();
  PipelineGeometryShader geometry_shader = PipelineGeometryShader::kNone;
  PipelinePrimitiveTopology primitive_topology;
  if (tessellated) {
    switch (primitive_processing_result.host_primitive_type) {
      case xenos::PrimitiveType::kTriangleList:
      case xenos::PrimitiveType::kQuadList:
      case xenos::PrimitiveType::kTrianglePatch:
      case xenos::PrimitiveType::kQuadPatch:
        primitive_topology = PipelinePrimitiveTopology::kPatchList;
        break;
      default:
        return false;
    }
    description_out.tessellation_mode = primitive_processing_result.tessellation_mode;
  } else {
    // Fallback expansion in the vertex shader for primitive types that may be
    // emulated through geometry shaders on GS-capable hosts.
    if (host_vertex_shader_type == Shader::HostVertexShaderType::kPointListAsTriangleStrip ||
        host_vertex_shader_type == Shader::HostVertexShaderType::kRectangleListAsTriangleStrip) {
      if (primitive_processing_result.host_primitive_type != xenos::PrimitiveType::kTriangleStrip ||
          !primitive_processing_result.host_primitive_reset_enabled) {
        assert_always();
        return false;
      }
      primitive_topology = PipelinePrimitiveTopology::kTriangleStrip;
    } else {
      switch (primitive_processing_result.host_primitive_type) {
        case xenos::PrimitiveType::kPointList:
          geometry_shader = PipelineGeometryShader::kPointList;
          primitive_topology = PipelinePrimitiveTopology::kPointList;
          break;
        case xenos::PrimitiveType::kLineList:
          primitive_topology = PipelinePrimitiveTopology::kLineList;
          break;
        case xenos::PrimitiveType::kLineStrip:
          primitive_topology = PipelinePrimitiveTopology::kLineStrip;
          break;
        case xenos::PrimitiveType::kTriangleList:
          primitive_topology = PipelinePrimitiveTopology::kTriangleList;
          break;
        case xenos::PrimitiveType::kTriangleFan:
          // Triangle fans on Vulkan are expected to be converted to triangle
          // lists in primitive processing for parity with D3D12 behavior.
          assert_always();
          return false;
        case xenos::PrimitiveType::kTriangleStrip:
          primitive_topology = PipelinePrimitiveTopology::kTriangleStrip;
          break;
        case xenos::PrimitiveType::kRectangleList:
          geometry_shader = PipelineGeometryShader::kRectangleList;
          primitive_topology = PipelinePrimitiveTopology::kTriangleList;
          break;
        case xenos::PrimitiveType::kQuadList:
          geometry_shader = PipelineGeometryShader::kQuadList;
          primitive_topology = PipelinePrimitiveTopology::kLineListWithAdjacency;
          break;
        default:
          return false;
      }
    }
  }
  description_out.geometry_shader = geometry_shader;
  description_out.primitive_topology = primitive_topology;
  description_out.primitive_restart = primitive_processing_result.host_primitive_reset_enabled;

  description_out.depth_clamp_enable =
      device_properties.depthClamp && regs.Get<reg::PA_CL_CLIP_CNTL>().clip_disable;

  // Tessellated draws are patch-domain polygonal primitives regardless of guest
  // register ambiguity in non-explicit major mode configurations.
  bool primitive_polygonal = tessellated ? true : draw_util::IsPrimitivePolygonal(regs);
  bool rasterization_enabled = draw_util::IsRasterizationPotentiallyDone(regs, primitive_polygonal);
  if (!rasterization_enabled) {
    // Keep parity with D3D12 by fully disabling rasterization for draws where
    // only non-raster stages (for instance, vertex memexport) can have side
    // effects.
    description_out.rasterizer_discard = 1;
    return true;
  }
  if (primitive_polygonal) {
    // Vulkan only allows the polygon mode to be set for both faces - pick the
    // most special one (more likely to represent the developer's deliberate
    // intentions - fill is very generic, wireframe is common in debug, points
    // are for pretty unusual things, but closer to debug purposes too - on the
    // Xenos, points have the lowest register value and triangles have the
    // highest) based on which faces are not culled.
    bool cull_front = pa_su_sc_mode_cntl.cull_front;
    bool cull_back = pa_su_sc_mode_cntl.cull_back;
    description_out.cull_front = cull_front;
    description_out.cull_back = cull_back;
    if (device_properties.fillModeNonSolid) {
      xenos::PolygonType polygon_type = xenos::PolygonType::kTriangles;
      if (!cull_front) {
        polygon_type = std::min(polygon_type, pa_su_sc_mode_cntl.polymode_front_ptype);
      }
      if (!cull_back) {
        polygon_type = std::min(polygon_type, pa_su_sc_mode_cntl.polymode_back_ptype);
      }
      if (pa_su_sc_mode_cntl.poly_mode != xenos::PolygonModeEnable::kDualMode) {
        polygon_type = xenos::PolygonType::kTriangles;
      }
      switch (polygon_type) {
        case xenos::PolygonType::kPoints:
        case xenos::PolygonType::kLines:
          // Keep non-triangle polygon fill behavior aligned with D3D12 (which
          // maps both point and line polygon modes to wireframe).
          description_out.polygon_mode = PipelinePolygonMode::kLine;
          break;
        case xenos::PolygonType::kTriangles:
          description_out.polygon_mode = PipelinePolygonMode::kFill;
          break;
        default:
          assert_unhandled_case(polygon_type);
          return false;
      }
    } else {
      description_out.polygon_mode = PipelinePolygonMode::kFill;
    }
    description_out.front_face_clockwise = pa_su_sc_mode_cntl.face != 0;
  } else {
    description_out.polygon_mode = PipelinePolygonMode::kFill;
  }
  if (tessellated && REXCVAR_GET(vulkan_tessellation_wireframe) &&
      device_properties.fillModeNonSolid) {
    description_out.polygon_mode = PipelinePolygonMode::kLine;
  }

  if (render_target_cache_.GetPath() == RenderTargetCache::Path::kHostRenderTargets) {
    if (render_pass_key.depth_and_color_used & 1) {
      if (normalized_depth_control.z_enable) {
        description_out.depth_write_enable = normalized_depth_control.z_write_enable;
        description_out.depth_compare_op = normalized_depth_control.zfunc;
      } else {
        description_out.depth_compare_op = xenos::CompareFunction::kAlways;
      }
      if (normalized_depth_control.stencil_enable) {
        description_out.stencil_test_enable = 1;
        description_out.stencil_front_fail_op = normalized_depth_control.stencilfail;
        description_out.stencil_front_pass_op = normalized_depth_control.stencilzpass;
        description_out.stencil_front_depth_fail_op = normalized_depth_control.stencilzfail;
        description_out.stencil_front_compare_op = normalized_depth_control.stencilfunc;
        if (primitive_polygonal && normalized_depth_control.backface_enable) {
          description_out.stencil_back_fail_op = normalized_depth_control.stencilfail_bf;
          description_out.stencil_back_pass_op = normalized_depth_control.stencilzpass_bf;
          description_out.stencil_back_depth_fail_op = normalized_depth_control.stencilzfail_bf;
          description_out.stencil_back_compare_op = normalized_depth_control.stencilfunc_bf;
        } else {
          description_out.stencil_back_fail_op = description_out.stencil_front_fail_op;
          description_out.stencil_back_pass_op = description_out.stencil_front_pass_op;
          description_out.stencil_back_depth_fail_op = description_out.stencil_front_depth_fail_op;
          description_out.stencil_back_compare_op = description_out.stencil_front_compare_op;
        }
      }
    }

    // Color blending and write masks (filled only for the attachments present
    // in the render pass object).
    uint32_t render_pass_color_rts = render_pass_key.depth_and_color_used >> 1;
    assert_true(device_properties.independentBlend);
    uint32_t render_pass_color_rts_remaining = render_pass_color_rts;
    uint32_t color_rt_index;
    while (bit_inline::bit_scan_forward(render_pass_color_rts_remaining, &color_rt_index)) {
      render_pass_color_rts_remaining &= ~(uint32_t(1) << color_rt_index);
      WritePipelineRenderTargetDescription(
          regs.Get<reg::RB_BLENDCONTROL>(reg::RB_BLENDCONTROL::rt_register_indices[color_rt_index]),
          (normalized_color_mask >> (color_rt_index * 4)) & 0b1111,
          description_out.render_targets[color_rt_index]);
    }
  }

  return true;
}

// wwe2k14 patch (Session 93, idea 7): the Vulkan values of the description's cull / depth / stencil fields, by the formulas
// EnsurePipelineCreated builds the static pipeline state with (a pipeline with these fields dynamic draws as the static one would).
VulkanPipelineCache::EdsState VulkanPipelineCache::EdsStateOf(const PipelineDescription& description) {
  EdsState s;
  s.cull_mode = (description.cull_front ? uint32_t(VK_CULL_MODE_FRONT_BIT) : 0u) |
                (description.cull_back ? uint32_t(VK_CULL_MODE_BACK_BIT) : 0u);
  s.front_face = description.front_face_clockwise ? uint32_t(VK_FRONT_FACE_CLOCKWISE) : uint32_t(VK_FRONT_FACE_COUNTER_CLOCKWISE);
  if (description.depth_write_enable || description.depth_compare_op != xenos::CompareFunction::kAlways) {
    s.depth_test_enable = 1;
    s.depth_write_enable = description.depth_write_enable ? 1u : 0u;
    s.depth_compare_op = uint32_t(VK_COMPARE_OP_NEVER) + uint32_t(description.depth_compare_op);
  } else {
    s.depth_compare_op = uint32_t(VK_COMPARE_OP_ALWAYS);
  }
  if (description.stencil_test_enable) {
    s.stencil_test_enable = 1;
    s.stencil_ops[0][0] = uint32_t(VK_STENCIL_OP_KEEP) + uint32_t(description.stencil_front_fail_op);
    s.stencil_ops[0][1] = uint32_t(VK_STENCIL_OP_KEEP) + uint32_t(description.stencil_front_pass_op);
    s.stencil_ops[0][2] = uint32_t(VK_STENCIL_OP_KEEP) + uint32_t(description.stencil_front_depth_fail_op);
    s.stencil_ops[0][3] = uint32_t(VK_COMPARE_OP_NEVER) + uint32_t(description.stencil_front_compare_op);
    s.stencil_ops[1][0] = uint32_t(VK_STENCIL_OP_KEEP) + uint32_t(description.stencil_back_fail_op);
    s.stencil_ops[1][1] = uint32_t(VK_STENCIL_OP_KEEP) + uint32_t(description.stencil_back_pass_op);
    s.stencil_ops[1][2] = uint32_t(VK_STENCIL_OP_KEEP) + uint32_t(description.stencil_back_depth_fail_op);
    s.stencil_ops[1][3] = uint32_t(VK_COMPARE_OP_NEVER) + uint32_t(description.stencil_back_compare_op);
  }
  return s;
}

void VulkanPipelineCache::CollapseForEds(PipelineDescription& description) {
  const bool depth_used =
      description.pixel_shader_hash == 0 &&
      (description.depth_write_enable || description.depth_compare_op != xenos::CompareFunction::kAlways);
  description.cull_front = 0;
  description.cull_back = 0;
  description.front_face_clockwise = 0;
  description.depth_write_enable = 0;
  description.depth_compare_op = xenos::CompareFunction(0);
  description.stencil_test_enable = 0;
  description.stencil_front_fail_op = xenos::StencilOp(0);
  description.stencil_front_pass_op = xenos::StencilOp(0);
  description.stencil_front_depth_fail_op = xenos::StencilOp(0);
  description.stencil_front_compare_op = xenos::CompareFunction(0);
  description.stencil_back_fail_op = xenos::StencilOp(0);
  description.stencil_back_pass_op = xenos::StencilOp(0);
  description.stencil_back_depth_fail_op = xenos::StencilOp(0);
  description.stencil_back_compare_op = xenos::CompareFunction(0);
  description.eds_dynamic = 1;
  description.eds_depth_used = depth_used ? 1 : 0;
}

bool VulkanPipelineCache::GetCurrentStateDescription(
    const VulkanShader::VulkanTranslation* vertex_shader,
    const VulkanShader::VulkanTranslation* pixel_shader,
    const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
    reg::RB_DEPTHCONTROL normalized_depth_control, uint32_t normalized_color_mask,
    VulkanRenderTargetCache::RenderPassKey render_pass_key, PipelineDescription& description_out) {
  if (!GetCurrentStateDescriptionFull(vertex_shader, pixel_shader, primitive_processing_result,
                                      normalized_depth_control, normalized_color_mask, render_pass_key,
                                      description_out)) {
    return false;
  }
  if (!eds_active_) {
    return true;  // the key as it always was
  }
  current_eds_ = EdsStateOf(description_out);
  // The collapse statistics: each distinct old key counted once; one whose collapsed key was seen already is a pipeline saved.
  const uint64_t full_key = description_out.GetHash();
  if (full_key != eds_last_full_key_) {
    eds_last_full_key_ = full_key;
    if (eds_full_keys_.insert(full_key).second) {
      ++eds_stats_.full_keys_seen;
      PipelineDescription collapsed = description_out;
      CollapseForEds(collapsed);
      if (!eds_collapsed_keys_.insert(collapsed.GetHash()).second) {
        ++eds_stats_.keys_collapsed;
      }
    }
  }
  CollapseForEds(description_out);
  return true;
}

bool VulkanPipelineCache::ArePipelineRequirementsMet(const PipelineDescription& description) const {
  // wwe2k14 patch (Session 93, idea 7): a record made with the other extended-dynamic-state mode is another pipeline (its key says so).
  if ((description.eds_dynamic != 0) != eds_active_) {
    return false;
  }
  VkShaderStageFlags vertex_shader_stage =
      Shader::IsHostVertexShaderTypeDomain(
          SpirvShaderTranslator::Modification(description.vertex_shader_modification)
              .vertex.host_vertex_shader_type)
          ? VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT
          : VK_SHADER_STAGE_VERTEX_BIT;
  if (!(guest_shader_vertex_stages_ & vertex_shader_stage)) {
    return false;
  }
  // wwe2k14 patch (Session 24, DLSS): a stored LOD-bias variant only with gpu_dlss (no extra pipelines otherwise).
  if (SpirvShaderTranslator::Modification(description.pixel_shader_modification).pixel.dlss_lod_bias_quarters &&
      !REXCVAR_GET(gpu_dlss)) {
    return false;
  }
  // wwe2k14 patch (Session 24, OBJMV, review fix): a stored transform-feedback variant needs gpu_dlss_objmv latched
  // (and with it the device feature); a default run must not recreate it from the pipeline storage.
  if (SpirvShaderTranslator::Modification(description.vertex_shader_modification).vertex.xfb_position &&
      !command_processor_.ObjmvLatched()) {
    return false;
  }

  const ui::vulkan::VulkanDevice::Properties& device_properties =
      command_processor_.GetVulkanDevice()->properties();

  if (!device_properties.geometryShader &&
      description.geometry_shader != PipelineGeometryShader::kNone) {
    return false;
  }

  // Keep fan handling consistent with D3D12 by always using primitive
  // processor conversion to triangle lists instead of native fan topologies.
  if (description.primitive_topology == PipelinePrimitiveTopology::kTriangleFan) {
    return false;
  }

  if (!device_properties.depthClamp && description.depth_clamp_enable) {
    return false;
  }

  if (!device_properties.pointPolygons && description.polygon_mode == PipelinePolygonMode::kPoint) {
    return false;
  }

  if (!device_properties.fillModeNonSolid &&
      description.polygon_mode != PipelinePolygonMode::kFill) {
    return false;
  }

  assert_true(device_properties.independentBlend);

  if (!device_properties.constantAlphaColorBlendFactors) {
    uint32_t color_rts_remaining = description.render_pass_key.depth_and_color_used >> 1;
    uint32_t color_rt_index;
    while (rex::bit_scan_forward(color_rts_remaining, &color_rt_index)) {
      color_rts_remaining &= ~(uint32_t(1) << color_rt_index);
      const PipelineRenderTarget& color_rt = description.render_targets[color_rt_index];
      if (color_rt.src_color_blend_factor == PipelineBlendFactor::kConstantAlpha ||
          color_rt.src_color_blend_factor == PipelineBlendFactor::kOneMinusConstantAlpha ||
          color_rt.dst_color_blend_factor == PipelineBlendFactor::kConstantAlpha ||
          color_rt.dst_color_blend_factor == PipelineBlendFactor::kOneMinusConstantAlpha) {
        return false;
      }
    }
  }

  return true;
}

uint32_t VulkanPipelineCache::GetTessellationPatchControlPointCount(
    Shader::HostVertexShaderType host_vertex_shader_type,
    xenos::TessellationMode tessellation_mode) {
  switch (tessellation_mode) {
    case xenos::TessellationMode::kDiscrete:
    case xenos::TessellationMode::kContinuous:
      switch (host_vertex_shader_type) {
        case Shader::HostVertexShaderType::kTriangleDomainCPIndexed:
          return 3;
        case Shader::HostVertexShaderType::kTriangleDomainPatchIndexed:
          return 1;
        case Shader::HostVertexShaderType::kQuadDomainCPIndexed:
          return 4;
        case Shader::HostVertexShaderType::kQuadDomainPatchIndexed:
          return 1;
        default:
          return 0;
      }
    case xenos::TessellationMode::kAdaptive:
      switch (host_vertex_shader_type) {
        case Shader::HostVertexShaderType::kTriangleDomainPatchIndexed:
          return 3;
        case Shader::HostVertexShaderType::kQuadDomainPatchIndexed:
          return 4;
        default:
          return 0;
      }
    default:
      return 0;
  }
}

VkShaderModule VulkanPipelineCache::GetTessellationVertexShader(bool adaptive) {
  bool& attempted = adaptive ? tessellation_adaptive_vertex_shader_attempted_
                             : tessellation_indexed_vertex_shader_attempted_;
  VkShaderModule& shader_module =
      adaptive ? tessellation_adaptive_vertex_shader_ : tessellation_indexed_vertex_shader_;
  if (attempted) {
    return shader_module;
  }
  attempted = true;

  std::vector<uint32_t> shader_spirv;
  std::string compile_error;
  std::string source = adaptive ? GetTessellationAdaptiveVertexShaderGlsl()
                                : GetTessellationIndexedVertexShaderGlsl();
  if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_VERTEX_BIT, source, shader_spirv,
                                             compile_error)) {
    REXGPU_ERROR(
        "VulkanPipelineCache: Failed to compile tessellation {} vertex "
        "shader: {}",
        adaptive ? "adaptive" : "indexed", compile_error);
    shader_module = VK_NULL_HANDLE;
    return shader_module;
  }

  shader_module = ui::vulkan::util::CreateShaderModule(command_processor_.GetVulkanDevice(),
                                                       shader_spirv.data(),
                                                       sizeof(uint32_t) * shader_spirv.size());
  if (shader_module == VK_NULL_HANDLE) {
    REXGPU_ERROR(
        "VulkanPipelineCache: Failed to create tessellation {} vertex "
        "shader module",
        adaptive ? "adaptive" : "indexed");
  }
  return shader_module;
}

VkShaderModule VulkanPipelineCache::GetTessellationControlShader(
    Shader::HostVertexShaderType host_vertex_shader_type,
    xenos::TessellationMode tessellation_mode) {
  TessellationControlShaderKey key;
  key.host_vertex_shader_type = host_vertex_shader_type;
  key.tessellation_mode = tessellation_mode;
  auto it = tessellation_control_shaders_.find(key);
  if (it != tessellation_control_shaders_.end()) {
    return it->second;
  }

  uint32_t input_control_points =
      GetTessellationPatchControlPointCount(host_vertex_shader_type, tessellation_mode);
  uint32_t output_control_points = 0;
  if (tessellation_mode == xenos::TessellationMode::kAdaptive) {
    output_control_points = 1;
  } else if (IsPatchIndexedHostVertexShaderType(host_vertex_shader_type)) {
    output_control_points = 1;
  } else if (IsTriangleDomainHostVertexShaderType(host_vertex_shader_type)) {
    output_control_points = 3;
  } else if (IsQuadDomainHostVertexShaderType(host_vertex_shader_type)) {
    output_control_points = 4;
  }
  if (!input_control_points || !output_control_points) {
    tessellation_control_shaders_.emplace(key, VK_NULL_HANDLE);
    return VK_NULL_HANDLE;
  }

  std::vector<uint32_t> shader_spirv;
  std::string compile_error;
  std::string source = GetTessellationControlShaderGlsl(
      host_vertex_shader_type, tessellation_mode, input_control_points, output_control_points);
  if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, source,
                                             shader_spirv, compile_error)) {
    REXGPU_ERROR(
        "VulkanPipelineCache: Failed to compile tessellation control shader "
        "0x{:08X}: {}",
        key.key, compile_error);
    tessellation_control_shaders_.emplace(key, VK_NULL_HANDLE);
    return VK_NULL_HANDLE;
  }

  VkShaderModule shader_module = ui::vulkan::util::CreateShaderModule(
      command_processor_.GetVulkanDevice(), shader_spirv.data(),
      sizeof(uint32_t) * shader_spirv.size());
  if (shader_module == VK_NULL_HANDLE) {
    REXGPU_ERROR(
        "VulkanPipelineCache: Failed to create tessellation control shader "
        "module 0x{:08X}",
        key.key);
  }
  tessellation_control_shaders_.emplace(key, shader_module);
  return shader_module;
}

bool VulkanPipelineCache::GetGeometryShaderKey(
    PipelineGeometryShader geometry_shader_type,
    SpirvShaderTranslator::Modification vertex_shader_modification,
    SpirvShaderTranslator::Modification pixel_shader_modification, GeometryShaderKey& key_out) {
  if (geometry_shader_type == PipelineGeometryShader::kNone) {
    return false;
  }
  // For kPointListAsTriangleStrip, output_point_parameters has a different
  // meaning (the coordinates, not the size). However, the AsTriangleStrip host
  // vertex shader types are needed specifically when geometry shaders are not
  // supported as fallbacks.
  if (vertex_shader_modification.vertex.host_vertex_shader_type ==
          Shader::HostVertexShaderType::kPointListAsTriangleStrip ||
      vertex_shader_modification.vertex.host_vertex_shader_type ==
          Shader::HostVertexShaderType::kRectangleListAsTriangleStrip) {
    assert_always();
    return false;
  }
  GeometryShaderKey key;
  key.type = geometry_shader_type;
  key.interpolator_count = rex::bit_count(vertex_shader_modification.vertex.interpolator_mask);
  key.user_clip_plane_count = vertex_shader_modification.vertex.user_clip_plane_count;
  key.user_clip_plane_cull = vertex_shader_modification.vertex.user_clip_plane_cull;
  key.has_vertex_kill_and = vertex_shader_modification.vertex.vertex_kill_and;
  key.has_point_size = vertex_shader_modification.vertex.output_point_parameters;
  key.has_point_coordinates = pixel_shader_modification.pixel.param_gen_point;
  key.point_ps_ucp_mode = vertex_shader_modification.vertex.point_ps_ucp_mode;
  key_out = key;
  return true;
}

VkShaderModule VulkanPipelineCache::GetGeometryShader(GeometryShaderKey key) {
  auto it = geometry_shaders_.find(key);
  if (it != geometry_shaders_.end()) {
    return it->second;
  }

  std::vector<spv::Id> id_vector_temp;
  std::vector<unsigned int> uint_vector_temp;

  spv::ExecutionMode input_primitive_execution_mode = spv::ExecutionMode(0);
  uint32_t input_primitive_vertex_count = 0;
  spv::ExecutionMode output_primitive_execution_mode = spv::ExecutionMode(0);
  uint32_t output_max_vertices = 0;
  switch (key.type) {
    case PipelineGeometryShader::kPointList:
      // Point to a strip of 2 triangles.
      input_primitive_execution_mode = spv::ExecutionModeInputPoints;
      input_primitive_vertex_count = 1;
      output_primitive_execution_mode = spv::ExecutionModeOutputTriangleStrip;
      output_max_vertices = 4;
      break;
    case PipelineGeometryShader::kRectangleList:
      // Triangle to a strip of 2 triangles.
      input_primitive_execution_mode = spv::ExecutionModeTriangles;
      input_primitive_vertex_count = 3;
      output_primitive_execution_mode = spv::ExecutionModeOutputTriangleStrip;
      output_max_vertices = 4;
      break;
    case PipelineGeometryShader::kQuadList:
      // 4 vertices passed via a line list with adjacency to a strip of 2
      // triangles.
      input_primitive_execution_mode = spv::ExecutionModeInputLinesAdjacency;
      input_primitive_vertex_count = 4;
      output_primitive_execution_mode = spv::ExecutionModeOutputTriangleStrip;
      output_max_vertices = 4;
      break;
    default:
      assert_unhandled_case(key.type);
  }

  uint32_t clip_distance_count = key.user_clip_plane_cull ? 0 : key.user_clip_plane_count;
  uint32_t point_user_cull_distance_count =
      key.type == PipelineGeometryShader::kPointList && key.user_clip_plane_cull
          ? key.user_clip_plane_count
          : 0;
  uint32_t cull_distance_count = point_user_cull_distance_count + key.has_vertex_kill_and;
  bool point_recalculate_clip_distances = key.type == PipelineGeometryShader::kPointList &&
                                          clip_distance_count && key.point_ps_ucp_mode >= 2;
  bool point_recalculate_cull_distances = key.type == PipelineGeometryShader::kPointList &&
                                          point_user_cull_distance_count &&
                                          key.point_ps_ucp_mode >= 3;

  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Properties& device_properties = vulkan_device->properties();
  const ui::vulkan::VulkanDevice::Extensions& device_extensions = vulkan_device->extensions();
  spv::SpvBuildLogger builder_logger;
  spv::SpvBuildLogger* builder_logger_ptr = nullptr;
  spv::SpvVersion spirv_version;
  if (device_properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 2, 0)) {
    spirv_version = spv::Spv_1_5;
  } else if (device_extensions.ext_1_2_KHR_spirv_1_4) {
    spirv_version = spv::Spv_1_4;
  } else if (device_properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0)) {
    spirv_version = spv::Spv_1_3;
  } else {
    spirv_version = spv::Spv_1_0;
    // Keep the build log around for compatibility diagnostics on older paths.
    builder_logger_ptr = &builder_logger;
  }
  SpirvBuilder builder(spirv_version, (SpirvShaderTranslator::kSpirvMagicToolId << 16) | 1,
                       builder_logger_ptr);
  spv::Id ext_inst_glsl_std_450 = builder.import("GLSL.std.450");
  builder.addCapability(spv::CapabilityGeometry);
  if (clip_distance_count) {
    builder.addCapability(spv::CapabilityClipDistance);
  }
  if (cull_distance_count) {
    builder.addCapability(spv::CapabilityCullDistance);
  }
  bool denorm_flush_to_zero_float32 = device_properties.shaderDenormFlushToZeroFloat32;
  bool signed_zero_inf_nan_preserve_float32 =
      device_properties.shaderSignedZeroInfNanPreserveFloat32;
  bool rounding_mode_rte_float32 = device_properties.shaderRoundingModeRTEFloat32;
  if (spirv_version < spv::Spv_1_4) {
    if (denorm_flush_to_zero_float32 || signed_zero_inf_nan_preserve_float32 ||
        rounding_mode_rte_float32) {
      builder.addExtension("SPV_KHR_float_controls");
    }
  }
  builder.setMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
  builder.setSource(spv::SourceLanguageUnknown, 0);

  std::vector<spv::Id> main_interface;

  spv::Id type_void = builder.makeVoidType();
  spv::Id type_bool = builder.makeBoolType();
  spv::Id type_bool4 = builder.makeVectorType(type_bool, 4);
  spv::Id type_int = builder.makeIntType(32);
  spv::Id type_float = builder.makeFloatType(32);
  spv::Id type_float2 = builder.makeVectorType(type_float, 2);
  spv::Id type_float3 = builder.makeVectorType(type_float, 3);
  spv::Id type_float4 = builder.makeVectorType(type_float, 4);
  spv::Id type_float4_array_6 =
      builder.makeArrayType(type_float4, builder.makeUintConstant(6), sizeof(float) * 4);
  builder.addDecoration(type_float4_array_6, spv::DecorationArrayStride, sizeof(float) * 4);
  spv::Id type_clip_distances =
      clip_distance_count
          ? builder.makeArrayType(type_float, builder.makeUintConstant(clip_distance_count), 0)
          : spv::NoType;
  spv::Id type_cull_distances =
      cull_distance_count
          ? builder.makeArrayType(type_float, builder.makeUintConstant(cull_distance_count), 0)
          : spv::NoType;

  // System constants.
  // For points:
  // - float2 point_constant_diameter
  // - float2 point_screen_diameter_to_ndc_radius
  // - float3 ndc_scale
  // - float3 ndc_offset
  // - float4[6] user_clip_planes
  enum PointConstant : uint32_t {
    kPointConstantConstantDiameter,
    kPointConstantScreenDiameterToNdcRadius,
    kPointConstantNdcScale,
    kPointConstantNdcOffset,
    kPointConstantUserClipPlanes,
    kPointConstantCount,
  };
  spv::Id type_system_constants = spv::NoType;
  if (key.type == PipelineGeometryShader::kPointList) {
    id_vector_temp.clear();
    id_vector_temp.resize(kPointConstantCount);
    id_vector_temp[kPointConstantConstantDiameter] = type_float2;
    id_vector_temp[kPointConstantScreenDiameterToNdcRadius] = type_float2;
    id_vector_temp[kPointConstantNdcScale] = type_float3;
    id_vector_temp[kPointConstantNdcOffset] = type_float3;
    id_vector_temp[kPointConstantUserClipPlanes] = type_float4_array_6;
    type_system_constants = builder.makeStructType(id_vector_temp, "XeSystemConstants");
    builder.addMemberName(type_system_constants, kPointConstantConstantDiameter,
                          "point_constant_diameter");
    builder.addMemberDecoration(
        type_system_constants, kPointConstantConstantDiameter, spv::DecorationOffset,
        int(offsetof(SpirvShaderTranslator::SystemConstants, point_constant_diameter)));
    builder.addMemberName(type_system_constants, kPointConstantScreenDiameterToNdcRadius,
                          "point_screen_diameter_to_ndc_radius");
    builder.addMemberDecoration(
        type_system_constants, kPointConstantScreenDiameterToNdcRadius, spv::DecorationOffset,
        int(offsetof(SpirvShaderTranslator::SystemConstants, point_screen_diameter_to_ndc_radius)));
    builder.addMemberName(type_system_constants, kPointConstantNdcScale, "ndc_scale");
    builder.addMemberDecoration(type_system_constants, kPointConstantNdcScale,
                                spv::DecorationOffset,
                                int(offsetof(SpirvShaderTranslator::SystemConstants, ndc_scale)));
    builder.addMemberName(type_system_constants, kPointConstantNdcOffset, "ndc_offset");
    builder.addMemberDecoration(type_system_constants, kPointConstantNdcOffset,
                                spv::DecorationOffset,
                                int(offsetof(SpirvShaderTranslator::SystemConstants, ndc_offset)));
    builder.addMemberName(type_system_constants, kPointConstantUserClipPlanes, "user_clip_planes");
    builder.addMemberDecoration(
        type_system_constants, kPointConstantUserClipPlanes, spv::DecorationOffset,
        int(offsetof(SpirvShaderTranslator::SystemConstants, user_clip_planes)));
  }
  spv::Id uniform_system_constants = spv::NoResult;
  if (type_system_constants != spv::NoType) {
    builder.addDecoration(type_system_constants, spv::DecorationBlock);
    uniform_system_constants =
        builder.createVariable(spv::NoPrecision, spv::StorageClassUniform, type_system_constants,
                               "xe_uniform_system_constants");
    builder.addDecoration(uniform_system_constants, spv::DecorationDescriptorSet,
                          int(SpirvShaderTranslator::kDescriptorSetConstants));
    builder.addDecoration(uniform_system_constants, spv::DecorationBinding,
                          int(SpirvShaderTranslator::kConstantBufferSystem));
    main_interface.push_back(uniform_system_constants);
  }

  // Inputs and outputs - matching glslang order, in gl_PerVertex gl_in[],
  // user-defined outputs, user-defined inputs, out gl_PerVertex.

  spv::Id const_input_primitive_vertex_count =
      builder.makeUintConstant(input_primitive_vertex_count);

  // in gl_PerVertex gl_in[].
  // gl_Position.
  id_vector_temp.clear();
  uint32_t member_in_gl_per_vertex_position = uint32_t(id_vector_temp.size());
  id_vector_temp.push_back(type_float4);
  spv::Id const_member_in_gl_per_vertex_position =
      builder.makeIntConstant(int32_t(member_in_gl_per_vertex_position));
  // gl_ClipDistance.
  uint32_t member_in_gl_per_vertex_clip_distance = UINT32_MAX;
  spv::Id const_member_in_gl_per_vertex_clip_distance = spv::NoResult;
  if (clip_distance_count) {
    member_in_gl_per_vertex_clip_distance = uint32_t(id_vector_temp.size());
    id_vector_temp.push_back(type_clip_distances);
    const_member_in_gl_per_vertex_clip_distance =
        builder.makeIntConstant(int32_t(member_in_gl_per_vertex_clip_distance));
  }
  // gl_CullDistance.
  uint32_t member_in_gl_per_vertex_cull_distance = UINT32_MAX;
  if (cull_distance_count) {
    member_in_gl_per_vertex_cull_distance = uint32_t(id_vector_temp.size());
    id_vector_temp.push_back(type_cull_distances);
  }
  // Structure and array.
  spv::Id type_struct_in_gl_per_vertex = builder.makeStructType(id_vector_temp, "gl_PerVertex");
  builder.addMemberName(type_struct_in_gl_per_vertex, member_in_gl_per_vertex_position,
                        "gl_Position");
  builder.addMemberDecoration(type_struct_in_gl_per_vertex, member_in_gl_per_vertex_position,
                              spv::DecorationBuiltIn, spv::BuiltInPosition);
  if (clip_distance_count) {
    builder.addMemberName(type_struct_in_gl_per_vertex, member_in_gl_per_vertex_clip_distance,
                          "gl_ClipDistance");
    builder.addMemberDecoration(type_struct_in_gl_per_vertex, member_in_gl_per_vertex_clip_distance,
                                spv::DecorationBuiltIn, spv::BuiltInClipDistance);
  }
  if (cull_distance_count) {
    builder.addMemberName(type_struct_in_gl_per_vertex, member_in_gl_per_vertex_cull_distance,
                          "gl_CullDistance");
    builder.addMemberDecoration(type_struct_in_gl_per_vertex, member_in_gl_per_vertex_cull_distance,
                                spv::DecorationBuiltIn, spv::BuiltInCullDistance);
  }
  builder.addDecoration(type_struct_in_gl_per_vertex, spv::DecorationBlock);
  spv::Id type_array_in_gl_per_vertex =
      builder.makeArrayType(type_struct_in_gl_per_vertex, const_input_primitive_vertex_count, 0);
  spv::Id in_gl_per_vertex = builder.createVariable(spv::NoPrecision, spv::StorageClassInput,
                                                    type_array_in_gl_per_vertex, "gl_in");
  main_interface.push_back(in_gl_per_vertex);

  uint32_t output_location = 0;

  // Interpolators outputs.
  std::array<spv::Id, xenos::kMaxInterpolators> out_interpolators;
  for (uint32_t i = 0; i < key.interpolator_count; ++i) {
    spv::Id out_interpolator =
        builder.createVariable(spv::NoPrecision, spv::StorageClassOutput, type_float4,
                               fmt::format("xe_out_interpolator_{}", i).c_str());
    out_interpolators[i] = out_interpolator;
    builder.addDecoration(out_interpolator, spv::DecorationLocation, int(output_location));
    builder.addDecoration(out_interpolator, spv::DecorationInvariant);
    main_interface.push_back(out_interpolator);
    ++output_location;
  }

  // Point coordinate output.
  spv::Id out_point_coordinates = spv::NoResult;
  if (key.has_point_coordinates) {
    out_point_coordinates = builder.createVariable(spv::NoPrecision, spv::StorageClassOutput,
                                                   type_float2, "xe_out_point_coordinates");
    builder.addDecoration(out_point_coordinates, spv::DecorationLocation, int(output_location));
    builder.addDecoration(out_point_coordinates, spv::DecorationInvariant);
    main_interface.push_back(out_point_coordinates);
    ++output_location;
  }

  uint32_t input_location = 0;

  // Interpolator inputs.
  std::array<spv::Id, xenos::kMaxInterpolators> in_interpolators;
  for (uint32_t i = 0; i < key.interpolator_count; ++i) {
    spv::Id in_interpolator = builder.createVariable(
        spv::NoPrecision, spv::StorageClassInput,
        builder.makeArrayType(type_float4, const_input_primitive_vertex_count, 0),
        fmt::format("xe_in_interpolator_{}", i).c_str());
    in_interpolators[i] = in_interpolator;
    builder.addDecoration(in_interpolator, spv::DecorationLocation, int(input_location));
    main_interface.push_back(in_interpolator);
    ++input_location;
  }

  // Point size input.
  spv::Id in_point_size = spv::NoResult;
  if (key.has_point_size) {
    in_point_size = builder.createVariable(
        spv::NoPrecision, spv::StorageClassInput,
        builder.makeArrayType(type_float, const_input_primitive_vertex_count, 0),
        "xe_in_point_size");
    builder.addDecoration(in_point_size, spv::DecorationLocation, int(input_location));
    main_interface.push_back(in_point_size);
    ++input_location;
  }

  // out gl_PerVertex.
  // gl_Position.
  id_vector_temp.clear();
  uint32_t member_out_gl_per_vertex_position = uint32_t(id_vector_temp.size());
  id_vector_temp.push_back(type_float4);
  spv::Id const_member_out_gl_per_vertex_position =
      builder.makeIntConstant(int32_t(member_out_gl_per_vertex_position));
  // gl_ClipDistance.
  uint32_t member_out_gl_per_vertex_clip_distance = UINT32_MAX;
  spv::Id const_member_out_gl_per_vertex_clip_distance = spv::NoResult;
  if (clip_distance_count) {
    member_out_gl_per_vertex_clip_distance = uint32_t(id_vector_temp.size());
    id_vector_temp.push_back(type_clip_distances);
    const_member_out_gl_per_vertex_clip_distance =
        builder.makeIntConstant(int32_t(member_out_gl_per_vertex_clip_distance));
  }
  // Structure.
  spv::Id type_struct_out_gl_per_vertex = builder.makeStructType(id_vector_temp, "gl_PerVertex");
  builder.addMemberName(type_struct_out_gl_per_vertex, member_out_gl_per_vertex_position,
                        "gl_Position");
  builder.addMemberDecoration(type_struct_out_gl_per_vertex, member_out_gl_per_vertex_position,
                              spv::DecorationBuiltIn, spv::BuiltInPosition);
  if (clip_distance_count) {
    builder.addMemberName(type_struct_out_gl_per_vertex, member_out_gl_per_vertex_clip_distance,
                          "gl_ClipDistance");
    builder.addMemberDecoration(type_struct_out_gl_per_vertex,
                                member_out_gl_per_vertex_clip_distance, spv::DecorationBuiltIn,
                                spv::BuiltInClipDistance);
  }
  builder.addDecoration(type_struct_out_gl_per_vertex, spv::DecorationBlock);
  spv::Id out_gl_per_vertex = builder.createVariable(spv::NoPrecision, spv::StorageClassOutput,
                                                     type_struct_out_gl_per_vertex, "");
  builder.addDecoration(out_gl_per_vertex, spv::DecorationInvariant);
  main_interface.push_back(out_gl_per_vertex);

  // Begin the main function.
  std::vector<spv::Id> main_param_types;
  std::vector<std::vector<spv::Decoration>> main_precisions;
  spv::Block* main_entry;
  spv::Function* main_function = builder.makeFunctionEntry(
      spv::NoPrecision, type_void, "main", main_param_types, main_precisions, &main_entry);
  spv::Instruction* entry_point =
      builder.addEntryPoint(spv::ExecutionModelGeometry, main_function, "main");
  for (spv::Id interface_id : main_interface) {
    entry_point->addIdOperand(interface_id);
  }
  builder.addExecutionMode(main_function, input_primitive_execution_mode);
  builder.addExecutionMode(main_function, spv::ExecutionModeInvocations, 1);
  builder.addExecutionMode(main_function, output_primitive_execution_mode);
  builder.addExecutionMode(main_function, spv::ExecutionModeOutputVertices,
                           int(output_max_vertices));
  if (denorm_flush_to_zero_float32) {
    builder.addCapability(spv::CapabilityDenormFlushToZero);
    builder.addExecutionMode(main_function, spv::ExecutionModeDenormFlushToZero, 32);
  }
  if (signed_zero_inf_nan_preserve_float32) {
    builder.addCapability(spv::CapabilitySignedZeroInfNanPreserve);
    builder.addExecutionMode(main_function, spv::ExecutionModeSignedZeroInfNanPreserve, 32);
  }
  if (rounding_mode_rte_float32) {
    builder.addCapability(spv::CapabilityRoundingModeRTE);
    builder.addExecutionMode(main_function, spv::ExecutionModeRoundingModeRTE, 32);
  }

  // Note that after every OpEmitVertex, all output variables are undefined.

  // Discard the whole primitive if any vertex has a NaN position (may also be
  // set to NaN for emulation of vertex killing with the OR operator).
  for (uint32_t i = 0; i < input_primitive_vertex_count; ++i) {
    id_vector_temp.clear();
    id_vector_temp.push_back(builder.makeIntConstant(int32_t(i)));
    id_vector_temp.push_back(const_member_in_gl_per_vertex_position);
    spv::Id position_is_nan = builder.createUnaryOp(
        spv::OpAny, type_bool,
        builder.createUnaryOp(
            spv::OpIsNan, type_bool4,
            builder.createLoad(
                builder.createAccessChain(spv::StorageClassInput, in_gl_per_vertex, id_vector_temp),
                spv::NoPrecision)));
    spv::Block& discard_predecessor = *builder.getBuildPoint();
    spv::Block& discard_then_block = builder.makeNewBlock();
    spv::Block& discard_merge_block = builder.makeNewBlock();
    builder.createSelectionMerge(&discard_merge_block, spv::SelectionControlDontFlattenMask);
    {
      std::unique_ptr<spv::Instruction> branch_conditional_op(
          std::make_unique<spv::Instruction>(spv::OpBranchConditional));
      branch_conditional_op->addIdOperand(position_is_nan);
      branch_conditional_op->addIdOperand(discard_then_block.getId());
      branch_conditional_op->addIdOperand(discard_merge_block.getId());
      branch_conditional_op->addImmediateOperand(1);
      branch_conditional_op->addImmediateOperand(2);
      discard_predecessor.addInstruction(std::move(branch_conditional_op));
    }
    discard_then_block.addPredecessor(&discard_predecessor);
    discard_merge_block.addPredecessor(&discard_predecessor);
    builder.setBuildPoint(&discard_then_block);
    builder.createNoResultOp(spv::OpReturn);
    builder.setBuildPoint(&discard_merge_block);
  }

  // Cull the whole primitive if any cull distance for all vertices in the
  // primitive is < 0.
  // For point lists with ps_ucp_mode 3, user cull plane distances are
  // calculated per expanded vertex later.
  if (cull_distance_count) {
    spv::Id const_member_in_gl_per_vertex_cull_distance =
        builder.makeIntConstant(int32_t(member_in_gl_per_vertex_cull_distance));
    spv::Id const_float_0 = builder.makeFloatConstant(0.0f);
    spv::Id cull_condition = spv::NoResult;
    uint32_t cull_distance_start =
        point_recalculate_cull_distances ? point_user_cull_distance_count : 0;
    if (cull_distance_start < cull_distance_count) {
      for (uint32_t i = cull_distance_start; i < cull_distance_count; ++i) {
        for (uint32_t j = 0; j < input_primitive_vertex_count; ++j) {
          id_vector_temp.clear();
          id_vector_temp.push_back(builder.makeIntConstant(int32_t(j)));
          id_vector_temp.push_back(const_member_in_gl_per_vertex_cull_distance);
          id_vector_temp.push_back(builder.makeIntConstant(int32_t(i)));
          spv::Id cull_distance_is_negative = builder.createBinOp(
              spv::OpFOrdLessThan, type_bool,
              builder.createLoad(builder.createAccessChain(spv::StorageClassInput, in_gl_per_vertex,
                                                           id_vector_temp),
                                 spv::NoPrecision),
              const_float_0);
          if (cull_condition != spv::NoResult) {
            cull_condition = builder.createBinOp(spv::OpLogicalAnd, type_bool, cull_condition,
                                                 cull_distance_is_negative);
          } else {
            cull_condition = cull_distance_is_negative;
          }
        }
      }
      assert_true(cull_condition != spv::NoResult);
      spv::Block& discard_predecessor = *builder.getBuildPoint();
      spv::Block& discard_then_block = builder.makeNewBlock();
      spv::Block& discard_merge_block = builder.makeNewBlock();
      builder.createSelectionMerge(&discard_merge_block, spv::SelectionControlDontFlattenMask);
      {
        std::unique_ptr<spv::Instruction> branch_conditional_op(
            std::make_unique<spv::Instruction>(spv::OpBranchConditional));
        branch_conditional_op->addIdOperand(cull_condition);
        branch_conditional_op->addIdOperand(discard_then_block.getId());
        branch_conditional_op->addIdOperand(discard_merge_block.getId());
        branch_conditional_op->addImmediateOperand(1);
        branch_conditional_op->addImmediateOperand(2);
        discard_predecessor.addInstruction(std::move(branch_conditional_op));
      }
      discard_then_block.addPredecessor(&discard_predecessor);
      discard_merge_block.addPredecessor(&discard_predecessor);
      builder.setBuildPoint(&discard_then_block);
      builder.createNoResultOp(spv::OpReturn);
      builder.setBuildPoint(&discard_merge_block);
    }
  }

  switch (key.type) {
    case PipelineGeometryShader::kPointList: {
      // Expand the point sprite, with left-to-right, top-to-bottom UVs.

      spv::Id const_int_0 = builder.makeIntConstant(0);
      spv::Id const_int_1 = builder.makeIntConstant(1);
      spv::Id const_float_0 = builder.makeFloatConstant(0.0f);

      // Load the point diameter in guest pixels.
      id_vector_temp.clear();
      id_vector_temp.push_back(builder.makeIntConstant(int32_t(kPointConstantConstantDiameter)));
      id_vector_temp.push_back(const_int_0);
      spv::Id point_guest_diameter_x =
          builder.createLoad(builder.createAccessChain(spv::StorageClassUniform,
                                                       uniform_system_constants, id_vector_temp),
                             spv::NoPrecision);
      id_vector_temp.back() = const_int_1;
      spv::Id point_guest_diameter_y =
          builder.createLoad(builder.createAccessChain(spv::StorageClassUniform,
                                                       uniform_system_constants, id_vector_temp),
                             spv::NoPrecision);
      if (key.has_point_size) {
        // The vertex shader's header writes -1.0 to point_size by default, so
        // any non-negative value means that it was overwritten by the
        // translated vertex shader, and needs to be used instead of the
        // constant size. The per-vertex diameter is already clamped in the
        // vertex shader (combined with making it non-negative).
        id_vector_temp.clear();
        // 0 is the input primitive vertex index.
        id_vector_temp.push_back(const_int_0);
        spv::Id point_vertex_diameter = builder.createLoad(
            builder.createAccessChain(spv::StorageClassInput, in_point_size, id_vector_temp),
            spv::NoPrecision);
        spv::Id point_vertex_diameter_written = builder.createBinOp(
            spv::OpFOrdGreaterThanEqual, type_bool, point_vertex_diameter, const_float_0);
        point_guest_diameter_x =
            builder.createTriOp(spv::OpSelect, type_float, point_vertex_diameter_written,
                                point_vertex_diameter, point_guest_diameter_x);
        point_guest_diameter_y =
            builder.createTriOp(spv::OpSelect, type_float, point_vertex_diameter_written,
                                point_vertex_diameter, point_guest_diameter_y);
      }

      // 4D5307F1 has zero-size snowflakes, drop them quicker, and also drop
      // points with a constant size of zero since point lists may also be used
      // as just "compute" with memexport.
      spv::Id point_size_not_zero =
          builder.createBinOp(spv::OpLogicalAnd, type_bool,
                              builder.createBinOp(spv::OpFOrdGreaterThan, type_bool,
                                                  point_guest_diameter_x, const_float_0),
                              builder.createBinOp(spv::OpFOrdGreaterThan, type_bool,
                                                  point_guest_diameter_y, const_float_0));
      spv::Block& point_size_zero_predecessor = *builder.getBuildPoint();
      spv::Block& point_size_zero_then_block = builder.makeNewBlock();
      spv::Block& point_size_zero_merge_block = builder.makeNewBlock();
      builder.createSelectionMerge(&point_size_zero_merge_block,
                                   spv::SelectionControlDontFlattenMask);
      {
        std::unique_ptr<spv::Instruction> branch_conditional_op(
            std::make_unique<spv::Instruction>(spv::OpBranchConditional));
        branch_conditional_op->addIdOperand(point_size_not_zero);
        branch_conditional_op->addIdOperand(point_size_zero_merge_block.getId());
        branch_conditional_op->addIdOperand(point_size_zero_then_block.getId());
        branch_conditional_op->addImmediateOperand(2);
        branch_conditional_op->addImmediateOperand(1);
        point_size_zero_predecessor.addInstruction(std::move(branch_conditional_op));
      }
      point_size_zero_then_block.addPredecessor(&point_size_zero_predecessor);
      point_size_zero_merge_block.addPredecessor(&point_size_zero_predecessor);
      builder.setBuildPoint(&point_size_zero_then_block);
      builder.createNoResultOp(spv::OpReturn);
      builder.setBuildPoint(&point_size_zero_merge_block);

      // Transform the diameter in the guest screen coordinates to radius in the
      // normalized device coordinates, and then to the clip space by
      // multiplying by W.
      id_vector_temp.clear();
      id_vector_temp.push_back(
          builder.makeIntConstant(int32_t(kPointConstantScreenDiameterToNdcRadius)));
      id_vector_temp.push_back(const_int_0);
      spv::Id point_radius_x = builder.createNoContractionBinOp(
          spv::OpFMul, type_float, point_guest_diameter_x,
          builder.createLoad(builder.createAccessChain(spv::StorageClassUniform,
                                                       uniform_system_constants, id_vector_temp),
                             spv::NoPrecision));
      id_vector_temp.back() = const_int_1;
      spv::Id point_radius_y = builder.createNoContractionBinOp(
          spv::OpFMul, type_float, point_guest_diameter_y,
          builder.createLoad(builder.createAccessChain(spv::StorageClassUniform,
                                                       uniform_system_constants, id_vector_temp),
                             spv::NoPrecision));
      id_vector_temp.clear();
      // 0 is the input primitive vertex index.
      id_vector_temp.push_back(const_int_0);
      id_vector_temp.push_back(const_member_in_gl_per_vertex_position);
      spv::Id point_position = builder.createLoad(
          builder.createAccessChain(spv::StorageClassInput, in_gl_per_vertex, id_vector_temp),
          spv::NoPrecision);
      spv::Id point_w = builder.createCompositeExtract(point_position, type_float, 3);
      point_radius_x =
          builder.createNoContractionBinOp(spv::OpFMul, type_float, point_radius_x, point_w);
      point_radius_y =
          builder.createNoContractionBinOp(spv::OpFMul, type_float, point_radius_y, point_w);

      // Load the inputs for the guest point.
      // Interpolators.
      std::array<spv::Id, xenos::kMaxInterpolators> point_interpolators;
      id_vector_temp.clear();
      // 0 is the input primitive vertex index.
      id_vector_temp.push_back(const_int_0);
      for (uint32_t i = 0; i < key.interpolator_count; ++i) {
        point_interpolators[i] = builder.createLoad(
            builder.createAccessChain(spv::StorageClassInput, in_interpolators[i], id_vector_temp),
            spv::NoPrecision);
      }
      // Positions.
      spv::Id point_x = builder.createCompositeExtract(point_position, type_float, 0);
      spv::Id point_y = builder.createCompositeExtract(point_position, type_float, 1);
      std::array<spv::Id, 2> point_edge_x, point_edge_y;
      for (uint32_t i = 0; i < 2; ++i) {
        spv::Op point_radius_add_op = i ? spv::OpFAdd : spv::OpFSub;
        point_edge_x[i] = builder.createNoContractionBinOp(point_radius_add_op, type_float, point_x,
                                                           point_radius_x);
        point_edge_y[i] = builder.createNoContractionBinOp(point_radius_add_op, type_float, point_y,
                                                           point_radius_y);
      };
      spv::Id point_z = builder.createCompositeExtract(point_position, type_float, 2);
      // Clip distances.
      spv::Id point_clip_distances = spv::NoResult;
      std::vector<spv::Id> point_user_clip_planes;
      point_user_clip_planes.reserve(clip_distance_count);
      spv::Id point_ndc_scale = spv::NoResult;
      spv::Id point_ndc_offset = spv::NoResult;
      if (point_recalculate_clip_distances || point_recalculate_cull_distances) {
        id_vector_temp.clear();
        id_vector_temp.push_back(builder.makeIntConstant(int32_t(kPointConstantNdcScale)));
        point_ndc_scale =
            builder.createLoad(builder.createAccessChain(spv::StorageClassUniform,
                                                         uniform_system_constants, id_vector_temp),
                               spv::NoPrecision);
        id_vector_temp.clear();
        id_vector_temp.push_back(builder.makeIntConstant(int32_t(kPointConstantNdcOffset)));
        point_ndc_offset =
            builder.createLoad(builder.createAccessChain(spv::StorageClassUniform,
                                                         uniform_system_constants, id_vector_temp),
                               spv::NoPrecision);
        uint32_t point_user_clip_plane_count =
            point_recalculate_cull_distances ? point_user_cull_distance_count : clip_distance_count;
        for (uint32_t i = 0; i < point_user_clip_plane_count; ++i) {
          id_vector_temp.clear();
          id_vector_temp.push_back(builder.makeIntConstant(int32_t(kPointConstantUserClipPlanes)));
          id_vector_temp.push_back(builder.makeIntConstant(int32_t(i)));
          point_user_clip_planes.push_back(builder.createLoad(
              builder.createAccessChain(spv::StorageClassUniform, uniform_system_constants,
                                        id_vector_temp),
              spv::NoPrecision));
        }
      } else if (clip_distance_count) {
        id_vector_temp.clear();
        // 0 is the input primitive vertex index.
        id_vector_temp.push_back(const_int_0);
        id_vector_temp.push_back(const_member_in_gl_per_vertex_clip_distance);
        point_clip_distances = builder.createLoad(
            builder.createAccessChain(spv::StorageClassInput, in_gl_per_vertex, id_vector_temp),
            spv::NoPrecision);
      }

      if (point_recalculate_cull_distances) {
        // Cull if any user clip plane has all expanded vertices outside.
        std::array<spv::Id, 4> point_guest_positions;
        for (uint32_t i = 0; i < 4; ++i) {
          uint32_t point_vertex_x = i >> 1;
          uint32_t point_vertex_y = i & 1;
          id_vector_temp.clear();
          id_vector_temp.push_back(point_edge_x[point_vertex_x]);
          id_vector_temp.push_back(point_edge_y[point_vertex_y]);
          id_vector_temp.push_back(point_z);
          id_vector_temp.push_back(point_w);
          spv::Id point_vertex_position =
              builder.createCompositeConstruct(type_float4, id_vector_temp);
          id_vector_temp.clear();
          id_vector_temp.push_back(
              builder.createCompositeExtract(point_vertex_position, type_float, 0));
          id_vector_temp.push_back(
              builder.createCompositeExtract(point_vertex_position, type_float, 1));
          id_vector_temp.push_back(
              builder.createCompositeExtract(point_vertex_position, type_float, 2));
          spv::Id point_vertex_position_xyz =
              builder.createCompositeConstruct(type_float3, id_vector_temp);
          spv::Id point_guest_position_xyz = builder.createNoContractionBinOp(
              spv::OpFSub, type_float3, point_vertex_position_xyz,
              builder.createNoContractionBinOp(spv::OpVectorTimesScalar, type_float3,
                                               point_ndc_offset, point_w));
          point_guest_position_xyz = builder.createNoContractionBinOp(
              spv::OpFDiv, type_float3, point_guest_position_xyz, point_ndc_scale);
          id_vector_temp.clear();
          id_vector_temp.push_back(
              builder.createCompositeExtract(point_guest_position_xyz, type_float, 0));
          id_vector_temp.push_back(
              builder.createCompositeExtract(point_guest_position_xyz, type_float, 1));
          id_vector_temp.push_back(
              builder.createCompositeExtract(point_guest_position_xyz, type_float, 2));
          id_vector_temp.push_back(point_w);
          point_guest_positions[i] = builder.createCompositeConstruct(type_float4, id_vector_temp);
        }
        spv::Id point_cull_condition = spv::NoResult;
        for (uint32_t i = 0; i < point_user_cull_distance_count; ++i) {
          spv::Id point_plane_cull_condition = spv::NoResult;
          for (uint32_t j = 0; j < 4; ++j) {
            spv::Id point_distance_is_negative = builder.createBinOp(
                spv::OpFOrdLessThan, type_bool,
                builder.createBinOp(spv::OpDot, type_float, point_guest_positions[j],
                                    point_user_clip_planes[i]),
                const_float_0);
            if (point_plane_cull_condition != spv::NoResult) {
              point_plane_cull_condition =
                  builder.createBinOp(spv::OpLogicalAnd, type_bool, point_plane_cull_condition,
                                      point_distance_is_negative);
            } else {
              point_plane_cull_condition = point_distance_is_negative;
            }
          }
          if (point_cull_condition != spv::NoResult) {
            point_cull_condition = builder.createBinOp(
                spv::OpLogicalOr, type_bool, point_cull_condition, point_plane_cull_condition);
          } else {
            point_cull_condition = point_plane_cull_condition;
          }
        }
        assert_true(point_cull_condition != spv::NoResult);
        spv::Block& point_cull_predecessor = *builder.getBuildPoint();
        spv::Block& point_cull_then_block = builder.makeNewBlock();
        spv::Block& point_cull_merge_block = builder.makeNewBlock();
        builder.createSelectionMerge(&point_cull_merge_block, spv::SelectionControlDontFlattenMask);
        {
          std::unique_ptr<spv::Instruction> branch_conditional_op(
              std::make_unique<spv::Instruction>(spv::OpBranchConditional));
          branch_conditional_op->addIdOperand(point_cull_condition);
          branch_conditional_op->addIdOperand(point_cull_then_block.getId());
          branch_conditional_op->addIdOperand(point_cull_merge_block.getId());
          branch_conditional_op->addImmediateOperand(1);
          branch_conditional_op->addImmediateOperand(2);
          point_cull_predecessor.addInstruction(std::move(branch_conditional_op));
        }
        point_cull_then_block.addPredecessor(&point_cull_predecessor);
        point_cull_merge_block.addPredecessor(&point_cull_predecessor);
        builder.setBuildPoint(&point_cull_then_block);
        builder.createNoResultOp(spv::OpReturn);
        builder.setBuildPoint(&point_cull_merge_block);
      }

      for (uint32_t i = 0; i < 4; ++i) {
        // Same interpolators for the entire sprite.
        for (uint32_t j = 0; j < key.interpolator_count; ++j) {
          builder.createStore(point_interpolators[j], out_interpolators[j]);
        }
        // Top-left, bottom-left, top-right, bottom-right order (chosen
        // arbitrarily, simply based on counterclockwise meaning front with
        // frontFace = VkFrontFace(0), but faceness is ignored for non-polygon
        // primitive types).
        uint32_t point_vertex_x = i >> 1;
        uint32_t point_vertex_y = i & 1;
        // Point coordinates.
        if (key.has_point_coordinates) {
          id_vector_temp.clear();
          id_vector_temp.push_back(builder.makeFloatConstant(float(point_vertex_x)));
          id_vector_temp.push_back(builder.makeFloatConstant(float(point_vertex_y)));
          builder.createStore(builder.makeCompositeConstant(type_float2, id_vector_temp),
                              out_point_coordinates);
        }
        // Position.
        id_vector_temp.clear();
        id_vector_temp.push_back(point_edge_x[point_vertex_x]);
        id_vector_temp.push_back(point_edge_y[point_vertex_y]);
        id_vector_temp.push_back(point_z);
        id_vector_temp.push_back(point_w);
        spv::Id point_vertex_position =
            builder.createCompositeConstruct(type_float4, id_vector_temp);
        id_vector_temp.clear();
        id_vector_temp.push_back(const_member_out_gl_per_vertex_position);
        builder.createStore(
            point_vertex_position,
            builder.createAccessChain(spv::StorageClassOutput, out_gl_per_vertex, id_vector_temp));
        // Clip distances.
        if (point_recalculate_clip_distances) {
          // Convert host clip space back to guest clip space before applying
          // user clip planes.
          id_vector_temp.clear();
          id_vector_temp.push_back(
              builder.createCompositeExtract(point_vertex_position, type_float, 0));
          id_vector_temp.push_back(
              builder.createCompositeExtract(point_vertex_position, type_float, 1));
          id_vector_temp.push_back(
              builder.createCompositeExtract(point_vertex_position, type_float, 2));
          spv::Id point_vertex_position_xyz =
              builder.createCompositeConstruct(type_float3, id_vector_temp);
          spv::Id point_guest_position_xyz = builder.createNoContractionBinOp(
              spv::OpFSub, type_float3, point_vertex_position_xyz,
              builder.createNoContractionBinOp(spv::OpVectorTimesScalar, type_float3,
                                               point_ndc_offset, point_w));
          point_guest_position_xyz = builder.createNoContractionBinOp(
              spv::OpFDiv, type_float3, point_guest_position_xyz, point_ndc_scale);
          id_vector_temp.clear();
          id_vector_temp.push_back(
              builder.createCompositeExtract(point_guest_position_xyz, type_float, 0));
          id_vector_temp.push_back(
              builder.createCompositeExtract(point_guest_position_xyz, type_float, 1));
          id_vector_temp.push_back(
              builder.createCompositeExtract(point_guest_position_xyz, type_float, 2));
          id_vector_temp.push_back(point_w);
          spv::Id point_guest_position =
              builder.createCompositeConstruct(type_float4, id_vector_temp);
          for (uint32_t j = 0; j < clip_distance_count; ++j) {
            id_vector_temp.clear();
            id_vector_temp.push_back(const_member_out_gl_per_vertex_clip_distance);
            id_vector_temp.push_back(builder.makeIntConstant(int32_t(j)));
            builder.createStore(builder.createBinOp(spv::OpDot, type_float, point_guest_position,
                                                    point_user_clip_planes[j]),
                                builder.createAccessChain(spv::StorageClassOutput,
                                                          out_gl_per_vertex, id_vector_temp));
          }
        } else if (clip_distance_count) {
          id_vector_temp.clear();
          id_vector_temp.push_back(const_member_out_gl_per_vertex_clip_distance);
          builder.createStore(point_clip_distances,
                              builder.createAccessChain(spv::StorageClassOutput, out_gl_per_vertex,
                                                        id_vector_temp));
        }
        // Emit the vertex.
        builder.createNoResultOp(spv::OpEmitVertex);
      }
      builder.createNoResultOp(spv::OpEndPrimitive);
    } break;

    case PipelineGeometryShader::kRectangleList: {
      // Construct a strip with the fourth vertex generated by mirroring a
      // vertex across the longest edge (the diagonal).
      //
      // Possible options:
      //
      // 0---1
      // |  /|
      // | / |  - 12 is the longest edge, strip 0123 (most commonly used)
      // |/  |    v3 = v0 + (v1 - v0) + (v2 - v0), or v3 = -v0 + v1 + v2
      // 2--[3]
      //
      // 1---2
      // |  /|
      // | / |  - 20 is the longest edge, strip 1203
      // |/  |
      // 0--[3]
      //
      // 2---0
      // |  /|
      // | / |  - 01 is the longest edge, strip 2013
      // |/  |
      // 1--[3]

      spv::Id const_int_0 = builder.makeIntConstant(0);
      spv::Id const_int_1 = builder.makeIntConstant(1);
      spv::Id const_int_2 = builder.makeIntConstant(2);
      spv::Id const_int_3 = builder.makeIntConstant(3);

      // Get squares of edge lengths to choose the longest edge.
      // [0] - 12, [1] - 20, [2] - 01.
      spv::Id edge_lengths[3];
      id_vector_temp.resize(3);
      id_vector_temp[1] = const_member_in_gl_per_vertex_position;
      for (uint32_t i = 0; i < 3; ++i) {
        id_vector_temp[0] = builder.makeIntConstant(int32_t((1 + i) % 3));
        id_vector_temp[2] = const_int_0;
        spv::Id edge_0_x = builder.createLoad(
            builder.createAccessChain(spv::StorageClassInput, in_gl_per_vertex, id_vector_temp),
            spv::NoPrecision);
        id_vector_temp[2] = const_int_1;
        spv::Id edge_0_y = builder.createLoad(
            builder.createAccessChain(spv::StorageClassInput, in_gl_per_vertex, id_vector_temp),
            spv::NoPrecision);
        id_vector_temp[0] = builder.makeIntConstant(int32_t((2 + i) % 3));
        id_vector_temp[2] = const_int_0;
        spv::Id edge_1_x = builder.createLoad(
            builder.createAccessChain(spv::StorageClassInput, in_gl_per_vertex, id_vector_temp),
            spv::NoPrecision);
        id_vector_temp[2] = const_int_1;
        spv::Id edge_1_y = builder.createLoad(
            builder.createAccessChain(spv::StorageClassInput, in_gl_per_vertex, id_vector_temp),
            spv::NoPrecision);
        spv::Id edge_x = builder.createBinOp(spv::OpFSub, type_float, edge_1_x, edge_0_x);
        spv::Id edge_y = builder.createBinOp(spv::OpFSub, type_float, edge_1_y, edge_0_y);
        edge_lengths[i] = builder.createBinOp(
            spv::OpFAdd, type_float, builder.createBinOp(spv::OpFMul, type_float, edge_x, edge_x),
            builder.createBinOp(spv::OpFMul, type_float, edge_y, edge_y));
      }

      // Choose the index of the first vertex in the strip based on which edge
      // is the longest, and calculate the indices of the other vertices.
      spv::Id vertex_indices[3];
      // If 12 > 20 && 12 > 01, then 12 is the longest edge, and the strip is
      // 0123. Otherwise, if 20 > 01, then 20 is the longest, and the strip is
      // 1203, but if not, 01 is the longest, and the strip is 2013.
      vertex_indices[0] = builder.createTriOp(
          spv::OpSelect, type_int,
          builder.createBinOp(spv::OpLogicalAnd, type_bool,
                              builder.createBinOp(spv::OpFOrdGreaterThan, type_bool,
                                                  edge_lengths[0], edge_lengths[1]),
                              builder.createBinOp(spv::OpFOrdGreaterThan, type_bool,
                                                  edge_lengths[0], edge_lengths[2])),
          const_int_0,
          builder.createTriOp(spv::OpSelect, type_int,
                              builder.createBinOp(spv::OpFOrdGreaterThan, type_bool,
                                                  edge_lengths[1], edge_lengths[2]),
                              const_int_1, const_int_2));
      for (uint32_t i = 1; i < 3; ++i) {
        // vertex_indices[i] = (vertex_indices[0] + i) % 3
        spv::Id vertex_index_without_wrapping = builder.createBinOp(
            spv::OpIAdd, type_int, vertex_indices[0], builder.makeIntConstant(int32_t(i)));
        vertex_indices[i] = builder.createTriOp(
            spv::OpSelect, type_int,
            builder.createBinOp(spv::OpSLessThan, type_bool, vertex_index_without_wrapping,
                                const_int_3),
            vertex_index_without_wrapping,
            builder.createBinOp(spv::OpISub, type_int, vertex_index_without_wrapping, const_int_3));
      }

      // Initialize the point coordinates output for safety if this shader type
      // is used with has_point_coordinates for some reason.
      spv::Id const_point_coordinates_zero = spv::NoResult;
      if (key.has_point_coordinates) {
        spv::Id const_float_0 = builder.makeFloatConstant(0.0f);
        id_vector_temp.clear();
        id_vector_temp.push_back(const_float_0);
        id_vector_temp.push_back(const_float_0);
        const_point_coordinates_zero = builder.makeCompositeConstant(type_float2, id_vector_temp);
      }

      // Emit the triangle in the strip that consists of the original vertices.
      for (uint32_t i = 0; i < 3; ++i) {
        spv::Id vertex_index = vertex_indices[i];
        // Interpolators.
        id_vector_temp.clear();
        id_vector_temp.push_back(vertex_index);
        for (uint32_t j = 0; j < key.interpolator_count; ++j) {
          builder.createStore(
              builder.createLoad(builder.createAccessChain(spv::StorageClassInput,
                                                           in_interpolators[j], id_vector_temp),
                                 spv::NoPrecision),
              out_interpolators[j]);
        }
        // Point coordinates.
        if (key.has_point_coordinates) {
          builder.createStore(const_point_coordinates_zero, out_point_coordinates);
        }
        // Position.
        id_vector_temp.clear();
        id_vector_temp.push_back(vertex_index);
        id_vector_temp.push_back(const_member_in_gl_per_vertex_position);
        spv::Id vertex_position = builder.createLoad(
            builder.createAccessChain(spv::StorageClassInput, in_gl_per_vertex, id_vector_temp),
            spv::NoPrecision);
        id_vector_temp.clear();
        id_vector_temp.push_back(const_member_out_gl_per_vertex_position);
        builder.createStore(
            vertex_position,
            builder.createAccessChain(spv::StorageClassOutput, out_gl_per_vertex, id_vector_temp));
        // Clip distances.
        if (clip_distance_count) {
          id_vector_temp.clear();
          id_vector_temp.push_back(vertex_index);
          id_vector_temp.push_back(const_member_in_gl_per_vertex_clip_distance);
          spv::Id vertex_clip_distances = builder.createLoad(
              builder.createAccessChain(spv::StorageClassInput, in_gl_per_vertex, id_vector_temp),
              spv::NoPrecision);
          id_vector_temp.clear();
          id_vector_temp.push_back(const_member_out_gl_per_vertex_clip_distance);
          builder.createStore(vertex_clip_distances,
                              builder.createAccessChain(spv::StorageClassOutput, out_gl_per_vertex,
                                                        id_vector_temp));
        }
        // Emit the vertex.
        builder.createNoResultOp(spv::OpEmitVertex);
      }

      // Construct the fourth vertex.
      // Interpolators.
      for (uint32_t i = 0; i < key.interpolator_count; ++i) {
        spv::Id in_interpolator = in_interpolators[i];
        id_vector_temp.clear();
        id_vector_temp.push_back(vertex_indices[0]);
        spv::Id vertex_interpolator_v0 = builder.createLoad(
            builder.createAccessChain(spv::StorageClassInput, in_interpolator, id_vector_temp),
            spv::NoPrecision);
        id_vector_temp[0] = vertex_indices[1];
        spv::Id vertex_interpolator_v01 = builder.createNoContractionBinOp(
            spv::OpFSub, type_float4,
            builder.createLoad(
                builder.createAccessChain(spv::StorageClassInput, in_interpolator, id_vector_temp),
                spv::NoPrecision),
            vertex_interpolator_v0);
        id_vector_temp[0] = vertex_indices[2];
        spv::Id vertex_interpolator_v3 = builder.createNoContractionBinOp(
            spv::OpFAdd, type_float4, vertex_interpolator_v01,
            builder.createLoad(
                builder.createAccessChain(spv::StorageClassInput, in_interpolator, id_vector_temp),
                spv::NoPrecision));
        builder.createStore(vertex_interpolator_v3, out_interpolators[i]);
      }
      // Point coordinates.
      if (key.has_point_coordinates) {
        builder.createStore(const_point_coordinates_zero, out_point_coordinates);
      }
      // Position.
      id_vector_temp.clear();
      id_vector_temp.push_back(vertex_indices[0]);
      id_vector_temp.push_back(const_member_in_gl_per_vertex_position);
      spv::Id vertex_position_v0 = builder.createLoad(
          builder.createAccessChain(spv::StorageClassInput, in_gl_per_vertex, id_vector_temp),
          spv::NoPrecision);
      id_vector_temp[0] = vertex_indices[1];
      spv::Id vertex_position_v01 = builder.createNoContractionBinOp(
          spv::OpFSub, type_float4,
          builder.createLoad(
              builder.createAccessChain(spv::StorageClassInput, in_gl_per_vertex, id_vector_temp),
              spv::NoPrecision),
          vertex_position_v0);
      id_vector_temp[0] = vertex_indices[2];
      spv::Id vertex_position_v3 = builder.createNoContractionBinOp(
          spv::OpFAdd, type_float4, vertex_position_v01,
          builder.createLoad(
              builder.createAccessChain(spv::StorageClassInput, in_gl_per_vertex, id_vector_temp),
              spv::NoPrecision));
      id_vector_temp.clear();
      id_vector_temp.push_back(const_member_out_gl_per_vertex_position);
      builder.createStore(
          vertex_position_v3,
          builder.createAccessChain(spv::StorageClassOutput, out_gl_per_vertex, id_vector_temp));
      // Clip distances.
      for (uint32_t i = 0; i < clip_distance_count; ++i) {
        spv::Id const_int_i = builder.makeIntConstant(int32_t(i));
        id_vector_temp.clear();
        id_vector_temp.push_back(vertex_indices[0]);
        id_vector_temp.push_back(const_member_in_gl_per_vertex_clip_distance);
        id_vector_temp.push_back(const_int_i);
        spv::Id vertex_clip_distance_v0 = builder.createLoad(
            builder.createAccessChain(spv::StorageClassInput, in_gl_per_vertex, id_vector_temp),
            spv::NoPrecision);
        id_vector_temp[0] = vertex_indices[1];
        spv::Id vertex_clip_distance_v01 = builder.createNoContractionBinOp(
            spv::OpFSub, type_float,
            builder.createLoad(
                builder.createAccessChain(spv::StorageClassInput, in_gl_per_vertex, id_vector_temp),
                spv::NoPrecision),
            vertex_clip_distance_v0);
        id_vector_temp[0] = vertex_indices[2];
        spv::Id vertex_clip_distance_v3 = builder.createNoContractionBinOp(
            spv::OpFAdd, type_float, vertex_clip_distance_v01,
            builder.createLoad(
                builder.createAccessChain(spv::StorageClassInput, in_gl_per_vertex, id_vector_temp),
                spv::NoPrecision));
        id_vector_temp.clear();
        id_vector_temp.push_back(const_member_out_gl_per_vertex_clip_distance);
        id_vector_temp.push_back(const_int_i);
        builder.createStore(
            vertex_clip_distance_v3,
            builder.createAccessChain(spv::StorageClassOutput, out_gl_per_vertex, id_vector_temp));
      }
      // Emit the vertex.
      builder.createNoResultOp(spv::OpEmitVertex);
      builder.createNoResultOp(spv::OpEndPrimitive);
    } break;

    case PipelineGeometryShader::kQuadList: {
      // Initialize the point coordinates output for safety if this shader type
      // is used with has_point_coordinates for some reason.
      spv::Id const_point_coordinates_zero = spv::NoResult;
      if (key.has_point_coordinates) {
        spv::Id const_float_0 = builder.makeFloatConstant(0.0f);
        id_vector_temp.clear();
        id_vector_temp.push_back(const_float_0);
        id_vector_temp.push_back(const_float_0);
        const_point_coordinates_zero = builder.makeCompositeConstant(type_float2, id_vector_temp);
      }

      // Build the triangle strip from the original quad vertices in the
      // 0, 1, 3, 2 order (like specified for GL_QUAD_STRIP).
      // TODO(Triang3l): Find the correct decomposition of quads into triangles
      // on the real hardware.
      for (uint32_t i = 0; i < 4; ++i) {
        spv::Id const_vertex_index = builder.makeIntConstant(int32_t(i ^ (i >> 1)));
        // Interpolators.
        id_vector_temp.clear();
        id_vector_temp.push_back(const_vertex_index);
        for (uint32_t j = 0; j < key.interpolator_count; ++j) {
          builder.createStore(
              builder.createLoad(builder.createAccessChain(spv::StorageClassInput,
                                                           in_interpolators[j], id_vector_temp),
                                 spv::NoPrecision),
              out_interpolators[j]);
        }
        // Point coordinates.
        if (key.has_point_coordinates) {
          builder.createStore(const_point_coordinates_zero, out_point_coordinates);
        }
        // Position.
        id_vector_temp.clear();
        id_vector_temp.push_back(const_vertex_index);
        id_vector_temp.push_back(const_member_in_gl_per_vertex_position);
        spv::Id vertex_position = builder.createLoad(
            builder.createAccessChain(spv::StorageClassInput, in_gl_per_vertex, id_vector_temp),
            spv::NoPrecision);
        id_vector_temp.clear();
        id_vector_temp.push_back(const_member_out_gl_per_vertex_position);
        builder.createStore(
            vertex_position,
            builder.createAccessChain(spv::StorageClassOutput, out_gl_per_vertex, id_vector_temp));
        // Clip distances.
        if (clip_distance_count) {
          id_vector_temp.clear();
          id_vector_temp.push_back(const_vertex_index);
          id_vector_temp.push_back(const_member_in_gl_per_vertex_clip_distance);
          spv::Id vertex_clip_distances = builder.createLoad(
              builder.createAccessChain(spv::StorageClassInput, in_gl_per_vertex, id_vector_temp),
              spv::NoPrecision);
          id_vector_temp.clear();
          id_vector_temp.push_back(const_member_out_gl_per_vertex_clip_distance);
          builder.createStore(vertex_clip_distances,
                              builder.createAccessChain(spv::StorageClassOutput, out_gl_per_vertex,
                                                        id_vector_temp));
        }
        // Emit the vertex.
        builder.createNoResultOp(spv::OpEmitVertex);
      }
      builder.createNoResultOp(spv::OpEndPrimitive);
    } break;

    default:
      assert_unhandled_case(key.type);
  }

  // End the main function.
  builder.leaveFunction();

  // Serialize the shader code.
  std::vector<unsigned int> shader_code;
  builder.dump(shader_code);

  // Create the shader module, and store the handle even if creation fails not
  // to try to create it again later.
  VkShaderModule shader_module = ui::vulkan::util::CreateShaderModule(
      command_processor_.GetVulkanDevice(), reinterpret_cast<const uint32_t*>(shader_code.data()),
      sizeof(uint32_t) * shader_code.size());
  if (shader_module == VK_NULL_HANDLE) {
    REXGPU_ERROR(
        "VulkanPipelineCache: Failed to create the primitive type geometry "
        "shader 0x{:08X}",
        key.key);
  }
  geometry_shaders_.emplace(key, shader_module);
  return shader_module;
}

bool VulkanPipelineCache::TryGetPipelineCreationArgumentsForDescription(
    const PipelineDescription& description,
    std::pair<const PipelineDescription, Pipeline>* pipeline,
    PipelineCreationArguments& creation_arguments, bool for_placeholder) {
  auto fail = [&](const char* reason) {
    REXGPU_ERROR(
        "VulkanPipelineCache: TryGetPipelineCreationArgumentsForDescription "
        "failed: {} (vs={:016X}, ps={:016X}, topo={}, geom={}, tess_mode={}, "
        "for_placeholder={}, render_pass_key=0x{:08X})",
        reason, description.vertex_shader_hash, description.pixel_shader_hash,
        uint32_t(description.primitive_topology), uint32_t(description.geometry_shader),
        uint32_t(description.tessellation_mode), uint32_t(for_placeholder),
        description.render_pass_key.key);
    return false;
  };

  if (!pipeline) {
    return fail("null_pipeline_storage");
  }
  if (!ArePipelineRequirementsMet(description)) {
    return fail("pipeline_requirements_not_met");
  }

  auto vertex_shader_it = shaders_.find(description.vertex_shader_hash);
  if (vertex_shader_it == shaders_.end()) {
    return fail("vertex_shader_not_found");
  }
  auto* vertex_shader = static_cast<VulkanShader::VulkanTranslation*>(
      vertex_shader_it->second->GetTranslation(description.vertex_shader_modification));
  if (!vertex_shader || !vertex_shader->is_translated() || !vertex_shader->is_valid()) {
    return fail("vertex_shader_translation_missing_or_invalid");
  }
  SpirvShaderTranslator::Modification vertex_shader_modification(
      description.vertex_shader_modification);
  Shader::HostVertexShaderType host_vertex_shader_type =
      vertex_shader_modification.vertex.host_vertex_shader_type;
  bool tessellated = Shader::IsHostVertexShaderTypeDomain(host_vertex_shader_type);

  VulkanShader::VulkanTranslation* pixel_shader = nullptr;
  if (description.pixel_shader_hash && !for_placeholder) {
    auto pixel_shader_it = shaders_.find(description.pixel_shader_hash);
    if (pixel_shader_it == shaders_.end()) {
      return fail("pixel_shader_not_found");
    }
    pixel_shader = static_cast<VulkanShader::VulkanTranslation*>(
        pixel_shader_it->second->GetTranslation(description.pixel_shader_modification));
    if (!pixel_shader || !pixel_shader->is_translated() || !pixel_shader->is_valid()) {
      return fail("pixel_shader_translation_missing_or_invalid");
    }
  }
  SpirvShaderTranslator::Modification pixel_shader_modification(
      description.pixel_shader_modification);

  VkShaderModule tessellation_vertex_shader = VK_NULL_HANDLE;
  VkShaderModule tessellation_control_shader = VK_NULL_HANDLE;
  uint32_t tessellation_patch_control_points = 0;
  if (tessellated) {
    xenos::TessellationMode tessellation_mode = description.tessellation_mode;
    tessellation_patch_control_points =
        GetTessellationPatchControlPointCount(host_vertex_shader_type, tessellation_mode);
    if (!tessellation_patch_control_points) {
      return fail("tessellation_patch_control_points_zero");
    }
    tessellation_vertex_shader =
        GetTessellationVertexShader(tessellation_mode == xenos::TessellationMode::kAdaptive);
    if (tessellation_vertex_shader == VK_NULL_HANDLE) {
      return fail("tessellation_vertex_shader_unavailable");
    }
    tessellation_control_shader =
        GetTessellationControlShader(host_vertex_shader_type, tessellation_mode);
    if (tessellation_control_shader == VK_NULL_HANDLE) {
      return fail("tessellation_control_shader_unavailable");
    }
  }

  VkShaderModule geometry_shader = VK_NULL_HANDLE;
  GeometryShaderKey geometry_shader_key;
  if (GetGeometryShaderKey(description.geometry_shader, vertex_shader_modification,
                           pixel_shader_modification, geometry_shader_key)) {
    geometry_shader = GetGeometryShader(geometry_shader_key);
    if (geometry_shader == VK_NULL_HANDLE) {
      return fail("geometry_shader_unavailable");
    }
  }

  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  bool use_dynamic_rendering =
      REXCVAR_GET(vulkan_dynamic_rendering) && vulkan_device->properties().dynamicRendering;
  VkRenderPass render_pass = VK_NULL_HANDLE;
  if (!use_dynamic_rendering) {
    render_pass =
        render_target_cache_.GetPath() == RenderTargetCache::Path::kPixelShaderInterlock
            ? render_target_cache_.GetFragmentShaderInterlockRenderPass()
            : render_target_cache_.GetHostRenderTargetsRenderPass(description.render_pass_key);
    if (render_pass == VK_NULL_HANDLE) {
      return fail("render_pass_unavailable");
    }
  }

  const PipelineLayoutProvider* pipeline_layout = command_processor_.GetPipelineLayout(
      (!for_placeholder && pixel_shader) ? static_cast<const VulkanShader&>(pixel_shader->shader())
                                               .GetTextureBindingsAfterTranslation()
                                               .size()
                                         : 0,
      (!for_placeholder && pixel_shader) ? static_cast<const VulkanShader&>(pixel_shader->shader())
                                               .GetSamplerBindingsAfterTranslation()
                                               .size()
                                         : 0,
      static_cast<const VulkanShader&>(vertex_shader->shader())
          .GetTextureBindingsAfterTranslation()
          .size(),
      static_cast<const VulkanShader&>(vertex_shader->shader())
          .GetSamplerBindingsAfterTranslation()
          .size());
  if (!pipeline_layout) {
    return fail("pipeline_layout_unavailable");
  }

  creation_arguments.pipeline = pipeline;
  creation_arguments.pipeline_layout = pipeline_layout;
  creation_arguments.vertex_shader = vertex_shader;
  creation_arguments.pixel_shader = for_placeholder ? nullptr : pixel_shader;
  creation_arguments.tessellation_vertex_shader = tessellation_vertex_shader;
  creation_arguments.tessellation_control_shader = tessellation_control_shader;
  creation_arguments.tessellation_patch_control_points = tessellation_patch_control_points;
  creation_arguments.geometry_shader = geometry_shader;
  creation_arguments.render_pass = render_pass;
  return true;
}

bool VulkanPipelineCache::EnsurePipelineCreated(const PipelineCreationArguments& creation_arguments,
                                                VkShaderModule fragment_shader_override) {
  // wwe2k14 patch (Session 55): vulkan_pipeline_debug_delay_ms (before the "already made" check is fine: a slow
  // driver is slow wherever the creation runs; callers only reach here to create).
  if (const int32_t delay = REXCVAR_GET(vulkan_pipeline_debug_delay_ms); delay > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
  }
  VkPipeline existing_pipeline =
      creation_arguments.pipeline->second.pipeline.load(std::memory_order_acquire);
  bool is_placeholder =
      creation_arguments.pipeline->second.is_placeholder.load(std::memory_order_acquire);
  bool creating_placeholder = fragment_shader_override != VK_NULL_HANDLE;
  if (existing_pipeline != VK_NULL_HANDLE) {
    if (!is_placeholder || creating_placeholder) {
      return true;
    }
  }

  // This function preferably should validate the description to prevent
  // unsupported behavior that may be dangerous/crashing because pipelines can
  // be created from the disk storage.

  rex::perf::frame_events::Get().pipelines.fetch_add(1, std::memory_order_relaxed);  // Session 73
  if (creation_arguments.pixel_shader) {
    REXGPU_INFO("Creating graphics pipeline state with VS {:016X}, PS {:016X}",
                creation_arguments.vertex_shader->shader().ucode_data_hash(),
                creation_arguments.pixel_shader->shader().ucode_data_hash());
  } else {
    REXGPU_INFO("Creating graphics pipeline state with VS {:016X}",
                creation_arguments.vertex_shader->shader().ucode_data_hash());
  }

  const PipelineDescription& description = creation_arguments.pipeline->first;
  if (!ArePipelineRequirementsMet(description)) {
    assert_always(
        "When creating a new pipeline, the description must not require "
        "unsupported features, and when loading the pipeline storage, "
        "pipelines with unsupported features must be filtered out");
    return false;
  }

  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();

  bool edram_fragment_shader_interlock =
      render_target_cache_.GetPath() == RenderTargetCache::Path::kPixelShaderInterlock;

  bool tessellated = description.primitive_topology == PipelinePrimitiveTopology::kPatchList;

  std::array<VkPipelineShaderStageCreateInfo, 5> shader_stages;
  uint32_t shader_stage_count = 0;

  // Vertex or tessellation evaluation shader (plus helper stages for tessellation).
  assert_true(creation_arguments.vertex_shader->is_translated());
  if (!creation_arguments.vertex_shader->is_valid()) {
    return false;
  }
  if (tessellated) {
    if (creation_arguments.tessellation_vertex_shader == VK_NULL_HANDLE ||
        creation_arguments.tessellation_control_shader == VK_NULL_HANDLE ||
        !creation_arguments.tessellation_patch_control_points) {
      return false;
    }

    VkPipelineShaderStageCreateInfo& shader_stage_vertex = shader_stages[shader_stage_count++];
    shader_stage_vertex.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_vertex.pNext = nullptr;
    shader_stage_vertex.flags = 0;
    shader_stage_vertex.stage = VK_SHADER_STAGE_VERTEX_BIT;
    shader_stage_vertex.module = creation_arguments.tessellation_vertex_shader;
    shader_stage_vertex.pName = "main";
    shader_stage_vertex.pSpecializationInfo = nullptr;

    VkPipelineShaderStageCreateInfo& shader_stage_tess_control =
        shader_stages[shader_stage_count++];
    shader_stage_tess_control.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_tess_control.pNext = nullptr;
    shader_stage_tess_control.flags = 0;
    shader_stage_tess_control.stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    shader_stage_tess_control.module = creation_arguments.tessellation_control_shader;
    shader_stage_tess_control.pName = "main";
    shader_stage_tess_control.pSpecializationInfo = nullptr;

    VkPipelineShaderStageCreateInfo& shader_stage_tess_eval = shader_stages[shader_stage_count++];
    shader_stage_tess_eval.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_tess_eval.pNext = nullptr;
    shader_stage_tess_eval.flags = 0;
    shader_stage_tess_eval.stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    shader_stage_tess_eval.module = creation_arguments.vertex_shader->shader_module();
    assert_true(shader_stage_tess_eval.module != VK_NULL_HANDLE);
    shader_stage_tess_eval.pName = "main";
    shader_stage_tess_eval.pSpecializationInfo = nullptr;
  } else {
    VkPipelineShaderStageCreateInfo& shader_stage_vertex = shader_stages[shader_stage_count++];
    shader_stage_vertex.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_vertex.pNext = nullptr;
    shader_stage_vertex.flags = 0;
    shader_stage_vertex.stage = VK_SHADER_STAGE_VERTEX_BIT;
    shader_stage_vertex.module = creation_arguments.vertex_shader->shader_module();
    assert_true(shader_stage_vertex.module != VK_NULL_HANDLE);
    shader_stage_vertex.pName = "main";
    shader_stage_vertex.pSpecializationInfo = nullptr;
  }
  // Geometry shader.
  if (creation_arguments.geometry_shader != VK_NULL_HANDLE) {
    if (tessellated) {
      return false;
    }
    VkPipelineShaderStageCreateInfo& shader_stage_geometry = shader_stages[shader_stage_count++];
    shader_stage_geometry.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_geometry.pNext = nullptr;
    shader_stage_geometry.flags = 0;
    shader_stage_geometry.stage = VK_SHADER_STAGE_GEOMETRY_BIT;
    shader_stage_geometry.module = creation_arguments.geometry_shader;
    shader_stage_geometry.pName = "main";
    shader_stage_geometry.pSpecializationInfo = nullptr;
  }
  // Fragment shader.
  VkPipelineShaderStageCreateInfo& shader_stage_fragment = shader_stages[shader_stage_count++];
  shader_stage_fragment.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shader_stage_fragment.pNext = nullptr;
  shader_stage_fragment.flags = 0;
  shader_stage_fragment.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  shader_stage_fragment.module = VK_NULL_HANDLE;
  shader_stage_fragment.pName = "main";
  shader_stage_fragment.pSpecializationInfo = nullptr;
  if (fragment_shader_override != VK_NULL_HANDLE) {
    shader_stage_fragment.module = fragment_shader_override;
  } else if (creation_arguments.pixel_shader) {
    assert_true(creation_arguments.pixel_shader->is_translated());
    if (!creation_arguments.pixel_shader->is_valid()) {
      return false;
    }
    shader_stage_fragment.module = creation_arguments.pixel_shader->shader_module();
    assert_true(shader_stage_fragment.module != VK_NULL_HANDLE);
  } else {
    if (edram_fragment_shader_interlock) {
      shader_stage_fragment.module = depth_only_fragment_shader_;
    } else if (render_target_cache_.depth_float24_convert_in_pixel_shader() &&
               (description.render_pass_key.depth_and_color_used & 1) &&
               (description.eds_dynamic ? description.eds_depth_used != 0
                                        : (description.depth_compare_op != xenos::CompareFunction::kAlways ||
                                           description.depth_write_enable)) &&
               description.render_pass_key.depth_format ==
                   xenos::DepthRenderTargetFormat::kD24FS8) {
      shader_stage_fragment.module = render_target_cache_.depth_float24_round()
                                         ? depth_float24_round_fragment_shader_
                                         : depth_float24_truncate_fragment_shader_;
      if (shader_stage_fragment.module == VK_NULL_HANDLE) {
        return false;
      }
    }
  }
  if (shader_stage_fragment.module == VK_NULL_HANDLE) {
    --shader_stage_count;
  }

  VkPipelineVertexInputStateCreateInfo vertex_input_state = {};
  vertex_input_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineInputAssemblyStateCreateInfo input_assembly_state;
  input_assembly_state.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly_state.pNext = nullptr;
  input_assembly_state.flags = 0;
  switch (description.primitive_topology) {
    case PipelinePrimitiveTopology::kPointList:
      input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
      assert_false(description.primitive_restart);
      if (description.primitive_restart) {
        return false;
      }
      break;
    case PipelinePrimitiveTopology::kLineList:
      input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
      assert_false(description.primitive_restart);
      if (description.primitive_restart) {
        return false;
      }
      break;
    case PipelinePrimitiveTopology::kLineStrip:
      input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
      break;
    case PipelinePrimitiveTopology::kTriangleList:
      input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      assert_false(description.primitive_restart);
      if (description.primitive_restart) {
        return false;
      }
      break;
    case PipelinePrimitiveTopology::kTriangleStrip:
      input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
      break;
    case PipelinePrimitiveTopology::kTriangleFan:
      // Keep parity with D3D12 by requiring triangle fan to list conversion in
      // PrimitiveProcessor rather than emitting native fan pipelines.
      assert_always();
      return false;
    case PipelinePrimitiveTopology::kLineListWithAdjacency:
      input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
      assert_false(description.primitive_restart);
      if (description.primitive_restart) {
        return false;
      }
      break;
    case PipelinePrimitiveTopology::kPatchList:
      input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
      assert_false(description.primitive_restart);
      if (description.primitive_restart) {
        return false;
      }
      break;
    default:
      assert_unhandled_case(description.primitive_topology);
      return false;
  }
  input_assembly_state.primitiveRestartEnable = description.primitive_restart ? VK_TRUE : VK_FALSE;

  VkPipelineViewportStateCreateInfo viewport_state;
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.pNext = nullptr;
  viewport_state.flags = 0;
  viewport_state.viewportCount = 1;
  viewport_state.pViewports = nullptr;
  viewport_state.scissorCount = 1;
  viewport_state.pScissors = nullptr;

  VkPipelineRasterizationStateCreateInfo rasterization_state = {};
  rasterization_state.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterization_state.rasterizerDiscardEnable = description.rasterizer_discard ? VK_TRUE : VK_FALSE;
  rasterization_state.depthClampEnable = description.depth_clamp_enable ? VK_TRUE : VK_FALSE;
  switch (description.polygon_mode) {
    case PipelinePolygonMode::kFill:
      rasterization_state.polygonMode = VK_POLYGON_MODE_FILL;
      break;
    case PipelinePolygonMode::kLine:
      rasterization_state.polygonMode = VK_POLYGON_MODE_LINE;
      break;
    case PipelinePolygonMode::kPoint:
      rasterization_state.polygonMode = VK_POLYGON_MODE_POINT;
      break;
    default:
      assert_unhandled_case(description.polygon_mode);
      return false;
  }
  rasterization_state.cullMode = VK_CULL_MODE_NONE;
  if (description.cull_front) {
    rasterization_state.cullMode |= VK_CULL_MODE_FRONT_BIT;
  }
  if (description.cull_back) {
    rasterization_state.cullMode |= VK_CULL_MODE_BACK_BIT;
  }
  rasterization_state.frontFace =
      description.front_face_clockwise ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
  // Depth bias is dynamic (even toggling - pipeline creation is expensive).
  // "If no depth attachment is present, r is undefined" in the depth bias
  // formula, though Z has no effect on anything if a depth attachment is not
  // used (the guest shader can't access Z), enabling only when there's a
  // depth / stencil attachment for correctness.
  rasterization_state.depthBiasEnable =
      (!edram_fragment_shader_interlock && (description.render_pass_key.depth_and_color_used & 0b1))
          ? VK_TRUE
          : VK_FALSE;
  // TODO(Triang3l): Wide lines.
  rasterization_state.lineWidth = 1.0f;

  bool subpass_has_attachments =
      !edram_fragment_shader_interlock && description.render_pass_key.depth_and_color_used != 0;
  VkSampleMask sample_mask = UINT32_MAX;
  VkPipelineMultisampleStateCreateInfo multisample_state = {};
  multisample_state.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  if (description.rasterizer_discard) {
    // Keep rasterizer-discard pipelines independent from guest MSAA state, as
    // done by D3D12 when rasterization is disabled.
    multisample_state.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  } else if (!edram_fragment_shader_interlock && !subpass_has_attachments) {
    // Keep parity with D3D12 host-render-target path, where draws without
    // color/depth attachments must run at 1x sample count.
    multisample_state.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  } else if (description.render_pass_key.msaa_samples == xenos::MsaaSamples::k2X &&
             !render_target_cache_.IsMsaa2xSupported(subpass_has_attachments)) {
    // Using sample 0 as 0 and 3 as 1 for 2x instead (not exactly the same
    // sample locations, but still top-left and bottom-right - however, this can
    // be adjusted with custom sample locations).
    multisample_state.rasterizationSamples = VK_SAMPLE_COUNT_4_BIT;
    // Keep parity with D3D12 ROV: sample masks are ignored without attachments.
    if (subpass_has_attachments) {
      sample_mask = 0b1001;
      multisample_state.pSampleMask = &sample_mask;
    }
  } else {
    multisample_state.rasterizationSamples =
        VkSampleCountFlagBits(uint32_t(1) << uint32_t(description.render_pass_key.msaa_samples));
  }
  if (description.sample_rate_shading &&
      multisample_state.rasterizationSamples != VK_SAMPLE_COUNT_1_BIT) {
    multisample_state.sampleShadingEnable = VK_TRUE;
    multisample_state.minSampleShading = 1.0f;
  }

  VkPipelineDepthStencilStateCreateInfo depth_stencil_state = {};
  depth_stencil_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth_stencil_state.pNext = nullptr;
  if (!edram_fragment_shader_interlock) {
    if (description.depth_write_enable ||
        description.depth_compare_op != xenos::CompareFunction::kAlways) {
      depth_stencil_state.depthTestEnable = VK_TRUE;
      depth_stencil_state.depthWriteEnable = description.depth_write_enable ? VK_TRUE : VK_FALSE;
      depth_stencil_state.depthCompareOp =
          VkCompareOp(uint32_t(VK_COMPARE_OP_NEVER) + uint32_t(description.depth_compare_op));
    }
    if (description.stencil_test_enable) {
      depth_stencil_state.stencilTestEnable = VK_TRUE;
      depth_stencil_state.front.failOp =
          VkStencilOp(uint32_t(VK_STENCIL_OP_KEEP) + uint32_t(description.stencil_front_fail_op));
      depth_stencil_state.front.passOp =
          VkStencilOp(uint32_t(VK_STENCIL_OP_KEEP) + uint32_t(description.stencil_front_pass_op));
      depth_stencil_state.front.depthFailOp = VkStencilOp(
          uint32_t(VK_STENCIL_OP_KEEP) + uint32_t(description.stencil_front_depth_fail_op));
      depth_stencil_state.front.compareOp = VkCompareOp(
          uint32_t(VK_COMPARE_OP_NEVER) + uint32_t(description.stencil_front_compare_op));
      depth_stencil_state.back.failOp =
          VkStencilOp(uint32_t(VK_STENCIL_OP_KEEP) + uint32_t(description.stencil_back_fail_op));
      depth_stencil_state.back.passOp =
          VkStencilOp(uint32_t(VK_STENCIL_OP_KEEP) + uint32_t(description.stencil_back_pass_op));
      depth_stencil_state.back.depthFailOp = VkStencilOp(
          uint32_t(VK_STENCIL_OP_KEEP) + uint32_t(description.stencil_back_depth_fail_op));
      depth_stencil_state.back.compareOp = VkCompareOp(
          uint32_t(VK_COMPARE_OP_NEVER) + uint32_t(description.stencil_back_compare_op));
    }
  }

  VkPipelineColorBlendStateCreateInfo color_blend_state = {};
  color_blend_state.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  VkPipelineColorBlendAttachmentState color_blend_attachments[xenos::kMaxColorRenderTargets] = {};
  if (!edram_fragment_shader_interlock) {
    uint32_t color_rts_used = description.render_pass_key.depth_and_color_used >> 1;
    {
      static const VkBlendFactor kBlendFactorMap[] = {
          VK_BLEND_FACTOR_ZERO,
          VK_BLEND_FACTOR_ONE,
          VK_BLEND_FACTOR_SRC_COLOR,
          VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
          VK_BLEND_FACTOR_DST_COLOR,
          VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
          VK_BLEND_FACTOR_SRC_ALPHA,
          VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
          VK_BLEND_FACTOR_DST_ALPHA,
          VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
          VK_BLEND_FACTOR_CONSTANT_COLOR,
          VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
          VK_BLEND_FACTOR_CONSTANT_ALPHA,
          VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
          VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
      };
      // 8 entries for safety since 3 bits from the guest are passed directly.
      static const VkBlendOp kBlendOpMap[] = {VK_BLEND_OP_ADD,
                                              VK_BLEND_OP_SUBTRACT,
                                              VK_BLEND_OP_MIN,
                                              VK_BLEND_OP_MAX,
                                              VK_BLEND_OP_REVERSE_SUBTRACT,
                                              VK_BLEND_OP_ADD,
                                              VK_BLEND_OP_ADD,
                                              VK_BLEND_OP_ADD};
      assert_true(vulkan_device->properties().independentBlend);
      uint32_t color_rts_remaining = color_rts_used;
      uint32_t color_rt_index;
      while (rex::bit_scan_forward(color_rts_remaining, &color_rt_index)) {
        color_rts_remaining &= ~(uint32_t(1) << color_rt_index);
        VkPipelineColorBlendAttachmentState& color_blend_attachment =
            color_blend_attachments[color_rt_index];
        const PipelineRenderTarget& color_rt = description.render_targets[color_rt_index];
        if (color_rt.src_color_blend_factor != PipelineBlendFactor::kOne ||
            color_rt.dst_color_blend_factor != PipelineBlendFactor::kZero ||
            color_rt.color_blend_op != xenos::BlendOp::kAdd ||
            color_rt.src_alpha_blend_factor != PipelineBlendFactor::kOne ||
            color_rt.dst_alpha_blend_factor != PipelineBlendFactor::kZero ||
            color_rt.alpha_blend_op != xenos::BlendOp::kAdd) {
          color_blend_attachment.blendEnable = VK_TRUE;
          color_blend_attachment.srcColorBlendFactor =
              kBlendFactorMap[uint32_t(color_rt.src_color_blend_factor)];
          color_blend_attachment.dstColorBlendFactor =
              kBlendFactorMap[uint32_t(color_rt.dst_color_blend_factor)];
          color_blend_attachment.colorBlendOp = kBlendOpMap[uint32_t(color_rt.color_blend_op)];
          color_blend_attachment.srcAlphaBlendFactor =
              kBlendFactorMap[uint32_t(color_rt.src_alpha_blend_factor)];
          color_blend_attachment.dstAlphaBlendFactor =
              kBlendFactorMap[uint32_t(color_rt.dst_alpha_blend_factor)];
          color_blend_attachment.alphaBlendOp = kBlendOpMap[uint32_t(color_rt.alpha_blend_op)];
        }
        color_blend_attachment.colorWriteMask = VkColorComponentFlags(color_rt.color_write_mask);
      }
    }
    color_blend_state.attachmentCount = 32 - rex::lzcnt(color_rts_used);
    color_blend_state.pAttachments = color_blend_attachments;
  }

  std::array<VkDynamicState, 14> dynamic_states;
  VkPipelineDynamicStateCreateInfo dynamic_state;
  dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_state.pNext = nullptr;
  dynamic_state.flags = 0;
  dynamic_state.dynamicStateCount = 0;
  dynamic_state.pDynamicStates = dynamic_states.data();
  // Regardless of whether some of this state actually has any effect on the
  // pipeline, marking all as dynamic because otherwise, binding any pipeline
  // with such state not marked as dynamic will cause the dynamic state to be
  // invalidated (again, even if it has no effect).
  dynamic_states[dynamic_state.dynamicStateCount++] = VK_DYNAMIC_STATE_VIEWPORT;
  dynamic_states[dynamic_state.dynamicStateCount++] = VK_DYNAMIC_STATE_SCISSOR;
  if (!edram_fragment_shader_interlock) {
    dynamic_states[dynamic_state.dynamicStateCount++] = VK_DYNAMIC_STATE_DEPTH_BIAS;
    dynamic_states[dynamic_state.dynamicStateCount++] = VK_DYNAMIC_STATE_BLEND_CONSTANTS;
    dynamic_states[dynamic_state.dynamicStateCount++] = VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK;
    dynamic_states[dynamic_state.dynamicStateCount++] = VK_DYNAMIC_STATE_STENCIL_WRITE_MASK;
    dynamic_states[dynamic_state.dynamicStateCount++] = VK_DYNAMIC_STATE_STENCIL_REFERENCE;
    // wwe2k14 patch (Session 93, idea 7): vulkan_extended_dynamic_state - the cull mode, front face, depth and stencil state the
    // command processor sets before every draw that uses this pipeline (VulkanCommandProcessor::UpdateEdsState).
    if (description.eds_dynamic) {
      dynamic_states[dynamic_state.dynamicStateCount++] = VK_DYNAMIC_STATE_CULL_MODE_EXT;
      dynamic_states[dynamic_state.dynamicStateCount++] = VK_DYNAMIC_STATE_FRONT_FACE_EXT;
      dynamic_states[dynamic_state.dynamicStateCount++] = VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE_EXT;
      dynamic_states[dynamic_state.dynamicStateCount++] = VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE_EXT;
      dynamic_states[dynamic_state.dynamicStateCount++] = VK_DYNAMIC_STATE_DEPTH_COMPARE_OP_EXT;
      dynamic_states[dynamic_state.dynamicStateCount++] = VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE_EXT;
      dynamic_states[dynamic_state.dynamicStateCount++] = VK_DYNAMIC_STATE_STENCIL_OP_EXT;
    }
  }

  VkPipelineTessellationStateCreateInfo tessellation_state = {};
  if (tessellated) {
    tessellation_state.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
    tessellation_state.patchControlPoints = creation_arguments.tessellation_patch_control_points;
  }

  VkPipelineRenderingCreateInfo pipeline_rendering_create_info = {};
  VkFormat color_attachment_formats[xenos::kMaxColorRenderTargets] = {};
  bool use_dynamic_rendering =
      REXCVAR_GET(vulkan_dynamic_rendering) && vulkan_device->properties().dynamicRendering;
  if (use_dynamic_rendering) {
    pipeline_rendering_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipeline_rendering_create_info.pNext = nullptr;
    pipeline_rendering_create_info.viewMask = 0;

    uint32_t color_attachment_count = 0;
    const auto& key = description.render_pass_key;
    xenos::ColorRenderTargetFormat color_formats[] = {
        key.color_0_view_format, key.color_1_view_format, key.color_2_view_format,
        key.color_3_view_format};
    for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
      if (key.depth_and_color_used & (1 << (1 + i))) {
        color_attachment_formats[i] = render_target_cache_.GetColorVulkanFormat(color_formats[i]);
        color_attachment_count = i + 1;
      }
    }
    pipeline_rendering_create_info.colorAttachmentCount = color_attachment_count;
    pipeline_rendering_create_info.pColorAttachmentFormats =
        color_attachment_count ? color_attachment_formats : nullptr;

    if (key.depth_and_color_used & 0b1) {
      VkFormat depth_format = render_target_cache_.GetDepthVulkanFormat(key.depth_format);
      pipeline_rendering_create_info.depthAttachmentFormat = depth_format;
      pipeline_rendering_create_info.stencilAttachmentFormat = depth_format;
    }
  }

  VkGraphicsPipelineCreateInfo pipeline_create_info;
  pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_create_info.pNext = use_dynamic_rendering ? &pipeline_rendering_create_info : nullptr;
  pipeline_create_info.flags = 0;
  pipeline_create_info.stageCount = shader_stage_count;
  pipeline_create_info.pStages = shader_stages.data();
  pipeline_create_info.pVertexInputState = &vertex_input_state;
  pipeline_create_info.pInputAssemblyState = &input_assembly_state;
  pipeline_create_info.pTessellationState = tessellated ? &tessellation_state : nullptr;
  pipeline_create_info.pViewportState = &viewport_state;
  pipeline_create_info.pRasterizationState = &rasterization_state;
  pipeline_create_info.pMultisampleState = &multisample_state;
  pipeline_create_info.pDepthStencilState = &depth_stencil_state;
  pipeline_create_info.pColorBlendState = &color_blend_state;
  pipeline_create_info.pDynamicState = &dynamic_state;
  if (creation_arguments.pipeline_layout == nullptr) {
    return false;
  }
  pipeline_create_info.layout = creation_arguments.pipeline_layout->GetPipelineLayout();
  pipeline_create_info.renderPass =
      use_dynamic_rendering ? VK_NULL_HANDLE : creation_arguments.render_pass;
  pipeline_create_info.subpass = 0;
  pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
  pipeline_create_info.basePipelineIndex = -1;

  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  VkPipeline pipeline;
  const auto s58_t0 = std::chrono::steady_clock::now();  // wwe2k14 patch (Session 58): s58_stats.h
  // wwe2k14 patch (Session 63): the on-disk pipeline cache (VK_NULL_HANDLE when off).
  VkResult create_result = dfn.vkCreateGraphicsPipelines(device, disk_pipeline_cache::Get(), 1,
                                                         &pipeline_create_info, nullptr, &pipeline);
  if (create_result == VK_SUCCESS) disk_pipeline_cache::NoteCreated();
  s58::g_stock_pipelines.Add(s58_t0);
  if (create_result != VK_SUCCESS) {
    uint64_t ps_hash = creation_arguments.pixel_shader
                           ? creation_arguments.pixel_shader->shader().ucode_data_hash()
                           : 0;
    REXGPU_ERROR(
        "VulkanPipelineCache: vkCreateGraphicsPipelines failed (result={}, vs={:016X}, "
        "ps={:016X}, topo={}, geom={}, tess_mode={}, patch_cp={}, render_pass_key=0x{:08X}, "
        "dynamic_rendering={})",
        int32_t(create_result), creation_arguments.vertex_shader->shader().ucode_data_hash(),
        ps_hash, uint32_t(description.primitive_topology), uint32_t(description.geometry_shader),
        uint32_t(description.tessellation_mode),
        creation_arguments.tessellation_patch_control_points, description.render_pass_key.key,
        uint32_t(use_dynamic_rendering));
    return false;
  }
  bool was_placeholder =
      creation_arguments.pipeline->second.is_placeholder.load(std::memory_order_acquire);
  VkPipeline old_pipeline =
      creation_arguments.pipeline->second.pipeline.exchange(pipeline, std::memory_order_acq_rel);
  creation_arguments.pipeline->second.pipeline_layout.store(creation_arguments.pipeline_layout,
                                                            std::memory_order_release);
  if (creating_placeholder) {
    creation_arguments.pipeline->second.is_placeholder.store(true, std::memory_order_release);
  } else {
    creation_arguments.pipeline->second.is_placeholder.store(false, std::memory_order_release);
    if (was_placeholder && old_pipeline != VK_NULL_HANDLE) {
      std::lock_guard<std::mutex> lock(deferred_destroy_lock_);
      deferred_destroy_pipelines_.emplace_back(command_processor_.GetCurrentSubmission(),
                                               old_pipeline);
    }
  }

  return true;
}

void VulkanPipelineCache::CreationThread(size_t thread_index) {
  while (true) {
    PipelineCreationArguments creation_arguments;
    {
      std::unique_lock<std::mutex> lock(creation_request_lock_);
      if (thread_index >= creation_threads_shutdown_from_ || creation_queue_.empty()) {
        if (creation_completion_set_event_ && creation_threads_busy_ == 0) {
          creation_completion_set_event_ = false;
          creation_completion_event_->Set();
        }
        if (thread_index >= creation_threads_shutdown_from_) {
          return;
        }
        creation_request_cond_.wait(lock);
        continue;
      }
      creation_arguments = creation_queue_.top();
      creation_queue_.pop();
      ++creation_threads_busy_;
    }

    bool created = EnsurePipelineCreated(creation_arguments);
    if (!created) {
      bool has_placeholder =
          creation_arguments.pipeline->second.is_placeholder.load(std::memory_order_acquire);
      VkPipeline pipeline =
          creation_arguments.pipeline->second.pipeline.load(std::memory_order_acquire);
      if (has_placeholder && pipeline != VK_NULL_HANDLE) {
        // Keep the placeholder resident and stop waiting for a real pipeline.
        creation_arguments.pipeline->second.is_placeholder.store(false, std::memory_order_release);
      } else {
        creation_arguments.pipeline->second.pipeline.store(VK_NULL_HANDLE,
                                                           std::memory_order_release);
        // wwe2k14 patch (Session 6, H1): a failed pending creation (queued by
        // vulkan_pipeline_no_submit_wait without a placeholder) must not stay
        // pending forever - its draws would be skipped and, with
        // vulkan_async_skip_incomplete_frames, no frame presented again. Its
        // next use retries synchronously in ConfigurePipeline, as without
        // async creation. Without the setting the flag is never set here.
        if (has_placeholder) {
          creation_arguments.pipeline->second.is_placeholder.store(false,
                                                                   std::memory_order_release);
        }
      }
    }

    {
      std::lock_guard<std::mutex> lock(creation_request_lock_);
      --creation_threads_busy_;
    }
  }
}

void VulkanPipelineCache::CreateQueuedPipelinesOnProcessorThread() {
  assert_false(creation_threads_.empty());
  while (true) {
    PipelineCreationArguments creation_arguments;
    {
      std::lock_guard<std::mutex> lock(creation_request_lock_);
      if (creation_queue_.empty()) {
        break;
      }
      creation_arguments = creation_queue_.top();
      creation_queue_.pop();
    }
    CreateQueuedPipelineOnProcessorThread(creation_arguments);
  }
}

// wwe2k14 patch (Session 55): vulkan_pipeline_sync_offscreen - wait for one pending pipeline only. Taken out of the
// creation queue (under its lock no creation thread can take it) and made here; when a creation thread is already
// making it, waited for until that thread publishes it (or fails: the flag is cleared either way).
void VulkanPipelineCache::AwaitPendingPipeline(std::pair<const PipelineDescription, Pipeline>* entry) {
  PipelineCreationArguments mine;
  bool found = false;
  {
    std::lock_guard<std::mutex> lock(creation_request_lock_);
    std::vector<PipelineCreationArguments> others;
    others.reserve(creation_queue_.size());
    while (!creation_queue_.empty()) {
      if (!found && creation_queue_.top().pipeline == entry) {
        mine = creation_queue_.top();
        found = true;
      } else {
        others.push_back(creation_queue_.top());
      }
      creation_queue_.pop();
    }
    for (const PipelineCreationArguments& a : others) creation_queue_.push(a);
  }
  if (found) {
    CreateQueuedPipelineOnProcessorThread(mine);
    return;
  }
  // (At most 2 s: a pipeline still pending then is skipped as before rather than holding the command processor.)
  // Session 57 review: a real deadline - 10,000 sleeps of 200 us were 10-20 s on Windows (the sleep rounds up to the
  // timer tick) and ~156 s with the window covered (the coarse timer), should a driver compile hang.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (entry->second.is_placeholder.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
}

void VulkanPipelineCache::CreateQueuedPipelineOnProcessorThread(const PipelineCreationArguments& creation_arguments) {
  {
    bool created = EnsurePipelineCreated(creation_arguments);
    if (!created) {
      bool has_placeholder =
          creation_arguments.pipeline->second.is_placeholder.load(std::memory_order_acquire);
      VkPipeline pipeline =
          creation_arguments.pipeline->second.pipeline.load(std::memory_order_acquire);
      if (has_placeholder && pipeline != VK_NULL_HANDLE) {
        creation_arguments.pipeline->second.is_placeholder.store(false, std::memory_order_release);
      } else {
        creation_arguments.pipeline->second.pipeline.store(VK_NULL_HANDLE,
                                                           std::memory_order_release);
        // wwe2k14 patch (Session 6, H1): see CreationThread.
        if (has_placeholder) {
          creation_arguments.pipeline->second.is_placeholder.store(false,
                                                                   std::memory_order_release);
        }
      }
    }
  }
}

void VulkanPipelineCache::ProcessDeferredPipelineDestructions(bool force_all) {
  std::vector<VkPipeline> pipelines_to_destroy;
  uint64_t completed_submission = command_processor_.GetCompletedSubmission();
  {
    std::lock_guard<std::mutex> lock(deferred_destroy_lock_);
    if (deferred_destroy_pipelines_.empty()) {
      return;
    }
    auto it = deferred_destroy_pipelines_.begin();
    while (it != deferred_destroy_pipelines_.end()) {
      if (force_all || it->first <= completed_submission) {
        pipelines_to_destroy.push_back(it->second);
        it = deferred_destroy_pipelines_.erase(it);
      } else if (!force_all) {
        // The queue is sorted by submission index.
        break;
      } else {
        ++it;
      }
    }
  }
  if (pipelines_to_destroy.empty()) {
    return;
  }
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  for (VkPipeline pipeline : pipelines_to_destroy) {
    if (pipeline != VK_NULL_HANDLE) {
      dfn.vkDestroyPipeline(device, pipeline, nullptr);
    }
  }
}

void VulkanPipelineCache::StorageWriteThread() {
  ShaderStoredHeader shader_header;
  // Don't leak anything in unused bits.
  std::memset(&shader_header, 0, sizeof(shader_header));

  std::vector<uint32_t> ucode_guest_endian;
  ucode_guest_endian.reserve(0xFFFF);

  bool flush_shaders = false;
  bool flush_pipelines = false;

  while (true) {
    if (flush_shaders) {
      flush_shaders = false;
      assert_not_null(shader_storage_file_);
      fflush(shader_storage_file_);
    }
    if (flush_pipelines) {
      flush_pipelines = false;
      assert_not_null(pipeline_storage_file_);
      fflush(pipeline_storage_file_);
    }

    const Shader* shader = nullptr;
    PipelineStoredDescription pipeline_description;
    bool write_pipeline = false;
    {
      std::unique_lock<std::mutex> lock(storage_write_request_lock_);
      if (storage_write_thread_shutdown_) {
        return;
      }
      if (!storage_write_shader_queue_.empty()) {
        shader = storage_write_shader_queue_.front();
        storage_write_shader_queue_.pop_front();
      } else if (storage_write_flush_shaders_) {
        storage_write_flush_shaders_ = false;
        flush_shaders = true;
      }
      if (!storage_write_pipeline_queue_.empty()) {
        std::memcpy(&pipeline_description, &storage_write_pipeline_queue_.front(),
                    sizeof(pipeline_description));
        storage_write_pipeline_queue_.pop_front();
        write_pipeline = true;
      } else if (storage_write_flush_pipelines_) {
        storage_write_flush_pipelines_ = false;
        flush_pipelines = true;
      }
      if (!shader && !write_pipeline) {
        storage_write_request_cond_.wait(lock);
        continue;
      }
    }

    if (shader) {
      shader_header.ucode_data_hash = shader->ucode_data_hash();
      shader_header.ucode_dword_count = shader->ucode_dword_count();
      shader_header.type = shader->type();
      assert_not_null(shader_storage_file_);
      fwrite(&shader_header, sizeof(shader_header), 1, shader_storage_file_);
      if (shader_header.ucode_dword_count) {
        ucode_guest_endian.resize(shader_header.ucode_dword_count);
        // Need to swap because the hash is calculated for the shader with guest
        // endianness.
        memory::copy_and_swap(ucode_guest_endian.data(), shader->ucode_dwords(),
                              shader_header.ucode_dword_count);
        fwrite(ucode_guest_endian.data(), shader_header.ucode_dword_count * sizeof(uint32_t), 1,
               shader_storage_file_);
      }
    }

    if (write_pipeline) {
      assert_not_null(pipeline_storage_file_);
      fwrite(&pipeline_description, sizeof(pipeline_description), 1, pipeline_storage_file_);
    }
  }
}

}  // namespace rex::graphics::vulkan
