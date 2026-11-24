// wwe2k14 patch (Session 92, M4b; docs\native\d3d\S92_M3_M5_DESIGN.md section 2, contract S92_M4B_CONTRACT.md): the draw description from the
// record decoder's typed state, as a SHADOW (owned_front_draw.h).
//
// BuildDraw is NativeDescribeDraw + the fields IssueDraw sets before the claim (vulkan\command_processor.cpp ~7902-8198 and ~11458-11510),
// with the register reads replaced by OwnedRecordState's typed fields (the decoder, owned_record.h): the shader translations by typed modification
// (pipeline_cache typed overloads), the viewport / scissor / polygon offset / normalized masks by draw_util's typed overloads, the primitive
// by PrimitiveProcessor::Process(result, DrawInput). What it does NOT build (named in the window line): the constants and bindings (layout, sets,
// dynamic offsets: M4c), the alpha-learn texture key (reads pixel float constants: M4c), the fast clear's fill fields and the claim-time fields
// (truth_*, cover, bounds, xfb).
//
// Register numbers do not appear here: the state is typed.
#include "draw_record.h"  // (Session 95, r3cmd)
#include "draw_slot.h"  // (Session 95: gpu_d3d_draw_slot)
#include "owned_front_draw.h"
#include "shader_swap.h"  // Session 94 (gpu_native_instance_world): the funnel counters
#include "owned_front_bind.h"

#include <rex/cvar.h>
#include <rex/graphics/free_counters.h>  // (M4f-C)
#include <rex/graphics/primitive_processor.h>
#include <rex/graphics/d3d_tex.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/graphics/vulkan/texture_cache.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/d3d_feed.h>  // (Session 92, M4d) D3dFeedCrowd: the member whose draw the crowd code is issuing
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/native/native_renderer.h>
#include <rex/graphics/vulkan/pipeline_cache.h>
#include <rex/hash.h>
#include <rex/logging.h>
#include <rex/ui/vulkan/fsr.h>

#include "../../d3d_shader_tag.h"
#include "../../pipeline/shader/spirv_relaxed_precision.h"
#include "owned_record.h"
#include "shader_swap.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <map>
#include <unordered_map>

REXCVAR_DECLARE(uint32_t, gpu_d3d_owned_front);
REXCVAR_DECLARE(bool, gpu_dlaa_jitter);
REXCVAR_DECLARE(bool, gpu_dlss_fsr);
REXCVAR_DECLARE(int32_t, gpu_dlaa_jitter_min_width);
REXCVAR_DECLARE(double, gpu_wide_factor);
REXCVAR_DECLARE(double, gpu_tall_factor);
REXCVAR_DECLARE(int32_t, gpu_wide_min_width);
REXCVAR_DECLARE(bool, gpu_wide_all);
REXCVAR_DECLARE(bool, gpu_native_lean);
REXCVAR_DECLARE(bool, gpu_native_swap_skip_translation);
REXCVAR_DECLARE(bool, gpu_native_swap_skip_translation_pixel);
REXCVAR_DECLARE(bool, gpu_native_vs_swap);
REXCVAR_DECLARE(bool, gpu_native_shader_swap);
REXCVAR_DECLARE(bool, gpu_native_post_depth_unrounded_debug);
REXCVAR_DECLARE(bool, gpu_native_truth_skip_restores);
REXCVAR_DECLARE(bool, gpu_allow_invalid_fetch_constants);
REXCVAR_DECLARE(bool, gpu_native_owned_alpha_learn);
REXCVAR_DECLARE(bool, gpu_native_owned_alpha_learn_constants);

// Session 93 (ideas 4 and 13; the read: out\s93\r3_match_gap.md in the game repo)
REXCVAR_DECLARE(bool, gpu_d3d_typed_latch);  // (Session 95: defined in owned_front_own.cpp)
REXCVAR_DECLARE(std::string, gpu_native_truth_bounds_vs);  // (Session 95, r3cmd: a record keeps the answer of TruthWantsBounds, so the list is part of the externals)
REXCVAR_DECLARE(int32_t, gpu_d3d_draw_record);  // (Session 95, r3cmd: owned_front_own.cpp)
REXCVAR_DEFINE_BOOL(gpu_d3d_typed_memo_reuse, false, "GPU/Native",
                    "Session 93: a typed draw's preparation (OwnedFrontPrepare) is taken from the memo on a hit - the decoder's epochs or StateHash + the shader pair, the "
                    "externals and the translation generation - instead of Build; only the primitive pass (once a draw, handed to DrawFromBase) and the fetch-dependent keys "
                    "are derived again. The memo is 512 slots, 2-way. Off = the 64-slot template memo of M4f-A.");
REXCVAR_DEFINE_BOOL(gpu_d3d_typed_memo_verify, false, "GPU/Native",
                    "Session 93: with gpu_d3d_typed_memo_reuse, every hit is rebuilt from scratch (Build) and compared field by field with what the hit gave; mismatches are "
                    "counted and logged ([d3d][memo]). The result of the hit is still the one drawn.");
REXCVAR_DEFINE_BOOL(gpu_d3d_typed_repeat, false, "GPU/Native",
                    "Session 93: under the cut a typed draw is recorded as the repeat base, and the next draw item that differs only in the vertex float constants (decoder "
                    "epochs, shaders, draw words, shared memory epoch) goes through RepeatDrawFast over the typed constants - an instance of the base's draw when the constants sit "
                    "one stride on (vulkan_repeat_draw_merge). Off = a typed draw ends the repeat chain.");
REXCVAR_DEFINE_BOOL(gpu_d3d_typed_timers, true, "GPU/Native",
                    "Session 93: the typed draw's clocks (two steady_clock reads in TypedDraw, two in OwnedFrontPrepare): the 'us a frame' figures of the [d3d][front] M4f-A line. "
                    "Off = no clock reads (the figures print 0).");
// Session 94 (idea 3, B7): the typed draw's per-draw CP work, trimmed. Off = byte-identical to Session 93.
REXCVAR_DEFINE_BOOL(gpu_d3d_typed_lean_bind, false, "GPU/Native",
                    "Session 94: trims the typed draw's per-draw work in DrawFromBase: the sampler parameters are kept per stage and rebuilt only when the decoder's fetch epoch, the shader or the "
                    "bindings changed; the system constants block is rebuilt only when its inputs changed (decoder epochs + the draw's own words + the primitive pass fields) and the block still "
                    "holds what the last build left; Externals reads its settings once a frame; TruthWantsBounds parses its list once a frame. Logs [d3d][typed] draw us/frame (sampled 1 frame in 8).");
namespace rex::graphics {
bool IsSelfConsistentInvalidTextureFetch(const xenos::xe_gpu_texture_fetch_t& fetch);
}

namespace rex::graphics::vulkan {
// Session 94: sampled split of DrawFromBase (index: 0 bind = samplers + textures + descriptors, 1 constants, 2 claim; [9] = sampled draws, [4] = sampler memo hits,
// [5] = sampler rebuilds, [6] = constants skipped, [7] = constants built, [8] = verify mismatches).
uint64_t g_lean_bind_stat[14] = {};
}  // namespace rex::graphics::vulkan

namespace rex::graphics::vulkan {
namespace native {
bool NativeVsExpandNote(Shader::HostVertexShaderType host_vs_type, xenos::PrimitiveType host_primitive, bool truth);  // Session 94 (native_renderer.cpp)
}

uint64_t g_owned_repeat_base_serial = 0;  // Session 93: bumped by DrawFromBase whenever a repeat base is recorded (any path)
using native::Draw;

// ---- the compared fields (X-macro: one name per field)
#define OWNED_FRONT_FIELDS(X)                                                                                                       \
  X(vertex_shader_hash) X(pixel_shader_hash) X(pixel_spec_key) X(pixel_ext) X(vertex_mul_key) X(vertex_ext) X(vertex_modification) \
  X(pixel_modification) X(vertex_shader_module) X(pixel_shader_module) X(float24_depth) X(color_quantize) X(ps_float24) X(ps_twin)   \
  X(ps_quantize) X(layout_counts) X(layout) X(restore) X(restore_pages) X(viewport_x) X(viewport_y) X(viewport_w) X(viewport_h)                \
  X(viewport_min_depth) X(viewport_max_depth) X(scissor) X(guest_primitive) X(host_primitive) X(host_vs_type) X(polygonal)           \
  X(raster_supported) X(raster_topology) X(raster_restart) X(raster_expanded) X(raster_geometry_shader) X(raster_cull_mode)         \
  X(raster_front_face) X(raster_depth_test) X(raster_depth_write) X(raster_depth_compare) X(raster_bias_constant)                   \
  X(raster_bias_slope) X(raster_stencil) X(raster_stencil_ops) X(raster_clip_disable) X(stencil_ref) X(stencil_refs)                \
  X(stencil_read_masks) X(stencil_write_masks) X(color_mask_all) X(blend_constants) X(occlusion_query) X(surface_info)              \
  X(color_info) X(depth_info) X(vertex_count) X(screen_space) X(memexport) X(color_write_mask) X(blend_control) X(indexed)         \
  X(index_buffer) X(index_offset) X(index_type) X(merge_stride) X(float_count) X(vertex_module_absent) X(pixel_module_absent)
enum Field : uint32_t {
#define X(n) F_##n,
  OWNED_FRONT_FIELDS(X)
#undef X
      kFieldCount
};
const char* const kFieldNames[kFieldCount] = {
#define X(n) #n,
    OWNED_FRONT_FIELDS(X)
#undef X
};

// why a claim was not built / compared
enum Skip : uint32_t {
  kSkipNone,
  kSkipNoRecordFast,      // a clear the library draws its own way (fast clear), no record draw item
  kSkipNoRecordOther,     // no record draw item: an inline draw, a segment draw, a draw the handler did not bracket
  kSkipNoRecordRepeat,    // a repeat claim with no record item (the crowd members the handler's crowd code issues after its draw)
  kSkipNoVertexShader,    // the typed state names no vertex shader the processor knows
  kSkipUnanalyzed,        // a shader the old path had not analysed (the typed path would not have been here)
  kSkipWouldNotDraw,      // the typed derivation says the old path would not have claimed this draw
  kSkipPrimitiveFailed,   // PrimitiveProcessor::Process refused the typed input
  kSkipCopyMode,          // the typed state says edram copy mode
  kSkipWouldConvert,      // the draw needs an index conversion that is not in the cache (the dry pass cannot do it)
  kSkipShaderHash,        // the typed vertex shader is not the old path's (counted as a vertex_shader_hash mismatch)
  kSkipMemexport,         // M4f-A: a memexport draw (the old path)
  kSkipXfb,               // M4f-A: captured positions (the old path)
  kSkipReplMismatch,      // M4f-A: the early replacement decision and the native ext name different translations
  kSkipModuleMissing,     // M4f-A: a module the draw needs could not be made
  kSkipLayout,            // M4f-A: no pipeline layout
  kSkipShaderState,       // M4f-A: the active shaders are not the typed state's
  kSkipCount
};
const char* const kSkipNames[kSkipCount] = {"built",           "no record item (fast clear)", "no record item (inline / segment / other)", "no record item (repeat: crowd members)",
                                            "no vertex shader", "shader not analysed",         "typed path would not draw",
                                            "primitive failed", "copy mode",           "index conversion not cached (would convert)",
                                            "vertex shader differs", "memexport", "captured positions", "replacement decision mismatch", "module missing", "no pipeline layout", "active shader differs"};

struct Window {
  uint64_t claims[kOwnedFrontKinds] = {};
  uint64_t built[kOwnedFrontKinds] = {};
  uint64_t mismatch_draws = 0;
  uint64_t skip[kSkipCount] = {};
  uint64_t by_field[kFieldCount] = {};
  uint64_t not_compared_memexport_index = 0;
  uint64_t not_compared_gs = 0;  // draws whose geometry shader handle was not looked up (the lookup creates)
  uint64_t build_ns = 0;
};
Window g_w;
Window g_winl;  // M4e: the inline draws' own window (swapped with g_w around their claims)
bool g_inl_logged[kFieldCount] = {};
uint32_t g_inl_logged_n = 0;
uint64_t g_frame_first = 0, g_last_print_frame = 0;
bool g_typed_draw_active = false;       // M4f-A: set around a typed draw's DrawFromBase
uint64_t g_typed_shadow_skipped = 0;
bool g_field_logged[kFieldCount] = {};
uint32_t g_logged = 0;

inline uint64_t NowNs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
inline uint32_t Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

// The few things DlaaJitterOn / the static helpers of command_processor.cpp do, restated here (they are file-static there).
static bool DlaaJitterOn() { return REXCVAR_GET(gpu_dlaa_jitter) && !(REXCVAR_GET(gpu_dlss_fsr) && rex::ui::vulkan::fsr::Failed()); }
static uint32_t SwapInterpolatorMask(bool native_enabled, const Shader* pixel_shader) {
  if (!pixel_shader || !native_enabled) return 0;
  return shader_swap::ReplacementInterpolatorMask(pixel_shader->ucode_data_hash());
}

// ---------------------------------------------------------------------------------------------------------------------------------
// M4d: what a template memo entry holds besides its Draw: the viewport derivation's result (the system constants need it) and the viewport
// origin before the DLAA jitter. TemplateOut is what a from-scratch Build reports for the entry.
struct TemplateOut {
  draw_util::ViewportInfo vinfo;
  float base_x = 0, base_y = 0;
};
struct TemplateIn : TemplateOut {
  const Draw* draw = nullptr;
};
TemplateOut g_scratch_out;  // the last from-scratch Build's report for a memo entry

struct OwnedFrontAccess {
  using VCP = VulkanCommandProcessor;

  // The shader the typed state names for a stage: the processor's active shader when it is that one, else a find-only lookup by hash
  // (nullptr if the pipeline cache has none: the draw then shows as a shader hash mismatch). Nothing is loaded or created.
  static VulkanShader* ShaderOf(VCP& cp, uint32_t type, const OwnedRecordState::ShaderId& id) {
    if (!id.dwords || (!id.address && !(id.hash && (REXCVAR_GET(gpu_d3d_owned_front) & 16u || rex::graphics::free_counters::g_cut.active)))) return nullptr;  // (M4f-C: under the cut an inline shader named by its words alone is found by its hash too) (M4e: an inline draw's shader may be named by its words alone: address 0, a hash)
    Shader* active = type == 0 ? cp.active_vertex_shader() : cp.active_pixel_shader();
    if (active && active->ucode_data_hash() == id.hash) return static_cast<VulkanShader*>(active);
    return cp.pipeline_cache_->FindShader(id.hash);
  }

  // A translation already made for (modification, native ext): its module, else VK_NULL_HANDLE with present = false. Never translates.
  static VkShaderModule ModuleOf(VulkanShader& shader, uint64_t modification, uint32_t native_ext, bool& present) {
    auto* t = static_cast<VulkanShader::VulkanTranslation*>(shader.GetTranslation(Shader::TranslationKey(modification, native_ext)));
    present = t && t->is_translated() && t->shader_module() != VK_NULL_HANDLE;
    return present ? t->shader_module() : VK_NULL_HANDLE;
  }

  // Session 93: the fields of a typed Draw that are a function of the primitive pass (Build's tail, and the memo hit's refresh). Returns shader_32bit_index_dma.
  static bool ApplyPrimitive(VCP& cp, const OwnedRecordState& s, const PrimitiveProcessor::ProcessingResult& primitive,
                             Shader::HostVertexShaderType host_vs_type, Draw& d) {
    const auto& device_properties = cp.GetVulkanDevice()->properties();
    const bool shader_32bit_index_dma = !device_properties.fullDrawIndexUint32 &&
                                        primitive.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA &&
                                        s.draw_initiator.index_size == xenos::IndexFormat::kInt32 &&
                                        host_vs_type == Shader::HostVertexShaderType::kVertex;
    const bool emulated_indexed =
        !(primitive.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kNone || shader_32bit_index_dma);
    d.indexed = emulated_indexed;
    d.index_buffer = VK_NULL_HANDLE;
    d.index_offset = 0;
    if (emulated_indexed) {
      switch (primitive.index_buffer_type) {
        case PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA:
          d.index_buffer = cp.shared_memory_->buffer();
          d.index_offset = primitive.guest_index_base;
          break;
        case PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted: {
          const auto b = cp.primitive_processor_->GetConvertedIndexBuffer(primitive.host_index_buffer_handle);
          d.index_buffer = b.first;
          d.index_offset = b.second;
          break;
        }
        case PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForAuto:
        case PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForDMA: {
          const auto b = cp.primitive_processor_->GetBuiltinIndexBuffer(primitive.host_index_buffer_handle);
          d.index_buffer = b.first;
          d.index_offset = b.second;
          break;
        }
        default:
          break;
      }
    }
    d.index_type = primitive.host_index_format == xenos::IndexFormat::kInt16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
    d.vertex_count = shader_32bit_index_dma ? 0 : primitive.host_draw_vertex_count;
    return shader_32bit_index_dma;
  }

  // Session 93: Draw::texture_key (the alpha learn's key: the pixel shader's texture base addresses + its float constants), from the typed words.
  static uint64_t TextureKey(const OwnedRecordState& s, VulkanShader* pixel_shader) {
    uint64_t h = 0x9E3779B97F4A7C15ull;
    for (const Shader::TextureBinding& binding : pixel_shader->texture_bindings()) {
      h = (h ^ uint64_t(FetchBase(s, binding.fetch_constant))) * 0xBF58476D1CE4E5B9ull;
    }
    const Shader::ConstantRegisterMap& pixel_constants = pixel_shader->constant_register_map();
    for (uint32_t i = 0; i < 4 && REXCVAR_GET(gpu_native_owned_alpha_learn_constants); ++i) {
      uint64_t entry = pixel_constants.float_bitmap[i];
      while (entry) {
        const uint32_t index = uint32_t(std::countr_zero(entry));
        entry &= entry - 1;
        const uint32_t at = (i * 64 + index) << 2;
        for (uint32_t c = 0; c < 4; ++c) h = (h ^ uint64_t(s.ps_float[at + c])) * 0xBF58476D1CE4E5B9ull;
      }
    }
    return h;
  }

  // Builds `d` from the typed state `s`. `fast_clear` is the old description's flag (set before NativeDescribeDraw, it changes what the native
  // renderer wants). Returns kSkipNone when built.
  // M4d: with `tmpl` (a template memo entry's Draw of the same record state - epoch, shader pair, primitive, mode) the parts that are a function of
  // the render state alone (viewport, scissor, depth / stencil / cull / bias) are taken from it; the rest - everything the draw's own words, the
  // fetch constants and the translations decide - is derived again (the delta). `tmpl` is never combined with `bi`.
  static Skip Build(VCP& cp, const OwnedRecordState& s, const Draw& old, Draw& d, bool& vs_absent, bool& ps_absent, bool& gs_skipped,
                    OwnedBindInputs* bi, const TemplateIn* tmpl = nullptr, TemplateOut* tout = nullptr, OwnedTypedPrep* own = nullptr) {
    const bool fast_clear = own ? false : old.fast_clear;  // (M4f-A: a token draw is a DMA draw; the clear fast path needs auto-index)
    using xenos::EdramMode;
    native::NativeRenderer& native = *cp.native_;
    const auto& device_properties = cp.GetVulkanDevice()->properties();
    const EdramMode edram_mode = s.modecontrol.edram_mode;
    if (edram_mode == EdramMode::kCopy) return kSkipCopyMode;

    VulkanShader* vertex_shader = ShaderOf(cp, 0, s.shader[0]);
    if (!vertex_shader) return kSkipNoVertexShader;
    if (own) {
      // M4f-A: the typed draw is the processor's draw now: the shaders it names are the ones the record loaded (find-only lookups of another shader
      // are the shadow's; here a different active shader is a fallback).
      if (cp.active_vertex_shader() != vertex_shader) return kSkipShaderState;
      if (!vertex_shader->is_ucode_analyzed()) cp.pipeline_cache_->AnalyzeShaderUcode(*vertex_shader);
    }
    if (!vertex_shader->is_ucode_analyzed()) return kSkipUnanalyzed;
    const bool polygonal = draw_util::IsPrimitivePolygonal(s.output_path_cntl.path_select == xenos::VGTOutputPath::kTessellationEnable,
                                                           s.draw_initiator.prim_type);
    const bool rasterization_done =
        draw_util::IsRasterizationPotentiallyDone(s.modecontrol, s.program_cntl, s.surface_info, s.sc_mode_cntl, polygonal);
    const bool memexport_used_vertex = vertex_shader->memexport_eM_written() != 0 && device_properties.vertexPipelineStoresAndAtomics;
    if (s.surface_info.surface_pitch == 0 && rasterization_done) return kSkipWouldNotDraw;

    VulkanShader* pixel_shader = nullptr;
    if (rasterization_done) {
      if (edram_mode == EdramMode::kColorDepth) {
        pixel_shader = ShaderOf(cp, 1, s.shader[1]);
        if (own && !pixel_shader && cp.active_pixel_shader()) return kSkipShaderState;  // (M4f-A review P2: a missing pixel shader would drop the colour writes)
        if (pixel_shader) {
          if (own) {
            if (cp.active_pixel_shader() != pixel_shader) return kSkipShaderState;
            if (!pixel_shader->is_ucode_analyzed()) cp.pipeline_cache_->AnalyzeShaderUcode(*pixel_shader);
          }
          if (!pixel_shader->is_ucode_analyzed()) return kSkipUnanalyzed;
          if (!draw_util::IsPixelShaderNeededWithRasterization(*pixel_shader, s.modecontrol, s.color_control, s.color_mask.value,
                                                               s.color_info)) {
            pixel_shader = nullptr;
          } else if (own) {
            spirv_relaxed_precision::NoteShaderPair(vertex_shader->ucode_data_hash(), pixel_shader->ucode_data_hash());  // (Session 73, as IssueDraw)
          }
        }
      }
    } else if (!memexport_used_vertex) {
      return kSkipWouldNotDraw;
    }
    const bool memexport_used_pixel = pixel_shader && pixel_shader->memexport_eM_written() != 0;
    const bool memexport = memexport_used_vertex || memexport_used_pixel;
    if (own && memexport_used_pixel && !device_properties.fragmentStoresAndAtomics) return kSkipMemexport;  // (the stock draw refuses it with an error)
    if (own && memexport && !(REXCVAR_GET(gpu_d3d_owned_front) & 64u)) return kSkipMemexport;  // (M4f-B, bit 6: DrawFromBase makes the memexport requests and the read-back)
    const reg::RB_DEPTHCONTROL depth_control = draw_util::GetNormalizedDepthControl(s.modecontrol, s.depth_control);
    const uint32_t color_mask =
        pixel_shader ? draw_util::GetNormalizedColorMask(s.modecontrol, s.color_mask.value, s.color_info, pixel_shader->writes_color_targets()) : 0;

    // Which translation the draw uses (the early decision of IssueDraw, ~10663-10690).
    const bool skip_vertex_on = REXCVAR_GET(gpu_native_swap_skip_translation) && REXCVAR_GET(gpu_native_vs_swap);
    const bool skip_pixel_on = REXCVAR_GET(gpu_native_swap_skip_translation_pixel) && REXCVAR_GET(gpu_native_shader_swap);
    const bool lean_frame = (skip_vertex_on || skip_pixel_on) && REXCVAR_GET(gpu_native_lean) && native.enabled() && native.replacing();
    bool pixel_repl = false, vertex_repl = false;
    if (skip_pixel_on && pixel_shader && lean_frame) {
      pixel_repl = cp.pipeline_cache_->UsesReplacementBindings(
          pixel_shader, shader_swap::Takes(pixel_shader->ucode_data_hash(), false, s.color_control.alpha_to_mask_enable)
                            ? uint32_t(SpirvShaderTranslator::kNativeExtSwap | SpirvShaderTranslator::kNativeExtOwnBindings)
                            : 0u);
    }
    // (IssueDraw ~10687: a draw that does not use the pixel replacement runs on the shader's real analysis, prepared before the interpolator mask)
    if (own && skip_pixel_on && pixel_shader && !pixel_repl && pixel_shader->analysis_registered()) {
      cp.pipeline_cache_->PrepareFullAnalysis(*pixel_shader);
    }
    uint32_t ps_param_gen_pos = UINT32_MAX;
    uint32_t interpolator_mask =
        pixel_shader ? (vertex_shader->writes_interpolators() &
                        pixel_shader->GetInterpolatorInputMask(s.program_cntl, s.context_misc, ps_param_gen_pos))
                     : 0;
    interpolator_mask |= SwapInterpolatorMask(native.enabled(), pixel_shader);

    // Primitive processing, from the typed draw words.
    PrimitiveProcessor::DrawInput input;
    input.draw_initiator = s.draw_initiator;
    input.dma_size = s.dma_size;
    input.dma_base = s.dma_base;
    input.output_path_cntl = s.output_path_cntl;
    input.hos_cntl = s.hos_cntl;
    input.su_sc_mode_cntl = s.sc_mode_cntl;
    input.multi_prim_ib_reset_indx = s.reset_indx;
    PrimitiveProcessor::ProcessingResult primitive;
    bool would_convert = false;
    // (the dry pass: no shared memory request, no conversion, no allocation, no write watch - Process(..., dry_run))
    if (own) {
      // M4f-A: the real pass (an index conversion the dry pass would refuse is made; DrawFromBase asks again and gets the cached result).
      if (!cp.primitive_processor_->Process(primitive, input)) return kSkipPrimitiveFailed;
    } else if (!cp.primitive_processor_->Process(primitive, input, true, &would_convert)) {
      return would_convert ? kSkipWouldConvert : kSkipPrimitiveFailed;
    }
    if (!primitive.host_draw_vertex_count) return kSkipWouldNotDraw;
    const Shader::HostVertexShaderType host_vs_type = primitive.host_vertex_shader_type;

    VulkanPipelineCache::TypedKeyState keys;
    keys.program_cntl = s.program_cntl;
    keys.hos_cntl = s.hos_cntl;
    keys.clip_cntl = s.clip_cntl;
    keys.draw_initiator = s.draw_initiator;
    keys.surface_info = s.surface_info;
    keys.context_misc = s.context_misc;
    keys.interpolator_cntl = s.interpolator_cntl;
    keys.depth_info = s.depth_info;
    keys.color_control = s.color_control;
    keys.fetch = s.fetch;
    const SpirvShaderTranslator::Modification vertex_modification = cp.pipeline_cache_->GetCurrentVertexShaderModification(
        *vertex_shader, host_vs_type, interpolator_mask, ps_param_gen_pos != UINT32_MAX, keys);
    SpirvShaderTranslator::Modification pixel_modification =
        pixel_shader ? cp.pipeline_cache_->GetCurrentPixelShaderModification(*pixel_shader, interpolator_mask, ps_param_gen_pos,
                                                                              depth_control, keys)
                     : SpirvShaderTranslator::Modification(0);
    if (skip_vertex_on && lean_frame && host_vs_type == Shader::HostVertexShaderType::kVertex) {
      const uint32_t wanted = cp.NativeVertexExt(vertex_shader);
      if (wanted & SpirvShaderTranslator::kNativeExtVsSwap) {
        vertex_repl = cp.pipeline_cache_->UsesReplacementBindings(vertex_shader, wanted | SpirvShaderTranslator::kNativeExtOwnBindings) &&
                      shader_swap::VertexFits(*vertex_shader, vertex_modification.value);
      }
    }
    bool vs_present = false;
    VkShaderModule vertex_base_module = VK_NULL_HANDLE;
    if (own) {
      // M4f-A (IssueDraw ~10759-10792): captured positions are the old path's; else the translations are really made - the emulated pair, with the
      // early replacement decision's deferral (the module of a deferred translation stays null, as NativeDescribeDraw finds it).
      if (vertex_modification.vertex.xfb_position) return kSkipXfb;
      auto* vertex_translation = static_cast<VulkanShader::VulkanTranslation*>(vertex_shader->GetOrCreateTranslation(vertex_modification.value));
      auto* pixel_translation =
          pixel_shader ? static_cast<VulkanShader::VulkanTranslation*>(pixel_shader->GetOrCreateTranslation(pixel_modification.value)) : nullptr;
      const uint32_t defer_emulated = lean_frame ? ((vertex_repl ? 1u : 0u) | (pixel_repl ? 2u : 0u)) : 0u;
      if (!cp.pipeline_cache_->EnsureShadersTranslated(vertex_translation, pixel_translation, defer_emulated)) return kSkipModuleMissing;
      vertex_base_module = vertex_translation->shader_module();
      vs_present = vertex_base_module != VK_NULL_HANDLE;
      own->vertex_endian_key = vertex_modification.vertex.vertex_fetch_endian_spec ? (1u | (uint32_t(vertex_modification.vertex.vertex_fetch_endians) << 1)) : 0u;  // (VertexEndianKeyOf, command_processor.cpp)
    } else {
      vertex_base_module = ModuleOf(*vertex_shader, vertex_modification.value, 0, vs_present);
    }

    // ---- NativeDescribeDraw's body
    d.fast_clear = fast_clear;
    // The constants / bindings are M4c's: not built here, but the native renderer's wants (TruthReason refuses a draw with no layout) read them,
    // so they are taken from the old description as inputs only (the failing first check: color_quantize / pixel module differed for want of a layout).
    d.layout = old.layout;
    std::copy(std::begin(old.sets), std::end(old.sets), d.sets);
    d.sets_bound = old.sets_bound;
    std::copy(std::begin(old.dynamic_offsets), std::end(old.dynamic_offsets), d.dynamic_offsets);
    d.dynamic_offset_count = old.dynamic_offset_count;
    d.pixel_spec_key = pixel_shader ? cp.pipeline_cache_->GetNativePixelSpecKey(pixel_shader, keys) : 0;
    d.pixel_ext = pixel_shader ? PixelExt(cp, s, pixel_shader) : 0;
    if (pixel_repl && (d.pixel_ext & SpirvShaderTranslator::kNativeExtSwap)) d.pixel_ext |= SpirvShaderTranslator::kNativeExtOwnBindings;
    VulkanPipelineCache::ApplyNativePixelSpec(pixel_modification, d.pixel_spec_key);
    d.vertex_shader_hash = vertex_shader->ucode_data_hash();
    d.pixel_shader_hash = pixel_shader ? pixel_shader->ucode_data_hash() : 0;
    d.restore = false;
    if (pixel_shader && REXCVAR_GET(gpu_native_truth_skip_restores)) {
      const uint32_t n = cp.native_restore_shader_count_.load(std::memory_order_acquire);
      for (uint32_t i = 0; i < n; ++i) {
        const VCP::NativeRestoreShader& r = cp.native_restore_shaders_[i];
        if (r.hash != d.pixel_shader_hash) continue;
        d.restore = true;
        d.restore_color_page = FetchBase(s, r.color_fetch);
        d.restore_depth_page = FetchBase(s, r.depth_fetch);
        break;
      }
    }
    d.vertex_shader = vertex_base_module;
    d.vertex_mul_key = cp.pipeline_cache_->GetNativeVertexMulKey();
    uint64_t native_vertex_modification = vertex_modification.value;
    d.vertex_ext = 0;
    uint32_t vertex_ext_wanted = cp.NativeVertexExt(vertex_shader);
    if (vertex_repl && (vertex_ext_wanted & SpirvShaderTranslator::kNativeExtVsSwap)) vertex_ext_wanted |= SpirvShaderTranslator::kNativeExtOwnBindings;
    if ((d.vertex_mul_key || vertex_ext_wanted) && host_vs_type == Shader::HostVertexShaderType::kVertex) {
      SpirvShaderTranslator::Modification mul_modification(native_vertex_modification);
      VulkanPipelineCache::ApplyNativeVertexSpec(mul_modification, d.vertex_mul_key);
      bool mul_present = false;
      VkShaderModule mul_module = VK_NULL_HANDLE;
      if (own) {
        // M4f-A (NativeDescribeDraw ~7986-8003): NATIVE's own vertex translation, made when it is not there yet.
        auto* mul_translation = static_cast<VulkanShader::VulkanTranslation*>(
            vertex_shader->GetOrCreateNativeTranslation(mul_modification.value, vertex_ext_wanted));
        if (mul_translation && cp.pipeline_cache_->EnsureShadersTranslated(mul_translation, nullptr) &&
            mul_translation->shader_module() != VK_NULL_HANDLE) {
          mul_module = mul_translation->shader_module();
          mul_present = true;
        }
      } else {
        mul_module = ModuleOf(*vertex_shader, mul_modification.value, vertex_ext_wanted, mul_present);
      }
      if (mul_present) {
        d.vertex_shader = mul_module;
        native_vertex_modification = mul_modification.value;
        d.vertex_ext = vertex_ext_wanted;
      }
    }
    if (own && d.vertex_shader == VK_NULL_HANDLE) {
      // (NativeDescribeDraw ~8008: this draw keeps the emulated vertex module, whose translation was deferred for a registered shader)
      auto* emulated = static_cast<VulkanShader::VulkanTranslation*>(vertex_shader->GetOrCreateTranslation(vertex_modification.value));
      if (emulated && !emulated->is_translated() && cp.pipeline_cache_->EnsureShadersTranslated(emulated, nullptr)) {
        d.vertex_shader = emulated->shader_module();
      }
    }
    vs_absent = d.vertex_shader == VK_NULL_HANDLE;  // (the shadow looks up what exists; the own mode made it)
    if (own && vs_absent) return kSkipModuleMissing;
    d.vertex_modification = native_vertex_modification;
    d.pixel_modification = pixel_shader ? pixel_modification.value : 0;
    d.ps_float24 = d.ps_twin = false;
    d.ps_quantize = 0;
    const bool pixel_replacement = cp.pipeline_cache_->UsesReplacementBindings(pixel_shader, d.pixel_ext);
    const bool vertex_replacement = cp.pipeline_cache_->UsesReplacementBindings(vertex_shader, d.vertex_ext);
    d.layout_counts[0] = uint16_t(pixel_shader ? pixel_shader->TextureBindingsFor(pixel_replacement).size() : 0);
    d.layout_counts[1] = uint16_t(pixel_shader ? pixel_shader->SamplerBindingsFor(pixel_replacement).size() : 0);
    d.layout_counts[2] = uint16_t(vertex_shader->TextureBindingsFor(vertex_replacement).size());
    d.layout_counts[3] = uint16_t(vertex_shader->SamplerBindingsFor(vertex_replacement).size());
    if (own) {
      // M4f-A (IssueDraw ~11022, the replacing frame's path): the layout from the shaders' binding counts of the early replacement decision, made
      // when it is not there (GetPipelineLayout creates).
      const auto* provider = cp.GetPipelineLayout(pixel_shader ? pixel_shader->TextureBindingsFor(pixel_repl).size() : 0,
                                                  pixel_shader ? pixel_shader->SamplerBindingsFor(pixel_repl).size() : 0,
                                                  vertex_shader->TextureBindingsFor(vertex_repl).size(),
                                                  vertex_shader->SamplerBindingsFor(vertex_repl).size());
      if (!provider) return kSkipLayout;
      d.layout = provider->GetPipelineLayout();
      own->layout = static_cast<const VCP::PipelineLayout*>(provider);
      // (the early decision and the native ext name different translations: the descriptors would be made for the other one)
      if ((pixel_shader && pixel_repl != cp.pipeline_cache_->UsesReplacementBindings(pixel_shader, d.pixel_ext)) ||
          vertex_repl != cp.pipeline_cache_->UsesReplacementBindings(vertex_shader, d.vertex_ext)) {
        return kSkipReplMismatch;
      }
    } else if (bi) {
      // M4c: the pipeline layout from the typed counts (a find-only lookup in the processor's layout cache: GetPipelineLayout creates on a miss);
      // the old description's is the fallback, counted.
      VCP::PipelineLayoutKey key;
      // (the lists IssueDraw's early replacement decision picks - what UpdateBindings and GetPipelineLayout use)
      key.texture_count_pixel = pixel_shader ? pixel_shader->TextureBindingsFor(pixel_repl).size() : 0;
      key.sampler_count_pixel = pixel_shader ? pixel_shader->SamplerBindingsFor(pixel_repl).size() : 0;
      key.texture_count_vertex = vertex_shader->TextureBindingsFor(vertex_repl).size();
      key.sampler_count_vertex = vertex_shader->SamplerBindingsFor(vertex_repl).size();
      const auto it = cp.pipeline_layouts_.find(key);
      if (it != cp.pipeline_layouts_.end()) {
        d.layout = it->second.GetPipelineLayout();
        OwnedFrontBindNoteLayout(true, d.layout == old.layout);
      } else {
        OwnedFrontBindNoteLayout(false, true);
      }
    }

    // Viewport (IssueDraw ~11064-11150, UpdateDynamicState, NativeDescribeDraw ~8039-8050).
    const bool host_render_targets_used = cp.render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets;
    const uint32_t scale_x = cp.texture_cache_->draw_resolution_scale_x(), scale_y = cp.texture_cache_->draw_resolution_scale_y();
    draw_util::ViewportInfo viewport_info;
    if (tmpl) {
      // (the template's viewport before the DLAA jitter; the jitter is applied now, as the scratch build does - not part of the key)
      viewport_info = tmpl->vinfo;
      d.viewport = tmpl->draw->viewport;
      d.viewport.x = tmpl->base_x;
      d.viewport.y = tmpl->base_y;
      if (DlaaJitterOn() && !cp.dlaa_scene_done_ && viewport_info.xy_extent[0] &&
          viewport_info.xy_extent[0] >= uint32_t(std::max(0, REXCVAR_GET(gpu_dlaa_jitter_min_width))) * scale_x) {
        d.viewport.x += cp.dlaa_jitter_[0];
        d.viewport.y += cp.dlaa_jitter_[1];
      }
      d.scissor = tmpl->draw->scissor;
    } else {
    draw_util::HostViewportRegs vr;
    vr.clip_cntl = s.clip_cntl;
    vr.vte_cntl = s.vte_cntl;
    vr.sc_mode_cntl = s.sc_mode_cntl;
    vr.su_vtx_cntl = s.su_vtx_cntl;
    vr.window_offset = s.window_offset;
    vr.depth_info = s.depth_info;
    std::memcpy(vr.vport, s.vport, sizeof(vr.vport));
    draw_util::GetHostViewportInfo(vr, scale_x, scale_y, false, device_properties.maxViewportDimensions[0],
                                   device_properties.maxViewportDimensions[1], true, depth_control,
                                   host_render_targets_used && cp.render_target_cache_->depth_float24_convert_in_pixel_shader(),
                                   host_render_targets_used, pixel_shader && pixel_shader->writes_depth(), viewport_info,
                                   cp.tiling_viewport_fix_);
    const float tall = float(REXCVAR_GET(gpu_tall_factor));
    if (const float wide = float(REXCVAR_GET(gpu_wide_factor));
        (wide > 1.001f && wide < 8.0f) || (tall > 1.001f && tall < 8.0f)) {
      const bool scene = !cp.dlaa_scene_done_ && viewport_info.xy_extent[0] &&
                         viewport_info.xy_extent[0] >= uint32_t(std::max(0, REXCVAR_GET(gpu_wide_min_width))) * scale_x;
      const bool screen_space = s.draw_initiator.prim_type == xenos::PrimitiveType::kRectangleList || !s.vte_cntl.vport_x_scale_ena;
      const bool post_after_marker = cp.hide_hud_after_pass_ && cp.IsWidePostVs(vertex_shader->ucode_data_hash());
      if ((scene && !screen_space) || (cp.hide_hud_after_pass_ && !post_after_marker) || cp.wide_menu_frame_ || REXCVAR_GET(gpu_wide_all)) {
        if (wide > 1.001f && wide < 8.0f) {
          viewport_info.ndc_scale[0] /= wide;
          viewport_info.ndc_offset[0] /= wide;
        }
        if (tall > 1.001f && tall < 8.0f) {
          viewport_info.ndc_scale[1] /= tall;
          viewport_info.ndc_offset[1] /= tall;
        }
      }
    }
    VkViewport viewport;
    if (viewport_info.xy_extent[0] && viewport_info.xy_extent[1]) {
      viewport.x = float(viewport_info.xy_offset[0]);
      viewport.y = float(viewport_info.xy_offset[1]);
      viewport.width = float(viewport_info.xy_extent[0]);
      viewport.height = float(viewport_info.xy_extent[1]);
    } else {
      viewport.x = -1.0f;
      viewport.y = -1.0f;
      viewport.width = 1.0f;
      viewport.height = 1.0f;
    }
    viewport.minDepth = viewport_info.z_min;
    viewport.maxDepth = viewport_info.z_max;
    if (tout) {
      tout->vinfo = viewport_info;
      tout->base_x = viewport.x;
      tout->base_y = viewport.y;
    }
    if (DlaaJitterOn() && !cp.dlaa_scene_done_ && viewport_info.xy_extent[0] &&
        viewport_info.xy_extent[0] >= uint32_t(std::max(0, REXCVAR_GET(gpu_dlaa_jitter_min_width))) * scale_x) {
      viewport.x += cp.dlaa_jitter_[0];
      viewport.y += cp.dlaa_jitter_[1];
    }
    d.viewport = viewport;
    if (host_render_targets_used && cp.native_ && native.truth() && depth_control.z_enable &&
        s.depth_info.depth_format == xenos::DepthRenderTargetFormat::kD24FS8) {
      d.viewport.minDepth = std::min(1.0f, d.viewport.minDepth * 2.0f);
      d.viewport.maxDepth = std::min(1.0f, d.viewport.maxDepth * 2.0f);
    }
    draw_util::Scissor scissor;
    draw_util::GetScissor(s.window_scissor_tl, s.window_scissor_br, s.window_offset, s.screen_scissor_tl, s.screen_scissor_br,
                          s.surface_info.surface_pitch, scissor);
    d.scissor.offset.x = int32_t(scissor.offset[0] * scale_x);
    d.scissor.offset.y = int32_t(scissor.offset[1] * scale_y);
    d.scissor.extent.width = scissor.extent[0] * scale_x;
    d.scissor.extent.height = scissor.extent[1] * scale_y;
    }  // !tmpl

    // Session 95 (Lane D): a counter for draws whose viewport has minDepth > maxDepth (the guest's Z range inverted), template draws
    // included. Core Vulkan only defines [0, 1] bounds, and drivers differ on the order; the value is not changed here. The first
    // one logs, then every 2^n-th.
    if (viewport_info.z_min > viewport_info.z_max) {
      static std::atomic<uint64_t> inverted_depth_draws{0};
      const uint64_t n = inverted_depth_draws.fetch_add(1, std::memory_order_relaxed) + 1;
      if ((n & (n - 1)) == 0) {
        REXGPU_INFO("[d3d][viewport] draws with minDepth > maxDepth: {} so far (this one {} > {})", n, viewport_info.z_min,
                    viewport_info.z_max);
      }
    }

    d.guest_primitive = uint8_t(s.draw_initiator.prim_type);
    d.host_primitive = uint8_t(primitive.host_primitive_type);
    d.host_vs_type = uint8_t(host_vs_type);
    d.polygonal = polygonal;
    native::DrawRaster& raster = d.raster;
    raster.supported = host_vs_type == Shader::HostVertexShaderType::kVertex && polygonal &&
                       (primitive.host_primitive_type == xenos::PrimitiveType::kTriangleList ||
                        primitive.host_primitive_type == xenos::PrimitiveType::kTriangleStrip);
    raster.topology = primitive.host_primitive_type == xenos::PrimitiveType::kTriangleStrip ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP
                                                                                           : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    raster.primitive_restart = primitive.host_primitive_reset_enabled;
    if (!raster.supported && native::NativeVsExpandNote(host_vs_type, primitive.host_primitive_type, native.truth())) {
      // Session 94 (nogs): see NativeDescribeDraw - the vertex shader expands it, no geometry shader.
      raster.expanded = true;
    } else if (!raster.supported &&
        (polygonal || primitive.host_primitive_type == xenos::PrimitiveType::kPointList ||
         (native.truth() && primitive.host_primitive_type == xenos::PrimitiveType::kRectangleList))) {
      // The geometry shader lookup creates one on a miss: not made here. The condition is typed; whether a shader exists, and its topology
      // and handle, are taken from the old description (counted).
      if (own) {
        // M4f-A (NativeDescribeDraw ~8079-8090): the geometry shader the emulated pipeline expands the primitives with, made when it is not there.
        VkPrimitiveTopology expanded_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        raster.geometry_shader = cp.pipeline_cache_->GetNativeGeometryShader(primitive, vertex_modification, pixel_modification, expanded_topology);
        if (raster.geometry_shader != VK_NULL_HANDLE) {
          raster.expanded = true;
          raster.topology = expanded_topology;
        }
      } else {
      gs_skipped = true;
      }
      if (!own && old.raster.expanded) {
        raster.expanded = true;
        raster.topology = old.raster.topology;
        raster.geometry_shader = old.raster.geometry_shader;
      }
    }
    if (tmpl) {
      raster.cull_mode = tmpl->draw->raster.cull_mode;
      raster.front_face_clockwise = tmpl->draw->raster.front_face_clockwise;
      raster.depth_test = tmpl->draw->raster.depth_test;
      raster.depth_write = tmpl->draw->raster.depth_write;
      raster.depth_compare = tmpl->draw->raster.depth_compare;
      raster.depth_bias_constant = tmpl->draw->raster.depth_bias_constant;
      raster.depth_bias_slope = tmpl->draw->raster.depth_bias_slope;
      raster.stencil = tmpl->draw->raster.stencil;
      raster.stencil_ops = tmpl->draw->raster.stencil_ops;
      raster.clip_disable = tmpl->draw->raster.clip_disable;
      d.stencil_ref = tmpl->draw->stencil_ref;
      for (uint32_t i = 0; i < 2; ++i) {
        d.stencil_refs[i] = tmpl->draw->stencil_refs[i];
        d.stencil_read_masks[i] = tmpl->draw->stencil_read_masks[i];
        d.stencil_write_masks[i] = tmpl->draw->stencil_write_masks[i];
      }
    } else {
    raster.cull_mode = (s.sc_mode_cntl.cull_front ? VK_CULL_MODE_FRONT_BIT : 0) | (s.sc_mode_cntl.cull_back ? VK_CULL_MODE_BACK_BIT : 0);
    if (!polygonal) raster.cull_mode = VK_CULL_MODE_NONE;
    raster.front_face_clockwise = s.sc_mode_cntl.face != 0;
    raster.depth_test = depth_control.z_enable;
    raster.depth_write = depth_control.z_enable && depth_control.z_write_enable;
    raster.depth_compare = depth_control.z_enable ? VkCompareOp(uint32_t(depth_control.zfunc)) : VK_COMPARE_OP_ALWAYS;
    {
      float bias_constant = 0.0f, bias_slope = 0.0f;
      draw_util::GetPreferredFacePolygonOffset(s.sc_mode_cntl, s.poly_offset, polygonal, bias_slope, bias_constant);
      bias_constant *= s.depth_info.depth_format == xenos::DepthRenderTargetFormat::kD24S8 ? draw_util::kD3D10PolygonOffsetFactorUnorm24
                                                                                           : draw_util::kD3D10PolygonOffsetFactorFloat24;
      bias_slope *= xenos::kPolygonOffsetScaleSubpixelUnit * float(std::max(scale_x, scale_y));
      {
        const uint32_t surface_info = s.surface_info.value, msaa = (surface_info >> 16) & 3;
        if (native.truth() && native.TruthTwinPitch(std::max(1u, ((surface_info & 0x3FFF) * (msaa >= 2 ? 2u : 1u) + 79) / 80))) {
          bias_slope *= cp.NativeTruthDrawRatio();
        }
      }
      raster.depth_bias_constant = bias_constant;
      raster.depth_bias_slope = bias_slope;
    }
    raster.stencil = depth_control.stencil_enable;
    d.stencil_ref = uint8_t(s.stencil_front.stencilref);
    if (depth_control.stencil_enable) {
      const bool back = polygonal && depth_control.backface_enable;
      const auto ops = [](xenos::StencilOp fail, xenos::StencilOp zpass, xenos::StencilOp zfail, xenos::CompareFunction func) {
        return (uint32_t(fail) & 7) | ((uint32_t(zpass) & 7) << 3) | ((uint32_t(zfail) & 7) << 6) | ((uint32_t(func) & 7) << 9);
      };
      const uint32_t front_ops = ops(depth_control.stencilfail, depth_control.stencilzpass, depth_control.stencilzfail, depth_control.stencilfunc);
      const uint32_t back_ops = back ? ops(depth_control.stencilfail_bf, depth_control.stencilzpass_bf, depth_control.stencilzfail_bf,
                                           depth_control.stencilfunc_bf)
                                     : front_ops;
      raster.stencil_ops = front_ops | (back_ops << 12);
      const reg::RB_STENCILREFMASK front_mask = s.stencil_front;
      const reg::RB_STENCILREFMASK back_mask = back ? s.stencil_back : front_mask;
      d.stencil_refs[0] = uint8_t(front_mask.stencilref);
      d.stencil_refs[1] = uint8_t(back_mask.stencilref);
      d.stencil_read_masks[0] = uint8_t(front_mask.stencilmask);
      d.stencil_read_masks[1] = uint8_t(back_mask.stencilmask);
      d.stencil_write_masks[0] = uint8_t(front_mask.stencilwritemask);
      d.stencil_write_masks[1] = uint8_t(back_mask.stencilwritemask);
    }
    raster.clip_disable = s.clip_cntl.clip_disable;
    }  // !tmpl
    d.color_mask_all = color_mask;
    for (uint32_t i = 0; i < 4; ++i) d.blend_constants[i] = s.blend_color[i];
    d.occlusion_query = cp.active_occlusion_query_.valid;
    d.surface_info = s.surface_info.value;
    d.color_info = s.color_info[0].value;
    d.depth_info = s.depth_info.value;
    d.vertex_count = primitive.host_draw_vertex_count;
    d.screen_space = s.draw_initiator.prim_type == xenos::PrimitiveType::kRectangleList || !s.vte_cntl.vport_x_scale_ena;
    d.memexport = memexport;
    d.color_write_mask = color_mask & 0xF;
    d.blend_control = s.blend[0].value;
    if (pixel_shader && native.WantsPixelShader(d)) {
      d.float24_depth = native.WantsFloat24Depth(d);
      d.color_quantize = native.WantsColorQuantize(d);
      const bool float24 = d.float24_depth && !REXCVAR_GET(gpu_native_post_depth_unrounded_debug);
      const uint32_t msaa = (d.surface_info >> 16) & 3;
      const bool twin =
          native.truth() && native.TruthTwinPitch(std::max(1u, ((d.surface_info & 0x3FFF) * (msaa >= 2 ? 2u : 1u) + 79) / 80));
      d.ps_float24 = float24;
      d.ps_twin = twin;
      d.ps_quantize = uint8_t(d.color_quantize);
      // (GetNativePixelShaderTranslation's modification, looked up only)
      SpirvShaderTranslator::Modification native_pixel = pixel_modification;
      native_pixel.pixel.native_host_render_targets = 1;
      native_pixel.pixel.native_twin_position = twin ? 1 : 0;
      native_pixel.pixel.native_color_quantize = d.color_quantize & 3;
      native_pixel.pixel.depth_stencil_mode = float24 ? SpirvShaderTranslator::Modification::DepthStencilMode::kFloat24Rounding
                                                      : SpirvShaderTranslator::Modification::DepthStencilMode::kNoModifiers;
      bool present = false;
      if (own) {
        // M4f-A (NativeDescribeDraw ~8189): NATIVE's pixel translation for host render targets, made when it is not there yet.
        VulkanShader::VulkanTranslation* translation = cp.pipeline_cache_->GetNativePixelShaderTranslation(
            *pixel_shader, pixel_modification, d.float24_depth && !REXCVAR_GET(gpu_native_post_depth_unrounded_debug), d.color_quantize, twin, d.pixel_ext);
        d.pixel_shader = translation ? translation->shader_module() : VK_NULL_HANDLE;
        present = d.pixel_shader != VK_NULL_HANDLE;
      } else {
        d.pixel_shader = ModuleOf(*pixel_shader, native_pixel.value, d.pixel_ext, present);
      }
      ps_absent = !present;
      if (own && !present) return kSkipModuleMissing;
    }
    if (own) {
      // Session 82 (alpha learn), NativeDescribeDraw ~7934: the base addresses of the pixel shader's textures (+ its float constants) - from the typed words.
      d.texture_key = 0;
      if (pixel_shader && REXCVAR_GET(gpu_native_owned_alpha_learn)) d.texture_key = TextureKey(s, pixel_shader);
    }

    // ---- IssueDraw's fields set after the description (~11421-11505)
    const bool shader_32bit_index_dma = ApplyPrimitive(cp, s, primitive, host_vs_type, d);
    d.merge_stride = vertex_modification.vertex.instanced_float_constants_stride;
    d.float_count = vertex_shader->constant_register_map().float_count;

    // ---- M4c: what the constants / bindings shadow needs (this function's own derivations + find-only reads of the old path's state)
    if (bi) {
      bi->s = &s;
      bi->vs = vertex_shader;
      bi->ps = pixel_shader;
      bi->polygonal = polygonal;
      bi->primitive = primitive;
      bi->shader_32bit_index_dma = shader_32bit_index_dma;
      for (uint32_t i = 0; i < 3; ++i) {
        bi->ndc_scale[i] = viewport_info.ndc_scale[i];
        bi->ndc_offset[i] = viewport_info.ndc_offset[i];
      }
      bi->xy_extent[0] = viewport_info.xy_extent[0];
      bi->xy_extent[1] = viewport_info.xy_extent[1];
      bi->depth_control = depth_control;
      bi->color_mask = color_mask;
      bi->used_texture_mask = vertex_shader->UsedTextureMaskFor(vertex_repl) | (pixel_shader ? pixel_shader->UsedTextureMaskFor(pixel_repl) : 0);
      bi->scale_x = scale_x;
      bi->scale_y = scale_y;
      const auto& rtc = *cp.render_target_cache_;
      bi->fsi = rtc.GetPath() == RenderTargetCache::Path::kPixelShaderInterlock;
      bi->host_rt = host_render_targets_used;
      bi->rg16_truncated = rtc.IsFixedRG16TruncatedToMinus1To1();
      bi->rgba16_truncated = rtc.IsFixedRGBA16TruncatedToMinus1To1();
      bi->native_truth = cp.NativeTruthActive();
      bi->gamma_as_unorm16 = rtc.gamma_render_target_as_unorm16();
      bi->image_view_format_swizzle = device_properties.imageViewFormatSwizzle;
      if (own) {
        // M4f-A: the lean base's fields and this draw's inputs. (The texture cache's per-texture answers the system constants take - the
        // resolution-scaled bits, the host swizzles - are read after RequestTextures, in OwnedFrontSystemConstants; the shadow's find-only view
        // / sampler / active-binding comparisons are not needed here.)
        bi->scaled_from_cache = 0;
        own->draw = &d;
        own->vertex_shader = vertex_shader;
        own->pixel_shader = pixel_shader;
        own->polygonal = polygonal;
        own->host_vertex_shader_type = host_vs_type;
        own->used_texture_mask = bi->used_texture_mask;
        own->depth_control = depth_control;
        own->color_mask = color_mask;
        own->viewport_info = viewport_info;
        own->merge_stride = d.merge_stride;
        own->float_count = d.float_count;
        bool ps_vertex_fetches = false;
        if (pixel_shader) {
          ps_vertex_fetches = !pixel_shader->vertex_bindings().empty();
          const Shader::ConstantRegisterMap& map = pixel_shader->constant_register_map();
          for (uint32_t i = 0; i < rex::countof(map.vertex_fetch_bitmap); ++i) {
            if (map.vertex_fetch_bitmap[i]) ps_vertex_fetches = true;
          }
        }
        own->pixel_shader_vertex_fetches = ps_vertex_fetches;
        own->shared_memory_access_stages = cp.shared_memory_->GetGuestDrawAccessStages(ps_vertex_fetches || memexport_used_pixel);  // (as the stock draw: a pixel memexport accesses the shared memory)
        own->host_primitive_reset_enabled = primitive.host_primitive_reset_enabled;
        own->host_primitive_type = primitive.host_primitive_type;
        own->pixel_spec_key = d.pixel_spec_key;
        own->vertex_mul_key = d.vertex_mul_key;
        own->pixel_ext = d.pixel_ext;
        own->vertex_ext = d.vertex_ext;
        own->input = input;
        own->vs_float = s.vs_float;
        own->ps_float = s.ps_float;
        own->bool_loop = s.bool_loop;
        own->fetch = s.fetch;
        own->surface_info = s.surface_info.value;
        own->color_info = s.color_info[0].value;
        own->depth_info = s.depth_info.value;
        return kSkipNone;
      }
      for (uint32_t i = 0; i < 32; ++i) {
        if (!(bi->used_texture_mask & (1u << i))) continue;
        if (cp.texture_cache_->IsActiveTextureResolutionScaled(i)) bi->scaled_from_cache |= 1u << i;
        bi->host_swizzle_from_cache[i] = cp.texture_cache_->GetActiveTextureHostSwizzle(i);
      }
      const auto fill = [&](uint32_t stage, const std::vector<VulkanShader::TextureBinding>* textures, size_t samplers) {
        bi->texture_count[stage] = textures ? uint32_t(std::min<size_t>(textures->size(), 32)) : 0;
        bi->sampler_count[stage] = uint32_t(samplers);
        for (uint32_t t = 0; t < bi->texture_count[stage]; ++t) {
          bi->textures[stage][t].fetch_constant = uint8_t((*textures)[t].fetch_constant);
          bi->textures[stage][t].is_signed = bool((*textures)[t].is_signed);
        }
      };
      fill(0, &vertex_shader->TextureBindingsFor(vertex_repl), vertex_shader->SamplerBindingsFor(vertex_repl).size());
      fill(1, pixel_shader ? &pixel_shader->TextureBindingsFor(pixel_repl) : nullptr, pixel_shader ? pixel_shader->SamplerBindingsFor(pixel_repl).size() : 0);
      for (uint32_t i = 0; i < 32; ++i) {
        uint32_t base_page = 0, format = 0, width = 0, height = 0;
        bi->active[i].valid = cp.texture_cache_->GetActiveTextureInfo(i, base_page, format, width, height);
        bi->active[i].base_page = base_page;
        bi->active[i].format = format;
        bi->active[i].width = width;
        bi->active[i].height = height;
      }
      for (uint32_t i = 0; i < 96; ++i) {
        bi->vertex_state[i].address = cp.vertex_buffer_states_[i].address;
        bi->vertex_state[i].size = cp.vertex_buffer_states_[i].size;
      }
      GatherBindings(cp, s, *bi, vertex_shader, pixel_shader, vertex_repl, pixel_repl);
      // the old path's constant sources: its system block, and the register image its banks were gathered from
      const RegisterFile& regs = cp.DrawRegs();
      bi->old_sys = &cp.system_constants_;
      bi->old_vs_float = &regs[XE_GPU_REG_SHADER_CONSTANT_000_X];
      bi->old_ps_float = &regs[XE_GPU_REG_SHADER_CONSTANT_256_X];
      bi->old_bool_loop = &regs[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031];
      bi->old_fetch = &regs[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0];
    }
    return kSkipNone;
  }

  // M4c: the image views and samplers the typed state implies, by find-only lookups (TextureCache::FindTextureOnly, VulkanTexture::FindView,
  // the sampler map), next to the raw views and the samplers the old path bound.
  static void GatherBindings(VCP& cp, const OwnedRecordState& s, OwnedBindInputs& bi, const VulkanShader* vs, const VulkanShader* ps,
                             bool vertex_repl, bool pixel_repl) {
    using TC = VulkanTextureCache;
    TC& tc = *cp.texture_cache_;
    struct Typed {
      bool done = false, valid = false, is_3d = false;
      xenos::DataDimension dimension = xenos::DataDimension(0);
      VkImageView views[2] = {};  // [0] unsigned, [1] signed
    };
    Typed typed[32];
    const auto derive = [&](uint32_t idx) -> Typed& {
      Typed& t = typed[idx];
      if (t.done) return t;
      t.done = true;
      xenos::xe_gpu_texture_fetch_t fetch;
      std::memcpy(&fetch, s.fetch + 6 * idx, sizeof(fetch));
      TC::TextureKey key;
      key.MakeInvalid();
      uint8_t signs = uint8_t(xenos::TextureSign::kUnsigned) * uint8_t(0b01010101);
      // (a bad fetch type makes BindingInfoFromFetchConstant log a warning: test it first, as the silent derivation)
      if (fetch.type == xenos::FetchConstantType::kTexture ||
          (fetch.type == xenos::FetchConstantType::kInvalidTexture && (REXCVAR_GET(gpu_allow_invalid_fetch_constants) || IsSelfConsistentInvalidTextureFetch(fetch)))) {
        TC::BindingInfoFromFetchConstant(fetch, key, &signs);
      }
      if (d3d_tex::g_active.load(std::memory_order_relaxed) && d3d_tex::Enabled() && d3d_tex::UseIds()) {  // the bound game object's key
        d3d_tex::Table& table = d3d_tex::Table::Get();
        uint32_t id = table.BoundId(idx);
        if (!id) {
          bool never = true;
          id = table.Recall(idx, s.fetch + 6 * idx, never);
        }
        d3d_tex::Entry e;
        if (id && table.Find(id, e) && fetch.type == xenos::FetchConstantType::kTexture) {
          xenos::xe_gpu_texture_fetch_t object_fetch;
          std::memcpy(&object_fetch, e.words, sizeof(object_fetch));
          object_fetch.dword_4 = fetch.dword_4;
          TC::TextureKey object_key;
          TC::BindingInfoFromFetchConstant(object_fetch, object_key, nullptr);
          if (object_key.is_valid && !(d3d_tex::VerifyKeys() && object_key != key)) {
            key = object_key;
            signs = texture_util::SwizzleSigns(fetch);
          }
        }
      }
      if (!key.is_valid) return t;
      t.valid = true;
      t.dimension = key.dimension;
      t.is_3d = key.dimension == xenos::DataDimension::k3D;
      const uint32_t host_swizzle = TC::GuestToHostSwizzle(fetch.swizzle, tc.GetHostFormatSwizzle(key));
      const TC::HostFormatPair& pair = tc.GetHostFormatPair(key);
      const bool uses_unsigned = texture_util::IsAnySignNotSigned(signs), uses_signed = texture_util::IsAnySignSigned(signs);
      const bool u_ok = pair.format_unsigned.format != VK_FORMAT_UNDEFINED, s_ok = pair.format_signed.format != VK_FORMAT_UNDEFINED;
      if (tc.IsSignedVersionSeparateForFormat(key)) {
        if (uses_unsigned && u_ok) {
          if (auto* tex = static_cast<TC::VulkanTexture*>(tc.FindTextureOnly(key))) t.views[0] = tex->FindView(false, host_swizzle);
        }
        if (uses_signed && s_ok) {
          TC::TextureKey signed_key = key;
          signed_key.signed_separate = 1;
          if (auto* tex = static_cast<TC::VulkanTexture*>(tc.FindTextureOnly(signed_key))) t.views[1] = tex->FindView(true, host_swizzle);
        }
      } else if (auto* tex = static_cast<TC::VulkanTexture*>(tc.FindTextureOnly(key))) {
        if (uses_unsigned && u_ok) t.views[0] = tex->FindView(false, host_swizzle);
        if (uses_signed && s_ok) t.views[1] = tex->FindView(true, host_swizzle);
      }
      return t;
    };
    const auto u64 = [](auto handle) {
      uint64_t bits = 0;
      std::memcpy(&bits, &handle, std::min(sizeof(handle), sizeof(bits)));
      return bits;
    };
    for (uint32_t st = 0; st < 2; ++st) {
      const VulkanShader* shader = st ? ps : vs;
      if (!shader) continue;
      const bool repl = st ? pixel_repl : vertex_repl;
      const auto& textures = shader->TextureBindingsFor(repl);
      for (uint32_t t = 0; t < bi.texture_count[st]; ++t) {
        const auto& b = textures[t];
        const uint32_t idx = b.fetch_constant & 31;
        Typed& ty = derive(idx);
        OwnedBindInputs::ViewCheck& v = bi.view[st][t];
        if ((tc.native_owned_mask_ | tc.native_served_mask_) & (1u << idx)) {
          ++bi.skipped_native_views;  // (the old path binds NativeOwnedView / NativeServeView, not the cache texture's view)
          continue;
        }
        if (ty.valid && !TC::AreDimensionsCompatible(b.dimension, ty.dimension)) continue;  // (the null view is bound; not compared)
        const auto& raw = tc.vulkan_texture_bindings_[idx];
        const VkImageView old_view = b.is_signed ? raw.image_view_signed : raw.image_view_unsigned;
        if (ty.valid && ty.is_3d && (b.dimension == xenos::FetchOpDimension::k1D || b.dimension == xenos::FetchOpDimension::k2D)) {
          ++bi.skipped_3d_views;  // (the old path makes that view lazily in GetActiveBindingOrNullImageView)
          continue;
        }
        v.compared = true;
        v.old_view = u64(old_view);
        v.typed_view = u64(b.is_signed ? ty.views[1] : ty.views[0]);
      }
      const auto& bindings = shader->SamplerBindingsFor(repl);
      const auto& old_list = st ? cp.current_samplers_pixel_ : cp.current_samplers_vertex_;
      bi.sampler_old_count[st] = uint32_t(old_list.size());
      const uint32_t n = uint32_t(std::min<size_t>({bindings.size(), old_list.size(), 32}));
      bi.sampler_checked[st] = n;
      for (uint32_t i = 0; i < n; ++i) {
        xenos::xe_gpu_texture_fetch_t fetch;
        std::memcpy(&fetch, s.fetch + 6 * (bindings[i].fetch_constant & 31), sizeof(fetch));
        OwnedBindInputs::SamplerCheck& sm = bi.sampler[st][i];
        if (fetch.type != xenos::FetchConstantType::kTexture && !(fetch.type == xenos::FetchConstantType::kInvalidTexture && (REXCVAR_GET(gpu_allow_invalid_fetch_constants) || IsSelfConsistentInvalidTextureFetch(fetch)))) {
          sm.old_params = sm.typed_params = old_list[i].first.value;  // (a bad fetch type: GetSamplerParametersForFetch would only log; not compared)
          sm.old_sampler = sm.typed_sampler = u64(old_list[i].second);
          continue;
        }
        const TC::SamplerParameters params = tc.GetSamplerParametersForFetch(fetch, bindings[i]);
        sm.old_params = old_list[i].first.value;
        sm.typed_params = params.value;
        sm.old_sampler = u64(old_list[i].second);
        const auto it = tc.samplers_.find(params);
        sm.typed_sampler = it != tc.samplers_.end() ? u64(it->second.sampler) : 0;
      }
    }
  }

  static uint32_t FetchBase(const OwnedRecordState& s, uint32_t index) {
    xenos::xe_gpu_texture_fetch_t fetch;
    std::memcpy(&fetch, s.fetch + 6 * size_t(index & 31), sizeof(fetch));
    return fetch.base_address;
  }

  // NativePixelExt (vulkan\command_processor.cpp ~7820): the epilogue-idle word from the typed state (what UpdateSystemConstantValues derives:
  // the alpha test function, alpha to coverage, the colour exponent bias and the gamma flag of every colour target the shader writes).
  static uint32_t PixelExt(VCP& cp, const OwnedRecordState& s, const Shader* pixel_shader) {
    if (!pixel_shader || !cp.NativeRendererEnabled()) return 0;
    uint32_t ext = 0;
    const VulkanPipelineCache::NativeSpecLatch& spec = cp.pipeline_cache_->NativeSpec();
    const bool a2c = s.color_control.alpha_to_mask_enable;
    if (spec.ps_epilogue) {
      const xenos::CompareFunction alpha_func =
          s.color_control.alpha_test_enable ? s.color_control.alpha_func : xenos::CompareFunction::kAlways;
      bool idle = alpha_func == xenos::CompareFunction::kAlways && !a2c;
      for (uint32_t i = 0; idle && i < 4; ++i) {
        if (!pixel_shader->writes_color_target(i)) continue;
        const reg::RB_COLOR_INFO ci = s.color_info[i];
        int32_t exp_bias = ci.color_exp_bias;
        if (cp.render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets && !cp.NativeTruthActive() &&
            ((ci.color_format == xenos::ColorRenderTargetFormat::k_16_16 && !cp.render_target_cache_->IsFixedRG16TruncatedToMinus1To1()) ||
             (ci.color_format == xenos::ColorRenderTargetFormat::k_16_16_16_16 &&
              !cp.render_target_cache_->IsFixedRGBA16TruncatedToMinus1To1()))) {
          exp_bias -= 5;
        }
        const uint32_t bias_bits = UINT32_C(0x3F800000) + (uint32_t(exp_bias) << 23);
        const bool gamma = !cp.render_target_cache_->gamma_render_target_as_unorm16() &&
                           ci.color_format == xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA;
        idle = bias_bits == 0x3F800000u && !gamma;
      }
      if (idle) ext |= SpirvShaderTranslator::kNativeExtEpilogueIdle;
    }
    if (spec.fma & 1) ext |= SpirvShaderTranslator::kNativeExtFma;
    if (REXCVAR_GET(gpu_native_shader_swap) &&
        shader_swap::Takes(pixel_shader->ucode_data_hash(), (ext & SpirvShaderTranslator::kNativeExtEpilogueIdle) != 0, a2c)) {
      ext |= SpirvShaderTranslator::kNativeExtSwap;
      ext |= shader_swap::AblateExt(pixel_shader->ucode_data_hash());  // (Session 95, B2)
    }
    return ext;
  }

  static uint64_t Frame(VCP& cp) { return cp.frame_current_; }

  // M4d: the processor-side inputs the template parts depend on besides the typed state (viewport derivation: the DLAA / wide / HUD flags, the
  // tiling viewport fix, the jitter), hashed with the frame; and the occlusion flag the Draw carries.
  // The inputs the frame number used to stand for: the resolution scale, the wide / tall / DLAA settings, the NATIVE truth state and its draw ratio,
  // the host render target path, and a cache generation (lean_cache_gen_: the caches the entries' handles come from were cleared). The DLAA jitter is
  // NOT here: it is applied to the template's viewport after the lookup.
  static uint64_t Externals(VCP& cp) {
    struct {
      uint64_t cache_gen;
      uint32_t scale_x, scale_y, wide, tall, wide_min, dlaa_min, flags, ratio;
      uint32_t latch_gen, pad;  // (Session 94: 0 unless gpu_d3d_typed_lean_bind latched a change of the settings)
    } in;
    std::memset(&in, 0, sizeof(in));
    in.cache_gen = cp.lean_cache_gen_;
    in.scale_x = cp.texture_cache_->draw_resolution_scale_x();
    in.scale_y = cp.texture_cache_->draw_resolution_scale_y();
    // Session 94 (gpu_d3d_typed_lean_bind): the settings (cvars) are read once a frame, not once a draw (~8 reads, 690 draws a frame); a setting changed
    // mid-frame takes effect on the next frame. The processor's own state (flags below, the cache generation, the scale, the ratio) is read every call.
    struct SettingsLatch {
      uint64_t frame = UINT64_MAX;
      uint32_t wide, tall, wide_min, dlaa_min;
      bool wide_all, dlaa_on;
      // Bumped whenever a refresh sees other values than the latched ones. Build reads the settings live, so a memo entry made while the latch was stale (a mid-frame change)
      // holds the new settings' content under the old hash; the generation in the hash keeps it from ever matching again (it only grows).
      uint32_t gen = 0;
    };
    static SettingsLatch latch;
    if (!REXCVAR_GET(gpu_d3d_typed_lean_bind) || latch.frame != cp.frame_current_) {
      const SettingsLatch old = latch;
      const float w = float(REXCVAR_GET(gpu_wide_factor)), t = float(REXCVAR_GET(gpu_tall_factor));
      std::memcpy(&latch.wide, &w, 4);
      std::memcpy(&latch.tall, &t, 4);
      latch.wide_min = uint32_t(REXCVAR_GET(gpu_wide_min_width));
      latch.dlaa_min = uint32_t(REXCVAR_GET(gpu_dlaa_jitter_min_width));
      latch.wide_all = REXCVAR_GET(gpu_wide_all);
      latch.dlaa_on = DlaaJitterOn();
      if (old.frame != UINT64_MAX && (old.wide != latch.wide || old.tall != latch.tall || old.wide_min != latch.wide_min || old.dlaa_min != latch.dlaa_min ||
                                      old.wide_all != latch.wide_all || old.dlaa_on != latch.dlaa_on)) {
        ++latch.gen;
      }
      latch.frame = cp.frame_current_;
    }
    in.latch_gen = REXCVAR_GET(gpu_d3d_typed_lean_bind) ? latch.gen : 0;
    in.wide = latch.wide;
    in.tall = latch.tall;
    in.wide_min = latch.wide_min;
    in.dlaa_min = latch.dlaa_min;
    const bool host_rt = cp.render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets;
    in.flags = (cp.dlaa_scene_done_ ? 1u : 0u) | (cp.hide_hud_after_pass_ ? 2u : 0u) | (cp.wide_menu_frame_ ? 4u : 0u) |
               (cp.tiling_viewport_fix_ ? 8u : 0u) | (latch.wide_all ? 16u : 0u) | (latch.dlaa_on ? 32u : 0u) |
               ((cp.native_ && cp.native_->truth()) ? 64u : 0u) | (cp.NativeTruthActive() ? 128u : 0u) | (host_rt ? 256u : 0u) |
               ((host_rt && cp.render_target_cache_->depth_float24_convert_in_pixel_shader()) ? 512u : 0u);
    const float ratio = cp.NativeTruthDrawRatio();
    std::memcpy(&in.ratio, &ratio, 4);
    return XXH3_64bits(&in, sizeof(in));
  }
  static uint64_t Submission(VCP& cp) { return cp.GetCurrentSubmission(); }
  // ---- Session 93 (gpu_d3d_typed_memo_reuse / gpu_d3d_typed_repeat): the processor-side inputs of the typed memo.
  // Externals plus the other things Build's state part reads from the processor and the settings (the native renderer's frame latches, the swap / lean settings,
  // the pipeline cache's specialisation latch, the DLSS bias, the objmv / merge latches, the restore shader count): one hash, per draw.
  static uint32_t BoundsListHash() {
    const std::string& l = REXCVAR_GET(gpu_native_truth_bounds_vs);
    return l.empty() ? 0u : uint32_t(XXH3_64bits(l.data(), l.size()));
  }
  static uint64_t ExternalsReuse(VCP& cp) {
    struct {
      uint64_t base;
      uint32_t flags, owned_front, spec, dlss, restore_n, pad[2];
    } in;
    std::memset(&in, 0, sizeof(in));
    in.base = Externals(cp);
    native::NativeRenderer& native = *cp.native_;
    // Session 95 (gpu_d3d_typed_latch): the settings' bits are read once a frame; a flip mid-frame applies next frame and bumps the generation in the hash (as Externals does).
    uint32_t cvar_bits, owned_front_now;
    if (REXCVAR_GET(gpu_d3d_typed_latch)) {
      static struct { uint64_t frame = UINT64_MAX; uint32_t bits = 0, of = 0, gen = 0, bounds = 0; } cvl;
      if (cvl.frame != cp.frame_current_) {
        const uint32_t nb = (REXCVAR_GET(gpu_native_swap_skip_translation) ? 4u : 0u) | (REXCVAR_GET(gpu_native_swap_skip_translation_pixel) ? 8u : 0u) |
                            (REXCVAR_GET(gpu_native_vs_swap) ? 16u : 0u) | (REXCVAR_GET(gpu_native_shader_swap) ? 32u : 0u) | (REXCVAR_GET(gpu_native_lean) ? 64u : 0u) |
                            (REXCVAR_GET(gpu_native_truth_skip_restores) ? 128u : 0u) | (REXCVAR_GET(gpu_native_post_depth_unrounded_debug) ? 256u : 0u) |
                            (REXCVAR_GET(gpu_native_owned_alpha_learn) ? 512u : 0u) | (REXCVAR_GET(gpu_native_owned_alpha_learn_constants) ? 1024u : 0u) |
                            (REXCVAR_GET(gpu_allow_invalid_fetch_constants) ? 8192u : 0u);
        const uint32_t nof = REXCVAR_GET(gpu_d3d_owned_front);
        const uint32_t nbv = BoundsListHash();
        if (cvl.frame != UINT64_MAX && (nb != cvl.bits || nof != cvl.of || nbv != cvl.bounds)) ++cvl.gen;
        cvl.bounds = nbv;
        cvl.bits = nb;
        cvl.of = nof;
        cvl.frame = cp.frame_current_;
      }
      cvar_bits = cvl.bits;
      owned_front_now = cvl.of;
      in.pad[0] = cvl.gen;
      in.pad[1] = cvl.bounds;
    } else {
      in.pad[1] = BoundsListHash();
      cvar_bits = (REXCVAR_GET(gpu_native_swap_skip_translation) ? 4u : 0u) | (REXCVAR_GET(gpu_native_swap_skip_translation_pixel) ? 8u : 0u) |
                  (REXCVAR_GET(gpu_native_vs_swap) ? 16u : 0u) | (REXCVAR_GET(gpu_native_shader_swap) ? 32u : 0u) | (REXCVAR_GET(gpu_native_lean) ? 64u : 0u) |
                  (REXCVAR_GET(gpu_native_truth_skip_restores) ? 128u : 0u) | (REXCVAR_GET(gpu_native_post_depth_unrounded_debug) ? 256u : 0u) |
                  (REXCVAR_GET(gpu_native_owned_alpha_learn) ? 512u : 0u) | (REXCVAR_GET(gpu_native_owned_alpha_learn_constants) ? 1024u : 0u) |
                  (REXCVAR_GET(gpu_allow_invalid_fetch_constants) ? 8192u : 0u);
      owned_front_now = REXCVAR_GET(gpu_d3d_owned_front);
    }
    in.flags = (native.enabled() ? 1u : 0u) | (native.replacing() ? 2u : 0u) | cvar_bits | (cp.objmv_latched_ ? 2048u : 0u) | (cp.repeat_merge_latched_ ? 4096u : 0u) |
               (cp.active_occlusion_query_.valid ? 16384u : 0u) | (native.TruthTwinPitch(0xFFFFFFFFu) ? 32768u : 0u);
    in.owned_front = owned_front_now;
    const VulkanPipelineCache::NativeSpecLatch& sp = cp.pipeline_cache_->NativeSpec();
    in.spec = (sp.native ? 1u : 0u) | (sp.endian ? 2u : 0u) | (sp.ps_texture ? 4u : 0u) | (sp.ps_implicit_lod ? 8u : 0u) | (sp.ps_epilogue ? 16u : 0u) |
              (sp.vs_fetch_finite ? 32u : 0u) | (sp.ps_wanted ? 64u : 0u) | (uint32_t(sp.ps_mul) << 8) | (uint32_t(sp.vs_mul) << 12) | (uint32_t(sp.fma) << 16);
    in.dlss = cp.DlssLodBiasQuarters();
    in.restore_n = cp.native_restore_shader_count_.load(std::memory_order_acquire);
    return XXH3_64bits(&in, sizeof(in));
  }
  static const void* ActiveVs(VCP& cp) { return static_cast<const Shader*>(cp.active_vertex_shader()); }
  static const void* ActivePs(VCP& cp) { return static_cast<const Shader*>(cp.active_pixel_shader()); }
  static bool ProcessInto(VCP& cp, const PrimitiveProcessor::DrawInput& input, PrimitiveProcessor::ProcessingResult& result) {
    return cp.primitive_processor_->Process(result, input);
  }
  static uint32_t EndianKeyNow(VCP& cp, const Shader& vertex_shader, const VulkanPipelineCache::TypedKeyState& keys) {
    return cp.pipeline_cache_->GetVertexFetchEndianKey(vertex_shader, keys);
  }
  static uint32_t PixelSpecKeyNow(VCP& cp, const Shader* pixel_shader, const VulkanPipelineCache::TypedKeyState& keys) {
    return pixel_shader ? cp.pipeline_cache_->GetNativePixelSpecKey(pixel_shader, keys) : 0;
  }
  // The viewport origin of a memo entry with this frame's DLAA jitter (Build's `tmpl` branch, restated).
  static void ViewportJitter(VCP& cp, Draw& d, const TemplateOut& out) {
    d.viewport.x = out.base_x;
    d.viewport.y = out.base_y;
    const uint32_t scale_x = cp.texture_cache_->draw_resolution_scale_x();
    if (DlaaJitterOn() && !cp.dlaa_scene_done_ && out.vinfo.xy_extent[0] &&
        out.vinfo.xy_extent[0] >= uint32_t(std::max(0, REXCVAR_GET(gpu_dlaa_jitter_min_width))) * scale_x) {
      d.viewport.x += cp.dlaa_jitter_[0];
      d.viewport.y += cp.dlaa_jitter_[1];
    }
  }
  static uint64_t RepeatEpoch(VCP& cp) { return cp.shared_memory_->repeat_epoch(); }
  // Session 95 (r3cmd, gpu_d3d_draw_record).
  static bool DlaaJitter() { return DlaaJitterOn(); }
  static const SpirvShaderTranslator::SystemConstants& SystemConstantsOf(VCP& cp) { return cp.system_constants_; }
  static bool RequestIndexSpan(VCP& cp, uint32_t base, uint32_t bytes) {
    SharedMemory::RamReaderScope ram_reader(*cp.shared_memory_, SharedMemory::RamReader::kIndex);
    return cp.shared_memory_->RequestRange(base, bytes);
  }
  // The block a build under the same inputs left. A build writes some fields only when the draw reads them, so what a record's block holds of the others is older than the live block's: no shader reads
  // them. The selector of the storage window is the window step's, written after this.
  static bool SystemConstantsFromRecord(VCP& cp, OwnedBindInputs& bi, const DrawRec& r) {
    using SC = SpirvShaderTranslator::SystemConstants;
    if (!bi.image_view_format_swizzle) return false;  // (the host swizzles of the bindings are inputs then: the full build)
    if (d3d_tex::g_active.load(std::memory_order_relaxed) && d3d_tex::Enabled() && d3d_tex::UseIds() && !d3d_tex::VerifyKeys()) return false;  // (the signs can change without a fetch word moving)
    bi.scaled_from_cache = 0;
    if (bi.scale_x > 1 || bi.scale_y > 1) {
      uint32_t remaining = bi.used_texture_mask;
      while (remaining) {
        const uint32_t i = uint32_t(std::countr_zero(remaining));
        remaining &= remaining - 1;
        if (cp.texture_cache_->IsActiveTextureResolutionScaled(i)) bi.scaled_from_cache |= 1u << i;
      }
    }
    if (bi.scaled_from_cache != r.scaled) return false;
    cp.D3dTypedSystemConstantsBegin();
    constexpr size_t kWin = offsetof(SC, shared_memory_window_base_dwords), kWinEnd = kWin + sizeof(cp.system_constants_.shared_memory_window_base_dwords);
    uint8_t* const a = reinterpret_cast<uint8_t*>(&cp.system_constants_);
    const uint8_t* const b = reinterpret_cast<const uint8_t*>(&r.sys);
    if (std::memcmp(a, b, kWin) != 0 || std::memcmp(a + kWinEnd, b + kWinEnd, sizeof(SC) - kWinEnd) != 0) {
      std::memcpy(a, b, kWin);
      std::memcpy(a + kWinEnd, b + kWinEnd, sizeof(SC) - kWinEnd);
      cp.current_constant_buffers_up_to_date_ &= ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferSystem);
    }
    return true;
  }
  // Session 95 (gpu_d3d_draw_slot): the same step on a slot hit, with the same checks of what is outside the words. The block is compared by content hash, not by bytes: the same hash = nothing to do; else the
  // record's block goes in without reading the live one.
  static bool SystemConstantsFromSlot(VCP& cp, OwnedBindInputs& bi, const DrawRec& r, uint64_t sys_hash) {
    using SC = SpirvShaderTranslator::SystemConstants;
    if (!bi.image_view_format_swizzle) return false;
    if (d3d_tex::g_active.load(std::memory_order_relaxed) && d3d_tex::Enabled() && d3d_tex::UseIds() && !d3d_tex::VerifyKeys()) return false;
    bi.scaled_from_cache = 0;
    if (bi.scale_x > 1 || bi.scale_y > 1) {
      uint32_t remaining = bi.used_texture_mask;
      while (remaining) {
        const uint32_t i = uint32_t(std::countr_zero(remaining));
        remaining &= remaining - 1;
        if (cp.texture_cache_->IsActiveTextureResolutionScaled(i)) bi.scaled_from_cache |= 1u << i;
      }
    }
    if (bi.scaled_from_cache != r.scaled) return false;
    cp.D3dTypedSystemConstantsBegin();
    if (g_draw_slot_sys_hash == sys_hash) {
      ++g_draw_slots.stats.sys_same;
      return true;
    }
    constexpr size_t kWin = offsetof(SC, shared_memory_window_base_dwords), kWinEnd = kWin + sizeof(cp.system_constants_.shared_memory_window_base_dwords);
    uint8_t* const a = reinterpret_cast<uint8_t*>(&cp.system_constants_);
    const uint8_t* const b = reinterpret_cast<const uint8_t*>(&r.sys);
    std::memcpy(a, b, kWin);
    std::memcpy(a + kWinEnd, b + kWinEnd, sizeof(SC) - kWinEnd);
    cp.current_constant_buffers_up_to_date_ &= ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferSystem);
    g_draw_slot_sys_hash = sys_hash;
    ++g_draw_slots.stats.sys_copied;
    return true;
  }
  static void Jitter(VCP& cp, float out[2]) {
    out[0] = cp.dlaa_jitter_[0];
    out[1] = cp.dlaa_jitter_[1];
  }
  static bool OcclusionValid(VCP& cp) { return cp.active_occlusion_query_.valid; }
  // Session 94 (gpu_d3d_typed_lean_bind): SystemConstantsInto without the work that cannot change the block.
  //  - The texture cache is asked only for what the build reads: the resolution-scaled bits at a draw resolution scale above 1, the host swizzles without
  //    imageViewFormatSwizzle, both for the used textures only (the build reads nothing else of them).
  //  - The build (BuildSystemConstants) is a function of its inputs plus, for the fields it writes conditionally (clip planes, point sizes, the signs of
  //    used textures, the FSI render target words), the block's earlier contents. So it is skipped when (a) every input equals the last build's - the
  //    decoder's render and fetch epochs (valid only while the state is the live decoder state, as the S93 memo's fast path requires), the draw's own
  //    words and the primitive pass fields it reads, the bind inputs - and (b) the block still equals what that build left (any other writer, the old
  //    UpdateSystemConstantValues, shows up as a difference and forces a rebuild). Re-running the build on identical inputs over an identical block
  //    writes the same bytes, so the skip changes nothing. shared_memory_window_base_dwords is the storage-window selector's, written after this step
  //    (command_processor.cpp ~19325) and not by the build: excluded from (b).
  //    With gpu_d3d_typed_memo_verify every skip is checked against a from-scratch build into a scratch block.
  static void SystemConstantsIntoLean(VCP& cp, OwnedBindInputs& bi) {
    using SC = SpirvShaderTranslator::SystemConstants;
    bi.scaled_from_cache = 0;
    const bool need_scaled = bi.scale_x > 1 || bi.scale_y > 1;
    const bool need_swizzle = !bi.image_view_format_swizzle;
    if (need_scaled || need_swizzle) {
      uint32_t remaining = bi.used_texture_mask;
      while (remaining) {
        const uint32_t i = uint32_t(std::countr_zero(remaining));
        remaining &= remaining - 1;
        if (need_scaled && cp.texture_cache_->IsActiveTextureResolutionScaled(i)) bi.scaled_from_cache |= 1u << i;
        if (need_swizzle) bi.host_swizzle_from_cache[i] = cp.texture_cache_->GetActiveTextureHostSwizzle(i);
      }
    }
    cp.D3dTypedSystemConstantsBegin();
    struct Key {
      // tex_words: XXH3 of the 6 fetch words of every used texture slot, in place of the fetch epoch (the vertex fetch constants share the fetch bank and move it on most draws).
      uint64_t render, tex_words;
      uint32_t init, dma_size, indx_offset;
      uint32_t used, scale_x, scale_y, color_mask, depth_control, scaled;
      uint32_t flags;
      uint32_t p_index_type, p_reset, p_vs_type, p_tess, p_endian, p_line_loop, p_guest_base, p_guest_prim;
      uint32_t polygonal, s32;
      uint32_t ndc[6], xy[2];
      uint32_t swz[32];
    } key;
    std::memset(&key, 0, sizeof(key));
    const OwnedRecordState& s = *bi.s;
    // BindingFor can take the signs from the d3d_tex table (ids mode without key verify), which changes without a fetch word moving: no skip in that mode.
    const bool ids_mode = d3d_tex::g_active.load(std::memory_order_relaxed) && d3d_tex::Enabled() && d3d_tex::UseIds() && !d3d_tex::VerifyKeys();
    const bool live = !ids_mode && rex::graphics::free_counters::g_cut.active && &s == &OwnedRecordStateNow();
    bool skipped = false;
    static Key last_key;
    static SC last_after;
    static bool last_valid = false;
    if (live) {
      const OwnedRecordEpochs& ep = OwnedRecordEpochsNow();
      key.render = ep.render;
      {
        uint32_t tw[32 * 6];
        size_t n = 0;
        uint32_t remaining = bi.used_texture_mask;
        while (remaining) {
          const uint32_t i = uint32_t(std::countr_zero(remaining));
          remaining &= remaining - 1;
          std::memcpy(tw + n, s.fetch + 6 * size_t(i), 6 * sizeof(uint32_t));
          n += 6;
        }
        key.tex_words = XXH3_64bits(tw, n * sizeof(uint32_t));
      }
      key.init = s.draw_initiator.value;
      key.dma_size = s.dma_size.value;
      key.indx_offset = s.indx_offset.value;
      key.used = bi.used_texture_mask;
      key.scale_x = bi.scale_x;
      key.scale_y = bi.scale_y;
      key.color_mask = bi.color_mask;
      static_assert(sizeof(bi.depth_control) == sizeof(uint32_t));
      std::memcpy(&key.depth_control, &bi.depth_control, sizeof(uint32_t));
      key.scaled = bi.scaled_from_cache;
      key.flags = (bi.fsi ? 1u : 0u) | (bi.host_rt ? 2u : 0u) | (bi.rg16_truncated ? 4u : 0u) | (bi.rgba16_truncated ? 8u : 0u) | (bi.native_truth ? 16u : 0u) |
                  (bi.gamma_as_unorm16 ? 32u : 0u) | (bi.image_view_format_swizzle ? 64u : 0u) |
                  (REXCVAR_GET(gpu_allow_invalid_fetch_constants) ? 128u : 0u);
      const PrimitiveProcessor::ProcessingResult& p = bi.primitive;
      key.p_index_type = uint32_t(p.index_buffer_type);
      key.p_reset = p.host_primitive_reset_enabled ? 1u : 0u;
      key.p_vs_type = uint32_t(p.host_vertex_shader_type);
      key.p_tess = uint32_t(p.tessellation_mode);
      key.p_endian = uint32_t(p.host_shader_index_endian);
      key.p_line_loop = p.line_loop_closing_index;
      key.p_guest_base = p.guest_index_base;
      key.p_guest_prim = uint32_t(p.guest_primitive_type);
      key.polygonal = bi.polygonal ? 1u : 0u;
      key.s32 = bi.shader_32bit_index_dma ? 1u : 0u;
      std::memcpy(key.ndc, bi.ndc_scale, sizeof(bi.ndc_scale));
      std::memcpy(key.ndc + 3, bi.ndc_offset, sizeof(bi.ndc_offset));
      key.xy[0] = bi.xy_extent[0];
      key.xy[1] = bi.xy_extent[1];
      if (need_swizzle) {
        uint32_t remaining = bi.used_texture_mask;
        while (remaining) {
          const uint32_t i = uint32_t(std::countr_zero(remaining));
          remaining &= remaining - 1;
          key.swz[i] = bi.host_swizzle_from_cache[i];
        }
      }
      if (last_valid && std::memcmp(&key, &last_key, sizeof(Key)) == 0) {
        const SC& cur = cp.system_constants_;
        constexpr size_t kWin = offsetof(SC, shared_memory_window_base_dwords), kWinEnd = kWin + sizeof(cur.shared_memory_window_base_dwords);
        const uint8_t* a = reinterpret_cast<const uint8_t*>(&cur);
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&last_after);
        skipped = std::memcmp(a, b, kWin) == 0 && std::memcmp(a + kWinEnd, b + kWinEnd, sizeof(SC) - kWinEnd) == 0;  // (b)
      }
    }
    if (skipped) {
      ++g_lean_bind_stat[6];
      if (REXCVAR_GET(gpu_d3d_typed_memo_verify)) {
        SC scratch = cp.system_constants_;
        OwnedFrontBindBuildSystem(bi, scratch);
        scratch.shared_memory_window_base_dwords = cp.system_constants_.shared_memory_window_base_dwords;
        if (std::memcmp(&scratch, &cp.system_constants_, sizeof(SC)) != 0) {
          ++g_lean_bind_stat[8];
          static uint32_t logged = 0;
          if (logged < 16) {
            ++logged;
            REXGPU_WARN("[d3d][typed] lean_bind verify: a skipped system constants build would have changed the block (#{})", logged);
          }
          cp.system_constants_ = scratch;  // (what the full build gives is what is drawn)
          cp.current_constant_buffers_up_to_date_ &= ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferSystem);
        }
      }
      return;
    }
    ++g_lean_bind_stat[7];
    const SC before = cp.system_constants_;
    OwnedFrontBindBuildSystem(bi, cp.system_constants_);
    if (std::memcmp(&before, &cp.system_constants_, sizeof(before)) != 0) {
      cp.current_constant_buffers_up_to_date_ &= ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferSystem);
    }
    if (live) {
      last_key = key;
      last_after = cp.system_constants_;
      last_valid = true;
    } else {
      last_valid = false;
    }
  }

  // M4f-A: the system constants of a typed draw into the processor's own block (see OwnedFrontSystemConstants).
  static void SystemConstantsInto(VCP& cp, OwnedBindInputs& bi) {
    if (REXCVAR_GET(gpu_d3d_typed_lean_bind)) {
      SystemConstantsIntoLean(cp, bi);
      return;
    }
    bi.scaled_from_cache = 0;
    for (uint32_t i = 0; i < 32; ++i) {
      if (!(bi.used_texture_mask & (1u << i))) continue;
      if (cp.texture_cache_->IsActiveTextureResolutionScaled(i)) bi.scaled_from_cache |= 1u << i;
      bi.host_swizzle_from_cache[i] = cp.texture_cache_->GetActiveTextureHostSwizzle(i);
    }
    cp.D3dTypedSystemConstantsBegin();
    const SpirvShaderTranslator::SystemConstants before = cp.system_constants_;
    OwnedFrontBindBuildSystem(bi, cp.system_constants_);
    if (std::memcmp(&before, &cp.system_constants_, sizeof(before)) != 0) {
      cp.current_constant_buffers_up_to_date_ &= ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferSystem);
    }
  }
};

// ---------------------------------------------------------------------------------------------------------------------------------
namespace {

struct Compare {
  uint32_t kind;
  uint64_t vs_hash, ps_hash;
  bool any = false;
  // (M4d: the counters and the log tag are the caller's; the defaults are M4b's.)
  uint64_t* by_field_p = g_w.by_field;
  bool* logged_p = g_field_logged;
  uint32_t* nlogged_p = &g_logged;
  const char* tag = "M4b";
  void Note(Field f, uint64_t old_v, uint64_t typed_v) {
    any = true;
    ++by_field_p[f];
    if (!logged_p[f] && *nlogged_p < 96) {
      logged_p[f] = true;
      ++*nlogged_p;
      REXGPU_INFO("[d3d][front] {} mismatch #{}: {} old {:X} typed {:X} (claim {}, vertex shader {:016X}, pixel shader {:016X})", tag, *nlogged_p,
                  kFieldNames[f], old_v, typed_v, kind == kOwnedFrontFull ? "full" : kind == kOwnedFrontLean ? "lean" : "repeat", vs_hash,
                  ps_hash);
    }
  }
  template <class T>
  void Eq(Field f, const T& a, const T& b) {
    if (a != b) Note(f, uint64_t(a), uint64_t(b));
  }
  void EqF(Field f, float a, float b) {
    if (Bits(a) != Bits(b)) Note(f, Bits(a), Bits(b));
  }
  void EqH(Field f, const void* a, const void* b) {  // handles
    if (a != b) Note(f, uint64_t(reinterpret_cast<uintptr_t>(a)), uint64_t(reinterpret_cast<uintptr_t>(b)));
  }
};

// ---------------------------------------------------------------------------------------------------------------------------------
// M4d (latch bit 3): the template memo and the crowd shadow.
//
// Template: a memo keyed by (render-state epoch, shader pair, primitive, mode) holding a prepared typed Draw. The epoch is a hash of the typed
// render-state fields the memo's parts depend on (everything of the record state but the draw's own words, the per-draw vertex index offset
// words, the user clip planes and the constant banks) plus the processor-side inputs the viewport derivation reads (frame, DLAA scene flag,
// HUD / menu flags, the tiling viewport fix, the jitter). On a hit - for every claim kind - the Draw is built again from the memo + the delta
// (Build with `tmpl`: viewport, scissor, depth / stencil / cull / bias taken from the memo; primitive processing, shader modifications and
// modules, the fetch-dependent keys derived again) and compared with the from-scratch build. A key that finds an entry of the same shader pair
// and primitive with another epoch is an invalidation (the entry is replaced).
// Repeat: the previous built draw, kept with a key of (epoch, shader pair, primitive, mode, draw words, fetch constants, occlusion): a draw
// whose key equals it is the typed form of the old path's repeat (only vertex float constants changed), and its Draw is the previous one
// verbatim; compared with the from-scratch build.
struct TmplWindow {
  uint64_t lookups[kOwnedFrontKinds] = {}, hits[kOwnedFrontKinds] = {}, misses[kOwnedFrontKinds] = {};
  uint64_t invalidations = 0, evictions = 0, inval_state = 0, inval_externals = 0, inval_both = 0, hits_cross_frame = 0;
  uint64_t compared = 0, not_built = 0, mismatch_draws = 0;
  uint64_t member_mismatch_draws = 0, member_by_field[kFieldCount] = {};  // the crowd members' own M4b compares, kept apart from M4b's
  uint64_t by_field[kFieldCount] = {};
  uint64_t rep_found[kOwnedFrontKinds] = {};  // the typed repeat key equals the previous built draw's
  uint64_t rep_claim_unmatched = 0;           // the old path claimed a repeat, the typed key did not match
  uint64_t rep_mismatch_draws = 0;
  uint64_t rep_by_field[kFieldCount] = {};
  uint64_t member_draws = 0;  // claims built from a crowd member's typed state (no record item of their own)
  // crowd
  uint64_t crowd_items = 0, crowd_no_snapshot = 0, crowd_no_banks = 0;
  uint64_t c76_compared = 0, c76_mismatch = 0, pretake_compared = 0, pretake_mismatch = 0;
  uint64_t ids_mapped = 0, id_conflicts = 0, id_zero = 0;
  uint64_t ns = 0;
};
TmplWindow g_t;
bool g_t_logged_field[kFieldCount] = {}, g_t_logged_rep[kFieldCount] = {}, g_t_logged_member[kFieldCount] = {};
uint32_t g_t_logged_member_n = 0;
uint32_t g_t_logged = 0, g_t_crowd_logged = 0;

constexpr uint32_t kMemoSlots = 64;
struct TmplEntry {
  bool valid = false;
  uint64_t vs = 0, ps = 0, state_hash = 0, ext_hash = 0, frame = 0;
  uint32_t prim = 0, mode = 0;
  TemplateOut out;
  Draw draw;
};
TmplEntry g_memo[kMemoSlots];
struct RepLast {
  bool valid = false;
  uint64_t key = 0;
  uint64_t seq = 0;  // OwnedRecordDrawSeq() of the draw item it was built for
  Draw draw;
};
RepLast g_rep;
std::unordered_map<uint32_t, uint64_t> g_id_fwd;  // old packet id (the item's arena address) -> typed id (token seq << 8 | item index), this frame
std::unordered_map<uint64_t, uint32_t> g_id_rev;
uint64_t g_id_frame = UINT64_MAX;

// Session 93: the counters of the typed memo reuse and the typed repeat (the `[d3d][memo]` line), and the typed repeat tracker.
struct ReuseWindow {
  uint64_t hits_fast = 0, hits_hashed = 0, hits_cross_frame = 0, misses = 0, refused = 0;
  uint64_t demoted_shader = 0, demoted_primitive = 0, demoted_fetch_keys = 0, not_stored_restore = 0, evictions = 0;
  uint64_t verify_compared = 0, verify_mismatch_draws = 0;
  uint64_t by_field[kFieldCount] = {};
  std::map<const char*, uint64_t> prep_mismatch;  // (keyed by the literal's address)
  uint64_t rep_checked = 0, rep_candidates = 0, rep[3] = {};  // OwnedRepeatOutcome: tried, drawn, refused
};
ReuseWindow g_rw;
bool g_rw_logged[kFieldCount] = {};
uint32_t g_rw_nlogged = 0;
struct TRepKey {
  bool valid = false;
  uint64_t serial = 0, seq = 0, render = 0, ps_float = 0, bool_loop = 0, fetch = 0, ext = 0, vs = 0, ps = 0, bin_mask = 0, bin_select = 0;
  uint32_t prim = 0, mode = 0, init = 0, dma_base = 0, dma_size = 0, occlusion = 0, viz = 0, idx[3] = {};
  float jitter[2] = {};
};
uint64_t g_repeat_base_serial_dummy = 0;
TRepKey g_trep, g_trep_prev;  // the key recorded after the last typed draw; the one before it, which the draw in hand is compared with

// The render-state part of the key: the typed fields the memo's parts are a function of (see the comment at the top of this block).
uint64_t StateHash(const OwnedRecordState& s) {
  const uint8_t* b = reinterpret_cast<const uint8_t*>(&s);
  constexpr size_t kA = offsetof(OwnedRecordState, user_clip), kB0 = offsetof(OwnedRecordState, window_offset),
                   kB1 = offsetof(OwnedRecordState, min_vtx_indx), kC0 = offsetof(OwnedRecordState, reset_indx),
                   kC1 = offsetof(OwnedRecordState, draw_initiator), kD0 = offsetof(OwnedRecordState, program_cntl),
                   kD1 = offsetof(OwnedRecordState, vs_float);
  uint64_t h = XXH3_64bits(b, kA);
  h = XXH3_64bits_withSeed(b + kB0, kB1 - kB0, h);
  h = XXH3_64bits_withSeed(b + kC0, kC1 - kC0, h);
  h = XXH3_64bits_withSeed(b + kD0, kD1 - kD0, h);
  return h;
}

// The full field compare of two Draws (o = the from-scratch build, d = the memo / template build). Modules, handles and the index buffer are
// compared as they are: both sides are typed lookups made at the same instant.
void CompareDraws(Compare& c, const Draw& o, const Draw& d) {
  c.Eq(F_vertex_shader_hash, o.vertex_shader_hash, d.vertex_shader_hash);
  c.Eq(F_pixel_shader_hash, o.pixel_shader_hash, d.pixel_shader_hash);
  c.Eq(F_pixel_spec_key, o.pixel_spec_key, d.pixel_spec_key);
  c.Eq(F_pixel_ext, o.pixel_ext, d.pixel_ext);
  c.Eq(F_vertex_mul_key, o.vertex_mul_key, d.vertex_mul_key);
  c.Eq(F_vertex_ext, o.vertex_ext, d.vertex_ext);
  c.Eq(F_vertex_modification, o.vertex_modification, d.vertex_modification);
  c.Eq(F_pixel_modification, o.pixel_modification, d.pixel_modification);
  c.EqH(F_vertex_shader_module, o.vertex_shader, d.vertex_shader);
  c.EqH(F_pixel_shader_module, o.pixel_shader, d.pixel_shader);
  c.Eq(F_float24_depth, o.float24_depth, d.float24_depth);
  c.Eq(F_color_quantize, o.color_quantize, d.color_quantize);
  c.Eq(F_ps_float24, o.ps_float24, d.ps_float24);
  c.Eq(F_ps_twin, o.ps_twin, d.ps_twin);
  c.Eq(F_ps_quantize, o.ps_quantize, d.ps_quantize);
  for (uint32_t i = 0; i < 4; ++i) c.Eq(F_layout_counts, o.layout_counts[i], d.layout_counts[i]);
  c.EqH(F_layout, reinterpret_cast<const void*>(o.layout), reinterpret_cast<const void*>(d.layout));
  c.Eq(F_restore, o.restore, d.restore);
  if (o.restore || d.restore) {
    c.Eq(F_restore_pages, uint64_t(o.restore_color_page) << 32 | o.restore_depth_page, uint64_t(d.restore_color_page) << 32 | d.restore_depth_page);
  }
  c.EqF(F_viewport_x, o.viewport.x, d.viewport.x);
  c.EqF(F_viewport_y, o.viewport.y, d.viewport.y);
  c.EqF(F_viewport_w, o.viewport.width, d.viewport.width);
  c.EqF(F_viewport_h, o.viewport.height, d.viewport.height);
  if (!(o.viewport.minDepth == d.viewport.minDepth)) c.Note(F_viewport_min_depth, Bits(o.viewport.minDepth), Bits(d.viewport.minDepth));
  if (!(o.viewport.maxDepth == d.viewport.maxDepth)) c.Note(F_viewport_max_depth, Bits(o.viewport.maxDepth), Bits(d.viewport.maxDepth));
  if (o.scissor.offset.x != d.scissor.offset.x || o.scissor.offset.y != d.scissor.offset.y || o.scissor.extent.width != d.scissor.extent.width ||
      o.scissor.extent.height != d.scissor.extent.height) {
    c.Note(F_scissor, (uint64_t(uint16_t(o.scissor.offset.x)) << 48) | (uint64_t(uint16_t(o.scissor.offset.y)) << 32) |
                          (uint64_t(uint16_t(o.scissor.extent.width)) << 16) | uint16_t(o.scissor.extent.height),
           (uint64_t(uint16_t(d.scissor.offset.x)) << 48) | (uint64_t(uint16_t(d.scissor.offset.y)) << 32) |
               (uint64_t(uint16_t(d.scissor.extent.width)) << 16) | uint16_t(d.scissor.extent.height));
  }
  c.Eq(F_guest_primitive, o.guest_primitive, d.guest_primitive);
  c.Eq(F_host_primitive, o.host_primitive, d.host_primitive);
  c.Eq(F_host_vs_type, o.host_vs_type, d.host_vs_type);
  c.Eq(F_polygonal, o.polygonal, d.polygonal);
  c.Eq(F_raster_supported, o.raster.supported, d.raster.supported);
  c.Eq(F_raster_topology, uint32_t(o.raster.topology), uint32_t(d.raster.topology));
  c.Eq(F_raster_restart, o.raster.primitive_restart, d.raster.primitive_restart);
  c.Eq(F_raster_expanded, o.raster.expanded, d.raster.expanded);
  c.EqH(F_raster_geometry_shader, o.raster.geometry_shader, d.raster.geometry_shader);
  c.Eq(F_raster_cull_mode, uint32_t(o.raster.cull_mode), uint32_t(d.raster.cull_mode));
  c.Eq(F_raster_front_face, o.raster.front_face_clockwise, d.raster.front_face_clockwise);
  c.Eq(F_raster_depth_test, o.raster.depth_test, d.raster.depth_test);
  c.Eq(F_raster_depth_write, o.raster.depth_write, d.raster.depth_write);
  c.Eq(F_raster_depth_compare, uint32_t(o.raster.depth_compare), uint32_t(d.raster.depth_compare));
  c.EqF(F_raster_bias_constant, o.raster.depth_bias_constant, d.raster.depth_bias_constant);
  c.EqF(F_raster_bias_slope, o.raster.depth_bias_slope, d.raster.depth_bias_slope);
  c.Eq(F_raster_stencil, o.raster.stencil, d.raster.stencil);
  c.Eq(F_raster_stencil_ops, o.raster.stencil_ops, d.raster.stencil_ops);
  c.Eq(F_raster_clip_disable, o.raster.clip_disable, d.raster.clip_disable);
  c.Eq(F_stencil_ref, o.stencil_ref, d.stencil_ref);
  for (uint32_t i = 0; i < 2; ++i) {
    c.Eq(F_stencil_refs, o.stencil_refs[i], d.stencil_refs[i]);
    c.Eq(F_stencil_read_masks, o.stencil_read_masks[i], d.stencil_read_masks[i]);
    c.Eq(F_stencil_write_masks, o.stencil_write_masks[i], d.stencil_write_masks[i]);
  }
  c.Eq(F_color_mask_all, o.color_mask_all, d.color_mask_all);
  for (uint32_t i = 0; i < 4; ++i) c.EqF(F_blend_constants, o.blend_constants[i], d.blend_constants[i]);
  c.Eq(F_occlusion_query, o.occlusion_query, d.occlusion_query);
  c.Eq(F_surface_info, o.surface_info, d.surface_info);
  c.Eq(F_color_info, o.color_info, d.color_info);
  c.Eq(F_depth_info, o.depth_info, d.depth_info);
  c.Eq(F_vertex_count, o.vertex_count, d.vertex_count);
  c.Eq(F_screen_space, o.screen_space, d.screen_space);
  c.Eq(F_memexport, o.memexport, d.memexport);
  c.Eq(F_color_write_mask, o.color_write_mask, d.color_write_mask);
  c.Eq(F_blend_control, o.blend_control, d.blend_control);
  c.Eq(F_indexed, o.indexed, d.indexed);
  c.EqH(F_index_buffer, o.index_buffer, d.index_buffer);
  c.Eq(F_index_offset, uint64_t(o.index_offset), uint64_t(d.index_offset));
  c.Eq(F_index_type, uint32_t(o.index_type), uint32_t(d.index_type));
  c.Eq(F_merge_stride, o.merge_stride, d.merge_stride);
  c.Eq(F_float_count, o.float_count, d.float_count);
}

// Called for every claim the M4b build produced (`scratch` is that from-scratch Draw, `sout` what its Build reported for an entry).
void TemplateShadow(VulkanCommandProcessor& cp, const OwnedRecordState& s, const Draw& old, const Draw& scratch, const TemplateOut& sout, uint32_t kind) {
  const uint64_t t0 = NowNs();
  ++g_t.lookups[kind];
  const uint64_t frame = OwnedFrontAccess::Frame(cp);
  const uint64_t vs = s.shader[0].hash, ps = s.shader[1].hash;
  const uint32_t prim = uint32_t(s.draw_initiator.prim_type);
  const uint32_t mode = uint32_t(s.modecontrol.edram_mode) | (uint32_t(s.draw_initiator.major_mode) << 8);
  const uint64_t state_hash = StateHash(s);
  const uint64_t ext_hash = OwnedFrontAccess::Externals(cp);
  struct {
    uint64_t vs, ps;
    uint32_t prim, mode;
  } slot_key{vs, ps, prim, mode};
  TmplEntry& e = g_memo[XXH3_64bits(&slot_key, sizeof(slot_key)) % kMemoSlots];
  const bool same_pair = e.valid && e.vs == vs && e.ps == ps && e.prim == prim && e.mode == mode;
  if (same_pair && e.state_hash == state_hash && e.ext_hash == ext_hash) {
    ++g_t.hits[kind];
    if (e.frame != frame) ++g_t.hits_cross_frame;
    Draw h;
    bool a = false, b = false, g = false;
    TemplateIn in;
    static_cast<TemplateOut&>(in) = e.out;
    in.draw = &e.draw;
    const Skip sk = OwnedFrontAccess::Build(cp, s, old, h, a, b, g, nullptr, &in, nullptr);
    if (sk != kSkipNone) {
      ++g_t.not_built;
    } else {
      ++g_t.compared;
      Compare c;
      c.kind = kind;
      c.vs_hash = old.vertex_shader_hash;
      c.ps_hash = old.pixel_shader_hash;
      c.by_field_p = g_t.by_field;
      c.logged_p = g_t_logged_field;
      c.nlogged_p = &g_t_logged;
      c.tag = "M4d template (memo + delta vs from scratch)";
      CompareDraws(c, scratch, h);
      if (c.any) ++g_t.mismatch_draws;
    }
  } else {
    ++g_t.misses[kind];
    if (same_pair) {
      ++g_t.invalidations;
      ++(e.state_hash != state_hash ? (e.ext_hash != ext_hash ? g_t.inval_both : g_t.inval_state) : g_t.inval_externals);
    } else if (e.valid) {
      ++g_t.evictions;
    }
    e.valid = true;
    e.vs = vs;
    e.ps = ps;
    e.state_hash = state_hash;
    e.ext_hash = ext_hash;
    e.frame = frame;
    e.prim = prim;
    e.mode = mode;
    e.out = sout;
    e.draw = scratch;
  }

  // The repeat case. The key holds everything the old path's repeat needs unchanged (RepeatIsSameDraw + the dirty state it checks) besides the
  // vertex float constants: the typed render state, the shaders, the draw words, the fetch constants, the pixel float / bool / loop banks, the user
  // clip planes, the vertex index words, the bin state, the shared memory repeat epoch (index / vertex buffer contents, texture watches), the
  // submission and the frame (and the jitter, the occlusion flag: the Draw carries them verbatim).
  // The previous draw is the previous draw ITEM (a skipped item, an event or a copy breaks it: g_rep.seq).
  const uint64_t seq = OwnedRecordDrawSeq();
  const bool chain = g_rep.valid && (g_rep.seq == seq || g_rep.seq + 1 == seq);
  struct {
    uint64_t state, ext, vs, ps, fetch, ps_float, bool_loop, user_clip, frame, submission, repeat_epoch, bin_mask, bin_select;
    uint32_t prim, mode, init, dma_base, dma_size, occlusion, idx[4], viz;
    float jitter[2];
  } rk;
  std::memset(&rk, 0, sizeof(rk));
  rk.state = state_hash;
  rk.ext = ext_hash;
  rk.vs = vs;
  rk.ps = ps;
  rk.fetch = XXH3_64bits(s.fetch, 190 * sizeof(uint32_t));  // (words 190-191 are the inline path's staging address)
  rk.ps_float = XXH3_64bits(s.ps_float, sizeof(s.ps_float));
  rk.bool_loop = XXH3_64bits(s.bool_loop, sizeof(s.bool_loop));
  rk.user_clip = XXH3_64bits(s.user_clip, sizeof(s.user_clip));
  rk.frame = frame;
  rk.submission = OwnedFrontAccess::Submission(cp);
  rk.repeat_epoch = OwnedFrontAccess::RepeatEpoch(cp);
  rk.bin_mask = s.bin_mask;
  rk.bin_select = s.bin_select;
  rk.prim = prim;
  rk.mode = mode;
  rk.init = s.draw_initiator.value;
  rk.dma_base = s.dma_base;
  rk.dma_size = s.dma_size.value;
  rk.occlusion = OwnedFrontAccess::OcclusionValid(cp) ? 1u : 0u;
  rk.idx[0] = s.min_vtx_indx.value;
  rk.idx[1] = s.max_vtx_indx.value;
  rk.idx[2] = s.indx_offset.value;
  rk.viz = s.draw_viz_query;
  OwnedFrontAccess::Jitter(cp, rk.jitter);
  const uint64_t key = XXH3_64bits(&rk, sizeof(rk));
  if (chain && s.banks_valid && g_rep.key == key) {
    ++g_t.rep_found[kind];
    Compare c;
    c.kind = kind;
    c.vs_hash = old.vertex_shader_hash;
    c.ps_hash = old.pixel_shader_hash;
    c.by_field_p = g_t.rep_by_field;
    c.logged_p = g_t_logged_rep;
    c.nlogged_p = &g_t_logged;
    c.tag = "M4d repeat (previous draw verbatim vs from scratch)";
    CompareDraws(c, scratch, g_rep.draw);
    if (c.any) ++g_t.rep_mismatch_draws;
  } else if (kind == kOwnedFrontRepeat) {
    ++g_t.rep_claim_unmatched;
  }
  g_rep.valid = true;
  g_rep.seq = seq;
  g_rep.key = key;
  g_rep.draw = scratch;
  g_t.ns += NowNs() - t0;
}

void PrintM4dWindow(VulkanCommandProcessor& cp) {
  const TmplWindow& w = g_t;
  const auto top = [](const uint64_t* by_field) {
    uint32_t order[kFieldCount];
    for (uint32_t i = 0; i < kFieldCount; ++i) order[i] = i;
    std::sort(order, order + kFieldCount, [&](uint32_t a, uint32_t b) { return by_field[a] > by_field[b]; });
    std::string t;
    for (uint32_t k = 0; k < 8 && by_field[order[k]]; ++k) {
      t += (t.empty() ? "" : ", ") + std::string(kFieldNames[order[k]]) + " " + std::to_string(by_field[order[k]]);
    }
    return t.empty() ? std::string("none") : t;
  };
  const uint64_t frames = std::max<uint64_t>(OwnedFrontAccess::Frame(cp) - g_frame_first, 1);
  REXGPU_INFO(
      "[d3d][front] M4d frames {}-{}: template lookups {} (full {}, lean {}, repeat {}), hits {} ({}/{}/{}; {} from an earlier frame), misses {} ({}/{}/{}), "
      "invalidations {} (state {}, externals {}, both {}), evictions {}; hit draws built from memo + delta and compared {} (not built {}), draws with a "
      "mismatch {}; mismatches by field (top 8): {}; claims built from a crowd member's typed state {} (their M4b mismatches apart from M4b's: {} draws, by "
      "field: {}); shadow cost {:.1f} us a frame",
      g_frame_first, OwnedFrontAccess::Frame(cp), w.lookups[0] + w.lookups[1] + w.lookups[2], w.lookups[0], w.lookups[1], w.lookups[2],
      w.hits[0] + w.hits[1] + w.hits[2], w.hits[0], w.hits[1], w.hits[2], w.hits_cross_frame, w.misses[0] + w.misses[1] + w.misses[2], w.misses[0],
      w.misses[1], w.misses[2], w.invalidations, w.inval_state, w.inval_externals, w.inval_both, w.evictions, w.compared, w.not_built, w.mismatch_draws,
      top(w.by_field), w.member_draws, w.member_mismatch_draws, top(w.member_by_field), double(w.ns) / 1000.0 / double(frames));
  REXGPU_INFO(
      "[d3d][front] M4d repeat frames {}-{}: the typed repeat key (state, shaders, draw words, fetch, pixel floats, bool / loop, clip planes, index words, bin "
      "state, shared memory epoch, submission, frame) matched the previous draw item in {} claims (as the old path's: {}; typed repeat where the old path did "
      "NOT repeat - full {}, lean {} - to be explained), old repeat claims the typed key did not match {}, repeat compares with a mismatch {}, by field: {}. "
      "Repeat: state fields compared; the constants / dynamic offsets / sets are the current draw's (CompareDraws leaves sets, dynamic offsets and sets_bound "
      "out; the old repeat uploads the vertex floats again) - M4f takes constants + offsets from the current draw and only the state from the repeat",
      g_frame_first, OwnedFrontAccess::Frame(cp), w.rep_found[0] + w.rep_found[1] + w.rep_found[2], w.rep_found[2], w.rep_found[0], w.rep_found[1],
      w.rep_claim_unmatched, w.rep_mismatch_draws, top(w.rep_by_field));
  REXGPU_INFO(
      "[d3d][front] M4d crowd frames {}-{}: crowd draw items {} (no snapshot {}, no banks {}), c76-c78 compared {} (mismatches {}), the run match asked again "
      "with the typed constants (non-draining) {} (differs from the old answer {}), packet ids mapped {} (conflicts {}, typed id missing {})",
      g_frame_first, OwnedFrontAccess::Frame(cp), w.crowd_items, w.crowd_no_snapshot, w.crowd_no_banks, w.c76_compared, w.c76_mismatch, w.pretake_compared,
      w.pretake_mismatch, w.ids_mapped, w.id_conflicts, w.id_zero);
  g_t = TmplWindow{};
}

// M4e (bit 4): the window line of clears / resolves / inline draws. The inline draws' M4b-style description compare is g_winl (claims, built, mismatches by field,
// why not built); their M4c-style constants / bindings compare is the bind file's inline window.
void PrintM4eWindow(VulkanCommandProcessor& cp) {
  const Window& w = g_winl;
  uint64_t claims = 0, built = 0;
  for (uint32_t k = 0; k < kOwnedFrontKinds; ++k) claims += w.claims[k], built += w.built[k];
  uint32_t order[kFieldCount];
  for (uint32_t i = 0; i < kFieldCount; ++i) order[i] = i;
  std::sort(order, order + kFieldCount, [&](uint32_t a, uint32_t b) { return w.by_field[a] > w.by_field[b]; });
  std::string top;
  for (uint32_t k = 0; k < 8 && w.by_field[order[k]]; ++k) {
    top += (top.empty() ? "" : ", ") + std::string(kFieldNames[order[k]]) + " " + std::to_string(w.by_field[order[k]]);
  }
  if (top.empty()) top = "none";
  std::string skips;
  for (uint32_t k = 1; k < kSkipCount; ++k) {
    if (w.skip[k]) skips += (skips.empty() ? "" : ", ") + std::string(kSkipNames[k]) + " " + std::to_string(w.skip[k]);
  }
  if (skips.empty()) skips = "none";
  const uint64_t frames = std::max<uint64_t>(OwnedFrontAccess::Frame(cp) - g_frame_first, 1);
  REXGPU_INFO(
      "[d3d][front] M4e frames {}-{}: {}; inline draws: claims {} (full {}, lean {}, repeat {}), description built {} and compared, draws with a mismatch {}; mismatches by field (top 8): {}; not "
      "built: {}; constants + bindings (M4c builders) for inline draws: {}; description build+compare {:.1f} us a frame",
      g_frame_first, OwnedFrontAccess::Frame(cp), OwnedRecordM4eSummary(frames), claims, w.claims[0], w.claims[1], w.claims[2], built, w.mismatch_draws, top, skips,
      OwnedFrontBindInlineSummary(), double(w.build_ns) / 1000.0 / double(frames));
  g_winl = Window{};
}

void PrintWindow(VulkanCommandProcessor& cp) {
  if (REXCVAR_GET(gpu_d3d_owned_front) & 16u) PrintM4eWindow(cp);
  if (REXCVAR_GET(gpu_d3d_owned_front) & 4u) OwnedFrontBindPrintWindow(g_frame_first, OwnedFrontAccess::Frame(cp));
  if (REXCVAR_GET(gpu_d3d_owned_front) & 8u) PrintM4dWindow(cp);
  const Window& w = g_w;
  uint64_t claims = 0, built = 0;
  for (uint32_t k = 0; k < kOwnedFrontKinds; ++k) claims += w.claims[k], built += w.built[k];
  uint32_t order[kFieldCount];
  for (uint32_t i = 0; i < kFieldCount; ++i) order[i] = i;
  std::sort(order, order + kFieldCount, [&](uint32_t a, uint32_t b) { return w.by_field[a] > w.by_field[b]; });
  std::string top;
  for (uint32_t k = 0; k < 8 && w.by_field[order[k]]; ++k) {
    top += (top.empty() ? "" : ", ") + std::string(kFieldNames[order[k]]) + " " + std::to_string(w.by_field[order[k]]);
  }
  if (top.empty()) top = "none";
  std::string skips;
  for (uint32_t k = 1; k < kSkipCount; ++k) {
    if (w.skip[k]) skips += (skips.empty() ? "" : ", ") + std::string(kSkipNames[k]) + " " + std::to_string(w.skip[k]);
  }
  if (skips.empty()) skips = "none";
  const uint64_t frames = std::max<uint64_t>(OwnedFrontAccess::Frame(cp) - g_frame_first, 1);
  const OwnedRecordSnapStats& sn = OwnedRecordSnapCounters();
  REXGPU_INFO(
      "[d3d][front] M4b frames {}-{}: claims {} (full {}, lean {}, repeat {}), draws built {} (full {}, lean {}, repeat {}), compared {}, draws with a "
      "mismatch {}; mismatches by field (top 8): {}; not built: {}; geometry shader handle not looked up {}, memexport index buffer not compared {}; draw snapshots taken {} (past the cap "
      "{}), draw items given to the handler {}, paired {}, no snapshot {}, past the cap {}, desync {}; not built by the typed side at all (M4c / M4d / claim-time): layout, sets, dynamic offsets, texture key, fast clear fill, "
      "truth_*, cover, bounds, xfb; build+compare {:.1f} us a frame",
      g_frame_first, OwnedFrontAccess::Frame(cp), claims, w.claims[0], w.claims[1], w.claims[2], built, w.built[0], w.built[1], w.built[2], built,
      w.mismatch_draws, top, skips, w.not_compared_gs, w.not_compared_memexport_index, sn.taken, sn.overflow_items, sn.advanced, sn.paired,
      sn.no_snapshot, sn.overflow_draws, sn.desync, double(w.build_ns) / 1000.0 / double(frames));
  g_w = Window{};
  g_frame_first = OwnedFrontAccess::Frame(cp);
  g_last_print_frame = g_frame_first;
}

}  // namespace

namespace {
void ShadowDrawBody(VulkanCommandProcessor& cp, const native::Draw& old_draw, uint32_t kind) {
  ++g_w.claims[kind];
  if (!g_frame_first) g_frame_first = g_last_print_frame = OwnedFrontAccess::Frame(cp);
  const OwnedRecordState* s = OwnedRecordDrawNow();
  // M4d: the draw a crowd member's constants make (the crowd code issues it after its run's first member's item; no record item of its own).
  if (!s && kind != kOwnedFrontFull && (REXCVAR_GET(gpu_d3d_owned_front) & 8u) && D3dFeedCrowd().member_c76) {
    s = OwnedRecordCrowdMemberState(D3dFeedCrowd().member_c76);
    if (s) ++g_t.member_draws;
  }
  if (!s) {
    g_rep.valid = false;  // (M4d: a draw with no snapshot breaks the repeat chain)
    ++g_w.skip[old_draw.fast_clear ? kSkipNoRecordFast : kind == kOwnedFrontRepeat ? kSkipNoRecordRepeat : kSkipNoRecordOther];
  } else {
    const uint64_t t0 = NowNs();
    Draw d;
    bool vs_absent = false, ps_absent = false, gs_skipped = false;
    Skip skip;
    const bool bind_on = (REXCVAR_GET(gpu_d3d_owned_front) & 4u) != 0;  // M4c (with bits 0 + 1)
    static OwnedBindInputs bi;
    if (bind_on) bi = OwnedBindInputs{};
    if (s->shader[0].hash != old_draw.vertex_shader_hash) {
      // the typed vertex shader is not the one the old path draws with
      Compare c;
      c.kind = kind;
      c.vs_hash = old_draw.vertex_shader_hash;
      c.ps_hash = old_draw.pixel_shader_hash;
      if (s->inline_draw) {
        c.logged_p = g_inl_logged;
        c.nlogged_p = &g_inl_logged_n;
        c.tag = "M4e (inline)";
      }
      c.Eq(F_vertex_shader_hash, old_draw.vertex_shader_hash, s->shader[0].hash);
      ++g_w.mismatch_draws;
      skip = kSkipShaderHash;
    } else {
      skip = OwnedFrontAccess::Build(cp, *s, old_draw, d, vs_absent, ps_absent, gs_skipped, bind_on ? &bi : nullptr, nullptr, &g_scratch_out);
    }
    if (skip != kSkipNone) {
      g_rep.valid = false;  // (M4d)
      ++g_w.skip[skip];
    } else {
      ++g_w.built[kind];
      Compare c;
      c.kind = kind;
      c.vs_hash = old_draw.vertex_shader_hash;
      c.ps_hash = old_draw.pixel_shader_hash;
      if (s->inline_draw) {  // M4e: an inline draw is compared apart from M4b's (g_w is g_winl's storage for this call)
        c.logged_p = g_inl_logged;
        c.nlogged_p = &g_inl_logged_n;
        c.tag = "M4e (inline)";
      }
      if (s->crowd_member) {  // M4d: a crowd member's draw is counted apart from M4b's
        c.by_field_p = g_t.member_by_field;
        c.logged_p = g_t_logged_member;
        c.nlogged_p = &g_t_logged_member_n;
        c.tag = "M4b (crowd member)";
      }
      const Draw& o = old_draw;
      c.Eq(F_vertex_shader_hash, o.vertex_shader_hash, d.vertex_shader_hash);
      c.Eq(F_pixel_shader_hash, o.pixel_shader_hash, d.pixel_shader_hash);
      c.Eq(F_pixel_spec_key, o.pixel_spec_key, d.pixel_spec_key);
      c.Eq(F_pixel_ext, o.pixel_ext, d.pixel_ext);
      c.Eq(F_vertex_mul_key, o.vertex_mul_key, d.vertex_mul_key);
      c.Eq(F_vertex_ext, o.vertex_ext, d.vertex_ext);
      c.Eq(F_vertex_modification, o.vertex_modification, d.vertex_modification);
      c.Eq(F_pixel_modification, o.pixel_modification, d.pixel_modification);
      // (a module the old path has and the typed key does not find is its own field; never a translation made here)
      if (vs_absent && o.vertex_shader != VK_NULL_HANDLE) {
        c.Note(F_vertex_module_absent, uint64_t(reinterpret_cast<uintptr_t>(o.vertex_shader)), 0);
      } else {
        c.EqH(F_vertex_shader_module, o.vertex_shader, d.vertex_shader);
      }
      if (ps_absent && o.pixel_shader != VK_NULL_HANDLE) {
        c.Note(F_pixel_module_absent, uint64_t(reinterpret_cast<uintptr_t>(o.pixel_shader)), 0);
      } else {
        c.EqH(F_pixel_shader_module, o.pixel_shader, d.pixel_shader);
      }
      if (gs_skipped) ++g_w.not_compared_gs;
      c.Eq(F_float24_depth, o.float24_depth, d.float24_depth);
      c.Eq(F_color_quantize, o.color_quantize, d.color_quantize);
      c.Eq(F_ps_float24, o.ps_float24, d.ps_float24);
      c.Eq(F_ps_twin, o.ps_twin, d.ps_twin);
      c.Eq(F_ps_quantize, o.ps_quantize, d.ps_quantize);
      for (uint32_t i = 0; i < 4; ++i) c.Eq(F_layout_counts, o.layout_counts[i], d.layout_counts[i]);
      c.EqH(F_layout, reinterpret_cast<const void*>(o.layout), reinterpret_cast<const void*>(d.layout));
      c.Eq(F_restore, o.restore, d.restore);
      if (o.restore || d.restore) {
        c.Eq(F_restore_pages, uint64_t(o.restore_color_page) << 32 | o.restore_depth_page,
             uint64_t(d.restore_color_page) << 32 | d.restore_depth_page);
      }
      c.EqF(F_viewport_x, o.viewport.x, d.viewport.x);
      c.EqF(F_viewport_y, o.viewport.y, d.viewport.y);
      c.EqF(F_viewport_w, o.viewport.width, d.viewport.width);
      c.EqF(F_viewport_h, o.viewport.height, d.viewport.height);
      // (== : SetViewport keeps -0.0 and +0.0 as equal values)
      if (!(o.viewport.minDepth == d.viewport.minDepth)) c.Note(F_viewport_min_depth, Bits(o.viewport.minDepth), Bits(d.viewport.minDepth));
      if (!(o.viewport.maxDepth == d.viewport.maxDepth)) c.Note(F_viewport_max_depth, Bits(o.viewport.maxDepth), Bits(d.viewport.maxDepth));
      if (o.scissor.offset.x != d.scissor.offset.x || o.scissor.offset.y != d.scissor.offset.y || o.scissor.extent.width != d.scissor.extent.width ||
          o.scissor.extent.height != d.scissor.extent.height) {
        c.Note(F_scissor, (uint64_t(uint16_t(o.scissor.offset.x)) << 48) | (uint64_t(uint16_t(o.scissor.offset.y)) << 32) |
                              (uint64_t(uint16_t(o.scissor.extent.width)) << 16) | uint16_t(o.scissor.extent.height),
               (uint64_t(uint16_t(d.scissor.offset.x)) << 48) | (uint64_t(uint16_t(d.scissor.offset.y)) << 32) |
                   (uint64_t(uint16_t(d.scissor.extent.width)) << 16) | uint16_t(d.scissor.extent.height));
      }
      c.Eq(F_guest_primitive, o.guest_primitive, d.guest_primitive);
      c.Eq(F_host_primitive, o.host_primitive, d.host_primitive);
      c.Eq(F_host_vs_type, o.host_vs_type, d.host_vs_type);
      c.Eq(F_polygonal, o.polygonal, d.polygonal);
      c.Eq(F_raster_supported, o.raster.supported, d.raster.supported);
      c.Eq(F_raster_topology, uint32_t(o.raster.topology), uint32_t(d.raster.topology));
      c.Eq(F_raster_restart, o.raster.primitive_restart, d.raster.primitive_restart);
      c.Eq(F_raster_expanded, o.raster.expanded, d.raster.expanded);
      c.EqH(F_raster_geometry_shader, o.raster.geometry_shader, d.raster.geometry_shader);
      c.Eq(F_raster_cull_mode, uint32_t(o.raster.cull_mode), uint32_t(d.raster.cull_mode));
      c.Eq(F_raster_front_face, o.raster.front_face_clockwise, d.raster.front_face_clockwise);
      c.Eq(F_raster_depth_test, o.raster.depth_test, d.raster.depth_test);
      c.Eq(F_raster_depth_write, o.raster.depth_write, d.raster.depth_write);
      c.Eq(F_raster_depth_compare, uint32_t(o.raster.depth_compare), uint32_t(d.raster.depth_compare));
      c.EqF(F_raster_bias_constant, o.raster.depth_bias_constant, d.raster.depth_bias_constant);
      c.EqF(F_raster_bias_slope, o.raster.depth_bias_slope, d.raster.depth_bias_slope);
      c.Eq(F_raster_stencil, o.raster.stencil, d.raster.stencil);
      c.Eq(F_raster_stencil_ops, o.raster.stencil_ops, d.raster.stencil_ops);
      c.Eq(F_raster_clip_disable, o.raster.clip_disable, d.raster.clip_disable);
      c.Eq(F_stencil_ref, o.stencil_ref, d.stencil_ref);
      for (uint32_t i = 0; i < 2; ++i) {
        c.Eq(F_stencil_refs, o.stencil_refs[i], d.stencil_refs[i]);
        c.Eq(F_stencil_read_masks, o.stencil_read_masks[i], d.stencil_read_masks[i]);
        c.Eq(F_stencil_write_masks, o.stencil_write_masks[i], d.stencil_write_masks[i]);
      }
      c.Eq(F_color_mask_all, o.color_mask_all, d.color_mask_all);
      for (uint32_t i = 0; i < 4; ++i) c.EqF(F_blend_constants, o.blend_constants[i], d.blend_constants[i]);
      c.Eq(F_occlusion_query, o.occlusion_query, d.occlusion_query);
      c.Eq(F_surface_info, o.surface_info, d.surface_info);
      c.Eq(F_color_info, o.color_info, d.color_info);
      c.Eq(F_depth_info, o.depth_info, d.depth_info);
      c.Eq(F_vertex_count, o.vertex_count, d.vertex_count);
      c.Eq(F_screen_space, o.screen_space, d.screen_space);
      c.Eq(F_memexport, o.memexport, d.memexport);
      c.Eq(F_color_write_mask, o.color_write_mask, d.color_write_mask);
      c.Eq(F_blend_control, o.blend_control, d.blend_control);
      c.Eq(F_indexed, o.indexed, d.indexed);
      // (a memexport draw's guest DMA indices are read from a scratch copy by the old path: its buffer and offset are not the shared memory's)
      if (d.memexport && d.indexed && o.index_buffer != d.index_buffer) {
        ++g_w.not_compared_memexport_index;
      } else {
        c.EqH(F_index_buffer, o.index_buffer, d.index_buffer);
        c.Eq(F_index_offset, uint64_t(o.index_offset), uint64_t(d.index_offset));
      }
      c.Eq(F_index_type, uint32_t(o.index_type), uint32_t(d.index_type));
      c.Eq(F_merge_stride, o.merge_stride, d.merge_stride);
      c.Eq(F_float_count, o.float_count, d.float_count);
      if (c.any) ++(s->crowd_member ? g_t.member_mismatch_draws : g_w.mismatch_draws);
      if (s->inline_draw) {
        g_rep.valid = false;  // (M4e: an inline draw ends the typed repeat chain, as its IM_LOAD ends the old path's)
      } else if (REXCVAR_GET(gpu_d3d_owned_front) & 8u) {
        TemplateShadow(cp, *s, old_draw, d, g_scratch_out, kind);
      }  // M4d (the memo and the repeat key)
      if (bind_on) {
        bi.s = s;
        bi.frame = OwnedFrontAccess::Frame(cp);
        bi.kind = kind;
        bi.vs_hash = old_draw.vertex_shader_hash;
        bi.ps_hash = old_draw.pixel_shader_hash;
        OwnedFrontBindShadow(bi);
      }
    }
    g_w.build_ns += NowNs() - t0;
  }
}
}  // namespace

// M4e: an inline draw's claim is counted in its own window (g_winl, the M4b and M4c counters swapped in for the call), so M4b / M4c stay what they were.
void OwnedFrontShadowDraw(VulkanCommandProcessor& cp, const native::Draw& old_draw, uint32_t kind) {
  if (g_typed_draw_active) {  // (M4f-A review P5: a typed draw is compared with nothing - the shadows stay old-vs-typed evidence for the fallback draws)
    ++g_typed_shadow_skipped;
    return;
  }
  const OwnedRecordState* peek = OwnedRecordDrawNow();
  const bool inl = peek && peek->inline_draw;
  if (inl) {
    std::swap(g_w, g_winl);
    OwnedFrontBindBucket(true);
  }
  ShadowDrawBody(cp, old_draw, kind);
  if (inl) OwnedRecordInlineEnd();  // (M4e: the crowd code the stock draw runs after this claim never reads the inline snapshot)
  if (inl) {
    std::swap(g_w, g_winl);
    OwnedFrontBindBucket(false);
  }
  if (OwnedFrontAccess::Frame(cp) - g_last_print_frame >= 600) PrintWindow(cp);
}

// M4d: anything but a draw item (a copy / clear / inline record, a segment, an event, the present, register items) ends "the previous draw".
void OwnedFrontRepeatBreak() {
  g_rep.valid = false;
  g_trep.valid = false;  // (Session 93: any item that is not a typed draw ends the typed repeat chain)
  g_trep_prev.valid = false;
}

// M4d: the crowd shadow, called by the draw handler for every draw item of a frame the crowd code has runs queued for (`pre_called`), with the
// old path's inputs: the member constants c76-c78 it read from its register image, its packet id (the item's arena address) and PreTake's answer.
// Compares the typed state of the item (c76-c78 from its snapshot's vs_float bank, the crowd identity token seq << 8 | item index) with them.
void OwnedFrontCrowdItem(VulkanCommandProcessor& cp, const uint32_t* old_c76, uint32_t old_id, bool pre_called, bool pre_result, OwnedFrontCrowdAsk ask) {
  if (!pre_called) return;
  const uint64_t t0 = NowNs();
  TmplWindow& w = g_t;
  ++w.crowd_items;
  const OwnedRecordState* s = OwnedRecordDrawNow();
  if (!s) {
    ++w.crowd_no_snapshot;
  } else if (!s->banks_valid) {
    ++w.crowd_no_banks;
  } else {
    const uint32_t* typed = s->vs_float + 4 * 76;
    ++w.c76_compared;
    if (std::memcmp(typed, old_c76, 12 * sizeof(uint32_t)) != 0) {
      ++w.c76_mismatch;
      if (g_t_crowd_logged < 8) {
        ++g_t_crowd_logged;
        uint32_t k = 0;
        while (k < 12 && typed[k] == old_c76[k]) ++k;
        REXGPU_INFO("[d3d][front] M4d crowd mismatch #{}: c76-c78 word {} old {:08X} typed {:08X} (packet id {:08X}, typed id {:X})", g_t_crowd_logged, k,
                    old_c76[k], typed[k], old_id, s->crowd_id);
      }
    }
    // the packet id: one typed id per old arena address, within a frame (a tiling replay of the same packet is the same typed id)
    const uint64_t frame = OwnedFrontAccess::Frame(cp);
    if (frame != g_id_frame) {
      g_id_frame = frame;
      g_id_fwd.clear();
      g_id_rev.clear();
    }
    if (!s->crowd_id) {
      ++w.id_zero;
    } else {
      const auto f = g_id_fwd.find(old_id);
      const auto r = g_id_rev.find(s->crowd_id);
      if (f == g_id_fwd.end() && r == g_id_rev.end()) {
        g_id_fwd.emplace(old_id, s->crowd_id);
        g_id_rev.emplace(s->crowd_id, old_id);
        ++w.ids_mapped;
      } else if (f == g_id_fwd.end() || r == g_id_rev.end() || f->second != s->crowd_id || r->second != old_id) {
        ++w.id_conflicts;
        if (g_t_crowd_logged < 8) {
          ++g_t_crowd_logged;
          REXGPU_INFO("[d3d][front] M4d crowd id conflict #{}: packet id {:08X} typed id {:X} (earlier: packet id -> {:X}, typed id <- {:08X})", g_t_crowd_logged,
                      old_id, s->crowd_id, f != g_id_fwd.end() ? f->second : 0, r != g_id_rev.end() ? r->second : 0);
        }
      }
    }
    // the crowd run match from the typed constants (SceneCrowdMatchLocal: the same scan as PreTake over the local list, taking nothing from the queue)
    ++w.pretake_compared;
    if (ask(cp, typed) != pre_result) {
      ++w.pretake_mismatch;
      if (g_t_crowd_logged < 8) {
        ++g_t_crowd_logged;
        REXGPU_INFO("[d3d][front] M4d crowd mismatch #{}: PreTake with the typed c76-c78 answers {}, the old {} (packet id {:08X})", g_t_crowd_logged,
                    !pre_result, pre_result, old_id);
      }
    }
  }
  w.ns += NowNs() - t0;
}


// ---------------------------------------------------------------------------------------------------------------------------------
// M4f-A (latch bit 5, value 32): the typed state DRAWS the token draw items.
//
// OwnedFrontPrepare is Build in its own mode (modules / geometry shader / layout really made) over the paired snapshot, with the template memo
// of M4d's shape: a key of (shader pair, primitive, mode, state hash, externals hash) finds an entry whose viewport / scissor / depth / stencil /
// cull parts are taken (Build with `tmpl`); the rest - the fetch-dependent keys, the modifications, the modules, the primitive - is derived for
// every draw, which is what the old lean's per-draw validity checks (endian key, spec key, ext stale) amount to. d3d_direct.cpp's DrawFromBase
// then draws it: real primitive pass, samplers, textures, system constants, constants upload / descriptor sets, vertex ranges, claim.
namespace {
struct OwnWindow {
  uint64_t counts[kTypedOutcomeCount] = {};
  uint64_t memo_hits = 0, memo_misses = 0, hits_cross_frame = 0;
  uint64_t build_skip[kSkipCount] = {};
  uint64_t prepare_ns = 0;
  std::map<const char*, uint64_t> fallbacks;  // (keyed by the literal's address: no string work per fallback)
} g_own;
Draw g_own_draw;
Draw g_own_old_stub;  // (Build's `old` argument: only read by the shadow mode)
OwnedTypedPrep g_own_prep;
struct OwnMemo {
  bool valid = false;
  uint64_t vs = 0, ps = 0, state_hash = 0, ext_hash = 0, frame = 0;
  uint32_t prim = 0, mode = 0;
  TemplateOut out;
  Draw draw;
};
OwnMemo g_own_memo[kMemoSlots];
uint64_t g_own_first_frame = 0, g_own_last_print = 0;

void PrintOwnWindow(uint64_t frame) {
  const OwnWindow& w = g_own;
  std::string fb, skips;
  uint64_t fb_total = 0;
  for (const auto& kv : w.fallbacks) {
    fb += (fb.empty() ? "" : ", ") + std::string(kv.first) + " " + std::to_string(kv.second);
    fb_total += kv.second;
  }
  if (fb.empty()) fb = "none";
  for (uint32_t k = 1; k < kSkipCount; ++k) {
    if (w.build_skip[k]) skips += (skips.empty() ? "" : ", ") + std::string(kSkipNames[k]) + " " + std::to_string(w.build_skip[k]);
  }
  if (skips.empty()) skips = "none";
  const uint64_t frames = std::max<uint64_t>(frame - g_own_first_frame, 1);
  REXGPU_INFO(
      "[d3d][front] M4f-A frames {}-{}: token draw items offered {}, drawn by the typed front end {} (template memo: {} hits - {} from an earlier frame, {} misses), failed in the "
      "backend {}, handed to the old path {} by reason: {}; of those, refused while building (Build skip reasons): {}; typed draws whose shadow compare was not run (typed vs typed would prove nothing) {}; of the drawn, restore draws {}, claimed but pipeline still compiling (dropped once) {}; M4f-B typed: crowd members {} (declined to the old way {}), inline draws {} (handed to the old path {}), cover rectangles {}, bounds rectangles {}, memexport draws {}; typed prepare {:.1f} us a frame, whole typed draw {:.1f} us a frame",
      g_own_first_frame, frame, w.counts[kTypedItems], w.counts[kTypedDrawn], w.memo_hits, w.hits_cross_frame, w.memo_misses, w.counts[kTypedFailed],
      fb_total, fb, skips, g_typed_shadow_skipped, w.counts[kTypedRestores], w.counts[kTypedPlaceholder], w.counts[kTypedCrowdMembers], w.counts[kTypedCrowdDeclined], w.counts[kTypedInline],
      w.counts[kTypedInlineFallback], w.counts[kTypedCover], w.counts[kTypedBounds], w.counts[kTypedMemexport], double(w.prepare_ns) / 1000.0 / double(frames), double(w.counts[kTypedNs]) / 1000.0 / double(frames));
  if (REXCVAR_GET(gpu_d3d_typed_lean_bind)) {  // Session 94 (B7): the sampled split. Phases are timed on 1 frame in 8 (x8 here); "draw" is the whole typed draw (every frame).
    static bool on_logged = false;
    if (!on_logged) {
      on_logged = true;
      REXGPU_INFO("[d3d][typed] lean_bind on (sampler memo per stage, system constants skip, settings latched once a frame, bounds list parsed once a frame)");
    }
    const double per_frame = 8.0 / 1000.0 / double(frames);
    const uint64_t* st = g_lean_bind_stat;
    REXGPU_INFO("[d3d][typed] draw us/frame {:.0f} (prepare {:.0f}, bind {:.0f} = samplers {:.0f} + textures {:.0f} + plans {:.0f} + update bindings {:.0f}, constants {:.0f}, claim {:.0f}); sampler memo hits {} rebuilds {}; system constants skipped {} built {}; "
                "sampled draws {}; verify mismatches {}",
                double(w.counts[kTypedNs]) / 1000.0 / double(frames), double(w.prepare_ns) / 1000.0 / double(frames), double(st[0] + st[3] + st[10] + st[11] + st[12]) * per_frame, double(st[3]) * per_frame,
                double(st[10]) * per_frame, double(st[11]) * per_frame, double(st[12]) * per_frame, double(st[1]) * per_frame, double(st[2]) * per_frame, st[4], st[5], st[6], st[7], st[9], st[8]);
    {  // (Session 95, r6tex: the split of "textures", ticks per sampled call; log only)
      const uint64_t* t = g_tex_stat;
      const double calls = double(std::max<uint64_t>(t[kTsCalls], 1));
      REXGPU_INFO("[d3d][typed] textures split: calls {} whole {:.0f} ticks a call = before {:.0f} + base {:.0f} (of it slots {:.0f}, {} loads {:.0f} = ranges {:.0f} (valid {:.0f}, copy {:.0f}, {} pages) + load {:.0f} + watch {:.0f}, UpdateTextureBindingsImpl {:.0f} in {} calls) + loops {:.0f} (of it views {:.0f}); base returned early {}, slots kept {}, derived {}, found {}; loop slots {} (view calls {}, served {}, avg pages scanned {:.1f}), loops skipped {}, outdated resets {}; memo: slot hits {} stores {} texture hits {}, free slots {}, view hits {}, slots read again {}, verify mismatches {}",
                  t[kTsCalls], double(t[kTsWholeTicks]) / calls, double(t[kTsPreTicks]) / calls, double(t[kTsBaseTicks]) / calls, double(t[kTsSlotLoopTicks]) / calls, t[kTsLoads], double(t[kTsLoadTicks]) / calls, double(t[kTsRangeTicks]) / calls, double(t[kTsUpValidTicks]) / calls, double(t[kTsUpCopyTicks]) / calls, t[kTsUpPages], double(t[kTsLoadImplTicks]) / calls, double(t[kTsWatchTicks]) / calls, double(t[kTsImplTicks]) / calls, t[kTsImplCalls], double(t[kTsLoopTicks]) / calls, double(t[kTsViewTicks]) / calls,
                  t[kTsBaseEarly], t[kTsSlotsKept], t[kTsSlotsDerived], t[kTsSlotsFound], t[kTsLoopSlots], t[kTsViewCalls], t[kTsViewServed], double(t[kTsPagesScanned]) / double(std::max<uint64_t>(t[kTsViewCalls], 1)), t[kTsLoopSkips], t[kTsOutdatedResets],
                  t[kTsMemoHits], t[kTsMemoStores], t[kTsTextureHits], t[kTsFreeSlots], t[kTsViewMemoHits], t[kTsSlotAgain], g_tex_bind_mismatches);
      REXGPU_INFO("[d3d][typed] plans split, ticks a sampled draw: ticks {:.0f}, skin {:.0f} ({} skin draws), record plans {:.0f} ({} draws), built plans {:.0f} ({} draws)", double(t[kTsPlanTickTicks]) / calls, double(t[kTsPlanSkinTicks]) / calls,
                  t[kTsPlanSkinDraws], double(t[kTsPlanRecTicks]) / calls, t[kTsPlanRecDraws], double(t[kTsPlanBuildTicks]) / calls, t[kTsPlanBuildDraws]);
      std::memset(g_tex_stat, 0, sizeof(g_tex_stat));
      TexLoadCensusLog();
    }
    std::memset(g_lean_bind_stat, 0, sizeof(g_lean_bind_stat));
  }
  if (REXCVAR_GET(gpu_d3d_typed_memo_reuse) || REXCVAR_GET(gpu_d3d_typed_repeat)) {
    const ReuseWindow& r = g_rw;
    const uint64_t hits = r.hits_fast + r.hits_hashed, prepared = hits + r.misses + r.refused;
    std::string pm;
    for (const auto& kv : r.prep_mismatch) pm += (pm.empty() ? "" : ", ") + std::string(kv.first) + " " + std::to_string(kv.second);
    std::string fm;
    for (uint32_t k = 0; k < kFieldCount; ++k) {
      if (r.by_field[k]) fm += (fm.empty() ? "" : ", ") + std::string(kFieldNames[k]) + " " + std::to_string(r.by_field[k]);
    }
    REXGPU_INFO(
        "[d3d][memo] per frames {}-{}: typed draws prepared {} (reuse {}): hits {} ({:.1f}% of draws; epoch fast path {}, hashed {}, {} from an earlier frame), misses {} (refused while building {}), "
        "hits turned back into builds {} (shader state {}, primitive pass {}, fetch keys {}), restore draws not stored {}, evictions {}; verify {}: compared {}, draws with a mismatch {} "
        "(draw fields: {}; preparation fields: {}); typed repeats {}: checked {}, candidates {}, tried {}, repeated {} (RepeatDrawFast), refused {}; timers {}",
        g_own_first_frame, frame, prepared, REXCVAR_GET(gpu_d3d_typed_memo_reuse) ? "on" : "off", hits, prepared ? 100.0 * double(hits) / double(prepared) : 0.0, r.hits_fast, r.hits_hashed,
        r.hits_cross_frame, r.misses, r.refused, r.demoted_shader + r.demoted_primitive + r.demoted_fetch_keys, r.demoted_shader, r.demoted_primitive, r.demoted_fetch_keys,
        r.not_stored_restore, r.evictions, REXCVAR_GET(gpu_d3d_typed_memo_verify) ? "on" : "off", r.verify_compared, r.verify_mismatch_draws, fm.empty() ? "none" : fm,
        pm.empty() ? "none" : pm, REXCVAR_GET(gpu_d3d_typed_repeat) ? "on" : "off", r.rep_checked, r.rep_candidates, r.rep[kRepeatTried], r.rep[kRepeatDrawn], r.rep[kRepeatRefused],
        REXCVAR_GET(gpu_d3d_typed_timers) ? "on" : "off");
    g_rw = ReuseWindow{};
  }
  g_own = OwnWindow{};
  g_typed_shadow_skipped = 0;
  g_own_first_frame = frame;
  g_own_last_print = frame;
}
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------------------
// Session 93 (gpu_d3d_typed_memo_reuse): the typed draw's preparation taken from a memo instead of Build.
//
// An entry holds everything Build derived that is a function of the render state, the shader pair and the processor's settings (the "state part"): the Draw, the
// prepared fields of OwnedTypedPrep and the bind inputs. A hit derives again only what depends on the draw's own words or the fetch constants: the primitive pass
// (once - its result goes to DrawFromBase), the vertex fetch endian key and the pixel specialisation key (both compared with the entry's: another value is a miss),
// the viewport origin (DLAA jitter), the occlusion flag, the alpha learn's texture key and the index fields. The key is (shader pair, primitive, mode, StateHash,
// ExternalsReuse, the translation generation); under the cut the decoder's render epoch stands for StateHash when it has not moved since the previous lookup.
namespace {
constexpr uint32_t kReuseSlots = 512, kReuseWays = 2, kReuseSets = kReuseSlots / kReuseWays;
constexpr size_t kPrepHead = offsetof(OwnedTypedPrep, bi);       // the prepared fields (and the draw's own: input, bank pointers) before the bind inputs
constexpr size_t kBiHead = offsetof(OwnedBindInputs, textures);  // the bind inputs Build fills in the typed mode (the lists after it stay zero)
struct ReuseEntry {
  bool valid = false;
  uint64_t id = 0, used = 0, frame = 0;
  uint64_t vs = 0, ps = 0, state_hash = 0, ext = 0, gen = 0;
  uint32_t prim = 0, mode = 0;
  const void* active_vs = nullptr;
  const void* active_ps = nullptr;
  TemplateOut out;
  Draw draw;
  OwnedTypedPrep prep;
};
ReuseEntry g_reuse[kReuseSlots];
struct ReuseLast {
  bool valid = false;
  uint64_t id = 0, render = 0, vs = 0, ps = 0, ext = 0, gen = 0;
  uint32_t prim = 0, mode = 0;
  ReuseEntry* e = nullptr;
};
ReuseLast g_reuse_last;
bool g_sh_valid = false;
uint64_t g_sh_render = 0, g_sh_value = 0;
uint64_t g_reuse_clock = 0, g_reuse_ids = 0;
OwnedTypedPrep g_verify_prep;
Draw g_verify_draw, g_verify_stub;

inline void CopyPrepState(OwnedTypedPrep& to, const OwnedTypedPrep& from) {
  std::memcpy(static_cast<void*>(&to), static_cast<const void*>(&from), kPrepHead);
  std::memcpy(static_cast<void*>(&to.bi), static_cast<const void*>(&from.bi), kBiHead);
}

PrimitiveProcessor::DrawInput DrawInputOf(const OwnedRecordState& s) {
  PrimitiveProcessor::DrawInput input;
  input.draw_initiator = s.draw_initiator;
  input.dma_size = s.dma_size;
  input.dma_base = s.dma_base;
  input.output_path_cntl = s.output_path_cntl;
  input.hos_cntl = s.hos_cntl;
  input.su_sc_mode_cntl = s.sc_mode_cntl;
  input.multi_prim_ib_reset_indx = s.reset_indx;
  return input;
}

void NotePrepMismatch(const char* name, const OwnedRecordState&) {
  ++g_rw.prep_mismatch[name];
  static uint32_t logged = 0;
  if (logged < 24) {
    ++logged;
    REXGPU_INFO("[d3d][memo] verify mismatch #{}: preparation field {} differs between the hit and the from-scratch build", logged, name);
  }
}

// Verify: the from-scratch Build of the same draw (what a hit skips: every derivation but the primitive pass) against what the hit gave.
void VerifyHit(VulkanCommandProcessor& cp, const OwnedRecordState& s, const OwnedTypedPrep& p, const Draw& d) {
  using A = OwnedFrontAccess;
  OwnedTypedPrep& r = g_verify_prep;
  r.refusal = nullptr;
  Draw& rd = g_verify_draw;
  rd = Draw();
  g_verify_stub = Draw();
  bool a = false, b = false, g = false;
  TemplateOut scratch;
  const Skip sk = A::Build(cp, s, g_verify_stub, rd, a, b, g, &r.bi, nullptr, &scratch, &r);
  ++g_rw.verify_compared;
  if (sk != kSkipNone) {
    ++g_rw.verify_mismatch_draws;
    NotePrepMismatch(kSkipNames[sk], s);  // (the from-scratch build refused a draw the hit accepted)
    return;
  }
  Compare c;
  c.kind = kOwnedFrontFull;
  c.vs_hash = rd.vertex_shader_hash;
  c.ps_hash = rd.pixel_shader_hash;
  c.by_field_p = g_rw.by_field;
  c.logged_p = g_rw_logged;
  c.nlogged_p = &g_rw_nlogged;
  c.tag = "typed memo reuse (hit vs from scratch)";
  CompareDraws(c, rd, d);
  uint32_t bad = c.any ? 1 : 0;
#define VCMP(f) \
  if (!(r.f == p.f)) { NotePrepMismatch("prep." #f, s); bad = 1; }
  VCMP(vertex_shader) VCMP(pixel_shader) VCMP(polygonal) VCMP(host_vertex_shader_type) VCMP(used_texture_mask) VCMP(depth_control.value) VCMP(color_mask)
  VCMP(layout) VCMP(shared_memory_access_stages) VCMP(merge_stride) VCMP(float_count) VCMP(pixel_shader_vertex_fetches) VCMP(host_primitive_reset_enabled)
  VCMP(host_primitive_type) VCMP(vertex_endian_key) VCMP(pixel_spec_key) VCMP(vertex_mul_key) VCMP(pixel_ext) VCMP(vertex_ext) VCMP(surface_info) VCMP(color_info)
  VCMP(depth_info) VCMP(viewport_info.z_min) VCMP(viewport_info.z_max) VCMP(viewport_info.xy_extent[0]) VCMP(viewport_info.xy_extent[1])
  VCMP(viewport_info.xy_offset[0]) VCMP(viewport_info.xy_offset[1]) VCMP(bi.polygonal) VCMP(bi.shader_32bit_index_dma) VCMP(bi.xy_extent[0]) VCMP(bi.xy_extent[1])
  VCMP(bi.depth_control.value) VCMP(bi.color_mask) VCMP(bi.used_texture_mask) VCMP(bi.scale_x) VCMP(bi.scale_y) VCMP(bi.fsi) VCMP(bi.host_rt) VCMP(bi.rg16_truncated)
  VCMP(bi.rgba16_truncated) VCMP(bi.native_truth) VCMP(bi.gamma_as_unorm16) VCMP(bi.image_view_format_swizzle) VCMP(bi.vs) VCMP(bi.ps)
#undef VCMP
  for (uint32_t i = 0; i < 3; ++i) {
    if (Bits(r.bi.ndc_scale[i]) != Bits(p.bi.ndc_scale[i])) { NotePrepMismatch("prep.bi.ndc_scale", s); bad = 1; }
    if (Bits(r.bi.ndc_offset[i]) != Bits(p.bi.ndc_offset[i])) { NotePrepMismatch("prep.bi.ndc_offset", s); bad = 1; }
  }
  if (rd.texture_key != d.texture_key) { NotePrepMismatch("draw.texture_key", s); bad = 1; }
  if (bad) ++g_rw.verify_mismatch_draws;
}

void* g_prep_entry = nullptr;  // (Session 95, r3cmd: the memo entry the last PrepareReuse served or stored; nullptr = none)
uint64_t g_prep_entry_id = 0, g_prep_ext = 0, g_prep_gen = 0;
OwnedTypedPrep* PrepareReuse(VulkanCommandProcessor& cp, const OwnedRecordState& s, OwnedTypedPrep& p, uint64_t frame, uint64_t t0, bool timers) {
  using A = OwnedFrontAccess;
  g_prep_entry = nullptr;
  const uint64_t vs = s.shader[0].hash, ps = s.shader[1].hash;
  const uint32_t prim = uint32_t(s.draw_initiator.prim_type);
  const uint32_t mode = uint32_t(s.modecontrol.edram_mode) | (uint32_t(s.draw_initiator.major_mode) << 8);
  const uint64_t ext = A::ExternalsReuse(cp);
  const uint64_t gen = shader_swap::TranslationGeneration();
  const bool live = rex::graphics::free_counters::g_cut.active && &s == &OwnedRecordStateNow();  // (the decoder's epochs describe this state only when it is the live one)
  const uint64_t render = OwnedRecordEpochsNow().render;
  ReuseEntry* e = nullptr;
  uint64_t state_hash = 0;
  bool have_hash = false, fast = false;
  ReuseLast& last = g_reuse_last;
  if (live && last.valid && last.e && last.e->valid && last.e->id == last.id && last.render == render && last.vs == vs && last.ps == ps && last.prim == prim &&
      last.mode == mode && last.ext == ext && last.gen == gen) {
    e = last.e;
    fast = true;
  } else {
    if (live && g_sh_valid && g_sh_render == render) {
      state_hash = g_sh_value;
    } else {
      state_hash = StateHash(s);
      if (live) {
        g_sh_valid = true;
        g_sh_render = render;
        g_sh_value = state_hash;
      }
    }
    have_hash = true;
    struct {
      uint64_t vs, ps;
      uint32_t prim, mode;
    } slot_key{vs, ps, prim, mode};
    ReuseEntry* set = &g_reuse[(XXH3_64bits(&slot_key, sizeof(slot_key)) % kReuseSets) * kReuseWays];
    for (uint32_t w = 0; w < kReuseWays; ++w) {
      ReuseEntry& c = set[w];
      if (c.valid && c.vs == vs && c.ps == ps && c.prim == prim && c.mode == mode && c.state_hash == state_hash && c.ext == ext && c.gen == gen) {
        e = &c;
        break;
      }
    }
  }
  // ---- a hit: validity checks, the draw part, the refresh
  if (e && (A::ActiveVs(cp) != e->active_vs || A::ActivePs(cp) != e->active_ps)) {
    ++g_rw.demoted_shader;  // (Build decides: the active shaders are not the ones the entry was made for)
    e = nullptr;
  }
  if (e) {
    const PrimitiveProcessor::DrawInput input = DrawInputOf(s);
    PrimitiveProcessor::ProcessingResult primitive;
    if (!A::ProcessInto(cp, input, primitive) || !primitive.host_draw_vertex_count || primitive.host_vertex_shader_type != e->prep.host_vertex_shader_type ||
        primitive.host_primitive_reset_enabled != e->prep.host_primitive_reset_enabled || primitive.host_primitive_type != e->prep.host_primitive_type) {
      ++g_rw.demoted_primitive;
      e = nullptr;
    } else {
      VulkanPipelineCache::TypedKeyState keys;
      keys.fetch = s.fetch;
      if (e->prep.vertex_shader == nullptr || A::EndianKeyNow(cp, *e->prep.vertex_shader, keys) != e->prep.vertex_endian_key ||
          A::PixelSpecKeyNow(cp, e->prep.pixel_shader, keys) != e->prep.pixel_spec_key) {
        ++g_rw.demoted_fetch_keys;
        e = nullptr;
      } else {
        CopyPrepState(p, e->prep);
        p.refusal = nullptr;
        p.input = input;
        p.vs_float = s.vs_float;
        p.ps_float = s.ps_float;
        p.bool_loop = s.bool_loop;
        p.fetch = s.fetch;
        Draw& d = g_own_draw;
        d = e->draw;
        p.draw = &d;
        A::ViewportJitter(cp, d, e->out);
        d.occlusion_query = A::OcclusionValid(cp);
        d.texture_key = (e->prep.pixel_shader && REXCVAR_GET(gpu_native_owned_alpha_learn)) ? A::TextureKey(s, e->prep.pixel_shader) : 0;
        p.bi.s = &s;
        p.bi.shader_32bit_index_dma = A::ApplyPrimitive(cp, s, primitive, e->prep.host_vertex_shader_type, d);
        p.bi.primitive = primitive;
        if (timers) g_own.prepare_ns += NowNs() - t0;
        ++g_own.memo_hits;
        if (e->frame != frame) {
          ++g_own.hits_cross_frame;
          ++g_rw.hits_cross_frame;
        }
        e->frame = frame;
        e->used = ++g_reuse_clock;
        (fast ? g_rw.hits_fast : g_rw.hits_hashed) += 1;
        if (live) {
          last.valid = true;
          last.id = e->id;
          last.e = e;
          last.render = render;
          last.vs = vs;
          last.ps = ps;
          last.prim = prim;
          last.mode = mode;
          last.ext = ext;
          last.gen = gen;
        }
        if (REXCVAR_GET(gpu_d3d_typed_memo_verify)) VerifyHit(cp, s, p, d);
        g_prep_entry = e;
        g_prep_entry_id = e->id;
        g_prep_ext = ext;
        g_prep_gen = gen;
        return &p;
      }
    }
  }
  // ---- a miss: Build from scratch, then the entry
  Draw& d = g_own_draw;
  d = Draw();
  g_own_old_stub = Draw();
  bool vs_absent = false, ps_absent = false, gs_skipped = false;
  g_scratch_out = TemplateOut();
  const Skip sk = A::Build(cp, s, g_own_old_stub, d, vs_absent, ps_absent, gs_skipped, &p.bi, nullptr, &g_scratch_out, &p);
  if (timers) g_own.prepare_ns += NowNs() - t0;
  if (sk != kSkipNone) {
    ++g_own.build_skip[sk];
    ++g_rw.refused;
    p.refusal = kSkipNames[sk];
    return nullptr;
  }
  ++g_own.memo_misses;
  ++g_rw.misses;
  if (d.restore) {
    ++g_rw.not_stored_restore;  // (its restore pages come from the fetch words; such draws are rebuilt every time)
    return &p;
  }
  if (!have_hash) state_hash = StateHash(s);
  struct {
    uint64_t vs, ps;
    uint32_t prim, mode;
  } slot_key{vs, ps, prim, mode};
  ReuseEntry* set = &g_reuse[(XXH3_64bits(&slot_key, sizeof(slot_key)) % kReuseSets) * kReuseWays];
  ReuseEntry* victim = nullptr;
  for (uint32_t w = 0; w < kReuseWays; ++w) {
    ReuseEntry& c = set[w];
    if (c.valid && c.vs == vs && c.ps == ps && c.prim == prim && c.mode == mode && c.state_hash == state_hash) {  // (the same key with another externals / generation: replaced)
      victim = &c;
      break;
    }
  }
  if (!victim) {
    for (uint32_t w = 0; w < kReuseWays; ++w) {
      if (!set[w].valid) {
        victim = &set[w];
        break;
      }
    }
  }
  if (!victim) {
    victim = set[0].used <= set[1].used ? &set[0] : &set[1];
    ++g_rw.evictions;
  }
  victim->valid = true;
  victim->id = ++g_reuse_ids;
  victim->used = ++g_reuse_clock;
  victim->frame = frame;
  victim->vs = vs;
  victim->ps = ps;
  victim->prim = prim;
  victim->mode = mode;
  victim->state_hash = state_hash;
  victim->ext = ext;
  victim->gen = gen;  // (read before Build: a bump during or after it makes the entry miss once, never absorbs another thread's)
  victim->active_vs = A::ActiveVs(cp);
  victim->active_ps = A::ActivePs(cp);
  victim->out = g_scratch_out;
  victim->draw = d;
  CopyPrepState(victim->prep, p);
  if (live) {
    last.valid = true;
    last.id = victim->id;
    last.e = victim;
    last.render = render;
    last.vs = vs;
    last.ps = ps;
    last.prim = prim;
    last.mode = mode;
    last.ext = ext;
    last.gen = gen;
  }
  g_prep_entry = victim;
  g_prep_entry_id = victim->id;
  g_prep_ext = ext;
  g_prep_gen = gen;
  return &p;
}
}  // namespace

// =====================================================================================================================================================
// Session 95 (r3cmd, gpu_d3d_draw_record): the per-draw record (draw_record.h).
DrawRecStats g_draw_rec_stats;
uint64_t g_draw_rec_generation = 1;
namespace {
constexpr uint32_t kRecSets = 2048, kRecWays = 2;  // (12 MB when the switch is on; about 1,000 draws a frame)
DrawRec* g_recs = nullptr;
struct RecUsed {  // the fetch slots of the last shader pair: their words are part of the key
  const void* vs = nullptr;
  const void* ps = nullptr;
  uint32_t tex_count = 0, vf_count = 0;
  uint8_t tex[32], vf[96];
};
RecUsed g_rec_used;
struct RecHash {
  bool valid = false;
  uint64_t render = 0, bool_loop = 0, render_hash = 0, bool_hash = 0;
};
RecHash g_rec_hash;
uint64_t g_rec_last_print = 0;
}  // namespace

void DrawRecTick(uint64_t frame) {
  DrawRecStats& t = g_draw_rec_stats;
  if (frame - g_rec_last_print < 600) return;
  if (t.asked) {
    REXGPU_INFO("[d3d][record] frames {}-{}: draws asked {}: from the record {} ({:.1f}%), new {}, no key {}, stale generation or shaders {}, stale externals {}; parts that went the full way: system constants {}, vertex plan {}, claim {}; stored {}, "
                "refused storage {}; verify: compared {}, differing: draw {}, system constants {}, claim {}, plan {}, primitive {}, index offset (expected) {}",
                g_rec_last_print, frame, t.asked, t.hits, 100.0 * double(t.hits) / double(t.asked), t.misses, t.unkeyed, t.stale_prep, t.stale_ext, t.sys_fallback, t.plan_fallback, t.claim_fallback, t.stored, t.refused,
                t.verify_compared, t.verify_draw, t.verify_sys, t.verify_claim, t.verify_plan, t.verify_primitive, t.verify_index_offset);
  }
  if (t.refused) {
    REXGPU_INFO("[d3d][record] refused by: index {}, cover {}, box {}, skin {}, memexport {}, ring {}, crowd {}, restore {}, clear {}, occlusion {}, claim {}, no pipeline {}, failed {}", t.why[0], t.why[1], t.why[2], t.why[3], t.why[4], t.why[5],
                t.why[6], t.why[7], t.why[8], t.why[9], t.why[10], t.why[11], t.why[12]);
  }
  t = DrawRecStats();
  g_rec_last_print = frame;
}

namespace {
// The fetch slots of the shader pair.
RecUsed& RecUsedOf(const VulkanShader* vs, const VulkanShader* ps) {
  RecUsed& u = g_rec_used;
  if (u.vs != vs || u.ps != ps) {
    u.vs = vs;
    u.ps = ps;
    uint32_t tex = vs->UsedTextureMaskFor(false) | vs->UsedTextureMaskFor(true);
    uint32_t vf[3];
    for (uint32_t i = 0; i < 3; ++i) vf[i] = vs->constant_register_map().vertex_fetch_bitmap[i];
    if (ps) {
      tex |= ps->UsedTextureMaskFor(false) | ps->UsedTextureMaskFor(true);
      for (uint32_t i = 0; i < 3; ++i) vf[i] |= ps->constant_register_map().vertex_fetch_bitmap[i];
    }
    u.tex_count = u.vf_count = 0;
    while (tex) {
      u.tex[u.tex_count++] = uint8_t(std::countr_zero(tex));
      tex &= tex - 1;
    }
    for (uint32_t i = 0; i < 3; ++i) {
      while (vf[i]) {
        u.vf[u.vf_count++] = uint8_t(i * 32 + std::countr_zero(vf[i]));
        vf[i] &= vf[i] - 1;
      }
    }
  }
  return u;
}
// The render state (the decoder's epoch says when it moved; the draw's own words are hashed by the caller, whatever the head held of them when it was hashed), the bool / loop words.
RecHash& RecHashOf(const OwnedRecordState& s) {
  const OwnedRecordEpochs& ep = OwnedRecordEpochsNow();
  RecHash& h = g_rec_hash;
  if (!h.valid || h.render != ep.render) {
    h.render = ep.render;
    // (min / max vertex index and the index offset are left out of the memoised head: the render epoch does not tell when they move, and the key hashes them itself below)
    const uint8_t* sb = reinterpret_cast<const uint8_t*>(&s);
    const size_t after_idx = offsetof(OwnedRecordState, indx_offset) + sizeof(s.indx_offset);
    uint64_t head = XXH3_64bits(&s, offsetof(OwnedRecordState, min_vtx_indx));
    head = XXH3_64bits_withSeed(sb + after_idx, offsetof(OwnedRecordState, draw_initiator) - after_idx, head);
    h.render_hash = XXH3_64bits_withSeed(sb + offsetof(OwnedRecordState, program_cntl), offsetof(OwnedRecordState, vs_float) - offsetof(OwnedRecordState, program_cntl), head);
  }
  if (!h.valid || h.bool_loop != ep.bool_loop) {
    h.bool_loop = ep.bool_loop;
    h.bool_hash = XXH3_64bits(s.bool_loop, sizeof(s.bool_loop));
  }
  h.valid = true;
  return h;
}
}  // namespace

DrawRec* DrawRecFind(VulkanCommandProcessor& cp, const OwnedRecordState& s, bool& hit) {
  using A = OwnedFrontAccess;
  hit = false;
  DrawRecStats& t = g_draw_rec_stats;
  ++t.asked;
  if (!g_recs) {
    g_recs = new DrawRec[size_t(kRecSets) * kRecWays];
    REXGPU_INFO("[d3d][record] on: gpu_d3d_draw_record {} ({} slots of {} bytes)", REXCVAR_GET(gpu_d3d_draw_record), kRecSets * kRecWays, sizeof(DrawRec));
  }
  const VulkanShader* const vs = static_cast<const VulkanShader*>(A::ActiveVs(cp));
  const VulkanShader* const ps = static_cast<const VulkanShader*>(A::ActivePs(cp));
  if (!vs || !vs->is_ucode_analyzed() || (ps && !ps->is_ucode_analyzed()) || A::DlaaJitter()) {
    ++t.unkeyed;
    return nullptr;
  }
  RecUsed& u = RecUsedOf(vs, ps);
  RecHash& h = RecHashOf(s);
  struct {
    uint64_t render, bools, vs, ps;
    uint32_t initiator, dma_base, dma_size, indx_offset, min_vtx, max_vtx, viz;
    uint32_t pad;
    uint32_t words[32 * 6 + 96 * 2];
  } in;
  in.render = h.render_hash;
  in.bools = h.bool_hash;
  in.vs = s.shader[0].hash;
  in.ps = s.shader[1].hash;
  in.initiator = s.draw_initiator.value;
  in.dma_base = s.dma_base;
  in.dma_size = s.dma_size.value;
  in.indx_offset = s.indx_offset.value;
  in.min_vtx = s.min_vtx_indx.value;
  in.max_vtx = s.max_vtx_indx.value;
  in.viz = s.draw_viz_query;
  in.pad = 0;
  uint32_t n = 0;
  for (uint32_t i = 0; i < u.tex_count; ++i) {
    std::memcpy(in.words + n, s.fetch + 6 * size_t(u.tex[i]), 24);
    n += 6;
  }
  for (uint32_t i = 0; i < u.vf_count; ++i) {
    std::memcpy(in.words + n, s.fetch + 2 * size_t(u.vf[i]), 8);
    n += 2;
  }
  const uint64_t key = XXH3_64bits(&in, offsetof(decltype(in), words) + size_t(n) * 4);
  DrawRec* const set = g_recs + size_t(key % kRecSets) * kRecWays;
  const uint64_t frame = A::Frame(cp);
  for (uint32_t w = 0; w < kRecWays; ++w) {
    DrawRec& r = set[w];
    if (r.key == key && r.vs == in.vs && r.ps == in.ps && r.dma_base == in.dma_base && r.initiator == in.initiator) {
      r.frame = frame;
      hit = r.complete;
      if (!hit) ++t.misses;
      return &r;
    }
  }
  DrawRec& v = set[0].frame <= set[1].frame ? set[0] : set[1];
  v.key = key;
  v.vs = in.vs;
  v.ps = in.ps;
  v.dma_base = in.dma_base;
  v.initiator = in.initiator;
  v.frame = frame;
  v.complete = false;
  ++t.misses;
  return &v;
}

OwnedTypedPrep* DrawRecPrep(VulkanCommandProcessor& cp, const OwnedRecordState& s, DrawRec& r) {
  using A = OwnedFrontAccess;
  DrawRecStats& t = g_draw_rec_stats;
  if (r.generation != g_draw_rec_generation || A::ActiveVs(cp) != r.active_vs || A::ActivePs(cp) != r.active_ps) {
    ++t.stale_prep;
    return nullptr;
  }
  if (A::ExternalsReuse(cp) != r.ext || shader_swap::TranslationGeneration() != r.gen) {
    ++t.stale_ext;
    return nullptr;
  }
  // The primitive pass's one effect: the index span, requested as Process requests it.
  if (r.index_bytes) {
    if (!A::RequestIndexSpan(cp, r.index_base, r.index_bytes)) {
      ++t.stale_prep;
      return nullptr;
    }
  }
  PrimitiveProcessor::SetLastIndexLoadSpan(r.index_base, r.index_bytes, r.index_expansion);
  const uint64_t frame = A::Frame(cp);
  if (!g_own_first_frame) g_own_first_frame = g_own_last_print = frame;
  if (frame - g_own_last_print >= 600) PrintOwnWindow(frame);
  OwnedTypedPrep& p = g_own_prep;
  std::memcpy(static_cast<void*>(&p), r.prep_image, kPrepHead);
  std::memcpy(static_cast<void*>(&p.bi), r.prep_image + kPrepHead, kBiHead);
  p.refusal = nullptr;
  p.input = DrawInputOf(s);
  p.vs_float = s.vs_float;
  p.ps_float = s.ps_float;
  p.bool_loop = s.bool_loop;
  p.fetch = s.fetch;
  p.draw = &r.draw;
  p.bi.s = &s;
  p.bi.shader_32bit_index_dma = r.s32;
  p.bi.primitive = r.primitive;
  r.draw.occlusion_query = A::OcclusionValid(cp);  // (the viewport has no jitter: a record's draw is made without it, DrawRecFind)
  ++g_own.memo_hits;
  ++t.hits;
  return &p;
}

bool DrawRecSystemConstants(VulkanCommandProcessor& cp, void* ctx, const DrawRec& r) {
  if (!r.sys_ok || !OwnedFrontAccess::SystemConstantsFromRecord(cp, static_cast<OwnedTypedPrep*>(ctx)->bi, r)) {
    ++g_draw_rec_stats.sys_fallback;
    return false;
  }
  return true;
}

void DrawRecFill(VulkanCommandProcessor& cp, DrawRec& r, const OwnedRecordState& s, const OwnedTypedPrep& p, const native::Draw& claimed, const native::VertexInputPlan& plan, bool plan_kept) {
  using A = OwnedFrontAccess;
  DrawRecStats& t = g_draw_rec_stats;
  if (!g_prep_entry) {
    ++t.refused;
    r.complete = false;
    return;
  }
  static_assert(kPrepHead + kBiHead <= kDrawRecPrepHead + kDrawRecBiHead && alignof(OwnedTypedPrep) <= 16);
  std::memcpy(r.prep_image, static_cast<const void*>(&p), kPrepHead);
  std::memcpy(r.prep_image + kPrepHead, static_cast<const void*>(&p.bi), kBiHead);
  r.ext = g_prep_ext;
  r.gen = g_prep_gen;
  r.generation = g_draw_rec_generation;
  r.active_vs = A::ActiveVs(cp);
  r.active_ps = A::ActivePs(cp);
  r.draw = *p.draw;
  r.primitive = p.bi.primitive;
  r.s32 = p.bi.shader_32bit_index_dma;
  r.sys_ok = true;
  r.scaled = p.bi.scaled_from_cache;
  r.sys = A::SystemConstantsOf(cp);
  r.plan_ok = plan_kept;
  if (plan_kept) r.plan = plan;
  r.claim_ok = true;
  r.pipeline = claimed.pipeline;
  r.truth = claimed.truth;
  r.owned = claimed.owned;
  r.post = claimed.post;
  r.truth_color = claimed.truth_color;
  r.truth_depth = claimed.truth_depth;
  r.truth_class = claimed.truth_class;
  r.truth_sx = claimed.truth_sx;
  r.truth_sy = claimed.truth_sy;
  r.complete = true;
  ++t.stored;
}

void DrawRecVerify(const DrawRec& r, VulkanCommandProcessor& cp, const OwnedTypedPrep& p, const native::Draw& claimed, const native::VertexInputPlan& plan) {
  using A = OwnedFrontAccess;
  using SC = SpirvShaderTranslator::SystemConstants;
  DrawRecStats& t = g_draw_rec_stats;
  ++t.verify_compared;
  static uint32_t logged = 0;
  const auto note = [&](const char* what, uint64_t a, uint64_t b) {
    if (logged < 40) {
      ++logged;
      REXGPU_WARN("[d3d][record] verify: {} differs (fresh {:X}, record {:X}): vs {:016X} ps {:016X} frame {}", what, a, b, r.vs, r.ps, A::Frame(cp));
    }
  };
  // The Draw of the preparation.
  {
    Draw d = r.draw;
    d.occlusion_query = p.draw->occlusion_query;
    Compare c;
    c.kind = kOwnedFrontFull;
    c.vs_hash = p.draw->vertex_shader_hash;
    c.ps_hash = p.draw->pixel_shader_hash;
    c.by_field_p = g_rw.by_field;
    c.logged_p = g_rw_logged;
    c.nlogged_p = &g_rw_nlogged;
    c.tag = "draw record (record vs the full way)";
    CompareDraws(c, *p.draw, d);
    if (c.any) ++t.verify_draw;
    if (p.draw->texture_key != d.texture_key) note("texture key", p.draw->texture_key, d.texture_key);
  }
  // The primitive pass.
  {
    const PrimitiveProcessor::ProcessingResult& a = p.bi.primitive;
    const PrimitiveProcessor::ProcessingResult& b = r.primitive;
    if (a.index_buffer_type != b.index_buffer_type || a.guest_index_base != b.guest_index_base || a.host_draw_vertex_count != b.host_draw_vertex_count || a.guest_draw_vertex_count != b.guest_draw_vertex_count ||
        a.host_index_format != b.host_index_format || a.host_shader_index_endian != b.host_shader_index_endian || a.host_primitive_reset_enabled != b.host_primitive_reset_enabled ||
        a.host_primitive_type != b.host_primitive_type || a.host_vertex_shader_type != b.host_vertex_shader_type || a.line_loop_closing_index != b.line_loop_closing_index ||
        a.host_index_buffer_handle != b.host_index_buffer_handle) {
      ++t.verify_primitive;
      note("primitive", uint64_t(a.index_buffer_type), uint64_t(b.index_buffer_type));
    }
  }
  // The system constants: every byte but the storage window's selector.
  {
    const SC& a = A::SystemConstantsOf(cp);
    const SC& b = r.sys;
    constexpr size_t kWin = offsetof(SC, shared_memory_window_base_dwords), kWinEnd = kWin + sizeof(a.shared_memory_window_base_dwords);
    const uint8_t* x = reinterpret_cast<const uint8_t*>(&a);
    const uint8_t* y = reinterpret_cast<const uint8_t*>(&b);
    if (std::memcmp(x, y, kWin) != 0 || std::memcmp(x + kWinEnd, y + kWinEnd, sizeof(SC) - kWinEnd) != 0) {
      ++t.verify_sys;
      size_t at = 0;
      while (at < sizeof(SC) && (x[at] == y[at] || (at >= kWin && at < kWinEnd))) ++at;
      uint32_t va = 0, vb = 0;
      std::memcpy(&va, x + (at & ~size_t(3)), 4);
      std::memcpy(&vb, y + (at & ~size_t(3)), 4);
      if (logged < 40) {
        ++logged;
        REXGPU_WARN("[d3d][record] verify: system constants differ at byte {} (fresh {:08X}, record {:08X}): vs {:016X} ps {:016X}", at, va, vb, r.vs, r.ps);
      }
    }
  }
  if (claimed.pipeline != r.pipeline || claimed.truth != r.truth || claimed.owned != r.owned || claimed.post != r.post || claimed.truth_color != r.truth_color || claimed.truth_depth != r.truth_depth ||
      claimed.truth_class != r.truth_class || claimed.truth_sx != r.truth_sx || claimed.truth_sy != r.truth_sy) {
    ++t.verify_claim;
    note(claimed.pipeline != r.pipeline ? "claim pipeline" : claimed.truth_color != r.truth_color ? "claim colour" : claimed.truth_depth != r.truth_depth ? "claim depth" : "claim other",
         uint64_t(claimed.pipeline), uint64_t(r.pipeline));
  }
  // (The index offset is counted apart: a hit takes it live from Find, so a difference is expected and not a plan error.)
  if (plan.ok && r.plan_ok && plan.index_offset != r.plan.index_offset) ++t.verify_index_offset;
  if (r.plan_ok != plan.ok || (plan.ok && (plan.form != r.plan.form || plan.pool_index != r.plan.pool_index || plan.index_count != r.plan.index_count ||
                                           plan.first_vertex != r.plan.first_vertex || plan.vb_offsets[0] != r.plan.vb_offsets[0] || plan.index_min != r.plan.index_min || plan.index_max != r.plan.index_max))) {
    ++t.verify_plan;
    note(r.plan_ok != plan.ok ? "plan taken" : plan.form != r.plan.form ? "plan form" : plan.index_offset != r.plan.index_offset ? "plan index offset" : plan.index_count != r.plan.index_count ? "plan index count" :
         plan.first_vertex != r.plan.first_vertex ? "plan first vertex" : "plan other", plan.index_offset, r.plan.index_offset);
  }
}

// =====================================================================================================================================================
// Session 95 (gpu_d3d_draw_slot): the slot's side of the record (draw_slot.h).
namespace {
// The generation word: everything DrawRecPrep checks, and the record it is for. The slot takes it from the record's own fields, a draw from the live ones: equal = DrawRecPrep would pass.
// The fetch slots the shader pair uses, read live (not from the key's memo: that one is keyed on the pointers and lags when a shader's mask grows at its first translation). Part of the generation word, so a slot dies when a mask moves.
struct SlotMasks {
  uint32_t tex = 0, vf[3] = {0, 0, 0};
};
SlotMasks SlotMasksOf(const VulkanShader* vs, const VulkanShader* ps) {
  SlotMasks m;
  m.tex = vs->UsedTextureMaskFor(false) | vs->UsedTextureMaskFor(true);
  for (uint32_t i = 0; i < 3; ++i) m.vf[i] = vs->constant_register_map().vertex_fetch_bitmap[i];
  if (ps) {
    m.tex |= ps->UsedTextureMaskFor(false) | ps->UsedTextureMaskFor(true);
    for (uint32_t i = 0; i < 3; ++i) m.vf[i] |= ps->constant_register_map().vertex_fetch_bitmap[i];
  }
  return m;
}
uint64_t SlotGeneration(uint64_t key, uint64_t vs_hash, uint64_t ps_hash, const void* active_vs, const void* active_ps, uint64_t generation, uint64_t ext, uint64_t gen, uint32_t settings, const SlotMasks& m) {
  const struct {
    uint64_t key, vs_hash, ps_hash, active_vs, active_ps, generation, ext, gen, settings;
    uint32_t tex, vf0, vf1, vf2;
  } in{key, vs_hash, ps_hash, uint64_t(reinterpret_cast<uintptr_t>(active_vs)), uint64_t(reinterpret_cast<uintptr_t>(active_ps)), generation, ext, gen, settings, m.tex, m.vf[0], m.vf[1], m.vf[2]};
  return XXH3_64bits(&in, sizeof(in));
}
void SlotOwnWords(const OwnedRecordState& s, uint32_t own[7]) {
  own[0] = s.draw_initiator.value;
  own[1] = s.dma_base;
  own[2] = s.dma_size.value;
  own[3] = s.indx_offset.value;
  own[4] = s.min_vtx_indx.value;
  own[5] = s.max_vtx_indx.value;
  own[6] = s.draw_viz_query;
}
uint64_t SlotState(const RecHash& h) { return h.render_hash ^ std::rotl(h.bool_hash, 21); }
}  // namespace

uint64_t DrawSlotSysHash(const DrawRec& r) {
  using SC = SpirvShaderTranslator::SystemConstants;
  constexpr size_t kWin = offsetof(SC, shared_memory_window_base_dwords), kWinEnd = kWin + sizeof(r.sys.shared_memory_window_base_dwords);
  const uint8_t* const b = reinterpret_cast<const uint8_t*>(&r.sys);
  const uint64_t h = XXH3_64bits_withSeed(b + kWinEnd, sizeof(SC) - kWinEnd, XXH3_64bits(b, kWin));
  return h ? h : 1;
}

DrawRec* DrawSlotTry(VulkanCommandProcessor& cp, const OwnedRecordState& s, uint32_t settings) {
  using A = OwnedFrontAccess;
  DrawSlots& g = g_draw_slots;
  DrawSlotStats& t = g.stats;
  ++t.asked;
  DrawSlotMiss why = kSlotNoNext;
  DrawRec* const next = g.cursor ? const_cast<DrawRec*>(static_cast<const DrawRec*>(g.cursor->next_hint)) : nullptr;
  if (next) {
    const DrawSlot& slot = DrawSlotOf(next);
    if (slot.rec != next) {
      why = kSlotNotStored;
    } else if (!next->complete) {
      why = kSlotIncomplete;
    } else if (A::DlaaJitter()) {
      why = kSlotJitter;
    } else {
      DrawSlotLive live;
      const VulkanShader* const avs = static_cast<const VulkanShader*>(A::ActiveVs(cp));
      const VulkanShader* const aps = static_cast<const VulkanShader*>(A::ActivePs(cp));
      live.gen = avs ? SlotGeneration(next->key, s.shader[0].hash, s.shader[1].hash, avs, aps, g_draw_rec_generation, A::ExternalsReuse(cp), shader_swap::TranslationGeneration(), settings, SlotMasksOf(avs, aps)) : ~0ull;
      live.state = SlotState(RecHashOf(s));
      SlotOwnWords(s, live.own);
      live.fetch = s.fetch;
      why = DrawSlotCompare(slot, live);
    }
  }
  if (why == kSlotHit) {
    next->frame = A::Frame(cp);
    ++t.hits;
    return next;
  }
  ++t.misses;
  ++t.why[why];
  return nullptr;
}

OwnedTypedPrep* DrawSlotPrep(VulkanCommandProcessor& cp, const OwnedRecordState& s, DrawRec& r) {
  using A = OwnedFrontAccess;
  DrawRecStats& t = g_draw_rec_stats;
  ++t.asked;  // (DrawRecFind's count: the record's line stays the count of every typed draw)
  // The primitive pass's one effect: the index span, requested as Process requests it.
  if (r.index_bytes) {
    if (!A::RequestIndexSpan(cp, r.index_base, r.index_bytes)) {
      ++t.stale_prep;
      return nullptr;
    }
  }
  PrimitiveProcessor::SetLastIndexLoadSpan(r.index_base, r.index_bytes, r.index_expansion);
  const uint64_t frame = A::Frame(cp);
  if (!g_own_first_frame) g_own_first_frame = g_own_last_print = frame;
  if (frame - g_own_last_print >= 600) PrintOwnWindow(frame);
  // The head of an OwnedTypedPrep, in place: a hit reads the fields of the head and nothing of the lists after it (a full build of the constants takes DrawSlotFullPrep).
  OwnedTypedPrep& p = *reinterpret_cast<OwnedTypedPrep*>(r.prep_image);
  p.input = DrawInputOf(s);
  p.vs_float = s.vs_float;
  p.ps_float = s.ps_float;
  p.bool_loop = s.bool_loop;
  p.fetch = s.fetch;
  p.draw = &r.draw;
  p.bi.s = &s;
  r.draw.occlusion_query = A::OcclusionValid(cp);
  ++g_own.memo_hits;
  ++t.hits;
  return &p;
}

OwnedTypedPrep* DrawSlotFullPrep(const OwnedRecordState& s, DrawRec& r) {
  OwnedTypedPrep& p = g_own_prep;
  std::memcpy(static_cast<void*>(&p), r.prep_image, kPrepHead);
  std::memcpy(static_cast<void*>(&p.bi), r.prep_image + kPrepHead, kBiHead);
  p.refusal = nullptr;
  p.input = DrawInputOf(s);
  p.vs_float = s.vs_float;
  p.ps_float = s.ps_float;
  p.bool_loop = s.bool_loop;
  p.fetch = s.fetch;
  p.draw = &r.draw;
  p.bi.s = &s;
  p.bi.shader_32bit_index_dma = r.s32;
  p.bi.primitive = r.primitive;
  return &p;
}

bool DrawSlotSystemConstants(VulkanCommandProcessor& cp, void* ctx, const DrawRec& r) {
  const uint64_t sys_hash = DrawSlotAuxOf(&r).sys_hash;
  if (!r.sys_ok || !sys_hash || !OwnedFrontAccess::SystemConstantsFromSlot(cp, static_cast<OwnedTypedPrep*>(ctx)->bi, r, sys_hash)) {
    ++g_draw_rec_stats.sys_fallback;
    ++g_draw_slots.stats.sys_full;
    return false;
  }
  return true;
}

void DrawSlotEnsure(int32_t mode) {
  if (g_recs && !DrawSlotsBound()) DrawSlotsBind(g_recs, kRecSets * kRecWays, mode);
}

void DrawSlotLearn(DrawRec& r, const OwnedRecordState& s, uint32_t settings) {
  const VulkanShader* const vs = static_cast<const VulkanShader*>(r.active_vs);
  const VulkanShader* const ps = static_cast<const VulkanShader*>(r.active_ps);
  const SlotMasks m = SlotMasksOf(vs, ps);
  uint8_t tex[32], vf[96];
  uint32_t tex_count = 0, vf_count = 0;
  for (uint32_t t = m.tex; t; t &= t - 1) tex[tex_count++] = uint8_t(std::countr_zero(t));
  for (uint32_t i = 0; i < 3; ++i) {
    for (uint32_t v = m.vf[i]; v; v &= v - 1) vf[vf_count++] = uint8_t(i * 32 + std::countr_zero(v));
  }
  DrawSlotLive live;
  live.gen = SlotGeneration(r.key, r.vs, r.ps, r.active_vs, r.active_ps, r.generation, r.ext, r.gen, settings, m);
  live.state = SlotState(RecHashOf(s));
  SlotOwnWords(s, live.own);
  live.fetch = s.fetch;
  DrawSlotWrite(DrawSlotOf(&r), &r, DrawSlotIntern(m.tex, m.vf, tex_count, vf_count, tex, vf), live);
}

void DrawSlotVerifyFind(VulkanCommandProcessor& cp, DrawRec* slot_rec, DrawRec* found, bool found_complete) {
  using A = OwnedFrontAccess;
  using SC = SpirvShaderTranslator::SystemConstants;
  DrawSlotStats& t = g_draw_slots.stats;
  ++t.verify_compared;
  static uint32_t logged = 0;
  const auto note = [&](const char* what) {
    if (logged < 40) {
      ++logged;
      REXGPU_WARN("[d3d][slot] verify: {}: vs {:016X} ps {:016X} frame {}", what, slot_rec->vs, slot_rec->ps, A::Frame(cp));
    }
  };
  if (found != slot_rec || !found_complete) {
    ++t.verify_record;
    note("the slot's record is not the one the key finds");
    return;
  }
  const DrawRec& r = *slot_rec;
  if (r.generation != g_draw_rec_generation || A::ActiveVs(cp) != r.active_vs || A::ActivePs(cp) != r.active_ps || A::ExternalsReuse(cp) != r.ext || shader_swap::TranslationGeneration() != r.gen) {
    ++t.verify_stale;
    note("the slot's generation word passed a record DrawRecPrep refuses");
  }
  // A hit would have left the live block alone: it must hold the record's bytes.
  const uint64_t sys_hash = DrawSlotAuxOf(slot_rec).sys_hash;
  if (sys_hash && sys_hash == g_draw_slot_sys_hash) {
    const SC& a = A::SystemConstantsOf(cp);
    constexpr size_t kWin = offsetof(SC, shared_memory_window_base_dwords), kWinEnd = kWin + sizeof(a.shared_memory_window_base_dwords);
    const uint8_t* const x = reinterpret_cast<const uint8_t*>(&a);
    const uint8_t* const y = reinterpret_cast<const uint8_t*>(&r.sys);
    if (std::memcmp(x, y, kWin) != 0 || std::memcmp(x + kWinEnd, y + kWinEnd, sizeof(SC) - kWinEnd) != 0) {
      ++t.verify_sys;
      note("the live system constants are not the block their hash says");
    }
  }
}

void DrawSlotVerifyPrep(VulkanCommandProcessor& cp, const DrawRec& r, const OwnedTypedPrep& p) {
  const OwnedTypedPrep& k = *reinterpret_cast<const OwnedTypedPrep*>(r.prep_image);
  if (k.vertex_shader != p.vertex_shader || k.pixel_shader != p.pixel_shader || k.polygonal != p.polygonal || k.host_vertex_shader_type != p.host_vertex_shader_type || k.used_texture_mask != p.used_texture_mask ||
      k.color_mask != p.color_mask || k.layout != p.layout || k.shared_memory_access_stages != p.shared_memory_access_stages || k.merge_stride != p.merge_stride || k.float_count != p.float_count ||
      k.pixel_shader_vertex_fetches != p.pixel_shader_vertex_fetches || k.host_primitive_reset_enabled != p.host_primitive_reset_enabled || k.host_primitive_type != p.host_primitive_type ||
      k.vertex_endian_key != p.vertex_endian_key || k.pixel_spec_key != p.pixel_spec_key || k.vertex_mul_key != p.vertex_mul_key || k.pixel_ext != p.pixel_ext || k.vertex_ext != p.vertex_ext ||
      k.surface_info != p.surface_info || k.color_info != p.color_info || k.depth_info != p.depth_info) {
    ++g_draw_slots.stats.verify_prep;
    static uint32_t logged = 0;
    if (logged < 40) {
      ++logged;
      REXGPU_WARN("[d3d][slot] verify: the record's preparation is not the full way's: vs {:016X} ps {:016X} frame {}", r.vs, r.ps, OwnedFrontAccess::Frame(cp));
    }
  }
}

OwnedTypedPrep* OwnedFrontPrepare(VulkanCommandProcessor& cp, const OwnedRecordState& s) {
  using A = OwnedFrontAccess;
  OwnedTypedPrep& p = g_own_prep;
  p.refusal = nullptr;
  const uint64_t frame = A::Frame(cp);
  if (!g_own_first_frame) g_own_first_frame = g_own_last_print = frame;
  if (frame - g_own_last_print >= 600) PrintOwnWindow(frame);
  const bool timers = REXCVAR_GET(gpu_d3d_typed_timers);  // (Session 93: off = no clock reads)
  const uint64_t t0 = timers ? NowNs() : 0;
  if (REXCVAR_GET(gpu_d3d_typed_memo_reuse)) return PrepareReuse(cp, s, p, frame, t0, timers);
  const uint64_t vs = s.shader[0].hash, ps = s.shader[1].hash;
  const uint32_t prim = uint32_t(s.draw_initiator.prim_type);
  const uint32_t mode = uint32_t(s.modecontrol.edram_mode) | (uint32_t(s.draw_initiator.major_mode) << 8);
  const uint64_t state_hash = StateHash(s);
  const uint64_t ext_hash = A::Externals(cp);
  struct {
    uint64_t vs, ps;
    uint32_t prim, mode;
  } slot_key{vs, ps, prim, mode};
  OwnMemo& e = g_own_memo[XXH3_64bits(&slot_key, sizeof(slot_key)) % kMemoSlots];
  const bool hit = e.valid && e.vs == vs && e.ps == ps && e.prim == prim && e.mode == mode && e.state_hash == state_hash && e.ext_hash == ext_hash;
  Draw& d = g_own_draw;
  d = Draw();
  g_own_old_stub = Draw();
  bool vs_absent = false, ps_absent = false, gs_skipped = false;
  TemplateIn tin;
  if (hit) {
    static_cast<TemplateOut&>(tin) = e.out;
    tin.draw = &e.draw;
  }
  g_scratch_out = TemplateOut();
  const Skip sk = A::Build(cp, s, g_own_old_stub, d, vs_absent, ps_absent, gs_skipped, &p.bi, hit ? &tin : nullptr, hit ? nullptr : &g_scratch_out, &p);
  if (timers) g_own.prepare_ns += NowNs() - t0;
  if (sk != kSkipNone) {
    ++g_own.build_skip[sk];
    p.refusal = kSkipNames[sk];
    return nullptr;
  }
  if (hit) {
    ++g_own.memo_hits;
    if (e.frame != frame) ++g_own.hits_cross_frame;
  } else {
    ++g_own.memo_misses;
    e.valid = true;
    e.vs = vs;
    e.ps = ps;
    e.state_hash = state_hash;
    e.ext_hash = ext_hash;
    e.frame = frame;
    e.prim = prim;
    e.mode = mode;
    e.out = g_scratch_out;
    e.draw = d;
  }
  return &p;
}

// DrawFromBase's system constants step: the typed block into the processor's own system_constants_ (the old UpdateSystemConstantValues writes it
// field by field; BuildSystemConstants is the same body over the typed fields - 0 mismatches in the M4c checks), the constant buffer marked out of
// date when a byte changed (what the old function's `dirty` does), the skip cache's stored inputs dropped (it no longer describes the block).
void OwnedFrontSystemConstants(VulkanCommandProcessor& cp, void* ctx, const PrimitiveProcessor::ProcessingResult& primitive) {
  OwnedTypedPrep& p = *static_cast<OwnedTypedPrep*>(ctx);
  OwnedBindInputs& bi = p.bi;
  bi.primitive = primitive;
  OwnedFrontAccess::SystemConstantsInto(cp, bi);
}

void OwnedFrontTypedNote(uint32_t outcome, uint64_t n) { g_own.counts[outcome] += n; }
void OwnedFrontTypedFallback(const char* reason) {
  ++g_own.fallbacks[reason];
  if (rex::graphics::free_counters::g_cut.active) rex::graphics::free_counters::NoteCutDrop(reason);  // (M4f-C: under the cut a refusal is a DROPPED draw, counted by reason)
}
void OwnedFrontTypedAfter() { g_rep.valid = false; }  // (Session 93: the typed repeat tracker g_trep is set by OwnedFrontRepeatRecord and ended by OwnedFrontRepeatBreak)
void OwnedFrontTypedTick(VulkanCommandProcessor& cp) {  // the window line also prints when every draw is refused before OwnedFrontPrepare
  const uint64_t frame = OwnedFrontAccess::Frame(cp);
  if (!g_own_first_frame) g_own_first_frame = g_own_last_print = frame;
  if (frame - g_own_last_print >= 600) PrintOwnWindow(frame);
}
void OwnedFrontTypedActive(bool on) { g_typed_draw_active = on; }

// ---- Session 93 (gpu_d3d_typed_repeat): the typed repeat tracker (see the header).
namespace {
void FillTRep(TRepKey& k, VulkanCommandProcessor& cp, const OwnedRecordState& s) {
  using A = OwnedFrontAccess;
  const OwnedRecordEpochs& ep = OwnedRecordEpochsNow();
  k.valid = true;
  k.seq = OwnedRecordDrawSeq();
  k.serial = g_owned_repeat_base_serial;
  k.render = ep.render;
  k.ps_float = ep.ps_float;
  k.bool_loop = ep.bool_loop;
  k.fetch = ep.fetch;
  k.ext = A::ExternalsReuse(cp);
  k.vs = s.shader[0].hash;
  k.ps = s.shader[1].hash;
  k.bin_mask = s.bin_mask;
  k.bin_select = s.bin_select;
  k.prim = uint32_t(s.draw_initiator.prim_type);
  k.mode = uint32_t(s.modecontrol.edram_mode) | (uint32_t(s.draw_initiator.major_mode) << 8);
  k.init = s.draw_initiator.value;
  k.dma_base = s.dma_base;
  k.dma_size = s.dma_size.value;
  k.occlusion = A::OcclusionValid(cp) ? 1u : 0u;
  k.viz = s.draw_viz_query;
  k.idx[0] = s.min_vtx_indx.value;
  k.idx[1] = s.max_vtx_indx.value;
  k.idx[2] = s.indx_offset.value;
  A::Jitter(cp, k.jitter);
}
}  // namespace

void OwnedFrontRepeatBegin() {
  g_trep_prev = g_trep;
  g_trep.valid = false;
}

bool OwnedFrontRepeatCandidate(VulkanCommandProcessor& cp, const OwnedRecordState& s) {
  const TRepKey& a = g_trep_prev;
  if (!a.valid || !rex::graphics::free_counters::g_cut.active || &s != &OwnedRecordStateNow()) return false;
  TRepKey b;
  FillTRep(b, cp, s);
  ++g_rw.rep_checked;
  // the previous recorded base is the draw item just before this one, and nothing but the vertex float constants moved (the decoder's epochs: the render state, the
  // pixel floats, the bool / loop constants, the fetch constants), the shaders, the draw words, the externals (frame latches, settings), the jitter, the bin state
  const bool grp[7] = {
      a.seq + 1 == b.seq && a.serial == b.serial,
      a.render == b.render,
      a.ps_float == b.ps_float && a.bool_loop == b.bool_loop && a.fetch == b.fetch,
      a.ext == b.ext,
      a.vs == b.vs && a.ps == b.ps && a.bin_mask == b.bin_mask && a.bin_select == b.bin_select && a.prim == b.prim && a.mode == b.mode && a.init == b.init,
      a.dma_base == b.dma_base && a.dma_size == b.dma_size && a.idx[0] == b.idx[0] && a.idx[1] == b.idx[1] && a.idx[2] == b.idx[2],
      a.occlusion == b.occlusion && a.viz == b.viz && Bits(a.jitter[0]) == Bits(b.jitter[0]) && Bits(a.jitter[1]) == Bits(b.jitter[1])};
  uint32_t first_bad = 7;
  for (uint32_t i = 0; i < 7 && first_bad == 7; ++i) {
    if (!grp[i]) first_bad = i;
  }
  const bool same = first_bad == 7;
  if (shader_swap::InstanceWorldLatched()) {  // (Session 94: the funnel of the instancing, counted by group of the first difference - the census's "differ only in the VS floats" pairs do not need the same mesh, a repeat does)
    const uint64_t frame = OwnedFrontAccess::Frame(cp);
    shader_swap::NoteWorldStat(frame, shader_swap::kWsChecked);
    shader_swap::NoteWorldStat(frame, same ? shader_swap::kWsCandidate : shader_swap::kWsMiss, first_bad);
  }
  if (same) ++g_rw.rep_candidates;
  return same;
}

void OwnedFrontRepeatRecord(VulkanCommandProcessor& cp, const OwnedRecordState& s) {
  if (!rex::graphics::free_counters::g_cut.active || &s != &OwnedRecordStateNow()) {
    g_trep.valid = false;
    return;
  }
  FillTRep(g_trep, cp, s);
}

void OwnedFrontRepeatNote(uint32_t outcome) { ++g_rw.rep[outcome]; }
uint64_t OwnedFrontStateHashOf(const OwnedRecordState& s) { return StateHash(s); }  // (Session 95, r3cmd)
const char* OwnedFrontPrepareRefusal() { return g_own_prep.refusal ? g_own_prep.refusal : "unknown"; }
uint64_t OwnedFrontNowNs() { return NowNs(); }

}  // namespace rex::graphics::vulkan

// ---------------------------------------------------------------------------------------------------------------------------------
// M4f-B (latch bit 6, value 64): the cover / bounds rectangle inputs from the typed state (NativeCoverRect / NativeBoundsRect take a CoverInputs).
// Every field comes from the snapshot; the constant arrays are the snapshot's own (register layout), hashed in place. The ONE thing that is a register
// file is the CPU shader interpreter's register source (it reads its float constants, bool / loop constants, the vertex fetch words and the SQ_VS_CONST
// word): ShaderInterpreter::ConstantSource over the snapshot's arrays (CoverInputs::interp_regs null = the typed arrays); no register file is read.
namespace rex::graphics::vulkan {
namespace {
uint64_t g_cover_typed = 0;
}  // namespace

void OwnedFrontCoverInputs(const OwnedRecordState& s, CoverInputs& in) {
  static_assert(offsetof(OwnedRecordState, ps_float) == offsetof(OwnedRecordState, vs_float) + 1024 * sizeof(uint32_t), "banks contiguous");
  in.interp_regs = nullptr;  // (the interpreter reads the arrays below through its constant source - no register file)
  in.initiator = s.draw_initiator;
  in.sc_mode = s.sc_mode_cntl;
  in.reset_indx = s.reset_indx;
  in.surface_info = s.surface_info;
  in.color_control = s.color_control;
  in.window_offset = s.window_offset;
  in.vte = s.vte_cntl;
  in.clip = s.clip_cntl;
  in.su_vtx_cntl = s.su_vtx_cntl;
  in.indx_offset = s.indx_offset;
  in.min_indx = s.min_vtx_indx;
  in.max_indx = s.max_vtx_indx;
  in.vport[0] = s.vport[0];  // x scale
  in.vport[1] = s.vport[1];  // x offset
  in.vport[2] = s.vport[2];  // y scale
  in.vport[3] = s.vport[3];  // y offset
  draw_util::GetScissor(s.window_scissor_tl, s.window_scissor_br, s.window_offset, s.screen_scissor_tl, s.screen_scissor_br, s.surface_info.surface_pitch,
                        in.scissor);
  in.vs_const = s.vs_const.value;
  in.floats = s.vs_float;
  in.bool_loop = s.bool_loop;
  in.fetch = s.fetch;
  ++g_cover_typed;
}
uint64_t OwnedFrontCoverTypedCount() { return g_cover_typed; }

}  // namespace rex::graphics::vulkan
