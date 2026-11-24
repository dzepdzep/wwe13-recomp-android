#pragma once
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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <rex/graphics/pipeline/shader/spirv_builder.h>
#include <rex/graphics/pipeline/shader/translator.h>
#include <rex/graphics/xenos.h>
#include <rex/ui/vulkan/device.h>

namespace rex::graphics {

class SpirvShaderTranslator : public ShaderTranslator {
 public:
  // wwe2k14 patch (Session 66, native_ext): see native_ext().
  enum NativeExtBits : uint32_t {
    kNativeExtEpilogueIdle = UINT32_C(1) << 0,
    kNativeExtFma = UINT32_C(1) << 1,
    // wwe2k14 patch (Session 72, gpu_native_shader_swap): the title's replacement module instead of the translation's
    // (vulkan/native/shader_swap.h); the translator itself ignores it.
    kNativeExtSwap = UINT32_C(1) << 2,
    // wwe2k14 patch (Session 82, gpu_native_owned_alpha_learn): the replacement module with an alpha probe - the fragments
    // whose 8-bit alpha is 1 are discarded before the colour store (shader_swap::Build); a separate translation, used
    // only by a probe draw (colour writes off) inside an occlusion query; the translator itself ignores it.
    kNativeExtAlphaProbe = UINT32_C(1) << 3,
    // wwe2k14 patch (Session 87, stage 4, gpu_native_vs_swap): a VERTEX translation whose module is the title's own vertex
    // shader (vulkan/native/shader_swap.h, BuildVertex) instead of the translator's code; the translator itself ignores it.
    kNativeExtVsSwap = UINT32_C(1) << 4,
    // wwe2k14 patch (Session 87, stage 4, gpu_native_swap_skip_translation(_pixel)): this replacement translation (kNativeExtSwap /
    // kNativeExtVsSwap) has its OWN bindings - the images and samplers its SPIR-V declares, not the emulated translation's - and
    // was made without the translator. Part of the translation's key, so the identity of the bindings is the identity of the
    // module; set by the command processor for the draws whose descriptors were made for them.
    kNativeExtOwnBindings = UINT32_C(1) << 5,
    // wwe2k14 patch (Session 95, B2, gpu_native_shader_ablate): the pixel replacement is the flat ablation module (diagnostic; part of the translation key).
    kNativeExtAblate = UINT32_C(1) << 6,
    // wwe2k14 patch (Session 95, item D, gpu_native_vertex_mirror; S0 of out\s95\android_structural.md renamed it from kNativeExtVsMirror, the same bit): a VERTEX translation (with
    // kNativeExtVsSwap and kNativeExtOwnBindings) whose module is our vertex-input (_vi) variant of the replacement: the vertex comes from a vertex buffer, not from the shared memory. Part of the
    // translation's key. Lane C: without kNativeExtVsSwap it is a translated module that reads vertex attributes, by Shader::VertexInputForm(): the vfetches read inputs, r0.x is gl_VertexIndex as
    // it is (the draw guarantees the identity: out\s95\android_structural.md section 3.2) and no shared memory is declared.
    kNativeExtVsInput = UINT32_C(1) << 7,
    // wwe2k14 patch (Session 95, r3gpu, gpu_native_skin_once): a VERTEX translation (with kNativeExtVsSwap and kNativeExtOwnBindings) whose module is the replacement's skin-once (_so) variant: the
    // vertex is a row of skin once's output. Its own bit: a shader can have both a _vi and a _so module (gpu_native_skin_with_input). Part of the translation's key.
    kNativeExtVsSkinOnce = UINT32_C(1) << 8,
  };
  union Modification {
    // If anything in this is structure is changed in a way not compatible with
    // the previous layout, invalidate the pipeline storages by increasing this
    // version number (0xYYYYMMDD)!
    // TODO(Triang3l): Change to 0xYYYYMMDD once it's out of the rapid
    // prototyping stage (easier to do small granular updates with an
    // incremental counter).
    static constexpr uint32_t kVersion = 12;

    enum class DepthStencilMode : uint32_t {
      kNoModifiers,
      // Early fragment tests - enable if alpha test and alpha to coverage are
      // disabled; ignored if anything in the shader blocks early Z writing.
      kEarlyHint,
      // Similar to DXBC float24 conversion modes, remapping the guest [0, 1]
      // range to the host [0, 0.5] range.
      kFloat24Truncating,
      kFloat24Rounding,
    };

    struct {
      // uint32_t 0.
      // Interpolators written by the vertex shader and needed by the pixel
      // shader.
      uint32_t interpolator_mask : xenos::kMaxInterpolators;
      // For HostVertexShaderType kPointListAsTriangleStrip, whether to output
      // the point coordinates.
      // For other HostVertexShaderTypes (though truly reachable only for
      // kVertex), whether to output the point size.
      uint32_t output_point_parameters : 1;
      // Dynamically indexable register count from SQ_PROGRAM_CNTL.
      uint32_t dynamic_addressable_register_count : 8;
      uint32_t user_clip_plane_count : 3;
      uint32_t user_clip_plane_cull : 1;
      // Whether vertex killing with the "and" operator is used, and one more
      // cull distance needs to be written.
      uint32_t vertex_kill_and : 1;
      // PA_CL_CLIP_CNTL::ps_ucp_mode for point primitives.
      uint32_t point_ps_ucp_mode : 2;
      // uint32_t 1.
      // Pipeline stage and input configuration.
      Shader::HostVertexShaderType host_vertex_shader_type : Shader::kHostVertexShaderTypeBitCount;
      // For domain host vertex shader types only, xenos::TessellationMode.
      uint32_t tessellation_mode : 2;
      // wwe2k14 patch (Session 23, MERGE): vulkan_repeat_draw_merge. Non-zero (kVertex host shaders only): the
      // float constants are read at index + gl_InstanceIndex * this (in vec4s), from a 256-vec4 block - instance i
      // of a merged draw uses the i-th of consecutive constant uploads this far apart. 0 = the normal shader.
      uint32_t instanced_float_constants_stride : 7;
      // wwe2k14 patch (Session 24, OBJMV): kVertex host shaders only: gl_Position is captured with transform
      // feedback (buffer 0, offset 0, stride 16; Xfb execution mode). Decorations only, the code is the same.
      uint32_t xfb_position : 1;
      // wwe2k14 patch (Session 66, B2.1 gpu_native_vertex_endian_spec): the vertex fetch endian modes baked into the
      // translation instead of read from the fetch constants. 0 = the normal shader (the endian word is read and
      // branched on; vertex_fetch_endians is zero). 1 = vertex_fetch_endians holds the xenos::Endian (2 bits) of the
      // fetch constants of Shader::vertex_bindings(), in that order, for up to 8 of them (a shader with more is never
      // specialised); a fetch whose constant is not among the bindings still reads the word. Zero bits keep every
      // record stored before these fields existed valid (kVersion unchanged).
      // (Session 66, VS mul: 14 bits, 7 bindings - the game's vertex shaders have at most 7 fetches; the 2 bits that
      // were the 8th binding's are native_mul_mode now, so a stored record of a shader with 8 bindings would alias -
      // there is none in the game.)
      uint32_t vertex_fetch_endian_spec : 1;
      uint32_t vertex_fetch_endians : 14;
      // wwe2k14 patch (Session 66, VS mul, gpu_native_vs_mul_mode): NATIVE's translation of a kVertex host vertex
      // shader only (the emulated path never sets it): the Xenos multiply rule as for the pixel field native_mul_mode
      // (0 exact, 1 left out where both operands are proven finite, 2 plus `p != p ? 0 : p` on the rest, 3 left out
      // everywhere). native_mul_fetch_finite: vertex fetch results of float formats (and vertex texture fetch results)
      // count as finite for the proof (otherwise they are tainted; fetches of normalized / packed / integer formats are
      // always finite).
      uint32_t native_mul_mode : 2;
      uint32_t native_mul_fetch_finite : 1;
    } vertex;
    struct PixelShaderModification {
      // uint32_t 0.
      // Interpolators written by the vertex shader and needed by the pixel
      // shader.
      uint32_t interpolator_mask : xenos::kMaxInterpolators;
      uint32_t interpolators_centroid : xenos::kMaxInterpolators;
      // uint32_t 1.
      // Dynamically indexable register count from SQ_PROGRAM_CNTL.
      uint32_t dynamic_addressable_register_count : 8;
      uint32_t param_gen_enable : 1;
      uint32_t param_gen_interpolator : 4;
      // If param_gen_enable is set, this must be set for point primitives, and
      // must not be set for other primitive types - enables the point sprite
      // coordinates input, and also effects the flag bits in PsParamGen.
      uint32_t param_gen_point : 1;
      // For host render targets - depth / stencil output mode.
      DepthStencilMode depth_stencil_mode : 3;
      // wwe2k14 patch (Session 24, DLSS): texture LOD bias of -N / 4 for the scene draws of a frame DLSS upscales
      // (textures sampled for the output resolution, as DLSS asks); 0 = the normal shader.
      uint32_t dlss_lod_bias_quarters : 4;
      // wwe2k14 patch (Session 38, NATIVE): translated for host render targets by the native renderer's translator
      // (VulkanPipelineCache::GetNativePixelShaderTranslation) although the emulated renderer uses the fragment
      // shader interlock path: keeps the two translations of the same modification apart.
      uint32_t native_host_render_targets : 1;
      // wwe2k14 patch (Session 43, NATIVE QUANTIZE): native translations only - the colour outputs rounded as the
      // interlock shader packs them, for 8-bit / 16-bit native targets that then hold the packed values exactly:
      // 0 none, 1 k_8_8_8_8 (clamp 0-1, x 255 + 0.5 truncated, / 255), 2 k_16_16 / k_16_16_16_16 (NaN 0, clamp
      // -32..32, x 32767 / 32, rounded away from zero, / 32767: an snorm16 target).
      uint32_t native_color_quantize : 2;
      // wwe2k14 patch (Session 54, NATIVE draw sizes): native translations only - drawn into a truth image's draw
      // twin (its pixel position in the twin's pixels: the translator's SetPixelPositionScale).
      uint32_t native_twin_position : 1;
      // wwe2k14 patch (Session 66, PS spec 1, gpu_native_ps_texture_spec): bit i = the texture binding with index i
      // (Shader::TextureBinding::binding_index: one per fetch constant, in the order of first use) is known to have
      // every swizzled sign unsigned (texture_swizzled_signs byte 0 - no signed sample, no sign / gamma / bias switch).
      // Bindings 4 and up are never specialised. 0 = the normal shader (the byte is read and switched on).
      uint32_t tex_unsigned_mask : 4;
      // wwe2k14 patch (Session 66, PS spec 2, gpu_native_ps_mul_mode): NATIVE translations only (native_host_render_targets
      // set; ignored otherwise): the Xenos multiply rule (0 x anything = +0) - 0 = as before; 1 = left out where both
      // operands are proven finite; 2 = as 1, and `p != p ? 0 : p` on the rest; 3 = left out everywhere.
      uint32_t native_mul_mode : 2;
      // wwe2k14 patch (Session 66, PS spec 3, gpu_native_ps_implicit_lod): NATIVE translations only: a texture fetch
      // with computed LOD from the quad's derivatives (no register gradients; 1D, 2D and cube) samples with the
      // hardware's implicit LOD and Bias = the instruction / fetch constant LOD, instead of explicit gradients built
      // from coarse derivatives scaled by 2^LOD. 0 = as before.
      uint32_t native_implicit_lod : 1;
    } pixel;
    uint64_t value = 0;

    explicit Modification(uint64_t modification_value = 0) : value(modification_value) {
      static_assert_size(*this, sizeof(value));
    }

    uint32_t GetVertexClipDistanceCount() const {
      return vertex.user_clip_plane_cull ? 0 : vertex.user_clip_plane_count;
    }
    uint32_t GetVertexCullDistanceCount() const {
      return (vertex.user_clip_plane_cull ? vertex.user_clip_plane_count : 0) +
             vertex.vertex_kill_and;
    }
  };

  enum : uint32_t {
    kSysFlag_VertexIndexLoad_Shift,
    kSysFlag_ComputeOrPrimitiveVertexIndexLoad_Shift,
    kSysFlag_ComputeOrPrimitiveVertexIndexLoad32Bit_Shift,
    kSysFlag_ComputeOrPrimitiveVertexIndexReset_Shift,
    // For memexport compute fallback only.
    kSysFlag_ComputeMemExportPatchIndexFromInvocation_Shift,
    // For memexport compute fallback only.
    kSysFlag_ComputeMemExportPatchIndexInRegister1_Shift,
    // For memexport compute fallback only.
    kSysFlag_ComputeMemExportTriangleCPIndexed_Shift,
    // For memexport compute fallback only.
    kSysFlag_ComputeMemExportQuadCPIndexed_Shift,
    kSysFlag_XYDividedByW_Shift,
    kSysFlag_ZDividedByW_Shift,
    kSysFlag_WNotReciprocal_Shift,
    kSysFlag_PrimitivePolygonal_Shift,
    kSysFlag_PrimitiveLine_Shift,
    kSysFlag_MsaaSamples_Shift,
    kSysFlag_DepthFloat24_Shift = kSysFlag_MsaaSamples_Shift + xenos::kMsaaSamplesBits,
    kSysFlag_AlphaPassIfLess_Shift,
    kSysFlag_AlphaPassIfEqual_Shift,
    kSysFlag_AlphaPassIfGreater_Shift,
    kSysFlag_ConvertColor0ToGamma_Shift,
    kSysFlag_ConvertColor1ToGamma_Shift,
    kSysFlag_ConvertColor2ToGamma_Shift,
    kSysFlag_ConvertColor3ToGamma_Shift,

    kSysFlag_FSIDepthStencil_Shift,
    kSysFlag_FSIDepthPassIfLess_Shift,
    kSysFlag_FSIDepthPassIfEqual_Shift,
    kSysFlag_FSIDepthPassIfGreater_Shift,
    // 1 to write new depth to the depth buffer, 0 to keep the old one if the
    // depth test passes.
    kSysFlag_FSIDepthWrite_Shift,
    kSysFlag_FSIStencilTest_Shift,
    // If the depth / stencil test has failed, but resulted in a stencil value
    // that is different than the one currently in the depth buffer, write it
    // anyway and don't run the rest of the shader (to check if the sample may
    // be discarded some way) - use when alpha test and alpha to coverage are
    // disabled. Ignored by the shader if not applicable to it (like if it has
    // kill instructions or writes the depth output).
    // TODO(Triang3l): Investigate replacement with an alpha-to-mask flag,
    // checking `(flags & (alpha test | alpha to mask)) == (always | disabled)`,
    // taking into account the potential relation with occlusion queries (but
    // should be safe at least temporarily).
    kSysFlag_FSIDepthStencilEarlyWrite_Shift,

    kSysFlag_Count,

    // For HostVertexShaderType kVertex, if fullDrawIndexUint32 is not
    // supported (ignored otherwise), whether to fetch the index manually
    // (32-bit only - 16-bit indices are always fetched via the Vulkan index
    // buffer).
    kSysFlag_VertexIndexLoad = 1u << kSysFlag_VertexIndexLoad_Shift,
    // For HostVertexShaderTypes kMemExportCompute, kPointListAsTriangleStrip,
    // kRectangleListAsTriangleStrip, whether the vertex index needs to be
    // loaded from the index buffer (rather than using autogenerated indices),
    // and whether it's 32-bit. This is separate from kSysFlag_VertexIndexLoad
    // because the same system constants may be used for the memexporting
    // compute shader and the vertex shader for the same draw, but
    // kSysFlag_VertexIndexLoad may be not needed.
    kSysFlag_ComputeOrPrimitiveVertexIndexLoad =
        1u << kSysFlag_ComputeOrPrimitiveVertexIndexLoad_Shift,
    kSysFlag_ComputeOrPrimitiveVertexIndexLoad32Bit =
        1u << kSysFlag_ComputeOrPrimitiveVertexIndexLoad32Bit_Shift,
    kSysFlag_ComputeOrPrimitiveVertexIndexReset =
        1u << kSysFlag_ComputeOrPrimitiveVertexIndexReset_Shift,
    kSysFlag_ComputeMemExportPatchIndexFromInvocation =
        1u << kSysFlag_ComputeMemExportPatchIndexFromInvocation_Shift,
    kSysFlag_ComputeMemExportPatchIndexInRegister1 =
        1u << kSysFlag_ComputeMemExportPatchIndexInRegister1_Shift,
    kSysFlag_ComputeMemExportTriangleCPIndexed =
        1u << kSysFlag_ComputeMemExportTriangleCPIndexed_Shift,
    kSysFlag_ComputeMemExportQuadCPIndexed = 1u << kSysFlag_ComputeMemExportQuadCPIndexed_Shift,
    kSysFlag_XYDividedByW = 1u << kSysFlag_XYDividedByW_Shift,
    kSysFlag_ZDividedByW = 1u << kSysFlag_ZDividedByW_Shift,
    kSysFlag_WNotReciprocal = 1u << kSysFlag_WNotReciprocal_Shift,
    kSysFlag_PrimitivePolygonal = 1u << kSysFlag_PrimitivePolygonal_Shift,
    kSysFlag_PrimitiveLine = 1u << kSysFlag_PrimitiveLine_Shift,
    kSysFlag_DepthFloat24 = 1u << kSysFlag_DepthFloat24_Shift,
    kSysFlag_AlphaPassIfLess = 1u << kSysFlag_AlphaPassIfLess_Shift,
    kSysFlag_AlphaPassIfEqual = 1u << kSysFlag_AlphaPassIfEqual_Shift,
    kSysFlag_AlphaPassIfGreater = 1u << kSysFlag_AlphaPassIfGreater_Shift,
    kSysFlag_ConvertColor0ToGamma = 1u << kSysFlag_ConvertColor0ToGamma_Shift,
    kSysFlag_ConvertColor1ToGamma = 1u << kSysFlag_ConvertColor1ToGamma_Shift,
    kSysFlag_ConvertColor2ToGamma = 1u << kSysFlag_ConvertColor2ToGamma_Shift,
    kSysFlag_ConvertColor3ToGamma = 1u << kSysFlag_ConvertColor3ToGamma_Shift,
    kSysFlag_FSIDepthStencil = 1u << kSysFlag_FSIDepthStencil_Shift,
    kSysFlag_FSIDepthPassIfLess = 1u << kSysFlag_FSIDepthPassIfLess_Shift,
    kSysFlag_FSIDepthPassIfEqual = 1u << kSysFlag_FSIDepthPassIfEqual_Shift,
    kSysFlag_FSIDepthPassIfGreater = 1u << kSysFlag_FSIDepthPassIfGreater_Shift,
    kSysFlag_FSIDepthWrite = 1u << kSysFlag_FSIDepthWrite_Shift,
    kSysFlag_FSIStencilTest = 1u << kSysFlag_FSIStencilTest_Shift,
    kSysFlag_FSIDepthStencilEarlyWrite = 1u << kSysFlag_FSIDepthStencilEarlyWrite_Shift,
  };
  static_assert(kSysFlag_Count <= 32, "Too many flags in the system constants");

  // For Vulkan vertex memexport compute fallback dispatches.
  static constexpr uint32_t kMemExportComputeDispatchX = 65535;
  static constexpr uint32_t kMemExportComputeDispatchY = 65535;

  // IF SYSTEM CONSTANTS ARE CHANGED OR ADDED, THE FOLLOWING MUST BE UPDATED:
  // - SystemConstantIndex enum.
  // - Structure members in BeginTranslation.
  //
  // Using the std140 layout - vec2 must be aligned to 8 bytes, vec3 and vec4 to
  // 16 bytes.
  struct SystemConstants {
    uint32_t flags;
    uint32_t vertex_index_load_address;
    xenos::Endian vertex_index_endian;
    // For non-indexed line loop draws, host vertex index that should map to
    // guest vertex 0.
    uint32_t line_loop_closing_index;
    int32_t vertex_base_index;
    uint32_t vertex_index_reset;
    // Number of guest vertices processed by the memexport compute dispatch.
    uint32_t compute_memexport_vertex_count;
    // One bit per texture fetch constant indicating that it references a
    // draw-resolution-scaled texture.
    // wwe2k14 patch (Session 95, S0): moved here, into the free word at 28, so that user_clip_planes and everything after it obey std140 and the block needs neither scalarBlockLayout nor
    // uniformBufferStandardLayout. Our own vertex shaders read this block by offset: src\scene\xenos_vs.glsl (the asserts below hold the offsets).
    uint32_t textures_resolution_scaled;
    float user_clip_planes[6][4];

    float ndc_scale[3];
    float point_vertex_diameter_min;

    float ndc_offset[3];
    float point_vertex_diameter_max;

    float point_constant_diameter[2];
    // Diameter in guest screen coordinates > radius (0.5 * diameter) in the NDC
    // for the host viewport.
    float point_screen_diameter_to_ndc_radius[2];

    // Each byte contains post-swizzle TextureSign values for each of the needed
    // components of each of the 32 used texture fetch constants.
    uint32_t texture_swizzled_signs[8];

    // If the imageViewFormatSwizzle portability subset is not supported, the
    // component swizzle (taking both guest and host swizzles into account) to
    // apply to the result directly in the shader code. In each uint32_t,
    // swizzles for 2 texture fetch constants (in bits 0:11 and 12:23).
    uint32_t texture_swizzles[16];

    float alpha_test_reference;
    // Bits 0:7 - alpha-to-coverage dither offsets from RB_COLORCONTROL bits
    // 24:31, bit 8 - whether alpha-to-coverage is enabled.
    uint32_t alpha_to_mask;
    uint32_t edram_32bpp_tile_pitch_dwords_scaled;
    uint32_t edram_depth_base_dwords_scaled;

    float color_exp_bias[4];

    float edram_poly_offset_front_scale;
    float edram_poly_offset_back_scale;
    float edram_poly_offset_front_offset;
    float edram_poly_offset_back_offset;

    union {
      struct {
        uint32_t edram_stencil_front_reference_masks;
        uint32_t edram_stencil_front_func_ops;

        uint32_t edram_stencil_back_reference_masks;
        uint32_t edram_stencil_back_func_ops;
      };
      struct {
        uint32_t edram_stencil_front[2];
        uint32_t edram_stencil_back[2];
      };
    };

    uint32_t edram_rt_base_dwords_scaled[4];

    // RT format combined with RenderTargetCache::kPSIColorFormatFlag values
    // (pass via RenderTargetCache::AddPSIColorFormatFlags).
    uint32_t edram_rt_format_flags[4];

    // Render target blending options - RB_BLENDCONTROL, with only the relevant
    // options (factors and operations - AND 0x1FFF1FFF). If 0x00010001
    // (1 * src + 0 * dst), blending is disabled for the render target.
    uint32_t edram_rt_blend_factors_ops[4];

    // Format info - mask to apply to the old packed RT data, and to apply as
    // inverted to the new packed data, before storing (more or less the inverse
    // of the write mask packed like render target channels). This can be used
    // to bypass unpacking if blending is not used. If 0 and not blending,
    // reading the old data from the EDRAM buffer is not required.
    uint32_t edram_rt_keep_mask[4][2];

    // Format info - values to clamp the color to before blending or storing.
    // Low color, low alpha, high color, high alpha.
    float edram_rt_clamp[4][4];

    // The constant blend factor for the respective modes.
    float edram_blend_constant[4];

    // Tessellation helper shader constants.
    uint32_t vertex_index_min;
    uint32_t vertex_index_max;
    float tessellation_factor_range_min;
    float tessellation_factor_range_max;

    // wwe2k14 patch (Session 93, the per-draw storage window): the byte offset of the bound shared memory window,
    // in dwords. Subtracted from every shared memory address by the shaders (always 0 with the window mode off).
    // LAST, at offset 512: the SPIR-V block's member index is the enum value (kSystemConstantSharedMemoryWindowBase).
    uint32_t shared_memory_window_base_dwords;
  };
  // wwe2k14 patch (Session 95, S0): the std140 offsets, also read by src\scene\xenos_vs.glsl and xenos_ps.glsl.
  static_assert(offsetof(SystemConstants, textures_resolution_scaled) == 28);
  static_assert(offsetof(SystemConstants, user_clip_planes) == 32);
  static_assert(offsetof(SystemConstants, ndc_scale) == 128);
  static_assert(offsetof(SystemConstants, point_vertex_diameter_min) == 140);
  static_assert(offsetof(SystemConstants, ndc_offset) == 144);
  static_assert(offsetof(SystemConstants, point_vertex_diameter_max) == 156);
  static_assert(offsetof(SystemConstants, point_constant_diameter) == 160);
  static_assert(offsetof(SystemConstants, point_screen_diameter_to_ndc_radius) == 168);
  static_assert(offsetof(SystemConstants, texture_swizzled_signs) == 176);
  static_assert(offsetof(SystemConstants, texture_swizzles) == 208);
  static_assert(offsetof(SystemConstants, alpha_test_reference) == 272);
  static_assert(offsetof(SystemConstants, color_exp_bias) == 288);
  static_assert(offsetof(SystemConstants, vertex_index_min) == 496);
  static_assert(offsetof(SystemConstants, shared_memory_window_base_dwords) == 512);
  // wwe2k14 patch (Session 95, S0): the bytes the block is bound with: the whole 16-byte row of the last member, so the window base at 512 stays inside the range however wide a driver loads it.
  static constexpr uint32_t kSystemConstantsBoundBytes = 528;
  static_assert(sizeof(SystemConstants) <= kSystemConstantsBoundBytes && kSystemConstantsBoundBytes % 16 == 0);

  enum ConstantBuffer : uint32_t {
    kConstantBufferSystem,
    kConstantBufferFloatVertex,
    kConstantBufferFloatPixel,
    kConstantBufferBoolLoop,
    kConstantBufferFetch,

    kConstantBufferCount,
  };

  // The minimum limit for maxPerStageDescriptorStorageBuffers is 4, and for
  // maxStorageBufferRange it's 128 MB. These are the values of those limits on
  // Arm Mali as of November 2020. Xenia needs 512 MB shared memory to be bound,
  // therefore SSBOs must only be used for shared memory - all other storage
  // resources must be images or texel buffers.
  enum DescriptorSet : uint32_t {
    // According to the "Pipeline Layout Compatibility" section of the Vulkan
    // specification:
    // "Two pipeline layouts are defined to be "compatible for set N" if they
    //  were created with identically defined descriptor set layouts for sets
    //  zero through N, and if they were created with identical push constant
    //  ranges."
    // "Place the least frequently changing descriptor sets near the start of
    //  the pipeline layout, and place the descriptor sets representing the most
    //  frequently changing resources near the end. When pipelines are switched,
    //  only the descriptor set bindings that have been invalidated will need to
    //  be updated and the remainder of the descriptor set bindings will remain
    //  in place."
    // This is partially the reverse of the Direct3D 12's rule of placing the
    // most frequently changed descriptor sets in the beginning. Here all
    // descriptor sets with an immutable layout are placed first, in reverse
    // frequency of changing, and sets that may be different for different
    // pipeline states last.

    // Always the same descriptor set layouts for all pipeline layouts:

    // Never changed.
    kDescriptorSetSharedMemoryAndEdram,
    // Changed in case of changes in the data.
    kDescriptorSetConstants,

    // Mutable part of the pipeline layout:
    kDescriptorSetMutableLayoutsStart,

    // Rarely used at all, but may be changed at an unpredictable rate when
    // vertex textures are used (for example, for bones of an object, which may
    // consist of multiple draw commands with different materials).
    kDescriptorSetTexturesVertex = kDescriptorSetMutableLayoutsStart,
    // Per-material textures.
    kDescriptorSetTexturesPixel,

    kDescriptorSetCount,
  };
  static_assert(kDescriptorSetCount <= 4,
                "The number of descriptor sets used by translated shaders must be within "
                "the minimum Vulkan maxBoundDescriptorSets requirement of 4, which is "
                "the limit on most GPUs used in Android devices - Arm Mali, Imagination "
                "PowerVR, Qualcomm Adreno 6xx and older, as well as on old PC Nvidia "
                "drivers");

  // "Xenia Emulator Microcode Translator".
  // https://github.com/KhronosGroup/SPIRV-Headers/blob/c43a43c7cc3af55910b9bec2a71e3e8a622443cf/include/spirv/spir-v.xml#L79
  static constexpr uint32_t kSpirvMagicToolId = 26;

  struct Features {
    explicit Features(const ui::vulkan::VulkanDevice* vulkan_device);
    explicit Features(bool all = false);

    unsigned int spirv_version;

    uint32_t max_storage_buffer_range;

    bool full_draw_index_uint32;

    bool vertex_pipeline_stores_and_atomics;
    bool fragment_stores_and_atomics;

    bool clip_distance;
    bool cull_distance;

    bool image_view_format_swizzle;

    bool signed_zero_inf_nan_preserve_float32;
    bool denorm_flush_to_zero_float32;
    bool rounding_mode_rte_float32;

    bool fragment_shader_sample_interlock;

    bool demote_to_helper_invocation;
    bool sample_rate_shading;

    // wwe2k14 patch (Session 93, the per-draw storage window): the shared memory is ONE storage buffer binding that
    // is a window of the 512 MiB, whose base the host passes in a system constant (GPUs with a small
    // maxStorageBufferRange). Set by the command processor from its latched decision, never from the device.
    bool shared_memory_window = false;
  };

  SpirvShaderTranslator(const Features& features, bool native_2x_msaa_with_attachments,
                        bool native_2x_msaa_no_attachments, bool edram_fragment_shader_interlock,
                        uint32_t draw_resolution_scale_x = 1, uint32_t draw_resolution_scale_y = 1)
      : features_(features),
        native_2x_msaa_with_attachments_(native_2x_msaa_with_attachments),
        native_2x_msaa_no_attachments_(native_2x_msaa_no_attachments),
        edram_fragment_shader_interlock_(edram_fragment_shader_interlock),
        draw_resolution_scale_x_(draw_resolution_scale_x ? draw_resolution_scale_x : 1),
        draw_resolution_scale_y_(draw_resolution_scale_y ? draw_resolution_scale_y : 1) {}

  // wwe2k14 patch (Session 54, NATIVE draw sizes): host pixels per guest pixel for the pixel position a pixel shader
  // reads (param_gen) when it differs from the whole draw resolution scale - the native renderer's truth passes draw
  // at a fractional size (gpu_native_truth_draw_height). 0 = the draw resolution scale.
  void SetPixelPositionScale(float x, float y) {
    pixel_position_scale_x_ = x;
    pixel_position_scale_y_ = y;
  }

  uint64_t GetDefaultVertexShaderModification(
      uint32_t dynamic_addressable_register_count,
      Shader::HostVertexShaderType host_vertex_shader_type =
          Shader::HostVertexShaderType::kVertex) const override;
  uint64_t GetDefaultPixelShaderModification(
      uint32_t dynamic_addressable_register_count) const override;

  static constexpr uint32_t GetSharedMemoryStorageBufferCountLog2(
      uint32_t max_storage_buffer_range) {
    if (max_storage_buffer_range >= 512 * 1024 * 1024) {
      return 0;
    }
    if (max_storage_buffer_range >= 256 * 1024 * 1024) {
      return 1;
    }
    return 2;
  }
  uint32_t GetSharedMemoryStorageBufferCountLog2() const {
    if (features_.shared_memory_window) {
      return 0;  // wwe2k14 patch (Session 93): one binding, a window
    }
    return GetSharedMemoryStorageBufferCountLog2(features_.max_storage_buffer_range);
  }

  // Creates a special fragment shader without color outputs - this resets the
  // state of the translator.
  std::vector<uint8_t> CreateDepthOnlyFragmentShader(
      Modification::DepthStencilMode depth_stencil_mode =
          Modification::DepthStencilMode::kNoModifiers);

  // Common functions useful not only for the translator, but also for EDRAM
  // emulation via conventional render targets.

  // Converts the color value externally clamped to [0, 31.875] to 7e3 floating
  // point, with zeros in bits 10:31, rounding to the nearest even.
  static spv::Id PreClampedFloat32To7e3(SpirvBuilder& builder, spv::Id f32_scalar,
                                        spv::Id ext_inst_glsl_std_450);
  // Same as PreClampedFloat32To7e3, but clamps the input to [0, 31.875].
  static spv::Id UnclampedFloat32To7e3(SpirvBuilder& builder, spv::Id f32_scalar,
                                       spv::Id ext_inst_glsl_std_450);
  // Converts the 7e3 number in bits [f10_shift, f10_shift + 10) to a 32-bit
  // float.
  static spv::Id Float7e3To32(SpirvBuilder& builder, spv::Id f10_uint_scalar, uint32_t f10_shift,
                              bool result_as_uint, spv::Id ext_inst_glsl_std_450);
  // Converts the depth value externally clamped to the representable [0, 2)
  // range to 20e4 floating point, with zeros in bits 24:31, rounding to the
  // nearest even or towards zero. If remap_from_0_to_0_5 is true, it's assumed
  // that 0...1 is pre-remapped to 0...0.5 in the input.
  static spv::Id PreClampedDepthTo20e4(SpirvBuilder& builder, spv::Id f32_scalar,
                                       bool round_to_nearest_even, bool remap_from_0_to_0_5,
                                       spv::Id ext_inst_glsl_std_450);
  // Converts the 20e4 number in bits [f24_shift, f24_shift + 10) to a 32-bit
  // float.
  static spv::Id Depth20e4To32(SpirvBuilder& builder, spv::Id f24_uint_scalar, uint32_t f24_shift,
                               bool remap_to_0_to_0_5, bool result_as_uint,
                               spv::Id ext_inst_glsl_std_450);

 protected:
  void Reset() override;

  uint32_t GetModificationRegisterCount() const override;

  void StartTranslation() override;

  std::vector<uint8_t> CompleteTranslation() override;

  void PostTranslation() override;

  // wwe2k14 patch (Session 92, M6 translation store): see ShaderTranslator::StoreConfigFingerprint.
  uint64_t StoreConfigFingerprint() const override;
  void StoreDescribeTranslation(bool& native, float& scale_x, float& scale_y) const override;
  void StoreExportExtras(std::vector<uint8_t>& out) const override;
  bool StoreImportExtras(const std::vector<uint8_t>& in) override;

  void ProcessLabel(uint32_t cf_index) override;

  void ProcessExecInstructionBegin(const ParsedExecInstruction& instr) override;
  void ProcessExecInstructionEnd(const ParsedExecInstruction& instr) override;
  void ProcessLoopStartInstruction(const ParsedLoopStartInstruction& instr) override;
  void ProcessLoopEndInstruction(const ParsedLoopEndInstruction& instr) override;
  void ProcessJumpInstruction(const ParsedJumpInstruction& instr) override;
  void ProcessAllocInstruction(const ParsedAllocInstruction& instr, uint8_t export_eM) override;

  void ProcessVertexFetchInstruction(const ParsedVertexFetchInstruction& instr) override;
  void ProcessTextureFetchInstruction(const ParsedTextureFetchInstruction& instr) override;
  void ProcessAluInstruction(const ParsedAluInstruction& instr,
                             uint8_t memexport_eM_potentially_written_before) override;

 private:
  struct TextureBinding {
    uint32_t fetch_constant;
    // Stacked and 3D are separate TextureBindings.
    xenos::FetchOpDimension dimension;
    bool is_signed;

    spv::Id variable;
  };

  struct SamplerBinding {
    uint32_t fetch_constant;
    xenos::TextureFilter mag_filter;
    xenos::TextureFilter min_filter;
    xenos::TextureFilter mip_filter;
    xenos::AnisoFilter aniso_filter;

    spv::Id variable;
  };

  // Builder helpers.
  spv::Id SpirvSmearScalarResultOrConstant(spv::Id scalar, spv::Id vector_type);

  // wwe2k14 patch (Session 92, M6): PostTranslation's shader-level step (the first translation of a shader sets its texture /
  // sampler bindings), with the lists given - the translator's own after a translation, the stored ones on a store hit.
  void SetShaderBindingsFromTranslation(const std::vector<TextureBinding>& textures,
                                        const std::vector<SamplerBinding>& samplers);

  Modification GetSpirvShaderModification() const {
    return Modification(current_translation().modification());
  }

  bool IsSpirvVertexShader() const {
    return is_vertex_shader() && !Shader::IsHostVertexShaderTypeDomain(
                                     GetSpirvShaderModification().vertex.host_vertex_shader_type);
  }
  bool IsSpirvTessEvalShader() const {
    return is_vertex_shader() && Shader::IsHostVertexShaderTypeDomain(
                                     GetSpirvShaderModification().vertex.host_vertex_shader_type);
  }
  bool IsSpirvComputeShader() const {
    return is_vertex_shader() && GetSpirvShaderModification().vertex.host_vertex_shader_type ==
                                     Shader::HostVertexShaderType::kMemExportCompute;
  }
  bool IsSpirvRectangleListVertexLoopEnabled() const {
    return IsSpirvVertexShader() && GetSpirvShaderModification().vertex.host_vertex_shader_type ==
                                        Shader::HostVertexShaderType::kRectangleListAsTriangleStrip;
  }

  bool IsExecutionModeEarlyFragmentTests() const {
    return is_pixel_shader() &&
           GetSpirvShaderModification().pixel.depth_stencil_mode ==
               Modification::DepthStencilMode::kEarlyHint &&
           !edram_fragment_shader_interlock_ && current_shader().implicit_early_z_write_allowed();
  }
  bool IsWritingFloat24Depth() const {
    assert_true(is_pixel_shader());
    if (edram_fragment_shader_interlock_) {
      return false;
    }
    Modification::DepthStencilMode depth_stencil_mode =
        GetSpirvShaderModification().pixel.depth_stencil_mode;
    return depth_stencil_mode == Modification::DepthStencilMode::kFloat24Truncating ||
           depth_stencil_mode == Modification::DepthStencilMode::kFloat24Rounding;
  }
  bool IsSampleRate() const {
    assert_true(is_pixel_shader());
    return IsWritingFloat24Depth() && !current_shader().writes_depth();
  }

  uint32_t GetModificationInterpolatorMask() const {
    Modification modification = GetSpirvShaderModification();
    return is_vertex_shader() ? modification.vertex.interpolator_mask
                              : modification.pixel.interpolator_mask;
  }

  // Returns UINT32_MAX if PsParamGen doesn't need to be written.
  uint32_t GetPsParamGenInterpolator() const;

  // Must be called before emitting any SPIR-V operations that must be in a
  // block in translator callbacks to ensure that if the last instruction added
  // was something like OpBranch - in this case, an unreachable block is
  // created.
  void EnsureBuildPointAvailable();

  void StartVertexOrTessEvalShaderBeforeMain();
  void StartVertexOrTessEvalShaderInMain();
  void CompleteVertexOrTessEvalShaderInMain();

  void StartFragmentShaderBeforeMain();
  void StartFragmentShaderInMain();
  void CompleteFragmentShaderInMain();

  // Updates the current flow control condition (to be called in the beginning
  // of exec and in jumps), closing the previous conditionals if needed.
  // However, if the condition is not different, the instruction-level predicate
  // conditional also won't be closed - this must be checked separately if
  // needed (for example, in jumps).
  void UpdateExecConditionals(ParsedExecInstruction::Type type, uint32_t bool_constant_index,
                              bool condition);
  // Opens or reopens the predicate check conditional for the instruction.
  // Should be called before processing a non-control-flow instruction.
  void UpdateInstructionPredication(bool predicated, bool condition);
  // Closes the instruction-level predicate conditional if it's open, useful if
  // a control flow instruction needs to do some code which needs to respect the
  // current exec conditional, but can't itself be predicated.
  void CloseInstructionPredication();
  // Closes conditionals opened by exec and instructions within them (but not by
  // labels) and updates the state accordingly.
  void CloseExecConditionals();

  spv::Id GetStorageAddressingIndex(InstructionStorageAddressingMode addressing_mode,
                                    uint32_t storage_index, bool is_float_constant = false);
  // Loads unswizzled operand without sign modifiers as float4.
  spv::Id LoadOperandStorage(const InstructionOperand& operand);
  spv::Id ApplyOperandModifiers(spv::Id operand_value, const InstructionOperand& original_operand,
                                bool invert_negate = false, bool force_absolute = false);
  // Returns the requested components, with the operand's swizzle applied, in a
  // condensed form, but without negation / absolute value modifiers. The
  // storage is float4, no matter what the component count of original_operand
  // is (the storage will be either r# or c#, but the instruction may be
  // scalar).
  spv::Id GetUnmodifiedOperandComponents(spv::Id operand_storage,
                                         const InstructionOperand& original_operand,
                                         uint32_t components);
  spv::Id GetOperandComponents(spv::Id operand_storage, const InstructionOperand& original_operand,
                               uint32_t components, bool invert_negate = false,
                               bool force_absolute = false) {
    return ApplyOperandModifiers(
        GetUnmodifiedOperandComponents(operand_storage, original_operand, components),
        original_operand, invert_negate, force_absolute);
  }
  // If components are identical, the same Id will be written to both outputs.
  void GetOperandScalarXY(spv::Id operand_storage, const InstructionOperand& original_operand,
                          spv::Id& a_out, spv::Id& b_out, bool invert_negate = false,
                          bool force_absolute = false);
  // Gets the absolute value of the loaded operand if it's not absolute already.
  spv::Id GetAbsoluteOperand(spv::Id operand_storage, const InstructionOperand& original_operand);
  // The type of the value must be a float vector consisting of
  // rex::bit_count(result.GetUsedResultComponents()) elements, or (to replicate
  // a scalar into all used components) float, or the value can be spv::NoResult
  // if there's no result to store (like constants only).
  // can_store_memexport_address is for safety, to allow only proper MADs with
  // a stream constant to write to eA.
  void StoreResult(const InstructionResult& result, spv::Id value,
                   bool can_store_memexport_address = false);

  // For Shader Model 3 multiplication (+-0 or denormal * anything = +0),
  // replaces the value with +0 if the minimum of the two operands is 0. This
  // must be called with absolute values of operands - use GetAbsoluteOperand!
  spv::Id ZeroIfAnyOperandIsZero(spv::Id value, spv::Id operand_0_abs, spv::Id operand_1_abs);
  // Conditionally discard the current fragment. Changes the build point.
  void KillPixel(spv::Id condition, uint8_t memexport_eM_potentially_written_before);
  // Return type is a rex::bit_count(result.GetUsedResultComponents())-component
  // float vector or a single float, depending on whether it's a reduction
  // instruction (check getTypeId of the result), or returns spv::NoResult if
  // nothing to store.
  spv::Id ProcessVectorAluOperation(const ParsedAluInstruction& instr,
                                    uint8_t memexport_eM_potentially_written_before,
                                    bool& predicate_written);
  // Returns a float value to write to the previous scalar register and to the
  // destination. If the return value is ps itself (in the retain_prev case),
  // returns spv::NoResult (handled as a special case, so if it's retain_prev,
  // but don't need to write to anywhere, no OpLoad(ps) will be done).
  spv::Id ProcessScalarAluOperation(const ParsedAluInstruction& instr,
                                    uint8_t memexport_eM_potentially_written_before,
                                    bool& predicate_written);

  // Perform endian swap of a uint scalar or vector.
  spv::Id EndianSwap32Uint(spv::Id value, spv::Id endian);
  // wwe2k14 patch (Session 66, B2.1): the same swap for a known xenos::Endian: the operations the branches of
  // EndianSwap32Uint run for that mode, without the comparisons and the conditional blocks.
  spv::Id EndianSwap32UintFixed(spv::Id value, xenos::Endian endian);
  // Perform endian swap of a uint4 vector.
  spv::Id EndianSwap128Uint4(spv::Id value, spv::Id endian);

  spv::Id LoadUint32FromSharedMemory(spv::Id address_dwords_int);
  // wwe2k14 patch (Session 95, Lane C): under vertex_input_, the dword `word` of the vertex in the binding of `fetch_constant`, as a uint, in the guest's byte order (the input is the raw bytes).
  spv::Id LoadVertexInputWord(uint32_t fetch_constant, int32_t word);
  // If `replace_mask` is provided, the bits specified in the mask will be
  // replaced with those from the value via OpAtomicAnd/Or.
  // Bits of `value` not in `replace_mask` will be ignored.
  void StoreUint32ToSharedMemory(spv::Id value, spv::Id address_dwords_int,
                                 spv::Id replace_mask = spv::NoResult);

  bool IsMemoryExportSupported() const {
    if (is_pixel_shader()) {
      return features_.fragment_stores_and_atomics;
    }
    return features_.vertex_pipeline_stores_and_atomics || IsSpirvComputeShader();
  }

  bool IsMemoryExportUsed() const {
    return current_shader().memexport_eM_written() && IsMemoryExportSupported();
  }

  void ExportToMemory(uint8_t export_eM);

  // The source may be a floating-point scalar or a vector.
  spv::Id PWLGammaToLinear(spv::Id gamma, bool gamma_pre_saturated);
  spv::Id LinearToPWLGamma(spv::Id linear, bool linear_pre_saturated);

  size_t FindOrAddTextureBinding(uint32_t fetch_constant, xenos::FetchOpDimension dimension,
                                 bool is_signed);
  size_t FindOrAddSamplerBinding(uint32_t fetch_constant, xenos::TextureFilter mag_filter,
                                 xenos::TextureFilter min_filter, xenos::TextureFilter mip_filter,
                                 xenos::AnisoFilter aniso_filter);
  // `texture_parameters` need to be set up except for `sampler`, which will be
  // set internally, optionally doing linear interpolation between the an
  // existing value and the new one (the result location may be the same as for
  // the first lerp endpoint, but not across signedness).
  // wwe2k14 patch (Session 66, PS spec 1): whether the texture binding of this fetch constant is specialised as
  // unsigned (Modification::pixel::tex_unsigned_mask); tex_spec_unsigned_ is set for the fetch being translated and
  // read by SampleTexture / QueryTextureLod (a signed image is then never sampled).
  bool IsTextureSpecUnsigned(uint32_t fetch_constant) const;
  // wwe2k14 patch (Session 66, native_ext): the NATIVE-only specialisation word of the translation being made (the
  // modification has no bit left): pixel shaders kNativeExtEpilogueIdle (the colour epilogue's alpha test, alpha to
  // coverage, colour exponent bias and gamma conversion are all off for the draw: not emitted) and kNativeExtFma
  // (fused multiply-add where the multiply rule is already gone); kVertex vertex shaders kNativeExtFma.
  uint32_t native_ext() const { return current_translation().native_ext(); }
  // wwe2k14 patch (Session 66, PS spec 2): the multiply rule modes. mul_rule_mode_ is the translation's
  // (Modification::pixel::native_mul_mode in a NATIVE pixel translation, else 0); mul_rule_action_ is what
  // ZeroIfAnyOperandIsZero and the inline copy in the vector multiply do for the part of the ALU instruction being
  // translated: 0 the exact rule, 1 `p != p ? 0 : p`, 2 nothing. For modes 1 and 2 the operands are proven finite by
  // a forward pass done while translating (the exec order is the program order when the shader has no loops, jumps
  // or calls and addresses registers statically): per register component "may be inf / NaN", set by the scalar
  // rcp / rsq / log / exp / sqrt family and by operands that are, cleared by saturation and by strong (unpredicated,
  // unconditional) writes of clean values; interpolators, float constants and texture results count as finite.
  uint32_t mul_rule_mode_ = 0;
  // wwe2k14 patch (Session 66, gpu_native_fma): kNativeExtFma in a NATIVE pixel / kVertex vertex translation: a multiply
  // whose multiply rule is already gone (mul_rule_action_ 2, or no lane where the rule applies) and the add that
  // follows it in a mad / dot product are one Fma (single rounding: looks right, not exact).
  bool mul_fma_enabled_ = false;
  uint32_t mul_rule_action_ = 0;
  uint32_t mul_rule_action_vector_ = 0, mul_rule_action_scalar_ = 0;
  bool mul_rule_tracking_ = false;
  bool mul_rule_fetch_finite_ = true;  // (vertex shaders: native_mul_fetch_finite; pixel shaders: always)
  bool mul_rule_vector_tainted_ = false, mul_rule_scalar_tainted_ = false;
  bool mul_rule_exec_unconditional_ = true;
  bool mul_rule_ps_tainted_ = false;
  uint8_t mul_rule_reg_taint_[128] = {};
  bool MulRuleOperandTainted(const InstructionOperand& operand);
  void MulRuleBeginInstruction(const ParsedAluInstruction& instr);
  void MulRuleNoteResult(const InstructionResult& result, bool tainted, bool weak);
  void MulRuleEndInstruction(const ParsedAluInstruction& instr);
  bool tex_spec_unsigned_ = false;
  void SampleTexture(spv::Builder::TextureParameters& texture_parameters,
                     spv::ImageOperandsMask image_operands_mask, spv::Id image_unsigned,
                     spv::Id image_signed, spv::Id sampler, spv::Id is_any_unsigned,
                     spv::Id is_any_signed, spv::Id& result_unsigned_out,
                     spv::Id& result_signed_out, spv::Id lerp_factor = spv::NoResult,
                     spv::Id lerp_first_unsigned = spv::NoResult,
                     spv::Id lerp_first_signed = spv::NoResult);
  // `texture_parameters` need to be set up except for `sampler`, which will be
  // set internally.
  spv::Id QueryTextureLod(spv::Builder::TextureParameters& texture_parameters,
                          spv::Id image_unsigned, spv::Id image_signed, spv::Id sampler,
                          spv::Id is_all_signed);

  spv::Id LoadMsaaSamplesFromFlags();
  // Whether it's possible and worth skipping running the translated shader for
  // 2x2 quads.
  bool FSI_IsDepthStencilEarly() const {
    assert_true(edram_fragment_shader_interlock_);
    return !is_depth_only_fragment_shader_ && !current_shader().writes_depth() &&
           !current_shader().memexport_eM_written();
  }
  void FSI_LoadSampleMask(spv::Id msaa_samples);
  void FSI_LoadEdramOffsets(spv::Id msaa_samples);
  // The address must be a signed int. Whether the render target is 64bpp, if
  // present at all, must be a bool (if it's NoResult, 32bpp will be assumed).
  spv::Id FSI_AddSampleOffset(spv::Id sample_0_address, uint32_t sample_index,
                              spv::Id is_64bpp = spv::NoResult);
  // Updates main_fsi_sample_mask_. Must be called outside non-uniform control
  // flow because of taking derivatives of the fragment depth.
  void FSI_DepthStencilTest(spv::Id msaa_samples, bool sample_mask_potentially_narrowed_previouly);
  // Returns the first and the second 32 bits as two uints.
  std::array<spv::Id, 2> FSI_ClampAndPackColor(spv::Id color_float4, spv::Id format_with_flags);
  std::array<spv::Id, 4> FSI_UnpackColor(std::array<spv::Id, 2> color_packed,
                                         spv::Id format_with_flags);
  // The bounds must have the same number of components as the color or alpha.
  spv::Id FSI_FlushNaNClampAndInBlending(spv::Id color_or_alpha, spv::Id is_fixed_point,
                                         spv::Id min_value, spv::Id max_value);
  spv::Id FSI_ApplyColorBlendFactor(spv::Id value, spv::Id is_fixed_point, spv::Id clamp_min_value,
                                    spv::Id clamp_max_value, spv::Id factor, spv::Id source_color,
                                    spv::Id source_alpha, spv::Id dest_color, spv::Id dest_alpha,
                                    spv::Id constant_color, spv::Id constant_alpha);
  spv::Id FSI_ApplyAlphaBlendFactor(spv::Id value, spv::Id is_fixed_point, spv::Id clamp_min_value,
                                    spv::Id clamp_max_value, spv::Id factor, spv::Id source_alpha,
                                    spv::Id dest_alpha, spv::Id constant_alpha);
  // If source_color_clamped, dest_color, constant_color_clamped are
  // spv::NoResult, will blend the alpha. Otherwise, will blend the color.
  // The result will be unclamped (color packing is supposed to clamp it).
  spv::Id FSI_BlendColorOrAlphaWithUnclampedResult(
      spv::Id is_fixed_point, spv::Id clamp_min_value, spv::Id clamp_max_value,
      spv::Id source_color_clamped, spv::Id source_alpha_clamped, spv::Id dest_color,
      spv::Id dest_alpha, spv::Id constant_color_clamped, spv::Id constant_alpha_clamped,
      spv::Id equation, spv::Id source_factor, spv::Id dest_factor);

  Features features_;
  bool native_2x_msaa_with_attachments_;
  bool native_2x_msaa_no_attachments_;

  // For safety with different drivers (even though fragment shader interlock in
  // SPIR-V only has one control flow requirement - that both begin and end must
  // be dynamically executed exactly once in this order), adhering to the more
  // strict control flow limitations of OpenGL (GLSL) fragment shader interlock,
  // that begin and end are called only on the outermost level of the control
  // flow of the main function, and that there are no returns before either
  // (there's a single return from the shader).
  bool edram_fragment_shader_interlock_;
  uint32_t draw_resolution_scale_x_;
  uint32_t draw_resolution_scale_y_;
  float pixel_position_scale_x_ = 0.0f;  // wwe2k14 patch (Session 54): SetPixelPositionScale
  float pixel_position_scale_y_ = 0.0f;

  // Is currently writing the empty depth-only pixel shader, such as for depth
  // and stencil testing with fragment shader interlock.
  bool is_depth_only_fragment_shader_ = false;

  std::unique_ptr<SpirvBuilder> builder_;

  std::vector<spv::Id> id_vector_temp_;
  // For helper functions like operand loading, so they don't conflict with
  // id_vector_temp_ usage in bigger callbacks.
  std::vector<spv::Id> id_vector_temp_util_;
  std::vector<unsigned int> uint_vector_temp_;
  std::vector<unsigned int> uint_vector_temp_util_;

  spv::Id ext_inst_glsl_std_450_;

  spv::Id type_void_;

  union {
    struct {
      spv::Id type_bool_;
      spv::Id type_bool2_;
      spv::Id type_bool3_;
      spv::Id type_bool4_;
    };
    // Index = component count - 1.
    spv::Id type_bool_vectors_[4];
  };
  union {
    struct {
      spv::Id type_int_;
      spv::Id type_int2_;
      spv::Id type_int3_;
      spv::Id type_int4_;
    };
    spv::Id type_int_vectors_[4];
  };
  union {
    struct {
      spv::Id type_uint_;
      spv::Id type_uint2_;
      spv::Id type_uint3_;
      spv::Id type_uint4_;
    };
    spv::Id type_uint_vectors_[4];
  };
  union {
    struct {
      spv::Id type_float_;
      spv::Id type_float2_;
      spv::Id type_float3_;
      spv::Id type_float4_;
    };
    spv::Id type_float_vectors_[4];
  };

  spv::Id const_int_0_;
  spv::Id const_int4_0_;
  spv::Id const_uint_0_;
  spv::Id const_uint4_0_;
  union {
    struct {
      spv::Id const_float_0_;
      spv::Id const_float2_0_;
      spv::Id const_float3_0_;
      spv::Id const_float4_0_;
    };
    spv::Id const_float_vectors_0_[4];
  };
  union {
    struct {
      spv::Id const_float_1_;
      spv::Id const_float2_1_;
      spv::Id const_float3_1_;
      spv::Id const_float4_1_;
    };
    spv::Id const_float_vectors_1_[4];
  };
  // vec2(0.0, 1.0), to arbitrarily VectorShuffle non-constant and constant
  // components.
  spv::Id const_float2_0_1_;

  enum SystemConstantIndex : unsigned int {
    kSystemConstantFlags,
    kSystemConstantVertexIndexLoadAddress,
    kSystemConstantVertexIndexEndian,
    kSystemConstantLineLoopClosingIndex,
    kSystemConstantVertexBaseIndex,
    kSystemConstantVertexIndexReset,
    kSystemConstantComputeMemExportVertexCount,
    kSystemConstantTexturesResolutionScaled,
    kSystemConstantUserClipPlanes,
    kSystemConstantVertexIndexMin,
    kSystemConstantVertexIndexMax,
    kSystemConstantNdcScale,
    kSystemConstantPointVertexDiameterMin,
    kSystemConstantNdcOffset,
    kSystemConstantPointVertexDiameterMax,
    kSystemConstantPointConstantDiameter,
    kSystemConstantPointScreenDiameterToNdcRadius,
    kSystemConstantTextureSwizzledSigns,
    kSystemConstantTextureSwizzles,
    kSystemConstantAlphaTestReference,
    kSystemConstantAlphaToMask,
    kSystemConstantEdram32bppTilePitchDwordsScaled,
    kSystemConstantEdramDepthBaseDwordsScaled,
    kSystemConstantColorExpBias,
    kSystemConstantEdramPolyOffsetFrontScale,
    kSystemConstantEdramPolyOffsetBackScale,
    kSystemConstantEdramPolyOffsetFrontOffset,
    kSystemConstantEdramPolyOffsetBackOffset,
    kSystemConstantEdramStencilFront,
    kSystemConstantEdramStencilBack,
    kSystemConstantEdramRTBaseDwordsScaled,
    kSystemConstantEdramRTFormatFlags,
    kSystemConstantEdramRTBlendFactorsOps,
    // Accessed as float4[2], not float2[4], due to std140 array stride
    // alignment.
    kSystemConstantEdramRTKeepMask,
    kSystemConstantEdramRTClamp,
    kSystemConstantEdramBlendConstant,
    // wwe2k14 patch (Session 93): LAST (the enum value is the SPIR-V member index; see SystemConstants).
    kSystemConstantSharedMemoryWindowBase,
  };
  spv::Id uniform_system_constants_;
  spv::Id uniform_float_constants_;
  spv::Id uniform_bool_loop_constants_;
  spv::Id uniform_fetch_constants_;

  spv::Id buffers_shared_memory_;
  spv::Id buffer_edram_;

  // Not using combined images and samplers because
  // maxPerStageDescriptorSamplers is often lower than
  // maxPerStageDescriptorSampledImages, and for every fetch constant, there
  // are, for regular fetches, two bindings (unsigned and signed).
  std::vector<TextureBinding> texture_bindings_;
  std::vector<SamplerBinding> sampler_bindings_;

  // VS as VS only - int, or VS as compute (kMemExportCompute) only - uint3.
  spv::Id input_vertex_index_;
  // wwe2k14 patch (Session 95, Lane C): this translation has kNativeExtVsInput and is the translator's own module; its inputs by location (only the ones it reads are declared) and each one's size.
  bool vertex_input_ = false;
  spv::Id input_vertex_attributes_[kViMaxAttributes];
  uint32_t input_vertex_attribute_words_[kViMaxAttributes];
  // wwe2k14 patch (Session 23, MERGE): gl_InstanceIndex and gl_InstanceIndex * the float constant stride (an int
  // computed at the start of main), with instanced_float_constants_stride.
  spv::Id input_instance_index_;
  spv::Id main_instance_float_constant_offset_;
  // wwe2k14 patch (Session 94, gpu_d3d_decouple): gl_ViewIndex (an int input, made at its first use) and gl_ViewIndex * 256 (view_float_constants_block).
  spv::Id input_view_index_;
  spv::Id main_view_float_constant_offset_;
  spv::Id DecoupleViewIndexValue();  // (loads gl_ViewIndex; declares the MultiView capability and the input at the first use)
 public:
  // The number of views the owned passes render (2) while gpu_d3d_decouple is on (the device has multiview, TRUE NATIVE): decided once per process. The
  // translations then read the view index (the layer of a 2D fetch, the memory export guard) and are another translation-store key. 0 = as before.
  static uint32_t DecoupleViews();
  // Whether the kVertex host shaders read their float constants at index + gl_ViewIndex * 256 (gpu_d3d_decouple bit 3, value 8; phase 1B's data layout: one
  // 256-vec4 block per view, the descriptor range covering them all). The vertex modification has no free bit (both words are full), so this is a
  // process-wide decision like DecoupleViews, folded into the translation store's configuration fingerprint instead of the modification.
  static bool DecoupleViewConstantBlocks();
 private:
  // VS as TES only - int.
  spv::Id input_primitive_id_;
  // VS as TES only - float3.
  spv::Id input_tess_coord_;
  // VS as TES only - patch float4.
  spv::Id input_patch_control_point_indices_;
  // PS, only when needed - float2.
  spv::Id input_point_coordinates_;
  // PS, only when needed - float4.
  spv::Id input_fragment_coordinates_;
  // PS, only when needed - bool.
  spv::Id input_front_facing_;
  // PS, only when needed - int.
  spv::Id input_sample_id_;
  // PS, only when needed - int[1].
  spv::Id input_sample_mask_;

  // VS output or PS input, only the ones that are needed (spv::NoResult for the
  // unneeded interpolators), indexed by the guest interpolator index - float4.
  // The Qualcomm Adreno driver has strict requirements for stage linkage - as
  // Xenia uses separate variables, not an array (so the interpolation
  // qualifiers can be applied to each element separately), the interpolators
  // must also be separate variables in the other stage, including the geometry
  // shader (not just an array assuming that consecutive locations will be
  // linked as consecutive array elements, on Qualcomm, they won't be linked at
  // all).
  std::array<spv::Id, xenos::kMaxInterpolators> input_output_interpolators_;

  // VS, only for HostVertexShaderType::kPointListAsTriangleStrip when needed
  // for the PS - float2.
  spv::Id output_point_coordinates_;
  // VS, only when needed - float.
  spv::Id output_point_size_;

  enum OutputPerVertexMember : unsigned int {
    kOutputPerVertexMemberPosition,
    kOutputPerVertexMemberClipDistance,
    kOutputPerVertexMemberCullDistance,
    kOutputPerVertexMemberCountMax,
  };
  uint32_t output_per_vertex_member_count_;
  uint32_t output_per_vertex_member_clip_distance_;
  uint32_t output_per_vertex_member_cull_distance_;
  spv::Id type_output_per_vertex_;
  spv::Id output_per_vertex_;

  // With fragment shader interlock, variables in the main function.
  // Otherwise, framebuffer color attachment outputs.
  std::array<spv::Id, xenos::kMaxColorRenderTargets> output_or_var_fragment_data_;
  // For host render targets and only when needed - float.
  spv::Id output_fragment_depth_;
  // For host render targets and only when needed - int[1].
  spv::Id output_fragment_sample_mask_;

  std::vector<spv::Id> main_interface_;
  spv::Function* function_main_;
  spv::Id main_system_constant_flags_;
  // bool.
  spv::Id var_main_predicate_;
  // uint4.
  spv::Id var_main_loop_count_;
  // int4.
  spv::Id var_main_loop_address_;
  // int.
  spv::Id var_main_address_register_;
  // float.
  spv::Id var_main_previous_scalar_;
  // `base + index * stride` in dwords from the last vfetch_full as it may be
  // needed by vfetch_mini - int.
  spv::Id var_main_vfetch_address_;
  // float.
  spv::Id var_main_tfetch_lod_;
  // float3.
  spv::Id var_main_tfetch_gradients_h_;
  spv::Id var_main_tfetch_gradients_v_;
  // float4[register_count()].
  spv::Id var_main_registers_;
  // Memory export variables are created only when needed.
  // float4.
  spv::Id var_main_memexport_address_;
  // Each is float4.
  spv::Id var_main_memexport_data_[ucode::kMaxMemExportElementCount];
  // Bit field of which eM# elements have been written so far by the invocation
  // since the last memory write - uint.
  spv::Id var_main_memexport_data_written_;
  // If memory export is disabled in certain invocations or (if emulating some
  // primitive types without a geometry shader) at specific guest vertex loop
  // iterations because the translated shader is executed multiple times for the
  // same guest vertex or pixel, this contains whether memory export is allowed
  // in the current execution of the translated code.
  // bool.
  spv::Id main_memexport_allowed_;
  // VS rectangle-list fallback only - uint.
  spv::Id var_main_rectangle_guest_vertex_index_;
  // VS rectangle-list fallback only - float4[3] (gl_Position). Not the
  // gl_PerVertex block type, whose BuiltIn member decorations are only valid
  // for Input and Output variables.
  spv::Id var_main_rectangle_per_vertex_;
  // One per user clip plane, plus one more cull distance for vertex killing.
  static constexpr size_t kMaxRectanglePerVertexDistances = 8;
  // VS rectangle-list fallback only - float[3] for each clip and cull distance.
  // Separate variables rather than one float[3][count], to keep the
  // ArrayStride-decorated gl_PerVertex member type out of Function storage.
  std::array<spv::Id, kMaxRectanglePerVertexDistances> var_main_rectangle_clip_distances_;
  std::array<spv::Id, kMaxRectanglePerVertexDistances> var_main_rectangle_cull_distances_;
  // VS rectangle-list fallback only - float4[3] for each used interpolator.
  std::array<spv::Id, xenos::kMaxInterpolators> var_main_rectangle_interpolators_;
  // VS only - float3 (special exports).
  spv::Id var_main_point_size_edge_flag_kill_vertex_;
  // PS, only when needed - bool.
  spv::Id var_main_kill_pixel_;
  // PS, with fragment shader interlock and depth export - float.
  spv::Id var_main_fragment_depth_;
  // PS, only when writing to color render targets with fragment shader
  // interlock - uint.
  // Whether color buffers have been written to, if not written on the taken
  // execution path, don't export according to Direct3D 9 register documentation
  // (some games rely on this behavior).
  spv::Id var_main_fsi_color_written_;
  // Loaded by FSI_LoadSampleMask.
  // Can be modified on the outermost control flow level in the main function.
  // 0:3 - Per-sample coverage at the current stage of the shader's execution.
  //       Affected by things like gl_SampleMaskIn, early or late depth /
  //       stencil (always resets bits for failing, no matter if need to defer
  //       writing), alpha to coverage.
  // 4:7 - Depth write deferred mask - when early depth / stencil resulted in a
  //       different value for the sample (like different stencil if the test
  //       failed), but can't write it before running the shader because it's
  //       not known if the sample will be discarded by the shader, alphatest or
  //       AtoC.
  // Early depth / stencil rejection of the pixel is possible when both 0:3 and
  // 4:7 are zero.
  spv::Id main_fsi_sample_mask_;
  // Loaded by FSI_LoadEdramOffsets.
  // Including the depth render target base.
  spv::Id main_fsi_address_depth_;
  // Not including the render target base.
  spv::Id main_fsi_offset_32bpp_;
  spv::Id main_fsi_offset_64bpp_;
  // Loaded by FSI_DepthStencilTest for early depth / stencil, the depth /
  // stencil values to write at the end of the shader if the specified in
  // main_fsi_sample_mask_ and if the samples were not discarded later after the
  // early test.
  std::array<spv::Id, 4> main_fsi_late_write_depth_stencil_;
  spv::Block* main_fsi_early_depth_stencil_execute_quad_merge_;
  spv::Block* main_loop_header_;
  spv::Block* main_loop_continue_;
  // Convergence of loop body exits before per-iteration finalization in
  // rectangle-list vertex fallback; otherwise also the loop merge.
  spv::Block* main_loop_merge_;
  // Actual loop merge for rectangle-list vertex fallback; otherwise the same as
  // main_loop_merge_.
  spv::Block* main_loop_exit_;
  spv::Id main_loop_pc_next_;
  spv::Block* main_switch_header_;
  std::unique_ptr<spv::Instruction> main_switch_op_;
  spv::Block* main_switch_merge_;
  std::vector<spv::Id> main_switch_next_pc_phi_operands_;

  // If the exec bool constant / predicate conditional is open, block after it
  // (not added to the function yet).
  spv::Block* cf_exec_conditional_merge_;
  // If the instruction-level predicate conditional is open, block after it (not
  // added to the function yet).
  spv::Block* cf_instruction_predicate_merge_;
  // When cf_exec_conditional_merge_ is not null:
  // If the current exec conditional is based on a bool constant: the number of
  // the bool constant.
  // If it's based on the predicate value: kCfExecBoolConstantPredicate.
  uint32_t cf_exec_bool_constant_or_predicate_;
  static constexpr uint32_t kCfExecBoolConstantPredicate = UINT32_MAX;
  // When cf_exec_conditional_merge_ is not null, the expected bool constant or
  // predicate value for the current exec conditional.
  bool cf_exec_condition_;
  // When cf_instruction_predicate_merge_ is not null, the expected predicate
  // value for the current or the last instruction.
  bool cf_instruction_predicate_condition_;
  // Whether there was a `setp` in the current exec before the current
  // instruction, thus instruction-level predicate value can be different than
  // the exec-level predicate value, and can't merge two execs with the same
  // predicate condition anymore.
  bool cf_exec_predicate_written_;
};

}  // namespace rex::graphics
