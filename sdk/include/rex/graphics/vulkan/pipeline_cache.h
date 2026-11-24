#pragma once
#include <rex/graphics/cut_probe.h>  // wwe2k14 patch (Session 92, M4f-C): detector 2
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

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include <rex/hash.h>
#include <rex/platform.h>
#include <rex/thread.h>
#include <rex/graphics/primitive_processor.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/vulkan/render_target_cache.h>
#include <rex/graphics/vulkan/shader.h>
#include <rex/graphics/xenos.h>
#include <rex/ui/vulkan/api.h>

namespace rex::graphics::vulkan {

class VulkanCommandProcessor;

// TODO(Triang3l): Create a common base for both the Vulkan and the Direct3D
// implementations.
class VulkanPipelineCache {
 public:
  static constexpr size_t kLayoutUIDEmpty = 0;

  class PipelineLayoutProvider {
   public:
    virtual ~PipelineLayoutProvider() {}
    virtual VkPipelineLayout GetPipelineLayout() const = 0;

   protected:
    PipelineLayoutProvider() = default;
  };

  VulkanPipelineCache(VulkanCommandProcessor& command_processor, const RegisterFile& register_file,
                      VulkanRenderTargetCache& render_target_cache,
                      VkShaderStageFlags guest_shader_vertex_stages);
  ~VulkanPipelineCache();

  bool Initialize();
  void Shutdown();
  void InitializeShaderStorage(const std::filesystem::path& cache_root, uint32_t title_id,
                               bool blocking);
  void ShutdownShaderStorage();
  void EndSubmission();
  // wwe2k14 patch (Session 6, H1): creates the queued pipelines on the calling
  // (command processor) thread and waits for the creation threads to finish -
  // what EndSubmission does unless vulkan_pipeline_no_submit_wait.
  void AwaitPipelineCreation();

  VulkanShader* LoadShader(xenos::ShaderType shader_type, const uint32_t* host_address,
                           uint32_t dword_count);
  // Analyze shader microcode on the translator thread.
  // wwe2k14 patch (Session 87, stage 4, gpu_native_swap_skip_translation): a registered shader's analysis comes from its
  // registration instead of its microcode (implemented in pipeline_cache.cpp).
  void AnalyzeShaderUcode(Shader& shader);
  // The real analysis of a shader whose analysis was taken from its registration (the refusal fallback, or a translation
  // that needs the microcode): drops the registered one first; plain AnalyzeUcode otherwise.
  void PrepareFullAnalysis(Shader& shader);
  // wwe2k14 patch (Session 87, stage 4, gpu_native_swap_skip_translation_pixel): whether a pixel draw with this native ext uses the
  // replacement translation's bindings (SpirvShader::TextureBindingsFor(true)) rather than the emulated translation's.
  bool UsesReplacementBindings(const Shader* pixel_shader, uint32_t native_ext) const;
  void NoteNativePixelRefusal(const Shader& shader, const char* why);
  // wwe2k14 patch (Session 61, gpu_native_pipeline_storage): a shader loaded so far (from the shader storage or by a
  // draw), by its microcode hash; nullptr if none.
  VulkanShader* FindShader(uint64_t ucode_data_hash) const {
    auto it = shaders_.find(ucode_data_hash);
    return it != shaders_.end() ? it->second : nullptr;
  }
  // wwe2k14 patch (Session 61): the shader into the shader storage file if it is not there yet - as ConfigurePipeline
  // stores the shaders of the emulated path's pipelines; the native renderer's draws make none (their shaders were
  // never stored before: runs 001-018, 6 of 234 native records' shaders missing).
  void StoreShader(Shader& shader);

  // wwe2k14 patch (Session 92, M6, translate before play; implemented in m6_precompile.cpp - no data members: the queue is
  // file-level state there). The game's precompile mode translates the variants of the player's .xvar list into the translation
  // store (translation_store.h), on its first frames. One step per frame (a time budget in ms); frame = the native renderer's
  // frame counter (retry passes every 10 frames, 20 at most, for shaders not loaded yet). Waiting() = variants still to be
  // made (the precompile status counts them; 0 when the queue is empty, given up or the store is off).
  void M6PrecompileStep(uint64_t frame, uint32_t budget_ms);
  uint32_t M6PrecompileWaiting() const;
  void M6PrecompileProgress(uint64_t& done, uint64_t& total) const;  // Session 93: the precompile status line's "m6 D of T"

  // Retrieves the shader modification for the current state. The shader must
  // have microcode analyzed.
  SpirvShaderTranslator::Modification GetCurrentVertexShaderModification(
      const Shader& shader, Shader::HostVertexShaderType host_vertex_shader_type,
      uint32_t interpolator_mask, bool ps_param_gen_used) const;
  // wwe2k14 patch (Session 66, keycost): the specialisation settings (the cvars) and whether the native renderer is on,
  // read once per frame - the per-draw key functions and the lean / repeat / crowd guards use these instead of a dozen
  // cvar reads and calls each. A setting changed live takes effect at the next frame end, in every key at once (a
  // base draw's keys are only compared within its frame); the stored-record functions read the cvars themselves.
  struct NativeSpecLatch {
    bool valid = false;
    bool native = false;
    bool endian = false, ps_texture = false, ps_implicit_lod = false, ps_epilogue = false, vs_fetch_finite = false;
    uint8_t ps_mul = 0, vs_mul = 0, fma = 0;
    bool ps_wanted = false;  // any pixel shader key setting
  };
  const NativeSpecLatch& NativeSpec() const {
    if (!spec_latch_.valid) {
      RefreshNativeSpec();
    }
    return spec_latch_;
  }
  void RefreshNativeSpec() const;
  // wwe2k14 patch (Session 66, B2.1 gpu_native_vertex_endian_spec): the vertex fetch endian key of the shader against
  // the current fetch constants (what GetCurrentVertexShaderModification puts in the modification: bit 0 = valid,
  // bits 1-16 the modes); 0 with the cvar off. The lean / repeat / crowd paths compare it with the key their base
  // draw's pipeline was built with.
  uint32_t GetVertexFetchEndianKey(const Shader& shader) const;
  // wwe2k14 patch (Session 66, PS spec): see GetNativePixelSpecKey in pipeline_cache.cpp.
  // (census: the keys as if every setting were on - gpu_native_ps_census.)
  uint32_t GetNativePixelSpecKey(const Shader* pixel_shader, bool census = false) const;
  bool NativePixelSpecWanted() const;  // any of the pixel shader specialisation settings is on
  bool PixelSpecKeyChanged(const Shader* pixel_shader, uint32_t recorded_key) const;
  // The bits of a key from GetNativePixelSpecKey into NATIVE's copy of a pixel modification (the key is computed once
  // per draw, by the caller, so the module and the key the fast paths compare cannot differ).
  static void ApplyNativePixelSpec(SpirvShaderTranslator::Modification& modification, uint32_t key);
  // wwe2k14 patch (Session 66): a stored record's modification made for other settings than the current ones is
  // converted to the current ones (an unspecialised key for a setting that is off; for a setting that is on, the
  // specialisation the draws most likely have - only a prediction, a wrong one costs a missed prewarm). `converted` is
  // set when the value changed. Vertex: gpu_native_vertex_endian_spec - every binding (up to 8) endian 8-in-32, the
  // game's only mode (census). Native pixel: gpu_native_ps_texture_spec - the all-unsigned mask for the first
  // min(bindings, 4) texture bindings; the mul mode and the implicit LOD bit are set to the current settings.
  // (native: the record is NATIVE's - its vertex modification may carry the multiply rule bits, set to the current
  // gpu_native_vs_mul_mode; an emulated record never does: they are stripped.)
  uint64_t ConformStoredVertexModification(VulkanShader& vertex_shader, uint64_t modification, bool* converted,
                                           bool native);
  // wwe2k14 patch (Session 66, native_ext): the NATIVE specialisation words the draws will most likely have with the
  // current settings, for a stored record (the record has no room for them) - the prewarm builds the translations
  // with these; only a prediction (a wrong one costs a missed prewarm). `settings_only`: the part that does not depend
  // on the draw state (FMA).
  mutable NativeSpecLatch spec_latch_;
  uint32_t PredictNativePixelExt() const;
  uint32_t PredictNativeVertexExt(uint64_t vertex_modification, uint64_t vertex_shader_hash = 0) const;  // (Session 87: + our vertex shader swap)
  // Whether the settings that make a recorded draw's NATIVE ext words changed since (the lean / repeat / crowd guard; the
  // draw-state part of the pixel ext - the colour epilogue - is a function of render state registers those paths hold
  // equal to the base's).
  bool NativeExtStale(uint32_t recorded_pixel_ext, uint32_t recorded_vertex_ext, const Shader* pixel_shader,
                      const Shader* vertex_shader = nullptr) const;  // (Session 87: + our vertex shader swap)

  // wwe2k14 patch (Session 66, VS mul): NATIVE's vertex shader multiply rule key - bits 0-1 gpu_native_vs_mul_mode, bit
  // 2 gpu_native_vs_mul_fetch_finite; 0 with the mode 0 or without the native renderer. Computed once per draw
  // (NativeDescribeDraw: applied to NATIVE's own vertex translation of a kVertex draw) and compared by the lean /
  // repeat / crowd paths with the key of their base draw.
  uint32_t GetNativeVertexMulKey() const;
  bool VertexMulKeyChanged(uint32_t recorded_key) const;
  static void ApplyNativeVertexSpec(SpirvShaderTranslator::Modification& modification, uint32_t key);
  // (Session 69: mask_known - the record was stored with the texture-sign specialisation on, so a 0 mask is its own.)
  uint64_t ConformStoredNativePixelModification(VulkanShader& pixel_shader, uint64_t modification, bool* converted,
                                                bool mask_known = false);
  // The emulated path's modification of a NATIVE record's one (a stored record holds the native bits).
  static void StripNativePixelSpec(SpirvShaderTranslator::Modification& modification);
  SpirvShaderTranslator::Modification GetCurrentPixelShaderModification(
      const Shader& shader, uint32_t interpolator_mask, uint32_t param_gen_pos,
      reg::RB_DEPTHCONTROL normalized_depth_control) const;

  // wwe2k14 patch (Session 92, M4b; docs\native\d3d\S92_M3_M5_DESIGN.md section 2): the same derivations from the record decoder's
  // typed state (native\owned_record.h) instead of the register source - the words these functions read, as typed fields plus the
  // 32 fetch constants (6 words each). Declared beside the register-reading ones above; the shadow compares both per draw.
  struct TypedKeyState {
    reg::SQ_PROGRAM_CNTL program_cntl;
    reg::VGT_HOS_CNTL hos_cntl;
    reg::PA_CL_CLIP_CNTL clip_cntl;
    reg::VGT_DRAW_INITIATOR draw_initiator;
    reg::RB_SURFACE_INFO surface_info;
    reg::SQ_CONTEXT_MISC context_misc;
    reg::SQ_INTERPOLATOR_CNTL interpolator_cntl;
    reg::RB_DEPTH_INFO depth_info;
    reg::RB_COLORCONTROL color_control;
    const uint32_t* fetch = nullptr;  // 32 x 6 words
  };
  SpirvShaderTranslator::Modification GetCurrentVertexShaderModification(
      const Shader& shader, Shader::HostVertexShaderType host_vertex_shader_type, uint32_t interpolator_mask,
      bool ps_param_gen_used, const TypedKeyState& state) const;
  SpirvShaderTranslator::Modification GetCurrentPixelShaderModification(
      const Shader& shader, uint32_t interpolator_mask, uint32_t param_gen_pos, reg::RB_DEPTHCONTROL normalized_depth_control,
      const TypedKeyState& state) const;
  uint32_t GetVertexFetchEndianKey(const Shader& shader, const TypedKeyState& state) const;  // (not cached)
  uint32_t GetNativePixelSpecKey(const Shader* pixel_shader, const TypedKeyState& state) const;  // (not cached)
  bool PixelSpecKeyChanged(const Shader* pixel_shader, uint32_t recorded_key, const TypedKeyState& state) const;  // (Session 92, M4f-C: from the typed state)

  // wwe2k14 patch (Session 87, stage 4): defer_mask bit 0 / 1 = the emulated (ext-free) vertex / pixel translation of a shader
  // with a registered analysis is left untranslated (the caller does not use its module; whoever does asks again with 0).
  bool EnsureShadersTranslated(VulkanShader::VulkanTranslation* vertex_shader,
                               VulkanShader::VulkanTranslation* pixel_shader, uint32_t defer_mask = 0);
  // TODO(Triang3l): Return a deferred creation handle.
  bool ConfigurePipeline(
      VulkanShader::VulkanTranslation* vertex_shader, VulkanShader::VulkanTranslation* pixel_shader,
      const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
      reg::RB_DEPTHCONTROL normalized_depth_control, uint32_t normalized_color_mask,
      VulkanRenderTargetCache::RenderPassKey render_pass_key, VkPipeline& pipeline_out,
      const PipelineLayoutProvider*& pipeline_layout_out, void** pipeline_handle_out = nullptr);
  bool IsCreatingPipelines() const;

  // wwe2k14 patch (Session 93, idea 7): vulkan_extended_dynamic_state. The guest pipelines made while EdsActive() have their cull
  // mode, front face, depth and stencil state dynamic (VK_EXT_extended_dynamic_state), so draws that differ only in those share
  // one pipeline; the command processor sets the state (CurrentEdsState(), the draw ConfigurePipeline last described) in the
  // command buffer before each draw that uses such a pipeline. The Vulkan values, as the static pipeline state would hold them.
  struct EdsState {
    uint32_t cull_mode = 0;           // VkCullModeFlags
    uint32_t front_face = 0;          // VkFrontFace
    uint32_t depth_test_enable = 0;
    uint32_t depth_write_enable = 0;
    uint32_t depth_compare_op = 0;    // VkCompareOp
    uint32_t stencil_test_enable = 0;
    uint32_t stencil_ops[2][4] = {};  // per face: VkStencilOp fail, pass, depth fail; VkCompareOp
    bool operator==(const EdsState& o) const { return std::memcmp(this, &o, sizeof(*this)) == 0; }
  };
  // On: the setting, the device extension + feature, the quirks table (gpu_quirks_eds_allowed bit 1) and the host render target
  // path (the interlock path has no pipeline depth / stencil state). Decided once, at Initialize.
  bool EdsActive() const { return eds_active_; }
  const EdsState& CurrentEdsState() const { return current_eds_; }
  struct EdsStats {
    uint64_t pipelines_created = 0;  // new pipeline keys (created or queued), since boot
    uint64_t keys_collapsed = 0;     // draws whose full (old) key was new but whose collapsed key was known or already counted
    uint64_t full_keys_seen = 0;     // distinct old keys seen (what the pipeline count would have been)
  };
  EdsStats& eds_stats() { return eds_stats_; }
  // wwe2k14 patch (Session 38, NATIVE): this pixel shader translated for host render targets (colour written to
  // attachments, no EDRAM emulation) for the native renderer, from the emulated draw's modification. Kept in the
  // shader with the modification's native_host_render_targets bit set. Null if it can't be translated, or if its
  // texture bindings would differ from the emulated translation's (the native draw reuses the emulated draw's sets).
  // wwe2k14 patch (Session 43, NATIVE DEPTH): float24_depth - the shader writes its depth rounded to 20e4 (in guest
  // 0-1 units), as the interlock shader tests and stores it (post targets that test depth exactly).
  VulkanShader::VulkanTranslation* GetNativePixelShaderTranslation(VulkanShader& shader,
                                                                   SpirvShaderTranslator::Modification modification,
                                                                   bool float24_depth = false,
                                                                   uint32_t color_quantize = 0,
                                                                   bool twin_position = false,  // Session 54
                                                                   uint32_t native_ext = 0);    // Session 66
  // wwe2k14 patch (Session 42, NATIVE POST): the geometry shader the emulated pipeline expands this draw's primitives
  // with (point, rectangle and quad lists on hosts with geometry shaders) and the host topology it takes;
  // VK_NULL_HANDLE when the draw has no such expansion.
  VkShaderModule GetNativeGeometryShader(const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
                                         SpirvShaderTranslator::Modification vertex_modification,
                                         SpirvShaderTranslator::Modification pixel_modification,
                                         VkPrimitiveTopology& topology_out);
  void GetPipelineAndLayoutByHandle(void* handle, VkPipeline& pipeline_out,
                                    const PipelineLayoutProvider*& pipeline_layout_out,
                                    bool* is_placeholder_out = nullptr) const;

 private:
  REXPACKEDSTRUCT(ShaderStoredHeader, {
    uint64_t ucode_data_hash;

    uint32_t ucode_dword_count : 31;
    xenos::ShaderType type : 1;

    static constexpr uint32_t kVersion = 0x20201219;
  });

  enum class PipelineGeometryShader : uint32_t {
    kNone,
    kPointList,
    kRectangleList,
    kQuadList,
  };

  enum class PipelinePrimitiveTopology : uint32_t {
    kPointList,
    kLineList,
    kLineStrip,
    kTriangleList,
    kTriangleStrip,
    kTriangleFan,
    kLineListWithAdjacency,
    kPatchList,
  };

  enum class PipelinePolygonMode : uint32_t {
    kFill,
    kLine,
    kPoint,
  };

  enum class PipelineBlendFactor : uint32_t {
    kZero,
    kOne,
    kSrcColor,
    kOneMinusSrcColor,
    kDstColor,
    kOneMinusDstColor,
    kSrcAlpha,
    kOneMinusSrcAlpha,
    kDstAlpha,
    kOneMinusDstAlpha,
    kConstantColor,
    kOneMinusConstantColor,
    kConstantAlpha,
    kOneMinusConstantAlpha,
    kSrcAlphaSaturate,
  };

  // Update PipelineDescription::kVersion if anything is changed!
  REXPACKEDSTRUCT(PipelineRenderTarget, {
    PipelineBlendFactor src_color_blend_factor : 4;  // 4
    PipelineBlendFactor dst_color_blend_factor : 4;  // 8
    xenos::BlendOp color_blend_op : 3;               // 11
    PipelineBlendFactor src_alpha_blend_factor : 4;  // 15
    PipelineBlendFactor dst_alpha_blend_factor : 4;  // 19
    xenos::BlendOp alpha_blend_op : 3;               // 22
    uint32_t color_write_mask : 4;                   // 26
  });

  REXPACKEDSTRUCT(PipelineDescription, {
    uint64_t vertex_shader_hash;
    uint64_t vertex_shader_modification;
    // 0 if no pixel shader.
    uint64_t pixel_shader_hash;
    uint64_t pixel_shader_modification;
    VulkanRenderTargetCache::RenderPassKey render_pass_key;

    // Shader stages.
    PipelineGeometryShader geometry_shader : 2;  // 2
    // Input assembly.
    PipelinePrimitiveTopology primitive_topology : 3;  // 5
    uint32_t primitive_restart : 1;                    // 6
    xenos::TessellationMode tessellation_mode : 2;     // 8
    // Rasterization.
    uint32_t depth_clamp_enable : 1;       // 9
    PipelinePolygonMode polygon_mode : 2;  // 11
    uint32_t cull_front : 1;               // 12
    uint32_t cull_back : 1;                // 13
    uint32_t front_face_clockwise : 1;     // 14
    uint32_t rasterizer_discard : 1;       // 15
    // Depth / stencil.
    uint32_t depth_write_enable : 1;                      // 16
    xenos::CompareFunction depth_compare_op : 3;          // 19
    uint32_t stencil_test_enable : 1;                     // 20
    xenos::StencilOp stencil_front_fail_op : 3;           // 23
    xenos::StencilOp stencil_front_pass_op : 3;           // 26
    xenos::StencilOp stencil_front_depth_fail_op : 3;     // 29
    xenos::CompareFunction stencil_front_compare_op : 3;  // 32
    xenos::StencilOp stencil_back_fail_op : 3;            // 35

    xenos::StencilOp stencil_back_pass_op : 3;           // 3
    xenos::StencilOp stencil_back_depth_fail_op : 3;     // 6
    xenos::CompareFunction stencil_back_compare_op : 3;  // 9
    uint32_t sample_rate_shading : 1;                    // 10
    // wwe2k14 patch (Session 93, idea 7): vulkan_extended_dynamic_state. eds_dynamic = this pipeline has its cull mode, front
    // face, depth and stencil state dynamic (VK_EXT_extended_dynamic_state): cull_front / cull_back / front_face_clockwise /
    // depth_write_enable / depth_compare_op / stencil_* above are zero in the key then (the draw's values are VulkanPipelineCache::
    // EdsState), and eds_depth_used keeps the one thing the creation still needs of them: for a pipeline with no pixel shader,
    // whether the draw tests or writes depth (the depth-only fragment shader that rounds / truncates the depth). Both zero
    // with the setting off: the key and the stored records are byte for byte what they were (bits that were always zero).
    uint32_t eds_dynamic : 1;    // 11
    uint32_t eds_depth_used : 1; // 12

    // Filled only for the attachments present in the render pass object.
    PipelineRenderTarget render_targets[xenos::kMaxColorRenderTargets];

    // Including all the padding, for a stable hash.
    static constexpr uint32_t kVersion = 0x20260228;
    PipelineDescription() {
      Reset();
    }
    PipelineDescription(const PipelineDescription& description) {
      std::memcpy(this, &description, sizeof(*this));
    }
    PipelineDescription& operator=(const PipelineDescription& description) {
      std::memcpy(this, &description, sizeof(*this));
      return *this;
    }
    bool operator==(const PipelineDescription& description) const {
      return std::memcmp(this, &description, sizeof(*this)) == 0;
    }
    void Reset() {
      std::memset(this, 0, sizeof(*this));
    }
    uint64_t GetHash() const {
      return XXH3_64bits(this, sizeof(*this));
    }
    struct Hasher {
      size_t operator()(const PipelineDescription& description) const {
        return size_t(description.GetHash());
      }
    };
  });
  REXPACKEDSTRUCT(PipelineStoredDescription, {
    uint64_t description_hash;
    PipelineDescription description;
  });

  struct Pipeline {
    std::atomic<VkPipeline> pipeline{VK_NULL_HANDLE};
    // The layouts are owned by the VulkanCommandProcessor, and must not be
    // destroyed by it while the pipeline cache is active.
    std::atomic<const PipelineLayoutProvider*> pipeline_layout{nullptr};
    std::atomic<bool> is_placeholder{false};
    Pipeline() = default;
    Pipeline(const PipelineLayoutProvider* pipeline_layout_provider)
        : pipeline_layout(pipeline_layout_provider) {}
    Pipeline(const Pipeline& other)
        : pipeline(other.pipeline.load(std::memory_order_acquire)),
          pipeline_layout(other.pipeline_layout.load(std::memory_order_acquire)),
          is_placeholder(other.is_placeholder.load(std::memory_order_acquire)) {}
    Pipeline& operator=(const Pipeline& other) {
      if (this == &other) {
        return *this;
      }
      pipeline.store(other.pipeline.load(std::memory_order_acquire), std::memory_order_release);
      pipeline_layout.store(other.pipeline_layout.load(std::memory_order_acquire),
                            std::memory_order_release);
      is_placeholder.store(other.is_placeholder.load(std::memory_order_acquire),
                           std::memory_order_release);
      return *this;
    }
  };

  // Description that can be passed from the command processor thread to the
  // creation threads, with everything needed from caches pre-looked-up.
  struct PipelineCreationArguments {
    uint8_t priority = 0;
    std::pair<const PipelineDescription, Pipeline>* pipeline = nullptr;
    const PipelineLayoutProvider* pipeline_layout = nullptr;
    // Guest shader translation (VS or TES depending on host vertex type).
    const VulkanShader::VulkanTranslation* vertex_shader = nullptr;
    const VulkanShader::VulkanTranslation* pixel_shader = nullptr;
    // Non-guest stages for tessellation.
    VkShaderModule tessellation_vertex_shader = VK_NULL_HANDLE;
    VkShaderModule tessellation_control_shader = VK_NULL_HANDLE;
    uint32_t tessellation_patch_control_points = 0;
    VkShaderModule geometry_shader = VK_NULL_HANDLE;
    // VK_NULL_HANDLE when dynamic rendering is used.
    VkRenderPass render_pass = VK_NULL_HANDLE;
  };
  struct PipelineCreationArgumentsPriorityComparator {
    bool operator()(const PipelineCreationArguments& a, const PipelineCreationArguments& b) const {
      return a.priority < b.priority;
    }
  };

  union TessellationControlShaderKey {
    uint32_t key;
    struct {
      Shader::HostVertexShaderType host_vertex_shader_type : Shader::kHostVertexShaderTypeBitCount;
      xenos::TessellationMode tessellation_mode : 2;
    };

    TessellationControlShaderKey() : key(0) { static_assert_size(*this, sizeof(key)); }

    struct Hasher {
      size_t operator()(const TessellationControlShaderKey& key) const {
        return std::hash<uint32_t>{}(key.key);
      }
    };
    bool operator==(const TessellationControlShaderKey& other_key) const {
      return key == other_key.key;
    }
  };

  union GeometryShaderKey {
    uint32_t key;
    struct {
      PipelineGeometryShader type : 2;
      uint32_t interpolator_count : 5;
      uint32_t user_clip_plane_count : 3;
      uint32_t user_clip_plane_cull : 1;
      uint32_t has_vertex_kill_and : 1;
      uint32_t has_point_size : 1;
      uint32_t has_point_coordinates : 1;
      // PA_CL_CLIP_CNTL::ps_ucp_mode for point primitives.
      uint32_t point_ps_ucp_mode : 2;
    };

    GeometryShaderKey() : key(0) { static_assert_size(*this, sizeof(key)); }

    struct Hasher {
      size_t operator()(const GeometryShaderKey& key) const {
        return std::hash<uint32_t>{}(key.key);
      }
    };
    bool operator==(const GeometryShaderKey& other_key) const { return key == other_key.key; }
    bool operator!=(const GeometryShaderKey& other_key) const { return !(*this == other_key); }
  };

  VulkanShader* LoadShader(xenos::ShaderType shader_type, const uint32_t* host_address,
                           uint32_t dword_count, uint64_t data_hash);

  // Can be called from multiple threads.
  bool TranslateAnalyzedShader(SpirvShaderTranslator& translator,
                               VulkanShader::VulkanTranslation& translation);
  // wwe2k14 patch (Session 87, stage 4, gpu_native_swap_skip_verify): the real analysis and an emulated translation of the
  // same microcode in a scratch shader, compared field by field with what the registration gave (logged, counted).
  void VerifyRegisteredAnalysis(Shader& shader, const Shader::RegisteredAnalysis& analysis);

  void WritePipelineRenderTargetDescription(reg::RB_BLENDCONTROL blend_control, uint32_t write_mask,
                                            PipelineRenderTarget& render_target_out) const;
  bool GetCurrentStateDescription(
      const VulkanShader::VulkanTranslation* vertex_shader,
      const VulkanShader::VulkanTranslation* pixel_shader,
      const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
      reg::RB_DEPTHCONTROL normalized_depth_control, uint32_t normalized_color_mask,
      VulkanRenderTargetCache::RenderPassKey render_pass_key,
      PipelineDescription& description_out);
  // (Session 93, idea 7: the description as before, every field filled.)
  bool GetCurrentStateDescriptionFull(
      const VulkanShader::VulkanTranslation* vertex_shader,
      const VulkanShader::VulkanTranslation* pixel_shader,
      const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
      reg::RB_DEPTHCONTROL normalized_depth_control, uint32_t normalized_color_mask,
      VulkanRenderTargetCache::RenderPassKey render_pass_key,
      PipelineDescription& description_out) const;
  // (Session 93, idea 7) The Vulkan values of a description's cull / depth / stencil fields (the static pipeline state's formulas).
  static EdsState EdsStateOf(const PipelineDescription& description);
  // Zeroes those fields and sets eds_dynamic / eds_depth_used (a description from an older storage file, or a full one).
  static void CollapseForEds(PipelineDescription& description);

  // Whether the pipeline for the given description is supported by the device.
  bool ArePipelineRequirementsMet(const PipelineDescription& description) const;

  static bool GetGeometryShaderKey(PipelineGeometryShader geometry_shader_type,
                                   SpirvShaderTranslator::Modification vertex_shader_modification,
                                   SpirvShaderTranslator::Modification pixel_shader_modification,
                                   GeometryShaderKey& key_out);
  static uint32_t GetTessellationPatchControlPointCount(
      Shader::HostVertexShaderType host_vertex_shader_type,
      xenos::TessellationMode tessellation_mode);
  VkShaderModule GetTessellationVertexShader(bool adaptive);
  VkShaderModule GetTessellationControlShader(Shader::HostVertexShaderType host_vertex_shader_type,
                                              xenos::TessellationMode tessellation_mode);
  VkShaderModule GetGeometryShader(GeometryShaderKey key);
  bool TryGetPipelineCreationArgumentsForDescription(
      const PipelineDescription& description,
      std::pair<const PipelineDescription, Pipeline>* pipeline,
      PipelineCreationArguments& creation_arguments, bool for_placeholder = false);

  // Can be called from creation threads - all needed data must be fully set up
  // at the point of the call: shaders must be translated, and the pipeline
  // layout and render pass object (unless dynamic rendering is used) must be
  // available.
  bool EnsurePipelineCreated(const PipelineCreationArguments& creation_arguments,
                             VkShaderModule fragment_shader_override = VK_NULL_HANDLE);
  void CreationThread(size_t thread_index);
  void CreateQueuedPipelinesOnProcessorThread();
  // wwe2k14 patch (Session 55): one queued pipeline made on this thread; one pending pipeline waited for.
  void CreateQueuedPipelineOnProcessorThread(const PipelineCreationArguments& creation_arguments);
  void AwaitPendingPipeline(std::pair<const PipelineDescription, Pipeline>* entry);
  void ProcessDeferredPipelineDestructions(bool force_all);

  VulkanCommandProcessor& command_processor_;
  const RegisterFile& register_file_;
  // wwe2k14 patch (Session 87, D3D layer stage 3): the fetch constants the specialisation keys read, from the record's
  // register image while a record-fed draw runs (SetRegisterSource, command processor thread).
  const RegisterFile* register_source_ = nullptr;
  const RegisterFile& KeyRegisters() const {
    REX_CUT_PROBE(kKeyRegisters);
    return register_source_ ? *register_source_ : register_file_;
  }
 public:
  void SetRegisterSource(const RegisterFile* source) { register_source_ = source; }
 private:
  // wwe2k14 patch (Session 55): draws into narrower surfaces whose pipeline was made or awaited at once (logged at
  // powers of two).
  uint64_t offscreen_waits_ = 0;
  void NoteOffscreenWait(const char* how) {
    const uint64_t n = ++offscreen_waits_;
    if (!(n & (n - 1))) {
      REXGPU_INFO("[s55][sync_offscreen] {} pipelines for draws into surfaces narrower than the screen made or awaited "
                  "at once instead of skipping the draw (latest: {})", n, how);
    }
  }
  VulkanRenderTargetCache& render_target_cache_;
  VkShaderStageFlags guest_shader_vertex_stages_;

  // Temporary storage for AnalyzeUcode calls on the processor thread.
  string::StringBuffer ucode_disasm_buffer_;
  // Reusable shader translator on the command processor thread.
  std::unique_ptr<SpirvShaderTranslator> shader_translator_;
  // wwe2k14 patch (Session 38, NATIVE): a second translator for host render targets (GetNativePixelShaderTranslation).
  std::unique_ptr<SpirvShaderTranslator> native_shader_translator_;
  // wwe2k14 patch (Session 92, M6): NATIVE's translator, made on first use (GetNativePixelShaderTranslation, the precompile).
  SpirvShaderTranslator& NativeShaderTranslator();
  // The depth-only fragment shaders (made after the translation store opens when it is on: they go through the translator).
  bool EnsureInternalFragmentShaders();
  // One .xvar variant translated into the store on a scratch shader (m6_precompile.cpp): 0 translated, 1 held already,
  // 2 its shader is not loaded yet, 3 refused.
  int M6TranslateVariant(bool native, bool pixel, uint64_t hash, uint64_t modification, uint32_t native_ext, float scale_x,
                         float scale_y);

  struct LayoutUID {
    size_t uid;
    size_t vector_span_offset;
    size_t vector_span_length;
  };
  std::mutex layouts_mutex_;
  // Texture binding layouts of different shaders, for obtaining layout UIDs.
  std::vector<VulkanShader::TextureBinding> texture_binding_layouts_;
  // Map of texture binding layouts used by shaders, for obtaining UIDs. Keys
  // are XXH3 hashes of layouts, values need manual collision resolution using
  // layout_vector_offset:layout_length of texture_binding_layouts_.
  std::unordered_multimap<uint64_t, LayoutUID, rex::IdentityHasher<uint64_t>>
      texture_binding_layout_map_;

  // Ucode hash -> shader.
  std::unordered_map<uint64_t, VulkanShader*, rex::IdentityHasher<uint64_t>> shaders_;

  // Geometry shaders for Xenos primitive types not supported by Vulkan.
  // Stores VK_NULL_HANDLE if failed to create.
  std::unordered_map<GeometryShaderKey, VkShaderModule, GeometryShaderKey::Hasher>
      geometry_shaders_;

  // Fixed-function emulation shaders for tessellation.
  bool tessellation_indexed_vertex_shader_attempted_ = false;
  VkShaderModule tessellation_indexed_vertex_shader_ = VK_NULL_HANDLE;
  bool tessellation_adaptive_vertex_shader_attempted_ = false;
  VkShaderModule tessellation_adaptive_vertex_shader_ = VK_NULL_HANDLE;
  std::unordered_map<TessellationControlShaderKey, VkShaderModule,
                     TessellationControlShaderKey::Hasher>
      tessellation_control_shaders_;

  // Empty depth-only pixel shader for writing to depth buffer using fragment
  // shader interlock when no Xenos pixel shader provided.
  VkShaderModule depth_only_fragment_shader_ = VK_NULL_HANDLE;
  VkShaderModule placeholder_pixel_shader_ = VK_NULL_HANDLE;
  // Depth-only shaders for float24 emulation when no Xenos pixel shader is
  // provided in host render target mode.
  VkShaderModule depth_float24_truncate_fragment_shader_ = VK_NULL_HANDLE;
  VkShaderModule depth_float24_round_fragment_shader_ = VK_NULL_HANDLE;

  std::unordered_map<PipelineDescription, Pipeline, PipelineDescription::Hasher> pipelines_;

  // Previously used pipeline, to avoid lookups if the state wasn't changed.
  const std::pair<const PipelineDescription, Pipeline>* last_pipeline_ = nullptr;
  // wwe2k14 patch (Session 93, idea 7): see EdsActive(). The last draw's state, and the old (uncollapsed) keys seen (a hash of
  // each, for the collapse statistics only).
  bool eds_active_ = false;
  EdsState current_eds_;
  EdsStats eds_stats_;
  uint64_t eds_last_full_key_ = 0;
  std::unordered_set<uint64_t> eds_full_keys_;
  std::unordered_set<uint64_t> eds_collapsed_keys_;
  // <Submission index, pipeline>.
  std::deque<std::pair<uint64_t, VkPipeline>> deferred_destroy_pipelines_;
  std::mutex deferred_destroy_lock_;

  // Currently open shader storage path.
  std::filesystem::path shader_storage_cache_root_;
  uint32_t shader_storage_title_id_ = 0;

  // Shader storage output stream, for preload in the next emulator runs.
  FILE* shader_storage_file_ = nullptr;
  // For only writing shaders to the currently open storage once, incremented
  // when switching the storage.
  uint32_t shader_storage_index_ = 0;
  bool shader_storage_file_flush_needed_ = false;

  // Pipeline storage output stream, for preload in the next emulator runs.
  FILE* pipeline_storage_file_ = nullptr;
  bool pipeline_storage_file_flush_needed_ = false;

  // Thread for asynchronous writing to the storage streams.
  void StorageWriteThread();
  std::mutex storage_write_request_lock_;
  std::condition_variable storage_write_request_cond_;
  // Storage thread input is protected with storage_write_request_lock_, and the
  // thread is notified about its change via storage_write_request_cond_.
  std::deque<const Shader*> storage_write_shader_queue_;
  std::deque<PipelineStoredDescription> storage_write_pipeline_queue_;
  bool storage_write_flush_shaders_ = false;
  bool storage_write_flush_pipelines_ = false;
  bool storage_write_thread_shutdown_ = false;
  std::unique_ptr<rex::thread::Thread> storage_write_thread_;

  mutable std::mutex creation_request_lock_;
  std::condition_variable creation_request_cond_;
  std::priority_queue<PipelineCreationArguments, std::vector<PipelineCreationArguments>,
                      PipelineCreationArgumentsPriorityComparator>
      creation_queue_;
  size_t creation_threads_busy_ = 0;
  bool startup_loading_ = false;
  std::unique_ptr<rex::thread::Event> creation_completion_event_;
  bool creation_completion_set_event_ = false;
  size_t creation_threads_shutdown_from_ = SIZE_MAX;
  std::vector<std::unique_ptr<rex::thread::Thread>> creation_threads_;
};

}  // namespace rex::graphics::vulkan
