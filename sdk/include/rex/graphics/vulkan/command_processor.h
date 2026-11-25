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

#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <map>
#include <vector>

#include <rex/assert.h>
#include <rex/graphics/command_processor.h>
#include <rex/graphics/d3d_feed.h>  // wwe2k14 patch (Session 87, stage 3): D3dFeedDrawRegs
#include <rex/graphics/pipeline/shader/interpreter.h>  // wwe2k14 patch (Session 40, RENAME)
#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/vulkan/deferred_command_buffer.h>
#include <rex/graphics/vulkan/graphics_system.h>
#include <rex/graphics/vulkan/pipeline_cache.h>
#include <rex/graphics/vulkan/primitive_processor.h>
#include <rex/graphics/vulkan/render_target_cache.h>
#include <rex/graphics/vulkan/shader.h>
#include <rex/graphics/vulkan/shared_memory.h>
#include <rex/graphics/vulkan/texture_cache.h>
#include <rex/graphics/xenos.h>
#include <rex/hash.h>
#include <rex/system/kernel_state.h>
#include <rex/ui/vulkan/linked_type_descriptor_set_allocator.h>
#include <rex/ui/vulkan/presenter.h>
#include <rex/ui/vulkan/provider.h>
#include <rex/ui/vulkan/queue_submit_order.h>
#include <rex/ui/vulkan/upload_buffer_pool.h>

namespace rex::graphics::vulkan {

// wwe2k14 patch (Session 38, NATIVE): vulkan/native/native_renderer.h.
namespace native {
class NativeRenderer;
struct Draw;
}  // namespace native

// wwe2k14 patch (Session 92, M3a): the words a resolve (IssueCopy) works from - the registers' values from the register source, or a resolve
// record's words. No data member of the command processor; a plain value type.
struct CopyWords {
  uint32_t copy_control = 0, dest_base = 0, dest_pitch = 0, dest_info = 0;
  uint32_t surface_info = 0, color_info = 0, depth_info = 0, mode_control = 0;
};

// wwe2k14 patch (Session 92, M4f-B): the inputs of NativeCoverRect / NativeBoundsRect / the interpreter memo keys, in one place. Filled from the register
// source (FromRegisters: the same values the functions read before) or from the typed state (owned_front_draw.cpp). A plain value type; no data member
// of the command processor. The constant arrays use the register layout: floats = 512 vec4 (vertex 0-255, pixel 256-511), bool_loop = 8 bool words + 32
// loop words, fetch = 32 fetch constants of 6 words (a vertex fetch constant n = words 2n, 2n+1).
struct CoverInputs {
  const RegisterFile* interp_regs = nullptr;  // the CPU shader interpreter's register source (stock: the draw's; typed: a scratch filled from the typed arrays)
  reg::VGT_DRAW_INITIATOR initiator{};
  reg::PA_SU_SC_MODE_CNTL sc_mode{};
  reg::VGT_MULTI_PRIM_IB_RESET_INDX reset_indx{};
  reg::RB_SURFACE_INFO surface_info{};
  reg::RB_COLORCONTROL color_control{};
  reg::PA_SC_WINDOW_OFFSET window_offset{};
  reg::PA_CL_VTE_CNTL vte{};
  reg::PA_CL_CLIP_CNTL clip{};
  reg::PA_SU_VTX_CNTL su_vtx_cntl{};
  reg::VGT_INDX_OFFSET indx_offset{};
  reg::VGT_MIN_VTX_INDX min_indx{};
  reg::VGT_MAX_VTX_INDX max_indx{};
  float vport[4]{};  // x scale, x offset, y scale, y offset
  draw_util::Scissor scissor{};
  uint32_t vs_const = 0;  // SQ_VS_CONST
  const uint32_t* floats = nullptr;
  const uint32_t* bool_loop = nullptr;
  const uint32_t* fetch = nullptr;
  const uint8_t* fetch_memory = nullptr;  // (Session 95, item P) an inline draw on the vertex ring: the vertex fetch addresses are offsets in this block; nullptr = guest memory
  static CoverInputs FromRegisters(const RegisterFile& regs);
};

class VulkanCommandProcessor : public CommandProcessor {
 public:
  // WWE '13 port: copies guest memory [address, address + length) from the GPU's shared memory into the CPU's copy, now
  // (command thread; waits for the GPU). For game code that reads render-to-texture results on the CPU without a
  // read-back (Attitude Era Shawn Michaels' attire: XGUntileSurface). False when it could not (scaled resolves).
  bool ReadbackRangeForCpuNow(uint32_t address, uint32_t length);
  // wwe2k14 patch (Session 87, D3D layer F2): the copy-family record executor (native/d3d_copy_records.cpp). A friend
  // declaration only - no member, no layout change.
  friend struct D3dCopyAccess;
  // wwe2k14 patch (Session 87, D3D layer F4): the direct draw handler (native/d3d_direct.cpp). A friend declaration and one
  // non-virtual accessor (the dynamic offsets live in a type defined only in command_processor.cpp) - no data member, no
  // layout change.
  friend struct D3dDirectAccess;
  friend struct DiagAccess;  // wwe2k14 patch (Session 95, the diagnostic run: native/diag_layer.cpp). A friend declaration only.
  const uint32_t* D3dDirectDynamicOffsets(uint32_t& count) const;
  uint64_t OwnedConstantsHash() const;  // wwe2k14 patch (Session 95, gpu_owned_restore_skip): the bytes the last UpdateBindings gave the draw
  const uint8_t* OwnedConstantsBytes(uint32_t buffer, size_t& size) const;  // (the same, one constant buffer)
  // wwe2k14 patch (Session 92, M4f-A): the typed draw writes system_constants_ itself; the stored inputs of the skip cache no longer describe them.
  void D3dTypedSystemConstantsBegin();
  // wwe2k14 patch (Session 92, M4b-M4d: the record-fed front end, native/owned_front_*.cpp): the typed draw builder reaches
  // the processor's private members through this access struct. A friend declaration only - no member, no layout change.
  friend struct OwnedFrontAccess;
  // wwe2k14 patch (Session 87, D3D layer stage 3, native/d3d_feed.cpp): the record-fed draw. DrawRegs() is the register
  // source of the per-draw derivations (system constants, constants upload, repeat path, cover / bounds rectangles): the
  // record's register image while such a draw runs (D3dFeedDrawRegs(), command processor thread only), the register file
  // otherwise. A friend declaration and a non-virtual inline function - no data member, no layout change.
  friend struct D3dFeedAccess;
  const RegisterFile& DrawRegs() const {
    REX_CUT_PROBE(kDrawRegs);
    const RegisterFile* const image = D3dFeedDrawRegs();
    return image ? *image : *register_file_;
  }
  // wwe2k14 patch (Session 23, MERGE): vulkan_repeat_draw_merge - the instanced float constant stride (vec4s) for
  // this vertex shader's kVertex translations (SpirvShaderTranslator::Modification::vertex), 0 = not instanced.
  uint32_t RepeatMergeStrideVec4(const Shader& shader) const;
  // wwe2k14 patch (Session 24, OBJMV): gpu_dlss_objmv - this vertex shader's kVertex translations capture
  // gl_Position with transform feedback (SpirvShaderTranslator::Modification::vertex.xfb_position).
  bool ObjmvCaptureShader(const Shader& shader) const;
  bool ObjmvLatched() const { return objmv_latched_; }
  // wwe2k14 patch (Session 66, PS spec): whether the native renderer is on (the pixel shader specialisation is NATIVE's).
  bool NativeRendererEnabled() const;
  // wwe2k14 patch (Session 72, gpu_native_shader_swap): native/shader_swap.h.
  bool RegisterNativeShaderSwap(uint64_t guest_pixel_shader_hash, const uint32_t* words, size_t word_count,
                                const uint8_t* float_registers, size_t float_register_count,
                                const char* name, uint32_t flags) override;
  // wwe2k14 patch (Session 87, stage 4, gpu_native_vs_swap): native/shader_swap.h (RegisterVertex).
  bool RegisterNativeVertexSwap(uint64_t guest_vertex_shader_hash, const uint32_t* words, size_t word_count,
                                const uint8_t* float_registers, size_t float_register_count, const char* name,
                                uint32_t flags, uint32_t stream_stride_words, uint32_t stream_fetch_constant) override;
  // wwe2k14 patch (Session 72 part 4, gpu_native_truth_skip_restores): CommandProcessor's.
  bool RegisterNativeRestoreShader(uint64_t guest_pixel_shader_hash, uint32_t color_fetch, uint32_t depth_fetch,
                                   const char* name) override;
  // wwe2k14 patch (Session 24, DLSS): gpu_dlss_lod_bias for this draw's pixel shader (scene draws only), in quarters.
  uint32_t DlssLodBiasQuarters() const;

  // Single-descriptor layouts for use within a single frame.
  enum class SingleTransientDescriptorLayout {
    kStorageBufferCompute,
    kStorageBufferPairCompute,
    kCount,
  };

  class ScratchBufferAcquisition {
   public:
    explicit ScratchBufferAcquisition() = default;
    explicit ScratchBufferAcquisition(VulkanCommandProcessor& command_processor, VkBuffer buffer,
                                      VkPipelineStageFlags stage_mask, VkAccessFlags access_mask,
                                      uint32_t ring_index = UINT32_MAX)
        : command_processor_(&command_processor),
          buffer_(buffer),
          stage_mask_(stage_mask),
          access_mask_(access_mask),
          ring_index_(ring_index) {}

    ScratchBufferAcquisition(const ScratchBufferAcquisition& acquisition) = delete;
    ScratchBufferAcquisition& operator=(const ScratchBufferAcquisition& acquisition) = delete;

    ScratchBufferAcquisition(ScratchBufferAcquisition&& acquisition) {
      command_processor_ = acquisition.command_processor_;
      buffer_ = acquisition.buffer_;
      stage_mask_ = acquisition.stage_mask_;
      access_mask_ = acquisition.access_mask_;
      ring_index_ = acquisition.ring_index_;
      acquisition.command_processor_ = nullptr;
      acquisition.buffer_ = VK_NULL_HANDLE;
      acquisition.stage_mask_ = 0;
      acquisition.access_mask_ = 0;
      acquisition.ring_index_ = UINT32_MAX;
    }
    ScratchBufferAcquisition& operator=(ScratchBufferAcquisition&& acquisition) {
      if (this == &acquisition) {
        return *this;
      }
      command_processor_ = acquisition.command_processor_;
      buffer_ = acquisition.buffer_;
      stage_mask_ = acquisition.stage_mask_;
      access_mask_ = acquisition.access_mask_;
      ring_index_ = acquisition.ring_index_;
      acquisition.command_processor_ = nullptr;
      acquisition.buffer_ = VK_NULL_HANDLE;
      acquisition.stage_mask_ = 0;
      acquisition.access_mask_ = 0;
      acquisition.ring_index_ = UINT32_MAX;
      return *this;
    }

    ~ScratchBufferAcquisition() {
      // wwe2k14 patch (Session 7c, TEXLOAD): an entry of the texture load scratch ring (AcquireScratchGpuBufferRing)
      // keeps its own last usage.
      if (buffer_ != VK_NULL_HANDLE && ring_index_ != UINT32_MAX) {
        ScratchRingEntry& entry = command_processor_->scratch_ring_[ring_index_];
        assert_true(entry.used);
        assert_true(entry.buffer == buffer_);
        entry.last_stage_mask = stage_mask_;
        entry.last_access_mask = access_mask_;
        entry.last_usage_submission = command_processor_->GetCurrentSubmission();
        entry.used = false;
        return;
      }
      if (buffer_ != VK_NULL_HANDLE) {
        assert_true(command_processor_->scratch_buffer_used_);
        assert_true(command_processor_->scratch_buffer_ == buffer_);
        command_processor_->scratch_buffer_last_stage_mask_ = stage_mask_;
        command_processor_->scratch_buffer_last_access_mask_ = access_mask_;
        command_processor_->scratch_buffer_last_usage_submission_ =
            command_processor_->GetCurrentSubmission();
        command_processor_->scratch_buffer_used_ = false;
      }
    }

    // VK_NULL_HANDLE if failed to acquire or if moved.
    VkBuffer buffer() const { return buffer_; }
    // wwe2k14 patch (Session 7c, TEXLOAD): the scratch ring entry, UINT32_MAX for the single scratch buffer.
    uint32_t ring_index() const { return ring_index_; }

    VkPipelineStageFlags GetStageMask() const { return stage_mask_; }
    VkPipelineStageFlags SetStageMask(VkPipelineStageFlags new_stage_mask) {
      VkPipelineStageFlags old_stage_mask = stage_mask_;
      stage_mask_ = new_stage_mask;
      return old_stage_mask;
    }
    VkAccessFlags GetAccessMask() const { return access_mask_; }
    VkAccessFlags SetAccessMask(VkAccessFlags new_access_mask) {
      VkAccessFlags old_access_mask = access_mask_;
      access_mask_ = new_access_mask;
      return old_access_mask;
    }

   private:
    VulkanCommandProcessor* command_processor_ = nullptr;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkPipelineStageFlags stage_mask_ = 0;
    VkAccessFlags access_mask_ = 0;
    uint32_t ring_index_ = UINT32_MAX;  // wwe2k14 patch (Session 7c, TEXLOAD): the scratch ring entry, or none
  };

  VulkanCommandProcessor(VulkanGraphicsSystem* graphics_system, system::KernelState* kernel_state);
  ~VulkanCommandProcessor();

  void ClearCaches() override;
  void InvalidateGpuMemory() override;
  void InitializeShaderStorage(const std::filesystem::path& cache_root, uint32_t title_id,
                               bool blocking) override;

  // wwe2k14 patch (Session 93, the per-draw storage window): whether the shared memory reaches the shaders as ONE storage buffer
  // window per draw (latched in SetupContext, before the pipeline cache; see vulkan_shared_memory_window).
  bool SharedMemoryWindowOn() const { return window_on_; }

  ui::vulkan::VulkanDevice* GetVulkanDevice() const {
    return static_cast<const ui::vulkan::VulkanProvider*>(graphics_system_->provider())
        ->vulkan_device();
  }

  bool CompileGlslToSpirv(VkShaderStageFlagBits stage, std::string_view source,
                          std::vector<uint32_t>& spirv_out, std::string& error_out) const;

  // Returns the deferred drawing command list for the currently open
  // submission.
  // wwe2k14 patch (Session 21, HOIST): the current submission's hoist stream (vulkan_upload_hoist 2), which runs
  // before the deferred command buffer's commands in the same command buffer. The first call in a submission
  // records its opening barrier (all earlier queue work -> transfer); EndSubmission closes it (transfer -> all).
  DeferredCommandBuffer& BeginHoistedUpload();
  DeferredCommandBuffer& deferred_command_buffer() {
    assert_true(submission_open_);
    return deferred_command_buffer_;
  }

  bool submission_open() const { return submission_open_; }
  uint64_t GetCurrentSubmission() const {
    return submission_completed_ + uint64_t(submissions_in_flight_fences_.size()) + 1;
  }
  uint64_t GetCompletedSubmission() const { return submission_completed_; }
  // wwe2k14 patch (Session 7b, UPSKIP review): how many async submissions had their commands dropped (SUBMITFIX:
  // recording failed, or vkQueueSubmit kept failing) or abandoned - their recorded uploads never ran. Increased on
  // the submit thread before the submission's fence can be signaled. Any thread.
  uint64_t GetAsyncSubmitCommandsDropped() const {
    return async_submit_commands_dropped_.load(std::memory_order_acquire);
  }

  // Sparse binds are:
  // - In a single submission, all submitted in one vkQueueBindSparse.
  // - Sent to the queue without waiting for a semaphore.
  // Thus, multiple sparse binds between the completed and the current
  // submission, and within one submission, must not touch any overlapping
  // memory regions.
  void SparseBindBuffer(VkBuffer buffer, uint32_t bind_count, const VkSparseMemoryBind* binds,
                        VkPipelineStageFlags wait_stage_mask);

  uint64_t GetCurrentFrame() const { return frame_current_; }
  uint64_t GetCompletedFrame() const { return frame_completed_; }

  // Submission must be open to insert barriers. If no pipeline stages access
  // the resource in a synchronization scope, the stage masks should be 0 (top /
  // bottom of pipe should be specified only if explicitly needed). Returning
  // true if the barrier has actually been inserted and not dropped.
  bool PushBufferMemoryBarrier(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                               VkPipelineStageFlags src_stage_mask,
                               VkPipelineStageFlags dst_stage_mask, VkAccessFlags src_access_mask,
                               VkAccessFlags dst_access_mask,
                               uint32_t src_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
                               uint32_t dst_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
                               bool skip_if_equal = true);
  bool PushImageMemoryBarrier(VkImage image, const VkImageSubresourceRange& subresource_range,
                              VkPipelineStageFlags src_stage_mask,
                              VkPipelineStageFlags dst_stage_mask, VkAccessFlags src_access_mask,
                              VkAccessFlags dst_access_mask, VkImageLayout old_layout,
                              VkImageLayout new_layout,
                              uint32_t src_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
                              uint32_t dst_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
                              bool skip_if_equal = true);
  // wwe2k14 patch (Session 7b, BARNARROW): the union of the source stage masks of the barriers pushed but not
  // submitted yet (measurement only: which stages the next SubmitBarriers waits for).
  VkPipelineStageFlags GetPendingBarriersSrcStageMask() const {
    VkPipelineStageFlags src_stage_mask = current_pending_barrier_.src_stage_mask;
    for (const PendingBarrier& pending_barrier : pending_barriers_) {
      src_stage_mask |= pending_barrier.src_stage_mask;
    }
    return src_stage_mask;
  }
  // Returns whether any barriers have been submitted - if true is returned, the
  // render pass will also be closed.
  bool SubmitBarriers(bool force_end_render_pass);

  // If not started yet, begins a render pass from the render target cache.
  // Submission must be open.
  void SubmitBarriersAndEnterRenderTargetCacheRenderPass(
      VkRenderPass render_pass, const VulkanRenderTargetCache::Framebuffer* framebuffer);
  // Overload for transfer operations with dynamic rendering.
  void SubmitBarriersAndEnterRenderTargetCacheRenderPass(
      VkRenderPass render_pass, const VulkanRenderTargetCache::Framebuffer* framebuffer,
      VkImageView transfer_dest_view, bool transfer_dest_is_depth);
  // Must be called before doing anything outside the render pass scope,
  // including adding pipeline barriers that are not a part of the render pass
  // scope. Submission must be open.
  void EndRenderPass();

  VkDescriptorSetLayout GetSingleTransientDescriptorLayout(
      SingleTransientDescriptorLayout transient_descriptor_layout) const {
    return descriptor_set_layouts_single_transient_[size_t(transient_descriptor_layout)];
  }
  // A frame must be open.
  VkDescriptorSet AllocateSingleTransientDescriptor(
      SingleTransientDescriptorLayout transient_descriptor_layout);

  // The returned reference is valid until a cache clear.
  VkDescriptorSetLayout GetTextureDescriptorSetLayout(bool is_vertex, size_t texture_count,
                                                      size_t sampler_count);
  // The returned reference is valid until a cache clear.
  const VulkanPipelineCache::PipelineLayoutProvider* GetPipelineLayout(size_t texture_count_pixel,
                                                                       size_t sampler_count_pixel,
                                                                       size_t texture_count_vertex,
                                                                       size_t sampler_count_vertex);

  // Returns a single temporary GPU-side buffer within a submission for tasks
  // like texture untiling and resolving. May push a buffer memory barrier into
  // the initial usage. Submission must be open.
  ScratchBufferAcquisition AcquireScratchGpuBuffer(VkDeviceSize size,
                                                   VkPipelineStageFlags initial_stage_mask,
                                                   VkAccessFlags initial_access_mask);
  // wwe2k14 patch (Session 7c, TEXLOAD): vulkan_texture_load_scratch_ring. Like AcquireScratchGpuBuffer, but from a
  // ring of kScratchRingSize buffers of its own (each grown as needed), so consecutive texture loads write
  // different buffers and the dispatch of one load doesn't have to wait for the copy of the previous one. Each
  // entry tracks its own last usage (stages, accesses, submission). When an entry needs a barrier into the
  // initial usage, the other idle entries whose last usage needs the same kind of barrier get it in the same
  // pipeline barrier (no extra execution dependency), so their next acquisitions need none. Only one acquisition
  // of the whole ring at a time is expected (texture loads); submission must be open.
  ScratchBufferAcquisition AcquireScratchGpuBufferRing(VkDeviceSize size,
                                                       VkPipelineStageFlags initial_stage_mask,
                                                       VkAccessFlags initial_access_mask);
  // Frees the ring's buffers (once their last submission has completed), for when the setting is off.
  void ReleaseScratchRing();
  static constexpr uint32_t kScratchRingSize = 4;
  // Cumulative counts (command processor thread only), for the [s7][TEXLOAD] summary.
  struct ScratchRingStats {
    uint64_t acquisitions = 0;
    uint64_t barriers = 0;          // acquisitions that pushed a barrier for their own entry
    uint64_t pre_transitions = 0;   // idle entries transitioned along with such a barrier
    uint64_t grows = 0;             // buffers (re)created because an entry was too small
    uint64_t releases = 0;          // ReleaseScratchRing calls that freed something
  };
  const ScratchRingStats& scratch_ring_stats() const { return scratch_ring_stats_; }
  VkDeviceSize GetScratchRingSize() const {
    VkDeviceSize size = 0;
    for (const ScratchRingEntry& entry : scratch_ring_) {
      size += entry.size;
    }
    return size;
  }

  // wwe2k14 patch (Session 7c, TEXLOAD): GPU marks inside texture loads (gpu_anatomy_gpu_marks, measurement only),
  // and the frame-anatomy row a texture load belongs to. Marks: kBegin before the load's first command, kDispatch
  // after its untiling dispatches, kConvert after the float conversion pass, kCopy after the copy to the image.
  enum class TextureLoadMark : uint8_t { kBegin, kDispatch, kConvert, kCopy };
  bool gpu_marks_active() const { return gpu_marks_active_; }
  void GpuMarkTextureLoad(TextureLoadMark mark);
  // The frame-anatomy row index of the IssueDraw call in progress (between calls: the index the next row will get),
  // while a capture is running.
  uint32_t anatomy_row() const { return anatomy_gpu_row_; }

  // Binds a graphics pipeline for host-specific purposes, invalidating the
  // affected state. keep_dynamic_* must be false (to invalidate the dynamic
  // state after binding the pipeline with the same state being static, or if
  // the caller changes the dynamic state bypassing the VulkanCommandProcessor)
  // unless the caller has these state variables as dynamic and uses the
  // tracking in VulkanCommandProcessor to modify them.
  void BindExternalGraphicsPipeline(VkPipeline pipeline, bool keep_dynamic_depth_bias = false,
                                    bool keep_dynamic_blend_constants = false,
                                    bool keep_dynamic_stencil_mask_ref = false);
  void BindExternalComputePipeline(VkPipeline pipeline);
  void SetViewport(const VkViewport& viewport);
  void SetScissor(const VkRect2D& scissor);

  // Returns the text to display in the GPU backend name in the window title.
  std::string GetWindowTitleText() const;

  // wwe2k14 patch (Session 7c, CAUSES): running counts of what this command processor and its caches record,
  // for the frame-anatomy cause columns (AnatomyRowCauses), the cpf_sync / cpf_sync2 timeline events and the
  // [s7][CAUSES] summary. Measurement only: plain increments where the work is recorded, nothing else changes.
  // Command processor thread only. Never reset (readers take differences, which wrap correctly), except the two
  // stage masks, which are ORed since the cause window's start. Dispatches and render pass begins are counted by the
  // deferred command buffer (dispatches_recorded, rendering_begins_recorded).
  struct CauseCounts {
    uint32_t up_calls = 0;            // VulkanSharedMemory::UploadRanges calls that record copies
    uint32_t up_pages = 0;            // pages they upload
    uint32_t tex_loads = 0;           // VulkanTextureCache loads recorded (dispatches + copy to the image,
                                      // or one dispatch into the image - TEXDIRECT vulkan_texture_load_direct)
    uint32_t tex_loads_scaled = 0;    // of those, from the resolution-scaled resolve buffer
    uint64_t tex_load_bytes = 0;      // bytes of their source ranges
    uint32_t barrier_batches = 0;     // SubmitBarriers calls that recorded barriers, + barriers recorded directly
    uint32_t fsi_edram_barriers = 0;  // full-EDRAM-buffer self-barriers pushed by EDRAM write commits
    uint32_t desc_writes = 0;         // descriptor sets allocated or written
    uint32_t pipelines_created = 0;   // guest graphics pipelines created or queued (ConfigurePipeline, on success)
    uint32_t pipelines_pending = 0;   // draws skipped because their pipeline was still being created
    uint32_t pipeline_binds = 0;      // guest graphics pipeline binds that changed the bound pipeline
    VkPipelineStageFlags barrier_src_stages = 0;  // OR of the recorded barriers' stage masks in the window
    VkPipelineStageFlags barrier_dst_stages = 0;
  };
  CauseCounts& cause_counts() { return cause_counts_; }
  // wwe2k14 patch (Session 7c, CAUSES): gpu_anatomy_gpu_marks_mode 1, called by the deferred command buffer's
  // event hook (DeferredCommandBuffer::SetEventHook) right before the first event command after an unmarked row.
  void GpuMarksSparseBeforeEvent();

  // wwe2k14 patch (Session 39, NATIVE): for the native renderer's load and write-back: the EDRAM buffer (the
  // fragment shader interlock path only; VK_NULL_HANDLE otherwise) and the draw resolution scale.
  VkBuffer NativeEdramBuffer() const;
  void NativeResolutionScale(uint32_t& x, uint32_t& y) const;
  // wwe2k14 patch (Session 54, NATIVE draw sizes): the native renderer's truth pass draw ratio (1 = none; decided once).
  float NativeTruthDrawRatio();
  // wwe2k14 patch (Session 61, gpu_native_pipeline_storage): the pipeline cache, for the native renderer's stored
  // pipelines (their shaders by hash, translations, geometry shaders).
  VulkanPipelineCache& NativePipelineCache() { return *pipeline_cache_; }
  // wwe2k14 patch (Session 61, DLSS / DLAA object motion in truth frames): transform feedback around the next draw
  // recorded (a truth draw's capture of its positions; the emulated capture uses the same).
  void NativeXfbBegin(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size);
  void NativeXfbEnd();
  // wwe2k14 patch (Session 54): the native renderer bound its own pipeline, sets and dynamic state outside a draw.
  void NativeDropBindings() { NativeStateTouched(false); }
  // wwe2k14 patch (Session 45, NATIVE FRAME): the texture cache's running texture load counts (the frame census).
  void NativeTextureLoadCounters(uint64_t& loads, uint64_t& resolved_loads, uint64_t& resolved_bytes) const;
  // wwe2k14 patch (Session 46, NATIVE TRUTH M0): the render target cache's running EDRAM access counts (uses, fills,
  // resolve copies reading it, resolve clears), for the EDRAM census; zeros without a render target cache.
  void NativeEdramCounters(uint64_t counts[4]) const;
  // wwe2k14 patch (Session 46, NATIVE TRUTH): in a truth frame, the set the resolve's copy reads instead of the EDRAM
  // (NativeRenderer::TruthResolveSource); VK_NULL_HANDLE otherwise. And the render target cache's set over a buffer.
  VkDescriptorSet NativeTruthResolveSource(const draw_util::ResolveInfo& info);
  bool NativeTruthResolveDirect(const draw_util::ResolveInfo& info, const draw_util::ResolveCopyShaderConstants& constants,
                                VkDescriptorSet dest_set, uint32_t group_count_x, uint32_t group_count_y);
  VkDescriptorSet NativeResolveSetForBuffer(VkBuffer buffer);
  VkDeviceSize NativeEdramSizeBytes() const;
  // wwe2k14 patch (Session 45, NATIVE RESOLVE): see NativeRenderer::ResolveFromTarget (the render target cache's
  // resolve asks before its fast 32bpp dispatch).
  bool NativeResolveFromTarget(const draw_util::ResolveInfo& info,
                               const draw_util::ResolveCopyShaderConstants& constants, VkDescriptorSet dest_set,
                               uint32_t group_count_x, uint32_t group_count_y);
  // wwe2k14 patch (Session 45, NATIVE DEFER): a stock resolve copy is about to read the EDRAM.
  void NativeBeforeStockResolve();
  // wwe2k14 patch (Session 45, NATIVE TEXWRITE N3b): the stock fast copy by the native renderer's shader, filling the
  // texture it feeds (NativeRenderer::ResolveTextureOnly); false: dispatch the stock copy.
  bool NativeResolveTextureOnly(const draw_util::ResolveInfo& info,
                                const draw_util::ResolveCopyShaderConstants& constants, VkDescriptorSet dest_set,
                                uint32_t group_count_x, uint32_t group_count_y);
  // wwe2k14 patch (Session 45, NATIVE TEXWRITE): the texture cache's part (VulkanTextureCache::NativeTextureForResolve
  // / NativeTextureResolved), and the render target cache's call after the resolve's MarkRangeAsResolved.
  // (Session 59: `format` k_16_16 / k_16_16_16_16 too - their R32_UINT / R16G16B16A16_UINT storage views.)
  // (Session 79: allow_layer - a cube face, gpu_native_truth_fill_cube_faces.)
  VkDescriptorSet NativeTextureForResolve(uint32_t base_page, uint32_t pitch_div_32, uint32_t endianness,
                                          uint32_t& width_out, uint32_t& height_out, void*& handle_out, bool allow_scaled = false,
                                          xenos::TextureFormat format = xenos::TextureFormat::k_8_8_8_8,
                                          bool allow_layer = false, int32_t* layer_out = nullptr);
  void NativeTextureResolved(void* handle);
  // wwe2k14 patch (Session 47, NATIVE TRUTH M6): VulkanTextureCache::NativeTextureForCopy; and, after a resolve's copy
  // in a truth frame (before MarkRangeAsResolved), the native renderer fills the texture it feeds from its image.
  bool NativeTextureForCopy(uint32_t base_page, uint32_t pitch_div_32, uint32_t endianness, xenos::TextureFormat format,
                            VkImage& image_out, uint32_t& width_out, uint32_t& height_out, uint32_t& level_out,
                            bool& has_mips_out, void*& handle_out, const char*& why, uint32_t storage_bytes = 0,
                            VkDescriptorSet* storage_set_out = nullptr);
  void NativeTruthFillTexture(const draw_util::ResolveInfo& info);
  // wwe2k14 patch (Session 49, RAM TIES: gpu_ram_stats): NativeRenderer::TruthTimer / TruthNoteWriteBack.
  void NativeTruthTimer(uint32_t kind, bool begin);
  void NativeTruthNoteWriteBack(bool direct, uint32_t bytes);
  // wwe2k14 patch (Session 49, gpu_native_truth_skip_writeback): the render target cache's resolve calls
  // (NativeRenderer::TruthBeforeResolveWrite / TruthSetDest / TruthFillOnly / TruthAfterResolveMarked), and what the
  // native renderer needs for texture-only resolves (the texture cache's readable textures and pins, the shared
  // memory's buffer and skipped pages).
  void NativeTruthBeforeResolveWrite(const draw_util::ResolveInfo& info);
  void NativeTruthSetDest(uint64_t offset, uint32_t range);  // (Session 56 review: 64-bit, scaled offsets)
  // wwe2k14 patch (Session 52, gpu_native_truth_defer_dest_use): a truth-frame 1x resolve's compute-write use of the
  // shared memory, held until the dispatch that writes the destination is recorded (Flush); a texture-only resolve
  // writes no memory and drops it (Drop) - no barrier before it, none before the next draw.
  void NativeHoldResolveDestUse(uint32_t start, uint32_t length) {
    held_dest_use_ = true;
    held_dest_scaled_ = false;
    held_dest_start_ = start;
    held_dest_length_ = length;
  }
  // wwe2k14 patch (Session 59, gpu_native_truth_defer_dest_use_scaled): the same for a scaled resolve's write of the
  // scaled resolve buffer (VulkanTextureCache::UseScaledResolveBufferForWrite at Flush).
  void NativeHoldScaledResolveDestUse(uint64_t start, uint64_t length) {
    held_dest_use_ = true;
    held_dest_scaled_ = true;
    held_dest_scaled_start_ = start;
    held_dest_scaled_length_ = length;
  }
  void NativeFlushResolveDestUse();
  // wwe2k14 patch (Session 52, gpu_native_truth_cover_outdated): VulkanTextureCache::NativeSetCover.
  void NativeSetTextureCover(uint32_t x, uint32_t y, uint32_t w, uint32_t h);
  // wwe2k14 patch (Session 52, gpu_native_truth_batch_resolves): NativeRenderer::TruthFlushDeferred (a texture a native
  // compute write left in kComputeStorageWrite is about to be sampled).
  // (Session 80: `texture` - the one about to be used; null: any.)
  void NativeFlushDeferredResolves(void* texture = nullptr);
  // (Session 80, gpu_native_truth_serve_copies: NativeRenderer::TruthServeView, with the draw's target registers.)
  VkImageView NativeServeView(void* texture, uint32_t width, uint32_t height, VkFormat format,
                              const VkComponentMapping& components);
  // wwe2k14 patch (Session 52): VulkanTextureCache::native_hold_barrier_ / native_held_barrier_. Take returns whether a
  // barrier was held (and clears it); Push records a held one as the texture cache would have.
  void NativeHoldTextureBarrier(bool hold);
  bool NativeTakeHeldTextureBarrier(VkImage& image, VkPipelineStageFlags& src_stage, VkAccessFlags& src_access,
                                    VkImageLayout& old_layout);
  void NativeDropResolveDestUse() { held_dest_use_ = false; }
  bool held_dest_use_ = false;
  bool held_dest_scaled_ = false;  // wwe2k14 patch (Session 59)
  uint32_t held_dest_start_ = 0, held_dest_length_ = 0;
  uint64_t held_dest_scaled_start_ = 0, held_dest_scaled_length_ = 0;  // wwe2k14 patch (Session 59)
  void NativeTruthBeforeMemoryWrite(const draw_util::ResolveInfo& info);
  bool NativeTruthFillOnly(const draw_util::ResolveInfo& info, const draw_util::ResolveCopyShaderConstants& constants);
  bool NativeTruthResolveScaledFull(const draw_util::ResolveInfo& info, VkDescriptorSet dest_set);  // Session 52
  bool NativeTruthResolveFull1x(const draw_util::ResolveInfo& info, uint32_t dest_base, uint32_t dest_coordinate_info,
                                uint32_t dest_info, VkDescriptorSet dest_set);  // Session 59
  void NativeTruthAfterResolveMarked(const draw_util::ResolveInfo& info);
  // wwe2k14 patch (Session 72 part 4): NativeRenderer::TruthResolveRedundant (false without a native renderer).
  bool NativeTruthResolveRedundant(const draw_util::ResolveInfo& info);
  // wwe2k14 patch (Session 80, gpu_native_own_steps): NativeRenderer::OwnedTakeResolve (the copy of an owned step's
  // output: left out, the page is our image's) and OwnedView (a binding of such a page: our image's view, or null).
  bool NativeOwnedResolve(const draw_util::ResolveInfo& info, bool copy_known);
  // (is_signed: the fetch's signed view; servable: false for a binding the substitution cannot take - 3D, an array,
  // mips - refused, the page's producers not owned any more.)
  bool NativeOwnedPageFree(uint32_t base_page, uint32_t width, uint32_t height, uint32_t format);  // wwe2k14 patch (Session 95, r6tex)
  bool NativeOwnedReadAgain(uint32_t memo_index, uint32_t base_page, VkImageView view);  // wwe2k14 patch (Session 95, r6tex)
  uint32_t NativeOwnedViewMemoLast() const;
  VkImageView NativeOwnedView(uint32_t base_page, uint32_t width, uint32_t height, uint32_t format, uint32_t endian,
                              bool scaled_ok, const VkComponentMapping& components, bool is_signed = false,
                              bool servable = true, bool has_mips = false, bool is_cube = false);
  bool NativeResolveFace(uint32_t face_page, uint32_t endianness, xenos::TextureFormat format,
                         uint32_t& texture_page_out, uint32_t& width_out, uint32_t& height_out, uint32_t& layers_out,
                         uint32_t& layer_out, bool& scaled_out, uint32_t& extent_bytes_out);
  // (Session 82, P2, G4: where a copy into a texture of a known key lands - its base page, level 0 size, level count, the
  // level and whether it is a scaled texture; false when no load of the texture was seen. No texture state is touched.)
  bool NativeResolveTarget(uint32_t base_page, uint32_t pitch_div_32, uint32_t endianness, xenos::TextureFormat format,
                           uint32_t& texture_page_out, uint32_t& width_out, uint32_t& height_out, uint32_t& levels_out,
                           uint32_t& level_out, bool& scaled_out, uint32_t& extent_bytes_out);
  // (Session 82, the owned match frame, P2: the swap's source texture - noted every swap; an owned page's view if one
  // stands for it. A non-virtual member: the class's layout and virtuals are unchanged.)
  VkImageView NativeOwnedSwapView(uint32_t base_page, uint32_t width, uint32_t height, uint32_t format,
                                  uint32_t endian, bool scaled, const VkComponentMapping& components, bool servable);
  // (The owned pages changed: the texture cache's unchanged-bindings shortcut looks again - review.)
  void NativeOwnedPagesChanged();
  // (Whether resolves are read back by the CPU now - the readback mode, readback_resolve_override first.)
  bool NativeReadbackActive() const;
  // Session 95 r5 (non-virtual, no data; gpu_native_readback_quiet_frames): a read-back window that still holds the owned frame's lean paths back: open, and a copy was really read back lately.
  bool NativeReadbackHolds() const;
  // Session 82 (P2): a cube the texture cache has just learned from a load: the owned frame checks the cubes it made of its own
  // copies before the first sample against it (owned_feedback.cpp, OwnedCubeLearned).
  void NativeOwnedCubeLearned(uint32_t base_page, uint32_t width, uint32_t height, uint32_t faces, uint32_t stride_bytes);
  // Session 82 (P2, review): the owned copy just served was a derived cube face - mark its range resolved (at 1x).
  bool NativeOwnedMarkResolved();
  // Session 90 (Q, non-virtual members, no data): the copy NativeOwnedResolve just served is to be written into memory from the owned image
  // (the CPU reads it back) - the cache's destination block runs NativeOwnedWriteBackRun for it (false: nothing was written).
  bool NativeOwnedWriteBackArmed() const;
  bool NativeOwnedWriteBackRun(const draw_util::ResolveInfo& info, const draw_util::ResolveCopyShaderConstants& constants,
                               VkDescriptorSet dest_set, uint32_t group_count_x, uint32_t group_count_y);
  // Session 82 (P2): the resolve being issued is read back for the CPU (see command_processor.cpp).
  bool NativeCopyReadsBack() const;
  // (An owned step's replayed truth draw bound its own state: everything the command processor tracks is forgotten.)
  void NativeForgetState() { NativeStateTouched(false); }
  // wwe2k14 patch (Session 80, frame structure census): with the segment log, the next draw's sampled textures.
  void NativeLogDrawTextures(uint32_t used_texture_mask);
  uint32_t NativeTextureReadable(void* handle, bool allow_scaled = false, int32_t layer = -1,
                                 bool allow_depth = false) const;
  // wwe2k14 patch (Session 56, gpu_native_truth_skip_writeback_scaled): the scaled resolve buffer, a guest range's
  // scaled range (32bpp), marking a range written by compute, and the scaled loads' materialize entry.
  VkBuffer NativeScaledResolveBuffer() const;
  bool NativeScaledResolveRange(uint32_t start, uint32_t length, uint64_t& scaled_start, uint64_t& scaled_length) const;
  void NativeUseScaledResolveBufferForWrite(uint64_t scaled_start, uint64_t scaled_length);
  void NativeTruthMaterializeScaled(uint32_t start, uint32_t length);
  VkDescriptorSet NativeTextureForRead(void* handle, uint32_t& old_usage_out, bool allow_scaled = false,
                                       int32_t layer = -1, bool allow_depth = false);
  void NativeTextureEndRead(void* handle, uint32_t old_usage);
  // wwe2k14 patch (Session 56): VulkanTextureCache::SampleWrittenTextures.
  uint32_t NativeSampleWrittenTextures(void* const* keep, size_t keep_count);
  void NativeTexturePin(void* handle, bool pin);
  VulkanSharedMemory* NativeSharedMemory() const { return shared_memory_.get(); }
  bool NativeCacheClearRequested() const { return cache_clear_requested_; }
  bool NativeResolveReadback() const { return native_resolve_readback_; }  // the CPU reads this resolve's output
  bool NativeLastCopyTextureSampled() const { return !texture_cache_ || texture_cache_->native_last_copy_sampled(); }
  bool NativeTruthActive() const;  // this frame is a truth frame (gpu_native_truth on and not refused)
  VkDescriptorSetLayout NativeTextureSetLayout() const;
  // wwe2k14 patch (Session 45, NATIVE FRAME): gpu_native_segment_log is listing this frame.
  bool NativeSegmentLogActive() const;
  void NativeAfterResolveMarked();
  // wwe2k14 patch (Session 66, G1.0 resolve census, gpu_native_alias_census): a flag read once per hook, the hooks themselves
  // (NativeRenderer::AliasNoteSampled / AliasNoteLoad) only run while it counts.
  bool NativeAliasCensusActive() const { return native_alias_census_; }
  void NativeSetAliasCensusActive(bool active) { native_alias_census_ = active; }
  void NativeAliasSampled(void* texture);
  void NativeAliasLoad(uint32_t first_page, uint32_t page_count);

 protected:
  bool SetupContext() override;
  void ShutdownContext() override;

  void WriteRegister(uint32_t index, uint32_t value) override;
  void WriteRegistersFromMem(uint32_t start_index, uint32_t* base, uint32_t num_registers) override;
  // wwe2k14 patch (Session 6, CP1): for gpu_type0_bulk_writes_verify.
  bool GetRegisterWriteInvalidationState(RegisterWriteInvalidationState& state_out) const override;
  bool ExecutePacketType3_EVENT_WRITE_ZPD(memory::RingBuffer* reader, uint32_t packet,
                                          uint32_t count) override;

  void OnGammaRamp256EntryTableValueWritten() override;
  void OnGammaRampPWLValueWritten() override;
  void OnPrimaryBufferEnd() override;
  // wwe2k14 patch (Session 66, H1.1 readback batching): flushes the pending resolve readbacks where the guest could
  // see GPU-written memory (see FlushPendingReadbacks).
  void OnGuestVisibleWrite(GuestVisibleEvent event, uint32_t address) override;
  void PrepareForWait() override;

  void IssueSwap(uint32_t frontbuffer_ptr, uint32_t frontbuffer_width,
                 uint32_t frontbuffer_height) override;

  Shader* LoadShader(xenos::ShaderType shader_type, uint32_t guest_address,
                     const uint32_t* host_address, uint32_t dword_count) override;

  bool IssueDraw(xenos::PrimitiveType prim_type, uint32_t index_count,
                 IndexBufferInfo* index_buffer_info, bool major_mode_explicit) override;
  bool IssueCopy() override;
  // wwe2k14 patch (Session 50, LEAN DRAWS): gpu_native_lean_draws - see command_processor.cpp "LEAN DRAWS".
  bool IssueDrawLean(xenos::PrimitiveType prim_type, uint32_t index_count, IndexBufferInfo* index_buffer_info,
                     bool major_mode_explicit, uint32_t library_kind, bool& succeeded) override;
  // wwe2k14 patch (Session 60, CROWD BATCH): see the base class.
  uint32_t AppendCrowdInstances(const uint32_t* members, uint32_t count) override;
  bool OwnCrowdDraw(const uint32_t* members, uint32_t count) override;  // wwe2k14 patch (Session 80, own crowd)
  bool OwnCrowdActive() const override;
  bool RegisterNativeOwnCrowd(uint64_t guest_vertex_shader_hash, const uint32_t* vs_words, size_t vs_count,
                              const uint32_t* fs_words, size_t fs_count, uint32_t flags) override;
  struct OwnCrowdProgram {
    uint64_t vertex_shader_hash;
    std::vector<uint32_t> vs, fs;
    uint32_t flags;
    std::vector<uint32_t> vs_vi;  // Session 95 (P7): the attribute form of vs (gpu_native_crowd_input); empty: none
  };
  std::mutex own_crowd_mutex_;
  std::vector<std::shared_ptr<const OwnCrowdProgram>> own_crowd_programs_;
  std::vector<uint64_t> own_crowd_unknown_;  // (vertex shader ^ reason, logged once each)
  uint32_t crowd_batch_refusals_logged_ = 0;
  double gpu_frame_accumulated_us_ = 0.0;  // Session 73: this frame's submissions' GPU time so far (frame_events)
  // Session 73: the GPU busy work the game asks for (frame_events warm_kb), recorded at the swap.
  void KeepWarmWork();
  VkBuffer warm_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory warm_buffer_memory_ = VK_NULL_HANDLE;
  bool warm_failed_ = false;
  // b78: why the command thread waits for the GPU now (rex::perf::frame_events::WaitSite); set by WaitSiteScope.
  uint32_t wait_site_ = 0;
  struct WaitSiteScope {
    WaitSiteScope(VulkanCommandProcessor* cp, uint32_t site) : cp_(cp), old_(cp->wait_site_) { cp->wait_site_ = site; }
    ~WaitSiteScope() { cp_->wait_site_ = old_; }
    VulkanCommandProcessor* cp_;
    uint32_t old_;
  };
  uint32_t crowd_refusal_counts_[32] = {};  // Session 73: by reason, logged every 300 frames
  uint64_t crowd_refusal_log_frame_ = 0;

  // wwe2k14 patch (Session 6, COUNTERS): per-frame counters kept outside the base class.
  void CollectFrameCounters(FrameCounters& counters) override;
  // wwe2k14 patch (Session 7b, GPUMARKS): gpu_anatomy_gpu_marks, see GpuMarks.
  void AnatomyGpuFrameBoundary(const std::string& rows_path, int64_t frame) override;
  // wwe2k14 patch (Session 7c, CAUSES): the anatomy cause columns, see CauseCounts.
  bool AnatomyRowCauses(AnatomyCauses* out) override;
  // wwe2k14 patch (Session 7, EXTENTS-SHADOW).
  SharedMemory* GetSharedMemoryForExtents() const override { return shared_memory_.get(); }

 private:
  struct CommandBuffer {
    VkCommandPool pool;
    VkCommandBuffer buffer;
  };

  struct SparseBufferBind {
    VkBuffer buffer;
    size_t bind_offset;
    uint32_t bind_count;
  };

  union TextureDescriptorSetLayoutKey {
    uint32_t key;
    struct {
      // If texture and sampler counts are both 0, use
      // descriptor_set_layout_empty_ instead as these are owning references.
      uint32_t texture_count : 16;
      uint32_t sampler_count : 15;
      uint32_t is_vertex : 1;
    };

    TextureDescriptorSetLayoutKey() : key(0) { static_assert_size(*this, sizeof(key)); }

    struct Hasher {
      size_t operator()(const TextureDescriptorSetLayoutKey& key) const {
        return std::hash<decltype(key.key)>{}(key.key);
      }
    };
    bool operator==(const TextureDescriptorSetLayoutKey& other_key) const {
      return key == other_key.key;
    }
    bool operator!=(const TextureDescriptorSetLayoutKey& other_key) const {
      return !(*this == other_key);
    }
  };

  union PipelineLayoutKey {
    uint64_t key;
    struct {
      // Pixel textures in the low bits since those are varied much more
      // commonly.
      uint16_t texture_count_pixel;
      uint16_t sampler_count_pixel;
      uint16_t texture_count_vertex;
      uint16_t sampler_count_vertex;
    };

    PipelineLayoutKey() : key(0) { static_assert_size(*this, sizeof(key)); }

    struct Hasher {
      size_t operator()(const PipelineLayoutKey& key) const {
        return std::hash<decltype(key.key)>{}(key.key);
      }
    };
    bool operator==(const PipelineLayoutKey& other_key) const { return key == other_key.key; }
    bool operator!=(const PipelineLayoutKey& other_key) const { return !(*this == other_key); }
  };

  class PipelineLayout : public VulkanPipelineCache::PipelineLayoutProvider {
   public:
    explicit PipelineLayout(VkPipelineLayout pipeline_layout,
                            VkDescriptorSetLayout descriptor_set_layout_textures_vertex_ref,
                            VkDescriptorSetLayout descriptor_set_layout_textures_pixel_ref)
        : pipeline_layout_(pipeline_layout),
          descriptor_set_layout_textures_vertex_ref_(descriptor_set_layout_textures_vertex_ref),
          descriptor_set_layout_textures_pixel_ref_(descriptor_set_layout_textures_pixel_ref) {}
    VkPipelineLayout GetPipelineLayout() const override { return pipeline_layout_; }
    VkDescriptorSetLayout descriptor_set_layout_textures_vertex_ref() const {
      return descriptor_set_layout_textures_vertex_ref_;
    }
    VkDescriptorSetLayout descriptor_set_layout_textures_pixel_ref() const {
      return descriptor_set_layout_textures_pixel_ref_;
    }

   private:
    VkPipelineLayout pipeline_layout_;
    VkDescriptorSetLayout descriptor_set_layout_textures_vertex_ref_;
    VkDescriptorSetLayout descriptor_set_layout_textures_pixel_ref_;
  };

  struct UsedSingleTransientDescriptor {
    uint64_t frame;
    SingleTransientDescriptorLayout layout;
    VkDescriptorSet set;
  };

  struct UsedTextureTransientDescriptorSet {
    uint64_t frame;
    TextureDescriptorSetLayoutKey layout;
    VkDescriptorSet set;
  };

  enum SwapApplyGammaDescriptorSet : uint32_t {
    kSwapApplyGammaDescriptorSetRamp,
    kSwapApplyGammaDescriptorSetSource,

    kSwapApplyGammaDescriptorSetCount,
  };
  enum SwapApplyGammaComputeDescriptorSet : uint32_t {
    kSwapApplyGammaComputeDescriptorSetRamp,
    kSwapApplyGammaComputeDescriptorSetSource,
    kSwapApplyGammaComputeDescriptorSetDestination,

    kSwapApplyGammaComputeDescriptorSetCount,
  };
  struct SwapApplyGammaConstants {
    uint32_t size[2];
  };
  enum SwapFxaaDescriptorSet : uint32_t {
    kSwapFxaaDescriptorSetSource,
    kSwapFxaaDescriptorSetDestination,

    kSwapFxaaDescriptorSetCount,
  };
  struct SwapFxaaConstants {
    uint32_t size[2];
    float size_inv[2];
  };

  // Framebuffer for the current presenter's guest output image revision, and
  // its usage tracking.
  struct SwapFramebuffer {
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    uint64_t version = UINT64_MAX;
    uint64_t last_submission = 0;
  };

  // BeginSubmission and EndSubmission may be called at any time. If there's an
  // open non-frame submission, BeginSubmission(true) will promote it to a
  // frame. EndSubmission(true) will close the frame no matter whether the
  // submission has already been closed.
  // Unlike on Direct3D 12, submission boundaries do not imply any memory
  // barriers aside from an incoming host write (but not outgoing host read)
  // dependency.

  // Rechecks submission number and reclaims per-submission resources. Pass 0 as
  // the submission to await to simply check status, or pass
  // GetCurrentSubmission() to wait for all queue operations to be completed.
  void CheckSubmissionFenceAndDeviceLoss(uint64_t await_submission);
  // If is_guest_command is true, a new full frame - with full cleanup of
  // resources and, if needed, starting capturing - is opened if pending (as
  // opposed to simply resuming after mid-frame synchronization). Returns
  // whether a submission is open currently and the device is not lost.
  bool BeginSubmission(bool is_guest_command);
  // If is_swap is true, a full frame is closed - with, if needed, cache
  // clearing and stopping capturing. Returns whether the submission was done
  // successfully, if it has failed, leaves it open.
  bool EndSubmission(bool is_swap);
  uint32_t copies_in_submission_ = 0;  // WWE '13 port: vulkan_submit_after_copies
  bool AwaitAllQueueOperationsCompletion() {
    // wwe2k14 patch (Session 66, H1.1): any other wait first copies the pending resolve readbacks back (in order).
    if (!readback_pending_.empty()) {
      FlushPendingReadbacks(GuestVisibleEvent::kAwait);
    }
    CheckSubmissionFenceAndDeviceLoss(GetCurrentSubmission());
    return !submission_open_ && submissions_in_flight_fences_.empty();
  }
  // Keep primary-buffer-end submit behavior aligned with D3D12: only submit
  // when immediate submission is safe.
  bool CanEndSubmissionImmediately() const;

  void ClearTransientDescriptorPools();
  bool IssueCopy_ReadbackResolvePath();
  // wwe2k14 patch (Session 92, M3a): IssueCopy as one body fed by words (owned_info: a record's ResolveInfo under gpu_d3d_owned_resolve, else null =
  // the render target cache computes it from the registers); the read-back part of a resolve for the range it wrote; the owned frame's resolve
  // (native/owned_resolve.cpp). Non-virtual, no data members.
  bool IssueCopyFrom(const CopyWords& w, const draw_util::ResolveInfo* owned_info);
  bool ReadbackWritten(uint32_t written_address, uint32_t written_length, uint32_t dest_info_value);
  bool OwnedResolveRecord(const draw_util::ResolveInfo& info, uint32_t& written_address_out, uint32_t& written_length_out);

  // GPU crash diagnostics (vulkan_gpu_crash_diagnostics): checkpoints recorded
  // into the command stream, and a report logged once when the device is lost.
  enum class CrashCheckpointPhase : uint32_t {
    kSubmissionBegin = 0,
    kDrawPrepare = 1,
    kDraw = 2,
    kCopy = 3,
  };
  struct CrashCheckpointRecord {
    uint64_t id = 0;
    uint64_t submission = 0;
    uint64_t frame = 0;
    uint64_t vertex_shader_hash = 0;
    uint64_t pixel_shader_hash = 0;
    uint32_t guest_primitive_type = 0;
    uint32_t host_primitive_type = 0;
    uint32_t guest_index_count = 0;
    uint32_t host_vertex_count = 0;
    uint32_t index_buffer_type = 0;
    uint32_t guest_index_base = 0;
    uint32_t edram_mode = 0;
    uint32_t memexport_vertex = 0;
    uint32_t memexport_pixel = 0;
    uint32_t memexport_extent_start = 0;
    uint32_t memexport_extent_end = 0;
    uint32_t rb_surface_info = 0;
    uint32_t rb_color_info = 0;
    uint32_t rb_depth_info = 0;
    CrashCheckpointPhase last_phase = CrashCheckpointPhase::kSubmissionBegin;
  };
  bool crash_checkpoints_enabled() const;
  // Allocates a record (or reuses the current draw's) and records a checkpoint.
  CrashCheckpointRecord* BeginCrashCheckpointRecord();
  void InsertCrashCheckpoint(CrashCheckpointRecord& record, CrashCheckpointPhase phase);
  void ReportGpuCrashDiagnostics();
  // wwe2k14 patch (Session 8, FLIGHT): rex::perf::flight GPU provider (vulkan_flight_gpu_records).
  static std::string FlightGpuRecords(void* user, uint32_t max_records);
  bool IssueDraw_MemexportReadbackFullPath(uint32_t total_size);
  bool IssueDraw_MemexportReadbackFastPath(uint32_t total_size);
  // (Session 92, M4f-B) What a draw does after its draw call for memexport: the written ranges invalidated in the shared memory, the optional read-back. One body for
  // IssueDraw and the typed draw (memexport_ranges_ is the draw's).
  void IssueDraw_MemexportAfter(bool writes_possible);

  // wwe2k14 patch (Session 6, VIEWPORT): IssueDraw recomputed draw_util::GetHostViewportInfo on every
  // draw (0.7% of the command processor thread in heavy entrance frames). With vulkan_viewport_info_cache,
  // the last result is reused while every input of that function is bit-identical to the previous call.
  // Port of our D3D12 ViewportCacheKey (d3d12/command_processor.cpp), with the inputs that key misses
  // (RB_DEPTH_INFO, the half_pixel_offset setting, the resolution scale, the viewport bounds and the
  // thread's floating-point rounding / denormal mode) added.
  // The arguments are exactly GetHostViewportInfo's (Session 7, SEAMS: including negative_origin).
  void GetHostViewportInfoCached(const RegisterFile& regs, uint32_t draw_resolution_scale_x,
                                 uint32_t draw_resolution_scale_y, bool origin_bottom_left,
                                 uint32_t x_max, uint32_t y_max, bool allow_reverse_z,
                                 reg::RB_DEPTHCONTROL normalized_depth_control,
                                 bool convert_z_to_float24, bool full_float24_in_0_to_1,
                                 bool pixel_shader_writes_depth,
                                 draw_util::ViewportInfo& viewport_info_out,
                                 bool negative_origin);
  // Every value draw_util::GetHostViewportInfo's result depends on (see GetHostViewportInfoCached for the
  // list and why it's complete). Only whole 32-bit words, so there's no padding and == compares every bit.
  struct ViewportInfoCacheKey {
    // Registers.
    uint32_t pa_cl_clip_cntl;
    uint32_t pa_cl_vte_cntl;
    uint32_t pa_su_sc_mode_cntl;
    uint32_t pa_su_vtx_cntl;
    uint32_t pa_sc_window_offset;
    uint32_t rb_depth_info;
    uint32_t vport_regs[6];  // PA_CL_VPORT_XSCALE, XOFFSET, YSCALE, YOFFSET, ZSCALE, ZOFFSET.
    // Arguments.
    uint32_t normalized_depth_control;
    uint32_t draw_resolution_scale_x;
    uint32_t draw_resolution_scale_y;
    uint32_t x_max;
    uint32_t y_max;
    // Bits: kViewportInfoCacheFlag*.
    uint32_t flags;
    // Host floating-point control bits that guest code changes on this thread (rounding mode and
    // flush-to-zero / denormals-are-zero: MXCSR on x64, FPCR on arm64).
    uint32_t host_fp_control;
    bool operator==(const ViewportInfoCacheKey&) const = default;
  };
  static constexpr uint32_t kViewportInfoCacheFlagOriginBottomLeft = 1u << 0;
  static constexpr uint32_t kViewportInfoCacheFlagAllowReverseZ = 1u << 1;
  static constexpr uint32_t kViewportInfoCacheFlagConvertZToFloat24 = 1u << 2;
  static constexpr uint32_t kViewportInfoCacheFlagFullFloat24In0To1 = 1u << 3;
  static constexpr uint32_t kViewportInfoCacheFlagPixelShaderWritesDepth = 1u << 4;
  static constexpr uint32_t kViewportInfoCacheFlagHalfPixelOffset = 1u << 5;
  // wwe2k14 patch (Session 7, SEAMS): GetHostViewportInfo's negative_origin (gpu_tiling_viewport_fix).
  static constexpr uint32_t kViewportInfoCacheFlagNegativeOrigin = 1u << 6;
  ViewportInfoCacheKey viewport_info_cache_key_{};
  draw_util::ViewportInfo viewport_info_cache_value_{};
  bool viewport_info_cache_valid_ = false;
  // Statistics since stats_frame, logged as "[s6][VIEWPORT]" every 300 frames while the cache is on. A
  // window is also closed (and logged up to last_frame) when the cache is turned back on after being off,
  // so the off period is never counted.
  struct ViewportInfoCacheStats {
    uint64_t stats_frame = 0;
    uint64_t last_frame = 0;  // Frame of the latest lookup (0: none yet).
    uint64_t hits = 0;
    uint64_t misses_invalid = 0;  // Nothing cached (first draw, or the cache was off).
    uint64_t misses_viewport = 0;  // PA_CL_VPORT_* changed.
    uint64_t misses_registers = 0;  // Only other registers changed.
    uint64_t misses_arguments = 0;  // Only non-register inputs changed (arguments, settings, FP control).
    uint64_t verify_checks = 0;
    uint64_t verify_mismatches = 0;
    uint64_t verify_mismatches_total = 0;
    uint32_t verify_mismatches_logged = 0;
  };
  ViewportInfoCacheStats viewport_info_cache_stats_;

  // wwe2k14 patch (Session 7, SEAMS): gpu_tiling_viewport_fix as latched at the first draw of
  // tiling_viewport_fix_frame_, so every tile of a frame uses the same setting.
  bool tiling_viewport_fix_ = false;
  uint64_t tiling_viewport_fix_frame_ = 0;
  // gpu_tiling_viewport_fix_stats: computes the draw's host viewport with and without the fix (the same
  // arguments as IssueDraw's GetHostViewportInfo call, without the cache) and counts the differences.
  void TilingViewportFixStatsDraw(const RegisterFile& regs, uint32_t draw_resolution_scale_x,
                                  uint32_t draw_resolution_scale_y, uint32_t x_max, uint32_t y_max,
                                  reg::RB_DEPTHCONTROL normalized_depth_control,
                                  bool convert_z_to_float24, bool full_float24_in_0_to_1,
                                  bool pixel_shader_writes_depth);
  // Logs the open window (frames window_first_frame...last_frame) if it counted any draw.
  void TilingViewportFixStatsReport(const char* note);
  struct TilingViewportFixStats {
    bool active = false;  // A window is open (the stats setting was on at the latest draw).
    uint64_t window_first_frame = 0;
    uint64_t last_frame = 0;
    uint64_t draws = 0;
    uint64_t draws_fix_on = 0;  // Draws issued with gpu_tiling_viewport_fix on (as latched).
    uint64_t draws_clip_enabled = 0;  // Only clip-enabled draws can differ.
    // Draws whose ViewportInfo differs with and without the fix, split by whether a non-zero window
    // offset is applied to vertices (PA_SU_SC_MODE_CNTL vtx_window_offset_enable and PA_SC_WINDOW_OFFSET).
    // A draw counts only if the difference can change what is rendered: both viewports empty on some axis
    // (UpdateDynamicState then uses its dummy viewport either way) goes to differ_empty_only instead.
    uint64_t differ_window_offset = 0;
    uint64_t differ_no_window_offset = 0;
    uint64_t differ_empty_only = 0;
    // Smallest host viewport origin with the fix among the counted differing draws (0 if none).
    int32_t min_origin_with_fix[2] = {};
    // Distinct PA_SC_WINDOW_OFFSET values (plus vtx_window_offset_enable) seen in the window, first 8;
    // differ counts the same draws as differ_window_offset / differ_no_window_offset.
    struct WindowOffset {
      uint32_t window_offset;
      bool vtx_window_offset_enable;
      uint64_t draws;
      uint64_t differ;
    };
    static constexpr uint32_t kMaxWindowOffsets = 8;
    WindowOffset window_offsets[kMaxWindowOffsets] = {};
    uint32_t window_offset_count = 0;
    uint64_t window_offsets_dropped = 0;  // Draws with a value beyond the first 8.
    // Differing draws without a window offset (the case Canary's change also affects outside tiling): the
    // first 8 of the run are logged in detail.
    uint32_t no_window_offset_logged = 0;
  };
  TilingViewportFixStats tiling_viewport_fix_stats_;

  // wwe2k14 patch (Session 23, REPEAT): repeat draws - a draw whose only change against the draw recorded just
  // before it is the vertex shader float constants (CommandProcessor::repeat_dirty_), with the same shaders,
  // draw parameters, submission and frame, and no shared memory invalidation in between
  // (SharedMemory::repeat_epoch). repeat_prev_ describes the last recorded draw (valid only right after it was
  // recorded, cleared at the start of every IssueDraw); gpu_draw_repeat_stats counts what ends repeats.
  // Command processor thread only.
  struct RepeatPrev {
    bool valid = false;
    const void* vertex_shader = nullptr;
    const void* pixel_shader = nullptr;  // the active one (before the rasterization checks)
    uint32_t prim_type = 0;
    uint32_t index_count = 0;
    bool indexed = false;
    bool major_mode_explicit = false;
    uint32_t index_guest_base = 0;
    uint32_t index_length = 0;
    uint32_t index_format = 0;
    uint32_t index_endianness = 0;
    uint64_t submission = 0;
    uint64_t frame = 0;
    uint64_t epoch = 0;
    // The host draw as recorded (vulkan_repeat_draw_fast records the same one again).
    bool host_indexed = false;
    uint32_t host_vertex_count = 0;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint32_t sets_bound = 0;  // current_graphics_descriptor_sets_bound_up_to_date_ right after the draw
    // The draw's crash checkpoint fields (vulkan_gpu_crash_diagnostics), for the fast path's own checkpoints.
    bool crash_valid = false;
    CrashCheckpointRecord crash;
    // vulkan_repeat_draw_merge: the vertex shader's instanced float constant stride (vec4s; 0 = not instanced),
    // and the host draw the next repeat draw may add an instance to: its instance count so far, the constants
    // set and the dynamic offsets it was drawn with (instance i's vertex constants are at the vertex float
    // offset + i * stride).
    uint32_t merge_stride = 0;
    uint32_t float_count = 0;
    // wwe2k14 patch (Session 66, B2.1): the vertex fetch endian key of the recorded draw's pipeline.
    uint32_t vertex_endian_key = 0;
    // wwe2k14 patch (Session 66, PS spec): the pixel shader the draw used (nullptr: none) and its NATIVE specialisation key.
    const Shader* spec_pixel_shader = nullptr;
    uint32_t pixel_spec_key = 0;
    uint32_t vertex_mul_key = 0;  // wwe2k14 patch (Session 66, VS mul)
    uint32_t pixel_ext = 0, vertex_ext = 0;  // wwe2k14 patch (Session 66, native_ext)
    uint32_t run_instances = 0;
    VkDescriptorSet run_set = VK_NULL_HANDLE;
    uint32_t run_offsets[8] = {};
    // gpu_draw_repeat_stats: the pixel shader float constants the draw used (in register order, as uploaded).
    uint32_t ps_float_count = 0;
    float ps_float[256 * 4];
  };
  RepeatPrev repeat_prev_;
  struct RepeatStats {
    uint64_t window_first_frame = UINT64_MAX;
    uint64_t draws = 0;          // IssueDraw calls (not copies)
    uint64_t prev_valid = 0;     // ... right after a recorded draw
    uint64_t same_draw = 0;      // ... with the same shaders, parameters, submission, frame and memory epoch
    uint64_t nothing = 0;        // ... and no register written at all
    uint64_t vs_only = 0;        // ... and only vertex float constants written (the repeat draws)
    uint64_t ps_same = 0;        // ... vertex (or none) + pixel float constants written, pixel values unchanged
    uint64_t ps_changed = 0;     // ... pixel float constant values changed
    uint64_t fetch = 0, bool_loop = 0, other = 0, packet = 0;  // ... and these written (a draw can count in several)
    uint64_t diff_shader = 0, diff_params = 0, diff_submission = 0, diff_frame = 0, diff_epoch = 0;
    uint64_t packet_opcodes[128] = {};
    struct Counted {
      uint64_t key;
      uint64_t count;
    };
    static constexpr uint32_t kMaxCounted = 8;
    Counted other_registers[kMaxCounted] = {};
    Counted vs_only_shaders[kMaxCounted] = {};
    uint64_t run_count = 0;      // runs of consecutive repeat draws (vs_only, or ps_same) closed
    uint64_t run_draws = 0;      // draws in those runs, the first (non-repeat) draw included
    uint64_t run_max = 0;
    uint64_t run_current = 0;    // draws in the run being counted (0: none)
  };
  RepeatStats repeat_stats_;
  static void RepeatStatsCount(RepeatStats::Counted* counted, uint64_t key);
  void RepeatStatsClassify(uint32_t dirty, const Shader* vertex_shader, uint32_t prim_type, uint32_t index_count,
                           const IndexBufferInfo* index_buffer_info, bool major_mode_explicit, uint64_t epoch);
  void RepeatStatsLog();
  // Fills repeat_prev_ for a draw just recorded.
  void RepeatNoteRecorded(const Shader* vertex_shader, const Shader* used_pixel_shader, uint32_t prim_type,
                          uint32_t index_count, const IndexBufferInfo* index_buffer_info,
                          bool major_mode_explicit, uint64_t epoch, bool stats, bool host_indexed,
                          uint32_t host_vertex_count);
  // Whether the draw being issued is the same draw as repeat_prev_ (shaders, parameters, submission, frame,
  // memory epoch); the register / packet changes are checked by the caller.
  bool RepeatIsSameDraw(const Shader* vertex_shader, uint32_t prim_type, uint32_t index_count,
                        const IndexBufferInfo* index_buffer_info, bool major_mode_explicit,
                        uint64_t epoch) const;
  // vulkan_repeat_draw_fast: records a repeat draw (repeat_prev_ valid and the same draw, only vertex float
  // constants written) as its new vertex float constants, the constants set bound with their offsets and the
  // previous host draw again. Returns false (nothing done) when the conditions for it don't hold - the caller
  // then takes the full path.
  bool RepeatDrawFast(const Shader* vertex_shader, uint64_t epoch);
  struct RepeatFastStats {
    uint64_t window_first_frame = UINT64_MAX;
    uint64_t candidates = 0;  // repeat draws (vertex float constants only) with the setting on
    uint64_t fast = 0;        // recorded by the fast path
    // Fell back to the full path because of:
    uint64_t fallback_mode = 0;     // a measurement / debug mode, the render target path or constants mode
    uint64_t fallback_buffers = 0;  // another constant buffer marked out of date (a same-value write)
    uint64_t fallback_bound = 0;    // pipeline or descriptor sets bound changed since the previous draw
    uint64_t fallback_upload = 0;   // the constant upload or set preparation failed
    uint64_t merged = 0;            // of fast, added as an instance to the previous host draw
  };
  // wwe2k14 patch (Session 87, stage 3, for the [d3d][direct] census): which refusal RepeatDrawFast returned false for, last
  // (13 no dynamic-offsets state, 14 per-draw sets, 15 offsets verify, 16 anatomy, 17 gpu marks / recstats / skip frame,
  // 18 crash checkpoints without a valid record, 19 execute_unclipped_draw_vs_on_cpu outside a host-RT truth frame, 20 render
  // target path, 21 no pipeline layout; 1 is no longer set, 2 another constant buffer out of date, 3 pipeline changed, 4 sets bound changed,
  // 5 vertex endian key, 6 PS spec / VS mul / native ext, 7 texture request served from an image, 8 no native base draw,
  // 9 truth claim of the repeat failed, 10 unwritten native state, 11 uniform pool, 12 constants set preparation).
  uint32_t repeat_refusal_ = 0;
  // vulkan_repeat_draw_merge, latched at setup with vulkan_constants_dynamic_offsets.
  bool repeat_merge_latched_ = false;
  uint32_t repeat_merge_alignment_ = 0;
  std::vector<uint64_t> repeat_merge_shaders_;
  // A new host draw was recorded for repeat_prev_: it starts a run of one instance.
  void RepeatStartRun();

  // wwe2k14 patch (Session 23, CHAINPROBE): gpu_chain_probe_shaders (measurement only). For draws of the listed
  // vertex shaders, a hash of their inputs that change between frames (used vertex and pixel float constants, the
  // fetch constants, bool / loop constants), in draw order per frame; at each swap compared with the previous
  // frame's list: same input at the same position, or anywhere in it. [s23][CHAIN] every 300 frames.
  std::vector<uint64_t> chain_probe_shaders_;
  std::string chain_probe_list_;
  std::vector<uint64_t> chain_probe_cur_, chain_probe_prev_;
  uint64_t chain_probe_window_first_ = UINT64_MAX;
  uint64_t chain_probe_frames_ = 0, chain_probe_draws_ = 0, chain_probe_same_pos_ = 0, chain_probe_same_any_ = 0;
  uint64_t chain_probe_frames_all_same_ = 0;
  void ChainProbeDraw(const Shader& vertex_shader, const Shader* pixel_shader);
  void ChainProbeSwap();
  RepeatFastStats repeat_fast_stats_;
  void RepeatFastStatsMaybeLog();

  // wwe2k14 patch (Session 7c, NOOPSKIP): gpu_skip_noop_draws / gpu_skip_noop_draws_stats. IssueDraw calls
  // NoopSkipDraw (fragment shader interlock path, either setting on) right after the render target cache update,
  // with the state the draw would be drawn with: it classifies and counts the draw and returns whether IssueDraw
  // stops there (only with gpu_skip_noop_draws). NoopSkipFrameBoundary (from CollectFrameCounters) records
  // cpf_noop and logs the [s7][NOOPSKIP] summary. Measurement state only; command processor thread only.
  enum NoopSkipOutcome : uint32_t {
    // Can't change anything: skipped (or would be, with gpu_skip_noop_draws_stats only).
    kNoopSkipOk,
    // No color render target written, but (cpf_noop b):
    kNoopSkipRejectDepthWrite,    // depth test and depth write enabled
    kNoopSkipRejectStencilWrite,  // stencil enabled with a non-zero write mask and an operation other than keep
    // Nothing written to the EDRAM, but (cpf_noop c):
    kNoopSkipRejectMemexportVertex,  // the vertex shader exports to memory
    kNoopSkipRejectMemexportPixel,   // the pixel shader exports to memory
    kNoopSkipRejectOcclusionQuery,   // a host occlusion query (guest EVENT_WRITE_ZPD) is counting samples
    kNoopSkipOutcomeCount,
  };
  bool NoopSkipDraw(bool skip, bool is_rasterization_done, bool primitive_polygonal,
                    reg::RB_DEPTHCONTROL normalized_depth_control, uint32_t normalized_color_mask,
                    bool memexport_used_vertex, bool memexport_used_pixel);
  void NoopSkipFrameBoundary();
  void NoopSkipLogWindow(const char* note);
  struct NoopSkipStats {
    struct Counts {
      uint64_t checked = 0;  // draws classified by NoopSkipDraw
      uint64_t color = 0;    // of those, writing a color render target (not candidates, no outcome)
      uint64_t outcomes[kNoopSkipOutcomeCount] = {};
      uint64_t skipped = 0;  // of the kNoopSkipOk ones, skipped (gpu_skip_noop_draws on at that draw)
    };
    Counts frame;   // the frame in progress
    Counts window;  // the summary window, up to the previous frame boundary
    // The window opens at the first classified frame (either setting on at the frame boundary, or a draw
    // classified in the frame) and covers the next 300 frame numbers; frames in it with both settings off count
    // nothing and aren't in the per-frame rates.
    uint64_t window_first_frame = UINT64_MAX;  // UINT64_MAX: window not open
    uint64_t window_last_frame = 0;
    uint64_t window_frames_classified = 0;  // frames in the window that were classified
    uint64_t window_frames_skip_on = 0;     // of those, frames with gpu_skip_noop_draws on (or a draw skipped)
    // kNoopSkipOk draws in the window by guest shader pair (the frame anatomy's vs_hash / ps_hash), first 8.
    struct ShaderPair {
      uint64_t vs_hash;
      uint64_t ps_hash;
      uint64_t draws;
    };
    static constexpr uint32_t kMaxShaderPairs = 8;
    ShaderPair shader_pairs[kMaxShaderPairs] = {};
    uint32_t shader_pair_count = 0;
    uint64_t shader_pairs_dropped = 0;  // kNoopSkipOk draws with a pair beyond the first 8
  };
  NoopSkipStats noop_skip_stats_;

  void SplitPendingBarrier();

  void DestroyScratchBuffer();
  bool InitializeOcclusionQueryResources();
  void ShutdownOcclusionQueryResources();
  bool BeginGuestOcclusionQuery(uint32_t sample_count_address);
  bool EndGuestOcclusionQuery(uint32_t sample_count_address);
  bool AcquireOcclusionQueryIndex(uint32_t& host_index_out);
  void DisableHostOcclusionQueries();
  uint64_t NormalizeOcclusionSamples(uint64_t samples) const;
  void WriteGuestOcclusionResult(uint32_t sample_count_address, uint64_t samples);
  void InvalidateAllVertexBufferResidency();
  void InvalidateVertexBufferResidency(uint32_t vfetch_index);
  void InvalidateVertexBufferResidencyRange(uint32_t first_vfetch, uint32_t last_vfetch);
  struct ReadbackBuffer {
    VkBuffer buffers[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory memories[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    void* mapped_data[2] = {nullptr, nullptr};
    uint32_t sizes[2] = {0, 0};
    uint64_t submission_written[2] = {0, 0};
    uint32_t written_size[2] = {0, 0};
    uint32_t current_index = 0;
    uint64_t last_used_frame = 0;
    // wwe2k14 patch (Session 66, H1.1): the slot's GPU copy is recorded but its data is not in guest memory yet
    // (readback_pending_); the slot must not be overwritten or destroyed until FlushPendingReadbacks.
    bool pending[2] = {false, false};
  };
  // wwe2k14 patch (Session 66, H1.1 readback_resolve_batch): one deferred copy-back of a full-mode resolve readback.
  struct PendingReadback {
    uint64_t key = 0;     // readback_buffers_ key
    uint32_t slot = 0;    // the double-buffer index the GPU copied into
    uint32_t address = 0;  // guest physical destination
    uint32_t length = 0;
    // The [resolve_stats] line (gpu_resolve_log), decided when the resolve was issued, logged at flush time.
    bool stats = false;
    uint32_t stats_resolve_index = 0, stats_color_info = 0, stats_surface_info = 0;
    uint64_t stats_frame = 0;
  };
  // Copies the pending readbacks into guest memory in order after one end-submission + wait. Command processor thread.
  void FlushPendingReadbacks(GuestVisibleEvent reason);
  void ReadbackBatchMaybeLog();
  std::vector<PendingReadback> readback_pending_;
  bool readback_flushing_ = false;
  // Census since the last [readback][batch] line.
  uint64_t readback_batch_flushes_ = 0, readback_batch_entries_ = 0, readback_batch_max_pending_ = 0;
  uint64_t readback_batch_bytes_ = 0, readback_batch_wait_us_ = 0, readback_batch_entries_total_ = 0;
  uint64_t readback_batch_reason_flushes_[size_t(GuestVisibleEvent::kCount)] = {};
  uint64_t readback_batch_reason_entries_[size_t(GuestVisibleEvent::kCount)] = {};
  int64_t readback_batch_log_ns_ = 0;
  void EvictOldReadbackBuffers(std::unordered_map<uint64_t, ReadbackBuffer>& buffer_map);
  static constexpr uint32_t kReadbackBufferSizeIncrement = 16 * 1024 * 1024;
  static constexpr size_t kMaxReadbackBuffers = 256;
  static constexpr uint64_t kReadbackBufferEvictionAgeFrames = 60;
  static inline uint32_t AlignReadbackBufferSize(uint32_t size) {
    if (size < 1 * 1024 * 1024) {
      return rex::align(size, 256u * 1024u);
    }
    if (size < 4 * 1024 * 1024) {
      return rex::align(size, 1u * 1024u * 1024u);
    }
    return rex::align(size, kReadbackBufferSizeIncrement);
  }
  static inline uint64_t MakeReadbackResolveKey(uint32_t address, uint32_t length) {
    return (uint64_t(address) << 32) | uint64_t(length);
  }
  static inline uint64_t MakeMemexportReadbackKey(uint32_t first_base_address_dwords,
                                                  uint32_t total_size) {
    return (uint64_t(first_base_address_dwords) << 32) | uint64_t(total_size);
  }
  struct ResolveDownscaleConstants {
    uint32_t scale_x;
    uint32_t scale_y;
    uint32_t pixel_size_log2;
    uint32_t tile_count;
    uint32_t source_offset_bytes;
    uint32_t half_pixel_offset;
  };
  bool EnsureSwapFxaaSourceImage(uint32_t width, uint32_t height);
  void DestroySwapFxaaSourceImage();

  void UpdateDynamicState(const draw_util::ViewportInfo& viewport_info, bool primitive_polygonal,
                          reg::RB_DEPTHCONTROL normalized_depth_control);
  // wwe2k14 patch (Session 93, idea 7): after a guest pipeline is bound (every site that binds one): the cull / front face / depth /
  // stencil state it has dynamic, from the draw the pipeline cache last described. Nothing with vulkan_extended_dynamic_state off.
  void UpdateEdsState();
  void EdsStatsMaybeLog();
  void UpdateSystemConstantValues(
      bool primitive_polygonal,
      const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
      bool shader_32bit_index_dma, uint32_t compute_memexport_vertex_count,
      const draw_util::ViewportInfo& viewport_info, uint32_t used_texture_mask,
      reg::RB_DEPTHCONTROL normalized_depth_control, uint32_t normalized_color_mask);
  // wwe2k14 patch (Session 6, CP9): the original UpdateSystemConstantValues, which
  // UpdateSystemConstantValues calls with update_textures true while vulkan_system_constants_skip
  // is off. With update_textures false, the texture fields are left as they are.
  void UpdateSystemConstantValuesFull(
      bool primitive_polygonal,
      const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
      bool shader_32bit_index_dma, uint32_t compute_memexport_vertex_count,
      const draw_util::ViewportInfo& viewport_info, uint32_t used_texture_mask,
      reg::RB_DEPTHCONTROL normalized_depth_control, uint32_t normalized_color_mask,
      bool update_textures);
  // wwe2k14 patch (Session 92, M4f-A): the constant arrays UpdateBindings uploads, when they are not the register image's (a typed draw
  // passes the record decoder's state; same layouts: 256 x 4 float words for each stage, 8 + 32 bool / loop words, 32 x 6 fetch words).
  struct ConstSources {
    const uint32_t* vs_float;
    const uint32_t* ps_float;
    const uint32_t* bool_loop;
    const uint32_t* fetch;
  };
  // (Session 95: skip_window = an attribute draw, which reads no storage range: no window is selected for it.)
  bool UpdateBindings(const VulkanShader* vertex_shader, const VulkanShader* pixel_shader,
                      const ConstSources* sources = nullptr, bool skip_window = false);
  // wwe2k14 patch (Session 93, the per-draw storage window): at the top of UpdateBindings; picks the window of the shared memory
  // this draw's vertex fetches / memexport / index loads fall in, rebinds set 0 and writes the base system constant when it changes.
  void SharedMemoryWindowSelect(const VulkanShader* vertex_shader, const VulkanShader* pixel_shader, const uint32_t* src_fetch);
  void SharedMemoryWindowFrameEnd();  // the 600-frame counter line
  // Allocates a descriptor set and fills one or two VkWriteDescriptorSet
  // structure instances (for images and samplers).
  // The descriptor set layout must be the one for the given is_vertex,
  // texture_count, sampler_count (from GetTextureDescriptorSetLayout - may be
  // already available at the moment of the call, no need to locate it again).
  // Returns how many VkWriteDescriptorSet structure instances have been
  // written, or 0 if there was a failure to allocate the descriptor set or no
  // bindings were requested.
  uint32_t WriteTransientTextureBindings(bool is_vertex, uint32_t texture_count,
                                         uint32_t sampler_count,
                                         VkDescriptorSetLayout descriptor_set_layout,
                                         const VkDescriptorImageInfo* texture_image_info,
                                         const VkDescriptorImageInfo* sampler_image_info,
                                         VkWriteDescriptorSet* descriptor_set_writes_out);

  bool device_lost_ = false;
  bool gpu_crash_reported_ = false;
  static constexpr size_t kCrashCheckpointRingSize = 16384;
  std::vector<CrashCheckpointRecord> crash_checkpoint_ring_;
  uint64_t crash_checkpoint_next_id_ = 1;

  bool cache_clear_requested_ = false;

  // Host shader types that guest shaders can be translated into - they can
  // access the shared memory (via vertex fetch, memory export, or manual index
  // buffer reading) and textures.
  VkPipelineStageFlags guest_shader_pipeline_stages_ = 0;
  VkShaderStageFlags guest_shader_vertex_stages_ = 0;

  std::vector<VkFence> fences_free_;
  std::vector<VkSemaphore> semaphores_free_;

  bool submission_open_ = false;
  uint64_t submission_completed_ = 0;
  // In case vkQueueSubmit fails after something like a successful
  // vkQueueBindSparse, to wait correctly on the next attempt.
  std::vector<VkSemaphore> current_submission_wait_semaphores_;
  std::vector<VkPipelineStageFlags> current_submission_wait_stage_masks_;
  std::vector<VkFence> submissions_in_flight_fences_;
  std::deque<std::pair<uint64_t, VkSemaphore>> submissions_in_flight_semaphores_;

  static constexpr uint32_t kMaxFramesInFlight = 3;
  bool frame_open_ = false;
  // Tracks whether any draw in the current frame used an async placeholder
  // graphics pipeline and may have produced incomplete output.
  bool frame_used_async_placeholder_pipeline_ = false;
  // wwe2k14 patch (Session 57, red patches still reported): draws skipped while their pipeline was made since the last
  // resolve (emulated placeholders and truth draws / repeats), the narrowest and widest surface they went to (guest
  // pixels), and how many resolves after such draws were logged (gpu_log_skipped_draw_resolves).
  void NoteSkippedDraw(int64_t surface_info_word = -1);  // (M4f-C: a typed draw passes its own RB_SURFACE_INFO word)
  void LatchTilingViewportFix();  // (Session 92, M4f-C) the once-a-frame latch of gpu_tiling_viewport_fix, shared by IssueDraw and the typed draw
  uint32_t skipped_draws_since_resolve_ = 0;
  uint32_t skipped_draws_pitch_min_ = 0;
  uint32_t skipped_draws_pitch_max_ = 0;
  uint64_t skipped_draw_resolves_ = 0;
  // Guest frame index, since some transient resources can be reused across
  // submissions. Values updated in the beginning of a frame.
  uint64_t frame_current_ = 1;
  uint64_t frame_completed_ = 0;
  // Submission indices of frames that have already been submitted.
  uint64_t closed_frame_submissions_[kMaxFramesInFlight] = {};

  // <Submission where last used, resource>, sorted by the submission number.
  std::deque<std::pair<uint64_t, VkDeviceMemory>> destroy_memory_;
  std::deque<std::pair<uint64_t, VkBuffer>> destroy_buffers_;
  std::deque<std::pair<uint64_t, VkImageView>> destroy_image_views_;
  std::deque<std::pair<uint64_t, VkImage>> destroy_images_;
  std::deque<std::pair<uint64_t, VkFramebuffer>> destroy_framebuffers_;

  std::vector<CommandBuffer> command_buffers_writable_;
  std::deque<std::pair<uint64_t, CommandBuffer>> command_buffers_submitted_;
  DeferredCommandBuffer deferred_command_buffer_;
  // wwe2k14 patch (Session 21, HOIST): see BeginHoistedUpload.
  DeferredCommandBuffer hoist_command_buffer_;
  bool hoist_pending_ = false;

  // wwe2k14 patch (Session 7b, GPUMARKS): what a submission's command buffer writes around its recorded commands
  // for gpu_anatomy_gpu_marks (see GpuMarks): resets of the query ranges its marks use (at its start, before any
  // of them is written), a top-of-pipe timestamp at its start and a bottom-of-pipe one at its end. pool null:
  // nothing.
  struct GpuMarksSubmission {
    static constexpr uint32_t kMaxResets = 4;
    VkQueryPool pool = VK_NULL_HANDLE;
    uint32_t reset_count = 0;
    uint32_t reset_first[kMaxResets] = {};
    uint32_t reset_size[kMaxResets] = {};
    uint32_t cb_begin = UINT32_MAX;
    uint32_t cb_end = UINT32_MAX;
    bool cb_begin_tried = false;  // command processor thread bookkeeping (cb_begin allocated or dropped)
    // wwe2k14 patch (Session 7c, CAUSES): command processor thread bookkeeping - where among the anatomy rows
    // (GpuMarks::Mark::pos) the submission's command buffer (or, for a submission open across an anatomy frame
    // boundary, this frame's part of it) starts, for the rows_covered column.
    uint32_t begin_pos = 0;
  };

  // wwe2k14 patch (vulkan_async_submit): a thread that replays recorded non-swap submissions into their Vulkan
  // command buffers and submits them, in order, so the command processor thread can carry on translating.
  struct AsyncSubmitJob {
    std::vector<uintmax_t> stream;
    std::vector<uintmax_t> hoist_stream;  // wwe2k14 patch (Session 21, HOIST): replayed before stream
    CommandBuffer command_buffer;
    std::vector<VkSemaphore> wait_semaphores;
    std::vector<VkPipelineStageFlags> wait_stage_masks;
    VkFence fence;
    uint64_t submission;
    // wwe2k14 patch (Session 6, COUNTERS): GPU timestamp query pair to write, UINT32_MAX for none.
    uint32_t timestamp_slot = UINT32_MAX;
    // wwe2k14 patch (Session 7b, GPUMARKS).
    GpuMarksSubmission gpu_marks;
    // wwe2k14 patch (Session 6, DRAIN): non-zero for a swap submission sent to the submit thread
    // (vulkan_async_swap) - the ticket retired in async_swap_order_ once it is on the queue.
    uint64_t order_ticket = 0;
    std::chrono::steady_clock::time_point enqueue_time;
  };
  void AsyncSubmitThreadMain();
  void AsyncSubmitExecute(AsyncSubmitJob& job);
  // Waits until every queued job has been submitted (call before any other submission to the queue).
  void AsyncSubmitDrain();
  void AsyncSubmitStop();
  // wwe2k14 patch (Session 7, SUBMITFIX): the wait before the submit thread retries a failed queue submission.
  // Returns false if the thread is being stopped (give up then).
  bool AsyncSubmitRetryWait(uint32_t retry);
  // wwe2k14 patch (Session 6, DRAIN): counts a swap sent to the submit thread, logs "[s6][DRAIN]" statistics.
  void AsyncSwapUpdateStats(bool backlogged);
  std::thread async_submit_thread_;
  std::mutex async_submit_mutex_;
  std::condition_variable async_submit_work_cv_;
  std::condition_variable async_submit_idle_cv_;
  std::deque<AsyncSubmitJob> async_submit_queue_;
  std::vector<std::vector<uintmax_t>> async_submit_free_streams_;
  bool async_submit_busy_ = false;
  // wwe2k14 patch (Session 73): the newest submission the submit thread has handed to vkQueueSubmit (under
  // async_submit_mutex_). Its fence is only waited on / polled after that: vkQueueSubmit's fence must be externally
  // synchronized, and the command processor thread waited on fences the submit thread was submitting at the same time
  // (validation UNASSIGNED-Threading-MultipleThreads).
  uint64_t async_submitted_submission_ = 0;
  bool async_submit_stop_ = false;
  std::atomic<bool> async_submit_device_lost_{false};
  std::atomic<uint64_t> async_submit_commands_dropped_{0};  // wwe2k14 patch (Session 7b, UPSKIP review)
  // wwe2k14 patch (Session 6, DRAIN): vulkan_async_swap. Swap submissions go to the submit thread as well, and
  // the presenter orders its own queue operations after them with tickets of this (shared with the presenter's
  // guest output images, so it may outlive this object).
  std::shared_ptr<ui::vulkan::VulkanQueueSubmitOrder> async_swap_order_ =
      std::make_shared<ui::vulkan::VulkanQueueSubmitOrder>();
  // Set by EndSubmission(true) when it has sent the swap submission to the submit thread, for the guest output
  // refresher in IssueSwap.
  uint64_t async_swap_ticket_ = 0;
  // wwe2k14 patch (Session 7, SUBMITFIX): the ticket of the last swap submission sent to the submit thread, 0 once
  // known to be on the queue (after AsyncSubmitDrain). Command processor thread only.
  uint64_t async_swap_unretired_ticket_ = 0;
  // Statistics for the "[s6][DRAIN]" log. Swaps sent to the submit thread (command processor thread only).
  uint32_t async_swap_stats_swaps_ = 0;
  uint32_t async_swap_stats_backlogged_ = 0;
  bool async_swap_logged_active_ = false;
  // Time the swap jobs waited in the queue behind earlier jobs - what AsyncSubmitDrain would have blocked the
  // command processor thread for (written by the submit thread).
  std::atomic<uint64_t> async_swap_stats_queued_ns_{0};
  std::atomic<uint64_t> async_swap_stats_queued_ns_max_{0};

  // wwe2k14 patch (Session 6, COUNTERS): fetch constant writes that change the value (cpf_fc), compared with the
  // last value seen per texture fetch constant.
  void CountFetchConstantsWritten(uint32_t first_index, uint32_t last_index);
  uint32_t fetch_constants_seen_[32][6] = {};

  // wwe2k14 patch (Session 6, COUNTERS): GPU timestamps per submission (vulkan_gpu_timestamps). A top-of-pipe and
  // a bottom-of-pipe timestamp around each submission's command buffer, written into the real command buffer by
  // whichever thread records it (this one or the submit thread), read back only once the submission's fence is
  // known to be signaled, and recorded to the timeline as gpu_ts.
  static constexpr uint32_t kGpuTimestampSlots = 256;  // submissions in flight with timestamps
  // Returns the query pair (slot) to write for the submission being ended, or UINT32_MAX.
  uint32_t GpuTimestampsBeginSubmission(bool is_swap);
  bool GpuTimestampsInitialize();
  void GpuTimestampsWriteBegin(VkCommandBuffer command_buffer, uint32_t slot) const;
  void GpuTimestampsWriteEnd(VkCommandBuffer command_buffer, uint32_t slot) const;
  void GpuTimestampsRead();
  void GpuTimestampsShutdown();
  VkQueryPool gpu_timestamps_pool_ = VK_NULL_HANDLE;
  bool gpu_timestamps_unsupported_ = false;
  double gpu_timestamps_period_ns_ = 0.0;
  uint64_t gpu_timestamps_mask_ = 0;
  PFN_vkCmdWriteTimestamp gpu_timestamps_cmd_write_ = nullptr;
  PFN_vkGetQueryPoolResults gpu_timestamps_get_results_ = nullptr;
  // Submission whose timestamps are in each slot (0 = none), and whether it was a swap.
  std::array<uint64_t, kGpuTimestampSlots> gpu_timestamps_slot_submission_ = {};
  std::array<bool, kGpuTimestampSlots> gpu_timestamps_slot_is_swap_ = {};
  uint64_t gpu_timestamps_read_up_to_ = 0;

  // wwe2k14 patch (Session 7b, GPUMARKS): GPU timestamps per frame-anatomy row (gpu_anatomy_gpu_marks, measurement
  // only). State in command_processor.cpp (created at the first marked frame); gpu_marks_active_: the current
  // anatomy frame records marks (changes only at anatomy frame boundaries), the only thing the draw path tests.
  // wwe2k14 patch (Session 7c, CAUSES): kPrepBegin and kBlockEnd are gpu_anatomy_gpu_marks_mode 1 marks (see
  // GpuMarksSparseDrawPre).
  // wwe2k14 patch (Session 7c, TEXLOAD): kTexLoad* are the marks inside texture loads (GpuMarkTextureLoad).
  enum class GpuMarkKind : uint8_t {
    kCbBegin,
    kCbEnd,
    kDrawPre,
    kDrawPost,
    kCopyBefore,
    kCopyAfter,
    kPrepBegin,
    kBlockEnd,
    kTexLoadBegin,
    kTexLoadDispatch,
    kTexLoadConvert,
    kTexLoadCopy
  };
  struct GpuMarks;
  std::unique_ptr<GpuMarks> gpu_marks_;
  bool gpu_marks_active_ = false;
  // wwe2k14 patch (Session 16, MATTE): gpu_resolve_log / gpu_depth_dump state.
  int32_t matte_log_remaining_ = 0;
  uint32_t matte_resolve_index_ = 0;   // resolves since the last swap (log / file names)
  uint32_t matte_dumped_ = 0;          // resolves dumped for the armed request (file numbers)
  uint32_t matte_depth_dumped_ = 0;    // of which depth: the swap ends the request once this is non-zero
  bool matte_dump_is_depth_ = false;   // the resolve being dumped copies depth
  bool matte_force_readback_ = false;  // this resolve: synchronous readback into a file, guest memory untouched
  std::string matte_dump_file_;        // the .bin for this resolve
  uint32_t matte_regs_[8] = {};        // copy control, dest base / info / pitch, surface / depth / color info

  // wwe2k14 patch (Session 18, DLAA): sub-pixel camera jitter for the scene draws (gpu_dlaa_jitter).
 public:
  struct DlaaFrameJitter {
    uint64_t frame = 0;      // swaps processed by this command processor
    float x = 0, y = 0;      // host pixels added to the scene viewports' origin this frame
    uint32_t jittered = 0;   // scene draws jittered (last completed frame)
  };
  DlaaFrameJitter dlaa_last_jitter() const { return dlaa_last_; }

  // wwe2k14 patch (Session 18, DLAA step 2): the game's camera of each guest frame, from the app (any thread): the
  // composed view at the hand-off to the renderer (world -> view, row vectors, x right / y down / z forward), its
  // field of view (degrees) and near / far planes. `frame` = guest frames (swaps) before this one.
  void DlaaSetCamera(uint64_t frame, const float view[16], float fov_degrees, float near_z, float far_z);

 private:
  void DlaaOnSwap();
  uint64_t dlaa_frame_ = 0;
  float dlaa_jitter_[2] = {};
  bool dlaa_scene_done_ = false;   // the frame's scene depth resolve happened: later draws (post, HUD) unjittered
  uint32_t dlaa_jittered_draws_ = 0;
  // wwe2k14 patch (Session 24): gpu_dlaa_jitter_max_camera_px - the next frame's jitter is 0 while the camera moves
  // fast (the game blends its previous frame there: a changing jitter turns into moire, runs 1699 / 1700).
  bool dlaa_jitter_suspend_ = false;
  float dlaa_camera_px_ = 0.0f;
  uint64_t dlaa_jitter_suspended_frames_ = 0;
  uint32_t dlaa_log_left_ = 8;
  DlaaFrameJitter dlaa_last_;
  // Step 2: camera motion vectors (gpu_dlaa_motion / gpu_dlaa_debug).
  struct DlaaCamera {
    uint64_t frame = UINT64_MAX;
    double view[16] = {};
    float fov_degrees = 0, near_z = 0, far_z = 0;
  };
  std::mutex dlaa_camera_mutex_;
  DlaaCamera dlaa_cameras_[16];
  bool DlaaFindCamera(uint64_t frame, DlaaCamera& out);
  void DlaaMotionPass();
  bool DlaaEnsureMotionResources(uint32_t width, uint32_t height);
  uint32_t dlaa_color_regs_[4] = {};  // the frame's last scene-size colour resolve: dest base, pitch, height, info
  bool dlaa_color_valid_ = false;
  uint32_t dlaa_depth_regs_[4] = {};
  VkPipelineLayout dlaa_motion_layout_ = VK_NULL_HANDLE;
  VkPipeline dlaa_motion_pipeline_ = VK_NULL_HANDLE;
  bool dlaa_motion_failed_ = false;
  VkImage dlaa_motion_image_ = VK_NULL_HANDLE;
  VkDeviceMemory dlaa_motion_memory_ = VK_NULL_HANDLE;
  VkImageView dlaa_motion_view_ = VK_NULL_HANDLE;
  VkDescriptorSet dlaa_motion_set_ = VK_NULL_HANDLE;
  uint32_t dlaa_motion_size_[2] = {};
  bool dlaa_motion_initialized_layout_ = false;
  uint32_t dlaa_motion_log_left_ = 6;
  uint64_t dlaa_motion_passes_ = 0, dlaa_motion_no_camera_ = 0;
  // Our output colour image (the scene colour's host format, MUTABLE, R32_UINT storage view), copied into the
  // scene colour texture after the pass (debug views; later the DLAA result).
  bool DlaaEnsureOutputImage(VkFormat format, uint32_t width, uint32_t height);
  VkImage dlaa_out_image_ = VK_NULL_HANDLE;
  VkDeviceMemory dlaa_out_memory_ = VK_NULL_HANDLE;
  VkImageView dlaa_out_view_ = VK_NULL_HANDLE;
  VkDescriptorSet dlaa_out_set_ = VK_NULL_HANDLE;
  VkFormat dlaa_out_format_ = VK_FORMAT_UNDEFINED;
  uint32_t dlaa_out_size_[2] = {};
  // Step 3: DLAA through Streamline (gpu_dlaa).
  struct DlaaImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;          // typed view (format)
    VkImageView storage_view = VK_NULL_HANDLE;  // R32_UINT view (if requested)
    VkDescriptorSet set = VK_NULL_HANDLE;       // storage set for storage_view
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    VkImageCreateFlags flags = 0;
    uint32_t width = 0, height = 0;
  };
  bool DlaaEnsureImage(DlaaImage& img, VkFormat format, bool r32_storage_view, uint32_t width, uint32_t height);
  void DlaaRunDlss(const VulkanTextureCache::DlaaStorage& color, uint32_t width, uint32_t height, const double rep[16],
                   const double view_now[16], float fov_degrees, float near_z, float far_z, bool reset);
  DlaaImage dlaa_depthf_, dlaa_in_, dlaa_dlss_out_;
  bool dlaa_post_color_seen_ = false;  // this frame: the first scene-size colour resolve after the depth happened
  uint64_t dlaa_dlss_frames_ = 0;
  uint32_t dlaa_dlss_log_left_ = 6;
  // wwe2k14 patch (Session 24, DLSS): DLSS Super Resolution at the swap (gpu_dlss). The swap's gamma pass writes our
  // render-size image instead of the presenter's; DLSS upscales it with this frame's depth, camera motion and jitter
  // (from the motion pass at the scene depth resolve); a composite pass writes the presenter's guest output at the
  // output size (gpu_dlss_view: DLSS, plain upscales for comparison, the HUD on top).
  struct DlssPending {
    bool valid = false;
    uint64_t frame = 0;
    double rep[16] = {}, view_now[16] = {};
    float fov_degrees = 0, near_z = 0, far_z = 0, jitter[2] = {};
    bool reset = false;
    uint32_t width = 0, height = 0;
  };
  DlssPending dlss_pending_;
  // gpu_dlss on and our resources ready: the output size for a render-size guest output, else false.
  bool DlssOutputSize(uint32_t width, uint32_t height, uint32_t& out_width, uint32_t& out_height);
  bool DlssEnsureComposite();
  // wwe2k14 patch (Session 33, FSR): the swap's upscaler can run: AMD FSR 3 with gpu_dlss_fsr (any card; loads the
  // runtime on first use), else DLSS through Streamline.
  bool DlssUpscalerReady();
  // wwe2k14 patch (Session 33, HIDEHUD): gpu_hide_hud: the post chain's last pass has been drawn this frame.
  bool hide_hud_after_pass_ = false;
  bool wide_menu_frame_ = false;  // Session 33 (WIDE): the previous frame had no scene depth resolve (a menu)
  uint64_t hide_hud_vs_ = 0;
  std::string hide_hud_vs_text_;
  // Session 83 (WIDE): gpu_wide_post_vs, full-screen passes after the post chain's marker left unsqueezed.
  bool IsWidePostVs(uint64_t vs_hash);
  std::vector<uint64_t> wide_post_vs_;
  std::string wide_post_vs_text_;
  // Inside the swap's refresh callback, after the gamma pass wrote dlss_in_: DLSS (if this frame has its inputs) and
  // the composite into `guest_output_view` (GENERAL).
  void DlssRecord(VkImageView guest_output_view, uint32_t width, uint32_t height, uint32_t out_width,
                  uint32_t out_height, uint32_t swap_frame_index, bool have_prehud);
  // The HUD step (gpu_dlss_hud): right after the post chain's last full-screen pass (gpu_dlss_hud_after_vs), a copy
  // of the colour surface from the EDRAM buffer (the picture before the HUD); at the swap it goes through the same
  // gamma pass into dlss_prehud_ (DlssPrehudGamma: the gamma pipeline and its ramp set must be bound) and DLSS gets
  // it instead of the final picture; the composite adds the HUD (final - pre-HUD) on top.
  void DlssMaybeSnapshot(const Shader& vertex_shader);
  // wwe2k14 patch (Session 61): the DLSS snapshot + hide-HUD detection after a recorded draw (IssueDraw, IssueDrawLean).
  void AfterPostChainCheck(const Shader& vertex_shader);
  bool DlssPrehudGamma(uint32_t width, uint32_t height, uint32_t swap_frame_index);
  DlaaImage dlss_snap_[2];  // by frame parity (gpu_dlss_hud_delay: a debug comparison with the previous frame's)
  uint64_t dlss_snap_frame_ = UINT64_MAX;
  uint64_t dlss_snap_frames_[2] = {UINT64_MAX, UINT64_MAX};
  std::string dlss_hud_vs_text_;
  uint64_t dlss_hud_vs_ = 0;
  uint32_t dlss_snap_log_left_ = 6;
  VkPipelineLayout dlss_composite_layout_ = VK_NULL_HANDLE;
  VkPipeline dlss_composite_pipeline_ = VK_NULL_HANDLE;
  VkDescriptorPool dlss_descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet dlss_sets_[kMaxFramesInFlight][6] = {};  // storage out, DLSS out, final, pre-HUD, motion, depth
  VkDescriptorSet dlss_gamma_sets_[kMaxFramesInFlight][2] = {};  // pre-HUD gamma pass: sampled source, storage dest
  bool dlss_failed_ = false;
  DlaaImage dlss_in_, dlss_out_, dlss_prehud_;
  // The motion pass's depth and motion, copied per frame parity: the swap's DLSS reads the copy of its own frame (a
  // single image is overwritten by the next frame's pass before the swap's work runs: runs 1679 / 1681).
  DlaaImage dlss_mv_[2], dlss_depth_[2];
  DlaaImage dlaa_depth_copy_;  // the scene depth of an unscaled frame (no storage view on the texture), copied
  // wwe2k14 patch (Session 24, OBJMV): object motion vectors for the character draws (gpu_dlss_objmv). Their
  // positions are captured with transform feedback into objmv_buffers_[frame parity]; the motion pass rasterizes
  // each draw with last frame's matching draw (dlss_objmv.cs.glsl) into the motion image and objmv_mask_.
  static constexpr uint32_t kObjmvMaxVertices = 1u << 20;
  struct ObjmvDraw {
    uint64_t vs_hash = 0;
    uint32_t ordinal = 0;   // among this shader's captured draws in the frame
    uint32_t first = 0;     // first captured vertex in the frame's buffer
    uint32_t vertices = 0;  // captured vertices (3 per triangle)
    float viewport[4] = {};  // host viewport without the jitter
  };
  bool ObjmvEnsureBuffers();
  bool ObjmvBeginCapture(const Shader& vertex_shader, const PrimitiveProcessor::ProcessingResult& primitive);
  void ObjmvEndCapture();
  // wwe2k14 patch (Session 61): the capture's range without its begin (a truth draw captures in its own pass), and
  // giving the last one back.
  bool ObjmvReserve(const Shader& vertex_shader, const PrimitiveProcessor::ProcessingResult& primitive,
                    VkBuffer& buffer_out, VkDeviceSize& offset_out, VkDeviceSize& size_out);
  void ObjmvUnreserve();
  void ObjmvPass(uint32_t width, uint32_t height, const float jitter[2], float near_z, float far_z);
  bool objmv_latched_ = false;
  bool objmv_failed_ = false;
  std::vector<uint64_t> objmv_shaders_;
  VkBuffer objmv_buffers_[2] = {};
  VkDeviceMemory objmv_buffer_memory_[2] = {};
  std::vector<ObjmvDraw> objmv_draws_[2];
  uint64_t objmv_frames_[2] = {UINT64_MAX, UINT64_MAX};
  uint32_t objmv_cursor_ = 0;
  uint32_t objmv_used_[2] = {kObjmvMaxVertices, kObjmvMaxVertices};  // vertices written since the buffer's clear
  float objmv_viewport_[4] = {};
  VkPipelineLayout objmv_layout_ = VK_NULL_HANDLE;
  VkPipeline objmv_pipeline_ = VK_NULL_HANDLE;
  DlaaImage objmv_mask_;
  VkDescriptorSet objmv_mask_set_ = VK_NULL_HANDLE;
  VkImageView objmv_mask_set_view_ = VK_NULL_HANDLE;
  DlaaImage dlss_mask_[2];  // the mask per frame parity for the swap's DLSS (bias-current-colour hint)
  uint64_t objmv_captured_ = 0, objmv_skipped_ = 0, objmv_passes_ = 0;
  uint32_t objmv_log_left_ = 8;
  uint64_t dlss_frames_ = 0, dlss_last_eval_frame_ = UINT64_MAX;
  int dlss_reset_hold_ = 0;  // wwe2k14 patch (Session 27): frames still to reset after a detected cut
  uint32_t dlss_log_left_ = 8;
  // wwe2k14 patch (Session 7c, CAUSES): gpu_anatomy_gpu_marks_mode 1 for the frame being marked (only while
  // gpu_marks_active_).
  bool gpu_marks_sparse_ = false;
  // A bottom-of-pipe timestamp into the deferred command buffer (call only while gpu_marks_active_).
  void GpuMark(GpuMarkKind kind);
  // wwe2k14 patch (Session 7c, CAUSES): gpu_anatomy_gpu_marks_mode 1 (only while gpu_marks_active_ and
  // gpu_marks_sparse_): the draw's marks at its pre / post points, the bookkeeping after a mark, and the start
  // of a submission (any mode, while gpu_marks_ exists).
  void GpuMarksSparseDrawPre();
  void GpuMarksSparseDrawPost();
  void GpuMarksSparseNoteMark(GpuMarkKind kind);
  // The end of the row in progress for mode 1: an unmarked row opens a block (arms the event hook).
  void GpuMarksSparseRowEnd();
  void GpuMarksBeginSubmission();
  // Where the command stream is among the anatomy rows (GpuMarks::Mark::pos) for a mark that starts an interval
  // (interval_start: before the row's next commands) or ends one.
  uint32_t GpuMarksPosition(bool interval_start) const;
  bool GpuMarksAllocate(GpuMarkKind kind, uint32_t& query_out);
  bool GpuMarksInitialize();
  // At EndSubmission: the marks to write around the submission's command buffer (cb_end allocated here).
  GpuMarksSubmission GpuMarksEndSubmission();
  void GpuMarksWriteBegin(VkCommandBuffer command_buffer, const GpuMarksSubmission& marks) const;
  void GpuMarksWriteEnd(VkCommandBuffer command_buffer, const GpuMarksSubmission& marks) const;
  // Writes the marked frames whose submissions have all completed, oldest first, without waiting. Called at
  // anatomy frame boundaries and, after a capture, at swaps (never inside the timed draws).
  void GpuMarksRead();
  void GpuMarksWriteFrame(uint32_t slot);
  void GpuMarksLogSummary();
  void GpuMarksShutdown();
  // Review fix: submissions with marks whose command buffer the submit thread didn't submit (not recorded, or
  // its commands dropped by SUBMITFIX), so their queries were neither reset nor written for this use and must
  // not be read. Added on the submit thread before the vkQueueSubmit that signals the job's fence; read and
  // pruned by GpuMarksWriteFrame. Rarely non-empty (only on failing submissions).
  void GpuMarksNotRun(uint64_t submission);
  std::mutex gpu_marks_not_run_mutex_;
  std::vector<uint64_t> gpu_marks_not_run_;

  // wwe2k14 patch (Session 38, NATIVE): the native renderer (vulkan/native/native_renderer.h). The command processor
  // hands it each draw (IssueDraw), the end of each render pass (NativeAfterGuestRenderPass), the resolves
  // (IssueCopy) and the swap (IssueSwap); with gpu_native off it only runs the census and the per-frame latch.
  std::unique_ptr<native::NativeRenderer> native_;
  // wwe2k14 patch (Session 40, RENAME): EDRAM renaming (gpu_edram_rename; VulkanRenderTargetCache::kRenameSlots).
  // A run = consecutive passes of the same shape, each starting with a clear rectangle that overwrites every bit of
  // its region (depth, stencil, colour target 0), whose later draws stay inside that rectangle and whose resolves
  // read inside the run's EDRAM span. Pass k draws and resolves in slot k; the span is copied into every slot at the
  // run's start and the last slot back at its end. Outside the rectangle nothing in the run writes, so the copies
  // hold exactly what the EDRAM would; inside it each pass starts from its own clear.
  struct EdramRenameRun {
    bool active = false;
    uint32_t surface_info = 0, color_info = 0, depth_info = 0;
    int32_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;  // the clear's rectangle in guest pixels, half-open
    VkDeviceSize span_start = 0, span_bytes = 0;
    uint32_t slot = 0;
  } rename_;
  struct EdramRenameStats {
    uint64_t frames = 0, runs = 0, passes = 0, resolves = 0;
    uint64_t ends[8] = {};  // by EdramRenameEndReason
  } rename_stats_;
  enum EdramRenameEndReason : uint32_t {
    kRenameEndDraw, kRenameEndResolve, kRenameEndShape, kRenameEndSlots, kRenameEndFrame, kRenameEndSubmission,
    kRenameEndOther, kRenameEndCount
  };
  std::unique_ptr<ShaderInterpreter> rename_interpreter_;
  VkDescriptorPool rename_draw_set_pool_ = VK_NULL_HANDLE;
  std::unordered_map<VkDeviceSize, VkDescriptorSet> rename_draw_sets_;
  uint32_t rename_shared_memory_binding_count_ = 0;
  // Before render_target_cache_->Update of an interlock-path draw: start / continue / end a run.
  void EdramRenameOnDraw(xenos::PrimitiveType prim_type, const VulkanShader& vertex_shader,
                         const VulkanShader* pixel_shader, reg::RB_DEPTHCONTROL depth_control, uint32_t color_mask,
                         bool memexport);
  // Before a resolve: whether it reads the run's slot (else the run ends first).
  void EdramRenameOnResolve();
  void EdramRenameEnd(EdramRenameEndReason reason);
  // The clear rectangle of a pass-start clear draw (guest pixels, half-open), or false.
  bool EdramRenameClearRect(xenos::PrimitiveType prim_type, const VulkanShader& vertex_shader,
                            const VulkanShader* pixel_shader, reg::RB_DEPTHCONTROL depth_control, uint32_t color_mask,
                            bool memexport, int32_t rect[4]);
  // Why the last EdramRenameClearRect said no (its source line), for gpu_edram_rename_log.
  int rename_reject_line_ = 0;
  bool EdramRenameReject(int line) {
    rename_reject_line_ = line;
    return false;
  }
  bool EdramRenameBindSlot();
  VkDescriptorSet EdramRenameDrawSet(VkDeviceSize offset);

  // wwe2k14 patch (Session 41, FASTCLEAR): exact cheap clears (gpu_fast_clear, interlock path). A clear rectangle
  // (the game clears whole surfaces with a rectangle draw, often on a 4x-MSAA view: four samples a pixel through the
  // EDRAM emulation) whose every pixel gets the same values is drawn for its first pixel only (the scissor cut to it),
  // then VulkanRenderTargetCache::EdramFill copies that pixel's dwords over the rest of the rectangle.
  struct FastClear {
    bool active = false;
    int32_t rect[4] = {};  // guest pixels, half-open
    uint32_t samples_x = 1, samples_y = 1;
    VulkanRenderTargetCache::EdramFillSurface surfaces[2] = {};
    uint32_t surface_count = 0;
  } fast_clear_;
  struct FastClearStats {
    uint64_t frames = 0, rect_draws = 0, fills = 0, fill_pixels = 0, too_small = 0;
    std::unordered_map<int, uint64_t> rejects;  // by source line
  } fast_clear_stats_;
  int fast_clear_reject_line_ = 0;
  bool FastClearReject(int line) {
    fast_clear_reject_line_ = line;
    return false;
  }
  // Before render_target_cache_->Update: fills fast_clear_ (active) when the draw qualifies.
  bool FastClearDetect(xenos::PrimitiveType prim_type, const VulkanShader& vertex_shader,
                       const VulkanShader* pixel_shader, reg::RB_DEPTHCONTROL depth_control, uint32_t color_mask,
                       bool memexport);
  void FastClearFrameEnd();
  uint32_t native_sets_needed_ = 0;  // the descriptor sets the last UpdateBindings bound for its draw (bit per set)
  void NativeAfterGuestRenderPass();
  // wwe2k14 patch (Session 46, NATIVE TRUTH M5): the native renderer recorded with its own state (see the .cpp).
  void NativeStateTouched(bool keep_bindings);
  // wwe2k14 patch (Session 47, NATIVE TRUTH CPU): NativeDescribeDraw's last native pixel shader translation (truth
  // frames only; dropped when the caches are cleared).
  struct NativeTranslationCache {
    bool valid = false;
    const VulkanShader* shader = nullptr;
    uint64_t modification = 0;
    uint32_t ext = 0;  // wwe2k14 patch (Session 66, native_ext)
    bool float24 = false;
    uint32_t quantize = 0;
    bool twin = false;  // wwe2k14 patch (Session 54)
    VkShaderModule module = VK_NULL_HANDLE;
  } native_ps_cache_;
  // wwe2k14 patch (Session 66, keycost): NATIVE's own vertex translation of the last kVertex draw's shader, again where
  // the shader, its modification, the mul key and the ext word are the same (dropped when the caches are cleared).
  struct NativeVertexTranslationCache {
    bool valid = false;
    const VulkanShader* shader = nullptr;
    uint64_t base_modification = 0;
    uint32_t mul_key = 0, ext = 0;
    uint64_t native_modification = 0;
    VkShaderModule module = VK_NULL_HANDLE;
  } native_vs_cache_;
  // wwe2k14 patch (Session 72 part 4, gpu_native_truth_skip_restores): the registered restore shaders (an entry is
  // written before the count that publishes it; the command thread reads up to the count).
  struct NativeRestoreShader {
    uint64_t hash = 0;
    uint32_t color_fetch = 0, depth_fetch = 0;
  };
  std::array<NativeRestoreShader, 8> native_restore_shaders_{};
  std::atomic<uint32_t> native_restore_shader_count_{0};
  void NativeDescribeDraw(native::Draw& draw, const VulkanShader& vertex_shader, const VulkanShader* pixel_shader,
                          const VulkanShader::VulkanTranslation& vertex_translation,
                          SpirvShaderTranslator::Modification pixel_modification,
                          const PrimitiveProcessor::ProcessingResult& primitive, xenos::PrimitiveType guest_primitive,
                          bool primitive_polygonal, reg::RB_DEPTHCONTROL depth_control, uint32_t color_mask,
                          bool memexport, bool pixel_replacement = false, bool vertex_replacement = false);
  // wwe2k14 patch (Session 39, NATIVE): counts the native renderer's recordings that change the bound graphics state
  // (NativeAfterGuestRenderPass). One can happen in the middle of a draw's setup (its render pass ended after its
  // pipeline was bound); the draw then binds its pipeline, descriptor sets, viewport and scissor again before its
  // draw call (NativeRestoreDrawState).
  uint64_t native_state_epoch_ = 0;
  bool native_lean_mismatch_logged_ = false;  // wwe2k14 patch (Session 43, NATIVE LEAN)
  // wwe2k14 patch (Session 43, NATIVE COVER): the pixels a quad / rectangle list draw writes for sure - its corners
  // from the vertex shader run on the CPU form an axis-aligned rectangle, nothing can discard a pixel (no kill, no
  // alpha test, no alpha to coverage), one sample: rect = x0, y0, x1, y1 (guest pixels, exclusive), within the
  // scissor. False when that cannot be proven.
  // (Session 80, owned_step: an owned step's draw - also a 4-vertex strip, and a culled draw facing the kept way.)
  bool NativeCoverRect(xenos::PrimitiveType prim_type, const VulkanShader& vertex_shader,
                       const VulkanShader* pixel_shader, int32_t rect[4], bool owned_step = false,
                       const IndexBufferInfo* index_buffer_info = nullptr);
  // (Session 92, M4f-B) The same body over explicit inputs (the old signature fills them from the register source).
  bool NativeCoverRect(const CoverInputs& in, xenos::PrimitiveType prim_type, const VulkanShader& vertex_shader,
                       const VulkanShader* pixel_shader, int32_t rect[4], bool owned_step, const IndexBufferInfo* index_buffer_info);
  bool NativeBoundsRect(const CoverInputs& in, xenos::PrimitiveType prim_type, const VulkanShader& vertex_shader, uint32_t index_count,
                        const IndexBufferInfo* index_buffer_info, int32_t rect[4]);
  bool NativeCoverReject(int line, const VulkanShader& vertex_shader);
  // wwe2k14 patch (Session 56, gpu_native_truth_clear_attachments): the host depth a fast clear's quad / rectangle
  // writes, from its vertex shader run on the CPU, when every corner gives exactly 0 or exactly 1 through a plain
  // depth viewport (z_min 0, z_max 1) without depth bias; -1 otherwise.
  // (Session 80, gpu_native_truth_clear_load: with `color`, also the first interpolator every corner exports - set and
  // *color_ok true when all corners give the same finite value.)
  float NativeClearDepth(xenos::PrimitiveType prim_type, const VulkanShader& vertex_shader,
                         const draw_util::ViewportInfo& viewport_info, float* color = nullptr,
                         bool* color_ok = nullptr);
  std::vector<uint64_t> native_cover_rejects_;
  // wwe2k14 patch (Session 80, gpu_native_truth_bounds_vs): an upper bound of the guest pixels a triangle / quad draw
  // can touch - every vertex it uses (its indices read from memory for an indexed draw) through the vertex shader on
  // the CPU, the box of their screen positions widened by 2 pixels: rect = x0, y0, x1, y1 (guest pixels, exclusive).
  // False (rect empty) when it cannot be bounded (a non-positive or non-finite w, a texture-fetching shader, too many
  // vertices). Cached by its inputs (the shader, the draw's registers, all constants, the vertex and index bytes).
  bool NativeBoundsRect(xenos::PrimitiveType prim_type, const VulkanShader& vertex_shader, uint32_t index_count,
                        const IndexBufferInfo* index_buffer_info, int32_t rect[4]);
  struct NativeBoundsEntry {
    uint64_t key = 0;
    int32_t rect[4] = {};
    bool ok = false;
  };
  std::array<NativeBoundsEntry, 256> native_bounds_cache_{};  // (Session 87: 256 entries; 64 are used unless gpu_interp_memo)
  uint32_t native_bounds_next_ = 0;
  std::vector<uint32_t> native_bounds_indices_;
  // wwe2k14 patch (Session 87, step 4, gpu_interp_memo): the CPU vertex-shader interpreter (NativeCoverRect,
  // NativeClearDepth, FastClearDetect) runs a few vertices of a small shader per call - 4.8% of Bryan's command thread
  // on the Surface with NativeBoundsRect. Its result is a function of the shader, the constants the shader reads, the
  // vertex fetch constants and bytes it reads, and the vertex index; the exports of one run are kept by that key and
  // replayed to the caller's sink. InterpMemoKey: false (no memo) when the shader reads constants by address or a
  // fetched buffer is over 4 KB.
  bool InterpInputHashes(const VulkanShader& vertex_shader, bool fetch_bytes, uint64_t& constants_hash,
                         uint64_t& fetch_hash);
  bool InterpMemoKey(const VulkanShader& vertex_shader, uint64_t& key);
  bool InterpInputHashes(const VulkanShader& vertex_shader, bool fetch_bytes, uint64_t& constants_hash, uint64_t& fetch_hash, const CoverInputs& in);
  bool InterpMemoKey(const VulkanShader& vertex_shader, uint64_t& key, const CoverInputs& in);
  void InterpRun(ShaderInterpreter& interpreter, uint64_t inputs_key, uint32_t vertex_index, uint32_t user);
  void InterpMemoLog();
  struct InterpExport {
    uint32_t export_register = 0;
    uint32_t mask = 0;
    float value[4] = {};
  };
  static constexpr uint32_t kInterpMemoMaxExports = 8;
  struct InterpMemoEntry {
    uint64_t key = 0;
    uint32_t count = 0;
    InterpExport exports[kInterpMemoMaxExports];
  };
  std::array<InterpMemoEntry, 4096> interp_memo_{};
  struct InterpMemoStats {
    uint64_t first_frame = UINT64_MAX;
    uint64_t calls[4] = {};         // by user: 0 cover, 1 clear depth, 2 fast clear
    uint64_t hits[4] = {};
    uint64_t misses[4] = {};
    uint64_t no_key[4] = {};        // the memo was off or the key refused
    uint64_t mismatches = 0;        // verify: the replayed exports differ from a fresh run's
    uint64_t overflow = 0;          // a run exported more than kInterpMemoMaxExports registers
    uint64_t key_ns = 0;
    uint64_t run_ns = 0;            // misses: the interpreter itself
    uint64_t bounds_calls = 0;      // NativeBoundsRect lookups
    uint64_t bounds_hits = 0;       // ... found with the key in use
    uint64_t bounds_full_hits = 0;  // verify / stats: found with the all-constants key
    uint64_t bounds_used_hits = 0;  // verify / stats: found with the shader-reads key
    uint64_t bounds_mismatches = 0;
    uint64_t bounds_key_ns = 0;
    struct VsCount {  // the vertex shaders the memo's runs were for (the first 12 seen in the window)
      uint64_t hash = 0, calls = 0, misses = 0;
    } by_vs[12];
    uint32_t by_vs_count = 0;
  } interp_memo_stats_;
  uint64_t interp_memo_vs_ = 0;  // the vertex shader of the last InterpMemoKey
  uint64_t bounds_seen_full_[256] = {};  // gpu_interp_memo_verify: the all-constants keys seen (ring), for the comparison
  uint64_t bounds_seen_used_[256] = {};
  uint32_t bounds_seen_next_full_ = 0, bounds_seen_next_used_ = 0;
  // wwe2k14 patch (Session 87, step 4, gpu_cp_trim): the clear-load pixel shader list, upper-cased once.
  std::string clear_load_src_;
  std::string clear_load_upper_;
  // wwe2k14 patch (Session 39, NATIVE): the full path's last draw as the native renderer got it (the base of a
  // repeat run, vulkan_repeat_draw_fast): in a replacing frame a repeat of a claimed draw goes to the native renderer
  // with the repeat's constants instead of a host draw call.
  std::unique_ptr<native::Draw> native_repeat_draw_;
  bool native_repeat_draw_valid_ = false;
  // wwe2k14 patch (Session 50, LEAN DRAWS): what the full path derived from the render state for the last truth draw
  // it recorded (the base): valid while no register outside the constants and no state packet was written since
  // (repeat_dirty_), within the frame and with nothing else issued between (IssueDraw's full path, IssueCopy).
  struct LeanState {
    bool valid = false;
    const Shader* active_vertex_shader = nullptr;  // the base's active shaders (as the packets loaded them)
    const Shader* active_pixel_shader = nullptr;
    VulkanShader* pixel_shader = nullptr;  // the one it used (nullptr: none)
    uint32_t prim_type = 0;
    bool major_mode_explicit = false;
    bool primitive_polygonal = false;
    Shader::HostVertexShaderType host_vertex_shader_type = Shader::HostVertexShaderType::kVertex;
    uint32_t used_texture_mask = 0;
    reg::RB_DEPTHCONTROL normalized_depth_control;
    uint32_t normalized_color_mask = 0;
    draw_util::ViewportInfo viewport_info;
    const PipelineLayout* layout = nullptr;
    VkPipelineStageFlags shared_memory_access_stages = 0;
    uint32_t merge_stride = 0, float_count = 0;
    bool pixel_shader_vertex_fetches = false;  // its pixel shader fetches vertices (every page counts as accessed)
    bool host_primitive_reset_enabled = false;  // (review L1: from the draw packet, not in repeat_dirty_)
    xenos::PrimitiveType host_primitive_type = xenos::PrimitiveType::kNone;
    // wwe2k14 patch (Session 66, B2.1): the vertex fetch endian key (VulkanPipelineCache::GetVertexFetchEndianKey) the
    // base's pipeline was built with; a draw whose fetch constants give another key goes to the full path.
    uint32_t vertex_endian_key = 0;
    // wwe2k14 patch (Session 66, PS spec): the same for NATIVE's pixel shader (VulkanPipelineCache::GetNativePixelSpecKey).
    uint32_t pixel_spec_key = 0;
    uint32_t vertex_mul_key = 0;  // wwe2k14 patch (Session 66, VS mul): GetNativeVertexMulKey
    uint32_t pixel_ext = 0, vertex_ext = 0;  // wwe2k14 patch (Session 66, native_ext): the base draw's ext words
  };
  LeanState lean_;
  // wwe2k14 patch (Session 56, gpu_native_lean_same_values): the render-state registers 0x2000-0x23FF as they were
  // at the base (VGT_DMA_BASE / SIZE / DRAW_INITIATOR, the draw packet's own, not compared).
  std::array<uint32_t, 0x400> lean_regs_ = {};
  std::unique_ptr<native::Draw> native_lean_draw_;  // the lean draw (scratch, made at the first one)
  bool lean_mode_ok_ = false;  // latched at each frame end: no setting that needs the full path is on
  bool lean_residency_cache_off_ = false;  // vulkan_vertex_residency_cache_off, latched with it
  LeanState lean_verify_snapshot_;  // gpu_native_lean_draws_verify: what a lean draw would have used
  std::unique_ptr<native::Draw> native_lean_verify_draw_;  // and its native draw
  bool lean_verify_pending_ = false;
  enum LeanReason : uint32_t {
    kLeanTaken,
    kLeanNotLibrary,  // no library draw (or not kind 1 / 2) holds the packet
    kLeanMode,        // not a truth frame, or a diagnostic / feature mode that needs the full path
    kLeanNoBase,      // no base since the last full-path draw, copy or frame start
    kLeanDirty,       // a render-state register or a state packet written since the base
    kLeanShader,      // other shaders, primitive or major mode (Session 66: or another vertex fetch endian key)
    kLeanPrimitive,   // primitive processing gave another host vertex shader type
    kLeanSampler,     // sampler overflow
    kLeanClaim,       // the truth claim needs the full path (images changed, stencil, occlusion query, ...)
    kLeanRepeat,      // a repeat (vertex constants only) recorded by the repeat path
    kLeanVerify,      // gpu_native_lean_draws_verify: handed to the full path to be compared
    kLeanReasonCount,
  };
  struct LeanStats {
    uint64_t first_frame = UINT64_MAX;
    uint64_t reasons[kLeanReasonCount] = {};
    uint64_t verify_checked = 0, verify_mismatches = 0, verify_not_truth = 0;
    uint64_t cache_verify_checked = 0, cache_verify_mismatches = 0;  // (Session 87, stage 3) cross-frame restores compared
    uint64_t same_values = 0;  // Session 56: state written, the same values as at the base (gone lean)
    uint32_t verify_logged = 0;
    uint32_t mode_logged = 0;
  };
  LeanStats lean_stats_;
  std::map<uint32_t, uint64_t> lean_census_;  // wwe2k14 patch (Session 56): gpu_native_lean_dirty_census
  uint64_t lean_census_draws_ = 0, lean_census_sets_[3] = {};
  // wwe2k14 patch (Session 59, census with gpu_native_lean_dirty_census): the last 32 bases' render-state hashes
  // (0x2000-0x23FF without the draw packet's own registers) and vertex shaders; "state written" draws whose state
  // matches one of them (a multi-base lean path's hit rate), and those with writes outside the block or packets.
  std::array<std::pair<uint64_t, const Shader*>, 32> lean_base_ring_ = {};
  uint32_t lean_base_ring_next_ = 0;
  uint64_t lean_multi_checks_ = 0, lean_multi_hits_ = 0, lean_multi_outside_ = 0;
  uint64_t LeanStateHash() const;
  // wwe2k14 patch (Session 59, gpu_native_lean_multi_base): this frame's earlier lean bases, by render state - a draw
  // with no base or with render state written goes lean on the base whose render-state block (0x2000-0x23FF, without
  // the draw packet's own registers) it equals exactly, with the same active shaders and no packet or register outside
  // the block written since that base was captured (lean_outside_epoch_).
  struct LeanBaseEntry {
    uint64_t hash = 0;
    uint64_t frame = UINT64_MAX;
    uint64_t outside_epoch = 0;
    uint64_t cache_gen = 0;  // (Session 87, stage 3) lean_cache_gen_ at the capture: the cross-frame restore needs it equal
    LeanState state;
    std::array<uint32_t, 0x400> regs = {};
    std::unique_ptr<native::Draw> draw;
  };
  // (Session 87 stage 3: 64 instead of 16 - the entrance run with the layer found all 16 held in 83% of the misses.)
  std::array<LeanBaseEntry, 64> lean_bases_;
  uint32_t lean_bases_next_ = 0;
  // wwe2k14 patch (Session 87, stage 3, gpu_native_lean_state_cache): bumped when what a base's handles / derivations rest on
  // goes (the caches cleared, the resolution scale or a lean-mode setting changed); a base of another frame restores only
  // while it equals. lean_restore_cross_: the last LeanRestoreBase restored a base of an earlier frame.
  uint64_t lean_cache_gen_ = 1;
  uint64_t lean_cache_key_ = 0;
  bool lean_restore_cross_ = false;
  uint64_t lean_cache_verify_checked_ = 0, lean_cache_verify_mismatches_ = 0;  // (not reset with lean_stats_)
  bool lean_verify_cache_ = false;  // the pending LeanVerifyCompare checks a cross-frame restore (its claim images left out)
  uint64_t lean_outside_epoch_ = 0;
  uint64_t lean_multi_restored_ = 0;
  bool LeanRestoreBase(uint32_t prim_type, bool major_mode_explicit, bool allow_cross_frame = false);
  void LeanCaptureBase(const VulkanShader* vertex_shader, VulkanShader* pixel_shader, uint32_t prim_type,
                       bool major_mode_explicit, bool primitive_polygonal,
                       Shader::HostVertexShaderType host_vertex_shader_type, uint32_t used_texture_mask,
                       reg::RB_DEPTHCONTROL normalized_depth_control, uint32_t normalized_color_mask,
                       const draw_util::ViewportInfo& viewport_info, const PipelineLayout* layout,
                       VkPipelineStageFlags shared_memory_access_stages, uint32_t merge_stride, uint32_t float_count,
                       bool pixel_shader_vertex_fetches, bool host_primitive_reset_enabled,
                       xenos::PrimitiveType host_primitive_type, uint32_t vertex_endian_key,
                       uint32_t pixel_spec_key, uint32_t vertex_mul_key, uint32_t pixel_ext, uint32_t vertex_ext);
  // wwe2k14 patch (Session 66, native_ext): the NATIVE specialisation words for this draw's pixel shader (the colour epilogue's
  // state is read from system_constants_, current after UpdateSystemConstantValues; census: as if every setting was on).
  uint32_t NativePixelExt(const Shader* pixel_shader, bool census = false) const;
  uint32_t NativeVertexExt(const Shader* vertex_shader = nullptr) const;  // (Session 87: + our vertex shader swap)

  // wwe2k14 patch (Session 66, B2.0 gpu_native_vertex_census): log-only statistics for the native vertex input
  // decision (docs\native\S66_MAP_VERTEX.md): per vertex shader (once) its vfetch instructions and whether their
  // index is the plain vertex index (Shader::GatherVertexFetchCensus), per draw (counted by the path that recorded
  // it) the endian mode of each fetch constant the shader uses and the host vertex shader type. Latched at each
  // frame end (vertex_census_on_), so the draw paths pay one bool compare while it is off.
  enum VertexCensusKind : uint32_t {
    kVertexCensusFull,    // IssueDraw, an emulated host draw
    kVertexCensusNative,  // IssueDraw, handed to the native renderer
    kVertexCensusLean,    // IssueDrawLean
    kVertexCensusRepeat,  // RepeatDrawFast
    kVertexCensusCrowd,   // AppendCrowdInstances (one count per member batched)
    kVertexCensusKindCount,
  };
  struct VertexCensusCounts {
    uint64_t draws = 0, eligible = 0, no_fetch = 0;
    uint64_t kinds[kVertexCensusKindCount] = {};
    uint64_t types[16] = {};   // by Shader::HostVertexShaderType
    uint64_t endian[4] = {};   // per (draw, fetch constant), by xenos::Endian
  };
  struct VertexCensusShader {
    bool analyzed = false;
    uint32_t fetches = 0, plain = 0, bindings = 0;
    bool eligible = false;                                         // at least one vfetch, every index plain
    std::vector<std::pair<std::string, uint32_t>> formats;        // "name norm|int signed|unsigned" and its fetch count
    uint64_t draws = 0;
    uint32_t modes_seen = 0;                                      // bit m: some fetch constant had xenos::Endian m
    std::vector<uint32_t> keys;                                   // distinct (2 bits per binding) endian keys, up to 16
    bool keys_overflow = false;
  };
  struct VertexCensus {
    std::unordered_map<uint64_t, VertexCensusShader> shaders;
    uint32_t frames = 0;
    VertexCensusCounts frame, total;
    std::vector<Shader::VertexFetchCensusEntry> scratch;
  };
  bool vertex_census_on_ = false;  // either census below is counting (the draw paths' gate)
  bool vertex_census_v_ = false;   // gpu_native_vertex_census
  // wwe2k14 patch (Session 66, PS spec 1, gpu_native_ps_census): per pixel shader, the distinct NATIVE texture
  // specialisation keys (GetNativePixelSpecKey with the texture setting forced on) its draws had.
  struct PsCensusShader {
    uint64_t draws = 0, unspecialised_draws = 0;  // draws, and those whose key is 0
    std::vector<uint32_t> keys;                   // distinct keys, up to 32
    bool keys_overflow = false;
  };
  bool vertex_census_p_ = false;  // gpu_native_ps_census
  uint32_t ps_census_frames_ = 0;
  std::unordered_map<uint64_t, PsCensusShader> ps_census_shaders_;
  void PsCensusSummary();
  uint32_t vertex_census_last_type_ = 0;  // the host vertex shader type of the last full / native / lean draw
  VertexCensus vertex_census_;
  void VertexCensusDraw(const VulkanShader& vertex_shader, const Shader* pixel_shader, VertexCensusKind kind,
                        uint32_t host_vertex_shader_type, uint32_t weight);
  void VertexCensusFrameEnd();
  void VertexCensusSummary();
  void LeanVerifyCompare(const LeanState& now, const native::Draw& now_draw);
  void LeanFrameEnd();
  void NativeRestoreDrawState(VkPipeline pipeline, const PipelineLayout* pipeline_layout);

  // wwe2k14 patch (Session 7c, CAUSES): see CauseCounts (command_processor.cpp, "Session 7c, CAUSES").
  struct CauseSnapshot {
    CauseCounts counts;
    uint32_t dispatches = 0;
    uint32_t rendering_begins = 0;
  };
  CauseSnapshot CausesNow() const;
  // The sum of the counters whose change makes a row's preparation "eventful" for gpu_anatomy_gpu_marks_mode 1
  // (uploads, texture loads, dispatches, barrier batches, rendering begins, pipeline binds and creations): it
  // changes exactly when one of them does.
  uint64_t CausesEventSum() const;
  // During a capture, at a draw's post point (right after its draw call; copy false) or a resolve's before point
  // (right after its before mark; copy true): its row's cause window (its preparation) ends there, and the row's
  // columns are computed then (causes_row_cut_result_).
  void CausesCut(bool copy);
  // The window from causes_row_start_ to now into out (with the stage masks ORed since its start); the next window
  // starts now.
  void CausesCloseWindow(AnatomyCauses& out);
  // Review fix: discard what was counted since causes_row_start_, the next window starts now (a capture frame's
  // start, the end of a resolve's own commands, a new submission's command buffer).
  void CausesRestartWindow();
  // Every frame boundary (CollectFrameCounters): cpf_sync / cpf_sync2 and the [s7][CAUSES] summary.
  void CausesFrameBoundary();
  CauseCounts cause_counts_;
  CauseSnapshot causes_row_start_;            // the current cause window's start
  AnatomyCauses causes_row_cut_result_ = {};  // the row's columns, if cut (one of the two flags below)
  bool causes_row_cut_valid_ = false;         // in a draw, after its draw call (CausesCut(false))
  bool causes_copy_cut_valid_ = false;        // review fix: in a resolve, after its before point (CausesCut(true))
  CauseSnapshot causes_frame_start_;  // cpf_sync / cpf_sync2
  CauseSnapshot causes_log_start_;    // [s7][CAUSES]
  uint64_t causes_log_first_frame_ = 0;
  uint32_t causes_log_frames_ = 0;
  bool causes_log_timeline_ = true;  // the timeline was on at every frame boundary of the summary window

  // wwe2k14 patch (Session 20, RECSTATS): gpu_record_stats. Measurement only: the deferred command buffer's command
  // counts per frame, and per IssueDraw call (everything recorded inside it) for the draws whose vertex shader is in
  // gpu_record_stats_vs (the group) and for the rest. Latched at each frame boundary.
  using RecordStats = DeferredCommandBuffer::RecordStats;
  void RecStatsFrameBoundary();
  void RecStatsDrawBegin(uint64_t vertex_shader_hash);
  void RecStatsDrawEnd();
  int32_t recstats_frames_ = 0;  // the summary window (frames); 0 = off
  uint32_t recstats_window_frames_ = 0;
  uint64_t recstats_first_frame_ = 0;
  RecordStats recstats_window_start_;
  std::string recstats_vs_string_;
  std::vector<uint64_t> recstats_vs_;
  bool recstats_in_draw_ = false;
  bool recstats_draw_group_ = false;
  RecordStats recstats_draw_start_;
  RecordStats recstats_group_;  // summed over the group's IssueDraw calls in the window
  RecordStats recstats_rest_;   // the other IssueDraw calls (not copies, not skipped before the vertex shader check)
  uint64_t recstats_group_draws_ = 0;
  uint64_t recstats_rest_draws_ = 0;

  // wwe2k14 patch (Session 21, BARMAP): gpu_barrier_map. Measurement only: SubmitBarriers batches per call site.
  struct BarMapSite {
    uintptr_t stack[3] = {};
    uint64_t batches = 0;
    uint64_t commands = 0;
    uint64_t buffer_shared = 0;
    uint64_t buffer_edram = 0;
    uint64_t buffer_other = 0;
    uint64_t images = 0;
    uint64_t render_pass_ends = 0;
    VkPipelineStageFlags src_stages = 0;
    VkPipelineStageFlags dst_stages = 0;
  };
  void BarMapRecord();
 public:
  // wwe2k14 patch (Session 56, measurement only): `count` return addresses of the caller's callers (skipping `skip`
  // more), and the module base of this DLL (for tools\perf\barmap.py-style symbolizing).
  static uintptr_t DebugCaptureStack(uintptr_t* out, uint32_t count, uint32_t skip);
 private:
  void BarMapFrameBoundary();
  static void BarMapRecordModuleAnchor();
  int32_t barmap_frames_ = 0;  // the summary window (frames); 0 = off
  uint32_t barmap_window_frames_ = 0;
  uint64_t barmap_first_frame_ = 0;
  std::unordered_map<uint64_t, BarMapSite> barmap_sites_;

  std::vector<VkSparseMemoryBind> sparse_memory_binds_;
  std::vector<SparseBufferBind> sparse_buffer_binds_;
  // SparseBufferBind converted to VkSparseBufferMemoryBindInfo to this buffer
  // on submission (because pBinds should point to a place in std::vector, but
  // it may be reallocated).
  std::vector<VkSparseBufferMemoryBindInfo> sparse_buffer_bind_infos_temp_;
  VkPipelineStageFlags sparse_bind_wait_stage_mask_ = 0;

  // Temporary storage with reusable memory for creating descriptor set layouts.
  std::vector<VkDescriptorSetLayoutBinding> descriptor_set_layout_bindings_;
  // Temporary storage with reusable memory for writing image and sampler
  // descriptors.
  std::vector<VkDescriptorImageInfo> descriptor_write_image_info_;

  std::unique_ptr<ui::vulkan::VulkanUploadBufferPool> uniform_buffer_pool_;

  // wwe2k14 patch (Session 7b, REBAR): cpf_rebar events and the [s7][REBAR] summary (see RebarFrameBoundary in
  // command_processor.cpp). Measurement only; command processor thread only.
  struct RebarStats {
    // Cumulative counters of the shared memory upload pool, the shared memory and the uniform pool.
    struct Totals {
      uint64_t upload_bytes = 0;
      uint64_t upload_pages = 0;
      uint64_t upload_pages_device_local = 0;
      uint64_t upload_fallbacks = 0;
      uint64_t upload_pages_over_share = 0;
      uint64_t upload_copy_ticks = 0;
      uint64_t upload_verify_skipped = 0;
      uint64_t uniform_bytes = 0;
      uint64_t uniform_pages = 0;
      uint64_t uniform_pages_device_local = 0;
      uint64_t uniform_fallbacks = 0;
      uint64_t uniform_pages_over_share = 0;
    };
    Totals frame;   // at the previous frame boundary
    Totals window;  // at the start of the summary window
    uint64_t window_first_frame = UINT64_MAX;
  };
  RebarStats rebar_stats_;
  void RebarFrameBoundary();

  // Descriptor set layouts used by different shaders.
  VkDescriptorSetLayout descriptor_set_layout_empty_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_set_layout_constants_ = VK_NULL_HANDLE;
  std::array<VkDescriptorSetLayout, size_t(SingleTransientDescriptorLayout::kCount)>
      descriptor_set_layouts_single_transient_{};
  VkDescriptorSetLayout descriptor_set_layout_shared_memory_and_edram_ = VK_NULL_HANDLE;

  // Descriptor set layouts are referenced by pipeline_layouts_.
  std::unordered_map<TextureDescriptorSetLayoutKey, VkDescriptorSetLayout,
                     TextureDescriptorSetLayoutKey::Hasher>
      descriptor_set_layouts_textures_;
  // Pipeline layouts are referenced by VulkanPipelineCache.
  std::unordered_map<PipelineLayoutKey, PipelineLayout, PipelineLayoutKey::Hasher>
      pipeline_layouts_;

  // No specific reason for 32768, just the "too much" descriptor count from
  // Direct3D 12 PIX warnings.
  static constexpr uint32_t kLinkedTypeDescriptorPoolSetCount = 32768;
  static const VkDescriptorPoolSize kDescriptorPoolSizeUniformBuffer;
  static const VkDescriptorPoolSize kDescriptorPoolSizeStorageBuffer;
  static const VkDescriptorPoolSize kDescriptorPoolSizeTextures[2];
  ui::vulkan::LinkedTypeDescriptorSetAllocator transient_descriptor_allocator_uniform_buffer_;
  ui::vulkan::LinkedTypeDescriptorSetAllocator transient_descriptor_allocator_storage_buffer_;
  std::deque<UsedSingleTransientDescriptor> single_transient_descriptors_used_;
  std::array<std::vector<VkDescriptorSet>, size_t(SingleTransientDescriptorLayout::kCount)>
      single_transient_descriptors_free_;
  // <Usage frame, set>.
  std::deque<std::pair<uint64_t, VkDescriptorSet>> constants_transient_descriptors_used_;
  std::vector<VkDescriptorSet> constants_transient_descriptors_free_;

  ui::vulkan::LinkedTypeDescriptorSetAllocator transient_descriptor_allocator_textures_;
  std::deque<UsedTextureTransientDescriptorSet> texture_transient_descriptor_sets_used_;
  std::unordered_map<TextureDescriptorSetLayoutKey, std::vector<VkDescriptorSet>,
                     TextureDescriptorSetLayoutKey::Hasher>
      texture_transient_descriptor_sets_free_;
  // wwe2k14 patch (Session 6, CP5): vulkan_reuse_texture_bindings state (defined in
  // command_processor.cpp, created when the setting is first used).
  struct TextureBindingReuse;
  std::unique_ptr<TextureBindingReuse> texture_binding_reuse_;
  // wwe2k14 patch (Session 6, CP7): vulkan_constants_dynamic_offsets state (defined in
  // command_processor.cpp). Created in SetupContext only if the setting is on at startup - the
  // constants descriptor set layout, and with it every guest pipeline layout and pipeline, depends on
  // it, so it's latched there; null means the original transient constants sets.
  struct ConstantsDynamicOffsets;
  std::unique_ptr<ConstantsDynamicOffsets> constants_dynamic_offsets_;
  bool constants_dynamic_offsets_restart_warned_ = false;

  std::unique_ptr<VulkanSharedMemory> shared_memory_;
  bool native_resolve_readback_ = false;  // wwe2k14 patch (Session 49): inside IssueCopy_ReadbackResolvePath
  bool native_alias_census_ = false;  // wwe2k14 patch (Session 66, G1.0): NativeRenderer's gpu_native_alias_census is counting
  // wwe2k14 patch (Session 59, vulkan_constants_reuse_identical): per constant buffer kind, the last one written in
  // this frame (its bytes, where it is) - a draw whose buffer has the same bytes uses that copy.
  struct ConstantsReuseEntry {
    uint64_t frame = UINT64_MAX;
    size_t size = 0;
    VkDescriptorBufferInfo info = {};
    uint8_t* mapping = nullptr;
    std::array<uint8_t, 4096> bytes;
  };
  ConstantsReuseEntry constants_reuse_[5];
  uint64_t constants_reuse_hits_ = 0, constants_reuse_misses_ = 0, constants_reuse_bytes_saved_ = 0,
           constants_reuse_bytes_written_ = 0, constants_reuse_stats_frame_ = 0;
  bool ConstantsTryReuse(uint32_t kind, const void* bytes, size_t size);
  void ConstantsNoteWritten(uint32_t kind, const void* bytes, size_t size, uint8_t* mapping);
  // wwe2k14 patch (Session 59, readback_resolve_override_skip_recurring): per resolve destination base, the last frame
  // it was resolved in and how many frames in a row (gaps of at most one) it has been; and the skips counted.
  struct ReadbackDestHistory {
    uint64_t last_frame = 0;
    uint32_t streak = 0;
  };
  std::unordered_map<uint32_t, ReadbackDestHistory> readback_dest_history_;
  uint64_t readback_dest_pruned_frame_ = UINT64_MAX;
  uint64_t readback_recurring_skipped_ = 0, readback_recurring_kept_ = 0;

  std::unique_ptr<VulkanPrimitiveProcessor> primitive_processor_;

  std::unique_ptr<VulkanRenderTargetCache> render_target_cache_;

  std::unique_ptr<VulkanPipelineCache> pipeline_cache_;

  std::unique_ptr<VulkanTextureCache> texture_cache_;

  VkDescriptorPool shared_memory_and_edram_descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet shared_memory_and_edram_descriptor_set_;
  // wwe2k14 patch (Session 93, the per-draw storage window): see vulkan_shared_memory_window. Latched in SetupContext.
  bool window_on_ = false;
  uint32_t window_size_ = 0;      // W, bytes (a power of two)
  uint32_t window_stride_ = 0;    // S = W / 4, bytes
  uint32_t window_count_ = 0;     // kmax + 1
  uint32_t window_cur_ = 0;       // the window the bound set 0 and system_constants_.shared_memory_window_base_dwords are on
  VkDescriptorSet window_sets_[32] = {};
  struct WindowStats {
    uint64_t frames = 0, windowed = 0, changes = 0, over = 0, no_access = 0;
    uint64_t per_window[32] = {};
    uint32_t first_frame = 0;
  } window_stats_;
  uint32_t window_over_logged_ = 0;
  bool window_restart_warned_ = false;
  std::string window_cvar_latched_;  // the cvar's value at SetupContext (for the changed-while-running warning)

  // Bytes 0x0...0x3FF - 256-entry gamma ramp table with B10G10R10X2 data (read
  // as R10G10B10X2 with swizzle).
  // Bytes 0x400...0x9FF - 128-entry PWL R16G16 gamma ramp (R - base, G - delta,
  // low 6 bits of each are zero, 3 elements per entry).
  // kMaxFramesInFlight pairs of gamma ramps if in host-visible memory and
  // uploaded directly, one otherwise.
  VkDeviceMemory gamma_ramp_buffer_memory_ = VK_NULL_HANDLE;
  VkBuffer gamma_ramp_buffer_ = VK_NULL_HANDLE;
  // kMaxFramesInFlight pairs, only when the gamma ramp buffer is not
  // host-visible.
  VkDeviceMemory gamma_ramp_upload_buffer_memory_ = VK_NULL_HANDLE;
  VkBuffer gamma_ramp_upload_buffer_ = VK_NULL_HANDLE;
  VkDeviceSize gamma_ramp_upload_memory_size_;
  uint32_t gamma_ramp_upload_memory_type_;
  // Mapping of either gamma_ramp_buffer_memory_ (if it's host-visible) or
  // gamma_ramp_upload_buffer_memory_ (otherwise).
  void* gamma_ramp_upload_mapping_;
  std::array<VkBufferView, 2 * kMaxFramesInFlight> gamma_ramp_buffer_views_{};
  // UINT32_MAX if outdated.
  uint32_t gamma_ramp_256_entry_table_current_frame_ = UINT32_MAX;
  uint32_t gamma_ramp_pwl_current_frame_ = UINT32_MAX;

  VkDescriptorSetLayout swap_descriptor_set_layout_sampled_image_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout swap_descriptor_set_layout_combined_image_sampler_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout swap_descriptor_set_layout_storage_image_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout swap_descriptor_set_layout_uniform_texel_buffer_ = VK_NULL_HANDLE;

  // Descriptor pool for allocating descriptors needed for presentation, such as
  // the destination images and the gamma ramps.
  VkDescriptorPool swap_descriptor_pool_ = VK_NULL_HANDLE;
  // Interleaved 256-entry table and PWL texel buffer descriptors.
  // kMaxFramesInFlight pairs of gamma ramps if in host-visible memory and
  // uploaded directly, one otherwise.
  std::array<VkDescriptorSet, 2 * kMaxFramesInFlight> swap_descriptors_gamma_ramp_;
  // Sampled images.
  std::array<VkDescriptorSet, kMaxFramesInFlight> swap_descriptors_source_;
  // Combined image sampler descriptors for FXAA.
  std::array<VkDescriptorSet, kMaxFramesInFlight> swap_descriptors_fxaa_source_;
  // Storage image descriptors for the apply-gamma compute pass destination.
  std::array<VkDescriptorSet, kMaxFramesInFlight> swap_descriptors_destination_storage_;
  // Separate storage image descriptors for the FXAA compute destination.
  std::array<VkDescriptorSet, kMaxFramesInFlight> swap_descriptors_fxaa_destination_storage_;

  VkSampler swap_sampler_linear_clamp_ = VK_NULL_HANDLE;

  VkPipelineLayout swap_apply_gamma_pipeline_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout swap_apply_gamma_compute_pipeline_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout swap_fxaa_pipeline_layout_ = VK_NULL_HANDLE;
  // Has no dependencies on specific pipeline stages on both ends to simplify
  // use in different scenarios with different pipelines - use explicit barriers
  // for synchronization.
  VkRenderPass swap_apply_gamma_render_pass_ = VK_NULL_HANDLE;
  VkPipeline swap_apply_gamma_256_entry_table_pipeline_ = VK_NULL_HANDLE;
  VkPipeline swap_apply_gamma_pwl_pipeline_ = VK_NULL_HANDLE;
  VkPipeline swap_apply_gamma_256_entry_table_rb_swap_pipeline_ = VK_NULL_HANDLE;
  VkPipeline swap_apply_gamma_pwl_rb_swap_pipeline_ = VK_NULL_HANDLE;
  VkPipeline swap_apply_gamma_256_entry_table_fxaa_luma_pipeline_ = VK_NULL_HANDLE;
  VkPipeline swap_apply_gamma_pwl_fxaa_luma_pipeline_ = VK_NULL_HANDLE;
  VkPipeline swap_apply_gamma_compute_256_entry_table_pipeline_ = VK_NULL_HANDLE;
  VkPipeline swap_apply_gamma_compute_pwl_pipeline_ = VK_NULL_HANDLE;
  VkPipeline swap_apply_gamma_compute_256_entry_table_rb_swap_pipeline_ = VK_NULL_HANDLE;
  VkPipeline swap_apply_gamma_compute_pwl_rb_swap_pipeline_ = VK_NULL_HANDLE;
  VkPipeline swap_apply_gamma_compute_256_entry_table_fxaa_luma_pipeline_ = VK_NULL_HANDLE;
  VkPipeline swap_apply_gamma_compute_pwl_fxaa_luma_pipeline_ = VK_NULL_HANDLE;
  VkPipeline swap_apply_gamma_compute_256_entry_table_fxaa_luma_rb_swap_pipeline_ = VK_NULL_HANDLE;
  VkPipeline swap_apply_gamma_compute_pwl_fxaa_luma_rb_swap_pipeline_ = VK_NULL_HANDLE;
  VkPipeline swap_fxaa_pipeline_ = VK_NULL_HANDLE;
  VkPipeline swap_fxaa_extreme_pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout resolve_downscale_pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline resolve_downscale_pipeline_ = VK_NULL_HANDLE;
  VkBuffer resolve_downscale_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory resolve_downscale_buffer_memory_ = VK_NULL_HANDLE;
  uint32_t resolve_downscale_buffer_size_ = 0;

  VkImage swap_fxaa_source_image_ = VK_NULL_HANDLE;
  VkDeviceMemory swap_fxaa_source_image_memory_ = VK_NULL_HANDLE;
  VkImageView swap_fxaa_source_image_view_ = VK_NULL_HANDLE;
  uint32_t swap_fxaa_source_image_width_ = 0;
  uint32_t swap_fxaa_source_image_height_ = 0;
  uint64_t swap_fxaa_source_image_submission_ = 0;
  VkPipelineStageFlags swap_fxaa_source_stage_mask_ = 0;
  VkAccessFlags swap_fxaa_source_access_mask_ = 0;
  VkImageLayout swap_fxaa_source_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;

  std::array<SwapFramebuffer, ui::vulkan::VulkanPresenter::kMaxActiveGuestOutputImageVersions>
      swap_framebuffers_;

  // Pending pipeline barriers.
  std::vector<VkBufferMemoryBarrier> pending_barriers_buffer_memory_barriers_;
  std::vector<VkImageMemoryBarrier> pending_barriers_image_memory_barriers_;
  struct PendingBarrier {
    VkPipelineStageFlags src_stage_mask = 0;
    VkPipelineStageFlags dst_stage_mask = 0;
    size_t buffer_memory_barriers_offset = 0;
    size_t image_memory_barriers_offset = 0;
  };
  std::vector<PendingBarrier> pending_barriers_;
  PendingBarrier current_pending_barrier_;

  // GPU-local scratch buffer.
  static constexpr VkDeviceSize kScratchBufferSizeIncrement = 16 * 1024 * 1024;
  VkDeviceMemory scratch_buffer_memory_ = VK_NULL_HANDLE;
  VkBuffer scratch_buffer_ = VK_NULL_HANDLE;
  VkDeviceSize scratch_buffer_size_ = 0;
  VkPipelineStageFlags scratch_buffer_last_stage_mask_ = 0;
  VkAccessFlags scratch_buffer_last_access_mask_ = 0;
  uint64_t scratch_buffer_last_usage_submission_ = 0;
  bool scratch_buffer_used_ = false;

  // wwe2k14 patch (Session 7c, TEXLOAD): the texture load scratch ring (AcquireScratchGpuBufferRing). Destroyed with
  // the scratch buffer (DestroyScratchBuffer) or by ReleaseScratchRing.
  struct ScratchRingEntry {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkPipelineStageFlags last_stage_mask = 0;
    VkAccessFlags last_access_mask = 0;
    uint64_t last_usage_submission = 0;
    bool used = false;
  };
  std::array<ScratchRingEntry, kScratchRingSize> scratch_ring_{};
  uint32_t scratch_ring_next_ = 0;
  ScratchRingStats scratch_ring_stats_;
  // Destroys the entry's buffer now if its last submission has completed, otherwise once it has.
  void FreeScratchRingEntry(ScratchRingEntry& entry);

  static constexpr uint32_t kMaxOcclusionQueries = 8192;
  VkQueryPool occlusion_query_pool_ = VK_NULL_HANDLE;
  VkBuffer occlusion_query_readback_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory occlusion_query_readback_memory_ = VK_NULL_HANDLE;
  uint32_t occlusion_query_readback_memory_type_ = UINT32_MAX;
  VkDeviceSize occlusion_query_readback_memory_size_ = 0;
  uint8_t* occlusion_query_readback_mapping_ = nullptr;
  uint32_t occlusion_query_cursor_ = 0;
  bool occlusion_query_resources_available_ = false;
  struct ActiveOcclusionQuery {
    uint32_t sample_count_address = 0;
    uint32_t host_index = UINT32_MAX;
    bool valid = false;
  } active_occlusion_query_;
  struct VertexBufferState {
    uint32_t address = UINT32_MAX;
    uint32_t size = UINT32_MAX;
  };
  std::array<VertexBufferState, 96> vertex_buffer_states_{};
  uint64_t vertex_buffers_in_sync_[2] = {};
  // wwe2k14 patch (Session 6, RESID): vulkan_vertex_residency_check state and counters (defined in
  // command_processor.cpp, created when the setting is first used). Read-only towards the cache above.
  struct VertexResidencyCheck;
  std::unique_ptr<VertexResidencyCheck> vertex_residency_check_;
  VertexResidencyCheck* VertexResidencyCheckBeginDraw();
  void VertexResidencyCheckInSync(VertexResidencyCheck& check, uint32_t vfetch_index);
  void VertexResidencyCheckMatch(VertexResidencyCheck& check, uint32_t vfetch_index,
                                 const xenos::xe_gpu_vertex_fetch_t& vfetch_constant);
  void VertexResidencyCheckRequested(VertexResidencyCheck& check, uint32_t vfetch_index,
                                     const xenos::xe_gpu_vertex_fetch_t& vfetch_constant);
  std::unordered_map<uint64_t, ReadbackBuffer> readback_buffers_;
  std::unordered_map<uint64_t, ReadbackBuffer> memexport_readback_buffers_;

  // The current dynamic state of the graphics pipeline bind point. Note that
  // binding any pipeline to the bind point with static state (even if it's
  // unused, like depth bias being disabled, but the values themselves still not
  // declared as dynamic in the pipeline) invalidates such dynamic state.
  VkViewport dynamic_viewport_;
  VkRect2D dynamic_scissor_;
  // Dynamic fixed-function depth bias, blend constants, stencil state are
  // applicable only to the render target implementations where they are
  // actually involved.
  float dynamic_depth_bias_constant_factor_;
  float dynamic_depth_bias_slope_factor_;
  float dynamic_blend_constants_[4];
  // The stencil values are pre-initialized (to D3D11_DEFAULT_STENCIL_*, and the
  // initial values for front and back are the same for portability subset
  // safety) because they're updated conditionally to avoid changing the back
  // face values when stencil is disabled and the primitive type is changed
  // between polygonal and non-polygonal.
  uint32_t dynamic_stencil_compare_mask_front_ = UINT8_MAX;
  uint32_t dynamic_stencil_compare_mask_back_ = UINT8_MAX;
  uint32_t dynamic_stencil_write_mask_front_ = UINT8_MAX;
  uint32_t dynamic_stencil_write_mask_back_ = UINT8_MAX;
  uint32_t dynamic_stencil_reference_front_ = 0;
  uint32_t dynamic_stencil_reference_back_ = 0;
  bool dynamic_viewport_update_needed_;
  bool dynamic_scissor_update_needed_;
  bool dynamic_depth_bias_update_needed_;
  bool dynamic_blend_constants_update_needed_;
  bool dynamic_stencil_compare_mask_front_update_needed_;
  bool dynamic_stencil_compare_mask_back_update_needed_;
  bool dynamic_stencil_write_mask_front_update_needed_;
  bool dynamic_stencil_write_mask_back_update_needed_;
  bool dynamic_stencil_reference_front_update_needed_;
  bool dynamic_stencil_reference_back_update_needed_;

  // Currently used samplers.
  std::vector<std::pair<VulkanTextureCache::SamplerParameters, VkSampler>> current_samplers_vertex_;
  std::vector<std::pair<VulkanTextureCache::SamplerParameters, VkSampler>> current_samplers_pixel_;

  // Cache render pass currently started in the command buffer with the
  // framebuffer. For dynamic rendering, current_render_pass_ is VK_NULL_HANDLE
  // but in_render_pass_ is true.
  VkRenderPass current_render_pass_;
  const VulkanRenderTargetCache::Framebuffer* current_framebuffer_;
  bool in_render_pass_ = false;

  // Currently bound graphics pipeline, either from the pipeline cache (with
  // potentially deferred creation - current_external_graphics_pipeline_ is
  // VK_NULL_HANDLE in this case) or a non-Xenos one
  // (current_guest_graphics_pipeline_ is VK_NULL_HANDLE in this case).
  // TODO(Triang3l): Change to a deferred compilation handle.
  VkPipeline current_guest_graphics_pipeline_;
  VkPipeline current_external_graphics_pipeline_;
  VkPipeline current_external_compute_pipeline_;

  // Pipeline layout of the current guest graphics pipeline.
  const PipelineLayout* current_guest_graphics_pipeline_layout_;
  VkDescriptorBufferInfo
      current_constant_buffer_infos_[SpirvShaderTranslator::kConstantBufferCount];
  // Whether up-to-date data has been written to constant (uniform) buffers, and
  // the buffer infos in current_constant_buffer_infos_ point to them.
  uint32_t current_constant_buffers_up_to_date_;
  VkDescriptorSet current_graphics_descriptor_sets_[SpirvShaderTranslator::kDescriptorSetCount];
  // Whether descriptor sets in current_graphics_descriptor_sets_ point to
  // up-to-date data.
  uint32_t current_graphics_descriptor_set_values_up_to_date_;
  // Whether the descriptor sets currently bound to the command buffer - only
  // low bits for the descriptor set layouts that remained the same are kept
  // when changing the pipeline layout. May be out of sync with
  // current_graphics_descriptor_set_values_up_to_date_, but should be ensured
  // to be a subset of it at some point when it becomes important; bits for
  // non-existent descriptor set layouts may also be set, but need to be ignored
  // when they start to matter.
  uint32_t current_graphics_descriptor_sets_bound_up_to_date_;
  static_assert(SpirvShaderTranslator::kDescriptorSetCount <=
                    sizeof(current_graphics_descriptor_set_values_up_to_date_) * CHAR_BIT,
                "Bit fields storing descriptor set validity must be large enough");
  static_assert(SpirvShaderTranslator::kDescriptorSetCount <=
                    sizeof(current_graphics_descriptor_sets_bound_up_to_date_) * CHAR_BIT,
                "Bit fields storing descriptor set validity must be large enough");

  // Float constant usage masks of the last draw call.
  uint64_t current_float_constant_map_vertex_[4];
  uint64_t current_float_constant_map_pixel_[4];

  // System shader constants.
  SpirvShaderTranslator::SystemConstants system_constants_;
  // wwe2k14 patch (Session 6, CP9): vulkan_system_constants_skip state (defined in
  // command_processor.cpp, created when the setting is first used).
  struct SystemConstantsSkip;
  std::unique_ptr<SystemConstantsSkip> system_constants_skip_;

  // Temporary storage for memexport stream constants used in the draw.
  std::vector<draw_util::MemExportRange> memexport_ranges_;
};

}  // namespace rex::graphics::vulkan
