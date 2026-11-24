#pragma once
/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include <rex/graphics/shared_memory.h>
#include <rex/memory.h>
#include <rex/ui/vulkan/upload_buffer_pool.h>

namespace rex::graphics::vulkan {

class VulkanCommandProcessor;

class VulkanSharedMemory : public SharedMemory {
 public:
  VulkanSharedMemory(VulkanCommandProcessor& command_processor, memory::Memory& memory,
                     VkPipelineStageFlags guest_shader_pipeline_stages);
  ~VulkanSharedMemory() override;

  bool Initialize();
  void Shutdown(bool from_destructor = false);

  void CompletedSubmissionUpdated();
  void EndSubmission();

  enum class Usage {
    // Index buffer, vfetch, compute read, transfer source.
    kRead,
    // Index buffer, vfetch, memexport.
    kGuestDrawReadWrite,
    kComputeWrite,
    kTransferDestination,
  };
  // Inserts a pipeline barrier for the target usage, also ensuring consecutive
  // read-write accesses are ordered with each other.
  // wwe2k14 patch (Session 7b, BARNARROW): access_stages = the pipeline stages in which the GPU work recorded
  // after this call (until the next Use) reads or writes the buffer, a subset of the usage's stages; 0 (the
  // default) = all of the usage's stages (conservative). Tracked for vulkan_shared_memory_narrow_barrier.
  void Use(Usage usage, std::pair<uint32_t, uint32_t> written_range = {},
           VkPipelineStageFlags access_stages = 0);
  // wwe2k14 patch (Session 79, gpu_native_truth_defer_memory_resolves): one more compute write right after a
  // kComputeWrite use, to a range the writes since that use do not overlap (the caller's guarantee): no barrier between
  // them - the next barrier commits all of them (the written range grows to cover both). Use() if the last use was
  // another one.
  void UseAlsoComputeWrite(std::pair<uint32_t, uint32_t> written_range);
  // (Session 79 review fix: the tracked use, so a write recorded in the middle of another use - a deferred batch
  // flushed inside a draw - can hand the tracking back to it: Use(last_usage(), last_written_range()) afterwards.)
  Usage last_usage() const { return last_usage_; }
  std::pair<uint32_t, uint32_t> last_written_range() const { return last_written_range_; }

  // wwe2k14 patch (Session 7b, BARNARROW): access_stages for a guest draw: vertex input (a guest DMA index
  // buffer in the shared memory) and every guest vertex-side stage (vertex fetch, index loads in the vertex /
  // tessellation evaluation shader, vertex shader memexport), plus the fragment shader if the pixel shader
  // vertex-fetches or memexports (the only shared memory accesses the SPIR-V translator emits in a pixel
  // shader; the FSI EDRAM is a separate buffer).
  VkPipelineStageFlags GetGuestDrawAccessStages(bool pixel_shader_accesses_shared_memory) const {
    return VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
           (guest_shader_pipeline_stages_ & ~VkPipelineStageFlags(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)) |
           (pixel_shader_accesses_shared_memory ? VkPipelineStageFlags(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
                                                : VkPipelineStageFlags(0));
  }
  // wwe2k14 patch (Session 7b, BARNARROW): at each frame boundary - the cpf_bar timeline event and the
  // [s7][BARNARROW] log about every 300 frames.
  void RecordNarrowBarrierFrameCounters(uint64_t frame);

  VkBuffer buffer() const { return buffer_; }

  // wwe2k14 patch (Session 21, HOIST): vulkan_upload_hoist. An upload whose pages no GPU command recorded earlier in
  // the current submission accessed (requested, read in sync, or written) is copied in the submission's hoist
  // stream, which runs before everything else in its command buffer, instead of in place (no barriers or render
  // pass break at the draw). Accesses are tracked per page for the current submission. Command processor thread.
  void OnRangesRequested(const std::pair<uint32_t, uint32_t>* ranges, size_t count) override;
  void MarkHoistAccess(uint32_t start, uint32_t length);
  // wwe2k14 patch (Session 95, the [vinput] line): whether any page of the range was uploaded in the current submission. Command processor thread.
  bool UploadedThisSubmission(uint32_t start, uint32_t length);
  // At each frame boundary: the cpf_hoist timeline event and the [s21][HOIST] log about every 300 frames.
  void RecordHoistFrameCounters(uint64_t frame);

  // wwe2k14 patch (Session 7b, REBAR): for the command processor's cpf_rebar events and [s7][REBAR] summary
  // (command processor thread only). The upload buffer pool (null outside Initialize..Shutdown); the host
  // time of the UploadRanges copies into it (rex::chrono::Clock::QueryHostTickCount ticks, counted while the
  // timeline and vulkan_upload_copy_timing are on) and the gpu_upload_streaming_copy_verify checks skipped
  // because it may be device-local, both since creation.
  const ui::vulkan::VulkanUploadBufferPool* upload_buffer_pool() const { return upload_buffer_pool_.get(); }
  uint64_t upload_copy_ticks() const { return upload_copy_ticks_; }
  uint64_t upload_verify_skipped() const { return upload_verify_skipped_; }

  // wwe2k14 patch (Session 94, DIRECT): vulkan_shared_memory_direct. On unified memory without sparse binding the
  // 512 MiB buffer sits in a host-visible type and is mapped once; a page not touched by any GPU work that is still
  // in flight (per-page "last submission that touched it" stamps, folded from the hoist bitmap when a submission
  // starts) nor earlier in the submission being recorded is written by the CPU straight into the mapping instead of
  // staged + copied. Command processor thread only, like UploadRanges.
  struct DirectStats {
    uint64_t calls = 0;             // UploadRanges calls
    uint64_t pages_direct = 0;      // pages written into the mapping (mode 1: pages that would have been)
    uint64_t pages_inflight = 0;    // staged: touched by a submission that has not completed
    uint64_t pages_this_sub = 0;    // staged: touched earlier in this submission, or an access without a known range
    uint64_t calls_no_barrier = 0;  // calls with every range direct (no barrier, no render pass break)
    uint64_t flushes = 0;           // vkFlushMappedMemoryRanges calls (non-coherent types)
  };
  void RecordDirectFrameCounters(uint64_t frame);
  bool direct_mapped() const { return direct_mapping_ != nullptr; }

 protected:
  bool AllocateSparseHostGpuMemoryRange(uint32_t offset_allocations,
                                        uint32_t length_allocations) override;

  bool UploadRanges(const std::vector<std::pair<uint32_t, uint32_t>>& upload_page_ranges) override;

  // wwe2k14 patch (Session 7b, UPSKIP): gpu_upload_skip_unchanged_verify.
  bool RecordUploadSkipVerify(uint32_t page, const uint8_t* bytes) override;
  // wwe2k14 patch (Session 7b, UPSKIP review): async submissions whose commands were dropped.
  uint64_t GetLostUploadCount() const override;

 private:
  // wwe2k14 patch (Session 7b, UPSKIP): gpu_upload_skip_unchanged_verify - a ring of page-sized readback slots
  // (copies of a page of buffer_ recorded at the point the page would have been skipped) and the guest bytes
  // that were hashed there, compared in order once their submissions have completed. Command processor thread.
  void CheckUploadSkipVerifyReadbacks();
  void ShutdownUploadSkipVerify();
  static constexpr uint32_t kUploadSkipVerifySlots = 1024;
  struct UploadSkipVerifySlot {
    uint64_t submission;
    uint64_t lost_upload_count;  // GetLostUploadCount when recorded (changed: the readback may not have run)
    uint32_t page;
  };
  VkBuffer upload_skip_verify_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory upload_skip_verify_memory_ = VK_NULL_HANDLE;
  uint8_t* upload_skip_verify_mapping_ = nullptr;
  bool upload_skip_verify_coherent_ = false;
  bool upload_skip_verify_unavailable_ = false;  // creation failed, not retried
  std::vector<UploadSkipVerifySlot> upload_skip_verify_slots_;
  std::vector<uint8_t> upload_skip_verify_hashed_bytes_;
  uint32_t upload_skip_verify_head_ = 0;
  uint32_t upload_skip_verify_count_ = 0;

  void GetUsageMasks(Usage usage, VkPipelineStageFlags& stage_mask,
                     VkAccessFlags& access_mask) const;

  VulkanCommandProcessor& command_processor_;
  VkPipelineStageFlags guest_shader_pipeline_stages_;

  VkBuffer buffer_ = VK_NULL_HANDLE;
  uint32_t buffer_memory_type_;
  // Single for non-sparse, every allocation so far for sparse.
  std::vector<VkDeviceMemory> buffer_memory_;

  Usage last_usage_;
  std::pair<uint32_t, uint32_t> last_written_range_;

  std::unique_ptr<ui::vulkan::VulkanUploadBufferPool> upload_buffer_pool_;
  std::vector<VkBufferCopy> upload_regions_;

  // wwe2k14 patch (Session 6, RESID): gpu_upload_streaming_copy - the UploadRanges copy of one chunk, and its
  // counters (only counted while the setting is on), logged about every 300 frames from EndSubmission.
  // Command processor thread only, like UploadRanges.
  void CopyUploadChunkStreaming(uint8_t* dest, uint32_t guest_start, size_t size);
  void ReportUploadCopyStats(uint64_t frame);
  struct UploadCopyStats {
    uint64_t window_first_frame = UINT64_MAX;
    uint64_t memcpy_chunks = 0;
    uint64_t memcpy_bytes = 0;
    uint64_t streamed_chunks = 0;
    uint64_t streamed_bytes = 0;
    uint64_t verify_checks = 0;
    uint64_t verify_mismatches = 0;
    uint64_t verify_mismatches_raced = 0;
    uint64_t verify_mismatches_total = 0;
    uint64_t verify_mismatches_logged = 0;
  };
  UploadCopyStats upload_copy_stats_;

  // wwe2k14 patch (Session 7b, BARNARROW): vulkan_shared_memory_narrow_barrier. The pipeline stages that may
  // have accessed buffer_ since the last barrier Use pushed on it, plus one stage of that barrier's destination
  // scope (a later barrier with these stages in its source scope chains to it, which covers every access before
  // it). Kept whether the setting is on or off, so it can be toggled at any upload. Command processor thread
  // only, like Use.
  void NarrowUploadBarrierSource(VkPipelineStageFlags& src_stage_mask, VkAccessFlags& src_access_mask);
  void CountNarrowBarrierBatch();
  void FlushMergedHoistCopies();  // wwe2k14 patch (Session 95, HOISTMERGE)
  VkPipelineStageFlags stages_accessed_since_barrier_ = VK_PIPELINE_STAGE_TRANSFER_BIT;
  // The upload barrier Use just pushed has no fragment stage in its source scope (UploadRanges checks what
  // else is batched with it).
  bool narrow_barrier_batch_check_ = false;
  struct NarrowBarrierStats {
    uint32_t upload_barriers = 0;      // cpf_bar a: barriers pushed by Use for an upload copy
    uint32_t narrowed = 0;             // cpf_bar b: of those, no fragment stage in the source scope
    uint32_t narrowed_no_compute = 0;  // cpf_bar c: of those, also no compute stage
    uint32_t from_read = 0;            // upload barriers after kRead / kGuestDrawReadWrite (the ones narrowed)
    uint32_t tracked_fragment = 0;     // of those, a pixel shader accessed the buffer since the last barrier
    uint32_t tracked_compute = 0;      // of those, a compute dispatch accessed it since the last barrier
    uint32_t would_narrow = 0;         // of those, the narrowed mask has no fragment stage (on or off)
    uint32_t not_narrowable = 0;       // of those, kept full (a write access bit not covered; not expected)
    uint32_t batched_fragment = 0;     // narrowed, but submitted with another pending barrier from fragment
  };
  NarrowBarrierStats narrow_barrier_frame_stats_;
  NarrowBarrierStats narrow_barrier_window_stats_;
  uint64_t narrow_barrier_window_first_frame_ = UINT64_MAX;
  // wwe2k14 patch (Session 7b, REBAR): see upload_copy_ticks().
  // wwe2k14 patch (Session 21, HOIST): see OnRangesRequested. hoist_accessed_ has one bit per page of buffer_,
  // valid for hoist_accessed_submission_ (cleared lazily when the submission changes); hoist_accessed_all_ = some
  // access had no known range (every page counts as accessed until the submission ends).
  bool HoistPagesUntouched(const std::vector<std::pair<uint32_t, uint32_t>>& upload_page_ranges);
  void HoistMarkPages(uint32_t page_first, uint32_t page_count);
  void HoistSyncSubmission();
  std::vector<uint64_t> hoist_accessed_;
  uint64_t hoist_accessed_submission_ = UINT64_MAX;
  bool hoist_accessed_all_ = false;
  struct HoistStats {
    uint64_t calls = 0;           // UploadRanges calls
    uint64_t pages = 0;
    uint64_t hoistable_calls = 0;  // no page accessed earlier in the submission
    uint64_t hoistable_pages = 0;
    uint64_t hoisted_calls = 0;    // copied in the hoist stream (vulkan_upload_hoist 2)
    uint64_t hoisted_pages = 0;
    uint64_t blocked_all = 0;      // not hoistable because of an access without a known range
  };
  HoistStats hoist_frame_stats_;
  HoistStats hoist_window_stats_;
  uint64_t hoist_window_first_frame_ = UINT64_MAX;
  uint64_t hoist_window_frames_ = 0;

  // wwe2k14 patch (Session 94, DIRECT): see DirectStats.
  void DirectFoldStamps();
  // Splits the call's page ranges into direct runs and the rest (counts every page).
  void DirectClassify(const std::vector<std::pair<uint32_t, uint32_t>>& ranges,
                      std::vector<std::pair<uint32_t, uint32_t>>& direct_runs,
                      std::vector<std::pair<uint32_t, uint32_t>>& staged_runs);
  void DirectFlushRanges();
  int32_t direct_mode_ = 0;  // resolved at Initialize: 0 off, 1 count only, 2 active (mapped)
  uint8_t* direct_mapping_ = nullptr;
  bool direct_coherent_ = true;
  std::vector<uint64_t> direct_stamp_;  // per page: last submission that touched it
  uint64_t direct_all_stamp_ = 0;
  std::vector<std::pair<uint32_t, uint32_t>> direct_flush_pages_;  // pages written since the last EndSubmission
  std::vector<std::pair<uint32_t, uint32_t>> direct_runs_, direct_staged_;
  DirectStats direct_frame_stats_, direct_window_stats_;
  uint64_t direct_window_first_frame_ = UINT64_MAX;
  uint64_t direct_window_frames_ = 0;
  uint64_t upload_copy_ticks_ = 0;
  uint64_t upload_verify_skipped_ = 0;
  bool upload_verify_skip_logged_ = false;
  // wwe2k14 patch (Session 95): see UploadedThisSubmission. (Last in the class: the exe never sees past here.)
  void NoteUploadedPages(uint32_t page_first, uint32_t page_count);
  std::vector<uint64_t> uploaded_pages_;
  uint64_t uploaded_submission_ = UINT64_MAX;
};

}  // namespace rex::graphics::vulkan
