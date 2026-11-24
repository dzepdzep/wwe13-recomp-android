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
#include <cstring>
#include <utility>
#include <vector>

#include <rex/assert.h>
#include <rex/chrono/clock.h>
#include <rex/cvar.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/deferred_command_buffer.h>
#include <rex/graphics/vulkan/shared_memory.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/perf/timeline.h>
#include <rex/platform.h>
#include <rex/ui/vulkan/util.h>
#include "native/shader_swap.h"  // (Session 95: VertexMirrorLatched)

#if REX_ARCH_AMD64
#include <emmintrin.h>
#endif

REXCVAR_DEFINE_BOOL(vulkan_sparse_shared_memory, true, "GPU/Vulkan",
                    "Use sparse shared memory on Vulkan")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

// wwe2k14 patch (Session 94, DIRECT): see VulkanSharedMemory::UploadRanges and DirectClassify.
REXCVAR_DEFINE_INT32(vulkan_shared_memory_direct, 0, "GPU/Vulkan",
                     "wwe2k14 (Session 94): write guest pages straight into the mapped shared memory buffer "
                     "instead of staging + copying them, for pages no GPU work in flight or earlier in the "
                     "submission being recorded touches (per-page submission stamps). -1 = auto (only when the "
                     "buffer is not sparse, sits in a host-visible type and the device has unified memory); 0 = "
                     "off; 1 = count only (decide and count, still copy; works with sparse too); 2 = force where "
                     "a host-visible device-local type exists. Needs vulkan_upload_hoist > 0 and no "
                     "gpu_upload_skip_unchanged_verify. Takes effect after a restart; [s94][DIRECT] in the log")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

// wwe2k14 patch (Session 6, RESID): UploadRanges copies guest pages into the upload buffer (host-visible,
// uncached memory where the device has it) with memcpy, ~1.2% of the command processor in stacks_067 while
// clear_memory_page_state re-uploads CPU data every frame. Xenia Canary copies chunks over 8 KB with a
// streaming copy (memory::vastcpy, 4d15f0afea / a092551735; docs/xenia_canary_compare.md section 2).
REXCVAR_DEFINE_BOOL(gpu_upload_streaming_copy, false, "GPU/Vulkan",
                    "Copy shared memory upload chunks of 8 KB or more into the upload buffer with "
                    "non-temporal (streaming) SSE2 stores and an sfence, instead of memcpy; the same bytes. "
                    "Read per upload (toggle-safe)");
REXCVAR_DEFINE_BOOL(gpu_upload_streaming_copy_verify, false, "GPU/Vulkan",
                    "With gpu_upload_streaming_copy: compare every streamed chunk with guest memory (slow - "
                    "reads the upload buffer back), count and log mismatches ([s6][RESID])");
// wwe2k14 patch (Session 7b, REBAR): UploadRanges copies guest pages into the upload buffer pool and records a
// vkCmdCopyBuffer from there into the shared memory buffer - ~800-1,100 small copies per heavy frame while
// clear_memory_page_state re-uploads CPU data, each reading its source from system memory across PCIe. In
// device-local host-visible memory (Resizable BAR) the copy source is in VRAM. The pool's memory type is chosen
// when its first page is created (Initialize, then the first upload), so this is read once, at Initialize.
REXCVAR_DEFINE_BOOL(vulkan_upload_pool_device_local, false, "GPU/Vulkan",
                    "Put the shared memory upload buffers (the source of the guest memory upload copies) in "
                    "device-local host-visible memory (Resizable BAR) if the device has such memory in a "
                    "heap of at least vulkan_device_local_upload_min_heap_mb, instead of system memory; the "
                    "same bytes are written and copied. Takes effect after a restart; the [s7][REBAR] log "
                    "says which memory is used")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);
// Defined in command_processor.cpp.
REXCVAR_DECLARE(int32_t, vulkan_device_local_upload_min_heap_mb);
REXCVAR_DECLARE(int32_t, gpu_diag_stats_now);  // Session 95 (the diagnostic run)
// wwe2k14 patch (Session 21, HOIST): see VulkanSharedMemory::OnRangesRequested and UploadRanges.
REXCVAR_DEFINE_INT32(vulkan_upload_hoist, 0, "GPU/Vulkan",
                     "wwe2k14 (Session 21): 0 = off; 1 = count the shared memory uploads whose pages no GPU "
                     "command recorded earlier in the same submission accessed (cpf_hoist, [s21][HOIST]); 2 = "
                     "also copy those in a stream that runs first in the submission's command buffer (one "
                     "barrier pair per submission) instead of in place (a barrier pair and a render pass break "
                     "each). Read per upload; a submission with any access recorded while it was 0 hoists "
                     "nothing");
REXCVAR_DECLARE(bool, gpu_upload_skip_unchanged_verify);
// wwe2k14 patch (Session 95, HOISTMERGE): GPU audit row 3. 44-57 vkCmdCopyBuffer calls a frame sit in the hoist stream,
// one per UploadRanges call (a call only batches its own regions). With this, the hoisted copies of a whole submission
// from the same source buffer are one vkCmdCopyBuffer with all the regions, recorded when the submission ends
// (EndSubmission, before the hoist stream's closing barrier). Same bytes, same destination pages, same barriers; only
// the number of copy commands falls. Hoisted uploads only (in-place ones are untouched).
REXCVAR_DEFINE_BOOL(vulkan_upload_hoist_merge, false, "GPU/Vulkan",
                    "Session 95 (HOISTMERGE): record the hoisted shared memory upload copies of a submission as one "
                    "multi-region vkCmdCopyBuffer per source buffer, at the end of the submission, instead of one "
                    "command per UploadRanges call. Needs vulkan_upload_hoist 2. [s95][HOISTMERGE] in the log");
namespace {
// One SharedMemory exists; command processor thread only (UploadRanges and EndSubmission).
VkBuffer g_merge_src = VK_NULL_HANDLE;
std::vector<VkBufferCopy> g_merge_regions;
uint64_t g_merge_calls_in = 0, g_merge_calls_out = 0, g_merge_regions_total = 0, g_merge_frames_logged = 0;
}  // namespace
REXCVAR_DEFINE_BOOL(vulkan_debug_skip_draw_read_barrier, false, "GPU/Vulkan",
                    "Session 59 diagnostic (racy, measurement only): no shared-memory barrier for a read after a "
                    "compute write (resolves into memory, then draws) - the ceiling of range-aware read barriers")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 7b, REBAR): timing every UploadRanges chunk copy costs two host clock reads per chunk
// (about 2,000 per heavy frame), so it is opt-in rather than tied to the timeline - timeline runs without it
// measure what they did before. Turn it on for both runs of a vulkan_upload_pool_device_local on / off
// comparison.
REXCVAR_DEFINE_BOOL(vulkan_upload_copy_timing, false, "GPU/Vulkan",
                    "While the timeline is on: time the copies of guest memory into the shared memory upload "
                    "buffers (two host clock reads per upload chunk) for cpf_rebar a and the [s7][REBAR] "
                    "summary (0 without it). Read at every upload");

// wwe2k14 patch (Session 7b, BARNARROW): the barrier before each shared memory upload copy (Use with
// kTransferDestination after kRead / kGuestDrawReadWrite) has every stage that may read the buffer in its
// source scope, the fragment shader included - with the FSI render target path that waits for all the pixel
// work in flight, ~1,000 times per heavy frame while clear_memory_page_state re-uploads pages every frame.
// Use tracks the stages that actually accessed the buffer since its last barrier; with this setting the upload
// barrier waits only for those. The same hazards are covered (see NarrowUploadBarrierSource).
REXCVAR_DEFINE_BOOL(vulkan_shared_memory_narrow_barrier, false, "GPU/Vulkan",
                    "Before each shared memory upload copy, wait only for the pipeline stages that accessed "
                    "the shared memory buffer since its last barrier (draws: vertex stages, plus fragment if "
                    "the pixel shader vertex-fetches or memexports; compute texture loads and resolves; "
                    "copies) instead of every stage that may read it (fragment included = all the pixel work "
                    "in flight). Covers the same hazards. Read per upload (toggle-safe); cpf_bar counters and "
                    "a [s7][BARNARROW] log about every 300 frames either way");

// wwe2k14 patch (Session 7, PROTECT): page protection counters, defined in src/system/xmemory.cpp
// (rexruntime exports all symbols) - declared here, not in rex/system/xmemory.h, which the game executable
// includes.
namespace rex::memory {
void ResetPhysicalProtectStats();
void LogPhysicalProtectStats(uint64_t first_frame, uint64_t last_frame);
}  // namespace rex::memory
// wwe2k14 patch (Session 49, RAM TIES): the write-watch fault counters, defined in src/system/mmio_handler.cpp.
namespace rex::runtime {
void ResetWatchFaultStats();
void LogWatchFaultStats(uint64_t first_frame, uint64_t last_frame);
}  // namespace rex::runtime

namespace {

// wwe2k14 patch (Session 7, PROTECT): what UploadRanges hands to SharedMemory::MakeRangeValid (each chunk is
// one MakeRangeValid -> EnableAccessCallbacks in the three physical views), logged with the page protection
// counters about every 300 frames. Command processor thread only, like UploadRanges.
struct ProtectUploadStats {
  uint64_t window_first_frame = UINT64_MAX;
  uint64_t calls = 0;        // UploadRanges calls
  uint64_t multi_range = 0;  // of them with more than one upload range
  uint64_t ranges = 0;       // upload ranges
  uint64_t chunks = 0;       // MakeRangeValid calls (ranges split at upload buffer page ends)
  uint64_t pages = 0;        // pages made valid
};
ProtectUploadStats protect_upload_stats;

// wwe2k14 patch (Session 6, RESID): gpu_upload_streaming_copy.
constexpr size_t kUploadStreamingCopyMinSize = 8192;

// Copies size bytes like memcpy (the ranges must not overlap), storing whole 64-byte lines of the destination
// with non-temporal stores, so the destination isn't read for ownership or kept in the cache. The bytes before
// the first and after the last whole destination line are copied with memcpy. Ends with an sfence: non-
// temporal stores are weakly ordered, and the fence makes them globally visible before any store this thread
// makes afterwards (handing the commands to the submit thread, vkQueueSubmit, vkFlushMappedMemoryRanges).
void StreamingCopy(uint8_t* dest, const uint8_t* source, size_t size) {
#if REX_ARCH_AMD64
  size_t head = std::min(size_t(0 - reinterpret_cast<uintptr_t>(dest)) & 63, size);
  if (head) {
    std::memcpy(dest, source, head);
    dest += head;
    source += head;
    size -= head;
  }
  __m128i* dest_vectors = reinterpret_cast<__m128i*>(dest);
  const __m128i* source_vectors = reinterpret_cast<const __m128i*>(source);
  for (size_t lines = size >> 6; lines; --lines) {
    __m128i vector_0 = _mm_loadu_si128(source_vectors);
    __m128i vector_1 = _mm_loadu_si128(source_vectors + 1);
    __m128i vector_2 = _mm_loadu_si128(source_vectors + 2);
    __m128i vector_3 = _mm_loadu_si128(source_vectors + 3);
    _mm_stream_si128(dest_vectors, vector_0);
    _mm_stream_si128(dest_vectors + 1, vector_1);
    _mm_stream_si128(dest_vectors + 2, vector_2);
    _mm_stream_si128(dest_vectors + 3, vector_3);
    source_vectors += 4;
    dest_vectors += 4;
  }
  size_t tail = size & 63;
  if (tail) {
    std::memcpy(dest_vectors, source_vectors, tail);
  }
  _mm_sfence();
#else
  std::memcpy(dest, source, size);
#endif
}

}  // namespace

namespace rex::graphics::vulkan {

VulkanSharedMemory::VulkanSharedMemory(VulkanCommandProcessor& command_processor,
                                       memory::Memory& memory,
                                       VkPipelineStageFlags guest_shader_pipeline_stages)
    : SharedMemory(memory),
      command_processor_(command_processor),
      guest_shader_pipeline_stages_(guest_shader_pipeline_stages) {}

VulkanSharedMemory::~VulkanSharedMemory() {
  Shutdown(true);
}

// wwe2k14 patch (Session 95, Lane D review 6): set while Initialize runs again as the fallback after a plain buffer failed to
// allocate with sparse switched off by a GPU rule (Turnip): sparse binding is tried then even though the cvar says false.
static bool g_sparse_retry = false;

bool VulkanSharedMemory::Initialize() {
  InitializeCommon();

  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  const VkBufferCreateFlags sparse_flags =
      VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;

  // Try to create a sparse buffer.
  VkBufferCreateInfo buffer_create_info;
  buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.pNext = nullptr;
  buffer_create_info.flags = sparse_flags;
  buffer_create_info.size = kBufferSize;
  buffer_create_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;  // (Session 95: attribute draws bind it directly)
  buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  buffer_create_info.queueFamilyIndexCount = 0;
  buffer_create_info.pQueueFamilyIndices = nullptr;
  if ((REXCVAR_GET(vulkan_sparse_shared_memory) || g_sparse_retry) &&
      vulkan_device->properties().sparseResidencyBuffer) {
    if (dfn.vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer_) == VK_SUCCESS) {
      VkMemoryRequirements buffer_memory_requirements;
      dfn.vkGetBufferMemoryRequirements(device, buffer_, &buffer_memory_requirements);
      if (rex::bit_scan_forward(buffer_memory_requirements.memoryTypeBits &
                                    vulkan_device->memory_types().device_local,
                                &buffer_memory_type_)) {
        uint32_t allocation_size_log2;
        rex::bit_scan_forward(std::max(uint64_t(buffer_memory_requirements.alignment), uint64_t(1)),
                              &allocation_size_log2);
        if (allocation_size_log2 < kBufferSizeLog2) {
          // Maximum of 1024 allocations in the worst case for all of the
          // buffer because of the overall 4096 allocation count limit on
          // Windows drivers.
          InitializeSparseHostGpuMemory(std::max(
              allocation_size_log2,
              std::max(kHostGpuMemoryOptimalSparseAllocationLog2, kBufferSizeLog2 - uint32_t(10))));
        } else {
          // Shouldn't happen on any real platform, but no point allocating the
          // buffer sparsely.
          dfn.vkDestroyBuffer(device, buffer_, nullptr);
          buffer_ = VK_NULL_HANDLE;
        }
      } else {
        REXGPU_ERROR(
            "Shared memory: Failed to get a device-local Vulkan memory type "
            "for the sparse buffer");
        dfn.vkDestroyBuffer(device, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
      }
    } else {
      REXGPU_ERROR("Shared memory: Failed to create the {} MB Vulkan sparse buffer",
                   kBufferSize >> 20);
    }
  }

  // wwe2k14 patch (Session 95, Lane D): one line saying which kind the shared memory is and why, for the launcher's report.txt.
  {
    const bool sparse_used = buffer_ != VK_NULL_HANDLE;
    VkPhysicalDeviceProperties real_properties = {};
    vulkan_device->vulkan_instance()->functions().vkGetPhysicalDeviceProperties(vulkan_device->physical_device(),
                                                                                &real_properties);
    const char* why = sparse_used ? "sparse-bound" : "one plain buffer";
    const char* reason = "";
    if (sparse_used && g_sparse_retry) reason = " (the plain buffer could not be allocated: sparse binding used instead of the rule)";
    if (!sparse_used) {
      if (!REXCVAR_GET(vulkan_sparse_shared_memory)) reason = " (vulkan_sparse_shared_memory=false, set by the settings or a GPU rule)";
      else if (!vulkan_device->properties().sparseResidencyBuffer) reason = " (the device has no sparseResidencyBuffer)";
      else reason = " (the sparse buffer could not be created)";
    }
    REXGPU_INFO("[sparse] shared memory: {}{}; sparseResidencyBuffer={}, residencyNonResidentStrict={}", why, reason,
                vulkan_device->properties().sparseResidencyBuffer ? "true" : "false",
                real_properties.sparseProperties.residencyNonResidentStrict ? "true" : "false");
  }
  // The plain buffer failed while sparse was off only by the cvar (a GPU rule such as Turnip's): try sparse once before giving up.
  auto plain_failed = [&]() -> bool {
    if (!g_sparse_retry && !REXCVAR_GET(vulkan_sparse_shared_memory) &&
        vulkan_device->properties().sparseResidencyBuffer) {
      REXGPU_WARN("[sparse] the plain {} MB shared memory buffer failed; retrying with sparse binding", kBufferSize >> 20);
      g_sparse_retry = true;
      Shutdown();
      const bool ok = Initialize();
      g_sparse_retry = false;
      return ok;
    }
    Shutdown();
    return false;
  };
  // Create a non-sparse buffer if there were issues with the sparse buffer.
  if (buffer_ == VK_NULL_HANDLE) {
    REXGPU_INFO(
        "Vulkan sparse binding is not used for shared memory emulation - video "
        "memory usage may increase significantly because a full {} MB buffer "
        "will be created",
        kBufferSize >> 20);
    buffer_create_info.flags &= ~sparse_flags;
    if (dfn.vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer_) != VK_SUCCESS) {
      REXGPU_ERROR("Shared memory: Failed to create the {} MB Vulkan buffer", kBufferSize >> 20);
      return plain_failed();
    }
    VkMemoryRequirements buffer_memory_requirements;
    dfn.vkGetBufferMemoryRequirements(device, buffer_, &buffer_memory_requirements);
    if (!rex::bit_scan_forward(
            buffer_memory_requirements.memoryTypeBits & vulkan_device->memory_types().device_local,
            &buffer_memory_type_)) {
      REXGPU_ERROR(
          "Shared memory: Failed to get a device-local Vulkan memory type for "
          "the buffer");
      Shutdown();
      return false;
    }
    // wwe2k14 patch (Session 94, DIRECT): a host-visible type to map, when wanted.
    const uint32_t default_memory_type = buffer_memory_type_;
    bool direct_type_chosen = false;
    {
      const int32_t mode = REXCVAR_GET(vulkan_shared_memory_direct);
      const ui::vulkan::VulkanDevice::MemoryTypes& mt = vulkan_device->memory_types();
      uint32_t supported =
          buffer_memory_requirements.memoryTypeBits & mt.device_local & mt.host_visible;
      // Unified memory: every device-local heap has a host-visible type (lazily allocated device-local types on
      // Adreno / Mali are not host-visible, so not a per-type test).
      bool unified = true;
      {
        VkPhysicalDeviceMemoryProperties mp;
        vulkan_device->vulkan_instance()->functions().vkGetPhysicalDeviceMemoryProperties(
            vulkan_device->physical_device(), &mp);
        for (uint32_t h = 0; h < mp.memoryHeapCount; ++h) {
          if (!(mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) {
            continue;
          }
          bool heap_visible = false;
          for (uint32_t t = 0; t < mp.memoryTypeCount; ++t) {
            if (mp.memoryTypes[t].heapIndex == h &&
                (mp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
              heap_visible = true;
            }
          }
          unified = unified && heap_visible;
        }
        // Session 94 M2a: a host-visible device-local heap smaller than the buffer (the PC without Resizable BAR: a 214 MB
        // heap) cannot hold it - drop such types before trying (the attempt was a validation error, VUID-01713).
        for (uint32_t t = 0; t < mp.memoryTypeCount; ++t) {
          if ((supported >> t & 1) && mp.memoryHeaps[mp.memoryTypes[t].heapIndex].size < kBufferSize) {
            supported &= ~(uint32_t(1) << t);
          }
        }
      }
      const char* reason = "off";
      if (mode == 0) {
        reason = "cvar 0";
      } else if (mode == 1) {
        reason = "count only, no mapping";
      } else if (!supported) {
        reason = "no host-visible device-local type for the buffer";
      } else if (mode == -1 && !unified) {
        reason = "auto: not unified memory (a device-local type is not host-visible)";
      } else if (REXCVAR_GET(vulkan_upload_hoist) <= 0) {
        reason = "vulkan_upload_hoist is off (no access tracking)";
      } else {
        uint32_t chosen = UINT32_MAX;
        const uint32_t pref[3] = {supported & mt.host_coherent & ~mt.host_cached,
                                  supported & mt.host_coherent, supported};
        for (uint32_t m : pref) {
          if (m && rex::bit_scan_forward(m, &chosen)) {
            break;
          }
        }
        if (chosen != UINT32_MAX) {
          buffer_memory_type_ = chosen;
          direct_type_chosen = true;
          direct_coherent_ = (mt.host_coherent >> chosen) & 1;
          reason = mode == 2 ? "forced" : "auto";
        }
      }
      direct_mode_ = direct_type_chosen ? 2 : (mode == 1 ? 1 : 0);
      REXGPU_INFO("[s94][DIRECT] init: mode {}, memory type {} ({}), coherent {}, sparse N - {}", mode,
                  buffer_memory_type_, direct_type_chosen ? "host-visible" : "device-local default",
                  direct_coherent_ ? "Y" : "N", reason);
    }
    VkMemoryAllocateInfo buffer_memory_allocate_info;
    VkMemoryAllocateInfo* buffer_memory_allocate_info_last = &buffer_memory_allocate_info;
    buffer_memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    buffer_memory_allocate_info.pNext = nullptr;
    buffer_memory_allocate_info.allocationSize = buffer_memory_requirements.size;
    buffer_memory_allocate_info.memoryTypeIndex = buffer_memory_type_;
    VkMemoryDedicatedAllocateInfo buffer_memory_dedicated_allocate_info;
    if (vulkan_device->extensions().ext_1_1_KHR_dedicated_allocation) {
      buffer_memory_allocate_info_last->pNext = &buffer_memory_dedicated_allocate_info;
      buffer_memory_allocate_info_last =
          reinterpret_cast<VkMemoryAllocateInfo*>(&buffer_memory_dedicated_allocate_info);
      buffer_memory_dedicated_allocate_info.sType =
          VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
      buffer_memory_dedicated_allocate_info.pNext = nullptr;
      buffer_memory_dedicated_allocate_info.image = VK_NULL_HANDLE;
      buffer_memory_dedicated_allocate_info.buffer = buffer_;
    }
    VkDeviceMemory buffer_memory;
    VkResult buffer_memory_result =
        dfn.vkAllocateMemory(device, &buffer_memory_allocate_info, nullptr, &buffer_memory);
    if (buffer_memory_result != VK_SUCCESS && direct_type_chosen) {
      // wwe2k14 patch (Session 94, DIRECT): the host-visible heap is too small - the default type, no mapping.
      REXGPU_INFO(
          "[s94][DIRECT] init: allocating {} MB in type {} failed - falling back to type {}, direct off",
          kBufferSize >> 20, buffer_memory_type_, default_memory_type);
      buffer_memory_type_ = default_memory_type;
      direct_type_chosen = false;
      direct_mode_ = 0;
      buffer_memory_allocate_info.memoryTypeIndex = buffer_memory_type_;
      buffer_memory_result =
          dfn.vkAllocateMemory(device, &buffer_memory_allocate_info, nullptr, &buffer_memory);
    }
    if (buffer_memory_result != VK_SUCCESS) {
      REXGPU_ERROR(
          "Shared memory: Failed to allocate {} MB of memory for the Vulkan "
          "buffer",
          kBufferSize >> 20);
      return plain_failed();
    }
    buffer_memory_.push_back(buffer_memory);
    if (dfn.vkBindBufferMemory(device, buffer_, buffer_memory, 0) != VK_SUCCESS) {
      REXGPU_ERROR("Shared memory: Failed to bind memory to the Vulkan buffer");
      Shutdown();
      return false;
    }
    if (direct_type_chosen) {
      void* mapping = nullptr;
      if (dfn.vkMapMemory(device, buffer_memory, 0, VK_WHOLE_SIZE, 0, &mapping) == VK_SUCCESS &&
          mapping) {
        direct_mapping_ = static_cast<uint8_t*>(mapping);
        REXGPU_INFO("[s94][DIRECT] init: mapped {} MB, coherent {}", kBufferSize >> 20,
                    direct_coherent_ ? "Y" : "N");
      } else {
        direct_mode_ = 0;
        REXGPU_INFO("[s94][DIRECT] init: vkMapMemory failed - reallocating in type {}, direct off",
                    default_memory_type);
        // Free and bind again in the default device-local type.
        dfn.vkDestroyBuffer(device, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        dfn.vkFreeMemory(device, buffer_memory, nullptr);
        buffer_memory_.clear();
        if (dfn.vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer_) != VK_SUCCESS) {
          Shutdown();
          return false;
        }
        buffer_memory_type_ = default_memory_type;
        buffer_memory_allocate_info.memoryTypeIndex = buffer_memory_type_;
        buffer_memory_dedicated_allocate_info.buffer = buffer_;
        VkDeviceMemory memory2;
        if (dfn.vkAllocateMemory(device, &buffer_memory_allocate_info, nullptr, &memory2) != VK_SUCCESS) {
          Shutdown();
          return false;
        }
        buffer_memory_.push_back(memory2);
        if (dfn.vkBindBufferMemory(device, buffer_, memory2, 0) != VK_SUCCESS) {
          Shutdown();
          return false;
        }
      }
    }
  } else {
    const int32_t mode = REXCVAR_GET(vulkan_shared_memory_direct);
    if (mode == 1) {
      direct_mode_ = 1;  // sparse buffer: count only
      REXGPU_INFO("[s94][DIRECT] init: mode 1, sparse Y - count only");
    } else if (mode != 0) {
      REXGPU_INFO("[s94][DIRECT] init: mode {}, sparse Y - direct off (the buffer is sparse)", mode);
    }
  }
  if (direct_mode_ > 0) {
    direct_stamp_.assign(size_t(kBufferSize >> page_size_log2()), 0);
    direct_all_stamp_ = 0;
  }

  // The first usage will likely be uploading.
  last_usage_ = Usage::kTransferDestination;
  last_written_range_ = std::make_pair<uint32_t, uint32_t>(0, 0);

  // wwe2k14 patch (Session 7b, REBAR): named for the [s7][REBAR] log; vulkan_upload_pool_device_local latched
  // here (restart required).
  upload_buffer_pool_ = std::make_unique<ui::vulkan::VulkanUploadBufferPool>(
      vulkan_device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      rex::align(ui::vulkan::VulkanUploadBufferPool::kDefaultPageSize, size_t(1)
                                                                           << page_size_log2()),
      "shared memory upload", REXCVAR_GET(vulkan_upload_pool_device_local),
      VkDeviceSize(std::max(REXCVAR_GET(vulkan_device_local_upload_min_heap_mb), int32_t(0))) << 20);

  return true;
}

void VulkanSharedMemory::Shutdown(bool from_destructor) {
  upload_buffer_pool_.reset();
  ShutdownUploadSkipVerify();  // wwe2k14 patch (Session 7b, UPSKIP)

  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, buffer_);
  direct_mapping_ = nullptr;  // wwe2k14 patch (Session 94, DIRECT): vkFreeMemory unmaps
  direct_mode_ = 0;
  for (VkDeviceMemory memory : buffer_memory_) {
    dfn.vkFreeMemory(device, memory, nullptr);
  }
  buffer_memory_.clear();

  // If calling from the destructor, the SharedMemory destructor will call
  // ShutdownCommon.
  if (!from_destructor) {
    ShutdownCommon();
  }
}

void VulkanSharedMemory::CompletedSubmissionUpdated() {
  upload_buffer_pool_->Reclaim(command_processor_.GetCompletedSubmission());
  // wwe2k14 patch (Session 7b, UPSKIP): returns right away with no readback pending.
  CheckUploadSkipVerifyReadbacks();
}

void VulkanSharedMemory::FlushMergedHoistCopies() {
  // wwe2k14 patch (Session 95, HOISTMERGE)
  if (g_merge_regions.empty()) {
    return;
  }
  command_processor_.BeginHoistedUpload().CmdVkCopyBuffer(g_merge_src, buffer_, uint32_t(g_merge_regions.size()),
                                                          g_merge_regions.data());
  ++g_merge_calls_out;
  g_merge_regions_total += g_merge_regions.size();
  g_merge_regions.clear();
}

void VulkanSharedMemory::EndSubmission() {
  FlushMergedHoistCopies();  // wwe2k14 patch (Session 95, HOISTMERGE): before the hoist stream is closed
  {
    const uint64_t merge_frame = command_processor_.GetCurrentFrame();
    if (g_merge_calls_in && merge_frame - g_merge_frames_logged >= 600) {
      REXGPU_INFO("[s95][HOISTMERGE] frames {}-{}: {} hoisted copy commands became {} ({:.1f} regions each)",
                  g_merge_frames_logged, merge_frame, g_merge_calls_in, g_merge_calls_out,
                  g_merge_calls_out ? double(g_merge_regions_total) / double(g_merge_calls_out) : 0.0);
      g_merge_frames_logged = merge_frame;
      g_merge_calls_in = g_merge_calls_out = g_merge_regions_total = 0;
    }
  }
  upload_buffer_pool_->FlushWrites();
  DirectFlushRanges();  // wwe2k14 patch (Session 94, DIRECT)
  // wwe2k14 patch (Session 6, CP2): gpu_request_ranges_fast counters (returns
  // right away if nothing was counted).
  ReportRequestRangesFastStats(command_processor_.GetCurrentFrame());
  ReportRamStats(command_processor_.GetCurrentFrame());  // wwe2k14 patch (Session 49, RAM TIES)
  // wwe2k14 patch (Session 6, RESID): same for gpu_upload_streaming_copy.
  ReportUploadCopyStats(command_processor_.GetCurrentFrame());
  // wwe2k14 patch (Session 7, PROTECT): page protection counters, about every 300 frames (a comparison per
  // submission otherwise).
  ProtectUploadStats& protect_stats = protect_upload_stats;
  uint64_t frame = command_processor_.GetCurrentFrame();
  if (protect_stats.window_first_frame == UINT64_MAX) {
    // Start the first window here, with the counters since startup dropped on both sides.
    rex::memory::ResetPhysicalProtectStats();
    rex::runtime::ResetWatchFaultStats();  // wwe2k14 patch (Session 49)
    protect_stats = ProtectUploadStats();
    protect_stats.window_first_frame = frame;
  } else if (frame - protect_stats.window_first_frame >= 300) {
    rex::memory::LogPhysicalProtectStats(protect_stats.window_first_frame, frame);
    rex::runtime::LogWatchFaultStats(protect_stats.window_first_frame, frame);  // wwe2k14 patch (Session 49)
    REXGPU_INFO(
        "[s7][PROTECT] frames {}-{} uploads: {} UploadRanges calls ({} with several ranges), {} ranges, "
        "{} chunks (MakeRangeValid calls), {} pages",
        protect_stats.window_first_frame, frame, protect_stats.calls, protect_stats.multi_range,
        protect_stats.ranges, protect_stats.chunks, protect_stats.pages);
    protect_stats = ProtectUploadStats();
    protect_stats.window_first_frame = frame;
  }
}

void VulkanSharedMemory::CopyUploadChunkStreaming(uint8_t* dest, uint32_t guest_start, size_t size) {
  // wwe2k14 patch (Session 6, RESID): gpu_upload_streaming_copy - the memcpy of UploadRanges, with
  // StreamingCopy for chunks of at least kUploadStreamingCopyMinSize bytes. Same source, destination and
  // size, so the same bytes end up in the upload buffer.
  const uint8_t* source = memory().TranslatePhysical(guest_start);
  UploadCopyStats& stats = upload_copy_stats_;
  if (size < kUploadStreamingCopyMinSize) {
    std::memcpy(dest, source, size);
    ++stats.memcpy_chunks;
    stats.memcpy_bytes += size;
    return;
  }
  StreamingCopy(dest, source, size);
  ++stats.streamed_chunks;
  stats.streamed_bytes += size;
  if (!REXCVAR_GET(gpu_upload_streaming_copy_verify)) {
    return;
  }
  // wwe2k14 patch (Session 7b, REBAR): the check reads the upload buffer back - never from device-local memory
  // (vulkan_upload_pool_device_local), where every read is an uncached round trip across PCIe. Skipped and
  // counted there instead.
  if (upload_buffer_pool_->may_have_device_local_pages()) {
    ++upload_verify_skipped_;
    if (!upload_verify_skip_logged_) {
      upload_verify_skip_logged_ = true;
      REXGPU_WARN(
          "[s7][REBAR] gpu_upload_streaming_copy_verify is skipped: the shared memory upload buffers may be in "
          "device-local memory (vulkan_upload_pool_device_local), which the CPU must not read back");
    }
    return;
  }
  if (direct_mapping_ && dest >= direct_mapping_ && dest < direct_mapping_ + kBufferSize) {
    return;  // wwe2k14 patch (Session 94, DIRECT): write-combined memory, never read back
  }
  // After the sfence in StreamingCopy, so the loads see the streamed bytes.
  ++stats.verify_checks;
  if (!std::memcmp(dest, source, size)) {
    return;
  }
  // MakeRangeValid protected these pages before the copy, so a guest write to them since then trapped and
  // cleared their valid bits before it was done. Pages still valid were not written through the watched
  // mappings - the copy itself differs (or something wrote guest memory without triggering the watch).
  bool raced = !IsRangeUpToDate(guest_start, uint32_t(size));
  ++stats.verify_mismatches;
  ++stats.verify_mismatches_total;
  if (raced) {
    ++stats.verify_mismatches_raced;
  }
  if (stats.verify_mismatches_logged < 16) {
    ++stats.verify_mismatches_logged;
    size_t first_difference = 0;
    while (first_difference < size && dest[first_difference] == source[first_difference]) {
      ++first_difference;
    }
    REXGPU_WARN(
        "[s6][RESID] upload streaming copy verify mismatch {}: guest {:08X}+{:X} (upload buffer {:p}), "
        "first difference at +{:X}{}",
        stats.verify_mismatches_total, guest_start, size, static_cast<const void*>(dest),
        first_difference,
        raced ? " - the pages were invalidated by a guest write meanwhile (benign, uploaded again later)"
              : " - NOT explained by a guest write invalidation (copy error, or an unwatched writer)");
  }
}

void VulkanSharedMemory::ReportUploadCopyStats(uint64_t frame) {
  // wwe2k14 patch (Session 6, RESID): about every 300 frames while gpu_upload_streaming_copy is used.
  UploadCopyStats& stats = upload_copy_stats_;
  if (!stats.memcpy_chunks && !stats.streamed_chunks) {
    return;
  }
  if (stats.window_first_frame == UINT64_MAX) {
    stats.window_first_frame = frame;
    return;
  }
  if (frame - stats.window_first_frame < 300) {
    return;
  }
  REXGPU_INFO(
      "[s6][RESID] upload copy frames {}-{}: streamed {} chunks ({} KB), memcpy {} chunks under {} KB "
      "({} KB); verify: {} checked, {} mismatches ({} raced with a guest write), {} mismatches in total",
      stats.window_first_frame, frame, stats.streamed_chunks, stats.streamed_bytes >> 10,
      stats.memcpy_chunks, kUploadStreamingCopyMinSize >> 10, stats.memcpy_bytes >> 10,
      stats.verify_checks, stats.verify_mismatches, stats.verify_mismatches_raced,
      stats.verify_mismatches_total);
  UploadCopyStats next_stats;
  next_stats.window_first_frame = frame;
  next_stats.verify_mismatches_total = stats.verify_mismatches_total;
  next_stats.verify_mismatches_logged = stats.verify_mismatches_logged;
  stats = next_stats;
}

void VulkanSharedMemory::Use(Usage usage, std::pair<uint32_t, uint32_t> written_range,
                             VkPipelineStageFlags access_stages) {
  written_range.first = std::min(written_range.first, kBufferSize);
  written_range.second = std::min(written_range.second, kBufferSize - written_range.first);
  assert_true(usage != Usage::kRead || !written_range.second);
  // wwe2k14 patch (Session 59, diagnostic vulkan_debug_skip_draw_read_barrier - racy, measurement only): a read after a
  // compute write records no barrier (the ceiling of range-aware read barriers before draws).
  if (usage == Usage::kRead && last_usage_ == Usage::kComputeWrite &&
      REXCVAR_GET(vulkan_debug_skip_draw_read_barrier)) [[unlikely]] {
    return;
  }
  // wwe2k14 patch (Session 7b, UPSKIP): a shader write to the buffer (resolve, memory export) follows - the pages
  // it may write lose their upload hashes (an empty range: every page). RangeWrittenByGpu drops them too.
  if (usage == Usage::kComputeWrite || usage == Usage::kGuestDrawReadWrite) {
    InvalidateUploadHashes(written_range.second ? written_range.first : 0,
                           written_range.second ? written_range.second : kBufferSize);
    // wwe2k14 patch (Session 21, HOIST): the pages it may write are accessed in this submission.
    MarkHoistAccess(written_range.second ? written_range.first : 0,
                    written_range.second ? written_range.second : kBufferSize);
  }
  if (last_usage_ != usage || last_written_range_.second) {
    VkPipelineStageFlags src_stage_mask, dst_stage_mask;
    VkAccessFlags src_access_mask, dst_access_mask;
    GetUsageMasks(last_usage_, src_stage_mask, src_access_mask);
    GetUsageMasks(usage, dst_stage_mask, dst_access_mask);
    // wwe2k14 patch (Session 7b, BARNARROW): the barrier before an upload copy - narrowed with
    // vulkan_shared_memory_narrow_barrier, counted either way.
    if (usage == Usage::kTransferDestination) {
      NarrowUploadBarrierSource(src_stage_mask, src_access_mask);
    }
    // wwe2k14 patch (Session 7b, BARNARROW): every access tracked so far is in this barrier's source scope
    // (directly, or through the stage kept from the previous barrier's destination scope), so it happens before
    // the work after it in the destination stages. Start over with one stage of that destination scope: a
    // later barrier with it in its source scope chains to this one. Prefer the transfer stage (always in an
    // upload barrier's source scope), then vertex input (every draw has it); kComputeWrite has only compute.
    VkPipelineStageFlags chain_stage_mask = dst_stage_mask & VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (!chain_stage_mask) {
      chain_stage_mask = dst_stage_mask & VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
    }
    if (!chain_stage_mask) {
      chain_stage_mask = dst_stage_mask;
    }
    stages_accessed_since_barrier_ = chain_stage_mask;
    VkDeviceSize offset, size;
    if (last_usage_ == usage) {
      // Committing the previous write, while not changing the access mask
      // (passing false as whether to skip the barrier if no masks are changed
      // for this reason).
      offset = VkDeviceSize(last_written_range_.first);
      size = VkDeviceSize(last_written_range_.second);
    } else {
      // Changing the stage and access mask - all preceding writes must be
      // available not only to the source stage, but to the destination as well.
      offset = 0;
      size = VK_WHOLE_SIZE;
      last_usage_ = usage;
    }
    command_processor_.PushBufferMemoryBarrier(
        buffer_, offset, size, src_stage_mask, dst_stage_mask, src_access_mask, dst_access_mask,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
  }
  last_written_range_ = written_range;
  // wwe2k14 patch (Session 7b, BARNARROW): the stages of the work recorded after this call. Within the usage's
  // stages (the source scope of the next barrier without the setting); 0 or anything outside them = all of
  // them.
  VkPipelineStageFlags usage_stage_mask;
  VkAccessFlags usage_access_mask;
  GetUsageMasks(usage, usage_stage_mask, usage_access_mask);
  access_stages &= usage_stage_mask;
  stages_accessed_since_barrier_ |= access_stages ? access_stages : usage_stage_mask;
}

// wwe2k14 patch (Session 79): see the header.
void VulkanSharedMemory::UseAlsoComputeWrite(std::pair<uint32_t, uint32_t> written_range) {
  if (last_usage_ != Usage::kComputeWrite || !last_written_range_.second) {
    Use(Usage::kComputeWrite, written_range);
    return;
  }
  written_range.first = std::min(written_range.first, kBufferSize);
  written_range.second = std::min(written_range.second, kBufferSize - written_range.first);
  if (!written_range.second) return;
  InvalidateUploadHashes(written_range.first, written_range.second);  // (as Use)
  MarkHoistAccess(written_range.first, written_range.second);
  const uint64_t begin = std::min(uint64_t(last_written_range_.first), uint64_t(written_range.first));
  const uint64_t end = std::max(uint64_t(last_written_range_.first) + last_written_range_.second,
                                uint64_t(written_range.first) + written_range.second);
  last_written_range_ = {uint32_t(begin), uint32_t(end - begin)};
  VkPipelineStageFlags usage_stage_mask;
  VkAccessFlags usage_access_mask;
  GetUsageMasks(Usage::kComputeWrite, usage_stage_mask, usage_access_mask);
  stages_accessed_since_barrier_ |= usage_stage_mask;
}

void VulkanSharedMemory::NarrowUploadBarrierSource(VkPipelineStageFlags& src_stage_mask,
                                                   VkAccessFlags& src_access_mask) {
  // wwe2k14 patch (Session 7b, BARNARROW): src_stage_mask / src_access_mask are GetUsageMasks(last_usage_) -
  // every stage and access the buffer may have had since the last barrier on it. Every GPU access of buffer_
  // is recorded after a Use naming its stages, with no barrier on buffer_ in between (draws, texture load
  // dispatches, resolve dispatches, copies; the FSI EDRAM is a different buffer), so:
  // - the accesses since the last barrier are in stages_accessed_since_barrier_, so in this source scope;
  // - every access before the last barrier is in that barrier's source scope (the full mask, or this same
  //   argument), and this source scope intersects that barrier's destination scope - in the chain stage kept
  //   in stages_accessed_since_barrier_, and in the stages of the access that barrier was pushed for (recorded
  //   between the two barriers) - which forms an execution dependency chain.
  // So every earlier access of the buffer happens before the copy (write-after-read and write-after-write),
  // as with the full mask. Read bits in the source access scope have no effect (availability operations only
  // apply to writes); the ones without a stage left are dropped (they would be invalid). Write bits stay as
  // they are, or nothing is narrowed.
  NarrowBarrierStats& stats = narrow_barrier_frame_stats_;
  ++stats.upload_barriers;
  narrow_barrier_batch_check_ = false;
  if (last_usage_ == Usage::kRead || last_usage_ == Usage::kGuestDrawReadWrite) {
    ++stats.from_read;
    // Never wider than the full mask. Transfer is kept where the full mask has it (after kRead); after
    // kGuestDrawReadWrite only draws accessed the buffer, and the chain stage is vertex input.
    VkPipelineStageFlags narrow_stage_mask =
        (stages_accessed_since_barrier_ | VK_PIPELINE_STAGE_TRANSFER_BIT) & src_stage_mask;
    if (narrow_stage_mask & VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT) {
      ++stats.tracked_fragment;
    }
    if (narrow_stage_mask & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
      ++stats.tracked_compute;
    }
    constexpr VkPipelineStageFlags kShaderStages =
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
        VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT |
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkAccessFlags narrow_access_mask = 0;
    if (narrow_stage_mask & VK_PIPELINE_STAGE_VERTEX_INPUT_BIT) {
      narrow_access_mask |= VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    }
    if (narrow_stage_mask & kShaderStages) {
      narrow_access_mask |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    }
    if (narrow_stage_mask & VK_PIPELINE_STAGE_TRANSFER_BIT) {
      narrow_access_mask |= VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    }
    narrow_access_mask &= src_access_mask;
    constexpr VkAccessFlags kReadAccesses = VK_ACCESS_INDEX_READ_BIT |
                                            VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    if (!narrow_stage_mask || ((src_access_mask & ~narrow_access_mask) & ~kReadAccesses)) {
      // Not expected (a write is always tracked with its stage) - keep the full masks.
      ++stats.not_narrowable;
    } else {
      if (!(narrow_stage_mask & VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)) {
        ++stats.would_narrow;
      }
      if (REXCVAR_GET(vulkan_shared_memory_narrow_barrier)) {
        src_stage_mask = narrow_stage_mask;
        src_access_mask = narrow_access_mask;
        narrow_barrier_batch_check_ = !(narrow_stage_mask & VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
      }
    }
  }
  if (!(src_stage_mask & VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)) {
    ++stats.narrowed;
    if (!(src_stage_mask & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)) {
      ++stats.narrowed_no_compute;
    }
  }
}

void VulkanSharedMemory::CountNarrowBarrierBatch() {
  // wwe2k14 patch (Session 7b, BARNARROW): measurement only. Called by UploadRanges between Use and
  // SubmitBarriers: if another pending barrier (an EDRAM buffer or image barrier) waits for the fragment
  // shader, the pipeline still waits for the pixel work at this point despite the narrowed shared memory
  // barrier.
  if (!narrow_barrier_batch_check_) {
    return;
  }
  narrow_barrier_batch_check_ = false;
  if (command_processor_.GetPendingBarriersSrcStageMask() & VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT) {
    ++narrow_barrier_frame_stats_.batched_fragment;
  }
}

void VulkanSharedMemory::RecordNarrowBarrierFrameCounters(uint64_t frame) {
  // wwe2k14 patch (Session 7b, BARNARROW): cpf_bar a = upload barriers issued, b = of those, without the
  // fragment stage in the source scope, c = of those, also without the compute stage. Then the log window.
  NarrowBarrierStats& frame_stats = narrow_barrier_frame_stats_;
  if (perf::timeline::Enabled()) {
    perf::timeline::Record("cpf_bar", frame_stats.upload_barriers, frame_stats.narrowed,
                           frame_stats.narrowed_no_compute);
  }
  NarrowBarrierStats& window = narrow_barrier_window_stats_;
  window.upload_barriers += frame_stats.upload_barriers;
  window.narrowed += frame_stats.narrowed;
  window.narrowed_no_compute += frame_stats.narrowed_no_compute;
  window.from_read += frame_stats.from_read;
  window.tracked_fragment += frame_stats.tracked_fragment;
  window.tracked_compute += frame_stats.tracked_compute;
  window.would_narrow += frame_stats.would_narrow;
  window.not_narrowable += frame_stats.not_narrowable;
  window.batched_fragment += frame_stats.batched_fragment;
  frame_stats = NarrowBarrierStats();
  if (narrow_barrier_window_first_frame_ == UINT64_MAX) {
    narrow_barrier_window_first_frame_ = frame;
    window = NarrowBarrierStats();
    return;
  }
  if (frame - narrow_barrier_window_first_frame_ < 300) {
    return;
  }
  if (window.upload_barriers) {
    REXGPU_INFO(
        "[s7][BARNARROW] frames {}-{} (vulkan_shared_memory_narrow_barrier {}): {} upload barriers, {} without "
        "the fragment stage ({} also without compute); {} after a read usage: {} with a pixel shader access "
        "tracked, {} with a compute access tracked, {} narrowable without fragment, {} not narrowable; {} "
        "narrowed but submitted with another pending barrier from the fragment stage",
        narrow_barrier_window_first_frame_, frame,
        REXCVAR_GET(vulkan_shared_memory_narrow_barrier) ? "on" : "off", window.upload_barriers,
        window.narrowed, window.narrowed_no_compute, window.from_read, window.tracked_fragment,
        window.tracked_compute, window.would_narrow, window.not_narrowable, window.batched_fragment);
  }
  window = NarrowBarrierStats();
  narrow_barrier_window_first_frame_ = frame;
}

bool VulkanSharedMemory::AllocateSparseHostGpuMemoryRange(uint32_t offset_allocations,
                                                          uint32_t length_allocations) {
  if (!length_allocations) {
    return true;
  }

  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  VkMemoryAllocateInfo memory_allocate_info;
  memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memory_allocate_info.pNext = nullptr;
  memory_allocate_info.allocationSize = length_allocations
                                        << host_gpu_memory_sparse_granularity_log2();
  memory_allocate_info.memoryTypeIndex = buffer_memory_type_;
  VkDeviceMemory memory;
  if (dfn.vkAllocateMemory(device, &memory_allocate_info, nullptr, &memory) != VK_SUCCESS) {
    REXGPU_ERROR("Shared memory: Failed to allocate sparse buffer memory");
    return false;
  }
  buffer_memory_.push_back(memory);

  VkSparseMemoryBind bind;
  bind.resourceOffset = offset_allocations << host_gpu_memory_sparse_granularity_log2();
  bind.size = memory_allocate_info.allocationSize;
  bind.memory = memory;
  bind.memoryOffset = 0;
  bind.flags = 0;
  VkPipelineStageFlags bind_wait_stage_mask =
      VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
      VK_PIPELINE_STAGE_TRANSFER_BIT;
  if (vulkan_device->properties().tessellationShader) {
    bind_wait_stage_mask |= VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT;
  }
  command_processor_.SparseBindBuffer(buffer_, 1, &bind, bind_wait_stage_mask);

  return true;
}

bool VulkanSharedMemory::UploadRanges(
    const std::vector<std::pair<uint32_t, uint32_t>>& all_upload_page_ranges) {
  if (all_upload_page_ranges.empty()) {
    return true;
  }
  // wwe2k14 patch (Session 94, DIRECT): split into runs written straight into the mapping and the rest (staged
  // exactly as before). Mode 1 only counts; the rest of the function then sees every range.
  const std::vector<std::pair<uint32_t, uint32_t>>* staged_source = &all_upload_page_ranges;
  bool direct_active = false;
  if (direct_mode_ > 0) {
    DirectClassify(all_upload_page_ranges, direct_runs_, direct_staged_);
    ++direct_frame_stats_.calls;
    direct_active = direct_mode_ == 2 && direct_mapping_ && REXCVAR_GET(vulkan_upload_hoist) > 0 &&
                    !REXCVAR_GET(gpu_upload_skip_unchanged_verify);
    if (direct_active) {
      staged_source = &direct_staged_;
    } else {
      direct_runs_.clear();
    }
  }
  const std::vector<std::pair<uint32_t, uint32_t>>& upload_page_ranges = *staged_source;
  if (direct_active && !direct_runs_.empty()) {
    const bool streaming_copy_direct = REXCVAR_GET(gpu_upload_streaming_copy);
    for (const std::pair<uint32_t, uint32_t>& run : direct_runs_) {
      const uint32_t run_bytes = run.second << page_size_log2();
      // Same order as the staged copy: valid first (a guest write during the copy invalidates the page).
      MakeRangeValid(run.first << page_size_log2(), run_bytes, false);
      ++protect_upload_stats.chunks;
      protect_upload_stats.pages += run.second;
      uint8_t* dest = direct_mapping_ + (size_t(run.first) << page_size_log2());
      if (upload_hash_tracking()) {
        CopyAndHashUploadPages(dest, run.first, run_bytes);
      } else if (streaming_copy_direct) {
        CopyUploadChunkStreaming(dest, run.first << page_size_log2(), size_t(run_bytes));
      } else {
        std::memcpy(dest, memory().TranslatePhysical(run.first << page_size_log2()), run_bytes);
      }
      // These pages are written in this submission now (a later upload of one stays staged, after this write).
      HoistMarkPages(run.first, run.second);
      NoteUploadedPages(run.first, run.second);  // (Session 95)
      if (!direct_coherent_) {
        direct_flush_pages_.push_back(run);
      }
    }
    ResetUploadHashNextPage();  // staged pages below the last direct page keep their hashes
    if (upload_page_ranges.empty()) {
      ++protect_upload_stats.calls;
      protect_upload_stats.ranges += direct_runs_.size();
      ++direct_frame_stats_.calls_no_barrier;
      return true;
    }
  }
  if (upload_page_ranges.empty()) {
    return true;
  }
  // wwe2k14 patch (Session 7c, CAUSES): measurement only - this call records its barrier and copies from here on.
  {
    VulkanCommandProcessor::CauseCounts& causes = command_processor_.cause_counts();
    ++causes.up_calls;
    for (const std::pair<uint32_t, uint32_t>& upload_range : upload_page_ranges) {
      causes.up_pages += upload_range.second;
    }
  }
  // wwe2k14 patch (Session 21, HOIST): vulkan_upload_hoist - copy in the hoist stream when no GPU command
  // recorded earlier in this submission accessed any of these pages (see OnRangesRequested).
  bool hoist = false;
  const int32_t hoist_mode = REXCVAR_GET(vulkan_upload_hoist);
  if (hoist_mode > 0) {
    HoistStats& hs = hoist_frame_stats_;
    uint32_t pages = 0;
    for (const std::pair<uint32_t, uint32_t>& r : upload_page_ranges) {
      pages += r.second;
    }
    ++hs.calls;
    hs.pages += pages;
    if (HoistPagesUntouched(upload_page_ranges)) {
      ++hs.hoistable_calls;
      hs.hoistable_pages += pages;
      // Not with the upload skip verify: its readbacks copy pages at their place in the main stream.
      if (hoist_mode >= 2 && !REXCVAR_GET(gpu_upload_skip_unchanged_verify)) {
        hoist = true;
        ++hs.hoisted_calls;
        hs.hoisted_pages += pages;
      }
    } else if (hoist_accessed_all_) {
      ++hs.blocked_all;
    }
  }
  if (!hoist) {
    // upload_page_ranges are sorted, use them to determine the range for the
    // ordering barrier.
    Use(Usage::kTransferDestination,
        std::make_pair(upload_page_ranges.front().first << page_size_log2(),
                       (upload_page_ranges.back().first + upload_page_ranges.back().second -
                        upload_page_ranges.front().first)
                           << page_size_log2()));
    CountNarrowBarrierBatch();  // wwe2k14 patch (Session 7b, BARNARROW): measurement only
    command_processor_.SubmitBarriers(true);
  }
  // In place, or the hoist stream (its opening barrier recorded at its first copy in the submission).
  DeferredCommandBuffer& command_buffer =
      hoist ? command_processor_.BeginHoistedUpload() : command_processor_.deferred_command_buffer();
  // wwe2k14 patch (Session 21, HOIST): these pages are written in this submission now (a later upload of one
  // of them must stay after this copy).
  if (hoist_mode > 0) {
    for (const std::pair<uint32_t, uint32_t>& r : upload_page_ranges) {
      HoistMarkPages(r.first, r.second);
    }
  }
  for (const std::pair<uint32_t, uint32_t>& r : upload_page_ranges) {
    NoteUploadedPages(r.first, r.second);  // (Session 95)
  }
  uint64_t submission_current = command_processor_.GetCurrentSubmission();
  bool successful = true;
  upload_regions_.clear();
  // wwe2k14 patch (Session 6, RESID): read once per call; nothing carries over between calls.
  const bool streaming_copy = REXCVAR_GET(gpu_upload_streaming_copy);
  // wwe2k14 patch (Session 7b, REBAR): host time of the copies into the upload buffer (timeline and
  // vulkan_upload_copy_timing on only - opt-in, as it adds two clock reads per chunk).
  const bool time_copies = perf::timeline::Enabled() && REXCVAR_GET(vulkan_upload_copy_timing);
  // wwe2k14 patch (Session 7, PROTECT): counters.
  ++protect_upload_stats.calls;
  protect_upload_stats.ranges += upload_page_ranges.size();
  if (upload_page_ranges.size() > 1) {
    ++protect_upload_stats.multi_range;
  }
  VkBuffer upload_buffer_previous = VK_NULL_HANDLE;
  // wwe2k14 patch (Session 95, HOISTMERGE): hoisted copies are collected across calls (flushed at EndSubmission).
  const bool merge_copies = hoist && REXCVAR_GET(vulkan_upload_hoist_merge);
  auto emit_copy = [&](VkBuffer src, const VkBufferCopy* regions, uint32_t count) {
    if (!merge_copies) {
      command_buffer.CmdVkCopyBuffer(src, buffer_, count, regions);
      return;
    }
    ++g_merge_calls_in;
    if (g_merge_src != src) {
      FlushMergedHoistCopies();
      g_merge_src = src;
    }
    g_merge_regions.insert(g_merge_regions.end(), regions, regions + count);
  };
  for (auto upload_range : upload_page_ranges) {
    uint32_t upload_range_start = upload_range.first;
    uint32_t upload_range_length = upload_range.second;
    while (upload_range_length) {
      VkBuffer upload_buffer;
      VkDeviceSize upload_buffer_offset, upload_buffer_size;
      uint8_t* upload_buffer_mapping = upload_buffer_pool_->RequestPartial(
          submission_current, upload_range_length << page_size_log2(),
          size_t(1) << page_size_log2(), upload_buffer, upload_buffer_offset, upload_buffer_size);
      if (upload_buffer_mapping == nullptr) {
        REXGPU_ERROR("Shared memory: Failed to get a Vulkan upload buffer");
        successful = false;
        break;
      }
      const uint64_t stat_valid_t = g_tex_stat_on ? TexStatTicks() : 0;  // (Session 95, r6tex: log only)
      MakeRangeValid(upload_range_start << page_size_log2(), uint32_t(upload_buffer_size), false);
      if (g_tex_stat_on) {
        g_tex_stat[kTsUpValidTicks] += TexStatTicks() - stat_valid_t;
        g_tex_stat[kTsUpPages] += upload_buffer_size >> page_size_log2();
      }
      const uint64_t stat_copy_t = g_tex_stat_on ? TexStatTicks() : 0;
      // wwe2k14 patch (Session 7, PROTECT): counters.
      ++protect_upload_stats.chunks;
      protect_upload_stats.pages += upload_buffer_size >> page_size_log2();
      const uint64_t copy_start = time_copies ? rex::chrono::Clock::QueryHostTickCount() : 0;  // REBAR
      if (upload_hash_tracking()) {
        // wwe2k14 patch (Session 7b, UPSKIP): the same bytes, hashed for gpu_upload_skip_unchanged(_stats).
        CopyAndHashUploadPages(upload_buffer_mapping, upload_range_start, uint32_t(upload_buffer_size));
      } else if (streaming_copy) {
        // wwe2k14 patch (Session 6, RESID)
        CopyUploadChunkStreaming(upload_buffer_mapping, upload_range_start << page_size_log2(),
                                 size_t(upload_buffer_size));
      } else {
        std::memcpy(upload_buffer_mapping,
                    memory().TranslatePhysical(upload_range_start << page_size_log2()),
                    upload_buffer_size);
      }
      if (time_copies) {  // wwe2k14 patch (Session 7b, REBAR)
        upload_copy_ticks_ += rex::chrono::Clock::QueryHostTickCount() - copy_start;
      }
      if (g_tex_stat_on) g_tex_stat[kTsUpCopyTicks] += TexStatTicks() - stat_copy_t;
      if (upload_buffer_previous != upload_buffer && !upload_regions_.empty()) {
        assert_true(upload_buffer_previous != VK_NULL_HANDLE);
        emit_copy(upload_buffer_previous, upload_regions_.data(), uint32_t(upload_regions_.size()));
        upload_regions_.clear();
      }
      upload_buffer_previous = upload_buffer;
      VkBufferCopy& upload_region = upload_regions_.emplace_back();
      upload_region.srcOffset = upload_buffer_offset;
      upload_region.dstOffset = VkDeviceSize(upload_range_start << page_size_log2());
      upload_region.size = upload_buffer_size;
      uint32_t upload_buffer_pages = uint32_t(upload_buffer_size >> page_size_log2());
      upload_range_start += upload_buffer_pages;
      upload_range_length -= upload_buffer_pages;
    }
    if (!successful) {
      break;
    }
  }
  if (!upload_regions_.empty()) {
    assert_true(upload_buffer_previous != VK_NULL_HANDLE);
    emit_copy(upload_buffer_previous, upload_regions_.data(), uint32_t(upload_regions_.size()));
    upload_regions_.clear();
  }
  return successful;
}

// wwe2k14 patch (Session 21, HOIST): vulkan_upload_hoist. Why a hoisted copy is safe: it runs at the start of
// its submission's command buffer, after a barrier from every stage of all earlier queue work (earlier
// submissions) to the transfer stage, and before a barrier from the transfer stage to every stage of what
// follows. Moving it earlier than its place in the stream only changes what GPU work between the two places
// sees, and only for its pages: work in the same submission recorded before the upload. Every GPU access of
// buffer_ is recorded after a request of its range (RequestRanges -> OnRangesRequested: draws' vertex fetch,
// index buffers, memexport streams, texture loads, resolve destinations) or, for vertex fetch constants the
// residency cache keeps in sync, after IssueDraw marks their last requested range (MarkHoistAccess); GPU
// writes also mark their range in Use (an unknown range = every page). An upload is hoisted only when none
// of its pages was marked earlier in the submission, so no work between the two places touches them. Its
// pages are marked afterwards, so a second upload of a page in the same submission stays in place (after the
// hoist stream's closing barrier). A submission in which any access was recorded while the setting was 0 is
// marked as all accessed (no hoisting), so toggling is safe.
// wwe2k14 patch (Session 95, the [vinput] line): one bit per page uploaded in the current submission, cleared lazily when the submission changes. A steady frame uploads nothing: the query then
// costs one compare.
void VulkanSharedMemory::NoteUploadedPages(uint32_t page_first, uint32_t page_count) {
  if (!shader_swap::VertexMirrorLatched()) return;  // (only the [vinput] line reads it: nothing at switch 0)
  const uint64_t submission = command_processor_.GetCurrentSubmission();
  if (submission != uploaded_submission_) {
    uploaded_pages_.assign(size_t(((kBufferSize >> page_size_log2()) + 63) >> 6), 0);
    uploaded_submission_ = submission;
  }
  for (uint32_t page = page_first, end = page_first + page_count; page < end; ++page) {
    if ((page >> 6) < uploaded_pages_.size()) uploaded_pages_[page >> 6] |= uint64_t(1) << (page & 63);
  }
}

bool VulkanSharedMemory::UploadedThisSubmission(uint32_t start, uint32_t length) {
  if (!length || uploaded_submission_ != command_processor_.GetCurrentSubmission()) return false;
  const uint32_t first = start >> page_size_log2(), last = uint32_t((uint64_t(start) + length - 1) >> page_size_log2());
  for (uint32_t page = first; page <= last && (page >> 6) < uploaded_pages_.size(); ++page) {
    if ((uploaded_pages_[page >> 6] >> (page & 63)) & 1) return true;
  }
  return false;
}

void VulkanSharedMemory::HoistSyncSubmission() {
  const uint64_t submission = command_processor_.GetCurrentSubmission();
  if (submission == hoist_accessed_submission_) {
    return;
  }
  DirectFoldStamps();  // wwe2k14 patch (Session 94, DIRECT): before the bitmap is cleared for the new submission
  hoist_accessed_submission_ = submission;
  const size_t words = size_t((kBufferSize >> page_size_log2()) + 63) >> 6;
  if (hoist_accessed_.size() != words) {
    hoist_accessed_.assign(words, 0);
  } else {
    std::fill(hoist_accessed_.begin(), hoist_accessed_.end(), uint64_t(0));
  }
  hoist_accessed_all_ = false;
}

void VulkanSharedMemory::HoistMarkPages(uint32_t page_first, uint32_t page_count) {
  HoistSyncSubmission();
  if (REXCVAR_GET(vulkan_upload_hoist) <= 0) {
    hoist_accessed_all_ = true;  // not tracked: nothing in this submission may be hoisted
    return;
  }
  const uint32_t page_total = kBufferSize >> page_size_log2();
  if (page_first >= page_total || !page_count) {
    return;
  }
  const uint32_t page_end = std::min(page_total, page_first + page_count);
  if (page_first == 0 && page_end == page_total) {
    hoist_accessed_all_ = true;  // the whole buffer (a memexport without known ranges, a pixel shader's vfetch)
    return;
  }
  for (uint32_t p = page_first; p < page_end;) {
    const uint32_t word = p >> 6, bit = p & 63;
    const uint32_t n = std::min(64 - bit, page_end - p);
    hoist_accessed_[word] |= (n == 64 ? ~uint64_t(0) : ((uint64_t(1) << n) - 1)) << bit;
    p += n;
  }
}

void VulkanSharedMemory::MarkHoistAccess(uint32_t start, uint32_t length) {
  if (!length) {
    return;
  }
  start = std::min(start, kBufferSize);
  length = std::min(length, kBufferSize - start);
  if (!length) {
    return;
  }
  const uint32_t page_first = start >> page_size_log2();
  const uint32_t page_last = (start + length - 1) >> page_size_log2();
  HoistMarkPages(page_first, page_last - page_first + 1);
}

void VulkanSharedMemory::OnRangesRequested(const std::pair<uint32_t, uint32_t>* ranges, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    MarkHoistAccess(ranges[i].first, ranges[i].second);
  }
}

bool VulkanSharedMemory::HoistPagesUntouched(
    const std::vector<std::pair<uint32_t, uint32_t>>& upload_page_ranges) {
  HoistSyncSubmission();
  if (hoist_accessed_all_) {
    return false;
  }
  const uint32_t page_total = kBufferSize >> page_size_log2();
  for (const std::pair<uint32_t, uint32_t>& r : upload_page_ranges) {
    const uint32_t page_end = std::min(page_total, r.first + r.second);
    for (uint32_t p = r.first; p < page_end;) {
      const uint32_t word = p >> 6, bit = p & 63;
      const uint32_t n = std::min(64 - bit, page_end - p);
      if (hoist_accessed_[word] & ((n == 64 ? ~uint64_t(0) : ((uint64_t(1) << n) - 1)) << bit)) {
        return false;
      }
      p += n;
    }
  }
  return true;
}

// wwe2k14 patch (Session 94, DIRECT): why a direct write is safe. The CPU writes straight into the buffer the GPU
// reads, so a page may only be written when no GPU work that has not finished (or that is still to be submitted)
// can read or write it, and no command recorded earlier in the submission being recorded does (the submission's
// vkQueueSubmit is a host-write -> device dependency for everything in it, so work recorded later sees the bytes).
// Every GPU access of the buffer is marked in the hoist bitmap of its submission (see the HOIST comment above); when
// a new submission starts the previous bitmap is folded into per-page stamps (the submission index that last
// touched the page; an access without a known range stamps every page). A page is written directly iff its bit is
// clear in the current bitmap, the whole-buffer flag is clear, and its stamp (and the whole-buffer stamp) is at most
// the last completed submission. Submissions recorded but not yet queued by the async submit thread have an index
// above completed, so their pages stay staged.
void VulkanSharedMemory::DirectFoldStamps() {
  if (direct_mode_ <= 0 || hoist_accessed_submission_ == UINT64_MAX) {
    return;
  }
  if (hoist_accessed_all_) {
    direct_all_stamp_ = std::max(direct_all_stamp_, hoist_accessed_submission_);
    return;
  }
  const uint32_t page_total = kBufferSize >> page_size_log2();
  for (size_t w = 0; w < hoist_accessed_.size(); ++w) {
    uint64_t bits = hoist_accessed_[w];
    while (bits) {
      uint32_t b = 0;
      rex::bit_scan_forward(bits, &b);
      bits &= bits - 1;
      const size_t page = (w << 6) + b;
      if (page < page_total) {
        direct_stamp_[page] = hoist_accessed_submission_;
      }
    }
  }
}

void VulkanSharedMemory::DirectClassify(const std::vector<std::pair<uint32_t, uint32_t>>& ranges,
                                        std::vector<std::pair<uint32_t, uint32_t>>& direct_runs,
                                        std::vector<std::pair<uint32_t, uint32_t>>& staged_runs) {
  direct_runs.clear();
  staged_runs.clear();
  HoistSyncSubmission();
  const uint64_t completed = command_processor_.GetCompletedSubmission();
  const bool all_this = hoist_accessed_all_;
  const bool all_inflight = direct_all_stamp_ > completed;
  const uint32_t page_total = kBufferSize >> page_size_log2();
  DirectStats& st = direct_frame_stats_;
  auto push = [](std::vector<std::pair<uint32_t, uint32_t>>& v, uint32_t first, uint32_t count) {
    if (!v.empty() && v.back().first + v.back().second == first) {
      v.back().second += count;
    } else {
      v.emplace_back(first, count);
    }
  };
  for (const std::pair<uint32_t, uint32_t>& r : ranges) {
    const uint32_t end = std::min(page_total, r.first + r.second);
    for (uint32_t p = r.first; p < end; ++p) {
      const bool this_sub = all_this || ((hoist_accessed_[p >> 6] >> (p & 63)) & 1);
      if (this_sub) {
        ++st.pages_this_sub;
        push(staged_runs, p, 1);
      } else if (all_inflight || direct_stamp_[p] > completed) {
        ++st.pages_inflight;
        push(staged_runs, p, 1);
      } else {
        ++st.pages_direct;
        push(direct_runs, p, 1);
      }
    }
  }
}

void VulkanSharedMemory::DirectFlushRanges() {
  if (direct_flush_pages_.empty() || direct_coherent_ || !direct_mapping_) {
    direct_flush_pages_.clear();
    return;
  }
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDeviceSize atom = std::max<VkDeviceSize>(vulkan_device->properties().nonCoherentAtomSize, 1);
  std::vector<VkMappedMemoryRange> ranges;
  ranges.reserve(direct_flush_pages_.size());
  std::sort(direct_flush_pages_.begin(), direct_flush_pages_.end());
  for (const std::pair<uint32_t, uint32_t>& r : direct_flush_pages_) {
    VkDeviceSize begin = VkDeviceSize(r.first) << page_size_log2();
    VkDeviceSize end = begin + (VkDeviceSize(r.second) << page_size_log2());
    begin = begin / atom * atom;
    end = std::min<VkDeviceSize>((end + atom - 1) / atom * atom, kBufferSize);
    if (!ranges.empty() && begin <= ranges.back().offset + ranges.back().size) {
      ranges.back().size = std::max(ranges.back().offset + ranges.back().size, end) - ranges.back().offset;
      continue;
    }
    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = buffer_memory_.front();
    range.offset = begin;
    range.size = end - begin;
    ranges.push_back(range);
  }
  direct_flush_pages_.clear();
  dfn.vkFlushMappedMemoryRanges(vulkan_device->device(), uint32_t(ranges.size()), ranges.data());
  ++direct_frame_stats_.flushes;
}

void VulkanSharedMemory::RecordDirectFrameCounters(uint64_t frame) {
  if (direct_mode_ <= 0) {
    return;
  }
  DirectStats& f = direct_frame_stats_;
  const uint64_t staged = f.pages_inflight + f.pages_this_sub;
  if (perf::timeline::Enabled() && (f.pages_direct || staged)) {
    // cpf_direct a = pages written directly, b = pages staged because in flight, c = pages staged because touched
    // this submission (or an access without a known range).
    perf::timeline::Record("cpf_direct", uint32_t(f.pages_direct), uint32_t(f.pages_inflight),
                           uint32_t(f.pages_this_sub));
  }
  DirectStats& w = direct_window_stats_;
  w.calls += f.calls;
  w.pages_direct += f.pages_direct;
  w.pages_inflight += f.pages_inflight;
  w.pages_this_sub += f.pages_this_sub;
  w.calls_no_barrier += f.calls_no_barrier;
  w.flushes += f.flushes;
  f = DirectStats();
  if (direct_window_first_frame_ == UINT64_MAX) {
    direct_window_first_frame_ = frame;
    direct_window_frames_ = 0;
  }
  ++direct_window_frames_;
  if (frame - direct_window_first_frame_ < 300) {
    return;
  }
  if (w.calls) {
    const double n = double(direct_window_frames_);
    const uint64_t page_kb = (uint64_t(1) << page_size_log2()) >> 10;
    const uint64_t total = w.pages_direct + w.pages_inflight + w.pages_this_sub;
    REXGPU_INFO(
        "[s94][DIRECT] frames {}-{} (mode {}), per frame: {:.1f} upload calls, {:.1f} pages direct ({:.0f} KB, "
        "{:.0f}% of {:.1f}), {:.1f} pages staged (in flight {:.1f}, this submission {:.1f}), {:.1f} calls with no "
        "barrier (all direct), {:.1f} flushes",
        direct_window_first_frame_, frame, direct_mode_, w.calls / n, w.pages_direct / n,
        double(w.pages_direct * page_kb) / n, total ? 100.0 * double(w.pages_direct) / double(total) : 0.0,
        total / n, (w.pages_inflight + w.pages_this_sub) / n, w.pages_inflight / n, w.pages_this_sub / n,
        w.calls_no_barrier / n, w.flushes / n);
  }
  w = DirectStats();
  direct_window_first_frame_ = frame;
  direct_window_frames_ = 0;
}

void VulkanSharedMemory::RecordHoistFrameCounters(uint64_t frame) {
  RecordDirectFrameCounters(frame);  // wwe2k14 patch (Session 94, DIRECT)
  // cpf_hoist a = upload calls, b = of those, hoistable, c = of those, hoisted (counted while the setting is on).
  HoistStats& f = hoist_frame_stats_;
  if (perf::timeline::Enabled() && f.calls) {
    perf::timeline::Record("cpf_hoist", uint32_t(f.calls), uint32_t(f.hoistable_calls),
                           uint32_t(f.hoisted_calls));
  }
  HoistStats& w = hoist_window_stats_;
  w.calls += f.calls;
  w.pages += f.pages;
  w.hoistable_calls += f.hoistable_calls;
  w.hoistable_pages += f.hoistable_pages;
  w.hoisted_calls += f.hoisted_calls;
  w.hoisted_pages += f.hoisted_pages;
  w.blocked_all += f.blocked_all;
  f = HoistStats();
  if (hoist_window_first_frame_ == UINT64_MAX) {
    hoist_window_first_frame_ = frame;
    hoist_window_frames_ = 0;
  }
  ++hoist_window_frames_;
  // Session 95 (the diagnostic run): gpu_diag_stats_now (command_processor.cpp) prints the totals so far at once.
  static int32_t diag_seen = 0;
  const int32_t diag_now = REXCVAR_GET(gpu_diag_stats_now);
  const bool diag_flush = diag_now != diag_seen;
  diag_seen = diag_now;
  if (frame - hoist_window_first_frame_ < 300 && !diag_flush) {
    return;
  }
  if (w.calls) {
    const double n = double(hoist_window_frames_);
    REXGPU_INFO(
        "[s21][HOIST] frames {}-{} (vulkan_upload_hoist {}), per frame: {:.1f} upload calls ({:.1f} pages), "
        "hoistable {:.1f} ({:.1f} pages), hoisted {:.1f} ({:.1f} pages); {:.1f} blocked by an access without a "
        "known range or recorded while off{}",
        hoist_window_first_frame_, frame, REXCVAR_GET(vulkan_upload_hoist), w.calls / n, w.pages / n,
        w.hoistable_calls / n, w.hoistable_pages / n, w.hoisted_calls / n, w.hoisted_pages / n,
        w.blocked_all / n, diag_flush ? fmt::format(" [diag-stats {}]", diag_now) : std::string());
  }
  w = HoistStats();
  hoist_window_first_frame_ = frame;
  hoist_window_frames_ = 0;
}

bool VulkanSharedMemory::RecordUploadSkipVerify(uint32_t page, const uint8_t* bytes) {
  // wwe2k14 patch (Session 7b, UPSKIP): gpu_upload_skip_unchanged_verify. Called from SharedMemory::
  // SkipUnchangedUploadPages for a page it would skip, before that request's UploadRanges (which then uploads the
  // page anyway): copies the page as the GPU buffer holds it at this point of the command stream to a readback
  // slot, keeping the hashed guest bytes for CheckUploadSkipVerifyReadbacks.
  UploadSkipVerifyStats& stats = upload_skip_verify_stats();
  const VkDeviceSize page_size = VkDeviceSize(1) << page_size_log2();
  if (upload_skip_verify_buffer_ == VK_NULL_HANDLE) {
    if (upload_skip_verify_unavailable_) {
      ++stats.dropped;
      return false;
    }
    const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
    const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();
    uint32_t memory_type = UINT32_MAX;
    void* mapping = nullptr;
    if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
            vulkan_device, page_size * kUploadSkipVerifySlots, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            ui::vulkan::util::MemoryPurpose::kReadback, upload_skip_verify_buffer_,
            upload_skip_verify_memory_, &memory_type) ||
        dfn.vkMapMemory(device, upload_skip_verify_memory_, 0, VK_WHOLE_SIZE, 0, &mapping) !=
            VK_SUCCESS) {
      REXGPU_ERROR("[s7][UPSKIP] verify: failed to create the readback buffer - verify is off");
      ShutdownUploadSkipVerify();
      upload_skip_verify_unavailable_ = true;
      ++stats.dropped;
      return false;
    }
    upload_skip_verify_mapping_ = static_cast<uint8_t*>(mapping);
    upload_skip_verify_coherent_ =
        (vulkan_device->memory_types().host_coherent & (uint32_t(1) << memory_type)) != 0;
    upload_skip_verify_slots_.resize(kUploadSkipVerifySlots);
    upload_skip_verify_hashed_bytes_.resize(size_t(page_size) * kUploadSkipVerifySlots);
    upload_skip_verify_head_ = 0;
    upload_skip_verify_count_ = 0;
  }
  if (upload_skip_verify_count_ >= kUploadSkipVerifySlots) {
    CheckUploadSkipVerifyReadbacks();
    if (upload_skip_verify_count_ >= kUploadSkipVerifySlots) {
      ++stats.dropped;
      return false;
    }
  }
  uint32_t slot = (upload_skip_verify_head_ + upload_skip_verify_count_) % kUploadSkipVerifySlots;
  // Every earlier write to the buffer (uploads, resolves, memory export) made visible to the transfer read.
  Use(Usage::kRead);
  command_processor_.SubmitBarriers(true);
  VkBufferCopy readback_region;
  readback_region.srcOffset = VkDeviceSize(page) << page_size_log2();
  readback_region.dstOffset = VkDeviceSize(slot) * page_size;
  readback_region.size = page_size;
  command_processor_.deferred_command_buffer().CmdVkCopyBuffer(buffer_, upload_skip_verify_buffer_, 1,
                                                               &readback_region);
  command_processor_.PushBufferMemoryBarrier(
      upload_skip_verify_buffer_, readback_region.dstOffset, page_size, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
  std::memcpy(upload_skip_verify_hashed_bytes_.data() + size_t(slot) * size_t(page_size), bytes,
              size_t(page_size));
  upload_skip_verify_slots_[slot].submission = command_processor_.GetCurrentSubmission();
  upload_skip_verify_slots_[slot].lost_upload_count = GetLostUploadCount();
  upload_skip_verify_slots_[slot].page = page;
  ++upload_skip_verify_count_;
  ++stats.recorded;
  return true;
}

uint64_t VulkanSharedMemory::GetLostUploadCount() const {
  // wwe2k14 patch (Session 7b, UPSKIP review): see SharedMemory::GetLostUploadCount.
  return command_processor_.GetAsyncSubmitCommandsDropped();
}

void VulkanSharedMemory::CheckUploadSkipVerifyReadbacks() {
  // wwe2k14 patch (Session 7b, UPSKIP): compares the readbacks of completed submissions, oldest first.
  if (!upload_skip_verify_count_) {
    return;
  }
  const uint64_t submission_completed = command_processor_.GetCompletedSubmission();
  if (upload_skip_verify_slots_[upload_skip_verify_head_].submission > submission_completed) {
    return;
  }
  if (!upload_skip_verify_coherent_) {
    const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
    VkMappedMemoryRange readback_memory_range = {};
    readback_memory_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    readback_memory_range.memory = upload_skip_verify_memory_;
    readback_memory_range.offset = 0;
    readback_memory_range.size = VK_WHOLE_SIZE;
    vulkan_device->functions().vkInvalidateMappedMemoryRanges(vulkan_device->device(), 1,
                                                              &readback_memory_range);
  }
  UploadSkipVerifyStats& stats = upload_skip_verify_stats();
  const size_t page_size = size_t(1) << page_size_log2();
  while (upload_skip_verify_count_) {
    const UploadSkipVerifySlot& slot = upload_skip_verify_slots_[upload_skip_verify_head_];
    if (slot.submission > submission_completed) {
      break;
    }
    // wwe2k14 patch (Session 7b, UPSKIP review): an async submission's commands were dropped since this readback
    // was recorded (counted before that submission's fence can be signaled, so seen here if it was this one):
    // the readback may never have run, so it is not compared (counted as dropped).
    if (GetLostUploadCount() != slot.lost_upload_count) {
      ++stats.dropped;
      upload_skip_verify_head_ = (upload_skip_verify_head_ + 1) % kUploadSkipVerifySlots;
      --upload_skip_verify_count_;
      continue;
    }
    const uint8_t* gpu_bytes = upload_skip_verify_mapping_ + size_t(upload_skip_verify_head_) * page_size;
    const uint8_t* hashed_bytes =
        upload_skip_verify_hashed_bytes_.data() + size_t(upload_skip_verify_head_) * page_size;
    ++stats.checked;
    if (std::memcmp(gpu_bytes, hashed_bytes, page_size)) {
      ++stats.mismatches;
      if (upload_skip_verify_mismatches_logged_ < 32) {
        ++upload_skip_verify_mismatches_logged_;
        size_t first_difference = 0;
        while (first_difference < page_size &&
               gpu_bytes[first_difference] == hashed_bytes[first_difference]) {
          ++first_difference;
        }
        size_t differing_bytes = 0;
        for (size_t i = first_difference; i < page_size; ++i) {
          differing_bytes += gpu_bytes[i] != hashed_bytes[i];
        }
        REXGPU_WARN(
            "[s7][UPSKIP] verify mismatch addr=0x{:08X}: the GPU buffer did not hold the guest bytes whose hash "
            "matched (first difference at +0x{:X}, {} bytes differ; submission {}, {} mismatches so far)",
            slot.page << page_size_log2(), first_difference, differing_bytes, slot.submission,
            stats.mismatches);
      }
    }
    upload_skip_verify_head_ = (upload_skip_verify_head_ + 1) % kUploadSkipVerifySlots;
    --upload_skip_verify_count_;
  }
}

void VulkanSharedMemory::ShutdownUploadSkipVerify() {
  // wwe2k14 patch (Session 7b, UPSKIP).
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  if (upload_skip_verify_mapping_) {
    dfn.vkUnmapMemory(device, upload_skip_verify_memory_);
    upload_skip_verify_mapping_ = nullptr;
  }
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, upload_skip_verify_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, upload_skip_verify_memory_);
  upload_skip_verify_head_ = 0;
  upload_skip_verify_count_ = 0;
}

void VulkanSharedMemory::GetUsageMasks(Usage usage, VkPipelineStageFlags& stage_mask,
                                       VkAccessFlags& access_mask) const {
  switch (usage) {
    case Usage::kComputeWrite:
      stage_mask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      access_mask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;  // Session 93: compute WRITES (were made available by nothing)
      return;
    case Usage::kTransferDestination:
      stage_mask = VK_PIPELINE_STAGE_TRANSFER_BIT;
      access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
      return;
    default:
      break;
  }
  stage_mask = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | guest_shader_pipeline_stages_;
  access_mask = VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;  // (Session 95: attribute draws)
  switch (usage) {
    case Usage::kRead:
      stage_mask |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
      access_mask |= VK_ACCESS_TRANSFER_READ_BIT;
      break;
    case Usage::kGuestDrawReadWrite:
      access_mask |= VK_ACCESS_SHADER_WRITE_BIT;
      break;
    default:
      assert_unhandled_case(usage);
  }
}

}  // namespace rex::graphics::vulkan
