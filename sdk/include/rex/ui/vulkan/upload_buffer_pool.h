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

#include <rex/ui/graphics_upload_buffer_pool.h>
#include <rex/ui/vulkan/device.h>

namespace rex {
namespace ui {
namespace vulkan {

class VulkanUploadBufferPool : public GraphicsUploadBufferPool {
 public:
  // wwe2k14 patch (Session 7b, REBAR): `name` (a string literal, or null) names the pool in the [s7][REBAR] log
  // of the memory type chosen (logged for every pool, when the first page is created). With
  // prefer_device_local, the pages go to a DEVICE_LOCAL | HOST_VISIBLE memory type (HOST_COHERENT preferred)
  // in a heap of at least device_local_min_heap_size bytes if the buffers can use one (Resizable BAR: the copy
  // source / uniform data in VRAM instead of system memory across PCIe), else to the usual host-visible type
  // (the log says why); the choice is then made (and logged) here, in the constructor, so
  // may_have_device_local_pages() is final from creation. Each pool keeps at most 1 / kDeviceLocalHeapShare
  // of that heap in device-local pages (later pages go to the usual type, until ClearCache frees them), so it
  // can't take all of a small heap (the 256 MB PCI BAR window without Resizable BAR). Latched here, for the
  // pool's lifetime. Without it (the defaults), nothing changes.
  VulkanUploadBufferPool(const VulkanDevice* vulkan_device, VkBufferUsageFlags usage,
                         size_t page_size = kDefaultPageSize, const char* name = nullptr,
                         bool prefer_device_local = false, VkDeviceSize device_local_min_heap_size = 0);

  uint8_t* Request(uint64_t submission_index, size_t size, size_t alignment, VkBuffer& buffer_out,
                   VkDeviceSize& offset_out);
  uint8_t* RequestPartial(uint64_t submission_index, size_t size, size_t alignment,
                          VkBuffer& buffer_out, VkDeviceSize& offset_out, VkDeviceSize& size_out);

  // wwe2k14 patch (Session 7b, REBAR).
  // Logs every memory heap and memory type of the device ([s7][REBAR]).
  static void LogMemoryHeapsAndTypes(const VulkanDevice* vulkan_device);
  bool prefers_device_local() const { return prefer_device_local_; }
  // wwe2k14 patch (Session 59, vulkan_uniform_pool_mirror): each page also gets a device-local copy (not host-visible)
  // that Request hands out as the buffer; the CPU writes the host page, and every range FlushWrites flushes is listed
  // (pending_mirror_copies) for the owner to copy host -> mirror before anything of the submission reads it. Set before
  // the first Request.
  struct MirrorCopy {
    VkBuffer source, mirror;
    VkDeviceSize offset, size;
  };
  void SetMirror(bool mirror) { mirror_ = mirror; }
  bool mirror() const { return mirror_; }
  std::vector<MirrorCopy>& pending_mirror_copies() { return pending_mirror_copies_; }
  // Some pages may be (or become) device-local host-visible memory chosen for prefer_device_local. The CPU must
  // not read them: write-combined and across PCIe, every read is an uncached round trip to the GPU. Final from
  // creation (the memory type is chosen in the constructor with prefer_device_local): false without
  // prefer_device_local or when no device-local type qualified (the usual type was chosen), and still true
  // after an allocation fallback (the earlier pages stay device-local).
  bool may_have_device_local_pages() const { return device_local_memory_type_ != UINT32_MAX; }
  // The memory type new pages are allocated from (UINT32_MAX before it is chosen or if there is none), and
  // the device-local one chosen for prefer_device_local (UINT32_MAX if none; differs from memory_type() after
  // an allocation fallback).
  uint32_t memory_type() const { return memory_type_ < kMemoryTypeUnavailable ? memory_type_ : UINT32_MAX; }
  uint32_t device_local_memory_type() const { return device_local_memory_type_; }
  // Counters since creation (owner thread only, like Request).
  struct Stats {
    uint64_t requested_bytes = 0;       // bytes returned by Request / RequestPartial
    uint64_t pages_created = 0;         // pages (buffers) created
    uint64_t pages_device_local = 0;    // of those, in the device-local memory type of prefer_device_local
    uint64_t allocation_fallbacks = 0;  // device-local page allocations that failed and used the usual type
    uint64_t pages_over_share = 0;      // pages put in the usual type as the pool held its share of the heap
  };
  // The most of the device-local heap one pool keeps in pages: 1 / kDeviceLocalHeapShare of it (a quarter, so
  // the two pools that may prefer it take at most half: about 53 MB each of a 214 MB BAR heap; with Resizable
  // BAR, gigabytes - never reached).
  static constexpr VkDeviceSize kDeviceLocalHeapShare = 4;
  const Stats& stats() const { return stats_; }

 protected:
  Page* CreatePageImplementation() override;

  void FlushPageWrites(Page* page, size_t offset, size_t size) override;

 private:
  struct VulkanPage : public Page {
    // Takes ownership of the buffer and its memory and mapping.
    VulkanPage(const VulkanDevice* vulkan_device, VkBuffer buffer, VkDeviceMemory memory,
               void* mapping, uint32_t memory_type)
        : vulkan_device_(vulkan_device),
          buffer_(buffer),
          memory_(memory),
          mapping_(mapping),
          memory_type_(memory_type) {}
    ~VulkanPage() override;
    const VulkanDevice* vulkan_device_;
    VkBuffer buffer_;
    VkDeviceMemory memory_;
    void* mapping_;
    // wwe2k14 patch (Session 7b, REBAR): pages of one pool may differ after an allocation fallback.
    uint32_t memory_type_;
    VkBuffer mirror_buffer_ = VK_NULL_HANDLE;  // wwe2k14 patch (Session 59)
    VkDeviceMemory mirror_memory_ = VK_NULL_HANDLE;
  };
  bool mirror_ = false;  // wwe2k14 patch (Session 59)
  std::vector<MirrorCopy> pending_mirror_copies_;
  void CreateMirror(VulkanPage& page);

  // wwe2k14 patch (Session 7b, REBAR): the memory type for buffers that can be in supported_types - the usual
  // host-visible one, or the device-local one with prefer_device_local (then also sets device_local_out and
  // usual_out; reason_out says why the device-local one was or wasn't chosen).
  uint32_t ChooseMemoryType(uint32_t supported_types, bool& device_local_out, uint32_t& usual_out,
                            const char*& reason_out) const;
  // wwe2k14 patch (Session 7b, REBAR): the original first-page memory type choice, moved out of
  // CreatePageImplementation unchanged (also chooses the device-local type and logs the choice): sets
  // memory_type_, allocation_size_ and maybe page_size_ from `buffer`, created from buffer_create_info with
  // page_size_ bytes, which it may replace with an expanded one. On failure, destroys `buffer`, sets
  // kMemoryTypeUnavailable and returns false.
  bool InitializeMemoryType(VkBuffer& buffer, VkBufferCreateInfo& buffer_create_info);
  // wwe2k14 patch (Session 7b, REBAR): fills buffer_create_info for a page buffer of page_size_ bytes.
  void FillBufferCreateInfo(VkBufferCreateInfo& buffer_create_info) const;

  const VulkanDevice* vulkan_device_;

  VkDeviceSize allocation_size_;
  static constexpr uint32_t kMemoryTypeUnknown = UINT32_MAX;
  static constexpr uint32_t kMemoryTypeUnavailable = kMemoryTypeUnknown - 1;
  uint32_t memory_type_ = UINT32_MAX;

  VkBufferUsageFlags usage_;

  // wwe2k14 patch (Session 7b, REBAR).
  const char* name_;
  bool prefer_device_local_;
  VkDeviceSize device_local_min_heap_size_;
  // The constructor couldn't create the probe buffer for the early choice: the usual type, as without
  // prefer_device_local (the choice must be known before anything is written).
  bool device_local_probe_failed_ = false;
  // The device-local type chosen for prefer_device_local (UINT32_MAX if none), and the usual type to fall back
  // to when an allocation from it fails or the pool holds its share of the heap (device_local_max_bytes_).
  uint32_t device_local_memory_type_ = UINT32_MAX;
  uint32_t usual_memory_type_ = UINT32_MAX;
  VkDeviceSize device_local_max_bytes_ = 0;
  bool device_local_share_logged_ = false;
  Stats stats_;
};

}  // namespace vulkan
}  // namespace ui
}  // namespace rex
