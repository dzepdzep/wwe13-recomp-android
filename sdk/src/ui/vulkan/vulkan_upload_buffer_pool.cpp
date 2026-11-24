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
#include <string>
#include <string_view>
#include <utility>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/ui/vulkan/upload_buffer_pool.h>
#include <rex/ui/vulkan/util.h>

namespace rex {
namespace ui {
namespace vulkan {

namespace {

// wwe2k14 patch (Session 7b, REBAR): memory property flags for the [s7][REBAR] logs.
std::string MemoryPropertyFlagsString(VkMemoryPropertyFlags flags) {
  static constexpr std::pair<VkMemoryPropertyFlags, const char*> kNames[] = {
      {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "DEVICE_LOCAL"},
      {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, "HOST_VISIBLE"},
      {VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "HOST_COHERENT"},
      {VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "HOST_CACHED"},
      {VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, "LAZILY_ALLOCATED"},
      {VK_MEMORY_PROPERTY_PROTECTED_BIT, "PROTECTED"},
  };
  std::string result;
  for (const auto& [bit, name] : kNames) {
    if (flags & bit) {
      if (!result.empty()) {
        result += '|';
      }
      result += name;
      flags &= ~bit;
    }
  }
  if (flags) {
    result += fmt::format("{}0x{:X}", result.empty() ? "" : "|", uint32_t(flags));
  }
  return result.empty() ? std::string("none") : result;
}

VkPhysicalDeviceMemoryProperties GetMemoryProperties(const VulkanDevice* vulkan_device) {
  VkPhysicalDeviceMemoryProperties memory_properties;
  vulkan_device->vulkan_instance()->functions().vkGetPhysicalDeviceMemoryProperties(
      vulkan_device->physical_device(), &memory_properties);
  return memory_properties;
}

}  // namespace

void VulkanUploadBufferPool::LogMemoryHeapsAndTypes(const VulkanDevice* vulkan_device) {
  // wwe2k14 patch (Session 7b, REBAR): without Resizable BAR, the DEVICE_LOCAL | HOST_VISIBLE memory is in the
  // 256 MB PCI BAR window (a 214 MB heap on NVIDIA); with it, in a heap about the size of VRAM.
  const VkPhysicalDeviceMemoryProperties memory_properties = GetMemoryProperties(vulkan_device);
  REXGPU_INFO("[s7][REBAR] Vulkan memory of {}: {} heaps, {} types (nonCoherentAtomSize {})",
              vulkan_device->properties().deviceName, memory_properties.memoryHeapCount,
              memory_properties.memoryTypeCount, vulkan_device->properties().nonCoherentAtomSize);
  for (uint32_t i = 0; i < memory_properties.memoryHeapCount; ++i) {
    const VkMemoryHeap& heap = memory_properties.memoryHeaps[i];
    REXGPU_INFO("[s7][REBAR]   heap {}: {} MB, flags 0x{:X}{}{}", i, heap.size >> 20, uint32_t(heap.flags),
                (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? " DEVICE_LOCAL" : "",
                (heap.flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT) ? " MULTI_INSTANCE" : "");
  }
  for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
    const VkMemoryType& type = memory_properties.memoryTypes[i];
    REXGPU_INFO("[s7][REBAR]   type {}: heap {}, {}", i, type.heapIndex,
                MemoryPropertyFlagsString(type.propertyFlags));
  }
}

uint32_t VulkanUploadBufferPool::ChooseMemoryType(uint32_t supported_types, bool& device_local_out,
                                                  uint32_t& usual_out, const char*& reason_out) const {
  // wwe2k14 patch (Session 7b, REBAR): without prefer_device_local, exactly the original choice.
  const VulkanDevice::MemoryTypes& memory_types = vulkan_device_->memory_types();
  const uint32_t usual = util::ChooseHostMemoryType(memory_types, supported_types, false);
  device_local_out = false;
  usual_out = usual;
  if (!prefer_device_local_) {
    reason_out = "device-local preference off";
    return usual;
  }
  if (device_local_probe_failed_) {
    reason_out = "device-local preference on, but the probe buffer could not be created at pool creation - "
                 "usual type kept";
    return usual;
  }
  const uint32_t candidates = supported_types & memory_types.device_local & memory_types.host_visible;
  if (!candidates) {
    reason_out = "device-local preference on, but no DEVICE_LOCAL | HOST_VISIBLE memory type can hold these "
                 "buffers - usual type kept";
    return usual;
  }
  const VkPhysicalDeviceMemoryProperties memory_properties = GetMemoryProperties(vulkan_device_);
  uint32_t big_enough = 0;
  for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
    if ((candidates & (uint32_t(1) << i)) &&
        memory_properties.memoryHeaps[memory_properties.memoryTypes[i].heapIndex].size >=
            device_local_min_heap_size_) {
      big_enough |= uint32_t(1) << i;
    }
  }
  if (!big_enough) {
    reason_out = "device-local preference on, but DEVICE_LOCAL | HOST_VISIBLE memory is only in heaps smaller "
                 "than the minimum (no Resizable BAR) - usual type kept";
    return usual;
  }
  // Coherent (no flushes needed) and uncached (write-combined: the writes don't go through the CPU caches)
  // preferred.
  uint32_t memory_type;
  if (rex::bit_scan_forward(big_enough & memory_types.host_coherent & ~memory_types.host_cached,
                            &memory_type) ||
      rex::bit_scan_forward(big_enough & memory_types.host_coherent, &memory_type) ||
      rex::bit_scan_forward(big_enough, &memory_type)) {
    device_local_out = true;
    reason_out = (memory_types.host_coherent & (uint32_t(1) << memory_type))
                     ? "device-local preference on: DEVICE_LOCAL | HOST_VISIBLE memory type chosen"
                     : "device-local preference on: DEVICE_LOCAL | HOST_VISIBLE memory type chosen (not "
                       "coherent: written ranges are flushed before submission, as for any non-coherent "
                       "upload memory)";
    return memory_type;
  }
  reason_out = "device-local preference on - usual type kept";
  return usual;
}

// Host-visible memory sizes are likely to be internally rounded to
// nonCoherentAtomSize (it's the memory mapping granularity, though as the map
// or flush range must be clamped to the actual allocation size as a special
// case, but it's still unlikely that the allocation won't be aligned to it), so
// try not to waste that padding.
REXCVAR_DEFINE_INT32(vulkan_upload_pool_keep_mb, 0, "GPU/Vulkan",
                     "Session 94 (low RAM): the shared-memory upload pool frees reclaimed pages above this many MB "
                     "(the peak burst of a level load otherwise stays allocated for the whole run). 0 = keep all, "
                     "as before. wwe2k14_low_ram sets 32. Restart required")
    .range(0, 1024);

VulkanUploadBufferPool::VulkanUploadBufferPool(const VulkanDevice* const vulkan_device,
                                               const VkBufferUsageFlags usage,
                                               const size_t page_size, const char* const name,
                                               const bool prefer_device_local,
                                               const VkDeviceSize device_local_min_heap_size)
    : GraphicsUploadBufferPool(size_t(
          rex::round_up(VkDeviceSize(page_size), vulkan_device->properties().nonCoherentAtomSize))),
      vulkan_device_(vulkan_device),
      usage_(usage),
      name_(name),
      prefer_device_local_(prefer_device_local),
      device_local_min_heap_size_(device_local_min_heap_size) {
  if (name && std::string_view(name) == "shared memory upload") {
    SetKeepBytes(size_t(std::max(REXCVAR_GET(vulkan_upload_pool_keep_mb), int32_t(0))) << 20);
  }
  // wwe2k14 patch (Session 7b, REBAR): with prefer_device_local, the memory type is chosen now, from a probe
  // buffer created exactly like the first page's (memoryTypeBits depends only on the buffer's flags and usage,
  // so the pages get the same choice), so whether the pages may be device-local is final before anything is
  // written - the CP7 shadow copies and the streaming copy verify skip follow the type actually chosen, not the
  // preference. Without prefer_device_local, the first page chooses, as originally.
  if (!prefer_device_local_) {
    return;
  }
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  VkBufferCreateInfo buffer_create_info;
  FillBufferCreateInfo(buffer_create_info);
  VkBuffer buffer;
  if (dfn.vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer) != VK_SUCCESS) {
    // Not known which memory the pages could be in: never device-local, the first page chooses the usual type.
    device_local_probe_failed_ = true;
    REXGPU_WARN(
        "[s7][REBAR] upload buffer pool '{}': creating a {} KB probe buffer to choose the memory type failed - "
        "device-local preference dropped (the usual memory type will be used)",
        name_ ? name_ : "unnamed", page_size_ >> 10);
    return;
  }
  if (InitializeMemoryType(buffer, buffer_create_info)) {
    // Only the choice was needed - the pages create their own buffers (of the final page_size_).
    dfn.vkDestroyBuffer(device, buffer, nullptr);
  }
}

uint8_t* VulkanUploadBufferPool::Request(uint64_t submission_index, size_t size, size_t alignment,
                                         VkBuffer& buffer_out, VkDeviceSize& offset_out) {
  size_t offset;
  const VulkanPage* page = static_cast<const VulkanPage*>(
      GraphicsUploadBufferPool::Request(submission_index, size, alignment, offset));
  if (!page) {
    return nullptr;
  }
  buffer_out = page->mirror_buffer_ != VK_NULL_HANDLE ? page->mirror_buffer_ : page->buffer_;  // (S59)
  offset_out = VkDeviceSize(offset);
  stats_.requested_bytes += size;  // wwe2k14 patch (Session 7b, REBAR)
  return reinterpret_cast<uint8_t*>(page->mapping_) + offset;
}

uint8_t* VulkanUploadBufferPool::RequestPartial(uint64_t submission_index, size_t size,
                                                size_t alignment, VkBuffer& buffer_out,
                                                VkDeviceSize& offset_out, VkDeviceSize& size_out) {
  size_t offset, size_obtained;
  const VulkanPage* page = static_cast<const VulkanPage*>(GraphicsUploadBufferPool::RequestPartial(
      submission_index, size, alignment, offset, size_obtained));
  if (!page) {
    return nullptr;
  }
  buffer_out = page->mirror_buffer_ != VK_NULL_HANDLE ? page->mirror_buffer_ : page->buffer_;  // (S59)
  offset_out = VkDeviceSize(offset);
  size_out = VkDeviceSize(size_obtained);
  stats_.requested_bytes += size_obtained;  // wwe2k14 patch (Session 7b, REBAR)
  return reinterpret_cast<uint8_t*>(page->mapping_) + offset;
}

void VulkanUploadBufferPool::FillBufferCreateInfo(VkBufferCreateInfo& buffer_create_info) const {
  buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.pNext = nullptr;
  buffer_create_info.flags = 0;
  buffer_create_info.size = VkDeviceSize(page_size_);
  buffer_create_info.usage = usage_ | (mirror_ ? VkBufferUsageFlags(VK_BUFFER_USAGE_TRANSFER_SRC_BIT) : 0);
  buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  buffer_create_info.queueFamilyIndexCount = 0;
  buffer_create_info.pQueueFamilyIndices = nullptr;
}

bool VulkanUploadBufferPool::InitializeMemoryType(VkBuffer& buffer,
                                                  VkBufferCreateInfo& buffer_create_info) {
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  VkMemoryRequirements memory_requirements;
  dfn.vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);
  // wwe2k14 patch (Session 7b, REBAR): ChooseMemoryType is util::ChooseHostMemoryType(..., false) unless
  // prefer_device_local.
  bool memory_type_device_local;
  uint32_t memory_type_usual;
  const char* memory_type_reason;
  memory_type_ = ChooseMemoryType(memory_requirements.memoryTypeBits, memory_type_device_local,
                                  memory_type_usual, memory_type_reason);
  if (memory_type_ == UINT32_MAX) {
    REXLOG_ERROR(
        "No host-visible memory types can store an Vulkan upload buffer with "
        "{} bytes",
        page_size_);
    memory_type_ = kMemoryTypeUnavailable;
    dfn.vkDestroyBuffer(device, buffer, nullptr);
    return false;
  }
  allocation_size_ = memory_requirements.size;
  if (allocation_size_ > page_size_) {
    // Try to occupy the allocation padding. If that's going to require even
    // more memory for some reason, don't.
    buffer_create_info.size = allocation_size_;
    VkBuffer buffer_expanded;
    if (dfn.vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer_expanded) == VK_SUCCESS) {
      VkMemoryRequirements memory_requirements_expanded;
      dfn.vkGetBufferMemoryRequirements(device, buffer_expanded, &memory_requirements_expanded);
      // wwe2k14 patch (Session 7b, REBAR): the same choice as above (the same arguments).
      bool memory_type_expanded_device_local;
      uint32_t memory_type_expanded_usual;
      const char* memory_type_expanded_reason;
      uint32_t memory_type_expanded =
          ChooseMemoryType(memory_requirements.memoryTypeBits, memory_type_expanded_device_local,
                           memory_type_expanded_usual, memory_type_expanded_reason);
      if (memory_requirements_expanded.size <= allocation_size_ &&
          memory_type_expanded != UINT32_MAX) {
        page_size_ = size_t(allocation_size_);
        allocation_size_ = memory_requirements_expanded.size;
        memory_type_ = memory_type_expanded;
        memory_type_device_local = memory_type_expanded_device_local;
        memory_type_usual = memory_type_expanded_usual;
        memory_type_reason = memory_type_expanded_reason;
        dfn.vkDestroyBuffer(device, buffer, nullptr);
        buffer = buffer_expanded;
      } else {
        dfn.vkDestroyBuffer(device, buffer_expanded, nullptr);
      }
    }
  }
  // wwe2k14 patch (Session 7b, REBAR): which memory each pool uses, once per pool (always logged), and with the
  // device-local type, the most of its heap this pool's pages may hold.
  const VkPhysicalDeviceMemoryProperties memory_properties = GetMemoryProperties(vulkan_device_);
  const VkMemoryType& chosen_type = memory_properties.memoryTypes[memory_type_];
  const VkDeviceSize chosen_heap_size = memory_properties.memoryHeaps[chosen_type.heapIndex].size;
  if (memory_type_device_local) {
    device_local_memory_type_ = memory_type_;
    usual_memory_type_ = memory_type_usual;
    device_local_max_bytes_ = chosen_heap_size / kDeviceLocalHeapShare;
  }
  REXGPU_INFO(
      "[s7][REBAR] upload buffer pool '{}' (usage 0x{:X}, pages of {} KB): memory type {} (heap {}, {} MB, "
      "{}); {} (minimum heap {} MB, usual type {}){}",
      name_ ? name_ : "unnamed", uint32_t(usage_), page_size_ >> 10, memory_type_, chosen_type.heapIndex,
      chosen_heap_size >> 20, MemoryPropertyFlagsString(chosen_type.propertyFlags), memory_type_reason,
      device_local_min_heap_size_ >> 20, memory_type_usual,
      memory_type_device_local
          ? fmt::format("; at most {} MB (1/{} of the heap) of device-local pages in this pool, later pages in "
                        "the usual type",
                        device_local_max_bytes_ >> 20, kDeviceLocalHeapShare)
          : std::string());
  return true;
}

GraphicsUploadBufferPool::Page* VulkanUploadBufferPool::CreatePageImplementation() {
  if (memory_type_ == kMemoryTypeUnavailable) {
    // Don't try to create everything again and again if totally broken.
    return nullptr;
  }

  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  // wwe2k14 patch (Session 7b, REBAR): the create info and the first-time memory type choice moved to
  // FillBufferCreateInfo and InitializeMemoryType (unchanged; the choice may have been made in the
  // constructor).
  VkBufferCreateInfo buffer_create_info;
  FillBufferCreateInfo(buffer_create_info);
  VkBuffer buffer;
  if (dfn.vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer) != VK_SUCCESS) {
    REXLOG_ERROR("Failed to create a Vulkan upload buffer with {} bytes", page_size_);
    return nullptr;
  }

  if (memory_type_ == kMemoryTypeUnknown && !InitializeMemoryType(buffer, buffer_create_info)) {
    return nullptr;
  }

  // wwe2k14 patch (Session 7b, REBAR): this page's memory type. With the device-local type, the usual one
  // instead while the pool's live device-local pages hold its share of the heap (device_local_max_bytes_), so
  // one pool never fills a small heap (the PCI BAR window without Resizable BAR) and the two pools that may use
  // it get fixed shares. Pages are freed by ClearCache (and, with vulkan_upload_pool_keep_mb, one reclaimed page per Reclaim), so the live ones are those in the two lists (the
  // page being replaced has already been moved to the submitted list).
  const bool device_local_fallback_possible = device_local_memory_type_ != UINT32_MAX &&
                                              usual_memory_type_ != UINT32_MAX &&
                                              usual_memory_type_ != device_local_memory_type_;
  uint32_t page_memory_type = memory_type_;
  if (device_local_fallback_possible && page_memory_type == device_local_memory_type_) {
    VkDeviceSize device_local_bytes = 0;
    for (const Page* list : {writable_first_, submitted_first_}) {
      for (const Page* page = list; page; page = page->next_) {
        if (static_cast<const VulkanPage*>(page)->memory_type_ == device_local_memory_type_) {
          device_local_bytes += allocation_size_;
        }
      }
    }
    if (device_local_bytes + allocation_size_ > device_local_max_bytes_) {
      page_memory_type = usual_memory_type_;
      ++stats_.pages_over_share;
      if (!device_local_share_logged_) {
        device_local_share_logged_ = true;
        REXGPU_WARN(
            "[s7][REBAR] upload buffer pool '{}': its device-local pages hold {} MB of its {} MB share of the "
            "heap - new pages use memory type {} while they do (counted as pages over the share; logged once)",
            name_ ? name_ : "unnamed", device_local_bytes >> 20, device_local_max_bytes_ >> 20,
            usual_memory_type_);
      }
    }
  }

  VkMemoryAllocateInfo memory_allocate_info;
  VkMemoryAllocateInfo* memory_allocate_info_last = &memory_allocate_info;
  memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memory_allocate_info.pNext = nullptr;
  memory_allocate_info.allocationSize = allocation_size_;
  memory_allocate_info.memoryTypeIndex = page_memory_type;
  VkMemoryDedicatedAllocateInfo memory_dedicated_allocate_info;
  if (vulkan_device_->extensions().ext_1_1_KHR_dedicated_allocation) {
    memory_allocate_info_last->pNext = &memory_dedicated_allocate_info;
    memory_allocate_info_last =
        reinterpret_cast<VkMemoryAllocateInfo*>(&memory_dedicated_allocate_info);
    memory_dedicated_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    memory_dedicated_allocate_info.pNext = nullptr;
    memory_dedicated_allocate_info.image = VK_NULL_HANDLE;
    memory_dedicated_allocate_info.buffer = buffer;
  }
  // wwe2k14 patch (Session 7b, REBAR): allocates and maps memory of type `type` (freed again if mapping fails).
  // The memory is mapped before it is bound to the buffer (mapping doesn't need the binding - the same result),
  // so after a failed device-local attempt the buffer is still unbound and can take memory of the usual type.
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapping = nullptr;
  bool mapping_failed = false;
  auto allocate_and_map = [&](const uint32_t type) -> VkResult {
    memory_allocate_info.memoryTypeIndex = type;
    mapping_failed = false;
    VkResult result = dfn.vkAllocateMemory(device, &memory_allocate_info, nullptr, &memory);
    if (result != VK_SUCCESS) {
      return result;
    }
    result = dfn.vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapping);
    if (result != VK_SUCCESS) {
      mapping_failed = true;
      dfn.vkFreeMemory(device, memory, nullptr);
      memory = VK_NULL_HANDLE;
    }
    return result;
  };
  VkResult allocate_result = allocate_and_map(page_memory_type);
  // wwe2k14 patch (Session 7b, REBAR): the device-local host-visible heap may be full (without Resizable BAR it
  // is a 214-256 MB window shared with the driver and other programs), or mapping it may fail: use the usual
  // type from now on.
  if (allocate_result != VK_SUCCESS && device_local_fallback_possible &&
      page_memory_type == device_local_memory_type_) {
    ++stats_.allocation_fallbacks;
    REXGPU_WARN(
        "[s7][REBAR] upload buffer pool '{}': {} a {} KB page of device-local memory type {} failed (VkResult "
        "{}) - this and later pages use memory type {}",
        name_ ? name_ : "unnamed", mapping_failed ? "mapping" : "allocating", allocation_size_ >> 10,
        page_memory_type, int32_t(allocate_result), usual_memory_type_);
    memory_type_ = usual_memory_type_;
    page_memory_type = memory_type_;
    allocate_result = allocate_and_map(page_memory_type);
  }
  if (allocate_result != VK_SUCCESS) {
    if (mapping_failed) {
      REXLOG_ERROR("Failed to map {} bytes of Vulkan upload buffer memory", allocation_size_);
    } else {
      REXLOG_ERROR("Failed to allocate {} bytes of Vulkan upload buffer memory", allocation_size_);
    }
    dfn.vkDestroyBuffer(device, buffer, nullptr);
    return nullptr;
  }

  if (dfn.vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
    REXLOG_ERROR("Failed to bind memory to a Vulkan upload buffer with {} bytes", page_size_);
    dfn.vkDestroyBuffer(device, buffer, nullptr);
    dfn.vkFreeMemory(device, memory, nullptr);
    return nullptr;
  }

  // wwe2k14 patch (Session 7b, REBAR): counters, and the page's own memory type for flushing.
  ++stats_.pages_created;
  if (page_memory_type == device_local_memory_type_) {
    ++stats_.pages_device_local;
  }
  VulkanPage* const page = new VulkanPage(vulkan_device_, buffer, memory, mapping, page_memory_type);
  if (mirror_) CreateMirror(*page);  // wwe2k14 patch (Session 59)
  return page;
}

// wwe2k14 patch (Session 59, vulkan_uniform_pool_mirror): the page's device-local copy (a type with DEVICE_LOCAL and,
// if possible, without HOST_VISIBLE: the card's own memory, not the BAR window). On failure the page has none and
// Request hands out the host page as before.
void VulkanUploadBufferPool::CreateMirror(VulkanPage& page) {
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  VkBufferCreateInfo info;
  FillBufferCreateInfo(info);
  info.size = allocation_size_;
  info.usage = usage_ | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkBuffer buffer;
  if (dfn.vkCreateBuffer(device, &info, nullptr, &buffer) != VK_SUCCESS) return;
  VkMemoryRequirements requirements;
  dfn.vkGetBufferMemoryRequirements(device, buffer, &requirements);
  const VkPhysicalDeviceMemoryProperties properties = GetMemoryProperties(vulkan_device_);
  uint32_t type = UINT32_MAX;
  for (int pass = 0; pass < 2 && type == UINT32_MAX; ++pass) {
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
      const VkMemoryPropertyFlags flags = properties.memoryTypes[i].propertyFlags;
      if (!(requirements.memoryTypeBits & (1u << i)) || !(flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
      if (pass == 0 && (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
      type = i;
      break;
    }
  }
  VkMemoryAllocateInfo allocate = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocate.allocationSize = requirements.size;
  allocate.memoryTypeIndex = type;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  if (type == UINT32_MAX || dfn.vkAllocateMemory(device, &allocate, nullptr, &memory) != VK_SUCCESS ||
      dfn.vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
    if (memory != VK_NULL_HANDLE) dfn.vkFreeMemory(device, memory, nullptr);
    dfn.vkDestroyBuffer(device, buffer, nullptr);
    return;
  }
  page.mirror_buffer_ = buffer;
  page.mirror_memory_ = memory;
}

void VulkanUploadBufferPool::FlushPageWrites(Page* page, size_t offset, size_t size) {
  const VulkanPage* vulkan_page = static_cast<const VulkanPage*>(page);
  if (vulkan_page->mirror_buffer_ != VK_NULL_HANDLE && size) {  // wwe2k14 patch (Session 59)
    pending_mirror_copies_.push_back(
        {vulkan_page->buffer_, vulkan_page->mirror_buffer_, VkDeviceSize(offset), VkDeviceSize(size)});
  }
  util::FlushMappedMemoryRange(vulkan_device_, vulkan_page->memory_, vulkan_page->memory_type_,
                               VkDeviceSize(offset), allocation_size_, VkDeviceSize(size));
}

VulkanUploadBufferPool::VulkanPage::~VulkanPage() {
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  dfn.vkDestroyBuffer(device, buffer_, nullptr);
  // Unmapping is done implicitly when the memory is freed.
  dfn.vkFreeMemory(device, memory_, nullptr);
  if (mirror_buffer_ != VK_NULL_HANDLE) dfn.vkDestroyBuffer(device, mirror_buffer_, nullptr);  // (Session 59)
  if (mirror_memory_ != VK_NULL_HANDLE) dfn.vkFreeMemory(device, mirror_memory_, nullptr);
}

}  // namespace vulkan
}  // namespace ui
}  // namespace rex
