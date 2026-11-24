// wwe2k14 patch (Session 95, plan item P): see inline_ring.h.
#include "inline_ring.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <vector>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/ui/vulkan/device.h>
#include <rex/ui/vulkan/util.h>

REXCVAR_DEFINE_BOOL(gpu_d3d_inline_vertex_ring, false, "GPU/Native",
                    "Session 95 (plan item P): under the cut, an inline draw's vertex bytes go into a host-visible ring, read through a descriptor set of its own in the place of the "
                    "shared memory's, not through guest memory: no write watch, no RequestRange, no transfer upload, and the draw never ends the owned render pass. Every shader takes it as it is. "
                    "A draw the ring cannot take stays on the guest memory path, counted by reason. [d3d][inline] ring lines. false = the guest memory path (default). Live")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

namespace rex::graphics::vulkan {

namespace {
// Measured (run 6552, the pyro): ~1000 draws of ~85 bytes, 16-byte aligned: ~130 KB a frame at most. A slice holds ~8 such frames, so a slice is entered again long after
// the GPU left it.
constexpr uint32_t kSliceBytes = 1u << 20;
// (An address of 0 is never given: a fetch constant at 0 is what an unset one looks like.)
constexpr uint32_t kFirstAddress = 256;
}  // namespace

InlineVertexRing& InlineRing() {
  static InlineVertexRing ring;
  return ring;
}

bool InlineVertexRing::Ensure(const ui::vulkan::VulkanDevice* device, VkDescriptorSetLayout set_layout, uint32_t binding_count) {
  if (set_ != VK_NULL_HANDLE) return true;
  if (failed_) return false;
  failed_ = true;  // (until everything below exists)
  device_ = device;
  const ui::vulkan::VulkanDevice::Functions& dfn = device->functions();
  const VkDevice vk_device = device->device();
  const VkDeviceSize total = VkDeviceSize(kSliceBytes) * kSlices;
  if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(device, total, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, ui::vulkan::util::MemoryPurpose::kUpload, buffer_, buffer_memory_,
                                                         &memory_type_, &memory_size_)) {
    REXGPU_ERROR("[d3d][inline] ring: the buffer could not be made; inline draws stay on guest memory");
    return false;
  }
  void* map = nullptr;
  if (dfn.vkMapMemory(vk_device, buffer_memory_, 0, VK_WHOLE_SIZE, 0, &map) != VK_SUCCESS) {
    REXGPU_ERROR("[d3d][inline] ring: the buffer could not be mapped; inline draws stay on guest memory");
    return false;
  }
  map_ = static_cast<uint8_t*>(map);
  const VkDescriptorPoolSize size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, binding_count};
  VkDescriptorPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &size;
  if (dfn.vkCreateDescriptorPool(vk_device, &pool_info, nullptr, &pool_) != VK_SUCCESS) return false;
  VkDescriptorSetAllocateInfo alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  alloc.descriptorPool = pool_;
  alloc.descriptorSetCount = 1;
  alloc.pSetLayouts = &set_layout;
  VkDescriptorSet set = VK_NULL_HANDLE;
  if (dfn.vkAllocateDescriptorSets(vk_device, &alloc, &set) != VK_SUCCESS) return false;
  // Every binding of the shared memory's array is the ring: an address in the ring is below the first split, so the shaders read the first, the others are only valid.
  std::vector<VkDescriptorBufferInfo> infos(binding_count, VkDescriptorBufferInfo{buffer_, 0, total});
  VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = set;
  write.dstBinding = 0;
  write.descriptorCount = binding_count;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo = infos.data();
  dfn.vkUpdateDescriptorSets(vk_device, 1, &write, 0, nullptr);
  set_ = set;
  cursor_ = kFirstAddress;
  failed_ = false;
  return true;
}

void InlineVertexRing::Destroy() {
  if (!device_) return;
  const ui::vulkan::VulkanDevice::Functions& dfn = device_->functions();
  const VkDevice vk_device = device_->device();
  if (pool_ != VK_NULL_HANDLE) dfn.vkDestroyDescriptorPool(vk_device, pool_, nullptr);
  if (map_) dfn.vkUnmapMemory(vk_device, buffer_memory_);
  if (buffer_ != VK_NULL_HANDLE) dfn.vkDestroyBuffer(vk_device, buffer_, nullptr);
  if (buffer_memory_ != VK_NULL_HANDLE) dfn.vkFreeMemory(vk_device, buffer_memory_, nullptr);
  *this = InlineVertexRing();
}

bool InlineVertexRing::Write(const uint32_t* words, uint32_t bytes, uint64_t submission, uint64_t completed, uint32_t& address) {
  const uint32_t aligned = (bytes + 15u) & ~15u;
  if (aligned > kSliceBytes - kFirstAddress) {
    NoteFallback(kTooBig);
    return false;
  }
  if (cursor_ + aligned > (slice_ + 1) * kSliceBytes) {
    const uint32_t next = (slice_ + 1) % kSlices;
    if (slice_free_after_[next] > completed) {
      NoteFallback(kBusy);
      return false;
    }
    slice_ = next;
    cursor_ = next ? next * kSliceBytes : kFirstAddress;
    ++window_.slice_moves;
  }
  address = cursor_;
  cursor_ += aligned;
  slice_free_after_[slice_] = submission + 1;
  uint32_t* const out = reinterpret_cast<uint32_t*>(map_ + address);
  const uint32_t count = (bytes + 3u) >> 2;
  for (uint32_t i = 0; i < count; ++i) out[i] = std::byteswap(words[i]);
  ui::vulkan::util::FlushMappedMemoryRange(device_, buffer_memory_, memory_type_, address, memory_size_, VkDeviceSize(count) * 4);  // (returns at once on coherent memory)
  ++window_.draws;
  window_.bytes += bytes;
  if (!engaged_) {
    engaged_ = true;
    REXGPU_INFO("[d3d][inline] ring on: {} slices of {} KB, host coherent {}: inline vertices stay out of guest memory", kSlices, kSliceBytes >> 10,
                (device_->memory_types().host_coherent >> memory_type_) & 1);
  }
  return true;
}

void InlineVertexRing::FrameLog(uint64_t frame) {
  if (window_first_frame_ == UINT64_MAX) window_first_frame_ = frame;
  if (frame - window_first_frame_ < 600) return;
  const Window& w = window_;
  uint64_t fell = 0;
  for (uint64_t n : w.fallbacks) fell += n;
  if (w.draws || fell) {
    REXGPU_INFO("[d3d][inline] ring frames {}-{}: {} draws on the ring ({} KB, slice moves {}, cover rectangles not needed {}), {} on guest memory (no objects {}, bad record {}, "
                "game's memory {}, interlock {}, not owned-only {}, no shader {}, other fetch {}, pixel fetch {}, memexport {}, too big {}, slice busy {})",
                window_first_frame_, frame, w.draws, w.bytes >> 10, w.slice_moves, w.cover_skipped, fell, w.fallbacks[kNoObjects], w.fallbacks[kBadRecord], w.fallbacks[kGuestMemory],
                w.fallbacks[kInterlock], w.fallbacks[kNotOwnedOnly], w.fallbacks[kNoShader], w.fallbacks[kOtherFetch], w.fallbacks[kPixelFetch], w.fallbacks[kMemexport], w.fallbacks[kTooBig], w.fallbacks[kBusy]);
  }
  window_ = Window();
  window_first_frame_ = frame;
}

}  // namespace rex::graphics::vulkan
