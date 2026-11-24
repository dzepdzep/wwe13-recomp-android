// wwe2k14 patch (Session 95, plan item P): the host vertex ring of the inline draws (gpu_d3d_inline_vertex_ring, default false).
//
// An inline draw (BeginVertices / EndVertices: particles, sprites, post quads) carries its vertex bytes in its record. Staged in guest memory they cost a write watch, a
// RequestRange and a transfer upload per draw; once a submission has more inline draws than the staging has pages (Brock's pyro: ~1000 against 256) the upload cannot be
// hoisted, and its barrier ends the owned render pass: one draw per pass. On the ring the bytes never reach guest memory: the CPU writes them into a host-visible buffer at
// record time (a host write before the submit needs no barrier) and the draw reads that buffer through a descriptor set of the ring's own, bound in the place of the shared
// memory's (same layout), with slot 31's address being the offset in the ring. The shaders read their vertices as ever, so this needs no shader variant and no new pipeline:
// our own and the translator's take it alike. Owned-only frames only: OwnedRecord is what binds a draw's own sets.
//
// The ring is kSlices slices. The write cursor moves through a slice and on to the next; a slice is entered only when the submissions that read it and the one after (a
// margin) have completed, else the draw goes the old way, counted. Nothing waits. All of it is the command thread's: the only state a job would have to own is the cursor
// (a job would take its own slice).
#pragma once

#include <cstdint>

#include <rex/ui/vulkan/api.h>

namespace rex::ui::vulkan {
class VulkanDevice;
}

namespace rex::graphics::vulkan {

class InlineVertexRing {
 public:
  static constexpr uint32_t kSlices = 4;

  // Why a draw stayed on the guest memory path.
  enum Fallback : uint32_t {
    kNoObjects,    // the buffer or its descriptor set could not be made
    kBadRecord,    // the record's counts do not add up (the old path drops it)
    kGuestMemory,  // the vertices are in the game's own allocation
    kInterlock,    // the pixel shader interlock path (its set also holds the EDRAM)
    kNotOwnedOnly,  // not an owned-only frame
    kNoShader,     // no analysed vertex shader
    kOtherFetch,   // the vertex shader fetches another stream than slot 31's
    kPixelFetch,   // the pixel shader fetches vertices
    kMemexport,    // a shader writes memory
    kTooBig,       // more bytes than a slice
    kBusy,         // the next slice is still being read
    kFallbackCount,
  };

  bool Ensure(const ui::vulkan::VulkanDevice* device, VkDescriptorSetLayout set_layout, uint32_t binding_count);
  void Destroy();

  // The record's vertex words (host order as the queue carries them, the bytes stored swapped twice) into the ring. `address` = the byte offset in the ring, never 0.
  // submission: the one being recorded; completed: the last one finished.
  bool Write(const uint32_t* words, uint32_t bytes, uint64_t submission, uint64_t completed, uint32_t& address);

  VkDescriptorSet set() const { return set_; }
  VkBuffer buffer() const { return buffer_; }  // (Session 95: attribute draws bind it directly, at slot 31's address)
  const uint8_t* memory() const { return map_; }

  void NoteFallback(Fallback why) { ++window_.fallbacks[why]; }
  void NoteCoverSkipped() { ++window_.cover_skipped; }
  void FrameLog(uint64_t frame);  // the ~600-frame [d3d][inline] ring line

 private:
  const ui::vulkan::VulkanDevice* device_ = nullptr;
  bool failed_ = false;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory buffer_memory_ = VK_NULL_HANDLE;
  uint32_t memory_type_ = 0;
  VkDeviceSize memory_size_ = 0;
  uint8_t* map_ = nullptr;
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
  VkDescriptorSet set_ = VK_NULL_HANDLE;
  uint32_t slice_ = 0;
  uint32_t cursor_ = 0;
  uint64_t slice_free_after_[kSlices] = {};  // a slice may be written again once this submission has completed
  bool engaged_ = false;
  struct Window {
    uint64_t draws = 0, bytes = 0, slice_moves = 0, cover_skipped = 0;
    uint64_t fallbacks[kFallbackCount] = {};
  } window_;
  uint64_t window_first_frame_ = UINT64_MAX;
};

// The one ring (the command thread's).
InlineVertexRing& InlineRing();

}  // namespace rex::graphics::vulkan
