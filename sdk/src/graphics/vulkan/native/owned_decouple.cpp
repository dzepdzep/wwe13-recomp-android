// wwe2k14 patch (Session 94, idea 1 phase 1A): rendering decoupled from the simulation - the multiview plumbing with identical views (A/A).
// Docs: docs\native\d3d\S93_DECOUPLE_DESIGN.md sections 2 (top block) and 6.
//
// With gpu_d3d_decouple on, TRUE NATIVE's owned passes render K = 2 views: the owned colour / depth images have K layers and the owned
// render passes (and the post / truth render passes the owned draws' pipelines are made with - render pass compatibility includes the view
// mask) carry the view mask 0b11, so every draw goes to both layers. This file holds the decision (DecoupleViews), the render pass chain, the
// copy helper (a copy touching an owned image is one region per layer) and the proof (the A/A compare of the presented image's layers).
//
// Layer convention (review flip): LAYER 0 is the picture of the tick - exactly what the build without the switch draws. Every view of an owned
// image that is not the framebuffer's (the fill / write-back sources, the sampler views, the swap's) looks at that layer only; copies out of an
// owned image read it; copies into one write every layer. In A/A mode (bit 2 alone) layer 0 is drawn from the same constants, so the layers must stay equal. With bit 3
// (the blend) layer 1 differs by design where the blend is active, and our own post / side-chain pixel shaders read the view's layer (their multiview variants, Session 95
// tools/shaders/build_multiview_ps.py); with bit 5 the swap shows layer 1 only. The compare's differing count is then the proof that the layers are not equal.

#include <rex/graphics/vulkan/native/native_renderer.h>

#include <algorithm>
#include <cstring>

#include <rex/cvar.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/deferred_command_buffer.h>
#include <rex/logging.h>
#include <rex/ui/presenter.h>
#include <rex/ui/vulkan/util.h>
#include "owned_dead_clear.h"  // (M2dEdramLess)

#include "decouple_blend.h"

REXCVAR_DECLARE(int32_t, gpu_d3d_decouple);  // (vulkan_device.cpp: defined with the device, which enables the feature)
REXCVAR_DECLARE(bool, gpu_native_owned_only);
REXCVAR_DECLARE(bool, gpu_native_vs_expand);
REXCVAR_DECLARE(bool, vulkan_force_expand_rectangle_lists_in_vs);
REXCVAR_DECLARE(bool, vulkan_force_expand_point_sprites_in_vs);
REXCVAR_DECLARE(bool, vulkan_force_convert_quad_lists_to_triangle_lists);  // Session 95 (F5)

REXCVAR_DECLARE(int32_t, gpu_d3d_decouple_device);  // (vulkan_device.cpp: bit 0 multiview enabled, bit 1 multiviewGeometryShader)

namespace rex::graphics::vulkan::native {

uint32_t NativeRenderer::DecoupleViews() {
  if (decouple_ready_) return decouple_views_;
  decouple_ready_ = true;
  const int32_t mode = REXCVAR_GET(gpu_d3d_decouple);
  if (!mode) return 0;
  const bool multiview = (REXCVAR_GET(gpu_d3d_decouple_device) & 1) != 0;
  REXGPU_INFO("[native] decouple: multiview {}", multiview ? "yes" : "no");
  if (!multiview) {
    REXGPU_INFO("[native] decouple: the device has no multiview (or lacks Vulkan 1.1): gpu_d3d_decouple does nothing");
    return 0;
  }
  if (!(REXCVAR_GET(gpu_d3d_decouple_device) & 2) && !(REXCVAR_GET(gpu_native_vs_expand) && REXCVAR_GET(vulkan_force_expand_rectangle_lists_in_vs) && REXCVAR_GET(vulkan_force_expand_point_sprites_in_vs) && REXCVAR_GET(vulkan_force_convert_quad_lists_to_triangle_lists))) {  // (Session 95: with gpu_native_vs_expand no geometry shader is ever built, so the feature is not needed)
    // (the emulated pipelines' geometry shaders - rectangles, point sprites - are only valid in a multiview pass with this feature)
    REXGPU_INFO("[native] decouple: multiview geometry shaders: no - the pipelines' geometry shaders cannot run in a multiview pass: gpu_d3d_decouple does nothing");
    return 0;
  }
  if (!REXCVAR_GET(gpu_native_owned_only) || !OwnedChains()) {
    REXGPU_INFO("[native] decouple: needs TRUE NATIVE (gpu_native_owned_only with the owned frame chains): gpu_d3d_decouple does nothing");
    return 0;
  }
  if (!M2dEdramLess()) {  // (Session 94 review)
    REXGPU_INFO("[native] decouple: needs TRUE NATIVE without an EDRAM buffer (the truth and post render passes stay 1-layer): gpu_d3d_decouple does nothing");
    return 0;
  }
  decouple_views_ = 2;
  REXGPU_INFO("[native] decouple: on, {} views per owned pass (mode {}{}): owned images {} layers, render passes with view mask 0b11, the picture is view {}",
              decouple_views_, mode, (mode & 4) ? ((mode & 8) ? ", compare: the layers are compared each frame (they differ where the blend is active)" : ", A/A: both views use the same constants, the layers are compared each frame") : "", decouple_views_,
              0);
  return decouple_views_;
}

VkImageSubresourceRange NativeRenderer::OwnedLastLayer(VkImageAspectFlags aspects) {
  return ui::vulkan::util::InitializeSubresourceRange(aspects, 0, VK_REMAINING_MIP_LEVELS, 0, 1);  // (review flip: layer 0 is the picture)
}

bool NativeRenderer::DecoupleAA() { return DecoupleViews() && (REXCVAR_GET(gpu_d3d_decouple) & 4) != 0; }

void NativeRenderer::DecoupleRenderPass(VkRenderPassCreateInfo& info, VkRenderPassMultiviewCreateInfo& mv) {
  const uint32_t views = DecoupleViews();
  if (!views) return;
  // (one subpass; a mask per subpass and a correlation mask: the views see nearly the same scene - a tiler may share the binning)
  static const uint32_t mask = 3;
  mv = {VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO};
  mv.subpassCount = 1;
  mv.pViewMasks = &mask;
  mv.correlationMaskCount = 1;
  mv.pCorrelationMasks = &mask;
  mv.pNext = info.pNext;
  info.pNext = &mv;
}

void NativeRenderer::OwnedCopyImage(DeferredCommandBuffer& cb, VkImage src, bool src_owned, VkImageLayout src_layout, VkImage dst, bool dst_owned,
                                    VkImageLayout dst_layout, uint32_t count, const VkImageCopy* regions) {
  const uint32_t layers = OwnedLayers();
  if (layers == 1 || (!src_owned && !dst_owned)) {
    cb.CmdVkCopyImage(src, src_layout, dst, dst_layout, count, regions);
    return;
  }
  std::vector<VkImageCopy> out;
  out.reserve(size_t(count) * (dst_owned ? layers : 1u));
  for (uint32_t i = 0; i < count; ++i) {
    if (dst_owned) {
      for (uint32_t l = 0; l < layers; ++l) {
        VkImageCopy c = regions[i];
        if (src_owned) c.srcSubresource.baseArrayLayer = l;
        c.dstSubresource.baseArrayLayer = l;
        out.push_back(c);
      }
    } else {
      VkImageCopy c = regions[i];
      c.srcSubresource.baseArrayLayer = 0;  // (review flip: layer 0 is the picture)
      out.push_back(c);
    }
  }
  cb.CmdVkCopyImage(src, src_layout, dst, dst_layout, uint32_t(out.size()), out.data());
}

// ---- the A/A proof -----------------------------------------------------------------------------------------------------------------

void NativeRenderer::DecoupleCompareSwap(int16_t image, uint32_t width, uint32_t height) {
  if (DecoupleViews()) {
    decouple_swap_image_ = image;
    decouple_swap_w_ = width;
    decouple_swap_h_ = height;
  }
}

void NativeRenderer::DecoupleCompareReap() {
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  const uint64_t completed = command_processor_.GetCompletedSubmission();
  for (DecoupleCompare& c : decouple_cmp_) {
    if (!c.pending || c.submission > completed || !c.mapped) continue;
    c.pending = false;
    VkMappedMemoryRange range = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, c.memory, 0, VK_WHOLE_SIZE};
    dfn.vkInvalidateMappedMemoryRanges(device, 1, &range);
    const size_t layer_bytes = size_t(c.width) * c.height * c.bytes_per_pixel;
    const uint8_t* a = static_cast<const uint8_t*>(c.mapped);
    const uint8_t* b = a + layer_bytes;
    uint64_t differing = 0;
    const uint32_t rw = c.rect_w ? std::min(c.rect_w, c.width) : c.width, rh = c.rect_h ? std::min(c.rect_h, c.height) : c.height;  // (review: only the presented rectangle)
    for (uint32_t y = 0; y < rh; ++y) {
      for (uint32_t x = 0; x < rw; ++x) {
        const size_t p = size_t(y) * c.width + x;
        if (std::memcmp(a + p * c.bytes_per_pixel, b + p * c.bytes_per_pixel, c.bytes_per_pixel) != 0) ++differing;
      }
    }
    ++decouple_compared_;
    decouple_diff_total_ += differing;
    if (differing) ++decouple_diff_frames_;
    decouple_diff_max_ = std::max(decouple_diff_max_, differing);
    static uint32_t logged = 0;
    if (differing && logged < 8) {
      ++logged;
      if (REXCVAR_GET(gpu_d3d_decouple) & 8) {  // (Session 95: with the blend on the layers differ by design)
        REXGPU_INFO("[native] decouple compare: the two views of the presented image {}x{} differ in {} pixels (expected with the blend on; frame compared at submission {})", c.width,
                    c.height, differing, c.submission);
      } else {
        REXGPU_ERROR("[native] decouple A/A: the two views of the presented image {}x{} differ in {} pixels (frame compared at submission {})", c.width,
                     c.height, differing, c.submission);
      }
    }
  }
}

void NativeRenderer::DecoupleCompareDestroy() {
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  if (!vulkan_device) return;
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  for (DecoupleCompare& c : decouple_cmp_) {
    if (c.mapped) dfn.vkUnmapMemory(device, c.memory);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, c.buffer);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, c.memory);
    c = DecoupleCompare();
  }
}

bool NativeRenderer::DecoupleTwoStep() {
  const int32_t mode = REXCVAR_GET(gpu_d3d_decouple);
  return DecoupleViews() && (mode & 8) && !(mode & 16) && !DecoupleAA() && DecoupleBlendTickWasActive();
}

// A 2D view of layer 1 (the in-between picture) of the image `last_layer_view` looks at (the same format and components). Kept in owned_views_ under a type no other view has (1D), so it is destroyed with them.
VkImageView NativeRenderer::OwnedLayer1View(VkImageView last_layer_view) {
  if (!DecoupleViews() || last_layer_view == VK_NULL_HANDLE) return VK_NULL_HANDLE;
  const OwnedViewEntry* base = nullptr;
  for (const OwnedViewEntry& v : owned_views_) {
    if (v.view == last_layer_view) {
      base = &v;
      break;
    }
  }
  if (!base || base->type != VK_IMAGE_VIEW_TYPE_2D) return VK_NULL_HANDLE;
  bool layered = false;  // (review: only an owned image made with OwnedLayers() layers; page images and others have one)
  for (const OwnedImage& o : owned_images_) {
    if (o.image == base->image && o.array_view != VK_NULL_HANDLE) layered = true;
  }
  if (!layered) return VK_NULL_HANDLE;
  for (const OwnedViewEntry& v : owned_views_) {
    if (v.type == VK_IMAGE_VIEW_TYPE_1D && v.image == base->image && v.format == base->format && !std::memcmp(&v.components, &base->components, sizeof(v.components))) return v.view;
  }
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view_info.image = base->image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = base->format;
  view_info.components = base->components;
  view_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 1, 1);  // (layer 1: the in-between picture)
  VkImageView view = VK_NULL_HANDLE;
  if (vulkan_device->functions().vkCreateImageView(vulkan_device->device(), &view_info, nullptr, &view) != VK_SUCCESS) return VK_NULL_HANDLE;
  const OwnedViewEntry entry = {base->image, base->components, view, base->format, VK_IMAGE_VIEW_TYPE_1D};  // (base may move when the vector grows: copied first)
  owned_views_.push_back(entry);
  return view;
}

void NativeRenderer::DecoupleFrameEnd() {
  const uint32_t views = DecoupleViews();
  if (!views) return;
  ++decouple_frames_;
  DecoupleBlendFrameEnd();  // (phase 1B: this tick's draw records become the previous tick's)
  const int16_t image = decouple_swap_image_;
  decouple_swap_image_ = -1;
  if (DecoupleAA()) {
    DecoupleCompareReap();
    // (every 4th frame: the compare is a diagnostic - a copy of both layers and a CPU pass over them)
    if (image >= 0 && size_t(image) < owned_images_.size() && owned_images_[image].width && owned_images_[image].image != VK_NULL_HANDLE &&
        (decouple_frames_ & 3) == 1) {
      OwnedImage& o = owned_images_[image];
      DecoupleCompare& c = decouple_cmp_[decouple_cmp_next_ & 1];
      const uint32_t bpp = o.kind == 2 ? 8u : 4u;
      const VkDeviceSize need = VkDeviceSize(o.width) * o.height * bpp * views;
      const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
      const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
      const VkDevice device = vulkan_device->device();
      if (!c.pending && (c.size < need || c.buffer == VK_NULL_HANDLE)) {
        if (c.mapped) dfn.vkUnmapMemory(device, c.memory);
        ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, c.buffer);
        ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, c.memory);
        c.mapped = nullptr;
        VkDeviceSize memory_size = 0;
        if (ui::vulkan::util::CreateDedicatedAllocationBuffer(vulkan_device, need, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                              ui::vulkan::util::MemoryPurpose::kReadback, c.buffer, c.memory, nullptr, &memory_size)) {
          c.size = need;
          if (dfn.vkMapMemory(device, c.memory, 0, VK_WHOLE_SIZE, 0, &c.mapped) != VK_SUCCESS) c.mapped = nullptr;
        } else {
          c.size = 0;
        }
      }
      if (!c.pending && c.buffer != VK_NULL_HANDLE && c.mapped && c.size >= need) {
        ++decouple_cmp_next_;
        OwnedEndPass();
        EndTruthPass();
        command_processor_.SubmitBarriers(true);
        DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
        const bool readable = o.readable;
        const VkPipelineStageFlags stages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                            VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = o.image;
        b.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);
        b.srcAccessMask = readable ? VkAccessFlags(0) : VkAccessFlags(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.oldLayout = readable ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        cb.CmdVkPipelineBarrier(stages, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkBufferImageCopy regions[2] = {};
        for (uint32_t l = 0; l < 2; ++l) {
          regions[l].bufferOffset = VkDeviceSize(l) * o.width * o.height * bpp;
          regions[l].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, l, 1};
          regions[l].imageExtent = {o.width, o.height, 1};
        }
        cb.CmdVkCopyImageToBuffer(o.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, c.buffer, 2, regions);
        b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.dstAccessMask = readable ? VkAccessFlags(VK_ACCESS_SHADER_READ_BIT)
                                   : VkAccessFlags(VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.newLayout = readable ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkMemoryBarrier to_host = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
        cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, stages | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0, nullptr, 1, &b);
        c.width = o.width;
        c.height = o.height;
        c.rect_w = decouple_swap_w_;
        c.rect_h = decouple_swap_h_;
        c.bytes_per_pixel = bpp;
        c.submission = command_processor_.GetCurrentSubmission();
        c.pending = true;
        OwnedTouch(image);
      }
    }
  }
  if (decouple_frames_ % 600 == 0) {
    const std::string blend_line = DecoupleBlendSummary(decouple_frames_, views, decouple_presents_, 0);
    if (!blend_line.empty()) {
      REXGPU_INFO("[native] {}", blend_line);
      decouple_presents_ = 0;  // (a window of its own)
    }
    {  // Session 95 (F2): the presenter's two-step counters (timer vs forced), which had no reader
      static ui::Presenter::DecoupleStats last;
      const ui::Presenter::DecoupleStats now = ui::Presenter::GetDecoupleStats();
      if (now.step0 || now.step1 || (REXCVAR_GET(gpu_d3d_decouple) & 32)) {
        REXGPU_INFO("[native] decouple present window: step0 {} step1 {} (timer {} forced {}) last tick {} us, last delay {} us{}", now.step0 - last.step0, now.step1 - last.step1,
                    now.delayed_timer - last.delayed_timer, now.delayed_forced - last.delayed_forced, now.last_tick_us, now.last_delay_us,
                    (REXCVAR_GET(gpu_d3d_decouple) & 32) ? " [view-1 proof mode: one present a tick, only view 1 shown]" : "");
      }
      last = now;
    }
    if (DecoupleAA()) {
      REXGPU_INFO("[native] decouple A/A: frames {}, views {}, differing pixels max {} (compared {} frames, {} with a difference, {} pixels in all)",
                  decouple_frames_, views, decouple_diff_max_, decouple_compared_, decouple_diff_frames_, decouple_diff_total_);
    } else {
      REXGPU_INFO("[native] decouple: frames {}, views {} (no compare: A/A is off)", decouple_frames_, views);
    }
  }
}

}  // namespace rex::graphics::vulkan::native
