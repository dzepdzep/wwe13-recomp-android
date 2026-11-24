// wwe2k14 patch (Session 82, the owned match frame, P0): the extension point for the copies of an owned pass the base
// cases (OwnedTakeResolve: a whole copy served from our image or filled into a 16-bit page image, a rectangle into a
// page of ours) do not take - page images of other formats, depth pages, dead copies, write-back. A later piece defines
// it here, in this file. Docs: docs\native\OWNED_FRAME_DESIGN.md section 3.4.
//
// Session 90 (Q, docs\native\d3d\S90_Q_DESIGN.md): the write-back. While the CPU is to read the bytes of a resolve (a read-back
// window of the app: the Create-a-Superstar render, the attire builds), a copy OwnedTakeResolve serves from our image writes no
// guest memory - the CPU would keep the old bytes. Until now such a copy was not served: the pass was replayed on the truth path
// and the game's own copy read it back. Now the copy is served exactly as outside a window and, in addition, written: the render
// target cache's destination block (render_target_cache.cpp, Resolve) calls OwnedWriteBackRun first in its direct chain, which runs
// the image -> memory resolve shaders of the truth path with OUR IMAGE as the source (kNativeResolveComputeShader / the scaled one,
// kNativeResolveScaledFullComputeShader, kNativeResolveFull1xComputeShader and their depth variants), into the same destination
// (the shared memory at 1x, the scaled resolve buffer at a scale); the cache then marks the range resolved and gives it to the
// read-back path as for any copy. No truth image, tile ownership, staging store or replay is involved (hard rule 6: the memory
// layout the CPU reads is the guest-memory contract, not EDRAM semantics).

#include <rex/graphics/vulkan/native/native_renderer.h>

#include <algorithm>
#include <cstring>

#include <rex/cvar.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/deferred_command_buffer.h>
#include <rex/logging.h>
#include <rex/ui/vulkan/util.h>

REXCVAR_DEFINE_INT32(gpu_native_owned_writeback_verify, 0, "GPU/Native",
                     "Session 90 (Q, diagnosis): 1 = the 8888 write-back forces alpha 255 (is the garbled attire an alpha difference?)");

namespace rex::graphics::vulkan::native {

namespace {
constexpr VkPipelineStageFlags kWbShaderStages =
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
constexpr VkPipelineStageFlags kWbDepthStages =
    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

VkImageMemoryBarrier WbBarrier(VkImage image, VkImageAspectFlags aspects, VkAccessFlags src, VkAccessFlags dst,
                               VkImageLayout from, VkImageLayout to) {
  VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = image;
  b.srcAccessMask = src;
  b.dstAccessMask = dst;
  b.oldLayout = from;
  b.newLayout = to;
  b.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(aspects);
  return b;
}

const char* const kWbPathNames[5] = {"8888 1x", "8888 scaled", "full 1x", "full scaled", "depth"};
}  // namespace

bool NativeRenderer::OwnedCopyExtra(OwnedCopyContext& ctx) {
  // (No other copy yet.)
  (void)ctx;
  return false;
}

void NativeRenderer::OwnedWbRefused(const std::string& why) {
  auto& v = owned_frame_stats_.wb_refused;
  auto it = std::find_if(v.begin(), v.end(), [&](const std::pair<std::string, uint32_t>& e) { return e.first == why; });
  if (it != v.end()) {
    ++it->second;
  } else if (v.size() < 32) {
    v.emplace_back(why, 1u);
  }
  static uint32_t logged = 0;
  if (logged < 24 || owned_frame_logging_) {
    ++logged;
    REXGPU_INFO("[native][owned] frame {}: read-back write-back not possible: {}", frame_, why);
  }
}

VkDescriptorSet NativeRenderer::OwnedWbPostSet(OwnedImage& o) {
  // The 8888 resolve shaders' source set (post-writeback layout: binding 0 the EDRAM buffer - never read inside the image -, 1 the colour,
  // 2 / 3 the coverage and depth stand-ins), one per image, rewritten only when the staging buffer changed.
  if (o.wb_set != VK_NULL_HANDLE && o.wb_set_buffer != truth_staging_) {
    owned_wb_spare_sets_.push_back({o.wb_set, command_processor_.GetCurrentSubmission()});
    o.wb_set = VK_NULL_HANDLE;
  }
  if (o.wb_set != VK_NULL_HANDLE) return o.wb_set;
  if (!EnsureResolveObjects() || !EnsureTruthStaging() || !EnsureTruthDummy() || post_writeback_set_layout_ == VK_NULL_HANDLE) {
    return VK_NULL_HANDLE;
  }
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  VkDescriptorSet set = VK_NULL_HANDLE;
  const uint64_t completed = command_processor_.GetCompletedSubmission();
  for (size_t i = 0; i < owned_wb_spare_sets_.size(); ++i) {
    if (owned_wb_spare_sets_[i].submission > completed) continue;
    set = owned_wb_spare_sets_[i].set;
    owned_wb_spare_sets_.erase(owned_wb_spare_sets_.begin() + i);
    break;
  }
  if (set == VK_NULL_HANDLE && OwnedAllocateSet(post_writeback_set_layout_, set) != VK_SUCCESS) return VK_NULL_HANDLE;
  VkDescriptorBufferInfo buffer_info = {truth_staging_, 0, VK_WHOLE_SIZE};
  VkDescriptorImageInfo color_info = {sampler_nearest(), o.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkDescriptorImageInfo stencil_info = {sampler_nearest(), truth_dummy_stencil_view_, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
  VkDescriptorImageInfo depth_info = {sampler_nearest(), truth_dummy_depth_view_, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
  VkWriteDescriptorSet writes[4] = {};
  for (uint32_t i = 0; i < 4; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = i ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  }
  writes[0].pBufferInfo = &buffer_info;
  writes[1].pImageInfo = &color_info;
  writes[2].pImageInfo = &stencil_info;
  writes[3].pImageInfo = &depth_info;
  dfn.vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
  o.wb_set = set;
  o.wb_set_buffer = truth_staging_;
  return set;
}

VkDescriptorSet NativeRenderer::OwnedWbColorSet(OwnedImage& o) {
  // The full / scaled-full shaders' source set (truth-store layout: binding 0 the staging buffer, 1 the colour) - the image's fill set.
  if (o.fill_set != VK_NULL_HANDLE && o.fill_set_buffer != truth_staging_) {
    OwnedSpareFillSet(o.fill_set);
    o.fill_set = VK_NULL_HANDLE;
  }
  if (o.fill_set != VK_NULL_HANDLE) return o.fill_set;
  if (!EnsureTruthStaging()) return VK_NULL_HANDLE;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  o.fill_set_buffer = truth_staging_;
  if (OwnedAcquireFillSet(o.fill_set) != VK_SUCCESS) {
    o.fill_set = VK_NULL_HANDLE;
    return VK_NULL_HANDLE;
  }
  VkDescriptorBufferInfo buffer_info = {truth_staging_, 0, VK_WHOLE_SIZE};
  VkDescriptorImageInfo image_info = {sampler_nearest(), o.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkWriteDescriptorSet writes[2] = {};
  for (uint32_t i = 0; i < 2; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = o.fill_set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
  }
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &buffer_info;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  writes[1].pImageInfo = &image_info;
  dfn.vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
  return o.fill_set;
}

VkDescriptorSet NativeRenderer::OwnedWbDepthSet(OwnedImage& o) {
  // The depth shaders' source set (truth-store layout: binding 0 the staging buffer, 1 the depth, 2 the stencil) - the image's depth fill set
  // (the views as OwnedFillDepth makes them).
  if (o.depth == VK_NULL_HANDLE || !EnsureTruthStaging()) return VK_NULL_HANDLE;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  if (o.depth_sample_view == VK_NULL_HANDLE) {
    VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = o.depth;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = post_ds_format_;
    view_info.subresourceRange = OwnedLastLayer(VK_IMAGE_ASPECT_DEPTH_BIT);  // (Session 94: the picture's layer)
    bool ok = dfn.vkCreateImageView(device, &view_info, nullptr, &o.depth_sample_view) == VK_SUCCESS;
    view_info.subresourceRange = OwnedLastLayer(VK_IMAGE_ASPECT_STENCIL_BIT);
    ok = ok && dfn.vkCreateImageView(device, &view_info, nullptr, &o.depth_stencil_view) == VK_SUCCESS;
    if (!ok) {
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device, o.depth_sample_view);
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device, o.depth_stencil_view);
      return VK_NULL_HANDLE;
    }
  }
  if (o.depth_fill_set != VK_NULL_HANDLE && o.depth_fill_set_buffer != truth_staging_) {
    OwnedSpareFillSet(o.depth_fill_set);
    o.depth_fill_set = VK_NULL_HANDLE;
  }
  if (o.depth_fill_set == VK_NULL_HANDLE) {
    o.depth_fill_set_buffer = truth_staging_;
    if (OwnedAcquireFillSet(o.depth_fill_set) != VK_SUCCESS) {
      o.depth_fill_set = VK_NULL_HANDLE;
      return VK_NULL_HANDLE;
    }
    VkDescriptorBufferInfo buffer_info = {truth_staging_, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo depth_info = {sampler_nearest(), o.depth_sample_view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo stencil_info = {sampler_nearest(), o.depth_stencil_view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[3] = {};
    for (uint32_t i = 0; i < 3; ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = o.depth_fill_set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = i ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    writes[0].pBufferInfo = &buffer_info;
    writes[1].pImageInfo = &depth_info;
    writes[2].pImageInfo = &stencil_info;
    dfn.vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
  }
  return o.depth_fill_set;
}

bool NativeRenderer::OwnedWbPlan(const draw_util::ResolveInfo& info, uint32_t sx, uint32_t sy, uint32_t w, uint32_t h, uint32_t gap,
                                 bool front, std::string& why) {
  // The write-back of a colour copy of the pending pass (OwnedTakeResolve: the shape is a plain copy of a rectangle of the pass's box, the
  // source format the pass's class's, no exponent bias / array): which writer, and that it can run (its pipelines exist). Nothing is recorded.
  OwnedPass& p = owned_pending_;
  const auto no = [&](const char* text) {
    why = text;
    return false;
  };
  if (!p.valid || p.image < 0 || size_t(p.image) >= owned_images_.size() || !owned_images_[p.image].width) return no("no source image");
  if (REXCVAR_GET(gpu_native_owned_writeback_verify) == 2) return no("verify mode 2: the old replay");
  const OwnedImage& o = owned_images_[p.image];
  if (o.kind != p.kind || (p.kind != 1 && p.kind != 2)) return no("source class");
  if (p.half) return no("half-size pass");
  if (uint32_t(info.copy_dest_coordinate_info.copy_sample_select) != 0) return no("sample select");
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const bool scaled = scale_x != 1 || scale_y != 1;
  if (!w || !h || (sx + w) * scale_x > o.width || (sy + h) * scale_y > o.height) return no("rectangle outside the image");
  const reg::RB_COPY_DEST_INFO d = info.copy_dest_info;
  uint32_t path, mode = 0;
  if (p.kind == 2) {
    path = 3;
    mode = d.copy_dest_format == xenos::ColorFormat::k_16_16 ? 1u : 3u;
    if (d.copy_dest_format != xenos::ColorFormat::k_16_16 && d.copy_dest_format != xenos::ColorFormat::k_16_16_16_16) {
      return no("16-bit source, other destination format");
    }
  } else if (d.copy_dest_format == xenos::ColorFormat::k_8_8_8_8) {
    path = scaled ? 1u : 0u;
  } else if (d.copy_dest_format == xenos::ColorFormat::k_8 || d.copy_dest_format == xenos::ColorFormat::k_4_4_4_4) {
    mode = (d.copy_dest_format == xenos::ColorFormat::k_8 ? 4u : 5u) | (d.copy_dest_swap ? 16u : 0u);
    path = scaled ? 3u : 2u;
  } else {
    return no("RGBA8 source, destination format without a writer");
  }
  if (!EnsureTruthStaging()) return no("no staging buffer");
  switch (path) {
    case 0:
      if (!EnsureResolveObjects() || !EnsureTruthDummy()) return no("no 8888 resolve pipeline");
      break;
    case 1:
      if (!EnsureResolveObjects() || !EnsureTruthDummy() || !EnsureResolveScaledPipeline()) return no("no scaled 8888 resolve pipeline");
      break;
    case 2:
      if (!EnsureResolveScaledFullPipeline() || !EnsureResolveFull1xPipeline()) return no("no full 1x resolve pipeline");
      break;
    default:
      if (!EnsureResolveScaledFullPipeline()) return no("no scaled-full resolve pipeline");
      break;
  }
  owned_wb_ = OwnedWriteBack();
  owned_wb_.path = path;
  owned_wb_.mode = mode;
  owned_wb_.image = p.image;
  owned_wb_.sx = sx;
  owned_wb_.sy = sy;
  owned_wb_.w = w;
  owned_wb_.h = h;
  owned_wb_.gap = gap;
  owned_wb_.front = front;
  owned_wb_.page = (info.copy_dest_base >> 12) & 0x1FFFF;
  owned_wb_.dest_format = uint32_t(d.copy_dest_format);
  return true;
}

bool NativeRenderer::OwnedWbPlanFill(const draw_util::ResolveInfo& info, std::string& why, const float* colour) {
  // Session 92, M2c (design 2.5, "filled"): a colour copy whose source surface has no image of ours (no pass drew it, no record holds it) writes a DEFINED value
  // into its destination instead of being dropped - transparent black, what a surface we never drew starts as (hard rule 6: not another surface's EDRAM
  // leftovers). Done with the machinery of the write-back: a cleared image of ours of the copy's size is the source of the 8888 writer (path 0 / 1), which the
  // render target cache's destination block runs (OwnedWriteBackRun) into the shared memory / the scaled resolve buffer exactly as for a served copy.
  // Only a k_8_8_8_8 destination of a 1x colour copy; any other shape is named and stays a drop. Nothing is recorded but the image's clear.
  const auto no = [&](const char* text) {
    why = text;
    return false;
  };
  if (info.IsCopyingDepth()) return no("fill: a depth source");
  const reg::RB_COPY_DEST_INFO d = info.copy_dest_info;
  if (d.copy_dest_format != xenos::ColorFormat::k_8_8_8_8) return no("fill: destination format is not k_8_8_8_8");
  if (d.copy_dest_exp_bias || d.copy_dest_array || uint32_t(info.copy_dest_coordinate_info.copy_sample_select) != 0) return no("fill: exp bias / array / sample select");
  if (info.color_edram_info.msaa_samples != xenos::MsaaSamples::k1X) return no("fill: a multisampled source");
  const uint32_t w = uint32_t(info.coordinate_info.width_div_8) * 8, h = uint32_t(info.height_div_8) * 8;
  if (!w || !h || w > 4096 || h > 4096) return no("fill: size");
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const bool scaled = scale_x != 1 || scale_y != 1;
  if (!EnsureTruthStaging() || !EnsureResolveObjects() || !EnsureTruthDummy()) return no("fill: no 8888 resolve pipeline");
  if (scaled && !EnsureResolveScaledPipeline()) return no("fill: no scaled 8888 resolve pipeline");
  const int16_t index = OwnedImageFor(w * scale_x, h * scale_y, 1);
  if (index < 0 || size_t(index) >= owned_images_.size() || !owned_images_[index].width) return no("fill: no image in the budget");
  OwnedImage& o = owned_images_[index];
  OwnedEndPass();
  EndTruthPass();
  command_processor_.SubmitBarriers(true);
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  // (review F5: the image may be new or left by a declined pass start - UNDEFINED layout, no transition made; the clear overwrites it whole, so the old layout is never read.)
  VkImageMemoryBarrier b = WbBarrier(o.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                     VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                         VK_ACCESS_SHADER_READ_BIT,
                                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  cb.CmdVkPipelineBarrier(kWbShaderStages | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                          nullptr, 1, &b);
  VkClearColorValue zero = {};
  if (colour) {  // (Session 92, M4f-B: a pended clear's value, rounded as the 8888 target stores it - OwnedBeginClearedPass's rounding)
    for (uint32_t i = 0; i < 4; ++i) {
      const float s = std::min(std::max(colour[i], 0.0f), 1.0f) * 255.0f;
      zero.float32[i] = std::trunc(s + 0.5f) / 255.0f;
    }
  }
  const VkImageSubresourceRange range = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);
  cb.CmdVkClearColorImage(o.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
  b = WbBarrier(o.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                          nullptr, 1, &b);
  o.readable = false;
  OwnedTouch(index);
  owned_wb_ = OwnedWriteBack();
  owned_wb_.path = scaled ? 1u : 0u;
  owned_wb_.mode = 0;
  owned_wb_.image = index;
  owned_wb_.sx = owned_wb_.sy = 0;
  owned_wb_.w = w;
  owned_wb_.h = h;
  owned_wb_.gap = 0;
  owned_wb_.front = false;
  owned_wb_.page = (info.copy_dest_base >> 12) & 0x1FFFF;
  owned_wb_.dest_format = uint32_t(d.copy_dest_format);
  return true;
}

bool NativeRenderer::OwnedWbPlanComposite(const draw_util::ResolveInfo& info, uint32_t w, uint32_t h, std::string& why) {
  // Session 92, M2d (Part 4, docs\native\d3d\S92_M1A_BOX_READ.md B): a colour copy larger than the box of the pass it reads (the 512x512 pitch-7 copies of a 512x256
  // pass that began from a held clear). On the 360 the rows past the pass's box are other surfaces' EDRAM tiles read as colour (aliasing - not modelled); here they are a
  // DEFINED fill, transparent black. The pass's own image is copied into the top of a zero-cleared image of the copy's size, which is the source of the 8888 writer
  // (path 0 / 1) exactly as OwnedWbPlanFill's zero image is. The pass stays pending and untouched (its image goes back to the layout it had).
  OwnedPass& p = owned_pending_;
  const auto no = [&](const char* text) {
    why = text;
    return false;
  };
  if (!p.valid || p.image < 0 || size_t(p.image) >= owned_images_.size() || !owned_images_[p.image].width) return no("composite: no source image");
  if (p.kind != 1 || p.half) return no("composite: source class / half-size pass");
  if (info.IsCopyingDepth()) return no("composite: a depth source");
  const reg::RB_COPY_DEST_INFO d = info.copy_dest_info;
  if (d.copy_dest_format != xenos::ColorFormat::k_8_8_8_8) return no("composite: destination format is not k_8_8_8_8");
  if (d.copy_dest_exp_bias || d.copy_dest_array || uint32_t(info.copy_dest_coordinate_info.copy_sample_select) != 0) return no("composite: exp bias / array / sample select");
  if (info.color_edram_info.msaa_samples != xenos::MsaaSamples::k1X) return no("composite: a multisampled source");
  if (!w || !h || w > 4096 || h > 4096 || w < p.width || h <= p.height) return no("composite: the copy is not larger than the pass box");
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const bool scaled = scale_x != 1 || scale_y != 1;
  if (!EnsureTruthStaging() || !EnsureResolveObjects() || !EnsureTruthDummy()) return no("composite: no 8888 resolve pipeline");
  if (scaled && !EnsureResolveScaledPipeline()) return no("composite: no scaled 8888 resolve pipeline");
  const int16_t src_index = p.image;
  const int16_t index = OwnedImageFor(w * scale_x, h * scale_y, 1);  // (may grow the vector: references are taken after)
  if (index < 0 || size_t(index) >= owned_images_.size() || !owned_images_[index].width || index == src_index) return no("composite: no image in the budget");
  OwnedImage& o = owned_images_[index];
  OwnedImage& s = owned_images_[src_index];
  const uint32_t copy_w = std::min({p.host_width, s.width, o.width});
  const uint32_t copy_h = std::min({p.host_height, s.height, o.height});
  if (!copy_w || !copy_h) return no("composite: empty box");
  OwnedEndPass();
  EndTruthPass();
  command_processor_.SubmitBarriers(true);
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const bool src_readable = s.readable;
  constexpr VkAccessFlags kAnyColour = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
  VkImageMemoryBarrier b[2];
  b[0] = WbBarrier(s.image, VK_IMAGE_ASPECT_COLOR_BIT, src_readable ? VkAccessFlags(0) : VkAccessFlags(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT), VK_ACCESS_TRANSFER_READ_BIT,
                   src_readable ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  // (the composite may be new or left by a declined start - UNDEFINED, the clear below overwrites it whole)
  b[1] = WbBarrier(o.image, VK_IMAGE_ASPECT_COLOR_BIT, kAnyColour, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  cb.CmdVkPipelineBarrier(kWbShaderStages | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                          nullptr, 2, b);
  const VkClearColorValue zero = {};
  const VkImageSubresourceRange range = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);
  cb.CmdVkClearColorImage(o.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
  VkMemoryBarrier clear_done = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &clear_done, 0, nullptr, 0, nullptr);
  VkImageCopy region = {};
  region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.extent = {copy_w, copy_h, 1};
  OwnedCopyImage(cb, s.image, true, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, o.image, true, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);  // (Session 94: per layer)
  b[0] = WbBarrier(s.image, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                   src_readable ? VkAccessFlags(VK_ACCESS_SHADER_READ_BIT) : VkAccessFlags(VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT),
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, src_readable ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  b[1] = WbBarrier(o.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                          kWbShaderStages | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 2, b);
  o.readable = false;
  OwnedTouch(index);
  OwnedTouch(src_index);
  owned_wb_ = OwnedWriteBack();
  owned_wb_.path = scaled ? 1u : 0u;
  owned_wb_.mode = 0;
  owned_wb_.image = index;
  owned_wb_.sx = owned_wb_.sy = 0;
  owned_wb_.w = w;
  owned_wb_.h = h;
  owned_wb_.gap = 0;
  owned_wb_.front = false;
  owned_wb_.page = (info.copy_dest_base >> 12) & 0x1FFFF;
  owned_wb_.dest_format = uint32_t(d.copy_dest_format);
  return true;
}

bool NativeRenderer::OwnedWbPlanDepth(const draw_util::ResolveInfo& info, std::string& why) {
  // The depth copy OwnedDepthCopy serves (k_24_8_FLOAT, the whole box into a texture's origin): the FULL1X_DEPTH / FULL_DEPTH writers with the
  // pass image's depth + stencil as the source.
  OwnedPass& p = owned_pending_;
  const auto no = [&](const char* text) {
    why = text;
    return false;
  };
  if (!p.valid || !p.depth || p.kind != 1 || p.depth_slot < 0 || size_t(p.depth_slot) >= OwnedSlotCount() || p.image < 0 ||
      size_t(p.image) >= owned_images_.size() || !owned_images_[p.image].width) {
    return no("the pass keeps no depth");
  }
  if (owned_images_[p.image].depth == VK_NULL_HANDLE) return no("no depth image");
  const OwnedSurface t = OwnedSlotKey(p.depth_slot);
  const reg::RB_COPY_DEST_INFO d = info.copy_dest_info;
  if (t.kind != 3 || !(t.format & 1u)) return no("not a 20e4 depth surface");
  if (d.copy_dest_swap || d.copy_dest_array || d.copy_dest_exp_bias || uint32_t(info.copy_dest_coordinate_info.copy_sample_select) != 0 ||
      uint32_t(d.copy_dest_format) != uint32_t(xenos::TextureFormat::k_24_8_FLOAT)) {
    return no("depth: destination shape");
  }
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const bool scaled = scale_x != 1 || scale_y != 1;
  if (!EnsureTruthStaging() || !EnsureResolveScaledFullPipeline() || !EnsureResolveDepthPipeline(scaled)) return no("no depth resolve pipeline");
  owned_wb_ = OwnedWriteBack();
  owned_wb_.path = 4;
  owned_wb_.mode = (t.format & 1u) | 2u;  // (20e4 depth; the stencil the image's)
  owned_wb_.image = p.image;
  owned_wb_.w = p.width;
  owned_wb_.h = p.height;
  owned_wb_.gap = scaled && info.depth_edram_info.fill_half_pixel_offset ? (scale_x / 2) | ((scale_y / 2) << 8) : 0u;
  owned_wb_.page = (info.copy_dest_base >> 12) & 0x1FFFF;
  owned_wb_.dest_format = uint32_t(d.copy_dest_format);
  return true;
}

bool NativeRenderer::OwnedWriteBackRun(const draw_util::ResolveInfo& info, const draw_util::ResolveCopyShaderConstants& constants,
                                       VkDescriptorSet dest_set, uint32_t group_count_x, uint32_t group_count_y) {
  // Called by the render target cache's destination block for a copy OwnedTakeResolve served and armed. Records, outside any render pass,
  // the image -> memory dispatch; false = nothing was written (counted, logged: the CPU keeps the old bytes).
  const OwnedWriteBack wb = owned_wb_;
  owned_wb_.armed = false;
  if (!wb.armed) return false;
  if (dest_set == VK_NULL_HANDLE) {
    OwnedWbRefused("armed, no destination set");
    return false;
  }
  if (wb.image < 0 || size_t(wb.image) >= owned_images_.size() || !owned_images_[wb.image].width) {
    OwnedWbRefused("armed, the source image is gone");
    return false;
  }
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const bool scaled = scale_x != 1 || scale_y != 1;
  OwnedEndPass();
  EndTruthPass();
  // (Review #1: texture-only records this copy overlaps go into memory BEFORE our bytes, as every stock direct writer does; a later call no-ops.)
  TruthResolveOverlaps(info, nullptr);
  OwnedTouch(wb.image);
  OwnedGraphNoteRead(wb.image);  // (Session 95, round 7: the dispatch samples the image)
  OwnedImage& o = owned_images_[wb.image];
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const uint32_t dest_coordinate_info = constants.dest_relative.dest_coordinate_info.packed;
  const uint32_t dest_info = constants.dest_relative.dest_info.value;
  const uint32_t dest_x0 = ((dest_coordinate_info >> 20) & 0xFu) * 8u;
  const uint32_t dest_y0 = ((dest_coordinate_info >> 24) & 0xFu) * 8u;
  const uint32_t dest_pitch = (dest_coordinate_info & 1023u) << 5;
  const uint32_t endian = dest_info & 7u;
  const uint32_t timer = OwnedTimerBegin("wb_copy", wb.w * scale_x, wb.h * scale_y);
  bool was_readable = o.readable;
  bool depth_was_read = o.depth_read;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkDescriptorSet source_set = VK_NULL_HANDLE;
  uint32_t pc[12] = {};
  uint32_t pc_words = 10;
  uint32_t dispatch_x = group_count_x, dispatch_y = group_count_y;
  if (wb.path == 0 || wb.path == 1) {
    layout = resolve_layout_;
    pipeline = wb.path == 1 ? resolve_scaled_pipeline_ : resolve_pipeline_;
    source_set = OwnedWbPostSet(o);
    pc[0] = constants.dest_relative.edram_info.packed;
    pc[1] = constants.dest_relative.coordinate_info.packed;
    pc[2] = dest_info;
    pc[3] = dest_coordinate_info;
    pc[4] = constants.dest_base;
    pc[5] = wb.sx * scale_x;
    pc[6] = wb.sy * scale_y;
    pc[7] = o.width;
    pc[8] = o.height;
    pc[9] = 1u | 0x200u | (REXCVAR_GET(gpu_native_owned_writeback_verify) == 1 ? 0x800u : 0u);  // RGBA8 (the packed values), every pixel from the image
    pc_words = 12;
  } else if (wb.path == 2) {
    layout = resolve_scaled_full_layout_;
    pipeline = resolve_full_1x_pipeline_;
    source_set = OwnedWbColorSet(o);
    if (constants.dest_base & 3u) {
      OwnedWbRefused("full 1x: destination base not dword aligned");
      OwnedTimerEnd(timer);
      return false;
    }
    pc[0] = wb.sx;
    pc[1] = wb.sy;
    pc[2] = wb.w;
    pc[3] = wb.h;
    pc[4] = wb.mode;
    pc[5] = constants.dest_base;
    pc[6] = dest_x0;
    pc[7] = dest_y0;
    pc[8] = dest_pitch;
    pc[9] = endian;
    const uint32_t per = (wb.mode & 15u) == 4u ? 4u : 2u;
    dispatch_x = ((wb.w + per - 1) / per + 7) / 8;
    dispatch_y = (wb.h + 7) / 8;
  } else if (wb.path == 3) {
    layout = resolve_scaled_full_layout_;
    pipeline = resolve_scaled_full_pipeline_;
    source_set = OwnedWbColorSet(o);
    if (!scaled && constants.dest_base != 0) {
      // (The shader addresses from the binding's start: at 1x the binding must start at the texture's base.)
      OwnedWbRefused("scaled-full at 1x: the destination is not at the binding's start");
      OwnedTimerEnd(timer);
      return false;
    }
    pc[0] = wb.sx * scale_x;
    pc[1] = wb.sy * scale_y;
    pc[2] = wb.w * scale_x;
    pc[3] = wb.h * scale_y;
    pc[4] = wb.mode;
    pc[5] = wb.gap;
    pc[6] = uint32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8;
    pc[7] = uint32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8;
    pc[8] = uint32_t(info.copy_dest_coordinate_info.pitch_aligned_div_32) << 5;
    pc[9] = uint32_t(info.copy_dest_info.copy_dest_endian) | (scale_x << 4) | (scale_y << 8);
    const uint32_t per = (wb.mode & 15u) == 4u ? 4u : (wb.mode & 15u) == 5u ? 2u : 1u;
    dispatch_x = ((pc[2] + per - 1) / per + 7) / 8;
    dispatch_y = (pc[3] + 7) / 8;
  } else {
    layout = resolve_scaled_full_layout_;
    pipeline = scaled ? resolve_depth_scaled_pipeline_ : resolve_depth_1x_pipeline_;
    source_set = OwnedWbDepthSet(o);
    OwnedNoteDepthRead(wb.image, "depth write-back to guest memory");  // (Session 93, gpu_owned_tiler_passes: the depth is written back)
    if (constants.dest_base & 3u) {
      OwnedWbRefused("depth: destination base not dword aligned");
      OwnedTimerEnd(timer);
      return false;
    }
    pc[0] = 0;
    pc[1] = 0;
    pc[2] = wb.w * scale_x;
    pc[3] = wb.h * scale_y;
    pc[4] = wb.mode;
    pc[5] = scaled ? wb.gap : constants.dest_base;
    pc[6] = dest_x0;
    pc[7] = dest_y0;
    pc[8] = dest_pitch;
    pc[9] = (dest_info & 7u) | (scaled ? (scale_x << 4) | (scale_y << 8) : 0u);
    dispatch_x = (pc[2] + 7) / 8;
    dispatch_y = (pc[3] + 7) / 8;
  }
  if (layout == VK_NULL_HANDLE || pipeline == VK_NULL_HANDLE || source_set == VK_NULL_HANDLE) {
    OwnedWbRefused(std::string("armed, no pipeline / source set for ") + kWbPathNames[wb.path]);
    OwnedTimerEnd(timer);
    return false;
  }
  command_processor_.SubmitBarriers(true);
  if (wb.path == 4) {
    VkImageMemoryBarrier b = WbBarrier(
        o.depth, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
        depth_was_read ? VkAccessFlags(0) : VkAccessFlags(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT), VK_ACCESS_SHADER_READ_BIT,
        depth_was_read ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    cb.CmdVkPipelineBarrier(kWbShaderStages | kWbDepthStages, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
  } else if (!was_readable) {
    VkImageMemoryBarrier b = WbBarrier(o.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
  }
  command_processor_.BindExternalComputePipeline(pipeline);
  const VkDescriptorSet sets[2] = {source_set, dest_set};
  cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 2, sets, 0, nullptr);
  cb.CmdVkPushConstants(layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_words * sizeof(uint32_t), pc);
  command_processor_.NativeFlushResolveDestUse();  // (the destination's use: this dispatch writes it)
  command_processor_.SubmitBarriers(true);
  cb.CmdVkDispatch(dispatch_x, dispatch_y, 1);
  if (wb.path == 4) {
    VkImageMemoryBarrier b = WbBarrier(
        o.depth, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0,
        depth_was_read ? VK_ACCESS_SHADER_READ_BIT
                       : VkAccessFlags(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        depth_was_read ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, depth_was_read ? kWbShaderStages : kWbDepthStages, 0, 0, nullptr, 0,
                            nullptr, 1, &b);
  } else if (!was_readable) {
    VkImageMemoryBarrier b = WbBarrier(o.image, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                       VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr,
                            1, &b);
  }
  OwnedTimerEnd(timer);
  truth_foreign_bind_ = true;  // (a compute pipeline and sets of ours bound)
  ++owned_frame_stats_.wb_written[wb.path];
  if (wb.front) ++owned_frame_stats_.wb_front;
  static uint32_t logged = 0;
  if (logged < 40 || owned_frame_logging_ || segment_log_frames_ > 0) {
    ++logged;
    REXGPU_INFO("[native][owned] frame {}: written from our image: page 0x{:05X}, {}x{} (guest) at {},{}, dest format {}, image {}, path {}{}",
                frame_, wb.page, wb.w, wb.h, wb.sx, wb.sy, wb.dest_format, wb.image, kWbPathNames[wb.path],
                wb.front ? " (a front-buffer page)" : "");
  }
  return true;
}

}  // namespace rex::graphics::vulkan::native
