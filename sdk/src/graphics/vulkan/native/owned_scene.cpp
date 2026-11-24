// wwe2k14 patch (Session 82, the owned match frame, P1: the scene, G8; docs\native\OWNED_FRAME_CONTRACTS.md "P1").
// The scene pass's depth copy (k_24_8_FLOAT, the whole box into a texture's origin) left out: the texture's samplers are
// given a depth page of ours - an R32_SFLOAT image, the texture cache's host format for the texture - filled from our
// depth image by the truth path's own depth conversion (kTruthFillComputeShader, mode 2: the depth quantised to the
// console's 20e4 grid, decoded to the float the texture load gives), with our depth image as the source (sampled
// through a depth-only view; DEPTH_STENCIL_ATTACHMENT <-> DEPTH_STENCIL_READ_ONLY around the dispatch, as the truth
// copy does). The stencil is not kept: nothing reads it from the texture (the truth path keeps it only for a memory
// record that is never written in this frame). The page is read through OwnedView (a depth page: R32_SFLOAT, the
// swizzle as the binding gives it).
//
// At a resolution scale the truth path does not use the fill for depth (gpu_native_truth_texture_fill_scaled_kinds: a
// cost choice - it resolves into the scaled buffer and loads the texture from it); ours is the fill alone, in host
// pixels, with the copy's half-pixel gap (the scaled resolve's FULL_DEPTH variant takes it the same way).

#include <rex/graphics/vulkan/native/native_renderer.h>
#include "owned_dead_clear.h"  // (Session 92, M2d)

#include <algorithm>

#include <rex/cvar.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/logging.h>
#include <rex/ui/vulkan/util.h>

REXCVAR_DEFINE_BOOL(gpu_native_owned_depth_direct, true, "GPU/Native",
                    "Session 82: the scene's depth copy (k_24_8_FLOAT, 1x, no half-pixel gap) is not converted into a page of its own: "
                    "the page is the pass image's depth, sampled through a depth view in SHADER_READ_ONLY (the conversion pass "
                    "only rounded the depth to the console's 20e4 grid). 1440p, a format that cannot filter and every "
                    "other case keep the conversion");

extern uint32_t g_dres_depth_blits;  // (Session 94: owned_steps.cpp)

namespace rex::graphics::vulkan::native {

namespace {
constexpr VkPipelineStageFlags kSceneShaderStages =
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
constexpr VkPipelineStageFlags kSceneAttachmentStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                                        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
constexpr VkPipelineStageFlags kSceneDepthStages =
    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

VkImageMemoryBarrier SceneBarrier(VkImage image, VkImageAspectFlags aspects, VkAccessFlags src, VkAccessFlags dst,
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
}  // namespace

// Session 82 (P1): a timed segment of owned GPU work (the pass log: gpu_native_truth_pass_log N) - the owned render
// passes, the converted starts' box copies, the copy-ins of dirty tiles, snapshots, the depth and page fills. The
// segment is one more [native][pass] line (kind 16, the label) in recording order, with the truth passes'. Off: one
// compare of a counter.
uint32_t NativeRenderer::OwnedTimerBegin(const char* label, uint32_t width, uint32_t height) {
  if (pass_log_frames_ <= 0 || !truth_ || !query_pool_ || !slot_queries_ok_) return UINT32_MAX;
  FrameSlot& slot = slots_[SlotOf(frame_)];
  if (slot.queries + 2 > QueryLimit()) {
    ++truth_counts_.untimed;
    return UINT32_MAX;
  }
  FrameSlot::Segment& segment = slot.segments.emplace_back();
  segment.kind = 16;
  segment.label = label;
  segment.pass_width = uint16_t(width);
  segment.pass_height = uint16_t(height);
  segment.native_begin = slot.queries++;
  command_processor_.deferred_command_buffer().CmdVkWriteTimestamp(
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, SlotOf(frame_) * kQueriesPerFrame + segment.native_begin);
  return (uint32_t(SlotOf(frame_)) << 24) | uint32_t(slot.segments.size() - 1);  // (the slot too: a pass may end after the frame counter moved)
}

void NativeRenderer::OwnedTimerEnd(uint32_t segment, uint32_t draws) {
  if (segment == UINT32_MAX) return;
  const uint32_t slot_index = segment >> 24;
  FrameSlot& slot = slots_[slot_index];
  segment &= 0xFFFFFFu;
  if (segment >= slot.segments.size() || slot.queries >= QueryLimit()) return;
  FrameSlot::Segment& s = slot.segments[segment];
  s.pass_draws = draws;
  s.native_end = slot.queries++;
  command_processor_.deferred_command_buffer().CmdVkWriteTimestamp(
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, slot_index * kQueriesPerFrame + s.native_end);
}

void NativeRenderer::OwnedUnmapExtent(uint32_t start_u, uint32_t length_u) {
  // Pages whose memory a copy writes are the game's again (or ours, below): the pages overlapping the extent go.
  if (owned_pages_.empty()) return;
  const uint64_t start = start_u, end = start + length_u;
  const size_t before = owned_pages_.size();
  owned_pages_.erase(std::remove_if(owned_pages_.begin(), owned_pages_.end(),
                                    [&](const OwnedPage& p) {
                                      return start < uint64_t(p.extent_start) + p.extent_length &&
                                             uint64_t(p.extent_start) < end;
                                    }),
                     owned_pages_.end());
  if (owned_pages_.size() != before) {
    command_processor_.NativeOwnedPagesChanged();
    OwnedWatchSync();
    OwnedLeanSettle(false);  // (Session 95: a copy of ours may have lost its last page unread)
  }
}

bool NativeRenderer::OwnedFillDepth(int16_t page_image, int16_t image, uint32_t width, uint32_t height, uint32_t gap, bool from_read) {
  // The depth page filled from our depth image: the views and the source set made once per image; the page written
  // whole (its old texels not kept), then readable; our depth back in its attachment layout.
  if (page_image < 0 || size_t(page_image) >= owned_page_images_.size() || image < 0 ||
      size_t(image) >= owned_images_.size() || !EnsureTruthStaging() || !EnsureTruthFillTexPipelines()) {
    return false;
  }
  OwnedGraphTripwire(image);  // (Session 95)
  OwnedGraphNoteRead(image);  // (Session 95, round 7: the fill's dispatch)
  OwnedTouch(image);
  OwnedImage& o = owned_images_[image];
  const OwnedPageImage& q = owned_page_images_[page_image];
  if (o.depth == VK_NULL_HANDLE) return false;
  OwnedNoteDepthRead(image, "OwnedFillDepth (depth page fill)");  // (Session 93, gpu_owned_tiler_passes: a depth read)
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
      return false;
    }
  }
  // (Its binding 0 names the staging buffer - statically used by the shader; a new staging buffer, a new set.)
  if (o.depth_fill_set != VK_NULL_HANDLE && o.depth_fill_set_buffer != truth_staging_) {
    OwnedSpareFillSet(o.depth_fill_set);
    o.depth_fill_set = VK_NULL_HANDLE;
  }
  if (o.depth_fill_set == VK_NULL_HANDLE) {
    o.depth_fill_set_buffer = truth_staging_;
    if (OwnedAcquireFillSet(o.depth_fill_set) != VK_SUCCESS) {
      o.depth_fill_set = VK_NULL_HANDLE;
      return false;
    }
    VkDescriptorBufferInfo buffer_info = {truth_staging_, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo depth_info = {sampler_nearest(), o.depth_sample_view,
                                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo stencil_info = {sampler_nearest(), o.depth_stencil_view,
                                          VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
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
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  command_processor_.SubmitBarriers(true);
  VkImageMemoryBarrier barriers[2] = {
      SceneBarrier(o.depth, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                   from_read ? 0 : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                   from_read ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL),
      SceneBarrier(q.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_GENERAL)};
  const uint32_t timer = OwnedTimerBegin("depth_fill", width, height);  // (Session 82, P1: the pass log)
  cb.CmdVkPipelineBarrier(kSceneShaderStages | kSceneAttachmentStages, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                          nullptr, 0, nullptr, 2, barriers);
  command_processor_.BindExternalComputePipeline(truth_fill_tex_pipelines_[0]);
  const VkDescriptorSet sets[2] = {o.depth_fill_set, q.storage_set};
  cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, truth_fill_tex_layout_, 0, 2, sets, 0, nullptr);
  // (origin, size, mode 2 - depth to the float the texture load gives -, the gap, the destination's first texel.)
  const uint32_t constants[10] = {0u, 0u, width, height, 2u, gap, 0u, 0u, 0u, 0u};
  cb.CmdVkPushConstants(truth_fill_tex_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), constants);
  cb.CmdVkDispatch((width + 7) / 8, (height + 7) / 8, 1);
  barriers[0] = SceneBarrier(o.depth, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0,
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, kSceneDepthStages, 0, 0, nullptr, 0, nullptr, 1,
                          barriers);
  barriers[1] = SceneBarrier(q.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                             VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, kSceneShaderStages, 0, 0, nullptr, 0, nullptr, 1,
                          &barriers[1]);
  OwnedTimerEnd(timer);
  truth_foreign_bind_ = true;  // (a compute pipeline and sets of ours bound)
  return true;
}

bool NativeRenderer::OwnedDepthCopy(const draw_util::ResolveInfo& info, uint32_t page, uint32_t texture_page,
                                    std::string& why) {
  OwnedPass& p = owned_pending_;
  const auto no = [&](const char* text) {
    why = text;
    return false;
  };
  if (!p.valid || !p.depth || p.kind != 1 || p.depth_slot < 0 || size_t(p.depth_slot) >= OwnedSlotCount() ||
      p.image < 0 || size_t(p.image) >= owned_images_.size()) {
    return no("the pass keeps no depth");
  }
  const OwnedSurface t = OwnedSlotKey(p.depth_slot);
  const draw_util::ResolveEdramInfo& edram = info.depth_edram_info;
  const reg::RB_COPY_DEST_INFO d = info.copy_dest_info;
  if (t.kind != 3 || !(t.format & 1u)) return no("not a 20e4 depth surface");
  if (edram.msaa_samples != xenos::MsaaSamples::k1X || d.copy_dest_array || d.copy_dest_exp_bias || d.copy_dest_swap ||
      uint32_t(info.copy_dest_coordinate_info.copy_sample_select) != 0) {
    return no("msaa / array / exp bias / swap / sample select");
  }
  if (uint32_t(d.copy_dest_format) != uint32_t(xenos::TextureFormat::k_24_8_FLOAT)) return no("destination format");
  if (info.depth_original_base != t.base || edram.pitch_tiles != t.row_tiles()) return no("another depth surface");
  // The copied tiles all the pass's (its depth tiles), and the copy the whole box into a texture's origin.
  uint32_t span_base = 0, row_length = 0, rows = 0, span_pitch = 0;
  info.GetCopyEdramTileSpan(span_base, row_length, rows, span_pitch);
  if (!rows || !row_length) return no("empty span");
  for (uint32_t r = 0; r < rows && !OwnedOnly(); ++r) {  // (Session 92, M2b: by surface under the latch - the base / pitch check above - no pass tiles)
    for (uint32_t c = 0; c < row_length; ++c) {
      if (!p.tiles.test((span_base + r * span_pitch + c) & 2047u)) return no("tiles outside the pass");
    }
  }
  const uint32_t tile_delta = (uint32_t(edram.base_tiles) - t.base) & 2047u;
  const uint32_t row_tiles = std::max(1u, t.row_tiles());
  const uint32_t sx = (tile_delta % row_tiles) * 80 + uint32_t(info.coordinate_info.edram_offset_x_div_8) * 8;
  const uint32_t sy = (tile_delta / row_tiles) * 16 + uint32_t(info.coordinate_info.edram_offset_y_div_8) * 8;
  const uint32_t w = uint32_t(info.coordinate_info.width_div_8) * 8, h = uint32_t(info.height_div_8) * 8;
  const uint32_t dx = info.copy_dest_base_x + uint32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8;
  const uint32_t dy = info.copy_dest_base_y + uint32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8;
  if (sx != 0 || sy != 0 || w != p.width || h != p.height || dx != 0 || dy != 0 || page != texture_page) {
    return no("not the whole box into a texture's origin");
  }
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  // (Session 94: a scene pass drawn at a ratio - its depth is brought up (nearest) into the depth of a full-size image first, and that image is the copy's source.)
  int16_t src = p.image;
  if (p.dres) {
    if (w * scale_x != p.host_width || h * scale_y != p.host_height) return no("size");
    // (Session 94: the pass's one full-size image (made at its start, depth_read reset there); a second depth copy of the pass gets a fresh one.)
    src = p.dres_depth_used || p.dres_full < 0 ? OwnedImageFor(p.host_width, p.host_height, 1) : p.dres_full;
    if (src < 0) return no("scaled scene pass: no full-size image for the depth blit");
    if (!OwnedEnsureDepth(src)) return no("scaled scene pass: no depth image at the full size");
  } else if (w * scale_x != owned_images_[p.image].width || h * scale_y != owned_images_[p.image].height) {
    return no("size");
  }
  const OwnedImage& image = owned_images_[src];
  // Everything that can fail before the pass is committed (after it the copy must succeed or the truth path runs it
  // from the dirty tiles, which stays exact but costs the conversion in the game's way).
  if (!EnsureTruthStaging() || !EnsureTruthFillTexPipelines()) return no("fill pipelines");
  if (p.dres && owned_depth_up_on_ && !OwnedDepthUpReady(p.image, src)) return no("scaled scene pass: the depth up-pass's set or framebuffer");  // (Session 95)
  OwnedUnmapExtent(info.copy_dest_extent_start, info.copy_dest_extent_length);
  const uint32_t gap = edram.fill_half_pixel_offset ? (scale_x / 2) | ((scale_y / 2) << 8) : 0u;
  if (owned_depth_direct_ok_ < 0) {
    const ui::vulkan::VulkanDevice* vd = command_processor_.GetVulkanDevice();
    VkFormatProperties properties = {};
    vd->vulkan_instance()->functions().vkGetPhysicalDeviceFormatProperties(vd->physical_device(), post_ds_format_,
                                                                          &properties);
    owned_depth_direct_ok_ = (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) ? 1 : 0;
  }
  const bool direct = REXCVAR_GET(gpu_native_owned_depth_direct) && owned_depth_direct_ok_ == 1 && gap == 0 &&
                      scale_x == 1 && scale_y == 1 && image.depth != VK_NULL_HANDLE;
  // (A second depth copy of the same pass - the game does one into 0x104AE twice - is another direct page of the same depth:
  // nothing writes it after the commit. The conversion cannot run on a depth already in SHADER_READ_ONLY.)
  if (image.depth_read && !direct) return no("the depth is shared already and this copy cannot be direct");
  const int16_t page_image =
      direct ? int16_t(-1) : OwnedPageImageFor(w * scale_x, h * scale_y, uint32_t(xenos::TextureFormat::k_24_8_FLOAT));
  if (!direct && page_image < 0) return no("no page image (budget / format)");
  OwnedCommit();  // (the colour's copy may have come first: then it is a no-op)
  if (p.dres) {
    if (!OwnedBlitUpDepth(p, src)) return no("scaled scene pass: the depth blit");
    p.dres_depth_used = true;
    ++g_dres_depth_blits;
  }
  if (direct) {
    OwnedNoteDepthRead(src, "direct depth copy (page shares the depth)");  // (Session 93, gpu_owned_tiler_passes: the depth is shared with a page)
    static uint32_t direct_logged = 0;
    if (direct_logged < 3) {
      ++direct_logged;
      REXGPU_INFO("[native][owned] frame {}: the depth copy into page 0x{:05X} is the image's depth itself (gpu_native_owned_depth_direct; no conversion pass)", frame_, page);
    }
    // (The depth stays where it is: attachment -> shader read once, here, after the pass's last draw; the next pass that
    // starts in this image begins from UNDEFINED, and the page keeps the image from being given to one until the game's
    // next copy into the texture replaces it - as the colour pages of the same pass.)
    if (!image.depth_read) {
    command_processor_.SubmitBarriers(true);
    const VkImageMemoryBarrier readable =
        SceneBarrier(image.depth, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    command_processor_.deferred_command_buffer().CmdVkPipelineBarrier(kSceneDepthStages, kSceneShaderStages, 0, 0, nullptr,
                                                                    0, nullptr, 1, &readable);
    owned_images_[src].depth_read = true;
    }
  } else if (!OwnedFillDepth(page_image, src, w * scale_x, h * scale_y, gap)) {
    return no("the fill could not be recorded");
  }
  OwnedPage pg;
  pg.page = page;
  pg.page_image = page_image;
  if (direct) {
    pg.image = src;
    pg.depth_direct = true;
  }
  pg.width = w;
  pg.height = h;
  pg.format = uint32_t(d.copy_dest_format);
  pg.endian = uint32_t(d.copy_dest_endian);
  pg.extent_start = info.copy_dest_extent_start;
  pg.extent_length = info.copy_dest_extent_length;
  pg.frame = frame_;
  pg.producers = p.producers;
  pg.key = p.key;
  pg.frame_owned = owned_frame_on_;
  pg.census = p.census;
  owned_pages_.push_back(pg);
  // (Session 82, P2: the depth surface a depth page was copied out of this frame - a later colour-less clear of it, the
  // game's end-of-frame depth clear before the front buffer's copy, is dead: OwnedDeadDepthClear.)
  if (p.depth_slot >= 0 && size_t(p.depth_slot) < OwnedSlotCount()) {
    owned_scene_depth_.base = OwnedSlotKey(p.depth_slot).base;
    owned_scene_depth_.pitch_tiles = p.pitch_tiles;
    owned_scene_depth_.width = w;
    owned_scene_depth_.height = h;
    owned_scene_depth_.frame = frame_ + 1;  // (0 = none)
    if (OwnedOnly() && owned_frame_stats_.frames < 4) M2dOrderLog(true, frame_, "scene depth copy made", fmt::format("depth slot {} (base {} pitch {}), {}x{}, page 0x{:05X}", p.depth_slot, owned_scene_depth_.base, p.pitch_tiles, w, h, page));
  }
  command_processor_.NativeOwnedPagesChanged();
  OwnedWatchSync();
  OwnedWatchRegister();
  if (segment_log_frames_ > 0) {
    REXGPU_INFO("[native][seg] frame {} owned depth copy left out: page 0x{:05X} ({}x{}, k_24_8_FLOAT) is our depth page "
                "image {} (filled from our image {}'s depth{})",
                frame_, page, w, h, page_image, src, direct ? ", direct: the image's depth itself" : "");
  }
  return true;
}

}  // namespace rex::graphics::vulkan::native
