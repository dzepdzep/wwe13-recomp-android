// wwe2k14 patch (Session 82, the owned match frame, P2, G4): the damage maps' mip chain - feedback textures of our own.
// Docs: docs\native\OWNED_FRAME_DESIGN.md (G4), CLAUDE.md hard rule 6 (never keep a 360 quirk just to match it).
//
// The game rebuilds each wrestler's damage map (a 1024x512 k_4_4_4_4 texture with mip levels) by drawing small blended
// rectangles into a 1040-wide surface and copying each into one level of the texture (draw a splat, copy it into level 0,
// redraw, copy into level 1 ...), the draws sampling the texture itself and the next frame's scene sampling it too.
// The 360's way - the texture's levels laid out in the packed mip tail, the surface's EDRAM tiles shared with other
// surfaces, 4-bit channels - is not kept here: the texture is an RGBA8 image of ours with real mip levels; each pass is
// drawn into a box image of ours (a defined start, cleared) and its rectangle copied into the level it names; samplers
// of the texture get a view over all levels (OwnedView). Close, not exact: 8-bit values (the 360 keeps 4 bits a
// channel); levels nothing wrote start as zero (the maps are blank at the match start); the packed tail's levels are
// never written (a copy into one is the game's own).

#include <rex/graphics/vulkan/native/native_renderer.h>

#include <algorithm>
#include <cstring>

#include <rex/graphics/vulkan/command_processor.h>
#include <rex/logging.h>
#include <rex/ui/vulkan/util.h>

REXCVAR_DECLARE(bool, gpu_native_owned_frame_cube_derive);
REXCVAR_DECLARE(bool, gpu_native_owned_frame_cube_derive_scaled);
REXCVAR_DECLARE(bool, gpu_native_owned_frame_strip_batch);

namespace rex::graphics::vulkan::native {

namespace {
constexpr VkPipelineStageFlags kFbShaderStages =
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
constexpr VkPipelineStageFlags kFbAttachmentStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                                     VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                                     VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
VkImageMemoryBarrier FbBarrier(VkImage image, uint32_t base_level, uint32_t level_count, VkAccessFlags src,
                               VkAccessFlags dst, VkImageLayout from, VkImageLayout to, uint32_t base_layer = 0,
                               uint32_t layer_count = 1) {
  VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = image;
  b.srcAccessMask = src;
  b.dstAccessMask = dst;
  b.oldLayout = from;
  b.newLayout = to;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, base_level, level_count, base_layer, layer_count};
  return b;
}
// Session 92 (cube derivation survives stale cubes): file-level state (no new members in the header the game exe includes).
std::vector<uint32_t> g_cube_last_copy;  // per owned_feedback_ index: the frame a face copy of ours last wrote that cube
uint32_t g_cube_stale_dropped = 0, g_cube_older_dropped = 0, g_cube_contradicted = 0, g_cube_reused = 0;
// Per owned_feedback_ index: the frame the entry was dropped (0xFFFFFFFF = never), the base page it had, and whether a learn dropped it.
std::vector<uint32_t> g_cube_drop_frame, g_cube_drop_page;
std::vector<uint8_t> g_cube_drop_learn;
void NoteCubeDrop(size_t i, uint32_t frame, uint32_t page, bool learn) {
  if (g_cube_drop_frame.size() <= i) {
    g_cube_drop_frame.resize(i + 1, 0xFFFFFFFFu);
    g_cube_drop_page.resize(i + 1, 0);
    g_cube_drop_learn.resize(i + 1, 0);
  }
  g_cube_drop_frame[i] = frame;
  g_cube_drop_page[i] = page;
  g_cube_drop_learn[i] = learn ? 1 : 0;
}
}  // namespace

int16_t NativeRenderer::OwnedBlitUp(const OwnedPass& p) {
  OwnedGraphTripwire(p.image);  // (Session 95)
  OwnedGraphNoteRead(p.image);  // (Session 95, round 7)
  const OwnedImage src = owned_images_[p.image];  // (a copy: OwnedImageFor may grow the vector)
  // (Session 94: a scaled scene pass - the first colour copy goes into its one full-size image; a later one gets a fresh image, the first page keeps its picture.)
  const bool use_full = p.dres && p.dres_full >= 0 && !p.dres_color_used;
  const int16_t index = use_full ? p.dres_full : OwnedImageFor(p.host_width, p.host_height, p.kind);
  if (index < 0) return -1;
  if (use_full) const_cast<OwnedPass&>(p).dres_color_used = true;
  OwnedTouch(p.image);
  OwnedTouch(index);
  OwnedImage& dst = owned_images_[index];
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  VkImageMemoryBarrier barriers[2] = {
      FbBarrier(src.image, 0, 1, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                src.readable ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 0, VK_REMAINING_ARRAY_LAYERS),  // (Session 94: every layer of an owned image)
      FbBarrier(dst.image, 0, 1, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_REMAINING_ARRAY_LAYERS)};
  cb.CmdVkPipelineBarrier(kFbShaderStages | kFbAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
  VkImageBlit blit = {};
  blit.srcSubresource = blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.srcOffsets[1] = {int32_t(src.width), int32_t(src.height), 1};
  blit.dstOffsets[1] = {int32_t(dst.width), int32_t(dst.height), 1};
  for (uint32_t layer = 0; layer < OwnedLayers(); ++layer) {  // (Session 94: layer by layer)
    blit.srcSubresource.baseArrayLayer = blit.dstSubresource.baseArrayLayer = layer;
    cb.CmdVkBlitImage(src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                      &blit, VK_FILTER_LINEAR);
  }
  // (The half image back to the layout it was in; the full one readable.)
  barriers[0] = FbBarrier(src.image, 0, 1, 0,
                          src.readable ? VK_ACCESS_SHADER_READ_BIT
                                       : VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          src.readable ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                       : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, VK_REMAINING_ARRAY_LAYERS);
  barriers[1] = FbBarrier(dst.image, 0, 1, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, VK_REMAINING_ARRAY_LAYERS);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, kFbShaderStages | kFbAttachmentStages, 0, 0, nullptr, 0,
                          nullptr, 2, barriers);
  dst.readable = true;
  return index;
}

bool NativeRenderer::OwnedBlitUpDepth(const OwnedPass& p, int16_t target) {
  // Session 94 (gpu_native_owned_draw_ratio): the scene pass's depth, drawn at a ratio, brought up to the full size (nearest: a depth cannot be
  // filtered) into the depth image of `target` (made by the caller), which ends in the attachment layout the depth copy's readers expect.
  if (p.image < 0 || size_t(p.image) >= owned_images_.size() || target < 0 || size_t(target) >= owned_images_.size()) return false;
  const OwnedImage& src = owned_images_[p.image];
  const OwnedImage& dst = owned_images_[target];
  if (src.depth == VK_NULL_HANDLE || dst.depth == VK_NULL_HANDLE) return false;
  OwnedNoteDepthRead(p.image, "dres depth blit");  // (Session 94: the tiler learning sees the depth read)
  OwnedTouch(p.image);
  OwnedTouch(target);
  if (owned_depth_up_on_) return OwnedDepthUpPass(p, target);  // (Session 95: no depth blit on this device)
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const VkImageAspectFlags both = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
  const VkImageLayout src_layout = src.depth_read ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  const VkPipelineStageFlags depth_stages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  command_processor_.SubmitBarriers(true);
  VkImageMemoryBarrier barriers[2] = {};
  for (VkImageMemoryBarrier& b : barriers) {
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange = {both, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS};  // (Session 95: every layer of a decoupled owned depth)
  }
  barriers[0].image = src.depth;
  barriers[0].srcAccessMask = src.depth_read ? 0 : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  barriers[0].oldLayout = src_layout;
  barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barriers[1].image = dst.depth;
  barriers[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                              VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  cb.CmdVkPipelineBarrier(kFbShaderStages | depth_stages | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                          nullptr, 2, barriers);
  VkImageBlit blit = {};
  blit.srcSubresource = blit.dstSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
  blit.srcOffsets[1] = {int32_t(src.width), int32_t(src.height), 1};
  blit.dstOffsets[1] = {int32_t(dst.width), int32_t(dst.height), 1};
  for (uint32_t layer = 0; layer < OwnedLayers(); ++layer) {  // (Session 95: layer by layer, as the colour blit above)
    blit.srcSubresource.baseArrayLayer = blit.dstSubresource.baseArrayLayer = layer;
    cb.CmdVkBlitImage(src.depth, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.depth, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                      VK_FILTER_NEAREST);
  }
  barriers[0].srcAccessMask = 0;
  barriers[0].dstAccessMask = src.depth_read ? VK_ACCESS_SHADER_READ_BIT
                                             : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barriers[0].newLayout = src_layout;
  barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barriers[1].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barriers[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, kFbShaderStages | depth_stages, 0, 0, nullptr, 0, nullptr, 2, barriers);
  owned_images_[target].depth_read = false;
  return true;
}

// Session 95: the owned depth up-pass - each full-size pixel takes the scaled pass's depth at the nearest source pixel, as the blit's nearest filter does. Our own objects
// (rule 6), no truth image or twin. Stencil is not carried (the blit does not carry it either): it starts cleared to 0.
namespace {
constexpr char kOwnedDepthUpVertex[] = R"(
#version 450
void main() {
  vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";
constexpr char kOwnedDepthUpFragment[] = R"(
#version 450
layout(set = 0, binding = 1) uniform sampler2D src_depth;
layout(push_constant) uniform Constants { vec2 ratio; vec2 max_coord; } c;
void main() {
  ivec2 p = ivec2(min(floor(gl_FragCoord.xy * c.ratio), c.max_coord));
  gl_FragDepth = texelFetch(src_depth, p, 0).r;
}
)";
}  // namespace

bool NativeRenderer::EnsureOwnedDepthUpObjects() {
  if (owned_depth_up_pipeline_ != VK_NULL_HANDLE) return true;
  if (owned_depth_up_failed_ || !EnsureWbSetLayout()) return false;
  owned_depth_up_failed_ = true;  // until everything below exists
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  const VkShaderStageFlagBits shader_stages[2] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT};
  const char* const sources[2] = {kOwnedDepthUpVertex, kOwnedDepthUpFragment};
  for (uint32_t i = 0; i < 2; ++i) {
    if (owned_depth_up_shaders_[i] != VK_NULL_HANDLE) continue;
    std::vector<uint32_t> spirv;
    std::string error;
    if (!command_processor_.CompileGlslToSpirv(shader_stages[i], sources[i], spirv, error)) {
      REXGPU_ERROR("[native][owned-dres] depth up-pass shader compilation failed: {}", error);
      return false;
    }
    owned_depth_up_shaders_[i] = ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
    if (owned_depth_up_shaders_[i] == VK_NULL_HANDLE) return false;
  }
  if (owned_depth_up_layout_ == VK_NULL_HANDLE) {
    // (Set 0: an image's depth source set, OwnedWbDepthSet - its binding 1 is visible to fragment shaders.)
    VkPushConstantRange push = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4 * sizeof(float)};
    VkPipelineLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &truth_store_set_layout_;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    if (dfn.vkCreatePipelineLayout(device, &layout_info, nullptr, &owned_depth_up_layout_) != VK_SUCCESS) {
      owned_depth_up_layout_ = VK_NULL_HANDLE;
      return false;
    }
  }
  if (owned_depth_up_render_pass_ == VK_NULL_HANDLE) {
    VkAttachmentDescription attachment = {};
    attachment.format = post_ds_format_;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference reference = {0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.pDepthStencilAttachment = &reference;
    VkRenderPassCreateInfo pass_info = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    pass_info.attachmentCount = 1;
    pass_info.pAttachments = &attachment;
    pass_info.subpassCount = 1;
    pass_info.pSubpasses = &subpass;
    if (dfn.vkCreateRenderPass(device, &pass_info, nullptr, &owned_depth_up_render_pass_) != VK_SUCCESS) {
      owned_depth_up_render_pass_ = VK_NULL_HANDLE;
      return false;
    }
  }
  VkPipelineShaderStageCreateInfo stages[2] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, owned_depth_up_shaders_[0], "main", nullptr},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, owned_depth_up_shaders_[1], "main", nullptr}};
  VkPipelineVertexInputStateCreateInfo vertex_input = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo input_assembly = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo viewport = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewport.viewportCount = 1;
  viewport.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo raster = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode = VK_CULL_MODE_NONE;
  raster.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo multisample = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo depth = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  depth.depthTestEnable = VK_TRUE;
  depth.depthWriteEnable = VK_TRUE;
  depth.depthCompareOp = VK_COMPARE_OP_ALWAYS;
  VkPipelineColorBlendStateCreateInfo blend_state = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic.dynamicStateCount = 2;
  dynamic.pDynamicStates = dynamic_states;
  VkGraphicsPipelineCreateInfo info = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  info.stageCount = 2;
  info.pStages = stages;
  info.pVertexInputState = &vertex_input;
  info.pInputAssemblyState = &input_assembly;
  info.pViewportState = &viewport;
  info.pRasterizationState = &raster;
  info.pMultisampleState = &multisample;
  info.pDepthStencilState = &depth;
  info.pColorBlendState = &blend_state;
  info.pDynamicState = &dynamic;
  info.layout = owned_depth_up_layout_;
  info.renderPass = owned_depth_up_render_pass_;
  if (dfn.vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &owned_depth_up_pipeline_) != VK_SUCCESS) {
    owned_depth_up_pipeline_ = VK_NULL_HANDLE;
    return false;
  }
  owned_depth_up_failed_ = false;
  return true;
}

bool NativeRenderer::OwnedDepthUpReady(int16_t source, int16_t target) {
  if (source < 0 || size_t(source) >= owned_images_.size() || target < 0 || size_t(target) >= owned_images_.size() || !EnsureOwnedDepthUpObjects()) return false;
  if (OwnedWbDepthSet(owned_images_[source]) == VK_NULL_HANDLE) return false;
  OwnedImage& dst = owned_images_[target];
  if (dst.depth == VK_NULL_HANDLE) return false;
  if (dst.framebuffer_depth_up != VK_NULL_HANDLE) return true;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  VkFramebufferCreateInfo fb_info = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  fb_info.renderPass = owned_depth_up_render_pass_;
  fb_info.attachmentCount = 1;
  fb_info.pAttachments = &dst.depth_view;
  fb_info.width = dst.width;
  fb_info.height = dst.height;
  fb_info.layers = 1;
  if (vulkan_device->functions().vkCreateFramebuffer(vulkan_device->device(), &fb_info, nullptr, &dst.framebuffer_depth_up) != VK_SUCCESS) {
    dst.framebuffer_depth_up = VK_NULL_HANDLE;
    return false;
  }
  return true;
}

bool NativeRenderer::OwnedDepthUpPass(const OwnedPass& p, int16_t target) {
  if (!OwnedDepthUpReady(p.image, target)) return false;  // (made before the commit: OwnedDepthCopy; nothing is done twice)
  OwnedImage& src = owned_images_[p.image];
  OwnedImage& dst = owned_images_[target];
  const VkDescriptorSet set = OwnedWbDepthSet(src);
  if (set == VK_NULL_HANDLE) return false;
  OwnedGraphNoteRead(p.image);  // (Session 95, round 7: this pass samples the depth; its framebuffer is no pass image's)
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const VkImageAspectFlags both = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
  const VkImageLayout src_layout = src.depth_read ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  const VkPipelineStageFlags depth_stages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  constexpr VkAccessFlags kDepthAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  command_processor_.SubmitBarriers(true);
  const uint32_t timer = OwnedTimerBegin("depth_up", dst.width, dst.height);
  VkImageMemoryBarrier barriers[2] = {};
  for (VkImageMemoryBarrier& b : barriers) {
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange = {both, 0, 1, 0, 1};
  }
  barriers[0].image = src.depth;
  barriers[0].srcAccessMask = src.depth_read ? 0 : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barriers[0].oldLayout = src_layout;
  barriers[0].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
  barriers[1].image = dst.depth;
  barriers[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT | kDepthAccess | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  barriers[1].dstAccessMask = kDepthAccess;
  barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barriers[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  cb.CmdVkPipelineBarrier(kFbShaderStages | depth_stages | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | depth_stages, 0, 0, nullptr, 0, nullptr, 2,
                          barriers);
  VkClearValue clear = {};
  clear.depthStencil = {1.0f, 0u};
  VkRenderPassBeginInfo begin_info = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  begin_info.renderPass = owned_depth_up_render_pass_;
  begin_info.framebuffer = dst.framebuffer_depth_up;
  begin_info.renderArea = {{0, 0}, {dst.width, dst.height}};
  begin_info.clearValueCount = 1;
  begin_info.pClearValues = &clear;
  cb.CmdVkBeginRenderPass(&begin_info, VK_SUBPASS_CONTENTS_INLINE);
  // (Through the command processor's tracking, so that its next SetViewport / SetScissor and pipeline are not taken as already bound.)
  command_processor_.SetViewport(VkViewport{0.0f, 0.0f, float(dst.width), float(dst.height), 0.0f, 1.0f});
  command_processor_.SetScissor(VkRect2D{{0, 0}, {dst.width, dst.height}});
  command_processor_.BindExternalGraphicsPipeline(owned_depth_up_pipeline_);
  cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, owned_depth_up_layout_, 0, 1, &set, 0, nullptr);
  // (Source pixels per target pixel, and the last source pixel.)
  const float constants[4] = {float(src.width) / float(dst.width), float(src.height) / float(dst.height), float(src.width) - 1.0f, float(src.height) - 1.0f};
  cb.CmdVkPushConstants(owned_depth_up_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(constants), constants);
  cb.CmdVkDraw(3, 1, 0, 0);
  cb.CmdVkEndRenderPass();
  barriers[0].srcAccessMask = 0;
  barriers[0].dstAccessMask = src.depth_read ? VK_ACCESS_SHADER_READ_BIT : kDepthAccess;
  barriers[0].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
  barriers[0].newLayout = src_layout;
  barriers[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  barriers[1].dstAccessMask = kDepthAccess | VK_ACCESS_SHADER_READ_BIT;
  barriers[1].oldLayout = barriers[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | depth_stages, kFbShaderStages | depth_stages, 0, 0, nullptr, 0, nullptr, 2, barriers);
  OwnedTimerEnd(timer);
  dst.depth_read = false;
  // (Whatever the owned frame or the processor binds next binds its own pipeline, sets and viewport.)
  owned_bound_ = OwnedBound();
  truth_bound_valid_ = false;
  truth_foreign_bind_ = true;
  command_processor_.NativeDropBindings();
  return true;
}

bool NativeRenderer::OwnedRectDrawn(const OwnedPass& p, uint32_t sx, uint32_t sy, uint32_t w, uint32_t h) const {
  // (The bounding box of the draws' rectangles, kept by OwnedRecord - their union is not tracked: close enough.)
  return p.drawn[2] > p.drawn[0] && p.drawn[0] <= int32_t(sx) && p.drawn[1] <= int32_t(sy) &&
         p.drawn[2] >= int32_t(sx + w) && p.drawn[3] >= int32_t(sy + h);
}

bool NativeRenderer::OwnedDeriveCubeFace(const draw_util::ResolveInfo& info, uint32_t sx, uint32_t sy, uint32_t w, uint32_t h,
                                         uint32_t& tex_page, uint32_t& layers, uint32_t& layer, bool& scaled,
                                         uint32_t& extent) {
  // A defined-start strip pass of six square faces side by side (the reflection / light cubes: 384x64 for 64x64 faces), each face
  // copied whole into its own page. A tiled k_8_8_8_8 face of whole 32x32 tiles (w = h = 32, 64 or 128) is w*h*4 bytes, the
  // faces lie one face apart (the cubes the cache learned: 0x1F462 + 4 pages a face), so face k of the strip is layer k and the
  // cube base page is this page less k faces. Anything else is not derived.
  const OwnedPass& p = owned_pending_;
  if (owned_cube_derive_off_ || !p.valid || !p.edram_free || w != h || (w != 32 && w != 64 && w != 128) || sy != 0 ||
      sx % w != 0 || p.width != 6 * w || p.height != h) {
    return false;
  }
  // (Review: the copy's destination is a w x w texture too - a face, not a strip into something else.)
  if (uint32_t(info.copy_dest_coordinate_info.pitch_aligned_div_32) * 32 != w ||
      uint32_t(info.copy_dest_coordinate_info.height_aligned_div_32) * 32 != h) {
    return false;
  }
  const uint32_t face = sx / w;
  const uint32_t stride_pages = (w * h * 4) >> 12;
  const uint32_t page = (info.copy_dest_base >> 12) & 0x1FFFF;
  if (!stride_pages || page < face * stride_pages) return false;
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  // (At a resolution scale the derived cubes' first row and column were dark - the copy's half-pixel gap, now filled in
  // OwnedFeedbackCopy; texdiff, 1440p: the cube differs in its first row / column only.)
  if ((scale_x > 1 || scale_y > 1) && !REXCVAR_GET(gpu_native_owned_frame_cube_derive_scaled)) return false;
  tex_page = page - face * stride_pages;
  layers = 6;
  layer = face;
  scaled = scale_x > 1 || scale_y > 1;
  extent = stride_pages * 6 * 4096;
  owned_derived_resolved_ = true;  // (the copy's range is marked resolved at 1x: render_target_cache.cpp)
  static uint32_t derived_logged = 0;
  if (derived_logged < 12) {
    ++derived_logged;
    REXGPU_INFO("[native][owned] frame {}: face {} of a cube the texture cache has not seen (page 0x{:05X}, {}x{}): derived the cube at "
                "page 0x{:05X}, 6 faces", frame_, face, page, w, h, tex_page);
  }
  return true;
}

void NativeRenderer::OwnedCubeLearned(uint32_t base_page, uint32_t width, uint32_t height, uint32_t faces,
                                      uint32_t stride_bytes) {
  // The cache has learned a cube from its first load: a cube of ours made before it (OwnedDeriveCubeFace) must be the same cube.
  // Our cube whose page lies inside this cube's pages but is not it - or differs in size or faces - was a wrong guess: dropped
  // (its sampler then reads the game's texture, which these face copies were never written to).
  if (owned_feedback_.empty() || !stride_bytes) return;
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const uint32_t pages = std::max(1u, (stride_bytes * faces) >> 12);
  for (size_t i = 0; i < owned_feedback_.size(); ++i) {
    OwnedFeedback& f = owned_feedback_[i];
    if (!f.cube || f.page == 0 || f.page < base_page || f.page >= base_page + pages) continue;
    const bool same = f.page == base_page && f.layers == faces && f.width == width * scale_x && f.height == height * scale_y;
    if (scale_x > 1) {
      REXGPU_INFO("[native][owned] frame {}: cube learned by the cache at page 0x{:05X} {}x{} {} faces stride {} (scale {}); ours at page 0x{:05X} {}x{} {} layers: {}",
                  frame_, base_page, width, height, faces, stride_bytes, scale_x, f.page, f.width, f.height, f.layers, same ? "same" : "DIFFERENT");
    }
    if (same) continue;
    REXGPU_WARN("[native][owned] frame {}: the cube derived at page 0x{:05X} ({}x{}, {} faces) is not the cube the cache learned at "
                "0x{:05X} ({}x{}, {} faces): dropped", frame_, f.page, f.width, f.height, f.layers, base_page, width, height, faces);
    const int16_t dead = int16_t(i);
    owned_pages_.erase(std::remove_if(owned_pages_.begin(), owned_pages_.end(),
                                      [&](const OwnedPage& q) { return q.feedback == dead; }),
                       owned_pages_.end());
    command_processor_.NativeOwnedPagesChanged();
    const uint32_t dropped_page = f.page;
    f.page = 0;
    if (OwnedChains()) {
      OwnedWatchSync();
      if (f.cube) NoteCubeDrop(i, uint32_t(frame_), dropped_page, true);
      // (Session 92, TRUE NATIVE: a learned cube overlapping ours is normally a STALE cube of ours - the game moved its cubes between
      // entrance segments - so only that one is dropped and deriving goes on; the next face copy derives the cube at its new base.
      // Only a cube our copies wrote in this very frame being contradicted is a wrong guess: derivation ends after 2 such cases.)
      if (i < g_cube_last_copy.size() && g_cube_last_copy[i] == uint32_t(frame_)) {
        ++g_cube_contradicted;
        if (g_cube_contradicted >= 2) owned_cube_derive_off_ = true;
      } else {
        ++g_cube_stale_dropped;
      }
      continue;
    }
    // (Review: our faces were never written to memory, so this cube stays blank until the game builds it again - the guess is
    // wrong for this game: no more cubes are derived; they are replayed on the truth path as before.)
    owned_cube_derive_off_ = true;
  }
}

void OwnedCubeCounters(uint32_t& stale, uint32_t& older, uint32_t& contradicted, uint32_t& reused, bool reset) {
  stale = g_cube_stale_dropped;
  older = g_cube_older_dropped;
  contradicted = g_cube_contradicted;
  reused = g_cube_reused;
  if (reset) g_cube_stale_dropped = g_cube_older_dropped = g_cube_reused = 0;  // (the contradictions are cumulative: they decide the kill switch)
}

int NativeRenderer::OwnedFeedbackCopy(const draw_util::ResolveInfo& info, uint32_t sx, uint32_t sy, uint32_t w,
                                      uint32_t h, uint32_t dx, uint32_t dy, bool swap, bool cube) {
  OwnedPass& p = owned_pending_;
  if (OwnedChains() && p.valid && !OwnedFeedbackSurface(p.base, p.pitch_tiles, p.kind)) owned_feedback_surfaces_.push_back({p.base, p.pitch_tiles, p.kind});
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  uint32_t tex_page = 0, w0 = 0, h0 = 0, levels = 0, level = 0, extent = 0, layer = 0, layers = 1;
  bool scaled = false;
  // (cube: the copy's destination page is a face's start page the texture cache learned from a load of the cube - a copy
  // to any other page is not ours (0); a face that cannot be served is -1: replayed, not paused.)
  const int fail = cube ? -1 : 0;
  if (cube) {
    if (!command_processor_.NativeResolveFace((info.copy_dest_base >> 12) & 0x1FFFF,
                                              uint32_t(info.copy_dest_info.copy_dest_endian),
                                              xenos::TextureFormat::k_8_8_8_8, tex_page, w0, h0, layers, layer, scaled,
                                              extent)) {
      // (Session 82, P2: not a cube the cache has seen - nothing sampled it yet. The cube of a 6-face strip: derived.)
      if (!REXCVAR_GET(gpu_native_owned_frame_cube_derive) ||
          !OwnedDeriveCubeFace(info, sx, sy, w, h, tex_page, layers, layer, scaled, extent)) {
        return 0;
      }
      w0 = w;
      h0 = h;
    }
    levels = 1;
  } else if (!command_processor_.NativeResolveTarget(
                 (info.copy_dest_texture_base >> 12) & 0x1FFFF,
                 uint32_t(info.copy_dest_coordinate_info.pitch_aligned_div_32),
                 uint32_t(info.copy_dest_info.copy_dest_endian), xenos::TextureFormat::k_4_4_4_4, tex_page, w0, h0,
                 levels, level, scaled, extent)) {
    return 0;  // (no load of that texture seen yet, or its packed tail: the game's copy)
  }
  if (scaled != (scale_x > 1 || scale_y > 1) || scale_x != scale_y) return fail;
  // (A cube's face is copied whole into its own start page: the copy's dest "base x, y" - an offset in the texture the
  // register file guesses around the face's page - means nothing there: the face's origin.)
  if (cube) {
    if (w != w0 || h != h0) return fail;
    dx = dy = 0;
  }
  const uint32_t lw = std::max(w0 >> level, 1u), lh = std::max(h0 >> level, 1u);
  if (level >= levels || dx + w > lw || dy + h > lh || !w || !h) return fail;
  const uint32_t scale = scale_x;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  // The feedback texture of this page: ours already, or a new one (zero: the maps start blank).
  int16_t fi = -1;
  for (size_t i = 0; i < owned_feedback_.size(); ++i) {
    if (owned_feedback_[i].page == tex_page) fi = int16_t(i);
  }
  const uint32_t host_w = w0 * scale, host_h = h0 * scale;
  const bool recreate = fi >= 0 && (owned_feedback_[fi].width != host_w || owned_feedback_[fi].height != host_h ||
                                    owned_feedback_[fi].levels != levels || owned_feedback_[fi].cube != cube ||
                                    owned_feedback_[fi].layers != layers);
  bool held = false;  // (a page of ours stands for it now: its levels hold what the game's copies made)
  for (const OwnedPage& q : owned_pages_) held = held || (fi >= 0 && q.feedback == fi);
  if ((fi < 0 || recreate) && cube && OwnedChains()) {
    // (Session 92, TRUE NATIVE: two cubes cannot share memory - an older derived cube of ours overlapping the new one at another base page
    // is the game's old cube, stale now: dropped.)
    const uint32_t new_pages = std::max(1u, layers * ((w0 * h0 * 4u) >> 12));
    for (size_t j = 0; j < owned_feedback_.size(); ++j) {
      OwnedFeedback& o = owned_feedback_[j];
      if (!o.cube || o.page == 0 || o.page == tex_page || scale == 0) continue;
      const uint32_t o_pages = std::max(1u, o.layers * (((o.width / scale) * (o.height / scale) * 4u) >> 12));
      if (o.page >= tex_page + new_pages || tex_page >= o.page + o_pages) continue;
      // (Views are left as the learn drop leaves them: commands recorded earlier may still use them.)
      const int16_t dead = int16_t(j);
      owned_pages_.erase(std::remove_if(owned_pages_.begin(), owned_pages_.end(),
                                        [&](const OwnedPage& q) { return q.feedback == dead; }),
                         owned_pages_.end());
      command_processor_.NativeOwnedPagesChanged();
      OwnedWatchSync();
      REXGPU_INFO("[native][owned] frame {}: the cube derived at page 0x{:05X} overlaps the new cube at 0x{:05X}: the older one dropped",
                  frame_, o.page, tex_page);
      NoteCubeDrop(j, uint32_t(frame_), o.page, false);
      o.page = 0;
      ++g_cube_older_dropped;
    }
  }
  bool reused = false;
  if (fi < 0 && cube && OwnedChains()) {
    // (Hardening: a cube re-derived at the base page a learn dropped within the last 2 frames - the cache keeps loading a cube that
    // contradicts ours before our strips run - is a contradiction too; the drop / re-derive loop must trip the switch.)
    for (size_t j = 0; j < owned_feedback_.size() && j < g_cube_drop_frame.size(); ++j) {
      if (owned_feedback_[j].page != 0 || !g_cube_drop_learn[j] || g_cube_drop_page[j] != tex_page) continue;
      g_cube_drop_learn[j] = 0;
      if (g_cube_drop_frame[j] != 0xFFFFFFFFu && uint32_t(frame_) - g_cube_drop_frame[j] <= 2) {
        ++g_cube_contradicted;
        if (g_cube_contradicted >= 2) owned_cube_derive_off_ = true;
      }
    }
    // (Reuse a dead cube entry of the same shape, dropped in an earlier frame: its image, memory and views stay in use instead of a
    // new image every move. It is left in SHADER_READ_ONLY, the layout the !held clear below starts from.)
    for (size_t j = 0; j < owned_feedback_.size() && j < g_cube_drop_frame.size(); ++j) {
      OwnedFeedback& o = owned_feedback_[j];
      if (o.page != 0 || !o.cube || o.image == VK_NULL_HANDLE || g_cube_drop_frame[j] == 0xFFFFFFFFu ||
          g_cube_drop_frame[j] >= uint32_t(frame_) || o.width != host_w || o.height != host_h || o.levels != levels ||
          o.layers != layers) {
        continue;
      }
      o.page = tex_page;
      fi = int16_t(j);
      reused = true;
      ++g_cube_reused;
      g_cube_drop_frame[j] = 0xFFFFFFFFu;
      g_cube_drop_learn[j] = 0;
      break;
    }
  }
  if ((fi < 0 || recreate) && !reused) {
    const uint64_t bytes =uint64_t(host_w) * host_h * 4u * 4u / 3u;
    if (owned_frame_on_ && OwnedOverBudget(bytes * layers)) {
      // (Session 87, chains, run 023: at 1440p the owned images fill the budget to within a few MB (510 of 512); one extra 2560x1440 image made before the
      // first feedback copy left no room for the 11 MB texture and every later copy of the damage / sweat maps failed here for ever - nothing was evicted for
      // it, as OwnedImageAvailable does for an image. Unused images are evicted first, then the budget is asked again.)
      if (OwnedChains()) OwnedEvictFor(bytes * layers);
      if (OwnedOverBudget(bytes * layers)) {
        static uint32_t budget_logged = 0;
        if (OwnedChains() && budget_logged < 8) {
          ++budget_logged;
          REXGPU_INFO("[native][owned] frame {}: no feedback texture for page 0x{:05X}: {} MB in use + {} MB over the owned-frame budget", frame_, tex_page,
                      owned_bytes_ >> 20, (bytes * layers) >> 20);
        }
        return fail;
      }
    }
    if (recreate) {
      // (A different size: the old texture's views and page go; the image stays as a dead entry until the end -
      // commands recorded earlier may still use it.)
      OwnedFeedback& old = owned_feedback_[fi];
      for (auto it = owned_views_.begin(); it != owned_views_.end();) {
        if (it->image == old.image) {
          dfn.vkDestroyImageView(vulkan_device->device(), it->view, nullptr);
          it = owned_views_.erase(it);
          ++owned_view_generation_;  // (Session 95, r6tex)
        } else {
          ++it;
        }
      }
      const int16_t dead = fi;
      owned_pages_.erase(std::remove_if(owned_pages_.begin(), owned_pages_.end(),
                                        [&](const OwnedPage& q) { return q.feedback == dead; }),
                         owned_pages_.end());
      command_processor_.NativeOwnedPagesChanged();
      old.page = 0;
      fi = -1;
    }
    OwnedFeedback f;
    f.page = tex_page;
    f.width = host_w;
    f.height = host_h;
    f.levels = levels;
    f.cube = cube;
    f.layers = layers;
    f.bytes = bytes * layers;
    VkImageCreateInfo image_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    if (cube) image_info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    image_info.extent = {host_w, host_h, 1};
    image_info.mipLevels = levels;
    image_info.arrayLayers = layers;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!ui::vulkan::util::CreateDedicatedAllocationImage(vulkan_device, image_info,
                                                          ui::vulkan::util::MemoryPurpose::kDeviceLocal, f.image,
                                                          f.memory)) {
      REXGPU_ERROR("[native][owned] could not make a {}x{} feedback texture", host_w, host_h);
      return fail;
    }
    owned_bytes_ += bytes * layers;
    owned_feedback_.push_back(f);
    fi = int16_t(owned_feedback_.size() - 1);
    REXGPU_INFO("[native][owned] feedback texture for page 0x{:05X}: {}x{}, {} levels, {} layers{} (RGBA8) made",
                tex_page, host_w, host_h, levels, layers, cube ? " (a cube)" : "");
  }
  OwnedFeedback& f = owned_feedback_[fi];
  if (cube && OwnedChains()) {  // (Session 92: the frame a copy of ours last wrote this cube - the narrow kill switch's evidence)
    if (g_cube_last_copy.size() <= size_t(fi)) g_cube_last_copy.resize(owned_feedback_.size(), 0xFFFFFFFFu);
    g_cube_last_copy[fi] = uint32_t(frame_);
  }
  OwnedCommit();  // (our pass's render pass ends; a defined start has nothing to mark dirty)
  p.feedback_pass = true;
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  // A texture of ours made now, or one no page of ours stands for any more (its texture was the game's again in between):
  // every level cleared to zero first.
  if (!held || !f.cleared_once) {
    const bool first_use = !f.cleared_once;
    const VkImageMemoryBarrier to_dst = FbBarrier(
        f.image, 0, VK_REMAINING_MIP_LEVELS, first_use ? 0 : VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        first_use ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_REMAINING_ARRAY_LAYERS);
    cb.CmdVkPipelineBarrier(kFbShaderStages | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_dst);
    const VkClearColorValue zero = {};
    const VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0,
                                         VK_REMAINING_ARRAY_LAYERS};
    cb.CmdVkClearColorImage(f.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &all);
    const VkImageMemoryBarrier to_read =
        FbBarrier(f.image, 0, VK_REMAINING_MIP_LEVELS, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0,
                  VK_REMAINING_ARRAY_LAYERS);
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, kFbShaderStages, 0, 0, nullptr, 0, nullptr, 1, &to_read);
    f.cleared_once = true;
  }
  // (Session 87, gpu_native_owned_frame_strip_batch: the six faces of a cube strip are one copy at its first face - 30 copies and 60
  // barriers a frame in Bryan's entrance become 5 and 10; the other five face copies are answered from it while the pass has not
  // drawn again. Resolution scale 1 only (no half-pixel gap regions); anything that does not fit takes the copy below.)
  const bool batch_shape = cube && scale == 1 && layers == 6 && w == w0 && h == h0 && p.valid && p.width == 6 * w0 &&
                           p.height == h0 && sy == 0 && sx == layer * w && REXCVAR_GET(gpu_native_owned_frame_strip_batch);
  OwnedStripBatch& sb = owned_strip_batch_;
  const bool answered = batch_shape && held && sb.valid && sb.image == p.image && sb.feedback == fi && sb.serial == owned_draw_serial_ &&
                        sb.width == w0 && (sb.mask & (1u << layer));
  const bool batch_first = batch_shape && !answered && layer == 0;
  if (answered) {
    ++owned_frame_stats_.strip_faces_answered;
  } else {
    sb.valid = false;
    // The rectangle: our pass's image -> the level (only that level changes layout, and back).
    OwnedTouch(p.image);
    const OwnedImage& m = owned_images_[p.image];
    const VkImageLayout from =
        m.readable ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    const uint32_t dst_layers = batch_first ? 6u : 1u;
    VkImageMemoryBarrier barriers[2] = {
        FbBarrier(m.image, 0, 1, m.readable ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_ACCESS_TRANSFER_READ_BIT, from, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 0, VK_REMAINING_ARRAY_LAYERS),  // (Session 94: every layer)
        FbBarrier(f.image, level, 1, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, layer, dst_layers)};
    cb.CmdVkPipelineBarrier(kFbShaderStages | kFbAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
    // (Session 82, P2: at a resolution scale the game's copy fills the first host columns / rows of its rectangle - never drawn - from
    // the first covered one (the truth fill's max(p, gap); fill_half_pixel_offset): texel (x, y) from (max(x, gx), max(y, gy)). Ours
    // copied them as drawn (nothing): a dark first row and column on every cube face, the rim light's thin outlines at 1440p.)
    const uint32_t gx = (scale > 1 && info.color_edram_info.fill_half_pixel_offset) ? std::min(scale / 2, 8u) : 0u;
    const uint32_t gy = gx;
    const uint32_t rw = w * scale, rh = h * scale;
    VkImageCopy regions[2 + 8 + 8 + 64];
    uint32_t region_count = 0;
    const auto add_region = [&](uint32_t src_x, uint32_t src_y, uint32_t dst_x, uint32_t dst_y, uint32_t ew, uint32_t eh) {
      VkImageCopy& r = regions[region_count++];
      r = {};
      r.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, layer, 1};
      r.srcOffset = {int32_t(sx * scale + src_x), int32_t(sy * scale + src_y), 0};
      r.dstOffset = {int32_t(dx * scale + dst_x), int32_t(dy * scale + dst_y), 0};
      r.extent = {ew, eh, 1};
    };
    if (batch_first) {
      for (uint32_t k = 0; k < 6; ++k) {
        VkImageCopy& r = regions[region_count++];
        r = {};
        r.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, k, 1};
        r.srcOffset = {int32_t(k * w0), 0, 0};
        r.dstOffset = {0, 0, 0};
        r.extent = {w0, h0, 1};
      }
    } else if (gx && rw > gx && rh > gy) {
      add_region(gx, gy, gx, gy, rw - gx, rh - gy);
      for (uint32_t y = 0; y < gy; ++y) add_region(gx, gy, gx, y, rw - gx, 1);
      for (uint32_t x = 0; x < gx; ++x) add_region(gx, gy, x, gy, 1, rh - gy);
      for (uint32_t x = 0; x < gx; ++x) {
        for (uint32_t y = 0; y < gy; ++y) add_region(gx, gy, x, y, 1, 1);
      }
    } else {
      add_region(0, 0, 0, 0, rw, rh);
    }
    OwnedCopyImage(cb, m.image, true, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, f.image, false, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region_count,
                   regions);  // (Session 94: from the picture's layer)
    barriers[0] = FbBarrier(m.image, 0, 1, 0,
                            m.readable ? VK_ACCESS_SHADER_READ_BIT
                                       : VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, from, 0, VK_REMAINING_ARRAY_LAYERS);
    barriers[1] = FbBarrier(f.image, level, 1, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, layer, dst_layers);
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, kFbShaderStages | kFbAttachmentStages, 0, 0, nullptr, 0,
                            nullptr, 2, barriers);
    if (batch_first) {
      sb.valid = true;
      sb.image = p.image;
      sb.feedback = fi;
      sb.serial = owned_draw_serial_;
      sb.width = w0;
      sb.mask = 0x3F;
      ++owned_frame_stats_.strip_batches;
    }
  }
  // The page: one per texture (the copies of every level update it).
  OwnedPage* page = nullptr;
  for (OwnedPage& q : owned_pages_) {
    if (q.page == tex_page) page = &q;
  }
  if (page && page->feedback != fi) {
    owned_pages_.erase(std::remove_if(owned_pages_.begin(), owned_pages_.end(),
                                      [&](const OwnedPage& q) { return q.page == tex_page; }),
                       owned_pages_.end());
    page = nullptr;
  }
  if (!page) {
    OwnedPage pg;
    pg.page = tex_page;
    pg.feedback = fi;
    pg.width = w0;
    pg.height = h0;
    pg.format = uint32_t(cube ? xenos::ColorFormat::k_8_8_8_8 : xenos::ColorFormat::k_4_4_4_4);
    pg.endian = uint32_t(info.copy_dest_info.copy_dest_endian);
    pg.swap = swap;
    pg.extent_start = tex_page << 12;
    pg.extent_length = extent;
    pg.frame = frame_;
    pg.producers = p.producers;
    pg.key = p.key;
    pg.frame_owned = owned_frame_on_;
    pg.census = p.census;
    owned_pages_.push_back(pg);
    command_processor_.NativeOwnedPagesChanged();
    OwnedWatchSync();
    OwnedWatchRegister();
    OwnedMarkLeftOut(owned_pages_.back());
  } else {
    page->frame = frame_;
    page->swap = swap;
    page->producers = p.producers;
    page->key = p.key;
    page->census = p.census;
  }
  if (segment_log_frames_ > 0) {
    REXGPU_INFO("[native][seg] frame {} owned rectangle {},{} {}x{} into level {} of our feedback texture (page "
                "0x{:05X}) at {},{}",
                frame_, sx, sy, w, h, level, tex_page, dx, dy);
  }
  return 1;
}

VkImageView NativeRenderer::OwnedFeedbackView(const OwnedPage& page, const VkComponentMapping& mapping) {
  const OwnedFeedback& f = owned_feedback_[page.feedback];
  // The texture's swizzle over our RGBA8: as for a k_8_8_8_8 page - the copy's red / blue swap undone by the view.
  VkComponentMapping components = mapping;
  const VkComponentSwizzle own[4] = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,
                                     VK_COMPONENT_SWIZZLE_A};
  VkComponentSwizzle* const parts[4] = {&components.r, &components.g, &components.b, &components.a};
  for (uint32_t i = 0; i < 4; ++i) {
    VkComponentSwizzle& s = *parts[i];
    if (s == VK_COMPONENT_SWIZZLE_IDENTITY) s = own[i];
    if (page.swap) {
      if (s == VK_COMPONENT_SWIZZLE_R) {
        s = VK_COMPONENT_SWIZZLE_B;
      } else if (s == VK_COMPONENT_SWIZZLE_B) {
        s = VK_COMPONENT_SWIZZLE_R;
      }
    }
  }
  for (const OwnedViewEntry& v : owned_views_) {
    if (v.image == f.image && v.format == VK_FORMAT_R8G8B8A8_UNORM &&
        v.type == (f.cube ? VK_IMAGE_VIEW_TYPE_CUBE : owned_view_type_) &&
        !std::memcmp(&v.components, &components, sizeof(components))) {
      return v.view;
    }
  }
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view_info.image = f.image;
  view_info.viewType = f.cube ? VK_IMAGE_VIEW_TYPE_CUBE : owned_view_type_;
  view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
  view_info.components = components;
  view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, f.levels, 0, f.layers};
  VkImageView view = VK_NULL_HANDLE;
  if (vulkan_device->functions().vkCreateImageView(vulkan_device->device(), &view_info, nullptr, &view) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  owned_views_.push_back({f.image, components, view, VK_FORMAT_R8G8B8A8_UNORM,
                          f.cube ? VK_IMAGE_VIEW_TYPE_CUBE : owned_view_type_});
  return view;
}

}  // namespace rex::graphics::vulkan::native
