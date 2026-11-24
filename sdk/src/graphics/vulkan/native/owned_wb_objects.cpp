// wwe2k14 patch (Session 92, M2d, Part 2): the objects the owned write-back / fill / page-image paths need, moved out of native_truth.cpp UNCHANGED so that the truth path
// can be compiled out (REX_TRUE_NATIVE_ONLY): the staging buffer and its sets, the dummy images, the fill-texture pipelines and the 8888 / full-1x / scaled-full / depth
// resolve pipelines (used by owned_copy.cpp, owned_scene.cpp, owned_steps.cpp). Their shader sources: owned_wb_shaders.h.

#include <rex/graphics/vulkan/native/native_renderer.h>
#include "owned_wb_shaders.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <fmt/format.h>
#include <rex/cvar.h>
#include <rex/graphics/free_counters.h>
#include <rex/logging.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/ui/vulkan/util.h>
#include <rex/ui/vulkan/instance.h>

namespace rex::graphics::vulkan::native {

bool NativeRenderer::EnsureWbSetLayout() {
  // The set layout of the fill / write-back / truth store sources (storage buffer; colour and stencil samplers), made once - moved unchanged from EnsureTruthObjects.
  // (Binding 2: a depth image's stencil - written and read only for depth images.)
  // (Session 79: the images visible to fragment shaders too - gpu_native_truth_alias_direct reads them in its load.)
  if (truth_store_set_layout_ != VK_NULL_HANDLE) return true;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  VkDescriptorSetLayoutBinding bindings[3] = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
       nullptr},
      {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
       nullptr}};
  VkDescriptorSetLayoutCreateInfo set_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  set_info.bindingCount = 3;
  set_info.pBindings = bindings;
  if (dfn.vkCreateDescriptorSetLayout(device, &set_info, nullptr, &truth_store_set_layout_) != VK_SUCCESS) {
    truth_store_set_layout_ = VK_NULL_HANDLE;
    return false;
  }
  return true;
}

bool NativeRenderer::EnsureTruthStaging() {
  const VkDeviceSize size = command_processor_.NativeEdramSizeBytes();
  if (!size) return false;
  if (truth_staging_ != VK_NULL_HANDLE && truth_staging_size_ == size && truth_staging_resolve_set_ != VK_NULL_HANDLE) {
    return true;
  }
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  if (truth_staging_ != VK_NULL_HANDLE) {
    Retired r;
    r.submission = command_processor_.GetCurrentSubmission();
    r.buffer = truth_staging_;
    r.memory[0] = truth_staging_memory_;
    retired_.push_back(r);
    truth_staging_ = VK_NULL_HANDLE;
    truth_staging_memory_ = VK_NULL_HANDLE;
    for (TruthImage& t : truth_images_) {
      if (t.store_sets[0] != VK_NULL_HANDLE) {
        truth_retired_sets_.push_back({command_processor_.GetCurrentSubmission(), t.store_sets[0]});
        t.store_sets[0] = VK_NULL_HANDLE;
      }
      if (t.resolve_set != VK_NULL_HANDLE) {  // (its binding 0 is the staging buffer)
        truth_retired_sets_.push_back({command_processor_.GetCurrentSubmission(), t.resolve_set});
        t.resolve_set = VK_NULL_HANDLE;
      }
    }
    for (TruthBundle& b : truth_spares_) {  // (review, Session 52: the pooled images' sets too)
      if (b.store_sets[0] != VK_NULL_HANDLE) {
        truth_retired_sets_.push_back({command_processor_.GetCurrentSubmission(), b.store_sets[0]});
        b.store_sets[0] = VK_NULL_HANDLE;
      }
      if (b.resolve_set != VK_NULL_HANDLE) {
        truth_retired_sets_.push_back({command_processor_.GetCurrentSubmission(), b.resolve_set});
        b.resolve_set = VK_NULL_HANDLE;
      }
    }
  }
  if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
          vulkan_device, size,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,  // (Session 47: TruthFillTexture's depth copies)
          ui::vulkan::util::MemoryPurpose::kDeviceLocal, truth_staging_, truth_staging_memory_)) {
    truth_staging_ = VK_NULL_HANDLE;
    truth_staging_memory_ = VK_NULL_HANDLE;
    return false;
  }
  truth_staging_size_ = size;
  truth_staging_resolve_set_ = command_processor_.NativeResolveSetForBuffer(truth_staging_);
  // The load shaders' set over it: a surface reading tiles another surface wrote gets them through the staging buffer
  // (stored there by their owner, loaded from there: the EDRAM's byte reinterpretation without the EDRAM).
  if (truth_staging_load_set_ != VK_NULL_HANDLE) {
    Retired r;
    r.submission = command_processor_.GetCurrentSubmission();
    r.replace_set = truth_staging_load_set_;
    retired_.push_back(r);
    truth_staging_load_set_ = VK_NULL_HANDLE;
  }
  if (EnsureReplaceObjects()) {
    const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
    VkDescriptorSetAllocateInfo alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = replace_descriptor_pool_;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &load_set_layout_;
    if (dfn.vkAllocateDescriptorSets(vulkan_device->device(), &alloc, &truth_staging_load_set_) == VK_SUCCESS) {
      VkDescriptorBufferInfo buffer_info = {truth_staging_, 0, VK_WHOLE_SIZE};
      VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      write.dstSet = truth_staging_load_set_;
      write.descriptorCount = 1;
      write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      write.pBufferInfo = &buffer_info;
      dfn.vkUpdateDescriptorSets(vulkan_device->device(), 1, &write, 0, nullptr);
    } else {
      truth_staging_load_set_ = VK_NULL_HANDLE;
    }
  }
  REXGPU_INFO("[native][truth] staging buffer {} MB", size >> 20);
  return truth_staging_resolve_set_ != VK_NULL_HANDLE;
}

bool NativeRenderer::EnsureTruthDummy() {
  if (truth_dummy_stencil_view_ != VK_NULL_HANDLE && truth_dummy_depth_view_ != VK_NULL_HANDLE) return true;
  if (truth_dummy_ds_ != VK_NULL_HANDLE) return false;  // (review 3: a failed earlier attempt; not retried)
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = post_ds_format_;
  info.extent = {1, 1, 1};
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (!ui::vulkan::util::CreateDedicatedAllocationImage(vulkan_device, info,
                                                        ui::vulkan::util::MemoryPurpose::kDeviceLocal, truth_dummy_ds_,
                                                        truth_dummy_ds_memory_)) {
    truth_dummy_ds_ = VK_NULL_HANDLE;
    truth_dummy_ds_memory_ = VK_NULL_HANDLE;
    return false;
  }
  VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view_info.image = truth_dummy_ds_;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = post_ds_format_;
  view_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_STENCIL_BIT);
  bool ok = dfn.vkCreateImageView(device, &view_info, nullptr, &truth_dummy_stencil_view_) == VK_SUCCESS;
  view_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT);
  ok = ok && dfn.vkCreateImageView(device, &view_info, nullptr, &truth_dummy_depth_view_) == VK_SUCCESS;
  if (!ok) return false;
  // Into the layout the sets name, once (never read: every pixel comes from the colour image).
  // Session 93 (idea 15, spec cleanliness): this runs from the write-back PLAN (OwnedWbPlan: the first front copy at boot, with the owned pass still
  // open - OwnedCommit ends it after the plan), so the barrier below was recorded inside an open render pass with no self-dependency
  // (VUID-vkCmdPipelineBarrier-None-07889, PC run 6475). Whatever is open ends first (a one-time event: the owned / truth pass resumes with LOAD).
  command_processor_.EndRenderPass();
  VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = truth_dummy_ds_;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barrier.subresourceRange =
      ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
  command_processor_.deferred_command_buffer().CmdVkPipelineBarrier(
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  return true;
}

bool NativeRenderer::EnsureTruthFillTexSmallPipelines() {
  // Session 59 (gpu_native_truth_fill_direct_small): TEXOUT 3 (r8ui) / 4 (r16ui) - the device's extended storage
  // formats.
  if (truth_fill_tex_pipelines_[2] != VK_NULL_HANDLE && truth_fill_tex_pipelines_[3] != VK_NULL_HANDLE) return true;
  if (truth_fill_tex_small_failed_ || !EnsureTruthFillTexPipelines()) return false;
  truth_fill_tex_small_failed_ = true;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  if (!vulkan_device->properties().shaderStorageImageExtendedFormats) return false;
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  for (int variant = 2; variant < 4; ++variant) {
    std::vector<uint32_t> spirv;
    std::string error;
    const std::string source = std::string("#version 450\n#define TEXOUT ") + (variant == 2 ? "3" : "4") + "\n" +
                               kTruthFillComputeShader;
    if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT, source, spirv, error)) {
      REXGPU_ERROR("[native][truth] direct small texture fill shader compilation failed: {}", error);
      return false;
    }
    const VkShaderModule shader =
        ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
    if (shader == VK_NULL_HANDLE) return false;
    VkComputePipelineCreateInfo info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader,
                  "main", nullptr};
    info.layout = truth_fill_tex_layout_;
    const bool ok = dfn.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                                                 &truth_fill_tex_pipelines_[variant]) == VK_SUCCESS;
    dfn.vkDestroyShaderModule(device, shader, nullptr);
    if (!ok) {
      truth_fill_tex_pipelines_[variant] = VK_NULL_HANDLE;
      return false;
    }
  }
  truth_fill_tex_small_failed_ = false;
  return true;
}

// Session 52 (gpu_native_truth_fill_direct): the pack shader with TEXOUT 1 (r32ui) / 2 (rgba16ui), on the store's set 0
// and the texture's storage set 1.
bool NativeRenderer::EnsureTruthFillTexPipelines() {
  if (truth_fill_tex_pipelines_[0] != VK_NULL_HANDLE && truth_fill_tex_pipelines_[1] != VK_NULL_HANDLE) return true;
  const VkDescriptorSetLayout texture_set_layout = command_processor_.NativeTextureSetLayout();
  if (truth_fill_tex_failed_ || !EnsureWbSetLayout() || texture_set_layout == VK_NULL_HANDLE) {
    return false;
  }
  truth_fill_tex_failed_ = true;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  if (truth_fill_tex_layout_ == VK_NULL_HANDLE) {
    const VkDescriptorSetLayout set_layouts[2] = {truth_store_set_layout_, texture_set_layout};
    VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 10 * sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 2;
    layout_info.pSetLayouts = set_layouts;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    if (dfn.vkCreatePipelineLayout(device, &layout_info, nullptr, &truth_fill_tex_layout_) != VK_SUCCESS) {
      truth_fill_tex_layout_ = VK_NULL_HANDLE;
      return false;
    }
  }
  for (int variant = 0; variant < 2; ++variant) {
    std::vector<uint32_t> spirv;
    std::string error;
    const std::string source = std::string("#version 450\n#define TEXOUT ") + (variant ? "2" : "1") + "\n" +
                               kTruthFillComputeShader;
    if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT, source, spirv, error)) {
      REXGPU_ERROR("[native][truth] direct texture fill shader compilation failed: {}", error);
      return false;
    }
    const VkShaderModule shader =
        ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
    if (shader == VK_NULL_HANDLE) return false;
    VkComputePipelineCreateInfo info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader,
                  "main", nullptr};
    info.layout = truth_fill_tex_layout_;
    const bool ok = dfn.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                                                 &truth_fill_tex_pipelines_[variant]) == VK_SUCCESS;
    dfn.vkDestroyShaderModule(device, shader, nullptr);
    if (!ok) {
      truth_fill_tex_pipelines_[variant] = VK_NULL_HANDLE;
      return false;
    }
  }
  truth_fill_tex_failed_ = false;
  return true;
}

bool NativeRenderer::EnsureResolveScaledFullPipeline() {
  if (resolve_scaled_full_pipeline_ != VK_NULL_HANDLE) return true;
  if (resolve_scaled_full_failed_ || !EnsureWbSetLayout()) return false;
  resolve_scaled_full_failed_ = true;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  if (resolve_scaled_full_layout_ == VK_NULL_HANDLE) {
    const VkDescriptorSetLayout set_layouts[2] = {
        truth_store_set_layout_, command_processor_.GetSingleTransientDescriptorLayout(
                                     VulkanCommandProcessor::SingleTransientDescriptorLayout::kStorageBufferCompute)};
    VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 10 * sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 2;
    layout_info.pSetLayouts = set_layouts;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    if (dfn.vkCreatePipelineLayout(device, &layout_info, nullptr, &resolve_scaled_full_layout_) != VK_SUCCESS) {
      resolve_scaled_full_layout_ = VK_NULL_HANDLE;
      return false;
    }
  }
  std::vector<uint32_t> spirv;
  std::string error;
  if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT,
                                             std::string("#version 450\n") + kNativeResolveScaledFullComputeShader,
                                             spirv, error)) {
    REXGPU_ERROR("[native][truth] scaled full resolve shader compilation failed: {}", error);
    return false;
  }
  const VkShaderModule shader =
      ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
  if (shader == VK_NULL_HANDLE) return false;
  VkComputePipelineCreateInfo info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader,
                "main", nullptr};
  info.layout = resolve_scaled_full_layout_;
  const bool ok = dfn.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                                               &resolve_scaled_full_pipeline_) == VK_SUCCESS;
  dfn.vkDestroyShaderModule(device, shader, nullptr);
  if (!ok) {
    resolve_scaled_full_pipeline_ = VK_NULL_HANDLE;
    return false;
  }
  resolve_scaled_full_failed_ = false;
  return true;
}

// Session 90 (Q, the read-back write-back, owned_copy.cpp): the pipelines TruthResolveFull1x / TruthResolveDepth1x make lazily inside their
// bodies, for a caller that must know they exist before it commits to using them. Same shaders, same layout (resolve_scaled_full_layout_,
// made by EnsureResolveScaledFullPipeline).
bool NativeRenderer::EnsureResolveFull1xPipeline() {
  if (resolve_full_1x_pipeline_ != VK_NULL_HANDLE) return true;
  if (resolve_full_1x_failed_ || resolve_scaled_full_layout_ == VK_NULL_HANDLE) return false;
  resolve_full_1x_failed_ = true;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  std::vector<uint32_t> spirv;
  std::string error;
  if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT,
                                             std::string("#version 450\n") + kNativeResolveFull1xComputeShader, spirv, error)) {
    REXGPU_ERROR("[native][truth] 1x full resolve shader compilation failed: {}", error);
    return false;
  }
  const VkShaderModule shader =
      ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
  if (shader == VK_NULL_HANDLE) return false;
  VkComputePipelineCreateInfo create = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  create.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader, "main", nullptr};
  create.layout = resolve_scaled_full_layout_;
  const bool ok = dfn.vkCreateComputePipelines(vulkan_device->device(), VK_NULL_HANDLE, 1, &create, nullptr,
                                               &resolve_full_1x_pipeline_) == VK_SUCCESS;
  dfn.vkDestroyShaderModule(vulkan_device->device(), shader, nullptr);
  if (!ok) {
    resolve_full_1x_pipeline_ = VK_NULL_HANDLE;
    return false;
  }
  resolve_full_1x_failed_ = false;
  return true;
}

bool NativeRenderer::EnsureResolveDepthPipeline(bool scaled) {
  VkPipeline& depth_pipeline = scaled ? resolve_depth_scaled_pipeline_ : resolve_depth_1x_pipeline_;
  bool& depth_failed = scaled ? resolve_depth_scaled_failed_ : resolve_depth_1x_failed_;
  if (depth_pipeline != VK_NULL_HANDLE) return true;
  if (depth_failed || resolve_scaled_full_layout_ == VK_NULL_HANDLE) return false;
  depth_failed = true;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  std::vector<uint32_t> spirv;
  std::string error;
  if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT,
                                             scaled ? std::string("#version 450\n#define FULL_DEPTH 1\n") + kNativeResolveScaledFullComputeShader
                                                    : std::string("#version 450\n#define FULL1X_DEPTH 1\n") + kNativeResolveFull1xComputeShader,
                                             spirv, error)) {
    REXGPU_ERROR("[native][truth] depth resolve shader compilation failed: {}", error);
    return false;
  }
  const VkShaderModule shader =
      ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
  if (shader == VK_NULL_HANDLE) return false;
  VkComputePipelineCreateInfo create = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  create.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader, "main", nullptr};
  create.layout = resolve_scaled_full_layout_;
  const bool ok = dfn.vkCreateComputePipelines(vulkan_device->device(), VK_NULL_HANDLE, 1, &create, nullptr, &depth_pipeline) == VK_SUCCESS;
  dfn.vkDestroyShaderModule(vulkan_device->device(), shader, nullptr);
  if (!ok) {
    depth_pipeline = VK_NULL_HANDLE;
    return false;
  }
  depth_failed = false;
  return true;
}

}  // namespace rex::graphics::vulkan::native
