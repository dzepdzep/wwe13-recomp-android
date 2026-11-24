// wwe2k14 patch (Session 80, OWN_RENDERER_PLAN.md: the crowd fully ours; gpu_native_own_crowd). Our own crowd draw
// inside NATIVE's truth pass: one instanced draw of a crowd run's members with the title's own program (its SPIR-V,
// src\scene\own_crowd_* in the game), our own pipeline layout and descriptor set, and a per-submission ring holding our
// copy of the member mesh (converted from the guest's big-endian vertices - the game rebuilds its crowd meshes every
// frame at a new address, run 5423, so each run converts its own), the members' matrices and the scene block.
// The pipeline is the crowd draw's own truth pipeline key (its depth, cull, blend and pass) with our shaders and layout.
// Map of the parts used: docs\native\OWN_CROWD_MAP.md (game repo).
// Session 95 (P7, gpu_native_crowd_input): the same ring, read as vertex attributes: the mesh as binding 0, the members as instance-rate binding 1, host-order indices from the index pool. The descriptor set
// and its dynamic offsets stay as they are (the attribute form reads no binding 4 or 5), so a merge and the owned pass's call list work unchanged. A run the plan refuses keeps the storage form.

#include "diag_layer.h"  // Session 95 (the diagnostic run)
#include "shader_swap.h"   // Session 95 (P7): VertexMirrorLatched
#include "vertex_input.h"  // Session 95 (P7)
#include <rex/graphics/vulkan/native/native_renderer.h>

#include <algorithm>
#include <cstring>

#include <rex/graphics/vulkan/command_processor.h>
#include <rex/hash.h>
#include <rex/logging.h>
#include <rex/ui/vulkan/util.h>

REXCVAR_DECLARE(bool, gpu_native_own_crowd_all);  // wwe2k14 patch (Session 87, crowd2): defined in command_processor.cpp
REXCVAR_DECLARE(uint32_t, gpu_d3d_wall);  // (Session 95: d3d_noring.cpp)
REXCVAR_DECLARE(bool, gpu_native_own_crowd_lean);   // wwe2k14 patch (Session 93): defined in vulkan/command_processor.cpp
REXCVAR_DECLARE(bool, gpu_native_own_crowd_merge);  // wwe2k14 patch (Session 93): defined in vulkan/command_processor.cpp
REXCVAR_DEFINE_BOOL(gpu_native_crowd_input, false, "GPU/Native",
                    "Session 95 (P7 of out\\s95\\android_structural.md): with gpu_native_vertex_input not 0, the crowd reads no vertex through a storage buffer. Own crowd's runs (the lit members and the flat "
                    "cards) take the member mesh as vertex binding 0, the members' matrices as instance-rate binding 1 (both in its ring) and host-order indices from the index pool; a run's first member, "
                    "recorded for its state only, selects no storage window and requests no vertex range; a crowd member drawn by itself goes through the attribute plan like any draw. A run the plan "
                    "refuses stays on own crowd's storage form, counted in the [vinput] line. Off: nothing changes.");

namespace rex::graphics::vulkan::native {

namespace {
constexpr uint32_t kOwnRingSlices = 4;
// (Session 87, crowd2, gpu_native_own_crowd_all: a lone draw is a run of one - up to ~460 calls a submission; and a slice keeps
// the converted meshes it made, so a mesh equal to one made earlier in the submission is not converted and stored again.
// Measured with 16 MB slices / 4096 sets (checks of 2026-11-04 04:45, entrance + match routes, 16 windows of 600 frames): a
// submission used at most 973 KB and 459 sets. Sized at twice that: 2 MB slices, 1024 sets, 8 MB of host-visible memory in
// all (the 4 MB / 512 sets before: 16 MB). A submission over it is refused and the game draws it - the census lists it
// as "handed back".)
constexpr VkDeviceSize kOwnRingSliceSizeAll = 2u << 20;
constexpr uint32_t kOwnSetsPerSliceAll = 1024;
// (Crowd2 review: with gpu_native_own_crowd_all off - the players' default path - the ring and the sets are what they were before
// crowd2: 4 MB slices, 512 sets, and no mesh dedupe.)
constexpr VkDeviceSize kOwnRingSliceSizeDefault = 4u << 20;
constexpr uint32_t kOwnSetsPerSliceDefault = 512;  // (Session 87: was 64 - the flat cards add a run per card block)

// Session 93 (gpu_native_own_crowd_lean): the buffers of a lean set are bound by dynamic offset with a fixed range (an exact range would
// need a set a call); a run whose mesh or members do not fit these is left to the game's draws. The ring is padded by the larger.
constexpr VkDeviceSize kLeanMeshRange = 512u << 10;    // 16384 lit-crowd vertices (32 B)
constexpr VkDeviceSize kLeanMembersRange = 64u << 10;  // 1365 members (48 B)
constexpr size_t kLeanSetCacheMax = 128;

// Session 95 (P7, gpu_native_crowd_input): the pipeline's vertex input (vertex_input_layout.h): the converted mesh (8 dwords a vertex, 4 for the cards), then the 12 dwords of a member, per instance.
constexpr uint32_t CrowdVi(bool card) { return (card ? 4u : 8u) | ((12u | kViInstanceRate) << 8); }
constexpr uint32_t kRestartStateBit = 1u << 4;  // (RasterKey)

uint32_t ByteSwap32(uint32_t v) {
  return (v >> 24) | ((v >> 8) & 0xFF00u) | ((v << 8) & 0xFF0000u) | (v << 24);
}

float HalfToFloat(uint16_t h) {
  const uint32_t sign = uint32_t(h & 0x8000u) << 16;
  uint32_t exponent = (h >> 10) & 0x1Fu, mantissa = h & 0x3FFu, bits;
  if (exponent == 0x1F) {
    bits = sign | 0x7F800000u | (mantissa << 13);
  } else if (exponent) {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  } else if (mantissa) {  // subnormal: normalise
    exponent = 113;
    while (!(mantissa & 0x400u)) {
      mantissa <<= 1;
      --exponent;
    }
    bits = sign | (exponent << 23) | ((mantissa & 0x3FFu) << 13);
  } else {
    bits = sign;
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}
}  // namespace

bool NativeRenderer::OwnCrowdEnsureObjects() {
  if (own_crowd_failed_) return false;
  if (own_crowd_layout_ != VK_NULL_HANDLE) return true;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  own_crowd_failed_ = true;  // (until everything below exists)
  // (Crowd2 review: the sizes by gpu_native_own_crowd_all as it stands when the objects are made.)
  const bool all = REXCVAR_GET(gpu_native_own_crowd_all);
  own_crowd_lean_ = REXCVAR_GET(gpu_native_own_crowd_lean);
  own_crowd_slice_size_ = all ? kOwnRingSliceSizeAll : kOwnRingSliceSizeDefault;
  own_crowd_sets_ = all ? kOwnSetsPerSliceAll : kOwnSetsPerSliceDefault;
  VkDescriptorSetLayoutBinding bindings[6] = {};
  const VkShaderStageFlags both = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  const VkDescriptorType types[6] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
  for (uint32_t i = 0; i < 6; ++i) bindings[i] = {i, types[i], 1, both, nullptr};
  VkDescriptorSetLayoutCreateInfo set_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  set_info.bindingCount = 6;
  set_info.pBindings = bindings;
  if (dfn.vkCreateDescriptorSetLayout(device, &set_info, nullptr, &own_crowd_set_layout_) != VK_SUCCESS) return false;
  VkPipelineLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &own_crowd_set_layout_;
  if (dfn.vkCreatePipelineLayout(device, &layout_info, nullptr, &own_crowd_layout_) != VK_SUCCESS) return false;
  if (own_crowd_lean_) {
    // (Session 93) The same set with the scene block, the mesh and the members as dynamic-offset buffers.
    VkDescriptorSetLayoutBinding dyn_bindings[6];
    std::memcpy(dyn_bindings, bindings, sizeof(bindings));
    dyn_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    dyn_bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    dyn_bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    VkDescriptorSetLayoutCreateInfo dyn_set_info = set_info;
    dyn_set_info.pBindings = dyn_bindings;
    if (dfn.vkCreateDescriptorSetLayout(device, &dyn_set_info, nullptr, &own_crowd_dyn_set_layout_) != VK_SUCCESS) {
      return false;
    }
    VkPipelineLayoutCreateInfo dyn_layout_info = layout_info;
    dyn_layout_info.pSetLayouts = &own_crowd_dyn_set_layout_;
    if (dfn.vkCreatePipelineLayout(device, &dyn_layout_info, nullptr, &own_crowd_dyn_layout_) != VK_SUCCESS) return false;
  }
  own_crowd_ring_total_ = own_crowd_slice_size_ * kOwnRingSlices + (own_crowd_lean_ ? kLeanMeshRange : 0);
  own_crowd_ring_vi_ = REXCVAR_GET(gpu_native_crowd_input) && shader_swap::VertexMirrorLatched();  // (Session 95, P7)
  if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
          vulkan_device, own_crowd_ring_total_,
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | (own_crowd_ring_vi_ ? VK_BUFFER_USAGE_VERTEX_BUFFER_BIT : VkBufferUsageFlags(0)),
          ui::vulkan::util::MemoryPurpose::kUpload, own_crowd_ring_, own_crowd_ring_memory_)) {
    return false;
  }
  if (dfn.vkMapMemory(device, own_crowd_ring_memory_, 0, VK_WHOLE_SIZE, 0, &own_crowd_ring_map_) != VK_SUCCESS) {
    return false;
  }
  const VkDescriptorPoolSize sizes[5] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, own_crowd_sets_},
                                         {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, own_crowd_sets_ * 3},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, own_crowd_sets_ * 2},
                                         {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, own_crowd_sets_},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, own_crowd_sets_ * 2}};
  VkDescriptorPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_info.maxSets = own_crowd_sets_;
  pool_info.poolSizeCount = own_crowd_lean_ ? 5 : 3;
  pool_info.pPoolSizes = sizes;
  own_crowd_slices_.resize(kOwnRingSlices);
  for (OwnCrowdSlice& slice : own_crowd_slices_) {
    if (dfn.vkCreateDescriptorPool(device, &pool_info, nullptr, &slice.pool) != VK_SUCCESS) return false;
  }
  own_crowd_failed_ = false;
  REXGPU_INFO("[native][own_crowd] objects made (ring {} x {} KB{})", kOwnRingSlices, own_crowd_slice_size_ >> 10,
              own_crowd_lean_ ? ", lean: dynamic-offset sets" : "");
  if (own_crowd_ring_vi_) REXGPU_INFO("[vinput] crowd on attributes: mesh binding 0, members instance-rate binding 1, pool indices (gpu_native_crowd_input)");
  return true;
}

void NativeRenderer::OwnCrowdDestroy() {
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  for (auto& [key, pipeline] : own_crowd_pipelines_) {
    if (pipeline != VK_NULL_HANDLE) dfn.vkDestroyPipeline(device, pipeline, nullptr);
  }
  own_crowd_pipelines_.clear();
  for (auto& [key, module] : own_crowd_modules_) dfn.vkDestroyShaderModule(device, module, nullptr);
  own_crowd_modules_.clear();
  own_crowd_module_ptr_.clear();
  own_crowd_open_ = {};
  own_crowd_lean_ = false;
  own_crowd_ring_vi_ = false;
  own_crowd_vi_index_ = {};
  for (OwnCrowdSlice& slice : own_crowd_slices_) {
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorPool, device, slice.pool);
  }
  own_crowd_slices_.clear();
  if (own_crowd_ring_map_) dfn.vkUnmapMemory(device, own_crowd_ring_memory_);
  own_crowd_ring_map_ = nullptr;
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, own_crowd_ring_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, own_crowd_ring_memory_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device, own_crowd_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device, own_crowd_dyn_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device, own_crowd_dyn_set_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device, own_crowd_set_layout_);
  own_crowd_failed_ = false;
}

VkShaderModule NativeRenderer::OwnCrowdModule(const uint32_t* words, size_t count) {
  const uint64_t key = XXH3_64bits(words, count * sizeof(uint32_t));
  auto found = own_crowd_modules_.find(key);
  if (found != own_crowd_modules_.end()) return found->second;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  VkShaderModuleCreateInfo info = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  info.codeSize = count * sizeof(uint32_t);
  info.pCode = words;
  VkShaderModule module = VK_NULL_HANDLE;
  if (vulkan_device->functions().vkCreateShaderModule(vulkan_device->device(), &info, nullptr, &module) !=
      VK_SUCCESS) {
    module = VK_NULL_HANDLE;
  }
  own_crowd_modules_[key] = module;
  return module;
}

// The member mesh: 7 big-endian dwords a vertex (position, normal, the half-float texture coordinate) to 8 floats.
void NativeRenderer::OwnCrowdConvertMesh(const OwnCrowdInput& in, float* out) {
  const uint32_t* guest = reinterpret_cast<const uint32_t*>(in.mesh_guest);
  if (in.card_mesh) {
    // Session 87: the flat cards' 4-dword vertices (position xyz, the half-float coordinate pair): only byte-swapped; our
    // card vertex shader unpacks the pair itself.
    uint32_t* words = reinterpret_cast<uint32_t*>(out);
    for (uint32_t i = 0; i < in.mesh_vertices * 4; ++i) words[i] = ByteSwap32(guest[i]);
    return;
  }
  for (uint32_t v = 0; v < in.mesh_vertices; ++v, guest += 7, out += 8) {
    uint32_t d[7];
    for (int i = 0; i < 7; ++i) d[i] = ByteSwap32(guest[i]);
    std::memcpy(out, d, 6 * sizeof(float));  // position, normal
    // The texture coordinate: the pixel shader samples (high half, low half) of the swapped dword (the game's vertex
    // shader fetches it into .yx).
    out[6] = HalfToFloat(uint16_t(d[6] >> 16));
    out[7] = HalfToFloat(uint16_t(d[6] & 0xFFFFu));
  }
}

// Session 95 (P7, gpu_native_crowd_input): the index this run draws with. The first member's plan is taken here (used once); the attribute form needs its shader, the plan, and indices that stay inside
// the mesh (an attribute read has no clamp; the storage form's shader clamps to the guest's range, and the plan proved that clamp changes nothing). Anything else: the first member's guest-order buffer
// and the storage form, as before.
NativeRenderer::OwnCrowdIndexUse NativeRenderer::OwnCrowdIndexFor(const OwnCrowdInput& in) {
  OwnCrowdIndexUse use;
  use.buffer = truth_last_index_.buffer;
  use.offset = truth_last_index_.offset;
  use.type = truth_last_index_.type;
  use.count = truth_last_index_.count;
  const OwnCrowdViIndex plan = own_crowd_vi_index_;
  own_crowd_vi_index_ = {};
  if (!own_crowd_ring_vi_) return use;
  ViRefusal why = kViRefusalCount;
  if (!plan.noted) {
    why = kViCrowd;  // (the run's first member was not held for us: no plan was made)
  } else if (!plan.ok) {
    why = ViRefusal(plan.why);
  } else if (!in.vs_vi_words) {
    why = kViNoForm;
  } else if (plan.last_vertex >= in.mesh_vertices) {
    why = kViCrowdMesh;
  }
  if (why != kViRefusalCount) {
    ++VertexInput().counters().crowd_why[why];
    return use;
  }
  use.attr = true;
  use.buffer = Mirror().buffer();
  use.offset = plan.offset;
  use.vertex_offset = plan.vertex_offset;
  use.clear_restart = plan.clear_restart;
  return use;
}

// (A run drawn: the [vinput] line's crowd part.)
void NativeRenderer::OwnCrowdNoteRun(const OwnCrowdIndexUse& use, uint32_t members) {
  if (!own_crowd_ring_vi_) return;
  VertexInputState::Counters& n = VertexInput().counters();
  if (use.attr) {
    ++n.crowd_runs;
    n.crowd_run_members += members;
  } else {
    ++n.refused[kViCrowd];
  }
}

bool NativeRenderer::RecordOwnCrowd(const OwnCrowdInput& in) {
  // Right after the run's first member, recorded as a truth draw of the same vertex shader into the open pass: our draw
  // goes into that pass with that draw's pipeline state.
  // Session 82 (P1 item 1, the owned frame): or into the owned pass the first member's state was recorded into
  // (OwnedRecord: state only, `owned_crowd_first_`) - the same pipeline (made against the truth passes' render pass,
  // compatible with the owned ones), our image and depth behind it.
  const bool in_owned = owned_crowd_first_ && owned_pending_.valid && owned_pending_.open;
  owned_crowd_first_ = false;
  own_crowd_call_sig_valid_ = own_crowd_sig_valid_;  // (Session 93: the state signature of this run's first member, used once)
  own_crowd_sig_valid_ = false;
  if (!truth_ || !(truth_pass_open_ || in_owned) || !(truth_last_recorded_ || in_owned) || !last_truth_valid_ ||
      truth_last_vs_hash_ != in.vertex_shader_hash || !truth_last_index_.indexed || truth_last_index_.attribute || !in.member_count) {  // (Session 95: `attribute`: host-order pool indices, this renderer's buffers are guest order)
    ++own_crowd_stats_.refused_state;
    return false;
  }
  if (!OwnCrowdEnsureObjects()) return false;
  const OwnCrowdIndexUse use = OwnCrowdIndexFor(in);  // (Session 95, P7)
  if (own_crowd_lean_) return RecordOwnCrowdLean(in, in_owned, use);  // (Session 93, gpu_native_own_crowd_lean)
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  // The per-submission ring slice (a slice is reused only once the GPU finished the submission that last used it).
  const uint64_t submission = command_processor_.GetCurrentSubmission();
  OwnCrowdSlice* slice = &own_crowd_slices_[own_crowd_slice_];
  if (slice->submission != submission) {
    const uint32_t next = (own_crowd_slice_ + 1) % kOwnRingSlices;
    OwnCrowdSlice& n = own_crowd_slices_[next];
    if (n.submission && n.submission > command_processor_.GetCompletedSubmission()) {
      ++own_crowd_stats_.refused_ring;
      return false;
    }
    dfn.vkResetDescriptorPool(device, n.pool, 0);
    n.used = 0;
    n.sets = 0;
    n.meshes.clear();
    ++own_crowd_stats_.submissions;
    n.submission = submission;
    own_crowd_slice_ = next;
    slice = &n;
  }
  const VkDeviceSize align = std::max<VkDeviceSize>(
      {vulkan_device->properties().minUniformBufferOffsetAlignment,
       vulkan_device->properties().minStorageBufferOffsetAlignment, VkDeviceSize(16)});
  const auto take = [&](VkDeviceSize size) -> VkDeviceSize {
    const VkDeviceSize at = (slice->used + align - 1) / align * align;
    if (at + size > own_crowd_slice_size_) return VK_WHOLE_SIZE;
    slice->used = at + size;
    return VkDeviceSize(own_crowd_slice_) * own_crowd_slice_size_ + at;
  };
  const VkDeviceSize members_size = VkDeviceSize(in.member_count) * 48;
  const VkDeviceSize mesh_size = VkDeviceSize(std::max<uint32_t>(in.mesh_vertices, 1)) * (in.card_mesh ? 16 : 32);
  // The mesh already converted in this submission (same guest bytes): its place in the ring again.
  // (Crowd2 review: only with gpu_native_own_crowd_all; off, every run converts its own copy as before.)
  const bool dedupe = REXCVAR_GET(gpu_native_own_crowd_all);
  const uint32_t mesh_words = in.mesh_vertices * (in.card_mesh ? 4 : 7);
  const uint64_t mesh_hash = !dedupe ? 0
                                     : XXH3_64bits(in.mesh_guest, size_t(mesh_words) * 4) ^
                                           (uint64_t(in.mesh_vertices) << 40) ^ (in.card_mesh ? 1ull : 0ull);
  VkDeviceSize mesh_at = VK_WHOLE_SIZE;
  for (const OwnCrowdSlice::Mesh& m : slice->meshes) {
    if (dedupe && m.hash == mesh_hash) {
      mesh_at = m.at;
      break;
    }
  }
  const bool mesh_new = mesh_at == VK_WHOLE_SIZE;
  const VkDeviceSize scene_at = take(sizeof(in.scene)), members_at = take(members_size);
  if (mesh_new) mesh_at = take(mesh_size);
  if (scene_at == VK_WHOLE_SIZE || members_at == VK_WHOLE_SIZE || mesh_at == VK_WHOLE_SIZE) {
    ++own_crowd_stats_.refused_ring;
    return false;
  }
  uint8_t* ring = static_cast<uint8_t*>(own_crowd_ring_map_);
  std::memcpy(ring + scene_at, in.scene, sizeof(in.scene));
  std::memcpy(ring + members_at, in.members, size_t(members_size));
  if (!mesh_new) ++own_crowd_stats_.mesh_shared;
  if (mesh_new) {
    ++own_crowd_stats_.mesh_converted;
    own_crowd_stats_.mesh_bytes += mesh_size;
    if (!dedupe) {
    } else if (!own_crowd_seen_meshes_.insert(mesh_hash).second) {
      ++own_crowd_stats_.mesh_repeat;
      own_crowd_stats_.mesh_repeat_bytes += mesh_size;
    } else if (own_crowd_seen_meshes_.size() > 4096) {
      own_crowd_seen_meshes_.clear();
    }
    OwnCrowdConvertMesh(in, reinterpret_cast<float*>(ring + mesh_at));
    if (dedupe && slice->meshes.size() < 32) slice->meshes.push_back({mesh_hash, mesh_at});
  }
  const VkMappedMemoryRange range = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, own_crowd_ring_memory_, 0,
                                     VK_WHOLE_SIZE};
  dfn.vkFlushMappedMemoryRanges(device, 1, &range);

  const VkShaderModule vs = use.attr ? OwnCrowdModule(in.vs_vi_words, in.vs_vi_count) : OwnCrowdModule(in.vs_words, in.vs_count);
  const VkShaderModule fs = OwnCrowdModule(in.fs_words, in.fs_count);
  if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE) {
    ++own_crowd_stats_.refused_objects;
    return false;
  }
  // The pipeline: the first member's truth pipeline key with our shaders and layout (made here, synchronously, once).
  PipelineKey key = last_truth_key_;
  key.vi = use.attr ? CrowdVi(in.card_mesh) : 0;  // (Session 95: 0 = our crowd shaders read their own buffers)
  if (use.clear_restart) key.state &= ~kRestartStateBit;
  key.vertex_shader = vs;
  key.pixel_shader = fs;
  key.layout = own_crowd_layout_;
  key.tint_hash = 0;
  key.geometry_shader = VK_NULL_HANDLE;
  VkPipeline& pipeline = own_crowd_pipelines_[key];
  if (pipeline == VK_NULL_HANDLE) {
    const float spec[3] = {};
    pipeline = CreatePipeline(Draw(), key, spec);
    if (pipeline == VK_NULL_HANDLE) {
      ++own_crowd_stats_.refused_objects;
      return false;
    }
    REXGPU_INFO("[native][own_crowd] pipeline made for vertex shader {:016X} (class {}, state {:08X})",
                in.vertex_shader_hash, key.variant & 7, key.state);
  }

  VkDescriptorSet set = VK_NULL_HANDLE;
  VkDescriptorSetAllocateInfo alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  alloc.descriptorPool = slice->pool;
  alloc.descriptorSetCount = 1;
  alloc.pSetLayouts = &own_crowd_set_layout_;
  if (dfn.vkAllocateDescriptorSets(device, &alloc, &set) != VK_SUCCESS) {
    ++own_crowd_stats_.refused_ring;
    return false;
  }
  ++slice->sets;
  own_crowd_stats_.peak_bytes = std::max<uint64_t>(own_crowd_stats_.peak_bytes, slice->used);
  own_crowd_stats_.peak_sets = std::max<uint64_t>(own_crowd_stats_.peak_sets, slice->sets);
  const VkDescriptorBufferInfo scene_info = {own_crowd_ring_, scene_at, sizeof(in.scene)};
  const VkDescriptorBufferInfo mesh_info = {own_crowd_ring_, mesh_at, mesh_size};
  const VkDescriptorBufferInfo members_info = {own_crowd_ring_, members_at, members_size};
  VkDescriptorImageInfo images[3];
  for (int i = 0; i < 3; ++i) {
    images[i] = {in.samplers[i], in.views[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  }
  VkWriteDescriptorSet writes[6] = {};
  for (uint32_t i = 0; i < 6; ++i) {
    writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[i].dstSet = set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
  }
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  writes[0].pBufferInfo = &scene_info;
  for (uint32_t i = 1; i <= 3; ++i) {
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[i].pImageInfo = &images[i - 1];
  }
  writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[4].pBufferInfo = &mesh_info;
  writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[5].pBufferInfo = &members_info;
  dfn.vkUpdateDescriptorSets(device, 6, writes, 0, nullptr);

  // The draw, in the open pass (viewport, scissor, depth bias and blend constants stay the first member's).
  OwnedCrowdCall call;
  call.pipeline = pipeline;
  call.layout = own_crowd_layout_;
  call.set = set;
  call.index_buffer = use.buffer;
  call.index_offset = use.offset;
  call.index_type = use.type;
  call.index_count = use.count;
  call.member_count = in.member_count;
  if (use.attr) {  // (Session 95, P7)
    call.vb_count = 2;
    call.vb[0] = call.vb[1] = own_crowd_ring_;
    call.vb_offsets[0] = mesh_at;
    call.vb_offsets[1] = members_at;
    call.vertex_offset = use.vertex_offset;
  }
  OwnCrowdRecordCall(call);
  if (in_owned) {
    // (Replayed into the truth images with its pass if that is lost; a committed pass is never replayed.)
    OwnedPass& p = owned_pending_;
    if (!p.committed) {
      call.after_draws = p.draws.size();
      p.crowd_calls.push_back(call);
    }
  }
  ++own_crowd_stats_.runs;
  own_crowd_stats_.members += in.member_count;
  own_crowd_stats_.vertices += in.mesh_vertices;
  if (in.card_mesh) {
    ++own_crowd_stats_.card_runs;
    own_crowd_stats_.card_members += in.member_count;
  }
  OwnCrowdNoteRun(use, in.member_count);  // (Session 95, P7)
  return true;
}

// Our crowd draw into the open pass (a truth pass or an owned one): pipeline, set, index buffer, the instanced draw.
void NativeRenderer::OwnCrowdRecordCall(const OwnedCrowdCall& call) {
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  VkDescriptorSet set = call.set;
  cb.CmdVkBindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, call.pipeline);
  if (call.dynamic) {
    cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, call.layout, 0, 1, &set, 3, call.dyn_offsets);
  } else {
    cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, call.layout, 0, 1, &set, 0, nullptr);
  }
  if (call.vb_count) cb.CmdVkBindVertexBuffers(0, call.vb_count, call.vb, call.vb_offsets);  // (Session 95, P7: the attribute form)
  cb.CmdVkBindIndexBuffer(call.index_buffer, call.index_offset, call.index_type);
  if (diag::g_armed) [[unlikely]] {  // (Session 95, the diagnostic run: no guest inputs here; its positions still show spikes)
    uint32_t aux = 0;
    if (const uint32_t id = diag::NoteOwnDraw(4, call.index_count, aux)) cb.CmdDiagMark(diag::kMarkNextDraw, id, aux);
  }
  cb.CmdVkDrawIndexed(call.index_count, call.member_count, 0, call.vertex_offset, 0);
  // Every binding the next truth draw relies on is another's now.
  truth_bound_pipeline_ = VK_NULL_HANDLE;
  truth_bound_index_buffer_ = VK_NULL_HANDLE;
  truth_bound_valid_ = false;
  truth_foreign_bind_ = true;
  truth_last_recorded_ = false;  // (nothing merges into our draw)
}

bool NativeRenderer::OwnCrowdLeanWanted() const {
  return own_crowd_lean_ || (own_crowd_layout_ == VK_NULL_HANDLE && REXCVAR_GET(gpu_native_own_crowd_lean));
}

// The shader module of one of the registered programs, by the words' address: the program's vectors are never resized or
// removed, so the address names the words for good; hashed once, then a find on the pointer.
VkShaderModule NativeRenderer::OwnCrowdModuleByPointer(const uint32_t* words, size_t count) {
  const auto found = own_crowd_module_ptr_.find(words);
  if (found != own_crowd_module_ptr_.end()) return found->second;
  const VkShaderModule module = OwnCrowdModule(words, count);
  if (module != VK_NULL_HANDLE) own_crowd_module_ptr_[words] = module;
  return module;
}

// wwe2k14 patch (Session 93, idea 5; gpu_native_own_crowd_lean / _merge): the same draw as RecordOwnCrowd with
//  - the shader modules found by pointer (no SPIR-V hashing a call),
//  - the written ring range flushed once a submission (OwnCrowdOnSubmissionEnd),
//  - a descriptor set made once for each triple of textures a submission (scene block, mesh and members by dynamic offset),
//  - the run appended to the call recorded last when everything equals and nothing was recorded after it (_merge).
bool NativeRenderer::RecordOwnCrowdLean(const OwnCrowdInput& in, bool in_owned, const OwnCrowdIndexUse& use) {
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  const uint64_t submission = command_processor_.GetCurrentSubmission();
  OwnCrowdSlice* slice = &own_crowd_slices_[own_crowd_slice_];
  if (slice->submission != submission) {
    const uint32_t next = (own_crowd_slice_ + 1) % kOwnRingSlices;
    OwnCrowdSlice& n = own_crowd_slices_[next];
    if (n.submission && n.submission > command_processor_.GetCompletedSubmission()) {
      ++own_crowd_stats_.refused_ring;
      return false;
    }
    dfn.vkResetDescriptorPool(device, n.pool, 0);
    n.used = 0;
    n.sets = 0;
    n.meshes.clear();
    n.set_cache.clear();
    n.dirty_lo = VK_WHOLE_SIZE;
    n.dirty_hi = 0;
    ++own_crowd_stats_.submissions;
    n.submission = submission;
    own_crowd_slice_ = next;
    slice = &n;
  }
  const VkDeviceSize members_size = VkDeviceSize(in.member_count) * 48;
  const VkDeviceSize mesh_size = VkDeviceSize(std::max<uint32_t>(in.mesh_vertices, 1)) * (in.card_mesh ? 16 : 32);
  if (members_size > kLeanMembersRange || mesh_size > kLeanMeshRange) {
    ++own_crowd_lean_stats_.size_declines;  // (the game draws this run)
    return false;
  }
  const bool dedupe = REXCVAR_GET(gpu_native_own_crowd_all);
  const uint32_t mesh_words = in.mesh_vertices * (in.card_mesh ? 4 : 7);
  // (Session 95, gpu_d3d_wall bit 32) The mesh's hash once a frame per mesh, not once a run (the same few meshes, ~60 runs a frame; the bytes of a mesh in use do not change inside a frame).
  uint64_t mesh_bytes_hash = 0;
  if (dedupe) {
    struct KeptHash {
      const void* at = nullptr;
      uint32_t words = 0;
      uint64_t frame = UINT64_MAX, hash = 0;
    };
    static KeptHash kept[256];
    KeptHash& k = kept[(reinterpret_cast<uintptr_t>(in.mesh_guest) >> 4) & 255];
    if ((REXCVAR_GET(gpu_d3d_wall) & 32u) && k.at == in.mesh_guest && k.words == mesh_words && k.frame == frame_) {
      mesh_bytes_hash = k.hash;
    } else {
      mesh_bytes_hash = XXH3_64bits(in.mesh_guest, size_t(mesh_words) * 4);
      k.at = in.mesh_guest;
      k.words = mesh_words;
      k.frame = frame_;
      k.hash = mesh_bytes_hash;
    }
  }
  const uint64_t mesh_hash = !dedupe ? 0 : mesh_bytes_hash ^ (uint64_t(in.mesh_vertices) << 40) ^ (in.card_mesh ? 1ull : 0ull);

  const VkShaderModule vs = use.attr ? OwnCrowdModuleByPointer(in.vs_vi_words, in.vs_vi_count) : OwnCrowdModuleByPointer(in.vs_words, in.vs_count);
  const VkShaderModule fs = OwnCrowdModuleByPointer(in.fs_words, in.fs_count);
  if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE) {
    ++own_crowd_stats_.refused_objects;
    return false;
  }
  PipelineKey key = last_truth_key_;
  key.vi = use.attr ? CrowdVi(in.card_mesh) : 0;  // (Session 95: 0 = our crowd shaders read their own buffers)
  if (use.clear_restart) key.state &= ~kRestartStateBit;
  key.vertex_shader = vs;
  key.pixel_shader = fs;
  key.layout = own_crowd_dyn_layout_;
  key.tint_hash = 0;
  key.geometry_shader = VK_NULL_HANDLE;

  // The call recorded last: equal in everything but the members (the counters compare even with the merge off; the merge also
  // needs the stream and the ring to be as that call left them).
  OwnCrowdOpen& open = own_crowd_open_;
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const bool merge_on = dedupe && REXCVAR_GET(gpu_native_own_crowd_merge);
  bool equal = false;
  if (dedupe && open.valid && open.submission == submission && open.slice == own_crowd_slice_ &&
      open.in_owned == in_owned && open.mesh_hash == mesh_hash && open.key == key &&
      open.index_buffer == use.buffer && open.index_offset == use.offset && open.index_type == use.type && open.index_count == use.count && open.vertex_offset == use.vertex_offset &&
      std::memcmp(open.views, in.views, sizeof(open.views)) == 0 &&
      std::memcmp(open.samplers, in.samplers, sizeof(open.samplers)) == 0 &&
      std::memcmp(open.scene, in.scene, sizeof(open.scene)) == 0) {
    equal = true;
    ++own_crowd_lean_stats_.eq_prev;
  }
  const bool adjacent = equal && cb.stream_offset() == open.stream_end;
  if (adjacent) ++own_crowd_lean_stats_.eq_prev_adjacent;
  // The merge (Session 93): this run's first member recorded its own state (pipeline, sets, viewport ...) after the call recorded
  // last, and that state is turned into no-ops when it equals what the call drew with (the signatures) and the stream holds nothing
  // but such state since the call (MergeInstancesIntoLastIndexedDraw). The members then join the call's instances.
  if (merge_on && equal) {
    if (!(open.sig_valid && own_crowd_call_sig_valid_ && open.sig.Equals(own_crowd_sig_))) {
      ++own_crowd_lean_stats_.merge_no_state;
    } else if (slice->used != open.members_end_rel ||
               VkDeviceSize(open.member_count + in.member_count) * 48 > kLeanMembersRange ||
               slice->used + members_size > own_crowd_slice_size_) {
      ++own_crowd_lean_stats_.merge_no_fit;
    } else {
      // (An owned pass's call list: the open call is the list's last, and at most its first member's state-only draw was kept after
      // it. A committed pass keeps no list - nothing replays it.)
      bool owned_ok = true, list_refused = false;
      OwnedPass& p = owned_pending_;
      const bool list = in_owned && !p.committed;
      // (The call list's last entry, when it is the open call.)
      const bool list_match = in_owned && p.valid && p.open && !p.crowd_calls.empty() &&
                              p.crowd_calls.back().set == open.set &&
                              p.crowd_calls.back().dyn_offsets[2] == uint32_t(open.members_at) &&
                              p.crowd_calls.back().member_count == open.member_count;
      if (in_owned) {
        owned_ok = p.valid && p.open;
        if (owned_ok && list) {
          owned_ok = list_match && p.draws.size() >= p.crowd_calls.back().after_draws;
          // (Every draw kept after the call must be a crowd first member: state only, replayed with no call after it.)
          for (size_t j = owned_ok ? p.crowd_calls.back().after_draws : 0; owned_ok && j < p.draws.size(); ++j) {
            if (!p.draws[j].own_skip_call) owned_ok = false;
          }
          list_refused = !owned_ok;
        }
      }
      bool refused_stream = false;
      if (!owned_ok) {
        if (list_refused) ++own_crowd_lean_stats_.merge_no_list;
        else ++own_crowd_lean_stats_.merge_no_fit;
      } else if (cb.MergeInstancesIntoLastIndexedDraw(in.member_count, open.index_count, open.stream_end, refused_stream)) {
        uint8_t* ring = static_cast<uint8_t*>(own_crowd_ring_map_);
        const VkDeviceSize at = open.members_at + VkDeviceSize(open.member_count) * 48;
        std::memcpy(ring + at, in.members, size_t(members_size));
        slice->used += members_size;
        slice->dirty_lo = std::min(slice->dirty_lo, at);
        slice->dirty_hi = std::max(slice->dirty_hi, at + members_size);
        open.member_count += in.member_count;
        open.members_end_rel = slice->used;
        if (list || (in_owned && list_match)) p.crowd_calls.back().member_count = open.member_count;
        // Everything the next draw relies on is another's, as after OwnCrowdRecordCall (the first member's binds are no-ops now,
        // the GPU holds our call's).
        truth_bound_pipeline_ = VK_NULL_HANDLE;
        truth_bound_index_buffer_ = VK_NULL_HANDLE;
        truth_bound_valid_ = false;
        truth_foreign_bind_ = true;
        truth_last_recorded_ = false;
        ++own_crowd_lean_stats_.merged_runs;
        own_crowd_lean_stats_.merged_members += in.member_count;
        ++own_crowd_stats_.runs;
        own_crowd_stats_.members += in.member_count;
        ++own_crowd_stats_.mesh_shared;
        own_crowd_stats_.vertices += in.mesh_vertices;
        if (in.card_mesh) {
          ++own_crowd_stats_.card_runs;
          own_crowd_stats_.card_members += in.member_count;
        }
        OwnCrowdNoteRun(use, in.member_count);  // (Session 95, P7)
        return true;
      } else if (refused_stream) {
        ++own_crowd_lean_stats_.merge_no_stream;
      } else {
        ++own_crowd_lean_stats_.merge_no_fit;
      }
    }
  }

  const VkDeviceSize align = std::max<VkDeviceSize>(
      {vulkan_device->properties().minUniformBufferOffsetAlignment,
       vulkan_device->properties().minStorageBufferOffsetAlignment, VkDeviceSize(16)});
  const auto take = [&](VkDeviceSize size) -> VkDeviceSize {
    const VkDeviceSize at = (slice->used + align - 1) / align * align;
    if (at + size > own_crowd_slice_size_) return VK_WHOLE_SIZE;
    slice->used = at + size;
    return VkDeviceSize(own_crowd_slice_) * own_crowd_slice_size_ + at;
  };
  VkDeviceSize mesh_at = VK_WHOLE_SIZE;
  for (const OwnCrowdSlice::Mesh& m : slice->meshes) {
    if (dedupe && m.hash == mesh_hash) {
      mesh_at = m.at;
      break;
    }
  }
  const bool mesh_new = mesh_at == VK_WHOLE_SIZE;
  // (The members last: a following run of the same call appends to them.)
  const VkDeviceSize scene_at = take(sizeof(in.scene));
  if (mesh_new) mesh_at = take(mesh_size);
  const VkDeviceSize members_at = take(members_size);
  if (scene_at == VK_WHOLE_SIZE || members_at == VK_WHOLE_SIZE || mesh_at == VK_WHOLE_SIZE) {
    ++own_crowd_stats_.refused_ring;
    return false;
  }
  uint8_t* ring = static_cast<uint8_t*>(own_crowd_ring_map_);
  std::memcpy(ring + scene_at, in.scene, sizeof(in.scene));
  std::memcpy(ring + members_at, in.members, size_t(members_size));
  VkDeviceSize lo = std::min(scene_at, members_at),
               hi = std::max(scene_at + sizeof(in.scene), members_at + members_size);
  if (!mesh_new) ++own_crowd_stats_.mesh_shared;
  if (mesh_new) {
    ++own_crowd_stats_.mesh_converted;
    own_crowd_stats_.mesh_bytes += mesh_size;
    if (!dedupe) {
    } else if (!own_crowd_seen_meshes_.insert(mesh_hash).second) {
      ++own_crowd_stats_.mesh_repeat;
      own_crowd_stats_.mesh_repeat_bytes += mesh_size;
    } else if (own_crowd_seen_meshes_.size() > 4096) {
      own_crowd_seen_meshes_.clear();
    }
    OwnCrowdConvertMesh(in, reinterpret_cast<float*>(ring + mesh_at));
    if (dedupe && slice->meshes.size() < 32) slice->meshes.push_back({mesh_hash, mesh_at});
    lo = std::min(lo, mesh_at);
    hi = std::max(hi, mesh_at + mesh_size);
  }
  slice->dirty_lo = std::min(slice->dirty_lo, lo);
  slice->dirty_hi = std::max(slice->dirty_hi, hi);

  VkPipeline& pipeline = own_crowd_pipelines_[key];
  if (pipeline == VK_NULL_HANDLE) {
    const float spec[3] = {};
    pipeline = CreatePipeline(Draw(), key, spec);
    if (pipeline == VK_NULL_HANDLE) {
      ++own_crowd_stats_.refused_objects;
      return false;
    }
    REXGPU_INFO("[native][own_crowd] pipeline made for vertex shader {:016X} (class {}, state {:08X}, lean)",
                in.vertex_shader_hash, key.variant & 7, key.state);
  }

  // The set: one for each triple of textures (and their samplers) in this submission.
  VkDescriptorSet set = VK_NULL_HANDLE;
  for (const OwnCrowdSlice::SetEntry& e : slice->set_cache) {
    if (std::memcmp(e.views, in.views, sizeof(e.views)) == 0 &&
        std::memcmp(e.samplers, in.samplers, sizeof(e.samplers)) == 0) {
      set = e.set;
      break;
    }
  }
  if (set != VK_NULL_HANDLE) {
    ++own_crowd_lean_stats_.sets_reused;
  } else {
    VkDescriptorSetAllocateInfo alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = slice->pool;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &own_crowd_dyn_set_layout_;
    if (dfn.vkAllocateDescriptorSets(device, &alloc, &set) != VK_SUCCESS) {
      ++own_crowd_stats_.refused_ring;
      return false;
    }
    ++slice->sets;
    ++own_crowd_lean_stats_.sets_made;
    own_crowd_stats_.peak_sets = std::max<uint64_t>(own_crowd_stats_.peak_sets, slice->sets);
    const VkDescriptorBufferInfo scene_info = {own_crowd_ring_, 0, sizeof(in.scene)};
    const VkDescriptorBufferInfo mesh_info = {own_crowd_ring_, 0, kLeanMeshRange};
    const VkDescriptorBufferInfo members_info = {own_crowd_ring_, 0, kLeanMembersRange};
    VkDescriptorImageInfo images[3];
    for (int i = 0; i < 3; ++i) {
      images[i] = {in.samplers[i], in.views[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    }
    VkWriteDescriptorSet writes[6] = {};
    for (uint32_t i = 0; i < 6; ++i) {
      writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      writes[i].dstSet = set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
    }
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    writes[0].pBufferInfo = &scene_info;
    for (uint32_t i = 1; i <= 3; ++i) {
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[i].pImageInfo = &images[i - 1];
    }
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    writes[4].pBufferInfo = &mesh_info;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    writes[5].pBufferInfo = &members_info;
    dfn.vkUpdateDescriptorSets(device, 6, writes, 0, nullptr);
    if (slice->set_cache.size() < kLeanSetCacheMax) {
      OwnCrowdSlice::SetEntry e;
      std::memcpy(e.views, in.views, sizeof(e.views));
      std::memcpy(e.samplers, in.samplers, sizeof(e.samplers));
      e.set = set;
      slice->set_cache.push_back(e);
    }
  }
  own_crowd_stats_.peak_bytes = std::max<uint64_t>(own_crowd_stats_.peak_bytes, slice->used);

  OwnedCrowdCall call;
  call.pipeline = pipeline;
  call.layout = own_crowd_dyn_layout_;
  call.set = set;
  call.index_buffer = use.buffer;
  call.index_offset = use.offset;
  call.index_type = use.type;
  call.index_count = use.count;
  call.member_count = in.member_count;
  call.dynamic = true;
  call.dyn_offsets[0] = uint32_t(scene_at);
  call.dyn_offsets[1] = uint32_t(mesh_at);
  call.dyn_offsets[2] = uint32_t(members_at);
  if (use.attr) {  // (Session 95, P7: the set stays as it is, the form's shader reads no binding 4 or 5; a merge appends members behind members_at, the binding's offset does not move)
    call.vb_count = 2;
    call.vb[0] = call.vb[1] = own_crowd_ring_;
    call.vb_offsets[0] = mesh_at;
    call.vb_offsets[1] = members_at;
    call.vertex_offset = use.vertex_offset;
  }
  OwnCrowdRecordCall(call);
  if (in_owned) {
    OwnedPass& p = owned_pending_;
    if (!p.committed) {
      call.after_draws = p.draws.size();
      p.crowd_calls.push_back(call);
    }
  }
  open.valid = true;
  open.sig_valid = own_crowd_call_sig_valid_;
  open.sig = own_crowd_sig_;
  open.in_owned = in_owned;
  open.submission = submission;
  open.mesh_hash = mesh_hash;
  open.slice = own_crowd_slice_;
  open.key = key;
  open.pipeline = pipeline;
  std::memcpy(open.views, in.views, sizeof(open.views));
  std::memcpy(open.samplers, in.samplers, sizeof(open.samplers));
  std::memcpy(open.scene, in.scene, sizeof(open.scene));
  open.index_buffer = call.index_buffer;
  open.index_offset = call.index_offset;
  open.index_type = call.index_type;
  open.index_count = call.index_count;
  open.vertex_offset = call.vertex_offset;
  open.member_count = in.member_count;
  open.members_at = members_at;
  open.members_end_rel = slice->used;
  open.stream_end = cb.stream_offset();
  open.set = set;
  ++own_crowd_lean_stats_.calls;
  ++own_crowd_stats_.runs;
  own_crowd_stats_.members += in.member_count;
  own_crowd_stats_.vertices += in.mesh_vertices;
  if (in.card_mesh) {
    ++own_crowd_stats_.card_runs;
    own_crowd_stats_.card_members += in.member_count;
  }
  OwnCrowdNoteRun(use, in.member_count);  // (Session 95, P7)
  return true;
}

// Session 93 (gpu_native_own_crowd_lean): the GPU reads the ring only once the submission is sent, so the ranges written are flushed
// here, once for each slice that has any (the legacy call flushed the whole buffer every call).
void NativeRenderer::OwnCrowdOnSubmissionEnd() {
  if (!own_crowd_lean_ || own_crowd_ring_memory_ == VK_NULL_HANDLE) return;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const VkDeviceSize atom = std::max<VkDeviceSize>(vulkan_device->properties().nonCoherentAtomSize, 1);
  for (uint32_t i = 0; i < own_crowd_slices_.size(); ++i) {
    OwnCrowdSlice& s = own_crowd_slices_[i];
    if (s.dirty_hi <= s.dirty_lo) continue;
    VkMappedMemoryRange range = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, own_crowd_ring_memory_, 0, VK_WHOLE_SIZE};
    range.offset = s.dirty_lo / atom * atom;
    const VkDeviceSize end = (s.dirty_hi + atom - 1) / atom * atom;
    range.size = end <= own_crowd_ring_total_ ? end - range.offset : VK_WHOLE_SIZE;
    vulkan_device->functions().vkFlushMappedMemoryRanges(vulkan_device->device(), 1, &range);
    ++own_crowd_lean_stats_.flushes;
    own_crowd_lean_stats_.flush_kb += (s.dirty_hi - s.dirty_lo + 1023) >> 10;
    s.dirty_lo = VK_WHOLE_SIZE;
    s.dirty_hi = 0;
  }
}

void NativeRenderer::OwnCrowdOnFrameEnd(bool from_hook) {
  if (from_hook) own_crowd_hook_seen_ = true;
  else if (own_crowd_hook_seen_) return;
  if (++own_crowd_stats_.frames >= 600) {
    if (own_crowd_stats_.runs || own_crowd_stats_.refused_state || own_crowd_stats_.refused_ring ||
        own_crowd_stats_.refused_objects) {
      REXGPU_INFO("[native][own_crowd] {} frames: {} runs, {} members drawn by us ({} mesh vertices converted; the flat "
                  "cards D5E2FCC9F76BC40C among them: {} runs, {} members); refused: state {}, ring {}, objects {}; "
                  "ring peak a submission: {} KB of {} KB, {} sets of {}; submissions {}; meshes converted {} ({} KB), "
                  "shared inside a submission {}, repeated from an earlier submission {} ({} KB)",
                  own_crowd_stats_.frames, own_crowd_stats_.runs, own_crowd_stats_.members, own_crowd_stats_.vertices,
                  own_crowd_stats_.card_runs, own_crowd_stats_.card_members, own_crowd_stats_.refused_state,
                  own_crowd_stats_.refused_ring, own_crowd_stats_.refused_objects, own_crowd_stats_.peak_bytes >> 10,
                  own_crowd_slice_size_ >> 10, own_crowd_stats_.peak_sets, own_crowd_sets_, own_crowd_stats_.submissions,
                  own_crowd_stats_.mesh_converted, own_crowd_stats_.mesh_bytes >> 10, own_crowd_stats_.mesh_shared,
                  own_crowd_stats_.mesh_repeat, own_crowd_stats_.mesh_repeat_bytes >> 10);
    }
    if (own_crowd_lean_ && (own_crowd_lean_stats_.calls || own_crowd_lean_stats_.merged_runs || own_crowd_lean_stats_.size_declines)) {
      REXGPU_INFO("[crowd][lean] {} frames: calls {}, merged runs {} ({} members; the merge is {}), runs equal to the previous call {} "
                  "(of them adjacent in the command stream {}), descriptor sets made {} / reused {}, ring flushes {} ({} KB), program "
                  "lookups without the scan {}, runs declined for size (mesh over 512 KB / members over 1365) {}, "
                  "equal runs not merged: first member's state differs {}, other commands in the stream {}, ring {}, owned call list {}",
                  own_crowd_stats_.frames, own_crowd_lean_stats_.calls, own_crowd_lean_stats_.merged_runs,
                  own_crowd_lean_stats_.merged_members, REXCVAR_GET(gpu_native_own_crowd_merge) ? "on" : "off",
                  own_crowd_lean_stats_.eq_prev, own_crowd_lean_stats_.eq_prev_adjacent, own_crowd_lean_stats_.sets_made,
                  own_crowd_lean_stats_.sets_reused, own_crowd_lean_stats_.flushes, own_crowd_lean_stats_.flush_kb,
                  own_crowd_lean_stats_.program_hits, own_crowd_lean_stats_.size_declines, own_crowd_lean_stats_.merge_no_state,
                  own_crowd_lean_stats_.merge_no_stream, own_crowd_lean_stats_.merge_no_fit, own_crowd_lean_stats_.merge_no_list);
    }
    if (own_crowd_lean_ && REXCVAR_GET(gpu_native_own_crowd_merge)) {
      // (log only) the command that refused each merge, top 6 kinds
      DeferredCommandBuffer& rcb = command_processor_.deferred_command_buffer();
      std::vector<std::pair<uint64_t, size_t>> kinds;
      for (size_t k = 0; k < DeferredCommandBuffer::kMergeRefusalKinds; ++k) {
        if (rcb.merge_refused_by_[k]) kinds.push_back({rcb.merge_refused_by_[k], k});
        rcb.merge_refused_by_[k] = 0;
      }
      std::sort(kinds.begin(), kinds.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      std::string text;
      for (size_t i = 0; i < kinds.size() && i < 6; ++i) {
        text += fmt::format("{}{} {}", i ? ", " : "", DeferredCommandBuffer::CommandName(kinds[i].second), kinds[i].first);
      }
      if (!text.empty()) REXGPU_INFO("[crowd][lean] merge refused by: {}", text);
    }
    own_crowd_stats_ = {};
    own_crowd_lean_stats_ = {};
  }
}

}  // namespace rex::graphics::vulkan::native
