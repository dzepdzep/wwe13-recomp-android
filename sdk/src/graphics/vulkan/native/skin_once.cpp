// wwe2k14 patch (Session 95, gpu_native_skin_once): each wrestler vertex skinned once a frame. See skin_once.h.

#include "skin_once.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

#include <rex/cvar.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/deferred_command_buffer.h>
#include <rex/hash.h>
#include <rex/logging.h>
#include <rex/system/xmemory.h>
#include <rex/ui/vulkan/util.h>

#include "shader_swap.h"

REXCVAR_DEFINE_BOOL(gpu_native_skin_bind_lean, false, "GPU/Native",
                    "Session 95 (r3gpu, default off; with gpu_native_skin_once): a skin-once shadow or rim draw whose shader reads the same constants as the draw before it (they differ only in the bones, "
                    "which skin once has used already) keeps that draw's constant slots: no descriptor bind between the draws of a shadow pass. Off = one bind a draw");
REXCVAR_DEFINE_BOOL(gpu_native_skin_with_input, false, "GPU/Native",
                    "Session 95 (r3gpu, default off, read at start; with gpu_native_skin_once and gpu_native_vertex_input): the two together: skin once writes the rows, the listed shaders' _so variants read "
                    "them as vertex attributes, a draw skin once refuses before it takes a job stays an attribute draw (one refused after that, by the claim or a pipeline still being made, goes to the plain draw, whose pipeline may be missing too: the _so pipelines are not stored). Off = skin once stands down while vertex input is on");
REXCVAR_DEFINE_BOOL(gpu_native_skin_once, false, "GPU/Native",
                    "Session 95 (read at start; needs gpu_native_vs_swap and the typed draw): each wrestler vertex is skinned once a frame by one compute dispatch into a vertex buffer the shadow, rim and "
                    "main passes read through vertex input (vulkan/native/skin_once.h), instead of every pass running the blend in its vertex shader. Off by default");

namespace rex::graphics::vulkan::native {

namespace {
constexpr const char* kRefusedNames[SkinOnce::kRefCount] = {"special", "base-ext", "not-fresh", "index", "fetch", "no-variant", "mirror", "init", "params-full", "out-full", "pipeline-pending", "claim-refused"};
inline uint64_t Mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  return h * 0xC2B2AE3D27D4EB4Full;
}
}  // namespace

SkinOnce::SkinOnce(VulkanCommandProcessor& cp) : cp_(cp) {}

SkinOnce::~SkinOnce() {
  const ui::vulkan::VulkanDevice* device = cp_.GetVulkanDevice();
  if (!device) return;
  const ui::vulkan::VulkanDevice::Functions& dfn = device->functions();
  const VkDevice d = device->device();
  if (pipeline_) dfn.vkDestroyPipeline(d, pipeline_, nullptr);
  if (module_) dfn.vkDestroyShaderModule(d, module_, nullptr);
  if (pool_) dfn.vkDestroyDescriptorPool(d, pool_, nullptr);
  if (pipeline_layout_) dfn.vkDestroyPipelineLayout(d, pipeline_layout_, nullptr);
  if (set_layout_) dfn.vkDestroyDescriptorSetLayout(d, set_layout_, nullptr);
  if (params_map_) dfn.vkUnmapMemory(d, params_memory_);
  if (params_buffer_) dfn.vkDestroyBuffer(d, params_buffer_, nullptr);
  if (params_memory_) dfn.vkFreeMemory(d, params_memory_, nullptr);
  if (out_buffer_) dfn.vkDestroyBuffer(d, out_buffer_, nullptr);
  if (out_memory_) dfn.vkFreeMemory(d, out_memory_, nullptr);
}

bool SkinOnce::EnsureInit(VkBuffer mirror_buffer) {
  if (ready_) return true;
  if (init_failed_) return false;
  init_failed_ = true;
  const ui::vulkan::VulkanDevice* device = cp_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = device->functions();
  const VkDevice d = device->device();
  size_t word_count = 0;
  const uint32_t* words = shader_swap::SkinOnceCompute(&word_count);
  if (!words) {
    REXGPU_WARN("[skin1] no compute shader registered: skin once stays off");
    return false;
  }
  const VkDeviceSize out_bytes = VkDeviceSize(kOutVertices) * 48, params_bytes = VkDeviceSize(kSlots) * kSlotRows * 16;
  if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(device, out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, ui::vulkan::util::MemoryPurpose::kDeviceLocal,
                                                         out_buffer_, out_memory_) ||
      !ui::vulkan::util::CreateDedicatedAllocationBuffer(device, params_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ui::vulkan::util::MemoryPurpose::kUpload, params_buffer_, params_memory_)) {
    REXGPU_WARN("[skin1] the buffers could not be made: skin once stays off");
    return false;
  }
  void* mapped = nullptr;
  if (dfn.vkMapMemory(d, params_memory_, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) return false;
  params_map_ = static_cast<uint32_t*>(mapped);

  VkDescriptorSetLayoutBinding bindings[4] = {};
  for (uint32_t i = 0; i < 4; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo set_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  set_info.bindingCount = 4;
  set_info.pBindings = bindings;
  if (dfn.vkCreateDescriptorSetLayout(d, &set_info, nullptr, &set_layout_) != VK_SUCCESS) return false;
  VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
  VkPipelineLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &set_layout_;
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &push;
  if (dfn.vkCreatePipelineLayout(d, &layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) return false;
  VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
  VkDescriptorPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  if (dfn.vkCreateDescriptorPool(d, &pool_info, nullptr, &pool_) != VK_SUCCESS) return false;
  VkDescriptorSetAllocateInfo alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  alloc.descriptorPool = pool_;
  alloc.descriptorSetCount = 1;
  alloc.pSetLayouts = &set_layout_;
  if (dfn.vkAllocateDescriptorSets(d, &alloc, &set_) != VK_SUCCESS) return false;
  const VkDescriptorBufferInfo infos[4] = {{mirror_buffer, 0, VK_WHOLE_SIZE}, {out_buffer_, 0, VK_WHOLE_SIZE}, {params_buffer_, 0, VK_WHOLE_SIZE}, {params_buffer_, 0, VK_WHOLE_SIZE}};
  VkWriteDescriptorSet writes[4] = {};
  for (uint32_t i = 0; i < 4; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = set_;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  dfn.vkUpdateDescriptorSets(d, 4, writes, 0, nullptr);
  VkShaderModuleCreateInfo module_info = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  module_info.codeSize = word_count * 4;
  module_info.pCode = words;
  if (dfn.vkCreateShaderModule(d, &module_info, nullptr, &module_) != VK_SUCCESS) return false;
  VkComputePipelineCreateInfo pipe_info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipe_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipe_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipe_info.stage.module = module_;
  pipe_info.stage.pName = "main";
  pipe_info.layout = pipeline_layout_;
  if (dfn.vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &pipe_info, nullptr, &pipeline_) != VK_SUCCESS) {
    REXGPU_WARN("[skin1] the compute pipeline could not be made: skin once stays off");
    return false;
  }
  init_failed_ = false;
  ready_ = true;
  return true;
}

void SkinOnce::Tick(uint64_t frame, uint64_t submission, uint64_t completed) {
  submission_ = submission;
  completed_ = completed;
  if (frame == frame_) return;
  frame_ = frame;
  jobs_.clear();
  // The frame's output space is in use up to the submission now open (a job can be read by any submission of its frame).
  for (auto it = segments_.rbegin(); it != segments_.rend() && it->submission == UINT64_MAX; ++it) it->submission = submission;
  Summary(frame);
}

// The palette rows a vertex range needs: 3 x (its largest USED bone byte + 1), scanned once per range (a mirror entry made again has a new place, and is scanned again). A byte whose blend weight is
// nothing does not count: run s95-skin1 001 read every byte and came to 153 rows a job on average against the real palettes' 4-86 (WRESTLER_DRAWS_MAP.md section 10), so the stale registers past
// the palette went into the hash and 136 jobs a frame served 158 draws. Such a bone gets the job's last matrix in the compute shader, weighted by nothing.
uint32_t SkinOnce::PaletteRows(const Request& q, uint32_t bones) {
  const uint32_t blend = (q.bools & 8u) ? 4 : (q.bools & 4u) ? 3 : (q.bools & 2u) ? 2 : 1;  // the position's blend; bones above it are the tangent's (b4)
  uint64_t key = Mix(Mix(Mix(uint64_t(q.address), q.mirror_offset), (uint64_t(q.first) << 32) | q.count), (uint64_t(q.stride_words) << 16) | (blend << 8) | bones);
  const auto it = scanned_.find(key);
  if (it != scanned_.end()) return it->second;
  if (scanned_.size() > 65536) scanned_.clear();
  const bool weighted = q.stride_words >= 11;
  const uint32_t dword = weighted ? 7 : 4;
  const uint32_t* src = q.mem->TranslatePhysical<const uint32_t*>(q.address + q.first * q.stride_words * 4);
  constexpr float kNothing = 1.0f / 65536.0f;
  uint32_t top = 0;
  for (uint32_t i = 0; i < q.count; ++i) {
    const uint32_t* vertex = src + size_t(i) * q.stride_words;
    float wa = 0.0f, wc = 0.0f, wb = 0.0f;
    if (weighted) {
      wa = std::bit_cast<float>(std::byteswap(vertex[4]));
      wc = std::bit_cast<float>(std::byteswap(vertex[5]));
      wb = std::bit_cast<float>(std::byteswap(vertex[6]));
    }
    // What each bone byte weighs (vs_ch.glsl's ChSkin; the normal's third bone gets what wa and wc leave).
    float w[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    if (blend == 4) {
      w[0] = wa, w[1] = wc, w[2] = 1.0f - wa - wc, w[3] = 1.0f - wa - wc - wb;
    } else if (blend == 3) {
      w[0] = wa, w[1] = wc, w[2] = 1.0f - wa - wc;
    } else if (blend == 2) {
      w[0] = wa, w[1] = 1.0f - wa;
    }
    const uint32_t v = std::byteswap(vertex[dword]);
    for (uint32_t k = 0; k < bones; ++k) {
      if (!(std::abs(w[k]) < kNothing)) top = std::max(top, (v >> (8 * k)) & 0xFFu);  // (a NaN weight counts)
    }
  }
  const uint32_t rows = std::min(3 * (top + 1), 180u);
  scanned_.emplace(key, rows);
  return rows;
}

bool SkinOnce::AllocateOut(uint32_t vertices, uint32_t* at) {
  while (!segments_.empty() && segments_.front().submission <= completed_) segments_.pop_front();
  if (vertices >= kOutVertices) return false;
  if (segments_.empty()) {
    if (out_head_ + vertices > kOutVertices) out_head_ = 0;
  } else {
    const uint32_t tail = segments_.front().start;
    if (out_head_ >= tail) {  // free: [head, end) and [0, tail)
      if (out_head_ + vertices > kOutVertices) {
        if (vertices >= tail) return false;
        out_head_ = 0;
      }
    } else if (out_head_ + vertices >= tail) {  // free: [head, tail)
      return false;
    }
  }
  *at = out_head_;
  if (!segments_.empty() && segments_.back().end == out_head_ && segments_.back().submission == UINT64_MAX) {
    segments_.back().end = out_head_ + vertices;
  } else {
    segments_.push_back({out_head_, out_head_ + vertices, UINT64_MAX});
  }
  out_head_ += vertices;
  return true;
}

bool SkinOnce::OpenSlot() {
  if (slot_ >= 0) return true;
  for (uint32_t i = 0; i < kSlots; ++i) {
    if (slot_submission_[i] <= completed_) {
      slot_ = int32_t(i);
      slot_rows_ = 0;
      chunks_ = 0;
      slot_submission_[i] = UINT64_MAX;
      return true;
    }
  }
  return false;
}

void SkinOnce::WriteJobFlags(const Job& j) {
  uint32_t* row = params_map_ + (size_t(j.row) + 1) * 4;
  row[0] = j.flags;
  row[2] = j.scale_bits;
}

bool SkinOnce::MakeJob(const Request& q, uint64_t key, uint32_t palette_rows, VkDeviceSize* offset) {
  if (!OpenSlot()) {
    ++window_.refused[kRefParams];
    return false;
  }
  const uint32_t chunks = (q.count + kChunk - 1) / kChunk;
  if (slot_rows_ + 2 + palette_rows + chunks_ + chunks + 1 > kSlotRows) {
    ++window_.refused[kRefParams];
    return false;
  }
  uint32_t at = 0;
  if (!AllocateOut(q.count, &at)) {
    ++window_.refused[kRefOut];
    return false;
  }
  const uint32_t slot_base = uint32_t(slot_) * kSlotRows;
  Job j;
  j.out_vertex = at;
  j.row = slot_base + slot_rows_;
  j.palette = j.row + 2;
  j.serial = serial_;
  j.flags = q.bools & (q.main ? 0x1Eu : 0x0Eu);
  j.scale_bits = 0x3F800000u;
  if (q.main) {
    j.main_set = true;
    j.flags |= q.style ? 0x100u : 0u;
    j.scale_bits = q.scale_bits;
  }
  uint32_t* row0 = params_map_ + size_t(j.row) * 4;
  row0[0] = uint32_t((q.mirror_offset >> 2) + uint64_t(q.first) * q.stride_words);
  row0[1] = q.stride_words;
  row0[2] = q.count;
  row0[3] = at;
  row0[5] = uint32_t(int32_t(j.palette) - 76);
  row0[7] = palette_rows;
  WriteJobFlags(j);
  std::memcpy(params_map_ + size_t(j.palette) * 4, q.vs_float + 4 * 76, size_t(palette_rows) * 16);
  slot_rows_ += 2 + palette_rows;
  for (uint32_t c = 0; c < chunks; ++c) {
    uint32_t* chunk = params_map_ + size_t(slot_base + kSlotRows - 1 - chunks_) * 4;
    chunk[0] = j.row;
    chunk[1] = c * kChunk;
    ++chunks_;
  }
  jobs_.emplace(key, j);
  ++window_.jobs;
  window_.job_vertices += q.count;
  window_.palette_rows += palette_rows;
  *offset = VkDeviceSize(at) * 48;
  if (!logged_first_) {
    logged_first_ = true;
    REXGPU_INFO("[skin1] first job (frame {}): {} vertices of {} dwords at 0x{:08X}, {} palette rows; each wrestler vertex is skinned once a frame", frame_, q.count, q.stride_words, q.address, palette_rows);
  }
  return true;
}

bool SkinOnce::GetJob(const Request& q, VkDeviceSize* offset) {
  if (!EnsureInit(q.mirror_buffer)) {
    ++window_.refused[kRefInit];
    return false;
  }
  const uint32_t b = q.bools;
  uint32_t bones = (b & 8u) ? 4 : (b & 4u) ? 3 : (b & 2u) ? 2 : 1;
  if (q.main && (b & 16u)) bones = std::max(bones, 3u);
  const uint32_t rows = PaletteRows(q, bones);
  uint64_t key = XXH3_64bits(q.vs_float + 4 * 76, size_t(rows) * 16);
  key = Mix(Mix(Mix(Mix(key, q.address), q.mirror_offset), (uint64_t(q.first) << 32) | q.count), (uint64_t(q.stride_words) << 8) | ((b >> 1) & 7u));
  const auto found = jobs_.find(key);
  if (found != jobs_.end()) {
    Job& j = found->second;
    const uint32_t main_flags = (q.bools & 0x1Eu) | (q.style ? 0x100u : 0u);
    if (!q.main || (j.main_set && j.flags == main_flags && j.scale_bits == q.scale_bits)) {
      ++window_.found;
      *offset = VkDeviceSize(j.out_vertex) * 48;
      return true;
    }
    if (!j.main_set && j.serial == serial_) {  // the side passes came first and the job has not gone out: the main pass's blend form is the job's
      j.main_set = true;
      j.flags = main_flags;
      j.scale_bits = q.scale_bits;
      WriteJobFlags(j);
      ++window_.found;
      *offset = VkDeviceSize(j.out_vertex) * 48;
      return true;
    }
    // Gone out with other main-pass inputs (or none): a job of its own.
    key = Mix(Mix(key, main_flags), q.scale_bits);
    const auto own = jobs_.find(key);
    if (own != jobs_.end()) {
      ++window_.found;
      *offset = VkDeviceSize(own->second.out_vertex) * 48;
      return true;
    }
    ++window_.own_main;
  }
  return MakeJob(q, key, rows, offset);
}

// wwe2k14 patch (Session 95, r6gamma, vulkan_hoist_skin_overlap): the caller records the dispatch inside the hoist stream, before its closing barrier, and that barrier names the shader write; the trailing barrier is then left out.
static bool g_merged_barrier = false;
void SkinOnceMergeBarrier(bool on) { g_merged_barrier = on; }

void SkinOnce::Record(DeferredCommandBuffer& hoist, uint64_t submission) {
  if (!chunks_) return;
  const ui::vulkan::VulkanDevice* device = cp_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = device->functions();
  // (The slot's bytes reach the GPU at the queue submit; a non-coherent type needs the flush.)
  VkMappedMemoryRange range = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, params_memory_, 0, VK_WHOLE_SIZE};
  dfn.vkFlushMappedMemoryRanges(device->device(), 1, &range);
  const uint32_t chunk_top = uint32_t(slot_) * kSlotRows + kSlotRows - 1;
  hoist.CmdVkBindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
  hoist.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &set_, 0, nullptr);
  hoist.CmdVkPushConstants(pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(chunk_top), &chunk_top);
  hoist.CmdVkDispatch(chunks_, 1, 1);
  if (!g_merged_barrier) {
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
  hoist.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  }
  slot_submission_[slot_] = submission;
  slot_ = -1;
  chunks_ = 0;
  ++serial_;
}

void SkinOnce::Summary(uint64_t frame) {
  if (frame < window_first_frame_ + 600) return;
  const uint64_t frames = std::max<uint64_t>(frame - window_first_frame_, 1);
  const uint64_t draws = window_.draws[0] + window_.draws[1] + window_.draws[2];
  if (draws) {
    const uint64_t index_vertices = window_.index_vertices[0] + window_.index_vertices[1] + window_.index_vertices[2];
    std::string refused;
    for (uint32_t i = 0; i < kRefCount; ++i) {
      if (window_.refused[i]) refused += fmt::format("{}{} {}", refused.empty() ? "" : ", ", kRefusedNames[i], window_.refused[i]);
    }
    REXGPU_INFO("[skin1] frames {}-{}: a frame {} draws read skinned rows (shadow {}, rim {}, main {}), {} index vertices ({} vertices named by the draws, counting a vertex once a draw) through them; skinned once: {} vertices in {} jobs a frame ({} draws found a job, {} main-pass jobs of their own, {} palette rows); refused {{{}}}",
                window_first_frame_, frame, draws / frames, window_.draws[0] / frames, window_.draws[1] / frames, window_.draws[2] / frames, index_vertices / frames, window_.draw_vertices / frames, window_.job_vertices / frames,
                window_.jobs / frames, window_.found / frames, window_.own_main / frames, window_.palette_rows / frames, refused.empty() ? "none" : refused);
  }
  window_ = Window{};
  window_first_frame_ = frame;
}

}  // namespace rex::graphics::vulkan::native
