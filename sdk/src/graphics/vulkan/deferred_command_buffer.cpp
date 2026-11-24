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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <rex/assert.h>
#include <rex/dbg.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/deferred_command_buffer.h>
#include <rex/math.h>

#include <algorithm>
#include <atomic>
#include <map>
#include <string>
#include <vector>

#include <rex/cvar.h>
#include <rex/logging.h>

#include "native/diag_layer.h"  // wwe2k14 patch (Session 95, the diagnostic run)

// wwe2k14 patch (Session 52): barrier merging at replay.
REXCVAR_DEFINE_BOOL(vulkan_merge_adjacent_barriers, false, "GPU/Vulkan",
                    "Session 52: pipeline barriers recorded back to back (no other command between them) are executed "
                    "as one vkCmdPipelineBarrier: the stage masks ORed, the memory, buffer and image barriers together "
                    "- at least the synchronisation of the separate calls, one wait-for-idle on the GPU instead of "
                    "several. Not merged: barriers with dependency flags, and two transitions of one image (the second "
                    "starts a new call). Census: [s52][barriers]")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(vulkan_barrier_census, 0, "GPU/Vulkan",
                     "Session 52 (diagnostic): N > 0 = every N recorded streams, a [s52][barrier kinds] line - the "
                     "pipeline barriers written (after vulkan_merge_adjacent_barriers) by their source and destination "
                     "stages and what they hold (m memory, b buffer, i image barriers), the 16 most frequent")
    .range(0, 100000)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// wwe2k14 patch (Session 56): a held image transition undone by the next one cancels.
REXCVAR_DEFINE_BOOL(vulkan_cancel_inverse_transitions, false, "GPU/Vulkan",
                    "Session 56 (with vulkan_merge_adjacent_barriers): an image barrier that undoes a held one (the same "
                    "image and subresources, the layouts the other way round, nothing recorded between them but binds "
                    "and state) cancels it - the image stays in its layout (e.g. a resolve's source going back to its "
                    "attachment layout and at once to shader reads again for the next resolve or fill): one barrier "
                    "call instead of two. Census: [s56][cancel]")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// wwe2k14 patch (Session 56): what the GPU's waits at barriers cost (an upper bound for in-run A/Bs).
REXCVAR_DEFINE_BOOL(vulkan_debug_barriers_nowait, false, "GPU/Vulkan",
                    "Session 56 (diagnostic, the picture may be WRONG with it on): every pipeline barrier is written "
                    "with no wait - top of pipe to bottom of pipe, no memory or buffer barriers, the image barriers "
                    "kept for their layout transitions with no access masks. For measuring what the barriers' "
                    "wait-for-idles cost")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

namespace rex::graphics::vulkan {

namespace {
// wwe2k14 patch (Session 52): the barrier-merge census (replay counts, read and logged by the replaying thread).
std::atomic<uint64_t> merge_census_calls{0}, merge_census_merged{0}, merge_census_streams{0};
// wwe2k14 patch (Session 56, vulkan_cancel_inverse_transitions): pairs of image transitions cancelled (recording).
std::atomic<uint64_t> cancel_census_cancelled{0};
// wwe2k14 patch (Session 52, vulkan_barrier_census): the recording thread's counts by kind.
struct BarrierKindKey {
  uint32_t src, dst, kinds;
  bool operator<(const BarrierKindKey& o) const {
    return src != o.src ? src < o.src : dst != o.dst ? dst < o.dst : kinds < o.kinds;
  }
};
thread_local std::map<BarrierKindKey, uint64_t> barrier_kinds;
thread_local uint64_t barrier_kind_streams = 0;
std::string StageNames(uint32_t mask) {
  static const std::pair<uint32_t, const char*> kNames[] = {
      {VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, "TOP"}, {VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, "IND"},
      {VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, "VI"}, {VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, "VS"},
      {VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT, "TCS"},
      {VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT, "TES"}, {VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT, "GS"},
      {VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, "FS"}, {VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, "EFT"},
      {VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, "LFT"}, {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, "CAO"},
      {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, "CS"}, {VK_PIPELINE_STAGE_TRANSFER_BIT, "XFER"},
      {VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, "BOTTOM"}, {VK_PIPELINE_STAGE_HOST_BIT, "HOST"},
      {VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, "ALLGFX"}, {VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, "ALL"}};
  std::string out;
  for (const auto& [bit, name] : kNames) {
    if (mask & bit) {
      if (!out.empty()) out += '|';
      out += name;
    }
  }
  return out.empty() ? std::string("0") : out;
}
}  // namespace

DeferredCommandBuffer::DeferredCommandBuffer(const VulkanCommandProcessor& command_processor,
                                             size_t initial_size)
    : command_processor_(command_processor) {
  command_stream_.reserve(initial_size / sizeof(uintmax_t));
}

void DeferredCommandBuffer::Reset() {
  // wwe2k14 patch (Session 52, vulkan_barrier_census): a stream ends here (the recording thread).
  if (const int32_t every = REXCVAR_GET(vulkan_barrier_census); every > 0 && ++barrier_kind_streams >= uint64_t(every)) {
    std::vector<std::pair<uint64_t, BarrierKindKey>> top;
    uint64_t total = 0;
    for (const auto& [k, n] : barrier_kinds) {
      top.push_back({n, k});
      total += n;
    }
    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::string text;
    for (size_t i = 0; i < top.size() && i < 16; ++i) {
      const BarrierKindKey& k = top[i].second;
      text += fmt::format("; {:.1f} {}->{} {}{}{}", double(top[i].first) / double(barrier_kind_streams),
                          StageNames(k.src), StageNames(k.dst), (k.kinds & 1) ? "m" : "", (k.kinds & 2) ? "b" : "",
                          (k.kinds & 4) ? "i" : "");
    }
    REXGPU_INFO("[s52][barrier kinds] per Reset of a deferred buffer (main and hoist; {} of them): {:.1f} barriers{}", barrier_kind_streams,
                double(total) / double(barrier_kind_streams), text);
    barrier_kinds.clear();
    barrier_kind_streams = 0;
  }
  barrier_pending_ = false;  // wwe2k14 patch (Session 52: dropped with the stream)
  // wwe2k14 patch (Session 56: likewise - a stream handed on wrote its barrier at its end)
  lazy_sync_pending_ = lazy_chain_ = in_lazy_pass_ = false;
  debug_last_constants_set_ = VK_NULL_HANDLE;  // wwe2k14 patch (Session 59, diagnostic)
  debug_last_layout_ = VK_NULL_HANDLE;
  lazy_attachments_.clear();
  command_stream_.clear();
  last_begin_offset_ = SIZE_MAX;  // wwe2k14 patch (Session 73)
  last_draw_end_ = SIZE_MAX;  // wwe2k14 patch (Session 23, MERGE)
}

// wwe2k14 patch (Session 7c, CAUSES): gpu_anatomy_gpu_marks_mode 1, see SetEventHook. Cleared before the call:
// the hook records a timestamp (not an event command) into this buffer, before the event command is written.
void DeferredCommandBuffer::CallEventHook() {
  VulkanCommandProcessor* hook = event_hook_;
  event_hook_ = nullptr;
  hook->GpuMarksSparseBeforeEvent();
}

bool DeferredCommandBuffer::GetDescriptorPayload(VkDescriptorType type, DescriptorPayload& payload_out,
                                                 size_t& element_size_out) {
  switch (type) {
    case VK_DESCRIPTOR_TYPE_SAMPLER:
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
    case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
      payload_out = DescriptorPayload::kImage;
      element_size_out = sizeof(VkDescriptorImageInfo);
      return true;
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
      payload_out = DescriptorPayload::kBuffer;
      element_size_out = sizeof(VkDescriptorBufferInfo);
      return true;
    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
      payload_out = DescriptorPayload::kTexelBufferView;
      element_size_out = sizeof(VkBufferView);
      return true;
    default:
      return false;
  }
}

bool DeferredCommandBuffer::CmdVkUpdateDescriptorSets(uint32_t write_count,
                                                      const VkWriteDescriptorSet* writes) {
  if (!write_count || write_count > kMaxDeferredDescriptorWrites) {
    return false;
  }
  // Same layout as the replay in ExecuteStream.
  size_t arguments_size = sizeof(ArgsVkUpdateDescriptorSets);
  for (uint32_t i = 0; i < write_count; ++i) {
    DescriptorPayload payload;
    size_t element_size;
    if (writes[i].pNext || !GetDescriptorPayload(writes[i].descriptorType, payload, element_size)) {
      return false;
    }
    arguments_size = rex::align(arguments_size, alignof(uintmax_t));
    arguments_size = rex::align(arguments_size + sizeof(ArgsDescriptorWrite), alignof(uintmax_t));
    arguments_size += element_size * writes[i].descriptorCount;
  }
  uint8_t* args_ptr =
      reinterpret_cast<uint8_t*>(WriteCommand(Command::kVkUpdateDescriptorSets, arguments_size));
  reinterpret_cast<ArgsVkUpdateDescriptorSets*>(args_ptr)->write_count = write_count;
  size_t offset_bytes = sizeof(ArgsVkUpdateDescriptorSets);
  for (uint32_t i = 0; i < write_count; ++i) {
    const VkWriteDescriptorSet& write = writes[i];
    DescriptorPayload payload;
    size_t element_size;
    GetDescriptorPayload(write.descriptorType, payload, element_size);
    offset_bytes = rex::align(offset_bytes, alignof(uintmax_t));
    auto& write_args = *reinterpret_cast<ArgsDescriptorWrite*>(args_ptr + offset_bytes);
    write_args.dst_set = write.dstSet;
    write_args.dst_binding = write.dstBinding;
    write_args.dst_array_element = write.dstArrayElement;
    write_args.descriptor_count = write.descriptorCount;
    write_args.descriptor_type = write.descriptorType;
    write_args.payload = payload;
    offset_bytes = rex::align(offset_bytes + sizeof(ArgsDescriptorWrite), alignof(uintmax_t));
    const void* infos = payload == DescriptorPayload::kImage    ? static_cast<const void*>(write.pImageInfo)
                        : payload == DescriptorPayload::kBuffer ? static_cast<const void*>(write.pBufferInfo)
                                                                : static_cast<const void*>(write.pTexelBufferView);
    std::memcpy(args_ptr + offset_bytes, infos, element_size * write.descriptorCount);
    offset_bytes += element_size * write.descriptorCount;
  }
  return true;
}

// wwe2k14 patch (Session 95, the diagnostic run): the one test when it is off; an armed frame's streams take the observing body.
void DeferredCommandBuffer::ExecuteStream(const std::vector<uintmax_t>& command_stream,
                                          VkCommandBuffer command_buffer) const {
  if (diag::g_observing.load(std::memory_order_relaxed)) [[unlikely]] {
    ExecuteStreamT<true>(command_stream, command_buffer);
  } else {
    ExecuteStreamT<false>(command_stream, command_buffer);
  }
}

template <bool kDiag>
void DeferredCommandBuffer::ExecuteStreamT(const std::vector<uintmax_t>& command_stream,
                                           VkCommandBuffer command_buffer) const {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  const ui::vulkan::VulkanDevice::Functions& dfn =
      command_processor_.GetVulkanDevice()->functions();
  const uintmax_t* stream = command_stream.data();
  size_t stream_remaining = command_stream.size();
  // wwe2k14 patch (Session 52, vulkan_merge_adjacent_barriers): scratch for merged barriers (the replaying thread's).
  const bool merge = REXCVAR_GET(vulkan_merge_adjacent_barriers);
  thread_local std::vector<VkMemoryBarrier> merge_memory;
  thread_local std::vector<VkBufferMemoryBarrier> merge_buffer;
  thread_local std::vector<VkImageMemoryBarrier> merge_image;
  uint64_t census_calls = 0, census_merged = 0;
  // wwe2k14 patch (Session 56, vulkan_debug_barriers_nowait): the barrier as written, or with no wait.
  const bool nowait = REXCVAR_GET(vulkan_debug_barriers_nowait);
  thread_local std::vector<VkImageMemoryBarrier> nowait_image;
  const auto write_barrier = [&](VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags flags,
                                 uint32_t memory_count, const VkMemoryBarrier* memory, uint32_t buffer_count,
                                 const VkBufferMemoryBarrier* buffer, uint32_t image_count,
                                 const VkImageMemoryBarrier* image) {
    if (!nowait) {
      dfn.vkCmdPipelineBarrier(command_buffer, src, dst, flags, memory_count, memory, buffer_count, buffer,
                               image_count, image);
      return;
    }
    nowait_image.assign(image, image + image_count);
    for (VkImageMemoryBarrier& b : nowait_image) b.srcAccessMask = b.dstAccessMask = 0;
    dfn.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 0, nullptr, 0, nullptr, image_count, nowait_image.data());
  };
  if constexpr (kDiag) diag::StreamBegin(command_buffer);
  while (stream_remaining) {
    const CommandHeader* header_ptr = reinterpret_cast<const CommandHeader*>(stream);
    const CommandHeader& header = *header_ptr;
    stream += kCommandHeaderSizeElements;
    stream_remaining -= kCommandHeaderSizeElements;

    switch (header.command) {
      case Command::kDiagMark: {  // wwe2k14 patch (Session 95, the diagnostic run)
        if constexpr (kDiag) {
          auto& args = *reinterpret_cast<const ArgsDiagMark*>(stream);
          diag::Mark(command_buffer, args.kind, args.id, args.aux);
        }
      } break;

      case Command::kVkBeginRenderPass: {
        auto& args = *reinterpret_cast<const ArgsVkBeginRenderPass*>(stream);
        size_t offset_bytes = sizeof(ArgsVkBeginRenderPass);
        VkRenderPassBeginInfo render_pass_begin_info;
        render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_begin_info.pNext = nullptr;
        render_pass_begin_info.renderPass = args.render_pass;
        render_pass_begin_info.framebuffer = args.framebuffer;
        render_pass_begin_info.renderArea = args.render_area;
        render_pass_begin_info.clearValueCount = args.clear_value_count;
        if (render_pass_begin_info.clearValueCount) {
          offset_bytes = rex::align(offset_bytes, alignof(VkClearValue));
          render_pass_begin_info.pClearValues = reinterpret_cast<const VkClearValue*>(
              reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
          offset_bytes += sizeof(VkClearValue) * render_pass_begin_info.clearValueCount;
        } else {
          render_pass_begin_info.pClearValues = nullptr;
        }
        if constexpr (kDiag) diag::BeginPass(command_buffer, args.render_pass, args.framebuffer);
        dfn.vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info, args.contents);
      } break;

      case Command::kVkBeginQuery: {
        auto& args = *reinterpret_cast<const ArgsVkBeginQuery*>(stream);
        if constexpr (kDiag) diag::GameQuery(true);
        dfn.vkCmdBeginQuery(command_buffer, args.query_pool, args.query, args.flags);
      } break;

      case Command::kVkBindDescriptorSets: {
        auto& args = *reinterpret_cast<const ArgsVkBindDescriptorSets*>(stream);
        size_t offset_bytes =
            rex::align(sizeof(ArgsVkBindDescriptorSets), alignof(VkDescriptorSet));
        const VkDescriptorSet* descriptor_sets = reinterpret_cast<const VkDescriptorSet*>(
            reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
        offset_bytes += sizeof(VkDescriptorSet) * args.descriptor_set_count;
        const uint32_t* dynamic_offsets = nullptr;
        if (args.dynamic_offset_count) {
          offset_bytes = rex::align(offset_bytes, alignof(uint32_t));
          dynamic_offsets = reinterpret_cast<const uint32_t*>(
              reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
          offset_bytes += sizeof(uint32_t) * args.dynamic_offset_count;
        }
        dfn.vkCmdBindDescriptorSets(command_buffer, args.pipeline_bind_point, args.layout,
                                    args.first_set, args.descriptor_set_count, descriptor_sets,
                                    args.dynamic_offset_count, dynamic_offsets);
        if constexpr (kDiag) {
          diag::BindSets(args.pipeline_bind_point, args.layout, args.first_set, args.descriptor_set_count, descriptor_sets,
                         args.dynamic_offset_count, dynamic_offsets);
        }
      } break;

      case Command::kVkBindIndexBuffer: {
        auto& args = *reinterpret_cast<const ArgsVkBindIndexBuffer*>(stream);
        dfn.vkCmdBindIndexBuffer(command_buffer, args.buffer, args.offset, args.index_type);
        if constexpr (kDiag) diag::BindIndexBuffer(args.buffer, args.offset, args.index_type);
      } break;

      case Command::kVkBindPipeline: {
        auto& args = *reinterpret_cast<const ArgsVkBindPipeline*>(stream);
        dfn.vkCmdBindPipeline(command_buffer, args.pipeline_bind_point, args.pipeline);
        if constexpr (kDiag) diag::BindPipeline(args.pipeline_bind_point, args.pipeline);
      } break;

      case Command::kVkBindVertexBuffers: {
        auto& args = *reinterpret_cast<const ArgsVkBindVertexBuffers*>(stream);
        size_t offset_bytes = rex::align(sizeof(ArgsVkBindVertexBuffers), alignof(VkBuffer));
        const VkBuffer* buffers = reinterpret_cast<const VkBuffer*>(
            reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
        offset_bytes =
            rex::align(offset_bytes + sizeof(VkBuffer) * args.binding_count, alignof(VkDeviceSize));
        const VkDeviceSize* offsets = reinterpret_cast<const VkDeviceSize*>(
            reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
        dfn.vkCmdBindVertexBuffers(command_buffer, args.first_binding, args.binding_count, buffers,
                                   offsets);
        if constexpr (kDiag) diag::BindVertexBuffers(args.first_binding, args.binding_count, buffers, offsets);
      } break;

      case Command::kVkClearAttachments: {
        auto& args = *reinterpret_cast<const ArgsVkClearAttachments*>(stream);
        size_t offset_bytes =
            rex::align(sizeof(ArgsVkClearAttachments), alignof(VkClearAttachment));
        const VkClearAttachment* attachments = reinterpret_cast<const VkClearAttachment*>(
            reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
        offset_bytes = rex::align(offset_bytes + sizeof(VkClearAttachment) * args.attachment_count,
                                  alignof(VkClearRect));
        const VkClearRect* rects = reinterpret_cast<const VkClearRect*>(
            reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
        dfn.vkCmdClearAttachments(command_buffer, args.attachment_count, attachments,
                                  args.rect_count, rects);
      } break;

      case Command::kVkClearColorImage: {
        auto& args = *reinterpret_cast<const ArgsVkClearColorImage*>(stream);
        dfn.vkCmdClearColorImage(
            command_buffer, args.image, args.image_layout, &args.color, args.range_count,
            reinterpret_cast<const VkImageSubresourceRange*>(
                reinterpret_cast<const uint8_t*>(stream) +
                rex::align(sizeof(ArgsVkClearColorImage), alignof(VkImageSubresourceRange))));
      } break;

      case Command::kVkCopyBuffer: {
        auto& args = *reinterpret_cast<const ArgsVkCopyBuffer*>(stream);
        dfn.vkCmdCopyBuffer(command_buffer, args.src_buffer, args.dst_buffer, args.region_count,
                            reinterpret_cast<const VkBufferCopy*>(
                                reinterpret_cast<const uint8_t*>(stream) +
                                rex::align(sizeof(ArgsVkCopyBuffer), alignof(VkBufferCopy))));
      } break;

      case Command::kVkCopyBufferToImage: {
        auto& args = *reinterpret_cast<const ArgsVkCopyBufferToImage*>(stream);
        dfn.vkCmdCopyBufferToImage(
            command_buffer, args.src_buffer, args.dst_image, args.dst_image_layout,
            args.region_count,
            reinterpret_cast<const VkBufferImageCopy*>(
                reinterpret_cast<const uint8_t*>(stream) +
                rex::align(sizeof(ArgsVkCopyBufferToImage), alignof(VkBufferImageCopy))));
      } break;

      // wwe2k14 patch (Session 18, DLAA).
      case Command::kHostCallback: {
        auto& args = *reinterpret_cast<const ArgsHostCallback*>(stream);
        args.fn(command_buffer, reinterpret_cast<const uint8_t*>(stream) +
                                    rex::align(sizeof(ArgsHostCallback), alignof(uintmax_t)));
      } break;

      // wwe2k14 patch (Session 18, DLAA).
      case Command::kVkCopyImage: {
        auto& args = *reinterpret_cast<const ArgsVkCopyImage*>(stream);
        dfn.vkCmdCopyImage(command_buffer, args.src_image, args.src_image_layout, args.dst_image,
                           args.dst_image_layout, args.region_count,
                           reinterpret_cast<const VkImageCopy*>(
                               reinterpret_cast<const uint8_t*>(stream) +
                               rex::align(sizeof(ArgsVkCopyImage), alignof(VkImageCopy))));
      } break;

      // wwe2k14 patch (Session 54, NATIVE without interlock).
      case Command::kVkClearDepthStencilImage: {
        auto& args = *reinterpret_cast<const ArgsVkClearDepthStencilImage*>(stream);
        dfn.vkCmdClearDepthStencilImage(command_buffer, args.image, args.image_layout, &args.value, 1, &args.range);
      } break;

      // wwe2k14 patch (Session 54, NATIVE draw sizes).
      case Command::kVkBlitImage: {
        auto& args = *reinterpret_cast<const ArgsVkBlitImage*>(stream);
        dfn.vkCmdBlitImage(command_buffer, args.src_image, args.src_image_layout, args.dst_image,
                           args.dst_image_layout, args.region_count,
                           reinterpret_cast<const VkImageBlit*>(
                               reinterpret_cast<const uint8_t*>(stream) +
                               rex::align(sizeof(ArgsVkBlitImage), alignof(VkImageBlit))),
                           args.filter);
      } break;

      // wwe2k14 patch (Session 7c, TEXDIRECT).
      case Command::kVkCopyImageToBuffer: {
        auto& args = *reinterpret_cast<const ArgsVkCopyImageToBuffer*>(stream);
        dfn.vkCmdCopyImageToBuffer(
            command_buffer, args.src_image, args.src_image_layout, args.dst_buffer,
            args.region_count,
            reinterpret_cast<const VkBufferImageCopy*>(
                reinterpret_cast<const uint8_t*>(stream) +
                rex::align(sizeof(ArgsVkCopyImageToBuffer), alignof(VkBufferImageCopy))));
      } break;

      case Command::kVkCopyQueryPoolResults: {
        auto& args = *reinterpret_cast<const ArgsVkCopyQueryPoolResults*>(stream);
        dfn.vkCmdCopyQueryPoolResults(command_buffer, args.query_pool, args.first_query,
                                      args.query_count, args.dst_buffer, args.dst_offset,
                                      args.stride, args.flags);
      } break;

      case Command::kVkDispatch: {
        auto& args = *reinterpret_cast<const ArgsVkDispatch*>(stream);
        dfn.vkCmdDispatch(command_buffer, args.group_count_x, args.group_count_y,
                          args.group_count_z);
      } break;

      case Command::kVkDraw: {
        auto& args = *reinterpret_cast<const ArgsVkDraw*>(stream);
        if constexpr (kDiag) diag::BeforeDraw(command_buffer, false, args.vertex_count, args.instance_count, args.first_vertex, 0, args.first_instance);
        dfn.vkCmdDraw(command_buffer, args.vertex_count, args.instance_count, args.first_vertex,
                      args.first_instance);
        if constexpr (kDiag) diag::AfterDraw(command_buffer);
      } break;

      case Command::kVkDrawIndexed: {
        auto& args = *reinterpret_cast<const ArgsVkDrawIndexed*>(stream);
        if constexpr (kDiag) diag::BeforeDraw(command_buffer, true, args.index_count, args.instance_count, args.first_index, args.vertex_offset, args.first_instance);
        dfn.vkCmdDrawIndexed(command_buffer, args.index_count, args.instance_count,
                             args.first_index, args.vertex_offset, args.first_instance);
        if constexpr (kDiag) diag::AfterDraw(command_buffer);
      } break;

      case Command::kVkEndQuery: {
        auto& args = *reinterpret_cast<const ArgsVkEndQuery*>(stream);
        dfn.vkCmdEndQuery(command_buffer, args.query_pool, args.query);
        if constexpr (kDiag) diag::GameQuery(false);
      } break;

      case Command::kVkEndRenderPass:
        dfn.vkCmdEndRenderPass(command_buffer);
        if constexpr (kDiag) diag::EndPass(command_buffer);
        break;

      case Command::kVkBeginRendering: {
        auto& args = *reinterpret_cast<const ArgsVkBeginRendering*>(stream);
        size_t offset_bytes =
            rex::align(sizeof(ArgsVkBeginRendering), alignof(VkRenderingAttachmentInfo));
        const VkRenderingAttachmentInfo* color_attachments =
            args.color_attachment_count
                ? reinterpret_cast<const VkRenderingAttachmentInfo*>(
                      reinterpret_cast<const uint8_t*>(stream) + offset_bytes)
                : nullptr;
        offset_bytes += sizeof(VkRenderingAttachmentInfo) * args.color_attachment_count;
        const VkRenderingAttachmentInfo* depth_attachment =
            args.has_depth_attachment ? reinterpret_cast<const VkRenderingAttachmentInfo*>(
                                            reinterpret_cast<const uint8_t*>(stream) + offset_bytes)
                                      : nullptr;
        if (args.has_depth_attachment) {
          offset_bytes += sizeof(VkRenderingAttachmentInfo);
        }
        const VkRenderingAttachmentInfo* stencil_attachment =
            args.has_stencil_attachment
                ? reinterpret_cast<const VkRenderingAttachmentInfo*>(
                      reinterpret_cast<const uint8_t*>(stream) + offset_bytes)
                : nullptr;
        VkRenderingInfo rendering_info = {};
        rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering_info.flags = args.flags;
        rendering_info.renderArea = args.render_area;
        rendering_info.layerCount = args.layer_count;
        rendering_info.viewMask = args.view_mask;
        rendering_info.colorAttachmentCount = args.color_attachment_count;
        rendering_info.pColorAttachments = color_attachments;
        rendering_info.pDepthAttachment = depth_attachment;
        rendering_info.pStencilAttachment = stencil_attachment;
        if constexpr (kDiag) diag::BeginDynamicPass(command_buffer, rendering_info);
        dfn.vkCmdBeginRendering(command_buffer, &rendering_info);
      } break;

      case Command::kVkEndRendering:
        dfn.vkCmdEndRendering(command_buffer);
        if constexpr (kDiag) diag::EndPass(command_buffer);
        break;

      case Command::kVkPipelineBarrier: {
        BarrierView view = ReadBarrier(stream);
        // wwe2k14 patch (Session 52, vulkan_merge_adjacent_barriers): the barriers right after this one join it.
        if (merge && !view.args->dependency_flags) {
          merge_memory.assign(view.memory, view.memory + view.args->memory_barrier_count);
          merge_buffer.assign(view.buffer, view.buffer + view.args->buffer_memory_barrier_count);
          merge_image.assign(view.image, view.image + view.args->image_memory_barrier_count);
          VkPipelineStageFlags src = view.args->src_stage_mask, dst = view.args->dst_stage_mask;
          uint32_t joined = 0;
          for (;;) {
            const uintmax_t* next = stream + header_ptr->arguments_size_elements;
            const size_t next_remaining = stream_remaining - header_ptr->arguments_size_elements;
            if (next_remaining < kCommandHeaderSizeElements) break;
            const CommandHeader& next_header = *reinterpret_cast<const CommandHeader*>(next);
            if (next_header.command != Command::kVkPipelineBarrier) break;
            const uintmax_t* next_args = next + kCommandHeaderSizeElements;
            const BarrierView v = ReadBarrier(next_args);
            if (v.args->dependency_flags) break;
            bool same_image = false;
            for (uint32_t i = 0; i < v.args->image_memory_barrier_count && !same_image; ++i) {
              for (const VkImageMemoryBarrier& b : merge_image) {
                if (b.image == v.image[i].image) {
                  same_image = true;
                  break;
                }
              }
            }
            if (same_image) break;
            src |= v.args->src_stage_mask;
            dst |= v.args->dst_stage_mask;
            merge_memory.insert(merge_memory.end(), v.memory, v.memory + v.args->memory_barrier_count);
            merge_buffer.insert(merge_buffer.end(), v.buffer, v.buffer + v.args->buffer_memory_barrier_count);
            merge_image.insert(merge_image.end(), v.image, v.image + v.args->image_memory_barrier_count);
            ++joined;
            // Step over this barrier: the loop's own advance then skips the one just joined.
            stream = next_args;
            stream_remaining = next_remaining - kCommandHeaderSizeElements;
            header_ptr = &next_header;
          }
          write_barrier(src, dst, 0, uint32_t(merge_memory.size()), merge_memory.data(),
                        uint32_t(merge_buffer.size()), merge_buffer.data(), uint32_t(merge_image.size()),
                        merge_image.data());
          ++census_calls;
          census_merged += joined;
        } else {
          write_barrier(view.args->src_stage_mask, view.args->dst_stage_mask, view.args->dependency_flags,
                        view.args->memory_barrier_count, view.memory, view.args->buffer_memory_barrier_count,
                        view.buffer, view.args->image_memory_barrier_count, view.image);
          ++census_calls;
        }
      } break;

      case Command::kVkPushConstants: {
        auto& args = *reinterpret_cast<const ArgsVkPushConstants*>(stream);
        dfn.vkCmdPushConstants(
            command_buffer, args.layout, args.stage_flags, args.offset, args.size,
            reinterpret_cast<const uint8_t*>(stream) + sizeof(ArgsVkPushConstants));
        if constexpr (kDiag) {
          diag::PushConstants(args.layout, args.stage_flags, args.offset, args.size,
                              reinterpret_cast<const uint8_t*>(stream) + sizeof(ArgsVkPushConstants));
        }
      } break;

      case Command::kVkResetQueryPool: {
        auto& args = *reinterpret_cast<const ArgsVkResetQueryPool*>(stream);
        dfn.vkCmdResetQueryPool(command_buffer, args.query_pool, args.first_query,
                                args.query_count);
      } break;

      case Command::kVkSetBlendConstants: {
        auto& args = *reinterpret_cast<const ArgsVkSetBlendConstants*>(stream);
        dfn.vkCmdSetBlendConstants(command_buffer, args.blend_constants);
      } break;

      case Command::kVkSetDepthBias: {
        auto& args = *reinterpret_cast<const ArgsVkSetDepthBias*>(stream);
        dfn.vkCmdSetDepthBias(command_buffer, args.depth_bias_constant_factor,
                              args.depth_bias_clamp, args.depth_bias_slope_factor);
      } break;

      case Command::kVkSetEdsState: {  // wwe2k14 patch (Session 93, idea 7)
        auto& args = *reinterpret_cast<const ArgsVkSetEdsState*>(stream);
        if (args.mask & kEdsCull) {
          dfn.vkCmdSetCullModeEXT(command_buffer, VkCullModeFlags(args.cull_mode));
          dfn.vkCmdSetFrontFaceEXT(command_buffer, VkFrontFace(args.front_face));
        }
        if (args.mask & kEdsDepth) {
          dfn.vkCmdSetDepthTestEnableEXT(command_buffer, args.depth_test_enable);
          dfn.vkCmdSetDepthWriteEnableEXT(command_buffer, args.depth_write_enable);
          dfn.vkCmdSetDepthCompareOpEXT(command_buffer, VkCompareOp(args.depth_compare_op));
        }
        if (args.mask & kEdsStencil) {
          dfn.vkCmdSetStencilTestEnableEXT(command_buffer, args.stencil_test_enable);
          for (uint32_t f = 0; f < 2; ++f) {
            dfn.vkCmdSetStencilOpEXT(command_buffer, f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT,
                                     VkStencilOp(args.stencil_ops[f][0]), VkStencilOp(args.stencil_ops[f][1]),
                                     VkStencilOp(args.stencil_ops[f][2]), VkCompareOp(args.stencil_ops[f][3]));
          }
        }
      } break;

      case Command::kVkSetScissor: {
        auto& args = *reinterpret_cast<const ArgsVkSetScissor*>(stream);
        dfn.vkCmdSetScissor(command_buffer, args.first_scissor, args.scissor_count,
                            reinterpret_cast<const VkRect2D*>(
                                reinterpret_cast<const uint8_t*>(stream) +
                                rex::align(sizeof(ArgsVkSetScissor), alignof(VkRect2D))));
        if constexpr (kDiag) {
          diag::SetScissor(args.first_scissor, args.scissor_count,
                           reinterpret_cast<const VkRect2D*>(reinterpret_cast<const uint8_t*>(stream) +
                                                             rex::align(sizeof(ArgsVkSetScissor), alignof(VkRect2D))));
        }
      } break;

      case Command::kVkSetStencilCompareMask: {
        auto& args = *reinterpret_cast<const ArgsSetStencilMaskReference*>(stream);
        dfn.vkCmdSetStencilCompareMask(command_buffer, args.face_mask, args.mask_reference);
      } break;

      case Command::kVkSetStencilReference: {
        auto& args = *reinterpret_cast<const ArgsSetStencilMaskReference*>(stream);
        dfn.vkCmdSetStencilReference(command_buffer, args.face_mask, args.mask_reference);
      } break;

      case Command::kVkSetStencilWriteMask: {
        auto& args = *reinterpret_cast<const ArgsSetStencilMaskReference*>(stream);
        dfn.vkCmdSetStencilWriteMask(command_buffer, args.face_mask, args.mask_reference);
      } break;

      case Command::kVkUpdateDescriptorSets: {
        const uint8_t* args_ptr = reinterpret_cast<const uint8_t*>(stream);
        const auto& args = *reinterpret_cast<const ArgsVkUpdateDescriptorSets*>(args_ptr);
        std::array<VkWriteDescriptorSet, kMaxDeferredDescriptorWrites> writes;
        size_t offset_bytes = sizeof(ArgsVkUpdateDescriptorSets);
        for (uint32_t i = 0; i < args.write_count; ++i) {
          offset_bytes = rex::align(offset_bytes, alignof(uintmax_t));
          const auto& write_args = *reinterpret_cast<const ArgsDescriptorWrite*>(args_ptr + offset_bytes);
          offset_bytes = rex::align(offset_bytes + sizeof(ArgsDescriptorWrite), alignof(uintmax_t));
          DescriptorPayload payload;
          size_t element_size;
          GetDescriptorPayload(write_args.descriptor_type, payload, element_size);
          const void* infos = args_ptr + offset_bytes;
          offset_bytes += element_size * write_args.descriptor_count;
          VkWriteDescriptorSet& write = writes[i];
          write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          write.pNext = nullptr;
          write.dstSet = write_args.dst_set;
          write.dstBinding = write_args.dst_binding;
          write.dstArrayElement = write_args.dst_array_element;
          write.descriptorCount = write_args.descriptor_count;
          write.descriptorType = write_args.descriptor_type;
          write.pImageInfo = payload == DescriptorPayload::kImage
                                 ? static_cast<const VkDescriptorImageInfo*>(infos)
                                 : nullptr;
          write.pBufferInfo = payload == DescriptorPayload::kBuffer
                                  ? static_cast<const VkDescriptorBufferInfo*>(infos)
                                  : nullptr;
          write.pTexelBufferView = payload == DescriptorPayload::kTexelBufferView
                                       ? static_cast<const VkBufferView*>(infos)
                                       : nullptr;
        }
        dfn.vkUpdateDescriptorSets(command_processor_.GetVulkanDevice()->device(), args.write_count,
                                   writes.data(), 0, nullptr);
      } break;

      case Command::kVkSetCheckpointNV: {
        auto& args = *reinterpret_cast<const ArgsVkSetCheckpointNV*>(stream);
        if (dfn.vkCmdSetCheckpointNV) {
          dfn.vkCmdSetCheckpointNV(command_buffer,
                                   reinterpret_cast<const void*>(uintptr_t(args.marker)));
        }
      } break;

      case Command::kVkSetViewport: {
        auto& args = *reinterpret_cast<const ArgsVkSetViewport*>(stream);
        dfn.vkCmdSetViewport(command_buffer, args.first_viewport, args.viewport_count,
                             reinterpret_cast<const VkViewport*>(
                                 reinterpret_cast<const uint8_t*>(stream) +
                                 rex::align(sizeof(ArgsVkSetViewport), alignof(VkViewport))));
        if constexpr (kDiag) {
          diag::SetViewport(args.first_viewport, args.viewport_count,
                            reinterpret_cast<const VkViewport*>(reinterpret_cast<const uint8_t*>(stream) +
                                                                rex::align(sizeof(ArgsVkSetViewport), alignof(VkViewport))));
        }
      } break;

      // wwe2k14 patch (Session 7b, GPUMARKS).
      case Command::kVkWriteTimestamp: {
        auto& args = *reinterpret_cast<const ArgsVkWriteTimestamp*>(stream);
        dfn.vkCmdWriteTimestamp(command_buffer, args.pipeline_stage, args.query_pool, args.query);
      } break;

      case Command::kNop:  // wwe2k14 patch (Session 79, NF-D): turned off after recording (NopRange)
        break;

      default:
        assert_unhandled_case(header.command);
        break;
    }

    stream += header_ptr->arguments_size_elements;
    stream_remaining -= header_ptr->arguments_size_elements;
  }
  if constexpr (kDiag) diag::StreamEnd(command_buffer);
  // wwe2k14 patch (Session 52): the barrier census, every 600 replayed streams (about 5-10 s).
  merge_census_calls += census_calls;
  merge_census_merged += census_merged;
  if (merge && (merge_census_streams.fetch_add(1) + 1) % 600 == 0) {
    const uint64_t calls = merge_census_calls.exchange(0), merged = merge_census_merged.exchange(0);
    REXGPU_INFO("[s52][barriers] last 600 streams: {} barrier calls, {} recorded barriers joined into them ({:.1f}%); "
                "[s56][cancel] {} pairs of image transitions cancelled",
                calls, merged, calls + merged ? 100.0 * double(merged) / double(calls + merged) : 0.0,
                cancel_census_cancelled.exchange(0));
  }
}

// wwe2k14 patch (Session 95, gpu_owned_graph): see GraphEvent in the header. Read only; a command turned off (kNop) is not an event.
bool DeferredCommandBuffer::ScanStream(std::vector<GraphEvent>& out, size_t max_events) const {
  const size_t base = out.size();
  const uintmax_t* const data = command_stream_.data();
  size_t at = 0;
  const size_t size = command_stream_.size();
  const auto push = [&](GraphEvent::Kind kind, size_t header_at, size_t next) -> GraphEvent& {
    out.emplace_back();
    GraphEvent& e = out.back();
    e.kind = kind;
    e.at = uint32_t(header_at);
    e.next = uint32_t(next);
    e.args = uint32_t(header_at + kCommandHeaderSizeElements);
    return e;
  };
  while (at < size) {
    const CommandHeader& header = *reinterpret_cast<const CommandHeader*>(data + at);
    const size_t next = at + kCommandHeaderSizeElements + header.arguments_size_elements;
    if (next > size) return false;
    const uintmax_t* args = data + at + kCommandHeaderSizeElements;
    switch (header.command) {
      case Command::kVkPipelineBarrier: {
        const BarrierView view = ReadBarrier(args);
        for (uint32_t i = 0; i < view.args->image_memory_barrier_count; ++i) {
          GraphEvent& e = push(GraphEvent::kBarrierImage, at, next);
          e.a = reinterpret_cast<uint64_t>(view.image[i].image);
          e.old_layout = int32_t(view.image[i].oldLayout);
          e.new_layout = int32_t(view.image[i].newLayout);
          e.aspect = uint8_t(view.image[i].subresourceRange.aspectMask & 0xFF);
        }
      } break;
      case Command::kVkBeginRenderPass: {
        const auto& a = *reinterpret_cast<const ArgsVkBeginRenderPass*>(args);
        GraphEvent& e = push(GraphEvent::kBegin, at, next);
        e.a = reinterpret_cast<uint64_t>(a.render_pass);
        e.b = reinterpret_cast<uint64_t>(a.framebuffer);
      } break;
      case Command::kVkBeginRendering:
        push(GraphEvent::kBegin, at, next);  // (dynamic rendering: no render pass / framebuffer to name - an unknown pass)
        break;
      case Command::kVkEndRenderPass:
      case Command::kVkEndRendering:
        push(GraphEvent::kEnd, at, next);
        break;
      case Command::kVkCopyImage: {
        const auto& a = *reinterpret_cast<const ArgsVkCopyImage*>(args);
        GraphEvent& e = push(GraphEvent::kCopyImage, at, next);
        e.a = reinterpret_cast<uint64_t>(a.src_image);
        e.b = reinterpret_cast<uint64_t>(a.dst_image);
        if (a.region_count) {
          const auto* r = reinterpret_cast<const VkImageCopy*>(reinterpret_cast<const uint8_t*>(args) +
                                                              rex::align(sizeof(ArgsVkCopyImage), alignof(VkImageCopy)));
          e.src_aspect = uint8_t(r[0].srcSubresource.aspectMask & 0xFF);
          e.aspect = uint8_t(r[0].dstSubresource.aspectMask & 0xFF);
        }
      } break;
      case Command::kVkBlitImage: {
        const auto& a = *reinterpret_cast<const ArgsVkBlitImage*>(args);
        GraphEvent& e = push(GraphEvent::kBlit, at, next);
        e.a = reinterpret_cast<uint64_t>(a.src_image);
        e.b = reinterpret_cast<uint64_t>(a.dst_image);
        if (a.region_count) {
          const auto* r = reinterpret_cast<const VkImageBlit*>(reinterpret_cast<const uint8_t*>(args) +
                                                              rex::align(sizeof(ArgsVkBlitImage), alignof(VkImageBlit)));
          e.src_aspect = uint8_t(r[0].srcSubresource.aspectMask & 0xFF);
          e.aspect = uint8_t(r[0].dstSubresource.aspectMask & 0xFF);
        }
      } break;
      case Command::kVkClearColorImage: {
        const auto& a = *reinterpret_cast<const ArgsVkClearColorImage*>(args);
        GraphEvent& e = push(GraphEvent::kClearColor, at, next);
        e.b = reinterpret_cast<uint64_t>(a.image);
        e.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
      } break;
      case Command::kVkClearDepthStencilImage: {
        const auto& a = *reinterpret_cast<const ArgsVkClearDepthStencilImage*>(args);
        GraphEvent& e = push(GraphEvent::kClearDepth, at, next);
        e.b = reinterpret_cast<uint64_t>(a.image);
        e.aspect = uint8_t(a.range.aspectMask & 0xFF);
      } break;
      case Command::kVkCopyImageToBuffer: {
        const auto& a = *reinterpret_cast<const ArgsVkCopyImageToBuffer*>(args);
        GraphEvent& e = push(GraphEvent::kImageToBuffer, at, next);
        e.a = reinterpret_cast<uint64_t>(a.src_image);
        if (a.region_count) {
          const auto* r = reinterpret_cast<const VkBufferImageCopy*>(reinterpret_cast<const uint8_t*>(args) +
                                                                    rex::align(sizeof(ArgsVkCopyImageToBuffer), alignof(VkBufferImageCopy)));
          e.src_aspect = uint8_t(r[0].imageSubresource.aspectMask & 0xFF);
        }
      } break;
      case Command::kVkCopyBufferToImage: {
        const auto& a = *reinterpret_cast<const ArgsVkCopyBufferToImage*>(args);
        GraphEvent& e = push(GraphEvent::kBufferToImage, at, next);
        e.b = reinterpret_cast<uint64_t>(a.dst_image);
        if (a.region_count) {
          const auto* r = reinterpret_cast<const VkBufferImageCopy*>(reinterpret_cast<const uint8_t*>(args) +
                                                                    rex::align(sizeof(ArgsVkCopyBufferToImage), alignof(VkBufferImageCopy)));
          e.aspect = uint8_t(r[0].imageSubresource.aspectMask & 0xFF);
        }
      } break;
      case Command::kVkDispatch:
        push(GraphEvent::kDispatch, at, next);
        break;
      case Command::kHostCallback:
        push(GraphEvent::kHostCallback, at, next);
        break;
      case Command::kVkBeginQuery:
      case Command::kVkEndQuery:
      case Command::kVkCopyQueryPoolResults:
        push(GraphEvent::kQuery, at, next);
        break;
      default:
        break;
    }
    if (out.size() - base > max_events) return false;
    at = next;
  }
  return at == size;
}

DeferredCommandBuffer::BarrierView DeferredCommandBuffer::ReadBarrier(const uintmax_t* stream) {
  BarrierView view = {};
  view.args = reinterpret_cast<const ArgsVkPipelineBarrier*>(stream);
  size_t offset = sizeof(ArgsVkPipelineBarrier);
  if (view.args->memory_barrier_count) {
    offset = rex::align(offset, alignof(VkMemoryBarrier));
    view.memory = reinterpret_cast<const VkMemoryBarrier*>(reinterpret_cast<const uint8_t*>(stream) + offset);
    offset += sizeof(VkMemoryBarrier) * view.args->memory_barrier_count;
  }
  if (view.args->buffer_memory_barrier_count) {
    offset = rex::align(offset, alignof(VkBufferMemoryBarrier));
    view.buffer = reinterpret_cast<const VkBufferMemoryBarrier*>(reinterpret_cast<const uint8_t*>(stream) + offset);
    offset += sizeof(VkBufferMemoryBarrier) * view.args->buffer_memory_barrier_count;
  }
  if (view.args->image_memory_barrier_count) {
    offset = rex::align(offset, alignof(VkImageMemoryBarrier));
    view.image = reinterpret_cast<const VkImageMemoryBarrier*>(reinterpret_cast<const uint8_t*>(stream) + offset);
  }
  return view;
}

void DeferredCommandBuffer::CmdVkPipelineBarrier(
    VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask,
    VkDependencyFlags dependency_flags, uint32_t memory_barrier_count,
    const VkMemoryBarrier* memory_barriers, uint32_t buffer_memory_barrier_count,
    const VkBufferMemoryBarrier* buffer_memory_barriers, uint32_t image_memory_barrier_count,
    const VkImageMemoryBarrier* image_memory_barriers) {
  // wwe2k14 patch (Session 56): the barrier held after lazy render passes goes first (joined with this one).
  if (lazy_chain_) {
    lazy_break_command_ = uint32_t(Command::kVkPipelineBarrier);  // (Session 59, diagnostic)
    if (capture_break_stack) CaptureBreakStack();                 // (Session 79, diagnostic)
  }
  lazy_chain_ = false;
  if (lazy_sync_pending_ && !in_lazy_pass_) FlushLazySync();
  BeforeEventCommand();  // wwe2k14 patch (Session 7c, CAUSES)
  // wwe2k14 patch (Session 20, RECSTATS)
  ++record_stats_.c[RecordStats::kBarrierBatches];
  record_stats_.c[RecordStats::kMemoryBarriers] += memory_barrier_count;
  record_stats_.c[RecordStats::kBufferBarriers] += buffer_memory_barrier_count;
  record_stats_.c[RecordStats::kImageBarriers] += image_memory_barrier_count;
  // wwe2k14 patch (Session 52, vulkan_merge_adjacent_barriers): held back and joined (see IsStateCommand).
  if (!REXCVAR_GET(vulkan_merge_adjacent_barriers) || dependency_flags) {
    FlushPendingBarrier();
    WriteBarrierCommand(src_stage_mask, dst_stage_mask, dependency_flags, memory_barrier_count, memory_barriers,
                        buffer_memory_barrier_count, buffer_memory_barriers, image_memory_barrier_count,
                        image_memory_barriers);
    return;
  }
  // wwe2k14 patch (Session 56, vulkan_cancel_inverse_transitions): transitions undoing held ones cancel with them.
  thread_local std::vector<VkImageMemoryBarrier> kept_image_barriers;
  if (barrier_pending_ && image_memory_barrier_count && REXCVAR_GET(vulkan_cancel_inverse_transitions)) {
    kept_image_barriers.clear();
    for (uint32_t i = 0; i < image_memory_barrier_count; ++i) {
      const VkImageMemoryBarrier& b = image_memory_barriers[i];
      auto held = std::find_if(pending_image_barriers_.begin(), pending_image_barriers_.end(),
                               [&](const VkImageMemoryBarrier& p) {
                                 return p.image == b.image && p.oldLayout == b.newLayout &&
                                        p.newLayout == b.oldLayout && p.oldLayout != p.newLayout &&
                                        p.srcQueueFamilyIndex == b.srcQueueFamilyIndex &&
                                        p.dstQueueFamilyIndex == b.dstQueueFamilyIndex &&
                                        std::memcmp(&p.subresourceRange, &b.subresourceRange,
                                                    sizeof(VkImageSubresourceRange)) == 0;
                               });
      if (held != pending_image_barriers_.end()) {
        pending_image_barriers_.erase(held);
        ++cancel_census_cancelled;
      } else {
        kept_image_barriers.push_back(b);
      }
    }
    if (kept_image_barriers.size() != image_memory_barrier_count) {
      image_memory_barrier_count = uint32_t(kept_image_barriers.size());
      image_memory_barriers = kept_image_barriers.data();
    }
  }
  if (barrier_pending_) {
    // Two transitions of one image are not joined: the earlier one is written first.
    bool same_image = false;
    for (uint32_t i = 0; i < image_memory_barrier_count && !same_image; ++i) {
      for (const VkImageMemoryBarrier& b : pending_image_barriers_) {
        if (b.image == image_memory_barriers[i].image) {
          same_image = true;
          break;
        }
      }
    }
    if (same_image) {
      FlushPendingBarrierSlow();
    } else {
      ++merge_census_merged;
    }
  }
  if (!barrier_pending_) {
    barrier_pending_ = true;
    pending_src_stage_mask_ = 0;
    pending_dst_stage_mask_ = 0;
    pending_memory_barriers_.clear();
    pending_buffer_barriers_.clear();
    pending_image_barriers_.clear();
  }
  pending_src_stage_mask_ |= src_stage_mask;
  pending_dst_stage_mask_ |= dst_stage_mask;
  pending_memory_barriers_.insert(pending_memory_barriers_.end(), memory_barriers,
                                  memory_barriers + memory_barrier_count);
  pending_buffer_barriers_.insert(pending_buffer_barriers_.end(), buffer_memory_barriers,
                                  buffer_memory_barriers + buffer_memory_barrier_count);
  pending_image_barriers_.insert(pending_image_barriers_.end(), image_memory_barriers,
                                 image_memory_barriers + image_memory_barrier_count);
}

// wwe2k14 patch (Session 52): the commands a held-back barrier may be written after.
bool DeferredCommandBuffer::IsStateCommand(Command command) {
  switch (command) {
    case Command::kVkBindDescriptorSets:
    case Command::kVkBindIndexBuffer:
    case Command::kVkBindPipeline:
    case Command::kVkBindVertexBuffers:
    case Command::kVkPushConstants:
    case Command::kVkSetBlendConstants:
    case Command::kVkSetDepthBias:
    case Command::kVkSetEdsState:
    case Command::kVkSetScissor:
    case Command::kVkSetStencilCompareMask:
    case Command::kVkSetStencilReference:
    case Command::kVkSetStencilWriteMask:
    case Command::kVkSetViewport:
      return true;  // (Review: not checkpoints - a hang in the barrier would be blamed on the command after it.)
    default:
      return false;
  }
}

// wwe2k14 patch (Session 56, gpu_native_truth_lazy_pass_deps): see CmdVkBeginRenderPassLazy. The stages and accesses
// are those of the truth passes' render pass dependencies (NativeRenderer's post render passes).
namespace {
constexpr VkPipelineStageFlags kLazyAttachmentStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                                       VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
constexpr VkAccessFlags kLazyAttachmentWrites =
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
constexpr VkAccessFlags kLazyAttachmentAccesses = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | kLazyAttachmentWrites |
                                                  VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
}  // namespace

void DeferredCommandBuffer::AddLazyImageBarriers(const LazyAttachment* attachments, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    const LazyAttachment& a = attachments[i];
    if (a.image == VK_NULL_HANDLE ||
        std::any_of(lazy_barriers_.begin(), lazy_barriers_.end(),
                    [&](const VkImageMemoryBarrier& b) { return b.image == a.image; })) {
      continue;
    }
    const bool color = (a.aspect & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
    VkImageMemoryBarrier& b = lazy_barriers_.emplace_back();
    b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = color ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                      (color ? VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                             : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    b.oldLayout = b.newLayout = a.layout;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = a.image;
    b.subresourceRange = {a.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
  }
}

void DeferredCommandBuffer::FlushLazySync() {
  lazy_sync_pending_ = false;
  // The external dependency out of the pass: its attachment work before the attachment stages and compute after it.
  if (lazy_style_ == 1) {
    lazy_barriers_.clear();
    AddLazyImageBarriers(lazy_attachments_.data(), lazy_attachments_.size());
    lazy_attachments_.clear();
    if (!lazy_barriers_.empty()) {
      CmdVkPipelineBarrier(kLazyAttachmentStages, kLazyAttachmentStages | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                           nullptr, 0, nullptr, uint32_t(lazy_barriers_.size()), lazy_barriers_.data());
    }
    return;
  }
  lazy_attachments_.clear();
  const VkAccessFlags ds = lazy_style_ == 2 ? VkAccessFlags(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
                                            : VkAccessFlags(0);
  const VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, kLazyAttachmentWrites & ~ds,
                                   (kLazyAttachmentAccesses & ~ds) | VK_ACCESS_SHADER_READ_BIT};
  CmdVkPipelineBarrier(kLazyAttachmentStages, kLazyAttachmentStages | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &barrier, 0, nullptr, 0, nullptr);
}

void DeferredCommandBuffer::CmdVkBeginRenderPassLazy(const VkRenderPassBeginInfo* render_pass_begin,
                                                     VkSubpassContents contents, const LazyAttachment* attachments,
                                                     uint32_t attachment_count, uint32_t mode, uint32_t style) {
  bool sync = !lazy_chain_ && (mode & 2);
  lazy_last_reason_ = sync ? 1 : 0;
  for (uint32_t i = 0; i < attachment_count && !sync; ++i) {
    sync = std::any_of(lazy_attachments_.begin(), lazy_attachments_.end(),
                       [&](const LazyAttachment& a) { return a.image == attachments[i].image; });
    if (sync) lazy_last_reason_ = 2 + i;  // (2 colour, 3 depth)
  }
  if (sync && style == 1) {
    // One barrier: the passes' attachments since the last one (the dependency out of them) and this pass's own (the
    // dependency into it: attachment work and compute before its attachment stages).
    lazy_barriers_.clear();
    if (lazy_sync_pending_) AddLazyImageBarriers(lazy_attachments_.data(), lazy_attachments_.size());
    AddLazyImageBarriers(attachments, attachment_count);
    lazy_sync_pending_ = false;
    lazy_attachments_.clear();
    CmdVkPipelineBarrier(kLazyAttachmentStages | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         kLazyAttachmentStages | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                         uint32_t(lazy_barriers_.size()), lazy_barriers_.data());
    ++lazy_begins_synced_;
  } else if (sync) {
    lazy_style_ = style;
    if (lazy_sync_pending_) FlushLazySync();
    lazy_attachments_.clear();  // (the barrier below orders every earlier pass's attachment writes too)
    // The external dependency into the pass: attachment work and compute before its attachment stages.
    const VkAccessFlags ds = style == 2 ? VkAccessFlags(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
                                       : VkAccessFlags(0);
    const VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, kLazyAttachmentWrites & ~ds,
                                     kLazyAttachmentAccesses & ~ds};
    CmdVkPipelineBarrier(kLazyAttachmentStages | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, kLazyAttachmentStages, 0, 1,
                         &barrier, 0, nullptr, 0, nullptr);
    ++lazy_begins_synced_;
  } else {
    ++lazy_begins_free_;
  }
  lazy_style_ = style;
  lazy_attachments_.insert(lazy_attachments_.end(), attachments, attachments + attachment_count);
  lazy_end_sync_ = (mode & 1) != 0;
  in_lazy_pass_ = true;  // (before the begin is written: the held barrier after earlier lazy passes stays held)
  CmdVkBeginRenderPass(render_pass_begin, contents);
}

void DeferredCommandBuffer::CmdVkEndRenderPassLazy() {
  CmdVkEndRenderPass();
  in_lazy_pass_ = false;
  lazy_sync_pending_ = lazy_sync_pending_ || lazy_end_sync_;
  lazy_chain_ = true;
}

void DeferredCommandBuffer::FlushPendingBarrierSlow() {
  barrier_pending_ = false;
  WriteBarrierCommand(pending_src_stage_mask_, pending_dst_stage_mask_, 0,
                      uint32_t(pending_memory_barriers_.size()), pending_memory_barriers_.data(),
                      uint32_t(pending_buffer_barriers_.size()), pending_buffer_barriers_.data(),
                      uint32_t(pending_image_barriers_.size()), pending_image_barriers_.data());
}

void DeferredCommandBuffer::WriteBarrierCommand(
    VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask,
    VkDependencyFlags dependency_flags, uint32_t memory_barrier_count,
    const VkMemoryBarrier* memory_barriers, uint32_t buffer_memory_barrier_count,
    const VkBufferMemoryBarrier* buffer_memory_barriers, uint32_t image_memory_barrier_count,
    const VkImageMemoryBarrier* image_memory_barriers) {
  if (REXCVAR_GET(vulkan_barrier_census) > 0) {  // wwe2k14 patch (Session 52)
    ++barrier_kinds[{src_stage_mask, dst_stage_mask,
                     (memory_barrier_count ? 1u : 0u) | (buffer_memory_barrier_count ? 2u : 0u) |
                         (image_memory_barrier_count ? 4u : 0u)}];
  }
  size_t arguments_size = sizeof(ArgsVkPipelineBarrier);
  size_t memory_barriers_offset = 0;
  if (memory_barrier_count) {
    arguments_size = rex::align(arguments_size, alignof(VkMemoryBarrier));
    memory_barriers_offset = arguments_size;
    arguments_size += sizeof(VkMemoryBarrier) * memory_barrier_count;
  }
  size_t buffer_memory_barriers_offset = 0;
  if (buffer_memory_barrier_count) {
    arguments_size = rex::align(arguments_size, alignof(VkBufferMemoryBarrier));
    buffer_memory_barriers_offset = arguments_size;
    arguments_size += sizeof(VkBufferMemoryBarrier) * buffer_memory_barrier_count;
  }
  size_t image_memory_barriers_offset = 0;
  if (image_memory_barrier_count) {
    arguments_size = rex::align(arguments_size, alignof(VkImageMemoryBarrier));
    image_memory_barriers_offset = arguments_size;
    arguments_size += sizeof(VkImageMemoryBarrier) * image_memory_barrier_count;
  }
  uint8_t* args_ptr =
      reinterpret_cast<uint8_t*>(WriteCommand(Command::kVkPipelineBarrier, arguments_size));
  auto& args = *reinterpret_cast<ArgsVkPipelineBarrier*>(args_ptr);
  args.src_stage_mask = src_stage_mask;
  args.dst_stage_mask = dst_stage_mask;
  args.dependency_flags = dependency_flags;
  args.memory_barrier_count = memory_barrier_count;
  args.buffer_memory_barrier_count = buffer_memory_barrier_count;
  args.image_memory_barrier_count = image_memory_barrier_count;
  if (memory_barrier_count) {
    std::memcpy(args_ptr + memory_barriers_offset, memory_barriers,
                sizeof(VkMemoryBarrier) * memory_barrier_count);
  }
  if (buffer_memory_barrier_count) {
    std::memcpy(args_ptr + buffer_memory_barriers_offset, buffer_memory_barriers,
                sizeof(VkBufferMemoryBarrier) * buffer_memory_barrier_count);
  }
  if (image_memory_barrier_count) {
    std::memcpy(args_ptr + image_memory_barriers_offset, image_memory_barriers,
                sizeof(VkImageMemoryBarrier) * image_memory_barrier_count);
  }
}

void DeferredCommandBuffer::CmdVkBeginRendering(const VkRenderingInfo* rendering_info) {
  assert_null(rendering_info->pNext);
  // wwe2k14 patch (Session 7c, CAUSES)
  BeforeEventCommand();
  ++rendering_begins_recorded_;
  ++record_stats_.c[RecordStats::kRenderingBegins];  // wwe2k14 patch (Session 20, RECSTATS)

  size_t arguments_size =
      rex::align(sizeof(ArgsVkBeginRendering), alignof(VkRenderingAttachmentInfo));
  size_t color_attachments_offset = arguments_size;
  arguments_size += sizeof(VkRenderingAttachmentInfo) * rendering_info->colorAttachmentCount;
  size_t depth_attachment_offset = arguments_size;
  if (rendering_info->pDepthAttachment) {
    arguments_size += sizeof(VkRenderingAttachmentInfo);
  }
  size_t stencil_attachment_offset = arguments_size;
  if (rendering_info->pStencilAttachment) {
    arguments_size += sizeof(VkRenderingAttachmentInfo);
  }

  uint8_t* args_ptr =
      reinterpret_cast<uint8_t*>(WriteCommand(Command::kVkBeginRendering, arguments_size));
  auto& args = *reinterpret_cast<ArgsVkBeginRendering*>(args_ptr);
  args.flags = rendering_info->flags;
  args.render_area = rendering_info->renderArea;
  args.layer_count = rendering_info->layerCount;
  args.view_mask = rendering_info->viewMask;
  args.color_attachment_count = rendering_info->colorAttachmentCount;
  args.has_depth_attachment = rendering_info->pDepthAttachment != nullptr;
  args.has_stencil_attachment = rendering_info->pStencilAttachment != nullptr;

  if (rendering_info->colorAttachmentCount) {
    std::memcpy(args_ptr + color_attachments_offset, rendering_info->pColorAttachments,
                sizeof(VkRenderingAttachmentInfo) * rendering_info->colorAttachmentCount);
  }
  if (rendering_info->pDepthAttachment) {
    std::memcpy(args_ptr + depth_attachment_offset, rendering_info->pDepthAttachment,
                sizeof(VkRenderingAttachmentInfo));
  }
  if (rendering_info->pStencilAttachment) {
    std::memcpy(args_ptr + stencil_attachment_offset, rendering_info->pStencilAttachment,
                sizeof(VkRenderingAttachmentInfo));
  }
}

// wwe2k14 patch (Session 79, diagnostic): see the header. (Skips this function and its caller in this file.)
void DeferredCommandBuffer::CaptureBreakStack() {
  uintptr_t frames[7] = {};
  lazy_break_base_ = VulkanCommandProcessor::DebugCaptureStack(frames, 7, 1);
  for (uint32_t i = 0; i < 5; ++i) lazy_break_stack_[i] = frames[i + 1];
}

void* DeferredCommandBuffer::WriteCommand(Command command, size_t arguments_size_bytes) {
  // wwe2k14 patch (Session 56): the barrier after lazy render passes goes before GPU work outside a lazy pass.
  if (command != Command::kVkPipelineBarrier && !KeepsLazyChain(command)) {
    if (lazy_sync_pending_ && !in_lazy_pass_) FlushLazySync();
    if (lazy_chain_) {
      lazy_break_command_ = uint32_t(command);       // wwe2k14 patch (Session 59, diagnostic)
      if (capture_break_stack) CaptureBreakStack();  // (Session 79, diagnostic)
    }
    lazy_chain_ = false;
  }
  // wwe2k14 patch (Session 52): a held-back barrier goes before any command that is not a bind or state.
  if (barrier_pending_ && command != Command::kVkPipelineBarrier && !IsStateCommand(command)) {
    FlushPendingBarrierSlow();
  }
  size_t arguments_size_elements =
      (arguments_size_bytes + sizeof(uintmax_t) - 1) / sizeof(uintmax_t);
  size_t offset = command_stream_.size();
  command_stream_.resize(offset + kCommandHeaderSizeElements + arguments_size_elements);
  CommandHeader& header = *reinterpret_cast<CommandHeader*>(command_stream_.data() + offset);
  header.command = command;
  header.arguments_size_elements = uint32_t(arguments_size_elements);
  return command_stream_.data() + (offset + kCommandHeaderSizeElements);
}

}  // namespace rex::graphics::vulkan
