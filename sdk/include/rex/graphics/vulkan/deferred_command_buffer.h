#pragma once
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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <rex/assert.h>
#include <rex/math.h>
#include <rex/ui/vulkan/api.h>

namespace rex::graphics::vulkan {

class VulkanCommandProcessor;

class DeferredCommandBuffer {
 public:
  DeferredCommandBuffer(const VulkanCommandProcessor& command_processor,
                        size_t initial_size_bytes = 1024 * 1024);

  void Reset();
  void Execute(VkCommandBuffer command_buffer) {
    if (lazy_sync_pending_) FlushLazySync();  // wwe2k14 patch (Session 56: the stream's end)
    FlushPendingBarrier();  // wwe2k14 patch (Session 52)
    ExecuteStream(command_stream_, command_buffer);
  }

  // wwe2k14 patch (vulkan_async_submit): the recorded stream can be handed to the submit thread, which replays
  // it with ExecuteStream (only reads the stream and the device's function table).
  void SwapStream(std::vector<uintmax_t>& other) {
    if (lazy_sync_pending_) FlushLazySync();  // wwe2k14 patch (Session 56: the stream's end)
    FlushPendingBarrier();  // wwe2k14 patch (Session 52)
    command_stream_.swap(other);
    last_draw_end_ = SIZE_MAX;  // wwe2k14 patch (Session 23, MERGE)
    last_begin_offset_ = SIZE_MAX;  // wwe2k14 patch (Session 73)
  }

  // wwe2k14 patch (Session 73): the last recorded vkCmdBeginRenderPass (its arguments' place in the stream, SIZE_MAX
  // after a Reset / SwapStream), and replacing its render pass by a compatible one (different load / store operations)
  // once the pass's draws are known. False (nothing changed) if the stream has moved on or the pass differs.
  size_t last_begin_render_pass_offset() const { return last_begin_offset_; }
  // wwe2k14 patch (Session 79, NF-D dead passes): the stream's end (the next command's offset, in elements), and
  // turning every command recorded in [begin_offset, end_offset) into kNop (skipped on replay). The range must start on
  // a command boundary; false (nothing changed) if it does not end on one.
  size_t stream_offset() const { return command_stream_.size(); }
  // wwe2k14 patch (Session 80, gpu_native_truth_drop_dead_copies): write what is held back (the lazy passes' barrier,
  // a pending barrier) now, as Execute does - so a range recorded next holds only its own commands (NopRange).
  void FlushHeldCommands() {
    if (lazy_sync_pending_) FlushLazySync();
    FlushPendingBarrier();
  }
  bool NopRange(size_t begin_offset, size_t end_offset) {
    size_t at = begin_offset;
    while (at < end_offset && at < command_stream_.size()) {
      const auto& header = *reinterpret_cast<const CommandHeader*>(command_stream_.data() + at);
      at += kCommandHeaderSizeElements + header.arguments_size_elements;
    }
    if (at != end_offset) return false;
    for (at = begin_offset; at < end_offset;) {
      auto& header = *reinterpret_cast<CommandHeader*>(command_stream_.data() + at);
      at += kCommandHeaderSizeElements + header.arguments_size_elements;
      header.command = Command::kNop;
    }
    return true;
  }
  // wwe2k14 patch (Session 80, gpu_native_truth_drop_dead_passes): whether only commands that read and write no image
  // or buffer (state commands, timestamps, checkpoints, barriers, no-ops) were recorded from `offset` (a command
  // boundary) to the stream's end.
  bool OnlyStateSince(size_t offset) const {
    size_t at = offset;
    while (at < command_stream_.size()) {
      const auto& header = *reinterpret_cast<const CommandHeader*>(command_stream_.data() + at);
      if (!KeepsLazyChain(header.command) && header.command != Command::kVkPipelineBarrier &&
          header.command != Command::kNop) {
        return false;
      }
      at += kCommandHeaderSizeElements + header.arguments_size_elements;
    }
    return at == command_stream_.size();
  }
  // wwe2k14 patch (Session 80, gpu_native_truth_drop_dead_passes): the render pass recorded from the vkCmdBeginRenderPass
  // whose arguments are at `begin_args_offset` (last_begin_render_pass_offset) to `end_offset` (right after its end)
  // turned off: its begin, draws and end become kNop; its state commands, timestamps and checkpoints stay (later
  // commands were recorded relying on the state it set). False (nothing changed) if the range holds anything else
  // (a barrier, a clear, a query) or is not on command boundaries.
  // (Session 80, gpu_native_own_steps_drop_clears: with_clears - its attachment clears turned off too.)
  bool NopRenderPass(size_t begin_args_offset, size_t end_offset, bool with_clears = false, bool dry_run = false /* Session 95: only check */) {
    if (begin_args_offset == SIZE_MAX || begin_args_offset < kCommandHeaderSizeElements) return false;
    const size_t begin = begin_args_offset - kCommandHeaderSizeElements;
    if (begin >= end_offset || end_offset > command_stream_.size()) return false;
    const auto off = [with_clears](Command c) {
      return c == Command::kVkBeginRenderPass || c == Command::kVkEndRenderPass || c == Command::kVkDraw ||
             c == Command::kVkDrawIndexed || (with_clears && c == Command::kVkClearAttachments);
    };
    // (Session 95: an armed frame's per-draw marks go with their draws, so a dropped draw leaves no stale id for the next one.)
    const auto off_header = [&off](const CommandHeader& h) { return off(h.command) || IsDrawMark(h); };
    size_t at = begin;
    Command first = Command::kNop, last = Command::kNop;
    while (at < end_offset) {
      const auto& header = *reinterpret_cast<const CommandHeader*>(command_stream_.data() + at);
      // (with_clears: descriptor set writes kept too - a host-side update later draws rely on, no GPU work.)
      if (!off(header.command) && !KeepsLazyChain(header.command) && header.command != Command::kNop &&
          !(with_clears && header.command == Command::kVkUpdateDescriptorSets)) {
        return false;
      }
      if (at == begin) first = header.command;
      if (off(header.command)) last = header.command;  // (a draw mark does not count as one of these)
      at += kCommandHeaderSizeElements + header.arguments_size_elements;
    }
    if (at != end_offset || first != Command::kVkBeginRenderPass || last != Command::kVkEndRenderPass) return false;
    if (dry_run) return true;
    for (at = begin; at < end_offset;) {
      auto& header = *reinterpret_cast<CommandHeader*>(command_stream_.data() + at);
      at += kCommandHeaderSizeElements + header.arguments_size_elements;
      if (off_header(header)) header.command = Command::kNop;
    }
    return true;
  }
  // wwe2k14 patch (Session 80, owned passes): the draws and attachment clears from `begin_offset` to `end_offset` (both
  // command boundaries, inside one render pass: a pass's clear draws after its loads, up to its end command) turned
  // off; its state commands, timestamps, checkpoints and descriptor writes stay. False (nothing changed) if the range
  // holds anything else or is not on command boundaries.
  // (Session 95, gpu_owned_restore_skip: keep_clears - the range's attachment clears stay.)
  bool NopDraws(size_t begin_offset, size_t end_offset, bool keep_clears = false) {
    if (begin_offset == SIZE_MAX || begin_offset >= end_offset || end_offset > command_stream_.size()) return false;
    const auto off = [keep_clears](Command c) {
      return c == Command::kVkDraw || c == Command::kVkDrawIndexed || (!keep_clears && c == Command::kVkClearAttachments);
    };
    const auto off_header = [&off](const CommandHeader& h) { return off(h.command) || IsDrawMark(h); };  // (Session 95: see NopRenderPass)
    size_t at = begin_offset;
    while (at < end_offset) {
      const auto& header = *reinterpret_cast<const CommandHeader*>(command_stream_.data() + at);
      if (!off(header.command) && !KeepsLazyChain(header.command) && header.command != Command::kNop &&
          header.command != Command::kVkUpdateDescriptorSets && !(keep_clears && header.command == Command::kVkClearAttachments)) {
        return false;
      }
      at += kCommandHeaderSizeElements + header.arguments_size_elements;
    }
    if (at != end_offset) return false;
    for (at = begin_offset; at < end_offset;) {
      auto& header = *reinterpret_cast<CommandHeader*>(command_stream_.data() + at);
      at += kCommandHeaderSizeElements + header.arguments_size_elements;
      if (off_header(header)) header.command = Command::kNop;
    }
    return true;
  }
  // (Session 80, diagnostic) The commands from the begin whose arguments are at `begin_args_offset` to `end_offset`,
  // as numbers (Command), and whether the range ends on a command boundary.
  std::string DescribeRange(size_t begin_args_offset, size_t end_offset) const {
    std::string s;
    if (begin_args_offset == SIZE_MAX || begin_args_offset < kCommandHeaderSizeElements) return "no begin";
    size_t at = begin_args_offset - kCommandHeaderSizeElements;
    while (at < end_offset && at < command_stream_.size()) {
      const auto& header = *reinterpret_cast<const CommandHeader*>(command_stream_.data() + at);
      s += std::to_string(int(header.command)) + " ";
      at += kCommandHeaderSizeElements + header.arguments_size_elements;
    }
    if (at != end_offset) s += "(not on a boundary: " + std::to_string(at) + " vs " + std::to_string(end_offset) + ")";
    return s;
  }
  // wwe2k14 patch (Session 95, gpu_owned_graph): a read-only description of the recorded stream, for the frame graph's walk at a
  // submission's end (owned_graph.cpp). One event per image barrier entry, render pass begin / end, image copy, blit, clear,
  // image-to-buffer / buffer-to-image copy, dispatch, host callback and query command; draws, binds and state are not events.
  // `at` = the command's header offset (elements), `next` = the offset right after the command (NopRenderPass's end_offset).
  struct GraphEvent {
    enum Kind : uint8_t {
      kBarrierImage, kBegin, kEnd, kCopyImage, kBlit, kClearColor, kClearDepth, kImageToBuffer, kBufferToImage, kDispatch,
      kHostCallback, kQuery
    };
    uint32_t at = 0, next = 0, args = 0;  // (args: the offset of the command's arguments - a begin's is NopRenderPass's begin_args_offset)
    Kind kind = kBarrierImage;
    uint8_t aspect = 0;      // barrier: the aspect mask (low 8 bits); copies / blits / clears: the destination's aspect (first region)
    uint8_t src_aspect = 0;  // copies / blits: the source's aspect (first region)
    uint8_t pad = 0;
    int32_t old_layout = 0, new_layout = 0;  // barrier (VkImageLayout)
    uint64_t a = 0, b = 0;  // barrier: image; begin: render pass, framebuffer; copy / blit: source, destination image; clear: image
  };
  // Appends the stream's events to `out` (not cleared); false (out may be partial) when more than `max_events` events would be needed.
  // The caller flushes held commands first (FlushHeldCommands).
  bool ScanStream(std::vector<GraphEvent>& out, size_t max_events) const;
  // The command at `header_offset` turned off (kNop), only when it is an image copy, blit or image clear (never a barrier, a pass or a
  // dispatch). False (nothing changed) otherwise.
  bool NopImageCommandAt(size_t header_offset) {
    if (header_offset >= command_stream_.size()) return false;
    auto& header = *reinterpret_cast<CommandHeader*>(command_stream_.data() + header_offset);
    if (header.command != Command::kVkCopyImage && header.command != Command::kVkBlitImage &&
        header.command != Command::kVkClearColorImage && header.command != Command::kVkClearDepthStencilImage) {
      return false;
    }
    header.command = Command::kNop;
    return true;
  }
  bool PatchBeginRenderPass(size_t offset, VkRenderPass expected, VkRenderPass replacement) {
    if (offset == SIZE_MAX || offset < kCommandHeaderSizeElements || offset >= command_stream_.size()) return false;
    const CommandHeader& header =
        *reinterpret_cast<const CommandHeader*>(command_stream_.data() + offset - kCommandHeaderSizeElements);
    if (header.command != Command::kVkBeginRenderPass) return false;
    auto& args = *reinterpret_cast<ArgsVkBeginRenderPass*>(command_stream_.data() + offset);
    if (args.render_pass != expected) return false;
    args.render_pass = replacement;
    return true;
  }

  // wwe2k14 patch (Session 23, MERGE): vulkan_repeat_draw_merge. If the last command in the stream is a draw
  // (CmdVkDraw / CmdVkDrawIndexed as given), adds one instance to it and returns true; otherwise changes nothing.
  bool AddInstanceToLastDraw(bool indexed) {
    // (Session 52: a barrier recorded after the draw and not written yet is between them.)
    if (barrier_pending_ || last_draw_end_ != command_stream_.size() || last_draw_indexed_ != indexed) {
      return false;
    }
    uintmax_t* args = command_stream_.data() + last_draw_args_;
    if (indexed) {
      ++reinterpret_cast<ArgsVkDrawIndexed*>(args)->instance_count;
    } else {
      ++reinterpret_cast<ArgsVkDraw*>(args)->instance_count;
    }
    return true;
  }
  // wwe2k14 patch (Session 93, gpu_native_own_crowd_merge): as AddInstanceToLastDraw, `count` instances, only for an indexed draw that
  // still has `expect_index_count` indices and `expect_stream_end` as the stream's size (our own draw, nothing recorded after it).
  bool AddInstancesToLastIndexedDraw(uint32_t count, uint32_t expect_index_count, size_t expect_stream_end) {
    if (barrier_pending_ || last_draw_end_ != command_stream_.size() || !last_draw_indexed_ ||
        command_stream_.size() != expect_stream_end) {
      return false;
    }
    auto* args = reinterpret_cast<ArgsVkDrawIndexed*>(command_stream_.data() + last_draw_args_);
    if (args->index_count != expect_index_count) return false;
    args->instance_count += count;
    return true;
  }
  // wwe2k14 patch (Session 93, gpu_native_own_crowd_merge): as AddInstancesToLastIndexedDraw, but commands may follow the draw: from
  // `expect_stream_end` (the stream's size right after our draw) to the stream's end there may be only the state commands a crowd
  // run's first member recorded (pipeline, descriptor set, index buffer, viewport, scissor, depth bias, blend constants and stencil
  // binds - turned into no-ops: the caller compared their values with what our draw drew with) and host-side descriptor writes and
  // no-ops (kept). A barrier, timestamp, checkpoint, push constant, draw, pass command or anything else refuses (nothing changed;
  // `refused_stream` tells it from a draw that is not ours). The scan resumes where the last merge stopped.
  bool MergeInstancesIntoLastIndexedDraw(uint32_t count, uint32_t expect_index_count, size_t expect_stream_end,
                                         bool& refused_stream) {
    refused_stream = false;
    if (barrier_pending_ || last_draw_end_ != expect_stream_end || !last_draw_indexed_ ||
        expect_stream_end > command_stream_.size()) {
      return false;
    }
    auto* draw_args = reinterpret_cast<ArgsVkDrawIndexed*>(command_stream_.data() + last_draw_args_);
    if (draw_args->index_count != expect_index_count) return false;
    const size_t begin = merged_scan_end_ != SIZE_MAX && merged_scan_end_ >= expect_stream_end &&
                                 merged_scan_end_ <= command_stream_.size()
                             ? merged_scan_end_
                             : expect_stream_end;
    const auto nopable = [](Command c) {
      switch (c) {
        case Command::kVkBindDescriptorSets:
        case Command::kVkBindIndexBuffer:
        case Command::kVkBindPipeline:
        case Command::kVkSetBlendConstants:
        case Command::kVkSetDepthBias:
        case Command::kVkSetScissor:
        case Command::kVkSetStencilCompareMask:
        case Command::kVkSetStencilReference:
        case Command::kVkSetStencilWriteMask:
        case Command::kVkSetViewport:
          return true;
        default:
          return false;
      }
    };
    size_t at = begin;
    while (at < command_stream_.size()) {
      const auto& header = *reinterpret_cast<const CommandHeader*>(command_stream_.data() + at);
      if (!nopable(header.command) && header.command != Command::kNop && header.command != Command::kVkUpdateDescriptorSets &&
          header.command != Command::kVkSetCheckpointNV) {  // (a marker, kept in place)
        refused_stream = true;
        if (size_t(header.command) < kMergeRefusalKinds) ++merge_refused_by_[size_t(header.command)];  // (log only)
        return false;
      }
      at += kCommandHeaderSizeElements + header.arguments_size_elements;
    }
    if (at != command_stream_.size()) {
      refused_stream = true;
      return false;
    }
    for (at = begin; at < command_stream_.size();) {
      auto& header = *reinterpret_cast<CommandHeader*>(command_stream_.data() + at);
      at += kCommandHeaderSizeElements + header.arguments_size_elements;
      if (nopable(header.command)) header.command = Command::kNop;
    }
    draw_args->instance_count += count;
    debug_last_constants_set_ = VK_NULL_HANDLE;  // (the bind that could have set it is a no-op now)
    merged_scan_end_ = command_stream_.size();
    return true;
  }
  // (Session 93, log only) which command refused each merge: counts by Command value, and the names.
  static constexpr size_t kMergeRefusalKinds = 64;
  uint64_t merge_refused_by_[kMergeRefusalKinds] = {};
  static const char* CommandName(size_t c) {
    switch (Command(c)) {
      case Command::kVkBeginRenderPass: return "kVkBeginRenderPass";
      case Command::kVkBeginQuery: return "kVkBeginQuery";
      case Command::kVkBindDescriptorSets: return "kVkBindDescriptorSets";
      case Command::kVkBindIndexBuffer: return "kVkBindIndexBuffer";
      case Command::kVkBindPipeline: return "kVkBindPipeline";
      case Command::kVkBindVertexBuffers: return "kVkBindVertexBuffers";
      case Command::kVkClearAttachments: return "kVkClearAttachments";
      case Command::kVkClearColorImage: return "kVkClearColorImage";
      case Command::kVkCopyBuffer: return "kVkCopyBuffer";
      case Command::kVkCopyBufferToImage: return "kVkCopyBufferToImage";
      case Command::kVkCopyImageToBuffer: return "kVkCopyImageToBuffer";
      case Command::kVkCopyQueryPoolResults: return "kVkCopyQueryPoolResults";
      case Command::kVkDispatch: return "kVkDispatch";
      case Command::kVkDraw: return "kVkDraw";
      case Command::kVkDrawIndexed: return "kVkDrawIndexed";
      case Command::kVkEndQuery: return "kVkEndQuery";
      case Command::kVkEndRenderPass: return "kVkEndRenderPass";
      case Command::kVkBeginRendering: return "kVkBeginRendering";
      case Command::kVkEndRendering: return "kVkEndRendering";
      case Command::kVkPipelineBarrier: return "kVkPipelineBarrier";
      case Command::kVkPushConstants: return "kVkPushConstants";
      case Command::kVkResetQueryPool: return "kVkResetQueryPool";
      case Command::kVkSetBlendConstants: return "kVkSetBlendConstants";
      case Command::kVkSetCheckpointNV: return "kVkSetCheckpointNV";
      case Command::kVkSetDepthBias: return "kVkSetDepthBias";
      case Command::kVkSetEdsState: return "kVkSetEdsState";
      case Command::kVkSetScissor: return "kVkSetScissor";
      case Command::kVkSetStencilCompareMask: return "kVkSetStencilCompareMask";
      case Command::kVkSetStencilReference: return "kVkSetStencilReference";
      case Command::kVkSetStencilWriteMask: return "kVkSetStencilWriteMask";
      case Command::kVkSetViewport: return "kVkSetViewport";
      case Command::kVkUpdateDescriptorSets: return "kVkUpdateDescriptorSets";
      case Command::kVkWriteTimestamp: return "kVkWriteTimestamp";
      case Command::kVkCopyImage: return "kVkCopyImage";
      case Command::kHostCallback: return "kHostCallback";
      case Command::kVkBlitImage: return "kVkBlitImage";
      case Command::kVkClearDepthStencilImage: return "kVkClearDepthStencilImage";
      case Command::kNop: return "kNop";
      case Command::kDiagMark: return "kDiagMark";
      default: return "?";
    }
  }
  void ExecuteStream(const std::vector<uintmax_t>& stream, VkCommandBuffer command_buffer) const;
  // wwe2k14 patch (Session 95, the diagnostic run): native/diag_layer.h. A mark in the stream: the armed frame's edges and its draws, seen by the replay in order.
  // Written only on an armed frame; the plain replay skips it.
  void CmdDiagMark(uint32_t kind, uint32_t id, uint32_t aux) {
    auto& args = *reinterpret_cast<ArgsDiagMark*>(WriteCommand(Command::kDiagMark, sizeof(ArgsDiagMark)));
    args.kind = kind;
    args.id = id;
    args.aux = aux;
  }

  // wwe2k14 patch (Session 7c, CAUSES): measurement only, nothing recorded changes. Running counts of recorded
  // compute dispatches and render pass / dynamic rendering begins (never reset, not even by Reset; readers take
  // differences, which wrap correctly), for the frame-anatomy cause columns and the cpf_sync events.
  uint32_t dispatches_recorded() const { return dispatches_recorded_; }
  uint32_t rendering_begins_recorded() const { return rendering_begins_recorded_; }
  // wwe2k14 patch (Session 20, RECSTATS): measurement only. Running counts of recorded commands (never reset;
  // readers take differences), read by gpu_record_stats.
  struct RecordStats {
    enum : uint32_t {
      kBarrierBatches, kMemoryBarriers, kBufferBarriers, kImageBarriers, kRenderingBegins, kPipelineBinds,
      kDescriptorBinds, kDescriptorSets, kPushConstants, kPushConstantBytes, kVertexIndexBinds, kViewportScissor,
      kDraws, kDispatches, kCount
    };
    uint64_t c[kCount] = {};
  };
  const RecordStats& record_stats() const { return record_stats_; }
  // gpu_anatomy_gpu_marks_mode 1: while set, the command processor's GpuMarksSparseBeforeEvent is called (once,
  // the hook is cleared first) right before the next "event" command is written: a pipeline barrier, a render
  // pass / dynamic rendering begin or end, a pipeline bind, a dispatch, a copy or a clear (not descriptor set,
  // vertex / index buffer binds, push constants, dynamic state, draws, queries, timestamps or checkpoints).
  // nullptr: off (one pointer test per such command).
  void SetEventHook(VulkanCommandProcessor* hook) { event_hook_ = hook; }

  // render_pass_begin->pNext of all barriers must be null.
  void CmdVkBeginRenderPass(const VkRenderPassBeginInfo* render_pass_begin,
                            VkSubpassContents contents) {
    assert_null(render_pass_begin->pNext);
    BeforeEventCommand();  // wwe2k14 patch (Session 7c, CAUSES)
    ++rendering_begins_recorded_;
    ++record_stats_.c[RecordStats::kRenderingBegins];  // wwe2k14 patch (Session 20, RECSTATS)
    size_t arguments_size = sizeof(ArgsVkBeginRenderPass);
    uint32_t clear_value_count = render_pass_begin->clearValueCount;
    size_t clear_values_offset = 0;
    if (clear_value_count) {
      arguments_size = rex::align(arguments_size, alignof(VkClearValue));
      clear_values_offset = arguments_size;
      arguments_size += sizeof(VkClearValue) * clear_value_count;
    }
    uint8_t* args_ptr =
        reinterpret_cast<uint8_t*>(WriteCommand(Command::kVkBeginRenderPass, arguments_size));
    // wwe2k14 patch (Session 73): where its arguments are (PatchBeginRenderPass).
    last_begin_offset_ = size_t(reinterpret_cast<uintmax_t*>(args_ptr) - command_stream_.data());
    auto& args = *reinterpret_cast<ArgsVkBeginRenderPass*>(args_ptr);
    args.render_pass = render_pass_begin->renderPass;
    args.framebuffer = render_pass_begin->framebuffer;
    args.render_area = render_pass_begin->renderArea;
    args.clear_value_count = clear_value_count;
    args.contents = contents;
    if (clear_value_count) {
      std::memcpy(args_ptr + clear_values_offset, render_pass_begin->pClearValues,
                  sizeof(VkClearValue) * clear_value_count);
    }
  }

  void CmdVkBindDescriptorSets(VkPipelineBindPoint pipeline_bind_point, VkPipelineLayout layout,
                               uint32_t first_set, uint32_t descriptor_set_count,
                               const VkDescriptorSet* descriptor_sets,
                               uint32_t dynamic_offset_count, const uint32_t* dynamic_offsets) {
    // wwe2k14 patch (Session 59, diagnostic vulkan_debug_skip_constants_rebind: pictures wrong): a graphics bind of
    // only the constants set (1) with the set already bound - new dynamic offsets alone - is not recorded.
    if (debug_skip_constants_rebind && pipeline_bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
      if (layout != debug_last_layout_ || first_set == 0) debug_last_constants_set_ = VK_NULL_HANDLE;
      debug_last_layout_ = layout;
      if (first_set == 1 && descriptor_set_count == 1 && dynamic_offset_count) {
        if (descriptor_sets[0] == debug_last_constants_set_) return;
        debug_last_constants_set_ = descriptor_sets[0];
      }
    }
    ++record_stats_.c[RecordStats::kDescriptorBinds];  // wwe2k14 patch (Session 20, RECSTATS)
    record_stats_.c[RecordStats::kDescriptorSets] += descriptor_set_count;
    size_t arguments_size = rex::align(sizeof(ArgsVkBindDescriptorSets), alignof(VkDescriptorSet));
    size_t descriptor_sets_offset = arguments_size;
    arguments_size += sizeof(VkDescriptorSet) * descriptor_set_count;
    size_t dynamic_offsets_offset = 0;
    if (dynamic_offset_count) {
      arguments_size = rex::align(arguments_size, alignof(uint32_t));
      dynamic_offsets_offset = arguments_size;
      arguments_size += sizeof(uint32_t) * dynamic_offset_count;
    }
    uint8_t* args_ptr =
        reinterpret_cast<uint8_t*>(WriteCommand(Command::kVkBindDescriptorSets, arguments_size));
    auto& args = *reinterpret_cast<ArgsVkBindDescriptorSets*>(args_ptr);
    args.pipeline_bind_point = pipeline_bind_point;
    args.layout = layout;
    args.first_set = first_set;
    args.descriptor_set_count = descriptor_set_count;
    args.dynamic_offset_count = dynamic_offset_count;
    std::memcpy(args_ptr + descriptor_sets_offset, descriptor_sets,
                sizeof(VkDescriptorSet) * descriptor_set_count);
    if (dynamic_offset_count) {
      std::memcpy(args_ptr + dynamic_offsets_offset, dynamic_offsets,
                  sizeof(uint32_t) * dynamic_offset_count);
    }
  }

  void CmdVkBindIndexBuffer(VkBuffer buffer, VkDeviceSize offset, VkIndexType index_type) {
    ++record_stats_.c[RecordStats::kVertexIndexBinds];  // wwe2k14 patch (Session 20, RECSTATS)
    auto& args = *reinterpret_cast<ArgsVkBindIndexBuffer*>(
        WriteCommand(Command::kVkBindIndexBuffer, sizeof(ArgsVkBindIndexBuffer)));
    args.buffer = buffer;
    args.offset = offset;
    args.index_type = index_type;
  }

  void CmdVkBindPipeline(VkPipelineBindPoint pipeline_bind_point, VkPipeline pipeline) {
    BeforeEventCommand();  // wwe2k14 patch (Session 7c, CAUSES)
    ++record_stats_.c[RecordStats::kPipelineBinds];  // wwe2k14 patch (Session 20, RECSTATS)
    auto& args = *reinterpret_cast<ArgsVkBindPipeline*>(
        WriteCommand(Command::kVkBindPipeline, sizeof(ArgsVkBindPipeline)));
    args.pipeline_bind_point = pipeline_bind_point;
    args.pipeline = pipeline;
  }

  void CmdVkBindVertexBuffers(uint32_t first_binding, uint32_t binding_count,
                              const VkBuffer* buffers, const VkDeviceSize* offsets) {
    ++record_stats_.c[RecordStats::kVertexIndexBinds];  // wwe2k14 patch (Session 20, RECSTATS)
    size_t arguments_size = rex::align(sizeof(ArgsVkBindVertexBuffers), alignof(VkBuffer));
    size_t buffers_offset = arguments_size;
    arguments_size =
        rex::align(arguments_size + sizeof(VkBuffer) * binding_count, alignof(VkDeviceSize));
    size_t offsets_offset = arguments_size;
    arguments_size += sizeof(VkDeviceSize) * binding_count;
    uint8_t* args_ptr =
        reinterpret_cast<uint8_t*>(WriteCommand(Command::kVkBindVertexBuffers, arguments_size));
    auto& args = *reinterpret_cast<ArgsVkBindVertexBuffers*>(args_ptr);
    args.first_binding = first_binding;
    args.binding_count = binding_count;
    std::memcpy(args_ptr + buffers_offset, buffers, sizeof(VkBuffer) * binding_count);
    std::memcpy(args_ptr + offsets_offset, offsets, sizeof(VkDeviceSize) * binding_count);
  }

  void CmdClearAttachmentsEmplace(uint32_t attachment_count, VkClearAttachment*& attachments_out,
                                  uint32_t rect_count, VkClearRect*& rects_out) {
    BeforeEventCommand();  // wwe2k14 patch (Session 7c, CAUSES)
    size_t arguments_size = rex::align(sizeof(ArgsVkClearAttachments), alignof(VkClearAttachment));
    size_t attachments_offset = arguments_size;
    arguments_size = rex::align(arguments_size + sizeof(VkClearAttachment) * attachment_count,
                                alignof(VkClearRect));
    size_t rects_offset = arguments_size;
    arguments_size += sizeof(VkClearRect) * rect_count;
    uint8_t* args_ptr =
        reinterpret_cast<uint8_t*>(WriteCommand(Command::kVkClearAttachments, arguments_size));
    auto& args = *reinterpret_cast<ArgsVkClearAttachments*>(args_ptr);
    args.attachment_count = attachment_count;
    args.rect_count = rect_count;
    attachments_out = reinterpret_cast<VkClearAttachment*>(args_ptr + attachments_offset);
    rects_out = reinterpret_cast<VkClearRect*>(args_ptr + rects_offset);
  }
  void CmdVkClearAttachments(uint32_t attachment_count, const VkClearAttachment* attachments,
                             uint32_t rect_count, const VkClearRect* rects) {
    VkClearAttachment* attachments_arg;
    VkClearRect* rects_arg;
    CmdClearAttachmentsEmplace(attachment_count, attachments_arg, rect_count, rects_arg);
    std::memcpy(attachments_arg, attachments, sizeof(VkClearAttachment) * attachment_count);
    std::memcpy(rects_arg, rects, sizeof(VkClearRect) * rect_count);
  }

  VkImageSubresourceRange* CmdClearColorImageEmplace(VkImage image, VkImageLayout image_layout,
                                                     const VkClearColorValue* color,
                                                     uint32_t range_count) {
    BeforeEventCommand();  // wwe2k14 patch (Session 7c, CAUSES)
    const size_t header_size =
        rex::align(sizeof(ArgsVkClearColorImage), alignof(VkImageSubresourceRange));
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(WriteCommand(
        Command::kVkClearColorImage, header_size + sizeof(VkImageSubresourceRange) * range_count));
    auto& args = *reinterpret_cast<ArgsVkClearColorImage*>(args_ptr);
    args.image = image;
    args.image_layout = image_layout;
    args.color = *color;
    args.range_count = range_count;
    return reinterpret_cast<VkImageSubresourceRange*>(args_ptr + header_size);
  }
  void CmdVkClearColorImage(VkImage image, VkImageLayout image_layout,
                            const VkClearColorValue* color, uint32_t range_count,
                            const VkImageSubresourceRange* ranges) {
    std::memcpy(CmdClearColorImageEmplace(image, image_layout, color, range_count), ranges,
                sizeof(VkImageSubresourceRange) * range_count);
  }

  VkBufferCopy* CmdCopyBufferEmplace(VkBuffer src_buffer, VkBuffer dst_buffer,
                                     uint32_t region_count) {
    BeforeEventCommand();  // wwe2k14 patch (Session 7c, CAUSES)
    const size_t header_size = rex::align(sizeof(ArgsVkCopyBuffer), alignof(VkBufferCopy));
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kVkCopyBuffer, header_size + sizeof(VkBufferCopy) * region_count));
    auto& args = *reinterpret_cast<ArgsVkCopyBuffer*>(args_ptr);
    args.src_buffer = src_buffer;
    args.dst_buffer = dst_buffer;
    args.region_count = region_count;
    return reinterpret_cast<VkBufferCopy*>(args_ptr + header_size);
  }
  void CmdVkCopyBuffer(VkBuffer src_buffer, VkBuffer dst_buffer, uint32_t region_count,
                       const VkBufferCopy* regions) {
    std::memcpy(CmdCopyBufferEmplace(src_buffer, dst_buffer, region_count), regions,
                sizeof(VkBufferCopy) * region_count);
  }

  void CmdVkBeginQuery(VkQueryPool query_pool, uint32_t query, VkQueryControlFlags flags) {
    auto& args = *reinterpret_cast<ArgsVkBeginQuery*>(
        WriteCommand(Command::kVkBeginQuery, sizeof(ArgsVkBeginQuery)));
    args.query_pool = query_pool;
    args.query = query;
    args.flags = flags;
  }

  VkBufferImageCopy* CmdCopyBufferToImageEmplace(VkBuffer src_buffer, VkImage dst_image,
                                                 VkImageLayout dst_image_layout,
                                                 uint32_t region_count) {
    BeforeEventCommand();  // wwe2k14 patch (Session 7c, CAUSES)
    const size_t header_size =
        rex::align(sizeof(ArgsVkCopyBufferToImage), alignof(VkBufferImageCopy));
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(WriteCommand(
        Command::kVkCopyBufferToImage, header_size + sizeof(VkBufferImageCopy) * region_count));
    auto& args = *reinterpret_cast<ArgsVkCopyBufferToImage*>(args_ptr);
    args.src_buffer = src_buffer;
    args.dst_image = dst_image;
    args.dst_image_layout = dst_image_layout;
    args.region_count = region_count;
    return reinterpret_cast<VkBufferImageCopy*>(args_ptr + header_size);
  }
  void CmdVkCopyBufferToImage(VkBuffer src_buffer, VkImage dst_image,
                              VkImageLayout dst_image_layout, uint32_t region_count,
                              const VkBufferImageCopy* regions) {
    std::memcpy(CmdCopyBufferToImageEmplace(src_buffer, dst_image, dst_image_layout, region_count),
                regions, sizeof(VkBufferImageCopy) * region_count);
  }

  // wwe2k14 patch (Session 18, DLAA): calls fn(command_buffer, copy of `data`) when the buffer is executed (for a
  // library that records into a real command buffer itself: Streamline's slEvaluateFeature). The caller must
  // re-bind any state the callback may change.
  using HostCallback = void (*)(VkCommandBuffer command_buffer, const void* data);
  void CmdHostCallback(HostCallback fn, const void* data, size_t data_size) {
    BeforeEventCommand();
    const size_t header_size = rex::align(sizeof(ArgsHostCallback), alignof(uintmax_t));
    uint8_t* args_ptr =
        reinterpret_cast<uint8_t*>(WriteCommand(Command::kHostCallback, header_size + data_size));
    auto& args = *reinterpret_cast<ArgsHostCallback*>(args_ptr);
    args.fn = fn;
    args.data_size = data_size;
    std::memcpy(args_ptr + header_size, data, data_size);
  }

  // wwe2k14 patch (Session 18, DLAA): image to image (the DLAA pass's output into a cached texture).
  void CmdVkCopyImage(VkImage src_image, VkImageLayout src_image_layout, VkImage dst_image,
                      VkImageLayout dst_image_layout, uint32_t region_count, const VkImageCopy* regions) {
    BeforeEventCommand();  // A copy, like the other copies (Session 7c, CAUSES).
    const size_t header_size = rex::align(sizeof(ArgsVkCopyImage), alignof(VkImageCopy));
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kVkCopyImage, header_size + sizeof(VkImageCopy) * region_count));
    auto& args = *reinterpret_cast<ArgsVkCopyImage*>(args_ptr);
    args.src_image = src_image;
    args.src_image_layout = src_image_layout;
    args.dst_image = dst_image;
    args.dst_image_layout = dst_image_layout;
    args.region_count = region_count;
    std::memcpy(args_ptr + header_size, regions, sizeof(VkImageCopy) * region_count);
  }

  // wwe2k14 patch (Session 54, NATIVE without interlock): a new truth depth image cleared (no EDRAM to load from).
  void CmdVkClearDepthStencilImage(VkImage image, VkImageLayout image_layout, const VkClearDepthStencilValue* value,
                                   const VkImageSubresourceRange& range) {
    BeforeEventCommand();  // A transfer, like the copies.
    auto& args = *reinterpret_cast<ArgsVkClearDepthStencilImage*>(
        WriteCommand(Command::kVkClearDepthStencilImage, sizeof(ArgsVkClearDepthStencilImage)));
    args.image = image;
    args.image_layout = image_layout;
    args.value = *value;
    args.range = range;
  }

  // wwe2k14 patch (Session 54, NATIVE draw sizes): the truth images' draw twins scaled to and from their images.
  void CmdVkBlitImage(VkImage src_image, VkImageLayout src_image_layout, VkImage dst_image,
                      VkImageLayout dst_image_layout, uint32_t region_count, const VkImageBlit* regions,
                      VkFilter filter) {
    BeforeEventCommand();  // A copy, like the other copies.
    const size_t header_size = rex::align(sizeof(ArgsVkBlitImage), alignof(VkImageBlit));
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kVkBlitImage, header_size + sizeof(VkImageBlit) * region_count));
    auto& args = *reinterpret_cast<ArgsVkBlitImage*>(args_ptr);
    args.src_image = src_image;
    args.src_image_layout = src_image_layout;
    args.dst_image = dst_image;
    args.dst_image_layout = dst_image_layout;
    args.region_count = region_count;
    args.filter = filter;
    std::memcpy(args_ptr + header_size, regions, sizeof(VkImageBlit) * region_count);
  }

  // wwe2k14 patch (Session 7c, TEXDIRECT): for the vulkan_texture_load_direct_verify readback.
  void CmdVkCopyImageToBuffer(VkImage src_image, VkImageLayout src_image_layout, VkBuffer dst_buffer,
                              uint32_t region_count, const VkBufferImageCopy* regions) {
    BeforeEventCommand();  // A copy, like the other copies (Session 7c, CAUSES).
    const size_t header_size =
        rex::align(sizeof(ArgsVkCopyImageToBuffer), alignof(VkBufferImageCopy));
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(WriteCommand(
        Command::kVkCopyImageToBuffer, header_size + sizeof(VkBufferImageCopy) * region_count));
    auto& args = *reinterpret_cast<ArgsVkCopyImageToBuffer*>(args_ptr);
    args.src_image = src_image;
    args.src_image_layout = src_image_layout;
    args.dst_buffer = dst_buffer;
    args.region_count = region_count;
    std::memcpy(args_ptr + header_size, regions, sizeof(VkBufferImageCopy) * region_count);
  }

  void CmdVkCopyQueryPoolResults(VkQueryPool query_pool, uint32_t first_query, uint32_t query_count,
                                 VkBuffer dst_buffer, VkDeviceSize dst_offset, VkDeviceSize stride,
                                 VkQueryResultFlags flags) {
    auto& args = *reinterpret_cast<ArgsVkCopyQueryPoolResults*>(
        WriteCommand(Command::kVkCopyQueryPoolResults, sizeof(ArgsVkCopyQueryPoolResults)));
    args.query_pool = query_pool;
    args.first_query = first_query;
    args.query_count = query_count;
    args.dst_buffer = dst_buffer;
    args.dst_offset = dst_offset;
    args.stride = stride;
    args.flags = flags;
  }

  void CmdVkDispatch(uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z) {
    BeforeEventCommand();  // wwe2k14 patch (Session 7c, CAUSES)
    ++dispatches_recorded_;
    ++record_stats_.c[RecordStats::kDispatches];  // wwe2k14 patch (Session 20, RECSTATS)
    auto& args = *reinterpret_cast<ArgsVkDispatch*>(
        WriteCommand(Command::kVkDispatch, sizeof(ArgsVkDispatch)));
    args.group_count_x = group_count_x;
    args.group_count_y = group_count_y;
    args.group_count_z = group_count_z;
  }

  void CmdVkDraw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
                 uint32_t first_instance) {
    ++record_stats_.c[RecordStats::kDraws];  // wwe2k14 patch (Session 20, RECSTATS)
    auto& args = *reinterpret_cast<ArgsVkDraw*>(WriteCommand(Command::kVkDraw, sizeof(ArgsVkDraw)));
    NoteLastDraw(&args, false);  // wwe2k14 patch (Session 23, MERGE)
    args.vertex_count = vertex_count;
    args.instance_count = instance_count;
    args.first_vertex = first_vertex;
    args.first_instance = first_instance;
  }

  void CmdVkDrawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index,
                        int32_t vertex_offset, uint32_t first_instance) {
    ++record_stats_.c[RecordStats::kDraws];  // wwe2k14 patch (Session 20, RECSTATS)
    auto& args = *reinterpret_cast<ArgsVkDrawIndexed*>(
        WriteCommand(Command::kVkDrawIndexed, sizeof(ArgsVkDrawIndexed)));
    NoteLastDraw(&args, true);  // wwe2k14 patch (Session 23, MERGE)
    args.index_count = index_count;
    args.instance_count = instance_count;
    args.first_index = first_index;
    args.vertex_offset = vertex_offset;
    args.first_instance = first_instance;
  }

  void CmdVkEndQuery(VkQueryPool query_pool, uint32_t query) {
    auto& args = *reinterpret_cast<ArgsVkEndQuery*>(
        WriteCommand(Command::kVkEndQuery, sizeof(ArgsVkEndQuery)));
    args.query_pool = query_pool;
    args.query = query;
  }

  void CmdVkEndRenderPass() {
    BeforeEventCommand();  // wwe2k14 patch (Session 7c, CAUSES)
    WriteCommand(Command::kVkEndRenderPass, 0);
  }

  // wwe2k14 patch (Session 56, gpu_native_truth_lazy_pass_deps): a "lazy" render pass - its render pass object has no
  // external subpass dependencies (each one is a wait-for-idle on the GPU, even between passes on unrelated images),
  // and what they would synchronise is recorded as barriers, only where it is needed:
  // - after its end, the attachment writes of the lazy passes since the last such barrier are made available to what
  //   follows by one barrier (the stages and accesses of the external dependency out of the pass), held until the
  //   next command that does GPU work outside a lazy pass and recorded right before it - so it joins that command's
  //   own barrier when vulkan_merge_adjacent_barriers holds barriers back;
  // - at its begin, nothing when only state commands (binds, dynamic state, timestamps, checkpoints) were recorded
  //   since a lazy pass ended and none of `attachments` was an attachment of a lazy pass since that barrier;
  //   otherwise that barrier and the external dependency into the pass, as barriers (joined with
  //   vulkan_merge_adjacent_barriers).
  // `mode` (Session 56, gpu_native_truth_lazy_pass_mode, for finding what the synchronisation costs): bit 0 the
  // barrier after the pass, bit 1 the barrier before it after other work; the barrier before it on an attachment of
  // a lazy pass since the last barrier is always recorded. `style` (gpu_native_truth_lazy_pass_style): 0 the barriers
  // are memory barriers (all images), 1 image barriers of the attachments involved (their layouts kept), 2 memory
  // barriers without depth / stencil accesses (a diagnostic: not enough for depth).
  struct LazyAttachment {
    VkImage image;
    VkImageAspectFlags aspect;  // colour, or depth and stencil
    VkImageLayout layout;       // its attachment layout (kept)
  };
  void CmdVkBeginRenderPassLazy(const VkRenderPassBeginInfo* render_pass_begin, VkSubpassContents contents,
                                const LazyAttachment* attachments, uint32_t attachment_count, uint32_t mode = 3,
                                uint32_t style = 0);
  void CmdVkEndRenderPassLazy();
  // Running counts (never reset; readers take differences): lazy passes begun without a barrier, and with one.
  // (Session 56: the last begin's reason - 0 no barrier, 1 other work since a lazy pass ended, 2 an attachment of a
  // lazy pass since the last barrier.)
  uint32_t lazy_last_reason() const { return lazy_last_reason_; }
  // (Session 59, diagnostic: the command that last broke a lazy chain - Command as a number.)
  uint32_t lazy_break_command() const { return lazy_break_command_; }
  // wwe2k14 patch (Session 79, diagnostic): with capture_break_stack set (the native renderer's segment log), the call
  // stack (5 return addresses, and the module base) of the command that last broke a lazy chain.
  bool capture_break_stack = false;
  const uintptr_t* lazy_break_stack() const { return lazy_break_stack_; }
  uintptr_t lazy_break_base() const { return lazy_break_base_; }
  // (Session 59, diagnostic: set by the command processor from vulkan_debug_skip_constants_rebind.)
  bool debug_skip_constants_rebind = false;
  VkDescriptorSet debug_last_constants_set_ = VK_NULL_HANDLE;
  VkPipelineLayout debug_last_layout_ = VK_NULL_HANDLE;
  uint64_t lazy_begins_free() const { return lazy_begins_free_; }
  uint64_t lazy_begins_synced() const { return lazy_begins_synced_; }

  // Dynamic rendering (VK_KHR_dynamic_rendering / Vulkan 1.3).
  void CmdVkBeginRendering(const VkRenderingInfo* rendering_info);
  void CmdVkEndRendering() {
    BeforeEventCommand();  // wwe2k14 patch (Session 7c, CAUSES)
    WriteCommand(Command::kVkEndRendering, 0);
  }

  void CmdVkResetQueryPool(VkQueryPool query_pool, uint32_t first_query, uint32_t query_count) {
    auto& args = *reinterpret_cast<ArgsVkResetQueryPool*>(
        WriteCommand(Command::kVkResetQueryPool, sizeof(ArgsVkResetQueryPool)));
    args.query_pool = query_pool;
    args.first_query = first_query;
    args.query_count = query_count;
  }

  // pNext of all barriers must be null.
  void CmdVkPipelineBarrier(VkPipelineStageFlags src_stage_mask,
                            VkPipelineStageFlags dst_stage_mask, VkDependencyFlags dependency_flags,
                            uint32_t memory_barrier_count, const VkMemoryBarrier* memory_barriers,
                            uint32_t buffer_memory_barrier_count,
                            const VkBufferMemoryBarrier* buffer_memory_barriers,
                            uint32_t image_memory_barrier_count,
                            const VkImageMemoryBarrier* image_memory_barriers);

  void CmdVkPushConstants(VkPipelineLayout layout, VkShaderStageFlags stage_flags, uint32_t offset,
                          uint32_t size, const void* values) {
    ++record_stats_.c[RecordStats::kPushConstants];  // wwe2k14 patch (Session 20, RECSTATS)
    record_stats_.c[RecordStats::kPushConstantBytes] += size;
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kVkPushConstants, sizeof(ArgsVkPushConstants) + size));
    auto& args = *reinterpret_cast<ArgsVkPushConstants*>(args_ptr);
    args.layout = layout;
    args.stage_flags = stage_flags;
    args.offset = offset;
    args.size = size;
    std::memcpy(args_ptr + sizeof(ArgsVkPushConstants), values, size);
  }

  void CmdVkSetBlendConstants(const float* blend_constants) {
    auto& args = *reinterpret_cast<ArgsVkSetBlendConstants*>(
        WriteCommand(Command::kVkSetBlendConstants, sizeof(ArgsVkSetBlendConstants)));
    std::memcpy(args.blend_constants, blend_constants, sizeof(float) * 4);
  }

  void CmdVkSetDepthBias(float depth_bias_constant_factor, float depth_bias_clamp,
                         float depth_bias_slope_factor) {
    auto& args = *reinterpret_cast<ArgsVkSetDepthBias*>(
        WriteCommand(Command::kVkSetDepthBias, sizeof(ArgsVkSetDepthBias)));
    args.depth_bias_constant_factor = depth_bias_constant_factor;
    args.depth_bias_clamp = depth_bias_clamp;
    args.depth_bias_slope_factor = depth_bias_slope_factor;
  }

  // wwe2k14 patch (Session 93, idea 7): the extended dynamic state of the guest pipelines, in one command (VK_EXT_extended_dynamic_state;
  // only the parts in `mask` are set: kEds* bits).
  struct ArgsVkSetEdsState {
    uint32_t mask;
    uint32_t cull_mode;
    uint32_t front_face;
    uint32_t depth_test_enable;
    uint32_t depth_write_enable;
    uint32_t depth_compare_op;
    uint32_t stencil_test_enable;
    uint32_t stencil_ops[2][4];  // per face (front, back): fail, pass, depth fail, compare
  };

  // wwe2k14 patch (Session 93, idea 7): see ArgsVkSetEdsState (kEdsCull = 1: cull mode + front face, kEdsDepth = 2: depth test / write / compare op, kEdsStencil = 4: stencil test enable + ops).
  enum : uint32_t { kEdsCull = 1, kEdsDepth = 2, kEdsStencil = 4 };
  ArgsVkSetEdsState* CmdVkSetEdsState(uint32_t mask) {
    auto& args = *reinterpret_cast<ArgsVkSetEdsState*>(WriteCommand(Command::kVkSetEdsState, sizeof(ArgsVkSetEdsState)));
    args.mask = mask;
    return &args;
  }

  void CmdVkSetScissor(uint32_t first_scissor, uint32_t scissor_count, const VkRect2D* scissors) {
    ++record_stats_.c[RecordStats::kViewportScissor];  // wwe2k14 patch (Session 20, RECSTATS)
    const size_t header_size = rex::align(sizeof(ArgsVkSetScissor), alignof(VkRect2D));
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kVkSetScissor, header_size + sizeof(VkRect2D) * scissor_count));
    auto& args = *reinterpret_cast<ArgsVkSetScissor*>(args_ptr);
    args.first_scissor = first_scissor;
    args.scissor_count = scissor_count;
    std::memcpy(args_ptr + header_size, scissors, sizeof(VkRect2D) * scissor_count);
  }

  void CmdVkSetStencilCompareMask(VkStencilFaceFlags face_mask, uint32_t compare_mask) {
    auto& args = *reinterpret_cast<ArgsSetStencilMaskReference*>(
        WriteCommand(Command::kVkSetStencilCompareMask, sizeof(ArgsSetStencilMaskReference)));
    args.face_mask = face_mask;
    args.mask_reference = compare_mask;
  }

  void CmdVkSetStencilReference(VkStencilFaceFlags face_mask, uint32_t reference) {
    auto& args = *reinterpret_cast<ArgsSetStencilMaskReference*>(
        WriteCommand(Command::kVkSetStencilReference, sizeof(ArgsSetStencilMaskReference)));
    args.face_mask = face_mask;
    args.mask_reference = reference;
  }

  void CmdVkSetStencilWriteMask(VkStencilFaceFlags face_mask, uint32_t write_mask) {
    auto& args = *reinterpret_cast<ArgsSetStencilMaskReference*>(
        WriteCommand(Command::kVkSetStencilWriteMask, sizeof(ArgsSetStencilMaskReference)));
    args.face_mask = face_mask;
    args.mask_reference = write_mask;
  }

  // wwe2k14 patch (vulkan_deferred_descriptor_writes): vkUpdateDescriptorSets as a deferred command, so the
  // submit thread runs it (driver time) right before the commands recorded after it. The image / buffer infos
  // are copied. Only image, buffer and texel buffer descriptors, no pNext; returns false if unsupported (then
  // the caller updates immediately).
  static constexpr uint32_t kMaxDeferredDescriptorWrites = 16;
  bool CmdVkUpdateDescriptorSets(uint32_t write_count, const VkWriteDescriptorSet* writes);

  // VK_NV_device_diagnostic_checkpoints; the marker is an opaque number.
  void CmdVkSetCheckpointNV(uint64_t marker) {
    auto& args = *reinterpret_cast<ArgsVkSetCheckpointNV*>(
        WriteCommand(Command::kVkSetCheckpointNV, sizeof(ArgsVkSetCheckpointNV)));
    args.marker = marker;
  }

  void CmdVkSetViewport(uint32_t first_viewport, uint32_t viewport_count,
                        const VkViewport* viewports) {
    ++record_stats_.c[RecordStats::kViewportScissor];  // wwe2k14 patch (Session 20, RECSTATS)
    const size_t header_size = rex::align(sizeof(ArgsVkSetViewport), alignof(VkViewport));
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kVkSetViewport, header_size + sizeof(VkViewport) * viewport_count));
    auto& args = *reinterpret_cast<ArgsVkSetViewport*>(args_ptr);
    args.first_viewport = first_viewport;
    args.viewport_count = viewport_count;
    std::memcpy(args_ptr + header_size, viewports, sizeof(VkViewport) * viewport_count);
  }

  // wwe2k14 patch (Session 7b, GPUMARKS): timestamp marks between the recorded commands (gpu_anatomy_gpu_marks).
  void CmdVkWriteTimestamp(VkPipelineStageFlagBits pipeline_stage, VkQueryPool query_pool, uint32_t query) {
    auto& args = *reinterpret_cast<ArgsVkWriteTimestamp*>(
        WriteCommand(Command::kVkWriteTimestamp, sizeof(ArgsVkWriteTimestamp)));
    args.pipeline_stage = pipeline_stage;
    args.query_pool = query_pool;
    args.query = query;
  }

 private:
  enum class Command {
    kVkBeginRenderPass,
    kVkBeginQuery,
    kVkBindDescriptorSets,
    kVkBindIndexBuffer,
    kVkBindPipeline,
    kVkBindVertexBuffers,
    kVkClearAttachments,
    kVkClearColorImage,
    kVkCopyBuffer,
    kVkCopyBufferToImage,
    kVkCopyImageToBuffer,  // wwe2k14 patch (Session 7c, TEXDIRECT)
    kVkCopyQueryPoolResults,
    kVkDispatch,
    kVkDraw,
    kVkDrawIndexed,
    kVkEndQuery,
    kVkEndRenderPass,
    kVkBeginRendering,
    kVkEndRendering,
    kVkPipelineBarrier,
    kVkPushConstants,
    kVkResetQueryPool,
    kVkSetBlendConstants,
    kVkSetCheckpointNV,
    kVkSetDepthBias,
    kVkSetEdsState,  // wwe2k14 patch (Session 93, idea 7)
    kVkSetScissor,
    kVkSetStencilCompareMask,
    kVkSetStencilReference,
    kVkSetStencilWriteMask,
    kVkSetViewport,
    kVkUpdateDescriptorSets,
    kVkWriteTimestamp,  // wwe2k14 patch (Session 7b, GPUMARKS)
    kVkCopyImage,       // wwe2k14 patch (Session 18, DLAA)
    kHostCallback,      // wwe2k14 patch (Session 18, DLAA)
    kVkBlitImage,       // wwe2k14 patch (Session 54, NATIVE draw sizes)
    kVkClearDepthStencilImage,  // wwe2k14 patch (Session 54, NATIVE without interlock)
    kNop,  // wwe2k14 patch (Session 79, NF-D): a command turned off after recording (NopRange); skipped on replay
    kDiagMark,  // wwe2k14 patch (Session 95, the diagnostic run)
  };

  struct CommandHeader {
    Command command;
    uint32_t arguments_size_elements;
  };
  static constexpr size_t kCommandHeaderSizeElements =
      (sizeof(CommandHeader) + sizeof(uintmax_t) - 1) / sizeof(uintmax_t);

  struct ArgsDiagMark {  // wwe2k14 patch (Session 95, the diagnostic run)
    uint32_t kind, id, aux;
  };

  // wwe2k14 patch (Session 95, the diagnostic run): ExecuteStream's body; kDiag = the observing replay of an armed frame.
  template <bool kDiag>
  void ExecuteStreamT(const std::vector<uintmax_t>& command_stream, VkCommandBuffer command_buffer) const;

  struct ArgsVkBeginRenderPass {
    VkRenderPass render_pass;
    VkFramebuffer framebuffer;
    VkRect2D render_area;
    uint32_t clear_value_count;
    VkSubpassContents contents;
    // Followed by aligned optional VkClearValue[].
    static_assert(alignof(VkClearValue) <= alignof(uintmax_t));
  };

  struct ArgsVkBeginRendering {
    VkRenderingFlags flags;
    VkRect2D render_area;
    uint32_t layer_count;
    uint32_t view_mask;
    uint32_t color_attachment_count;
    bool has_depth_attachment;
    bool has_stencil_attachment;
    // Followed by:
    // - VkRenderingAttachmentInfo[color_attachment_count]
    // - VkRenderingAttachmentInfo depth (if has_depth_attachment)
    // - VkRenderingAttachmentInfo stencil (if has_stencil_attachment)
    static_assert(alignof(VkRenderingAttachmentInfo) <= alignof(uintmax_t));
  };

  struct ArgsVkBeginQuery {
    VkQueryPool query_pool;
    uint32_t query;
    VkQueryControlFlags flags;
  };

  struct ArgsVkBindDescriptorSets {
    VkPipelineBindPoint pipeline_bind_point;
    VkPipelineLayout layout;
    uint32_t first_set;
    uint32_t descriptor_set_count;
    uint32_t dynamic_offset_count;
    // Followed by aligned VkDescriptorSet[], optional uint32_t[].
    static_assert(alignof(VkDescriptorSet) <= alignof(uintmax_t));
  };

  struct ArgsVkBindIndexBuffer {
    VkBuffer buffer;
    VkDeviceSize offset;
    VkIndexType index_type;
  };

  struct ArgsVkBindPipeline {
    VkPipelineBindPoint pipeline_bind_point;
    VkPipeline pipeline;
  };

  struct ArgsVkBindVertexBuffers {
    uint32_t first_binding;
    uint32_t binding_count;
    // Followed by aligned VkBuffer[], VkDeviceSize[].
    static_assert(alignof(VkBuffer) <= alignof(uintmax_t));
    static_assert(alignof(VkDeviceSize) <= alignof(uintmax_t));
  };

  struct ArgsVkClearAttachments {
    uint32_t attachment_count;
    uint32_t rect_count;
    // Followed by aligned VkClearAttachment[], VkClearRect[].
    static_assert(alignof(VkClearAttachment) <= alignof(uintmax_t));
    static_assert(alignof(VkClearRect) <= alignof(uintmax_t));
  };

  struct ArgsVkClearColorImage {
    VkImage image;
    VkImageLayout image_layout;
    VkClearColorValue color;
    uint32_t range_count;
    // Followed by aligned VkImageSubresourceRange[].
    static_assert(alignof(VkImageSubresourceRange) <= alignof(uintmax_t));
  };

  struct ArgsVkCopyBuffer {
    VkBuffer src_buffer;
    VkBuffer dst_buffer;
    uint32_t region_count;
    // Followed by aligned VkBufferCopy[].
    static_assert(alignof(VkBufferCopy) <= alignof(uintmax_t));
  };

  struct ArgsVkCopyBufferToImage {
    VkBuffer src_buffer;
    VkImage dst_image;
    VkImageLayout dst_image_layout;
    uint32_t region_count;
    // Followed by aligned VkBufferImageCopy[].
    static_assert(alignof(VkBufferImageCopy) <= alignof(uintmax_t));
  };

  // wwe2k14 patch (Session 7c, TEXDIRECT).
  struct ArgsHostCallback {  // wwe2k14 patch (Session 18, DLAA)
    HostCallback fn;
    size_t data_size;
    // Followed by the data (aligned to uintmax_t).
  };

  struct ArgsVkClearDepthStencilImage {  // wwe2k14 patch (Session 54, NATIVE without interlock)
    VkImage image;
    VkImageLayout image_layout;
    VkClearDepthStencilValue value;
    VkImageSubresourceRange range;
  };

  struct ArgsVkBlitImage {  // wwe2k14 patch (Session 54, NATIVE draw sizes)
    VkImage src_image;
    VkImageLayout src_image_layout;
    VkImage dst_image;
    VkImageLayout dst_image_layout;
    uint32_t region_count;
    VkFilter filter;
    // Followed by aligned VkImageBlit[].
    static_assert(alignof(VkImageBlit) <= alignof(uintmax_t));
  };

  struct ArgsVkCopyImage {  // wwe2k14 patch (Session 18, DLAA)
    VkImage src_image;
    VkImageLayout src_image_layout;
    VkImage dst_image;
    VkImageLayout dst_image_layout;
    uint32_t region_count;
    // Followed by aligned VkImageCopy[].
    static_assert(alignof(VkImageCopy) <= alignof(uintmax_t));
  };

  struct ArgsVkCopyImageToBuffer {
    VkImage src_image;
    VkImageLayout src_image_layout;
    VkBuffer dst_buffer;
    uint32_t region_count;
    // Followed by aligned VkBufferImageCopy[].
    static_assert(alignof(VkBufferImageCopy) <= alignof(uintmax_t));
  };

  struct ArgsVkCopyQueryPoolResults {
    VkQueryPool query_pool;
    uint32_t first_query;
    uint32_t query_count;
    VkBuffer dst_buffer;
    VkDeviceSize dst_offset;
    VkDeviceSize stride;
    VkQueryResultFlags flags;
  };

  struct ArgsVkDispatch {
    uint32_t group_count_x;
    uint32_t group_count_y;
    uint32_t group_count_z;
  };

  struct ArgsVkDraw {
    uint32_t vertex_count;
    uint32_t instance_count;
    uint32_t first_vertex;
    uint32_t first_instance;
  };

  struct ArgsVkDrawIndexed {
    uint32_t index_count;
    uint32_t instance_count;
    uint32_t first_index;
    int32_t vertex_offset;
    uint32_t first_instance;
  };

  struct ArgsVkEndQuery {
    VkQueryPool query_pool;
    uint32_t query;
  };

  struct ArgsVkResetQueryPool {
    VkQueryPool query_pool;
    uint32_t first_query;
    uint32_t query_count;
  };

  struct ArgsVkPipelineBarrier {
    VkPipelineStageFlags src_stage_mask;
    VkPipelineStageFlags dst_stage_mask;
    VkDependencyFlags dependency_flags;
    uint32_t memory_barrier_count;
    uint32_t buffer_memory_barrier_count;
    uint32_t image_memory_barrier_count;
    // Followed by aligned optional VkMemoryBarrier[],
    // optional VkBufferMemoryBarrier[], optional VkImageMemoryBarrier[].
    static_assert(alignof(VkMemoryBarrier) <= alignof(uintmax_t));
    static_assert(alignof(VkBufferMemoryBarrier) <= alignof(uintmax_t));
    static_assert(alignof(VkImageMemoryBarrier) <= alignof(uintmax_t));
  };

  // wwe2k14 patch (Session 52, vulkan_merge_adjacent_barriers): a recorded barrier's arguments and arrays.
  struct BarrierView {
    const ArgsVkPipelineBarrier* args;
    const VkMemoryBarrier* memory;
    const VkBufferMemoryBarrier* buffer;
    const VkImageMemoryBarrier* image;
  };
  static BarrierView ReadBarrier(const uintmax_t* args);

  struct ArgsVkPushConstants {
    VkPipelineLayout layout;
    VkShaderStageFlags stage_flags;
    uint32_t offset;
    uint32_t size;
    // Followed by `size` bytes of values.
  };

  struct ArgsVkSetBlendConstants {
    float blend_constants[4];
  };

  struct ArgsVkSetDepthBias {
    float depth_bias_constant_factor;
    float depth_bias_clamp;
    float depth_bias_slope_factor;
  };

  struct ArgsVkSetScissor {
    uint32_t first_scissor;
    uint32_t scissor_count;
    // Followed by aligned VkRect2D[].
    static_assert(alignof(VkRect2D) <= alignof(uintmax_t));
  };

  struct ArgsSetStencilMaskReference {
    VkStencilFaceFlags face_mask;
    uint32_t mask_reference;
  };

  struct ArgsVkSetCheckpointNV {
    uint64_t marker;
  };

  // Followed by write_count ArgsDescriptorWrite, each followed by its descriptor_count infos (8-byte aligned).
  struct ArgsVkUpdateDescriptorSets {
    uint32_t write_count;
  };
  enum class DescriptorPayload : uint32_t { kImage, kBuffer, kTexelBufferView };
  struct ArgsDescriptorWrite {
    VkDescriptorSet dst_set;
    uint32_t dst_binding;
    uint32_t dst_array_element;
    uint32_t descriptor_count;
    VkDescriptorType descriptor_type;
    DescriptorPayload payload;
  };
  static bool GetDescriptorPayload(VkDescriptorType type, DescriptorPayload& payload_out,
                                   size_t& element_size_out);

  struct ArgsVkSetViewport {
    uint32_t first_viewport;
    uint32_t viewport_count;
    // Followed by aligned VkViewport[].
    static_assert(alignof(VkViewport) <= alignof(uintmax_t));
  };

  // wwe2k14 patch (Session 7b, GPUMARKS).
  struct ArgsVkWriteTimestamp {
    VkPipelineStageFlagBits pipeline_stage;
    VkQueryPool query_pool;
    uint32_t query;
  };

  void* WriteCommand(Command command, size_t arguments_size_bytes);

  // wwe2k14 patch (Session 52, vulkan_merge_adjacent_barriers): a barrier is held back until the next command that
  // does GPU work or orders it (anything but binds, push constants, dynamic state and checkpoints) and later
  // barriers join it - one vkCmdPipelineBarrier, the union of their synchronisation, written right before that
  // command (binds and state recorded in between come first: barriers do not order them).
  static bool IsStateCommand(Command command);
  void WriteBarrierCommand(VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask,
                           VkDependencyFlags dependency_flags, uint32_t memory_barrier_count,
                           const VkMemoryBarrier* memory_barriers, uint32_t buffer_memory_barrier_count,
                           const VkBufferMemoryBarrier* buffer_memory_barriers,
                           uint32_t image_memory_barrier_count, const VkImageMemoryBarrier* image_memory_barriers);
  void FlushPendingBarrier() {
    if (barrier_pending_) FlushPendingBarrierSlow();
  }
  void FlushPendingBarrierSlow();
  // wwe2k14 patch (Session 56): the held barrier after lazy render passes (see CmdVkBeginRenderPassLazy).
  void FlushLazySync();
  static bool KeepsLazyChain(Command command) {
    return IsStateCommand(command) || command == Command::kVkWriteTimestamp || command == Command::kVkSetCheckpointNV ||
           command == Command::kDiagMark;  // Session 95: an armed frame's marks do nothing on the GPU, so they must not change what the pass-dropping code does
  }
  // Session 95: a mark that belongs to one draw (diag_layer.h MarkKind kMarkNextDraw = 3, kMarkPrevDraw = 4).
  static bool IsDrawMark(const CommandHeader& header) {
    if (header.command != Command::kDiagMark) return false;
    const auto* args = reinterpret_cast<const ArgsDiagMark*>(reinterpret_cast<const uintmax_t*>(&header) + kCommandHeaderSizeElements);
    return args->kind == 3 || args->kind == 4;
  }
  size_t last_begin_offset_ = SIZE_MAX;  // wwe2k14 patch (Session 73): PatchBeginRenderPass
  bool lazy_sync_pending_ = false;  // lazy passes' attachment writes not yet made available by a barrier
  bool lazy_chain_ = false;         // a lazy pass ended and only KeepsLazyChain commands were recorded since
  bool in_lazy_pass_ = false;
  bool lazy_end_sync_ = true;  // the open lazy pass's mode bit 0
  uint32_t lazy_style_ = 0;     // the last lazy pass's style
  std::vector<LazyAttachment> lazy_attachments_;  // the lazy passes' attachments since the last barrier after them
  std::vector<VkImageMemoryBarrier> lazy_barriers_;  // scratch
  // Appends image barriers (no layout change) for `attachments` not in lazy_barriers_ yet.
  void AddLazyImageBarriers(const LazyAttachment* attachments, size_t count);
  uint64_t lazy_begins_free_ = 0, lazy_begins_synced_ = 0;
  uint32_t lazy_last_reason_ = 0;
  uint32_t lazy_break_command_ = UINT32_MAX;  // wwe2k14 patch (Session 59, diagnostic)
  uintptr_t lazy_break_stack_[5] = {};        // wwe2k14 patch (Session 79, diagnostic)
  uintptr_t lazy_break_base_ = 0;
  void CaptureBreakStack();
  bool barrier_pending_ = false;
  VkPipelineStageFlags pending_src_stage_mask_ = 0, pending_dst_stage_mask_ = 0;
  std::vector<VkMemoryBarrier> pending_memory_barriers_;
  std::vector<VkBufferMemoryBarrier> pending_buffer_barriers_;
  std::vector<VkImageMemoryBarrier> pending_image_barriers_;

  // wwe2k14 patch (Session 23, MERGE): where the last draw's arguments are (in elements) and the stream size right
  // after it (SIZE_MAX: none since the last reset / swap).
  void NoteLastDraw(const void* args, bool indexed) {
    last_draw_args_ = size_t(reinterpret_cast<const uintmax_t*>(args) - command_stream_.data());
    last_draw_end_ = command_stream_.size();
    last_draw_indexed_ = indexed;
    merged_scan_end_ = SIZE_MAX;  // (Session 93)
  }
  size_t merged_scan_end_ = SIZE_MAX;  // wwe2k14 patch (Session 93): where MergeInstancesIntoLastIndexedDraw stopped scanning after the last draw
  size_t last_draw_args_ = 0;
  size_t last_draw_end_ = SIZE_MAX;
  bool last_draw_indexed_ = false;

  // wwe2k14 patch (Session 7c, CAUSES): see SetEventHook and dispatches_recorded.
  void BeforeEventCommand() {
    if (event_hook_) {
      CallEventHook();
    }
  }
  void CallEventHook();
  VulkanCommandProcessor* event_hook_ = nullptr;
  uint32_t dispatches_recorded_ = 0;
  uint32_t rendering_begins_recorded_ = 0;
  RecordStats record_stats_;  // wwe2k14 patch (Session 20, RECSTATS)

  const VulkanCommandProcessor& command_processor_;

  // uintmax_t to ensure uint64_t and pointer alignment of all structures.
  std::vector<uintmax_t> command_stream_;
};

}  // namespace rex::graphics::vulkan
