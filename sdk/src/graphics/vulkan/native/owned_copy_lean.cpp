// wwe2k14 patch (Session 95, gpu_owned_copy_lean): fewer full-size copies when a pass goes on after the game's copies of it.
//
// Bryan's entrance on the Surface (run 059, out\s95\bryan_copies.md): 5 snapshots + 2 continue copies at 1280x720 a frame, 3.7 ms of 20.
//   bit 1: a whole colour copy is the image itself (no learned snapshot) where a snapshot buys nothing: a page already IS the image (a direct
//          depth page, an earlier colour copy), so the next draw moves the pass in any case and the old image stays as it is for every page;
//          or the pass has no depth, so that move costs one colour copy, as the snapshot would, and only if the pass really goes on.
//   bit 2: a restore draw (colour, depth, stencil all rewritten over the whole box) over a pass whose image pages stand on takes a fresh image
//          with nothing copied; the copy before such a restore (learned, per pass and copy number) is the image itself.
//   bit 4: where the pages left behind were replaced unread (learned from the last move of this pass and copy number), the pass stays in its
//          image and the pages get a copy; replaced unread again in this stream (the proof), the copy is turned off before replay. A read keeps it.
// Fallback everywhere: OwnedContinueCopy as it was. Rule 6: our own images and pages; nothing here models the EDRAM.
// State for a job system: all of it is the owned frame's (one thread, the decoder's side); the stream patch must run before the stream is handed on.

#include <rex/graphics/vulkan/native/native_renderer.h>

#include <algorithm>

#include <rex/cvar.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/deferred_command_buffer.h>
#include <rex/logging.h>
#include <rex/ui/vulkan/util.h>

REXCVAR_DEFINE_INT32(gpu_owned_copy_lean, 0, "GPU/Native",
                     "Session 95 (default 0 = off): fewer full-size copies when an owned pass goes on after the game's copies. Bits: 1 no learned "
                     "snapshot where the pass moves anyway or has no depth; 2 a restore draw over the whole box takes a fresh image, no copy; "
                     "4 pages replaced unread in the last frames: the pass stays, the pages' copy is turned off in the stream when proven unread "
                     "again. 7 = all. Needs gpu_native_owned_frame_continue_copy; bits 2 and 4 need the owned-only frame");
REXCVAR_DECLARE(bool, gpu_native_owned_frame_continue_copy);

namespace rex::graphics::vulkan::native {

namespace {
constexpr VkPipelineStageFlags kShaderStages =
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
constexpr VkPipelineStageFlags kAttachmentStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
constexpr VkPipelineStageFlags kAllStages = kShaderStages | kAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT;
// (As OwnedBarrier's callers: a transition from UNDEFINED is a write, the image's earlier writes in its first scope.)
constexpr VkAccessFlags kAnyWrite = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
constexpr VkImageAspectFlags kDS = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
constexpr VkAccessFlags kColourAttachment = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
constexpr VkAccessFlags kDepthAttachment = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

VkImageMemoryBarrier LeanBarrier(VkImage image, VkImageAspectFlags aspects, VkAccessFlags src, VkAccessFlags dst, VkImageLayout from,
                                 VkImageLayout to) {
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
uint32_t CopyBit(uint32_t copy_count) { return 1u << std::min(copy_count, 31u); }
}  // namespace

uint32_t NativeRenderer::OwnedLeanBits() const {
  return REXCVAR_GET(gpu_native_owned_frame_continue_copy) ? uint32_t(REXCVAR_GET(gpu_owned_copy_lean)) & 7u : 0u;
}

size_t NativeRenderer::OwnedLeanEntry(const OwnedPass& p) {
  for (size_t i = 0; i < owned_lean_learn_.size(); ++i) {
    const OwnedLeanLearn& e = owned_lean_learn_[i];
    if (e.base == p.base && e.pitch_tiles == p.pitch_tiles && e.kind == p.kind && e.width == p.width && e.height == p.height) return i;
  }
  OwnedLeanLearn e;
  e.base = p.base;
  e.pitch_tiles = p.pitch_tiles;
  e.kind = p.kind;
  e.width = p.width;
  e.height = p.height;
  owned_lean_learn_.push_back(e);
  return owned_lean_learn_.size() - 1;
}

bool NativeRenderer::OwnedLeanAlias(const OwnedPass& p) {
  // (Asked only when the learned rule wants a snapshot of this whole colour copy, with no half-pixel gap to fill.)
  const uint32_t bits = OwnedLeanBits();
  if (!bits || p.half || p.dres || OwnedLayers() != 1) return false;
  OwnedLeanStats& st = owned_lean_stats_;
  if (bits & 1) {
    if (!p.depth) {
      ++st.alias_no_depth;
      return true;
    }
    if (std::any_of(owned_pages_.begin(), owned_pages_.end(), [&](const OwnedPage& q) { return q.image == p.image; })) {
      ++st.alias_page_on_image;
      return true;
    }
  }
  if ((bits & 2) && OwnedOnly()) {
    const OwnedLeanLearn& e = owned_lean_learn_[OwnedLeanEntry(p)];
    const uint32_t bit = CopyBit(p.copy_count);
    if ((e.restore_next & bit) && !(e.plain_next & bit)) {
      ++st.alias_restore_next;
      return true;
    }
  }
  return false;
}

void NativeRenderer::OwnedLeanNoteDraw(const OwnedPass& p, const Draw& draw, const int32_t rect[4]) {
  // (A committed pass's draw, counted already: the first one since the last whole colour copy says what follows that copy.)
  const uint32_t bits = OwnedLeanBits();
  if (!bits) return;
  OwnedLeanStats& st = owned_lean_stats_;
  const uint32_t k = std::min(p.copy_count, 7u);
  ++st.draws_after[k];
  if (draw.truth_depth >= 0 && (draw.raster.depth_write || draw.raster.stencil)) ++st.depth_writes_after[k];
  if (!(bits & 2) || p.draws_since_color != 1) return;
  OwnedLeanLearn& e = owned_lean_learn_[OwnedLeanEntry(p)];
  // (Only a restore that can take a fresh image counts: before any other the snapshot stays the cheap way. The held clears are still there: dropped after this draw.)
  (OwnedLeanCovers(p, draw, rect) ? e.restore_next : e.plain_next) |= CopyBit(p.copy_count);
}

bool NativeRenderer::OwnedLeanCovers(const OwnedPass& p, const Draw& draw, const int32_t rect[4]) {
  // A restore that leaves nothing of the pass: colour whole and unblended, depth written with the test always, the box whole. A stencil the pass keeps
  // must be the held clears' whole stencil, which the restore's caller clears in the pass right after the draw (restore_stencil_cleared).
  if (!draw.restore || !p.depth || p.half || p.dres || !OwnedOnly() || OwnedLayers() != 1) return false;
  if (rect[0] != 0 || rect[1] != 0 || rect[2] < int32_t(p.host_width) || rect[3] < int32_t(p.host_height)) return false;
  if ((draw.color_write_mask & 0xF) != 0xF || (draw.color_mask_all & ~0xFu) || draw.blend_control != 0x00010001u) return false;
  if (draw.truth_depth < 0 || !draw.raster.depth_write || draw.raster.stencil) return false;
  if (draw.raster.depth_test && draw.raster.depth_compare != VK_COMPARE_OP_ALWAYS) return false;
  // (The proof the quad really reaches every pixel: its proven coverage, as `covered` in OwnedTakeDraw. A restore without one is not fresh: the move, as before.)
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  if (!(draw.cover[2] > draw.cover[0] && draw.cover[3] > draw.cover[1] && draw.cover[0] <= 0 && draw.cover[1] <= 0 &&
        int64_t(draw.cover[2]) * scale_x >= int64_t(p.host_width) && int64_t(draw.cover[3]) * scale_y >= int64_t(p.host_height))) {
    ++owned_lean_stats_.no_cover;
    return false;
  }
  if (!p.stencil) return true;
  const ClearsInfo& hc = owned_held_.info;
  return !owned_held_.draws.empty() && owned_held_.color == draw.truth_color && owned_held_.depth == p.depth_slot && hc.valid && hc.stencil &&
         hc.stencil_touched && hc.rect[0] == 0 && hc.rect[1] == 0 && hc.rect[2] == int32_t(p.host_width) && hc.rect[3] == int32_t(p.host_height);
}

bool NativeRenderer::OwnedLeanFresh(OwnedPass& p) {
  // As OwnedContinueCopy, nothing copied: the next draw rewrites every pixel of colour and depth; the old image stays for its pages.
  if (p.image < 0 || size_t(p.image) >= owned_images_.size()) return false;
  const int16_t old_index = p.image;
  const uint32_t w = owned_images_[old_index].width, h = owned_images_[old_index].height, kind = owned_images_[old_index].kind;
  if (!OwnedImageAvailable(w, h, kind, true)) return false;
  OwnedEndPass();
  EndTruthPass();
  command_processor_.EndRenderPass();
  command_processor_.SubmitBarriers(true);
  const int16_t new_index = OwnedImageFor(w, h, kind);
  if (new_index < 0 || new_index == old_index || !OwnedEnsureDepth(new_index)) return false;
  OwnedDepthAccount(new_index);
  OwnedImage& d = owned_images_[new_index];
  VkImageMemoryBarrier barriers[2] = {
      LeanBarrier(d.image, VK_IMAGE_ASPECT_COLOR_BIT, kAnyWrite, kColourAttachment, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL),
      LeanBarrier(d.depth, kDS, kAnyWrite, kDepthAttachment, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)};
  command_processor_.deferred_command_buffer().CmdVkPipelineBarrier(kAllStages, kAttachmentStages, 0, 0, nullptr, 0, nullptr, 2, barriers);
  d.readable = false;
  d.depth_read = false;
  p.image = new_index;
  OwnedTouch(old_index);
  OwnedTouch(new_index);
  ++owned_frame_stats_.detached;
  ++owned_lean_stats_.fresh;
  command_processor_.NativeOwnedPagesChanged();
  return true;
}

bool NativeRenderer::OwnedLeanDetach(OwnedPass& p, size_t learn, uint32_t bit) {
  // The pass stays in its image; a fresh image takes what its pages hold, and the pages go there. The copy with the source's transitions (read layouts,
  // and back) is one range of the stream: turned off whole by OwnedLeanSettle, the source is as the flags say and the destination's barriers still ran.
  OwnedLeanStats& st = owned_lean_stats_;
  if (!OwnedOnly() || p.half || p.dres || OwnedLayers() != 1 || owned_wb_planned_ || command_processor_.NativeReadbackHolds()) return false;
  if (p.image < 0 || size_t(p.image) >= owned_images_.size()) return false;
  // (This draw's own texture request was given a view of the image: it goes on in a copy, as ever. S87's VUID.)
  if (std::find(owned_sampled_images_.begin(), owned_sampled_images_.end(), p.image) != owned_sampled_images_.end()) {
    ++st.detach_refused;
    return false;
  }
  bool colour = false, depth = false;
  for (const OwnedPage& q : owned_pages_) {
    if (q.image != p.image) continue;
    (q.depth_direct ? depth : colour) = true;
  }
  const int16_t old_index = p.image;
  const uint32_t w = owned_images_[old_index].width, h = owned_images_[old_index].height, kind = owned_images_[old_index].kind;
  if ((colour && !owned_images_[old_index].readable) ||
      (depth && (!owned_images_[old_index].depth_read || owned_images_[old_index].depth == VK_NULL_HANDLE)) ||
      !OwnedImageAvailable(w, h, kind, depth)) {
    ++st.detach_refused;
    return false;
  }
  OwnedEndPass();
  EndTruthPass();
  command_processor_.EndRenderPass();
  command_processor_.SubmitBarriers(true);
  const int16_t new_index = OwnedImageFor(w, h, kind);
  if (new_index < 0 || new_index == old_index || (depth && !OwnedEnsureDepth(new_index))) {
    ++st.detach_refused;
    return false;
  }
  if (depth) {
    OwnedNoteDepthRead(old_index, "lean detach");
    OwnedDepthAccount(new_index);
  }
  const OwnedImage& s = owned_images_[old_index];  // (after every call that may grow the vector)
  OwnedImage& d = owned_images_[new_index];
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const uint32_t timer = OwnedTimerBegin("lean_detach", w, h);
  VkImageMemoryBarrier barriers[2];
  uint32_t count = 0;
  if (colour) {
    barriers[count++] = LeanBarrier(d.image, VK_IMAGE_ASPECT_COLOR_BIT, kAnyWrite, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  }
  if (depth) {
    barriers[count++] = LeanBarrier(d.depth, kDS, kAnyWrite, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  }
  cb.CmdVkPipelineBarrier(kAllStages, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, count, barriers);
  cb.FlushHeldCommands();
  const size_t begin = cb.stream_offset();
  count = 0;
  if (colour) {
    barriers[count++] = LeanBarrier(s.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  }
  if (depth) {
    barriers[count++] = LeanBarrier(s.depth, kDS, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  }
  cb.CmdVkPipelineBarrier(kAllStages, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, count, barriers);
  VkImageCopy region = {};
  region.extent = {w, h, 1};
  if (colour) {
    region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    OwnedCopyImage(cb, s.image, true, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, d.image, true, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  }
  if (depth) {
    region.srcSubresource = region.dstSubresource = {kDS, 0, 0, 1};
    OwnedCopyImage(cb, s.depth, true, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, d.depth, true, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  }
  count = 0;
  if (colour) {
    barriers[count++] = LeanBarrier(s.image, VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
  if (depth) {
    barriers[count++] = LeanBarrier(s.depth, kDS, 0, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, kAllStages, 0, 0, nullptr, 0, nullptr, count, barriers);
  cb.FlushHeldCommands();
  const size_t end = cb.stream_offset();
  count = 0;
  if (colour) {
    barriers[count++] = LeanBarrier(d.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
  if (depth) {
    barriers[count++] = LeanBarrier(d.depth, kDS, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, kAllStages, 0, 0, nullptr, 0, nullptr, count, barriers);
  OwnedTimerEnd(timer);
  d.readable = colour;
  d.depth_read = depth;
  for (OwnedPage& q : owned_pages_) {
    if (q.image == old_index) q.image = new_index;
  }
  OwnedTouch(old_index);
  OwnedTouch(new_index);
  ++owned_frame_stats_.detached;
  ++st.detached;
  OwnedLeanPending pending;
  pending.image = new_index;
  pending.begin = begin;
  pending.end = end;
  pending.submission = command_processor_.GetCurrentSubmission();
  pending.learn = learn;
  pending.bit = bit;
  owned_lean_pending_.push_back(pending);
  command_processor_.NativeOwnedPagesChanged();
  OwnedWatchSync();
  return true;
}

bool NativeRenderer::OwnedLeanGoOn(OwnedPass& p, const Draw& draw, const int32_t rect[4]) {
  // The committed pass draws on while pages stand on its image (OwnedRecord). False: as OwnedContinueCopy's false.
  if (OwnedRsGoOn(p, draw, rect)) return true;  // (Session 95, gpu_owned_restore_skip)
  const uint32_t bits = OwnedLeanBits();
  const int16_t rs_old = p.image;
  if (!bits) {
    if (!OwnedContinueCopy(p)) return false;
    OwnedRsMoved(rs_old);
    return true;
  }
  OwnedLeanStats& st = owned_lean_stats_;
  if ((bits & 2) && draw.restore) {
    if (OwnedLeanCovers(p, draw, rect) && OwnedLeanFresh(p)) return true;
    ++st.fresh_missed;
  }
  const size_t learn = OwnedLeanEntry(p);
  const uint32_t bit = CopyBit(p.copy_count);
  // (A repeat draws with its base's descriptor sets: views asked for before this move, which a detach would leave on the pass's own image.)
  if ((bits & 4) && (owned_lean_learn_[learn].unread & bit)) {
    if (draw.repeat) {
      ++st.detach_refused;
    } else if (OwnedLeanDetach(p, learn, bit)) {
      return true;
    }
  }
  const int16_t old_index = p.image;
  if (!OwnedContinueCopy(p)) return false;
  ++st.moved;
  OwnedRsMoved(old_index);
  // (Watched only: were the pages it left behind replaced unread?)
  if ((bits & 4) && OwnedOnly()) {
    OwnedLeanPending pending;
    pending.image = old_index;
    pending.submission = command_processor_.GetCurrentSubmission();
    pending.learn = learn;
    pending.bit = bit;
    owned_lean_pending_.push_back(pending);
  }
  return true;
}

void NativeRenderer::OwnedLeanNoteRead(int16_t image) {
  // (OwnedView, every page view handed out. A draw left out after its request counts too: on the safe side.)
  for (OwnedLeanPending& q : owned_lean_pending_) q.read = q.read || q.image == image;
}

void NativeRenderer::OwnedLeanSettle(bool submission_end) {
  if (owned_lean_pending_.empty()) return;
  OwnedLeanStats& st = owned_lean_stats_;
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const uint64_t submission = command_processor_.GetCurrentSubmission();
  size_t kept = 0;
  for (size_t i = 0; i < owned_lean_pending_.size(); ++i) {
    const OwnedLeanPending q = owned_lean_pending_[i];
    const bool held = std::any_of(owned_pages_.begin(), owned_pages_.end(), [&](const OwnedPage& pg) { return pg.image == q.image; });
    OwnedLeanLearn& e = owned_lean_learn_[q.learn];
    if (!q.read && !held) {
      // (The proof: every page that stood on this image is gone, nothing asked for a view of it since. Its copy is dead.)
      e.unread |= q.bit;
      if (q.end > q.begin) {
        if (q.submission == submission && q.end <= cb.stream_offset() && cb.NopRange(q.begin, q.end)) {
          ++st.dropped;
          if (!owned_lean_first_drop_) {
            owned_lean_first_drop_ = true;
            REXGPU_INFO("[native][lean] frame {}: first copy turned off in the stream: the pages of image {} were replaced unread", frame_, q.image);
          }
        } else {
          ++st.nop_failed;
        }
      } else {
        ++st.watched_unread;
      }
      continue;
    }
    if (q.read || submission_end) {
      e.unread &= ~q.bit;
      if (q.end > q.begin) {
        if (q.read) {
          ++st.kept_read;
        } else {
          ++st.kept_held;
        }
      }
      continue;
    }
    owned_lean_pending_[kept++] = q;
  }
  owned_lean_pending_.resize(kept);
}

void NativeRenderer::OwnedLeanFrame() {
  const uint32_t bits = OwnedLeanBits();
  if (!bits) return;
  // (What follows a copy can change with the scene: a plain draw's mark lives 128 frames; a wrong guess costs one move, never a pixel.)
  if ((frame_ & 127) == 0) {
    for (OwnedLeanLearn& e : owned_lean_learn_) e.plain_next = 0;
  }
  OwnedLeanStats& st = owned_lean_stats_;
  if (!owned_lean_first_line_) {
    owned_lean_first_line_ = true;
    REXGPU_INFO("[native][lean] on: gpu_owned_copy_lean bits {} (1 no snapshot where it buys nothing, 2 restore into a fresh image, 4 unread copies off)", bits);
  }
  if (++st.frames < 600) return;
  const double f = 1.0 / double(st.frames);
  std::string after;
  for (uint32_t k = 0; k < 8; ++k) {
    if (st.draws_after[k]) after += fmt::format(" {}:{:.1f}/{:.1f}", k, st.draws_after[k] * f, st.depth_writes_after[k] * f);
  }
  REXGPU_INFO("[native][lean] frame {}: per frame: snapshots left out {:.2f} (no depth {:.2f}, a page on the image {:.2f}, restore next {:.2f}), restores in a fresh image {:.2f} "
              "(missed {:.2f}, restore checks with no proven coverage {:.2f}), moved as before {:.2f}, pages detached {:.2f} (refused {:.2f}), copies turned off {:.2f} (kept: read {:.2f}, held {:.2f}; failed {:.2f}), moves watched unread {:.2f}; "
              "draws after copy k / writing depth or stencil:{}",
              frame_, (st.alias_no_depth + st.alias_page_on_image + st.alias_restore_next) * f, st.alias_no_depth * f, st.alias_page_on_image * f, st.alias_restore_next * f, st.fresh * f,
              st.fresh_missed * f, st.no_cover * f, st.moved * f, st.detached * f, st.detach_refused * f, st.dropped * f, st.kept_read * f, st.kept_held * f, st.nop_failed * f, st.watched_unread * f, after);
  st = OwnedLeanStats();
}

}  // namespace rex::graphics::vulkan::native
