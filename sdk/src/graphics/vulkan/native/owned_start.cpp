// wwe2k14 patch (Session 82, the owned match frame, P0): the pass starts the base cases (cleared, covered -
// OwnedTakeDraw) do not give. Docs: docs\native\OWNED_FRAME_DESIGN.md sections 0 and 3.2; CLAUDE.md hard rule 6
// (never keep a 360 quirk just to match it).
//
// The defined start. A draw that blends onto, or does not cover, the box of its surface would read, on the console,
// whatever other surfaces left in those EDRAM tiles (G3 blends onto leftovers of the shadow chain and the small renders,
// the reductions onto the chain's earlier tiles). That is an accident of the 360's shared tile memory, not something
// the game draws: our pass starts from a defined box instead - cleared to transparent black - and nothing of it is
// a truth image's: no tiles are claimed, nothing is marked dirty, and no truth path loads, stores or copies for it.
// (Session 82's first converted start loaded the box through the truth path and copied it: removed.) The picture is
// then close, not identical, to owned off wherever a leftover showed - measured with owned_check's tolerance.

#include <rex/graphics/vulkan/native/native_renderer.h>

#include <cstdlib>
#include <rex/cvar.h>

#include <algorithm>
#include <cmath>

#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/deferred_command_buffer.h>
#include <rex/logging.h>
#include <rex/ui/vulkan/util.h>

REXCVAR_DEFINE_STRING(gpu_native_owned_frame_defined_skip, "", "GPU/Native",
                      "Session 82 (diagnostic): pixel shaders (hex hashes, comma-separated) whose draws get no defined start - they stay "
                      "on the truth path (bisecting a picture difference of the owned frame)");

REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_loaded_from_parked, true, "GPU/Native",
                    "Session 90 (Q): a draw that keeps content and finds no parked pass to resume (a loaded start) takes the same-surface parked defined starts "
                    "into its new image from their own images (their drawn rectangles copied in) instead of replaying their draws on the truth path. "
                    "Off = the Session 87 replay");

REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_defined_start_16, false, "GPU/Native",
                    "Session 87 (entrance passes, reason start): the defined start (a cleared box) also for the 16-bit class-2 targets "
                    "(k_16_16 / k_16_16_16_16, the shadow-map renders): a colour-only draw that is neither covering nor preceded by whole "
                    "clears. The entrance replay refuses 212 such two-draw 512x512 renders per 600 frames. Default off");

namespace rex::graphics::vulkan::native {

bool NativeRenderer::OwnedConvertible(const Draw& draw, const int32_t rect[4], bool depth_used, uint32_t scale_x,
                                      uint32_t scale_y) const {
  // RGBA8, colour only. The box is the rectangle from the surface's origin to the draw's far corner; a draw away from the
  // origin (Session 82, P2, G4: the damage-map splats at x 382, y 158 of a 1040-wide surface) leaves the rest of the box
  // cleared - a copy of it is trusted only inside a draw's rectangle (OwnedRectDrawn).
  const auto rt_format = xenos::ColorRenderTargetFormat((draw.color_info >> 16) & 0xF);
  const bool class2_16 = (REXCVAR_GET(gpu_native_owned_frame_defined_start_16) || OwnedChains()) && draw.truth_class == 2 &&
                         (rt_format == xenos::ColorRenderTargetFormat::k_16_16 ||
                          rt_format == xenos::ColorRenderTargetFormat::k_16_16_16_16);
  return (draw.truth_class == 1 || class2_16) && !depth_used && !draw.raster.stencil && (draw.color_write_mask & 0xF) != 0 &&
         rect[0] >= 0 && rect[1] >= 0 && rect[2] > 0 && rect[3] > 0 && rect[2] % int32_t(scale_x) == 0 &&
         rect[3] % int32_t(scale_y) == 0 && rect[2] <= 4096 && rect[3] <= 4096;
}

bool NativeRenderer::OwnedLoadParkedInto(int16_t image, bool has_content, const std::vector<OwnedPass>& passes, uint32_t scale_x,
                                         uint32_t scale_y) {
  // Session 90 (Q): the drawn rectangles of parked defined starts (oldest first: the list is in park order) copied from their own images into `image` -
  // outside any render pass, before the new pass's first barrier. `has_content`: the image holds the truth path's content already (OwnedLoadFromTruth left
  // it in its attachment layout); else it is cleared to transparent black first. A parked pass is newer than the truth content under it (a truth draw over
  // its tiles drops it first), so its rectangle goes over it.
  if (image < 0 || size_t(image) >= owned_images_.size()) return false;
  struct Copy {
    int16_t src;
    VkImageCopy region;
  };
  std::vector<Copy> copies;
  for (const OwnedPass& q : passes) {
    if (q.image < 0 || size_t(q.image) >= owned_images_.size() || q.image == image) continue;
    const OwnedImage& s = owned_images_[q.image];
    const OwnedImage& n = owned_images_[image];
    if (q.drawn[2] <= q.drawn[0] || q.drawn[3] <= q.drawn[1]) continue;
    const int32_t x0 = std::max(q.drawn[0], 0) * int32_t(scale_x), y0 = std::max(q.drawn[1], 0) * int32_t(scale_y);
    const int32_t x1 = std::min({q.drawn[2] * int32_t(scale_x), int32_t(s.width), int32_t(n.width)});
    const int32_t y1 = std::min({q.drawn[3] * int32_t(scale_y), int32_t(s.height), int32_t(n.height)});
    if (x1 <= x0 || y1 <= y0) continue;
    Copy c = {};
    c.src = q.image;
    c.region.srcSubresource = c.region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.region.srcOffset = c.region.dstOffset = {x0, y0, 0};
    c.region.extent = {uint32_t(x1 - x0), uint32_t(y1 - y0), 1};
    copies.push_back(c);
  }
  command_processor_.EndRenderPass();
  command_processor_.SubmitBarriers(true);
  const OwnedImage& o = owned_images_[image];
  OwnedTouch(image);
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const auto barrier = [](VkImage img, VkAccessFlags src, VkAccessFlags dst, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.oldLayout = from;
    b.newLayout = to;
    b.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);
    return b;
  };
  constexpr VkPipelineStageFlags kStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                                           VK_PIPELINE_STAGE_TRANSFER_BIT;
  constexpr VkAccessFlags kAnyWrite = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                      VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  const VkAccessFlags attachment_access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  const uint32_t timer = OwnedTimerBegin("load_parked", o.width, o.height);
  VkImageMemoryBarrier b = barrier(o.image, has_content ? VkAccessFlags(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT) : kAnyWrite,
                                   VK_ACCESS_TRANSFER_WRITE_BIT,
                                   has_content ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  cb.CmdVkPipelineBarrier(kStages, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
  if (!has_content) {  // (the box outside the copied rectangles is transparent black, as the converted start had it)
    const VkClearColorValue zero = {};
    const VkImageSubresourceRange range = ui::vulkan::util::InitializeSubresourceRange();
    cb.CmdVkClearColorImage(o.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
    b = barrier(o.image, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
  }
  for (const Copy& c : copies) {
    OwnedImage& s = owned_images_[c.src];
    OwnedTouch(c.src);
    const VkImageLayout src_layout = s.readable ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkImageMemoryBarrier sb = barrier(s.image, s.readable ? VkAccessFlags(0) : VkAccessFlags(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT),
                                      VK_ACCESS_TRANSFER_READ_BIT, src_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    cb.CmdVkPipelineBarrier(kStages, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &sb);
    OwnedCopyImage(cb, s.image, true, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, o.image, true, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c.region);  // (Session 94: per layer)
    sb = barrier(s.image, VK_ACCESS_TRANSFER_READ_BIT, s.readable ? VkAccessFlags(VK_ACCESS_SHADER_READ_BIT) : attachment_access,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, src_layout);
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, kStages, 0, 0, nullptr, 0, nullptr, 1, &sb);
    // (the next copy's write into the new image after this one: a barrier of its own)
    b = barrier(o.image, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
  }
  b = barrier(o.image, VK_ACCESS_TRANSFER_WRITE_BIT, attachment_access, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, kStages, 0, 0, nullptr, 0, nullptr, 1, &b);
  OwnedTimerEnd(timer);
  owned_images_[image].readable = false;
  static uint32_t logged = 0;
  if (logged < 12 || owned_frame_logging_) {
    ++logged;
    REXGPU_INFO("[native][owned] frame {}: a loaded start takes {} parked defined start(s) from their own images ({} rectangles copied; truth content {})",
                frame_, passes.size(), copies.size(), has_content ? "loaded first" : "none");
  }
  return true;
}

bool NativeRenderer::OwnedStartExtra(OwnedStartContext& ctx) {
  const Draw& draw = *ctx.draw;
  {
    // (Diagnostic: the pixel shaders listed get no defined start.)
    const std::string skip = REXCVAR_GET(gpu_native_owned_frame_defined_skip);
    for (size_t k = 0; k < skip.size();) {
      const size_t end = std::min(skip.find(',', k), skip.size());
      const unsigned long long hash = std::strtoull(skip.substr(k, end - k).c_str(), nullptr, 16);
      if (hash && hash == ctx.draw->pixel_shader_hash) {
        ctx.reason = kOwnedReasonStart;
        ctx.detail = "defined start skipped";
        return false;
      }
      k = end + 1;
    }
  }
  const int32_t* rect = ctx.rect;
  const uint32_t scale_x = ctx.scale_x, scale_y = ctx.scale_y;
  const bool depth_used =
      draw.truth_depth >= 0 && (draw.raster.depth_test || draw.raster.depth_write || draw.raster.stencil);
  if (!OwnedConvertible(draw, rect, depth_used, scale_x, scale_y)) {
    ctx.reason = kOwnedReasonStart;
    ctx.detail = "coverage";
    return false;
  }
  const OwnedPassKey key = OwnedKeyOf(draw);
  if (OwnedPaused(key)) {
    ctx.reason = kOwnedReasonPaused;
    ctx.detail = "paused";
    return false;
  }
  if (OwnedPresentKeyed(key)) {
    ctx.reason = kOwnedReasonPresent;
    ctx.detail = "front buffer";
    return false;
  }
  // (Session 87, chains: a draw on a surface that already has content does not start from transparent black. Our own parked pass of that surface is taken
  // up again - the newest one whose box holds the draw - or, with none, the box is loaded once from the truth path (below); the transparent-black
  // converted start stays for a surface with nothing in it.)
  // (Which draws get a start that keeps content: only a draw that READS the destination - blended with a destination factor, a partial colour write mask, a
  // pixel shader that reads the target. An opaque draw that writes its rectangle whole does not care what was under it: it keeps the converted start. And
  // not a scratch surface the feedback path has seen.)
  const bool unblended_draw = (draw.color_write_mask & 0xF) == 0xF && !(draw.color_mask_all & ~0xFu) && draw.blend_control == 0x00010001u;
  const bool keeps_content = OwnedChains() && owned_frame_on_ && (!unblended_draw || PostNeedsDst(draw)) &&
                             !OwnedFeedbackSurface(draw.color_info & 0xFFF, std::max(1u, ((draw.surface_info & 0x3FFF) * draw.truth_sx + 79) / 80), uint32_t(draw.truth_class));
  if (keeps_content && !owned_pending_.valid) {
    const uint32_t rpitch = std::max(1u, ((draw.surface_info & 0x3FFF) * draw.truth_sx + 79) / 80);
    for (size_t pj = owned_parked_.size(); pj-- > 0;) {
      const OwnedPass& q = owned_parked_[pj];
      if (q.base != (draw.color_info & 0xFFF) || q.pitch_tiles != rpitch || q.kind != uint32_t(draw.truth_class) || !q.committed || rect[0] < 0 || rect[1] < 0 ||
          rect[2] > int32_t(q.host_width) || rect[3] > int32_t(q.host_height)) {
        continue;
      }
      OwnedPass taken = std::move(owned_parked_[pj]);
      owned_parked_.erase(owned_parked_.begin() + pj);
      owned_pending_ = std::move(taken);
      ++owned_frame_stats_.ch_resumed;
      if (owned_pending_.carried_defined) ++owned_frame_stats_.ch_carried_taken_draw;
      ctx.start = kOwnedStartResumed;
      return true;
    }
  }
  if (keeps_content && !owned_pending_.valid && OwnedOnly()) {
    // Session 92, M2a (design 4 item 4): the surface's newest record - a pass that ended - is resumed like a parked one (no load from the truth path
    // exists): the newest one whose box holds the draw and whose colour no later pass has replaced. None: the start below is the converted one (a
    // cleared box) - counted, the first named, so a missing record shows as a picture question, not silently.
    const uint32_t rpitch = std::max(1u, ((draw.surface_info & 0x3FFF) * draw.truth_sx + 79) / 80);
    for (size_t ri = owned_newest_.size(); ri-- > 0;) {
      const OwnedPass& q = owned_newest_[ri];
      if (q.base != (draw.color_info & 0xFFF) || q.pitch_tiles != rpitch || q.kind != uint32_t(draw.truth_class) || !q.committed || q.rec_colour_gone ||
          rect[0] < 0 || rect[1] < 0 || rect[2] > int32_t(q.host_width) || rect[3] > int32_t(q.host_height)) {
        continue;
      }
      OwnedPass taken = std::move(owned_newest_[ri]);
      owned_newest_.erase(owned_newest_.begin() + ri);
      owned_pending_ = std::move(taken);
      owned_pending_.valid = true;
      owned_pending_.committed = true;
      owned_pending_.open = false;
      ++m2a_loaded_from_record_;
      ctx.start = kOwnedStartResumed;
      return true;
    }
    ++m2a_loaded_no_record_;
    static uint32_t no_record_logged = 0;
    if (no_record_logged < 12) {
      ++no_record_logged;
      REXGPU_INFO("[native][owned-only] frame {}: a start that keeps content finds no record of its surface: ps {:016X} surface {}/{} class {} rect {},{}-{},{} ({} records)",
                  frame_, draw.pixel_shader_hash, draw.color_info & 0xFFF, rpitch, draw.truth_class, rect[0], rect[1], rect[2], rect[3], owned_newest_.size());
    }
  }
  // (Session 90, Q: the same-surface parked DEFINED starts are not replayed on the truth path for the load - their pixels are in their own images, which are
  // copied into the new image below (their drawn rectangles), oldest first. Only the ones whose image cannot be copied keep the replay.)
  std::vector<OwnedPass> composite;
  if (keeps_content) {
    const uint32_t rpitch2 = std::max(1u, ((draw.surface_info & 0x3FFF) * draw.truth_sx + 79) / 80);
    if (REXCVAR_GET(gpu_native_owned_frame_loaded_from_parked)) {
      for (const OwnedPass& q : owned_parked_) {
        if (q.base == (draw.color_info & 0xFFF) && q.pitch_tiles == rpitch2 && q.kind == uint32_t(draw.truth_class) && q.edram_free && q.committed && !(q.chain_parked && !q.draws.empty()) &&
            q.image >= 0 && size_t(q.image) < owned_images_.size() && owned_images_[q.image].width && owned_images_[q.image].kind == q.kind &&
            q.host_width && q.host_height && q.host_width <= owned_images_[q.image].width && q.host_height <= owned_images_[q.image].height) {
          composite.push_back(q);
        }
      }
    }
    // (review A4: the parked defined starts of this surface that are not copied are replayed before the load - 'the same surface starts again' would drop them
    // with their draws unplayed. The replay records truth draws - a truth render pass is left open - so it runs HERE, before the passes are ended and our image is
    // made: run 197-203's crash was a barrier recorded inside that open truth pass, VUID-vkCmdPipelineBarrier-None-07889.)
    OwnedDropParkedIf("a loaded start of the same surface", [&](const OwnedPass& q) {
      if (!(q.base == (draw.color_info & 0xFFF) && q.pitch_tiles == rpitch2 && q.kind == uint32_t(draw.truth_class))) return false;
      for (const OwnedPass& c : composite) {
        if (c.image == q.image && c.census == q.census) return false;  // (copied below, dropped after)
      }
      return true;
    });
  }
  uint32_t width = uint32_t(rect[2]), height = uint32_t(rect[3]);
  if (rect[0] != 0 || rect[1] != 0) {
    // (Session 82, P2, G4: a draw away from the origin is one of several the game puts into one surface (the damage maps'
    // splats at x 382, y 158) - the box is the viewport's extent (the render area the game set), up to the surface's
    // width, so the next draws fit it. A draw at the origin keeps the box it always had: its copies may be whole ones.)
    const uint32_t surface_w = std::max(1u, ((draw.surface_info & 0x3FFF) * draw.truth_sx + 79) / 80) * 80u * scale_x;
    const uint32_t vp_r = uint32_t(std::max(0.0f, std::ceil(draw.viewport.x + draw.viewport.width)));
    const uint32_t vp_b = uint32_t(std::max(0.0f, std::ceil(draw.viewport.y + draw.viewport.height)));
    width = std::max(width, std::min(vp_r, surface_w));
    height = std::max(height, std::min(vp_b, 4096u));
    width = (width + scale_x - 1) / scale_x * scale_x;
    height = (height + scale_y - 1) / scale_y * scale_y;
  }
  for (const OwnedPass& q : composite) {  // (the new box holds every pass copied into it)
    width = std::max(width, q.host_width);
    height = std::max(height, q.host_height);
  }
  if (!OwnedImageAvailable(width, height, draw.truth_class, false)) {
    ctx.reason = kOwnedReasonPool;
    ctx.detail = "no image within the budget";
    return false;
  }
  OwnedPass& p = owned_pending_;
  // Our pass is a pass of its own: whatever is open ends first (a clears' pass of the truth path stays as it ran).
  EndTruthPass();
  owned_clears_ended_ = false;
  command_processor_.EndRenderPass();
  command_processor_.SubmitBarriers(true);
  const int16_t index = OwnedImageFor(width, height, draw.truth_class);
  if (index < 0) {
    ctx.reason = owned_failed_ ? kOwnedReasonFailed : kOwnedReasonPool;
    ctx.detail = "image";
    return false;
  }
  const uint32_t pitch = std::max(1u, ((draw.surface_info & 0x3FFF) * draw.truth_sx + 79) / 80);
  const uint32_t guest_w = width / scale_x, guest_h = height / scale_y;
  // (chains) The surface has content in the truth path: our image is given it once, and the pass starts with LOAD.
  const bool loaded = keeps_content && OwnedLoadFromTruth(draw.truth_color, index, width, height);
  // (Session 90, Q: then the parked defined starts' pixels, from their own images. The image is left in its attachment layout for a render pass that loads.)
  const bool composited = !composite.empty() && OwnedLoadParkedInto(index, loaded, composite, scale_x, scale_y);
  p.valid = true;
  p.kind = draw.truth_class;
  p.is_64bpp = draw.truth_class == 2 && xenos::IsColorRenderTargetFormat64bpp(xenos::ColorRenderTargetFormat((draw.color_info >> 16) & 0xF));
  p.committed = p.started = p.open = false;
  p.cleared = false;
  p.depth = p.stencil = false;
  p.draws_depth = p.draws_stencil = false;
  p.start = uint32_t(kOwnedPassClearColor);  // (the box starts as transparent black)
  p.loaded_start = false;
  p.edram_free = true;
  p.chain_pass = p.chain_unkeyed = p.chain_parked = false;  // (Session 87, review F6)
  p.carried_defined = false;  // (Session 90)
  p.feedback_pass = false;
  p.half = false;
  p.dres = false;
  p.dres_full = -1;
  p.dres_color_used = p.dres_depth_used = false;
  p.drawn[0] = p.drawn[1] = p.drawn[2] = p.drawn[3] = 0;
  p.clears_held = false;
  p.held_clears.clear();
  p.nonorigin = rect[0] != 0 || rect[1] != 0 || width != uint32_t(rect[2]) || height != uint32_t(rect[3]);
  p.clear_values[0] = p.clear_values[1] = VkClearValue();
  if (loaded && command_processor_.NativeReadbackActive()) {  // (Session 90, Q: a loaded start inside a read-back window splits a pass: counted, named)
    ++owned_frame_stats_.wb_window_loads;
    static uint32_t split_logged = 0;
    if (split_logged < 12) {
      ++split_logged;
      REXGPU_INFO("[native][owned] frame {}: LOADED START INSIDE A READ-BACK WINDOW: ps {:016X} surface {}/{} (truth content loaded into our image)", frame_,
                  draw.pixel_shader_hash, draw.color_info & 0xFFF, std::max(1u, ((draw.surface_info & 0x3FFF) * draw.truth_sx + 79) / 80));
    }
  }
  if (loaded) {  // (nothing of the box is defined by the draw: the pass holds what the surface held, and commits and marks dirty like a covered pass)
    p.start = uint32_t(kOwnedPassResume);
    p.loaded_start = true;
    p.edram_free = false;
    p.nonorigin = false;
  } else if (composited) {
    // (Session 90, Q: a defined start whose box already holds the parked passes' pixels: the render pass loads it; their drawn rectangles are drawn.)
    p.start = uint32_t(kOwnedPassResume);
    p.loaded_start = true;
    for (const OwnedPass& q : composite) {
      if (q.drawn[2] <= q.drawn[0] || q.drawn[3] <= q.drawn[1]) continue;
      if (p.drawn[2] <= p.drawn[0]) {
        for (int k = 0; k < 4; ++k) p.drawn[k] = q.drawn[k];
      } else {
        p.drawn[0] = std::min(p.drawn[0], q.drawn[0]);
        p.drawn[1] = std::min(p.drawn[1], q.drawn[1]);
        p.drawn[2] = std::max(p.drawn[2], q.drawn[2]);
        p.drawn[3] = std::max(p.drawn[3], q.drawn[3]);
      }
    }
  }
  p.base = draw.color_info & 0xFFF;
  p.pitch_tiles = pitch;
  p.color_slot = draw.truth_color;
  p.depth_slot = -1;
  p.width = guest_w;
  p.height = guest_h;
  p.host_width = width;
  p.host_height = height;
  p.image = index;
  p.frame = frame_;
  p.last_tick = ++owned_tick_;  // (Session 92, M2a)
  p.rec_colour_gone = p.rec_depth_gone = false;
  p.draws_since_color = p.draws_since_depth = 0;
  p.copy_color_page = p.copy_depth_page = 0;
  p.submission = command_processor_.GetCurrentSubmission();
  // (Only to match a copy to this pass: its source tiles.)
  p.tiles = std::bitset<2048>();  // (Session 92, M2b: no tiles under the latch)
  if (!OwnedOnly() && draw.truth_color >= 0 && size_t(draw.truth_color) < OwnedSlotCount()) {
    const OwnedSurface t = OwnedSlotKey(draw.truth_color);
    p.tiles = EdramTiles(t.base, t.pitch_tiles, t.is_64bpp(), 0, 0, guest_w, guest_h);
  }
  if (composited) {  // (Session 90, Q: taken into the new image - a reason that reads nothing: no replay)
    owned_frame_stats_.wb_loaded_parked += uint32_t(composite.size());
    OwnedDropParkedIf("taken into a loaded start", [&](const OwnedPass& q) {
      for (const OwnedPass& c : composite) {
        if (c.image == q.image && c.census == q.census) return true;
      }
      return false;
    });
  }
  OwnedDropParkedIf("the same surface starts again", [&](const OwnedPass& q) {
    return q.base == p.base && q.pitch_tiles == p.pitch_tiles && q.kind == p.kind;
  });
  OwnedSupersedeRecords(p.base, p.pitch_tiles, p.kind, -1);  // (Session 92, M2a, review D4)
  owned_restorable_.erase(std::remove_if(owned_restorable_.begin(), owned_restorable_.end(), [&](const OwnedPass& q) {
    return q.base == p.base && q.pitch_tiles == p.pitch_tiles && q.kind == p.kind;
  }), owned_restorable_.end());
  p.draws.clear();
  p.producers.clear();
  p.clears_begin = p.clears_draws = p.clears_end_cmd = p.clears_end = SIZE_MAX;
  p.clears_loaded = p.clears_ours = p.clears_depth = p.clears_stencil = false;
  p.clears_submission = p.submission;
  p.key = key;
  p.start_kind = loaded ? kOwnedStartResumed : kOwnedStartConverted;  // (named "converted" in the census: a start for a draw that is not covered)
  p.census = OwnedFrameNewPass(draw);
  OwnedFramePass& r = owned_frame_passes_[p.census];
  r.start = p.start_kind;
  r.box_w = guest_w;
  r.box_h = guest_h;
  ++truth_counts_.owned_passes;
  if (loaded) {
    ++owned_frame_stats_.ch_loaded;
  } else {
    ++owned_frame_stats_.converted;
    if (p.base == 0 && p.pitch_tiles == 16) ++owned_frame_stats_.ch_conv016;
  }
  ctx.start = p.start_kind;
  if (segment_log_frames_ > 0) {
    REXGPU_INFO("[native][seg] frame {} owned pass: defined start (cleared box), our image {} ({}x{}), ps {:016X}", frame_,
                index, width, height, draw.pixel_shader_hash);
  }
  return true;
}

}  // namespace rex::graphics::vulkan::native
