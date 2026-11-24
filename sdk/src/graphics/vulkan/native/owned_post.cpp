// wwe2k14 patch (Session 82, the owned match frame, P2): the post effects' copies and the presentation. Docs:
// docs\native\OWNED_FRAME_CONTRACTS.md section P2, OWNED_FRAME_MAP_SCENE_POST.md sections B and D.
//
//  - OwnedCopyBox: the 320x184 copies of the 320x180 post passes. The copy's rectangle is rounded up to 8 pixels
//    (RB_COPY_DEST_INFO / the extent registers count in 8s), so a 180-row box is copied as 184 rows. The rows past the
//    box land in the padding of the destination texture (the texture is 320x180; its guest memory is tiled to 32-row
//    blocks, so rows 180-191 exist as padding): no sampler reads them (the page's samplers must be 320x180 - OwnedView
//    refuses any other size), and the texture-only path of the truth copy cannot hold them either (it keeps the copy
//    as a memory write for that reason alone: native_truth.cpp, `x0 + w > texture_w || y0 + h > texture_h`). So the
//    page is the box: exact for everything that reads it.
//  - OwnedSwapView / OwnedPresentOk: the presentation. The swap reads its source as a texture of the front buffer's
//    page (fetch constant 0); an owned page for it gives the presenter our image's view. A copy into a front buffer is
//    left out only when the swap has read that page lately with exactly the page's size / format / byte order /
//    scaled flag (else the swap would read a stale texture), otherwise it stays the truth path's (reason `present`).

#include <rex/graphics/vulkan/native/native_renderer.h>

#include <rex/graphics/vulkan/command_processor.h>
#include "owned_dead_clear.h"
#include <algorithm>
#include <bitset>
#include <cstring>
#include <vector>

#include <rex/cvar.h>
REXCVAR_DECLARE(uint32_t, gpu_d3d_owned_front);  // (d3d_noring.cpp) bit 5 = M4f-A, bit 6 = M4f-B
REXCVAR_DECLARE(bool, gpu_native_owned_dead_held);  // (owned_steps.cpp)
#include <rex/logging.h>

REXCVAR_DEFINE_BOOL(gpu_native_owned_dead_depth_stale, false, "GPU/Native",                    "Session 92 M2d (review D1: off - the mark is never reset at a frame boundary and cut the next frame's quads too; E1 uses the dead clear's pending value by tick instead): M2b review F4: marking the depth stale at a dead depth clear cut Jericho (Retro)'s yRect quads off their depth (328k draws dropped, E1 0; run 013 with it off: 0 dropped, E1 373) - off by default; on only to study the stale-depth case");

REXCVAR_DEFINE_BOOL(gpu_native_owned_post_half, false, "GPU/Native",
                    "Session 82 (look lever, off by default): the owned frame's post chain - the downsample A7AD68BE, the "
                    "blur A5C9AC54 and the combine 1233F68C - drawn at half size (the combine's picture linearly "
                    "brought up to the full size for the front buffer). Softer; 720p only. Judged by eye");

REXCVAR_DEFINE_BOOL(gpu_native_owned_shadow_half, false, "GPU/Native",
                    "Session 82 (look lever, off by default; low-spec): the owned frame's 8-bit shadow maps - the ring mask, the "
                    "props map and the second map's 74 casters (RGBA8 passes cleared with depth on the 560-wide surface, "
                    "512x512 / 512x256) - drawn at half size (256), sampled as they are (softer shadows). The wrestler maps A / "
                    "B (16-bit) stay full size. 720p only. Judged by eye");

namespace rex::graphics::vulkan::native {

// Session 92 (E2): read and reset by OwnedFrameSummary (owned_steps.cpp).
uint32_t g_s92_held_not_aging = 0;

// Session 92, M2d (Part 3) and M4f-B: see owned_dead_clear.h (the pending clear of a surface slot, one table).
OwnedPendClearRec g_pend_clear[kPendClearMax];
uint32_t g_m2d_draw_index = 0;
namespace {
Draw g_pend_draw[kPendClearMax];  // (the colour clear as the game issued it: held again just before the next draw of its slot)
std::vector<int16_t> g_pend_retry;  // (slots whose pend was held again and not yet erased by a newer clear: a second drop of it is final)
// The M4f-B window counters (printed by OwnedPendPrint as "[native][pend]").
struct PendStats {
  uint32_t made_colour = 0, made_depth = 0, by_origin[kPendOriginCount] = {};
  uint32_t consumed_start = 0, consumed_depth_given = 0, consumed_depth_given_dead = 0, consumed_copy = 0;
  uint32_t superseded_by_clear = 0, superseded_in_table = 0;
  uint32_t dropped_unconsumed = 0, dropped_unservable = 0, dropped_retried = 0, dropped_overwritten = 0, dropped_table_full = 0, dropped_no_value = 0;
  uint32_t refusals_pended = 0, refusals_unpendable = 0, refusals_empty = 0;
  uint32_t unpendable_by[6] = {};  // 0 preconditions (renderer / frame off), 1 stencil-only, 2 colour format has no class, 3 no surface slot, 4 no value, 5 not the latch
  uint32_t copy_partial = 0;       // a copy larger than the pended rectangle (not filled from it)
  uint32_t dropped_own_draw = 0, dropped_not_whole = 0, frame_end_dropped = 0, recomposed = 0;
};
PendStats g_pend;
uint32_t g_pend_empty_total = 0;
bool g_pend_frame_end = false;
}  // namespace
bool OwnedPendBitOn() { return (REXCVAR_GET(gpu_d3d_owned_front) & 96u) == 96u; }
uint32_t OwnedPendEmptyTotal() { return g_pend_empty_total; }
void OwnedPendSetFrameEnd(bool on) { g_pend_frame_end = on; }
void OwnedPendRetryDone(int32_t slot) { g_pend_retry.erase(std::remove(g_pend_retry.begin(), g_pend_retry.end(), int16_t(slot)), g_pend_retry.end()); }
const OwnedDeadDClearRec* DeadDClearFor(int32_t slot, uint64_t epoch) {
  for (const OwnedPendClearRec& r : g_pend_clear) {
    if (r.slot == slot && r.slot >= 0 && !r.colour && r.frame == epoch) return &r;
  }
  return nullptr;
}
void PendNoteGivenDepth(const OwnedPendClearRec* r) {
  if (!r) return;
  OwnedPendClearRec* w = const_cast<OwnedPendClearRec*>(r);
  w->used = true;
  ++g_pend.consumed_depth_given;
  if (r->origin == kPendDead) ++g_pend.consumed_depth_given_dead;
}
namespace {
uint32_t g_m2d_watch_pages[8] = {};
uint32_t g_m2d_watch_count = 0, g_m2d_watch_logged = 0;
uint32_t g_m2d_watch_last_page = 0;  // (the page of the last watched read, until the next draw names its shaders)
}  // namespace
void M2dWatchAdd(uint32_t page) {
  for (uint32_t i = 0; i < g_m2d_watch_count; ++i) {
    if (g_m2d_watch_pages[i] == page) return;
  }
  if (g_m2d_watch_count < 8) g_m2d_watch_pages[g_m2d_watch_count++] = page;
}
void M2dWatchRead(uint32_t page, uint64_t frame, uint32_t w, uint32_t h, uint32_t format) {
  if (!g_m2d_watch_count || g_m2d_watch_logged >= 12) return;
  for (uint32_t i = 0; i < g_m2d_watch_count; ++i) {
    if (g_m2d_watch_pages[i] != page) continue;
    ++g_m2d_watch_logged;
    g_m2d_watch_last_page = page;
    REXGPU_INFO("[native][owned-only][composite] frame {}: page 0x{:05X} (written as a composite) is requested as a texture: fetch {}x{} format {} (the draw's shaders follow)", frame, page, w, h, format);
    return;
  }
}
void M2dWatchDraw(uint64_t frame, uint64_t vs, uint64_t ps) {
  if (!g_m2d_watch_last_page) return;
  REXGPU_INFO("[native][owned-only][composite] frame {}: ... sampled by the draw vs {:016X} ps {:016X} (page 0x{:05X})", frame, vs, ps, g_m2d_watch_last_page);
  g_m2d_watch_last_page = 0;
}
void M2dOrderLog(bool wanted, uint64_t frame, const char* what, const std::string& detail) {
  static uint32_t logged = 0;
  if (!wanted || logged >= 800) return;
  ++logged;
  REXGPU_INFO("[native][owned-only][order] frame {} draw {}: {} {}", frame, g_m2d_draw_index, what, detail);
}

bool NativeRenderer::OwnedPostHalfWanted(const Draw& draw, bool covered, bool cleared, uint32_t scale_x, uint32_t scale_y,
                                         uint32_t width, uint32_t height, bool depth_kept, uint32_t pitch) const {
  if (scale_x != 1 || scale_y != 1 || ((width | height) & 1)) return false;
  // (The shadow maps: a cleared start with depth into a 560-wide RGBA8 surface, 512 x 512 / 512 x 256.)
  // (Session 95, F5: the pass's first draw must be a shadow caster's - the wrestlers' map shader DA494C93 or the props' circle shadow D05BFEA1 - and
  // not the ring mask (vertex shaders A96C8C4D / A878C3FD: its page 0x1F5A2 feeds ~96 arena draws and a downscale chain that expect the full size);
  // never inside a read-back window, where no read-back copy may be refused. Session 95 r5: a quiet window counts as closed here, so a read-back can land on a half-size map after all: it is refused
  // ("half-size pass" in the read-back line) and the gate shuts for its 30 frames, full-size maps again, a sharpness step. Not seen in the Surface soak's logs; the LOW check inside a window judges it.)
  if (REXCVAR_GET(gpu_native_owned_shadow_half) && cleared && depth_kept && draw.truth_class == 1 && pitch == 7 &&
      width == 512 && (height == 512 || height == 256) && !command_processor_.NativeReadbackHolds() &&
      (draw.pixel_shader_hash == 0xDA494C9383E18961ull || draw.pixel_shader_hash == 0xD05BFEA12F4350ADull) &&
      draw.vertex_shader_hash != 0xA96C8C4D28D28F80ull && draw.vertex_shader_hash != 0xA878C3FD5E27FB35ull) {
    return true;
  }
  if (!REXCVAR_GET(gpu_native_owned_post_half) || !covered || cleared) return false;
  return draw.pixel_shader_hash == 0xA7AD68BE21C46E7Aull || draw.pixel_shader_hash == 0xA5C9AC54E4868DC1ull ||
         draw.pixel_shader_hash == 0x1233F68C15FDF948ull;
}

bool NativeRenderer::OwnedCopyBox(const OwnedPass& p, uint32_t sx, uint32_t sy, uint32_t& w, uint32_t& h) const {
  if (sx + w <= p.width && sy + h <= p.height) return true;
  if (!owned_frame_on_ || sx >= p.width || sy >= p.height) return false;
  const uint32_t over_w = sx + w > p.width ? sx + w - p.width : 0;
  const uint32_t over_h = sy + h > p.height ? sy + h - p.height : 0;
  // (The copy's 8-pixel rounding: the rows / columns it adds are fewer than 8.)
  if (over_w >= 8 || over_h >= 8) return false;
  w -= over_w;
  h -= over_h;
  return true;
}

bool NativeRenderer::OwnedDrawOverwrites(const Draw& draw, const OwnedPass& p) const {
  if (p.depth || p.stencil || p.kind != 1 || draw.truth_color < 0 || draw.truth_class != 1) return false;
  if ((draw.color_write_mask & 0xF) != 0xF || (draw.color_mask_all & ~0xFu) || draw.blend_control != 0x00010001u) {
    return false;
  }
  if (draw.fast_clear || draw.restore || draw.memexport || draw.occlusion_query || draw.xfb_buffer != VK_NULL_HANDLE ||
      !draw.instance_count) {
    return false;
  }
  if (draw.truth_depth >= 0 && (draw.raster.depth_test || draw.raster.depth_write || draw.raster.stencil)) return false;
  if (draw.truth_sx != 1 || draw.truth_sy != 1) return false;
  const auto format = xenos::ColorRenderTargetFormat((draw.color_info >> 16) & 0xF);
  if (format != xenos::ColorRenderTargetFormat::k_8_8_8_8) return false;
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  int32_t rect[4];
  TruthDrawRect(draw, rect);
  if (rect[0] != 0 || rect[1] != 0 || rect[2] <= 0 || rect[3] <= 0) return false;
  // (Whole tiles: 80 x 16 guest pixels.)
  if (rect[2] % int32_t(80 * scale_x) || rect[3] % int32_t(16 * scale_y)) return false;
  if (!(draw.cover[2] > draw.cover[0] && draw.cover[3] > draw.cover[1] && draw.cover[0] <= 0 && draw.cover[1] <= 0 &&
        draw.cover[2] * int32_t(scale_x) >= rect[2] && draw.cover[3] * int32_t(scale_y) >= rect[3])) {
    return false;
  }
  const uint32_t pitch = std::max(1u, ((draw.surface_info & 0x3FFF) * draw.truth_sx + 79) / 80);
  const std::bitset<2048> tiles = EdramTiles(draw.color_info & 0xFFF, pitch, false, 0, 0, uint32_t(rect[2]) / scale_x,
                                             uint32_t(rect[3]) / scale_y);
  if (OwnedOnly()) return false;  // (Session 92, M2b: no pass tiles under the latch; not reached - OwnedTakeDrawImpl's not_taken returns first)
  return (p.tiles & ~tiles).none();
}

bool NativeRenderer::OwnedDeadDepthClear(const Draw& draw) {
  if (!owned_frame_on_ || owned_replaying_ || !draw.fast_clear || draw.truth_depth < 0) return false;
  const auto& sd = owned_scene_depth_;
  if (!sd.frame || sd.frame != frame_ + 1) return false;  // (this frame's depth copy only)
  if ((draw.color_write_mask & 0xF) != 0 || !draw.raster.depth_write) return false;
  const uint32_t pitch = std::max(1u, ((draw.surface_info & 0x3FFF) * draw.truth_sx + 79) / 80);
  if ((draw.depth_info & 0xFFF) != sd.base || pitch != sd.pitch_tiles) return false;
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  int32_t rect[4];
  TruthDrawRect(draw, rect);
  // (The whole of the depth page's box.)
  if (rect[0] != 0 || rect[1] != 0 || uint32_t(rect[2]) < sd.width * scale_x || uint32_t(rect[3]) < sd.height * scale_y) {
    return false;
  }
  // (Nothing owned waiting for its copy may hold these depth tiles.)
  if (owned_pending_.valid && (OwnedOnly() ? (owned_pending_.depth && owned_pending_.depth_slot == draw.truth_depth) : (OwnedDrawTiles(draw) & owned_pending_.tiles).any())) return false;  // (Session 92, M2b: by depth surface under the latch)
  const int32_t rec = OwnedFrameNewPass(draw);
  OwnedFramePass& r = owned_frame_passes_[rec];
  r.owned = true;
  r.start = kOwnedStartCleared;
  r.reason = kOwnedReasonNone;
  r.box_w = sd.width;
  r.box_h = sd.height;
  r.draws = 1;
  ++truth_counts_.owned_clears_dropped;
  if (OwnedOnly() && draw.clear_depth < 0.0f) {
    for (OwnedPendClearRec& e : g_pend_clear) {
      if (e.slot == draw.truth_depth && !e.colour) e.slot = -1;  // (review D4: an unknown value must not leave an older one as the answer)
    }
  }
  if (OwnedOnly() && draw.clear_depth >= 0.0f) {
    // Session 92, M2d (Part 3): the clear stays undrawn but leaves its value as the slot's PENDING CLEAR (E1, owned_steps.cpp, gives a colour-only pass a depth image
    // cleared to it when nothing newer holds the slot's depth). One record per slot (replaced); a draw without a known clear value leaves none.
    // (M4f-B: the same table every pending clear is in - OwnedPendClearMake with its origin and the depth page's box.)
    OwnedPendClearMake(draw, kPendDead, false, sd.width, sd.height);
  }
  if (OwnedOnly() && owned_frame_stats_.frames < 4) M2dOrderLog(true, frame_, "dead depth clear", fmt::format("of depth slot {} (base {} pitch {}), value {} stencil {}{}, box {}x{}, vs {:016X} ps {:016X}",
              draw.truth_depth, sd.base, sd.pitch_tiles, draw.clear_depth, draw.raster.stencil ? "written " : "untouched ", draw.stencil_refs[0], sd.width, sd.height,
              draw.vertex_shader_hash, draw.pixel_shader_hash));
  if (OwnedOnly() && REXCVAR_GET(gpu_native_owned_dead_depth_stale)) OwnedMarkDepthStale(draw.truth_depth);  // (review F4: the dead clear is not recorded - the depth of any record / parked pass of that slot is no longer the newest, as for a dropped depth clear)
  return true;
}

VkImageView NativeRenderer::OwnedSwapView(uint32_t base_page, uint32_t width, uint32_t height, uint32_t format,
                                          uint32_t endian, bool scaled, const VkComponentMapping& mapping,
                                          bool servable) {
  // The swap's source as it is read now (every swap, with the switch on or off - a flip off keeps the pages).
  {
    OwnedSwapSeen* slot = nullptr;
    for (OwnedSwapSeen& s : owned_swap_seen_) {
      if (s.page == base_page) slot = &s;
    }
    if (!slot) slot = &owned_swap_seen_[owned_swap_next_++ & 3];
    slot->page = base_page;
    slot->width = width;
    slot->height = height;
    slot->format = format;
    slot->endian = endian;
    slot->scaled = scaled;
    slot->frame = frame_;
  }
  if (owned_pages_.empty()) {
    // (Session 95, phone black r083: the frame's only pass dropped as dead leaves no page at all, and this return came before OwnedViewPre, the one place a read of a dropped page brings the pass back: the swap read it
    // every frame, "dropped-pass pages sampled 0", and the screen stayed black. The read counts here too; the pass is drawn again from its next frame.)
    if (!owned_dead_pages_.empty() && REXCVAR_GET(gpu_native_owned_dead_held)) OwnedViewPre(base_page, width, height, format);
    return VK_NULL_HANDLE;
  }
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const bool draw_scaled = scale_x > 1 || scale_y > 1;
  owned_view_type_ = VK_IMAGE_VIEW_TYPE_2D;
  const VkImageView view = OwnedView(base_page, width, height, format, endian, scaled == draw_scaled, mapping,
                                     /*is_signed=*/false, servable);
  owned_view_type_ = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  // (Session 82 integration: an owned page the hook cannot serve - the swap reads a texture whose copy was left out:
  // its pass is refused from now on, logged; the core's IssueSwap check moved here.)
  if (view == VK_NULL_HANDLE) OwnedCheckSwapPage(base_page);
  if (view != VK_NULL_HANDLE) {
    if (DecoupleViews()) {
    for (const OwnedPage& pg : owned_pages_) {  // (Session 94, gpu_d3d_decouple: the image the swap shows - its layers are compared at the frame's end; review: only with the switch on)
      if (pg.page == base_page && pg.image >= 0 && pg.feedback < 0 && pg.page_image < 0) {
        DecoupleCompareSwap(pg.image, width * (scaled ? scale_x : 1u), height * (scaled ? scale_y : 1u));  // (review: the image is draw-resolution scaled when the swap texture is)
        break;
      }
    }
    }
    ++owned_frame_stats_.presented;
    static uint32_t logged = 0;
    if (logged < 6 || segment_log_frames_ > 0) {
      ++logged;
      REXGPU_INFO("[native][owned] frame {}: the swap presents our image for page 0x{:05X} ({}x{}, format {})", frame_,
                  base_page, width, height, format);
    }
  }
  return view;
}

bool NativeRenderer::OwnedPresentOk(uint32_t page, uint32_t width, uint32_t height, uint32_t format, uint32_t endian,
                                    bool scaled) const {
  // Session 92 (E2, TRUE NATIVE): a frame held while pipelines are made returns before the swap (no OwnedSwapView), so frame_ runs on while no
  // page is read: staleness is measured against the newest swap seen of any page then, not frame_.
  uint64_t now = frame_;
  if (OwnedChains()) {
    uint64_t newest = 0;
    for (const OwnedSwapSeen& s : owned_swap_seen_) newest = std::max<uint64_t>(newest, s.frame);
    now = std::min<uint64_t>(now, newest);
  }
  for (const OwnedSwapSeen& s : owned_swap_seen_) {
    // (Seen lately: the front buffers alternate - a page is read every second frame or so.)
    if (s.page == page && s.page && s.width == width && s.height == height && s.format == format &&
        s.endian == endian && s.scaled == scaled && s.frame + 16 >= now) {
      if (s.frame + 16 < frame_) ++g_s92_held_not_aging;
      return true;
    }
  }
  return false;
}

// ---- Session 92, M4f-B: the pending clear of a surface slot (owned_dead_clear.h) ---------------------------------------------------------------------------

namespace {
// The record a new pending clear of `slot` goes into: the slot's own (replaced), else a free one, else the oldest (dropped, counted).
OwnedPendClearRec* PendSlotFor(int32_t slot) {
  OwnedPendClearRec* rec = nullptr;
  for (OwnedPendClearRec& e : g_pend_clear) {
    if (e.slot == slot) rec = &e;
  }
  if (rec) {
    if (!rec->used && rec->origin != kPendDead) ++g_pend.superseded_in_table;
    return rec;
  }
  for (OwnedPendClearRec& e : g_pend_clear) {
    if (e.slot < 0) return &e;
  }
  for (OwnedPendClearRec& e : g_pend_clear) {
    if (!rec || e.tick < rec->tick) rec = &e;
  }
  if (!rec->used && rec->origin != kPendDead) ++g_pend.dropped_table_full;
  return rec;
}
void PendRetryRemove(int32_t slot) {
  g_pend_retry.erase(std::remove(g_pend_retry.begin(), g_pend_retry.end(), int16_t(slot)), g_pend_retry.end());
}
}  // namespace

void OwnedPendNoteRefusal(int kind) {
  if (kind == 5) {
    ++g_pend.refusals_empty;
    ++g_pend_empty_total;
  } else if (kind == 6) {
    ++g_pend.unpendable_by[5];  // (bit 64 on, but not an owned-only frame: the old path ran - not a loss)
  } else if (kind >= 0 && kind < 5) {
    ++g_pend.refusals_unpendable;
    ++g_pend.unpendable_by[kind];
  }
}

bool NativeRenderer::OwnedPendClearMake(const Draw& d, int origin, bool order_work, uint32_t box_w, uint32_t box_h, const int32_t* require_rect) {
  if (!OwnedOnly() || !d.fast_clear) return false;
  if (g_pend_frame_end && origin != kPendRefusal && origin != kPendDead) {  // (review fix 9: what the frame's end flushes was used by nothing - dropped there, counted on its own)
    ++g_pend.frame_end_dropped;
    return false;
  }
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  int32_t rect[4];
  TruthDrawRect(d, rect);
  const uint32_t pitch = std::max(1u, ((d.surface_info & 0x3FFF) * d.truth_sx + 79) / 80);
  bool made = false, retried = false, counted = false;
  // ---- the colour part ----
  if (d.truth_color >= 0 && (d.color_write_mask & 0xF) != 0 && d.clear_color_known && d.truth_class >= 1 && d.truth_class <= 2) {
    const uint32_t base = d.color_info & 0xFFF;
    bool final_drop = false;
    if (require_rect && std::memcmp(rect, require_rect, sizeof(rect)) != 0) {  // (review fix 6: a held set pends only as ONE whole value)
      ++g_pend.dropped_not_whole;
      final_drop = true;
      counted = true;
    }
    if (origin == kPendHeldDrop) {  // (the clear was held again from a pend and dropped again: final - a clear is never pended twice)
      const auto it = std::find(g_pend_retry.begin(), g_pend_retry.end(), int16_t(d.truth_color));
      if (it != g_pend_retry.end()) {
        g_pend_retry.erase(it);
        ++g_pend.dropped_retried;
        final_drop = true;
        retried = true;
      }
    }
    if (!final_drop) {
      if (order_work) {
        // The order rule: nothing that holds the surface may take a draw that comes after this clear. What the game's own clear does in OwnedHoldClear: the held clears
        // before it become their pass, the pending pass of the surface is parked (or its committed end), the parked passes and records of the surface lose their colour.
        static const char* const kWhy = "a held clear over its tiles or surface";
        if (!owned_held_.draws.empty()) OwnedFlushHeld(true);
        if (owned_pending_.valid && owned_pending_.base == base && owned_pending_.pitch_tiles == pitch && owned_pending_.kind == uint32_t(d.truth_class)) {
          if (owned_pending_.committed) {
            OwnedEndCommitted(kWhy);
          } else {
            OwnedParkOrReplay(kWhy);
          }
        }
        OwnedDropParkedIf(kWhy, [&](const OwnedPass& q) { return q.base == base && q.pitch_tiles == pitch && q.kind == uint32_t(d.truth_class); });
      }
      OwnedSupersedeRecords(base, pitch, d.truth_class, -1);
      OwnedPendClearRec* rec = PendSlotFor(d.truth_color);
      const size_t index = size_t(rec - g_pend_clear);
      *rec = OwnedPendClearRec();
      rec->slot = d.truth_color;
      rec->colour = true;
      for (uint32_t i = 0; i < 4; ++i) rec->colour_value[i] = d.clear_color[i];
      for (uint32_t i = 0; i < 4; ++i) rec->rect[i] = rect[i];
      rec->colour_mask = d.color_write_mask & 0xF;
      rec->base = base;
      rec->pitch_tiles = pitch;
      rec->kind = uint32_t(d.truth_class);
      rec->frame = truth_claim_epoch_;
      rec->tick = ++owned_tick_;
      rec->origin = uint8_t(origin);
      g_pend_draw[index] = d;
      {  // (review fix 3: the depth part has its own record for E1 - the stored colour clear carries no depth / stencil, so a re-held clear cannot clear a depth surface a later draw reuses)
        Draw& st = g_pend_draw[index];
        st.truth_depth = -1;
        st.depth_info = 0;
        st.raster.depth_test = st.raster.depth_write = false;
        st.raster.stencil = false;
        st.clear_depth = -1.0f;
      }
      ++g_pend.made_colour;
      ++g_pend.by_origin[origin];
      made = true;
    }
  }
  // ---- the depth part ----
  if (d.truth_depth >= 0 && d.raster.depth_write && d.clear_depth >= 0.0f) {
    if (origin == kPendRefusal) OwnedMarkDepthStale(d.truth_depth);  // (a record or a pass holding that depth loses it, as a dropped depth clear leaves it)
    OwnedPendClearRec* rec = PendSlotFor(d.truth_depth);
    *rec = OwnedPendClearRec();
    rec->slot = d.truth_depth;
    rec->depth = d.clear_depth;
    // (review D3: the cleared stencil is trusted only for a whole stencil clear - the test OwnedDepthClearInPass uses)
    const auto stencil_whole = [](uint32_t ops) { return ((ops >> 3) & 7) == uint32_t(VK_STENCIL_OP_REPLACE) && ((ops >> 9) & 7) == uint32_t(VK_COMPARE_OP_ALWAYS); };
    rec->stencil_written = d.raster.stencil && stencil_whole(d.raster.stencil_ops & 0xFFF) && stencil_whole((d.raster.stencil_ops >> 12) & 0xFFF) &&
                           d.stencil_write_masks[0] == 0xFF && d.stencil_write_masks[1] == 0xFF && d.stencil_refs[0] == d.stencil_refs[1];
    rec->stencil = rec->stencil_written ? uint32_t(d.stencil_refs[0]) : 0u;
    if (box_w && box_h) {
      rec->box_w = box_w;
      rec->box_h = box_h;
    } else if (rect[0] == 0 && rect[1] == 0 && rect[2] > 0 && rect[3] > 0) {
      rec->box_w = uint32_t(rect[2]) / scale_x;
      rec->box_h = uint32_t(rect[3]) / scale_y;
    }
    for (uint32_t i = 0; i < 4; ++i) rec->rect[i] = rect[i];
    rec->frame = truth_claim_epoch_;  // (the registry epoch, not the frame: the record stays the slot's newest answer until a newer holder appears - E1 compares ticks)
    rec->tick = ++owned_tick_;
    rec->origin = uint8_t(origin);
    ++g_pend.made_depth;
    ++g_pend.by_origin[origin];
    made = true;
  }
  if (made && origin == kPendRefusal) ++g_pend.refusals_pended;
  if (!made && !retried && !counted && origin != kPendDead && origin != kPendRefusal) ++g_pend.dropped_no_value;  // (a clear with no colour value we know, no class, or no depth value: nothing to remember)
  return made;
}

void NativeRenderer::OwnedPendClearMaterialize(const Draw& draw) {
  // The order rule's consumer: a draw of a slot with a pended colour clear comes AFTER that clear, so the clear is held again right here - exactly as if the game had
  // issued it just before this draw (OwnedHoldClear: the same tests, the same pass start from held clears, the same flush if the draw does not take them). Held again and
  // dropped again at a flush is final (the retry list). Never later than the draw: this runs at the entry of OwnedTakeDraw.
  if (!OwnedOnly() || draw.truth_color < 0 || owned_replaying_ || owned_flushing_held_) return;
  bool any_colour = false;
  for (const OwnedPendClearRec& r : g_pend_clear) any_colour = any_colour || (r.slot >= 0 && r.colour);
  if (!any_colour) return;
  for (size_t i = 0; i < kPendClearMax; ++i) {
    OwnedPendClearRec& r = g_pend_clear[i];
    if (!r.colour || r.slot != draw.truth_color) continue;
    if (r.frame != truth_claim_epoch_) {  // (the registry was rebuilt: its slots mean something else now)
      ++g_pend.dropped_unconsumed;
      r = OwnedPendClearRec();
      return;
    }
    Draw clear = g_pend_draw[i];
    r = OwnedPendClearRec();
    if (std::find(g_pend_retry.begin(), g_pend_retry.end(), draw.truth_color) == g_pend_retry.end()) g_pend_retry.push_back(draw.truth_color);
    if (OwnedHoldClear(clear)) {
      ++g_pend.consumed_start;
    } else {
      ++g_pend.dropped_unservable;
      PendRetryRemove(draw.truth_color);
    }
    return;
  }
}

void NativeRenderer::OwnedPendClearEraseFor(const Draw& c) {
  // A newer clear of the slot: the pend is not the slot's content any more. Whole over the pend's rectangle = superseded. A PARTIAL newer clear cannot replace a whole
  // older one: the older colour clear is held again first (review fix 6), so the two compose in the held set exactly as the game's two clears would; a depth pend that
  // is covered only in part is dropped (counted).
  if (!OwnedOnly()) return;
  const bool colour = c.truth_color >= 0 && (c.color_write_mask & 0xF) != 0;
  const bool depth = c.truth_depth >= 0 && c.raster.depth_write;
  if (!colour && !depth) return;
  if (colour) PendRetryRemove(c.truth_color);  // (review fix 5: a new clear of the slot ends the retry mark, whatever the table held)
  int32_t rect[4];
  TruthDrawRect(c, rect);
  for (size_t i = 0; i < kPendClearMax; ++i) {
    OwnedPendClearRec& r = g_pend_clear[i];
    if (r.slot < 0) continue;
    if (!((colour && r.colour && r.slot == c.truth_color) || (depth && !r.colour && r.slot == c.truth_depth))) continue;
    if (rect[0] <= r.rect[0] && rect[1] <= r.rect[1] && rect[2] >= r.rect[2] && rect[3] >= r.rect[3]) {
      if (!r.used && r.origin != kPendDead) ++g_pend.superseded_by_clear;
    } else if (r.colour) {
      Draw old = g_pend_draw[i];
      r = OwnedPendClearRec();
      ++g_pend.recomposed;
      if (std::find(g_pend_retry.begin(), g_pend_retry.end(), old.truth_color) == g_pend_retry.end()) g_pend_retry.push_back(old.truth_color);
      if (OwnedHoldClear(old)) {
        ++g_pend.consumed_start;
      } else {
        ++g_pend.dropped_unservable;
        PendRetryRemove(old.truth_color);
      }
      continue;
    } else if (!r.used && r.origin != kPendDead) {
      ++g_pend.dropped_overwritten;
    }
    r = OwnedPendClearRec();
  }
}

void NativeRenderer::OwnedPendClearDropNewerThan(int32_t colour_slot, uint64_t tick) {
  // Review fix 2: a held clear of the draw's own colour slot dropped at the flush inside that draw's own OwnedTakeDrawImpl was pended after the draw had been
  // recorded - consuming it at the next draw of the slot would wipe the draw. Dropped, counted ("made under its own draw").
  for (OwnedPendClearRec& r : g_pend_clear) {
    if (r.slot == colour_slot && r.slot >= 0 && r.colour && r.tick > tick) {
      ++g_pend.dropped_own_draw;
      r = OwnedPendClearRec();
    }
  }
}

bool NativeRenderer::OwnedPendColourFor(uint32_t base, uint32_t pitch_tiles, int32_t x0, int32_t y0, int32_t x1, int32_t y1, float value[4]) {
  for (OwnedPendClearRec& r : g_pend_clear) {
    if (r.slot < 0 || !r.colour || r.frame != truth_claim_epoch_ || r.base != base || r.pitch_tiles != pitch_tiles || r.kind != 1) continue;
    if (r.colour_mask != 0xF) return false;  // (review fix 7: a partial-channel clear is not a uniform value)
    if (!(r.rect[0] <= x0 && r.rect[1] <= y0 && r.rect[2] >= x1 && r.rect[3] >= y1)) {  // (the copy's real source rectangle, in host pixels)
      ++g_pend.copy_partial;
      return false;
    }
    for (uint32_t i = 0; i < 4; ++i) value[i] = r.colour_value[i];
    r.used = true;
    ++g_pend.consumed_copy;
    return true;
  }
  return false;
}

void NativeRenderer::OwnedPendClearFrameEnd() {
  // Never consumed = dropped, counted (nothing read it). The dead depth clear stays across frames, as M2d made it (E1 compares ticks).
  for (OwnedPendClearRec& r : g_pend_clear) {
    if (r.slot < 0 || r.origin == kPendDead) continue;
    if (!r.used) ++g_pend.dropped_unconsumed;
    r = OwnedPendClearRec();
  }
  g_pend_retry.clear();
}

void NativeRenderer::OwnedPendClearReset() {
  for (OwnedPendClearRec& r : g_pend_clear) r = OwnedPendClearRec();
  g_pend_retry.clear();
}

void OwnedPendPrint(uint64_t first_frame, uint64_t frames) {
  if (!OwnedPendBitOn()) {  // (review fix 9: not printed, and the counters not carried over, in a run without bits 32 + 64 - the dead depth clears alone must not make the line appear)
    g_pend = PendStats();
    return;
  }
  const PendStats& s = g_pend;
  REXGPU_INFO("[native][pend] frames {}-{}: pending clears made {} (colour {}, depth {}) [from D3dClear refusals {}, held clears dropped at a flush {}, pending depth-only clears flushed {} / dropped {}, dead depth clears {}]; "
              "consumed [pass start (held again before a draw) {}, depth given to a colour-only pass {} (dead depth clears {}), copy filled {}]; superseded [by a newer clear {}, replaced in the table {}]; "
              "dropped [unconsumed at frame end {}, unservable (held again and refused) {}, held again and dropped again {}, partly overwritten {}, table full {}, no value or class to remember {}]; "
              "D3dClear refusals pended {} (unpendable {} [no renderer or frame {}, stencil only {}, colour format without a class {}, no surface slot {}, no value {}, not the latch {}], empty rectangle {}); copies larger than the pended rectangle {}; pends made under their own draw and dropped {}; held sets not one whole value {}; older pends re-held to compose with a partial clear {}; held / depth clears dropped at the frame's end (used by nothing) {}",
              first_frame, first_frame + frames - 1, s.made_colour + s.made_depth, s.made_colour, s.made_depth, s.by_origin[kPendRefusal], s.by_origin[kPendHeldDrop], s.by_origin[kPendDepthFlush], s.by_origin[kPendDepthDrop],
              s.by_origin[kPendDead], s.consumed_start, s.consumed_depth_given, s.consumed_depth_given_dead, s.consumed_copy, s.superseded_by_clear, s.superseded_in_table, s.dropped_unconsumed, s.dropped_unservable,
              s.dropped_retried, s.dropped_overwritten, s.dropped_table_full, s.dropped_no_value, s.refusals_pended, s.refusals_unpendable, s.unpendable_by[0], s.unpendable_by[1], s.unpendable_by[2], s.unpendable_by[3], s.unpendable_by[4], s.unpendable_by[5],
              s.refusals_empty, s.copy_partial, s.dropped_own_draw, s.dropped_not_whole, s.recomposed, s.frame_end_dropped);
  g_pend = PendStats();
}

}  // namespace rex::graphics::vulkan::native
