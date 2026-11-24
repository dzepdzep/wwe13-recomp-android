#pragma once
// wwe2k14 patch (Session 92, M2d, Part 3): the pending value of a dead depth clear (owned_post.cpp OwnedDeadDepthClear) and the per-window order log
// of {scene depth copy, dead depth clear, colour-only pass given depth}. File-level state shared by owned_post.cpp, owned_scene.cpp and
// owned_steps.cpp (no header the game exe includes changes). Docs: docs\native\d3d\S92_M1A_BOX_READ.md section A.
#include <cstdint>
#include <string>

namespace rex::graphics::vulkan::native {

// Session 92, M4f-B (gpu_d3d_owned_front bit 64): the PENDING CLEAR of a surface slot - ONE mechanism. M2d's dead depth clear (a depth-only whole clear the
// owned frame leaves undrawn: "dead", nothing of ours is cleared) is one origin of it; the others are the clears the owned frame could not begin a pass with
// (a record clear D3dClear was refused, held clears dropped at a flush, pending depth-only clears flushed / dropped). One record per slot (a colour value, or
// depth + stencil), the slot's content until a newer holder appears (E1: tick compare); consumed by the slot's next pass start (a colour clear is held again
// just before the draw: OwnedPendClearMaterialize; a depth value is given to a colour-only pass by E1) or by a copy of the slot (the defined fill writer with
// the value); never consumed = dropped at the frame's end, counted (the dead depth clear stays across frames, as M2d made it).
// Docs: docs\native\d3d\S92_M4F_DESIGN.md section 2.
enum OwnedPendOrigin : uint8_t { kPendDead = 0, kPendRefusal, kPendHeldDrop, kPendDepthFlush, kPendDepthDrop, kPendOriginCount };
struct OwnedPendClearRec {
  int32_t slot = -1;
  bool colour = false;  // true: a colour clear (colour_value, rect); false: depth (+ stencil)
  float colour_value[4] = {};
  uint32_t colour_mask = 0;  // the clear's colour write mask (a copy fills only a whole 0xF)
  float depth = 0.0f;
  uint32_t stencil = 0;
  bool stencil_written = false;
  uint32_t box_w = 0, box_h = 0;                 // guest pixels, from the origin (0 = a clear that does not start at the origin: no box)
  int32_t rect[4] = {};                          // host pixels (TruthDrawRect)
  uint32_t base = 0, pitch_tiles = 0, kind = 0;  // a colour clear: the surface it clears (the copy's lookup)
  uint64_t frame = 0;                            // the registry epoch (truth_claim_epoch_) it was made in
  uint64_t tick = 0;                             // owned_tick_ then
  uint8_t origin = kPendDead;
  bool used = false;  // consumed at least once (E1 gave depth from it / a copy was filled from it)
};
using OwnedDeadDClearRec = OwnedPendClearRec;  // (M2d's name, still the type E1 reads)
constexpr size_t kPendClearMax = 8;
extern OwnedPendClearRec g_pend_clear[kPendClearMax];
extern uint32_t g_m2d_draw_index;  // draws seen by OwnedTakeDrawImpl this frame (reset at the frame's end) - the order log's draw index

// Part 4 (the 512x512 box copies written as a composite): the destination pages of those copies are watched - who samples them afterwards is logged
// (a texture request of the page: its size and format; the draw that follows: its shaders). Logs the first 12, then stays silent.
void M2dWatchAdd(uint32_t page);
void M2dWatchRead(uint32_t page, uint64_t frame, uint32_t w, uint32_t h, uint32_t format);
void M2dWatchDraw(uint64_t frame, uint64_t vs, uint64_t ps);

// The pending DEPTH clear of the slot made in the registry epoch `epoch`, or null (colour pends are not answered here).
const OwnedDeadDClearRec* DeadDClearFor(int32_t slot, uint64_t epoch);  // (epoch = NativeRenderer::truth_claim_epoch_)
// Bit 64 of gpu_d3d_owned_front with bit 32 (M4f-B on): the hold is effective, refusals pend, the flush drops feed the pend table.
bool OwnedPendBitOn();
// D3dClear's (d3d_copy_records.cpp) own refusals, counted in the window line: 0-4 a clear that could not be pended (preconditions / stencil only / colour format
// without a class / no surface slot / no value), 5 an empty rectangle (nothing to clear, handled).
void OwnedPendNoteRefusal(int kind);
// Monotonic count of empty-rectangle clears handled by D3dClear (Execute does not count those as held).
uint32_t OwnedPendEmptyTotal();
// The frame end flushes held clears / pending depth clears that nothing used: they are dropped there, not pended (counted on their own).
void OwnedPendSetFrameEnd(bool on);
// A re-held pended clear of this colour slot became a pass: it is no longer on the retry list.
void OwnedPendRetryDone(int32_t slot);
// E1 gave a colour-only pass depth from this pending clear (counted; the record is marked used).
void PendNoteGivenDepth(const OwnedPendClearRec* r);
// The `[native][pend]` window line (Session 92, M4f-B); resets its counters. Printed only when bit 64 is on or something was counted.
void OwnedPendPrint(uint64_t first_frame, uint64_t frames);
// One line of the order log ([native][owned-only][order]); the caller passes whether this is one of the window's first 4 frames.
void M2dOrderLog(bool wanted, uint64_t frame, const char* what, const std::string& detail);

}  // namespace rex::graphics::vulkan::native

namespace rex::graphics::vulkan {
// render_target_cache.cpp (M2d review E2 / E4 / E5): the cache made no EDRAM buffer; stock EDRAM paths asked for anyway (and those before the renderer's first frame boundary).
bool M2dEdramLess();
uint32_t M2dEdramLessDrops();
uint32_t M2dEdramLessBootDrops();
void M2dRendererFrameSeen();
}  // namespace rex::graphics::vulkan
