// wwe2k14 patch (Session 95, item K, gpu_native_owned_shadow_skip, default false): the dependent-pass skip of the wrestlers' shadow chain.
// gpu_shadow_half_rate drops a wrestler map's draws + copy on alternate frames; the reductions (901118D1 / 0DC910FA) and the two blurs
// (7C650AA1 / 694A6D14) that follow read only those maps, so in a frame where EVERY map was skipped they would recompute last frame's
// pages from the same inputs. They are skipped too (draws and copy): the pages keep last frame's image. No static layer (read_K.md section 1).
// Any doubt = draw (mixed frames, a pass not drawn the frame before, an unexpected size, a copy that also clears is kept).
// State is file statics (one chain; a job running the chain would own this struct). Command thread only.
#pragma once

#include <cstdint>

namespace rex::graphics::vulkan {
namespace shadow_skip {

bool Wanted();  // the cvar
// A draw whose pixel shader is a chain pass: true = skip it. key = the NfRt0 key of its target; color_info / surface_info = the draw's
// (the target is matched by base / pitch in tiles / format, not by size: the second map's chain has the same shaders);
// maps_drawn / maps_skipped = the frame's wrestler maps so far (half-rate bookkeeping, counted at their copies); frame = shadow_frame_.
bool PreDraw(uint64_t ps_hash, uint32_t key, uint32_t color_info, uint32_t surface_info, uint64_t frame, uint32_t maps_drawn,
             uint32_t maps_skipped);
// A wrestler map's decision (its first draw): skipped = true puts the target in the skipped list, false takes it out.
void NoteMap(uint32_t key, bool skipped);
// A draw on `key` was not skipped: the target leaves the skipped list (a later pass on the same EDRAM surface is its own).
void NoteDrawn(uint32_t key);
// IssueCopy of a colour target (copy_src_select 0): true = the copy belongs to a skipped pass and is dropped (EVERY copy of the pass,
// until a draw on the target is not skipped or the frame ends). clears = the copy also clears (kept).
bool DropCopy(uint32_t key, bool clears);
// Session 95 (resolves, gpu_native_owned_strip_half): the light-cube strip passes (vs D261C3CC, ps 8479D8DA on a 384x64 target, 5 passes of 1 draw and 6
// copies in an entrance frame) on alternate frames: the draw is skipped and every copy of the pass is dropped (DropCopy above), the cubes keep last
// frame's faces. Independent of gpu_shadow_half_rate. Decided once per frame, at the first strip draw (never two skipped frames in a row).
bool StripWanted();
bool StripPreDraw(uint64_t ps_hash, uint32_t key, uint32_t color_info, uint32_t surface_info, uint64_t frame);
// Session 95 (resolves, gpu_native_owned_ring_half): the ring's shadow chain (ps D03E7D4C into a 512x256 target on base 0 pitch 7, then its two reductions
// (901118D1 on base 2040 pitch 2, 0DC910FA on 2047 pitch 1) and the blur 632189BA on base 0 pitch 7) on alternate frames, as a unit: the ring map is
// light-space and the ring does not move, the pages keep last frame's. Independent of gpu_shadow_half_rate. A dependent pass is skipped only in a frame where
// the ring map was skipped and the pass was drawn the frame before.
bool RingWanted();
bool RingPreDraw(uint64_t ps_hash, uint32_t key, uint32_t color_info, uint32_t surface_info, uint64_t frame);
// Session 95 (r7 contract 3, gpu_native_half_phase, default off): one counter (shadow_frame_) decides every half-rate pass. Shadow maps draw on even
// frames (stagger 0); the strip, the ring chain and the crowd pose passes are skipped only on EVEN frames (so they draw on odd ones), never two
// frames in a row. Without it the strip / ring alternate from their first draw and the poses use another counter, so their phase is fixed by their first draw (the poses already dropped on even frames; ring and strip landed on either parity per run).
bool HalfPhase();
// Census: what = 0 shadow map, 1 strip, 2 ring, 3 pose; one call per decision. Logged by parity every ~600 frames.
void NotePhase(int what, uint64_t frame, bool drawn);
void FrameEnd(uint64_t frame);  // per-frame reset + the ~600-frame summary line

}  // namespace shadow_skip
}  // namespace rex::graphics::vulkan
