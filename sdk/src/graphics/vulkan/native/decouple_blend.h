// wwe2k14 patch (Session 94, idea 1 phase 1B; docs\native\d3d\S93_DECOUPLE_DESIGN.md sections 2, 3, 6): the draw match across two guest ticks and the blended
// vertex constants of view 1 (the in-between) (gpu_d3d_decouple bit 3). Two calls: DecoupleBlendDraw from the typed draw (once per typed draw item) and DecoupleBlendFrameEnd from
// the owned frame's end (once per guest tick). Reads the typed state, writes only its own statics (command thread only). Nothing here runs without the switch.
#pragma once

#include <cstdint>
#include <string>

namespace rex::memory {
class Memory;
}

namespace rex::graphics::vulkan {

struct OwnedRecordState;
class VulkanShader;

// The vertex float bank view 1 of this draw sees (1024 words, the typed bank's layout, valid until the next call), or nullptr: view 1 = view 0 (the picture) for this draw (nothing to
// blend, an unmatched draw with no camera replacement, a cut, or the blend is not active this tick). `vs` = the draw's vertex shader.
const uint32_t* DecoupleBlendDraw(rex::memory::Memory* memory, const OwnedRecordState& s, const VulkanShader* vs, bool crowd_member);

// Session 95 (F5): view 1's camera (c56-c59, 16 floats) for a draw that is not one of our vertex shaders (the own crowd), BY VALUE: the blend this tick decided for exactly this camera block
// (set by the draws of our arena shaders that carry it). False = view 1 sees the tick's own camera (not active, never decided, a camera cut). `vs_float` = the draw's 1024-word bank.
bool DecoupleBlendCamera(const uint32_t* vs_float, float out[16]);

// A guest tick ended: this tick's records become the previous tick's. Returns whether the NEXT tick's views differ (the blend is active: the game's frame word matched
// gpu_d3d_decouple_word when the tick began); the presenter shows two views per tick only then.
void DecoupleBlendFrameEnd();

// Whether the tick that just ended was blended (what the two-step present follows), and the 600-frame summary line (empty: not yet).
bool DecoupleBlendTickWasActive();
std::string DecoupleBlendSummary(uint64_t frames, uint32_t views, uint64_t presents, uint64_t two_step_ticks);

}  // namespace rex::graphics::vulkan
