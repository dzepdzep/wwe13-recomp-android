// wwe2k14 patch (Session 93, LOG-ONLY phase 0 of docs\native\d3d\S93_MIRROR_DESIGN.md (ideas 11 + 13) and S93_DECOUPLE_DESIGN.md):
// the census behind gpu_d3d_phase0_census. Two calls only: Phase0Draw from the typed draw (once per draw item the typed draw is given) and Phase0FrameEnd
// from the base class' SceneCrowdFrameEnd hook (every guest frame, also under the cut). It reads the typed state and guest memory, never writes either.
// File statics only (no data members in the SDK headers the game exe includes). Off (the cvar) = neither call does anything.
#pragma once

#include <cstdint>

namespace rex::memory {
class Memory;
}

namespace rex::graphics::vulkan {

struct OwnedRecordState;
class VulkanShader;

// `vs` = the vertex shader the draw uses (nullptr when unknown: the bindings are then not read); `crowd_member` = a crowd run's member draw (counted, not matched).
void Phase0Draw(rex::memory::Memory* memory, const OwnedRecordState& s, const VulkanShader* vs, bool crowd_member);
// One guest frame boundary; prints the window lines every 600 of them.
void Phase0FrameEnd();

}  // namespace rex::graphics::vulkan
