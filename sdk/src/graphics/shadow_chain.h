// wwe2k14 patch (Session 85, gpu_dynamic_shadows false): the shadow chain's state.
//
// File-level state, NOT members of CommandProcessor: the game exe compiles that class's inline accessors against the
// header, so a data member added there moves the members behind it (Session 85 review). One GPU command thread.
//
// What the chain is: the render surfaces (colour info + depth info, pitch left out) that drew a known dynamic shadow
// caster while priming (the wrestler / props shadow maps A and B, the circle shadows, the ring mask casters), with their
// own flat fills and blurs. Primed = run kPrime frames with the casters skipped (the maps' textures are written once as
// empty = lit); after that every draw and copy of those surfaces is skipped, so the textures keep that content.
// Never the chain: a surface that drew the rim-light pass (ps D03E7D4C), its blur (632189BA) or the copy quad (7ABDFEFB).
#pragma once

#include <cstdint>
#include <vector>

namespace wwe_shadow_chain {

constexpr uint32_t kPrime = 8;          // frames run with casters skipped before the chain is skipped
constexpr size_t kMaxDests = 64;        // known copy destinations; more = fail safe (never skip)
constexpr size_t kMaxKeys = 16;

struct State {
  bool off = false;              // gpu_dynamic_shadows false, latched at the swap (a toggle never acts inside a frame)
  bool skip = false;             // this frame: the chain's draws and copies are skipped
  bool disabled = false;         // too many destinations / surfaces: normal rendering until the chain goes quiet
  bool reprime_pending = false;  // a copy to an unknown destination was skipped: prime again from the next frame
  uint32_t primed = 0, casters = 0, seen = 0;  // consecutive priming frames; this frame's caster / chain draws
  std::vector<uint64_t> keys;      // chain surfaces
  std::vector<uint64_t> excluded;  // surfaces that drew the rim pass / its blur / the copy quad
  std::vector<uint32_t> dests;     // copy destinations (RB_COPY_DEST_BASE) seen while priming
  uint64_t draws_skipped = 0, copies_skipped = 0, reprimes = 0, frames_skipped = 0, frames_primed = 0;
};

inline State g;

}  // namespace wwe_shadow_chain

// (Defined in vulkan/command_processor.cpp, where the cvar lives; called by the swap in command_processor.cpp.)
void ShadowChainFrameEnd(uint64_t frame);
