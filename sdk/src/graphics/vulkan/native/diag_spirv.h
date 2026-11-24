// wwe2k14 patch (Session 95, the diagnostic run; out\s95\diag_design.md section 5, "Numbers"): the SPIR-V passes of the vertex truth.
//
// PatchPositionCapture copies a vertex module and changes only its tail: at each OpReturn of the entry point the final gl_Position is
// bitcast to uvec4 into a new output (location N) and gl_VertexIndex into a second one (location N + 1), and gl_Position is replaced by
// a pixel chosen from the vertex index, gl_PointSize = 1. All the module's maths stays as it is, so what comes out of location N is what
// the real shader computed, bit-exact (NaN and denormals survive), and it needs only the rasteriser: no stores from the vertex stage
// (Mali-G57 has no vertexPipelineStoresAndAtomics), no transform feedback.
//
// No SDK dependencies: tools\native\diag_spirv_test.cpp builds this file alone and runs it on modules from glslang through spirv-val.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rex::graphics::vulkan::diag {

// The pixel a vertex index is written to in the 256 x 256 capture target. A 32-bit index the guest stored big-endian reaches the
// shader byte-swapped (its low bits are the same for every vertex), so both halves are folded in. The id target says which index a
// pixel really holds (two indices may fold together: the later one wins, nothing is mixed).
constexpr uint32_t kCaptureSize = 256;
inline uint32_t CaptureSlot(uint32_t raw_index) { return (raw_index ^ (raw_index >> 16)) & 0xFFFFu; }

struct PositionPatch {
  std::vector<uint32_t> words;
  uint32_t location = 0;  // the position bits' output; the index is location + 1
};
// False (out untouched, why set) when the module cannot be patched: not a single vertex entry point, no gl_Position, no room under
// max_locations (a location is 4 components: maxVertexOutputComponents / 4, and the fragment side's limit).
bool PatchPositionCapture(const uint32_t* words, size_t count, uint32_t max_locations, PositionPatch& out, std::string* why);

// The fragment shaders of the capture passes (made here, our own code; nothing from the game).
// Numbers: input location N (flat uvec4) to colour target 0, N + 1 (flat uint) to colour target 1.
std::vector<uint32_t> CaptureFragmentModule(uint32_t location);
// Solo: every fragment white (a silhouette).
std::vector<uint32_t> SoloFragmentModule();

}  // namespace rex::graphics::vulkan::diag
