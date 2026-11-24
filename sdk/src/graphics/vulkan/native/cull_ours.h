// wwe2k14 patch (Session 93, idea 13: docs\native\d3d\S93_MIRROR_DESIGN.md section 4): the draw cut of OUR arena draws the camera cannot see (gpu_d3d_cull_ours,
// default false). The typed draw (D3dDirectAccess::TypedDraw, under the cut) asks Evaluate for every draw item; true = the draw is skipped (counted as a cut skip;
// nothing of it is drawn, requested or recorded). The test is the census' (draw_bounds.h): the mesh's object-space box through the model matrix c76-c78 and the
// view-projection c56-c59, a 1% margin; only for the shaders whose position is exactly that (shader_swap::VertexBoxCullable). Everything that makes a skip unsafe
// is a counted refusal: see the list in cull_ours.cpp. File statics only. Off (the cvar) = nothing here runs.
#pragma once

#include <cstdint>

namespace rex::memory {
class Memory;
}

namespace rex::graphics::vulkan {

struct OwnedRecordState;

namespace cull_ours {

struct Request {
  rex::memory::Memory* mem = nullptr;
  const OwnedRecordState* s = nullptr;
  bool crowd_member = false;
  bool occlusion_query = false;  // an occlusion query is active
  bool truth_pass_open = false;  // the renderer's pass is open (a skipped first draw must not decide a pass's structure)
  bool shaders_known = false;    // the active vertex + pixel shaders are the draw's, analysed, neither writes memory (memexport), and the vertex shader has the one 8-dword binding in the registered fetch constant
  // The guard only the command processor can answer; asked only for a draw the box test would cull: the host clip space of the draw is the guest's (x, y: scale 1, offset ~0).
  // Fills the draw's host viewport transform {scale x, scale y, offset x, offset y}; false = refuse (empty viewport, wide / tall factor).
  bool (*viewport)(void* ctx, float* host_xy) = nullptr;
  void* viewport_ctx = nullptr;
};

// Whether the vertex shader is one of ours with a position the box test knows (shader_swap::VertexBoxCullable); cached per shader. CullableFetch = its stream's fetch constant.
bool CullableShader(uint64_t vs_hash);
uint32_t CullableFetch(uint64_t vs_hash);
// true = skip the draw. Always call NoteDrawn(s) for a draw that is not skipped (it carries the pass identity the next draw's guard compares with).
bool Evaluate(const Request& r);
void NoteDrawn(const OwnedRecordState& s);
// Anything that is not a drawn typed draw item (a refused or old-path draw, an inline draw, a clear / resolve, the end of the renderer's pass) ends "the pass the previous
// drawn draw continues": the next draw is drawn.
void NoteBreak();
// One guest frame boundary; prints the window line every 600 of them.
void FrameEnd();

}  // namespace cull_ours
}  // namespace rex::graphics::vulkan
