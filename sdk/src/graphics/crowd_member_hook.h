// wwe2k14 patch (Session 92, M4f-B; gpu_d3d_owned_front bit 6, value 64): the crowd member loop's draw callback and the typed constant arrays the
// own-crowd functions read. New file, SDK-internal (not a header the game exe includes); no data member of any class.
//
// SceneCrowdAfterDraw (command_processor.cpp) draws a crowd run's members one by one: the old way writes the member's c76-c78 into the register file /
// the record image (WriteRegister), draws through IssueDrawLean / IssueDraw and puts c76-c78 back. While a TYPED token draw's item is current, the
// direct path (native/d3d_direct.cpp) installs a callback: the member's 12 words go into a copy of the item's typed state, the typed draw draws it,
// nothing is written to the register file or the image and nothing needs putting back. OwnCrowdDraw / AppendCrowdInstances (vulkan/command_processor.cpp)
// read their float constants, fetch words and surface word from the OwnedCrowdSource while one is installed.
#pragma once

#include <cstdint>

namespace rex::graphics {

// Returns true when the member (c76 = its 12 words c76-c78) was drawn by the callback; `ok` is then the draw's success.
using SceneCrowdMemberDraw = bool (*)(void* ctx, const uint32_t* c76, bool& ok);
void SceneCrowdSetMemberDraw(SceneCrowdMemberDraw fn, void* ctx);  // fn nullptr = the old way
bool SceneCrowdMemberDrawInstalled();

struct OwnedCrowdSource {
  uint32_t* vs_float = nullptr;        // 256 vec4: the vertex constants (the member's c76-c78 are written here by AppendCrowdInstances)
  const uint32_t* ps_float = nullptr;  // 256 vec4
  const uint32_t* fetch = nullptr;     // 32 x 6 words
  uint32_t color_info = 0;             // RB_COLOR_INFO (a log line only)
  const void* keys = nullptr;          // M4f-C: a VulkanPipelineCache::TypedKeyState of the draw (the key derivations under the cut read it, not the register source)
  uint32_t clip_cntl = 0, vte_cntl = 0, vport_xscale = 0;  // M4f-C: PA_CL_CLIP_CNTL / PA_CL_VTE_CNTL words and the viewport x scale bits (the crowd cull reads them)
};
void OwnedCrowdSourceSet(const OwnedCrowdSource* source);  // nullptr = the register source
const OwnedCrowdSource* OwnedCrowdSourceGet();

}  // namespace rex::graphics
