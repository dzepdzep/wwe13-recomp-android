// wwe2k14 patch (Session 95, r3cmd, gpu_d3d_draw_record): the per-draw record. The game draws nearly the same draw list frame to frame: a census of typed draws
// (gpu_d3d_plan_census) found 91% (Bryan's entrance) to 98% (the match) with every input but the float constants the same as a draw of an earlier frame, and almost none with the same
// vertex floats. So a record holds what a typed draw derives from everything BUT the float constants, under a key of those inputs: the render state, the shader pair, the draw's own words,
// the fetch slots its shaders use, the bool / loop words. A draw with a record skips the derivation; the constants, the shared memory and texture requests and the owned pass step run as
// ever. Each part of the record has its own validity check, and a part that does not hold takes the full path for that draw. Command thread only.
#pragma once

#include <rex/graphics/vulkan/native/native_renderer.h>
#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include <rex/graphics/primitive_processor.h>

#include "vertex_input.h"

namespace rex::graphics::vulkan {

struct OwnedRecordState;
struct OwnedTypedPrep;
class VulkanCommandProcessor;

constexpr size_t kDrawRecPrepHead = 768, kDrawRecBiHead = 768;
extern uint64_t g_draw_rec_generation;  // moves when what a record holds handles of goes (pipelines, surfaces)

struct DrawRec {
  uint64_t key = 0, frame = 0;
  uint64_t vs = 0, ps = 0;  // (a second look at a key match)
  // wwe2k14 patch (Session 95, r6tex, gpu_tex_bind_memo bit 8): the record of the draw after this one last time. The draw list repeats frame to frame, and a record is kilobytes read once a frame: it is
  // out of the cache by the time its draw comes round (here on the Surface's 6 MB L3 with the GPU). Only ever an address to fetch ahead: never read through.
  const void* next_hint = nullptr;
  uint32_t dma_base = 0, initiator = 0;
  bool complete = false;
  // Prepare: the preparation (OwnedTypedPrep's state part), the externals and shaders it holds under, Build's Draw with the primitive pass applied, the primitive pass and the index span it requests.
  // (Session 95, gpu_d3d_draw_slot: one image, the bind inputs at their own offset in OwnedTypedPrep, so a slot hit uses it where it lies; only the head of an OwnedTypedPrep: the lists after it are not here.)
  alignas(16) uint8_t prep_image[kDrawRecPrepHead + kDrawRecBiHead];
  uint64_t ext = 0, gen = 0, generation = 0;
  const void* active_vs = nullptr;
  const void* active_ps = nullptr;
  native::Draw draw;
  PrimitiveProcessor::ProcessingResult primitive{};
  bool s32 = false, index_expansion = false;
  uint32_t index_base = 0, index_bytes = 0;
  // The system constants the draw's build left, and the one texture-cache input they read.
  bool sys_ok = false;
  uint32_t scaled = 0;
  SpirvShaderTranslator::SystemConstants sys{};
  // The vertex input plan (attribute draws).
  bool plan_ok = false;
  native::VertexInputPlan plan;
  // The claim's result (OwnedClaim).
  bool claim_ok = false;
  VkPipeline pipeline = VK_NULL_HANDLE;
  bool truth = false, owned = false, post = false;
  int16_t truth_color = -1, truth_depth = -1;
  uint8_t truth_class = 1, truth_sx = 1, truth_sy = 1;
};

struct DrawRecStats {
  uint64_t asked = 0, hits = 0, misses = 0, stale_prep = 0, unkeyed = 0, stale_ext = 0, sys_fallback = 0, plan_fallback = 0, claim_fallback = 0, stored = 0, refused = 0;
  uint64_t why[13] = {};
  uint64_t verify_compared = 0, verify_draw = 0, verify_sys = 0, verify_claim = 0, verify_plan = 0, verify_primitive = 0, verify_index_offset = 0;
};
extern DrawRecStats g_draw_rec_stats;

// The record slot of this draw's inputs: hit = complete and usable; nullptr = this draw has no key (a shader not analysed yet).
DrawRec* DrawRecFind(VulkanCommandProcessor& cp, const OwnedRecordState& s, bool& hit);
// The preparation from the record: nullptr = its validity failed, the full preparation runs (the slot is filled again).
OwnedTypedPrep* DrawRecPrep(VulkanCommandProcessor& cp, const OwnedRecordState& s, DrawRec& r);
// DrawFromBase's system constants step from the record: false = the full build runs.
bool DrawRecSystemConstants(VulkanCommandProcessor& cp, void* ctx, const DrawRec& r);
// After the draw went the full way: fill the slot (or, with verify, compare a complete one with what the full way gave).
void DrawRecFill(VulkanCommandProcessor& cp, DrawRec& r, const OwnedRecordState& s, const OwnedTypedPrep& p, const native::Draw& claimed, const native::VertexInputPlan& plan, bool plan_kept);
void DrawRecVerify(const DrawRec& r, VulkanCommandProcessor& cp, const OwnedTypedPrep& p, const native::Draw& claimed, const native::VertexInputPlan& plan);
void DrawRecTick(uint64_t frame);

}  // namespace rex::graphics::vulkan
