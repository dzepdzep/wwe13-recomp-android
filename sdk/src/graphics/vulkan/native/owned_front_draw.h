// wwe2k14 patch (Session 92, M4b; docs\native\d3d\S92_M3_M5_DESIGN.md section 2, contract S92_M4B_CONTRACT.md): the draw description built
// from the record decoder's typed state, as a SHADOW.
//
// At each claim the old path makes (the full description, a lean draw, a repeat), OwnedFrontShadowDraw builds the same native::Draw from the
// typed state of the record's draw item (owned_record.h; the draw handler d3d_direct.cpp brackets the item with OwnedRecordDrawBegin / End)
// and compares it with the one the old path built, field by field. The old path still draws. Latch: gpu_d3d_owned_front bit 1 (with bit 0,
// the decoder).
#pragma once

#include <cstdint>

#include <rex/graphics/primitive_processor.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/vulkan/native/native_renderer.h>
#include <rex/graphics/vulkan/command_processor.h>

#include "owned_front_bind.h"

namespace rex::graphics::vulkan {

class VulkanCommandProcessor;

enum OwnedFrontClaimKind : uint32_t { kOwnedFrontFull = 0, kOwnedFrontLean = 1, kOwnedFrontRepeat = 2, kOwnedFrontKinds = 3 };

void OwnedFrontShadowDraw(VulkanCommandProcessor& cp, const native::Draw& old_draw, uint32_t kind);

// M4d (gpu_d3d_owned_front bit 3): the crowd shadow. The draw handler calls it for a draw item with the old path's crowd inputs (the member constants
// c76-c78 it read, its packet id, whether PreTake was asked and what it answered); the typed state's are compared. OwnedFrontShadowDraw also reads
// D3dFeedCrowd().member_c76 to build a crowd member's draw from the typed state.
// (`ask` = the handler's SceneCrowdMatchLocal, a captureless lambda in its friend scope: the processor's member is protected.)
void OwnedFrontRepeatBreak();  // M4d: ends the typed repeat chain (called by the decoder for items that are not draws)
using OwnedFrontCrowdAsk = bool (*)(VulkanCommandProcessor&, const uint32_t*);
void OwnedFrontCrowdItem(VulkanCommandProcessor& cp, const uint32_t* old_c76, uint32_t old_id, bool pre_called, bool pre_result, OwnedFrontCrowdAsk ask);

// ---- M4f-A (gpu_d3d_owned_front bit 5, value 32): the typed state DRAWS the token draw items.
//
// OwnedFrontPrepare builds, from the paired snapshot, what the lean body (d3d_direct.cpp DrawFromBase) takes from a base: the native::Draw (modules
// really created, the geometry shader made, the pipeline layout made), the fields of the lean base, and the inputs of this draw (the typed constant
// banks and fetch words, the draw's primitive input, the system constants' inputs). Every refusal is a counted fallback reason (the old path draws).
struct OwnedTypedPrep {
  native::Draw* draw = nullptr;  // storage owned by the .cpp (valid until the next Prepare)
  // the fields of the lean base (VulkanCommandProcessor::LeanState)
  VulkanShader* vertex_shader = nullptr;
  VulkanShader* pixel_shader = nullptr;
  bool polygonal = false;
  Shader::HostVertexShaderType host_vertex_shader_type = Shader::HostVertexShaderType::kVertex;
  uint32_t used_texture_mask = 0;
  reg::RB_DEPTHCONTROL depth_control{};
  uint32_t color_mask = 0;
  draw_util::ViewportInfo viewport_info{};
  const void* layout = nullptr;  // VulkanCommandProcessor::PipelineLayout
  uint32_t shared_memory_access_stages = 0;
  uint32_t merge_stride = 0, float_count = 0;
  bool pixel_shader_vertex_fetches = false;
  bool host_primitive_reset_enabled = false;
  xenos::PrimitiveType host_primitive_type = xenos::PrimitiveType::kNone;
  uint32_t vertex_endian_key = 0, pixel_spec_key = 0, vertex_mul_key = 0, pixel_ext = 0, vertex_ext = 0;
  // this draw's own inputs
  PrimitiveProcessor::DrawInput input{};
  const uint32_t* vs_float = nullptr;
  const uint32_t* ps_float = nullptr;
  const uint32_t* bool_loop = nullptr;
  const uint32_t* fetch = nullptr;
  uint32_t surface_info = 0, color_info = 0, depth_info = 0;
  OwnedBindInputs bi;  // the system constants' inputs (the typed block builder)
  const char* refusal = nullptr;  // set when OwnedFrontPrepare refuses: the reason name
};
// Returns the prepared draw (storage owned by the .cpp), or nullptr with p.refusal set.
OwnedTypedPrep* OwnedFrontPrepare(VulkanCommandProcessor& cp, const OwnedRecordState& s);
// DrawFromBase's system constants step for a typed draw (ctx = the OwnedTypedPrep).
void OwnedFrontSystemConstants(VulkanCommandProcessor& cp, void* ctx, const PrimitiveProcessor::ProcessingResult& primitive);
// Counters of the typed draws (window line `[d3d][front] M4f-A`).
enum OwnedTypedOutcome : uint32_t { kTypedItems, kTypedDrawn, kTypedFailed, kTypedNs, kTypedRestores, kTypedPlaceholder,
                                  kTypedCrowdMembers, kTypedCrowdDeclined, kTypedInline, kTypedInlineFallback, kTypedCover, kTypedBounds, kTypedMemexport, kTypedOutcomeCount };
void OwnedFrontTypedNote(uint32_t outcome, uint64_t n = 1);
void OwnedFrontTypedFallback(const char* reason);  // reason: a string literal (pointer-keyed)
void OwnedFrontTypedAfter();                       // after a typed draw: the shadows' repeat chain ends
const char* OwnedFrontPrepareRefusal();             // the last refusal of OwnedFrontPrepare (a Build skip name)
uint64_t OwnedFrontStateHashOf(const OwnedRecordState& s);  // (Session 95, r3cmd: StateHash)
uint64_t OwnedFrontNowNs();
extern uint64_t g_lean_bind_stat[14];  // Session 94 (gpu_d3d_typed_lean_bind): see owned_front_draw.cpp
void OwnedFrontTypedTick(VulkanCommandProcessor& cp);
// M4f-B: NativeCoverRect / NativeBoundsRect inputs from the typed state (see owned_front_draw.cpp).
void OwnedFrontCoverInputs(const OwnedRecordState& s, CoverInputs& in);
uint64_t OwnedFrontCoverTypedCount();
void OwnedFrontTypedActive(bool on);  // typed draw running: the shadows (OwnedFrontShadowDraw) do not run for it

// ---- Session 93 (gpu_d3d_typed_repeat): the typed repeat tracker. A typed draw whose Draw was recorded as the repeat base (RepeatNoteRecorded) leaves a key of what the
// old repeat needs unchanged (the decoder's epochs but the vertex floats, the shaders, the draw words, the externals, the frame / submission / jitter); the next draw
// item that matches it except in the vertex float constants is the old path's repeat (RepeatDrawFast, which instances it into the base's draw when it can).
enum OwnedRepeatOutcome : uint32_t { kRepeatTried, kRepeatDrawn, kRepeatRefused, kRepeatOutcomeCount };
extern uint64_t g_owned_repeat_base_serial;
void OwnedFrontRepeatBegin();                                                              // start of a typed draw: the recorded key moves to `previous`, nothing is recorded
bool OwnedFrontRepeatCandidate(VulkanCommandProcessor& cp, const OwnedRecordState& s);  // `s` is the next draw item and equals `previous` but in the vertex floats
void OwnedFrontRepeatRecord(VulkanCommandProcessor& cp, const OwnedRecordState& s);       // after the draw: this draw is the base the next one is compared with
void OwnedFrontRepeatNote(uint32_t outcome);

}  // namespace rex::graphics::vulkan
