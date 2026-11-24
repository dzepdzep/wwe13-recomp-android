// wwe2k14 patch (Session 95, Lane B2 of out\s95\android_structural.md; gpu_native_vertex_input): the attribute draws. A typed draw whose vertices and indices pass section 3.2's identity reads its
// vertices as vertex attributes, with the shared memory buffer (or the inline ring) bound directly as vertex buffers and its DMA indices from the index pool: the vertex stage reads no storage buffer
// and selects no storage window. D3dDirectAccess::BuildVertexInputPlan (owned_front_own.cpp) decides per draw, before UpdateBindings; every refusal keeps the draw exactly as it was and is counted by
// name in the [vinput] line. A draw whose attribute pipeline is still being made is dropped like any draw with a pending pipeline: once decided, an attribute draw has selected no window, so there is
// no plain fallback behind the claim.
// State: this object is the command thread's (a job that planned a draw would own a copy of the latched switches and its own counters; the form table would be shared, filled ahead of the job).
#pragma once

#include <cstdint>
#include <unordered_map>

#include <rex/graphics/pipeline/shader/vertex_input_layout.h>
#include <rex/logging/api.h>
#include <rex/ui/vulkan/api.h>

#include "index_pool.h"

namespace rex::graphics::vulkan::native {

// Why a draw stayed on storage, in the order section 3.2 checks (each failure counted by name).
enum ViRefusal : uint32_t {
  kViNotTyped,       // the old lean path, or a draw continuing a claimed base
  kViNotOwnedOnly,   // not an owned-only frame
  kViCrowd,          // a crowd member or a run's first member; with gpu_native_crowd_input, a run own crowd drew on its storage form (why: crowd_why)
  kViMerge,          // an instanced (merged constants) base
  kViXfb,            // transform feedback
  kViFastClear,
  kViInstanceWorld,
  kViViewBlocks,
  kViVsExpand,       // the vertex shader expands points / rectangles
  kViMemexport,
  kViPsFetch,        // the pixel shader fetches vertices
  kViLineLoop,       // a line loop's closing index
  kViTranslatedOff,  // a translated shader at level 1
  kViNoForm,         // no _vi module, or the translated shader's form is not eligible
  kViVariantRefused, // the variant could not be made (translation or pipeline)
  kViAttributes,     // over the attribute limit
  kViFetchType,
  kViFetchSize,
  kViFetchEndian,
  kViFetchRange,
  kViPoolNone,       // the pool could not be made
  kViPoolFull,
  kViPoolTooBig,
  kViGpuWritten,     // the index range holds GPU-written data
  kViIndexRange,     // the pool could not read the index range
  kViIdentity,       // an index plus the base leaves the VGT clamp, 24 bits or goes below 0
  kViResetValues,    // reset values in the data under gpu_native_index_no_restart, and not a stitched strip (gpu_native_index_stitch off, or not a triangle strip)
  kViConverted,      // kHostConverted indices
  kViBuiltinDma,     // kHostBuiltinForDMA indices
  kViNoIndexIdentity,  // a draw without guest indices leaves the clamp
  kViCrowdMesh,      // an own crowd run's indices reach past its mesh (only in crowd_why)
  kViRefusalCount,
};

// One translation's attribute form (by plain module, modification and ext; made once, cleared with the caches). The pipeline is made by the claim, not here.
struct ViForm {
  VkShaderModule module = VK_NULL_HANDLE;
  uint32_t vi = 0;  // the pipeline key's: one byte per binding, the stride in dwords
  uint32_t binding_count = 0;
  uint8_t fetch_constant[kViMaxBindings] = {};
  bool translated = false;
  bool failed = false;  // no form: never retried
  ViRefusal refusal = kViNoForm;
};

// What BuildVertexInputPlan decided for one draw.
struct VertexInputPlan {
  bool ok = false;
  ViForm* form = nullptr;
  VkBuffer vb[kViMaxBindings] = {};
  VkDeviceSize vb_offsets[kViMaxBindings] = {};
  // Session 95 (P7, gpu_native_crowd_input): a run's first member held for own crowd: recorded for its state only, no draw call, so it reads no vertex: no storage window, no vertex range requested.
  // Its indices are planned for own crowd (NativeRenderer::OwnCrowdNoteIndex); `ok` stays false, the state stays the plain draw's.
  bool crowd_state = false;
  bool pool_index = false;  // DMA indices: from the pool at index_offset (else the draw's own: none or built-in)
  VkDeviceSize index_offset = 0;
  uint32_t index_count = 0;  // (P2b) a stitched strip: the elements to draw instead of the draw's own count (0 = the draw's own)
  int32_t first_vertex = 0;  // vertexOffset of an indexed draw, firstVertex of the others
  // Session 95 (r3cmd, gpu_d3d_draw_record): what the pool answered, the check of a record's plan.
  uint32_t index_min = 0, index_max = 0;
  bool index_has_reset = false, index_stitched = false;
};

class VertexInputState {
 public:
  VertexInputState() { rex::RegisterQuitFlush(&quit_req_, &quit_ack_); }
  ~VertexInputState() { rex::UnregisterQuitFlush(&quit_req_); }
  VertexInputState(const VertexInputState&) = delete;
  VertexInputState& operator=(const VertexInputState&) = delete;
  struct Counters {
    uint64_t ours = 0, translated = 0, ring = 0;  // attribute draws drawn (ring: whatever the shader)
    uint64_t pending = 0;                         // attribute draws dropped: pipeline still being made
    uint64_t fell_back = 0;                       // planned attribute draws handed back to the typed fallback (in neither total)
    uint64_t uploaded = 0;                        // attribute draws with a range uploaded in this submission
    uint64_t with_reset = 0;                      // DMA draws whose indices hold a reset value, taken or refused
    uint64_t stitched = 0, stitched_in = 0, stitched_out = 0;  // (P2b) attribute draws drawn from a stitched strip; their guest elements and the elements drawn
    uint64_t refused[kViRefusalCount] = {};
    // Session 95 (P7, gpu_native_crowd_input): first members recorded for state only (no vertex read at all); own crowd's runs on attributes (and their members); crowd member draws on attributes
    // (also in ours / translated); why a run stayed on own crowd's storage form (the run is in refused[kViCrowd]).
    uint64_t crowd_state = 0, crowd_runs = 0, crowd_run_members = 0, crowd_member_draws = 0;
    uint64_t crowd_why[kViRefusalCount] = {};
  };
  Counters& counters() { return counters_; }
  // Latched once a frame.
  bool no_restart() const { return no_restart_; }
  bool stitch() const { return stitch_; }
  bool index_verify() const { return index_verify_; }
  bool crowd_input() const { return crowd_input_; }  // Session 95 (P7): gpu_native_crowd_input

  // This translation's form; *fresh = it was just added, the caller fills it.
  ViForm& Form(VkShaderModule module, uint64_t modification, uint32_t ext, bool* fresh);
  void ClearForms() { forms_.clear(); }

  // Once a frame, before the pool's own FrameEnd (the same 600-frame windows): the latches, and the [vinput] line of a finished window. `window_changes` = the storage window's running count.
  void FrameTick(uint64_t frame, IndexPool& pool, uint64_t window_changes);

 private:
  std::atomic<bool> quit_req_{false}, quit_ack_{false};  // (Session 95) rex::RegisterQuitFlush: the quit closes the last [vinput] window
  Counters counters_;
  std::unordered_map<uint64_t, ViForm> forms_;
  bool no_restart_ = false, index_verify_ = false, stitch_ = false, crowd_input_ = false;
  uint64_t frame_ = UINT64_MAX, window_first_frame_ = 0;
  uint64_t window_changes_last_ = 0, window_changes_ = 0;
  uint64_t callback_calls_last_ = 0;
};

}  // namespace rex::graphics::vulkan::native
