// wwe2k14 patch (Session 87 F4 / Session 92 M4f-A, M4f-C): the friend of VulkanCommandProcessor shared by native/d3d_direct.cpp (the old record-fed lean
// path) and native/owned_front_own.cpp (the typed draw: TypedDraw, DrawFromBase, the cut's executor entries). Command processor thread only.
// SDK-internal header (not included by the game exe).
#pragma once

#include <rex/graphics/vulkan/native/native_renderer.h>

#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/d3d_direct.h>
#include <rex/graphics/d3d_feed.h>
#include <rex/graphics/registers.h>

#include <array>
#include <cstdint>

#include "d3d_feed.h"
#include "owned_record.h"
#include "owned_front_draw.h"
#include "vertex_input.h"
#include "draw_record.h"  // (Session 95, r3cmd)
#include "index_pool.h"  // (Session 95: gpu_d3d_draw_slot)
#include "../../crowd_member_hook.h"

namespace rex::graphics::vulkan {

// The census: relaxed counters on the command thread, one line per kPrintEvery draw items.
struct DirectCensus {
  uint64_t items = 0;       // DRAW items offered
  uint64_t taken = 0;       // drawn by the lean variant (new template + patch)
  uint64_t repeats = 0;     // ... of those, by the repeat fast path (vertex float constants only)
  uint64_t to_stock_lean = 0;  // handed to the stock IssueDrawLean (a special mode)
  uint64_t to_full = 0;     // handed to IssueDraw (the template did not hold)
  uint64_t fallback_mode = 0;     // the draw packet path (anatomy / extents / det skip / no native renderer)
  uint64_t fallback_source = 0;   // the draw packet path (not a plain DMA draw)
  uint64_t same_values = 0;       // render state re-sent with the template's values (kept the template)
  uint64_t restored = 0;          // template restored from the multi-base ring
  uint64_t repeat_refused[24] = {};  // RepeatDrawFast refusals by VulkanCommandProcessor::repeat_refusal_ code (0 = not set)
  uint64_t dirty_misses = 0;      // "state written" handbacks after a failed restore
  uint64_t dirty_outside = 0;     // ... of those, with a register outside 0x2000-0x23FF or a packet written
  uint64_t diff_regs[0x400] = {}; // ... and for each render-state register (index - 0x2000) how often it differed from the base's
  uint64_t cross_restores = 0;   // restores of a base of an earlier frame (gpu_native_lean_state_cache)
  uint64_t cross_claimed = 0;     // ... of those, claimed again and drawn
  uint64_t cache_verify_handbacks = 0;  // ... of those, handed to the full path to be compared (gpu_native_lean_state_cache_verify)
  uint64_t restored_for_shaders = 0;  // ... of those, restored because the shaders / primitive differed from the base's
  uint64_t restore_off = 0;       // restore not tried: gpu_native_lean_multi_base is off
  uint64_t restore_outside = 0;   // restore not tried: a register outside 0x2000-0x23FF or a state packet was written
  uint64_t restore_miss = 0;      // restore tried, no base of this frame has this state and these shaders
  uint64_t restore_ring_full = 0; // ... of the misses, seen with all 16 slots holding this frame's bases
  uint64_t reasons[16] = {};  // by VulkanCommandProcessor::LeanReason (11 values)
  uint64_t range_hits = 0;        // vertex ranges skipped by the epoch stamp
  uint64_t range_requests = 0;    // vertex ranges requested for real
  uint64_t range_mismatches = 0;  // verify: a stamp hit whose range was not up to date
  uint32_t mismatch_logged = 0;
  uint64_t next_print = 4096;
};

// One stamp a vertex fetch slot (the stock indexing: 0..95): the range requested successfully, and the page-validity
// epoch read before that request.
struct RangeStamp {
  uint32_t address = UINT32_MAX;  // dwords, as the fetch constant holds them
  uint32_t size = UINT32_MAX;
  uint64_t epoch = UINT64_MAX;
};
inline DirectCensus g_direct_census;
inline std::array<RangeStamp, 96> g_direct_stamps;

// The friend of VulkanCommandProcessor (command_processor.h): the direct handler's way to the draw state.
struct D3dDirectAccess {
  using VCP = VulkanCommandProcessor;
  enum class Result { kTaken, kPass, kFull };

  static bool DrawItem(CommandProcessor& base, rex::memory::RingBuffer* reader);
  // own: the record-fed variant (Session 87, stage 3) - the derivations read cp.DrawRegs() (the record's register image)
  // and Process is fed from `item`; false = F4's draw, reading the register file.
  static Result Lean(VCP& cp, xenos::PrimitiveType prim_type, uint32_t index_count,
                     VCP::IndexBufferInfo* index_buffer_info, bool major_mode_explicit, bool& succeeded, bool own,
                     const D3dFeedItem& item);
  static void Print(VCP& cp);

  // (Session 92, M4f-A) Where DrawFromBase takes the draw's own inputs from. The old lean path leaves everything at the defaults but `own` / `dirty` /
  // `input` / the three surface words (the register image); a typed draw (OwnedFrontDrawItem's) sets `typed` and the typed state's arrays.
  struct DrawSources {
    bool own = false;       // old path: the record-fed variant
    bool typed = false;     // the typed memo path
    uint32_t dirty = 0;     // repeat_dirty_ as it was (what a refusal gives back)
    const PrimitiveProcessor::DrawInput* input = nullptr;  // Process's input; nullptr = from the registers
    const uint32_t* fetch = nullptr;                        // 32 x 6 fetch words (textures, samplers, vertex ranges); nullptr = the register image
    const VCP::ConstSources* consts = nullptr;              // UpdateBindings' arrays; nullptr = the register image
    void (*sysconst)(VCP&, void*, const PrimitiveProcessor::ProcessingResult&) = nullptr;  // nullptr = UpdateSystemConstantValues
    void* ctx = nullptr;
    uint32_t rb_surface_info = 0, rb_color_info = 0, rb_depth_info = 0;  // the crash record's words
    VCP::LeanReason* why = nullptr;  // typed: out, the reason of a kFull
    bool* cover_wanted = nullptr;    // typed: out, the refusal was a wanted cover / bounds rectangle
    const uint32_t* mx_floats = nullptr;  // M4f-B: the 512 float constants (typed banks), for the memexport stream constants; nullptr = no typed memexport
    uint32_t mx_vs_base = 0, mx_ps_base = 0;
    bool crowd_member = false;  // M4f-B: a crowd run's member drawn typed: the draw is kept as the base the next members merge into (AppendCrowdInstances)
    const OwnedRecordState* cover_state = nullptr;  // M4f-B (latch bit 6): the typed cover / bounds inputs come from this state; nullptr = a wanted rectangle is a fallback
    const PrimitiveProcessor::ProcessingResult* primitive = nullptr;  // Session 93 (gpu_d3d_typed_memo_reuse): Process's result, made once by OwnedFrontPrepare; nullptr = Process runs here
    bool record_repeat = false;  // Session 93 (gpu_d3d_typed_repeat): a typed draw keeps its Draw as the repeat base (RepeatNoteRecorded) instead of ending the chain
    VkDescriptorSet inline_ring_set = VK_NULL_HANDLE;  // Session 95 (item P): the draw reads its vertices through this set, not the shared memory's: nothing of it is requested
    const uint8_t* inline_ring_memory = nullptr;       // (the ring's mapping, for a cover / bounds rectangle)
    uint32_t vs_const_word = 0;  // Session 95 (the diagnostic run): SQ_VS_CONST of a typed draw
    DrawRec* rec = nullptr;      // Session 95 (r3cmd, gpu_d3d_draw_record): this draw's record slot
    bool rec_hit = false;        // the record is complete and its preparation was used
    bool rec_verify = false;     // the record is complete, the draw went the full way, compare
    const OwnedRecordState* rec_state = nullptr;
    OwnedTypedPrep* rec_prep = nullptr;
    bool rec_no_bind = false;    // the typed draw's descriptor sets are bound by OwnedRecord alone
    bool slot_on = false;        // Session 95 (gpu_d3d_draw_slot): a draw that is not a hit leaves its words in its record's slot
    bool slot_hit = false;       // the slot's words equal the draw's (rec_hit is set too): the sampler list, the plan's pool answer and the system constants come from the slot's side
    bool slot_verify = false;    // the slot's words equal the draw's, the draw went the full way, compare
    uint32_t slot_settings = 0;  // the setting in the generation word
  };
  static Result DrawFromBase(VCP& cp, const VCP::LeanState& L, const native::Draw& base, VulkanShader* vertex_shader,
                             xenos::PrimitiveType prim_type, uint32_t index_count, VCP::IndexBufferInfo* index_buffer_info,
                             bool major_mode_explicit, bool& succeeded, const DrawSources& src, bool restored, bool cross_frame,
                             uint64_t repeat_epoch);
  // The typed token draw (gpu_d3d_owned_front bit 5): true = drawn (or failed in the backend: `succeeded`), false = the old path draws it.
  // Session 95 (out\s95\android_structural.md section 3.2): whether this draw reads its vertices as attributes; every refusal is counted by name. Before UpdateBindings; reads system_constants_.
  static bool RecordVertexInputPlan(VCP& cp, const DrawRec& r, const PrimitiveProcessor::ProcessingResult& ppr, native::VertexInputPlan& plan, native::IndexPool::HeldIndex* held);  // (Session 95, r3cmd; held: gpu_d3d_draw_slot)
  static void BuildVertexInputPlan(VCP& cp, const native::Draw& base, VulkanShader* vertex_shader, bool pixel_shader_vertex_fetches, const PrimitiveProcessor::ProcessingResult& ppr,
                                   const DrawSources& src, bool cross_frame, bool mx_possible, native::VertexInputPlan& plan);
  static bool TypedDraw(VCP& cp, const OwnedRecordState* state, xenos::PrimitiveType prim_type, uint32_t index_count,
                        VCP::IndexBufferInfo* index_buffer_info, bool major_mode_explicit, bool& succeeded, bool crowd_blocked, bool crowd_member = false);
  // M4f-B: a crowd run's member (or its first member, drawn by the crowd code's guard) through the typed draw; the context lives in DrawItem's frame.
  struct CrowdHook {
    VCP* cp = nullptr;
    xenos::PrimitiveType prim_type{};
    uint32_t index_count = 0;
    VCP::IndexBufferInfo* index_buffer_info = nullptr;
    bool major_mode_explicit = false;
    OwnedCrowdSource source;
    uint32_t vs_copy[1024] = {};  // a private copy of the vertex float bank: AppendCrowdInstances writes the members' c76-c78 into it (never a snapshot)
    uint32_t drawn = 0, declined = 0;
    bool cut = false;  // M4f-C: a declined member is DROPPED (counted), never drawn the old way
    bool repeat_base = false;  // Session 95 (gpu_d3d_wall bit 128): the last member went through the typed draw: it is the repeat base of the next one
  };
  static bool CrowdMember(void* ctx, const uint32_t* c76, bool& ok);
  // M4f-B: the inline draw (0x7E) through the typed draw (called by CommandProcessor::ExecuteD3dInlineDraw); false = the old path draws it.
  static bool InlineDraw(VCP& cp, uint32_t viz_query, uint32_t initiator, bool& drawn);
  // M4f-C (the cut): the executor's entry points, on the LIVE typed state (owned_front_own.cpp). Final: true = drawn, or dropped and counted by reason.
  static bool CutDraw(CommandProcessor& cp, const OwnedRecordState& live, uint32_t item_index);
  static bool CutCullOurs(VCP& cp, const OwnedRecordState& s, bool crowd_member);  // Session 93 (gpu_d3d_cull_ours): true = an off-screen draw of ours, not drawn
  static void PostLiteClearRevive(VCP& cp, const uint32_t* sem, uint32_t count);  // Session 95: a clear of a dead post target revives it
  static const char* CutPreDrawSkip(VCP& cp, const OwnedRecordState& s);  // M4f-C: hide HUD / shadow chain / half-rate shadows / crowd density from typed words; nullptr = draw
  static bool CutInline(CommandProcessor& cp, const OwnedRecordState& live, const uint32_t* record, uint32_t dwords);
  // Session 95 (item P): the record's vertices onto the host vertex ring; false = this draw stays on guest memory (counted by reason). `address` = its offset in the ring.
  static bool InlineRingStage(VCP& cp, const uint32_t* record, uint32_t dwords, uint32_t& address);
};

}  // namespace rex::graphics::vulkan
