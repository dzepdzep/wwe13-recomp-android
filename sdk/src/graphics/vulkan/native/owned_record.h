// wwe2k14 patch (Session 92, M4a; docs\native\d3d\S92_M3_M5_DESIGN.md section 2): the record decoder's typed state.
//
// The record-fed front end of the owned frame reads the D3D layer's record in host order straight from the queue storage and
// decodes it into THIS state: typed fields (the `reg::` bit layouts are the data format of the game's words) plus the arrays our
// shaders read as they are (the VS / PS float banks, bool / loop, the 32 fetch constants). Delta semantics: a record carries the
// dirty blocks only, so the state persists across records.
//
// What this is NOT: a register file. There is no word array indexed by register number and no RegisterFile here; register numbers
// appear in one place only, the decoder's table in owned_record.cpp (register -> field). M4a runs it as a SHADOW (the existing path
// still draws and a compare at the end of each item counts the differences from the feed image); M4b..M4f read it to draw.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <rex/graphics/registers.h>

namespace rex::graphics {
class CommandProcessor;
}

namespace rex::graphics::vulkan {

struct OwnedRecordState {
  // ---- surfaces
  reg::RB_MODECONTROL modecontrol{};
  reg::RB_SURFACE_INFO surface_info{};
  reg::RB_COLOR_INFO color_info[4]{};
  reg::RB_DEPTH_INFO depth_info{};
  // ---- raster / clip / viewport / window / scissors
  reg::PA_SU_SC_MODE_CNTL sc_mode_cntl{};
  reg::PA_SU_VTX_CNTL su_vtx_cntl{};
  reg::PA_SU_POINT_SIZE point_size{};
  reg::PA_SU_POINT_MINMAX point_minmax{};
  float poly_offset[4]{};  // front scale, front offset, back scale, back offset
  reg::PA_CL_CLIP_CNTL clip_cntl{};
  reg::PA_CL_VTE_CNTL vte_cntl{};
  float vport[6]{};  // x scale, x offset, y scale, y offset, z scale, z offset
  float user_clip[24]{};  // 6 planes x (x, y, z, w)
  reg::PA_SC_WINDOW_OFFSET window_offset{};
  reg::PA_SC_WINDOW_SCISSOR_TL window_scissor_tl{};
  reg::PA_SC_WINDOW_SCISSOR_BR window_scissor_br{};
  reg::PA_SC_SCREEN_SCISSOR_TL screen_scissor_tl{};
  reg::PA_SC_SCREEN_SCISSOR_BR screen_scissor_br{};
  reg::PA_SC_VIZ_QUERY viz_query_cntl{};
  // ---- depth / stencil / colour / blend
  reg::RB_DEPTHCONTROL depth_control{};
  reg::RB_STENCILREFMASK stencil_front{};
  reg::RB_STENCILREFMASK stencil_back{};
  reg::RB_COLORCONTROL color_control{};
  float alpha_ref = 0;
  reg::RB_BLENDCONTROL blend[4]{};
  float blend_color[4]{};  // red, green, blue, alpha
  reg::RB_COLOR_MASK color_mask{};
  // ---- primitive / VGT
  reg::VGT_MIN_VTX_INDX min_vtx_indx{};
  reg::VGT_MAX_VTX_INDX max_vtx_indx{};
  reg::VGT_INDX_OFFSET indx_offset{};
  reg::VGT_MULTI_PRIM_IB_RESET_INDX reset_indx{};
  reg::VGT_OUTPUT_PATH_CNTL output_path_cntl{};
  reg::VGT_HOS_CNTL hos_cntl{};
  float hos_tess[2]{};  // max, min tessellation level
  // ---- the draw item's own words
  reg::VGT_DRAW_INITIATOR draw_initiator{};
  uint32_t dma_base = 0;
  reg::VGT_DMA_SIZE dma_size{};
  // ---- shader control
  reg::SQ_PROGRAM_CNTL program_cntl{};
  reg::SQ_CONTEXT_MISC context_misc{};
  reg::SQ_INTERPOLATOR_CNTL interpolator_cntl{};
  reg::SQ_VS_CONST vs_const{};
  reg::SQ_PS_CONST ps_const{};
  // ---- the arrays our shaders read as they are (xenos_vs.glsl layout): words, 4 per vec4 constant
  uint32_t vs_float[256 * 4]{};
  uint32_t ps_float[256 * 4]{};
  uint32_t bool_loop[8 + 32]{};  // bool constants (256 bits) then the 32 loop constants
  uint32_t fetch[32 * 6]{};      // the 32 fetch constants, 6 words each
  // ---- not registers: the draw item and the shader identities
  uint32_t draw_viz_query = 0;
  uint64_t crowd_id = 0;       // the draw's crowd identity: token sequence << 16 | item index (16 bits) (command_processor.cpp kD3dTokenMagic handler)
  uint32_t token_seq = 0;      // the current token's sequence number
  struct ShaderId {
    uint32_t address = 0;      // as the record carries it (bit 31 = guest virtual)
    uint32_t dwords = 0;
    uint64_t hash = 0;         // XXH3 of the microcode words, as the pipeline cache keys its shaders
  };
  ShaderId shader[2]{};        // 0 vertex, 1 pixel
  uint64_t bin_mask = 0;       // the tiling predication state (kNrRegs bin ops)
  uint64_t bin_select = 0;
  uint32_t banks_valid = 0;  // M4c: set in a snapshot taken with the constant banks (latch bit 2 at the time of the queue item)
  uint32_t crowd_member = 0;  // M4d: set in the state OwnedRecordCrowdMemberState derives (a crowd member's draw: c76-c78 are the member's)
  uint32_t inline_draw = 0;   // M4e: set in the snapshot of an inline (0x7E) draw
  uint32_t inline_ring = 0;   // Session 95 (item P): set around an inline draw whose vertices are on the host vertex ring (slot 31's address is an offset in the ring)
};

// The decoder's state (command thread only). M4b onwards read it.
const OwnedRecordState& OwnedRecordStateNow();

// M4b: the typed state AS OF each DRAW item of a queue item (a token holds several items; the decoder runs ahead of the old path, which
// executes them one by one). Taken at every token DRAW item when gpu_d3d_owned_front bit 1 is set (all but the shader constant banks, which M4c reads: with
// bit 2 of the latch the banks are copied too), in order, at most 512 per queue item. The old path's draw handler (d3d_direct.cpp D3dDirectDrawItem) pairs them
// SEQUENTIALLY: OwnedRecordDrawAdvance() at the very top of every draw item it is given (killed, fallback or drawn), so draw N of the
// queue item pairs with snapshot N; OwnedRecordDrawBegin() then checks the item's own words against it (a mismatch is a desync, counted,
// and nothing is paired until the next queue item). OwnedRecordDrawNow() is the paired snapshot while the draw runs, else nullptr.
void OwnedRecordDrawAdvance();
const OwnedRecordState* OwnedRecordDrawBegin(uint32_t viz_query, uint32_t initiator, uint32_t dma_base, uint32_t dma_size);
void OwnedRecordDrawEnd();
const OwnedRecordState* OwnedRecordDrawNow();
struct OwnedRecordSnapStats {
  uint64_t taken = 0;           // snapshots stored
  uint64_t overflow_items = 0;  // draw items past the 512 cap (no snapshot)
  uint64_t advanced = 0;        // draw items the old path's handler was given
  uint64_t paired = 0;          // ... that paired with a snapshot
  uint64_t no_snapshot = 0;     // ... in a queue item with no such snapshot (a segment's replay, a draw the decoder did not see)
  uint64_t overflow_draws = 0;  // ... that fell past the cap
  uint64_t desync = 0;          // ... whose words differ from the snapshot's (the rest of the queue item is not paired)
};
OwnedRecordSnapStats& OwnedRecordSnapCounters();
// M4d: counts every draw item the old path's handler is given (OwnedRecordDrawAdvance); a typed repeat needs consecutive numbers.
uint64_t OwnedRecordDrawSeq();

// M4d (latch bit 3): the typed state of a crowd member's draw. The crowd code (SceneCrowdAfterDraw) issues each member's draw after its run's
// first member's item; the member differs from that item only in the constants c76-c78 (12 words, host order). Returns the last paired
// snapshot of this queue item with c76-c78 replaced by `c76` (crowd_member set), or nullptr when there is none / it carries no banks.
// Valid until the next call.
const OwnedRecordState* OwnedRecordCrowdMemberState(const uint32_t* c76);

// ---- M4e (gpu_d3d_owned_front bit 4, value 16): clears, resolves and inline draws built from the typed state, as a SHADOW.
//
// Inline draws: the decoder takes a snapshot of the typed state at each 0x7E record (needs bit 1, with bit 2 for the banks; kept per queue
// item, at most 32). ExecuteD3dInlineDraw brackets its draw with OwnedRecordInlineBegin / End, which pairs the record's snapshot by sequence
// number (once: a tiling replay of the same sequence finds it used, counted) and makes it OwnedRecordDrawNow() for the claim's shadow
// (OwnedFrontShadowDraw / the bind shadow: built from the typed state, compared with the old path's description). The staged vertex
// address (slot 31 words 4-5) is the old path's own, taken as an input. Nothing the old path sees changes.
void OwnedRecordInlineBegin(uint32_t sequence, uint32_t viz_query, uint32_t initiator, uint32_t fetch_word4, uint32_t fetch_word5);
void OwnedRecordInlineEnd();

// Clears: what ClearStock did, read back at the point of its draw (the registers it wrote, the vertices it staged); compared with the clear the
// typed state + the record describe (surface, rect, values, masks, modes). Resolves: M3's CopyWords against the typed state's view.
struct OwnedClearOld {
  uint32_t surface_info = 0, color_info = 0, depth_info = 0, color_mask = 0, depth_control = 0, stencil_refmask = 0, mode_control = 0, aa_config = 0,
           vtx_cntl = 0;
  uint32_t vertex[3][7] = {};  // x y z r g b a as float bits
};
void OwnedRecordClearShadow(const uint32_t* sem, uint32_t count, const OwnedClearOld& old);
struct OwnedResolveOld {
  uint32_t copy_control = 0, dest_base = 0, dest_pitch = 0, dest_info = 0, surface_info = 0, color_info = 0, depth_info = 0, mode_control = 0,
           vtx_cntl = 0;
};
void OwnedRecordResolveShadow(const uint32_t* sem, uint32_t count, const OwnedResolveOld& old);
// The decoder's part of the M4e window line (clears, resolves, inline pairing); resets its window.
std::string OwnedRecordM4eSummary(uint64_t frames);

// ---- M4f-C (gpu_d3d_owned_front bit 128, THE CUT): the decoder as the executor.
//
// With the cut on (read where the queue attaches) d3d_noring.cpp hands kNrPackets / kNrRegs items to the `run` hook (d3d_owned_record.h) instead of
// the register-file machinery: the decoder walks the item IN ORDER and executes it on its live typed state (OwnedRecordStateNow()) - no snapshot, no
// second parse, no ExecutePacket, no arena, no feed image. RUN items store into the typed state and make the notifications (D3dFeedAccess::Notify's);
// SHADER items call LoadShader once; DRAW items, the 0x7D clear / resolve records and the 0x7E inline draw call the ENTRY POINTS below with the live state.
//
// An entry point that is not wired (nullptr: phase 1 of M4f-C) makes the executor refuse the WHOLE item before it touches anything (the item goes the
// existing way, counted by reason in free_counters::g_cut.old_path); a wired one is final: its answer is true when it drew / cleared / resolved, or dropped
// the work and counted it itself, false when the work failed (counted as a dropped item). Nothing falls back to the old path once an item has started.
// `record` is the 0x7D / 0x7E record's payload from its magic word, in host order, `dwords` long.
struct OwnedExecEntries {
  bool (*draw)(CommandProcessor& cp, const OwnedRecordState& live, uint32_t item_index) = nullptr;
  bool (*clear)(CommandProcessor& cp, const OwnedRecordState& live, const uint32_t* record, uint32_t dwords) = nullptr;
  bool (*resolve)(CommandProcessor& cp, const OwnedRecordState& live, const uint32_t* record, uint32_t dwords) = nullptr;
  bool (*inline_draw)(CommandProcessor& cp, const OwnedRecordState& live, const uint32_t* record, uint32_t dwords) = nullptr;
};
OwnedExecEntries& OwnedExecEntrySlots();

// The decoder's dirty epochs: each is bumped when a stored field / bank changes VALUE (not on a rewrite of the same words). `render` covers every typed
// field except the draw's own words (base vertex, initiator, DMA base / size) and the banks. Command thread only; the typed memo keys read them.
struct OwnedRecordEpochs {
  uint64_t render = 0, vs_float = 0, ps_float = 0, bool_loop = 0, fetch = 0;
};
const OwnedRecordEpochs& OwnedRecordEpochsNow();

}  // namespace rex::graphics::vulkan

namespace rex::memory {
class Memory;
}

namespace rex::graphics::vulkan {
// ---- Session 95 (item N2, owned_scout.cpp): the decoder's item grammar applied to a PRIVATE typed state, for the scout thread.
// Pure: it reads the register -> field table (built once on the command thread: OwnedRecordScoutPrepare) and writes only `st`. No notification, no
// epoch, no counter, no LoadShader (the shader identity is the XXH3 of the microcode, the shadow decoder's own), no guest store, no entry point. `kind` is
// kNrPackets or kNrRegs; `payload` the item's words in host order. `draw` is called at every token DRAW item after its words were stored.
struct OwnedScoutApplyHooks {
  void (*draw)(void* ctx, const OwnedRecordState& st) = nullptr;
  void* ctx = nullptr;
};
void OwnedRecordScoutPrepare();
void OwnedRecordScoutApply(OwnedRecordState& st, rex::memory::Memory* memory, uint32_t kind, const uint32_t* payload, uint32_t dwords,
                           const OwnedScoutApplyHooks& hooks);

// ---- M4f-C: the targets a typed draw serves its textures against under the cut (NativeServeView): the typed RB words, set around the draw; nullptr = registers.
struct OwnedServeTargets {
  uint32_t surface_info = 0, color_info[4] = {}, color_mask = 0;
};
void OwnedServeTargetsSet(const OwnedServeTargets* targets);  // (vulkan/command_processor.cpp)
// Session 93 (gpu_d3d_typed_repeat): the typed source of RepeatDrawFast (the vertex float constants, a VulkanPipelineCache::TypedKeyState); nullptr, nullptr = the register file.
void OwnedRepeatSourceSet(const uint32_t* vs_float, const void* keys);  // (vulkan/command_processor.cpp)
// Session 94 (gpu_d3d_decouple bit 3): view 1 (the in-between)'s vertex float constants of the draw being issued (a blended copy of the typed bank); nullptr = the same bank as view 0 (the picture).
void OwnedViewFloatSet(const uint32_t* vs_float_view0);  // (vulkan/command_processor.cpp)

// The executor's pieces the entry points (owned_front_own.cpp) reach: the live state (mutable: the inline entry marks it), a register-range store with the
// notifications (the inline draw's staged address, slot 31 words 4-5), and the clear / resolve records run by the copy executor (d3d_copy_records.cpp).
OwnedRecordState& OwnedRecordStateLive();
void OwnedExecStore(uint32_t first, uint32_t n, const uint32_t* values);
bool OwnedExecCopy(CommandProcessor& cp, uint32_t kind, const uint32_t* sem, uint32_t sem_dwords);
}  // namespace rex::graphics::vulkan

namespace rex::graphics::vulkan {
// ---- M4f-C: the shadow chain / crowd density predicates of IssueDraw, from words (vulkan/command_processor.cpp), for the typed draw under the cut.
bool OwnedCrowdDensityRejects(uint64_t vs_hash, uint32_t surface_info, const uint32_t* c76, int32_t density);
uint32_t OwnedDynamicShadowKind(uint64_t ps_hash, uint64_t vs_hash);
uint64_t OwnedShadowChainKey(uint32_t color_info, uint32_t depth_info);
bool OwnedShadowChainPitchOk(uint32_t surface_info, uint64_t ps_hash, bool copy);
bool OwnedShadowChainNotShadow(uint64_t ps_hash);
bool OwnedShadowChainHas(const std::vector<uint64_t>& v, uint64_t key);
void OwnedDynamicShadowsNoteSkip(uint32_t kind, uint64_t frame);
}  // namespace rex::graphics::vulkan
