// wwe2k14 patch (Session 92, M4a; docs\native\d3d\S92_M3_M5_DESIGN.md section 2, contract S92_M4A_CONTRACT.md): the record
// decoder, as a SHADOW.
//
// d3d_noring.cpp calls `before` for every queue item with the item's words in HOST order, straight from the queue storage (no
// arena copy, no byte swap, no ExecutePacket), and `after` once the existing path has run the item. This file:
//   before: decodes the item into OwnedRecordState (owned_record.h): the 0x7C token's RUN / SHADER / DRAW items, the Registers
//           item's ops (write, read-modify-write, the tiling bin ops; the texture-table ops are only counted), and counts the
//           other packet records (0x7D copy, 0x7E inline: M4e) and the present (the window tick).
//   after:  compares every typed field with the feed image (D3dFeedAccess::ImageValues) the existing path just updated, and the
//           shader identities with the base's active shaders. Counts by field; the first 8 differences are logged with both
//           values and the record's sequence number.
// The existing path draws the frame unchanged. Latch: gpu_d3d_owned_front, bit 0 (defined in d3d_noring.cpp).
//
// Register numbers appear ONLY in kFields below (register -> field) and in the bank bounds of StoreWord. The state is typed
// fields plus the shader arrays; the per-register bit sets (carried / read by nobody) are census bookkeeping, not state.
#include "owned_record.h"
#include "owned_scout.h"       // (Session 95, item N2) the job slots' draw key
#include "d3d_feed.h"          // (M4f-C) D3dFeedAccess::Cut*: the processor-side effects of the executor
#include "owned_front_draw.h"  // (M4d) OwnedFrontRepeatBreak

#include <rex/cvar.h>
#include <rex/graphics/command_processor.h>
#include <rex/graphics/d3d_feed.h>
#include <rex/graphics/free_counters.h>  // (M4f-C) the cut's counters
#include <rex/graphics/d3d_copy.h>
#include <rex/graphics/d3d_inline_draw.h>
#include <rex/graphics/d3d_noring.h>
#include <rex/graphics/d3d_owned_record.h>
#include <rex/graphics/d3d_tex.h>
#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/graphics/register_file.h>
#include <rex/hash.h>
#include <rex/logging.h>
#include <rex/system/xmemory.h>

#include "../../d3d_shader_tag.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

REXCVAR_DECLARE(uint32_t, gpu_d3d_owned_front);
REXCVAR_DECLARE(uint32_t, gpu_d3d_wall);  // (Session 95: d3d_noring.cpp)
REXCVAR_DEFINE_BOOL(gpu_d3d_decode_timer_sample, false, "GPU/Native",
                    "Session 95: the decoder's per-item timer (the '[d3d][front] decode us a frame' figure) reads the clock on 1 item in 8 and scales by 8, instead of twice "
                    "on every item (~1.5% of the command thread). Off = timed on every item, as before.");

REXCVAR_DEFINE_BOOL(gpu_d3d_inline_merge, false, "GPU/Native",
                    "Session 95 r5: runs of particles (yParticle's four-vertex inline strips) that differ only in their matrix are drawn as one strip, each matrix put into its vertices. "
                    "Same shaders, pipeline and order. Only with the shader's camera fade off (its own test); anything else stays a draw of its own. [d3d][inline] merge lines. Off by default.");

namespace rex::graphics::vulkan {

namespace {

constexpr uint32_t kTokenMagic = 0xD3D10002u;  // command_processor.cpp kD3dTokenMagic
constexpr uint32_t kRegCount = uint32_t(RegisterFile::kRegisterCount);
constexpr uint32_t kFloatLo = 0x4000, kFloatPixelLo = 0x4400, kFloatHi = 0x4800;
constexpr uint32_t kFetchLo = 0x4800, kFetchHi = 0x48C0;
constexpr uint32_t kBoolLo = 0x4900, kBoolHi = 0x4928;
constexpr uint32_t kWindowPresents = 600;  // as the consumer's [d3d][noring] line

OwnedRecordState g_s;
}  // namespace
extern uint64_t g_draw_seq;  // (defined below, outside the anonymous namespace)
namespace {

// ---- M4f-C (the cut): the executor's state. g_x is the processor while an item is being EXECUTED (Run, or the present's fetch writes under the cut); null
// in the shadow, where every function below behaves exactly as before the cut existed.
CommandProcessor* g_x = nullptr;
OwnedRecordEpochs g_ep;
OwnedExecEntries g_entries;
bool g_in_regs = false;  // executing a Registers item (drops are counted by index there; a RUN's words with no field on their own counter)
free_counters::CutCounters& g_cc = free_counters::g_cut;
using Feed = D3dFeedAccess;

// ---- M4b: the state as of each draw item of the current queue item (bit 1 of the latch)
constexpr uint32_t kMaxSnaps = 512;
std::vector<OwnedRecordState> g_snaps;  // [0, g_snap_n) are live
uint32_t g_snap_n = 0;       // stored
uint32_t g_snap_seen = 0;    // draw items decoded in this queue item (stored or past the cap)
uint32_t g_snap_adv = 0;     // draw items the old path's handler has been given in this queue item
bool g_snap_on = false;
bool g_snap_banks = false;  // M4c (latch bit 2): the snapshots carry the shader constant banks too
bool g_snap_desync = false;
const OwnedRecordState* g_snap_cand = nullptr;  // the snapshot of the draw item just advanced to
const OwnedRecordState* g_snap_cur = nullptr;
const OwnedRecordState* g_snap_last = nullptr;  // M4d: the last snapshot paired in this queue item (the crowd members' base)
OwnedRecordState g_member;                      // M4d: OwnedRecordCrowdMemberState's result
OwnedRecordSnapStats g_snap_stats;

// Everything but the shader constant banks (with M4c on, those too): the fields before vs_float, the banks (M4c), the fetch constants, and the tail after them.
void CopySnapshot(OwnedRecordState& d) {
  std::memcpy(&d, &g_s, offsetof(OwnedRecordState, vs_float));
  if (g_snap_banks) {  // vs_float, ps_float, bool_loop are contiguous, fetch follows
    std::memcpy(d.vs_float, g_s.vs_float, offsetof(OwnedRecordState, fetch) - offsetof(OwnedRecordState, vs_float));
  }
  std::memcpy(d.fetch, g_s.fetch, sizeof(d.fetch));
  std::memcpy(reinterpret_cast<uint8_t*>(&d) + offsetof(OwnedRecordState, draw_viz_query),
              reinterpret_cast<const uint8_t*>(&g_s) + offsetof(OwnedRecordState, draw_viz_query),
              sizeof(OwnedRecordState) - offsetof(OwnedRecordState, draw_viz_query));
  d.banks_valid = g_snap_banks ? 1u : 0u;
}

void TakeSnapshot() {
  ++g_snap_seen;
  if (g_snap_n >= kMaxSnaps) {
    ++g_snap_stats.overflow_items;
    return;
  }
  if (g_snaps.size() <= g_snap_n) g_snaps.resize(g_snap_n + 16);
  CopySnapshot(g_snaps[g_snap_n++]);
  ++g_snap_stats.taken;
}

// ---- M4e (bit 4): the clear / resolve / inline records of the current queue item, as the decoder saw them
constexpr uint32_t kMaxCopyEntries = 64, kMaxInlineSnaps = 32, kCopySemMax = 24;
struct CopyEntry {
  uint32_t kind = 0, seq = 0, count = 0;
  bool used = false;
  uint32_t sem[kCopySemMax] = {};
  // the typed state right after the decoder applied the record (resolve: surface, the source's colour info, depth info, vertex rounding);
  // a clear: the depth info the clear leaves alone when it clears no depth / stencil
  uint32_t surface_info = 0, color_info = 0, depth_info = 0, su_vtx_cntl = 0;
};
CopyEntry g_copy[kMaxCopyEntries];
uint32_t g_copy_n = 0;
struct InlineSnap {
  uint32_t seq = 0;
  bool used = false;
  OwnedRecordState s;
};
std::vector<InlineSnap> g_inl;  // sized once
uint32_t g_inl_n = 0;

enum : uint32_t {
  kCfSurface, kCfColorInfo, kCfDepthInfo, kCfColorWrite, kCfDepthControl, kCfStencil, kCfMode, kCfAa, kCfVtxCntl, kCfRect, kCfDepthValue, kCfColorValue,
  kCfCount
};
const char* const kCfNames[kCfCount] = {"surface_info", "color_info", "depth_info", "color_written", "depth_control", "stencil", "mode_control",
                                        "aa_config", "vtx_cntl", "rect", "depth_value", "color_value"};
enum : uint32_t {
  kRfCopyControl, kRfDestBase, kRfDestPitch, kRfDestInfo, kRfSurface, kRfColorInfo, kRfDepthInfo, kRfMode, kRfVtxCntl, kRfCount
};
const char* const kRfNames[kRfCount] = {"copy_control", "dest_base", "dest_pitch", "dest_info", "surface_info", "color_info", "depth_info", "mode_control", "vtx_cntl"};
struct M4eWindow {
  uint64_t clears_decoded = 0, clears_compared = 0, clears_unpaired = 0, clears_never_run = 0, clear_mismatch_records = 0;
  uint64_t clear_by_field[kCfCount] = {};
  uint64_t resolves_decoded = 0, resolves_compared = 0, resolves_unpaired = 0, resolves_never_run = 0, resolve_mismatch_records = 0, resolve_depth_source = 0;
  uint64_t resolve_by_field[kRfCount] = {};
  uint64_t inl_decoded = 0, inl_taken = 0, inl_overflow = 0, inl_paired = 0, inl_replays = 0, inl_unpaired = 0, inl_desync = 0;
  uint64_t ns = 0;
};
M4eWindow g_e;
bool g_cf_logged[kCfCount] = {}, g_rf_logged[kRfCount] = {};
uint32_t g_e_logged = 0;

// ---- the decoder's table: register -> field (count words from `reg`, at `offset` bytes into OwnedRecordState). The only place
// where render-state register numbers live.
struct FieldDef {
  uint16_t reg;
  uint16_t offset;
  uint16_t count;
  const char* name;
};
#define OR_F(R, m) {XE_GPU_REG_##R, uint16_t(offsetof(OwnedRecordState, m)), 1, #m}
#define OR_FI(R, m, i) {XE_GPU_REG_##R, uint16_t(offsetof(OwnedRecordState, m) + 4 * (i)), 1, #m "[" #i "]"}
#define OR_FN(R, m, n) {XE_GPU_REG_##R, uint16_t(offsetof(OwnedRecordState, m)), n, #m}
const FieldDef kFields[] = {
    // surfaces
    OR_F(RB_MODECONTROL, modecontrol),
    OR_F(RB_SURFACE_INFO, surface_info),
    OR_FI(RB_COLOR_INFO, color_info, 0),
    OR_FI(RB_COLOR1_INFO, color_info, 1),
    OR_FI(RB_COLOR2_INFO, color_info, 2),
    OR_FI(RB_COLOR3_INFO, color_info, 3),
    OR_F(RB_DEPTH_INFO, depth_info),
    // raster / clip / viewport / window / scissors
    OR_F(PA_SU_SC_MODE_CNTL, sc_mode_cntl),
    OR_F(PA_SU_VTX_CNTL, su_vtx_cntl),
    OR_F(PA_SU_POINT_SIZE, point_size),
    OR_F(PA_SU_POINT_MINMAX, point_minmax),
    OR_FN(PA_SU_POLY_OFFSET_FRONT_SCALE, poly_offset, 4),
    OR_F(PA_CL_CLIP_CNTL, clip_cntl),
    OR_F(PA_CL_VTE_CNTL, vte_cntl),
    OR_FN(PA_CL_VPORT_XSCALE, vport, 6),
    OR_FN(PA_CL_UCP_0_X, user_clip, 24),
    OR_F(PA_SC_WINDOW_OFFSET, window_offset),
    OR_F(PA_SC_WINDOW_SCISSOR_TL, window_scissor_tl),
    OR_F(PA_SC_WINDOW_SCISSOR_BR, window_scissor_br),
    OR_F(PA_SC_SCREEN_SCISSOR_TL, screen_scissor_tl),
    OR_F(PA_SC_SCREEN_SCISSOR_BR, screen_scissor_br),
    OR_F(PA_SC_VIZ_QUERY, viz_query_cntl),
    // depth / stencil / colour / blend
    OR_F(RB_DEPTHCONTROL, depth_control),
    OR_F(RB_STENCILREFMASK, stencil_front),
    OR_F(RB_STENCILREFMASK_BF, stencil_back),
    OR_F(RB_COLORCONTROL, color_control),
    OR_F(RB_ALPHA_REF, alpha_ref),
    OR_FI(RB_BLENDCONTROL0, blend, 0),
    OR_FI(RB_BLENDCONTROL1, blend, 1),
    OR_FI(RB_BLENDCONTROL2, blend, 2),
    OR_FI(RB_BLENDCONTROL3, blend, 3),
    OR_FN(RB_BLEND_RED, blend_color, 4),
    OR_F(RB_COLOR_MASK, color_mask),
    // primitive / VGT
    OR_F(VGT_MIN_VTX_INDX, min_vtx_indx),
    OR_F(VGT_MAX_VTX_INDX, max_vtx_indx),
    OR_F(VGT_INDX_OFFSET, indx_offset),
    OR_F(VGT_MULTI_PRIM_IB_RESET_INDX, reset_indx),
    OR_F(VGT_OUTPUT_PATH_CNTL, output_path_cntl),
    OR_F(VGT_HOS_CNTL, hos_cntl),
    OR_FN(VGT_HOS_MAX_TESS_LEVEL, hos_tess, 2),
    OR_F(VGT_DRAW_INITIATOR, draw_initiator),
    OR_F(VGT_DMA_BASE, dma_base),
    OR_F(VGT_DMA_SIZE, dma_size),
    // shader control
    OR_F(SQ_PROGRAM_CNTL, program_cntl),
    OR_F(SQ_CONTEXT_MISC, context_misc),
    OR_F(SQ_INTERPOLATOR_CNTL, interpolator_cntl),
    OR_F(SQ_VS_CONST, vs_const),
    OR_F(SQ_PS_CONST, ps_const),
};
#undef OR_F
#undef OR_FI
#undef OR_FN
constexpr uint32_t kFieldCount = uint32_t(sizeof(kFields) / sizeof(kFields[0]));
// Pseudo-fields for the shader arrays in the statistics.
enum : uint32_t { kBankVsFloat = kFieldCount, kBankPsFloat, kBankBoolLoop, kBankFetch, kStatCount };

// register -> (field index + 1, byte offset into the state); built once from kFields.
constexpr uint16_t kNone = 0xFFFF;
uint16_t g_reg_field[kRegCount];
uint16_t g_reg_offset[kRegCount];
bool g_table_built = false;
void BuildTable() {
  if (g_table_built) return;
  for (uint32_t r = 0; r < kRegCount; ++r) g_reg_field[r] = g_reg_offset[r] = kNone;
  for (uint32_t i = 0; i < kFieldCount; ++i) {
    for (uint32_t w = 0; w < kFields[i].count; ++w) {
      const uint32_t reg = kFields[i].reg + w;
      if (reg < kRegCount) {
        g_reg_field[reg] = uint16_t(i);
        g_reg_offset[reg] = uint16_t(kFields[i].offset + 4 * w);
      }
    }
  }
  g_table_built = true;
}

// ---- census bookkeeping (bits per register: NOT state)
uint64_t g_carried[(kRegCount + 63) / 64];  // a record carried this register
uint64_t g_unread[(kRegCount + 63) / 64];   // ... and no field reads it
inline bool Bit(const uint64_t* b, uint32_t r) { return (b[r >> 6] >> (r & 63)) & 1; }
inline void SetBit(uint64_t* b, uint32_t r) { b[r >> 6] |= uint64_t(1) << (r & 63); }

// Where a difference is put: a decoder bug (carried), init-time state (never carried), or right after an item kind whose writes the
// decoder applies from its own reading of the old path (the bucket is named, never hidden).
enum : uint32_t { kClassCarried, kClassNever, kClassClear, kClassResolve, kClassInline, kClassPresent, kClassCount };
const char* const kClassName[kClassCount] = {"carried", "never carried", "after a clear record", "after a resolve record",
                                              "after an inline draw", "after a present"};

struct Window {
  uint64_t tokens = 0, copies = 0, inlines = 0, other_packets = 0, packet_framing_errors = 0;
  uint64_t reg_items = 0, reg_writes = 0, reg_rmw = 0, reg_rmw_unmapped = 0, reg_bin = 0, reg_tex = 0;
  uint64_t runs = 0, run_words = 0, draws = 0, shader_items = 0, token_errors = 0;
  uint64_t compares = 0, compares_skipped = 0;
  uint64_t shader_checked = 0, shader_mismatch = 0;
  uint64_t mismatch_carried = 0, mismatch_never = 0;
  uint64_t by_class[kClassCount] = {};
  uint64_t by_field[kClassCount][kStatCount] = {};
  uint64_t segments = 0, segment_dwords = 0;
  uint64_t ns = 0;
  uint64_t presents = 0;
};
Window g_w;
uint64_t g_total_packets = 0, g_total_tokens = 0, g_total_draws = 0;  // cumulative, for the match with the consumer's line
uint64_t g_frame_first = 0, g_present_total = 0;
uint32_t g_shader_logged = 0;
bool g_shader_pending[2] = {false, false};  // a shader item in the item being run
bool g_shader_check = false;     // the item held a draw (token or inline): both active shaders are compared
bool g_item_draw = false;        // the item ended on a token DRAW: the per-draw words are compared
bool g_fetch31_private = false;  // fetch[190..191] hold the inline path's own staging address: left out of the compare
uint32_t g_last_foreign = 0;     // class of the last non-token packet item (0 = none since the last token)
bool g_field_logged[kClassCount][kStatCount] = {};
uint32_t g_logged_total = 0;
uint64_t g_item_start_ns = 0;

inline uint64_t NowNs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

const char* StatName(uint32_t s) {
  if (s < kFieldCount) return kFields[s].name;
  switch (s) {
    case kBankVsFloat: return "vs_float";
    case kBankPsFloat: return "ps_float";
    case kBankBoolLoop: return "bool_loop";
    default: return "fetch";
  }
}

// ---- storing a register value into its field
inline uint8_t* StateBytes() { return reinterpret_cast<uint8_t*>(&g_s); }

// (M4f-C) the draw's own words are not render state: a change of them does not bump the render epoch.
inline bool IsPerDrawReg(uint32_t reg) {
  return reg == XE_GPU_REG_VGT_INDX_OFFSET || reg == XE_GPU_REG_VGT_DRAW_INITIATOR || reg == XE_GPU_REG_VGT_DMA_BASE || reg == XE_GPU_REG_VGT_DMA_SIZE;
}

// Stores one register's value; true when the stored value differs from the one held (M4f-C: bumps the dirty epochs; while executing, a register with
// no field and no bank - no reader - is dropped and counted by index).
bool StoreWord(uint32_t reg, uint32_t v) {
  if (reg >= kRegCount) {
    if (g_x) ++g_cc.reg_out_of_range;
    return false;
  }
  SetBit(g_carried, reg);
  bool changed = false;
  if (reg >= kFloatLo && reg < kFloatPixelLo) {
    uint32_t& d = g_s.vs_float[reg - kFloatLo];
    if ((changed = d != v)) ++g_ep.vs_float;
    d = v;
  } else if (reg >= kFloatPixelLo && reg < kFloatHi) {
    uint32_t& d = g_s.ps_float[reg - kFloatPixelLo];
    if ((changed = d != v)) ++g_ep.ps_float;
    d = v;
  } else if (reg >= kFetchLo && reg < kFetchHi) {
    uint32_t& d = g_s.fetch[reg - kFetchLo];
    if ((changed = d != v)) ++g_ep.fetch;
    d = v;
    if (reg == kFetchLo + 190 || reg == kFetchLo + 191) g_fetch31_private = false;  // slot 31 words 4-5 written by a record
  } else if (reg >= kBoolLo && reg < kBoolHi) {
    uint32_t& d = g_s.bool_loop[reg - kBoolLo];
    if ((changed = d != v)) ++g_ep.bool_loop;
    d = v;
  } else if (g_reg_offset[reg] != kNone) {
    uint32_t old;
    std::memcpy(&old, StateBytes() + g_reg_offset[reg], sizeof(old));
    changed = old != v;
    if (changed && !IsPerDrawReg(reg)) ++g_ep.render;
    std::memcpy(StateBytes() + g_reg_offset[reg], &v, sizeof(v));
  } else {
    SetBit(g_unread, reg);  // carried, read by no field
    if (g_x) {
      if (g_in_regs) {  // a Registers-item op: dropped by index
        ++g_cc.reg_dropped;
        ++g_cc.dropped_by_index[reg];
      } else {  // a RUN / inline word with no typed field
        ++g_cc.run_words_no_field;
      }
    }
  }
  return changed;
}

bool LoadWord(uint32_t reg, uint32_t* out) {
  if (reg >= kRegCount) return false;
  if (g_x) {  // (M4f-C, executing: a read-modify-write of a bank register reads the bank; the shadow keeps counting it unmapped, as before)
    if (reg >= kFloatLo && reg < kFloatPixelLo) {
      *out = g_s.vs_float[reg - kFloatLo];
      return true;
    }
    if (reg >= kFloatPixelLo && reg < kFloatHi) {
      *out = g_s.ps_float[reg - kFloatPixelLo];
      return true;
    }
    if (reg >= kFetchLo && reg < kFetchHi) {
      *out = g_s.fetch[reg - kFetchLo];
      return true;
    }
    if (reg >= kBoolLo && reg < kBoolHi) {
      *out = g_s.bool_loop[reg - kBoolLo];
      return true;
    }
  }
  if (g_reg_offset[reg] == kNone) return false;
  std::memcpy(out, StateBytes() + g_reg_offset[reg], sizeof(*out));
  return true;
}

// ---- M4f-C: a run of `n` values for registers first.. EXECUTED. The same class split as D3dFeedAccess::Apply (floats 0x4000-0x47FF, fetch 0x4800-0x48BF,
// bool / loop 0x4900-0x4927, everything else word by word); each bank piece is stored in one copy, the change is found by comparing it with what the
// bank held (the memcmp the feed does against its image), and the notifications Notify makes are made for the piece (D3dFeedAccess::CutNotify).
void ExecStoreRun(uint32_t first, uint32_t n, const uint32_t* v) {
  uint32_t i = 0;
  while (i < n) {
    const uint32_t reg = first + i;
    if (reg >= kRegCount) {
      g_cc.reg_out_of_range += n - i;
      return;
    }
    uint32_t end = first + n;
    if (reg >= kFloatLo && reg < kFloatHi) {
      end = std::min(end, kFloatHi);
      bool vs_changed = false, ps_changed = false;
      if (reg < kFloatPixelLo) {  // the vertex half
        const uint32_t len = std::min(end, kFloatPixelLo) - reg;
        uint32_t* dst = g_s.vs_float + (reg - kFloatLo);
        vs_changed = std::memcmp(dst, v + i, size_t(len) * sizeof(uint32_t)) != 0;
        if (vs_changed) std::memcpy(dst, v + i, size_t(len) * sizeof(uint32_t));
      }
      if (end > kFloatPixelLo) {  // the pixel half
        const uint32_t from = std::max(reg, kFloatPixelLo);
        uint32_t* dst = g_s.ps_float + (from - kFloatPixelLo);
        const size_t bytes = size_t(end - from) * sizeof(uint32_t);
        ps_changed = std::memcmp(dst, v + i + (from - reg), bytes) != 0;
        if (ps_changed) std::memcpy(dst, v + i + (from - reg), bytes);
      }
      g_ep.vs_float += vs_changed ? 1 : 0;
      g_ep.ps_float += ps_changed ? 1 : 0;
      // (Session 95, gpu_d3d_wall bit 4) The same words again: the constant buffers already hold them, nothing to mark (the banks are only ever written here).
      if (vs_changed || ps_changed || !(REXCVAR_GET(gpu_d3d_wall) & 4u)) Feed::CutNotify(*g_x, Feed::Piece::kFloat, reg, end - reg, vs_changed || ps_changed, nullptr);
    } else if (reg >= kFetchLo && reg < kFetchHi) {
      end = std::min(end, kFetchHi);
      const uint32_t len = end - reg;
      uint32_t* dst = g_s.fetch + (reg - kFetchLo);
      const bool changed = std::memcmp(dst, v + i, size_t(len) * sizeof(uint32_t)) != 0;
      // (Session 95, gpu_d3d_wall bit 4) The words that really differ: texture slots and vertex streams outside them stay in sync, an all-equal write marks nothing (D3dFeedAccess::CutFetchChanged).
      uint32_t diff_first = 0, diff_count = 0;
      const bool narrow = (REXCVAR_GET(gpu_d3d_wall) & 4u) != 0;
      if (narrow && changed) {
        while (dst[diff_first] == v[i + diff_first]) ++diff_first;
        uint32_t diff_last = len - 1;
        while (dst[diff_last] == v[i + diff_last]) --diff_last;
        diff_count = diff_last - diff_first + 1;
      }
      if (changed) std::memcpy(dst, v + i, size_t(len) * sizeof(uint32_t));
      if (changed) ++g_ep.fetch;
      if (reg <= kFetchLo + 191 && end > kFetchLo + 190) g_fetch31_private = false;  // slot 31 words 4-5 written by a record
      if (narrow) {
        Feed::CutFetchChanged(*g_x, reg, len, reg + diff_first, diff_count, g_s.fetch);
      } else {
        Feed::CutNotify(*g_x, Feed::Piece::kFetch, reg, len, changed, g_s.fetch);
      }
    } else if (reg >= kBoolLo && reg < kBoolHi) {
      end = std::min(end, kBoolHi);
      const uint32_t len = end - reg;
      uint32_t* dst = g_s.bool_loop + (reg - kBoolLo);
      const bool changed = std::memcmp(dst, v + i, size_t(len) * sizeof(uint32_t)) != 0;
      if (changed) std::memcpy(dst, v + i, size_t(len) * sizeof(uint32_t));
      if (changed) ++g_ep.bool_loop;
      if (changed || !(REXCVAR_GET(gpu_d3d_wall) & 4u)) Feed::CutNotify(*g_x, Feed::Piece::kBoolLoop, reg, len, changed, nullptr);
    } else {
      for (const uint32_t boundary : {kFloatLo, kFloatHi, kFetchHi, kBoolLo, kBoolHi}) {  // the next class boundary above reg
        if (boundary > reg) {
          end = std::min(end, boundary);
          break;
        }
      }
      for (uint32_t r = reg; r < end; ++r) StoreWord(r, v[i + (r - reg)]);  // (render state: the epoch is bumped by a changed field; no reader: dropped)
    }
    i += end - reg;
  }
}

// A run of values the way the caller's mode wants it: executed (notified) under the cut, a plain store in the shadow.
void StoreRunAny(uint32_t first, uint32_t n, const uint32_t* v) {
  if (g_x) {
    ExecStoreRun(first, n, v);
  } else {
    for (uint32_t k = 0; k < n; ++k) StoreWord(first + k, v[k]);
  }
}

// ---- the token (command_processor.cpp ExecutePacketType3_D3D_TOKEN's item grammar)
void DecodeToken(rex::memory::Memory* memory, const uint32_t* w, uint32_t count) {
  if (count < 2 || w[0] != kTokenMagic) {
    ++g_w.token_errors;
    return;
  }
  const uint32_t seq = w[1];
  g_s.token_seq = seq;
  uint32_t i = 2, item_index = 0;
  while (i < count) {
    const uint32_t header = w[i];
    const uint32_t kind = header >> 28;
    uint32_t used = 0;
    if (kind == 1) {
      const uint32_t n = header & 0x0FFFFFFF;
      used = 2 + n;
      if (n == 0 || used > count - i) {
        ++g_w.token_errors;
        return;
      }
      const uint32_t first = w[i + 1];
      StoreRunAny(first, n, w + i + 2);
      ++g_w.runs;
      g_w.run_words += n;
      if (g_x) {
        ++g_cc.runs;
        g_cc.run_words += n;
      }
    } else if (kind == 2) {
      used = 3;
      if (used > count - i || (header & 0xF) > 1) {
        ++g_w.token_errors;
        return;
      }
      const uint32_t type = header & 0xF;
      const uint32_t addr = w[i + 1] & ~0x3u;
      const uint32_t dwords = w[i + 2] & 0xFFFF;
      // Identity as the token handler gives it to LoadShader: the microcode at the address, hashed as the pipeline cache keys it.
      // Identified here, not loaded a second time.
      const uint32_t* ptr = D3dShaderMicrocodePtrOf(memory, addr);
      OwnedRecordState::ShaderId& id = g_s.shader[type];
      id.address = addr;
      id.dwords = dwords;
      if (g_x) {
        // (M4f-C) executed: LoadShader once, the identity is the loaded shader's own microcode hash (no second XXH3), the stage's active shader set.
        id.hash = Feed::CutLoadShader(*g_x, type, addr, ptr, dwords);
        ++g_cc.shader_items;
        if (!id.hash) ++g_cc.shader_load_failed;
      } else {
        id.hash = XXH3_64bits(ptr, size_t(dwords) * sizeof(uint32_t));
        g_shader_pending[type] = true;
        g_shader_check = true;
      }
      ++g_w.shader_items;
    } else if (kind == 3) {
      used = 6;
      if (used > count - i) {
        ++g_w.token_errors;
        return;
      }
      StoreWord(XE_GPU_REG_VGT_INDX_OFFSET, w[i + 1]);  // the chunk's base vertex, written as a run of one
      g_s.draw_viz_query = w[i + 2];
      StoreWord(XE_GPU_REG_VGT_DRAW_INITIATOR, w[i + 3]);
      StoreWord(XE_GPU_REG_VGT_DMA_BASE, w[i + 4]);
      StoreWord(XE_GPU_REG_VGT_DMA_SIZE, w[i + 5]);
      g_s.crowd_id = (uint64_t(seq) << 16) | (item_index & 0xFFFF);  // (M4d review C3: a token can hold more than 256 items)
      g_item_draw = true;
      g_shader_check = true;
      ++g_w.draws;
      ++g_total_draws;
      if (g_x) {
        // (M4f-C) executed, in order, on the live state: one call into the draw entry point (no snapshot). Before the native renderer is enabled
        // (the boot frame) the draw is dropped and counted.
        ++g_cc.draw_items;
        ++g_draw_seq;  // (the typed repeat chain counts draw items: one per DRAW item, as the old handler's Advance did)
        scout::DrawItem();  // (Session 95, owned_scout.h: the draw's key for the job slots; a flag test when gpu_jobs is off)
        if (!Feed::CutReady(*g_x)) {
          ++g_cc.draws_before_owned;
        } else if (!g_entries.draw) {
          ++g_cc.draws_no_entry;  // (Run refuses a token with a DRAW item while the entry is not wired: not reachable)
        } else {
          if (!g_entries.draw(*g_x, g_s, item_index)) ++g_cc.entries_failed;
        }
        scout::ClearKey();
      } else if (g_snap_on) {
        TakeSnapshot();
      }
    } else {
      ++g_w.token_errors;
      return;
    }
    i += used;
    ++item_index;
  }
}

// The identity of a shader the way LoadShader keys it: XXH3 over the microcode words.
void SetShaderId(uint32_t type, uint32_t addr, uint32_t dwords, const uint32_t* words) {
  OwnedRecordState::ShaderId& id = g_s.shader[type];
  id.address = addr;
  id.dwords = dwords;
  id.hash = XXH3_64bits(words, size_t(dwords) * sizeof(uint32_t));
  g_shader_pending[type] = true;
  g_shader_check = true;
}

// 0x7E inline draw (d3d_inline_draw.cpp ExecutePacketType3_D3D_INLINE + ExecuteD3dInlineDraw; layout d3d_inline_draw.cpp ~59-105): 21 fixed
// words, 3 per run, the values, the vs / ps inline words, the vertex bytes. What the old path applies: the runs, the shader binds,
// VGT_INDX_OFFSET = 0, slot 31 words 0-3 (words 4-5 are the consumer's own staging address: left out of the compare), the three
// flush-fetch zeros, and the draw's initiator (auto-index, primitive type, vertex count).
bool DecodeInline(rex::memory::Memory* memory, const uint32_t* w, uint32_t count) {  // (M4f-C: false = a framing error, nothing to execute)
  OwnedFrontRepeatBreak();  // (M4d)
  constexpr uint32_t kFixed = 21;
  if (count < kFixed || w[0] != d3d::kInlineTokenMagic) {
    ++g_w.token_errors;
    return false;
  }
  const uint32_t flags = w[2], nruns = w[11], nvalues = w[12], vs_inline = w[15], ps_inline = w[18], vbytes = w[20];
  const uint64_t need = uint64_t(kFixed) + 3ull * nruns + nvalues + vs_inline + ps_inline + ((uint64_t(vbytes) + 3) >> 2);
  if (need != count) {
    ++g_w.token_errors;
    return false;
  }
  // (the old path rejects the whole record when a shader named by no address carries fewer words than it claims: d3d_inline_draw.cpp)
  if ((flags & d3d::kInlineHasVertexShader) && !w[13] && w[14] && w[14] > vs_inline) {
    ++g_w.token_errors;
    if (g_x) free_counters::NoteCutDrop("inline record rejected: a shader without an address carries fewer words than it claims");
    return false;
  }
  if ((flags & d3d::kInlineHasPixelShader) && !w[16] && w[17] && w[17] > ps_inline) {
    ++g_w.token_errors;
    if (g_x) free_counters::NoteCutDrop("inline record rejected: a shader without an address carries fewer words than it claims");
    return false;
  }
  const uint32_t* runs = w + kFixed;
  const uint32_t* values = runs + 3 * size_t(nruns);
  const uint32_t* vs_words = values + nvalues;
  const uint32_t* ps_words = vs_words + vs_inline;
  for (uint32_t r = 0; r < nruns; ++r) {
    const uint32_t first = runs[3 * r], n = runs[3 * r + 1], off = runs[3 * r + 2];
    if (uint64_t(off) + n > nvalues) {
      ++g_w.token_errors;
      return false;
    }
    StoreRunAny(first, n, values + off);
    ++g_w.runs;
    g_w.run_words += n;
    if (g_x) {
      ++g_cc.runs;
      g_cc.run_words += n;
    }
  }
  const auto bind = [&](uint32_t type, uint32_t addr, uint32_t dwords, const uint32_t* inline_words, uint32_t inline_count) {
    if (!dwords) return;
    ++g_w.shader_items;
    if (addr) {
      if (g_x) {  // (M4f-C executing: LoadShader once, the identity is the loaded shader's own hash)
        OwnedRecordState::ShaderId& id = g_s.shader[type];
        id.address = addr;
        id.dwords = dwords;
        id.hash = Feed::CutLoadShader(*g_x, type, addr, D3dShaderMicrocodePtrOf(memory, addr), dwords);
        if (!id.hash) ++g_cc.shader_load_failed;
      } else {
        SetShaderId(type, addr, dwords, D3dShaderMicrocodePtrOf(memory, addr));
      }
    } else {
      // (the inline words are raw guest bytes stored swapped twice: the old path hashes them byte-swapped back)
      static thread_local std::vector<uint32_t> raw;
      raw.resize(inline_count);
      for (uint32_t k = 0; k < inline_count; ++k) raw[k] = std::byteswap(inline_words[k]);
      if (g_x) {
        OwnedRecordState::ShaderId& id = g_s.shader[type];
        id.address = 0;
        id.dwords = dwords;
        id.hash = Feed::CutLoadShader(*g_x, type, 0, raw.data(), dwords);
        if (!id.hash) ++g_cc.shader_load_failed;
      } else {
        SetShaderId(type, 0, dwords, raw.data());
      }
    }
  };
  if (flags & d3d::kInlineHasVertexShader) bind(0, w[13], w[14], vs_words, vs_inline);
  if (flags & d3d::kInlineHasPixelShader) bind(1, w[16], w[17], ps_words, ps_inline);
  StoreWord(XE_GPU_REG_VGT_INDX_OFFSET, 0);
  StoreRunAny(XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + 6 * 31, 4, w + 7);  // (executed: slot 31 is notified like any fetch write)
  g_fetch31_private = true;  // words 4-5: the consumer's staging address and size
  for (uint32_t k = 0; k < 3; ++k) StoreWord(0x5000 + k, 0);
  StoreWord(XE_GPU_REG_VGT_DRAW_INITIATOR, (w[3] & 0x3Fu) | (2u << 6) | (w[4] << 16));
  g_s.draw_viz_query = w[6];
  // M4e (bit 4, with bit 1): the typed state as of this inline draw, for the claim's shadow (not while executing: no snapshots under the cut).
  if (!g_x && (REXCVAR_GET(gpu_d3d_owned_front) & 16u) != 0) {
    ++g_e.inl_decoded;
    if (g_snap_on) {
      if (g_inl.empty()) g_inl.resize(kMaxInlineSnaps);
      if (g_inl_n < kMaxInlineSnaps) {
        InlineSnap& is = g_inl[g_inl_n++];
        CopySnapshot(is.s);
        is.s.inline_draw = 1;
        is.seq = w[1];
        is.used = false;
        ++g_e.inl_taken;
      } else {
        ++g_e.inl_overflow;
      }
    }
  }
  return true;
}

// 0x7D copy-family record (d3d_copy.h; the executors d3d_copy_records.cpp): [0] magic [1] seq [2] kind [3] sem_dwords, then the block.
// A clear restores every register it writes except the initiator its draw leaves (rectangle list | auto-index | 3 vertices). A
// resolve leaves RB_SURFACE_INFO, the source's RB_COLOR_INFO, RB_DEPTH_INFO and PA_SU_VTX_CNTL written; its RB_COPY_* / clear-value
// registers are not state any field holds (counted by the carried-unread list).
bool DecodeCopy(const uint32_t* w, uint32_t count) {  // (M4f-C: false = a framing error, nothing to execute)
  OwnedFrontRepeatBreak();  // (M4d)
  if (count < kD3dCopyHeaderDwords || w[0] != kD3dCopyMagic || count != kD3dCopyHeaderDwords + w[3]) {
    ++g_w.token_errors;
    return false;
  }
  const uint32_t kind = w[2];
  const uint32_t* sem = w + kD3dCopyHeaderDwords;
  if (kind == kD3dCopyClear) {
    StoreWord(XE_GPU_REG_VGT_DRAW_INITIATOR, 8u | (2u << 6) | (3u << 16));
    g_last_foreign = kClassClear;
  } else if (kind == kD3dCopyResolve && w[3] >= d3d_resolve_sem::kDwords) {
    namespace rs = d3d_resolve_sem;
    static const uint32_t kColorInfoRegs[4] = {XE_GPU_REG_RB_COLOR_INFO, XE_GPU_REG_RB_COLOR1_INFO, XE_GPU_REG_RB_COLOR2_INFO,
                                               XE_GPU_REG_RB_COLOR3_INFO};
    const uint32_t src = sem[rs::kCopyControl] & 7u;  // copy_src_select (a depth resolve, 4, writes no colour info)
    StoreWord(XE_GPU_REG_RB_SURFACE_INFO, sem[rs::kSurfaceInfo]);
    if (src < 4) StoreWord(kColorInfoRegs[src], sem[rs::kColorInfo]);
    StoreWord(XE_GPU_REG_RB_DEPTH_INFO, sem[rs::kDepthInfo]);
    StoreWord(XE_GPU_REG_PA_SU_VTX_CNTL, sem[rs::kVtxCntl]);
    g_last_foreign = kClassResolve;
  }
  // M4e (bit 4): remember the record and the typed state it leaves, for the executor's shadow compare (a clear stores nothing: its draw's registers are
  // restored; the depth info it leaves alone is the state's as of this record).
  if (!g_x && (REXCVAR_GET(gpu_d3d_owned_front) & 16u) != 0 && (kind == kD3dCopyClear || kind == kD3dCopyResolve)) {
    if (kind == kD3dCopyClear) ++g_e.clears_decoded; else ++g_e.resolves_decoded;
    if (g_copy_n < kMaxCopyEntries && w[3] <= kCopySemMax) {
      CopyEntry& e = g_copy[g_copy_n++];
      e.kind = kind;
      e.seq = w[1];
      e.count = w[3];
      e.used = false;
      std::memcpy(e.sem, sem, size_t(w[3]) * sizeof(uint32_t));
      e.surface_info = g_s.surface_info.value;
      const uint32_t src = (kind == kD3dCopyResolve && w[3] >= d3d_resolve_sem::kDwords) ? (sem[d3d_resolve_sem::kCopyControl] & 7u) : 4u;
      e.color_info = src < 4 ? g_s.color_info[src].value : 0;
      e.depth_info = g_s.depth_info.value;
      e.su_vtx_cntl = g_s.su_vtx_cntl.value;
    }
  }
  return true;
}

// kNrPresent payload: VdSwap's words, type-0 packets writing fetch constant 0 (0x4800 x 6), then XE_SWAP.
// (M4f-C: under the cut the present's fetch constant 0 is not stored at all - see the line below.)
void DecodePresent(const uint32_t* p, uint32_t dwords) {
  uint32_t i = 0;
  while (i < dwords && (p[i] >> 30) == 0) {
    const uint32_t base = p[i] & 0x7FFF, n = ((p[i] >> 16) & 0x3FFF) + 1;
    if (n > dwords - i - 1) return;
    if (!free_counters::g_cut.active) StoreRunAny(base, n, p + i + 1);  // (under the cut the present's fetch 0 is NOT stored: the old as-data present wrote it nowhere, and an unnotified store would leave the typed bank and the bound texture disagreeing)
    i += 1 + n;
  }
  g_last_foreign = kClassPresent;
}

// (M4f-C) A clear / resolve / inline record, after the decoder applied its typed effects: handed to its entry point on the live state. Before the native
// renderer is enabled (the boot frame) it is dropped and counted.
void ExecRecord(uint32_t opcode, const uint32_t* w, uint32_t count) {
  bool (*fn)(CommandProcessor&, const OwnedRecordState&, const uint32_t*, uint32_t) = nullptr;
  if (opcode == 0x7D) {
    if (w[2] == kD3dCopyGprSplit) {
      ++g_cc.records_noop;
      return;
    }
    fn = w[2] == kD3dCopyClear ? g_entries.clear : w[2] == kD3dCopyResolve ? g_entries.resolve : nullptr;
  } else {
    fn = g_entries.inline_draw;
  }
  if (!fn) {  // (a 0x7D of a kind that is not clear / resolve / GPR split: counted)
    if (g_x) free_counters::NoteCutDrop("0x7D record of an unknown kind skipped");
    return;
  }
  if (!Feed::CutReady(*g_x)) {
    ++g_cc.records_before_owned;
    return;
  }
  ++g_cc.records_executed;
  if (!fn(*g_x, g_s, w, count)) ++g_cc.entries_failed;
}

// Session 95 r5 (gpu_d3d_inline_merge): Brock's pyro. A particle is its own inline draw: four vertices, a triangle strip, 2,700 of them a frame (PC run 003 of live_s95r5), and each one costs the command
// thread a whole draw (Surface, 14 W: 27 ms in those frames). Run after run of them differ only in their matrix (c228-c230): each has its billboard's rotation and translation. A run becomes ONE inline
// draw of the same strip: the first record's state as it stands, the vertices of all, joined by repeated vertices that make zero-area triangles, the parity of each quad kept.
// Each particle's own matrix goes into its vertices, in the first one's space: position' = M0^-1 (M position + t - t0), so M0 position' + t0 = M position + t. Only where the shader uses the position through
// the matrix alone: the camera fade of yParticle reads the raw position, so the fade must be off (c255.y > c201.x: the shader's own test). Same shaders, same pipeline, same order of the particles: the blend
// sees what it saw; the only change is rounding. A particle whose record writes anything else that differs, any other item, the present: the run is drawn first. Nothing is held across another item.
// State for a job system: all of it is the decoder's (one thread).
constexpr uint64_t kParticleVs = 0xC3570DF53E3AC981ull;  // yParticle (src\scene\arena_vs_particle.vert in the game)
constexpr uint32_t kParticleStride = 40, kParticleWords = 10, kParticleVerts = 4, kMaxBatch = 1024;
constexpr uint32_t kMatrixFirstWord = 228 * 4, kMatrixWords = 12;  // (three rows; the fourth, c231, is state like any other)
struct InlineBatch {
  bool open = false;
  std::vector<uint32_t> rec;  // the first record, then the joined vertices (host order floats)
  uint32_t particles = 0, vertices = 0;
  float t0[3] = {}, m[kMatrixWords] = {};  // the first particle's translation; the latest particle's matrix
  double inverse[9] = {};
  // the ~600-frame line
  uint64_t first_frame = UINT64_MAX, draws = 0, merged = 0, largest = 0;
  bool engaged = false;
  // A Registers item of texture Binds alone, between two particles (the layer pushes one ahead of a record that rewrites its fetch constant): held. The next particle joining with the same fetch words
  // makes it say nothing new (it is dropped); anything else comes after the batch's draw, this item first.
  std::vector<uint32_t> held;
  uint64_t binds_dropped = 0;
  uint64_t ended[9] = {};  // what ended a batch: another shape, another head, vertex floats, pixel floats, fetch words, another register, a token, a copy, a Registers item or another item
} g_inline_batch;

void DecodeRegs(const uint32_t* p, uint32_t dwords);

float VsFloat(uint32_t word) {
  float f;
  std::memcpy(&f, &g_s.vs_float[word], 4);
  return f;
}

bool InlineBatchShape(const uint32_t* w, uint32_t count) {
  if (count < 21 || (w[3] & 0x3Fu) != uint32_t(xenos::PrimitiveType::kTriangleStrip) || w[4] != kParticleVerts || w[5] != kParticleStride || w[15] || w[18] || w[20] != kParticleVerts * kParticleStride ||
      (w[2] & d3d::kInlineUseGuestVertexMemory)) {
    return false;
  }
  return count == 21 + 3ull * w[11] + w[12] + kParticleVerts * kParticleWords;
}

bool InlineBatchOpen(const uint32_t* w, uint32_t count) {
  InlineBatch& b = g_inline_batch;
  if (!InlineBatchShape(w, count) || g_s.shader[0].hash != kParticleVs || !Feed::CutReady(*g_x)) return false;
  if (!(VsFloat(255 * 4 + 1) > VsFloat(201 * 4))) return false;  // (the camera fade is on: it reads the raw position)
  // (Review r5, 4: position' keeps rows 0-2 only; the shader's w is dot(c231, position), so c231 must not read the position.)
  if (VsFloat(231 * 4) != 0.0f || VsFloat(231 * 4 + 1) != 0.0f || VsFloat(231 * 4 + 2) != 0.0f) return false;
  double m[9];
  for (uint32_t r = 0; r < 3; ++r) {
    for (uint32_t c = 0; c < 3; ++c) m[r * 3 + c] = VsFloat((228 + r) * 4 + c);
  }
  const double c0 = m[4] * m[8] - m[5] * m[7], c1 = m[5] * m[6] - m[3] * m[8], c2 = m[3] * m[7] - m[4] * m[6];
  const double det = m[0] * c0 + m[1] * c1 + m[2] * c2;
  if (!(std::abs(det) > 1e-9)) return false;
  b.inverse[0] = c0 / det;
  b.inverse[1] = (m[2] * m[7] - m[1] * m[8]) / det;
  b.inverse[2] = (m[1] * m[5] - m[2] * m[4]) / det;
  b.inverse[3] = c1 / det;
  b.inverse[4] = (m[0] * m[8] - m[2] * m[6]) / det;
  b.inverse[5] = (m[2] * m[3] - m[0] * m[5]) / det;
  b.inverse[6] = c2 / det;
  b.inverse[7] = (m[1] * m[6] - m[0] * m[7]) / det;
  b.inverse[8] = (m[0] * m[4] - m[1] * m[3]) / det;
  for (uint32_t i = 0; i < kMatrixWords; ++i) b.m[i] = VsFloat(kMatrixFirstWord + i);
  for (uint32_t i = 0; i < 3; ++i) b.t0[i] = b.m[i * 4 + 3];
  b.rec.assign(w, w + count);
  b.particles = 1;
  b.vertices = kParticleVerts;
  b.open = true;
  return true;
}

// A later particle: read only (nothing of its record is stored). True = it joined the batch.
bool InlineBatchTake(const uint32_t* w, uint32_t count) {
  InlineBatch& b = g_inline_batch;
  const uint32_t* const f = b.rec.data();
  if (b.particles >= kMaxBatch || !InlineBatchShape(w, count)) {
    ++b.ended[0];
    return false;
  }
  if (w[2] != f[2] || w[3] != f[3] || w[6] != f[6] || std::memcmp(w + 7, f + 7, 16) || w[13] != f[13] || w[14] != f[14] || w[16] != f[16] || w[17] != f[17]) {
    ++b.ended[1];
    return false;
  }
  const uint32_t nruns = w[11], nvalues = w[12];
  const uint32_t* const runs = w + 21;
  const uint32_t* const values = runs + 3 * size_t(nruns);
  float m[kMatrixWords];
  std::memcpy(m, b.m, sizeof(m));
  for (uint32_t r = 0; r < nruns; ++r) {
    const uint32_t first = runs[3 * r], n = runs[3 * r + 1], off = runs[3 * r + 2];
    if (uint64_t(off) + n > nvalues) return false;
    for (uint32_t k = 0; k < n; ++k) {
      const uint32_t reg = first + k, v = values[off + k];
      if (reg >= kFloatLo && reg < kFloatPixelLo) {
        const uint32_t word = reg - kFloatLo;
        if (word >= kMatrixFirstWord && word < kMatrixFirstWord + kMatrixWords) {
          std::memcpy(&m[word - kMatrixFirstWord], &v, 4);
        } else if (g_s.vs_float[word] != v) {
          ++b.ended[2];
          return false;
        }
      } else if (reg >= kFloatPixelLo && reg < kFloatHi) {
        if (g_s.ps_float[reg - kFloatPixelLo] != v) {
          ++b.ended[3];
          return false;
        }
      } else if (reg >= kFetchLo && reg < kFetchHi) {
        if (g_s.fetch[reg - kFetchLo] != v) {
          ++b.ended[4];
          return false;
        }
      } else if (reg >= kBoolLo && reg < kBoolHi) {
        if (g_s.bool_loop[reg - kBoolLo] != v) {
          ++b.ended[5];
          return false;
        }
      } else if (reg >= kRegCount) {
        ++b.ended[5];
        return false;
      } else if (g_reg_offset[reg] != kNone) {  // (a typed field: the same value says nothing; a register no field reads, the flush zeros among them, carries nothing)
        uint32_t held;
        std::memcpy(&held, StateBytes() + g_reg_offset[reg], sizeof(held));
        if (held != v) {
          ++b.ended[5];
          return false;
        }
      }
    }
  }
  const uint32_t* const v = values + nvalues;
  const auto vertex = [&](uint32_t i) {
    const size_t at = b.rec.size();
    b.rec.insert(b.rec.end(), v + i * kParticleWords, v + (i + 1) * kParticleWords);
    float p[3];
    std::memcpy(p, &b.rec[at], sizeof(p));
    double d[3];
    for (uint32_t r = 0; r < 3; ++r) d[r] = double(m[r * 4]) * p[0] + double(m[r * 4 + 1]) * p[1] + double(m[r * 4 + 2]) * p[2] + double(m[r * 4 + 3]) - b.t0[r];
    for (uint32_t r = 0; r < 3; ++r) p[r] = float(b.inverse[r * 3] * d[0] + b.inverse[r * 3 + 1] * d[1] + b.inverse[r * 3 + 2] * d[2]);
    std::memcpy(&b.rec[at], p, sizeof(p));
  };
  // The strip goes on: the last vertex again, this quad's first twice (zero-area triangles; an even number of them, so this quad's triangles keep their facing), then the quad.
  const size_t last = b.rec.size() - kParticleWords;
  b.rec.resize(b.rec.size() + kParticleWords);
  std::memcpy(&b.rec[last + kParticleWords], &b.rec[last], kParticleWords * 4);
  vertex(0);
  vertex(0);
  vertex(1);
  vertex(2);
  vertex(3);
  b.vertices += 6;
  ++b.particles;
  std::memcpy(b.m, m, sizeof(m));
  if (!b.held.empty()) {
    ++b.binds_dropped;
    b.held.clear();
  }
  return true;
}

bool InlineBatchHold(const uint32_t* p, uint32_t dwords) {
  if (!dwords || (dwords & 3u)) return false;
  for (uint32_t i = 0; i < dwords; i += 4) {
    if (p[i] != d3d_tex::kOpBind) return false;
  }
  InlineBatch& b = g_inline_batch;
  b.held.insert(b.held.end(), p, p + dwords);
  return true;
}

void InlineBatchDraw();
void InlineBatchFlush() {
  InlineBatch& b = g_inline_batch;
  if (!b.open) return;
  InlineBatchDraw();
  if (!b.held.empty()) {  // (after the draw, as it came)
    static thread_local std::vector<uint32_t> items;
    items.swap(b.held);
    b.held.clear();
    DecodeRegs(items.data(), uint32_t(items.size()));
  }
}

void InlineBatchDraw() {
  InlineBatch& b = g_inline_batch;
  b.open = false;
  d3d_tex::WriterScope tex_writer(d3d_tex::kWInline, true);  // (the first record's Binds are this draw's)
  uint32_t* const w = b.rec.data();
  if (b.particles > 1) {
    w[4] = b.vertices;
    w[20] = b.vertices * kParticleStride;
    StoreWord(XE_GPU_REG_VGT_DRAW_INITIATOR, (w[3] & 0x3Fu) | (2u << 6) | (b.vertices << 16));
  }
  ExecRecord(0x7E, w, uint32_t(b.rec.size()));
  // What the game left in the registers is the last particle's.
  if (b.particles > 1) {
    uint32_t v[kMatrixWords];
    std::memcpy(v, b.m, sizeof(v));
    StoreRunAny(kFloatLo + kMatrixFirstWord, kMatrixWords, v);
  }
  ++b.draws;
  b.merged += b.particles;
  b.largest = std::max<uint64_t>(b.largest, b.particles);
  const uint64_t frame = Feed::CutFrame(*g_x);
  if (b.first_frame == UINT64_MAX) b.first_frame = frame;
  if (!b.engaged && b.particles > 1) {
    b.engaged = true;
    REXGPU_INFO("[d3d][inline] merge on: frame {}: {} particles drawn as one strip of {} vertices", frame, b.particles, b.vertices);
  }
  if (frame - b.first_frame >= 600) {
    REXGPU_INFO("[d3d][inline] merge frames {}-{}: {} particles in {} draws, the largest run {}; Binds with nothing new {}; ended by: shape {}, head {}, vertex floats {}, pixel floats {}, fetch {}, register {}, "
                "token {}, copy {}, other item {}", b.first_frame, frame, b.merged, b.draws, b.largest, b.binds_dropped, b.ended[0], b.ended[1], b.ended[2], b.ended[3], b.ended[4], b.ended[5], b.ended[6], b.ended[7], b.ended[8]);
    b.first_frame = frame;
    b.draws = b.merged = b.largest = b.binds_dropped = 0;
    std::fill(std::begin(b.ended), std::end(b.ended), 0);
  }
}

// kNrPackets payload: one or more packets by value (type-3 headers), host order.
void DecodePackets(rex::memory::Memory* memory, const uint32_t* p, uint32_t dwords) {
  uint32_t i = 0;
  while (i < dwords) {
    const uint32_t header = p[i];
    if ((header >> 30) == 3) {
      const uint32_t opcode = (header >> 8) & 0x7F;
      const uint32_t count = ((header >> 16) & 0x3FFF) + 1;
      if (count > dwords - i - 1) {
        ++g_w.packet_framing_errors;
        return;
      }
      g_item_draw = false;
      if (opcode != 0x7E && g_inline_batch.open) {  // (Session 95 r5: the batch's draw comes before anything else of the stream)
        ++g_inline_batch.ended[opcode == 0x7C ? 6 : 7];
        InlineBatchFlush();
      }
      if (opcode == 0x7C) {
        ++g_w.tokens;
        ++g_total_tokens;
        g_last_foreign = 0;
        if (g_x) {
          ++g_cc.tokens;
          d3d_tex::WriterScope tex_writer(d3d_tex::kWToken, true);  // (the token handler's: who writes fetch constants, a Bind belongs to this token)
          DecodeToken(memory, p + i + 1, count);
        } else {
          DecodeToken(memory, p + i + 1, count);
        }
      } else if (opcode == 0x7D) {
        ++g_w.copies;
        if (g_x) {
          d3d_tex::WriterScope tex_writer(d3d_tex::kWCopy, false);  // (a copy token has no Bind to end)
          if (DecodeCopy(p + i + 1, count)) ExecRecord(opcode, p + i + 1, count);
        } else {
          DecodeCopy(p + i + 1, count);
        }
      } else if (opcode == 0x7E) {
        ++g_w.inlines;
        if (g_x) {
          // (Session 95 r5, gpu_d3d_inline_merge) A particle that differs from the open batch only in its translation joins it; its record is not applied.
          if (g_inline_batch.open && InlineBatchTake(p + i + 1, count)) {
            g_last_foreign = kClassInline;
            i += 1 + count;
            continue;
          }
          InlineBatchFlush();
          d3d_tex::WriterScope tex_writer(d3d_tex::kWInline, true);
          if (DecodeInline(memory, p + i + 1, count)) {
            if (REXCVAR_GET(gpu_d3d_inline_merge) && InlineBatchOpen(p + i + 1, count)) {
              tex_writer.ends_binds = false;  // (its draw is still to come: InlineBatchFlush ends the Binds)
            } else {
              ExecRecord(opcode, p + i + 1, count);
            }
          }
        } else {
          DecodeInline(memory, p + i + 1, count);
        }
        g_last_foreign = kClassInline;
      } else {
        ++g_w.other_packets;
        if (g_x) free_counters::NoteCutDrop("foreign type-3 packet in a packet item skipped (the item's records ran)");
      }
      i += 1 + count;
    } else if ((header >> 30) == 2) {
      ++i;  // a type-2 nop
    } else {
      InlineBatchFlush();
      ++g_w.other_packets;  // a type-0 / type-1 packet by value: the item's packets are not decoded further
      return;
    }
  }
}

// kNrRegs payload: quads {op, register, a, b}.
void DecodeRegs(const uint32_t* p, uint32_t dwords) {
  ++g_w.reg_items;
  // (M4f-C, executing) the texture-table ops are applied once, as the writer of a Registers item (d3d_noring.cpp's `apply(false)`: the drawing side).
  std::optional<d3d_tex::WriterScope> tex_writer;
  if (g_x) tex_writer.emplace(d3d_tex::kWRegsItem);
  for (uint32_t i = 0; i + 4 <= dwords; i += 4) {
    const uint32_t op = p[i], reg = p[i + 1], a = p[i + 2], b = p[i + 3];
    if (op >= d3d_tex::kOpDefine0 && op <= d3d_tex::kOpBind) {
      ++g_w.reg_tex;  // the texture object table: M4c
      if (g_x) {
        ++g_cc.reg_tex;
        if (d3d_tex::Enabled()) {
          d3d_tex::Table& table = d3d_tex::Table::Get();
          if (op <= d3d_tex::kOpDefine3) {
            table.Define(op - d3d_tex::kOpDefine0, reg, a, b, 0);
          } else if (op == d3d_tex::kOpDestroy) {
            table.Destroy(reg);
          } else if (op == d3d_tex::kOpChanged) {
            table.Changed(reg);
          } else {
            table.Bind(reg, a);
          }
        }
      }
    } else if (op >= d3d_noring_field::kRegsOpBinMaskLo && op <= d3d_noring_field::kRegsOpBinSelectHi) {
      ++g_w.reg_bin;
      if (g_x) {
        ++g_cc.reg_bin;
        Feed::CutBinOp(*g_x, op, a);  // (the processor's predication state, both sides as the old path)
      }
      if (op == d3d_noring_field::kRegsOpBinMaskLo) {
        g_s.bin_mask = (g_s.bin_mask & 0xFFFFFFFF00000000ull) | a;
      } else if (op == d3d_noring_field::kRegsOpBinMaskHi) {
        g_s.bin_mask = (g_s.bin_mask & 0xFFFFFFFFull) | (uint64_t(a) << 32);
      } else if (op == d3d_noring_field::kRegsOpBinSelectLo) {
        g_s.bin_select = (g_s.bin_select & 0xFFFFFFFF00000000ull) | a;
      } else {
        g_s.bin_select = (g_s.bin_select & 0xFFFFFFFFull) | (uint64_t(a) << 32);
      }
    } else if (op == d3d_noring_field::kRegsOpRmw) {
      uint32_t v = 0;
      if (LoadWord(reg & 0x1FFF, &v)) {
        const uint32_t nv = (v & a) | b;
        StoreRunAny(reg & 0x1FFF, 1, &nv);
        ++g_w.reg_rmw;
        if (g_x) ++g_cc.reg_rmw;
      } else {
        ++g_w.reg_rmw_unmapped;  // a read-modify-write of a register no field holds
        SetBit(g_carried, std::min(reg & 0x1FFF, kRegCount - 1));
        SetBit(g_unread, std::min(reg & 0x1FFF, kRegCount - 1));
        if (g_x) {  // no reader: dropped, counted by index
          ++g_cc.reg_dropped;
          ++g_cc.dropped_by_index[std::min(reg & 0x1FFF, kRegCount - 1)];
        }
      }
    } else {
      StoreRunAny(reg, 1, &a);
      ++g_w.reg_writes;
      if (g_x) ++g_cc.reg_writes;
    }
  }
}

// ---- the compare
const char* ItemKindName(uint32_t kind) {
  switch (kind) {
    case kNrPackets: return "packets";
    case kNrRegs: return "regs";
    default: return "other";
  }
}

void NoteMismatch(uint32_t stat, uint32_t reg, uint32_t word, uint32_t typed, uint32_t image, uint32_t kind) {
  const bool carried = Bit(g_carried, reg);
  // The class: right after a non-token packet item the difference is put in that kind's own bucket (named, never hidden); else
  // carried (a decoder bug) or never carried (init-time state).
  const uint32_t cls = g_last_foreign ? g_last_foreign : (carried ? kClassCarried : kClassNever);
  (carried ? g_w.mismatch_carried : g_w.mismatch_never)++;
  ++g_w.by_class[cls];
  ++g_w.by_field[cls][stat];
  // Logged: the first difference of each field in each class (cap 96 for the run), so init-state differences cannot use up the
  // budget before a carried one.
  if (!g_field_logged[cls][stat] && g_logged_total < 96) {
    g_field_logged[cls][stat] = true;
    ++g_logged_total;
    char name[64];
    if (stat < kFieldCount && kFields[stat].count == 1) {
      std::snprintf(name, sizeof name, "%s", kFields[stat].name);
    } else {
      std::snprintf(name, sizeof name, "%s[%u]", StatName(stat), word);
    }
    REXGPU_INFO("[d3d][front] shadow mismatch #{} ({}): {} (register {:04X}) typed {:08X} image {:08X}, after a {} item, token seq {}, register {}",
                g_logged_total, kClassName[cls], name, reg, typed, image, ItemKindName(kind), g_s.token_seq,
                carried ? "carried by a record" : "never carried by a record");
  }
}

void CompareBank(const uint32_t* typed, const uint32_t* image, uint32_t words, uint32_t reg_lo, uint32_t stat, uint32_t kind) {
  if (std::memcmp(typed, image, size_t(words) * sizeof(uint32_t)) == 0) return;
  for (uint32_t i = 0; i < words; ++i) {
    if (typed[i] != image[i]) NoteMismatch(stat, reg_lo + i, i, typed[i], image[i], kind);
  }
}

void CheckShaders(CommandProcessor& cp) {
  // Shader identity against what the existing path has active, after any item that held a SHADER item or a draw (token or inline):
  // both stages, so a shader the inline path set without restoring is seen too.
  if (!g_shader_check) return;
  g_shader_check = false;
  g_shader_pending[0] = g_shader_pending[1] = false;
  for (uint32_t t = 0; t < 2; ++t) {
    if (!g_s.shader[t].dwords) continue;  // nothing identified for this stage yet
    const Shader* active = t == 0 ? cp.active_vertex_shader() : cp.active_pixel_shader();
    ++g_w.shader_checked;
    if (!active || active->ucode_data_hash() != g_s.shader[t].hash) {
      ++g_w.shader_mismatch;
      if (g_shader_logged++ < 8) {
        REXGPU_INFO("[d3d][front] shader identity mismatch: {} typed {:016X} (address {:08X}, {} dwords), the existing path active shader {:016X}",
                    t == 0 ? "vertex" : "pixel", g_s.shader[t].hash, g_s.shader[t].address, g_s.shader[t].dwords,
                    active ? active->ucode_data_hash() : 0ull);
      }
    }
  }
}

void Compare(CommandProcessor& cp, uint32_t kind) {
  CheckShaders(cp);
  if (!(D3dFeedState() & kD3dFeedValid)) {
    ++g_w.compares_skipped;  // a foreign packet in between, or the feed is not on: nothing to compare against
    return;
  }
  const auto image_values = D3dFeedHandlerSlots().image_values;
  const uint32_t* image = image_values ? image_values() : nullptr;
  if (!image) {
    ++g_w.compares_skipped;
    return;
  }
  ++g_w.compares;
  const uint8_t* base = StateBytes();
  for (uint32_t i = 0; i < kFieldCount; ++i) {
    const FieldDef& d = kFields[i];
    for (uint32_t w = 0; w < d.count; ++w) {
      const uint32_t reg = d.reg + w;
      // The per-draw words (base vertex, DMA base / size, initiator) are the draw's own: compared only when the item ended on a token DRAW.
      if (!g_item_draw && (reg == XE_GPU_REG_VGT_INDX_OFFSET || reg == XE_GPU_REG_VGT_DMA_BASE || reg == XE_GPU_REG_VGT_DMA_SIZE ||
                           reg == XE_GPU_REG_VGT_DRAW_INITIATOR)) {
        continue;
      }
      uint32_t typed;
      std::memcpy(&typed, base + d.offset + 4 * w, sizeof(typed));
      const uint32_t img = image[reg];
      if (typed != img) NoteMismatch(i, reg, w, typed, img, kind);
    }
  }
  CompareBank(g_s.vs_float, image + kFloatLo, 1024, kFloatLo, kBankVsFloat, kind);
  CompareBank(g_s.ps_float, image + kFloatPixelLo, 1024, kFloatPixelLo, kBankPsFloat, kind);
  CompareBank(g_s.bool_loop, image + kBoolLo, uint32_t(kBoolHi - kBoolLo), kBoolLo, kBankBoolLoop, kind);
  // (fetch[190..191] = slot 31 words 4-5: the inline path's own staging address, left out while an inline draw wrote them last)
  CompareBank(g_s.fetch, image + kFetchLo, g_fetch31_private ? 190u : uint32_t(kFetchHi - kFetchLo), kFetchLo, kBankFetch, kind);
}

// ---- the window line
std::string Ranges(const uint64_t* bits, const uint32_t* image, bool only_set_in_image) {
  std::string out;
  uint32_t ranges = 0, total = 0;
  uint32_t r = 0;
  while (r < kRegCount) {
    const bool in = only_set_in_image ? (!Bit(g_carried, r) && image && image[r] != 0 &&
                                         ((r >= 0x2000 && r < 0x2400) || (r >= 0x4000 && r < 0x4930)))
                                      : Bit(bits, r);
    if (!in) {
      ++r;
      continue;
    }
    const uint32_t start = r;
    while (r < kRegCount) {
      const bool in2 = only_set_in_image ? (!Bit(g_carried, r) && image && image[r] != 0 &&
                                            ((r >= 0x2000 && r < 0x2400) || (r >= 0x4000 && r < 0x4930)))
                                         : Bit(bits, r);
      if (!in2) break;
      ++r;
    }
    total += r - start;
    if (++ranges <= 40) {
      char buf[24];
      if (r - start == 1) {
        std::snprintf(buf, sizeof buf, "%s%X", out.empty() ? "" : ",", start);
      } else {
        std::snprintf(buf, sizeof buf, "%s%X-%X", out.empty() ? "" : ",", start, r - 1);
      }
      out += buf;
    }
  }
  if (ranges > 40) out += ",... (" + std::to_string(ranges) + " ranges)";
  if (out.empty()) out = "none";
  return out + " [" + std::to_string(total) + " registers]";
}

void PrintWindow() {
  const Window& w = g_w;
  // top 8 fields of one class
  const auto top_of = [&](uint32_t cls) {
    uint32_t order[kStatCount];
    for (uint32_t i = 0; i < kStatCount; ++i) order[i] = i;
    std::sort(order, order + kStatCount, [&](uint32_t a, uint32_t b) { return w.by_field[cls][a] > w.by_field[cls][b]; });
    std::string top;
    for (uint32_t k = 0; k < 8; ++k) {
      const uint32_t s = order[k];
      if (!w.by_field[cls][s]) break;
      top += (top.empty() ? "" : ", ") + std::string(StatName(s)) + " " + std::to_string(w.by_field[cls][s]);
    }
    return top;
  };
  const auto image_values = D3dFeedHandlerSlots().image_values;
  const uint32_t* image = image_values ? image_values() : nullptr;
  const uint64_t frames = std::max<uint64_t>(w.presents, 1);
  REXGPU_INFO(
      "[d3d][front] frames {}-{}: packet items decoded {} (tokens {}, copy {}, inline {}, other {}; since the start {} tokens, {} packet "
      "records in all), register items {} (writes {}, rmw {}, rmw of an unread register {}, bin ops {}, texture ops {}), runs {} ({} words), "
      "draw items {} (since the start {}), shader items {}, framing errors {}; segment items seen {} ({} dwords, not decoded); "
      "shadow compares {} (skipped, image not valid {}): mismatches: carried {}, never carried {}, after a clear {}, after a resolve {}, "
      "after an inline draw {}, after a present {}; shader identities checked {}, mismatched {}; decode {:.1f} us a frame",
      g_frame_first, g_frame_first + w.presents - 1, w.tokens + w.copies + w.inlines + w.other_packets, w.tokens, w.copies,
      w.inlines, w.other_packets, g_total_tokens, g_total_packets, w.reg_items, w.reg_writes, w.reg_rmw, w.reg_rmw_unmapped,
      w.reg_bin, w.reg_tex, w.runs, w.run_words, w.draws, g_total_draws, w.shader_items, w.token_errors + w.packet_framing_errors,
      w.segments, w.segment_dwords, w.compares, w.compares_skipped, w.by_class[kClassCarried], w.by_class[kClassNever],
      w.by_class[kClassClear], w.by_class[kClassResolve], w.by_class[kClassInline], w.by_class[kClassPresent], w.shader_checked,
      w.shader_mismatch, double(w.ns) / 1000.0 / double(frames));
  for (uint32_t cls = 0; cls < kClassCount; ++cls) {
    if (w.by_class[cls]) REXGPU_INFO("[d3d][front] mismatches ({}) by field: {}", kClassName[cls], top_of(cls));
  }
  REXGPU_INFO("[d3d][front] registers carried, read by no field: {}", Ranges(g_unread, nullptr, false));
  REXGPU_INFO("[d3d][front] registers holding a value that no record ever carried (seen only before the records): {}",
              Ranges(nullptr, image, true));
  g_w = Window{};
  g_frame_first = g_present_total;
}

// ---- the hooks
void Before(CommandProcessor& cp, rex::memory::Memory* memory, uint32_t kind, const uint32_t* f, const uint32_t* payload,
            uint32_t payload_dwords) {
  (void)cp;
  BuildTable();
  const uint64_t t0 = NowNs();
  g_item_start_ns = t0;
  g_item_draw = false;
  // M4b: the previous item's draws have all run; this item's snapshots start afresh.
  g_snap_n = 0;
  g_snap_seen = 0;
  g_snap_adv = 0;
  g_snap_desync = false;
  g_snap_cand = g_snap_cur = g_snap_last = nullptr;
  // M4e: records decoded in the previous item whose executor never reached the shadow (dropped, or a replay path) are counted, then forgotten.
  for (uint32_t i = 0; i < g_copy_n; ++i) {
    if (!g_copy[i].used) ++(g_copy[i].kind == kD3dCopyClear ? g_e.clears_never_run : g_e.resolves_never_run);
  }
  g_copy_n = 0;
  g_inl_n = 0;
  g_snap_on = (REXCVAR_GET(gpu_d3d_owned_front) & 2u) != 0;
  g_snap_banks = g_snap_on && (REXCVAR_GET(gpu_d3d_owned_front) & 4u) != 0;
  // M4d: the typed repeat chain survives only draw items (and the fence / barrier / pad bookkeeping): any other item ends it.
  if (kind != kNrPackets && kind != kNrFence && kind != kNrBarrier && kind != kNrPad) OwnedFrontRepeatBreak();
  switch (kind) {
    case kNrSegment:  // (R4: pieces of a library segment are not decoded; counted so the bar line shows whether it matters)
      ++g_w.segments;
      g_w.segment_dwords += f[d3d_noring_field::kSegLength];
      break;
    case kNrPackets:
      ++g_total_packets;
      DecodePackets(memory, payload, payload_dwords);
      break;
    case kNrRegs:
      DecodeRegs(payload, payload_dwords);
      break;
    case kNrPresent:
      ++g_w.presents;
      ++g_present_total;
      // (M4f-C: under the cut DecodePresent does NOT store the present's fetch constant 0 - the old as-data present wrote it nowhere (RequestSwapTexture reads it
      // from the item), and a store would leave the typed bank and the bound texture disagreeing)
      DecodePresent(payload, payload_dwords);
      if (g_w.presents >= kWindowPresents) PrintWindow();
      break;
    default:
      break;
  }
  g_w.ns += NowNs() - t0;
}

void After(CommandProcessor& cp, uint32_t kind) {
#if REX_TRUE_NATIVE_ONLY
  (void)cp; (void)kind;
  return;  // (the TRUE NATIVE proof build: the shadow compare against the feed image is out)
#endif
  if (kind != kNrPackets && kind != kNrRegs) return;
  const uint64_t t0 = NowNs();
  Compare(cp, kind);
  g_w.ns += NowNs() - t0;
}

// ---- M4f-C: the executor (the `run` hook)

// Would Run take this kNrPackets payload whole? kCutOldCount = yes; else the reason it goes the existing way (nothing has been touched). A token is
// taken when every record in the payload is a 0x7C token (no DRAW item unless the draw entry is wired), a 0x7D clear / resolve or a 0x7E inline draw
// whose entry is wired; anything else (another packet record, a type-0 / type-1 packet) is the old path's.
uint32_t Handleable(const uint32_t* p, uint32_t dwords) {
  uint32_t i = 0;
  while (i < dwords) {
    const uint32_t header = p[i];
    if ((header >> 30) == 3) {
      const uint32_t opcode = (header >> 8) & 0x7F;
      const uint32_t count = ((header >> 16) & 0x3FFF) + 1;
      if (count > dwords - i - 1) return free_counters::kCutOldFraming;
      const uint32_t* w = p + i + 1;
      if (opcode == 0x7C) {
        if (!g_entries.draw) {  // a token with a DRAW item cannot be executed yet: walk its items (a malformed token is DecodeToken's to count)
          uint32_t at = 2;
          while (at < count) {
            const uint32_t item = w[at], kind = item >> 28;
            const uint32_t used = kind == 1 ? 2u + (item & 0x0FFFFFFFu) : kind == 2 ? 3u : kind == 3 ? 6u : 0u;
            if (!used || used > count - at) break;
            if (kind == 3) return free_counters::kCutOldDraw;
            at += used;
          }
        }
      } else if (opcode == 0x7D) {
        if (count < kD3dCopyHeaderDwords) return free_counters::kCutOldFraming;
        if (w[2] == kD3dCopyClear) {
          if (!g_entries.clear) return free_counters::kCutOldClear;
        } else if (w[2] == kD3dCopyResolve) {
          if (!g_entries.resolve) return free_counters::kCutOldResolve;
        } else if (w[2] == kD3dCopyGprSplit) {
          // census only (the 360's shader register split): a no-op, taken
        } else {
          // (a 0x7D of another kind: skipped and counted by the executor)
        }
      } else if (opcode == 0x7E) {
        if (!g_entries.inline_draw) return free_counters::kCutOldInline;
      } else {
        // (a foreign type-3 packet: skipped and counted by the executor, the 0x7C / 0x7D / 0x7E records of the item still run)
      }
      i += 1 + count;
    } else if ((header >> 30) == 2) {
      ++i;  // a type-2 nop
    } else {
      return free_counters::kCutOldOther;
    }
  }
  return free_counters::kCutOldCount;
}

bool Run(CommandProcessor& cp, rex::memory::Memory* memory, uint32_t kind, const uint32_t* f, const uint32_t* payload, uint32_t payload_dwords) {
  (void)f;
  if (kind != kNrPackets && kind != kNrRegs) return false;
  BuildTable();
  if (kind == kNrPackets) {
    const uint32_t why = Handleable(payload, payload_dwords);
    if (why != free_counters::kCutOldCount) {
      ++g_cc.old_path[why];
      static const char* const kWhy[free_counters::kCutOldCount] = {"packet item refused: draw entry missing", "packet item refused: clear entry missing", "packet item refused: resolve entry missing", "packet item refused: inline entry missing", "packet item refused: a packet that is not a 0x7C / 0x7D / 0x7E record", "packet item refused: packet framing error"};
      free_counters::NoteCutDrop(kWhy[why]);
      return false;
    }
  }
  // (Session 95) gpu_d3d_decode_timer_sample: time 1 item in 8 and scale the sum by 8 (the two clock reads an item were ~1.5% of the command thread).
  static uint32_t timer_tick = 0;
  const bool timer_sampled = REXCVAR_GET(gpu_d3d_decode_timer_sample);
  const bool timed = !timer_sampled || (timer_tick++ & 7u) == 0;
  const uint64_t t0 = timed ? NowNs() : 0;
  g_x = &cp;
  g_snap_on = false;  // (no snapshots under the cut)
  g_s.banks_valid = 1;  // (the live state always carries the banks)
  Feed::CutFetchSource(cp, g_s.fetch);  // the texture cache reads the typed fetch constants for the session
  cut_probe::g_on = Feed::CutReady(cp);  // detector 2: armed as soon as the native renderer is enabled
  if (cut_probe::g_on) cut_probe::g_armed_window = true;
  if (kind == kNrPackets) {
    ++g_cc.items_packets;
    ++g_total_packets;
    DecodePackets(memory, payload, payload_dwords);
  } else {
    ++g_cc.items_regs;
    if (!g_inline_batch.open || !InlineBatchHold(payload, payload_dwords)) {  // (Session 95 r5)
      if (g_inline_batch.open) ++g_inline_batch.ended[8];
      InlineBatchFlush();
      OwnedFrontRepeatBreak();  // (M4d: any item that is not a draw item ends the typed repeat chain)
      g_in_regs = true;
      DecodeRegs(payload, payload_dwords);
      g_in_regs = false;
    }
  }
  g_x = nullptr;
  if (timed) g_cc.ns += (NowNs() - t0) * (timer_sampled ? 8u : 1u);
  return true;
}

bool EntriesWired() { return g_entries.draw && g_entries.clear && g_entries.resolve && g_entries.inline_draw; }

// (Session 95 r5) An item that does not go through Run (a fence, a present, a segment ...): the open particle batch is drawn first.
void InlineFlushHook(CommandProcessor& cp) {
  if (!g_inline_batch.open) return;
  ++g_inline_batch.ended[8];
  g_x = &cp;
  g_s.banks_valid = 1;
  Feed::CutFetchSource(cp, g_s.fetch);
  InlineBatchFlush();
  g_x = nullptr;
}

// gpu_d3d_cut_poison: the register file's ranges the executor now owns - every typed field's register, 0x2000-0x23FF, the float / fetch / bool-loop
// constants and 0x5000-0x5002; nothing below 0x2000. check = false fills them; check = true counts the ones that changed, refills them, returns the count.
constexpr uint32_t kPoison = 0xBADC0DE5u;
uint64_t Poison(uint32_t* values, bool check) {
  uint64_t changed = 0;
  const auto one = [&](uint32_t r) {
    if (r < 0x2000 || r >= kRegCount) return;
    if (check && values[r] != kPoison) {
      ++changed;
      if (cut_probe::g_poison_n < 4) cut_probe::g_poison_regs[cut_probe::g_poison_n++] = r + 1;
    }
    values[r] = kPoison;
  };
  const auto range = [&](uint32_t lo, uint32_t hi) {
    for (uint32_t r = lo; r < hi; ++r) one(r);
  };
  for (uint32_t i = 0; i < kFieldCount; ++i) range(kFields[i].reg, kFields[i].reg + kFields[i].count);
  range(0x2000, 0x2400);
  range(0x4000, 0x48C0);
  range(0x4900, 0x4928);
  range(0x5000, 0x5003);
  return changed;
}

bool g_registered = [] {
  OwnedRecordHooks& h = OwnedRecordHookSlots();
  h.before = &Before;
  h.after = &After;
  h.run = &Run;
  h.entries_wired = &EntriesWired;
  h.inline_flush = &InlineFlushHook;
  h.poison = &Poison;
  h.scout_enter = &scout::DrainEnter;  // (Session 95, owned_scout.cpp)
  h.scout_leave = &scout::DrainLeave;
  h.scout_item = &scout::BeginItem;
  h.scout_present = &scout::NotePresent;
  h.scout_guest_write = &scout::NoteGuestWrite;
  return true;
}();

}  // namespace

const OwnedRecordState& OwnedRecordStateNow() { return g_s; }
OwnedExecEntries& OwnedExecEntrySlots() { return g_entries; }
OwnedRecordState& OwnedRecordStateLive() { return g_s; }
void OwnedExecStore(uint32_t first, uint32_t n, const uint32_t* values) { ExecStoreRun(first, n, values); }
const OwnedRecordEpochs& OwnedRecordEpochsNow() { return g_ep; }

uint64_t g_draw_seq = 0;
uint64_t OwnedRecordDrawSeq() { return g_draw_seq; }

void OwnedRecordDrawAdvance() {
  ++g_draw_seq;
  g_snap_cur = g_snap_cand = g_snap_last = nullptr;
  ++g_snap_stats.advanced;
  const uint32_t index = g_snap_adv++;
  if (g_snap_desync) return;
  if (index < g_snap_n) {
    g_snap_cand = &g_snaps[index];
  } else if (index < g_snap_seen) {
    ++g_snap_stats.overflow_draws;
  } else {
    ++g_snap_stats.no_snapshot;
  }
}

const OwnedRecordState* OwnedRecordDrawBegin(uint32_t viz_query, uint32_t initiator, uint32_t dma_base, uint32_t dma_size) {
  g_snap_cur = nullptr;
  const OwnedRecordState* s = g_snap_cand;
  g_snap_cand = nullptr;
  if (!s) return nullptr;
  if (s->draw_viz_query != viz_query || s->draw_initiator.value != initiator || s->dma_base != dma_base || s->dma_size.value != dma_size) {
    g_snap_desync = true;
    ++g_snap_stats.desync;
    return nullptr;
  }
  ++g_snap_stats.paired;
  g_snap_cur = s;
  return s;
}
void OwnedRecordDrawEnd() {
  if (g_snap_cur) g_snap_last = g_snap_cur;
  g_snap_cur = nullptr;
}
const OwnedRecordState* OwnedRecordCrowdMemberState(const uint32_t* c76) {
  // (M4f-C: under the cut the member's base is the LIVE state - the item being drawn; no snapshot exists)
  const OwnedRecordState* base = free_counters::g_cut.active ? &g_s : g_snap_last;
  if (!base || !base->banks_valid) return nullptr;
  g_member = *base;
  std::memcpy(g_member.vs_float + 4 * 76, c76, 12 * sizeof(uint32_t));
  g_member.crowd_member = 1;
  return &g_member;
}
const OwnedRecordState* OwnedRecordDrawNow() { return g_snap_cur; }
OwnedRecordSnapStats& OwnedRecordSnapCounters() { return g_snap_stats; }

}  // namespace rex::graphics::vulkan

// =================================================================================================================================
// M4e (gpu_d3d_owned_front bit 4): the shadow compares of the clear / resolve / inline records.
namespace rex::graphics::vulkan {

namespace {

CopyEntry* FindCopy(uint32_t kind, const uint32_t* sem, uint32_t count) {
  // The executor is given the record's words only: matched by kind, size and content, the oldest unused first (the decoder ran ahead in order).
  for (uint32_t i = 0; i < g_copy_n; ++i) {
    CopyEntry& e = g_copy[i];
    if (e.used || e.kind != kind || e.count != count) continue;
    if (std::memcmp(e.sem, sem, size_t(count) * sizeof(uint32_t)) == 0) return &e;
  }
  return nullptr;
}

void NoteClear(const CopyEntry& e, uint32_t f, uint32_t old_v, uint32_t typed_v) {
  ++g_e.clear_by_field[f];
  if (!g_cf_logged[f] && g_e_logged < 48) {
    g_cf_logged[f] = true;
    ++g_e_logged;
    REXGPU_INFO("[d3d][front] M4e clear mismatch #{}: {} old {:08X} typed {:08X} (record seq {}, flags {:X}, pass rt {:X})", g_e_logged, kCfNames[f], old_v,
                typed_v, e.seq, e.sem[0], e.sem[d3d_clear_sem::kPassRt]);
  }
}
void NoteResolve(const CopyEntry& e, uint32_t f, uint32_t old_v, uint32_t typed_v) {
  ++g_e.resolve_by_field[f];
  if (!g_rf_logged[f] && g_e_logged < 48) {
    g_rf_logged[f] = true;
    ++g_e_logged;
    REXGPU_INFO("[d3d][front] M4e resolve mismatch #{}: {} M3 word {:08X} typed {:08X} (record seq {}, copy control {:08X})", g_e_logged, kRfNames[f], old_v,
                typed_v, e.seq, e.sem[d3d_resolve_sem::kCopyControl]);
  }
}

}  // namespace

void OwnedRecordClearShadow(const uint32_t* sem, uint32_t count, const OwnedClearOld& o) {
  using namespace d3d_clear_sem;
  const uint64_t t0 = NowNs();
  CopyEntry* e = FindCopy(kD3dCopyClear, sem, count);
  if (!e || count < kDwords) {
    ++g_e.clears_unpaired;
    g_e.ns += NowNs() - t0;
    return;
  }
  e->used = true;
  ++g_e.clears_compared;
  bool any = false;
  const auto cmp = [&](uint32_t f, uint32_t old_v, uint32_t typed_v) {
    if (old_v != typed_v) {
      any = true;
      NoteClear(*e, f, old_v, typed_v);
    }
  };
  // The clear the typed state + the record describe (what NativeRenderer::D3dClear reads, and the fill the stock draw stands for).
  const uint32_t flags = sem[kFlags];
  const bool depth = (flags & 0x10) != 0, stencil = (flags & 0x20) != 0;
  const bool colour = sem[kPassRt] != 0xFFFFFFFFu;
  const uint32_t msaa = (sem[kSurfaceInfo] >> 16) & 3;
  cmp(kCfSurface, o.surface_info, sem[kSurfaceInfo]);
  // (the colour target is part of the effect only when colour is written: a depth-only clear's RB_COLOR_INFO is not a target)
  const bool old_depth_only = (o.mode_control & 7u) == uint32_t(xenos::EdramMode::kDepthOnly);
  const uint32_t old_colour_mask = old_depth_only ? 0u : (o.color_mask & 0xFu);
  cmp(kCfColorWrite, old_colour_mask, colour ? 0xFu : 0u);
  cmp(kCfColorInfo, old_colour_mask ? o.color_info : 0u, colour ? sem[kColorInfo] : 0u);
  // (a clear that clears neither depth nor stencil leaves RB_DEPTH_INFO as the state had it: the typed state's, as of the record)
  cmp(kCfDepthInfo, o.depth_info, (depth || stencil) ? sem[kDepthInfo] : e->depth_info);
  cmp(kCfDepthControl, o.depth_control, (depth ? 0x76u : 0u) | (stencil ? 0x8701u : 0u));
  cmp(kCfStencil, stencil ? (o.stencil_refmask & 0xFFu) : 0u, stencil ? (sem[kStencil] & 0xFFu) : 0u);
  cmp(kCfMode, o.mode_control & 7u, uint32_t(colour ? xenos::EdramMode::kColorDepth : xenos::EdramMode::kDepthOnly));
  cmp(kCfAa, o.aa_config, msaa == 0 ? 0u : msaa == 1 ? 0x8001u : 0xC003u);
  cmp(kCfVtxCntl, o.vtx_cntl & 1u, sem[kVtxCntl] & 1u);
  // The rectangle the three staged vertices stand for (l,t) (r,t) (r,b), less the half pixel the vertex rounding adds back.
  {
    const auto rounded = [&](uint32_t bits) {
      float v;
      std::memcpy(&v, &bits, sizeof(v));
      return int32_t(std::lround(v + ((o.vtx_cntl & 1) ? 0.0f : 0.5f)));
    };
    const int32_t old_rect[4] = {rounded(o.vertex[0][0]), rounded(o.vertex[0][1]), rounded(o.vertex[1][0]), rounded(o.vertex[2][1])};
    bool differs = false;
    for (uint32_t i = 0; i < 4; ++i) differs |= uint32_t(old_rect[i]) != sem[kRect + i];
    if (differs) {
      any = true;
      NoteClear(*e, kCfRect, uint32_t(old_rect[0]) ^ uint32_t(old_rect[1]) ^ uint32_t(old_rect[2]) ^ uint32_t(old_rect[3]),
                sem[kRect] ^ sem[kRect + 1] ^ sem[kRect + 2] ^ sem[kRect + 3]);
    }
  }
  cmp(kCfDepthValue, o.vertex[0][2], depth ? sem[kDepth] : 0u);
  {
    bool differs = false;
    for (uint32_t v = 0; v < 3; ++v) {
      for (uint32_t i = 0; i < 4; ++i) differs |= o.vertex[v][3 + i] != sem[kColor + i];
    }
    if (differs) {
      any = true;
      NoteClear(*e, kCfColorValue, o.vertex[0][3], sem[kColor]);
    }
  }
  if (any) ++g_e.clear_mismatch_records;
  g_e.ns += NowNs() - t0;
}

void OwnedRecordResolveShadow(const uint32_t* sem, uint32_t count, const OwnedResolveOld& o) {
  namespace rs = d3d_resolve_sem;
  const uint64_t t0 = NowNs();
  CopyEntry* e = FindCopy(kD3dCopyResolve, sem, count);
  if (!e || count < rs::kDwords) {
    ++g_e.resolves_unpaired;
    g_e.ns += NowNs() - t0;
    return;
  }
  e->used = true;
  ++g_e.resolves_compared;
  bool any = false;
  const auto cmp = [&](uint32_t f, uint32_t old_v, uint32_t typed_v) {
    if (old_v != typed_v) {
      any = true;
      NoteResolve(*e, f, old_v, typed_v);
    }
  };
  // The typed view of the words M3 feeds IssueCopyFrom: the surface / colour / depth info and the vertex rounding as the typed state holds them after
  // the record; the copy registers (RB_COPY_*) are no state field: the record's own words; the mode is the library's copy mode.
  const uint32_t src = sem[rs::kCopyControl] & 7u;
  cmp(kRfCopyControl, o.copy_control, sem[rs::kCopyControl]);
  cmp(kRfDestBase, o.dest_base, sem[rs::kDestBase]);
  cmp(kRfDestPitch, o.dest_pitch, sem[rs::kDestPitch]);
  cmp(kRfDestInfo, o.dest_info, sem[rs::kDestInfo]);
  cmp(kRfSurface, o.surface_info, e->surface_info);
  if (src < 4) {
    cmp(kRfColorInfo, o.color_info, e->color_info);
  } else {
    ++g_e.resolve_depth_source;  // a depth resolve: the state holds no colour info for it (the old path writes none either)
  }
  cmp(kRfDepthInfo, o.depth_info, e->depth_info);
  cmp(kRfMode, o.mode_control, uint32_t(xenos::EdramMode::kCopy));
  cmp(kRfVtxCntl, o.vtx_cntl, e->su_vtx_cntl);
  if (any) ++g_e.resolve_mismatch_records;
  g_e.ns += NowNs() - t0;
}

void OwnedRecordInlineBegin(uint32_t sequence, uint32_t viz_query, uint32_t initiator, uint32_t fetch_word4, uint32_t fetch_word5) {
  g_snap_cur = nullptr;
  InlineSnap* hit = nullptr;
  bool replay = false;
  for (uint32_t i = g_inl_n; i-- > 0;) {
    InlineSnap& is = g_inl[i];
    if (is.seq != sequence) continue;
    if (is.used) {
      replay = true;
    } else {
      hit = &is;
      break;
    }
  }
  if (!hit) {
    ++(replay ? g_e.inl_replays : g_e.inl_unpaired);
    return;
  }
  hit->used = true;
  if (hit->s.draw_viz_query != viz_query || hit->s.draw_initiator.value != initiator) {
    ++g_e.inl_desync;
    return;
  }
  hit->s.fetch[190] = fetch_word4;  // slot 31 words 4-5: the old path's own staging address and size, taken as an input
  hit->s.fetch[191] = fetch_word5;
  ++g_e.inl_paired;
  g_snap_cur = &hit->s;
}

void OwnedRecordInlineEnd() { g_snap_cur = nullptr; }

std::string OwnedRecordM4eSummary(uint64_t frames) {
  const M4eWindow& w = g_e;
  const auto top = [](const char* const* names, const uint64_t* counts, uint32_t n) {
    std::string out;
    uint32_t order[16];
    for (uint32_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order, order + n, [&](uint32_t a, uint32_t b) { return counts[a] > counts[b]; });
    for (uint32_t k = 0; k < 8 && k < n && counts[order[k]]; ++k) {
      out += (out.empty() ? "" : ", ") + std::string(names[order[k]]) + " " + std::to_string(counts[order[k]]);
    }
    return out.empty() ? std::string("none") : out;
  };
  char buf[1600];
  std::snprintf(buf, sizeof buf,
                "clears paired %llu of %llu decoded (real tests: depth_info when no depth/stencil is cleared, colour written / color_info, mode compared; the other fields are the same record words on both sides; records with a mismatch %llu: %s; no pair %llu, executor never reached %llu); "
                "resolves paired %llu of %llu decoded (coverage: the old fields are the record words the decoder stored; mismatches %llu: %s; no pair %llu, never reached %llu; depth-source resolves, colour info not compared %llu); "
                "inline records decoded %llu, snapshots %llu (past the cap %llu), paired with the old path's draw %llu, tiling replays (snapshot already used) %llu, no snapshot %llu, "
                "desync %llu; clear+resolve compare %.1f us a frame",
                (unsigned long long)w.clears_compared, (unsigned long long)w.clears_decoded, (unsigned long long)w.clear_mismatch_records,
                top(kCfNames, w.clear_by_field, kCfCount).c_str(), (unsigned long long)w.clears_unpaired, (unsigned long long)w.clears_never_run,
                (unsigned long long)w.resolves_compared, (unsigned long long)w.resolves_decoded, (unsigned long long)w.resolve_mismatch_records,
                top(kRfNames, w.resolve_by_field, kRfCount).c_str(), (unsigned long long)w.resolves_unpaired, (unsigned long long)w.resolves_never_run,
                (unsigned long long)w.resolve_depth_source, (unsigned long long)w.inl_decoded, (unsigned long long)w.inl_taken,
                (unsigned long long)w.inl_overflow, (unsigned long long)w.inl_paired, (unsigned long long)w.inl_replays, (unsigned long long)w.inl_unpaired,
                (unsigned long long)w.inl_desync, double(w.ns) / 1000.0 / double(frames ? frames : 1));
  g_e = M4eWindow{};
  return buf;
}

}  // namespace rex::graphics::vulkan

// =================================================================================================================================
// Session 95 (item N2): the scout's apply. The same grammar as DecodePackets / DecodeToken / DecodeInline / DecodeCopy / DecodeRegs above, on a private
// state and with none of their effects but the typed stores (see owned_record.h). Every store is the same store as StoreWord's field / bank branch.
namespace rex::graphics::vulkan {

namespace {

void ScoutStoreWord(OwnedRecordState& st, uint32_t reg, uint32_t v) {
  if (reg >= kRegCount) return;
  if (reg >= kFloatLo && reg < kFloatPixelLo) {
    st.vs_float[reg - kFloatLo] = v;
  } else if (reg >= kFloatPixelLo && reg < kFloatHi) {
    st.ps_float[reg - kFloatPixelLo] = v;
  } else if (reg >= kFetchLo && reg < kFetchHi) {
    st.fetch[reg - kFetchLo] = v;
  } else if (reg >= kBoolLo && reg < kBoolHi) {
    st.bool_loop[reg - kBoolLo] = v;
  } else if (g_reg_offset[reg] != kNone) {
    std::memcpy(reinterpret_cast<uint8_t*>(&st) + g_reg_offset[reg], &v, sizeof(v));
  }
}

// A run of n values: the banks in one copy per piece, everything else word by word.
void ScoutStoreRun(OwnedRecordState& st, uint32_t first, uint32_t n, const uint32_t* v) {
  uint32_t i = 0;
  while (i < n) {
    const uint32_t reg = first + i;
    if (reg >= kRegCount) return;
    uint32_t end = first + n;
    if (reg >= kFloatLo && reg < kFloatHi) {
      end = std::min(end, kFloatHi);
      if (reg < kFloatPixelLo) {
        const uint32_t len = std::min(end, kFloatPixelLo) - reg;
        std::memcpy(st.vs_float + (reg - kFloatLo), v + i, size_t(len) * sizeof(uint32_t));
      }
      if (end > kFloatPixelLo) {
        const uint32_t from = std::max(reg, kFloatPixelLo);
        std::memcpy(st.ps_float + (from - kFloatPixelLo), v + i + (from - reg), size_t(end - from) * sizeof(uint32_t));
      }
    } else if (reg >= kFetchLo && reg < kFetchHi) {
      end = std::min(end, kFetchHi);
      std::memcpy(st.fetch + (reg - kFetchLo), v + i, size_t(end - reg) * sizeof(uint32_t));
    } else if (reg >= kBoolLo && reg < kBoolHi) {
      end = std::min(end, kBoolHi);
      std::memcpy(st.bool_loop + (reg - kBoolLo), v + i, size_t(end - reg) * sizeof(uint32_t));
    } else {
      for (const uint32_t boundary : {kFloatLo, kFloatHi, kFetchHi, kBoolLo, kBoolHi}) {
        if (boundary > reg) {
          end = std::min(end, boundary);
          break;
        }
      }
      for (uint32_t r = reg; r < end; ++r) ScoutStoreWord(st, r, v[i + (r - reg)]);
    }
    i += end - reg;
  }
}

bool ScoutLoadWord(const OwnedRecordState& st, uint32_t reg, uint32_t* out) {
  if (reg >= kRegCount) return false;
  if (reg >= kFloatLo && reg < kFloatPixelLo) return *out = st.vs_float[reg - kFloatLo], true;
  if (reg >= kFloatPixelLo && reg < kFloatHi) return *out = st.ps_float[reg - kFloatPixelLo], true;
  if (reg >= kFetchLo && reg < kFetchHi) return *out = st.fetch[reg - kFetchLo], true;
  if (reg >= kBoolLo && reg < kBoolHi) return *out = st.bool_loop[reg - kBoolLo], true;
  if (g_reg_offset[reg] == kNone) return false;
  std::memcpy(out, reinterpret_cast<const uint8_t*>(&st) + g_reg_offset[reg], sizeof(*out));
  return true;
}

void ScoutShaderId(OwnedRecordState& st, uint32_t type, uint32_t addr, uint32_t dwords, const uint32_t* words) {
  OwnedRecordState::ShaderId& id = st.shader[type];
  id.address = addr;
  id.dwords = dwords;
  id.hash = XXH3_64bits(words, size_t(dwords) * sizeof(uint32_t));
}

void ScoutToken(OwnedRecordState& st, rex::memory::Memory* memory, const uint32_t* w, uint32_t count, const OwnedScoutApplyHooks& hooks) {
  if (count < 2 || w[0] != kTokenMagic) return;
  const uint32_t seq = w[1];
  st.token_seq = seq;
  uint32_t i = 2, item_index = 0;
  while (i < count) {
    const uint32_t header = w[i];
    const uint32_t kind = header >> 28;
    uint32_t used = 0;
    if (kind == 1) {
      const uint32_t n = header & 0x0FFFFFFF;
      used = 2 + n;
      if (n == 0 || used > count - i) return;
      ScoutStoreRun(st, w[i + 1], n, w + i + 2);
    } else if (kind == 2) {
      used = 3;
      if (used > count - i || (header & 0xF) > 1) return;
      const uint32_t type = header & 0xF;
      const uint32_t addr = w[i + 1] & ~0x3u;
      const uint32_t dwords = w[i + 2] & 0xFFFF;
      ScoutShaderId(st, type, addr, dwords, D3dShaderMicrocodePtrOf(memory, addr));
    } else if (kind == 3) {
      used = 6;
      if (used > count - i) return;
      ScoutStoreWord(st, XE_GPU_REG_VGT_INDX_OFFSET, w[i + 1]);
      st.draw_viz_query = w[i + 2];
      ScoutStoreWord(st, XE_GPU_REG_VGT_DRAW_INITIATOR, w[i + 3]);
      ScoutStoreWord(st, XE_GPU_REG_VGT_DMA_BASE, w[i + 4]);
      ScoutStoreWord(st, XE_GPU_REG_VGT_DMA_SIZE, w[i + 5]);
      st.crowd_id = (uint64_t(seq) << 16) | (item_index & 0xFFFF);
      if (hooks.draw) hooks.draw(hooks.ctx, st);
    } else {
      return;
    }
    i += used;
    ++item_index;
  }
}

void ScoutInline(OwnedRecordState& st, rex::memory::Memory* memory, const uint32_t* w, uint32_t count) {
  constexpr uint32_t kFixed = 21;
  if (count < kFixed || w[0] != d3d::kInlineTokenMagic) return;
  const uint32_t flags = w[2], nruns = w[11], nvalues = w[12], vs_inline = w[15], ps_inline = w[18], vbytes = w[20];
  const uint64_t need = uint64_t(kFixed) + 3ull * nruns + nvalues + vs_inline + ps_inline + ((uint64_t(vbytes) + 3) >> 2);
  if (need != count) return;
  if ((flags & d3d::kInlineHasVertexShader) && !w[13] && w[14] && w[14] > vs_inline) return;
  if ((flags & d3d::kInlineHasPixelShader) && !w[16] && w[17] && w[17] > ps_inline) return;
  const uint32_t* runs = w + kFixed;
  const uint32_t* values = runs + 3 * size_t(nruns);
  const uint32_t* vs_words = values + nvalues;
  const uint32_t* ps_words = vs_words + vs_inline;
  for (uint32_t r = 0; r < nruns; ++r) {
    const uint32_t first = runs[3 * r], n = runs[3 * r + 1], off = runs[3 * r + 2];
    if (uint64_t(off) + n > nvalues) return;
    ScoutStoreRun(st, first, n, values + off);
  }
  std::vector<uint32_t> raw;  // (inline shader words: raw guest bytes stored swapped twice; the old path hashes them byte-swapped back)
  const auto bind = [&](uint32_t type, uint32_t addr, uint32_t dwords, const uint32_t* inline_words, uint32_t inline_count) {
    if (!dwords) return;
    if (addr) {
      ScoutShaderId(st, type, addr, dwords, D3dShaderMicrocodePtrOf(memory, addr));
    } else {
      raw.resize(inline_count);
      for (uint32_t k = 0; k < inline_count; ++k) raw[k] = std::byteswap(inline_words[k]);
      ScoutShaderId(st, type, 0, dwords, raw.data());
    }
  };
  if (flags & d3d::kInlineHasVertexShader) bind(0, w[13], w[14], vs_words, vs_inline);
  if (flags & d3d::kInlineHasPixelShader) bind(1, w[16], w[17], ps_words, ps_inline);
  ScoutStoreWord(st, XE_GPU_REG_VGT_INDX_OFFSET, 0);
  ScoutStoreRun(st, XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + 6 * 31, 4, w + 7);  // (slot 31 words 4-5, the consumer's staging address, are NOT known here)
  for (uint32_t k = 0; k < 3; ++k) ScoutStoreWord(st, 0x5000 + k, 0);
  ScoutStoreWord(st, XE_GPU_REG_VGT_DRAW_INITIATOR, (w[3] & 0x3Fu) | (2u << 6) | (w[4] << 16));
  st.draw_viz_query = w[6];
}

void ScoutCopy(OwnedRecordState& st, const uint32_t* w, uint32_t count) {
  if (count < kD3dCopyHeaderDwords || w[0] != kD3dCopyMagic || count != kD3dCopyHeaderDwords + w[3]) return;
  const uint32_t kind = w[2];
  const uint32_t* sem = w + kD3dCopyHeaderDwords;
  if (kind == kD3dCopyClear) {
    ScoutStoreWord(st, XE_GPU_REG_VGT_DRAW_INITIATOR, 8u | (2u << 6) | (3u << 16));
  } else if (kind == kD3dCopyResolve && w[3] >= d3d_resolve_sem::kDwords) {
    namespace rs = d3d_resolve_sem;
    static const uint32_t kColorInfoRegs[4] = {XE_GPU_REG_RB_COLOR_INFO, XE_GPU_REG_RB_COLOR1_INFO, XE_GPU_REG_RB_COLOR2_INFO, XE_GPU_REG_RB_COLOR3_INFO};
    const uint32_t src = sem[rs::kCopyControl] & 7u;
    ScoutStoreWord(st, XE_GPU_REG_RB_SURFACE_INFO, sem[rs::kSurfaceInfo]);
    if (src < 4) ScoutStoreWord(st, kColorInfoRegs[src], sem[rs::kColorInfo]);
    ScoutStoreWord(st, XE_GPU_REG_RB_DEPTH_INFO, sem[rs::kDepthInfo]);
    ScoutStoreWord(st, XE_GPU_REG_PA_SU_VTX_CNTL, sem[rs::kVtxCntl]);
  }
}

}  // namespace

void OwnedRecordScoutPrepare() { BuildTable(); }

void OwnedRecordScoutApply(OwnedRecordState& st, rex::memory::Memory* memory, uint32_t kind, const uint32_t* p, uint32_t dwords,
                           const OwnedScoutApplyHooks& hooks) {
  if (kind == kNrRegs) {
    for (uint32_t i = 0; i + 4 <= dwords; i += 4) {
      const uint32_t op = p[i], reg = p[i + 1], a = p[i + 2];
      if (op >= d3d_tex::kOpDefine0 && op <= d3d_tex::kOpBind) {
        continue;  // the texture object table: not typed state
      } else if (op >= d3d_noring_field::kRegsOpBinMaskLo && op <= d3d_noring_field::kRegsOpBinSelectHi) {
        if (op == d3d_noring_field::kRegsOpBinMaskLo) st.bin_mask = (st.bin_mask & 0xFFFFFFFF00000000ull) | a;
        else if (op == d3d_noring_field::kRegsOpBinMaskHi) st.bin_mask = (st.bin_mask & 0xFFFFFFFFull) | (uint64_t(a) << 32);
        else if (op == d3d_noring_field::kRegsOpBinSelectLo) st.bin_select = (st.bin_select & 0xFFFFFFFF00000000ull) | a;
        else st.bin_select = (st.bin_select & 0xFFFFFFFFull) | (uint64_t(a) << 32);
      } else if (op == d3d_noring_field::kRegsOpRmw) {
        uint32_t v = 0;
        if (ScoutLoadWord(st, reg & 0x1FFF, &v)) {
          const uint32_t nv = (v & a) | p[i + 3];
          ScoutStoreWord(st, reg & 0x1FFF, nv);
        }
      } else {
        ScoutStoreWord(st, reg, a);
      }
    }
    return;
  }
  if (kind != kNrPackets) return;
  uint32_t i = 0;
  while (i < dwords) {
    const uint32_t header = p[i];
    if ((header >> 30) == 3) {
      const uint32_t opcode = (header >> 8) & 0x7F;
      const uint32_t count = ((header >> 16) & 0x3FFF) + 1;
      if (count > dwords - i - 1) return;
      if (opcode == 0x7C) ScoutToken(st, memory, p + i + 1, count, hooks);
      else if (opcode == 0x7D) ScoutCopy(st, p + i + 1, count);
      else if (opcode == 0x7E) ScoutInline(st, memory, p + i + 1, count);
      i += 1 + count;
    } else if ((header >> 30) == 2) {
      ++i;
    } else {
      return;
    }
  }
}

}  // namespace rex::graphics::vulkan
