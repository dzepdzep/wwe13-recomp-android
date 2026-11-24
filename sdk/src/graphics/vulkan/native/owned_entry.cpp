// wwe2k14 patch (Session 92, M2b; docs\native\d3d\S92_M2_DESIGN.md sections 2.2, 2.3 and 3 "M2b", contract docs\native\d3d\S92_M2BC_CONTRACT.md):
// the owned entry and the surface registry of the owned-only latch (gpu_native_owned_only, owned_steps.cpp).
//
// Under the latch nothing is handed to a truth path - there is none to hand to. A draw the command processor offers is claimed by OwnedClaim (always
// true: a draw it cannot record is claimed and dropped, counted by reason) and entered by OwnedEnter (OwnedTakeDraw, else OwnedDropDraw); the truth
// claim (ClaimTruth) and the truth recorder (TruthRecordDraw) are not called. A surface is identified by the game's own surface words - base, pitch in
// 80-sample tiles, class, format; what the bound surface object holds - never by the EDRAM tiles it would share with another surface on the console
// (CLAUDE.md hard rule 6). The registry (owned_surfaces_) is a list of those keys: no image, no tile, no owner map. The slot stays an int16_t, so every
// owned-code use of Draw::truth_color / truth_depth and OwnedPass::color_slot / depth_slot keeps its meaning; only the table behind it changes
// (owned_surf_mode_: the whole frame reads one table).

#include <rex/graphics/vulkan/native/native_renderer.h>
#include "native_truth_reasons.h"

#include <algorithm>
#include <array>
#include <vector>
#include <string_view>

#include <rex/cvar.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/logging.h>

#include "owned_front_draw.h"
#include "shader_swap.h"  // Session 94: InstanceWorldLatched

REXCVAR_DECLARE(uint32_t, gpu_d3d_owned_front);
REXCVAR_DECLARE(int32_t, gpu_native_truth_debug_draws);
REXCVAR_DECLARE(bool, gpu_native_truth_async_pipelines);
REXCVAR_DECLARE(bool, async_shader_compilation);
REXCVAR_DECLARE(bool, gpu_native_truth_pipeline_wait_offscreen);
REXCVAR_DECLARE(int32_t, gpu_native_truth_pipeline_skip_min_pitch);
REXCVAR_DECLARE(int32_t, gpu_native_truth_pipeline_skip_max_pitch);

namespace rex::graphics::vulkan::native {

namespace {

// (The reason codes are native_truth_reasons.h's.)
constexpr size_t kMaxSurfaces = 512;  // (no image behind a key: far more than the truth images' 96)

struct EntryStats {
  uint64_t claims = 0, lean = 0, repeat = 0;
  uint32_t surfaces_new = 0, would_alias = 0, table_full = 0;
};
EntryStats g_entry;

}  // namespace

// The window line's figures (owned_steps.cpp OwnedFrameSummary).
void OwnedEntryCounters(uint64_t& claims, uint64_t& lean, uint64_t& repeat, uint32_t& surfaces_new, uint32_t& would_alias, uint32_t& table_full, bool reset) {
  claims = g_entry.claims;
  lean = g_entry.lean;
  repeat = g_entry.repeat;
  surfaces_new = g_entry.surfaces_new;
  would_alias = g_entry.would_alias;
  table_full = g_entry.table_full;
  if (reset) g_entry = EntryStats();
}

int16_t NativeRenderer::OwnedFindSurface(uint32_t base, uint32_t pitch_tiles, uint32_t kind, uint32_t format) {
  for (size_t i = 0; i < owned_surfaces_.size(); ++i) {
    const OwnedSurface& s = owned_surfaces_[i];
    if (s.base == base && s.pitch_tiles == pitch_tiles && s.kind == kind && s.format == format) return int16_t(i);
  }
  if (owned_surfaces_.size() >= kMaxSurfaces) {
    // (review F3: "table full" counts keys refused, once each - a small set of the refused keys; the draws they drop are counted by OwnedNoteClaimDrop)
    static std::vector<std::array<uint32_t, 4>> refused;
    const std::array<uint32_t, 4> key = {base, pitch_tiles, kind, format};
    if (std::find(refused.begin(), refused.end(), key) == refused.end() && refused.size() < 256) {
      refused.push_back(key);
      ++g_entry.table_full;
    }
    return -1;
  }
  // A new key. Diagnostic (design 2.3, last bullet): on the console a surface at this base with other words would have shared its pixels with this one;
  // here they are two surfaces and the new one starts defined. Counted as an upper bound of "would-have-aliased" (a key sharing a base with an earlier key,
  // whether or not its first use reads anything) - the first ones named. Unconfirmed which of them read before a write: judged by pictures.
  for (const OwnedSurface& s : owned_surfaces_) {
    if (s.base != base || s.base == kTruthScratchBase) continue;
    ++g_entry.would_alias;
    static uint32_t logged = 0;
    if (logged < 12) {
      ++logged;
      REXGPU_INFO("[native][owned-only] frame {}: new surface key base {} pitch {} kind {} format {} shares its base with base {} pitch {} kind {} format {} "
                  "(on the console they would share pixels; here two surfaces, the new one starts defined)",
                  frame_, base, pitch_tiles, kind, format, s.base, s.pitch_tiles, s.kind, s.format);
    }
    break;  // (one count per new key)
  }
  OwnedSurface& added = owned_surfaces_.emplace_back();
  added.base = base;
  added.pitch_tiles = pitch_tiles;
  added.kind = kind;
  added.format = format;
  added.first_frame = uint32_t(frame_);
  ++g_entry.surfaces_new;
  return int16_t(owned_surfaces_.size() - 1);
}

int16_t NativeRenderer::SurfaceSlot(uint32_t base, uint32_t pitch_tiles, uint32_t kind, uint32_t format) {
  return owned_surf_mode_ ? OwnedFindSurface(base, pitch_tiles, kind, format) : TruthFindImage(base, pitch_tiles, kind, format);
}

NativeRenderer::OwnedSurface NativeRenderer::OwnedSlotKey(int16_t slot) const {
  OwnedSurface key;
  if (slot < 0) return key;
  if (owned_surf_mode_) {
    if (size_t(slot) < owned_surfaces_.size()) key = owned_surfaces_[slot];
  } else if (size_t(slot) < truth_images_.size()) {
    const TruthImage& t = truth_images_[slot];
    key.base = t.base;
    key.pitch_tiles = t.pitch_tiles;
    key.kind = t.kind;
    key.format = t.format;
  }
  return key;
}

int NativeRenderer::OwnedSourceDrawn(uint32_t base, uint32_t row_tiles, bool depth) const {
  // Whether a pass of ours ever drew the surface a copy reads (the registry key: base and pitch, any class of the kind; M2c's "find out first").
  int best = 0;
  for (const OwnedSurface& s : owned_surfaces_) {
    if (s.base != base || (s.kind == 3) != depth) continue;
    if (s.row_tiles() != row_tiles && s.pitch_tiles != row_tiles) continue;
    if (!s.last_draw_frame) best = std::max(best, 1);
    else best = std::max(best, s.last_draw_frame == uint32_t(frame_) + 1 ? 3 : 2);
  }
  return best;
}

bool NativeRenderer::OwnedClaim(Draw& draw) {
  ++g_entry.claims;
  // Session 92, M4b (gpu_d3d_owned_front bit 1): the same description built from the record decoder's typed state, compared with this one.
  if (REXCVAR_GET(gpu_d3d_owned_front) & 2u) OwnedFrontShadowDraw(command_processor_, draw, kOwnedFrontFull);
  // Always claimed: a refusal (a TruthReason - the limits of our pipelines) claims the draw and OwnedEnter drops it, counted and named by the reason.
  const auto refuse = [&](uint32_t reason) {
    OwnedNoteClaimDrop(reason, draw);
    draw.claim_refused = uint8_t(reason);
    draw.truth = draw.post = draw.owned = true;
    draw.truth_epoch = truth_claim_epoch_;
    return true;
  };
  const uint32_t reason = TruthReason(draw, true);
  if (reason) return refuse(reason);
  if (const int32_t limit = REXCVAR_GET(gpu_native_truth_debug_draws); limit > 0 && truth_frame_draws_ >= limit) return refuse(kTruthDebug);
  ++truth_frame_draws_;
  const uint32_t msaa = (draw.surface_info >> 16) & 3;  // 0 1x, 1 2x, 2 4x
  const uint32_t sx = msaa >= 2 ? 2 : 1, sy = msaa >= 1 ? 2 : 1;
  const uint32_t pitch_tiles = std::max(1u, ((draw.surface_info & 0x3FFF) * sx + 79) / 80);
  const bool writes_color = (draw.color_write_mask & 0xF) != 0;
  const bool uses_depth = draw.raster.depth_test || draw.raster.depth_write || draw.raster.stencil;
  int16_t color = -1, depth = -1;
  if (writes_color) {
    color = OwnedFindSurface(draw.color_info & 0xFFF, pitch_tiles, TruthClass(draw), (draw.color_info >> 16) & 0xF);
    if (color < 0) return refuse(kTruthImages);
    owned_surfaces_[color].last_draw_frame = uint32_t(frame_) + 1;
  }
  if (uses_depth) {
    depth = OwnedFindSurface(draw.depth_info & 0xFFF, pitch_tiles, 3, (draw.depth_info >> 16) & 1);
    if (depth < 0) return refuse(kTruthImages);
    owned_surfaces_[depth].last_draw_frame = uint32_t(frame_) + 1;
  }
  // (No stencil tile proof: stencil is ours - kept in our depth images; a stencil draw on a surface whose stencil we never wrote starts from 0, counted by
  // OwnedCountUnprovenStencil's successors in the pass code.)
  draw.truth = true;
  draw.post = true;
  draw.truth_color = color;
  draw.truth_depth = depth;
  draw.truth_class = uint8_t(writes_color ? TruthClass(draw) : 1u);
  draw.truth_sx = uint8_t(sx);
  draw.truth_sy = uint8_t(sy);
  // The async pipeline skip, as the truth claim has it (Session 47 / 55 / 57 / 58): a pipeline being made on the worker thread - the draw is claimed and
  // skipped (OwnedDropDraw counts "skipped for no pipeline") - only for a screen-wide surface; a narrower one waits for its pipeline instead.
  bool pending = false;
  const bool async = REXCVAR_GET(gpu_native_truth_async_pipelines) && REXCVAR_GET(async_shader_compilation);
  const bool may_skip = (pitch_tiles >= uint32_t(REXCVAR_GET(gpu_native_truth_pipeline_skip_min_pitch)) &&
                         pitch_tiles <= uint32_t(REXCVAR_GET(gpu_native_truth_pipeline_skip_max_pitch))) ||
                        !REXCVAR_GET(gpu_native_truth_pipeline_wait_offscreen);
  const size_t pipelines_before = pipelines_.size();
  draw.pipeline = GetPipeline(draw, async && may_skip ? &pending : nullptr);
  if (async && !may_skip && pipelines_.size() != pipelines_before) ++truth_counts_.waited;
  if (pending) {
    ++truth_counts_.compiling;
    draw.owned = true;
    draw.truth_epoch = truth_claim_epoch_;
    return true;
  }
  if (draw.pipeline == VK_NULL_HANDLE) {
    draw.truth = draw.post = false;
    return refuse(kTruthNoPipeline);  // (as the truth claim under the latch had it)
  }
  draw.owned = true;
  draw.truth_epoch = truth_claim_epoch_;
  return true;
}

// A repeat has its base's registers but for the constants (the command processor's repeat path): whatever OwnedClaim found for the base holds for it. The
// registry only grows, so a slot never goes stale within a mode (the epoch is bumped when the mode changes). No allocation: this is hot (CPU).
bool NativeRenderer::OwnedClaimRepeat(Draw& draw) {
  if (!draw.truth || !draw.owned || !truth_ || !replacing_ || draw.truth_epoch != truth_claim_epoch_ || draw.occlusion_query ||
      draw.pipeline == VK_NULL_HANDLE || REXCVAR_GET(gpu_native_truth_debug_draws) > 0 || (draw.raster.stencil && !draw.fast_clear)) {
    return Claim(draw);
  }
  if (REXCVAR_GET(gpu_d3d_owned_front) & 2u) OwnedFrontShadowDraw(command_processor_, draw, kOwnedFrontRepeat);
  ++g_entry.repeat;
  declined_pending_ = false;
  claim_retry_ = false;
  draw.declined = false;
  ++truth_frame_draws_;
  return true;
}

// Session 95 (r3cmd, gpu_d3d_draw_record): a draw with its record's claim: what OwnedClaim found under the same inputs holds (TruthReason and the surfaces are functions of the Draw; the registry only
// grows, a slot never goes stale within a mode, the epoch moves with the mode). Its two per-draw effects are kept: the frame count and the surfaces' last draw.
bool NativeRenderer::OwnedClaimRecord(Draw& draw) {
  if (!owned_surf_mode_ || !draw.truth || !draw.owned || !truth_ || !replacing_ || draw.occlusion_query ||
      draw.pipeline == VK_NULL_HANDLE || REXCVAR_GET(gpu_native_truth_debug_draws) > 0 || draw.fast_clear || draw.xfb_buffer != VK_NULL_HANDLE) {
    return false;
  }
  ++g_entry.claims;
  draw.truth_epoch = truth_claim_epoch_;  // (it moves every frame: what it guards under the truth path is the truth images' owners; a surface slot does not go stale)
  declined_pending_ = false;
  claim_retry_ = false;
  draw.declined = false;
  ++truth_frame_draws_;
  if (draw.truth_color >= 0) owned_surfaces_[size_t(draw.truth_color)].last_draw_frame = uint32_t(frame_) + 1;
  if (draw.truth_depth >= 0) owned_surfaces_[size_t(draw.truth_depth)].last_draw_frame = uint32_t(frame_) + 1;
  return true;
}

bool NativeRenderer::OwnedClaimLean(Draw& draw) {
  if (!draw.truth || !draw.owned || !truth_ || !replacing_ || draw.truth_epoch != truth_claim_epoch_ || draw.occlusion_query ||
      draw.pipeline == VK_NULL_HANDLE || REXCVAR_GET(gpu_native_truth_debug_draws) > 0 || (draw.raster.stencil && !draw.fast_clear) || draw.fast_clear ||
      draw.xfb_buffer != VK_NULL_HANDLE) {  // (Session 61: a captured draw's copy would capture into its range again)
    return false;
  }
  if (REXCVAR_GET(gpu_d3d_owned_front) & 2u) OwnedFrontShadowDraw(command_processor_, draw, kOwnedFrontLean);
  ++g_entry.lean;
  declined_pending_ = false;
  claim_retry_ = false;
  draw.declined = false;
  ++truth_frame_draws_;
  return true;
}

void NativeRenderer::OwnedEnter(const Draw& draw) {
  // TruthRecordDraw's latch branch, without the truth recorder: the owned frame takes the draw, else it is dropped (counted, named). A claim that was
  // refused (a TruthReason) is a drop at once; a draw with no pipeline yet is skipped there, as ever.
  // Session 94 review (world instancing): a draw that is dropped or not recorded must not leave the previous draw's merge
  // flag set (else a repeat of THIS draw would add an instance to the earlier, unrelated draw). OwnedRecord sets it again for
  // a recorded instanceable draw. Gated so the off path is unchanged (mirrors native_truth.cpp's clear at its draw start).
  if (shader_swap::InstanceWorldLatched()) truth_last_recorded_ = false;
  if (!draw.claim_refused && OwnedTakeDraw(draw)) return;
  OwnedDropDraw(draw);
}

}  // namespace rex::graphics::vulkan::native
