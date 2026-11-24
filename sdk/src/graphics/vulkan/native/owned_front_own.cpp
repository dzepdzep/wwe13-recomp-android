// wwe2k14 patch (Session 92, M4f-C): the typed draw (TypedDraw, DrawFromBase - the one lean body, CrowdMember) and the cut's executor entries.
// Moved out of d3d_direct.cpp so the REX_TRUE_NATIVE_ONLY proof build can leave that file (the old record-fed lean path, the token draw handler) out.
// DrawFromBase is still called by the old lean path (d3d_direct.cpp Lean): one body.
#include "diag_layer.h"  // Session 95 (the diagnostic run)
#include <rex/graphics/vulkan/native/native_renderer.h>

#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/d3d_direct.h>
#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include "shader_swap.h"  // wwe2k14 patch (Session 87, pixel skip)
#include <rex/graphics/registers.h>
#include <rex/graphics/util/bit_inline.h>
#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/memory/ring_buffer.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <bit>
#include <utility>
#include <vector>

#include "../../shadow_chain.h"  // wwe_shadow_chain::g (the lean path's own gate for the shadow chain)
#include <rex/hash.h>  // (Session 95, r3gpu: XXH3)
#include "d3d_feed.h"  // (Session 87, stage 3) the record's register image
#include "owned_record.h"  // (Session 92, M4b) the typed state of this draw item
#include "owned_front_draw.h"  // (Session 92, M4d) the crowd shadow
#include "../../crowd_member_hook.h"  // (Session 92, M4f-B)

#include "d3d_direct_access.h"
#include "phase0_census.h"  // (Session 93, gpu_d3d_phase0_census: log-only census)
#include "owned_shadow_skip.h"  // (Session 95, item K: gpu_native_owned_shadow_skip)
#include "cull_ours.h"  // (Session 93, gpu_d3d_cull_ours: our off-screen draws are not drawn)
#include "owned_scout.h"  // (Session 95, item N2: cover / bounds results made ahead by jobs)
#include "decouple_blend.h"  // (Session 94, gpu_d3d_decouple phase 1B: the draw match across ticks + the blended view 0 constants)
#include "inline_ring.h"  // (Session 95, item P: gpu_d3d_inline_vertex_ring)
#include "index_pool.h"  // (Session 95)
#include "skin_once.h"  // (Session 95: gpu_native_skin_once)
#include "draw_record.h"  // (Session 95, r3cmd: gpu_d3d_draw_record)
#include "draw_slot.h"  // (Session 95: gpu_d3d_draw_slot)
#include "vertex_input.h"  // (Session 95: gpu_native_vertex_input)
#include "vertex_mirror.h"  // (Session 95: gpu_native_skin_once's vertex source)
#include "native_truth_reasons.h"  // (Session 95: kTruthNoPipeline)
#include <rex/graphics/util/draw.h>
#include <rex/graphics/free_counters.h>
#include <rex/graphics/d3d_copy.h>
#include <rex/graphics/d3d_inline_draw.h>
REXCVAR_DECLARE(bool, gpu_d3d_own_full);
REXCVAR_DECLARE(int32_t, gpu_tex_bind_memo);  // (Session 95, r6tex: pipeline\texture\cache.cpp)
REXCVAR_DECLARE(int32_t, gpu_d3d_draw_slot);  // (Session 95: draw_slot.cpp)
REXCVAR_DECLARE(bool, gpu_sampler_mip_point);
REXCVAR_DECLARE(uint32_t, gpu_d3d_wall);  // (Session 95: d3d_noring.cpp)
namespace rex::graphics {
bool SceneCrowdPreTakeFoundNothing();  // (Session 95: command_processor.cpp)
}
REXCVAR_DECLARE(bool, gpu_d3d_own_crowd);
REXCVAR_DECLARE(uint32_t, gpu_d3d_owned_front);
REXCVAR_DECLARE(bool, gpu_d3d_typed_memo_reuse);
REXCVAR_DECLARE(bool, gpu_d3d_typed_repeat);
REXCVAR_DECLARE(bool, gpu_d3d_phase0_census);
REXCVAR_DECLARE(bool, gpu_d3d_cull_ours);  // (Session 93: defined in cull_ours.cpp)
REXCVAR_DECLARE(bool, gpu_d3d_inline_vertex_ring);  // (Session 95: defined in inline_ring.cpp)
REXCVAR_DECLARE(bool, gpu_native_vinput_census);  // (Session 95, P0: defined in index_pool.cpp)
REXCVAR_DECLARE(bool, gpu_native_skin_bind_lean);  // (Session 95, r3gpu: defined in skin_once.cpp)
REXCVAR_DECLARE(bool, gpu_native_vertex_mirror_verify);  // (Session 95: the same)
REXCVAR_DECLARE(int32_t, gpu_d3d_decouple);  // (Session 94: ui/vulkan/vulkan_device.cpp)
REXCVAR_DECLARE(double, gpu_wide_factor);
REXCVAR_DECLARE(double, gpu_tall_factor);
REXCVAR_DECLARE(bool, gpu_wide_all);  // (Session 93: defined in phase0_census.cpp)
REXCVAR_DECLARE(bool, gpu_d3d_typed_timers);
REXCVAR_DECLARE(bool, gpu_d3d_typed_lean_bind);  // (Session 94, B7: defined in owned_front_draw.cpp)
REXCVAR_DECLARE(bool, gpu_d3d_typed_memo_verify);
REXCVAR_DEFINE_BOOL(gpu_d3d_sampler_memo_fix, false, "GPU/Native",
                    "Session 95: the typed draw's sampler-parameter memo (gpu_d3d_typed_lean_bind) hits. Off: it never hit under TRUE NATIVE (the old IssueDraw's repeat_dirty_ fetch bit "
                    "is never cleared by typed draws, so the hit test always failed). On: the memo ignores that bit (the six fetch words of every binding are compared anyway) and keeps "
                    "a direct-mapped table of 16 entries per stage indexed by shader, so alternating shaders hit too (needs gpu_d3d_typed_lean_bind); a hit rewrites the processor's sampler list to what a rebuild gives. Same parameters, same picture.");
REXCVAR_DECLARE(bool, gpu_allow_invalid_fetch_constants);
REXCVAR_DECLARE(int32_t, anisotropic_override);
REXCVAR_DECLARE(bool, gpu_native_lean_draws);
REXCVAR_DECLARE(bool, gpu_native_lean_draws_verify);
REXCVAR_DECLARE(bool, gpu_native_lean_dirty_census);
REXCVAR_DECLARE(bool, gpu_native_lean_multi_base);
REXCVAR_DECLARE(bool, gpu_native_lean);
REXCVAR_DECLARE(bool, gpu_native_truth_host_rt);
REXCVAR_DECLARE(bool, gpu_det_skip_host_draws);
REXCVAR_DECLARE(bool, gpu_shadow_half_rate);
REXCVAR_DECLARE(bool, gpu_depth_of_field);  // (Session 95: post lite)
REXCVAR_DECLARE(bool, vulkan_repeat_draw_fast);
REXCVAR_DECLARE(int32_t, gpu_scene_crowd_density);
REXCVAR_DECLARE(int32_t, gpu_debug_draw_census_count);
REXCVAR_DEFINE_BOOL(gpu_d3d_direct, false, "GPU/Native",
                    "Session 87 (D3D layer F4, experimental, needs the game's D3D layer): the token's draw items are drawn by a lean-draw variant that takes its "
                    "knowledge from the record instead of the draw packet path. A draw it cannot take goes the stock way (counted, [d3d][direct]). (Defined here, "
                    "with the typed draw, because the base library reads it too; the handler is d3d_direct.cpp, left out of the TRUE NATIVE proof build.)");
REXCVAR_DEFINE_BOOL(gpu_d3d_direct_verify, false, "GPU/Native",
                    "Session 87 (D3D layer F4): with gpu_d3d_direct, every vertex-range stamp hit is checked against "
                    "SharedMemory::IsRangeUpToDate (a mismatch is logged, counted and requested for real).");

REXCVAR_DEFINE_INT32(gpu_d3d_draw_record, 0, "GPU/Native",
                      "Session 95 (r3cmd): the per-draw record (draw_record.h): a typed draw whose inputs but the float constants equal an earlier frame's draw takes what that draw derived (preparation, system constants, "
                      "vertex plan, claim) from a record, each part under its own validity check. 0 = off; 1 = on; 2 = verify: every draw goes the full way and is compared with its record ([d3d][record]).");
REXCVAR_DEFINE_BOOL(gpu_d3d_typed_latch, false, "GPU/Native",
                    "Session 95: TypedDraw / DrawFromBase read their per-draw settings (22 cvars, 690 draws a frame) once a frame instead of once a draw; a setting changed "
                    "mid-frame takes effect on the next frame. Same values, same picture. Off = every draw reads them live, as before.");

namespace rex::graphics::vulkan {

extern bool g_update_bindings_no_bind;  // (Session 95, r3cmd: command_processor.cpp)

namespace {
DirectCensus& g_c = g_direct_census;
// Session 95 (gpu_d3d_typed_latch): the settings of the typed draw, read once a frame on the command thread (a job that runs a draw would own its own copy).
struct TypedSettings {
  uint64_t frame = UINT64_MAX;
  bool typed_repeat, timers, phase0, cull_ours, lean_draws, lean_verify, lean_census, half_rate, lean, host_rt, repeat_fast, memo_reuse, lean_bind, memo_verify, memo_fix, invalid_fetch, direct_verify;
  int32_t census_count, crowd_density, decouple, aniso, draw_record, draw_slot;
  uint32_t owned_front;
};
const TypedSettings& TypedSettingsNow(uint64_t frame) {
  static TypedSettings t;
  if (t.frame == frame) return t;
  t.typed_repeat = REXCVAR_GET(gpu_d3d_typed_repeat);
  t.timers = REXCVAR_GET(gpu_d3d_typed_timers);
  t.phase0 = REXCVAR_GET(gpu_d3d_phase0_census);
  t.cull_ours = REXCVAR_GET(gpu_d3d_cull_ours);
  t.lean_draws = REXCVAR_GET(gpu_native_lean_draws);
  t.lean_verify = REXCVAR_GET(gpu_native_lean_draws_verify);
  t.lean_census = REXCVAR_GET(gpu_native_lean_dirty_census);
  t.half_rate = REXCVAR_GET(gpu_shadow_half_rate);
  t.lean = REXCVAR_GET(gpu_native_lean);
  t.host_rt = REXCVAR_GET(gpu_native_truth_host_rt);
  t.repeat_fast = REXCVAR_GET(vulkan_repeat_draw_fast);
  t.memo_reuse = REXCVAR_GET(gpu_d3d_typed_memo_reuse);
  t.lean_bind = REXCVAR_GET(gpu_d3d_typed_lean_bind);
  t.memo_verify = REXCVAR_GET(gpu_d3d_typed_memo_verify);
  t.memo_fix = REXCVAR_GET(gpu_d3d_sampler_memo_fix);
  t.invalid_fetch = REXCVAR_GET(gpu_allow_invalid_fetch_constants);
  t.direct_verify = REXCVAR_GET(gpu_d3d_direct_verify);
  t.census_count = REXCVAR_GET(gpu_debug_draw_census_count);
  t.crowd_density = REXCVAR_GET(gpu_scene_crowd_density);
  t.decouple = REXCVAR_GET(gpu_d3d_decouple);
  t.aniso = REXCVAR_GET(anisotropic_override);
  t.owned_front = REXCVAR_GET(gpu_d3d_owned_front);
  t.draw_record = REXCVAR_GET(gpu_d3d_draw_record);
  t.draw_slot = REXCVAR_GET(gpu_d3d_draw_slot);
  t.frame = frame;
  static bool logged = false;
  if (!logged) {
    logged = true;
    REXGPU_INFO("[d3d][typed] latch: the typed draw's settings are read once a frame (gpu_d3d_typed_latch)");
  }
  return t;
}
// The setting `cvar`: the latched copy `field` with gpu_d3d_typed_latch on, the live cvar otherwise. `lat` is null when the switch is off.
#define TLV(lat, field, cvar) ((lat) ? (lat)->field : REXCVAR_GET(cvar))
std::array<RangeStamp, 96>& g_stamps = g_direct_stamps;
}  // namespace

REXCVAR_DEFINE_BOOL(gpu_native_post_lite_drop, false, "GPU/Native",
                    "Session 95 (item R, P1; set by wwe2k14_post_lite): the post passes whose output the last match / entrance combine did not use "
                    "are not drawn, and their copies are skipped (the page keeps the last image): the star glare passes (75641266, 9C84E33D) while "
                    "c32.x of the entrance combine is 0, the bloom + glare chain while c31.x and c32.x are 0, the depth-of-field downsample and blur "
                    "(A7AD68BE, A5C9AC54) while gpu_depth_of_field is off. The passes draw again from the frame after a constant turns nonzero "
                    "(that one frame reads the page as last copied, which can be many frames old). The combines are never touched. Only with the TRUE NATIVE draw path");

REXCVAR_DEFINE_BOOL(gpu_native_post_lite_near, false, "GPU/Native",
                    "Session 95 (default off; with gpu_native_post_lite_drop): the entrance's second downsample + blur pair (9C85DA5F, 632189BA on the 320-wide surface) is not drawn while the last "
                    "entrance combine's near blur cannot reach a pixel (c21.x <= 0, depth never past it) and its fetch 2 reads that pair's page: Bryan's frame draws one pair before the wrestlers "
                    "(the far blur, fetch 6) and one after them that nothing uses. The pair draws again from the frame after c21.x turns positive");

namespace {
// (Session 95, post lite) What the last combine did not use, and what was dropped. Command thread only; a job would own this struct.
struct PostLiteState {
  // (gpu_native_post_lite_near) This frame's pairs in order: the pages their blurs were copied to; the pair the last combine's near blur reads.
  static constexpr uint32_t kPairs = 4;
  uint32_t blur_page[kPairs] = {}, blur_seen = 0, scene_page = 0;
  uint32_t last_blur_page[kPairs] = {}, last_blurs = 0;
  bool blur_copy_pending = false;
  int32_t near_pair = -1;
  int32_t near_down = -1;  // the pair whose downsample was dropped (the blur is dropped only with it)
  uint64_t near_dropped = 0;
  bool near_logged = false;
  uint64_t combine_frame = 0;  // the frame the last combine was seen (0: none)
  bool glare_off = false, bloom_off = false, entrance = false;  // (entrance: the last combine was the entrance's)
  uint64_t frame = 0;
  uint32_t frame_drops = 0;
  uint64_t drops_sum = 0, frames_with_drops = 0, frames_seen = 0, glare = 0, bloom = 0, dof = 0;
  uint64_t next_log = 0;
  bool logged = false;
};
PostLiteState g_post_lite;
constexpr uint64_t kPlEntranceCombine = 0x4E2CD8031D93DE3Eull, kPlMatchCombine = 0x1233F68C15FDF948ull;
}  // namespace

// Session 95 (r3cmd, gpu_d3d_plan_census: log only): how much of a frame's draw list repeats frame to frame, which decides what a per-draw record can skip.
REXCVAR_DEFINE_BOOL(gpu_d3d_plan_census, false, "GPU/Native",
                    "Session 95 (r3cmd, log only): per 600 frames, how many typed draws have the inputs of a draw of the frame before (state, shaders, draw words, used fetch slots, bool / loop words), and of those how "
                    "many have the same vertex / pixel float constants. Costs CPU; for measuring. [d3d][plan] census");
namespace {
struct PlanCensus {
  struct E { uint64_t frame = 0, prev_frame = 0, vs = 0, ps = 0; uint32_t uses = 0; };
  std::unordered_map<uint64_t, E> map;
  std::unordered_map<uint64_t, uint64_t> alt[3];
  uint64_t alt_last[3] = {}, alt_earlier[3] = {};
  uint64_t draws = 0, seen_last = 0, seen_earlier = 0, same_vs = 0, same_ps = 0, same_both = 0, dup_in_frame = 0, last_print = 0, frames = 0, last_frame = 0, epoch_moves = 0, last_epoch = 0;
} g_pc;
uint64_t PlanCensusFloats(const uint32_t* bank, const uint64_t* bitmap) {
  uint64_t h = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    uint64_t m = bitmap[i];
    while (m) {
      const uint32_t b = uint32_t(std::countr_zero(m));
      m &= m - 1;
      h = XXH3_64bits_withSeed(bank + (i * 64 + b) * 4, 16, h);
    }
  }
  return h;
}
void PlanCensusDraw(uint64_t frame, uint64_t epoch, const OwnedRecordState& s, const VulkanShader* vs, const VulkanShader* ps) {
  PlanCensus& c = g_pc;
  if (frame != c.last_frame) {
    c.last_frame = frame;
    ++c.frames;
  }
  if (epoch != c.last_epoch) {
    c.last_epoch = epoch;
    ++c.epoch_moves;
  }
  uint64_t h = OwnedFrontStateHashOf(s);
  const uint32_t words[] = {uint32_t(s.shader[0].hash), uint32_t(s.shader[0].hash >> 32), uint32_t(s.shader[1].hash), uint32_t(s.shader[1].hash >> 32), s.draw_initiator.value, s.dma_base, s.dma_size.value, s.indx_offset.value};
  h = XXH3_64bits_withSeed(words, sizeof(words), h);
  h = XXH3_64bits_withSeed(s.bool_loop, sizeof(s.bool_loop), h);
  h = XXH3_64bits_withSeed(s.fetch, sizeof(s.fetch), h);
  // (the same with the fetch slots the shaders use only, and without the render state / without the fetch words: which part makes a draw differ)
  {
    uint32_t tex = 0, vf[3] = {};
    if (vs) {
      tex |= vs->UsedTextureMaskFor(false) | vs->UsedTextureMaskFor(true);
      for (uint32_t i = 0; i < 3; ++i) vf[i] |= vs->constant_register_map().vertex_fetch_bitmap[i];
    }
    if (ps) {
      tex |= ps->UsedTextureMaskFor(false) | ps->UsedTextureMaskFor(true);
      for (uint32_t i = 0; i < 3; ++i) vf[i] |= ps->constant_register_map().vertex_fetch_bitmap[i];
    }
    uint64_t hu = XXH3_64bits(words, sizeof(words));
    const uint64_t hw = hu;
    uint32_t m = tex;
    while (m) {
      const uint32_t b = uint32_t(std::countr_zero(m));
      m &= m - 1;
      hu = XXH3_64bits_withSeed(s.fetch + b * 6, 24, hu);
    }
    for (uint32_t i = 0; i < 3; ++i) {
      uint32_t v = vf[i];
      while (v) {
        const uint32_t b = uint32_t(std::countr_zero(v));
        v &= v - 1;
        hu = XXH3_64bits_withSeed(s.fetch + (i * 32 + b) * 2, 8, hu);
      }
    }
    const uint64_t k2 = XXH3_64bits_withSeed(s.bool_loop, sizeof(s.bool_loop), hu ^ OwnedFrontStateHashOf(s));
    const uint64_t keys[3] = {k2, hu, hw};
    for (uint32_t i = 0; i < 3; ++i) {
      uint64_t& f = c.alt[i][keys[i]];
      if (f + 1 == frame || f == frame) ++c.alt_last[i];
      else if (f) ++c.alt_earlier[i];
      f = frame;
    }
  }
  const uint64_t fv = vs ? PlanCensusFloats(s.vs_float, vs->constant_register_map().float_bitmap) : 0;
  const uint64_t fp = ps ? PlanCensusFloats(s.ps_float, ps->constant_register_map().float_bitmap) : 0;
  ++c.draws;
  PlanCensus::E& e = c.map[h];
  if (e.frame == frame) {
    ++c.dup_in_frame;
    if (e.prev_frame + 1 == frame) ++c.seen_last;
  } else if (e.frame + 1 == frame) {
    ++c.seen_last;
    c.same_vs += e.vs == fv;
    c.same_ps += e.ps == fp;
    c.same_both += e.vs == fv && e.ps == fp;
  } else if (e.frame) {
    ++c.seen_earlier;
  }
  if (e.frame != frame) e.prev_frame = e.frame;
  e.frame = frame;
  e.vs = fv;
  e.ps = fp;
  if (frame - c.last_print >= 600) {
    REXGPU_INFO("[d3d][plan] census frames {}-{}: typed draws {} ({} a frame): inputs seen the frame before {} ({:.1f}%), seen earlier {}, twice in one frame {}; of those seen the frame before: same vertex floats {}, same pixel floats {}, "
                "both {}; page epoch moves {} a frame; table {}; sizeof Draw {} prep {} state {}", c.last_print, frame, c.draws, c.draws / std::max<uint64_t>(c.frames, 1), c.seen_last, c.draws ? 100.0 * double(c.seen_last) / double(c.draws) : 0.0, c.seen_earlier,
                c.dup_in_frame, c.same_vs, c.same_ps, c.same_both, c.epoch_moves / std::max<uint64_t>(c.frames, 1), c.map.size(), sizeof(native::Draw), sizeof(OwnedTypedPrep), sizeof(OwnedRecordState));
    REXGPU_INFO("[d3d][plan] census: used fetch slots + state + bool: last frame {} earlier {}; used fetch + draw words only: last frame {} earlier {}; draw words only: last frame {} earlier {}", c.alt_last[0], c.alt_earlier[0], c.alt_last[1],
                c.alt_earlier[1], c.alt_last[2], c.alt_earlier[2]);
    for (uint32_t i = 0; i < 3; ++i) {
      c.alt_last[i] = c.alt_earlier[i] = 0;
      if (c.alt[i].size() > (1u << 20)) c.alt[i].clear();
    }
    c.draws = c.seen_last = c.seen_earlier = c.same_vs = c.same_ps = c.same_both = c.dup_in_frame = c.frames = c.epoch_moves = 0;
    c.last_print = frame;
    if (c.map.size() > (1u << 20)) c.map.clear();
  }
}
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// (Session 92, M4f-A) The typed token draw. The old lean path's stand-down list first (the modes whose per-draw work only the stock wrapper does),
// then OwnedFrontPrepare (the typed state's description, modules really made), then the one lean body, DrawFromBase, fed from the typed state.
bool D3dDirectAccess::TypedDraw(VCP& cp, const OwnedRecordState* state, xenos::PrimitiveType prim_type, uint32_t index_count,
                                VCP::IndexBufferInfo* index_buffer_info, bool major_mode_explicit, bool& succeeded, bool crowd_blocked, bool crowd_member) {
  OwnedFrontTypedTick(cp);
  OwnedFrontTypedNote(kTypedItems);
  const TypedSettings* const lat = REXCVAR_GET(gpu_d3d_typed_latch) ? &TypedSettingsNow(cp.frame_current_) : nullptr;  // (Session 95)
  const bool typed_repeat_on = TLV(lat, typed_repeat, gpu_d3d_typed_repeat) && free_counters::g_cut.active;  // (Session 93)
  if (typed_repeat_on) OwnedFrontRepeatBegin();  // (the previous recorded base moves to `previous`; this draw records its own when it succeeds)
  const bool timers = TLV(lat, timers, gpu_d3d_typed_timers);
  const auto fallback = [&](const char* why) {
    cull_ours::NoteBreak();  // (Session 93: a draw the old path makes, or none: the pass the cull guard follows is not known any more)
    OwnedFrontTypedFallback(why);
    return false;
  };
  if (crowd_blocked) return fallback("crowd run queued / crowd draw");
  const OwnedRecordState* s = state ? state : OwnedRecordDrawNow();
  if (!s) return fallback("no snapshot (past the 512 cap, desync, or a draw the decoder did not see)");
  if (!s->banks_valid) return fallback("snapshot without constant banks");
  static const bool plan_census = REXCVAR_GET(gpu_d3d_plan_census);
  if (plan_census) [[unlikely]] {
    if (!crowd_member) PlanCensusDraw(cp.frame_current_, cp.shared_memory_->repeat_epoch(), *s, static_cast<const VulkanShader*>(cp.active_vertex_shader()), static_cast<const VulkanShader*>(cp.active_pixel_shader()));
  }
  if (TLV(lat, phase0, gpu_d3d_phase0_census)) [[unlikely]] {  // (Session 93: log-only census; reads the typed state, writes nothing)
    Phase0Draw(cp.memory_, *s, static_cast<const VulkanShader*>(cp.active_vertex_shader()), crowd_member);
  }
  if (free_counters::g_cut.active) {
    if (const char* skip = CutPreDrawSkip(cp, *s)) {  // (a setting skips the draw: counted on its own, not a drop)
      free_counters::NoteCutSkip(skip);
      succeeded = true;
      return true;
    }
    if (TLV(lat, cull_ours, gpu_d3d_cull_ours)) [[unlikely]] {  // (Session 93, idea 13: our draw the camera cannot see is not drawn; every other draw is noted as the pass the next one continues)
      if (CutCullOurs(cp, *s, crowd_member)) {
        free_counters::NoteCutSkip("cull ours: off-screen draw");
        cp.repeat_prev_.valid = false;  // (as the other skips: the repeat base and the lean base end here)
        cp.lean_.valid = false;
        succeeded = true;
        return true;
      }
      cull_ours::NoteDrawn(*s);
    }
  }
  if (!TLV(lat, lean_draws, gpu_native_lean_draws) || TLV(lat, lean_verify, gpu_native_lean_draws_verify) || TLV(lat, lean_census, gpu_native_lean_dirty_census) ||
      TLV(lat, census_count, gpu_debug_draw_census_count) > 0) {
    return fallback("lean diagnostics on");
  }
  const bool cut = free_counters::g_cut.active;  // (under the cut no setting drops a draw: the shadow chain, half-rate shadows, crowd density and hide HUD are typed predicates below)
  if (!cut && (wwe_shadow_chain::g.off || TLV(lat, crowd_density, gpu_scene_crowd_density) < 100 || TLV(lat, half_rate, gpu_shadow_half_rate))) {
    return fallback("shadow chain / half rate shadows / crowd density");
  }
  if (!cp.native_ || !cp.native_->truth() || !TLV(lat, lean, gpu_native_lean) ||
      (cp.render_target_cache_->GetPath() != RenderTargetCache::Path::kPixelShaderInterlock && !TLV(lat, host_rt, gpu_native_truth_host_rt))) {
    return fallback("not a truth frame");
  }
  if (cp.anatomy_active_ || cp.gpu_marks_active_ || cp.recstats_frames_ > 0 || (!cut && cp.skip_frame_draws_) || cp.rename_.active ||
      (!cut && cp.hide_hud_after_pass_) || !cp.chain_probe_shaders_.empty() || cp.debug_skip_vertex_shader_hash_ || cp.DrawProbeActive() ||
      cp.native_->CoverStatsActive() || (!cut && !cp.lean_mode_ok_)) {
    return fallback("special mode (anatomy / marks / recstats / skip / rename / hide HUD / probes / lean mode off)");
  }
  // (half2_ is on in these runs and the old lean path does not stand down for it: Half2OnCpDraw ran above for this item, the draw kind is known - no fallback)
  {
    // (M4f-A review P4) A draw whose cover rectangle or vertex bounds the claim wants is the old path's (NativeCoverRect / NativeBoundsRect read the
    // registers): decided from the typed state's prim type and shader hashes BEFORE anything is built or requested. DrawFromBase keeps the same test
    // as a backstop for what only the built description knows.
    const bool has_ps = s->shader[1].dwords != 0;
    const bool owned_cover = has_ps && cp.native_->OwnedWantsCover(s->shader[1].hash);
    const xenos::PrimitiveType pt = s->draw_initiator.prim_type;
    const bool wants_cover = (pt == xenos::PrimitiveType::kQuadList || pt == xenos::PrimitiveType::kRectangleList ||
                              (owned_cover && pt == xenos::PrimitiveType::kTriangleStrip)) &&
                             (cp.native_->TruthWantsCover() || owned_cover);
    const bool wants_bounds = !cp.active_occlusion_query_.valid && cp.native_->TruthWantsBounds(s->shader[0].hash);
    if ((wants_cover || wants_bounds) && !(TLV(lat, owned_front, gpu_d3d_owned_front) & 64u)) return fallback("cover or bounds rectangle wanted");
  }
  const uint64_t t0 = timers ? OwnedFrontNowNs() : 0;
  if (!cp.BeginSubmission(true)) {
    succeeded = false;
    OwnedFrontTypedNote(kTypedFailed);
    return true;  // (the old path would fail the same way; the error is the caller's)
  }
  // The page-validity epoch of this draw: read before anything of it is requested (a later invalidation ends a stamp).
  const uint64_t repeat_epoch = cp.shared_memory_->repeat_epoch();
  cp.LatchTilingViewportFix();  // (as IssueDraw: the once-a-frame latch, before the description reads tiling_viewport_fix_)
  // (Session 93, gpu_d3d_typed_repeat) A draw whose only change since the previous recorded base is the vertex float constants (the decoder's epochs, the draw words,
  // the shared memory epoch: OwnedFrontRepeatCandidate) is the old path's repeat: RepeatDrawFast over the typed constants (an instance merged into the base's
  // draw when the constants sit one stride on, else the base's Draw drawn again). Anything it declines is drawn in full below.
  const bool typed_record = typed_repeat_on && !crowd_member && !s->inline_draw;
  if (typed_record && shader_swap::InstanceWorldLatched() && !(TLV(lat, repeat_fast, vulkan_repeat_draw_fast) && cp.repeat_prev_.valid && cp.native_repeat_draw_valid_)) {
    shader_swap::NoteWorldStat(cp.frame_current_, shader_swap::kWsNoBase);  // (Session 94: no repeat base to compare with)
  }
  if (typed_record && TLV(lat, repeat_fast, vulkan_repeat_draw_fast) && cp.repeat_prev_.valid && cp.native_repeat_draw_valid_ && OwnedFrontRepeatCandidate(cp, *s)) {
    VulkanShader* const repeat_vs = static_cast<VulkanShader*>(cp.active_vertex_shader());
    if (repeat_vs && cp.RepeatIsSameDraw(repeat_vs, uint32_t(prim_type), index_count, index_buffer_info, major_mode_explicit, repeat_epoch)) {
      OwnedFrontRepeatNote(kRepeatTried);
      VulkanPipelineCache::TypedKeyState repeat_keys;
      repeat_keys.fetch = s->fetch;
      OwnedRepeatSourceSet(s->vs_float, &repeat_keys);
      ++cp.repeat_fast_stats_.candidates;
      const uint64_t merged_before = cp.repeat_fast_stats_.merged;
      shader_swap::NoteWorldStat(cp.frame_current_, shader_swap::kWsTried);
      const bool repeated = cp.RepeatDrawFast(repeat_vs, repeat_epoch);
      OwnedRepeatSourceSet(nullptr, nullptr);
      if (!repeated) shader_swap::NoteWorldStat(cp.frame_current_, shader_swap::kWsRefused, cp.repeat_refusal_);
      else if (cp.repeat_fast_stats_.merged == merged_before) shader_swap::NoteWorldStat(cp.frame_current_, shader_swap::kWsDrawnApart);
      if (repeated) {
        OwnedFrontRepeatNote(kRepeatDrawn);
        OwnedFrontTypedNote(kTypedDrawn);
        if (timers) OwnedFrontTypedNote(kTypedNs, OwnedFrontNowNs() - t0);
        OwnedFrontRepeatRecord(cp, *s);  // (the base stays the same draw; its key moves on to this draw's constants)
        OwnedFrontTypedAfter();
        succeeded = true;
        return true;
      }
      OwnedFrontRepeatNote(kRepeatRefused);
      cp.repeat_prev_.valid = false;  // (as the old lean path: the full path starts a new run)
    }
  }
  // Session 95 (r3cmd, gpu_d3d_draw_record): this draw's record. A crowd run's member or held first member, an inline draw, the diagnostic run: never.
  DrawRec* rec = nullptr;
  bool rec_hit = false, rec_verify = false;
  bool slot_hit = false, slot_verify = false;  // Session 95 (gpu_d3d_draw_slot)
  int32_t slot_mode = 0;
  uint32_t slot_settings = 0;
  const int32_t rec_mode_raw = TLV(lat, draw_record, gpu_d3d_draw_record);
  const int32_t rec_mode = (rec_mode_raw == 1 || rec_mode_raw == 2) ? rec_mode_raw : 0;  // (any other value is off)
  if (rec_mode && cut && !crowd_member && !s->inline_draw && !cp.own_crowd_skip_draw_ && !diag::g_armed && TLV(lat, memo_reuse, gpu_d3d_typed_memo_reuse) && s == &OwnedRecordStateNow()) {
    DrawRecTick(cp.frame_current_);
    uint32_t skin_role = 0, skin_stride = 0;
    const bool skin_draw = shader_swap::SkinOnceLatched() && shader_swap::SkinOnceRow(s->shader[0].hash, &skin_role, &skin_stride);  // (a skin draw is never stored: no key, no Find)
    // Session 95 (gpu_d3d_draw_slot): the slot the cursor expects first; a miss is today's lookup, and the cursor moves to what it finds. Needs the record's own mode 1: the verify of the record has no hits.
    const int32_t slot_raw = TLV(lat, draw_slot, gpu_d3d_draw_slot);
    slot_mode = (rec_mode == 1 && (slot_raw == 1 || slot_raw == 2)) ? slot_raw : 0;
    DrawRec* slot_rec = nullptr;
    if (slot_mode) {
      DrawSlotTick(cp.frame_current_);
      DrawSlotsFrame(cp.frame_current_);
      const int32_t aniso = REXCVAR_GET(anisotropic_override);  // (live, as the sampler params are built from the live value)
      slot_settings = (aniso > -1 && aniso < 6 ? uint32_t(aniso) + 1 : 0u) | (REXCVAR_GET(gpu_sampler_mip_point) ? 0x100u : 0u);  // (what GetSamplerParameters reads besides the words; gpu_allow_invalid_fetch_constants is in the externals)
      if (!skin_draw) slot_rec = DrawSlotTry(cp, *s, slot_settings);
    }
    if (slot_rec && slot_mode == 1) {
      rec = slot_rec;
      rec_hit = slot_hit = true;
    } else if (!skin_draw) {
      rec = DrawRecFind(cp, *s, rec_hit);
      if (slot_mode) {
        DrawSlotEnsure(slot_mode);
        if (slot_rec) {
          DrawSlotVerifyFind(cp, slot_rec, rec, rec_hit);
          slot_verify = rec_hit && rec == slot_rec;
        } else if (rec_hit) {
          ++g_draw_slots.stats.resyncs;
        }
      }
    }
    if (slot_mode) {
      if (rec) DrawSlotAdvance(rec, (REXCVAR_GET(gpu_tex_bind_memo) & 8) != 0);
    } else if (REXCVAR_GET(gpu_tex_bind_memo) & 8) {  // (Session 95, r6tex)
      static DrawRec* previous = nullptr;
      if (previous && rec) previous->next_hint = rec;
      if (rec && rec->next_hint) {
        const char* const next = static_cast<const char*>(rec->next_hint);
        for (size_t at = 0; at < sizeof(DrawRec); at += 64) __builtin_prefetch(next + at, 0, 2);
      }
      if (rec) previous = rec;
    }
    if (rec_hit && (rec_mode == 2 || slot_verify)) {
      rec_hit = false;
      rec_verify = true;
    }
  }
  OwnedTypedPrep* p = !rec_hit ? nullptr : slot_hit ? DrawSlotPrep(cp, *s, *rec) : DrawRecPrep(cp, *s, *rec);
  if (!p) {
    rec_hit = slot_hit = false;
    p = OwnedFrontPrepare(cp, *s);
  }
  if (!p) return fallback("build refused (see the Build skip reasons)");
  VCP::LeanState L;
  L.valid = true;
  L.pixel_shader = p->pixel_shader;
  L.prim_type = uint32_t(prim_type);
  L.major_mode_explicit = major_mode_explicit;
  L.primitive_polygonal = p->polygonal;
  L.host_vertex_shader_type = p->host_vertex_shader_type;
  L.used_texture_mask = p->used_texture_mask;
  L.normalized_depth_control = p->depth_control;
  L.normalized_color_mask = p->color_mask;
  L.viewport_info = p->viewport_info;
  L.layout = static_cast<const VCP::PipelineLayout*>(p->layout);
  L.shared_memory_access_stages = p->shared_memory_access_stages;
  L.merge_stride = p->merge_stride;
  L.float_count = p->float_count;
  L.pixel_shader_vertex_fetches = p->pixel_shader_vertex_fetches;
  L.host_primitive_reset_enabled = p->host_primitive_reset_enabled;
  L.host_primitive_type = p->host_primitive_type;
  L.vertex_endian_key = p->vertex_endian_key;
  L.pixel_spec_key = p->pixel_spec_key;
  L.vertex_mul_key = p->vertex_mul_key;
  L.pixel_ext = p->pixel_ext;
  L.vertex_ext = p->vertex_ext;
  const VCP::ConstSources consts{p->vs_float, p->ps_float, p->bool_loop, p->fetch};
  VCP::LeanReason why = VCP::kLeanTaken;
  bool cover_wanted = false;
  DrawSources src;
  src.typed = true;
  src.crowd_member = crowd_member;
  src.dirty = cp.repeat_dirty_;
  src.input = &p->input;
  src.fetch = p->fetch;
  src.consts = &consts;
  src.vs_const_word = s->vs_const.value;  // (Session 95, the diagnostic run)
  src.sysconst = &OwnedFrontSystemConstants;
  src.ctx = p;
  src.rb_surface_info = p->surface_info;
  src.rb_color_info = p->color_info;
  src.rb_depth_info = p->depth_info;
  src.why = &why;
  src.cover_wanted = &cover_wanted;
  if (TLV(lat, memo_reuse, gpu_d3d_typed_memo_reuse)) src.primitive = &p->bi.primitive;  // (Session 93: Process ran once, in OwnedFrontPrepare)
  src.record_repeat = typed_record;
  src.rec = rec;
  src.rec_hit = rec_hit;
  src.rec_verify = rec_verify;
  src.rec_state = s;
  src.rec_prep = p;
  src.rec_no_bind = rec_mode == 1 && cut;
  src.slot_on = slot_mode != 0;
  src.slot_hit = slot_hit;
  src.slot_verify = slot_verify;
  src.slot_settings = slot_settings;
  if (s->inline_ring) {  // (Session 95, item P)
    src.inline_ring_set = InlineRing().set();
    src.inline_ring_memory = InlineRing().memory();
  }
  if (TLV(lat, owned_front, gpu_d3d_owned_front) & 64u) {
    src.cover_state = s;  // M4f-B: cover / bounds from the typed state
    // M4f-B: memexport stream constants from the typed banks (vs_float and ps_float are one 512-vec4 array, the register layout)
    src.mx_floats = s->vs_float;
    src.mx_vs_base = s->vs_const.base;
    src.mx_ps_base = s->ps_const.base;
  }
  static OwnedServeTargets serve;  // (M4f-C, the cut: textures are served against the last typed draw's RB words - also by serves made outside TypedDraw (deferred serves, serve-ahead, crowd draws) - never the register source)
  if (free_counters::g_cut.active) {
    serve.surface_info = s->surface_info.value;
    for (uint32_t i = 0; i < 4; ++i) serve.color_info[i] = s->color_info[i].value;
    serve.color_mask = p->color_mask;
    OwnedServeTargetsSet(&serve);
  }
  OwnedFrontTypedActive(true);
  // wwe2k14 patch (Session 94, gpu_d3d_decouple bit 3, phase 1B): view 1 (the in-between) vertex constants of this draw (a blended copy of the bank, or none = the same as view 0, the picture).
  const bool view_blend = (TLV(lat, decouple, gpu_d3d_decouple) & 8) != 0 && SpirvShaderTranslator::DecoupleViewConstantBlocks() && cp.native_->DecoupleViewsOn();
  if (view_blend) OwnedViewFloatSet(DecoupleBlendDraw(cp.memory_, *s, static_cast<const VulkanShader*>(p->vertex_shader), crowd_member));
  const Result r = DrawFromBase(cp, L, *p->draw, p->vertex_shader, prim_type, index_count, index_buffer_info, major_mode_explicit, succeeded, src,
                                false, true, repeat_epoch);
  if (view_blend) OwnedViewFloatSet(nullptr);
  OwnedFrontTypedActive(false);
  if (r == Result::kFull) {
    const char* name = cover_wanted ? "cover or bounds rectangle wanted"
                       : why == VCP::kLeanPrimitive ? "primitive (host draw type changed, 32-bit index DMA, or processing refused)"
                       : why == VCP::kLeanSampler  ? "sampler"
                       : why == VCP::kLeanClaim    ? "claim declined"
                       : why == VCP::kLeanMode     ? "vertex fetch constant not a vertex fetch"
                       : why == VCP::kLeanShader   ? "layout missing"
                                                   : "other";
    return fallback(name);
  }
  OwnedFrontTypedNote(succeeded ? kTypedDrawn : kTypedFailed);
  if (succeeded && p->draw->restore) OwnedFrontTypedNote(kTypedRestores);
  if (succeeded && cp.native_lean_draw_ && cp.native_lean_draw_->pipeline == VK_NULL_HANDLE && cp.native_lean_draw_->claim_refused == 0) OwnedFrontTypedNote(kTypedPlaceholder);
  if (timers) OwnedFrontTypedNote(kTypedNs, OwnedFrontNowNs() - t0);
  if (succeeded && typed_record) OwnedFrontRepeatRecord(cp, *s);
  OwnedFrontTypedAfter();
  return true;
}

// Session 95 (out\s95\android_structural.md section 3.2): the identity an attribute draw must satisfy, in order, each failure counted by name. Whole-buffer bindings read the same bytes the storage
// path reads: neither XeVertexAddress nor the translator bounds a fetch, so the size is not a bound here either; the only protection is this identity, because an attribute draw has no index clamp.
void D3dDirectAccess::BuildVertexInputPlan(VCP& cp, const native::Draw& base, VulkanShader* vertex_shader, bool pixel_shader_vertex_fetches, const PrimitiveProcessor::ProcessingResult& ppr,
                                           const DrawSources& src, bool cross_frame, bool mx_possible, native::VertexInputPlan& plan) {
  using PP = PrimitiveProcessor;
  native::VertexInputState& vin = cp.native_->VertexInput();
  native::VertexInputState::Counters& n = vin.counters();
  const auto no = [&](native::ViRefusal why) { ++n.refused[why]; };
  cp.native_->OwnCrowdClearIndex();  // (P7: a first member's plan lives only until the next draw; the held branch below notes this one's)
  const SpirvShaderTranslator::SystemConstants& sc = cp.system_constants_;
  const int64_t base_index = sc.vertex_base_index, vgt_min = sc.vertex_index_min, vgt_max = sc.vertex_index_max, limit = int64_t(1) << 24;
  // Item 4's DMA step, shared with a crowd run's first member: the pool and the identity. The shader would compute (index + base), clamped to the draw's range, 24 bits: the plan holds only when that
  // is the index plus the base. kViRefusalCount = it holds.
  const auto pool_index = [&](native::IndexPool::Index* index, bool may_stitch) -> native::ViRefusal {
    native::IndexPool& pool = cp.native_->Mirror();
    native::IndexPool::Ctx ctx;
    ctx.mem = cp.memory_;
    ctx.shared_memory = cp.shared_memory_.get();
    ctx.submission = cp.GetCurrentSubmission();
    ctx.completed = cp.GetCompletedSubmission();
    ctx.frame = cp.frame_current_;
    ctx.verify = vin.index_verify();
    const auto refusal = [](native::IndexPool::Why why) {
      switch (why) {
        case native::IndexPool::kOk: return native::kViRefusalCount;
        case native::IndexPool::kNoPool: return native::kViPoolNone;
        case native::IndexPool::kPoolFull: return native::kViPoolFull;
        case native::IndexPool::kTooBig: return native::kViPoolTooBig;
        case native::IndexPool::kGpuWritten: return native::kViGpuWritten;
        default: return native::kViIndexRange;
      }
    };
    const auto find = [&](bool stitch) {
      return refusal(pool.Find(ctx, ppr.guest_index_base, ppr.host_draw_vertex_count, ppr.host_index_format == xenos::IndexFormat::kInt16, uint32_t(ppr.host_shader_index_endian),
                               ppr.host_primitive_reset_enabled, index, stitch));
    };
    if (const native::ViRefusal why = find(false); why != native::kViRefusalCount) return why;
    const int64_t lo = int64_t(index->min) + base_index, hi = int64_t(index->max) + base_index;
    n.with_reset += index->has_reset;
    if (lo < 0 || lo < vgt_min || hi > vgt_max || hi >= limit) return native::kViIdentity;
    if (index->has_reset && vin.no_restart()) {
      // (P2b) With restart off a reset value would be a vertex: a triangle strip takes its own entry, one strip without them, in the same pipeline as a strip that never had any.
      // A crowd run's first member (may_stitch false) has no count of its own to pass on: it stays refused.
      if (!may_stitch || !vin.stitch() || ppr.host_primitive_type != xenos::PrimitiveType::kTriangleStrip || base.raster.topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP) return native::kViResetValues;
      if (const native::ViRefusal why = find(true); why != native::kViRefusalCount) return why;
      // The stitched Find may have remade its entry from newer bytes than the first one saw: its own min / max are the ones that must hold.
      const int64_t slo = int64_t(index->min) + base_index, shi = int64_t(index->max) + base_index;
      if (slo < 0 || slo < vgt_min || shi > vgt_max || shi >= limit) return native::kViIdentity;
      plan.index_count = index->count;
      plan.index_stitched = true;
    }
    plan.index_min = index->min;
    plan.index_max = index->max;
    plan.index_has_reset = index->has_reset;
    return native::kViRefusalCount;
  };
  // 1. The mode.
  if (!src.typed || !cross_frame || !src.fetch) return no(native::kViNotTyped);
  if (!cp.native_->OwnedOnlyActive()) return no(native::kViNotOwnedOnly);
  // Session 95 (P7, gpu_native_crowd_input): a member drawn by itself is an ordinary draw from here on (it is never a stored base: attr.ok below). A held first member is recorded for its state only and
  // reads no vertex; what it needs is the index plan of the run own crowd draws right after it.
  if (cp.own_crowd_skip_draw_ && vin.crowd_input()) {
    plan.crowd_state = true;
    ++n.crowd_state;
    native::NativeRenderer::OwnCrowdViIndex crowd;
    crowd.noted = true;
    if (ppr.index_buffer_type == PP::ProcessedIndexBufferType::kGuestDMA) {
      native::IndexPool::Index index;
      const native::ViRefusal why = pool_index(&index, false);
      crowd.why = why;
      crowd.ok = why == native::kViRefusalCount;
      crowd.offset = index.offset;
      crowd.vertex_offset = sc.vertex_base_index;
      crowd.last_vertex = uint32_t(int64_t(index.max) + base_index);
      crowd.clear_restart = vin.no_restart();
    } else {
      crowd.why = ppr.index_buffer_type == PP::ProcessedIndexBufferType::kHostConverted ? native::kViConverted : native::kViBuiltinDma;
    }
    cp.native_->OwnCrowdNoteIndex(crowd);
    return;
  }
  if (cp.own_crowd_skip_draw_ || (src.crowd_member && !vin.crowd_input())) return no(native::kViCrowd);
  if (base.merge_stride) return no(native::kViMerge);
  if (base.xfb_buffer != VK_NULL_HANDLE) return no(native::kViXfb);
  if (base.fast_clear) return no(native::kViFastClear);
  if (shader_swap::InstanceWorldLatched()) return no(native::kViInstanceWorld);
  if (SpirvShaderTranslator::DecoupleViewConstantBlocks()) return no(native::kViViewBlocks);
  if (ppr.host_vertex_shader_type != Shader::HostVertexShaderType::kVertex || base.raster.expanded) return no(native::kViVsExpand);  // (an expanded draw has a geometry shader: no attribute key of it is prewarmable)
  if (mx_possible) return no(native::kViMemexport);
  if (pixel_shader_vertex_fetches) return no(native::kViPsFetch);
  if (ppr.line_loop_closing_index != 0) return no(native::kViLineLoop);
  // 2. The form: made once per translation.
  bool fresh = false;
  native::ViForm& form = vin.Form(base.vertex_shader, base.vertex_modification, base.vertex_ext, &fresh);
  if (fresh) {
    form.failed = true;
    if (cp.native_->VertexInputFormOf(*vertex_shader, base.vertex_modification, base.vertex_ext, &form)) {
      const uint32_t ext = base.vertex_ext | SpirvShaderTranslator::kNativeExtVsInput;
      auto* translation = static_cast<VulkanShader::VulkanTranslation*>(vertex_shader->GetOrCreateNativeTranslation(base.vertex_modification, ext));
      if (translation && cp.pipeline_cache_->EnsureShadersTranslated(translation, nullptr) && translation->shader_module() != VK_NULL_HANDLE) {
        form.module = translation->shader_module();
        form.failed = false;
      } else {
        form.refusal = native::kViVariantRefused;
      }
    }
  }
  if (form.failed) return no(form.refusal);
  // 3. Each binding. Ring draws: slot 31's stream, its address an offset in the ring (InlineRingStage refused any other stream).
  const bool ring = src.inline_ring_set != VK_NULL_HANDLE;
  const VkBuffer buffer = ring ? InlineRing().buffer() : cp.shared_memory_->buffer();
  for (uint32_t i = 0; i < form.binding_count; ++i) {
    xenos::xe_gpu_vertex_fetch_t vf;
    std::memcpy(&vf, src.fetch + 2 * size_t(form.fetch_constant[i]), sizeof(vf));
    if (vf.type != xenos::FetchConstantType::kVertex) return no(native::kViFetchType);
    if (!vf.size) return no(native::kViFetchSize);
    if (vf.endian != xenos::Endian::k8in32) return no(native::kViFetchEndian);
    const uint64_t offset = uint64_t(vf.address) << 2;
    if (!ring && offset >= SharedMemory::kBufferSize) return no(native::kViFetchRange);
    plan.vb[i] = buffer;
    plan.vb_offsets[i] = offset;
  }
  // 4. The indices, by type.
  switch (ppr.index_buffer_type) {
    case PP::ProcessedIndexBufferType::kGuestDMA: {
      native::IndexPool::Index index;
      if (const native::ViRefusal why = pool_index(&index, true); why != native::kViRefusalCount) return no(why);
      plan.pool_index = true;
      plan.index_offset = index.offset;
      break;
    }
    case PP::ProcessedIndexBufferType::kNone:
    case PP::ProcessedIndexBufferType::kHostBuiltinForAuto: {
      const int64_t hi = base_index + int64_t(ppr.guest_draw_vertex_count) - 1;
      if (base_index < 0 || base_index < vgt_min || hi > vgt_max || hi >= limit) return no(native::kViNoIndexIdentity);
      break;
    }
    case PP::ProcessedIndexBufferType::kHostConverted:
      return no(native::kViConverted);
    default:
      return no(native::kViBuiltinDma);
  }
  plan.form = &form;
  plan.first_vertex = sc.vertex_base_index;
  plan.ok = true;
}

// Session 95 (r3cmd, gpu_d3d_draw_record): the plan of a record. The key holds what the plan is a function of but the index bytes: the pool answers for those (its watch and hash checks), and the plan holds
// while the answer's range, reset values and stitched count are the plan's: the identity with the draw's clamp was proven on them. False = the plan is made again.
bool D3dDirectAccess::RecordVertexInputPlan(VCP& cp, const DrawRec& r, const PrimitiveProcessor::ProcessingResult& ppr, native::VertexInputPlan& plan, native::IndexPool::HeldIndex* held) {
  if (!r.plan_ok || !r.plan.form || r.plan.form->failed) {
    if (r.plan_ok) ++g_draw_rec_stats.plan_fallback;
    return false;
  }
  native::VertexInputState& vin = cp.native_->VertexInput();
  cp.native_->OwnCrowdClearIndex();
  if (r.plan.pool_index && held && held->entry && !vin.index_verify() && cp.native_->Mirror().Touch(*held, cp.GetCurrentSubmission(), cp.frame_current_)) {
    // (Session 95, gpu_d3d_draw_slot) The pool's answer is the one kept: the same entry unmarked, its hash check not due, and its space in use by this submission as a lookup would leave it.
    plan = r.plan;
    plan.index_offset = held->offset;
    vin.counters().with_reset += r.plan.index_has_reset;
    ++g_draw_slots.stats.plan_held;
  } else if (r.plan.pool_index) {
    native::IndexPool::Ctx ctx;
    ctx.mem = cp.memory_;
    ctx.shared_memory = cp.shared_memory_.get();
    ctx.submission = cp.GetCurrentSubmission();
    ctx.completed = cp.GetCompletedSubmission();
    ctx.frame = cp.frame_current_;
    ctx.verify = vin.index_verify();
    native::IndexPool::Index index;
    if (cp.native_->Mirror().Find(ctx, ppr.guest_index_base, ppr.host_draw_vertex_count, ppr.host_index_format == xenos::IndexFormat::kInt16, uint32_t(ppr.host_shader_index_endian),
                                  ppr.host_primitive_reset_enabled, &index, r.plan.index_stitched) != native::IndexPool::kOk ||
        index.min != r.plan.index_min || index.max != r.plan.index_max || index.has_reset != r.plan.index_has_reset || (r.plan.index_stitched && index.count != r.plan.index_count)) {
      ++g_draw_rec_stats.plan_fallback;
      return false;
    }
    plan = r.plan;
    plan.index_offset = index.offset;
    vin.counters().with_reset += index.has_reset;
    if (held) {  // (the answer moved or a check came due: the pool has answered, keep what it answered)
      ++g_draw_slots.stats.plan_asked;
      cp.native_->Mirror().Hold(ppr.guest_index_base, ppr.host_draw_vertex_count, ppr.host_index_format == xenos::IndexFormat::kInt16, uint32_t(ppr.host_shader_index_endian), ppr.host_primitive_reset_enabled,
                                r.plan.index_stitched, r.plan.index_min, r.plan.index_max, r.plan.index_has_reset, r.plan.index_count, held);
    }
  } else {
    plan = r.plan;
  }
  // (the streams' buffer is the live one, as BuildVertexInputPlan takes it: a record's draw is never on the ring)
  for (uint32_t i = 0; i < plan.form->binding_count; ++i) plan.vb[i] = cp.shared_memory_->buffer();
  return true;
}

// The lean body (Session 50 / 87): everything after the validity checks, for a base that holds. `L` and `base` are the old lean base or the typed
// memo's; `src` says where the draw's own inputs come from (see DrawSources).
D3dDirectAccess::Result D3dDirectAccess::DrawFromBase(VCP& cp, const VCP::LeanState& L, const native::Draw& base,
                                                      VulkanShader* vertex_shader, xenos::PrimitiveType prim_type,
                                                      uint32_t index_count, VCP::IndexBufferInfo* index_buffer_info,
                                                      bool major_mode_explicit, bool& succeeded, const DrawSources& src,
                                                      bool restored, bool cross_frame, uint64_t repeat_epoch) {
  using R = Result;
  DirectCensus& c = g_c;
  VCP::LeanStats& stats = cp.lean_stats_;
  (void)restored;
  const auto full = [&](VCP::LeanReason reason) {
    if (!src.typed) {
      ++stats.reasons[reason];
      ++c.reasons[reason];
    }
    if (src.why) *src.why = reason;
    cp.repeat_dirty_ = src.dirty;
    return R::kFull;
  };
  if (!src.typed) cp.repeat_dirty_ = 0;
  // The typed fetch words (the texture cache reads fetch constants through them while this draw runs).
  struct FetchScope {
    VulkanTextureCache* tc;
    const uint32_t* prev;  // (M4f-C: restored, not nulled - under the cut the session source is the typed state's fetch words)
    ~FetchScope() { tc->SetFetchSource(prev); }
  } fetch_scope{cp.texture_cache_.get(), cp.texture_cache_->fetch_source()};
  if (src.fetch) cp.texture_cache_->SetFetchSource(src.fetch);
  VulkanShader* const pixel_shader = L.pixel_shader;
  const ui::vulkan::VulkanDevice::Properties& device_properties = cp.GetVulkanDevice()->properties();
  cp.memexport_ranges_.clear();
  // (Session 92, M4f-B) A typed memexport draw: the stream ranges from the typed constants (AddMemExportRanges over the banks). The description says whether the draw
  // exports (base.memexport: a vertex export only where the device can store from the vertex stage).
  const bool mx_writes = src.mx_floats && base.memexport;
  const bool mx_vertex = mx_writes && vertex_shader->memexport_eM_written() != 0 && device_properties.vertexPipelineStoresAndAtomics;
  const bool mx_pixel = mx_writes && pixel_shader && pixel_shader->memexport_eM_written() != 0;
  if (mx_vertex) draw_util::AddMemExportRanges(src.mx_floats, src.mx_vs_base, *vertex_shader, cp.memexport_ranges_);
  if (mx_pixel) draw_util::AddMemExportRanges(src.mx_floats, src.mx_ps_base, *pixel_shader, cp.memexport_ranges_);
  const bool mx_possible = mx_vertex || mx_pixel;

  if (!cp.BeginSubmission(true)) {
    succeeded = false;
    return R::kTaken;
  }
  VCP::CrashCheckpointRecord* crash_record = nullptr;
  if (cp.crash_checkpoints_enabled()) {
    crash_record = cp.BeginCrashCheckpointRecord();
    cp.InsertCrashCheckpoint(*crash_record, VCP::CrashCheckpointPhase::kDrawPrepare);
  }
  PrimitiveProcessor::ProcessingResult primitive_processing_result;
  if (src.primitive) {
    primitive_processing_result = *src.primitive;  // (Session 93: made once by OwnedFrontPrepare for this draw)
  } else if (src.input) {
    if (!cp.primitive_processor_->Process(primitive_processing_result, *src.input)) {
      return full(VCP::kLeanPrimitive);
    }
  } else if (!cp.primitive_processor_->Process(primitive_processing_result)) {
    return full(VCP::kLeanPrimitive);
  }
  if (!primitive_processing_result.host_draw_vertex_count) {
    cp.anatomy_skip_reason_ = VCP::kAnatomySkipNoVertices;
    cp.lean_.valid = false;
    cp.repeat_prev_.valid = false;
    ++stats.reasons[VCP::kLeanTaken];
    succeeded = true;
    return R::kTaken;
  }
  if (primitive_processing_result.host_vertex_shader_type != L.host_vertex_shader_type ||
      primitive_processing_result.host_primitive_reset_enabled != L.host_primitive_reset_enabled ||
      primitive_processing_result.host_primitive_type != L.host_primitive_type) {
    return full(VCP::kLeanPrimitive);
  }
  const bool shader_32bit_index_dma =
      !device_properties.fullDrawIndexUint32 &&
      primitive_processing_result.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA &&
      index_buffer_info->format == xenos::IndexFormat::kInt32 &&  // (the draw's VGT_DRAW_INITIATOR.index_size)
      primitive_processing_result.host_vertex_shader_type == Shader::HostVertexShaderType::kVertex;
  if (shader_32bit_index_dma) {
    return full(VCP::kLeanPrimitive);
  }

  // Session 94 (gpu_d3d_typed_lean_bind): the typed draw's trims, and the sampled split of the draw's phases (1 frame in 8; clock reads only then).
  const TypedSettings* const lat = (src.typed && REXCVAR_GET(gpu_d3d_typed_latch)) ? &TypedSettingsNow(cp.frame_current_) : nullptr;  // (Session 95)
  const bool lean_bind = src.typed && TLV(lat, lean_bind, gpu_d3d_typed_lean_bind);
  const bool lean_sample = lean_bind && (cp.frame_current_ & 7) == 0;
  uint64_t lean_t = lean_sample ? OwnedFrontNowNs() : 0;
  const auto lean_lap = [&](uint32_t slot) {
    if (!lean_sample) return;
    const uint64_t now = OwnedFrontNowNs();
    g_lean_bind_stat[slot] += now - lean_t;
    lean_t = now;
  };
  if (lean_sample) ++g_lean_bind_stat[9];
  // Samplers: the parameters again only when fetch constants changed (they come from the texture fetch constants);
  // UseSampler every time (the submission may be another one).
  for (uint32_t j = 0; j < 2; ++j) {
    std::vector<std::pair<VulkanTextureCache::SamplerParameters, VkSampler>>& shader_samplers =
        j ? cp.current_samplers_pixel_ : cp.current_samplers_vertex_;
    const VulkanShader* shader = j ? pixel_shader : vertex_shader;
    if (!shader) {
      continue;
    }
    // Session 95 (gpu_d3d_draw_slot): the parameters are a function of the used fetch words, the binding list and three settings. The words and the settings are in the slot's compare; the list is the shader's
    // for the record's ext and has_replacement_bindings, read live, so it is looked at here (a change there is a rebuild, as any draw). Else the list as the draw that stored it left it. UseSampler below still
    // runs for each: the submission may be another one.
    const DrawSlotAux* const slot_aux = src.slot_hit ? &DrawSlotAuxOf(src.rec) : nullptr;
    const bool slot_list = slot_aux && (j ? cp.pipeline_cache_->UsesReplacementBindings(pixel_shader, L.pixel_ext) : cp.pipeline_cache_->UsesReplacementBindings(vertex_shader, L.vertex_ext)) == slot_aux->replacement[j] &&
                           shader->SamplerBindingsFor(slot_aux->replacement[j]).size() == slot_aux->sampler_count[j];
    if (slot_list) {
      const DrawSlotAux& aux = *slot_aux;
      (j ? shader_swap::PixelSamplersReplacement() : shader_swap::VertexSamplersReplacement()) = aux.replacement[j];
      shader_samplers.resize(aux.sampler_count[j]);
      for (uint32_t k = 0; k < aux.sampler_count[j]; ++k) shader_samplers[k].first = aux.params[j][k];
      ++g_lean_bind_stat[4];
    } else {
    // (Session 87, pixel skip: the list is the one of the translation the base draw's native ext picks; rebuilt too when the pixel
    // samplers in hand were made from the other one.)
    const bool replacement_list = j ? cp.pipeline_cache_->UsesReplacementBindings(pixel_shader, L.pixel_ext)
                                    : cp.pipeline_cache_->UsesReplacementBindings(vertex_shader, L.vertex_ext);
    bool& held_replacement = j ? shader_swap::PixelSamplersReplacement() : shader_swap::VertexSamplersReplacement();
    // Session 94 (gpu_d3d_typed_lean_bind): a typed draw's sampler parameters are a function of the typed fetch words (the decoder's fetch epoch, valid while src.fetch is the
    // live decoder state), the shader's binding list and the two settings GetSamplerParameters reads besides. The memo keeps the parameters this code built last for the stage;
    // it hits when all of those are unchanged AND the processor's own list still equals them (the old IssueDraw path rebuilds that list too, so it is compared, not trusted).
    // Session 95 (gpu_d3d_sampler_memo_fix, needs gpu_d3d_typed_lean_bind: memo_live): the processor-list comparison and the repeat_dirty_ fetch bit are dropped from the hit test (typed draws never
    // clear that bit, so it blocked every hit) and the memo is a direct-mapped table of kMemoWays entries per stage indexed by the shader pointer; a hit rewrites the processor's list.
    // A hit skips GetSamplerParameters for every binding (UseSampler below still runs every draw: the submission may be another one). gpu_d3d_typed_memo_verify recomputes
    // on every hit and compares.
    struct SamplerMemo {
      bool valid = false;
      const VulkanShader* shader = nullptr;
      uint64_t hash = 0;
      std::vector<uint32_t> words;  // the 6 fetch words of each binding, as the parameters were built from them
      const void* bindings = nullptr;
      size_t count = 0;
      uint32_t extra = 0;
      std::vector<VulkanTextureCache::SamplerParameters> params;
    };
    constexpr uint32_t kMemoWays = 16;
    static SamplerMemo sampler_memo[2][kMemoWays];
    const bool memo_fix = TLV(lat, memo_fix, gpu_d3d_sampler_memo_fix);  // (Session 95)
    SamplerMemo& memo = sampler_memo[j][memo_fix ? ((reinterpret_cast<uintptr_t>(shader) >> 6) & (kMemoWays - 1)) : 0];
    const bool memo_live = lean_bind && src.fetch;  // (the words themselves are compared: no epoch, no live-state condition)
    uint32_t memo_extra = 0;
    bool memo_hit = false;
    if (memo_live && (memo_fix || !(src.dirty & VCP::kRepeatDirtyFetch)) && !restored && held_replacement == replacement_list) {
      memo_extra = (TLV(lat, aniso, anisotropic_override) > -1 && TLV(lat, aniso, anisotropic_override) < 6 ? uint32_t(TLV(lat, aniso, anisotropic_override)) + 1 : 0u) |
                   (TLV(lat, invalid_fetch, gpu_allow_invalid_fetch_constants) ? 0x100u : 0u);
      const std::vector<VulkanShader::SamplerBinding>& bl = shader->SamplerBindingsFor(replacement_list);
      if (memo.valid && memo.shader == shader && memo.hash == shader->ucode_data_hash() && memo.bindings == bl.data() && memo.count == bl.size() &&
          memo.extra == memo_extra && memo.words.size() == bl.size() * 6 && memo.params.size() == bl.size() &&
          (memo_fix || shader_samplers.size() == memo.params.size())) {
        memo_hit = true;
        for (size_t k = 0; k < memo.params.size(); ++k) {
          if (k >= bl.size() || (!memo_fix && shader_samplers[k].first != memo.params[k]) ||
              std::memcmp(&memo.words[6 * k], src.fetch + 6 * size_t(bl[k].fetch_constant), 6 * sizeof(uint32_t)) != 0) {
            memo_hit = false;
            break;
          }
        }
        if (memo_hit && TLV(lat, memo_verify, gpu_d3d_typed_memo_verify)) {
          for (size_t k = 0; k < bl.size(); ++k) {
            if (cp.texture_cache_->GetSamplerParameters(bl[k]) != memo.params[k]) {
              memo_hit = false;
              ++g_lean_bind_stat[8];
              static uint32_t logged = 0;
              if (logged < 16) {
                ++logged;
                REXGPU_WARN("[d3d][typed] lean_bind verify: sampler memo hit would have given another parameter (stage {}, binding {}, #{})", j, k, logged);
              }
              break;
            }
          }
        }
      }
    }
    if (memo_hit) {
      ++g_lean_bind_stat[4];
      if (memo_fix) {  // (Session 95: the list as a rebuild would leave it; the samplers themselves are taken below)
        shader_samplers.resize(memo.params.size());
        for (size_t k = 0; k < memo.params.size(); ++k) shader_samplers[k].first = memo.params[k];
      }
    } else if (src.typed || (src.dirty & VCP::kRepeatDirtyFetch) || restored || held_replacement != replacement_list) {
      const std::vector<VulkanShader::SamplerBinding>& bindings = shader->SamplerBindingsFor(replacement_list);
      held_replacement = replacement_list;
      shader_samplers.clear();
      shader_samplers.reserve(bindings.size());
      for (const VulkanShader::SamplerBinding& binding : bindings) {
        ++cp.frame_counters_.sampler_parameters;
        shader_samplers.emplace_back(cp.texture_cache_->GetSamplerParameters(binding), VK_NULL_HANDLE);
      }
      if (memo_live) {
        ++g_lean_bind_stat[5];
        memo.valid = true;
        memo.shader = shader;
        memo.hash = shader->ucode_data_hash();
        memo.bindings = bindings.data();
        memo.count = bindings.size();
        // (Session 95: read live, not latched: the parameters just built above read the settings live, so the key stored with them must be what they were built from; the compare above stays latched)
        memo.extra = (REXCVAR_GET(anisotropic_override) > -1 && REXCVAR_GET(anisotropic_override) < 6 ? uint32_t(REXCVAR_GET(anisotropic_override)) + 1 : 0u) |
                     (REXCVAR_GET(gpu_allow_invalid_fetch_constants) ? 0x100u : 0u);
        memo.words.resize(bindings.size() * 6);
        for (size_t k = 0; k < bindings.size(); ++k) std::memcpy(&memo.words[6 * k], src.fetch + 6 * size_t(bindings[k].fetch_constant), 6 * sizeof(uint32_t));
        memo.params.clear();
        for (const auto& sp : shader_samplers) memo.params.push_back(sp.first);
      }
    }
    }  // (Session 95: not a slot hit)
    for (std::pair<VulkanTextureCache::SamplerParameters, VkSampler>& pair : shader_samplers) {
      bool overflowed;
      pair.second = cp.texture_cache_->UseSampler(pair.first, overflowed);
      if (pair.second == VK_NULL_HANDLE) {
        cp.lean_.valid = false;
        return full(VCP::kLeanSampler);
      }
    }
  }
  lean_lap(3);  // (Session 95 r5: the split of "bind": samplers, textures, the plans, UpdateBindings)
  const uint64_t tex_stat_t = lean_sample ? TexStatTicks() : 0;  // (Session 95, r6tex: log only)
  g_tex_stat_on = lean_sample;
  cp.native_->TruthServeBegin();
  cp.texture_cache_->RequestTextures(L.used_texture_mask);
  cp.native_->TruthServeEnd();
  g_tex_stat_on = false;
  if (lean_sample) g_tex_stat[kTsWholeTicks] += TexStatTicks() - tex_stat_t;
  if (cp.native_->segment_log_active()) cp.NativeLogDrawTextures(L.used_texture_mask);
  lean_lap(10);

  const VCP::PipelineLayout* const pipeline_layout = L.layout;
  if (!pipeline_layout) {
    return full(VCP::kLeanShader);
  }
  if (cp.current_guest_graphics_pipeline_layout_ != pipeline_layout) {
    if (!cp.current_guest_graphics_pipeline_layout_) {
      cp.current_graphics_descriptor_sets_bound_up_to_date_ = 0;
    }
    cp.current_guest_graphics_pipeline_layout_ = pipeline_layout;
  }
  if (src.sysconst) {
    if (src.slot_hit) {  // (Session 95, gpu_d3d_draw_slot)
      if (!DrawSlotSystemConstants(cp, src.ctx, *src.rec)) {
        g_draw_slot_sys_hash = 0;
        src.sysconst(cp, DrawSlotFullPrep(*src.rec_state, *src.rec), primitive_processing_result);  // (a full build reads the lists a record's head does not hold)
      }
    } else {
      g_draw_slot_sys_hash = 0;  // (the block is written below by another hand)
      if (!src.rec_hit || !DrawRecSystemConstants(cp, src.ctx, *src.rec)) {  // (Session 95, r3cmd: the record's block)
        src.sysconst(cp, src.ctx, primitive_processing_result);  // (Session 92, M4f-A: the typed state's system constants)
      }
    }
  } else {
    cp.UpdateSystemConstantValues(L.primitive_polygonal, primitive_processing_result, false, 0,
                                  L.viewport_info, L.used_texture_mask, L.normalized_depth_control,
                                  L.normalized_color_mask);
  }
  lean_lap(1);
  // ---- Session 95 (P0: gpu_native_vinput_census, log only): what the vertex-input design has to serve, read here because the system constants above hold this draw's index base and clamp. Nothing
  // below changes the draw: the DMA indices go through the mirror pool dry (an entry is made and watched) and the result is only counted. Order of the classes: crowd, ring, ours, translated.
  static const bool vinput_census = REXCVAR_GET(gpu_native_vinput_census);
  const bool vertex_input = shader_swap::VertexMirrorLatched();  // (Session 95: gpu_native_vertex_input, read at start)
  uint64_t plan_t = lean_sample ? TexStatTicks() : 0;  // (Session 95, r6tex: the split of "plans", log only)
  const auto plan_lap = [&](uint32_t ticks, uint32_t draws) {
    if (!lean_sample) return;
    const uint64_t now = TexStatTicks();
    g_tex_stat[ticks] += now - plan_t;
    if (draws) ++g_tex_stat[draws];
    plan_t = now;
  };
  if (vertex_input) {  // (the [vinput] line reads the pool's counters of the window the pool's own FrameEnd is about to close)
    native::IndexPool& pool = cp.native_->Mirror();
    cp.native_->VertexInput().FrameTick(cp.frame_current_, pool, cp.window_stats_.changes);
    pool.FrameEnd(cp.frame_current_);
  }
  plan_lap(kTsPlanTickTicks, 0);
  if (vinput_census) [[unlikely]] {
    native::IndexPool& m = cp.native_->Mirror();
    m.FrameEnd(cp.frame_current_);
    using Census = native::IndexPool::Census;
    using PP = PrimitiveProcessor;
    Census& k = m.census();
    const PP::ProcessingResult& ppr = primitive_processing_result;
    const SpirvShaderTranslator::SystemConstants& sc = cp.system_constants_;
    const bool crowd = src.crowd_member || cp.own_crowd_skip_draw_, ring = src.inline_ring_set != VK_NULL_HANDLE;
    const bool ours = (base.vertex_ext & SpirvShaderTranslator::kNativeExtVsSwap) != 0;
    const uint32_t type = std::min<uint32_t>(uint32_t(ppr.index_buffer_type), 4);
    ++(src.typed ? k.typed : k.not_typed);
    ++k.cls[crowd ? Census::kCrowd : ring ? Census::kRing : ours ? Census::kOurs : Census::kTranslated];
    if (crowd) {
      ++k.crowd_index[type];
    } else {
      ++k.index[type];
      if (ring && ours) ++k.ring_ours;
      if (!ours) ++k.not_ours[vertex_shader->ucode_data_hash()];
      k.restart += ppr.host_primitive_reset_enabled;
      k.base_nonzero += sc.vertex_base_index != 0;
      k.clamp_trivial += sc.vertex_index_min == 0 && sc.vertex_index_max >= 0xFFFFFFu;
      k.memexport += mx_possible;
      k.ps_fetch += L.pixel_shader_vertex_fetches;
      k.line_loop += ppr.line_loop_closing_index != 0;
      const bool plain_vs = ppr.host_vertex_shader_type == Shader::HostVertexShaderType::kVertex;
      k.vs_expand += !plain_vs;
      // The streams, read from the typed fetch words (the first refusal of section 3.2 item 3 counts; ring draws hold the ring's slot, so only the guest's own).
      if (src.fetch && !ring) {
        bool bad_type = false, bad_size = false, bad_endian = false, bad_range = false;
        for (const Shader::VertexBinding& binding : vertex_shader->vertex_bindings()) {
          xenos::xe_gpu_vertex_fetch_t vf;
          std::memcpy(&vf, src.fetch + 2 * size_t(binding.fetch_constant), sizeof(vf));
          bad_type |= vf.type != xenos::FetchConstantType::kVertex;
          bad_size |= !vf.size;
          bad_endian |= vf.endian != xenos::Endian::k8in32;
          bad_range |= (uint64_t(vf.address) << 2) >= SharedMemory::kBufferSize;
        }
        k.fetch_type += bad_type;
        k.fetch_size += !bad_type && bad_size;
        k.fetch_endian += !bad_type && !bad_size && bad_endian;
        k.fetch_range += !bad_type && !bad_size && !bad_endian && bad_range;
      }
      if (ppr.index_buffer_type == PP::ProcessedIndexBufferType::kGuestDMA && plain_vs) {
        native::IndexPool::Ctx ctx;
        ctx.mem = cp.memory_;
        ctx.shared_memory = cp.shared_memory_.get();
        ctx.submission = cp.GetCurrentSubmission();
        ctx.completed = cp.GetCompletedSubmission();
        ctx.frame = cp.frame_current_;
        native::IndexPool::Index index;
        ++k.dma_asked;
        k.dma_restart += ppr.host_primitive_reset_enabled;
        if (m.Find(ctx, ppr.guest_index_base, ppr.host_draw_vertex_count, ppr.host_index_format == xenos::IndexFormat::kInt16, uint32_t(ppr.host_shader_index_endian),
                   ppr.host_primitive_reset_enabled, &index) == native::IndexPool::kOk) {
          ++k.dma_found;
          if (index.has_reset) {
            ++k.dma_reset;
            k.dma_reset_tri_strip += ppr.host_primitive_type == xenos::PrimitiveType::kTriangleStrip;
            k.dma_reset_line_strip += ppr.host_primitive_type == xenos::PrimitiveType::kLineStrip;
          }
          const int64_t lo = int64_t(index.min) + sc.vertex_base_index, hi = int64_t(index.max) + sc.vertex_base_index;
          k.dma_identity += lo < 0 || lo < int64_t(sc.vertex_index_min) || hi > int64_t(sc.vertex_index_max) || hi >= (int64_t(1) << 24);
        }
      } else if ((ppr.index_buffer_type == PP::ProcessedIndexBufferType::kNone || ppr.index_buffer_type == PP::ProcessedIndexBufferType::kHostBuiltinForAuto) && plain_vs) {
        const int64_t lo = sc.vertex_base_index, hi = lo + int64_t(ppr.guest_draw_vertex_count) - 1;
        k.plain_identity += lo < 0 || lo < int64_t(sc.vertex_index_min) || hi > int64_t(sc.vertex_index_max) || hi >= (int64_t(1) << 24);
      }
    }
  }
  // ---- Session 95 (out\s95\android_structural.md, gpu_native_vertex_input): does this draw read its vertices as attributes? Decided here, before UpdateBindings, because such a draw selects no storage
  // window; after this point there is no way back to the plain draw (a pending pipeline drops it, as any draw). The module, the buffers and the index source are set on native_draw below.
  const Shader::ConstantRegisterMap& constant_map_vertex = vertex_shader->constant_register_map();
  const bool on_ring = src.inline_ring_set != VK_NULL_HANDLE;  // (Session 95, item P: the one stream the draw fetches is on the ring: no guest range to request, no page to watch)
  // (Session 95, r3gpu: skin once decides first; a listed draw it takes is not planned as an attribute draw, one it refuses is, gpu_native_skin_with_input. Nothing here needs UpdateBindings: a job of a draw that returns there is rows nobody reads.)
  struct SkinPlan {
    bool ok = false;
    VkShaderModule module = VK_NULL_HANDLE;
    native::VertexMirror::PipeMemo* memo = nullptr;
    VkDeviceSize vertex_off = 0, index_off = 0;
    int32_t base_index = 0;
    uint32_t role = 0;
  } skin;
  // ---- Session 95 (gpu_native_skin_once, skin_once.h): a typed draw of a listed wrestler shader reads this frame's skinned rows through its _so variant: the same claim and fallbacks as an attribute
  // draw, with the rows as its vertex buffer, the mirror's host-order indices as its index buffer and vertexOffset = -(the smallest index): row 0 is the draw's first vertex.
  uint32_t skin_row_bits = 0, skin_stride = 0;
  if (src.typed && shader_swap::SkinOnceLatched() && shader_swap::SkinOnceRow(vertex_shader->ucode_data_hash(), &skin_row_bits, &skin_stride)) [[unlikely]] {
    native::SkinOnce& so = cp.native_->Skin();
    native::VertexMirror& m = cp.native_->SkinMirror();
    using S = native::SkinOnce;
    m.FrameEnd(cp.frame_current_);
    so.Tick(cp.frame_current_, cp.GetCurrentSubmission(), cp.GetCompletedSubmission());
    const PrimitiveProcessor::ProcessingResult& ppr = primitive_processing_result;
    if (src.crowd_member || cp.own_crowd_skip_draw_ || restored || on_ring || !cp.native_->OwnedOnlyActive() || base.merge_stride || base.xfb_buffer != VK_NULL_HANDLE || base.fast_clear ||
        ppr.host_vertex_shader_type != Shader::HostVertexShaderType::kVertex || SpirvShaderTranslator::DecoupleViewConstantBlocks() || !src.fetch || !src.consts || mx_possible ||
        L.pixel_shader_vertex_fetches) {
      so.Refuse(S::kRefSpecial);
    } else if (!(base.vertex_ext & SpirvShaderTranslator::kNativeExtVsSwap) || !(base.vertex_ext & SpirvShaderTranslator::kNativeExtOwnBindings)) {
      so.Refuse(S::kRefBaseExt);
    } else if (!cross_frame) {
      so.Refuse(S::kRefNotFresh);
    } else if (ppr.index_buffer_type != PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA || ppr.line_loop_closing_index != 0) {
      so.Refuse(S::kRefIndex);
    } else {
      bool only_stream = true;
      for (uint32_t i = 0; i < rex::countof(constant_map_vertex.vertex_fetch_bitmap); ++i) {
        const uint32_t want = (95u >> 5) == i ? (uint32_t(1) << (95u & 31)) : 0u;
        if (constant_map_vertex.vertex_fetch_bitmap[i] != want) only_stream = false;
      }
      xenos::xe_gpu_vertex_fetch_t vf;
      std::memcpy(&vf, src.fetch + 2 * size_t(95), sizeof(vf));
      if (!only_stream || vf.type != xenos::FetchConstantType::kVertex || !vf.size || vf.endian != xenos::Endian::k8in32) {
        so.Refuse(S::kRefFetch);
      } else {
        native::VertexMirror::PipeMemo& pm = m.pipe_memo()[native::VertexMirror::MemoKey(base.vertex_shader, base.vertex_modification, base.vertex_ext)];
        if (!pm.failed && pm.module == VK_NULL_HANDLE) {
          const uint32_t ext = base.vertex_ext | SpirvShaderTranslator::kNativeExtVsSkinOnce;
          if (!cp.pipeline_cache_->UsesReplacementBindings(vertex_shader, ext) || !shader_swap::VertexSkinFits(*vertex_shader, base.vertex_modification)) {
            pm.failed = true;
          } else {
            auto* translation = static_cast<VulkanShader::VulkanTranslation*>(vertex_shader->GetOrCreateNativeTranslation(base.vertex_modification, ext));
            if (translation && cp.pipeline_cache_->EnsureShadersTranslated(translation, nullptr) && translation->shader_module() != VK_NULL_HANDLE) {
              pm.module = translation->shader_module();
            } else {
              pm.failed = true;
            }
          }
        }
        if (pm.failed) {
          so.Refuse(S::kRefVariant);
        } else {
          native::VertexMirror::Ctx ctx;
          ctx.mem = cp.memory_;
          ctx.shared_memory = cp.shared_memory_.get();
          ctx.submission = cp.GetCurrentSubmission();
          ctx.completed = cp.GetCompletedSubmission();
          ctx.frame = cp.frame_current_;
          ctx.verify = REXCVAR_GET(gpu_native_vertex_mirror_verify);
          const bool is16 = ppr.host_index_format == xenos::IndexFormat::kInt16;
          native::VertexMirror::Index index;
          native::VertexMirror::Why why = m.IndexData(ctx, ppr.guest_index_base, ppr.host_draw_vertex_count, is16, uint32_t(ppr.host_shader_index_endian), ppr.host_primitive_reset_enabled, &index);
          const SpirvShaderTranslator::SystemConstants& sc = cp.system_constants_;
          const int64_t lo = int64_t(index.min) + sc.vertex_base_index, hi = int64_t(index.max) + sc.vertex_base_index;
          if (why != native::VertexMirror::kOk) {
            so.Refuse(S::kRefMirror);
          } else if (lo < 0 || hi >= (int64_t(1) << 24) || lo < int64_t(sc.vertex_index_min) || hi > int64_t(sc.vertex_index_max) || (uint64_t(hi) + 1) * skin_stride > uint64_t(vf.size)) {
            so.Refuse(S::kRefIndex);
          } else {
            VkDeviceSize vertex_off = 0;
            why = m.Vertex(ctx, vf.address << 2, vf.size << 2, &vertex_off);
            if (why != native::VertexMirror::kOk) {
              so.Refuse(S::kRefMirror);
            } else {
              S::Request q;
              q.mem = cp.memory_;
              q.mirror_buffer = m.buffer();
              q.mirror_offset = vertex_off;
              q.address = vf.address << 2;
              q.stride_words = skin_stride;
              q.first = uint32_t(lo);
              q.count = uint32_t(hi - lo + 1);
              q.vs_float = src.consts->vs_float;
              q.bools = src.consts->bool_loop[0];
              q.main = (skin_row_bits & 15u) == S::kRoleMain;
              q.style = (skin_row_bits >> 4) & 1u;
              if (q.main) {
                float c218w;
                std::memcpy(&c218w, src.consts->vs_float + 4 * 218 + 3, 4);
                const float scale = c218w + 1.0f;
                std::memcpy(&q.scale_bits, &scale, 4);
              }
              VkDeviceSize rows = 0;
              if (so.Get(q, &rows)) {
                skin.ok = true;
                skin.module = pm.module;
                skin.memo = &pm;
                skin.vertex_off = rows;
                skin.index_off = index.offset;
                skin.base_index = -int32_t(index.min);
                skin.role = (skin_row_bits & 15u) + 1;
              }
            }
          }
        }
      }
    }
  }
  plan_lap(kTsPlanSkinTicks, skin.ok ? kTsPlanSkinDraws : 0);
  native::VertexInputPlan attr;
  if (vertex_input && !skin.ok && !(src.rec_hit && RecordVertexInputPlan(cp, *src.rec, primitive_processing_result, attr, src.slot_hit ? &DrawSlotAuxOf(src.rec).held : nullptr))) {  // (a record hit is never a listed skin draw: those have no record)
    plan_lap(kTsPlanRecTicks, src.rec_hit ? kTsPlanRecDraws : 0);
    if (src.rec_hit) attr = native::VertexInputPlan();  // (only after a failed record plan; a fresh one is already empty)
    BuildVertexInputPlan(cp, base, vertex_shader, L.pixel_shader_vertex_fetches, primitive_processing_result, src, cross_frame, mx_possible, attr);
    plan_lap(kTsPlanBuildTicks, kTsPlanBuildDraws);
  }
  plan_lap(kTsPlanRecTicks, src.rec_hit ? kTsPlanRecDraws : 0);
  // A planned attribute draw that still falls back to the typed fallback (a claim refusal, a cover or mode check) is counted: it is in neither the attribute nor the storage total.
  const auto fell_back = [&]() { if (attr.ok) ++cp.native_->VertexInput().counters().fell_back; };
  lean_lap(11);
  g_update_bindings_no_bind = src.rec_no_bind && cp.native_->OwnedOnlyActive();  // (Session 95, r3cmd)
  const bool bindings_ok = cp.UpdateBindings(vertex_shader, pixel_shader, src.consts, attr.ok || attr.crowd_state);
  g_update_bindings_no_bind = false;
  if (!bindings_ok) {
    succeeded = false;
    cp.lean_.valid = false;
    cp.repeat_prev_.valid = false;
    ++stats.reasons[VCP::kLeanTaken];
    return R::kTaken;
  }
  lean_lap(12);

  // The vertex buffers: the stock lean loop, with the stamp in front of RequestRange (row 8).
  const RegisterFile* const regs = src.fetch ? nullptr : &cp.DrawRegs();  // (the old lean path; the typed draw reads src.fetch and never touches the register source)
  const bool verify = TLV(lat, direct_verify, gpu_d3d_direct_verify);
  // The stock residency of the draw's vertex ranges (a lambda: a skin draw whose pipeline turns out to be pending runs it late, after the claim). An attribute draw reads the same pages through its
  // vertex buffers: it requests them too.
  const auto request_vertex_ranges = [&]() -> std::optional<R> {
  for (uint32_t i = 0; !on_ring && i < rex::countof(constant_map_vertex.vertex_fetch_bitmap); ++i) {
    uint32_t vfetch_bits_remaining = constant_map_vertex.vertex_fetch_bitmap[i];
    uint32_t j;
    while (bit_inline::bit_scan_forward(vfetch_bits_remaining, &j)) {
      vfetch_bits_remaining &= ~(uint32_t(1) << j);
      const uint32_t vfetch_index = i * 32 + j;
      const uint64_t vfetch_bit = uint64_t(1) << (vfetch_index & 63);
      if ((cp.vertex_buffers_in_sync_[vfetch_index >> 6] & vfetch_bit) && !cp.lean_residency_cache_off_) {
        const VCP::VertexBufferState& in_sync_state = cp.vertex_buffer_states_[vfetch_index];
        cp.shared_memory_->MarkHoistAccess(in_sync_state.address << 2, in_sync_state.size << 2);
        // (Session 95 review: bit 4 keeps a re-written slot in sync - the skipped-page check the state-match branch ran stays; free when no page is skipped.)
        if (REXCVAR_GET(gpu_d3d_wall) & 4u) cp.shared_memory_->CheckSkipped(in_sync_state.address << 2, in_sync_state.size << 2);
        continue;
      }
      xenos::xe_gpu_vertex_fetch_t vfetch_constant;
      if (src.fetch) {  // (Session 92, M4f-A: the typed state's fetch words; vertex fetch constant n = words 2n, 2n+1)
        std::memcpy(&vfetch_constant, src.fetch + 2 * size_t(vfetch_index), sizeof(vfetch_constant));
      } else {
        vfetch_constant = regs->GetVertexFetch(vfetch_index);
      }
      if (vfetch_constant.type != xenos::FetchConstantType::kVertex) {
        cp.lean_.valid = false;
        return (fell_back(), full(VCP::kLeanMode));
      }
      VCP::VertexBufferState& state = cp.vertex_buffer_states_[vfetch_index];
      if (state.address == vfetch_constant.address && state.size == vfetch_constant.size &&
          !cp.lean_residency_cache_off_) {
        cp.vertex_buffers_in_sync_[vfetch_index >> 6] |= vfetch_bit;
        cp.shared_memory_->MarkHoistAccess(vfetch_constant.address << 2, vfetch_constant.size << 2);
        cp.shared_memory_->CheckSkipped(vfetch_constant.address << 2, vfetch_constant.size << 2);
        if (cp.shared_memory_->ram_stats_on()) {
          SharedMemory::RamReaderScope ram_reader(*cp.shared_memory_, SharedMemory::RamReader::kVertex);
          const std::pair<uint32_t, uint32_t> range(vfetch_constant.address << 2, vfetch_constant.size << 2);
          cp.shared_memory_->RamCountRequest(&range, 1);
        }
        continue;
      }
      const uint32_t range_start = vfetch_constant.address << 2;
      const uint32_t range_length = vfetch_constant.size << 2;
      RangeStamp& stamp = g_stamps[vfetch_index];
      if (stamp.epoch == repeat_epoch && stamp.address == vfetch_constant.address && stamp.size == vfetch_constant.size) {
        // The fetch constant and the page-validity epoch equal what this range was last requested (successfully) under:
        // no page stopped being valid and no watch fired since, so it still holds. Replayed: what the stock in-sync
        // branch does for a range it knows is valid.
        if (verify && !cp.shared_memory_->IsRangeUpToDate(range_start, range_length)) {
          ++c.range_mismatches;
          if (c.mismatch_logged < 16) {
            ++c.mismatch_logged;
            REXGPU_WARN("[d3d][direct] vertex range stamp mismatch (frame {}): fetch slot {} range 0x{:08X} + 0x{:X} is not "
                        "up to date at epoch {}",
                        cp.frame_current_, vfetch_index, range_start, range_length, repeat_epoch);
          }
          stamp.epoch = UINT64_MAX;  // (requested for real below)
        } else {
          ++c.range_hits;
          state.address = vfetch_constant.address;
          state.size = vfetch_constant.size;
          cp.vertex_buffers_in_sync_[vfetch_index >> 6] |= vfetch_bit;
          cp.shared_memory_->MarkHoistAccess(range_start, range_length);
          cp.shared_memory_->CheckSkipped(range_start, range_length);
          if (cp.shared_memory_->ram_stats_on()) {
            SharedMemory::RamReaderScope ram_reader(*cp.shared_memory_, SharedMemory::RamReader::kVertex);
            const std::pair<uint32_t, uint32_t> range(range_start, range_length);
            cp.shared_memory_->RamCountRequest(&range, 1);
          }
          continue;
        }
      }
      SharedMemory::RamReaderScope ram_reader(*cp.shared_memory_, SharedMemory::RamReader::kVertex);
      ++c.range_requests;
      if (!cp.shared_memory_->RequestRange(range_start, range_length)) {
        REXGPU_ERROR("Failed to request vertex buffer at 0x{:08X} (size {}) in the shared memory", range_start,
                     range_length);
        stamp.epoch = UINT64_MAX;
        succeeded = false;
        cp.lean_.valid = false;
        cp.repeat_prev_.valid = false;
        ++stats.reasons[VCP::kLeanTaken];
        return R::kTaken;
      }
      stamp.address = vfetch_constant.address;
      stamp.size = vfetch_constant.size;
      stamp.epoch = repeat_epoch;
      state.address = vfetch_constant.address;
      state.size = vfetch_constant.size;
      cp.vertex_buffers_in_sync_[vfetch_index >> 6] |= vfetch_bit;
    }
  }
  return std::nullopt;
  };
  // (Session 95, P7: a held first member reads no vertex, and own crowd reads the mesh from guest memory itself. If own crowd declines, the guard's draw is a full typed draw and requests its own.)
  if (!attr.crowd_state && !skin.ok) {  // (Session 95: a skin draw reads the job's rows; if it goes back to plain, its ranges are requested then)
    if (const std::optional<R> r = request_vertex_ranges()) return *r;
  }
  // A pixel shader's vertex fetches: every page counts as accessed in this submission.
  if (L.pixel_shader_vertex_fetches) {
    cp.shared_memory_->MarkHoistAccess(0, SharedMemory::kBufferSize);
  }
  // (Session 92, M4f-B) The memexport draw's streams: requested in the shared memory, the extent the draw reads and writes; a guest DMA index buffer is read
  // from a snapshot (no read / write aliasing through the shared memory) - what IssueDraw does.
  uint32_t mx_start = UINT32_MAX, mx_end = 0;
  VCP::ScratchBufferAcquisition dma_index_scratch;
  const auto mx_fail = [&]() {  // (review 7: a failed memexport request ends every base the old and the typed path hold)
    succeeded = false;
    cp.lean_.valid = false;
    cp.repeat_prev_.valid = false;
    if (src.typed) cp.native_repeat_draw_valid_ = false;
    return R::kTaken;
  };
  if (mx_possible) {
    for (const draw_util::MemExportRange& range : cp.memexport_ranges_) {
      const uint32_t base_bytes = range.base_address_dwords << 2;
      SharedMemory::RamReaderScope memexport_reader(*cp.shared_memory_, SharedMemory::RamReader::kMemexport);
      if (!cp.shared_memory_->RequestRange(base_bytes, range.size_bytes)) {
        REXGPU_ERROR("Failed to request memexport stream at 0x{:08X} (size {}) in the shared memory", base_bytes, range.size_bytes);
        return mx_fail();
      }
      mx_start = std::min(mx_start, base_bytes);
      mx_end = std::max(mx_end, base_bytes + range.size_bytes);
    }
    if (cp.memexport_ranges_.empty() && !cp.shared_memory_->RequestRange(0, SharedMemory::kBufferSize)) {
      REXGPU_ERROR("Failed to request full shared memory residency for unresolved memexport destinations");
      return mx_fail();
    }
    if (primitive_processing_result.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA) {
      const VkDeviceSize size = VkDeviceSize(primitive_processing_result.host_draw_vertex_count) *
                                (primitive_processing_result.host_index_format == xenos::IndexFormat::kInt16 ? sizeof(uint16_t) : sizeof(uint32_t));
      if (size) {
        dma_index_scratch = cp.AcquireScratchGpuBuffer(size, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        if (dma_index_scratch.buffer() == VK_NULL_HANDLE) return mx_fail();
        cp.shared_memory_->Use(VulkanSharedMemory::Usage::kRead, {}, VK_PIPELINE_STAGE_TRANSFER_BIT);
        cp.SubmitBarriers(true);
        VkBufferCopy region = {};
        region.srcOffset = primitive_processing_result.guest_index_base;
        region.dstOffset = 0;
        region.size = size;
        cp.deferred_command_buffer_.CmdVkCopyBuffer(cp.shared_memory_->buffer(), dma_index_scratch.buffer(), 1, &region);
        cp.PushBufferMemoryBarrier(dma_index_scratch.buffer(), 0, size, dma_index_scratch.GetStageMask(), VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                                   dma_index_scratch.GetAccessMask(), VK_ACCESS_INDEX_READ_BIT);
        dma_index_scratch.SetStageMask(VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
        dma_index_scratch.SetAccessMask(VK_ACCESS_INDEX_READ_BIT);
      }
    }
    if (mx_start < mx_end) {
      cp.shared_memory_->Use(VulkanSharedMemory::Usage::kGuestDrawReadWrite, std::make_pair(mx_start, mx_end - mx_start), L.shared_memory_access_stages);
    } else {
      cp.shared_memory_->Use(VulkanSharedMemory::Usage::kGuestDrawReadWrite, std::make_pair(0u, SharedMemory::kBufferSize), L.shared_memory_access_stages);
    }
  } else if (!on_ring && !skin.ok) {  // (Session 95: a ring or skin draw does not read the shared memory, so no barrier for it; the next draw that does has its own Use. An attribute draw reads it at the vertex input stage)
    cp.shared_memory_->Use(VulkanSharedMemory::Usage::kRead, {}, L.shared_memory_access_stages | (attr.ok ? VkPipelineStageFlags(VK_PIPELINE_STAGE_VERTEX_INPUT_BIT) : VkPipelineStageFlags(0)));
  }

  const bool emulated_indexed =
      primitive_processing_result.index_buffer_type != PrimitiveProcessor::ProcessedIndexBufferType::kNone;
  std::pair<VkBuffer, VkDeviceSize> index_buffer = {VK_NULL_HANDLE, 0};
  switch (primitive_processing_result.index_buffer_type) {
    case PrimitiveProcessor::ProcessedIndexBufferType::kNone:
      break;
    case PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA:
      if (dma_index_scratch.buffer() != VK_NULL_HANDLE) {
        index_buffer.first = dma_index_scratch.buffer();
        index_buffer.second = 0;
      } else {
        index_buffer.first = cp.shared_memory_->buffer();
        index_buffer.second = primitive_processing_result.guest_index_base;
      }
      break;
    case PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted:
      index_buffer =
          cp.primitive_processor_->GetConvertedIndexBuffer(primitive_processing_result.host_index_buffer_handle);
      break;
    case PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForAuto:
    case PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForDMA:
      index_buffer =
          cp.primitive_processor_->GetBuiltinIndexBuffer(primitive_processing_result.host_index_buffer_handle);
      break;
    default:
      return (fell_back(), full(VCP::kLeanPrimitive));
  }
  const VkIndexType index_type = primitive_processing_result.host_index_format == xenos::IndexFormat::kInt16
                                     ? VK_INDEX_TYPE_UINT16
                                     : VK_INDEX_TYPE_UINT32;

  // The base's native draw with this draw's own bindings, buffers and counts, claimed as the base was.
  if (!cp.native_lean_draw_) cp.native_lean_draw_ = std::make_unique<native::Draw>();
  native::Draw& native_draw = *cp.native_lean_draw_;
  native_draw = base;
  native_draw.restore = src.typed ? base.restore : false;  // (M4f-A review P1: a typed draw keeps the restore its own fetch words made)
  std::copy(std::begin(cp.current_graphics_descriptor_sets_), std::end(cp.current_graphics_descriptor_sets_),
            native_draw.sets);
  native_draw.sets_bound = cp.native_sets_needed_;
  if (on_ring) {  // (this draw only: the current set stays the shared memory's, and UpdateBindings above bound it into the stream again)
    native_draw.sets[SpirvShaderTranslator::kDescriptorSetSharedMemoryAndEdram] = src.inline_ring_set;
    cp.native_->OwnedForgetSharedMemorySet();
  }
  {
    uint32_t offset_count = 0;
    const uint32_t* offsets = cp.D3dDirectDynamicOffsets(offset_count);
    if (offsets) {
      std::copy(offsets, offsets + offset_count, native_draw.dynamic_offsets);
      native_draw.dynamic_offset_count = offset_count;
    }
  }
  native_draw.occlusion_query = cp.active_occlusion_query_.valid;
  native_draw.vertex_count = primitive_processing_result.host_draw_vertex_count;
  native_draw.instance_count = 1;
  native_draw.indexed = emulated_indexed;
  native_draw.index_buffer = index_buffer.first;
  native_draw.index_offset = index_buffer.second;
  native_draw.index_type = index_type;
  // (Session 95, gpu_native_skin_once) A skin draw: the _so variant, the job's rows as its one binding and the mirror's host-order indices; the claim below makes its pipeline. Never a stored record
  // (no_store) or a stored base. drop_skin puts the plain draw back; the variant's vertex input is the 12-dword rows, the same attribute rule as any attribute draw (vi 12: three uvec4).
  const auto drop_skin = [&]() {
    native_draw.vertex_shader = base.vertex_shader;
    native_draw.vertex_ext = base.vertex_ext;
    native_draw.pipeline = VK_NULL_HANDLE;
    native_draw.attribute = false;
    native_draw.no_store = base.no_store;
    native_draw.skin_side_key = 0;  // (Session 95, r3gpu)
    native_draw.vi = 0;
    native_draw.vb_count = 0;
    native_draw.vb[0] = VK_NULL_HANDLE;
    native_draw.vb_offsets[0] = 0;
    native_draw.vertex_offset = 0;
    native_draw.index_buffer = index_buffer.first;
    native_draw.index_offset = index_buffer.second;
  };
  if (skin.ok) {
    native_draw.vertex_shader = skin.module;
    native_draw.vertex_ext |= SpirvShaderTranslator::kNativeExtVsSkinOnce;
    native_draw.attribute = true;
    native_draw.no_store = true;
    native_draw.vi = 12;
    native_draw.vb_count = 1;
    native_draw.vb[0] = cp.native_->Skin().buffer();
    native_draw.vb_offsets[0] = skin.vertex_off;
    native_draw.vertex_offset = skin.base_index;
    native_draw.index_buffer = cp.native_->SkinMirror().buffer();
    native_draw.index_offset = skin.index_off;
    native_draw.plain_restart = native_draw.raster.primitive_restart;
    native_draw.guest_index_base = primitive_processing_result.guest_index_base;
    // (Session 95, r3gpu, gpu_native_skin_bind_lean) The side passes' _so shaders (vs_skin_shadow_so / vs_skin_rim_so) read c54-c59, c64-c67, c162, c255 and the position half of the system constants.
    // The hash of those: two draws that share it take the same constants, so the second keeps the first's slots (OwnedRecord). The pixel side (shadow_map.frag c168, rim_light_map none, the epilogue) goes in too.
    native_draw.skin_side_key = 0;
    if (REXCVAR_GET(gpu_native_skin_bind_lean) && skin.role - 1 != uint32_t(native::SkinOnce::kRoleMain)) {
      uint64_t h = 0x9E3779B97F4A7C15ull ^ skin.role;
      const auto mix = [&h](const void* bytes, size_t size) { h = XXH3_64bits_withSeed(bytes, size, h); };
      mix(src.consts->vs_float + 4 * 54, 4 * 14 * sizeof(uint32_t));   // c54-c67
      mix(src.consts->vs_float + 4 * 162, 4 * sizeof(uint32_t));
      mix(src.consts->vs_float + 4 * 255, 4 * sizeof(uint32_t));
      const SpirvShaderTranslator::SystemConstants& sc = cp.system_constants_;
      mix(&sc.flags, sizeof(sc.flags));
      mix(sc.ndc_scale, sizeof(sc.ndc_scale));
      mix(sc.ndc_offset, sizeof(sc.ndc_offset));
      mix(&sc.alpha_test_reference, sizeof(sc.alpha_test_reference));
      mix(sc.color_exp_bias, sizeof(sc.color_exp_bias));
      mix(&native_draw.dynamic_offsets[SpirvShaderTranslator::kConstantBufferFloatPixel], sizeof(uint32_t));  // (the pixel floats: the same slot means the same bytes)
      native_draw.skin_side_key = h | 1u;
      static uint64_t keyed_total = 0;  // (proof: the key store ran; the pass side prints keyed / seen / kept)
      if (keyed_total++ % 60000 == 0) REXGPU_INFO("[skin1] bind lean: side draw {} keyed (role {})", keyed_total, skin.role - 1);
    }
  }
  if (attr.ok) {  // (Session 95) the attribute form and its buffers; the claim below makes its pipeline (a cross-frame claim: the pipeline is looked up there)
    const native::ViForm& form = *attr.form;
    native_draw.vertex_shader = form.module;
    native_draw.vertex_ext |= SpirvShaderTranslator::kNativeExtVsInput;
    native_draw.attribute = true;
    native_draw.sync_pipeline = src.crowd_member;  // (P7: review finding 2)
    native_draw.vi = form.vi;
    native_draw.vb_count = form.binding_count;
    std::copy_n(attr.vb, form.binding_count, native_draw.vb);
    std::copy_n(attr.vb_offsets, form.binding_count, native_draw.vb_offsets);
    native_draw.vertex_offset = attr.first_vertex;
    if (attr.pool_index) {
      native_draw.index_buffer = cp.native_->Mirror().buffer();
      native_draw.index_offset = attr.index_offset;
      if (attr.index_count) native_draw.vertex_count = attr.index_count;  // (P2b: a stitched strip has its own count)
    }
    // Restart is patched here only, never on the primitive result: the lean checks compare that. A record keeps the guest's bit (plain_restart).
    native_draw.plain_restart = native_draw.raster.primitive_restart;
    native_draw.guest_index_base = primitive_processing_result.guest_index_base;
    if (cp.native_->VertexInput().no_restart()) native_draw.raster.primitive_restart = false;
  }
  // (Session 93) The cover / bounds predicates are pure within a draw: asked once (the shader's cover program flag always, the other two when the draw gets that far).
  const bool cover_owned_step = !src.rec_hit && pixel_shader && cp.native_->OwnedWantsCover(pixel_shader->ucode_data_hash());
  int8_t truth_cover_memo = src.rec_hit ? 0 : -1, bounds_vs_memo = src.rec_hit ? 0 : -1;  // (Session 95, r3cmd: a record's draw wants neither: it was stored because of that)
  const auto truth_wants_cover = [&]() {
    if (truth_cover_memo < 0) truth_cover_memo = cp.native_->TruthWantsCover() ? 1 : 0;
    return truth_cover_memo == 1;
  };
  const auto bounds_vs_wanted = [&]() {
    if (bounds_vs_memo < 0) bounds_vs_memo = cp.native_->TruthWantsBounds(vertex_shader->ucode_data_hash()) ? 1 : 0;
    return bounds_vs_memo == 1;
  };
  if (src.typed) {
    // (Session 92, M4f-A: a draw whose cover rectangle or vertex bounds the claim wants is the old path's - NativeCoverRect / NativeBoundsRect read
    // the registers; counted by the caller as "cover or bounds wanted".)
    const bool owned_cover = cover_owned_step;
    const bool wants_cover = (prim_type == xenos::PrimitiveType::kQuadList || prim_type == xenos::PrimitiveType::kRectangleList ||
                              (owned_cover && prim_type == xenos::PrimitiveType::kTriangleStrip)) &&
                             (truth_wants_cover() || owned_cover);
    const bool wants_bounds = !native_draw.fast_clear && native_draw.xfb_buffer == VK_NULL_HANDLE && !native_draw.occlusion_query &&
                              !native_draw.memexport && bounds_vs_wanted();
    if ((wants_cover || wants_bounds) && !src.cover_state) {
      if (src.cover_wanted) *src.cover_wanted = true;
      return (fell_back(), full(VCP::kLeanMode));
    }
  }
  // (Session 92, M4f-B) A typed draw's cover / bounds rectangles: the same two functions over the typed state's inputs (CoverInputs); the stock draw's
  // inputs are its register source's, filled by the functions' own signatures.
  CoverInputs typed_cover;
  bool cover_filled = false;  // (filled on first use: it copies the constant banks for the interpreter)
  const auto fill_cover = [&]() {
    if (!cover_filled) {
      cover_filled = true;
      OwnedFrontCoverInputs(*src.cover_state, typed_cover);
      typed_cover.fetch_memory = src.inline_ring_memory;
    }
  };
  // (Session 95, item P) The interpreter ran 4 vertices of every particle for a rectangle nothing reads: a cover narrows a screen-space draw (TruthDrawRect), or proves a draw
  // that replaces what is under it (no blending) or writes depth. A blended world-space draw without depth writes is none of those.
  const bool cover_unused = on_ring && !base.screen_space && base.blend_control != 0x00010001u && !base.raster.depth_write;
  const bool cover_asked = src.cover_state && pixel_shader && (prim_type == xenos::PrimitiveType::kQuadList || prim_type == xenos::PrimitiveType::kRectangleList ||
                                                 (cover_owned_step && prim_type == xenos::PrimitiveType::kTriangleStrip)) &&
                                (truth_wants_cover() || cover_owned_step);
  if (cover_unused && cover_asked) InlineRing().NoteCoverSkipped();
  const bool cover_wanted_now = !cover_unused && cover_asked;
  bool bounds_ran = false;  // (counted after the claim: a draw handed back is not a typed rectangle)
  // (Session 95, owned_scout.h) A result a job made ahead of this draw, when its key and input fingerprint match; else the functions below run as before.
  scout::Lookup jobs(src.cover_state, src.crowd_member, *vertex_shader, pixel_shader);
  if (jobs.active() && !src.rec_hit) jobs.Note(cover_owned_step, truth_wants_cover(), bounds_vs_wanted());  // (Session 95, r3cmd: a record hit has no cover answers to note)
  const auto cover_typed = [&]() {
    bool jok = false;
    int32_t jline = 0;
    if (jobs.Cover(jok, native_draw.cover, jline)) {
      if (!jok && jline) cp.NativeCoverReject(jline, *vertex_shader);
      return jok;
    }
    fill_cover();
    const bool r = cp.NativeCoverRect(typed_cover, prim_type, *vertex_shader, pixel_shader, native_draw.cover, cover_owned_step, index_buffer_info);
    jobs.VerifyCover(r, native_draw.cover);
    return r;
  };
  std::fill(std::begin(native_draw.cover), std::end(native_draw.cover), 0);
  if (!cover_unused && (prim_type == xenos::PrimitiveType::kQuadList || prim_type == xenos::PrimitiveType::kRectangleList ||
       (cover_owned_step && prim_type == xenos::PrimitiveType::kTriangleStrip)) &&
      (truth_wants_cover() || cover_owned_step) &&
      !(src.cover_state ? cover_typed()
                        : cp.NativeCoverRect(prim_type, *vertex_shader, pixel_shader, native_draw.cover, cover_owned_step,
                                             index_buffer_info))) {
    std::fill(std::begin(native_draw.cover), std::end(native_draw.cover), 0);
  }
  std::fill(std::begin(native_draw.bounds), std::end(native_draw.bounds), 0);
  if (!native_draw.fast_clear && native_draw.xfb_buffer == VK_NULL_HANDLE && !native_draw.occlusion_query &&
      !native_draw.memexport && bounds_vs_wanted()) {
    if (src.cover_state) {
      bounds_ran = true;
      const int jo = jobs.Bounds(native_draw.bounds);
      if (jo >= 0) {
        cp.native_->TruthNoteBounds(uint32_t(jo), 0.0);
      } else {
        fill_cover();
        jobs.VerifyBounds(cp.NativeBoundsRect(typed_cover, prim_type, *vertex_shader, index_count, index_buffer_info, native_draw.bounds), native_draw.bounds);
      }
    } else {
      cp.NativeBoundsRect(prim_type, *vertex_shader, index_count, index_buffer_info, native_draw.bounds);
    }
  }
  lean_t = lean_sample ? OwnedFrontNowNs() : 0;
  bool claimed;
  const auto claim_fresh = [&]() {
    // The cached description carries no claim: the draw is claimed as a fresh one (the images, the epoch, the pipeline).
    native_draw.truth = native_draw.owned = native_draw.post = native_draw.declined = false;
    native_draw.pipeline = VK_NULL_HANDLE;
    native_draw.truth_color = native_draw.truth_depth = -1;
    native_draw.truth_class = 1;
    native_draw.truth_sx = native_draw.truth_sy = 1;
    native_draw.truth_epoch = 0;
    const bool ok = cp.native_->Claim(native_draw) && native_draw.truth && native_draw.owned &&
                    (src.typed || native_draw.pipeline != VK_NULL_HANDLE);  // (what ClaimTruthLean demands of a base's draw; a typed draw with no pipeline yet is dropped below, once, as IssueDraw does)
    if (ok) ++c.cross_claimed;
    return ok;
  };
  if (cross_frame) {
    // (Session 95, r3cmd) The record's claim, for the same draw (its plan held): else the fresh one, whose answer is the same or newer.
    claimed = false;
    if (src.rec_hit && src.rec->claim_ok && attr.ok == src.rec->plan_ok && !skin.ok) {
      const DrawRec& r = *src.rec;
      native_draw.truth = r.truth;
      native_draw.owned = r.owned;
      native_draw.post = r.post;
      native_draw.declined = false;
      native_draw.claim_refused = 0;
      native_draw.pipeline = r.pipeline;
      native_draw.truth_color = r.truth_color;
      native_draw.truth_depth = r.truth_depth;
      native_draw.truth_class = r.truth_class;
      native_draw.truth_sx = r.truth_sx;
      native_draw.truth_sy = r.truth_sy;
      claimed = cp.native_->OwnedClaimRecord(native_draw);  // (sets truth_epoch)
      if (!claimed) ++g_draw_rec_stats.claim_fallback;
    }
    if (!claimed) claimed = claim_fresh();
  } else {
    claimed = cp.native_->ClaimTruthLean(native_draw);
  }
  if (skin.ok && !claimed) {
    // The claim was refused with the variant. If it is the pipeline (made now, synchronously, and not possible for the variant's key): the variant is never tried again for this module, the draw is
    // claimed again as the plain one and requests its ranges late. Any other reason is the plain draw's too: it goes back to plain and the refusal stands.
    const bool no_pipeline = native_draw.claim_refused == native::kTruthNoPipeline;
    if (no_pipeline) ++cp.native_->SkinMirror().counters().fb[native::VertexMirror::kFbNoPipeline];
    cp.native_->Skin().Refuse(native::SkinOnce::kRefClaim);
    drop_skin();
    skin.ok = false;
    if (no_pipeline) {
      if (skin.memo) skin.memo->failed = true;
      native_draw.claim_refused = 0;
      claimed = claim_fresh();
      if (claimed) {
        if (const std::optional<R> r = request_vertex_ranges()) return *r;
        cp.shared_memory_->Use(VulkanSharedMemory::Usage::kRead, {}, L.shared_memory_access_stages);
      }
    }
  }
  if (attr.ok && !claimed && native_draw.claim_refused == native::kTruthNoPipeline) {
    // (Session 95) The variant's pipeline cannot be made: never tried again for this form, and this one draw goes back to the typed fallback, counted as fell back.
    attr.form->failed = true;
    attr.form->refusal = native::kViVariantRefused;
  }
  lean_lap(2);
  if (!claimed) {
    return (fell_back(), full(VCP::kLeanClaim));
  }
  if (skin.ok) {
    native::VertexMirror& m = cp.native_->SkinMirror();
    if (native_draw.pipeline == VK_NULL_HANDLE) {
      // The variant's pipeline is still being made (the claim queued it): this draw is the plain one. The plain module, ext and index buffer back, the plain pipeline looked up, and the stream's
      // residency requested now - exactly what the draw would have done without skin once.
      ++m.counters().fb[native::VertexMirror::kFbPipelinePending];
      cp.native_->Skin().Refuse(native::SkinOnce::kRefPending);
      drop_skin();
      skin.ok = false;
      bool plain_pending = false;
      if (!native_draw.claim_refused) native_draw.pipeline = cp.native_->MirrorPipeline(native_draw, &plain_pending);  // (GetPipeline: made already for a plain draw that has been drawn before)
      if (const std::optional<R> r = request_vertex_ranges()) return *r;
      cp.shared_memory_->Use(VulkanSharedMemory::Usage::kRead, {}, L.shared_memory_access_stages);
    } else {
      ++m.counters().mirrored;
      cp.native_->Skin().NoteTaken(native::SkinOnce::Role(skin.role - 1), primitive_processing_result.host_draw_vertex_count);
    }
  }
  if (attr.ok) {  // (Session 95: a pending pipeline drops the draw below, like any plain draw's)
    native::VertexInputState::Counters& vin = cp.native_->VertexInput().counters();
    if (native_draw.pipeline == VK_NULL_HANDLE) {
      ++vin.pending;
    } else {
      ++(on_ring ? vin.ring : attr.form->translated ? vin.translated : vin.ours);
      if (attr.index_count) {
        ++vin.stitched;
        vin.stitched_in += primitive_processing_result.host_draw_vertex_count;
        vin.stitched_out += attr.index_count;
      }
      vin.crowd_member_draws += src.crowd_member;  // (Session 95, P7)
      for (uint32_t i = 0; !on_ring && i < native_draw.vb_count; ++i) {
        xenos::xe_gpu_vertex_fetch_t vf;
        std::memcpy(&vf, src.fetch + 2 * size_t(attr.form->fetch_constant[i]), sizeof(vf));
        if (cp.shared_memory_->UploadedThisSubmission(uint32_t(vf.address) << 2, uint32_t(vf.size) << 2)) {
          ++vin.uploaded;
          break;
        }
      }
    }
  }
  if (src.cover_state) {
    if (cover_wanted_now) OwnedFrontTypedNote(kTypedCover);
    if (bounds_ran) OwnedFrontTypedNote(kTypedBounds);
    if (mx_possible) OwnedFrontTypedNote(kTypedMemexport);
  }
  if (src.typed && native_draw.pipeline == VK_NULL_HANDLE) {
    // (M4f-A review P3) Claimed but its pipeline is still being made: the draw is dropped once by the owned frame (OnDrawPre / OnDraw below), and the frame
    // is held back like any async placeholder - as IssueDraw does for a truth draw without a pipeline. The old path does not claim it a second time.
    cp.frame_used_async_placeholder_pipeline_ = true;
    cp.NoteSkippedDraw(src.typed ? int64_t(src.rb_surface_info) : -1);
  }
  if (cp.native_->census_active()) {
    cp.native_->CensusDraw(vertex_shader->ucode_data_hash(), pixel_shader ? pixel_shader->ucode_data_hash() : 0, false);
  }
  if (cp.in_render_pass_) cp.EndRenderPass();
  cp.SubmitBarriers(false);
  if (crash_record) {
    crash_record->vertex_shader_hash = vertex_shader->ucode_data_hash();
    crash_record->pixel_shader_hash = pixel_shader ? pixel_shader->ucode_data_hash() : 0;
    crash_record->guest_primitive_type = uint32_t(prim_type);
    crash_record->host_primitive_type = uint32_t(primitive_processing_result.host_primitive_type);
    crash_record->guest_index_count = index_count;
    crash_record->host_vertex_count = primitive_processing_result.host_draw_vertex_count;
    crash_record->index_buffer_type = uint32_t(primitive_processing_result.index_buffer_type);
    crash_record->guest_index_base = primitive_processing_result.guest_index_base;
    crash_record->edram_mode = uint32_t(xenos::EdramMode::kColorDepth);
    crash_record->memexport_vertex = mx_vertex;
    crash_record->memexport_pixel = mx_pixel;
    crash_record->memexport_extent_start = mx_start;
    crash_record->memexport_extent_end = mx_end;
    crash_record->rb_surface_info = src.rb_surface_info;
    crash_record->rb_color_info = src.rb_color_info;
    crash_record->rb_depth_info = src.rb_depth_info;
    cp.InsertCrashCheckpoint(*crash_record, VCP::CrashCheckpointPhase::kDraw);
  }
  ++cp.frame_counters_.draws_issued;
  native_draw.own_skip_call = cp.own_crowd_skip_draw_;
  native_draw.diag_id = 0;  // (Session 95, the diagnostic run)
  if (diag::g_armed) [[unlikely]] {
    diag::DrawInputs in;
    const RegisterFile* const draw_regs = src.consts ? nullptr : &cp.DrawRegs();
    in.draw = &native_draw;
    in.vertex_shader = vertex_shader;
    in.floats = src.consts ? src.consts->vs_float : &draw_regs->values[XE_GPU_REG_SHADER_CONSTANT_000_X];
    in.ps_floats = src.consts ? src.consts->ps_float : &draw_regs->values[XE_GPU_REG_SHADER_CONSTANT_256_X];
    in.bool_loop = src.consts ? src.consts->bool_loop : &draw_regs->values[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031];
    in.fetch = src.consts ? src.consts->fetch : &draw_regs->values[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0];
    in.vs_const = src.consts ? src.vs_const_word : draw_regs->values[XE_GPU_REG_SQ_VS_CONST];
    in.index_count = index_count;
    if (index_buffer_info && index_buffer_info->guest_base) {
      in.indexed = true;
      in.guest_index_base = index_buffer_info->guest_base;
      in.index_32bit = index_buffer_info->format == xenos::IndexFormat::kInt32;
      in.index_endian = uint32_t(index_buffer_info->endianness);
    }
    in.inline_draw = src.inline_ring_set != VK_NULL_HANDLE;
    diag::NoteDraw(cp, in);
  }
  cp.native_->OnDrawPre(native_draw);
  cp.native_->OnDraw(native_draw);
  // (Session 95, r3cmd) What this draw went through the full way for is its record. Only a draw whose parts are functions of the key: not one whose cover or vertex box reads the constants, not a draw
  // with a converted index buffer or a primitive pass that looked at the index bytes, not a restore, a memexport or a skin draw, and only with a pipeline in hand.
  if (src.rec && !src.rec_hit) {
    const PrimitiveProcessor::ProcessingResult& ppr = primitive_processing_result;
    using PT = PrimitiveProcessor::ProcessedIndexBufferType;
    const bool index_ok = ppr.index_buffer_type == PT::kNone || ppr.index_buffer_type == PT::kHostBuiltinForAuto ||
                          (ppr.index_buffer_type == PT::kGuestDMA && (!ppr.host_primitive_reset_enabled || (src.rec_state->reset_indx.value & 0xFFFF) == 0xFFFF) && ppr.host_index_format == xenos::IndexFormat::kInt16);
    // (a strip, quad or rectangle that does not have the 4 or 3 vertices of a rectangle: NativeCoverRect refuses it before it reads anything but the draw's words; every strip of the game asks.)
    const bool cover_of_words = cover_asked && index_count == (prim_type == xenos::PrimitiveType::kRectangleList ? 3u : 4u);
    const bool wants_box = cover_of_words || (!native_draw.fast_clear && native_draw.xfb_buffer == VK_NULL_HANDLE && !native_draw.occlusion_query && !native_draw.memexport && bounds_vs_wanted());
    const bool storable = index_ok && !wants_box && !skin.ok && !mx_possible && !on_ring && !attr.crowd_state && !base.restore && !base.fast_clear && !base.memexport && base.xfb_buffer == VK_NULL_HANDLE &&
                          !native_draw.occlusion_query && !native_draw.claim_refused && native_draw.pipeline != VK_NULL_HANDLE && succeeded;
    if (src.rec_verify) {
      const DrawRecStats before = g_draw_rec_stats;
      DrawRecVerify(*src.rec, cp, *src.rec_prep, native_draw, attr);
      if (src.slot_verify) {  // (Session 95, gpu_d3d_draw_slot: everything a hit takes from the record and the slot's side, against what the full way just gave)
        DrawSlotStats& st = g_draw_slots.stats;
        const DrawRecStats& now = g_draw_rec_stats;
        st.verify_parts += (now.verify_draw - before.verify_draw) + (now.verify_sys - before.verify_sys) + (now.verify_claim - before.verify_claim) + (now.verify_plan - before.verify_plan) + (now.verify_primitive - before.verify_primitive);
        DrawSlotVerifyPrep(cp, *src.rec, *src.rec_prep);
        DrawSlotAux& aux = DrawSlotAuxOf(src.rec);
        for (uint32_t j = 0; j < 2; ++j) {
          const auto& list = j ? cp.current_samplers_pixel_ : cp.current_samplers_vertex_;
          if (!(j ? pixel_shader : vertex_shader)) continue;
          bool same = list.size() == aux.sampler_count[j] && (j ? shader_swap::PixelSamplersReplacement() : shader_swap::VertexSamplersReplacement()) == aux.replacement[j];
          for (size_t k = 0; same && k < list.size(); ++k) same = list[k].first == aux.params[j][k];
          st.verify_samplers += !same;
        }
        if (attr.ok && attr.pool_index && cp.native_->Mirror().HeldGood(aux.held, cp.frame_current_) && aux.held.offset != attr.index_offset) ++st.verify_plan;
        if (now.verify_sys == before.verify_sys) g_draw_slot_sys_hash = aux.sys_hash;  // (the live block is the record's: the next draw's check of a hash skip reads it)
      }
    } else if (storable) {
      uint32_t span_base = 0, span_bytes = 0;
      bool span_expansion = false;
      PrimitiveProcessor::LastIndexLoadSpan(span_base, span_bytes, span_expansion);
      src.rec->index_base = span_base;
      src.rec->index_bytes = span_bytes;
      src.rec->index_expansion = span_expansion;
      DrawRecFill(cp, *src.rec, *src.rec_state, *src.rec_prep, native_draw, attr, attr.ok);
    } else {
      src.rec->complete = false;
      ++g_draw_rec_stats.refused;
      DrawRecStats& t = g_draw_rec_stats;
      t.why[0] += !index_ok; t.why[1] += cover_of_words; t.why[2] += wants_box && !cover_of_words; t.why[3] += skin.ok; t.why[4] += mx_possible; t.why[5] += on_ring; t.why[6] += attr.crowd_state; t.why[7] += base.restore;
      t.why[8] += base.fast_clear; t.why[9] += native_draw.occlusion_query; t.why[10] += native_draw.claim_refused != 0; t.why[11] += native_draw.pipeline == VK_NULL_HANDLE; t.why[12] += !succeeded;
    }
  }
  // (Session 95, gpu_d3d_draw_slot) A draw that was not a hit leaves its words beside its complete record: its key just matched them, so the slot says exactly what the record is for. A hit changes nothing here.
  if (src.slot_on && !src.slot_hit && !src.slot_verify && src.rec && src.rec->complete && succeeded) {
    DrawRec& r = *src.rec;
    DrawSlotAux& aux = DrawSlotAuxOf(&r);
    if (cp.current_samplers_vertex_.size() > 32 || cp.current_samplers_pixel_.size() > 32) {
      DrawSlotOf(&r).rec = nullptr;
      ++g_draw_slots.stats.refused;
    } else {
      for (uint32_t j = 0; j < 2; ++j) {
        const auto& list = j ? cp.current_samplers_pixel_ : cp.current_samplers_vertex_;
        aux.sampler_count[j] = uint32_t(list.size());
        for (size_t k = 0; k < list.size(); ++k) aux.params[j][k] = list[k].first;
      }
      aux.replacement[0] = shader_swap::VertexSamplersReplacement();
      aux.replacement[1] = shader_swap::PixelSamplersReplacement();
      aux.sys_hash = r.sys_ok ? DrawSlotSysHash(r) : 0;
      aux.held = native::IndexPool::HeldIndex();
      const PrimitiveProcessor::ProcessingResult& ppr = primitive_processing_result;
      if (r.plan_ok && r.plan.pool_index) {
        cp.native_->Mirror().Hold(ppr.guest_index_base, ppr.host_draw_vertex_count, ppr.host_index_format == xenos::IndexFormat::kInt16, uint32_t(ppr.host_shader_index_endian), ppr.host_primitive_reset_enabled,
                                  r.plan.index_stitched, r.plan.index_min, r.plan.index_max, r.plan.index_has_reset, r.plan.index_count, &aux.held);
      }
      DrawSlotLearn(r, *src.rec_state, src.slot_settings);
    }
  }
  cp.NativeStateTouched(!cp.native_->TakeTruthForeignBind());
  if (mx_possible) cp.IssueDraw_MemexportAfter(true);  // (M4f-B: the written ranges invalidated, the optional read-back - IssueDraw's own body)
  if ((src.typed && !src.crowd_member && !src.record_repeat) || attr.ok || attr.crowd_state || skin.ok) {  // (P7: a held first member has no ranges or window: it is no repeat base either)  // (Session 95: an attribute draw is not recorded as a repeat base)
    // (Session 92, M4f-A) The old path's repeat / lean bases describe an earlier old draw: none of them may continue across a typed draw.
    cp.lean_.valid = false;
    cp.repeat_prev_.valid = false;
    cp.native_repeat_draw_valid_ = false;
  } else {
  if (src.typed && src.record_repeat) cp.lean_.valid = false;  // (Session 93: a recorded typed base; the old lean base still ends)
  ++g_owned_repeat_base_serial;
  if (!cp.native_repeat_draw_) cp.native_repeat_draw_ = std::make_unique<native::Draw>();  // (the cut: the old lean path never ran, so nothing allocated the repeat base)
  *cp.native_repeat_draw_ = native_draw;
  cp.native_repeat_draw_valid_ = true;
  cp.RepeatNoteRecorded(vertex_shader, pixel_shader, uint32_t(prim_type), index_count, index_buffer_info,
                        major_mode_explicit, repeat_epoch, false, emulated_indexed,
                        primitive_processing_result.host_draw_vertex_count);
  cp.repeat_prev_.crash_valid = crash_record != nullptr;
  if (crash_record) cp.repeat_prev_.crash = *crash_record;
  cp.repeat_prev_.merge_stride = L.merge_stride;
  cp.repeat_prev_.float_count = L.float_count;
  cp.repeat_prev_.vertex_endian_key = L.vertex_endian_key;
  cp.repeat_prev_.spec_pixel_shader = pixel_shader;
  cp.repeat_prev_.pixel_spec_key = L.pixel_spec_key;
  cp.repeat_prev_.vertex_mul_key = L.vertex_mul_key;
  cp.repeat_prev_.pixel_ext = L.pixel_ext;
  cp.repeat_prev_.vertex_ext = L.vertex_ext;
}
  if (cp.vertex_census_on_) [[unlikely]] {
    cp.VertexCensusDraw(*vertex_shader, pixel_shader, VCP::kVertexCensusLean, uint32_t(L.host_vertex_shader_type), 1);
  }
  if ((!src.typed || src.crowd_member || src.record_repeat) && !attr.ok && !attr.crowd_state && !skin.ok) cp.RepeatStartRun();
  cp.AfterPostChainCheck(*vertex_shader);
  ++stats.reasons[VCP::kLeanTaken];
  succeeded = true;
  return R::kTaken;
}

// ---------------------------------------------------------------------------------------------------------------------
// (Session 92, M4f-B, latch bit 6) A crowd run's member (or its first member, drawn by the crowd code's guard) through the typed draw: the member's 12 words
// replace c76-c78 in a copy of the item's typed state (OwnedRecordCrowdMemberState), the vertex float buffer is marked out of date (the member's constants
// are not what the ring holds), the typed draw draws it. Nothing is written to the register file or the image. False = the old way draws it (counted).
bool D3dDirectAccess::CrowdMember(void* ctx, const uint32_t* c76, bool& ok) {
  CrowdHook& h = *static_cast<CrowdHook*>(ctx);
  VCP& cp = *h.cp;
  // (Session 95, gpu_d3d_wall bit 128) A run's members differ from each other only in c76-c78, so after one member went the whole way the next one is that draw again with its own vertex floats: the
  // repeat path (the constants uploaded, the base's native draw recorded), not the whole typed draw (the audience atlas: up to ~430 members a frame, each ~14,000 cycles; the windows with them cost
  // twice the others). RepeatIsSameDraw holds the epoch: a page invalidated since sends the member the whole way, which makes it the new base. A base with a cover or vertex box is not repeated.
  const bool member_repeat = h.cut && (REXCVAR_GET(gpu_d3d_wall) & 128u) && !diag::g_armed;
  if (member_repeat && h.repeat_base) {
    static uint64_t tried = 0, repeated_count = 0;
    Shader* const vs = cp.active_vertex_shader();
    const uint64_t epoch = cp.shared_memory_->repeat_epoch();
    const native::Draw* const base = cp.native_repeat_draw_.get();
    bool repeated = false;
    ++tried;
    if (vs && base && cp.repeat_prev_.valid && cp.native_repeat_draw_valid_ && !base->fast_clear && !(base->cover[2] > base->cover[0]) && !(base->bounds[2] > base->bounds[0]) &&
        cp.RepeatIsSameDraw(vs, uint32_t(h.prim_type), h.index_count, h.index_buffer_info, h.major_mode_explicit, epoch)) {
      std::memcpy(h.vs_copy + 4 * 76, c76, 12 * sizeof(uint32_t));
      VulkanPipelineCache::TypedKeyState keys;
      keys.fetch = h.source.fetch;
      OwnedRepeatSourceSet(h.vs_copy, &keys);
      cp.current_constant_buffers_up_to_date_ &= ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatVertex);
      repeated = cp.RepeatDrawFast(vs, epoch);
      OwnedRepeatSourceSet(nullptr, nullptr);
    }
    if (repeated) ++repeated_count;
    if ((tried & 0xFFFF) == 1) REXGPU_INFO("[d3d][wall] crowd members: repeats tried {}, drawn {}; last refusal {}", tried, repeated_count, cp.repeat_refusal_);
    if (repeated) {
      ++h.drawn;
      ok = true;
      return true;
    }
    h.repeat_base = false;
  }
  const OwnedRecordState* ms = OwnedRecordCrowdMemberState(c76);
  if (!ms) {
    ++h.declined;
    if (h.cut) {
      free_counters::NoteCutDrop("crowd member: no base state / no banks");
      ok = true;
      return true;
    }
    return false;
  }
  std::memcpy(h.vs_copy, ms->vs_float, sizeof(h.vs_copy));
  h.source.vs_float = h.vs_copy;
  h.source.ps_float = ms->ps_float;
  h.source.fetch = ms->fetch;
  h.source.color_info = ms->color_info[0].value;
  cp.current_constant_buffers_up_to_date_ &= ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatVertex);
  bool succeeded = true;
  OwnedCrowdSourceSet(&h.source);
  if (!TypedDraw(cp, ms, h.prim_type, h.index_count, h.index_buffer_info, h.major_mode_explicit, succeeded, false, true)) {
    ++h.declined;
    if (h.cut) {  // (M4f-C: the refusal is already counted by reason in TypedDraw; the member is dropped, never drawn the old way)
      ok = true;
      return true;
    }
    OwnedCrowdSourceSet(nullptr);  // (the next batch comes from the register path: the old way drew the member)
    return false;
  }
  ++h.drawn;
  ok = succeeded;
  h.repeat_base = member_repeat && succeeded;
  return true;
}

// (Session 92, M4f-B) An inline (0x7E) draw: ExecutePacketType3Draw's work for an auto-indexed draw, on the typed state paired by OwnedRecordInlineBegin.
}  // namespace rex::graphics::vulkan


// =====================================================================================================================================================
// Session 92, M4f-C: THE CUT's entry points (gpu_d3d_owned_front bit 128). The decoder (owned_record.cpp Run) executes the queue's items in order on its
// live typed state and calls these, one per DRAW item / clear / resolve / inline record. Nothing here reads the register file or a snapshot: the state is
// `live` (OwnedRecordStateNow()). A draw this cannot take is DROPPED and counted by reason (free_counters::NoteCutDrop); there is no old path.
REXCVAR_DECLARE(bool, gpu_dlaa);
REXCVAR_DECLARE(bool, gpu_dlaa_jitter);

namespace rex::graphics::vulkan {

namespace {
// DLAA / DLSS read registers captured by the old draws (the jitter, the camera): not supported under the cut. Logged once, counted by draw.
void NoteDlssUnderCut() {
  // DLSS / DLAA are forced off while the cut is latched (d3d_noring.cpp CutEnforceSettings). The FSR jitter (gpu_dlaa_jitter) is drawn: its own counter.
}
}  // namespace

bool D3dDirectAccess::CutDraw(CommandProcessor& base, const OwnedRecordState& s, uint32_t) {
  VCP& cp = static_cast<VCP&>(base);
  const auto drop = [&](const char* why) {
    free_counters::NoteCutDrop(why);
    return true;
  };
  if (s.draw_initiator.source_select != xenos::SourceSelect::kDMA) return drop("draw: not a DMA draw");
  if (cp.anatomy_active_ || cp.D3dDirectExtentsActive() || REXCVAR_GET(gpu_det_skip_host_draws)) {
    return drop("draw: special mode (anatomy / extents shadow / det skip)");
  }
  NoteDlssUnderCut();
  const reg::VGT_DRAW_INITIATOR init = s.draw_initiator;
  const uint32_t index_size_bytes = init.index_size == xenos::IndexFormat::kInt16 ? sizeof(uint16_t) : sizeof(uint32_t);
  VCP::IndexBufferInfo index_buffer_info;
  index_buffer_info.guest_base = s.dma_base & ~(index_size_bytes - 1);
  index_buffer_info.endianness = s.dma_size.swap_mode;
  index_buffer_info.format = init.index_size;
  index_buffer_info.length = s.dma_size.num_words * index_size_bytes;
  index_buffer_info.count = init.num_indices;
  cp.half2_matched_kind_ = 0;  // (no Half2OnCpDraw under the cut: the draw kind is known)
  ++(cp.native_ib_depth_ ? cp.native_census_.draws_in_ib : cp.native_census_.draws_in_ring);
  if (s.viz_query_cntl.viz_query_ena) {
    ++cp.native_census_.viz_query_draws;
    cp.native_census_.viz_query_dropped += s.viz_query_cntl.kill_pix_post_hi_z ? 1 : 0;
    if (s.viz_query_cntl.kill_pix_post_hi_z) return true;  // (the old path drops such a draw the same way)
  }
  const bool major_mode_explicit = xenos::IsMajorModeExplicit(init.major_mode, init.prim_type);
  ++cp.frame_counters_.issue_draw;
  // The crowd code: one packet id per item (never two items sharing it), the item's own identity (token sequence << 16 | item index) folded.
  const uint64_t cid = s.crowd_id;
  const uint32_t crowd_id = (uint32_t(cid ^ (cid >> 32)) << 1) | 1u;
  cp.SceneCrowdItemAt(crowd_id);
  const uint32_t* c76 = s.vs_float + 4 * 76;
  const bool queues = !cp.scene_crowd_local_.empty() || cp.scene_crowd_queued_.load(std::memory_order_relaxed);
  const bool pre = queues && cp.SceneCrowdPreTake(c76);
  cp.own_crowd_skip_draw_ = pre;
  if (cp.SceneCrowdCensusBegin(cp.own_crowd_skip_draw_, c76)) cp.own_crowd_skip_draw_ = true;
  bool succeeded = true;
  const bool taken = TypedDraw(cp, &s, init.prim_type, init.num_indices, &index_buffer_info, major_mode_explicit, succeeded, false);
  // (Session 95, gpu_d3d_wall bit 2) The runs were searched for this draw's c76 a moment ago and none matched: unless a draw is held for the crowd program, a verify run is under way, more runs were queued
  // since, or the list needs its ageing, SceneCrowdAfterDraw would change nothing: no 4 KB bank copy, no key state, no call. The old bases below are the old path's: the typed draw ended them already.
  const bool crowd_nothing = (REXCVAR_GET(gpu_d3d_wall) & 2u) && queues && !pre && SceneCrowdPreTakeFoundNothing() && !cp.own_crowd_skip_draw_ && !cp.scene_crowd_verify_active_ &&
                             !(crowd_id == cp.scene_crowd_matched_packet_ && cp.counter_ == cp.scene_crowd_matched_frame_) &&  // (review: the predicated-tiling stand-down runs in AfterDraw)
                             cp.scene_crowd_local_.size() <= 1024 && !cp.scene_crowd_queued_.load(std::memory_order_relaxed);
  if (!crowd_nothing && (cp.scene_crowd_verify_active_ || queues || cp.own_crowd_skip_draw_ || !cp.scene_crowd_local_.empty() ||
      cp.scene_crowd_queued_.load(std::memory_order_relaxed))) {
    // The crowd code on the typed state: c76-c78 and the packet id are this item's; the first member's draw and the member loop call back into the typed
    // draw (CrowdMember); the own-crowd functions read the typed arrays. The register file is not read or written.
    CrowdHook hook;
    hook.cp = &cp;
    hook.cut = true;
    hook.prim_type = init.prim_type;
    hook.index_count = init.num_indices;
    hook.index_buffer_info = &index_buffer_info;
    hook.major_mode_explicit = major_mode_explicit;
    std::memcpy(hook.vs_copy, s.vs_float, sizeof(hook.vs_copy));
    hook.source.vs_float = hook.vs_copy;
    hook.source.ps_float = s.ps_float;
    hook.source.fetch = s.fetch;
    hook.source.color_info = s.color_info[0].value;
    VulkanPipelineCache::TypedKeyState keys;  // (the key derivations of the run's batches read the typed state, not the register source)
    keys.program_cntl = s.program_cntl;
    keys.hos_cntl = s.hos_cntl;
    keys.clip_cntl = s.clip_cntl;
    keys.draw_initiator = s.draw_initiator;
    keys.surface_info = s.surface_info;
    keys.context_misc = s.context_misc;
    keys.interpolator_cntl = s.interpolator_cntl;
    keys.depth_info = s.depth_info;
    keys.color_control = s.color_control;
    keys.fetch = s.fetch;
    hook.source.keys = &keys;
    hook.source.clip_cntl = s.clip_cntl.value;
    hook.source.vte_cntl = s.vte_cntl.value;
    std::memcpy(&hook.source.vport_xscale, &s.vport[0], sizeof(uint32_t));
    uint32_t c76_copy[12];
    std::memcpy(c76_copy, c76, sizeof(c76_copy));
    SceneCrowdSetMemberDraw(&D3dDirectAccess::CrowdMember, &hook);
    OwnedCrowdSourceSet(&hook.source);
    cp.SceneCrowdAfterDraw(init.prim_type, init.num_indices, &index_buffer_info, major_mode_explicit, 1u, c76_copy, crowd_id);
    SceneCrowdSetMemberDraw(nullptr, nullptr);
    OwnedCrowdSourceSet(nullptr);
    OwnedFrontTypedNote(kTypedCrowdMembers, hook.drawn);
    OwnedFrontTypedNote(kTypedCrowdDeclined, hook.declined);
    if (hook.drawn) cp.current_constant_buffers_up_to_date_ &= ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatVertex);
    cp.lean_.valid = false;
    cp.repeat_prev_.valid = false;
    cp.native_repeat_draw_valid_ = false;
  }
  cp.SceneCrowdCensusEnd();
  cp.SceneCrowdItemAt(0);
  if (taken && !succeeded) {
    REXGPU_ERROR("D3D token draw({}, {}): Failed in backend (typed, the cut; major_mode={}, explicit_major={}, path_select={}, tess_mode={}, edram_mode={})",
                 uint32_t(init.num_indices), uint32_t(init.prim_type), uint32_t(init.major_mode), uint32_t(major_mode_explicit),
                 uint32_t(s.output_path_cntl.path_select), uint32_t(s.hos_cntl.tess_mode), uint32_t(s.modecontrol.edram_mode));
    free_counters::NoteCutDrop("draw: failed in the backend");
  }
  return true;
}

bool D3dDirectAccess::InlineRingStage(VCP& cp, const uint32_t* w, uint32_t count, uint32_t& address) {
  InlineVertexRing& ring = InlineRing();
  ring.FrameLog(cp.frame_current_);
  const auto refuse = [&](InlineVertexRing::Fallback why) {
    ring.NoteFallback(why);
    return false;
  };
  const uint32_t* words = nullptr;
  uint32_t bytes = 0;
  bool guest_memory = false;
  if (!D3dInlineVertexWords(w, count, words, bytes, guest_memory)) return refuse(InlineVertexRing::kBadRecord);
  if (guest_memory) return refuse(InlineVertexRing::kGuestMemory);
  if (cp.render_target_cache_->GetPath() == RenderTargetCache::Path::kPixelShaderInterlock) return refuse(InlineVertexRing::kInterlock);
  if (!cp.native_ || !cp.native_->OwnedOnlyActive()) return refuse(InlineVertexRing::kNotOwnedOnly);  // (only OwnedRecord binds a draw's own sets; the truth path trusts the current ones)
  // The ring stands in for the whole shared memory in this draw: nothing else of it may be read or written. (Slot 31's stream is vertex fetch constant 95.)
  Shader* const vs = cp.active_vertex_shader();
  Shader* const ps = cp.active_pixel_shader();
  if (!vs) return refuse(InlineVertexRing::kNoShader);
  if (!vs->is_ucode_analyzed()) cp.pipeline_cache_->AnalyzeShaderUcode(*static_cast<VulkanShader*>(vs));
  if (ps && !ps->is_ucode_analyzed()) cp.pipeline_cache_->AnalyzeShaderUcode(*static_cast<VulkanShader*>(ps));
  if (!vs->is_ucode_analyzed() || (ps && !ps->is_ucode_analyzed())) return refuse(InlineVertexRing::kNoShader);
  const uint32_t* const vs_fetch = vs->constant_register_map().vertex_fetch_bitmap;
  if (vs_fetch[0] || vs_fetch[1] || (vs_fetch[2] & ~(1u << 31))) return refuse(InlineVertexRing::kOtherFetch);
  if (ps) {
    const uint32_t* const ps_fetch = ps->constant_register_map().vertex_fetch_bitmap;
    if (ps_fetch[0] || ps_fetch[1] || ps_fetch[2]) return refuse(InlineVertexRing::kPixelFetch);
  }
  if (vs->memexport_eM_written() || (ps && ps->memexport_eM_written())) return refuse(InlineVertexRing::kMemexport);
  if (!ring.Ensure(cp.GetVulkanDevice(), cp.descriptor_set_layout_shared_memory_and_edram_, cp.rename_shared_memory_binding_count_)) {
    return refuse(InlineVertexRing::kNoObjects);
  }
  return ring.Write(words, bytes, cp.GetCurrentSubmission(), cp.GetCompletedSubmission(), address);
}

// The 0x7E inline draw on the live state. DecodeInline has stored the record's runs, bound its shaders (CutLoadShader) and set the draw's words; this stages the
// vertex bytes (InlineStage, the one body), stores slot 31 words 4-5 (the staged address and size) with the fetch notifications, and draws through the typed draw.
bool D3dDirectAccess::CutInline(CommandProcessor& base, const OwnedRecordState&, const uint32_t* w, uint32_t count) {
  VCP& cp = static_cast<VCP&>(base);
  cull_ours::NoteBreak();  // (Session 93: an inline draw is not a draw the cull guard follows)
  OwnedRecordState& s = OwnedRecordStateLive();
  const auto drop = [&](const char* why) {
    OwnedFrontTypedNote(kTypedItems);
    OwnedFrontTypedNote(kTypedInlineFallback);
    free_counters::NoteCutDrop(why);
    return true;
  };
  // (Session 95, item P) On the host vertex ring `physical` is an offset in the ring: slot 31 carries it to the shaders as ever, and only this draw's descriptor set differs.
  uint32_t physical = 0;
  const bool on_ring = REXCVAR_GET(gpu_d3d_inline_vertex_ring) && InlineRingStage(cp, w, count, physical);
  if (!on_ring) {
    bool staged = false;
    physical = D3dInlineStageWords(cp.memory_, w, count, staged);
    if (!staged) return drop("inline: vertex staging failed");
  }
  const uint32_t dwords = (w[4] * w[5]) >> 2;
  const uint32_t words45[2] = {(physical & ~3u) | 3u, 0x10000002u | ((dwords & 0xFFFFFFu) << 2)};
  OwnedExecStore(XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + 6 * 31 + 4, 2, words45);
  if (cp.anatomy_active_ || cp.D3dDirectExtentsActive() || REXCVAR_GET(gpu_det_skip_host_draws)) {
    return drop("inline: special mode (extents / det skip / anatomy)");
  }
  NoteDlssUnderCut();
  const reg::VGT_DRAW_INITIATOR init = s.draw_initiator;
  // A 3-vertex rectangle list may be a fast clear: drawn through the typed draw like any inline draw (a clear drawn as a draw gives the same pixels), counted on its own.
  if (init.prim_type == xenos::PrimitiveType::kRectangleList && init.num_indices == 3) free_counters::NoteCutSkip("inline 3-vertex rectangle list drawn as a draw (possible fast clear)");
  // (An inline draw is never a crowd run's member: the crowd code is skipped for it.)
  const bool major_mode_explicit = xenos::IsMajorModeExplicit(init.major_mode, init.prim_type);
  const auto census = [&]() {
    cp.half2_matched_kind_ = 0;
    ++(cp.native_ib_depth_ ? cp.native_census_.draws_in_ib : cp.native_census_.draws_in_ring);
    if (s.viz_query_cntl.viz_query_ena) {
      ++cp.native_census_.viz_query_draws;
      cp.native_census_.viz_query_dropped += s.viz_query_cntl.kill_pix_post_hi_z ? 1 : 0;
    }
  };
  if (s.viz_query_cntl.viz_query_ena && s.viz_query_cntl.kill_pix_post_hi_z) {
    census();
    OwnedFrontTypedNote(kTypedItems);
    OwnedFrontTypedNote(kTypedInline);
    OwnedFrontTypedNote(kTypedDrawn);
    return true;
  }
  cp.own_crowd_skip_draw_ = false;
  VCP::IndexBufferInfo index_buffer_info;  // (auto index: no buffer)
  bool succeeded = true;
  s.inline_draw = 1;
  s.inline_ring = on_ring;
  const bool taken = TypedDraw(cp, &s, init.prim_type, init.num_indices, &index_buffer_info, major_mode_explicit, succeeded, false);
  s.inline_draw = 0;
  s.inline_ring = 0;
  if (!taken) {
    OwnedFrontTypedNote(kTypedInlineFallback);  // (the reason is counted as a drop by TypedDraw's fallback)
    return true;
  }
  census();
  ++cp.frame_counters_.issue_draw;
  OwnedFrontTypedNote(kTypedInline);
  if (!succeeded) {
    REXGPU_ERROR("D3D_INLINE_DRAW({}, {}, {}): Failed in backend (typed draw, the cut)", uint32_t(init.num_indices), uint32_t(init.prim_type),
                 uint32_t(init.source_select));
    free_counters::NoteCutDrop("inline: failed in the backend");
  }
  return true;
}

namespace {
// The clear / resolve records: the copy executor from the record's host-order block (clears: B's owned clear + the pending table, never ClearStock once the native
// renderer exists - the executor drops a boot clear before it gets here; resolves: the M3a owned resolve).
bool CutClear(CommandProcessor& cp, const OwnedRecordState&, const uint32_t* record, uint32_t) {
  cull_ours::NoteBreak();
  // (Session 95, post lite: a clear of a dead target revives it - what the copies after it hold is the clear's, not a dropped pass's)
  D3dDirectAccess::PostLiteClearRevive(static_cast<VulkanCommandProcessor&>(cp), record + kD3dCopyHeaderDwords, record[3]);
  return OwnedExecCopy(cp, kD3dCopyClear, record + kD3dCopyHeaderDwords, record[3]);
}
bool CutResolve(CommandProcessor& cp, const OwnedRecordState&, const uint32_t* record, uint32_t) {
  cull_ours::NoteBreak();
  return OwnedExecCopy(cp, kD3dCopyResolve, record + kD3dCopyHeaderDwords, record[3]);
}
bool g_cut_wired = [] {
  OwnedExecEntries& e = OwnedExecEntrySlots();
  e.draw = &D3dDirectAccess::CutDraw;
  e.clear = &CutClear;
  e.resolve = &CutResolve;
  e.inline_draw = &D3dDirectAccess::CutInline;
  return true;
}();
}  // namespace

}  // namespace rex::graphics::vulkan

// ---------------------------------------------------------------------------------------------------------------------
// (Session 92, M4f-C) The per-draw skips of IssueDraw that a player setting turns on, for the typed draw under the cut: hide HUD, the shadow chain (dynamic
// shadows off), crowd density and half-rate wrestler shadows. The same predicates as vulkan/command_processor.cpp IssueDraw (~10441-10640), from the typed
// state's words (shader hashes, RB_COLOR_INFO / RB_DEPTH_INFO / RB_SURFACE_INFO, c76-c78) instead of the register file. Returns the reason a draw is
// skipped (counted as a skip, never a drop), nullptr = draw it.
REXCVAR_DECLARE(bool, gpu_hide_hud);
REXCVAR_DECLARE(int32_t, readback_resolve_override);

namespace rex::graphics::vulkan {

// Session 93 (gpu_d3d_cull_ours): true = this draw of ours is proven outside the view (cull_ours.cpp lists every guard). The guards only the command processor can answer are here.
bool D3dDirectAccess::CutCullOurs(VCP& cp, const OwnedRecordState& s, bool crowd_member) {
  cull_ours::Request req;
  req.mem = cp.memory_;
  req.s = &s;
  req.crowd_member = crowd_member;
  req.occlusion_query = cp.active_occlusion_query_.valid;
  const uint64_t vs_hash = s.shader[0].hash;
  if (cull_ours::CullableShader(vs_hash)) {
    req.truth_pass_open = cp.native_ && cp.native_->TruthPassOpen();
    // both shaders the draw's own, analysed, neither writes memory (the memexport test needs the analysis, as CutPreDrawSkip)
    Shader* const vs = cp.active_vertex_shader();
    Shader* const ps = cp.active_pixel_shader();
    bool known = vs && vs->ucode_data_hash() == vs_hash;
    if (known && !vs->is_ucode_analyzed()) cp.pipeline_cache_->AnalyzeShaderUcode(*static_cast<VulkanShader*>(vs));
    // the one binding the box test reads: a single vertex binding, in the registered fetch constant, 8 dwords a vertex
    known = known && vs->is_ucode_analyzed() && vs->vertex_bindings().size() == 1 && uint32_t(vs->vertex_bindings()[0].fetch_constant) == cull_ours::CullableFetch(vs_hash) &&
            vs->vertex_bindings()[0].stride_words == 8;
    if (known && !vs->is_ucode_analyzed()) cp.pipeline_cache_->AnalyzeShaderUcode(*static_cast<VulkanShader*>(vs));
    known = known && vs->is_ucode_analyzed() && vs->memexport_eM_written() == 0;
    if (known && s.shader[1].dwords) {
      known = ps && ps->ucode_data_hash() == s.shader[1].hash;
      if (known && !ps->is_ucode_analyzed()) cp.pipeline_cache_->AnalyzeShaderUcode(*static_cast<VulkanShader*>(ps));
      known = known && ps->is_ucode_analyzed() && ps->memexport_eM_written() == 0;
    }
    req.shaders_known = known;
    struct Ctx {
      VCP* cp;
      const OwnedRecordState* s;
    } ctx{&cp, &s};
    // The host clip space of the draw: the viewport the draw would get (OwnedFrontPrepare's call); its ndc scale / offset on x / y go into the box test. No wide / tall factor.
    req.viewport = [](void* c, float* host) -> bool {
      const Ctx& x = *static_cast<const Ctx*>(c);
      VCP& p = *x.cp;
      const OwnedRecordState& st = *x.s;
      if (REXCVAR_GET(gpu_wide_factor) > 1.001 || REXCVAR_GET(gpu_tall_factor) > 1.001 || REXCVAR_GET(gpu_wide_all)) return false;
      draw_util::HostViewportRegs vr;
      vr.clip_cntl = st.clip_cntl;
      vr.vte_cntl = st.vte_cntl;
      vr.sc_mode_cntl = st.sc_mode_cntl;
      vr.su_vtx_cntl = st.su_vtx_cntl;
      vr.window_offset = st.window_offset;
      vr.depth_info = st.depth_info;
      std::memcpy(vr.vport, st.vport, sizeof(vr.vport));
      const bool host_rt = p.render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets;
      const ui::vulkan::VulkanDevice::Properties& props = p.GetVulkanDevice()->properties();
      draw_util::ViewportInfo vi;
      draw_util::GetHostViewportInfo(vr, p.texture_cache_->draw_resolution_scale_x(), p.texture_cache_->draw_resolution_scale_y(), false, props.maxViewportDimensions[0],
                                     props.maxViewportDimensions[1], true, draw_util::GetNormalizedDepthControl(st.modecontrol, st.depth_control),
                                     host_rt && p.render_target_cache_->depth_float24_convert_in_pixel_shader(), host_rt, false, vi, p.tiling_viewport_fix_);
      if (!vi.xy_extent[0] || !vi.xy_extent[1]) return false;
      host[0] = vi.ndc_scale[0];
      host[1] = vi.ndc_scale[1];
      host[2] = vi.ndc_offset[0];
      host[3] = vi.ndc_offset[1];
      return std::isfinite(host[0]) && std::isfinite(host[1]) && std::isfinite(host[2]) && std::isfinite(host[3]);
    };
    req.viewport_ctx = &ctx;
    return cull_ours::Evaluate(req);
  }
  return cull_ours::Evaluate(req);
}

void PostLiteNoteCopy(uint32_t dest_base) {
  PostLiteState& pl = g_post_lite;
  if (!pl.blur_copy_pending) return;
  pl.blur_copy_pending = false;
  if (pl.blur_seen && pl.blur_seen <= PostLiteState::kPairs) pl.blur_page[pl.blur_seen - 1] = (dest_base >> 12) & 0x1FFFF;
}

void D3dDirectAccess::PostLiteClearRevive(VCP& cp, const uint32_t* sem, uint32_t count) {
  if (cp.nf_dead_targets_.empty() || count <= d3d_clear_sem::kColorInfo) return;
  const uint32_t key = VCP::NfRt0KeyOf(sem[d3d_clear_sem::kColorInfo], sem[d3d_clear_sem::kSurfaceInfo]);
  const auto it = std::find(cp.nf_dead_targets_.begin(), cp.nf_dead_targets_.end(), key);
  if (it != cp.nf_dead_targets_.end()) cp.nf_dead_targets_.erase(it);
}

const char* D3dDirectAccess::CutPreDrawSkip(VCP& cp, const OwnedRecordState& s) {
  // IssueDraw: after the post chain's last pass the rest of the frame is the HUD (hide HUD)
  if (cp.hide_hud_after_pass_ && REXCVAR_GET(gpu_hide_hud)) {
    cp.repeat_prev_.valid = false;
    return "hide HUD (after the post chain)";
  }
  Shader* const vs = cp.active_vertex_shader();
  if (!vs) return nullptr;
  const uint64_t vs_hash = s.shader[0].hash;
  const uint64_t ps_hash = s.shader[1].dwords ? s.shader[1].hash : 0;  // (a depth-only draw has no pixel shader: hash 0)
  if (!vs->is_ucode_analyzed()) cp.pipeline_cache_->AnalyzeShaderUcode(*static_cast<VulkanShader*>(vs));  // (as Build does: the memexport test needs the analysis)
  const bool memexport_used_vertex = vs->is_ucode_analyzed() && vs->memexport_eM_written() != 0 && cp.GetVulkanDevice()->properties().vertexPipelineStoresAndAtomics;
  // Session 95 (item R, P1, gpu_native_post_lite_drop): post passes the last combine did not use are not drawn; the copies out of a dead
  // target are skipped by IssueCopyFrom's existing nf_dead_targets_ test (the page keeps the last image, mapped and defined).
  if (REXCVAR_GET(gpu_native_post_lite_drop) && !memexport_used_vertex) {
    PostLiteState& pl = g_post_lite;
    const uint64_t frame = cp.frame_current_;
    if (frame != pl.frame) {
      if (pl.frame) {
        ++pl.frames_seen;
        if (pl.frame_drops) {
          ++pl.frames_with_drops;
          pl.drops_sum += pl.frame_drops;
        }
      }
      pl.frame = frame;
      pl.frame_drops = 0;
      cp.nf_dead_targets_.clear();  // (a target is dead for the frame its draws were dropped in)
      std::copy(std::begin(pl.blur_page), std::end(pl.blur_page), std::begin(pl.last_blur_page));
      pl.last_blurs = pl.blur_seen;
      std::fill(std::begin(pl.blur_page), std::end(pl.blur_page), 0u);
      pl.blur_seen = 0;
      pl.blur_copy_pending = false;
      if (frame >= pl.next_log && pl.frames_seen) {
        pl.next_log = frame + 600;
        REXGPU_INFO("[s95][post_lite] {} frames: passes dropped per frame {:.1f} (draws dropped {}: glare {}, bloom chain {}, DoF {}), frames with drops {}, "
                    "copies skipped {} (kept because they clear: {})", pl.frames_seen, double(pl.drops_sum) / double(pl.frames_seen), pl.glare + pl.bloom + pl.dof,
                    pl.glare, pl.bloom, pl.dof, pl.frames_with_drops, cp.nf_census_.glare_copies_skipped, cp.nf_census_.glare_copies_kept_clear);
        if (pl.near_dropped) REXGPU_INFO("[s95][post_lite] near-blur pair passes dropped so far {}", pl.near_dropped);
      }
    }
    if (ps_hash == kPlEntranceCombine || ps_hash == kPlMatchCombine) {
      const bool entrance = ps_hash == kPlEntranceCombine;
      const auto zero = [&](uint32_t reg) { return (s.ps_float[4 * reg] << 1) == 0u; };  // c.x == +-0.0f
      pl.combine_frame = frame;
      pl.entrance = entrance;
      pl.glare_off = entrance && zero(32);
      pl.bloom_off = entrance && zero(31);
      // (Session 95, gpu_native_post_lite_near: z = 1 - depth is never below 0, so with c21.x <= 0 the near blur's branch is never taken; its page only matters if it is that fetch's alone.)
      pl.near_pair = -1;
      pl.near_down = -1;
      float c21x;
      std::memcpy(&c21x, &s.ps_float[4 * 21], sizeof(c21x));
      if (entrance && REXCVAR_GET(gpu_native_post_lite_near) && c21x <= 0.0f) {
        const uint32_t page2 = (s.fetch[2 * 6 + 1] >> 12) & 0x1FFFF, page6 = (s.fetch[6 * 6 + 1] >> 12) & 0x1FFFF;
        pl.scene_page = (s.fetch[0 * 6 + 1] >> 12) & 0x1FFFF;
        for (uint32_t k = 0; k < std::min(pl.blur_seen, PostLiteState::kPairs); ++k) {
          if (pl.blur_page[k] == page2 && page2 != page6) {
            pl.near_pair = int32_t(k);
            break;
          }
        }
        // (A frame the pair was dropped in has no copy of it: the pair stays dropped while the last drawn frame's list says so.)
        if (pl.near_pair < 0) {
          for (uint32_t k = 0; k < std::min(pl.last_blurs, PostLiteState::kPairs); ++k) {
            if (pl.last_blur_page[k] == page2 && page2 != page6) {
              pl.near_pair = int32_t(k);
              break;
            }
          }
        }
      }
    }
    const char* why = nullptr;
    if (pl.combine_frame && frame - pl.combine_frame <= 3) {
      // (632189BA is also the entrance depth-of-field blur on 320x180 surfaces: the glare / bloom chain is on the 160-pixel-pitch ones only)
      const bool small_target = (s.surface_info.value & 0x3FFF) == 160;
      const bool quarter_target = (s.surface_info.value & 0x3FFF) == 320;
      if (quarter_target && ps_hash == 0x9C85DA5F0879DA76ull) {
        // (The pair's downsample: of the scene's page, with the blurs before it counted. The same shader also fills a 256x256 box of this surface, from a shadow map.)
        const bool near_hit = pl.near_pair >= 0 && pl.glare_off && pl.bloom_off && int32_t(pl.blur_seen) == pl.near_pair && ((s.fetch[0 * 6 + 1] >> 12) & 0x1FFFF) == pl.scene_page;
        if (near_hit) {
          why = "post lite: near blur not used by the combine";
          pl.near_down = pl.near_pair;
        }
      } else if (quarter_target && ps_hash == 0x632189BAFAE233F0ull) {
        const uint32_t k = pl.blur_seen++;
        if (pl.near_pair >= 0 && pl.glare_off && pl.bloom_off && int32_t(k) == pl.near_pair && pl.near_down == pl.near_pair) {
          why = "post lite: near blur not used by the combine";
          if (k < PostLiteState::kPairs) pl.blur_page[k] = pl.last_blur_page[k];
        } else {
          pl.blur_copy_pending = true;
        }
      }
      if (why) {
        ++pl.near_dropped;
        if (!pl.near_logged) {
          pl.near_logged = true;
          REXGPU_INFO("[s95][post_lite] frame {}: first near-blur pair pass dropped (pair {}, its page 0x{:X}; the combine's c21.x is not above 0)", frame, pl.near_pair, pl.last_blur_page[std::min<uint32_t>(pl.near_pair, PostLiteState::kPairs - 1)]);
        }
      }
      if (why) {
      } else if (small_target && pl.glare_off && (ps_hash == 0x75641266549A40C0ull || ps_hash == 0x9C84E33D39058FB0ull)) {
        ++pl.glare;
        why = "post lite: star glare not used by the combine";
      } else if (small_target && pl.glare_off && pl.bloom_off &&
                 (ps_hash == 0x156FA9A105B45F9Bull || ps_hash == 0x632189BAFAE233F0ull || ps_hash == 0x755AC86C4301F2BBull)) {
        // (the bloom chain feeds the glare's first pass, and 156FA9A1 starts both: dropped only when neither is used)
        ++pl.bloom;
        why = "post lite: bloom chain not used by the combine";
      } else if (!REXCVAR_GET(gpu_depth_of_field) && (!pl.entrance || (pl.glare_off && pl.bloom_off)) &&
                 // (the entrance's bloom / glare bright pass samples the DoF blur's page: the DoF pair is dropped in an entrance only when neither is used)
                 (ps_hash == 0xA7AD68BE21C46E7Aull || ps_hash == 0xA5C9AC54E4868DC1ull)) {
        ++pl.dof;
        why = "post lite: depth of field off";
      }
    }
    const uint32_t dead_key = VCP::NfRt0KeyOf(s.color_info[0].value, s.surface_info.value);
    if (why) {
      if (std::find(cp.nf_dead_targets_.begin(), cp.nf_dead_targets_.end(), dead_key) == cp.nf_dead_targets_.end()) cp.nf_dead_targets_.push_back(dead_key);
      ++pl.frame_drops;
      if (!pl.logged) {
        pl.logged = true;
        REXGPU_INFO("[s95][post_lite] frame {}: first dropped post pass (ps {:016X}: {}); its target's copies are skipped, the page keeps the last image", frame,
                    ps_hash, why);
      }
      cp.repeat_prev_.valid = false;
      cp.lean_.valid = false;
      return why;
    }
    // (a drawn draw into a dead target makes that target's copies live again, as IssueDraw does for the old path)
    if (!cp.nf_dead_targets_.empty()) {
      const auto it = std::find(cp.nf_dead_targets_.begin(), cp.nf_dead_targets_.end(), dead_key);
      if (it != cp.nf_dead_targets_.end()) cp.nf_dead_targets_.erase(it);
    }
  }
  // gpu_skip_frame_draws (load turbo): the old path's rule - never a memexport draw (its results are written to guest memory)
  if (cp.skip_frame_draws_) {
    Shader* const ps = cp.active_pixel_shader();
    if (ps && !ps->is_ucode_analyzed()) cp.pipeline_cache_->AnalyzeShaderUcode(*static_cast<VulkanShader*>(ps));
    const bool memexport_used_pixel = ps && ps->is_ucode_analyzed() && ps->memexport_eM_written() != 0;
    if (!memexport_used_vertex && !memexport_used_pixel) {
      ++cp.skip_frame_draws_skipped_;
      return "load turbo (gpu_skip_frame_draws)";
    }
  }
  const uint32_t color_info = s.color_info[0].value, depth_info = s.depth_info.value, surface_info = s.surface_info.value;
  if (wwe_shadow_chain::g.off) {
    namespace sc = wwe_shadow_chain;
    sc::State& chain = sc::g;
    const uint32_t kind = OwnedDynamicShadowKind(ps_hash, vs_hash);
    const uint64_t chain_key = OwnedShadowChainKey(color_info, depth_info);
    if (kind) {
      ++chain.casters;
      if (!OwnedShadowChainHas(chain.excluded, chain_key) && !OwnedShadowChainHas(chain.keys, chain_key)) {
        if (chain.keys.size() < sc::kMaxKeys) {
          chain.keys.push_back(chain_key);
        } else {
          chain.disabled = true;
        }
      }
      OwnedDynamicShadowsNoteSkip(kind, cp.frame_current_);
      cp.repeat_prev_.valid = false;
      cp.lean_.valid = false;
      return "dynamic shadows off: caster draw";
    }
    if (REXCVAR_GET(readback_resolve_override) <= 0) {
      if (OwnedShadowChainNotShadow(ps_hash)) {
        if (!OwnedShadowChainHas(chain.excluded, chain_key) && chain.excluded.size() < 64) chain.excluded.push_back(chain_key);
        const auto it = std::find(chain.keys.begin(), chain.keys.end(), chain_key);
        if (it != chain.keys.end()) {
          chain.keys.erase(it);
          if (chain.skip) chain.reprime_pending = true;
        }
      } else if (OwnedShadowChainHas(chain.keys, chain_key) && OwnedShadowChainPitchOk(surface_info, ps_hash, false)) {
        ++chain.seen;
        if (chain.skip && !memexport_used_vertex) {
          ++chain.draws_skipped;
          cp.repeat_prev_.valid = false;
          cp.lean_.valid = false;
          return "dynamic shadows off: chain surface";
        }
      }
    }
  }
  // gpu_scene_crowd_density: a crowd member above the percentage (not a run's first member our crowd renderer holds)
  if (const int32_t density = REXCVAR_GET(gpu_scene_crowd_density);
      density < 100 && !cp.own_crowd_skip_draw_ && OwnedCrowdDensityRejects(vs_hash, surface_info, s.vs_float + 4 * 76, density)) {
    cp.repeat_prev_.valid = false;
    cp.lean_.valid = false;
    return "crowd density";
  }
  // gpu_shadow_half_rate: the wrestlers' shadow maps on alternate frames (a map's first draw decides for the whole map)
  if (REXCVAR_GET(gpu_shadow_half_rate) && REXCVAR_GET(readback_resolve_override) <= 0 && !memexport_used_vertex && ps_hash == 0xDA494C9383E18961ull) {
    const uint32_t key = VCP::NfRt0KeyOf(color_info, surface_info);
    if (!cp.shadow_open_ || cp.shadow_open_key_ != key) {
      uint32_t index = 0;
      for (const auto& done : cp.shadow_maps_done_) {
        if (done.first == key) index = done.second;
      }
      auto map = std::find_if(cp.shadow_maps_.begin(), cp.shadow_maps_.end(),
                              [key, index](const VCP::ShadowMap& m) { return m.key == key && m.index == index; });
      if (map == cp.shadow_maps_.end()) {
        cp.shadow_maps_.push_back({key, index, UINT64_MAX});
        map = cp.shadow_maps_.end() - 1;
      }
      // (gpu_native_owned_shadow_skip: no stagger by index - every map skips on the same frames, so the passes that read them can skip too)
      const uint32_t stagger = (shadow_skip::Wanted() || shadow_skip::HalfPhase()) ? 0 : index;
      const bool skip_map = map->last_drawn + 1 == cp.shadow_frame_ && ((cp.shadow_frame_ + stagger) & 1);
      if (!skip_map) map->last_drawn = cp.shadow_frame_;
      shadow_skip::NotePhase(0, cp.shadow_frame_, !skip_map);
      shadow_skip::NoteMap(key, skip_map);
      cp.shadow_open_ = true;
      cp.shadow_open_key_ = key;
      cp.shadow_open_index_ = index;
      cp.shadow_open_skip_ = skip_map;
    }
    if (cp.shadow_open_skip_) {
      ++cp.shadow_skipped_;
      cp.repeat_prev_.valid = false;
      cp.lean_.valid = false;
      return "half-rate wrestler shadow map";
    }
    ++cp.shadow_drawn_;
  }
  // (Session 95, F6) the skipped map's other draws - the props map's circle shadows (PS D05BFEA1) and the pass's flat fills - are dropped at the
  // source too, as the dynamic-shadow block does, so no clear-only pass of it is left. Only while its pass is open (before its first copy).
  if (REXCVAR_GET(gpu_shadow_half_rate) && shadow_skip::Wanted() && cp.shadow_open_ && cp.shadow_open_skip_ && !memexport_used_vertex &&
      (ps_hash == 0xD05BFEA12F4350ADull || ps_hash == 0xA4A965C189287B99ull) &&
      VCP::NfRt0KeyOf(color_info, surface_info) == cp.shadow_open_key_) {
    ++cp.shadow_skipped_;
    cp.repeat_prev_.valid = false;
    cp.lean_.valid = false;
    return "half-rate wrestler shadow map: circle / flat draw";
  }
  // gpu_native_owned_shadow_skip (Session 95, K): the passes that read only the maps, in a frame where every map was skipped
  if (shadow_skip::Wanted() && REXCVAR_GET(gpu_shadow_half_rate) && REXCVAR_GET(readback_resolve_override) <= 0 && !memexport_used_vertex) {
    if (shadow_skip::PreDraw(ps_hash, VCP::NfRt0KeyOf(color_info, surface_info), color_info, surface_info, cp.shadow_frame_,
                             cp.shadow_frame_maps_drawn_, cp.shadow_frame_maps_skipped_)) {
      cp.repeat_prev_.valid = false;
      cp.lean_.valid = false;
      return "shadow skip: dependent pass";
    }
  }
  // Session 95 (resolves, gpu_native_owned_ring_half): the ring's shadow chain on alternate frames
  if (shadow_skip::RingWanted() && REXCVAR_GET(readback_resolve_override) <= 0 && !memexport_used_vertex) {
    if (shadow_skip::RingPreDraw(ps_hash, VCP::NfRt0KeyOf(color_info, surface_info), color_info, surface_info, cp.shadow_frame_)) {
      cp.repeat_prev_.valid = false;
      cp.lean_.valid = false;
      return "ring half rate: ring shadow chain pass";
    }
  }
  // Session 95 (resolves, gpu_native_owned_strip_half): the light-cube strips on alternate frames (draw skipped, every copy of the pass dropped)
  if (shadow_skip::StripWanted() && REXCVAR_GET(readback_resolve_override) <= 0 && !memexport_used_vertex) {
    if (shadow_skip::StripPreDraw(ps_hash, VCP::NfRt0KeyOf(color_info, surface_info), color_info, surface_info, cp.shadow_frame_)) {
      cp.repeat_prev_.valid = false;
      cp.lean_.valid = false;
      return "strip half rate: light-cube strip pass";
    }
  }
  shadow_skip::NoteDrawn(VCP::NfRt0KeyOf(color_info, surface_info));  // (a draw that is drawn: its target is no longer a skipped one)
  return nullptr;
}

}  // namespace rex::graphics::vulkan
