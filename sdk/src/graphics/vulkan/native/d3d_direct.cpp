// wwe2k14 patch (Session 87, D3D layer family F4; docs\native\d3d\F4_S2_CONTRACTS.md, map F = F_direct_and_noring.md in the
// game repository): the D3D token's draw item executed by a lean-draw variant that takes its knowledge from the record.
//
// Where it sits. CommandProcessor::ExecutePacketType3_D3D_TOKEN applies a token's register runs and shader loads as before
// (the register file stays the backend's input: UpdateBindings, the system constants, PrimitiveProcessor and the native
// draw description all read it; replacing that is a constants layer of its own, not this file). For each DRAW item it
// calls D3dDirectDrawItem (include\rex\graphics\d3d_direct.h) with --gpu_d3d_direct on. This file registers the handler.
//
// What the handler does instead of ExecutePacketType3Draw + IssueDrawLean (map F section 1.1 row numbers):
//  - rows 3 (inference of what changed) and the Half 2 attribution gate: the library-draw kind is KNOWN (a token is a
//    library indexed draw), and the dirty classes are the ones the token's own writes noted (CommandProcessor::repeat_dirty_,
//    set by exactly the registers the record carried; no register diff). gpu_native_lean_benign_writes is forced on for the
//    layer: a shader item (IM_LOAD's work) and the library's trailing flush-fetch registers are the record's own, and the
//    lean template compares the shaders itself.
//  - row 3 / 17 (the state epoch): a render-state write is judged by VALUE against the template's registers
//    (gpu_native_lean_same_values behaviour, always on here), so a block the library re-sent unchanged keeps the template
//    (draw description, pipeline objects, viewport, system-constant inputs, layout, claim) and only the per-draw pieces run.
//  - row 8 (vertex ranges): each used vertex fetch slot's range is stamped with SharedMemory::repeat_epoch() read before
//    the draw requested anything (the page-validity epoch: it moves whenever any page stopped being valid or a watch fired);
//    a slot whose fetch constant and the epoch both equal the stamp skips RequestRange and replays the accounting the
//    stock in-sync branch does (hoist mark, skipped-page check, RAM stats). vulkan_vertex_residency_cache_off stays true:
//    this is not the in_sync bit. gpu_d3d_direct_verify checks every stamp hit against IsRangeUpToDate.
//  - rows 10 / 11 / 16: kept as the lean path has them (RequestTextures returns on its in-sync mask after one atomic load;
//    UseSampler hits its per-submission cache; UpdateBindings uploads only the buffers the writes marked out of date).
//  - rows 1, 2, 7, 13-15, 18: kept (shader translation by hash + modification, pipeline cache, claim, owned frame).
// A draw this path cannot take is handed back: kPass (a special mode: the stock IssueDrawLean decides), kFull (the lean
// template does not hold: IssueDraw runs with repeat_dirty_ as it was), or the draw packet path (the DRAW item was not a
// plain DMA draw, or a capture mode is on). All counted and printed ([d3d][direct]).
//
// Command processor thread only. No game data, no addresses: the token's words are the game layer's.
#include <rex/graphics/vulkan/native/native_renderer.h>

#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/d3d_direct.h>
#include <rex/graphics/d3d_feed.h>  // (Session 87, stage 3)
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
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "../../shadow_chain.h"  // wwe_shadow_chain::g (the lean path's own gate for the shadow chain)
#include "d3d_feed.h"  // (Session 87, stage 3) the record's register image
#include "owned_record.h"  // (Session 92, M4b) the typed state of this draw item
#include "owned_front_draw.h"  // (Session 92, M4d) the crowd shadow
#include "../../crowd_member_hook.h"
#include "d3d_direct_access.h"  // (Session 92, M4f-B)

REXCVAR_DECLARE(bool, gpu_d3d_direct);  // (defined in owned_front_own.cpp: read by the base library too)
REXCVAR_DECLARE(bool, gpu_d3d_direct_verify);  // (defined in owned_front_own.cpp with DrawFromBase)
REXCVAR_DEFINE_INT32(gpu_d3d_direct_abi, 1, "GPU/Native",
                     "Session 87 (D3D layer F4): capability marker - this build has the direct draw handler. Read-only, do "
                     "not set")
    .range(1, 1);

REXCVAR_DECLARE(bool, gpu_native_lean_state_cache_verify);
REXCVAR_DECLARE(bool, gpu_d3d_own_full);
REXCVAR_DECLARE(bool, gpu_d3d_own_crowd);
REXCVAR_DECLARE(uint32_t, gpu_d3d_owned_front);
REXCVAR_DECLARE(bool, gpu_native_lean_draws);
REXCVAR_DECLARE(bool, gpu_native_lean_draws_verify);
REXCVAR_DECLARE(bool, gpu_native_lean_dirty_census);
REXCVAR_DECLARE(bool, gpu_native_lean_multi_base);
REXCVAR_DECLARE(bool, gpu_native_lean);
REXCVAR_DECLARE(bool, gpu_native_truth_host_rt);
REXCVAR_DECLARE(bool, gpu_det_skip_host_draws);
REXCVAR_DECLARE(bool, gpu_shadow_half_rate);
REXCVAR_DECLARE(bool, vulkan_repeat_draw_fast);
REXCVAR_DECLARE(int32_t, gpu_scene_crowd_density);
REXCVAR_DECLARE(int32_t, gpu_debug_draw_census_count);

namespace rex::graphics::vulkan {

namespace {
DirectCensus& g_c = g_direct_census;
std::array<RangeStamp, 96>& g_stamps = g_direct_stamps;
constexpr uint32_t kPrintEvery = 4096;
}  // namespace


namespace {
bool g_registered = [] {
  D3dDirectSetDrawHandler(&D3dDirectAccess::DrawItem);
  return true;
}();
}  // namespace

void D3dDirectAccess::Print(VCP& cp) {
  DirectCensus& c = g_c;
  REXGPU_INFO(
      "[d3d][direct] frame {}: {} draw items: {} drawn direct ({} of them repeats), {} to the stock lean path (special "
      "mode), {} to IssueDraw (template did not hold: no base {}, state written {}, shaders {}, primitive {}, sampler {}, "
      "claim {}, vertex fetch / mode {}, repeat path refused {}, other {}), {} to the draw packet path (mode {}, not a DMA "
      "draw {}); render state re-sent with the template's values {}, bases restored {} ({} of them for other shaders; "
      "restore not tried: {} switch off, {} outside write; {} misses, {} of them with every base slot held this frame); vertex "
      "ranges: {} stamped hits, {} requested, {} verify mismatches; repeat path refused by (mode detail: no dyn-offsets state {}, per-draw sets {}, offsets verify {}, "
      "anatomy {}, marks/recstats/skip {}, crash checkpoints {}, unclipped VS on CPU {}, render target path {}, no pipeline "
      "layout {}): mode (old code) {}, other buffer out "
      "of date {}, pipeline changed {}, sets bound changed {}, endian key {}, PS spec/VS mul/ext {}, texture served {}, no "
      "native base {}, truth claim {}, unwritten native {}, pool page {}, set preparation {}, unset {}",
      cp.frame_current_, c.items, c.taken, c.repeats, c.to_stock_lean, c.to_full, c.reasons[VCP::kLeanNoBase],
      c.reasons[VCP::kLeanDirty], c.reasons[VCP::kLeanShader], c.reasons[VCP::kLeanPrimitive], c.reasons[VCP::kLeanSampler],
      c.reasons[VCP::kLeanClaim], c.reasons[VCP::kLeanMode], c.reasons[VCP::kLeanRepeat],
      // (the reasons that are not printed above: what is left of to_full)
      c.to_full - std::min<uint64_t>(c.to_full, c.reasons[VCP::kLeanNoBase] + c.reasons[VCP::kLeanDirty] +
                                                    c.reasons[VCP::kLeanShader] + c.reasons[VCP::kLeanPrimitive] +
                                                    c.reasons[VCP::kLeanSampler] + c.reasons[VCP::kLeanClaim] +
                                                    c.reasons[VCP::kLeanMode] + c.reasons[VCP::kLeanRepeat]),
      c.fallback_mode + c.fallback_source, c.fallback_mode, c.fallback_source, c.same_values,
      c.restored, c.restored_for_shaders, c.restore_off, c.restore_outside, c.restore_miss, c.restore_ring_full,
      c.range_hits, c.range_requests, c.range_mismatches, c.repeat_refused[13], c.repeat_refused[14], c.repeat_refused[15], c.repeat_refused[16],
      c.repeat_refused[17], c.repeat_refused[18], c.repeat_refused[19], c.repeat_refused[20], c.repeat_refused[21],
      c.repeat_refused[1], c.repeat_refused[2], c.repeat_refused[3],
      c.repeat_refused[4], c.repeat_refused[5], c.repeat_refused[6], c.repeat_refused[7], c.repeat_refused[8],
      c.repeat_refused[9], c.repeat_refused[10], c.repeat_refused[11], c.repeat_refused[12], c.repeat_refused[0]);
  if (c.dirty_misses) {
    // The ten registers that most often differ from the base's at a "state written" handback.
    std::array<std::pair<uint64_t, uint32_t>, 0x400> order;
    for (uint32_t r = 0; r < 0x400; ++r) order[r] = {c.diff_regs[r], r};
    std::partial_sort(order.begin(), order.begin() + 10, order.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::string top;
    for (int i = 0; i < 10 && order[i].first; ++i) {
      top += fmt::format("{}0x{:04X} x{}", i ? ", " : "", 0x2000 + order[i].second, order[i].first);
    }
    REXGPU_INFO("[d3d][direct] state written handbacks after a failed restore: {} ({} with an outside write or packet); registers "
                "differing from the base's, most often: {}", c.dirty_misses, c.dirty_outside, top);
  }
  if (c.cross_restores || c.cache_verify_handbacks) {
    REXGPU_INFO(
        "[d3d][direct] state cache (cross-frame bases) frame {}: {} restores of an earlier frame's base, {} claimed again and "
        "drawn; verify: {} handed to the full path, {} compared, {} differing from the fresh derivation (cache generation "
        "{})",
        cp.frame_current_, c.cross_restores, c.cross_claimed, c.cache_verify_handbacks,
        cp.lean_cache_verify_checked_, cp.lean_cache_verify_mismatches_, cp.lean_cache_gen_);
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// The DRAW item: ExecutePacketType3Draw's work for a plain DMA draw, with the dispatch changed.
bool D3dDirectAccess::DrawItem(CommandProcessor& base, rex::memory::RingBuffer* reader) {
  using Feed = D3dFeedAccess;
  if (!REXCVAR_GET(gpu_d3d_direct)) {
    D3dFeedLastPacket() = 0x3FB;  // (Session 90: the census key of a Foreign that no packet announced - was the previous item's)
    D3dFeedForeign(base);  // (runs the feed took while the switch was on: the stock draw reads the register file)
    return false;
  }
  VCP& cp = static_cast<VCP&>(base);
  DirectCensus& c = g_c;
  ++c.items;
  // (Session 92, M4b: the draw item pairs with the decoder's snapshot of the same position - killed, fallback or drawn.)
  if (REXCVAR_GET(gpu_d3d_owned_front) & 2u) OwnedRecordDrawAdvance();
  // The modes whose per-draw work only the stock wrapper does (frame-anatomy rows, the extents shadow, the det skip) or
  // that have no native renderer to draw into.
  if (cp.anatomy_active_ || cp.D3dDirectExtentsActive() || REXCVAR_GET(gpu_det_skip_host_draws) || !cp.native_ ||
      !cp.native_->enabled()) {
    ++c.fallback_mode;
    D3dFeedLastPacket() = 0x3FC; D3dFeedForeign(base);
    return false;
  }
  cp.repeat_benign_ = true;  // (the record's shader loads / trailing flush registers are not state a template cares about)
  const size_t saved = reader->read_offset();
  const uintptr_t item_ptr = uintptr_t(reader->read_ptr());  // (crowd2 review B3: this item's address)
  const uint64_t crowd_writes_before_item = cp.SceneCrowdRegisterWrites();
  const uint64_t crowd_restored_before_item = D3dFeedCrowd().restored_image + D3dFeedCrowd().restored_regfile;  // (Session 90, K2)
  const uint32_t viz_query_condition = reader->ReadAndSwap<uint32_t>();
  reg::VGT_DRAW_INITIATOR vgt_draw_initiator;
  vgt_draw_initiator.value = reader->ReadAndSwap<uint32_t>();
  const uint32_t vgt_dma_base = reader->ReadAndSwap<uint32_t>();
  reg::VGT_DMA_SIZE vgt_dma_size;
  vgt_dma_size.value = reader->ReadAndSwap<uint32_t>();
  if (vgt_draw_initiator.source_select != xenos::SourceSelect::kDMA) {
    reader->set_read_offset(saved);
    ++c.fallback_source;
    D3dFeedLastPacket() = 0x3FC; D3dFeedForeign(base);
    return false;
  }
  // (Session 87, stage 3) The record-fed variant: the register image feeds this draw (BeginDraw), the register file is
  // not written for it. Verify mode keeps the stock draw and compares the image after it.
  const D3dFeedItem feed_item{viz_query_condition, vgt_draw_initiator.value, vgt_dma_base, vgt_dma_size.value};
  const Feed::Mode feed_mode = Feed::ModeNow(cp);
  bool own = feed_mode == Feed::Mode::kOwn;
  D3dFeedCensus& fc = Feed::Census();
  // The register file brought up to date for a draw the record-fed path does not take.
  const auto flush_for_draw = [&]() {
    if (D3dFeedState() & kD3dFeedPending) {
      if (feed_mode == Feed::Mode::kOff) ++fc.fb_mode;
      ++fc.flush_by_draw;
      D3dFeedFlush(base);
    }
  };
  // Is this draw a crowd run's first member (its c76-c78 match a queued run)? The crowd code then reads the register file
  // before the draw and writes c76-c78 into it after: the stock way, and the image is stale afterwards.
  bool crowd_hit = false;
  if (feed_mode != Feed::Mode::kOff) {
    Feed::BeginDraw(cp, feed_item, feed_mode);
    crowd_hit = Feed::CrowdMatch(cp);
    ++fc.draws_seen;
    fc.crowd_hits += crowd_hit ? 1 : 0;
    if (own && crowd_hit) {
      if (REXCVAR_GET(gpu_d3d_own_crowd)) {
        // gpu_d3d_own_crowd: the crowd code runs on the image (its reads: the register source; its per-member c76-c78 writes:
        // the image, pending for the register file), so the first member's draw is described from the record like any other.
        ++fc.draws_crowd_image;
      } else {
        ++fc.fb_crowd;
        Feed::EndDraw(cp);
        own = false;
      }
    }
  }
  const bool crowd_on_image = own && crowd_hit;  // (the crowd code's writes go to the image: it stays valid)
  bool scope_ended = false;                        // the draw was handed back below: the register file is the source after all
  if (!own) flush_for_draw();
  // (From here on the item is ours: the stock wrapper's own steps for this draw, in its order.)
  if (!own) {
    cp.WriteRegister(XE_GPU_REG_VGT_DRAW_INITIATOR, vgt_draw_initiator.value);
    cp.WriteRegister(XE_GPU_REG_VGT_DMA_BASE, vgt_dma_base);
    cp.WriteRegister(XE_GPU_REG_VGT_DMA_SIZE, vgt_dma_size.value);
  }
  const uint32_t index_size_bytes =
      vgt_draw_initiator.index_size == xenos::IndexFormat::kInt16 ? sizeof(uint16_t) : sizeof(uint32_t);
  VCP::IndexBufferInfo index_buffer_info;
  index_buffer_info.guest_base = vgt_dma_base & ~(index_size_bytes - 1);
  index_buffer_info.endianness = vgt_dma_size.swap_mode;
  index_buffer_info.format = vgt_draw_initiator.index_size;
  index_buffer_info.length = vgt_dma_size.num_words * index_size_bytes;
  index_buffer_info.count = vgt_draw_initiator.num_indices;

  cp.half2_matched_kind_ = 0;
  if (cp.half2_) {
    cp.Half2OnCpDraw(cp.half2_packet_at_, false,
                     {uint32_t(vgt_draw_initiator.prim_type), uint32_t(vgt_draw_initiator.num_indices),
                      index_buffer_info.guest_base & 0x1FFFFFFF,
                      vgt_draw_initiator.index_size == xenos::IndexFormat::kInt32});
  }

  const RegisterFile& regs = cp.DrawRegs();  // (the record's image while the record-fed variant runs; else the register file)
  const auto viz_query = regs.Get<reg::PA_SC_VIZ_QUERY>();
  ++(cp.native_ib_depth_ ? cp.native_census_.draws_in_ib : cp.native_census_.draws_in_ring);
  if (viz_query.viz_query_ena) {
    ++cp.native_census_.viz_query_draws;
    cp.native_census_.viz_query_dropped += viz_query.kill_pix_post_hi_z ? 1 : 0;
  }
  if (!(viz_query.viz_query_ena && viz_query.kill_pix_post_hi_z)) {
    const bool major_mode_explicit =
        xenos::IsMajorModeExplicit(vgt_draw_initiator.major_mode, vgt_draw_initiator.prim_type);
    ++cp.frame_counters_.issue_draw;
    // (Session 92, M4b: gpu_d3d_owned_front bit 1 - the typed state of this draw item, for the shadow draw description at the claim.)
    const bool owned_front_draw = (REXCVAR_GET(gpu_d3d_owned_front) & 2u) != 0;
    if (owned_front_draw) OwnedRecordDrawBegin(viz_query_condition, vgt_draw_initiator.value, vgt_dma_base, vgt_dma_size.value);
    // (Session 80, own crowd: a crowd run's first member left to our own crowd renderer.)
    // (A record-fed draw is no crowd run's first member - CrowdMatch above - and PreTake reads the lagging register file.)
    // (Crowd2 review B3: this item's own address for the crowd code's predicated-tiling stand-down check, not the token's.)
    // (Session 92, M4d: the crowd functions take the member constants and the packet id from the caller; the old path hands in its register
    // image's c76-c78 and the item's arena address - the same values they read themselves before.)
    const uint32_t crowd_old_id = uint32_t(int64_t(item_ptr - uintptr_t(cp.memory_->TranslatePhysical(0))) + cp.half2_exec_bias_) | 1u;
    cp.SceneCrowdItemAt(crowd_old_id);
    const bool crowd_pre_called = (!own || crowd_on_image) && (!cp.scene_crowd_local_.empty() || cp.scene_crowd_queued_.load(std::memory_order_relaxed));
    const bool crowd_pre = crowd_pre_called && cp.SceneCrowdPreTake(cp.SceneCrowdC76());
    cp.own_crowd_skip_draw_ = crowd_pre;
    // (Session 92, M4d: gpu_d3d_owned_front bit 3 - the same inputs from the typed state, compared; PreTake is asked a second time with the typed
    // constants).)
    if ((REXCVAR_GET(gpu_d3d_owned_front) & 8u) != 0) OwnedFrontCrowdItem(cp, cp.SceneCrowdC76(), crowd_old_id, crowd_pre_called, crowd_pre, [](VCP& c, const uint32_t* c76) { return c.SceneCrowdMatchLocal(c76); });
    // (Session 87, crowd2: the census, and with gpu_native_own_crowd_all a lone draw of the three instanced crowd shaders
    // - a fed one too, its per-member c76-c78 from the record's image - as a run of one for our crowd program.)
    if (cp.SceneCrowdCensusBegin(cp.own_crowd_skip_draw_)) cp.own_crowd_skip_draw_ = true;
    bool draw_succeeded = true;
    const xenos::PrimitiveType prim_type = vgt_draw_initiator.prim_type;
    const uint32_t index_count = vgt_draw_initiator.num_indices;
    // (Session 92, M4f-A: gpu_d3d_owned_front bit 5 with the decoder's snapshots and banks (bits 1, 2): the typed state draws this item; any draw it
    // cannot take - counted by reason - goes the old way below, nothing of it done twice.)
    bool typed_taken = false;
    const OwnedRecordState* typed_snapshot = nullptr;
    if (owned_front_draw && (REXCVAR_GET(gpu_d3d_owned_front) & 36u) == 36u) {
      // (the same gate as the after-draw crowd code below: with the record-fed variant and no crowd image the crowd queues are not this draw's business)
      // (M4f-B, latch bit 6: the crowd code's calls take the typed way - the first member's draw, the member loop - so a crowd run's draws are typed too)
      const bool crowd_blocked = !(REXCVAR_GET(gpu_d3d_owned_front) & 64u) &&
                                 (crowd_pre_called || cp.own_crowd_skip_draw_ ||
                                  ((!own || crowd_on_image) && (cp.scene_crowd_verify_active_ || !cp.scene_crowd_local_.empty() ||
                                                               cp.scene_crowd_queued_.load(std::memory_order_relaxed))));
      typed_snapshot = OwnedRecordDrawNow();
      typed_taken = TypedDraw(cp, nullptr, prim_type, index_count, &index_buffer_info, major_mode_explicit, draw_succeeded, crowd_blocked);
    }
    const Result result = typed_taken ? Result::kTaken
                                      : Lean(cp, prim_type, index_count, &index_buffer_info, major_mode_explicit, draw_succeeded, own, feed_item);
    if (typed_taken) {
      // (counted by the typed front end: [d3d][front] M4f-A)
    } else if (result == Result::kTaken) {
      ++c.taken;
      if (own) ++fc.draws_direct;
    } else {
      if (own) {
        // gpu_d3d_own_full: the full derivation reads the record's image too (the scope stays), unless the image says this
        // draw is a copy (IssueCopy reads the register file) - then the draw is handed back by name.
        const bool copy_mode = cp.DrawRegs().Get<reg::RB_MODECONTROL>().edram_mode == xenos::EdramMode::kCopy;
        if (REXCVAR_GET(gpu_d3d_own_full) && !copy_mode) {
          ++fc.draws_full_image;
          (result == Result::kPass ? fc.full_image_pass : fc.full_image_template)++;
        } else {
          (result == Result::kPass ? fc.fb_pass : fc.fb_template)++;
          if (copy_mode) ++fc.fb_copy_mode;
          scope_ended = true;  // (the crowd code, if any, runs on the register file below: the image is stale after it)
          // Handed back to F4's path: the register file first (every pending run and the draw's words), then the stock
          // derivations read it.
          Feed::EndDraw(cp);
          flush_for_draw();
        }
      }
      bool lean_ok = false;
      if (result == Result::kPass) {
        ++c.to_stock_lean;
        if (cp.IssueDrawLean(prim_type, index_count, &index_buffer_info, major_mode_explicit, 1u, lean_ok)) {
          draw_succeeded = lean_ok;
        } else {
          draw_succeeded = cp.IssueDraw(prim_type, index_count, &index_buffer_info, major_mode_explicit);
        }
      } else {
        ++c.to_full;
        draw_succeeded = cp.IssueDraw(prim_type, index_count, &index_buffer_info, major_mode_explicit);
      }
    }
    if (owned_front_draw) OwnedRecordDrawEnd();  // (the crowd code's own draws below are not this item's)
    if (((!own || crowd_on_image) && (cp.scene_crowd_verify_active_ || !cp.scene_crowd_local_.empty() ||
                 cp.scene_crowd_queued_.load(std::memory_order_relaxed))) || cp.own_crowd_skip_draw_) {
      if (typed_taken && typed_snapshot && (REXCVAR_GET(gpu_d3d_owned_front) & 64u)) {
        // (M4f-B) The crowd code on the typed state: c76-c78 and the packet id are the typed item's; its first-member draw and its member loop call
        // back into the typed draw (CrowdMember); the own-crowd functions read the typed arrays (OwnedCrowdSource). The register file and the image
        // are not written.
        CrowdHook hook;
        hook.cp = &cp;
        hook.prim_type = prim_type;
        hook.index_count = index_count;
        hook.index_buffer_info = &index_buffer_info;
        hook.major_mode_explicit = major_mode_explicit;
        std::memcpy(hook.vs_copy, typed_snapshot->vs_float, sizeof(hook.vs_copy));
        hook.source.vs_float = hook.vs_copy;
        hook.source.ps_float = typed_snapshot->ps_float;
        hook.source.fetch = typed_snapshot->fetch;
        hook.source.color_info = typed_snapshot->color_info[0].value;
        // (review 3: the packet id keeps its old meaning - the item's arena address; the typed crowd_id folded lost the low bit that separates items 2k / 2k+1)
        uint32_t c76_copy[12];
        std::memcpy(c76_copy, typed_snapshot->vs_float + 4 * 76, sizeof(c76_copy));
        SceneCrowdSetMemberDraw(&D3dDirectAccess::CrowdMember, &hook);
        OwnedCrowdSourceSet(&hook.source);
        cp.SceneCrowdAfterDraw(prim_type, index_count, &index_buffer_info, major_mode_explicit, 1u, c76_copy, crowd_old_id);
        SceneCrowdSetMemberDraw(nullptr, nullptr);
        OwnedCrowdSourceSet(nullptr);
        OwnedFrontTypedNote(kTypedCrowdMembers, hook.drawn);
        OwnedFrontTypedNote(kTypedCrowdDeclined, hook.declined);
        // The uploaded vertex float buffer holds the last member's c76-c78: the next draw uploads from its own state.
        if (hook.drawn) cp.current_constant_buffers_up_to_date_ &= ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatVertex);
        // (review 4) A member's draw is a merge base inside the run only: no other draw continues from it.
        cp.lean_.valid = false;
        cp.repeat_prev_.valid = false;
        cp.native_repeat_draw_valid_ = false;
      } else {
        cp.SceneCrowdAfterDraw(prim_type, index_count, &index_buffer_info, major_mode_explicit, 1u, cp.SceneCrowdC76(), crowd_old_id);
      }
    }
    cp.SceneCrowdCensusEnd();
    cp.SceneCrowdItemAt(0);
    if (!draw_succeeded) {
      const RegisterFile& state = cp.DrawRegs();
      REXGPU_ERROR(
          "D3D token draw({}, {}): Failed in backend (major_mode={}, explicit_major={}, path_select={}, tess_mode={}, "
          "edram_mode={})",
          uint32_t(vgt_draw_initiator.num_indices), uint32_t(vgt_draw_initiator.prim_type),
          uint32_t(vgt_draw_initiator.major_mode), uint32_t(major_mode_explicit),
          uint32_t(state.Get<reg::VGT_OUTPUT_PATH_CNTL>().path_select), uint32_t(state.Get<reg::VGT_HOS_CNTL>().tess_mode),
          uint32_t(state.Get<reg::RB_MODECONTROL>().edram_mode));
    }
  }
  if (own) Feed::EndDraw(cp);
  // (Crowd2 review B3: also when SceneCrowdAfterDraw wrote c76-c78 itself - our draw declined and the game drew the run.)
  const bool crowd_wrote = own && cp.SceneCrowdRegisterWrites() != crowd_writes_before_item;
  // (Session 90, K2: SceneCrowdAfterDraw puts c76-c78 back after a run it drew itself, so the register file and the image are what the
  // game believes again: the catch-up is skipped for an item whose crowd writes were all restored, kept (and counted by clause) for
  // any write that was not.)
  const bool crowd_restored = D3dFeedCrowd().restored_image + D3dFeedCrowd().restored_regfile != crowd_restored_before_item;
  const bool crowd_clause = crowd_hit && (!crowd_on_image || scope_ended);
  if (crowd_restored && !crowd_wrote && (crowd_clause || crowd_on_image)) {
    ++D3dFeedCrowd().skipped_restored;
    if (feed_mode == Feed::Mode::kVerify) Feed::VerifyDraw(cp);
  } else if (crowd_clause || crowd_wrote) {
    D3dFeedCrowdState& cs = D3dFeedCrowd();
    if (crowd_wrote) ++cs.catchup_unrestored;
    else if (scope_ended) ++cs.catchup_scope_ended;
    else ++cs.catchup_crowd_hit;
    // The crowd code (batched members, SceneCrowdAfterDraw) writes vertex constants c76-c78 into the register file
    // directly, not by a packet: the image is stale after such a draw (verify run: 39 of 299,008 draws differed, all in
    // 0x4130-0x413B).
    if (crowd_wrote && (REXCVAR_GET(gpu_d3d_owned_front) & 32u)) {
      // (M4f-A review P6) The crowd code uploaded its own c76-c78 and did not put them back: the typed banks (the game's record) and what the ring holds
      // differ, so the vertex float buffer is written again by the next draw (typed or old).
      cp.current_constant_buffers_up_to_date_ &= ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatVertex);
    }
    D3dFeedLastPacket() = 0x3FD;  // (Session 90: after a crowd draw - the census key; before, the flush was credited to the last item's kind)
    Feed::Foreign(cp);
  } else if (feed_mode == Feed::Mode::kVerify) {
    Feed::VerifyDraw(cp);
  }
  if (c.items >= c.next_print) {
    c.next_print = c.items + kPrintEvery;
    Print(cp);
    Feed::Print(cp);
  }
  return true;
}
// ---------------------------------------------------------------------------------------------------------------------
// The lean variant. Structure and every check of VulkanCommandProcessor::IssueDrawLean for a draw it takes, with the
// changes listed at the top of this file; the diagnostics of the stock one (draw census, dirty census, verify) make this
// path stand down (kPass: the stock path owns them).
D3dDirectAccess::Result D3dDirectAccess::Lean(VCP& cp, xenos::PrimitiveType prim_type, uint32_t index_count,
                                              VCP::IndexBufferInfo* index_buffer_info, bool major_mode_explicit,
                                              bool& succeeded, bool own, const D3dFeedItem& item) {
  using R = Result;
  DirectCensus& c = g_c;
  if (!REXCVAR_GET(gpu_native_lean_draws) || REXCVAR_GET(gpu_native_lean_draws_verify) ||
      REXCVAR_GET(gpu_native_lean_dirty_census) || REXCVAR_GET(gpu_debug_draw_census_count) > 0) {
    return R::kPass;
  }
  VCP::LeanStats& stats = cp.lean_stats_;
  const uint32_t dirty = cp.repeat_dirty_;
  // The special draws the stock path sends to the full path by hash / register (the shadow caster and chain draws, the
  // crowd density, the shadow half rate): the stock IssueDrawLean decides these.
  if (wwe_shadow_chain::g.off || REXCVAR_GET(gpu_scene_crowd_density) < 100 || REXCVAR_GET(gpu_shadow_half_rate)) {
    return R::kPass;
  }
  // State outside the compared block changed - earlier bases no longer hold.
  const bool outside_written = (dirty & (VCP::kRepeatDirtyPacket | VCP::kRepeatDirtyOtherOutside)) != 0;
  if (outside_written) ++cp.lean_outside_epoch_;
  const bool multi_base = REXCVAR_GET(gpu_native_lean_multi_base);
  bool restored = false;
  cp.lean_restore_cross_ = false;
  cp.lean_verify_cache_ = false;  // (a previous cross-frame verify whose full draw captured no base: not this draw's)
  cp.repeat_census_regs_.clear();
  const auto full = [&](VCP::LeanReason reason) {
    ++stats.reasons[reason];
    ++c.reasons[reason];
    cp.repeat_dirty_ = dirty;
    return R::kFull;
  };
  if (!cp.native_ || !cp.native_->truth() || !REXCVAR_GET(gpu_native_lean) ||
      (cp.render_target_cache_->GetPath() != RenderTargetCache::Path::kPixelShaderInterlock &&
       !REXCVAR_GET(gpu_native_truth_host_rt)) ||
      cp.anatomy_active_ || cp.gpu_marks_active_ || cp.recstats_frames_ > 0 || cp.skip_frame_draws_ || cp.rename_.active ||
      cp.hide_hud_after_pass_ || !cp.chain_probe_shaders_.empty() || cp.debug_skip_vertex_shader_hash_ ||
      cp.DrawProbeActive() || cp.native_->CoverStatsActive() || !cp.lean_mode_ok_) {
    return R::kPass;  // (the stock path logs which mode)
  }
  // A restore attempt, counted (why it did not happen: switch off, outside write, no base with this state / these shaders).
  const auto try_restore = [&]() {
    if (!multi_base) {
      ++c.restore_off;
      return false;
    }
    if (outside_written) {
      ++c.restore_outside;
      return false;
    }
    if (cp.LeanRestoreBase(uint32_t(prim_type), major_mode_explicit, /*allow_cross_frame=*/true)) return true;
    ++c.restore_miss;
    uint32_t held = 0;
    for (const auto& e : cp.lean_bases_) held += (e.frame == cp.frame_current_ && e.draw) ? 1 : 0;
    c.restore_ring_full += held == cp.lean_bases_.size() ? 1 : 0;
    return false;
  };
  if (!cp.lean_.valid) {
    if (!try_restore()) {
      return full(VCP::kLeanNoBase);
    }
    restored = true;
    ++c.restored;
  }
  // Only constants since the base: its render state holds - or the render state was re-sent with the base's values.
  if (!restored && (dirty & (VCP::kRepeatDirtyOther | VCP::kRepeatDirtyPacket))) {
    const uint32_t* now = cp.DrawRegs().values;  // (the record's image in the record-fed variant)
    constexpr uint32_t kDma = 0x21FA - 0x2000, kAfter = 0x21FD - 0x2000;  // VGT_DMA_BASE .. VGT_DRAW_INITIATOR
    if ((dirty & (VCP::kRepeatDirtyPacket | VCP::kRepeatDirtyOtherOutside)) ||
        std::memcmp(now + 0x2000, cp.lean_regs_.data(), kDma * sizeof(uint32_t)) != 0 ||
        std::memcmp(now + 0x2000 + kAfter, cp.lean_regs_.data() + kAfter, (0x400 - kAfter) * sizeof(uint32_t)) != 0) {
      if (!try_restore()) {
        // (Census, log only: which render-state registers differ from the base's - what the restore key is made of.)
        ++c.dirty_misses;
        if (dirty & (VCP::kRepeatDirtyPacket | VCP::kRepeatDirtyOtherOutside)) ++c.dirty_outside;
        for (uint32_t r = 0; r < 0x400; ++r) {
          if (r >= kDma && r < kAfter) continue;
          if (now[0x2000 + r] != cp.lean_regs_[r]) ++c.diff_regs[r];
        }
        return full(VCP::kLeanDirty);
      }
      restored = true;
      ++c.restored;
    } else {
      ++stats.same_values;
      ++c.same_values;
    }
  }
  auto vertex_shader = static_cast<VulkanShader*>(cp.active_vertex_shader());
  if (!vertex_shader || vertex_shader != cp.lean_.active_vertex_shader ||
      cp.active_pixel_shader() != cp.lean_.active_pixel_shader || uint32_t(prim_type) != cp.lean_.prim_type ||
      major_mode_explicit != cp.lean_.major_mode_explicit) {
    // (Session 87 stage 3) Other shaders / primitive with the same render state: an earlier base of this frame may hold
    // exactly this pair (LeanRestoreBase compares the state, the shaders, the primitive and the mode).
    if (restored || !vertex_shader || !try_restore()) {
      return full(VCP::kLeanShader);
    }
    restored = true;
    ++c.restored;
    ++c.restored_for_shaders;
  }
  if ((cp.pipeline_cache_->NativeSpec().endian || cp.lean_.vertex_endian_key) &&
      cp.pipeline_cache_->GetVertexFetchEndianKey(*vertex_shader) != cp.lean_.vertex_endian_key) {
    return full(VCP::kLeanShader);
  }
  if (cp.pipeline_cache_->PixelSpecKeyChanged(cp.lean_.pixel_shader, cp.lean_.pixel_spec_key) ||
      cp.pipeline_cache_->VertexMulKeyChanged(cp.lean_.vertex_mul_key) ||
      cp.pipeline_cache_->NativeExtStale(cp.lean_.pixel_ext, cp.lean_.vertex_ext, cp.lean_.pixel_shader, vertex_shader)) {
    return full(VCP::kLeanShader);
  }
  // (Session 87, stage 3, gpu_native_lean_state_cache) The base is an earlier frame's: its frame-independent description
  // stands, its claim does not. Verify mode hands the draw to the full path and compares what that derives with it.
  const bool cross_frame = restored && cp.lean_restore_cross_;
  if (cross_frame) {
    ++c.cross_restores;
    if (REXCVAR_GET(gpu_native_lean_state_cache_verify)) {
      cp.lean_verify_snapshot_ = cp.lean_;
      if (!cp.native_lean_verify_draw_) cp.native_lean_verify_draw_ = std::make_unique<native::Draw>();
      *cp.native_lean_verify_draw_ = *cp.native_repeat_draw_;
      cp.lean_verify_pending_ = true;
      cp.lean_verify_cache_ = true;
      ++c.cache_verify_handbacks;
      return full(VCP::kLeanVerify);
    }
  }
  // The page-validity epoch of this draw: read before anything of it is requested (a later invalidation ends a stamp).
  const uint64_t repeat_epoch = cp.shared_memory_->repeat_epoch();
  // A repeat (only vertex float constants, the same draw parameters): the repeat path, as IssueDraw takes it.
  if (REXCVAR_GET(vulkan_repeat_draw_fast) && cp.repeat_prev_.valid &&
      !(dirty & ~uint32_t(VCP::kRepeatDirtyVsFloat)) &&
      cp.RepeatIsSameDraw(vertex_shader, uint32_t(prim_type), index_count, index_buffer_info, major_mode_explicit,
                          repeat_epoch)) {
    cp.repeat_dirty_ = 0;
    ++cp.repeat_fast_stats_.candidates;
    cp.repeat_refusal_ = 0;
    if (cp.RepeatDrawFast(vertex_shader, repeat_epoch)) {
      if (cp.vertex_census_on_) [[unlikely]] {
        cp.VertexCensusDraw(*vertex_shader, cp.repeat_prev_.spec_pixel_shader, VCP::kVertexCensusRepeat,
                            cp.vertex_census_last_type_, 1);
      }
      ++stats.reasons[VCP::kLeanRepeat];
      ++c.repeats;
      succeeded = true;
      return R::kTaken;
    }
    cp.repeat_prev_.valid = false;
    ++c.repeat_refused[std::min<uint32_t>(cp.repeat_refusal_, 23)];
    return full(VCP::kLeanRepeat);
  }
  if (!cp.native_repeat_draw_valid_ || !cp.native_repeat_draw_ || !cp.native_repeat_draw_->truth ||
      !cp.native_repeat_draw_->owned || cp.native_repeat_draw_->pipeline == VK_NULL_HANDLE) {
    return full(VCP::kLeanNoBase);
  }
  // (Session 92, M4f-A) From here the body is DrawFromBase: one body for the old lean base (this path: cp.lean_ and the base draw, the register image's
  // constants / fetch words) and the typed memo path (OwnedFrontDrawItem: the typed state's).
  DrawSources src;
  src.own = own;
  src.dirty = dirty;
  PrimitiveProcessor::DrawInput process_input;
  if (own) {
    // The record-fed variant: the draw's words from the DRAW item, the four render state registers from the image.
    process_input = PrimitiveProcessor::DrawInput::FromRegisters(cp.DrawRegs());
    process_input.draw_initiator.value = item.initiator;
    process_input.dma_size.value = item.dma_size;
    process_input.dma_base = item.dma_base;
    src.input = &process_input;
  }
  {
    const RegisterFile& rf = cp.DrawRegs();  // (the record's image while the record-fed variant runs; else the register file)
    src.rb_surface_info = rf.Get<reg::RB_SURFACE_INFO>().value;
    src.rb_color_info = rf[XE_GPU_REG_RB_COLOR_INFO];
    src.rb_depth_info = rf[XE_GPU_REG_RB_DEPTH_INFO];
  }
  return DrawFromBase(cp, cp.lean_, *cp.native_repeat_draw_, vertex_shader, prim_type, index_count, index_buffer_info,
                      major_mode_explicit, succeeded, src, restored, cross_frame, repeat_epoch);
}


// The same stand-downs as the token draw item (modes with no native renderer, the extents shadow, the det skip, queued crowd runs) and one of its own:
// a 3-vertex rectangle list may be a fast clear (FastClearDetect reads the registers), left to the old path and counted.
bool D3dDirectAccess::InlineDraw(VCP& cp, uint32_t viz_query_condition, uint32_t initiator_value, bool& drawn) {
  (void)viz_query_condition;
  const auto fallback = [&](const char* why) {
    OwnedFrontTypedNote(kTypedItems);  // (offered = drawn + failed + handed to the old path: an early refusal is an offered item too)
    OwnedFrontTypedNote(kTypedInlineFallback);
    OwnedFrontTypedFallback(why);
    return false;
  };
  const OwnedRecordState* s = OwnedRecordDrawNow();
  if (!s || !s->inline_draw) return fallback("inline: no paired snapshot");
  if (cp.anatomy_active_ || cp.D3dDirectExtentsActive() || REXCVAR_GET(gpu_det_skip_host_draws) || !cp.native_ || !cp.native_->enabled()) {
    return fallback("inline: special mode (no native renderer / extents / det skip / anatomy)");
  }
  // The old draw asks PreTake with the draw's c76-c78 whenever runs are queued: an inline draw is no crowd run's first member, so it is asked the same way (the
  // queue moves to the local list as PreTake does) and only a verify run in progress or a match hands it to the old path.
  if (cp.scene_crowd_verify_active_) return fallback("inline: crowd verify in progress");
  if ((!cp.scene_crowd_local_.empty() || cp.scene_crowd_queued_.load(std::memory_order_relaxed)) && s->banks_valid &&
      cp.SceneCrowdPreTake(s->vs_float + 4 * 76)) {
    return fallback("inline: a crowd run's first member");
  }
  reg::VGT_DRAW_INITIATOR init;
  init.value = initiator_value;
  if (init.prim_type == xenos::PrimitiveType::kRectangleList && init.num_indices == 3) return fallback("inline: 3-vertex rectangle list (possible fast clear)");
  const bool major_mode_explicit = xenos::IsMajorModeExplicit(init.major_mode, init.prim_type);
  // (review 2) The viz-kill test first: the old path counts and drops such a draw. The census / half2 / issue_draw steps are done once, by whoever draws.
  const auto census = [&]() {
    cp.half2_matched_kind_ = 0;
    if (cp.half2_) cp.Half2OnCpDraw(cp.half2_packet_at_, false, {uint32_t(init.prim_type), uint32_t(init.num_indices), 0, false});
    ++(cp.native_ib_depth_ ? cp.native_census_.draws_in_ib : cp.native_census_.draws_in_ring);
    if (s->viz_query_cntl.viz_query_ena) {
      ++cp.native_census_.viz_query_draws;
      cp.native_census_.viz_query_dropped += s->viz_query_cntl.kill_pix_post_hi_z ? 1 : 0;
    }
  };
  if (s->viz_query_cntl.viz_query_ena && s->viz_query_cntl.kill_pix_post_hi_z) {
    census();
    drawn = true;  // (the old path drops it the same way)
    OwnedFrontTypedNote(kTypedItems);
    OwnedFrontTypedNote(kTypedInline);
    OwnedFrontTypedNote(kTypedDrawn);
    return true;
  }
  cp.own_crowd_skip_draw_ = false;
  VCP::IndexBufferInfo index_buffer_info;  // (auto index: no buffer)
  bool succeeded = true;
  if (!TypedDraw(cp, s, init.prim_type, init.num_indices, &index_buffer_info, major_mode_explicit, succeeded, false)) {
    OwnedFrontTypedNote(kTypedInlineFallback);
    return false;
  }
  census();
  ++cp.frame_counters_.issue_draw;
  OwnedFrontTypedNote(kTypedInline);
  if (!succeeded) {
    // (review 1) A draw that fails in the backend is logged as ExecutePacketType3Draw logs it and does not end the buffer: the item is consumed.
    REXGPU_ERROR("D3D_INLINE_DRAW({}, {}, {}): Failed in backend (typed draw)", uint32_t(init.num_indices), uint32_t(init.prim_type),
                 uint32_t(init.source_select));
  }
  drawn = true;
  return true;
}

bool OwnedFrontInlineDraw(CommandProcessor& base, uint32_t viz_query, uint32_t initiator, bool& drawn) {
  return D3dDirectAccess::InlineDraw(static_cast<VulkanCommandProcessor&>(base), viz_query, initiator, drawn);
}

}  // namespace rex::graphics::vulkan
