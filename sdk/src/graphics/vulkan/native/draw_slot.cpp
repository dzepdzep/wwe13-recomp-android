// wwe2k14 patch (Session 95, gpu_d3d_draw_slot): see draw_slot.h.
#include "draw_slot.h"

#include <algorithm>
#include <cstring>

#include <rex/cvar.h>
#include <rex/logging.h>

REXCVAR_DEFINE_INT32(gpu_d3d_draw_slot, 0, "GPU/Native",
                     "Session 95: the draw slot (draw_slot.h), needs gpu_d3d_draw_record 1. 1 = a cursor walks last frame's draw order and a draw whose words equal the slot's skips the record's key and gather, the "
                     "1.5 KB copy, the sampler rebuild, the plan's pool lookup and the system constants' compare; a miss takes DrawRecFind and moves the cursor. 2 = verify: a matched draw goes the full way and is compared. "
                     "0 = off. Float constants, vertex range checks, texture requests, the owned pass step and the cull are never skipped");

namespace rex::graphics::vulkan {

DrawSlots g_draw_slots;
uint64_t g_draw_slot_sys_hash = 0;

void DrawSlotsBind(DrawRec* recs, uint32_t count, int32_t mode) {
  DrawSlots& g = g_draw_slots;
  g.recs = recs;
  g.count = count;
  g.slots = new DrawSlot[count];
  g.aux = new DrawSlotAux[count];
  REXGPU_INFO("[d3d][slot] on: gpu_d3d_draw_slot {} ({} slots of {} bytes, {} bytes beside each)", mode, count, sizeof(DrawSlot), sizeof(DrawSlotAux));
}

void DrawSlotsFrame(uint64_t frame) {
  DrawSlots& g = g_draw_slots;
  if (frame == g.last_frame) return;
  if (frame != g.last_frame + 1) {
    for (uint32_t i = 0; i < g.count; ++i) g.slots[i].rec = nullptr;
    g.cursor = nullptr;
  }
  g.last_frame = frame;
}

DrawSlotMiss DrawSlotCompare(const DrawSlot& slot, const DrawSlotLive& live) {
  if (slot.gen != live.gen) return kSlotGeneration;
  if (slot.state != live.state) return kSlotState;
  if (std::memcmp(slot.own, live.own, sizeof(slot.own)) != 0) return kSlotOwn;
  const DrawSlotUsed& u = *slot.used;
  const uint32_t* w = slot.Words();
  for (uint32_t i = 0; i < u.tex_count; ++i, w += 6) {
    if (std::memcmp(w, live.fetch + 6 * size_t(u.tex[i]), 24) != 0) return kSlotFetch;
  }
  for (uint32_t i = 0; i < u.vf_count; ++i, w += 2) {
    if (std::memcmp(w, live.fetch + 2 * size_t(u.vf[i]), 8) != 0) return kSlotFetch;
  }
  return kSlotHit;
}

void DrawSlotWrite(DrawSlot& slot, DrawRec* r, const DrawSlotUsed* used, const DrawSlotLive& live) {
  const uint32_t n = used->tex_count * 6 + used->vf_count * 2;
  if (n > kDrawSlotInlineWords && n > slot.capacity) {
    delete[] slot.spill;
    slot.spill = new uint32_t[n];
    slot.capacity = uint16_t(n);
  }
  slot.count = uint16_t(n);
  uint32_t* w = n > kDrawSlotInlineWords ? slot.spill : slot.words;
  for (uint32_t i = 0; i < used->tex_count; ++i, w += 6) std::memcpy(w, live.fetch + 6 * size_t(used->tex[i]), 24);
  for (uint32_t i = 0; i < used->vf_count; ++i, w += 2) std::memcpy(w, live.fetch + 2 * size_t(used->vf[i]), 8);
  slot.used = used;
  slot.gen = live.gen;
  slot.state = live.state;
  std::memcpy(slot.own, live.own, sizeof(slot.own));
  slot.rec = r;
  ++g_draw_slots.stats.stored;
}

const DrawSlotUsed* DrawSlotIntern(uint32_t tex_mask, const uint32_t* vf_mask, uint32_t tex_count, uint32_t vf_count, const uint8_t* tex, const uint8_t* vf) {
  std::unique_ptr<DrawSlotUsed>& e = g_draw_slots.used[std::make_tuple(tex_mask, vf_mask[0], vf_mask[1], vf_mask[2])];
  if (!e) {
    e = std::make_unique<DrawSlotUsed>();
    e->tex_count = tex_count;
    e->vf_count = vf_count;
    std::copy_n(tex, tex_count, e->tex);
    std::copy_n(vf, vf_count, e->vf);
  }
  return e.get();
}

void DrawSlotAdvance(DrawRec* r, bool whole) {
  DrawSlots& g = g_draw_slots;
  if (g.cursor && g.cursor->next_hint != r) g.cursor->next_hint = r;
  g.cursor = r;
  const DrawRec* const next = static_cast<const DrawRec*>(r->next_hint);
  if (!next) return;
  const DrawSlot* const slot = &DrawSlotOf(next);
  __builtin_prefetch(slot, 0, 2);
  __builtin_prefetch(reinterpret_cast<const char*>(slot) + 64, 0, 2);
  __builtin_prefetch(next, 0, 2);
  const DrawSlotAux* const aux = &DrawSlotAuxOf(next);
  __builtin_prefetch(aux, 0, 2);
  __builtin_prefetch(&aux->params[1][0], 0, 2);
  if (whole) {
    const char* const bytes = reinterpret_cast<const char*>(next);
    for (size_t at = 0; at < sizeof(DrawRec); at += 64) __builtin_prefetch(bytes + at, 0, 2);
  }
}

void DrawSlotTick(uint64_t frame) {
  DrawSlots& g = g_draw_slots;
  DrawSlotStats& t = g.stats;
  if (frame - g.last_print < 600) return;
  if (t.asked) {
    REXGPU_INFO("[d3d][slot] frames {}-{}: draws asked {}: cursor hits {} ({:.1f}%), resyncs {}, misses {} (no next {}, not stored {}, incomplete {}, jitter {}, generation {}, state {}, draw words {}, fetch words {}); "
                "stored {}, refused {}; plan held {}, asked the pool {}; system constants same {}, copied {}, full build {}; verify: compared {}, mismatches {} (record {}, stale {}, preparation {}, parts {}, samplers {}, plan {}, "
                "system constants {})",
                g.last_print, frame, t.asked, t.hits, 100.0 * double(t.hits) / double(t.asked), t.resyncs, t.misses, t.why[kSlotNoNext], t.why[kSlotNotStored], t.why[kSlotIncomplete], t.why[kSlotJitter], t.why[kSlotGeneration],
                t.why[kSlotState], t.why[kSlotOwn], t.why[kSlotFetch], t.stored, t.refused, t.plan_held, t.plan_asked, t.sys_same, t.sys_copied, t.sys_full, t.verify_compared, t.Mismatches(), t.verify_record, t.verify_stale,
                t.verify_prep, t.verify_parts, t.verify_samplers, t.verify_plan, t.verify_sys);
  }
  t = DrawSlotStats();
  g.last_print = frame;
}

}  // namespace rex::graphics::vulkan
