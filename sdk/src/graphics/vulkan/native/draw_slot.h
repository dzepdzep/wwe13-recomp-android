// wwe2k14 patch (Session 95, gpu_d3d_draw_slot): the draw slot. The record (draw_record.h) answers "what did a draw with these inputs derive", but asking costs a key: the used fetch words gathered and
// hashed, a table walk, a validity check, a 1.5 KB copy. The draw list repeats frame to frame, so a cursor walks last frame's order (DrawRec::next_hint) and the next draw is first compared with the slot of the
// record the cursor expects: the words themselves, no key. Equal = a slot hit: the record's preparation is used where it lies, and the sampler list, the plan's pool answer and the system constants come
// from what the slot's side (DrawSlotAux) keeps. Different = today's DrawRecFind, and the cursor moves to the record found: frame-to-frame drift in a moving shot costs one lookup, not the frame's walk.
// A hit never skips the float constants, the vertex range checks, the texture requests, the owned pass step or the cull: all of those run as for any draw. Command thread only: a job that runs draws would own
// one DrawSlots of its own, with its own cursor.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <tuple>

#include <rex/graphics/vulkan/texture_cache.h>

#include "draw_record.h"
#include "index_pool.h"

namespace rex::graphics::vulkan {

// The fetch slots a shader pair uses, once per pair: a slot points at its pair's list, so a compare needs neither the shaders' masks nor a gather.
struct DrawSlotUsed {
  uint32_t tex_count = 0, vf_count = 0;
  uint8_t tex[32] = {}, vf[96] = {};
};

constexpr uint32_t kDrawSlotInlineWords = 14;  // (two textures and a stream; a longer list is in `spill`)

// One slot per record, in the record's order. `gen` moves with everything DrawRecPrep checks; `state` is the render state and the bool / loop words as the decoder's epochs memoise them.
struct alignas(64) DrawSlot {
  DrawRec* rec = nullptr;  // nullptr = nothing stored
  const DrawSlotUsed* used = nullptr;
  uint64_t gen = 0;
  uint64_t state = 0;
  uint32_t own[7] = {};  // the draw's own words: initiator, DMA base and size, index offset, min / max vertex index, the viz query
  uint16_t count = 0;         // the used fetch words
  uint16_t capacity = 0;      // of `spill`
  uint32_t* spill = nullptr;  // the words when they are more than the inline ones (owned)
  uint32_t words[kDrawSlotInlineWords] = {};
  const uint32_t* Words() const { return count > kDrawSlotInlineWords ? spill : words; }
};
static_assert(sizeof(DrawSlot) == 128, "a slot is two cache lines");

// What the hit takes from the slot's side, beside the record.
struct DrawSlotAux {
  uint64_t sys_hash = 0;  // of the record's system constants, the storage window's selector left out; 0 = none
  native::IndexPool::HeldIndex held;
  uint32_t sampler_count[2] = {};
  bool replacement[2] = {};  // the binding list the parameters were made from (shader_swap's held flag)
  VulkanTextureCache::SamplerParameters params[2][32];
};

enum DrawSlotMiss : uint32_t { kSlotHit, kSlotNoNext, kSlotNotStored, kSlotIncomplete, kSlotJitter, kSlotGeneration, kSlotState, kSlotOwn, kSlotFetch, kSlotMissCount };

struct DrawSlotStats {
  uint64_t asked = 0, hits = 0, resyncs = 0, misses = 0;
  uint64_t why[kSlotMissCount] = {};
  uint64_t stored = 0, refused = 0, plan_held = 0, plan_asked = 0, sys_same = 0, sys_copied = 0, sys_full = 0;
  uint64_t verify_compared = 0, verify_record = 0, verify_stale = 0, verify_prep = 0, verify_parts = 0, verify_samplers = 0, verify_plan = 0, verify_sys = 0;
  uint64_t Mismatches() const { return verify_record + verify_stale + verify_prep + verify_parts + verify_samplers + verify_plan + verify_sys; }
};

// The live words of the draw being asked.
struct DrawSlotLive {
  uint64_t gen = 0, state = 0;
  uint32_t own[7] = {};
  const uint32_t* fetch = nullptr;
};

struct DrawSlots {
  DrawRec* recs = nullptr;  // the record table: slot i belongs to record i
  uint32_t count = 0;
  DrawSlot* slots = nullptr;
  DrawSlotAux* aux = nullptr;
  DrawRec* cursor = nullptr;  // the record of the last draw that had one
  uint64_t last_frame = 0, last_print = 0;
  std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, std::unique_ptr<DrawSlotUsed>> used;
  DrawSlotStats stats;
};
extern DrawSlots g_draw_slots;
// The content hash of the live system constants block when it is known to hold a record's bytes, 0 = not known. Every other writer of the block clears it.
extern uint64_t g_draw_slot_sys_hash;

void DrawSlotsBind(DrawRec* recs, uint32_t count, int32_t mode);
inline bool DrawSlotsBound() { return g_draw_slots.slots != nullptr; }
inline DrawSlot& DrawSlotOf(const DrawRec* r) { return g_draw_slots.slots[r - g_draw_slots.recs]; }
inline DrawSlotAux& DrawSlotAuxOf(const DrawRec* r) { return g_draw_slots.aux[r - g_draw_slots.recs]; }
// A gap in the slots' use (the switch was off, nothing drew): what a slot holds may be older than its record. Forgets every slot and the cursor.
void DrawSlotsFrame(uint64_t frame);
// The compare: every word, no hash of the fetch words.
DrawSlotMiss DrawSlotCompare(const DrawSlot& slot, const DrawSlotLive& live);
// Writes the slot's words (the record's own generation word comes from the caller).
void DrawSlotWrite(DrawSlot& slot, DrawRec* r, const DrawSlotUsed* used, const DrawSlotLive& live);
const DrawSlotUsed* DrawSlotIntern(uint32_t tex_mask, const uint32_t* vf_mask, uint32_t tex_count, uint32_t vf_count, const uint8_t* tex, const uint8_t* vf);
// The draw's record is known (hit or found): last frame's order is this frame's from here on. `whole` = also fetch the whole next record (gpu_tex_bind_memo bit 8).
void DrawSlotAdvance(DrawRec* r, bool whole);
void DrawSlotTick(uint64_t frame);

// owned_front_draw.cpp (they read the typed draw's private state).
// The record the cursor expects, when the draw's words equal its slot's; nullptr = a miss, counted by reason. `settings` = the settings the sampler parameters read besides the words and the externals.
DrawRec* DrawSlotTry(VulkanCommandProcessor& cp, const OwnedRecordState& s, uint32_t settings);
// The record's preparation where it lies (nullptr: the index span's request failed, the full preparation runs).
OwnedTypedPrep* DrawSlotPrep(VulkanCommandProcessor& cp, const OwnedRecordState& s, DrawRec& r);
// The whole OwnedTypedPrep a full build of the system constants needs (a record holds only its head).
OwnedTypedPrep* DrawSlotFullPrep(const OwnedRecordState& s, DrawRec& r);
// DrawFromBase's system constants step on a hit: false = the full build runs.
bool DrawSlotSystemConstants(VulkanCommandProcessor& cp, void* ctx, const DrawRec& r);
// After a draw that was not a hit, its record complete: the slot of the record, from this draw's words (DrawRecFind ran for them).
void DrawSlotLearn(DrawRec& r, const OwnedRecordState& s, uint32_t settings);
// The slots, once the record table exists.
void DrawSlotEnsure(int32_t mode);
// gpu_d3d_draw_slot 2: what the full way says about a draw the slot matched. Before the draw: the record found, its validity, the live block if the hash says it needs no copy. After it: the preparation.
void DrawSlotVerifyFind(VulkanCommandProcessor& cp, DrawRec* slot_rec, DrawRec* found, bool found_complete);
void DrawSlotVerifyPrep(VulkanCommandProcessor& cp, const DrawRec& r, const OwnedTypedPrep& p);
// The content hash of a record's block.
uint64_t DrawSlotSysHash(const DrawRec& r);

}  // namespace rex::graphics::vulkan
