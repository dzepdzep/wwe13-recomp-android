// wwe2k14 patch (Session 92, M4a): the call points between the host-queue consumer (src\graphics\d3d_noring.cpp, base library) and
// the record decoder of the Vulkan backend (vulkan\native\owned_record.cpp). Declarations only: the base never names a backend
// type (the pattern of d3d_feed.h). New header: nothing the game exe includes changes.
//
// gpu_d3d_owned_front (a bit mask, default 0 = off, everything unchanged): bit 0 = the M4a SHADOW. `before` reads the item
// (host order, straight from the queue storage) into the decoder's typed state BEFORE the existing path runs; `after` compares
// that state with the feed image the existing path has just updated. The existing path still draws the frame.
#pragma once

#include <cstdint>

namespace rex::memory {
class Memory;
}

namespace rex::graphics {

class CommandProcessor;
struct D3dNoringShared;

struct OwnedRecordHooks {
  // One queue item: kind (D3dNoringKind), fields f[0..5], payload (host order words), payload_dwords. Command thread only.
  void (*before)(CommandProcessor& cp, rex::memory::Memory* memory, uint32_t kind, const uint32_t* f, const uint32_t* payload,
                 uint32_t payload_dwords) = nullptr;
  // The same item after the existing path ran it.
  void (*after)(CommandProcessor& cp, uint32_t kind) = nullptr;
  // (Session 92, M4f-C; gpu_d3d_owned_front bit 128, THE CUT, read at the queue's attach) Executes one queue item IN ORDER on the decoder's live typed
  // state: the 0x7C token (RUN -> typed store + the notifications, SHADER -> LoadShader once, DRAW -> the draw entry point), the 0x7D / 0x7E records
  // (their entry points) and the Registers item (register ops). Only kNrPackets / kNrRegs are looked at. True = handled (the consumer does not
  // call `before`, D3dNoringExecute or `after` for the item); false = the item is untouched and goes the existing way (counted by reason: an item
  // whose draw / clear / resolve / inline entry point is not wired, a packet record that is not one of those three, a framing error).
  // Under the cut `run` is FINAL for kNrPackets / kNrRegs: an item it cannot take is dropped and counted by reason, never given to the old path.
  bool (*run)(CommandProcessor& cp, rex::memory::Memory* memory, uint32_t kind, const uint32_t* f, const uint32_t* payload,
              uint32_t payload_dwords) = nullptr;
  // True when the executor's draw, clear, resolve and inline entry points are all wired: the cut latches only then (bit 128 does nothing before).
  bool (*entries_wired)() = nullptr;
  // gpu_d3d_cut_poison (proof): check = false fills the executor-owned register ranges of `values` (the register file) with a pattern; check = true
  // counts the registers of those ranges that no longer hold it (a writer survived), refills them and returns the count.
  uint64_t (*poison)(uint32_t* values, bool check) = nullptr;
  // Session 95 (item N, gpu_jobs; vulkan\native\owned_scout.cpp): the scout's call points, command thread. Each returns at its first line when gpu_jobs is 0.
  // enter / leave: around the drain loop (the queue to read ahead of the consumer); item: before each kNrPackets / kNrRegs item the cut runs, with the item's
  // start in dwords (the draws' key); present: at each kNrPresent item (the frame tick: switches, stats, the [jobs] line; the queue and memory let it switch on in the middle of a drain).
  void (*scout_enter)(D3dNoringShared* queue, rex::memory::Memory* memory) = nullptr;
  void (*scout_leave)() = nullptr;
  void (*scout_item)(uint32_t head) = nullptr;
  void (*scout_present)(D3dNoringShared* queue, rex::memory::Memory* memory) = nullptr;
  void (*scout_guest_write)() = nullptr;  // an item that can change guest memory under a job has run (a kNrInterrupt item)
  void (*inline_flush)(CommandProcessor& cp) = nullptr;  // (Session 95 r5) before an item `run` does not see: the open particle batch is drawn
};

inline OwnedRecordHooks& OwnedRecordHookSlots() {
  static OwnedRecordHooks hooks;
  return hooks;
}

}  // namespace rex::graphics
