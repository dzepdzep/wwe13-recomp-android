// wwe2k14 patch (Session 87, D3D layer stage 3: our renderer fed in its own terms; docs\native\d3d\S3_CONTRACT.md in the
// game repository): the call points between the base command processor and the backend's record-fed state.
//
// With --gpu_d3d_own_input (needs --gpu_d3d_direct) the token's register runs are not written into the emulated register
// file one register at a time (WriteRegister: per-register side effects, the register-diff of RepeatNoteRegisters). The
// backend keeps the record's register image (native/d3d_feed.cpp): a run is stored into it in one copy and the block-level
// notifications the writes would have made (dirty classes, constant buffers out of date, fetch constants written) are made
// from the record's blocks. The taken draw's derivations (constants upload, system constants, primitive processing, the
// texture / sampler / pipeline-key fetch-constant reads) read the image; the register file is NOT written for it.
//
// The register file then lags the image. It is brought up to date (the dirty ranges copied once, no notifications) only
// when something that is not a taken draw is about to read or write it: any packet other than the token (the generic
// check in CommandProcessor::ExecutePacket), a draw the fed path hands back, a run it does not model. After a packet that
// is not the token the image is stale and is re-copied from the register file (the ranges the derivations read) before
// the next run is applied.
//
// Declarations only: the base command processor never names a backend type (the pattern of d3d_direct.h).
#pragma once

#include <cstdint>

namespace rex::memory {
class RingBuffer;
}

namespace rex::graphics {

class CommandProcessor;

struct D3dFeedHandlers {
  // A RUN item of the token: `n` values for registers first .. first + n - 1 follow in the reader. True = taken (the
  // values consumed, the register file not written); false = the reader is where it was and the stock type-0 path runs
  // (the handler has already brought the register file up to date and ended the image's validity).
  bool (*run)(CommandProcessor& cp, rex::memory::RingBuffer* reader, uint32_t first, uint32_t n) = nullptr;
  // Copies the image's pending ranges into the register file.
  void (*flush)(CommandProcessor& cp) = nullptr;
  // A packet other than the token (or a nop / indirect buffer) is about to run: flush, then the image is stale.
  void (*foreign)(CommandProcessor& cp) = nullptr;
  // gpu_d3d_own_crowd: inside a record-fed draw a register write (the crowd code writes c76-c78 for each member it replays) goes
  // into the image as a run would (notified, pending for the register file), not into the register file. False = not taken
  // (not a register the image models): the caller writes the register file as it did.
  bool (*scoped_write)(CommandProcessor& cp, uint32_t index, uint32_t value) = nullptr;
  // The image's registers for code that stores whole groups (AppendCrowdInstances), and the pending mark for what it stored.
  // gpu_d3d_own_copy: the copy-family token (clear / resolve), the inline draw token and the ring's type-0 register packets run
  // on the image. scope_wanted: the mode is on (the foreign check then leaves those packets alone). scope_begin opens the
  // image scope for an executor (image valid, the derivations' register sources pointed at it; false = not opened, the stock
  // way), scope_end closes it.
  bool (*scope_wanted)(CommandProcessor& cp) = nullptr;
  bool (*scope_begin)(CommandProcessor& cp) = nullptr;
  void (*scope_end)(CommandProcessor& cp) = nullptr;
  // gpu_d3d_own_copy (Session 90, contract R): a REG_RMW packet (the game's own read-modify-write of one register) applied on the image: reads
  // come from the image, the result goes into it (notified, pending for the register file). False = not taken (the mode is off or a
  // register involved is not one the image models): the caller brings the register file up to date and runs the stock way.
  bool (*reg_rmw)(CommandProcessor& cp, uint32_t rmw_info, uint32_t and_mask, uint32_t or_mask) = nullptr;
  uint32_t* (*image_values)() = nullptr;
  void (*image_dirty)(uint32_t first, uint32_t count) = nullptr;
};

// Session 90 (contract K2): the crowd code's state restore and its census, shared between the base command processor (which
// restores c76-c78 after a run it drew itself), the Vulkan OwnCrowdDraw (why it declined) and the direct path / feed census.
struct D3dFeedCrowdState {
  uint64_t restored_image = 0;      // fallback runs whose member c76-c78 writes were put back in the image
  uint64_t restored_regfile = 0;    // ... in the register file
  uint64_t unrestored_runs = 0;     // runs that wrote c76-c78 and did not put them back (should stay 0)
  uint64_t skipped_restored = 0;    // register-file catch-ups skipped because the item's crowd writes were restored
  uint64_t catchup_crowd_hit = 0;   // catch-ups made: crowd run's first member, image not the source
  uint64_t catchup_scope_ended = 0; // ... the draw was handed back to the register file
  uint64_t catchup_unrestored = 0;  // ... crowd writes that were not restored
  uint64_t declined_runs = 0;       // runs our crowd program declined (the game's draws)
  uint64_t decline_by_reason[16] = {};  // OwnCrowdDraw's refusal reason (1-8 as logged by the decline lines, 10-12 below)
  uint32_t last_decline = 0;        // the reason of the last OwnCrowdDraw refusal (set by it; 0 = none)
  uint32_t decline_logged = 0;
  // Session 92, M4d (shadow only): the 12 words of the member whose draw the crowd code is issuing right now (nullptr outside a member's draw).
  // The typed shadow reads it to build that draw's typed state; nothing in the old path reads it.
  const uint32_t* member_c76 = nullptr;
};
inline D3dFeedCrowdState& D3dFeedCrowd() {
  static D3dFeedCrowdState state;
  return state;
}

inline D3dFeedHandlers& D3dFeedHandlerSlots() {
  static D3dFeedHandlers handlers;
  return handlers;
}

// bit 0: the register file lags the image (flush needed); bit 1: the image is valid (in step with the register file plus
// its pending ranges). Zero = nothing to do for a foreign packet. Command processor thread only.
inline constexpr uint32_t kD3dFeedPending = 1;
inline constexpr uint32_t kD3dFeedValid = 2;
inline uint32_t& D3dFeedState() {
  static uint32_t state = 0;
  return state;
}

// The register source of the record-fed draw in progress (the record's image), else null. See VulkanCommandProcessor::DrawRegs.
class RegisterFile;
inline const RegisterFile*& D3dFeedDrawRegs() {
  static const RegisterFile* image = nullptr;
  return image;
}

// Session 90 (K2): true while the D3D feed is live (its image holds state, or a record-fed draw is open): the crowd code's restore and
// its census run only then; with the feed off the crowd code behaves exactly as before.
inline bool D3dFeedLive() { return D3dFeedState() != 0 || D3dFeedDrawRegs() != nullptr; }

// The packet the foreign check last saw (type << 8 | opcode), for the flush census (which packets make the register file catch up).
inline uint32_t& D3dFeedLastPacket() {
  static uint32_t key = 0;
  return key;
}

// The no-ring Registers item being executed (diagnostic, set by D3dNoringExecute): first op << 24 | op count (capped at 255) << 16 | the
// first op's register. The census line of a foreign flush by such an item prints it.
inline uint32_t& D3dFeedItemInfo() {
  static thread_local uint32_t info = 0;
  return info;
}
inline bool D3dFeedScopeWanted(CommandProcessor& cp) {
  const auto h = D3dFeedHandlerSlots().scope_wanted;
  return h != nullptr && h(cp);
}
inline bool D3dFeedScopeBegin(CommandProcessor& cp) {
  const auto h = D3dFeedHandlerSlots().scope_begin;
  return h != nullptr && h(cp);
}
inline bool D3dFeedRegRmw(CommandProcessor& cp, uint32_t rmw_info, uint32_t and_mask, uint32_t or_mask) {
  const auto h = D3dFeedHandlerSlots().reg_rmw;
  return h != nullptr && h(cp, rmw_info, and_mask, or_mask);
}
inline void D3dFeedScopeEnd(CommandProcessor& cp) {
  const auto h = D3dFeedHandlerSlots().scope_end;
  if (h != nullptr) h(cp);
}

inline bool D3dFeedRun(CommandProcessor& cp, rex::memory::RingBuffer* reader, uint32_t first, uint32_t n) {
  const auto handler = D3dFeedHandlerSlots().run;
  return handler != nullptr && handler(cp, reader, first, n);
}
inline void D3dFeedFlush(CommandProcessor& cp) {
  if (D3dFeedState() & kD3dFeedPending) D3dFeedHandlerSlots().flush(cp);
}
inline void D3dFeedForeign(CommandProcessor& cp) {
  if (D3dFeedState()) D3dFeedHandlerSlots().foreign(cp);
}

}  // namespace rex::graphics
