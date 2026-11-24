// wwe2k14 patch (Session 87, D3D layer STAGE 2 "no ring"): the typed host queue between the game's D3D layer hooks (the
// producer: render thread, any guest thread) and the command processor thread (the consumer). Docs in the game repository:
// docs\native\d3d\F_direct_and_noring.md sections 2.3-2.9, docs\native\d3d\F4_S2_CONTRACTS.md, docs\native\d3d\STAGE2_NORING.md.
//
// What it replaces: the ring. With the layer's no-ring mode on, the library's ring submit (0x82CB5EF0) writes nothing to the
// ring and no WPTR; the segments (and the typed sync items) reach the command thread through this queue instead, strictly in
// program order. The consumer (src\graphics\d3d_noring.cpp, CommandProcessor::D3dNoringDrain) is called from the SDK
// command loop in front of the ring branch; with the cvar gpu_d3d_noring at 0 (default) the loop never looks at the queue.
//
// Shared by the SDK and the game exe as plain data: the exe keeps its OWN copy of this layout (src\d3d_noring.h), the SDK
// never links to the exe and the exe calls no SDK function - the exe allocates a D3dNoringShared, fills it and hands its
// address over in the cvar gpu_d3d_noring_shared (a uint64); the consumer reads it from there. The two are tied by the
// int capability cvar gpu_d3d_noring_abi (= kD3dNoringAbi below): change a field or an item layout here -> bump it here AND
// in the exe's copy.
//
// Entry in the storage ring (dwords, host byte order; an entry never wraps - a kNrPad entry fills the rest of the storage):
//   [0] kind | (total dwords of the entry << 8)   [1-6] the item's six fields   [7 ...] payload (total - 7 dwords)
// Items (field indexes below). All guest addresses are the GPU-form values the library itself would have put into the
// packet ((((a >> 20) + 512) & 0x1000) + (a & 0x1FFFFFFF), then |2 for the endian bits where a memory poll / write has them).
#pragma once

#include <atomic>
#include <cstdint>

namespace rex::thread {
class Event;
enum class WaitResult;
}

namespace rex::graphics {

inline constexpr int32_t kD3dNoringAbi = 2;  // the capability cvar gpu_d3d_noring_abi (an exe written for 2 keeps working)
// (Session 92, M5b) The block layout with the wake fields (consumer_idle, wake, wake_ctx, coher_dropped at the END of the struct). The SDK
// advertises it as the capability cvar gpu_d3d_noring_abi_max = 3; an exe that sees it may set the block's `abi` to 3, an exe that does not
// (or an ABI-2 exe) allocates / fills the block as before and the SDK never touches the extra fields (the old poll, counted).
inline constexpr int32_t kD3dNoringAbiWake = 3;
inline constexpr uint32_t kD3dNoringMagic = 0xD3D40001u;
inline constexpr uint32_t kD3dNoringHeaderDwords = 7;

enum D3dNoringKind : uint32_t {
  kNrPad = 0,
  kNrSegment = 1,        // a piece of a library segment in guest memory, executed as an INDIRECT_BUFFER (stage 2a/2b)
  kNrPackets = 2,        // packets by value (payload; the record tokens 0x7C / 0x7D / 0x7E - stage 2c)
  kNrFence = 3,          // the segment-end fence pair, written by the consumer after every item before it
  kNrFlipHandshake = 4,  // 0x82CB3300: the present's flip word handshake (scratch+4 = 1, interrupt, wait scratch+4 == 0)
  kNrSceneTiming = 5,    // 0x82CB6510 with callback 0x82C16978 / 0x82C16980 (scene-timing slots)
  kNrInterrupt = 6,      // 0x82CB6510 for any other callback
  kNrPresent = 7,        // VdSwap's words (fetch constants + XE_SWAP), executed as ring-level packets
  kNrBarrier = 8,        // the consumer publishes the ticket when it has executed everything before this item
  kNrRegs = 9,           // plain register writes / read-modify-writes (the display registers of the present, the 0x5C8 flush)
  kNrKindCount
};

// Field indexes (dwords 1-6 of the entry).
namespace d3d_noring_field {
// kNrSegment
inline constexpr uint32_t kSegAddress = 0;  // GPU-form address of the first word (what the IB packet holds)
inline constexpr uint32_t kSegLength = 1;   // length in dwords
inline constexpr uint32_t kSegFlags = 2;    // bit 0: INDIRECT_BUFFER_PFD (0x37) instead of INDIRECT_BUFFER (0x3F): the ring submit's first-byte 0x8F variant
// kNrPackets: payload = the packet words; field 0 = tag (diagnostics: the producer's family)
inline constexpr uint32_t kPktTag = 0;
// kNrFence
inline constexpr uint32_t kFenceValue = 0;     // device+11036: the fence number
inline constexpr uint32_t kFenceMarker = 1;    // (device+15324 & 3) | device+48: the heap progress marker
inline constexpr uint32_t kFenceMarkerAt = 2;  // GPU-form address of fence block +4, |2
inline constexpr uint32_t kFenceValueAt = 3;   // GPU-form address of fence block +0, |2
inline constexpr uint32_t kFenceFlags = 4;     // bit 0: the segment end's 0x5C8 write (device byte 11072 bit 0x80)
// kNrFlipHandshake / kNrSceneTiming / kNrInterrupt (the arguments of 0x82CB6510 + the handshake's own words)
inline constexpr uint32_t kIntFlags = 0;       // r5: bits 24-29 cpu mask, bit 0 / 1 / 2 / 0x80 as the emitter reads them
inline constexpr uint32_t kIntCallback = 1;    // r6: the guest callback
inline constexpr uint32_t kIntArgument = 2;    // r7: its argument
inline constexpr uint32_t kIntScratch = 3;     // GPU-form address of the scratch block (device+11028), no endian bits
inline constexpr uint32_t kIntOptions = 4;     // bit 0: pre-arm (scratch+4 = 1), bit 1: wait scratch+4 == 0 afterwards,
                                               // bit 2: the emitter's 0x5C8 write (misc flush) precedes the callback regs
// kNrPresent: payload = VdSwap's words (host order, trailing type-2 NOPs trimmed); fields 0-2 = front buffer, width, height
inline constexpr uint32_t kPresentFront = 0, kPresentWidth = 1, kPresentHeight = 2;
// kNrRegs: payload = quads {op, register, a, b}: op 1 = write register := a; op 2 = register := (register & a) | b
inline constexpr uint32_t kRegsOpWrite = 1, kRegsOpRmw = 2;
// Capability gpu_d3d_noring_ops = 1: the tiling predication state (SET_BIN_MASK_LO / _HI, SET_BIN_SELECT_LO / _HI): a = the value, register unused
inline constexpr uint32_t kRegsOpBinMaskLo = 3, kRegsOpBinMaskHi = 4, kRegsOpBinSelectLo = 5, kRegsOpBinSelectHi = 6;
// kNrBarrier
inline constexpr uint32_t kBarrierTicket = 0;
}  // namespace d3d_noring_field

// The shared block. Allocated and owned by the game exe; one producer lock (the exe's), one consumer (the command thread).
struct D3dNoringShared {
  uint32_t magic = kD3dNoringMagic;
  uint32_t abi = kD3dNoringAbi;
  uint32_t* storage = nullptr;  // capacity dwords
  uint32_t capacity = 0;        // dwords, a power of two
  alignas(64) std::atomic<uint32_t> tail{0};  // producer: dwords written so far (wraps at 2^32; position = tail & (capacity - 1))
  alignas(64) std::atomic<uint32_t> head{0};  // consumer: dwords consumed
  alignas(64) std::atomic<uint32_t> consumed_ticket{0};  // the last kNrBarrier ticket the consumer reached
  // The consumer's census (relaxed; the exe prints it with its own).
  std::atomic<uint64_t> items[kNrKindCount]{};
  std::atomic<uint64_t> segment_dwords{0};   // dwords executed from kNrSegment items (stage 2c: should be 0 in the scene)
  std::atomic<uint64_t> packet_dwords{0};    // dwords executed from kNrPackets / kNrPresent items
  std::atomic<uint64_t> fences{0};           // fence values written
  std::atomic<uint64_t> flip_waits{0};       // flip handshakes whose wait on the flip word really had to wait
  std::atomic<uint64_t> flip_wait_us{0};     // microseconds spent in those waits
  std::atomic<uint64_t> bad{0};              // framing / unknown-kind errors (loud)
  // ---- ABI 3 only (Session 92, M5b): present when `abi` >= 3 (an ABI-2 exe's block ends at `bad`; the SDK reads nothing below then). ----
  // The Dekker pair: the consumer stores consumer_idle = 1 (seq_cst) and THEN re-reads tail (seq_cst) before it sleeps; the producer stores
  // tail (seq_cst) and THEN reads consumer_idle (seq_cst) and, when 1, calls wake(wake_ctx). Either the consumer sees the new tail or the
  // producer sees the idle flag: no lost wake. wake / wake_ctx are filled by the consumer before it first stores consumer_idle = 1.
  alignas(64) std::atomic<uint32_t> consumer_idle{0};
  void (*wake)(void*) = nullptr;
  void* wake_ctx = nullptr;
  std::atomic<uint64_t> coher_dropped{0};    // producer side: kick coherency handshakes (0x82CB61C0) not emitted (wwe2k14_d3d_drop_coher)
};

// The worker loop's idle wait in milliseconds: 5 as before, 1 while the host queue is on (the exe cannot signal the loop's event, so
// the idle wait polls the queue more often). d3d_noring.cpp.
int D3dNoringIdleWaitMs();

// (Session 92, M5b) latch gpu_d3d_sync_events, effective only with the host queue AND gpu_d3d_sync_direct.
bool D3dNoringSyncEventsOn();
// The vblank event: MarkVblank signals it (after its dispatch), CommandProcessor::WaitGuestWord waits on it (timeout_ms is a guard).
void D3dNoringVblankSignal();
bool D3dNoringVblankWait(int timeout_ms);  // true = signalled (counted as a vblank wake), false = timeout
// The command loop's idle wait. Arm: 0 = not applicable (wait as before), 1 = armed (wait, then Disarm), 2 = work is queued (skip the wait).
// Disarm(timed_out): clears the flag and counts a wake, or a guard timeout (with work queued).
int D3dNoringIdleArm(rex::thread::Event* loop_event);
// Waits on the loop's own event AND the never-freed wake event (the target of the block's wake()).
rex::thread::WaitResult D3dNoringIdleWait(rex::thread::Event* loop_event, int timeout_ms);
void D3dNoringIdleDisarm(bool timed_out);
// The producer-side counter of dropped coherency handshakes since the last call (for the [d3d][free] line).
uint64_t D3dNoringTakeCoherDropped();

// Stage 2c: non-zero while the command thread executes a kNrPackets item (a record token by value): the library-draw kind its
// draws are (1 = the indexed draw token 0x7C, 2 = the inline draw token 0x7E, 0xFF = a record with no draw, e.g. the copy token
// 0x7D). Half2OnCpDraw reads it instead of matching the packet's address: the record knows what it is.
uint32_t D3dNoringRecordKind();

}  // namespace rex::graphics
