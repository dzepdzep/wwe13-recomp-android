// wwe2k14 patch (Session 48, HALF2): the game's own D3D library draws, as the guest device's shadow state gives
// them, handed to the command processor (docs\native\HALF2_MAP.md in the game repository). Step N5b / N6a: the
// command processor matches each of its draw packets to the library draw that wrote it (by guest address) and
// compares its register file with the device's shadow (gpu_native_half2 1). No game knowledge here: the game's hooks
// fill the runs (which registers, their values) and the float-constant group hashes from its device.
#pragma once

#include <atomic>
#include <cstdint>

namespace rex::graphics {

struct Half2Run {
  uint16_t first_register = 0;
  uint16_t count = 0;
};

// One library draw. Filled by the game between CommandProcessor::Half2BeginLibraryDraw (which reserves it in order)
// and Half2EndLibraryDraw (which publishes it); read by the command processor thread.
struct Half2LibraryDraw {
  static constexpr uint32_t kMaxRuns = 12;
  static constexpr uint32_t kMaxValues = 384;
  static constexpr uint32_t kFloatGroups = 128;  // 0x4000-0x47FF in groups of 16 dwords (4 constants)
  std::atomic<uint32_t> state{0};  // 0 free, 1 reserved (the game is filling it), 2 ready
  uint32_t kind = 0;           // the game's writer: 1 indexed draw, 2 inline vertices, 3 clear, 4 resolve
  uint32_t packets_begin = 0;  // guest physical address of the first word the writer wrote
  uint32_t packets_end = 0;    // one past its last word (0: unknown)
  uint32_t device = 0;         // the guest device (diagnostics)
  uint32_t queued_frame = 0;   // the command processor's frame count when it was reserved (stale entries are dropped)
  // The draw itself (kind 1), from the device: primitive type, index count (all chunks), the first index's guest
  // address (0: none) and 32-bit indices; the shaders' microcode (vertex: one per variant of the bound object, 0 =
  // none). The game passes guest virtual addresses; Half2EndLibraryDraw makes them physical.
  uint32_t prim = 0, index_count = 0, index_address = 0;
  bool index32 = false;
  uint32_t vs_ucode[2] = {}, ps_ucode = 0;
  // The last patched vertex-shader copy the library wrote inline (0x82CBE210: the variant's microcode copied into
  // the command stream and patched there), its first microcode word; 0 = none yet.
  uint32_t vs_inline_ucode = 0;
  // wwe2k14 patch (Session 51, STATE LOG): the state-log record holding this call's packets (gpu_native_half2 3; a
  // call made inside another call shares the outer call's record). 0: none.
  uint64_t log_seq = 0;
  // wwe2k14 patch (Session 79, NF-D look-ahead): for a draw of the watched pixel shader
  // (CommandProcessor::half2_watch_ps_ucode, the entrance combine), its pixel constant c32 from the device's float
  // shadow (the game's hooks, every half2 mode); has_ps_c32 false otherwise.
  bool has_ps_c32 = false;
  uint32_t ps_c32[4] = {};
  uint32_t run_count = 0;
  uint32_t value_count = 0;
  Half2Run runs[kMaxRuns];
  uint32_t values[kMaxValues];       // the runs' registers in order, host byte order
  uint64_t float_hash[kFloatGroups];  // Half2HashGroup of each 16-dword group of float constants
};

// The hash both sides use for a group of 16 float-constant dwords (host byte order).
inline uint64_t Half2HashGroup(const uint32_t* v) {
  uint64_t h = 0xCBF29CE484222325ull;
  for (uint32_t i = 0; i < 16; ++i) {
    h = (h ^ v[i]) * 0x100000001B3ull;
    h ^= h >> 29;
  }
  return h;
}

}  // namespace rex::graphics
