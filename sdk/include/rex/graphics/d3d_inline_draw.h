// wwe2k14 patch (Session 87, D3D LAYER F3): the inline-vertex draw record of the game's D3D layer.
//
// The game's D3D library draws immediate vertices (the HUD / 2D sprites, the titantron's YUV sprite, particles, every
// post-effect quad) through one pair, BeginVertices 0x82CB1520 / EndVertices 0x82CB1510. With the layer on, the game side
// (src\d3d_layer_inline.h) builds one InlineDrawRecord per call from the device object's shadows (the five dirty masks'
// blocks, slot 31's shadow words, the call's arguments) plus the vertex bytes the game wrote, hands it to the command
// processor inline in a private token packet (opcode kInlineTokenOpcode) into the
// ring so the record keeps its place in the stream. The command thread's token handler looks the record up by sequence and
// executes it (CommandProcessor::ExecuteD3dInlineDraw): the register values are stored, the shaders set, the vertex bytes
// put into guest-addressable memory of ours, and the draw issued as an auto-indexed draw - no PM4 for any of it.
//
// A record is NOT consumed by its first execution: the library's tiling mode replays a recorded segment once per strip, and
// a token executes again with it (the record is IN the token, in the guest command memory; the sequence only counts
// the repeats). Data members only in this header's own types; CommandProcessor gets three non-virtual
// member declarations (layout unchanged).
#pragma once

#include <cstdint>
#include <vector>

namespace rex::graphics::d3d {

inline constexpr uint32_t kInlineTokenOpcode = 0x7E;           // used by no packet of the SDK's tables nor by the library (F2: 0x7D, F1: 0x7C)
inline constexpr uint32_t kInlineTokenMagic = 0xD3D30001u;     // token payload: magic, sequence, then the record (layout below)

// A run of consecutive registers (host values, in register order) the flush would have written.
struct RegisterRun {
  uint32_t first_register = 0;
  uint32_t count = 0;
  uint32_t value_offset = 0;  // into InlineDrawRecord::values
};

// A shader the flush would have loaded: microcode in guest memory (guest_address != 0, the words are read from there when
// the record executes) or carried (inline_words, guest byte order as the memory holds them: the patched vertex-shader copy
// the library builds lives only in its command stream, so a layer that reads shadows carries it).
struct ShaderBind {
  uint32_t guest_address = 0;
  uint32_t dword_count = 0;
  std::vector<uint32_t> inline_words;
  bool present() const { return dword_count != 0; }
};

enum InlineFlags : uint32_t {
  kInlineTiled = 1u << 0,       // device+11068 bit 0 (the library's extents mode): the original wraps the draw in the deferred recording
  kInlineHasPixelShader = 1u << 1,
  kInlineHasVertexShader = 1u << 2,
  kInlineUseGuestVertexMemory = 1u << 3,  // read the vertices at guest_vertex_address instead of staging the carried bytes
};

struct InlineDrawRecord {
  uint32_t sequence = 0;
  uint32_t flags = 0;
  // The call: BeginVertices(device, primitive, vertex_count, stride_bytes).
  uint32_t primitive_type = 0;  // Xenos numbering (13 quad list, 6 strip, 4 list)
  uint32_t vertex_count = 0;
  uint32_t stride_bytes = 0;
  uint32_t viz_query_word = 0;  // the library's first draw word (viz query id / use), passed through
  // The state the flush would have written: register runs (the main state blocks, the float constant groups, bool / loop,
  // the fetch slots other than slot 31's override, the misc blocks), then the shaders.
  std::vector<RegisterRun> runs;
  std::vector<uint32_t> values;
  ShaderBind vertex_shader;
  ShaderBind pixel_shader;
  // Stream 0 = fetch slot 31 for this draw only: shadow words 0-3 (registers 0x48BA-0x48BD) as the library writes them;
  // words 4-5 (0x48BE-0x48BF) are formed at execution from where the vertex bytes are.
  uint32_t slot31_words[4] = {};
  // The vertex bytes the game wrote (count * stride, guest byte order). guest_vertex_address: where the library's own
  // allocation holds them (kInlineUseGuestVertexMemory).
  std::vector<uint8_t> vertices;
  uint32_t guest_vertex_address = 0;
};

}  // namespace rex::graphics::d3d

namespace rex::memory {
class Memory;
}

namespace rex::graphics {
// wwe2k14 patch (Session 92, M4f-C): the inline record's vertex bytes made addressable by the cut's executor - InlineStage (the one body the old path uses) on
// the record's words in HOST order, as the queue carries them (21 fixed words, runs, values, vs / ps inline words, vertex bytes). Returns the guest physical
// address of the vertices; ok false = the draw is dropped. Validated by the caller (the decoder checks the counts).
uint32_t D3dInlineStageWords(memory::Memory* memory, const uint32_t* w, uint32_t count, bool& ok);
// wwe2k14 patch (Session 95, plan item P): where the same record's vertex words are (host order, the bytes stored swapped twice) and how many bytes the draw reads, for
// the host vertex ring. False = the counts do not add up, or the bytes are short. guest_memory = the vertices are in the game's own allocation (nothing to carry).
bool D3dInlineVertexWords(const uint32_t* w, uint32_t count, const uint32_t*& words, uint32_t& bytes, bool& guest_memory);
}  // namespace rex::graphics
