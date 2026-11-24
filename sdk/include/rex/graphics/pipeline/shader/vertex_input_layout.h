// wwe2k14 patch (Session 95, S0 of out\s95\android_structural.md section 3.3): one rule for how a draw's vertex buffers map to vertex attributes, shared by the translator (which
// declares the inputs) and the pipeline code (which describes them). No Vulkan types here, so both can include it and pipeline workers read no shared table.
//
// `vi` is what a pipeline key holds: one byte per binding, the stride in dwords, binding 0 lowest. 0 = no vertex input. Session 95 (P7): bit 7 of a byte (kViInstanceRate) = that binding steps once per
// instance (own crowd's members; a stride is at most 64 dwords by the attribute limit, so the bit is free).
// Per binding the vertex is cut into uvec4 chunks, then a uvec2 and / or uint tail, never past the vertex: only R32G32B32A32_UINT, R32G32_UINT and R32_UINT are used. Locations run in order across
// the bindings. A pipeline's attributes may cover more of the vertex than its shader reads: the shader declares only the locations it reads (legal Vulkan), so strides alone fix the attributes.
#pragma once

#include <cstdint>

namespace rex::graphics {

constexpr uint32_t kViMaxBindings = 4, kViMaxAttributes = 16;
constexpr uint32_t kViInstanceRate = 0x80;

struct ViAttr {
  uint8_t binding, location, first_word, words;  // words: 4, 2 or 1
};

// The attributes of `vi` into `out`, in location order; returns how many. 0 = no vertex input, or over kViMaxAttributes (the caller refuses the draw by name).
constexpr uint32_t ViAttributes(uint32_t vi, ViAttr out[kViMaxAttributes]) {
  uint32_t count = 0;
  for (uint32_t binding = 0; binding < kViMaxBindings; ++binding) {
    const uint32_t stride = (vi >> (8 * binding)) & 0x7Fu;
    for (uint32_t word = 0; word < stride;) {
      const uint32_t left = stride - word;
      const uint32_t words = left >= 4 ? 4u : left >= 2 ? 2u : 1u;
      if (count == kViMaxAttributes) return 0;
      out[count] = {uint8_t(binding), uint8_t(count), uint8_t(word), uint8_t(words)};
      ++count;
      word += words;
    }
  }
  return count;
}

// Where dword `word` of a vertex in `binding` lands: the attribute's location and the component in it. false = that word is not in `vi`.
constexpr bool ViWord(uint32_t vi, uint32_t binding, uint32_t word, uint32_t* location, uint32_t* component) {
  ViAttr attrs[kViMaxAttributes] = {};
  const uint32_t count = ViAttributes(vi, attrs);
  for (uint32_t i = 0; i < count; ++i) {
    const ViAttr& a = attrs[i];
    if (a.binding == binding && word >= a.first_word && word < uint32_t(a.first_word) + a.words) {
      *location = a.location;
      *component = word - a.first_word;
      return true;
    }
  }
  return false;
}

namespace vi_check {
constexpr uint32_t Count(uint32_t vi) {
  ViAttr attrs[kViMaxAttributes] = {};
  return ViAttributes(vi, attrs);
}
constexpr uint32_t Words(uint32_t vi, uint32_t index) {
  ViAttr attrs[kViMaxAttributes] = {};
  ViAttributes(vi, attrs);
  return attrs[index].words;
}
constexpr uint32_t Place(uint32_t vi, uint32_t binding, uint32_t word) {  // location * 4 + component, or 0xFF
  uint32_t location = 0, component = 0;
  return ViWord(vi, binding, word, &location, &component) ? location * 4 + component : 0xFFu;
}
static_assert(Count(0) == 0);
static_assert(Count(8) == 2 && Words(8, 0) == 4 && Words(8, 1) == 4);                       // the arena stream
static_assert(Count(10) == 3 && Words(10, 2) == 2);                                         // the 10-dword stream: uvec4, uvec4, uvec2
static_assert(Count(3) == 2 && Words(3, 0) == 2 && Words(3, 1) == 1);                       // a tail of a uvec2 and a uint
static_assert(Count(7) == 3 && Words(7, 0) == 4 && Words(7, 1) == 2 && Words(7, 2) == 1);   // never past the vertex
static_assert(Count(8 | (3 << 8)) == 4 && Place(8 | (3 << 8), 1, 2) == 3 * 4);              // locations run on across bindings
static_assert(Place(10, 0, 9) == 2 * 4 + 1 && Place(10, 0, 10) == 0xFFu && Place(10, 1, 0) == 0xFFu);
static_assert(Count(64) == 16 && Count(65) == 0);                                           // over 16 attributes is refused
static_assert(Count(8 | ((12 | kViInstanceRate) << 8)) == 5 && Place(8 | ((12 | kViInstanceRate) << 8), 1, 0) == 2 * 4);  // own crowd: mesh, then the members at locations 2-4
static_assert(Count(4 | ((12 | kViInstanceRate) << 8)) == 4 && Place(4 | ((12 | kViInstanceRate) << 8), 1, 11) == 3 * 4 + 3);  // the cards' mesh
}  // namespace vi_check

}  // namespace rex::graphics
