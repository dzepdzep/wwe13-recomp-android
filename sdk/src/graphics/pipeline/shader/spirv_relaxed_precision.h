// wwe2k14 patch (Session 73, gpu_native_ps_relaxed_precision): an optional half-precision mode for the translated
// pixel shaders. A SPIR-V binary post-pass that marks the results of 32-bit float arithmetic and texture samples in a
// fragment-only module RelaxedPrecision (GLSL mediump). Phone GPUs (Adreno, Mali) may then run that maths at 16 bits
// (about twice the ALU rate and half the registers); desktop drivers ignore the decoration, so the PC picture stays the
// same. Kept at full precision: everything that flows (through SSA values and through Function / Private memory) into a
// texture coordinate / LOD / gradient / depth-compare operand, an array or vector index, an integer result (float to
// int conversions, bitcasts, packing) or a FragDepth store. No vertex shaders (every entry point must be Fragment).
//
// No dependencies besides glslang's SPIR-V header, so tools/android/fp16 in the main repo builds it on its own (with
// REX_SPIRV_RELAXED_PRECISION_STANDALONE defined, which leaves out the cvar and the logging).

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rex::graphics::spirv_relaxed_precision {

struct Stats {
  uint32_t decorated = 0;       // results given RelaxedPrecision by this pass
  uint32_t candidates = 0;      // float arithmetic / sample results of the listed kinds
  uint32_t kept_full = 0;       // candidates left at full precision (coordinates, indices, depth, integers)
  uint32_t already_relaxed = 0;  // candidates that already carried the decoration
};

// The module with the decorations added, in out. False (out untouched) for a module that is not fragment-only, that
// cannot be parsed, or that has nothing to decorate.
bool AddRelaxedPrecision(const uint32_t* words, size_t word_count, std::vector<uint32_t>& out, Stats* stats = nullptr);

#ifndef REX_SPIRV_RELAXED_PRECISION_STANDALONE
// The pass when gpu_native_ps_relaxed_precision is on (logged per shader, rate-limited). False = use the input as it
// is (the cvar is off, or nothing was decorated).
bool ApplyIfEnabled(const uint32_t* words, size_t word_count, uint64_t shader_hash, uint64_t modification,
                    std::vector<uint32_t>& out);

// Session 73 (Jarrod: "wrestlers are the most important part"): the characters' pixel shaders stay at full precision
// (half precision darkened and tinted the wrestlers' skin, replay 4711). A pixel shader is a character's when a draw
// (or a stored pipeline record) pairs it with one of the character vertex shaders (gpu_dlss_objmv_shaders, Session 16's
// classification), or when gpu_native_ps_full_precision_shaders lists it. Both are known before its module is made:
// the draw notes the pair before it asks for the translation, the storage load before the prewarm.
// Cheap when gpu_native_ps_relaxed_precision is off (one bool).
void NoteShaderPair(uint64_t vertex_shader_hash, uint64_t pixel_shader_hash);
bool IsFullPrecision(uint64_t pixel_shader_hash);
#endif

}  // namespace rex::graphics::spirv_relaxed_precision
