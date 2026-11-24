#pragma once
/**
 * wwe2k14 patch (Session 6, CP4/C5): header-inline equivalents of rex::lzcnt, rex::tzcnt and
 * rex::bit_scan_forward for the GPU plugin.
 *
 * The rex:: versions are declared out of line in include/rex/math.h and defined in rexruntime.dll
 * (src/core/math_msvc.cpp on Windows), so every call from rexgpu-xenos.dll is a cross-DLL call
 * through an import thunk - paid once per set bit in the command processor's per-draw bit loops
 * (UpdateBindings, UpdateSystemConstantValues, IssueDraw, RequestTextures...). include/rex/math.h is
 * also included by the game executable, so it is not changed; GPU code that wants the inline
 * versions calls bit_inline:: instead of rex::.
 *
 * Same overload set, return types and results as the rex:: functions, including zero input (checked
 * against the disassembly of the Release math_msvc.cpp.obj):
 * - lzcnt / tzcnt return the operand width (8, 16, 32 or 64) for 0, otherwise the number of leading
 *   / trailing zero bits.
 * - bit_scan_forward returns false for 0 and does not write *out_first_set_index (clang's
 *   _BitScanForward only stores for nonzero input; math.h documents the index as invalid then).
 *   Otherwise it stores the index of the lowest set bit and returns true. Note: the non-Windows
 *   rex::bit_scan_forward (math_gcc.cpp) stores 0xFFFFFFFF for 0 instead; no caller may rely on
 *   the index after a false return.
 */

#include <bit>
#include <cstdint>

namespace rex::graphics::bit_inline {

// lzcnt - count leading zeros. Returns the size of the input operand if value is zero.
inline uint8_t lzcnt(uint8_t v) {
  return static_cast<uint8_t>(std::countl_zero(v));
}
inline uint8_t lzcnt(uint16_t v) {
  return static_cast<uint8_t>(std::countl_zero(v));
}
inline uint8_t lzcnt(uint32_t v) {
  return static_cast<uint8_t>(std::countl_zero(v));
}
inline uint8_t lzcnt(uint64_t v) {
  return static_cast<uint8_t>(std::countl_zero(v));
}
inline uint8_t lzcnt(int8_t v) {
  return lzcnt(static_cast<uint8_t>(v));
}
inline uint8_t lzcnt(int16_t v) {
  return lzcnt(static_cast<uint16_t>(v));
}
inline uint8_t lzcnt(int32_t v) {
  return lzcnt(static_cast<uint32_t>(v));
}
inline uint8_t lzcnt(int64_t v) {
  return lzcnt(static_cast<uint64_t>(v));
}

// tzcnt - count trailing zeros. Returns the size of the input operand if value is zero.
inline uint8_t tzcnt(uint8_t v) {
  return static_cast<uint8_t>(std::countr_zero(v));
}
inline uint8_t tzcnt(uint16_t v) {
  return static_cast<uint8_t>(std::countr_zero(v));
}
inline uint8_t tzcnt(uint32_t v) {
  return static_cast<uint8_t>(std::countr_zero(v));
}
inline uint8_t tzcnt(uint64_t v) {
  return static_cast<uint8_t>(std::countr_zero(v));
}
inline uint8_t tzcnt(int8_t v) {
  return tzcnt(static_cast<uint8_t>(v));
}
inline uint8_t tzcnt(int16_t v) {
  return tzcnt(static_cast<uint16_t>(v));
}
inline uint8_t tzcnt(int32_t v) {
  return tzcnt(static_cast<uint32_t>(v));
}
inline uint8_t tzcnt(int64_t v) {
  return tzcnt(static_cast<uint64_t>(v));
}

// BitScanForward (bsf).
// Search the value from least significant bit (LSB) to the most significant bit
// (MSB) for a set bit (1).
// Returns false if no bits are set (the output index is not written then).
inline bool bit_scan_forward(uint32_t v, uint32_t* out_first_set_index) {
  if (!v) {
    return false;
  }
  *out_first_set_index = static_cast<uint32_t>(std::countr_zero(v));
  return true;
}
inline bool bit_scan_forward(uint64_t v, uint32_t* out_first_set_index) {
  if (!v) {
    return false;
  }
  *out_first_set_index = static_cast<uint32_t>(std::countr_zero(v));
  return true;
}
inline bool bit_scan_forward(int32_t v, uint32_t* out_first_set_index) {
  return bit_scan_forward(static_cast<uint32_t>(v), out_first_set_index);
}
inline bool bit_scan_forward(int64_t v, uint32_t* out_first_set_index) {
  return bit_scan_forward(static_cast<uint64_t>(v), out_first_set_index);
}

}  // namespace rex::graphics::bit_inline
