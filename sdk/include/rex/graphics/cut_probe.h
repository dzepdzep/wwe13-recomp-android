#pragma once
// wwe2k14 patch (Session 92, M4f-C; detector 2 of the cut): counts the calls of the register-file readers of the draw derivations while THE CUT is latched and
// the owned frame runs (gpu_d3d_owned_front bit 128). Under the cut nothing writes the register file for these ranges, so any such call is a reader that
// should have been given the typed state (bar 0). The first return addresses of each reader are kept. Command thread only; SDK-internal header.
#include <cstdint>

namespace rex::graphics::cut_probe {

enum Reader : uint32_t { kDrawRegs = 0, kTextureRegisterFile, kRenderTargetRegisterFile, kKeyRegisters, kFromRegisters, kCrowdC76, kReaderCount };
inline constexpr const char* kReaderNames[kReaderCount] = {"DrawRegs()", "TextureCache::register_file()", "RenderTargetCache::register_file()",
                                                           "KeyRegisters()", "DrawInput::FromRegisters", "SceneCrowdC76"};
inline bool g_on = false;  // set by d3d_noring.cpp: the cut is latched and the owned frame's mode is set
inline uint64_t g_count[kReaderCount] = {};
inline int64_t g_first[kReaderCount][3] = {};  // module-relative: return address minus the address of g_on in this module (stable across runs, not an ASLR pointer)
inline bool g_armed_window = false;  // the probe was armed at some point in this window
inline uint32_t g_poison_regs[4] = {0, 0, 0, 0};  // the first registers (index + 1) a check found rewritten in this window
inline uint32_t g_poison_n = 0;

inline void Note(uint32_t which, const void* ret) {
  const uint64_t n = g_count[which]++;
  if (n < 3) g_first[which][n] = int64_t(reinterpret_cast<intptr_t>(ret)) - int64_t(reinterpret_cast<intptr_t>(&g_on));
}

}  // namespace rex::graphics::cut_probe

#define REX_CUT_PROBE(which) \
  do {                       \
    if (::rex::graphics::cut_probe::g_on) ::rex::graphics::cut_probe::Note(::rex::graphics::cut_probe::which, __builtin_return_address(0)); \
  } while (0)
