// Training mode (launcher setting, WWE13_TRAINING=1): the CPU never reverses, for testing moves.
// The six "A.I. Reversals" rates of Options > Gameplay > Balancing (Standing Strike, Standing Grapple, Ground Strike,
// Ground Grapple, Finisher, Foreign Attack; 0-100, one byte each) are the first bytes of the game's options block at
// 0x8359DBEC (22 sliders, found by moving the first one 50 -> 35). They are held at 0 while training runs. The saves
// are a copy in training mode (jsdk content_manager.cpp), so the game saving these zeros never reaches the real save.
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <rex/logging.h>
#include "rex_compat.h"

namespace {
constexpr uint32_t kAiReversalRates = 0x8359DBECu;
constexpr int kAiReversalCount = 6;
}  // namespace

bool wwe13_training() {
  static const bool on = [] {
    const char* e = std::getenv("WWE13_TRAINING");
    const bool o = e && std::strcmp(e, "1") == 0;
    if (o) REXLOG_INFO("[wwe13-training] training mode: CPU reversals off, saves go to a copy");
    return o;
  }();
  return on;
}

// Called from the render-command hook (heap_diag.cpp, a guest thread, many times a frame): every 256th call.
void wwe13_training_tick(uint8_t* base) {
  if (!wwe13_training()) return;
  static uint32_t n = 0;
  if ((++n & 255) != 0) return;
  static const uint8_t want = [] {  // (WWE13_TRAINING_RATE: another rate, 0-100, for testing the switch itself)
    const char* e = std::getenv("WWE13_TRAINING_RATE");
    const int v = e && *e ? std::atoi(e) : 0;
    return uint8_t(v < 0 ? 0 : v > 100 ? 100 : v);
  }();
  static int logged = 0;
  for (int i = 0; i < kAiReversalCount; ++i) {
    const uint8_t v = PPC_LOAD_U8(kAiReversalRates + i);
    if (v != want) {
      if (logged++ < 6) REXLOG_INFO("[wwe13-training] A.I. reversal rate {} was {}: set to {}", i, v, want);
      PPC_STORE_U8(kAiReversalRates + i, want);
    }
  }
}
