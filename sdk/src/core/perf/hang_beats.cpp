/**
 * @file        core/perf/hang_beats.cpp
 * @brief       Signs of life from the threads a freeze stops (wwe2k14 Session 95; see rex/perf/hang_beats.h)
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#include <rex/perf/hang_beats.h>

#include <chrono>
#include <thread>

namespace rex::perf::hang_beats {

Beats& Get() {
  static Beats beats;
  return beats;
}

void TestBlock(TestSide side) {
  Beats& beats = Get();
  while (beats.test_side.load(std::memory_order_relaxed) == side) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (side == kTestFence) beats.fence_steps.fetch_add(1, std::memory_order_relaxed);
  }
}

}  // namespace rex::perf::hang_beats
