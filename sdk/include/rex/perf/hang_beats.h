/**
 * @file        rex/perf/hang_beats.h
 * @brief       Signs of life from the threads a freeze stops (wwe2k14 Session 95, piece HANG)
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 *
 * The GPU side bumps these; the game's guard thread (src/hang_guard.h in the wwe2k14 repository) reads them once a second
 * and names which side of a freeze stopped. A getter from one library, never an inline variable: each shared library would
 * get its own copy. Cheap: relaxed atomics.
 */

#pragma once

#include <atomic>
#include <cstdint>

namespace rex::perf::hang_beats {

// wwe2k14_test_hang: where a test stops a thread for good (the guest side stops in the game's own code).
enum TestSide : uint32_t { kTestNone = 0, kTestPaint = 1, kTestFence = 2 };

struct Beats {
  std::atomic<uint64_t> command{0};           // the command thread went round its loop (busy or idle)
  std::atomic<uint64_t> paints{0};            // a paint started
  std::atomic<uint64_t> presents{0};          // vkQueuePresentKHR accepted a picture
  std::atomic<uint64_t> fence_steps{0};       // a fence wait timed out one more 1 s step
  std::atomic<uint64_t> acquire_timeouts{0};  // Android: the bounded acquire gave up, that present was skipped
  std::atomic<uint32_t> in_paint{0};          // a thread is inside the paint now
  std::atomic<uint32_t> test_side{kTestNone};
  std::atomic<uint64_t> vk_used_mb{0};        // device-local heaps, from the last read (VK_EXT_memory_budget); 0 = none
  std::atomic<uint64_t> vk_budget_mb{0};
  std::atomic<bool> want_budget{false};       // the guard is on: read the budget every pass, not only for the [s58] line
};

Beats& Get();

// For a beat with one writer: no read-modify-write instruction.
inline void Beat(std::atomic<uint64_t>& beat) {
  beat.store(beat.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

// Test only: returns at once unless wwe2k14_test_hang chose this side, then never (a fence test still counts its steps).
void TestBlock(TestSide side);

}  // namespace rex::perf::hang_beats
