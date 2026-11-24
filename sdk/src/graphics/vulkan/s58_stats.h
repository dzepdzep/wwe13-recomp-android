// wwe2k14 patch (Session 58, problem reports): counters for the [s58][gpu] summary line that
// VulkanCommandProcessor::IssueSwap writes every gpu_session_stats_frames guest frames: pipelines made (the emulated
// path's guest pipelines and the native renderer's draw pipelines) with their creation times, and draws skipped while
// their pipeline was made.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace rex::graphics::vulkan::s58 {

struct PipelineStats {
  std::atomic<uint64_t> count{0}, total_us{0}, max_us{0};
  void Add(std::chrono::steady_clock::time_point t0) {
    const uint64_t us = uint64_t(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
    count.fetch_add(1, std::memory_order_relaxed);
    total_us.fetch_add(us, std::memory_order_relaxed);
    uint64_t m = max_us.load(std::memory_order_relaxed);
    while (us > m && !max_us.compare_exchange_weak(m, us, std::memory_order_relaxed)) {
    }
  }
};
inline PipelineStats g_stock_pipelines, g_native_pipelines;
inline std::atomic<uint64_t> g_skipped_draws{0};

}  // namespace rex::graphics::vulkan::s58
