/**
 * @file        rex/perf/frame_events.h
 * @brief       What a frame did besides drawing (wwe2k14 Session 73)
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 *
 * Counters the GPU side bumps when it makes something on the spot (a pipeline, a truth image, a shader module) or
 * waits for the GPU, and the GPU's own busy time per frame from the submissions' timestamps. The game reads them
 * at every presented frame (src/frame_stats.h in the wwe2k14 repository): a slow frame's log line says what happened
 * in it, and the per-frame CSV gets the GPU time. Cheap: relaxed atomics, bumped only when something is made.
 */

#pragma once

#include <atomic>
#include <cstdint>

namespace rex::perf::frame_events {

// Why the command thread waited for the GPU (b78: the slow-frame lines split "GPU waits" by these).
enum WaitSite : uint32_t {
  kWaitOther = 0,   // anything not tagged below
  kWaitFrame,       // a new frame's submission: the frame kMaxFramesInFlight back must be done
  kWaitCopy,        // a resolve / copy (IssueCopy: readbacks of render targets to guest memory)
  kWaitOcclusion,   // an occlusion query's result
  kWaitReadback,    // the deferred readbacks flushed
  kWaitDraw,        // a draw (sampler pool overflow)
  kWaitSiteCount,
};
inline constexpr const char* kWaitSiteNames[kWaitSiteCount] = {"other", "frame", "copy", "occlusion", "readback",
                                                               "draw"};

struct Counters {
  std::atomic<uint32_t> pipelines{0};    // graphics pipelines made (the native renderer's and the emulated path's)
  std::atomic<uint32_t> images{0};       // truth images (and rename spares) made
  std::atomic<uint32_t> shaders{0};      // shader modules made (a translation finished)
  std::atomic<uint64_t> gpu_wait_us{0};  // the command thread blocked on submission fences
  std::atomic<uint64_t> gpu_wait_us_by[kWaitSiteCount]{};  // b78: the same, by WaitSite
  // b78: every submission's GPU time added up (us), whether or not a frame ends with it: the GPU's busy share over a
  // stretch of wall time (the keep-warm control).
  std::atomic<uint64_t> gpu_busy_us_total{0};
  // The GPU's busy time (the sum of its submissions' timestamp spans) of the latest frame it finished, and how many
  // frames it has finished (a frame = the submissions up to and including a swap). 0 = not measured.
  std::atomic<uint32_t> gpu_frame_us{0};
  std::atomic<uint64_t> gpu_frames_done{0};
  // Set by the game: whether the GPU side should measure (the timestamps cost two queries per submission).
  std::atomic<bool> measure_gpu{false};
  // Texture loads (guest texture data converted on the GPU) and their guest size in KB.
  std::atomic<uint32_t> textures{0};
  std::atomic<uint64_t> texture_kb{0};
  // Set by the game: busy work for the GPU at the next swap, KB of buffer fill (one-shot; the GPU side takes it).
  // Keeps the phone's GPU clock up through light stretches (loads) so the next scene does not start at the lowest
  // clock (src/frame_stats.h "keep warm"). warm_kb_done: what the GPU side recorded.
  std::atomic<uint32_t> warm_kb{0};
  std::atomic<uint64_t> warm_kb_done{0};
  // Session 95: guest frames that reached vkQueuePresentKHR, and the time between the last two (us). Kept last: the game
  // exe and the SDK DLLs share this struct. An older exe on newer DLLs is safe; a newer exe on older DLLs reads past the
  // end of the DLL's Counters (garbage in presented), so build them together.
  std::atomic<uint64_t> presents{0};
  std::atomic<uint32_t> present_gap_us{0};
};

Counters& Get();

}  // namespace rex::perf::frame_events
