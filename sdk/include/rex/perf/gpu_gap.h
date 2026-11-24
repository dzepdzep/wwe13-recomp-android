/**
 * @file        perf/gpu_gap.h
 * @brief       What the GPU does between two frames (wwe2k14 patch, Session 95, item T)
 *
 * Measurement only (present_gpu_gap_stats, default off). The command processor hands in the timestamps of each of its
 * submissions and the Vulkan presenter those of each paint: one queue, one timestamp clock, so the time between
 * them is read on the GPU's own timeline, with no CPU wake-up in it. Every 600 frames one [pace][gpu-gap] line.
 * A separate header so that the headers the game exe includes stay unchanged. Defined in perf/gpu_gap.cpp (rexruntime).
 */
#pragma once

#include <cstdint>

namespace rex::perf::gpu_gap {

bool Enabled();
// From the presenter, once: ns per timestamp tick and the mask of the valid bits. Nothing is reported before it.
void SetTicks(double ns_per_tick, uint64_t tick_mask);
// A command processor submission / a presenter paint that has completed, in raw timestamp ticks. Any thread, any order.
void NoteSubmission(uint64_t begin_tick, uint64_t end_tick, bool is_swap);
void NotePaint(uint64_t begin_tick, uint64_t end_tick);

}  // namespace rex::perf::gpu_gap
