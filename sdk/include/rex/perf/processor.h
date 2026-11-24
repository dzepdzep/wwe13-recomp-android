/**
 * @file        perf/processor.h
 * @brief       Which logical processor a thread runs on, for the timeline (wwe2k14 patch, Session 6)
 *
 * A separate header so that perf/timeline.h (included by the app) stays unchanged. Defined in
 * perf/timeline.cpp (rexruntime).
 */
#pragma once

#include <cstdint>

namespace rex::perf {

// The logical processor the calling thread is running on right now: on Windows group * 64 + number
// (GetCurrentProcessorNumberEx), on Linux sched_getcpu(), 0 if unknown.
uint32_t CurrentProcessorNumber();

}  // namespace rex::perf
