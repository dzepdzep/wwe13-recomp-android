/**
 * @file        perf/flight.h
 * @brief       Flight-recorder access to the GPU plugin's recent-work ring (wwe2k14 patch, Session 8)
 *
 * The Vulkan command processor keeps a ring of checkpoint records (draws / copies / submission starts) for
 * vulkan_gpu_crash_diagnostics. It registers a provider here at SetupContext; the app's flight recorder calls
 * GpuRecords() from any thread when it writes a dump (hang / crash / abort). The provider answers only while the
 * setting vulkan_flight_gpu_records is true (default false); otherwise it returns a one-line note.
 * Exported from rexruntime, like perf/timeline.h, so the GPU plugin and the app share it.
 */
#pragma once

#include <cstdint>
#include <string>

namespace rex::perf::flight {

// Writes up to `max_records` newest records, newest first, one text line each ("  ...\n").
using GpuRecordsFn = std::string (*)(void* user, uint32_t max_records);

// nullptr unregisters (only if `user` matches the registered one).
void SetGpuRecordsProvider(GpuRecordsFn fn, void* user);
// The provider's text, or a one-line note if none is registered.
std::string GpuRecords(uint32_t max_records);

}  // namespace rex::perf::flight
