/**
 * @file        perf/flight.cpp
 * @brief       Flight-recorder GPU provider registry (wwe2k14 patch, Session 8). See perf/flight.h.
 */
#include <rex/perf/flight.h>

#include <mutex>

namespace rex::perf::flight {

namespace {
std::mutex& Mutex() {
  static std::mutex mutex;
  return mutex;
}
GpuRecordsFn g_fn = nullptr;
void* g_user = nullptr;
}  // namespace

void SetGpuRecordsProvider(GpuRecordsFn fn, void* user) {
  std::lock_guard<std::mutex> lock(Mutex());
  if (fn) {
    g_fn = fn;
    g_user = user;
  } else if (g_user == user) {
    g_fn = nullptr;
    g_user = nullptr;
  }
}

std::string GpuRecords(uint32_t max_records) {
  // A dump can happen while the GPU thread itself is stuck: never wait long for the registry lock.
  std::unique_lock<std::mutex> lock(Mutex(), std::try_to_lock);
  if (!lock.owns_lock()) return "  (GPU provider registry busy)\n";
  if (!g_fn) return "  (no GPU provider registered: not the Vulkan backend, or it is not set up)\n";
  return g_fn(g_user, max_records);
}

}  // namespace rex::perf::flight
