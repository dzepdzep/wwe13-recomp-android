/**
 * @file        perf/audio_tap.cpp
 * @brief       Audio frame tap registry (wwe2k14 patch, Session 18). See perf/audio_tap.h.
 */
#include <rex/perf/audio_tap.h>

#include <atomic>
#include <mutex>

namespace rex::perf::audio {

namespace {
std::mutex& Mutex() {
  static std::mutex mutex;
  return mutex;
}
std::atomic<TapFn> g_fn{nullptr};
void* g_user = nullptr;
}  // namespace

void SetTap(TapFn fn, void* user) {
  std::lock_guard<std::mutex> lock(Mutex());
  if (fn) {
    g_user = user;
    g_fn.store(fn);
  } else if (g_user == user) {
    g_fn.store(nullptr);
    g_user = nullptr;
  }
}

void Tap(const void* frame_be, uint32_t samples, uint32_t channels, uint32_t frequency) {
  if (!g_fn.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lock(Mutex());
  if (TapFn fn = g_fn.load()) fn(g_user, frame_be, samples, channels, frequency);
}

}  // namespace rex::perf::audio
