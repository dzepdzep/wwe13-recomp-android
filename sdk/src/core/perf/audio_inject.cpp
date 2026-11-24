/**
 * @file        perf/audio_inject.cpp
 * @brief       Audio injection registry (wwe2k14 patch, Session 101 lane 288). See perf/audio_inject.h.
 */
#include <rex/perf/audio_inject.h>

#include <atomic>
#include <cstring>

namespace rex::perf::audio {

namespace {
constexpr uint32_t kMaxSamples = 2048;  // stack scratch per plane; longer frames are left alone
std::atomic<InjectFn> g_fn{nullptr};
std::atomic<void*> g_user{nullptr};

inline float SwapF(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  u = __builtin_bswap32(u);
  std::memcpy(&v, &u, 4);
  return v;
}
}  // namespace

void SetInject(InjectFn fn, void* user) {
  if (fn) {
    g_user.store(user, std::memory_order_release);
    g_fn.store(fn, std::memory_order_release);
  } else if (g_user.load() == user) {
    g_fn.store(nullptr, std::memory_order_release);
    g_user.store(nullptr, std::memory_order_release);
  }
}

bool Inject(void* frame_be, uint32_t samples, uint32_t channels, uint32_t frequency) {
  InjectFn fn = g_fn.load(std::memory_order_acquire);
  if (!fn || channels < 2 || samples == 0 || samples > kMaxSamples) return false;
  float left[kMaxSamples];
  float right[kMaxSamples];
  std::memset(left, 0, samples * sizeof(float));
  std::memset(right, 0, samples * sizeof(float));
  if (!fn(g_user.load(std::memory_order_acquire), left, right, samples, frequency)) return false;
  float* fl = static_cast<float*>(frame_be);
  float* fr = fl + samples;
  for (uint32_t i = 0; i < samples; ++i) {
    fl[i] = SwapF(SwapF(fl[i]) + left[i]);
    fr[i] = SwapF(SwapF(fr[i]) + right[i]);
  }
  return true;
}

}  // namespace rex::perf::audio