/**
 * @file        perf/dlaa.cpp
 * @brief       DLAA camera sink registry (wwe2k14 patch, Session 18). See perf/dlaa.h.
 */
#include <rex/perf/dlaa.h>

#include <mutex>

namespace rex::perf::dlaa {

namespace {
std::mutex& Mutex() {
  static std::mutex mutex;
  return mutex;
}
CameraFn g_fn = nullptr;
void* g_user = nullptr;
}  // namespace

void SetCameraSink(CameraFn fn, void* user) {
  std::lock_guard<std::mutex> lock(Mutex());
  if (fn) {
    g_fn = fn;
    g_user = user;
  } else if (g_user == user) {
    g_fn = nullptr;
    g_user = nullptr;
  }
}

bool SubmitCamera(uint64_t frame, const float view[16], float fov_degrees, float near_z, float far_z) {
  std::lock_guard<std::mutex> lock(Mutex());
  if (!g_fn) return false;
  g_fn(g_user, frame, view, fov_degrees, near_z, far_z);
  return true;
}

}  // namespace rex::perf::dlaa
