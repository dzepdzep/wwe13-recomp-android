/**
 * @file        rex/core/fiber_android.cpp
 * @brief       Android backend for rex::thread::Fiber: not supported (stub)
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 *
 * wwe2k14 Android port: Bionic has no getcontext / makecontext / swapcontext. Titles that never create guest fibers
 * (WWE 2K14: no CreateFiber / ConvertThreadToFiber / SwitchToFiber in its code) only need thread fibers, which the
 * runtime makes for every guest thread. Creating or switching a fiber logs and aborts, so a title that does use fibers
 * fails loudly instead of misbehaving.
 */

#include <rex/platform.h>
#if REX_PLATFORM_ANDROID

#include <cstdlib>

#include <rex/logging.h>
#include <rex/thread/fiber.h>

namespace rex::thread {

thread_local Fiber* Fiber::tls_current_ = nullptr;

[[noreturn]] static void FibersUnsupported(const char* what) {
  REXSYS_ERROR("fibers are not supported on Android ({})", what);
  std::abort();
}

// The runtime converts every guest thread at start-up, fibers used or not (phone run 003). A thread fiber needs no saved
// context until something switches away from it, so this one works; Create / SwitchTo are the real fiber use.
Fiber* Fiber::ConvertCurrentThread() {
  auto* f = new Fiber();
  tls_current_ = f;
  return f;
}

Fiber* Fiber::Create(size_t, void (*)(void*), void*) {
  FibersUnsupported("Create");
}

void Fiber::SwitchTo(Fiber*) {
  FibersUnsupported("SwitchTo");
}

void Fiber::Destroy() {
  // Only thread fibers exist here (Create aborts).
  if (tls_current_ == this) {
    tls_current_ = nullptr;
  }
  delete this;
}

}  // namespace rex::thread

#endif  // REX_PLATFORM_ANDROID
