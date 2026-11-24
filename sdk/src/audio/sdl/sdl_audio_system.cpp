/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <rex/audio/flags.h>
#include <rex/audio/sdl/sdl_audio_driver.h>
#include <rex/audio/sdl/sdl_audio_system.h>
#include <rex/cvar.h>
#include <rex/logging.h>
#include <SDL3/SDL.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

// wwe2k14 patch (Session 27): SDL's audio subsystem is started here, on the UI thread, before the guest runs.
// SDL_PumpEvents (UI thread) calls SDL_UpdateAudio on every pump, which read-locks and unlocks
// current_audio.subsystem_rwlock, reading the pointer twice. SDL_InitAudio sets that pointer (and zeroes
// current_audio per driver it tries) without a lock. When the guest's audio driver ran the first
// SDL_InitSubSystem(SDL_INIT_AUDIO) on a guest thread, the UI thread could "lock" a null lock and then unlock the new
// one: ReleaseSRWLockShared on a lock it does not hold = 0xC0000264 (the boot crash in SDL_UpdateAudio,
// SDL_audio.c:2612, Sessions 22-23). With the subsystem started here, the driver's later init only adds a reference.
REXCVAR_DEFINE_BOOL(audio_sdl_init_on_ui_thread, true, "Audio",
                    "Start SDL's audio subsystem on the UI thread at setup (avoids a boot race in SDL_UpdateAudio)");

namespace rex::audio::sdl {

namespace {

// wwe2k14 patch (Session 76): a player's PC whose default playback device would not open (SDL_OpenAudioDeviceStream:
// "WASAPI can't initialize audio client") crashed at boot: CreateDriver failed, so XAudioRegisterRenderDriverClient
// returned an error (a 360 never does) and the game went on to lock a sound object it had never made
// (RtlEnterCriticalSection on a null pointer; reproduced with SDL_AUDIO_DRIVER set to a missing driver). With no
// device the game now gets this driver: each submitted frame is handed back at the device's pace (one 256-sample
// frame per 5.33 ms at 48 kHz, as SDLAudioDriver::SDLCallback does), so the game runs without sound.
class SilentAudioDriver final : public AudioDriver {
 public:
  SilentAudioDriver(memory::Memory* memory, rex::thread::Semaphore* semaphore)
      : AudioDriver(memory), semaphore_(semaphore), thread_([this] { Run(); }) {}
  ~SilentAudioDriver() override {
    {
      std::lock_guard lock(mutex_);
      quit_ = true;
    }
    cv_.notify_all();
    thread_.join();
  }
  void SubmitFrame(uint32_t /*frame_ptr*/) override {
    std::lock_guard lock(mutex_);
    ++queued_;
  }

 private:
  void Run() {
    using clock = std::chrono::steady_clock;
    const auto period = std::chrono::nanoseconds(1'000'000'000LL * 256 / 48000);
    auto next = clock::now();
    std::unique_lock lock(mutex_);
    while (!quit_) {
      next += period;
      if (clock::now() - next > std::chrono::milliseconds(100)) next = clock::now();  // after a stall: no burst
      cv_.wait_until(lock, next, [this] { return quit_; });
      if (quit_ || !queued_) continue;
      --queued_;
      lock.unlock();
      semaphore_->Release(1, nullptr);
      lock.lock();
    }
  }

  rex::thread::Semaphore* semaphore_;
  std::mutex mutex_;
  std::condition_variable cv_;
  uint32_t queued_ = 0;
  bool quit_ = false;
  std::thread thread_;  // last: starts once the members above exist
};

}  // namespace

std::unique_ptr<AudioSystem> SDLAudioSystem::Create(
    runtime::FunctionDispatcher* function_dispatcher) {
  return std::make_unique<SDLAudioSystem>(function_dispatcher);
}

SDLAudioSystem::SDLAudioSystem(runtime::FunctionDispatcher* function_dispatcher)
    : AudioSystem(function_dispatcher) {}

SDLAudioSystem::~SDLAudioSystem() {
  // Runtime::Shutdown on the UI thread (ReXApp::OnDestroy), after every driver has quit its own reference.
  if (sdl_audio_started_) {
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
    sdl_audio_started_ = false;
  }
}

X_STATUS SDLAudioSystem::Setup(system::KernelState* kernel_state) {
  // Runtime::Setup, called from ReXApp::OnInitialize / the resume callback on the UI thread, before the guest runs.
  // (Initialize() runs later on the Audio Worker thread, concurrently with the UI thread's event pumps.)
  if (REXCVAR_GET(audio_sdl_init_on_ui_thread)) {
    // The same hints the driver sets; they must be in place before the first init.
    SDL_SetHint(SDL_HINT_AUDIO_CATEGORY, "playback");
    SDL_SetAppMetadataProperty(SDL_PROP_APP_METADATA_NAME_STRING, "rexglue");
    if (SDL_InitSubSystem(SDL_INIT_AUDIO)) {
      sdl_audio_started_ = true;
      REXAPU_INFO("SDL audio subsystem started at runtime setup (driver: {})",
                  SDL_GetCurrentAudioDriver() ? SDL_GetCurrentAudioDriver() : "?");
    } else {
      // Not fatal: the driver tries again on its own thread (the old behaviour) and reports any failure.
      REXAPU_WARN("SDL_InitSubSystem(SDL_INIT_AUDIO) at setup failed: {}", SDL_GetError());
    }
  }
  return AudioSystem::Setup(kernel_state);
}

void SDLAudioSystem::Initialize() {
  AudioSystem::Initialize();
}

X_STATUS SDLAudioSystem::CreateDriver([[maybe_unused]] size_t index,
                                      rex::thread::Semaphore* semaphore, AudioDriver** out_driver) {
  assert_not_null(out_driver);
  auto driver = new SDLAudioDriver(memory_, semaphore);
  if (!driver->Initialize()) {
    driver->Shutdown();
    delete driver;
    // wwe2k14 patch (Session 76): no device: run silent instead of failing the game's registration (above).
    REXAPU_WARN("no audio device could be opened (client {}): the game runs without sound", index);
    *out_driver = new SilentAudioDriver(memory_, semaphore);
    return X_STATUS_SUCCESS;
  }

  *out_driver = driver;
  return X_STATUS_SUCCESS;
}

void SDLAudioSystem::DestroyDriver(AudioDriver* driver) {
  assert_not_null(driver);
  if (auto sdldriver = dynamic_cast<SDLAudioDriver*>(driver)) {
    sdldriver->Shutdown();
    delete sdldriver;
    return;
  }
  delete driver;  // wwe2k14 patch (Session 76): the SilentAudioDriver
}

}  // namespace rex::audio::sdl
