/**
 * @file        ui/windowed_app_context_sdl.cpp
 * @brief       SDL3 implementation of the windowed app UI loop context
 *
 * @copyright   Copyright (c) 2026 Tom Clay <tomc@tctechstuff.com>
 *              All rights reserved.
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#include <map>
#include <chrono>
#include <cstdlib>
#include <string>
#include <fmt/format.h>
#include <rex/ui/windowed_app_context_sdl.h>

#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include <SDL3/SDL.h>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/ui/flags.h>
#include <rex/ui/window_sdl.h>

// wwe2k14 Session 95 (piece LIFE, a test switch): the game sets this to "background" / "foreground" on the UI thread, and SDL's
// real app-state event goes through WatchEvent as it does on a phone (src/android_lifecycle.h in the game). A cvar and not a new
// exported function, so a game exe linked against an older SDK still links.
REXCVAR_DEFINE_STRING(wwe2k14_test_app_event, "", "wwe2k14",
                      "Session 95 (test only, set by --wwe2k14_test_background_at): background / foreground = send SDL's "
                      "app-state event through the event watcher, as Android does");

namespace rex::ui {

namespace {
// wwe2k14 Session 95: SDL never queues DID_ENTER_BACKGROUND / DID_ENTER_FOREGROUND, it only calls the event watchers with them
// (SDL_SendAppEvent, SDL_events.c), and it does so holding the watcher lock. WatchEvent turns each into this custom event
// (user.code: 1 = foreground); ProcessEvent handles it on the UI thread. File-static: windowed_app_context_sdl.h is in the
// game exe. SDL's Android pump goes on to block at once on a pause (GetLifecycleEventTimeout reads Android_Paused), so on a phone
// both events normally reach ProcessEvent after the return, background first.
uint32_t g_app_state_event_type = 0;
}  // namespace

SDLWindowedAppContext::~SDLWindowedAppContext() {
  if (event_watch_registered_) {
    SDL_RemoveEventWatch(WatchEvent, this);
  }
  // Execute leftover pending functions before the loop machinery goes away,
  // mirroring the shutdown contract documented in WindowedAppContext.
  ExecutePendingFunctionsFromUIThread();
  if (SDL_WasInit(SDL_INIT_VIDEO)) {
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
  }
}

bool SDLWindowedAppContext::Initialize() {
  // Picked before SDL_InitSubSystem, long before a graphics instance can say
  // which surface extensions it has, so the cvar is the escape hatch.
  std::string requested_driver = REXCVAR_GET(video_driver);
#if REX_PLATFORM_MAC
  // macOS presents via a CAMetalLayer surface obtained from the Cocoa driver.
  if (requested_driver.empty()) {
    requested_driver = "cocoa";
  }
#endif
  if (!requested_driver.empty()) {
    SDL_SetHint(SDL_HINT_VIDEO_DRIVER, requested_driver.c_str());
  }
  if (!SDL_InitSubSystem(SDL_INIT_VIDEO)) {
    REXLOG_ERROR("SDL_InitSubSystem(SDL_INIT_VIDEO) failed: {}", SDL_GetError());
    return false;
  }
  const char* video_driver_in_use = SDL_GetCurrentVideoDriver();
  REXLOG_INFO("SDL video driver: {}", video_driver_in_use ? video_driver_in_use : "unknown");
  uint32_t first = SDL_RegisterEvents(3);
  if (first == 0) {
    REXLOG_ERROR("SDL_RegisterEvents failed: {}", SDL_GetError());
    return false;
  }
  wakeup_event_type_ = first;
  paint_event_type_ = first + 1;
  g_app_state_event_type = first + 2;
  rex::cvar::RegisterChangeCallback("wwe2k14_test_app_event", [](std::string_view, std::string_view value) {
    if (value != "background" && value != "foreground") {
      return;
    }
    SDL_Event event{};
    event.type = value == "foreground" ? SDL_EVENT_DID_ENTER_FOREGROUND : SDL_EVENT_DID_ENTER_BACKGROUND;
    SDL_PushEvent(&event);  // the watchers first; ProcessEvent ignores the queued copy
  });
  if (!SDL_AddEventWatch(WatchEvent, this)) {
    REXLOG_ERROR("SDL_AddEventWatch failed: {}", SDL_GetError());
    return false;
  }
  event_watch_registered_ = true;
  return true;
}

void SDLWindowedAppContext::NotifyUILoopOfPendingFunctions() {
  // SDL_PushEvent is thread-safe by SDL contract.
  SDL_Event event{};
  event.type = wakeup_event_type_;
  SDL_PushEvent(&event);
}

void SDLWindowedAppContext::PlatformQuitFromUIThread() {
  // RunMainMessageLoop re-checks HasQuitFromUIThread after every event; a
  // wakeup guarantees SDL_WaitEvent returns promptly if the queue is empty.
  NotifyUILoopOfPendingFunctions();
}

int SDLWindowedAppContext::RunMainMessageLoop() {
  while (!HasQuitFromUIThread()) {
    SDL_Event event;
    if (!SDL_WaitEvent(&event)) {
      REXLOG_ERROR("SDL_WaitEvent failed: {}", SDL_GetError());
      return EXIT_FAILURE;
    }
    {  // WWE13 (diagnostic): UI events per type, every 5 s (WWE13_UI_EVENT_LOG=1)
      static const bool log_on = [] { const char* v = std::getenv("WWE13_UI_EVENT_LOG"); return v && *v == '1'; }();
      if (log_on) {
        static std::map<uint32_t, uint64_t> counts;
        static auto last = std::chrono::steady_clock::now();
        ++counts[event.type];
        const auto now = std::chrono::steady_clock::now();
        if (now - last > std::chrono::seconds(5)) {
          std::string line;
          for (auto& [type, n] : counts) line += fmt::format(" 0x{:X}={}", type, n);
          REXLOG_INFO("[wwe13-ui-events] 5 s:{} (wakeup 0x{:X}, paint 0x{:X})", line, wakeup_event_type_, paint_event_type_);
          counts.clear();
          last = now;
        }
      }
    }
    ProcessEvent(event);
  }
  return EXIT_SUCCESS;
}

void SDLWindowedAppContext::ProcessEvent(SDL_Event& event) {
  if (event.type == wakeup_event_type_) {
    ExecutePendingFunctionsFromUIThread();
    return;
  }
  if (event.type == g_app_state_event_type) {
    // wwe2k14 Session 71 / 95: the app's window surface goes away in the background
    // (WindowSDL::HandleAndroidSurfaceEvent). Foreground also works alone: OnSurfaceChanged(true) detaches first.
    // A newer background event already queued: the surface it reports is gone (or going), so attaching to the
    // window pointer SDL still holds could touch a freed window. Skip this foreground; the later one rebuilds.
    if (event.user.code != 0) {
      SDL_Event next{};
      if (SDL_PeepEvents(&next, 1, SDL_PEEKEVENT, g_app_state_event_type, g_app_state_event_type) > 0 &&
          next.user.code == 0) {
        REXLOG_INFO("app state: foreground skipped, a background event is queued behind it");
        return;
      }
    }
    std::vector<WindowSDL*> windows;
    for (const auto& [id, window] : windows_) windows.push_back(window);
    for (WindowSDL* window : windows) {
      window->HandleAndroidSurfaceEvent(event.user.code != 0);
    }
    return;
  }
  if (event.type == paint_event_type_) {
    // Cocoa may enqueue its quit request behind an already queued paint. Once
    // termination has been requested, CAMetalLayer may stop supplying
    // drawables, so entering the paint first can block forever in
    // -[CAMetalLayer nextDrawable] and prevent the quit event from being
    // processed. Give an already queued quit request priority over rendering.
    // Pump explicitly because a continuous stream of custom paint events can
    // otherwise keep SDL_WaitEvent from returning to Cocoa to collect the
    // application-menu quit request.
    SDL_PumpEvents();
    SDL_Event quit_event{};
    if (SDL_PeepEvents(&quit_event, 1, SDL_GETEVENT, SDL_EVENT_QUIT, SDL_EVENT_QUIT) > 0) {
      ProcessEvent(quit_event);
      if (HasQuitFromUIThread()) {
        return;
      }
    }
    if (WindowSDL* window = GetWindow(event.user.windowID)) {
      window->HandlePaintEvent();
    }
    return;
  }
  if (event.type >= SDL_EVENT_WINDOW_FIRST && event.type <= SDL_EVENT_WINDOW_LAST) {
    if (WindowSDL* window = GetWindow(event.window.windowID)) {
      window->HandleWindowEvent(event);
    }
    return;
  }
  switch (event.type) {
    case SDL_EVENT_QUIT:
      if (synchronously_handled_quit_events_ != 0) {
        --synchronously_handled_quit_events_;
      } else {
        ProcessQuitRequest();
      }
      break;
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP: {
      if (WindowSDL* window = GetWindow(event.key.windowID)) {
        window->HandleKeyEvent(event);
      }
      break;
    }
    case SDL_EVENT_TEXT_INPUT: {
      if (WindowSDL* window = GetWindow(event.text.windowID)) {
        window->HandleTextInputEvent(event);
      }
      break;
    }
    case SDL_EVENT_MOUSE_MOTION: {
      if (WindowSDL* window = GetWindow(event.motion.windowID)) {
        window->HandleMouseEvent(event);
      }
      break;
    }
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP: {
      if (WindowSDL* window = GetWindow(event.button.windowID)) {
        window->HandleMouseEvent(event);
      }
      break;
    }
    case SDL_EVENT_MOUSE_WHEEL: {
      if (WindowSDL* window = GetWindow(event.wheel.windowID)) {
        window->HandleMouseEvent(event);
      }
      break;
    }
    case SDL_EVENT_DROP_FILE: {
      if (WindowSDL* window = GetWindow(event.drop.windowID)) {
        window->HandleDropEvent(event);
      }
      break;
    }
    default:
      break;
  }
}

bool SDLCALL SDLWindowedAppContext::WatchEvent(void* userdata, SDL_Event* event) {
  auto* context = static_cast<SDLWindowedAppContext*>(userdata);
  if (event->type == SDL_EVENT_DID_ENTER_BACKGROUND || event->type == SDL_EVENT_DID_ENTER_FOREGROUND) {
    // wwe2k14 Session 95: this runs inside SDL's watcher lock, so nothing here may block. SDL_PeepEvents, not SDL_PushEvent:
    // no second pass through the watchers. It wakes a waiting SDL_WaitEvent as a push does (SDL_PeepEventsInternal).
    SDL_Event app_state{};
    app_state.type = g_app_state_event_type;
    app_state.user.code = event->type == SDL_EVENT_DID_ENTER_FOREGROUND ? 1 : 0;
    SDL_PeepEvents(&app_state, 1, SDL_ADDEVENT, 0, 0);
    return true;
  }
  if (event->type == SDL_EVENT_QUIT && SDL_IsMainThread() && context->IsInUIThread()) {
    // Cocoa stops making Metal drawables available as part of its termination
    // request. Handle the request synchronously while SDL is queueing it,
    // before rendering can enter another blocking nextDrawable call.
    ++context->synchronously_handled_quit_events_;
    context->ProcessQuitRequest();
  }
  return true;
}

void SDLWindowedAppContext::ProcessQuitRequest() {
  // Use the normal close-request path rather than terminating the message loop
  // directly. Window listeners use OnClosing to stop guest, audio and GPU
  // threads; bypassing it leaves the process hanging during teardown.
  std::vector<SDL_WindowID> window_ids;
  window_ids.reserve(windows_.size());
  for (const auto& [id, window] : windows_) {
    (void)window;
    window_ids.push_back(id);
  }
  for (SDL_WindowID id : window_ids) {
    if (WindowSDL* window = GetWindow(id)) {
      SDL_Event close_event{};
      close_event.type = SDL_EVENT_WINDOW_CLOSE_REQUESTED;
      close_event.window.windowID = id;
      window->HandleWindowEvent(close_event);
    }
  }
}

}  // namespace rex::ui
