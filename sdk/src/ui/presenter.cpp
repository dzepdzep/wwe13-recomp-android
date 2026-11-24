/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>

#include <rex/assert.h>
#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/perf/timeline.h>  // wwe2k14 patch (Session 19, LATENCY)
#include <rex/platform.h>
#include <rex/ui/presenter.h>
#include <rex/ui/window.h>

#if defined(REX_HAS_FIDELITYFX_RUNTIME) && REX_HAS_FIDELITYFX_RUNTIME
#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_upscale.h>
#endif

REXCVAR_DEFINE_BOOL(host_present_from_non_ui_thread, true, "UI/Presenter",
                    "Allow presentation from non-UI thread");

// wwe2k14 patch (Session 21, ONCE)
REXCVAR_DEFINE_BOOL(present_skip_repeats, REX_PLATFORM_ANDROID, "UI/Presenter",
                    "WWE13: no repaint of a guest frame already on screen (one repeat every 250 ms)");
REXCVAR_DEFINE_BOOL(present_guest_frames_once, false, "UI/Presenter",
                    "wwe2k14 (Session 21): when the UI thread presents (an overlay is registered), present only "
                    "when a new guest frame is ready (no repeats at UI ticks) and right away (a new frame wakes the "
                    "UI tick wait) - one present per guest frame. [s21][ONCE] log about every 5 s");
// wwe2k14 patch (Session 37): with present_guest_frames_once, a UI paint with no new guest frame re-requested the
// next paint at once, without the UI tick wait the present path has: the UI thread looped flat out between guest
// frames (~99% of a core in a match, run 2622; profile run 2626: SDL_PeepEvents / PumpEvents / PaintFromUIThread).
REXCVAR_DEFINE_BOOL(present_ui_idle_wait, false, "UI/Presenter",
                    "wwe2k14 (Session 37): with present_guest_frames_once, a UI paint that has no new guest frame "
                    "waits for the next UI tick (a new guest frame still wakes it at once) instead of re-requesting "
                    "the next paint immediately (the UI thread spun between guest frames)");
// wwe2k14 patch (Session 95, present_every_frame): see PaintGate below.
REXCVAR_DEFINE_BOOL(present_every_frame, false, "UI/Presenter",
                    "Session 95: every guest frame reaches a present. Under GPU load the paint ran two frames late and "
                    "the mailbox dropped the frame in between (Surface runs 119, 128, 130: 69-94% of swaps presented, in "
                    "long runs of every second frame). A frame is published only once the paint has taken the one before "
                    "it, so the mailbox never replaces an unpainted frame. Needs present_guest_frames_once and the Vulkan backend (the D3D12 paint does not report its frames: every 31st frame would time out). Logs [pace] "
                    "present_every_frame lines");
REXCVAR_DEFINE_INT32(present_every_frame_min_us, 4000, "UI/Presenter",
                     "Session 95: present_every_frame does not wait while frames come closer together than this, on average "
                     "(loads and fast-forwarded runs: more frames than the screen can show). The time waited at the publish is left out of the interval");
REXCVAR_DEFINE_INT32(present_every_frame_timeout_ms, 40, "UI/Presenter",
                     "Session 95: present_every_frame's longest wait for a paint; after a timeout the next 30 frames are "
                     "not waited for (a minimised window, a held paint)");
REXCVAR_DEFINE_INT32(present_paint_test_delay_us, 0, "UI/Presenter",
                     "Session 95, test only: every paint sleeps this long after it took its guest frame, as the Surface's "
                     "paint does under GPU load, so present_every_frame can be checked on a PC. 0 = off");
REXCVAR_DEFINE_INT32(present_ui_idle_wait_ms, 2, "UI/Presenter",
                     "wwe2k14 (Session 95): the longest present_ui_idle_wait waits for a new guest frame before the UI "
                     "thread looks at window events again (a new guest frame wakes it at once, so this is only the "
                     "idle poll: 2 = about 500 wakes a second, 8 = about 125); 1 to 50");
REXCVAR_DEFINE_INT32(present_ui_tick_fast_sleep_ms, 4, "UI/Presenter",
                     "wwe2k14 (Session 60): when the UI tick thread's WaitForVBlank returns in under 1 ms (a monitor "
                     "that is off or asleep: run 4075, ~96% of a core in the tick thread and the UI thread it wakes), "
                     "sleep this many ms instead of calling it again at once; 0 = off (the old busy loop)");

REXCVAR_DEFINE_BOOL(present_letterbox, true, "UI/Presenter",
                    "Enable letterboxing for non-native aspect ratios");

REXCVAR_DEFINE_INT32(present_safe_area_x, 90, "UI/Presenter",
                     "Horizontal safe area percentage (0-100)")
    .range(0, 100);

REXCVAR_DEFINE_INT32(present_safe_area_y, 90, "UI/Presenter",
                     "Vertical safe area percentage (0-100)")
    .range(0, 100);

#if defined(REX_HAS_FIDELITYFX_SDK)
REXCVAR_DEFINE_STRING(present_effect, "bilinear", "UI/Presenter",
                      "Guest output effect: bilinear, cas, fsr, fsr2, fsr3, sgsr (Snapdragon GSR 1, Vulkan)")
    .allowed({"bilinear", "cas", "fsr", "fsr2", "fsr3", "sgsr"})
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_DOUBLE(present_cas_additional_sharpness,
                      rex::ui::Presenter::GuestOutputPaintConfig::kCasAdditionalSharpnessDefault,
                      "UI/Presenter", "Additional CAS sharpness in [0, 1]")
    .range(rex::ui::Presenter::GuestOutputPaintConfig::kCasAdditionalSharpnessMin,
           rex::ui::Presenter::GuestOutputPaintConfig::kCasAdditionalSharpnessMax)
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_INT32(present_fsr_max_upsampling_passes,
                     rex::ui::Presenter::GuestOutputPaintConfig::kFsrMaxUpscalingPassesMax,
                     "UI/Presenter", "Maximum chained FSR EASU passes")
    .range(1, int32_t(rex::ui::Presenter::GuestOutputPaintConfig::kFsrMaxUpscalingPassesMax))
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_DOUBLE(present_fsr_sharpness_reduction,
                      rex::ui::Presenter::GuestOutputPaintConfig::kFsrSharpnessReductionDefault,
                      "UI/Presenter", "FSR RCAS sharpness reduction in stops")
    .range(rex::ui::Presenter::GuestOutputPaintConfig::kFsrSharpnessReductionMin,
           rex::ui::Presenter::GuestOutputPaintConfig::kFsrSharpnessReductionMax)
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_STRING(
    present_fsr_quality_mode, "auto", "UI/Presenter",
    "Temporal FSR quality mode: auto, nativeaa, quality, balanced, performance, ultra_performance")
    .allowed({"auto", "nativeaa", "quality", "balanced", "performance", "ultra_performance"})
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);
#else
REXCVAR_DEFINE_STRING(present_effect, "bilinear", "UI/Presenter",
                      "Guest output effect: bilinear, sgsr (Snapdragon GSR 1, Vulkan)")
    .allowed({"bilinear", "sgsr"})
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);
#endif

// wwe2k14 patch (Session 73, SGSR): the reference shader's EdgeSharpness (sgsr/v1/README.md: default 2.0, any
// value in [1.0, 2.0]).
REXCVAR_DEFINE_DOUBLE(present_sgsr_edge_sharpness,
                      rex::ui::Presenter::GuestOutputPaintConfig::kSgsrEdgeSharpnessDefault, "UI/Presenter",
                      "present_effect=sgsr: Snapdragon GSR edge sharpness in [1, 2]")
    .range(rex::ui::Presenter::GuestOutputPaintConfig::kSgsrEdgeSharpnessMin,
           rex::ui::Presenter::GuestOutputPaintConfig::kSgsrEdgeSharpnessMax)
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_BOOL(present_dither, false, "UI/Presenter",
                    "Enable output dithering in the final present pass")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_BOOL(present_allow_overscan_cutoff, false, "UI/Presenter",
                    "Allow overscan cutoff based on safe area settings")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

namespace {
using GuestOutputPaintConfig = rex::ui::Presenter::GuestOutputPaintConfig;

GuestOutputPaintConfig::Effect ParsePresentEffect(const std::string& effect_name) {
  std::string lowered = effect_name;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char c) { return char(std::tolower(c)); });
  if (lowered == "sgsr") {
    return GuestOutputPaintConfig::Effect::kSgsr;
  }
#if defined(REX_HAS_FIDELITYFX_SDK)
  if (lowered == "cas") {
    return GuestOutputPaintConfig::Effect::kCas;
  }
  if (lowered == "fsr") {
    return GuestOutputPaintConfig::Effect::kFsr;
  }
#if defined(REX_HAS_FIDELITYFX_RUNTIME) && REX_HAS_FIDELITYFX_RUNTIME
  if (lowered == "fsr2") {
    return GuestOutputPaintConfig::Effect::kFsr2;
  }
  if (lowered == "fsr3") {
    return GuestOutputPaintConfig::Effect::kFsr3;
  }
#else
  // wwe2k14 patch (Session 27, review): without the FidelityFX runtime, FSR 2 / 3 would run as one spatial EASU pass
  // (worse than the chained passes of "fsr"): use FSR 1.
  if (lowered == "fsr2" || lowered == "fsr3") {
    return GuestOutputPaintConfig::Effect::kFsr;
  }
#endif
#endif
  return GuestOutputPaintConfig::Effect::kBilinear;
}

#if defined(REX_HAS_FIDELITYFX_SDK)
bool IsTemporalFsrCompatibilityEffect(GuestOutputPaintConfig::Effect effect) {
  return effect == GuestOutputPaintConfig::Effect::kFsr2 ||
         effect == GuestOutputPaintConfig::Effect::kFsr3;
}

GuestOutputPaintConfig::FsrQualityMode ParsePresentFsrQualityMode(const std::string& mode_name) {
  std::string lowered = mode_name;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
    c = static_cast<unsigned char>(std::tolower(c));
    return c == '-' ? '_' : char(c);
  });
  if (lowered == "nativeaa" || lowered == "native_aa" || lowered == "native") {
    return GuestOutputPaintConfig::FsrQualityMode::kNativeAa;
  }
  if (lowered == "quality") {
    return GuestOutputPaintConfig::FsrQualityMode::kQuality;
  }
  if (lowered == "balanced") {
    return GuestOutputPaintConfig::FsrQualityMode::kBalanced;
  }
  if (lowered == "performance") {
    return GuestOutputPaintConfig::FsrQualityMode::kPerformance;
  }
  if (lowered == "ultra_performance" || lowered == "ultra") {
    return GuestOutputPaintConfig::FsrQualityMode::kUltraPerformance;
  }
  return GuestOutputPaintConfig::FsrQualityMode::kAuto;
}

bool GetFsrQualityModeApiValue(GuestOutputPaintConfig::FsrQualityMode mode,
                               uint32_t& api_mode_out) {
  switch (mode) {
    case GuestOutputPaintConfig::FsrQualityMode::kAuto:
      return false;
    case GuestOutputPaintConfig::FsrQualityMode::kNativeAa:
#if defined(REX_HAS_FIDELITYFX_RUNTIME) && REX_HAS_FIDELITYFX_RUNTIME
      api_mode_out = FFX_UPSCALE_QUALITY_MODE_NATIVEAA;
#else
      api_mode_out = 0;
#endif
      return true;
    case GuestOutputPaintConfig::FsrQualityMode::kQuality:
#if defined(REX_HAS_FIDELITYFX_RUNTIME) && REX_HAS_FIDELITYFX_RUNTIME
      api_mode_out = FFX_UPSCALE_QUALITY_MODE_QUALITY;
#else
      api_mode_out = 1;
#endif
      return true;
    case GuestOutputPaintConfig::FsrQualityMode::kBalanced:
#if defined(REX_HAS_FIDELITYFX_RUNTIME) && REX_HAS_FIDELITYFX_RUNTIME
      api_mode_out = FFX_UPSCALE_QUALITY_MODE_BALANCED;
#else
      api_mode_out = 2;
#endif
      return true;
    case GuestOutputPaintConfig::FsrQualityMode::kPerformance:
#if defined(REX_HAS_FIDELITYFX_RUNTIME) && REX_HAS_FIDELITYFX_RUNTIME
      api_mode_out = FFX_UPSCALE_QUALITY_MODE_PERFORMANCE;
#else
      api_mode_out = 3;
#endif
      return true;
    case GuestOutputPaintConfig::FsrQualityMode::kUltraPerformance:
#if defined(REX_HAS_FIDELITYFX_RUNTIME) && REX_HAS_FIDELITYFX_RUNTIME
      api_mode_out = FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE;
#else
      api_mode_out = 4;
#endif
      return true;
  }
  return false;
}

float GetFsrQualityModeRatioFallback(GuestOutputPaintConfig::FsrQualityMode mode) {
  switch (mode) {
    case GuestOutputPaintConfig::FsrQualityMode::kNativeAa:
      return 1.0f;
    case GuestOutputPaintConfig::FsrQualityMode::kQuality:
      return 1.5f;
    case GuestOutputPaintConfig::FsrQualityMode::kBalanced:
      return 1.7f;
    case GuestOutputPaintConfig::FsrQualityMode::kPerformance:
      return 2.0f;
    case GuestOutputPaintConfig::FsrQualityMode::kUltraPerformance:
      return 3.0f;
    case GuestOutputPaintConfig::FsrQualityMode::kAuto:
    default:
      return 1.0f;
  }
}

bool QueryTemporalFsrRenderResolutionFromQualityMode(uint32_t display_width,
                                                     uint32_t display_height,
                                                     GuestOutputPaintConfig::FsrQualityMode mode,
                                                     uint32_t& render_width_out,
                                                     uint32_t& render_height_out) {
  uint32_t api_mode = 0;
  if (!GetFsrQualityModeApiValue(mode, api_mode)) {
    return false;
  }

#if defined(REX_HAS_FIDELITYFX_RUNTIME) && REX_HAS_FIDELITYFX_RUNTIME
  ffxQueryDescUpscaleGetRenderResolutionFromQualityMode query_desc = {};
  query_desc.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE;
  query_desc.header.pNext = nullptr;
  query_desc.displayWidth = display_width;
  query_desc.displayHeight = display_height;
  query_desc.qualityMode = api_mode;
  query_desc.pOutRenderWidth = &render_width_out;
  query_desc.pOutRenderHeight = &render_height_out;
  if (ffxQuery(nullptr, &query_desc.header) == FFX_API_RETURN_OK && render_width_out &&
      render_height_out) {
    return true;
  }
#endif

  float ratio = GetFsrQualityModeRatioFallback(mode);
  render_width_out = std::max(uint32_t(1), uint32_t(float(display_width) / ratio + 0.5f));
  render_height_out = std::max(uint32_t(1), uint32_t(float(display_height) / ratio + 0.5f));
  return true;
}

void LogTemporalFsrCompatibilityPathOnce() {
  static std::atomic<bool> temporal_fsr_compatibility_logged = false;
  if (!temporal_fsr_compatibility_logged.exchange(true)) {
    REXLOG_WARN(
        "present_effect=fsr2/fsr3 uses an experimental temporal upscaler path "
        "with synthesized depth/motion inputs and may fall back to spatial FSR");
  }
}

void LogTemporalFsrQualityModeInputLimitOnce() {
  static std::atomic<bool> temporal_fsr_quality_mode_input_limit_logged = false;
  if (!temporal_fsr_quality_mode_input_limit_logged.exchange(true)) {
    REXLOG_WARN(
        "present_fsr_quality_mode requested a render size larger than the "
        "guest output; using guest output size");
  }
}
#endif  // defined(REX_HAS_FIDELITYFX_SDK)

GuestOutputPaintConfig BuildGuestOutputPaintConfigFromCVar() {
  GuestOutputPaintConfig config;
  GuestOutputPaintConfig::Effect parsed_effect = ParsePresentEffect(REXCVAR_GET(present_effect));
#if defined(REX_HAS_FIDELITYFX_SDK)
  if (IsTemporalFsrCompatibilityEffect(parsed_effect)) {
    LogTemporalFsrCompatibilityPathOnce();
  }
#endif
  config.SetAllowOverscanCutoff(REXCVAR_GET(present_allow_overscan_cutoff));
  config.SetEffect(parsed_effect);
  config.SetSgsrEdgeSharpness(float(REXCVAR_GET(present_sgsr_edge_sharpness)));
#if defined(REX_HAS_FIDELITYFX_SDK)
  config.SetCasAdditionalSharpness(float(REXCVAR_GET(present_cas_additional_sharpness)));
  config.SetFsrMaxUpsamplingPasses(
      uint32_t(std::max(int32_t(1), REXCVAR_GET(present_fsr_max_upsampling_passes))));
  config.SetFsrSharpnessReduction(float(REXCVAR_GET(present_fsr_sharpness_reduction)));
  config.SetFsrQualityMode(ParsePresentFsrQualityMode(REXCVAR_GET(present_fsr_quality_mode)));
#endif
  config.SetDither(REXCVAR_GET(present_dither));
  return config;
}

}  // namespace

namespace rex {
namespace ui {

void Presenter::FatalErrorHostGpuLossCallback([[maybe_unused]] bool is_responsible,
                                              [[maybe_unused]] bool statically_from_ui_thread) {
  rex::FatalError("Graphics device lost (probably due to an internal error)");
}

// wwe2k14 patch (Session 92, M5c): the held publish state. File static (the header is included by the game exe: no data members).
// Command thread only.
namespace {
struct HoldState {
  bool requested = false;
  Presenter* held_presenter = nullptr;
  Presenter::HoldStats stats;
};
HoldState g_hold;

// wwe2k14 patch (Session 94, gpu_d3d_decouple phase 1B): the two-step present - see Presenter::DecoupleStep. The delayed image waits as the writable one; a timer thread (or the next refresh)
// publishes it. All under g_dec.mutex.
struct DecoupleState {
  std::mutex mutex;
  std::condition_variable cv;
  std::thread thread;
  bool stop = false;
  int step = -1;
  Presenter* pending = nullptr;
  std::chrono::steady_clock::time_point target;
  std::chrono::steady_clock::time_point last_step0;
  bool have_last_step0 = false;
  uint64_t tick_us = 16667;  // the measured time between two step-0 publishes (smoothed)
  bool publishing = false;  // the timer thread is inside a publish (outside the mutex): the command thread waits for it before it touches the mailbox
  std::condition_variable idle_cv;
  Presenter::DecoupleStats stats;
};
// (Leaked on purpose: no static destructor at DLL detach; the timer thread is stopped and joined by ~Presenter.)
DecoupleState& g_dec = *new DecoupleState;

void DecoupleTimerMain() {
  std::unique_lock<std::mutex> lock(g_dec.mutex);
  while (!g_dec.stop) {
    if (!g_dec.pending) {
      g_dec.cv.wait(lock);
      continue;
    }
    if (g_dec.cv.wait_until(lock, g_dec.target) == std::cv_status::timeout || std::chrono::steady_clock::now() >= g_dec.target) {
      Presenter* p = g_dec.pending;
      if (p && std::chrono::steady_clock::now() >= g_dec.target) {
        g_dec.pending = nullptr;
        g_dec.publishing = true;
        ++g_dec.stats.delayed_timer;
        lock.unlock();
        p->PublishGuestOutput(true);  // (outside the mutex)
        lock.lock();
        g_dec.publishing = false;
        g_dec.idle_cv.notify_all();
      }
    }
  }
}
}  // namespace

void Presenter::DecoupleStep(int step) {
  std::lock_guard<std::mutex> lock(g_dec.mutex);
  g_dec.step = step;
}
Presenter::DecoupleStats Presenter::GetDecoupleStats() {
  std::lock_guard<std::mutex> lock(g_dec.mutex);
  return g_dec.stats;
}

Presenter::~Presenter() {
  if (g_hold.held_presenter == this) g_hold.held_presenter = nullptr;  // (M5c) a held frame of a dying presenter is dropped
  {
    std::unique_lock<std::mutex> lock(g_dec.mutex);
    g_dec.idle_cv.wait(lock, [] { return !g_dec.publishing; });
    if (g_dec.pending == this) g_dec.pending = nullptr;  // (Session 94) a delayed frame of a dying presenter is dropped
    if (g_dec.thread.joinable()) {  // the timer thread is stopped here (not at static destruction)
      g_dec.stop = true;
      lock.unlock();
      g_dec.cv.notify_all();
      g_dec.thread.join();
      lock.lock();
      g_dec.stop = false;
    }
  }
  // No intrusive lifetime management must be performed from UI drawers - defer
  // it if needed.
  assert_false(is_executing_ui_drawers_);

#if REX_PLATFORM_WIN32
  if (dxgi_ui_tick_thread_.joinable()) {
    {
      std::scoped_lock<std::mutex> dxgi_ui_tick_lock(dxgi_ui_tick_mutex_);
      dxgi_ui_tick_thread_shutdown_ = true;
    }
    dxgi_ui_tick_control_condition_.notify_all();
    dxgi_ui_tick_thread_.join();
  }
#endif  // XE_PLATFORM

  if (window_) {
    Window* old_window = window_;
    // Null the pointer to prevent an infinite loop between SetPresenter and
    // SetWindowSurfaceFromUIThread calling each other.
    window_ = nullptr;
    old_window->SetPresenter(nullptr);
  }
}

void Presenter::SetWindowSurfaceFromUIThread(Window* new_window, Surface* new_surface) {
  // No intrusive lifetime management must be performed from UI drawers - defer
  // it if needed.
  assert_false(is_executing_ui_drawers_);

  // There can't be a valid surface pointer without a window, as a surface is
  // created and owned by the window.
  assert_false(new_surface && !new_window);

  if (window_ == new_window && (!window_ || surface_ == new_surface)) {
    // Nothing has changed (or a recursive SetWindowSurfaceFromUIThread >
    // SetPresenter > SetWindowSurfaceFromUIThread call).
    return;
  }

  outdated_recovery_ = false;  // wwe2k14 patch (Session 92): must not outlive its surface

  // Disconnect from the current surface.
  if (surface_) {
    // Take ownership of painting, and also stop accepting paint requests from
    // the guest output thread - the window (which is required for making them)
    // may be going away, and there will be a forced paint when the connection
    // becomes available.
    SetPaintModeFromUIThread(PaintMode::kNone);
    DisconnectPaintingFromSurfaceFromUIThread(
        SurfacePaintConnectionState::kUnconnectedRetryAtStateChange);
    surface_ = nullptr;
    UpdateSurfaceMonitorFromUIThread(true);
  }

  if (window_ != new_window) {
    // The window pointer may be accessed by the guest output thread if painting
    // is possible (or was possible, but the paint attempt has resulted in the
    // implementation reporting that the connection has become outdated).
    // However, a painting connection is currently not established at all, so
    // it's safe to modify the window pointer here.

    // Detach from the old window if attaching to a different one or just
    // detaching. SetPresenter for the new window might have been called without
    // it having been called with nullptr for the old window.
    if (window_) {
      Window* old_window = window_;
      // Null the pointer to prevent an infinite loop between SetPresenter and
      // SetWindowSurfaceFromUIThread calling each other.
      window_ = nullptr;
      old_window->SetPresenter(nullptr);
    }

    // Attach to the new one.
    // This function is called from SetPresenter - don't need to notify the
    // window of this, as it itself has triggered this.
    window_ = new_window;
  }

  if (new_surface) {
    assert_true(paint_mode_ == PaintMode::kNone);
    surface_ = new_surface;
    UpdateSurfaceMonitorFromUIThread(true);
    assert_true(surface_paint_connection_state_ ==
                SurfacePaintConnectionState::kUnconnectedRetryAtStateChange);
    bool request_repaint;
    UpdateSurfacePaintConnectionFromUIThread(&request_repaint, true);
    // Request to paint as soon as possible in the UI thread if connected
    // successfully.
    if (request_repaint) {
      RequestPaintOrConnectionRecoveryViaWindow(true);
    }
  }
}

void Presenter::OnSurfaceMonitorUpdateFromUIThread(bool old_monitor_potentially_disconnected) {
  // No intrusive lifetime management must be performed from UI drawers - defer
  // it if needed.
  assert_false(is_executing_ui_drawers_);

  if (!surface_) {
    return;
  }

  UpdateSurfaceMonitorFromUIThread(old_monitor_potentially_disconnected);
  RecoverConnectionFromUIThread();
}

void Presenter::RecoverConnectionFromUIThread() {
  // wwe2k14 patch (Session 92): called on window RESTORED / EXPOSED / DISPLAY_CHANGED. Unlike
  // OnSurfaceResizeFromUIThread, it never rebuilds a healthy (paintable) swapchain; it only reconnects an outdated
  // or unconnected one, and starts the bounded retry loop if that fails.
  assert_false(is_executing_ui_drawers_);
  if (!surface_ || !window_) {
    return;
  }
  if (paint_mode_ == PaintMode::kGuestOutputThreadImmediately) {
    SetPaintModeFromUIThread(PaintMode::kUIThreadOnRequest);
  }
  if (surface_paint_connection_state_ == SurfacePaintConnectionState::kConnectedPaintable) {
    SetPaintModeFromUIThread(GetDesiredPaintModeFromUIThread(true));
    // Re-review (c): the window may come back (restored / back on a monitor) while no guest frame arrives; requests
    // are coalesced, so asking for one paint is cheap.
    RequestPaintOrConnectionRecoveryViaWindow(true);
    return;
  }
  if (surface_paint_connection_state_ ==
      SurfacePaintConnectionState::kUnconnectedSurfaceReportedUnusable) {
    return;
  }
  bool request_repaint = false;
  UpdateSurfacePaintConnectionFromUIThread(&request_repaint, true);
  if (surface_paint_connection_state_ != SurfacePaintConnectionState::kConnectedPaintable) {
    if (!outdated_recovery_) {
      outdated_retry_interval_ms_ = 100;
    }
    outdated_recovery_ = true;
    outdated_retry_ms_ = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::steady_clock::now().time_since_epoch())
                                      .count());
    request_repaint = true;
  }
  if (request_repaint) {
    RequestPaintOrConnectionRecoveryViaWindow(true);
  }
}

void Presenter::OnSurfaceResizeFromUIThread() {
  // No intrusive lifetime management must be performed from UI drawers - defer
  // it if needed.
  assert_false(is_executing_ui_drawers_);

  if (!surface_) {
    return;
  }

  // Let the UI thread take ownership of painting (so the connection can be
  // updated) in a smooth way - downgrade to kUIThreadOnRequest rather than
  // kNone, because a forced repaint may not be necessary if, for example, the
  // size internally turns out to be the same after the update, and in this case
  // the current image may be kept - but the new one must not be missed either
  // if it becomes available during the resize.
  if (paint_mode_ == PaintMode::kGuestOutputThreadImmediately) {
    SetPaintModeFromUIThread(PaintMode::kUIThreadOnRequest);
  }

  bool request_repaint;
  UpdateSurfacePaintConnectionFromUIThread(&request_repaint, true);

  // Request to repaint as soon as possible in the UI thread if needed.
  if (request_repaint) {
    RequestPaintOrConnectionRecoveryViaWindow(true);
  }
}

void Presenter::PaintFromUIThread(bool force_paint) {
  // If there is no surface, this will be a no-op, nothing outdated, nothing to
  // paint. However, an explicit monitor check is needed because UI framerate
  // limiting may be tied to signals from the OS for the monitor - but painting
  // may still occur, for instance, if drawing to a composition surface in the
  // OS (which will still be live even if the window goes outside any monitor).
  // But a surface check still won't cause harm, for simplicity.
  if (!InSurfaceOnMonitorFromUIThread()) {
    // wwe2k14 patch (Session 92): consume the request, or every later
    // RequestPaintOrConnectionRecoveryViaWindow returns early without pushing a paint. The monitor-update event
    // (OnSurfaceMonitorUpdateFromUIThread) asks again.
    ui_thread_paint_requested_.store(false, std::memory_order_relaxed);
    return;
  }

  // Defer changes to the paint mode as well as window paint requests, and do
  // them in this function so they're consistent with the assumptions made here.
  assert_false(is_in_ui_thread_paint_);
  is_in_ui_thread_paint_ = true;
  request_guest_output_paint_after_current_ui_thread_paint_ = false;
  request_ui_paint_after_current_ui_thread_paint_ = false;

  // Actualize the connection if the UI needs to be drawn if there was some
  // explicit paint request (the guest output has been refreshed, and the guest
  // output thread was asked not to present directly due as the UI needs to be
  // drawn, or some surface state change has happened so the guest output needs
  // to be displayed as soon as possible without waiting for the guest to
  // refresh it, or the guest output thread has been notified that the
  // connection has become outdated and has requested the UI thread to
  // reconnect).
  bool draw_ui = !ui_drawers_.empty();
  bool do_paint = force_paint || draw_ui;
  // Reset ui_thread_paint_requested_ unconditionally also, regardless of
  // whether the UI needs to be drawn - the flag may be set to try reconnecting,
  // for example.
  if (ui_thread_paint_requested_.exchange(false, std::memory_order_relaxed)) {
    do_paint = true;
  }
  PaintResult paint_result = PaintResult::kNotPresented;
  bool request_repaint_at_tick = false;
  bool request_repaint_immediately = false;
  skipped_repeat_ = false;  // WWE13 (present_skip_repeats)
  if (do_paint) {
    // Take ownership of painting if it's currently owned by the guest output
    // thread (downgrade from kGuestOutputThreadImmediately to
    // kUIThreadOnRequest - not to kNone so if during this paint a new guest
    // output frame is generated, the notification will still be sent to the UI
    // thread rather than dropped, so the frame won't be skipped). This is
    // needed to be able not only to paint, but also to try to recover from an
    // outdated surface.
    if (paint_mode_ == PaintMode::kGuestOutputThreadImmediately) {
      SetPaintModeFromUIThread(PaintMode::kUIThreadOnRequest);
    }
    // Try to recover from the connection becoming outdated in the previous
    // paint.
    if (surface_paint_connection_state_ == SurfacePaintConnectionState::kConnectedOutdated) {
      outdated_recovery_ = true;
      outdated_retry_interval_ms_ = 100;
      UpdateSurfacePaintConnectionFromUIThread(nullptr, false);
      outdated_retry_ms_ = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
      if (surface_paint_connection_state_ ==
          SurfacePaintConnectionState::kUnconnectedRetryAtStateChange) {
        REXLOG_INFO("Presenter: reconnect after an outdated swapchain failed, will retry (100 ms, backing off to 1 s)");
      } else if (surface_paint_connection_state_ != SurfacePaintConnectionState::kConnectedPaintable) {
        REXLOG_INFO("Presenter: reconnect after an outdated swapchain failed, the surface reported unusable (no retry)");
      }
    } else if (outdated_recovery_ &&
               surface_paint_connection_state_ ==
                   SurfacePaintConnectionState::kUnconnectedRetryAtStateChange) {
      // wwe2k14 patch (Session 92): the reconnect failed (zero-area surface, minimised, ...) and only a changed
      // size would retry; retry here at a bounded rate instead.
      const uint64_t retry_now_ms = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                 std::chrono::steady_clock::now().time_since_epoch())
                                                 .count());
      if (retry_now_ms - outdated_retry_ms_ >= outdated_retry_interval_ms_) {
        outdated_retry_ms_ = retry_now_ms;
        bool retry_repaint = false;
        UpdateSurfacePaintConnectionFromUIThread(&retry_repaint, false);
        if (surface_paint_connection_state_ == SurfacePaintConnectionState::kConnectedPaintable) {
          REXLOG_INFO("Presenter: reconnected after an outdated swapchain");
          outdated_retry_interval_ms_ = 100;
          if (retry_repaint) {
            request_repaint_immediately = true;
          }
        } else {
          outdated_retry_interval_ms_ = std::min<uint64_t>(outdated_retry_interval_ms_ * 2, 1000);
        }
      }
    }
    if (surface_paint_connection_state_ == SurfacePaintConnectionState::kConnectedPaintable) {
      outdated_recovery_ = false;
    }
    // If still paintable or recovered successfully, paint.
    if (surface_paint_connection_state_ == SurfacePaintConnectionState::kConnectedPaintable) {
      // The paint mode might have been set to kNone when the connection was
      // marked as outdated last time. Or, if wasn't reconnecting, there was
      // some other incorrect situation that caused the paint mode to be set to
      // kNone for an active connection. Make sure that the current paint mode
      // is consistent with painting from the UI thread.
      SetPaintModeFromUIThread(PaintMode::kUIThreadOnRequest);

      // wwe2k14 patch (Session 21, ONCE): present_guest_frames_once - present a guest frame once, as soon as
      // it is ready; a paint with no new guest frame (a UI tick, or the request of a frame a tick paint
      // already showed) presents nothing. force_paint (surface changes) always paints.
      const bool once = REXCVAR_GET(present_guest_frames_once);
      const bool new_guest_frame = guest_output_ready_seq_.load(std::memory_order_acquire) !=
                                   guest_output_presented_seq_.load(std::memory_order_relaxed);
      bool skip_present = false;
      if (once) {
        ++once_stats_.ui_paints;
        skip_present = !force_paint && !new_guest_frame;
        if (skip_present) {
          ++once_stats_.skipped;
        }
      }
      if (!skip_present) {
        // Limit the frame rate of the UI, usually to the monitor refresh rate,
        // in a way so that the UI won't be stealing all the remaining GPU
        // resources if it's repainted continuously, and the window system itself
        // doesn't limit the frame rate.
        if (once && new_guest_frame) {
          ++once_stats_.tick_waits_skipped;  // a new guest frame: present now
        } else {
          WaitForUITickFromUIThread();
        }

        paint_result = PaintAndPresent(draw_ui);
        if (paint_result == PaintResult::kNotPresentedImageLate) {
          // wwe2k14 patch (Session 95, present_acquire_fence): retry shortly; the sleep keeps it from a busy loop (a new
          // guest frame skips the tick wait above, and there is no tick on Android).
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          request_repaint_immediately = true;
        }
      } else if (draw_ui) {
        // wwe2k14 patch (Session 37): wait here until a new guest frame is ready (ForceUIThreadPaintTick sets the
        // force flag and notifies) or 2 ms pass. The UI tick wait itself returned at once here (run 2628: still 5.8
        // million UI paints per 5 s for 300 presents), so this waits on its condition with a timeout instead.
        if (REXCVAR_GET(present_ui_idle_wait)) {
#if REX_PLATFORM_WIN32
          std::unique_lock<std::mutex> idle_lock(dxgi_ui_tick_mutex_);
          if (!dxgi_ui_tick_force_requested_) {
            dxgi_ui_tick_signal_condition_.wait_for(idle_lock,
                                                    std::chrono::milliseconds(std::clamp(REXCVAR_GET(present_ui_idle_wait_ms), 1, 50)),
                                                    [this] { return dxgi_ui_tick_force_requested_; });
          }
          dxgi_ui_tick_force_requested_ = false;
#else
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif
        }
        // Keep the UI ticking (the next tick checks for a new guest frame again).
        request_ui_paint_after_current_ui_thread_paint_ = true;
      }
      if (once) {
        const uint64_t now_ms = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                                             std::chrono::steady_clock::now().time_since_epoch())
                                             .count());
        if (!once_log_last_ms_) {
          once_log_last_ms_ = now_ms;
        } else if (now_ms - once_log_last_ms_ >= 5000) {
          REXLOG_INFO("[s21][ONCE] {:.1f} s: {} UI paints, {} skipped (no new guest frame), {} presented "
                      "without the UI tick wait",
                      (now_ms - once_log_last_ms_) / 1000.0, once_stats_.ui_paints, once_stats_.skipped,
                      once_stats_.tick_waits_skipped);
          once_stats_ = OnceStats();
          once_log_last_ms_ = now_ms;
        }
      }
      if (surface_paint_connection_state_ == SurfacePaintConnectionState::kConnectedOutdated) {
        // Request another PaintFromUIThread which will try to recover from the
        // outdated connection in the next frame (not immediately, so the
        // windowing system has some time to prepare what may be required to
        // recover from it, such as to send a resize event).
        request_repaint_immediately = true;
      }
    }
    // If can't paint anymore, notify the paint mode refresh below (which is not
    // guaranteed to have access to have ownership of painting as it's taken
    // here only conditionally, thus can't know whether the connection is
    // actually in a paintable state).
    if (surface_paint_connection_state_ != SurfacePaintConnectionState::kConnectedPaintable) {
      SetPaintModeFromUIThread(PaintMode::kNone);
      if (outdated_recovery_ &&
          surface_paint_connection_state_ == SurfacePaintConnectionState::kUnconnectedRetryAtStateChange) {
        // wwe2k14 patch (Session 92): keep asking for a paint; the retry above is rate-limited to ~100 ms, and the
        // short sleep keeps the repaint requests from becoming a busy loop.
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        request_repaint_immediately = true;
      } else {
        outdated_recovery_ = false;
      }
    }
  }

  // Transfer the ownership of painting back to the guest output thread if it
  // was taken or if needed for any reason (however, it's taken conditionally -
  // no guarantees that the actual connection state is accessible here, so only
  // checking whether the mode is not kNone currently, not the connection
  // state), and overall synchronize the state taking into account both what has
  // been done in this function and what could have been done by the UI drawer
  // callbacks.
  if (paint_mode_ != PaintMode::kNone) {
    SetPaintModeFromUIThread(GetDesiredPaintModeFromUIThread(true));
  }
  is_in_ui_thread_paint_ = false;

  // Check if the device has been lost. There's no point in requesting repaint
  // if it has happened anyway, it won't be possible to satisfy such request
  // with the current Presenter.
  if (paint_result == PaintResult::kGpuLostExternally ||
      paint_result == PaintResult::kGpuLostResponsible) {
    if (host_gpu_loss_callback_) {
      host_gpu_loss_callback_(paint_result == PaintResult::kGpuLostResponsible, true);
    }
    // The loss callback might have destroyed the presenter, must not do
    // anything with `this` anymore.
    return;
  }

  // Request refresh if needed.
  // Can't check the exact paintability as the connection state may currently
  // be owned by the guest output thread, so check conservatively via
  // paint_mode_.
  if (paint_mode_ != PaintMode::kNone) {
    // Immediately paint the guest output if requested explicitly or if the UI
    // has hidden itself.
    // WWE13: not after a skipped repeat (present_skip_repeats) - that re-requested the paint at once, and the UI thread
    // spun through SDL_WaitEvent / SDL_PumpEvents at 100% of a big core between guest frames.
    if (request_guest_output_paint_after_current_ui_thread_paint_ ||
        (draw_ui && ui_drawers_.empty() && !skipped_repeat_)) {
      request_repaint_immediately = true;
    }
    if (request_ui_paint_after_current_ui_thread_paint_ && !ui_drawers_.empty()) {
      request_repaint_at_tick = true;
    }
  }
  if (request_repaint_at_tick || request_repaint_immediately) {
    RequestPaintOrConnectionRecoveryViaWindow(request_repaint_immediately);
  }
}


// wwe2k14 patch (Session 95, present_every_frame): the paint runs on the UI thread and takes the newest frame of the mailbox;
// a frame replaced before a paint takes it is never presented. Surface timelines (runs 119, 128, 130): with the GPU behind,
// a paint took 30+ ms between taking its frame and returning from the present (not in the acquire, not in the older-paint
// fence, not while holding the queue: the game's submissions went on meanwhile), ending about when the frames on the queue
// finished; by then the command thread had published two frames, so the next paint skipped one, and the pattern held
// itself: frames drawn and dropped in long runs of every second frame. Here the command thread waits at the publish, after
// the frame's submission is queued, until the paint has taken the frame before it. Holding a submission in place of this
// does not do it: the command thread keeps publishing ahead of the queue (PC proof run 003). The state belongs to the one
// presenter that publishes; a job would own this struct.
namespace {
struct PaintGate {
  static constexpr uint32_t kBackoffFrames = 30;
  std::mutex mutex;
  std::condition_variable cv;
  uint64_t published = 0;  // guest output thread: the newest frame published
  uint64_t consumed = 0;   // the newest frame a paint took
  uint32_t backoff = 0;
  bool have_last_publish = false;
  std::chrono::steady_clock::time_point last_publish;
  uint64_t average_us = 0;  // the time between frames, smoothed
  bool logged_on = false;
  struct Stats {
    uint32_t frames = 0, waits = 0, timeouts = 0, skipped_fast = 0, skipped_backoff = 0;
    uint64_t wait_us = 0, wait_max_us = 0;
  } stats;
};
PaintGate g_paint_gate;
}  // namespace

void Presenter::PaintGateBeforePublish(uint64_t refresh_seq) {
  if (!REXCVAR_GET(present_every_frame) || !REXCVAR_GET(present_guest_frames_once)) return;
  PaintGate& g = g_paint_gate;
  std::unique_lock<std::mutex> lock(g.mutex);
  const auto now = std::chrono::steady_clock::now();
  if (g.have_last_publish) {
    const uint64_t since_us = std::min<uint64_t>(
        uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(now - g.last_publish).count()), 100000);
    g.average_us = g.average_us ? (g.average_us * 15 + since_us) / 16 : since_us;
  }
  g.have_last_publish = true;
  if (!g.logged_on) {
    g.logged_on = true;
    REXLOG_INFO("[pace] present_every_frame on: a frame is published once the paint has taken the one before it (up to {} ms; "
                "not while frames come closer than {} us)",
                REXCVAR_GET(present_every_frame_timeout_ms), REXCVAR_GET(present_every_frame_min_us));
  }
  ++g.stats.frames;
  const uint64_t wanted = g.published;
  if (g.backoff) {
    --g.backoff;
    ++g.stats.skipped_backoff;
  } else if (g.average_us < uint64_t(std::max(REXCVAR_GET(present_every_frame_min_us), 0))) {
    ++g.stats.skipped_fast;
  } else if (g.consumed < wanted) {
    perf::timeline::Scope wait_scope("pgate_b", "pgate_e", uint32_t(wanted), uint32_t(refresh_seq));
    const bool ok = g.cv.wait_for(lock, std::chrono::milliseconds(std::max(REXCVAR_GET(present_every_frame_timeout_ms), 1)),
                                  [&g, wanted] { return g.consumed >= wanted; });
    const uint64_t us = uint64_t(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - now).count());
    ++g.stats.waits;
    g.stats.wait_us += us;
    g.stats.wait_max_us = std::max(g.stats.wait_max_us, us);
    if (!ok) {
      ++g.stats.timeouts;
      g.backoff = PaintGate::kBackoffFrames;
    }
  }
  g.published = refresh_seq;
  // The end of this publish: the next interval then leaves out the time waited here (else a paint cycle above the
  // threshold keeps the average above it and the exemption for loads latches shut).
  g.last_publish = std::chrono::steady_clock::now();
  if (g.stats.frames >= 600) {
    REXLOG_INFO("[pace] present_every_frame: 600 frames: {} waits for the paint, {:.2f} ms each, longest {:.2f} ms, {} timed out; "
                "not waited for: {} fast, {} after a timeout",
                g.stats.waits, g.stats.waits ? double(g.stats.wait_us) / 1000.0 / g.stats.waits : 0.0,
                double(g.stats.wait_max_us) / 1000.0, g.stats.timeouts, g.stats.skipped_fast, g.stats.skipped_backoff);
    g.stats = PaintGate::Stats();
  }
}

void Presenter::PaintGateNoteConsumed(uint64_t refresh_seq) {
  if (REXCVAR_GET(present_every_frame)) {
    PaintGate& g = g_paint_gate;
    {
      std::lock_guard<std::mutex> lock(g.mutex);
      g.consumed = std::max(g.consumed, refresh_seq);
    }
    g.cv.notify_all();
  }
  const int32_t delay_us = REXCVAR_GET(present_paint_test_delay_us);
  if (delay_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
}

void Presenter::HoldGuestOutputPublish(bool hold) { g_hold.requested = hold; }
bool Presenter::GuestOutputPublishHeld() { return g_hold.held_presenter != nullptr; }
Presenter::HoldStats Presenter::GetHoldStats() { return g_hold.stats; }

// wwe2k14 patch (Session 93, log only): the host present call in progress (see presenter.h).
namespace {
std::atomic<int64_t> g_host_call_start_ns{0};
std::atomic<const char*> g_host_call_what{nullptr};
int64_t HostCallNowNs() {
  return int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
}  // namespace
void Presenter::NoteHostPresentCall(const char* what) {
  if (what) {
    g_host_call_what.store(what, std::memory_order_relaxed);
    g_host_call_start_ns.store(HostCallNowNs(), std::memory_order_release);
  } else {
    g_host_call_start_ns.store(0, std::memory_order_release);
  }
}
uint64_t Presenter::HostPresentCallMs(const char** what_out) {
  const int64_t start = g_host_call_start_ns.load(std::memory_order_acquire);
  if (what_out) *what_out = g_host_call_what.load(std::memory_order_relaxed);
  if (!start) return 0;
  const int64_t d = HostCallNowNs() - start;
  return d > 0 ? uint64_t(d / 1000000) : 0;
}
bool Presenter::PublishHeldGuestOutput() {
  Presenter* p = g_hold.held_presenter;
  if (!p) return false;
  g_hold.held_presenter = nullptr;
  ++g_hold.stats.published_held;
  p->PublishGuestOutput(true);
  return true;
}

bool Presenter::RefreshGuestOutput(
    uint32_t frontbuffer_width, uint32_t frontbuffer_height, uint32_t display_aspect_ratio_x,
    uint32_t display_aspect_ratio_y,
    std::function<bool(GuestOutputRefreshContext& context)> refresher) {
  // wwe2k14 patch (Session 92, M5c): a frame still held from the previous refresh is published first (one writable image at a time).
  if (g_hold.held_presenter == this) {
    ++g_hold.stats.forced;
    PublishHeldGuestOutput();
  }
  {  // wwe2k14 patch (Session 94, gpu_d3d_decouple phase 1B): a delayed frame still pending is published first (one writable image at a time)
    std::unique_lock<std::mutex> lock(g_dec.mutex);
    g_dec.idle_cv.wait(lock, [] { return !g_dec.publishing; });
    if (g_dec.pending == this) {
      g_dec.pending = nullptr;
      ++g_dec.stats.delayed_forced;
      lock.unlock();
      PublishGuestOutput(true);  // (outside the mutex: the timer cannot take it any more)
    }
  }
  GuestOutputProperties& writable_properties =
      guest_output_properties_[guest_output_mailbox_writable_];
  writable_properties.frontbuffer_width = frontbuffer_width;
  writable_properties.frontbuffer_height = frontbuffer_height;
  writable_properties.display_aspect_ratio_x = display_aspect_ratio_x;
  writable_properties.display_aspect_ratio_y = display_aspect_ratio_y;
  writable_properties.is_8bpc = false;
  bool is_active = writable_properties.IsActive();
  if (is_active) {
    if (!RefreshGuestOutputImpl(guest_output_mailbox_writable_, frontbuffer_width,
                                frontbuffer_height, refresher, writable_properties.is_8bpc)) {
      // If failed to refresh, don't send the currently writable image to the
      // mailbox as it may be in an undefined state. Don't disable the guest
      // output either though because the failure may be something transient.
      return false;
    }
    guest_output_active_last_refresh_ = true;
    // wwe2k14 patch (Session 19, LATENCY): a = refresh sequence number of the image now becoming ready.
    writable_properties.refresh_seq = ++guest_output_refresh_seq_;
    writable_properties_seq_for_once_ = writable_properties.refresh_seq;  // wwe2k14 patch (Session 21, ONCE)
    perf::timeline::Record("gorefresh", uint32_t(writable_properties.refresh_seq));
  } else {
    // Request presenting a blank image if there was a true image previously,
    // but not now.
    if (!guest_output_active_last_refresh_) {
      return false;
    }
    guest_output_active_last_refresh_ = false;
  }

  // wwe2k14 patch (Session 92, M5c): the publish is held (the image stays the writable one until PublishHeldGuestOutput).
  if (is_active && g_hold.requested) {
    g_hold.held_presenter = this;
    ++g_hold.stats.held;
    perf::timeline::Record("gohold", uint32_t(writable_properties.refresh_seq));
    return true;
  }
  if (is_active) {  // wwe2k14 patch (Session 94, gpu_d3d_decouple phase 1B): the two-step present
    std::unique_lock<std::mutex> lock(g_dec.mutex);
    const int step = g_dec.step;
    g_dec.step = -1;
    const auto now = std::chrono::steady_clock::now();
    if (step == 0) {
      if (g_dec.have_last_step0) {
        const uint64_t us = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(now - g_dec.last_step0).count());
        if (us > 2000 && us < 200000) g_dec.tick_us = (g_dec.tick_us * 3 + us) / 4;
        g_dec.stats.last_tick_us = us;
      }
      g_dec.last_step0 = now;
      g_dec.have_last_step0 = true;
      ++g_dec.stats.step0;
    } else if (step == 1) {
      ++g_dec.stats.step1;
      const uint64_t delay_us = std::clamp<uint64_t>(g_dec.tick_us / 2, 2000, 60000);
      g_dec.stats.last_delay_us = delay_us;
      g_dec.pending = this;
      g_dec.target = std::max(now, g_dec.last_step0 + std::chrono::microseconds(delay_us));  // (half a tick after the step-0 publish, not after this refresh)
      if (!g_dec.thread.joinable()) g_dec.thread = std::thread(DecoupleTimerMain);
      lock.unlock();
      g_dec.cv.notify_all();
      return true;
    }
  }
  ++g_hold.stats.published_now;
  PublishGuestOutput(is_active);
  return is_active;
}

void Presenter::PublishGuestOutput(bool is_active) {
  // wwe2k14 patch (Session 95, present_every_frame): only for a paint on the UI thread (the guest output thread's own paint
  // has taken the last frame by now, and with no paint mode nothing will).
  if (is_active && REXCVAR_GET(present_every_frame)) {
    PaintMode paint_mode_now;
    {
      std::lock_guard<std::mutex> paint_mode_mutex_lock(paint_mode_mutex_);
      paint_mode_now = paint_mode_;
    }
    if (paint_mode_now == PaintMode::kUIThreadOnRequest) {
      PaintGateBeforePublish(writable_properties_seq_for_once_);
    }
  }
  // Make the new image the next to present on the host (the "ready" one),
  // replacing the one already specified as the next (dropping it instead of
  // enqueueing the new image after it) to achieve the lowest latency (also,
  // after switching from UI thread painting to doing it in the guest output
  // thread, will immediately recover to having the latest frame always sent to
  // the host present call on the CPU and all frames reaching a present call).
  uint32_t last_acquired_and_ready =
      guest_output_mailbox_acquired_and_ready_.load(std::memory_order_relaxed);
  // Desired acquired = current acquired (changed only by the consumers).
  // Desired ready = current writable.
  // memory_order_acq_rel to acquire the new writable image and to release the
  // current one (to let the consumers take it, from ready to acquired).
  while (!guest_output_mailbox_acquired_and_ready_.compare_exchange_weak(
      last_acquired_and_ready,
      (last_acquired_and_ready & 3) | (guest_output_mailbox_writable_ << 2),
      std::memory_order_acq_rel, std::memory_order_relaxed)) {}
  // Now, it's known that `ready == writable` on the host presentation side.
  // Take the next `writable` with this assumption about its current value in
  // mind.
  uint32_t last_acquired = last_acquired_and_ready & 3;
  if (last_acquired == guest_output_mailbox_writable_) {
    // The new image has already been acquired by the time the compare-exchange
    // loop has finished (acquired == ready == currently writable).
    // It's a valid situation from the ownership perspective, and the semantics
    // of the weak compare-exchange explicitly permit spurious `false` results.
    // (3 - a - b) % 3 cannot be used here, as (3 - a - a) % 3 results in `a` -
    // the same index.
    // Take any free image. Preferably using + 1, not ^ 1, so if the guest needs
    // to await any GPU work referencing the image, it will wait for the frame 3
    // frames ago, not 2, if this happens repeatedly.
    guest_output_mailbox_writable_ = (guest_output_mailbox_writable_ + 1) % 3;
  } else {
    // Take the image other than the last acquired one and the new one,
    // currently not accessible to the host presentation.
    guest_output_mailbox_writable_ = (3 - last_acquired - guest_output_mailbox_writable_) % 3;
  }

  // wwe2k14 patch (Session 21, ONCE): the newest ready image (for present_guest_frames_once).
  if (is_active) {
    guest_output_ready_seq_.store(writable_properties_seq_for_once_, std::memory_order_release);
  }
  // Trigger the presentation on the host.
  PaintResult paint_result = PaintResult::kNotPresented;
  {
    std::lock_guard<std::mutex> paint_mode_mutex_lock(paint_mode_mutex_);
    switch (paint_mode_) {
      case PaintMode::kNone:
        // Neither painting nor window paint requesting is accessible.
        break;
      case PaintMode::kUIThreadOnRequest:
        // Only window paint requesting is accessible.
        RequestPaintOrConnectionRecoveryViaWindow(true);
        break;
      case PaintMode::kGuestOutputThreadImmediately:
        // Both painting and window paint requesting are accessible.
        if (surface_paint_connection_state_ == SurfacePaintConnectionState::kConnectedPaintable) {
          paint_result = PaintAndPresent(false);
          if (surface_paint_connection_state_ == SurfacePaintConnectionState::kConnectedOutdated ||
              paint_result == PaintResult::kNotPresentedImageLate) {  // wwe2k14 patch (Session 95): retry a late image
            RequestPaintOrConnectionRecoveryViaWindow(true);
          }
        }
        break;
    }
  }
  // Handle GPU loss when not in the middle of the function anymore, and
  // lifecycle management from the GPU loss callback is fine on the UI thread.
  if (host_gpu_loss_callback_) {
    if (paint_result == PaintResult::kGpuLostResponsible) {
      host_gpu_loss_callback_(true, false);
    } else if (paint_result == PaintResult::kGpuLostExternally) {
      host_gpu_loss_callback_(false, false);
    }
  }
}

void Presenter::SetGuestOutputPaintConfigFromUIThread(const GuestOutputPaintConfig& new_config) {
  // For simplicity, this may be called externally repeatedly.
  // Lock the mutex only when something has been modified, and also don't
  // request UI thread guest output redraws when not needed.
  bool modified = false;
  bool request_repaint = false;
  if (guest_output_paint_config_.GetEffect() != new_config.GetEffect()) {
    modified = true;
    request_repaint = true;
  }
#if defined(REX_HAS_FIDELITYFX_SDK)
  if (guest_output_paint_config_.GetFsrSharpnessReduction() !=
      new_config.GetFsrSharpnessReduction()) {
    modified = true;
    if (new_config.GetEffect() == GuestOutputPaintConfig::Effect::kFsr ||
        new_config.GetEffect() == GuestOutputPaintConfig::Effect::kFsr2 ||
        new_config.GetEffect() == GuestOutputPaintConfig::Effect::kFsr3) {
      request_repaint = true;
    }
  }
  if (guest_output_paint_config_.GetCasAdditionalSharpness() !=
      new_config.GetCasAdditionalSharpness()) {
    modified = true;
    if (new_config.GetEffect() == GuestOutputPaintConfig::Effect::kCas ||
        new_config.GetEffect() == GuestOutputPaintConfig::Effect::kFsr ||
        new_config.GetEffect() == GuestOutputPaintConfig::Effect::kFsr2 ||
        new_config.GetEffect() == GuestOutputPaintConfig::Effect::kFsr3) {
      request_repaint = true;
    }
  }
#endif
  if (guest_output_paint_config_.GetDither() != new_config.GetDither()) {
    modified = true;
    request_repaint = true;
  }
#if defined(REX_HAS_FIDELITYFX_SDK)
  if (guest_output_paint_config_.GetFsrQualityMode() != new_config.GetFsrQualityMode()) {
    modified = true;
    if (new_config.GetEffect() == GuestOutputPaintConfig::Effect::kFsr2 ||
        new_config.GetEffect() == GuestOutputPaintConfig::Effect::kFsr3) {
      request_repaint = true;
    }
  }
#endif
  if (modified) {
    {
      std::unique_lock<std::mutex> config_lock(guest_output_paint_config_mutex_);
      guest_output_paint_config_ = new_config;
    }
    // Coarsely check the availability of painting and of the window (for
    // calling RequestPaint) via paint_mode_ because the actual painting
    // connection state may currently be owned not by the UI thread.
    if (request_repaint && paint_mode_ != PaintMode::kNone) {
      if (is_in_ui_thread_paint_) {
        // Defer until the end of the current paint if called from, for
        // instance, a UI drawer.
        request_guest_output_paint_after_current_ui_thread_paint_ = true;
      } else {
        RequestPaintOrConnectionRecoveryViaWindow(true);
      }
    }
  }
}

void Presenter::AddUIDrawerFromUIThread(UIDrawer* drawer, size_t z_order) {
  assert_not_null(drawer);
  // Obtain whether the iterator list was empty before erasing in case of
  // replacing with a new entry with a different Z order happens.
  bool drawers_were_empty = ui_drawers_.empty();
  uint64_t drawer_last_draw = UINT64_MAX;
  // Check if already added.
  for (auto it_existing = ui_drawers_.begin(); it_existing != ui_drawers_.end(); ++it_existing) {
    if (it_existing->second.drawer != drawer) {
      continue;
    }
    if (it_existing->first == z_order) {
      return;
    }
    // Keep the same last draw index to prevent the drawer from being executed
    // twice if increasing its Z order during the drawer loop.
    drawer_last_draw = it_existing->second.last_draw;
    // If removing the drawer that is the next in the current drawer loop, skip
    // it (in a multimap, only one element iterator is invalidated).
    if (is_executing_ui_drawers_ && ui_draw_next_iterator_ == it_existing) {
      ++ui_draw_next_iterator_;
    }
    ui_drawers_.erase(it_existing);
    break;
  }
  auto it_new = ui_drawers_.emplace(z_order, UIDrawerReference(drawer, drawer_last_draw));
  // If adding to the Z layer currently being processed (for drawing, from the
  // lowest to the highest), or to layers in between the current and the
  // previously next, make sure the new drawer is executed too.
  if (is_executing_ui_drawers_ && z_order >= ui_draw_current_z_order_ &&
      (ui_draw_next_iterator_ == ui_drawers_.end() || z_order < ui_draw_next_iterator_->first)) {
    ui_draw_next_iterator_ = it_new;
  }
  HandleUIDrawersChangeFromUIThread(drawers_were_empty);
}

void Presenter::RemoveUIDrawerFromUIThread(UIDrawer* drawer) {
  assert_not_null(drawer);
  for (auto it_existing = ui_drawers_.begin(); it_existing != ui_drawers_.end(); ++it_existing) {
    if (it_existing->second.drawer != drawer) {
      continue;
    }
    // If removing the drawer that is the next in the current drawer loop, skip
    // it (in a multimap, only one element iterator is invalidated).
    if (is_executing_ui_drawers_ && ui_draw_next_iterator_ == it_existing) {
      ++ui_draw_next_iterator_;
    }
    ui_drawers_.erase(it_existing);
    HandleUIDrawersChangeFromUIThread(false);
    return;
  }
}

void Presenter::RequestUIPaintFromUIThread() {
  if (is_in_ui_thread_paint_) {
    // The paint request will be done once in the end of PaintFromUIThread
    // according to the actual state at the moment that happens. It's common for
    // drawers to call this (even every frame), and no need to do too many OS
    // paint request calls.
    request_ui_paint_after_current_ui_thread_paint_ = true;
    return;
  }
  // The connection state may be owned by the guest output thread now rather
  // than the UI thread, check whether it's not pointless to make the request
  // coarsely via paint_mode_.
  if (!ui_drawers_.empty() && paint_mode_ != PaintMode::kNone) {
    // The window must be present, otherwise the conditions wouldn't have been
    // met.
    window_->RequestPaint();
  }
}

bool Presenter::InitializeCommonSurfaceIndependent() {
  {
    std::lock_guard<std::mutex> config_lock(guest_output_paint_config_mutex_);
    guest_output_paint_config_ = BuildGuestOutputPaintConfigFromCVar();
  }

  // Initialize UI frame rate limiting.
#if REX_PLATFORM_WIN32
  dxgi_ui_tick_thread_ = std::thread(&Presenter::DXGIUITickThread, this);
#endif  // XE_PLATFORM

  return true;
}

std::unique_lock<std::mutex> Presenter::ConsumeGuestOutput(
    uint32_t& mailbox_index_or_max_if_inactive_out, GuestOutputProperties* properties_out,
    GuestOutputPaintConfig* paint_config_out) {
  if (paint_config_out) {
    // Get the up-to-date guest output paint configuration settings set by the
    // UI thread.
    std::unique_lock<std::mutex> config_lock(guest_output_paint_config_mutex_);
    *paint_config_out = guest_output_paint_config_;
  }

  // Lock the mutex to make sure the image that will be acquired now is owned
  // exclusively by the calling thread for the time while this mutex is still
  // locked (it needs to be held by the consumer while working with anything
  // that depends on the image now being acquired or its index in the mailbox).
  std::unique_lock<std::mutex> consumer_lock(guest_output_mailbox_consumer_mutex_);
  // Acquire the up-to-date ready guest image (may be new, in this case the last
  // acquired one will be released, or still the same or no refresh has happened
  // since the last consumption).
  // memory_order_relaxed here because the ready index from this load will be
  // used directly only if it's the same as during the last consumption - thus
  // the image has already been acquired previously, no need for
  // memory_order_acquire (if the image has been acquired by a different
  // consumer though, the consumer mutex performs memory access ordering).
  uint32_t old_acquired_and_ready =
      guest_output_mailbox_acquired_and_ready_.load(std::memory_order_relaxed);
  // Desired acquired = current ready.
  // Desired ready = current ready (changed only by the producer).
  uint32_t desired_acquired_and_ready =
      (old_acquired_and_ready & ~uint32_t(3)) | (old_acquired_and_ready >> 2);
  // Either the same image as during the last consumption, or a new one, is
  // satisfying. However, if it's new, using memory_order_acq_rel to acquire the
  // new ready image (to make it acquired) and to release the old acquired image
  // (to let the producer take it as writable).
  while (old_acquired_and_ready != desired_acquired_and_ready &&
         !guest_output_mailbox_acquired_and_ready_.compare_exchange_weak(
             old_acquired_and_ready, desired_acquired_and_ready, std::memory_order_acq_rel,
             std::memory_order_relaxed)) {
    desired_acquired_and_ready =
        (old_acquired_and_ready & ~uint32_t(3)) | (old_acquired_and_ready >> 2);
  }
  uint32_t mailbox_index = desired_acquired_and_ready & 3;
  // Give the current acquired image to the caller, or UINT32_MAX if it's
  // inactive.
  const GuestOutputProperties& properties = guest_output_properties_[mailbox_index];
  mailbox_index_or_max_if_inactive_out = properties.IsActive() ? mailbox_index : UINT32_MAX;
  if (properties_out) {
    *properties_out = properties;
  }
  return std::move(consumer_lock);
}

Presenter::GuestOutputPaintFlow Presenter::GetGuestOutputPaintFlow(
    const GuestOutputProperties& properties, uint32_t host_rt_width, uint32_t host_rt_height,
    uint32_t max_rt_width, uint32_t max_rt_height, const GuestOutputPaintConfig& config) const {
  GuestOutputPaintFlow flow = {};

  // FIXME(Triang3l): Configuration variables racing with per-game config
  // loading.

  assert_not_zero(max_rt_width);
  assert_not_zero(max_rt_height);

  // Initialize one clear rectangle for the case of drawing no guest output, for
  // consistency with fewer state dependencies.
  flow.letterbox_clear_rectangle_count = 1;
  flow.letterbox_clear_rectangles[0].width = host_rt_width;
  flow.letterbox_clear_rectangles[0].height = host_rt_height;

  // For safety such as division by zero prevention.
  if (!properties.IsActive() || !host_rt_width || !host_rt_height ||
      !surface_width_in_paint_connection_ || !surface_height_in_paint_connection_) {
    return flow;
  }

  flow.properties = properties;

  // Multiplication-division rounding to the nearest.
  auto rescale_unsigned = [](uint32_t value, uint32_t new_scale, uint32_t old_scale) -> uint32_t {
    return uint32_t((uint64_t(value) * new_scale + (old_scale >> 1)) / old_scale);
  };
  auto rescale_signed = [](int32_t value, uint32_t new_scale, uint32_t old_scale) -> int32_t {
    // Plus old_scale / 2 for positive values, minus old_scale / 2 for
    // negative values for consistent rounding for both positive and
    // negative values (as the `/` operator rounds towards zero).
    // Example:
    // (-3 - 1) / 3 == -1
    // (-2 - 1) / 3 == -1
    // (-1 - 1) / 3 == 0
    // ---
    // (0 + 1) / 3 == 0
    // (1 + 1) / 3 == 0
    // (2 + 1) / 3 == 1
    return int32_t((int64_t(value) * new_scale + int32_t(old_scale >> 1) * (value < 0 ? -1 : 1)) /
                   old_scale);
  };

  // Final output location and dimensions.
  // All host location calculations are DPI-independent, conceptually depending
  // only on the aspect ratios, not the absolute values.
  uint32_t output_width, output_height;
  if (uint64_t(surface_width_in_paint_connection_) * properties.display_aspect_ratio_y >
      uint64_t(surface_height_in_paint_connection_) * properties.display_aspect_ratio_x) {
    // The window is wider that the source - crop along Y to preserve the aspect
    // ratio while stretching throughout the entire surface's width, then limit
    // the Y cropping via letterboxing or stretching along X.
    uint32_t present_safe_area;
    if (config.GetAllowOverscanCutoff() && REXCVAR_GET(present_safe_area_y) > 0 &&
        REXCVAR_GET(present_safe_area_y) < 100) {
      present_safe_area = uint32_t(REXCVAR_GET(present_safe_area_y));
    } else {
      present_safe_area = 100;
    }
    // Scale the desired width by the H:W aspect ratio (inverse of W:H) to get
    // the height.
    output_height =
        rescale_unsigned(surface_width_in_paint_connection_, properties.display_aspect_ratio_y,
                         properties.display_aspect_ratio_x);
    bool letterbox = false;
    if (output_height * present_safe_area > surface_height_in_paint_connection_ * 100) {
      // Don't crop out more than the safe area margin - letterbox or stretch.
      output_height = rescale_unsigned(surface_height_in_paint_connection_, 100, present_safe_area);
      letterbox = true;
    }
    if (letterbox && REXCVAR_GET(present_letterbox)) {
      output_width = rescale_unsigned(surface_height_in_paint_connection_ * 100,
                                      properties.display_aspect_ratio_x,
                                      properties.display_aspect_ratio_y * present_safe_area);
      // output_width might have been rounded up already by rescale_unsigned, so
      // rounding down in this division.
      flow.output_x = (int32_t(surface_width_in_paint_connection_) - int32_t(output_width)) / 2;
    } else {
      output_width = surface_width_in_paint_connection_;
      flow.output_x = 0;
    }
    // output_height might have been rounded up already by rescale_unsigned, so
    // rounding down in this division.
    flow.output_y = (int32_t(surface_height_in_paint_connection_) - int32_t(output_height)) / 2;
  } else {
    // The window is taller that the source - crop along X to preserve the
    // aspect ratio while stretching throughout the entire surface's height,
    // then limit the X cropping via letterboxing or stretching along Y.
    uint32_t present_safe_area;
    if (config.GetAllowOverscanCutoff() && REXCVAR_GET(present_safe_area_x) > 0 &&
        REXCVAR_GET(present_safe_area_x) < 100) {
      present_safe_area = uint32_t(REXCVAR_GET(present_safe_area_x));
    } else {
      present_safe_area = 100;
    }
    // Scale the desired height by the W:H aspect ratio to get the width.
    output_width =
        rescale_unsigned(surface_height_in_paint_connection_, properties.display_aspect_ratio_x,
                         properties.display_aspect_ratio_y);
    bool letterbox = false;
    if (output_width * present_safe_area > surface_width_in_paint_connection_ * 100) {
      // Don't crop out more than the safe area margin - letterbox or stretch.
      output_width = rescale_unsigned(surface_width_in_paint_connection_, 100, present_safe_area);
      letterbox = true;
    }
    if (letterbox && REXCVAR_GET(present_letterbox)) {
      output_height = rescale_unsigned(surface_width_in_paint_connection_ * 100,
                                       properties.display_aspect_ratio_y,
                                       properties.display_aspect_ratio_x * present_safe_area);
      // output_height might have been rounded up already by rescale_unsigned,
      // so rounding down in this division.
      flow.output_y = (int32_t(surface_height_in_paint_connection_) - int32_t(output_height)) / 2;
    } else {
      output_height = surface_height_in_paint_connection_;
      flow.output_y = 0;
    }
    // output_width might have been rounded up already by rescale_unsigned, so
    // rounding down in this division.
    flow.output_x = (int32_t(surface_width_in_paint_connection_) - int32_t(output_width)) / 2;
  }

  // Convert the location from surface pixels (which have 1:1 aspect ratio
  // relatively to the physical display) to render target pixels (the render
  // target size may be arbitrary with any aspect ratio, but if it's different
  // than the surface size, the OS is expected to stretch it to the surface
  // boundaries), preserving the aspect ratio.
  if (host_rt_width != surface_width_in_paint_connection_) {
    flow.output_x =
        rescale_signed(flow.output_x, host_rt_width, surface_width_in_paint_connection_);
    output_width =
        rescale_unsigned(output_width, host_rt_width, surface_width_in_paint_connection_);
  }
  if (host_rt_height != surface_height_in_paint_connection_) {
    flow.output_y =
        rescale_signed(flow.output_y, host_rt_height, surface_height_in_paint_connection_);
    output_height =
        rescale_unsigned(output_height, host_rt_height, surface_height_in_paint_connection_);
  }

  // The out-of-bounds checks are needed for correct letterbox calculations.
  // Though this normally shouldn't happen, but in case of rounding issues with
  // extreme values.
  int32_t output_right = flow.output_x + int32_t(output_width);
  int32_t output_bottom = flow.output_y + int32_t(output_height);
  // In rare cases (for example, when surface and render-target coordinate
  // spaces diverge unexpectedly), the computed output may become anchored at
  // the top-left and smaller than the host render target. This results in only
  // a top-left region being presented with black right/bottom areas.
  if (flow.output_x >= 0 && flow.output_y >= 0 && flow.output_x <= 1 && flow.output_y <= 1 &&
      output_width < host_rt_width && output_height < host_rt_height) {
    static std::atomic<bool> logged_top_left_expand = false;
    if (!logged_top_left_expand.exchange(true)) {
      REXLOG_WARN(
          "Presenter: expanding guest output from {}x{} to host render target "
          "{}x{} to avoid top-left-only presentation",
          output_width, output_height, host_rt_width, host_rt_height);
    }
    output_width = host_rt_width;
    output_height = host_rt_height;
    output_right = int32_t(output_width);
    output_bottom = int32_t(output_height);
  }
  if (!output_width || !output_height || output_right <= 0 || output_bottom <= 0 ||
      flow.output_x >= int32_t(host_rt_width) || flow.output_y >= int32_t(host_rt_height)) {
    return flow;
  }

  // The output image may have a part of it outside the final render target (if
  // using the overscan area to stretch the image to the entire surface while
  // preserving the guest aspect ratio if it differs from the host one, for
  // instance). While the final render target size is known to be within the
  // host render target / image size limit, the intermediate images may be
  // larger than that as they include the overscan area that will be outside the
  // screen. Make sure the intermediate images can't be larger than the maximum
  // render target size.
  uint32_t output_width_clamped = std::min(output_width, max_rt_width);
  uint32_t output_height_clamped = std::min(output_height, max_rt_height);

  // wwe2k14 patch (Session 73, SGSR): Snapdragon GSR 1 upscales and sharpens in one pass straight from the guest
  // output to the final output rectangle (any scale; no RCAS / bilinear after it). Like FSR, only when upscaling
  // along at least one axis; otherwise (or without backend support) the bilinear pass below is used.
  if (config.GetEffect() == GuestOutputPaintConfig::Effect::kSgsr && IsGuestOutputPaintSgsrSupported() &&
      (properties.frontbuffer_width < output_width || properties.frontbuffer_height < output_height)) {
    assert_true(flow.effect_count < flow.effects.size());
    flow.effect_output_sizes[flow.effect_count] = std::make_pair(output_width, output_height);
    flow.effects[flow.effect_count++] = GuestOutputPaintEffect::kSgsr;
    static uint64_t sgsr_logged_size = 0;  // once per output size
    if (const uint64_t size = (uint64_t(output_width) << 32) | output_height; size != sgsr_logged_size) {
      sgsr_logged_size = size;
      REXLOG_INFO("Presenter: SGSR upscale {}x{} -> {}x{}", properties.frontbuffer_width,
                  properties.frontbuffer_height, output_width, output_height);
    }
  }

#if defined(REX_HAS_FIDELITYFX_SDK)
  if (config.GetEffect() == GuestOutputPaintConfig::Effect::kCas ||
      config.GetEffect() == GuestOutputPaintConfig::Effect::kFsr ||
      config.GetEffect() == GuestOutputPaintConfig::Effect::kFsr2 ||
      config.GetEffect() == GuestOutputPaintConfig::Effect::kFsr3) {
    // FidelityFX Super Resolution and Contrast Adaptive Sharpening only work
    // good for up to 2x2 upscaling due to the way they fetch texels.
    // CAS is primarily a sharpening filter, not an upscaling one (its upscaling
    // eliminates reduces blurriness, but doesn't preserve the shapes of edges,
    // and executing it multiple times will only result in oversharpening. So,
    // using it for scales only of up to 2x2, then simply stretching with
    // bilinear filtering.
    // EASU of FSR, however, preserves edges, it's not supposed to blur them or
    // to make them jagged, so it can be executed multiple times - running
    // multiple EASU passes for scale factors of over 2x2.
    // Just one EASU pass rather than multiple for scaling to factors bigger
    // than 2x2 (especially significantly bigger, such as 1152x640 to 3840x2160,
    // or 3.333x3.375) results in blurry edges and an overall noisy look,
    // multiple passes improve visual stability.
    std::pair<uint32_t, uint32_t> ffx_last_size;
    if (flow.effect_count) {
      ffx_last_size = flow.effect_output_sizes[flow.effect_count - 1];
    } else {
      ffx_last_size.first = properties.frontbuffer_width;
      ffx_last_size.second = properties.frontbuffer_height;
    }
    bool is_temporal_effect = config.GetEffect() == GuestOutputPaintConfig::Effect::kFsr2 ||
                              config.GetEffect() == GuestOutputPaintConfig::Effect::kFsr3;
    bool temporal_quality_mode_forced =
        is_temporal_effect &&
        config.GetFsrQualityMode() != GuestOutputPaintConfig::FsrQualityMode::kAuto;
    if ((config.GetEffect() == GuestOutputPaintConfig::Effect::kFsr || is_temporal_effect) &&
        ((ffx_last_size.first < output_width_clamped ||
          ffx_last_size.second < output_height_clamped) ||
         temporal_quality_mode_forced)) {
      // AMD FidelityFX Super Resolution - upsample along at least one axis.
      // Using the output size clamped to the maximum render target size here as
      // EASU will always write to intermediate images, and RCAS supports only
      // 1:1.
      if (is_temporal_effect) {
        uint32_t temporal_input_width = ffx_last_size.first;
        uint32_t temporal_input_height = ffx_last_size.second;
        uint32_t quality_mode_render_width = 0;
        uint32_t quality_mode_render_height = 0;
        if (QueryTemporalFsrRenderResolutionFromQualityMode(
                output_width_clamped, output_height_clamped, config.GetFsrQualityMode(),
                quality_mode_render_width, quality_mode_render_height)) {
          quality_mode_render_width =
              std::max(uint32_t(1), std::min(quality_mode_render_width, output_width_clamped));
          quality_mode_render_height =
              std::max(uint32_t(1), std::min(quality_mode_render_height, output_height_clamped));
          if (quality_mode_render_width < temporal_input_width ||
              quality_mode_render_height < temporal_input_height) {
            temporal_input_width = quality_mode_render_width;
            temporal_input_height = quality_mode_render_height;
            if (temporal_input_width != ffx_last_size.first ||
                temporal_input_height != ffx_last_size.second) {
              assert_true(flow.effect_count < flow.effects.size());
              flow.effect_output_sizes[flow.effect_count] =
                  std::make_pair(temporal_input_width, temporal_input_height);
              // Pre-temporal quality mode may request a lower render
              // resolution. Use CAS resample since bilinear is not allowed in
              // intermediate passes.
              flow.effects[flow.effect_count++] = GuestOutputPaintEffect::kCasResample;
              ffx_last_size.first = temporal_input_width;
              ffx_last_size.second = temporal_input_height;
            }
          } else if (quality_mode_render_width > temporal_input_width ||
                     quality_mode_render_height > temporal_input_height) {
            LogTemporalFsrQualityModeInputLimitOnce();
          }
        }

        // A single temporal upscaler dispatch can target the final clamped
        // size directly, unlike the spatial multi-pass chain.
        ffx_last_size.first = output_width_clamped;
        ffx_last_size.second = output_height_clamped;
        assert_true(flow.effect_count < flow.effects.size());
        flow.effect_output_sizes[flow.effect_count] = ffx_last_size;
        flow.effects[flow.effect_count++] = GuestOutputPaintEffect::kFsrEasu;
      } else {
        uint32_t easu_max_passes = config.GetFsrMaxUpsamplingPasses();
        uint32_t easu_pass_count = 0;
        while (easu_pass_count < easu_max_passes &&
               (ffx_last_size.first < output_width_clamped ||
                ffx_last_size.second < output_height_clamped)) {
          ffx_last_size.first = std::min(ffx_last_size.first * uint32_t(2), output_width_clamped);
          ffx_last_size.second =
              std::min(ffx_last_size.second * uint32_t(2), output_height_clamped);
          assert_true(flow.effect_count < flow.effects.size());
          flow.effect_output_sizes[flow.effect_count] = ffx_last_size;
          flow.effects[flow.effect_count++] = GuestOutputPaintEffect::kFsrEasu;
          ++easu_pass_count;
        }
      }
      assert_true(flow.effect_count < flow.effects.size());
      flow.effect_output_sizes[flow.effect_count] = ffx_last_size;
      flow.effects[flow.effect_count++] = GuestOutputPaintEffect::kFsrRcas;
    } else {
      // AMD FidelityFX Contrast Adaptive Sharpening - sharpen or downsample, or
      // upsample up to 2x2 if CAS is specified to be used for upscaling too.
      // Using the unclamped output size as CAS may be the last pass - if a
      // bilinear pass is needed afterwards, and the CAS pass will be writing to
      // an intermediate image, the CAS pass output size will be clamped while
      // adding the bilinear stretch.
      std::pair<uint32_t, uint32_t> pre_cas_size = ffx_last_size;
      ffx_last_size.first = std::min(ffx_last_size.first * uint32_t(2), output_width);
      ffx_last_size.second = std::min(ffx_last_size.second * uint32_t(2), output_height);
      assert_true(flow.effect_count < flow.effects.size());
      flow.effect_output_sizes[flow.effect_count] = ffx_last_size;
      flow.effects[flow.effect_count++] = ffx_last_size == pre_cas_size
                                              ? GuestOutputPaintEffect::kCasSharpen
                                              : GuestOutputPaintEffect::kCasResample;
    }
  }
#endif  // defined(REX_HAS_FIDELITYFX_SDK)

  std::pair<uint32_t, uint32_t>* last_pre_bilinear_effect_size =
      flow.effect_count ? &flow.effect_output_sizes[flow.effect_count - 1] : nullptr;
  if (!last_pre_bilinear_effect_size || last_pre_bilinear_effect_size->first != output_width ||
      last_pre_bilinear_effect_size->second != output_height) {
    // If not using FidelityFX, or it has reached its upscaling capabilities,
    // but more is needed, stretch via bilinear filtering.
    // Clamp the output size of the last effect to the maximum render target
    // size because it will go to an intermediate image now.
    if (last_pre_bilinear_effect_size) {
#if defined(REX_HAS_FIDELITYFX_SDK)
      // RCAS only works for 1:1, clamping must be done explicitly for FSR.
      assert_false(flow.effects[flow.effect_count - 1] == GuestOutputPaintEffect::kFsrRcas &&
                   (last_pre_bilinear_effect_size->first > max_rt_width ||
                    last_pre_bilinear_effect_size->second > max_rt_height));
#endif
      last_pre_bilinear_effect_size->first =
          std::min(last_pre_bilinear_effect_size->first, max_rt_width);
      last_pre_bilinear_effect_size->second =
          std::min(last_pre_bilinear_effect_size->second, max_rt_height);
    }
    assert_true(flow.effect_count < flow.effects.size());
    flow.effect_output_sizes[flow.effect_count] = std::make_pair(output_width, output_height);
    flow.effects[flow.effect_count++] = GuestOutputPaintEffect::kBilinear;
  }

  assert_not_zero(flow.effect_count);

  if (config.GetDither()) {
    // Dithering must be applied only to the final effect since resampling and
    // sharpening filters may considering the dithering noise features and
    // amplify it.
    GuestOutputPaintEffect& last_effect = flow.effects[flow.effect_count - 1];
    switch (last_effect) {
      case GuestOutputPaintEffect::kBilinear:
        // Dithering has no effect for 1:1 copying of a 8bpc image.
        if (!properties.is_8bpc || flow.effect_count > 1 ||
            output_width != properties.frontbuffer_width ||
            output_height != properties.frontbuffer_height) {
          last_effect = GuestOutputPaintEffect::kBilinearDither;
        }
        break;
#if defined(REX_HAS_FIDELITYFX_SDK)
      case GuestOutputPaintEffect::kCasSharpen:
        last_effect = GuestOutputPaintEffect::kCasSharpenDither;
        break;
      case GuestOutputPaintEffect::kCasResample:
        last_effect = GuestOutputPaintEffect::kCasResampleDither;
        break;
      case GuestOutputPaintEffect::kFsrRcas:
        last_effect = GuestOutputPaintEffect::kFsrRcasDither;
        break;
#endif
      default:
        break;
    }
  }

#ifndef NDEBUG
  for (size_t i = 0; i + 1 < flow.effect_count; ++i) {
    assert_true(CanGuestOutputPaintEffectBeIntermediate(flow.effects[i]));
  }
  assert_true(CanGuestOutputPaintEffectBeFinal(flow.effects[flow.effect_count - 1]));
#endif

  // Calculate the letterbox geometry.
  if (flow.effect_count) {
    flow.letterbox_clear_rectangle_count = 0;
    uint32_t letterbox_mid_top = uint32_t(std::max(flow.output_y, int32_t(0)));
    // Top.
    if (letterbox_mid_top) {
      assert_true(flow.letterbox_clear_rectangle_count < flow.letterbox_clear_rectangles.size());
      GuestOutputPaintFlow::ClearRectangle& letterbox_clear_rectangle_top =
          flow.letterbox_clear_rectangles[flow.letterbox_clear_rectangle_count++];
      letterbox_clear_rectangle_top.x = 0;
      letterbox_clear_rectangle_top.y = 0;
      letterbox_clear_rectangle_top.width = host_rt_width;
      letterbox_clear_rectangle_top.height = letterbox_mid_top;
    }
    uint32_t letterbox_mid_bottom = std::min(uint32_t(output_bottom), host_rt_height);
    uint32_t letterbox_mid_height = letterbox_mid_bottom - letterbox_mid_top;
    // Middle-left.
    if (flow.output_x > 0) {
      assert_true(flow.letterbox_clear_rectangle_count < flow.letterbox_clear_rectangles.size());
      GuestOutputPaintFlow::ClearRectangle& letterbox_clear_rectangle_left =
          flow.letterbox_clear_rectangles[flow.letterbox_clear_rectangle_count++];
      letterbox_clear_rectangle_left.x = 0;
      letterbox_clear_rectangle_left.y = letterbox_mid_top;
      letterbox_clear_rectangle_left.width = uint32_t(flow.output_x);
      letterbox_clear_rectangle_left.height = letterbox_mid_height;
    }
    // Middle-right.
    if (uint32_t(output_right) < host_rt_width) {
      assert_true(flow.letterbox_clear_rectangle_count < flow.letterbox_clear_rectangles.size());
      GuestOutputPaintFlow::ClearRectangle& letterbox_clear_rectangle_right =
          flow.letterbox_clear_rectangles[flow.letterbox_clear_rectangle_count++];
      letterbox_clear_rectangle_right.x = uint32_t(output_right);
      letterbox_clear_rectangle_right.y = letterbox_mid_top;
      letterbox_clear_rectangle_right.width = host_rt_width - uint32_t(output_right);
      letterbox_clear_rectangle_right.height = letterbox_mid_height;
    }
    // Bottom.
    if (letterbox_mid_bottom < host_rt_height) {
      assert_true(flow.letterbox_clear_rectangle_count < flow.letterbox_clear_rectangles.size());
      GuestOutputPaintFlow::ClearRectangle& letterbox_clear_rectangle_top =
          flow.letterbox_clear_rectangles[flow.letterbox_clear_rectangle_count++];
      letterbox_clear_rectangle_top.x = 0;
      letterbox_clear_rectangle_top.y = letterbox_mid_bottom;
      letterbox_clear_rectangle_top.width = host_rt_width;
      letterbox_clear_rectangle_top.height = host_rt_height - letterbox_mid_bottom;
    }
  }

  return flow;
}

void Presenter::ExecuteUIDrawersFromUIThread(UIDrawContext& ui_draw_context) {
  // May be called by the implementations only when requested.
  assert_true(is_in_ui_thread_paint_);
  // Drawers can add or remove drawers (including themselves), need to ensure
  // iterator validity in this case.
  assert_false(is_executing_ui_drawers_);
  ui_draw_next_iterator_ = ui_drawers_.begin();
  is_executing_ui_drawers_ = true;
  while (ui_draw_next_iterator_ != ui_drawers_.end()) {
    // The current iterator may be invalidated, and ui_draw_next_iterator_ may
    // be changed, during the execution of the drawer if the list of the drawers
    // is modified by it - don't assume that after the call
    // ui_draw_next_iterator_ will be the same as
    // std::next(ui_draw_next_iterator_) before it.
    auto it_current = ui_draw_next_iterator_++;
    // Don't draw twice if already drawn in this frame (may happen if the Z
    // order of a drawer was increased from below the current one to above it by
    // one of the drawers).
    if (it_current->second.last_draw != ui_draw_current_) {
      ui_draw_current_z_order_ = it_current->first;
      it_current->second.last_draw = ui_draw_current_;
      it_current->second.drawer->Draw(ui_draw_context);
    }
  }
  is_executing_ui_drawers_ = false;
  ++ui_draw_current_;
}

void Presenter::SetPaintModeFromUIThread(PaintMode new_mode) {
  // Can be modified only from the UI thread, so can skip locking if it's the
  // same.
  if (paint_mode_ == new_mode) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(paint_mode_mutex_);
    paint_mode_ = new_mode;
  }
  UpdateUITicksNeededFromUIThread();
}

Presenter::PaintMode Presenter::GetDesiredPaintModeFromUIThread(bool is_paintable) const {
  if (!is_paintable) {
    // The only case when kNone can be returned, for surface connection updates
    // when it's known that the UI thread currently has access to the connection
    // lifecycle.
    return PaintMode::kNone;
  }
  if (!REXCVAR_GET(host_present_from_non_ui_thread)) {
    return PaintMode::kUIThreadOnRequest;
  }
  if (surface_paint_connection_has_implicit_vsync_) {
    // Don't be causing host vertical sync CPU waits in the thread generating
    // the guest output.
    return PaintMode::kUIThreadOnRequest;
  }
  if (!ui_drawers_.empty()) {
    // The UI can be drawn only by the UI thread, and it needs to be drawn -
    // paint in the UI thread.
    return PaintMode::kUIThreadOnRequest;
  }
  // Only the guest output needs to be drawn - let the guest output thread
  // present immediately for a lower latency.
  return PaintMode::kGuestOutputThreadImmediately;
}

void Presenter::DisconnectPaintingFromSurfaceFromUIThread(SurfacePaintConnectionState new_state) {
  assert_false(IsConnectedSurfacePaintConnectionState(new_state));
  if (IsConnectedSurfacePaintConnectionState(surface_paint_connection_state_)) {
    DisconnectPaintingFromSurfaceFromUIThreadImpl();
  }
  surface_paint_connection_state_ = new_state;
  surface_paint_connection_has_implicit_vsync_ = false;
  surface_width_in_paint_connection_ = 0;
  surface_height_in_paint_connection_ = 0;
}

void Presenter::UpdateSurfacePaintConnectionFromUIThread(bool* repaint_needed_out,
                                                         bool update_paint_mode_to_desired) {
  assert_not_null(surface_);

  // Validate that painting lifecycle is accessible by the UI thread currently,
  // not given to the guest output thread. The mode can be modified only by
  // the UI thread, so no need to lock the mutex.
  assert_true(paint_mode_ != PaintMode::kGuestOutputThreadImmediately);

  // Initialize repaint_needed_out for failure cases.
  if (repaint_needed_out) {
    *repaint_needed_out = false;
  }

  // If the connection state is kUnconnectedSurfaceReportedUnusable, the
  // implementation has reported that the surface is not usable by the presenter
  // at all, and it's pointless to retry connecting to it.
  if (surface_paint_connection_state_ !=
      SurfacePaintConnectionState::kUnconnectedSurfaceReportedUnusable) {
    uint32_t surface_width = 0, surface_height = 0;
    bool surface_area_available = surface_->GetSize(surface_width, surface_height);
    if (!surface_area_available) {
      // The surface is currently zero-area (or has become zero-area), try again
      // when it's resized.
      DisconnectPaintingFromSurfaceFromUIThread(
          SurfacePaintConnectionState::kUnconnectedRetryAtStateChange);
    } else {
      bool is_reconnect = IsConnectedSurfacePaintConnectionState(surface_paint_connection_state_);
      bool is_vsync_implicit = false;
      SurfacePaintConnectResult connect_result = ConnectOrReconnectPaintingToSurfaceFromUIThread(
          *surface_, surface_width, surface_height,
          surface_paint_connection_state_ == SurfacePaintConnectionState::kConnectedPaintable,
          is_vsync_implicit);
      switch (connect_result) {
        case SurfacePaintConnectResult::kSuccess:
          if (repaint_needed_out) {
            *repaint_needed_out = true;
          }
          // Fallthrough to common success handling.
        case SurfacePaintConnectResult::kSuccessUnchanged:
          // Don't know yet what the first result was (success or suboptimal).
          surface_paint_connection_was_optimal_at_successful_paint_ = false;
          surface_paint_connection_state_ = SurfacePaintConnectionState::kConnectedPaintable;
          surface_paint_connection_has_implicit_vsync_ = is_vsync_implicit;
          surface_width_in_paint_connection_ = surface_width;
          surface_height_in_paint_connection_ = surface_height;
          outdated_recovery_ = false;
          if (!is_reconnect && repaint_needed_out) {
            *repaint_needed_out = true;
          }
          break;
        case SurfacePaintConnectResult::kFailure:
          surface_paint_connection_state_ =
              SurfacePaintConnectionState::kUnconnectedRetryAtStateChange;
          break;
        case SurfacePaintConnectResult::kFailureSurfaceUnusable:
          surface_paint_connection_state_ =
              SurfacePaintConnectionState::kUnconnectedSurfaceReportedUnusable;
          break;
      }
    }
  }

  if (update_paint_mode_to_desired) {
    SetPaintModeFromUIThread(GetDesiredPaintModeFromUIThread(
        surface_paint_connection_state_ == SurfacePaintConnectionState::kConnectedPaintable));
  }
}

bool Presenter::RequestPaintOrConnectionRecoveryViaWindow(bool force_ui_thread_paint_tick) {
  // Can be called from any thread if an existing window_ is available in it,
  // and it's known to have a Surface that will be the same throughout this
  // call - not doing any checks whether this request can be satisfied
  // theoretically. For safety, check whether the window exists unconditionally.
  assert_not_null(window_);
  assert_not_null(surface_);
  if (ui_thread_paint_requested_.exchange(true, std::memory_order_relaxed)) {
    // Invalidation pending already, no need to do it twice.
    return false;
  }
  if (force_ui_thread_paint_tick) {
    ForceUIThreadPaintTick();
  }
  window_->RequestPaint();
  return true;
}

void Presenter::UpdateSurfaceMonitorFromUIThread(bool old_monitor_potentially_disconnected) {
  // For dropping the monitor when the window is closing and is losing its
  // surface, the existence of `surface_` (which implies that `window_` exists
  // too) must be the condition for a non-null monitor, not just the existence
  // of `window_`.
#if REX_PLATFORM_WIN32
  HMONITOR surface_new_win32_monitor = nullptr;
  if (surface_) {
    HWND hwnd = static_cast<HWND>(window_->GetNativeWindowHandle());
    // The HWND may be non-existent if the window has been closed and destroyed
    // (the HWND, not the rex::ui::Window) already.
    if (hwnd) {
      surface_new_win32_monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONULL);
    }
  }
  if (old_monitor_potentially_disconnected || surface_win32_monitor_ != surface_new_win32_monitor) {
    surface_win32_monitor_ = surface_new_win32_monitor;
    if (dxgi_ui_tick_factory_ && !dxgi_ui_tick_factory_->IsCurrent()) {
      // If a monitor has been newly connected, it won't appear in the old
      // factory, need to recreate it.
      {
        Microsoft::WRL::ComPtr<IDXGIOutput> old_factory_output_to_release;
        {
          std::scoped_lock<std::mutex> dxgi_ui_tick_lock(dxgi_ui_tick_mutex_);
          old_factory_output_to_release = std::move(dxgi_ui_tick_output_);
        }
      }
      dxgi_ui_tick_factory_.Reset();
    }
    if (!dxgi_ui_tick_factory_) {
      if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&dxgi_ui_tick_factory_)))) {
        REXLOG_ERROR("Presenter: Failed to create a DXGI factory");
      }
    }
    Microsoft::WRL::ComPtr<IDXGIOutput> new_dxgi_output;
    if (dxgi_ui_tick_factory_ && surface_new_win32_monitor) {
      new_dxgi_output =
          GetDXGIOutputForMonitor(dxgi_ui_tick_factory_.Get(), surface_new_win32_monitor);
    }
    // If the adapter was recreated, and the old output was released before its
    // destruction, notifying is still required - the vertical blank wait thread
    // might have entered the condition variable wait already as the output was
    // null.
    bool signal_dxgi_ui_tick_control;
    {
      std::unique_lock<std::mutex> dxgi_ui_tick_lock(dxgi_ui_tick_mutex_);
      bool dxgi_output_was_null = (dxgi_ui_tick_output_ == nullptr);
      dxgi_ui_tick_output_ = new_dxgi_output;
      signal_dxgi_ui_tick_control =
          dxgi_output_was_null && AreDXGIUITicksWaitable(dxgi_ui_tick_lock);
    }
    if (signal_dxgi_ui_tick_control) {
      dxgi_ui_tick_control_condition_.notify_all();
    }
  }
#endif  // XE_PLATFORM
}

bool Presenter::InSurfaceOnMonitorFromUIThread() const {
  if (!surface_) {
    return false;
  }
#if REX_PLATFORM_WIN32
  return surface_win32_monitor_ != nullptr;
#else
  return true;
#endif  // XE_PLATFORM
}

Presenter::PaintResult Presenter::PaintAndPresent(bool execute_ui_drawers) {
  assert_false(execute_ui_drawers && !is_in_ui_thread_paint_);
  assert_true(surface_paint_connection_state_ == SurfacePaintConnectionState::kConnectedPaintable);
  // WWE13 (Android): no repaint of a guest frame already on screen. The UI thread repainted the same image ~270 times a
  // second, every paint a full-screen pass on the phone's GPU in the game's way. A repeat still goes out every 250 ms.
  if (REXCVAR_GET(present_skip_repeats)) {
    static auto last_present = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (guest_output_ready_seq_.load(std::memory_order_acquire) == guest_output_presented_seq_.load(std::memory_order_relaxed) &&
        now - last_present < std::chrono::milliseconds(250)) {
      skipped_repeat_ = true;  // (no immediate repaint request for it: a new guest frame requests its own paint)
      return PaintResult::kNotPresented;
    }
    last_present = now;
  }
  PaintResult result = PaintAndPresentImpl(execute_ui_drawers);
  switch (result) {
    case PaintResult::kPresented:
      surface_paint_connection_was_optimal_at_successful_paint_ = true;
      break;
    case PaintResult::kPresentedSuboptimal:
      // Make outdated if previously optimal, now suboptimal, but don't cause
      // the connection to become outdated if it has been suboptimal from the
      // very beginning.
      if (surface_paint_connection_was_optimal_at_successful_paint_) {
        surface_paint_connection_state_ = SurfacePaintConnectionState::kConnectedOutdated;
      }
      break;
    case PaintResult::kNotPresentedConnectionOutdated:
      surface_paint_connection_state_ = SurfacePaintConnectionState::kConnectedOutdated;
      break;
    default:
      // Another issue not directly related to the surface connection.
      break;
  }
  return result;
}

void Presenter::HandleUIDrawersChangeFromUIThread(bool drawers_were_empty) {
  if (is_in_ui_thread_paint_) {
    // Defer the refresh so no dangerous lifecycle-related changes happen during
    // drawing.
    if (!ui_drawers_.empty()) {
      request_ui_paint_after_current_ui_thread_paint_ = true;
    }
    return;
  }

  if (paint_mode_ == PaintMode::kNone) {
    // Not connected, no point in refreshing (checking a more conservative
    // paint_mode_ because the actual connection state may currently be owned by
    // the guest output thread instead) or in toggling the ownership (the rest
    // of the function can assume it's not kNone).
    return;
  }

  if (ui_drawers_.empty() != drawers_were_empty) {
    // Require the UI thread to paint if it needs the UI, or let the guest
    // output thread paint immediately if not.
    SetPaintModeFromUIThread(GetDesiredPaintModeFromUIThread(true));
    // Make sure the ticks for limiting the UI frame rate are sent.
    UpdateUITicksNeededFromUIThread();
  }

  // Request painting so the changes to the UI drawer list are reflected as
  // quickly as possible.
  // RequestUIPaintFromUIThread is not enough, because a paint request is also
  // needed if disabling the UI, to force paint a frame without the UI as soon
  // as possible - it can't be dropped if ui_drawers_ is empty.
  // The coarse painting availability (and thus the availability of `window_`,
  // which is required for the paint mode to be anything else than kNone) has
  // already been checked above.
  ForceUIThreadPaintTick();
  window_->RequestPaint();
}

void Presenter::UpdateUITicksNeededFromUIThread() {
#if REX_PLATFORM_WIN32
  bool new_needed = AreUITicksNeededFromUIThread();
  if (dxgi_ui_ticks_needed_ == new_needed) {
    return;
  }
  bool signal_dxgi_ui_tick_control;
  {
    std::unique_lock<std::mutex> dxgi_ui_tick_lock(dxgi_ui_tick_mutex_);
    dxgi_ui_ticks_needed_ = new_needed;
    signal_dxgi_ui_tick_control = AreDXGIUITicksWaitable(dxgi_ui_tick_lock);
  }
  if (signal_dxgi_ui_tick_control) {
    dxgi_ui_tick_control_condition_.notify_all();
  }
#endif
}

void Presenter::WaitForUITickFromUIThread() {
#if REX_PLATFORM_WIN32
  if (!AreUITicksNeededFromUIThread()) {
    return;
  }
  std::unique_lock<std::mutex> dxgi_ui_tick_lock(dxgi_ui_tick_mutex_);
  uint64_t last_vblank_before_wait = dxgi_ui_tick_last_vblank_;
  while (true) {
    // Guest output present requests should interrupt the wait as quickly as
    // possible as they should be fulfilled as early as possible.
    if (dxgi_ui_tick_force_requested_) {
      dxgi_ui_tick_force_requested_ = false;
      return;
    }
    if (!AreDXGIUITicksWaitable(dxgi_ui_tick_lock)) {
      return;
    }
    if (dxgi_ui_tick_last_vblank_ > dxgi_ui_tick_last_draw_) {
      // If there have been multiple vblanks during the wait for some reason,
      // next time draw the UI immediately.
      dxgi_ui_tick_last_draw_ =
          std::min(last_vblank_before_wait + uint64_t(1), dxgi_ui_tick_last_vblank_);
      return;
    }
    dxgi_ui_tick_signal_condition_.wait(dxgi_ui_tick_lock);
  }
#endif  // XE_PLATFORM
}

void Presenter::ForceUIThreadPaintTick() {
#if REX_PLATFORM_WIN32
  std::scoped_lock<std::mutex> dxgi_ui_tick_lock(dxgi_ui_tick_mutex_);
  dxgi_ui_tick_force_requested_ = true;
  // wwe2k14 patch (Session 21, ONCE): wake a UI thread already waiting for the tick.
  if (REXCVAR_GET(present_guest_frames_once)) {
    dxgi_ui_tick_signal_condition_.notify_all();
  }
#endif  // XE_PLATFORM
}

#if REX_PLATFORM_WIN32
Microsoft::WRL::ComPtr<IDXGIOutput> Presenter::GetDXGIOutputForMonitor(IDXGIFactory1* factory,
                                                                       HMONITOR monitor) {
  Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
  for (UINT adapter_index = 0;
       SUCCEEDED(factory->EnumAdapters(adapter_index, adapter.ReleaseAndGetAddressOf()));
       ++adapter_index) {
    Microsoft::WRL::ComPtr<IDXGIOutput> output;
    for (UINT output_index = 0; SUCCEEDED(adapter->EnumOutputs(output_index, &output));
         ++output_index) {
      DXGI_OUTPUT_DESC output_desc;
      if (SUCCEEDED(output->GetDesc(&output_desc)) && output_desc.Monitor == monitor) {
        return std::move(output);
      }
    }
  }
  return nullptr;
}

void Presenter::DXGIUITickThread() {
  std::unique_lock<std::mutex> dxgi_ui_tick_lock(dxgi_ui_tick_mutex_);
  while (true) {
    if (dxgi_ui_tick_thread_shutdown_) {
      return;
    }
    if (!AreDXGIUITicksWaitable(dxgi_ui_tick_lock)) {
      dxgi_ui_tick_control_condition_.wait(dxgi_ui_tick_lock);
      continue;
    }
    // Wait for vertical blank, with the mutex unlocked (holding a new reference
    // to the current output while it's happening) so subscribers can still do
    // early-out checks.
    bool wait_succeeded;
    {
      Microsoft::WRL::ComPtr<IDXGIOutput> dxgi_output = dxgi_ui_tick_output_;
      dxgi_ui_tick_lock.unlock();
      // wwe2k14 patch (Session 60): a WaitForVBlank that returns at once (monitor off / asleep) made this a busy
      // loop that also woke the UI thread thousands of times a second; sleep instead. Counted, logged every 10 s.
      const auto wait_start = std::chrono::steady_clock::now();
      wait_succeeded = SUCCEEDED(dxgi_output->WaitForVBlank());
      const auto wait_end = std::chrono::steady_clock::now();
      static uint64_t s60_waits = 0, s60_fast = 0;
      static auto s60_log_time = wait_end;
      ++s60_waits;
      const int32_t fast_sleep_ms = REXCVAR_GET(present_ui_tick_fast_sleep_ms);
      if (wait_end - wait_start < std::chrono::milliseconds(1)) {
        ++s60_fast;
        if (fast_sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(fast_sleep_ms));
      }
      if (wait_end - s60_log_time >= std::chrono::seconds(10)) {
        REXLOG_INFO("[s60][ui_tick] {} vblank waits in 10 s, {} returned in under 1 ms{}", s60_waits, s60_fast,
                    s60_fast && fast_sleep_ms > 0 ? " (slept instead)" : "");
        s60_waits = s60_fast = 0;
        s60_log_time = wait_end;
      }
    }
    dxgi_ui_tick_lock.lock();
    if (wait_succeeded) {
      ++dxgi_ui_tick_last_vblank_;
    } else {
      // Lost the ability to wait for a vertical blank on this output, notify
      // the waiting threads, and wait for a new one.
      dxgi_ui_tick_output_.Reset();
    }
    dxgi_ui_tick_signal_condition_.notify_all();
  }
}
#endif  // XE_PLATFORM

}  // namespace ui
}  // namespace rex
