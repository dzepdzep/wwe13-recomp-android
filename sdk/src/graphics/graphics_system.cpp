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

#include <rex/det.h>  // wwe2k14 patch (Session 15): det_frames
#include <rex/graphics/graphics_system.h>
#include <rex/perf/timeline.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <rex/cvar.h>
#include <rex/graphics/command_processor.h>
#include <rex/graphics/d3d_noring.h>  // (Session 92, M5b) D3dNoringVblankSignal
#include <rex/graphics/flags.h>
REXCVAR_DECLARE(uint64_t, gpu_d3d_noring_shared);  // wwe2k14 patch (Session 101, lane 233): d3d_noring.cpp
#include <rex/graphics/free_counters.h>
#include <rex/kernel/xboxkrnl/video.h>
#include <rex/logging.h>
#include <rex/stream.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xthread.h>
#include <rex/ui/graphics_provider.h>
#include <rex/ui/window.h>
#include <rex/ui/windowed_app_context.h>

#if REX_PLATFORM_WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// Windows 10 1803+; older SDK headers lack the define.
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
#endif  // REX_PLATFORM_WIN32

// wwe2k14 patch (Session 95, spike): the command thread on a physical core of its own.
REXCVAR_DEFINE_BOOL(gpu_cmd_own_core, false, "GPU",
                    "Session 95: Windows, 4 or more physical cores: the GPU command thread gets the last core to itself "
                    "(the process's default CPU sets leave that core out, the thread selects both its logical processors, so "
                    "a system thread on one does not stall it). "
                    "Surface Bryan: identical frames took the thread 12.6 to 25 ms with ~3.6 logical processors busy on 4 "
                    "cores. Logs [pace] gpu_cmd_own_core.");
// wwe2k14 patch (Session 95, round 7): the same core kept free without the process-wide call.
REXCVAR_DEFINE_INT32(gpu_cmd_keep_core, 0, "GPU",
                     "Session 95: Windows, 4 or more physical cores, set at start: keep the GPU command thread's core free of "
                     "busy guest threads. 0 off. 1 hints only: the command thread gets that core as its ideal processor, and a "
                     "guest thread over 8% of a logical CPU gets another core as its ideal one. 2 keep-out: the same, but that "
                     "guest thread selects every logical processor except that core (CPU sets, per thread). The command thread "
                     "keeps every logical processor, there is no process-wide call, and each thread is placed once. Looks at "
                     "most every 2 s of wall time (checked every 60 vblanks) from the GPU VSync thread; ignored while gpu_cmd_own_core is on. "
                     "Logs [pace] gpu_cmd_keep_core.")
    .range(0, 2);

namespace rex::graphics {
#if REX_PLATFORM_WIN32
namespace {

using SetThreadCpuSets = BOOL(WINAPI*)(HANDLE, const ULONG*, ULONG);

SetThreadCpuSets FindSetThreadCpuSets() {
  const HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
  return kernel ? reinterpret_cast<SetThreadCpuSets>(GetProcAddress(kernel, "SetThreadSelectedCpuSets")) : nullptr;
}

// Which core the command thread gets: the last physical core of processor group 0. Both gpu_cmd_own_core and
// gpu_cmd_keep_core use this choice.
struct CommandCoreChoice {
  uint32_t own = 0;                 // the core's index
  uint32_t own_lp = UINT32_MAX;     // its first logical processor
  size_t cores = 0;                 // physical cores in group 0
  std::vector<ULONG> own_ids;       // CPU set ids of the core's logical processors
  std::vector<ULONG> others;        // CPU set ids of every other logical processor
  std::vector<uint32_t> other_lps;  // every other logical processor, each core's first one ahead of any second one
};

// False when the choice cannot be made: no CPU set functions, or under 4 physical cores. who = the switch the log lines
// name; nullptr logs nothing (the second caller: the first has said it).
bool ChooseCommandCore(const char* who, CommandCoreChoice& choice) {
  const HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
  using GetInfo = BOOL(WINAPI*)(PSYSTEM_CPU_SET_INFORMATION, ULONG, PULONG, HANDLE, ULONG);
  const auto get_info = kernel ? reinterpret_cast<GetInfo>(GetProcAddress(kernel, "GetSystemCpuSetInformation")) : nullptr;
  if (!get_info) {
    if (who) REXLOG_WARN("[pace] {}: no CPU set functions on this Windows", who);
    return false;
  }
  ULONG bytes = 0;
  get_info(nullptr, 0, &bytes, GetCurrentProcess(), 0);
  std::vector<uint8_t> buffer(bytes);
  if (!bytes || !get_info(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()), bytes, &bytes, GetCurrentProcess(), 0)) {
    if (who) REXLOG_WARN("[pace] {}: GetSystemCpuSetInformation failed ({})", who, GetLastError());
    return false;
  }
  struct Lp {
    ULONG id;
    uint32_t core, index;
  };
  std::vector<Lp> lps;
  for (ULONG offset = 0; offset < bytes;) {
    const auto* info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
    if (info->Type == CpuSetInformation && info->CpuSet.Group == 0) {
      lps.push_back({info->CpuSet.Id, info->CpuSet.CoreIndex, info->CpuSet.LogicalProcessorIndex});
    }
    offset += info->Size;
  }
  std::vector<uint32_t> cores;
  for (const Lp& lp : lps) {
    if (std::find(cores.begin(), cores.end(), lp.core) == cores.end()) cores.push_back(lp.core);
  }
  if (cores.size() < 4) {
    if (who) REXLOG_INFO("[pace] {}: {} physical cores, needs 4: not applied", who, cores.size());
    return false;
  }
  choice.cores = cores.size();
  choice.own = *std::max_element(cores.begin(), cores.end());
  std::vector<uint32_t> first_seen, later_lps;
  for (const Lp& lp : lps) {
    if (lp.core == choice.own) {
      choice.own_ids.push_back(lp.id);
      choice.own_lp = std::min(choice.own_lp, lp.index);
      continue;
    }
    choice.others.push_back(lp.id);
    if (std::find(first_seen.begin(), first_seen.end(), lp.core) == first_seen.end()) {
      first_seen.push_back(lp.core);
      choice.other_lps.push_back(lp.index);
    } else {
      later_lps.push_back(lp.index);
    }
  }
  choice.other_lps.insert(choice.other_lps.end(), later_lps.begin(), later_lps.end());
  return true;
}

// gpu_cmd_keep_core: a local of the GPU VSync thread, so nothing here is shared. A job would own this object whole.
class CommandCoreKeeper {
 public:
  // Called every 60 vblanks, looks at most every kMinLookSeconds of wall time (the load turbo runs vblanks far faster than
  // 60 Hz, so a vblank count is no clock). The mode is latched at the first call. The object table's lock is held only inside GetObjectsByType; the Windows calls run after it,
  // on references that keep each XThread, and so its host handle, alive.
  void Look(system::KernelState* kernel_state) {
    if (!latched_) {
      latched_ = true;
      mode_ = REXCVAR_GET(gpu_cmd_keep_core);
    }
    const int32_t mode = mode_;
    if (mode == 0 || unavailable_ || REXCVAR_GET(gpu_cmd_own_core)) return;
    if (last_look_ != 0 &&
        double(chrono::Clock::QueryHostTickCount() - last_look_) < kMinLookSeconds * double(chrono::Clock::QueryHostTickFrequency())) {
      return;
    }
    if (!chosen_) {
      set_thread_ = FindSetThreadCpuSets();
      chosen_ = set_thread_ && ChooseCommandCore(nullptr, choice_);
      if (!chosen_) {
        unavailable_ = true;
        return;
      }
    }
    const uint64_t frequency = chrono::Clock::QueryHostTickFrequency();
    const uint64_t began = chrono::Clock::QueryHostTickCount();
    const std::vector<system::object_ref<system::XThread>> threads =
        kernel_state->object_table()->GetObjectsByType<system::XThread>();
    const double wall_s = double(began - last_look_) / double(frequency);
    struct Busy {
      system::XThread* thread;
      void* handle;  // copied during the walk: handle() reads handles_ unlocked
      double share;
    };
    std::vector<Busy> busy;
    std::vector<Seen> seen;
    seen.reserve(threads.size());
    for (const system::object_ref<system::XThread>& thread : threads) {
      // Running: its Create has finished, so thread() is complete. Guest only, so never the command thread (a host thread).
      if (!thread || !thread->is_guest_thread() || !thread->is_running() || !thread->thread()) continue;
      FILETIME created, exited, kernel_time, user_time;
      if (!GetThreadTimes(thread->thread()->native_handle(), &created, &exited, &kernel_time, &user_time)) continue;
      const uint64_t cpu_100ns = FileTimeValue(kernel_time) + FileTimeValue(user_time);
      const uint32_t id = thread->thread_id();
      seen.push_back({id, cpu_100ns});
      if (std::find(placed_.begin(), placed_.end(), id) != placed_.end()) continue;
      const auto before = std::find_if(seen_.begin(), seen_.end(), [id](const Seen& s) { return s.id == id; });
      if (before == seen_.end()) continue;  // the first look at this thread: no share yet
      const double share = double(cpu_100ns - before->cpu_100ns) / 1.0e7 / wall_s;
      if (share > kBusyShare) busy.push_back({thread.get(), thread->thread()->native_handle(), share});
    }
    seen_ = std::move(seen);
    last_look_ = began;
    std::sort(busy.begin(), busy.end(), [](const Busy& a, const Busy& b) { return a.share > b.share; });
    for (const Busy& b : busy) {
      void* const handle = b.handle;
      placed_.push_back(b.thread->thread_id());
      if (mode == 2) {
        const BOOL ok = set_thread_(handle, choice_.others.data(), ULONG(choice_.others.size()));
        const uint64_t look_us = (chrono::Clock::QueryHostTickCount() - began) * 1000000 / frequency;
        REXLOG_INFO("[pace] gpu_cmd_keep_core: thread {:08X} (host {}) at {:.1f}% of a logical CPU: keep-out, selects {} logical "
                    "processors, not core index {}; {}; {} threads looked at in {} us",
                    b.thread->handle(), b.thread->thread()->system_id(), b.share * 100.0, choice_.others.size(), choice_.own,
                    ok ? "ok" : "failed", threads.size(), look_us);
      } else {
        const uint32_t lp = choice_.other_lps[next_lp_++ % choice_.other_lps.size()];
        PROCESSOR_NUMBER number = {};
        number.Number = BYTE(lp);
        const BOOL ok = SetThreadIdealProcessorEx(handle, &number, nullptr);
        const uint64_t look_us = (chrono::Clock::QueryHostTickCount() - began) * 1000000 / frequency;
        REXLOG_INFO("[pace] gpu_cmd_keep_core: thread {:08X} (host {}) at {:.1f}% of a logical CPU: ideal processor {}; {}; {} "
                    "threads looked at in {} us",
                    b.thread->handle(), b.thread->thread()->system_id(), b.share * 100.0, lp, ok ? "ok" : "failed",
                    threads.size(), look_us);
      }
    }
  }

 private:
  static constexpr double kBusyShare = 0.08;  // of one logical CPU, since the last look
  static constexpr double kMinLookSeconds = 2.0;
  struct Seen {
    uint32_t id;  // XThread::thread_id(): counts up, never reused
    uint64_t cpu_100ns;
  };
  static uint64_t FileTimeValue(const FILETIME& time) {
    return (uint64_t(time.dwHighDateTime) << 32) | time.dwLowDateTime;
  }

  CommandCoreChoice choice_;
  SetThreadCpuSets set_thread_ = nullptr;
  bool chosen_ = false, unavailable_ = false, latched_ = false;
  int32_t mode_ = 0;
  std::vector<Seen> seen_;
  std::vector<uint32_t> placed_;  // placed once, never touched again
  uint64_t last_look_ = 0;
  uint32_t next_lp_ = 0;
};

}  // namespace
#endif  // REX_PLATFORM_WIN32

// Called from the command thread itself. State: none kept; Windows holds the CPU sets (a job system would own this choice).
void ApplyCommandThreadOwnCore() {
  const bool own_core = REXCVAR_GET(gpu_cmd_own_core);
  const int32_t keep_core = REXCVAR_GET(gpu_cmd_keep_core);
  if (!own_core && keep_core == 0) return;
#if REX_PLATFORM_WIN32
  if (!own_core) {
    CommandCoreChoice choice;
    if (!ChooseCommandCore("gpu_cmd_keep_core", choice)) return;
    PROCESSOR_NUMBER number = {};
    number.Number = BYTE(choice.own_lp);
    const BOOL ok = SetThreadIdealProcessorEx(GetCurrentThread(), &number, nullptr);
    REXLOG_INFO("[pace] gpu_cmd_keep_core on: mode {}, command thread ideal processor {} (core index {} of {} cores), it keeps every "
                "logical processor; {}", keep_core, choice.own_lp, choice.own, choice.cores, ok ? "ok" : "failed");
    return;
  }
  if (keep_core != 0) REXLOG_INFO("[pace] gpu_cmd_keep_core: gpu_cmd_own_core is on, keep core not applied");
  const HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
  using SetProcess = BOOL(WINAPI*)(HANDLE, const ULONG*, ULONG);
  const auto set_process = kernel ? reinterpret_cast<SetProcess>(GetProcAddress(kernel, "SetProcessDefaultCpuSets")) : nullptr;
  const auto set_thread = FindSetThreadCpuSets();
  if (!set_process || !set_thread) {
    REXLOG_WARN("[pace] gpu_cmd_own_core: no CPU set functions on this Windows");
    return;
  }
  CommandCoreChoice choice;
  if (!ChooseCommandCore("gpu_cmd_own_core", choice)) return;
  const BOOL thread_ok = set_thread(GetCurrentThread(), choice.own_ids.data(), ULONG(choice.own_ids.size()));
  const BOOL process_ok = set_process(GetCurrentProcess(), choice.others.data(), ULONG(choice.others.size()));
  REXLOG_INFO("[pace] gpu_cmd_own_core on: command thread on {} logical processors from {} (core index {}), the process on {} others "
              "of {} cores; thread set {}, process set {}", choice.own_ids.size(), choice.own_lp, choice.own, choice.others.size(),
              choice.cores, thread_ok ? "ok" : "failed", process_ok ? "ok" : "failed");
#else
  REXLOG_INFO("[pace] {}: Windows only", own_core ? "gpu_cmd_own_core" : "gpu_cmd_keep_core");
#endif
}
}  // namespace rex::graphics


REXCVAR_DEFINE_STRING(swap_post_effect, "none", "GPU", "Swap post effect: none, fxaa, fxaa_extreme")
    .allowed({"none", "fxaa", "fxaa_extreme"})
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_BOOL(store_shaders, true, "GPU",
                    "Store shaders persistently and load them when loading games to avoid "
                    "runtime spikes and freezes when playing the game not for the first time.");

// wwe2k14 patch (Session 6, O1): the guest vblank comes from a Sleep(1) polling loop, so each one
// fires 0-2 ms after its deadline (timelines 071/073/074: p99 ~2.5 ms late, single intervals
// p1 14.8 / p99 18.6 ms). The command processor's WAIT_REG_MEM is released by the vblank, so that
// jitter moves its per-frame budget. With this on, the GPU VSync thread sleeps on a high-resolution
// waitable timer until shortly before the deadline and spins the rest. Same deadlines (fixed grid
// from the thread's start), same catch-up of missed vblanks. Read every loop iteration, so it can
// be flipped at any time.
REXCVAR_DEFINE_BOOL(
    gpu_vblank_precise_timer, false, "GPU",
    "Time the guest vblank with a high-resolution waitable timer and a short final spin "
    "instead of Sleep(1) polling (Windows 10 1803+ and vsync on only; otherwise it keeps "
    "the Sleep(1) polling)");
REXCVAR_DEFINE_DOUBLE(
    gpu_vblank_rate_override_hz, 0.0, "GPU",
    "wwe2k14 (Session 14, S3 slow-motion experiment): feed the guest vblank at this rate (Hz) "
    "instead of the video mode's (60); 0 = off. Read every vblank loop, only with vsync on");
// wwe2k14 patch (Session 15, sub-second loads): with vsync off the guest vblank runs at a fixed
// 1 kHz, polled with Sleep(1) (~1-2 ms per wake on Windows, vblanks fired in batches). A title that
// waits for N vblanks per present (WAIT_REG_MEM on its vblank-driven pacing word) is then held to
// ~N ms per frame even when the command processor needs far less (2K14's loading screens: CP busy
// 0.5 ms, frame 2.1 ms). > 0 = with vsync off, run the vblank at this rate on the high-resolution
// timer + spin (the gpu_vblank_precise_timer waiter, whatever that setting says). 0 = 1 kHz,
// Sleep(1) polling, as before. Read every vblank loop.
REXCVAR_DEFINE_DOUBLE(
    gpu_vblank_unthrottled_hz, 0.0, "GPU",
    "wwe2k14 (Session 15): with vsync off, run the guest vblank at this rate (Hz) on the precise "
    "timer instead of 1 kHz Sleep(1) polling; 0 = off (1 kHz)");
REXCVAR_DEFINE_INT32(
    gpu_vblank_precise_timer_spin_us, 1000, "GPU",
    "gpu_vblank_precise_timer: how long before each vblank deadline the timer wakes the "
    "thread to spin the rest, in microseconds (more = more CPU, fewer late vblanks)")
    .range(0, 16000);

namespace {

rex::graphics::CommandProcessor::SwapPostEffect ParseSwapPostEffect(
    const std::string& effect_name) {
  std::string lowered = effect_name;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
    c = static_cast<unsigned char>(std::tolower(c));
    return c == '-' ? '_' : char(c);
  });
  if (lowered == "fxaa") {
    return rex::graphics::CommandProcessor::SwapPostEffect::kFxaa;
  }
  if (lowered == "fxaa_extreme" || lowered == "extreme") {
    return rex::graphics::CommandProcessor::SwapPostEffect::kFxaaExtreme;
  }
  return rex::graphics::CommandProcessor::SwapPostEffect::kNone;
}
}  // namespace

namespace rex::graphics {

// Nvidia Optimus/AMD PowerXpress support.
// These exports force the process to trigger the discrete GPU in multi-GPU
// systems.
// https://developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/OptimusRenderingPolicies.pdf
// https://stackoverflow.com/questions/17458803/amd-equivalent-to-nvoptimusenablement
#if REX_PLATFORM_WIN32
extern "C" {
__declspec(dllexport) uint32_t NvOptimusEnablement = 0x00000001;
__declspec(dllexport) uint32_t AmdPowerXpressRequestHighPerformance = 1;
}  // extern "C"
#endif  // REX_PLATFORM_WIN32

#if REX_PLATFORM_WIN32
namespace {

// wwe2k14 patch (Session 6, O1): waits for a guest vblank deadline on a high-resolution waitable
// timer until spin_us before it, then spins on the host performance counter. The timer does not
// depend on the process timer resolution (raised to 1 ms by SDL's timeBeginPeriod, which Windows 11
// may ignore for occluded windows). The spin reads the host counter, not the guest clock:
// Clock::QueryGuestTickCount takes the tick mutex that the game's mftb reads also take.
class PreciseVblankWaiter {
 public:
  PreciseVblankWaiter() = default;
  PreciseVblankWaiter(const PreciseVblankWaiter&) = delete;
  PreciseVblankWaiter& operator=(const PreciseVblankWaiter&) = delete;
  ~PreciseVblankWaiter() {
    if (timer_) {
      CloseHandle(timer_);
    }
  }

  // Creates the timer on first use. False if the OS has no high-resolution waitable timers (before
  // Windows 10 1803); the caller then keeps the Sleep(1) polling.
  bool Available() {
    if (!timer_ && !unavailable_) {
      timer_ = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                      TIMER_ALL_ACCESS);
      if (!timer_) {
        unavailable_ = true;
        REXGPU_WARN(
            "[s6][O1] No high-resolution waitable timer (error {}), the guest vblank keeps "
            "Sleep(1) polling",
            GetLastError());
      }
    }
    return timer_ != nullptr;
  }

  // Returns once the guest tick count has reached guest_deadline, right away if it already has.
  // spun_us gets the length of the final spin (0 if the timer woke at or after the deadline). False
  // if the timer could not be armed: nothing was waited for and the caller falls back to Sleep(1)
  // this time.
  bool WaitUntil(uint64_t guest_deadline, int32_t spin_us, uint32_t& spun_us) {
    spun_us = 0;
    // The guest clock is the host performance counter times guest_tick_ratio (core/clock.cpp). The
    // guest clock is sampled first, so the host time it stands for is <= host_now and host_deadline
    // is never early.
    uint64_t guest_now = chrono::Clock::QueryGuestTickCount();
    uint64_t host_now = chrono::Clock::QueryHostTickCount();
    if (int64_t(guest_deadline - guest_now) <= 0) {
      return true;
    }
    // Guest ticks per host tick = first / second. Normally at most one vblank interval remains; the
    // caps only keep the arithmetic in range and the thread responsive under odd clock settings
    // (the caller re-checks).
    std::pair<uint64_t, uint64_t> ratio = chrono::Clock::guest_tick_ratio();
    uint64_t host_frequency = chrono::Clock::QueryHostTickFrequency();
    uint64_t remaining_guest = std::min(guest_deadline - guest_now, uint64_t(1) << 32);
    uint64_t remaining_host = (remaining_guest * ratio.second + ratio.first - 1) / ratio.first;
    remaining_host = std::min(remaining_host, host_frequency / 10);
    uint64_t host_deadline = host_now + remaining_host;
    uint64_t spin_ticks = uint64_t(std::max(spin_us, int32_t(0))) * host_frequency / 1000000;
    if (remaining_host > spin_ticks) {
      // Relative due time, in 100 ns units, negative.
      int64_t sleep_100ns = int64_t((remaining_host - spin_ticks) * 10000000 / host_frequency);
      if (sleep_100ns > 0) {
        LARGE_INTEGER due_time;
        due_time.QuadPart = -sleep_100ns;
        if (!SetWaitableTimer(timer_, &due_time, 0, nullptr, nullptr, FALSE)) {
          // Not expected; give up on the timer for good rather than flip between the two waits.
          REXGPU_WARN(
              "[s6][O1] SetWaitableTimer failed (error {}), the guest vblank goes back to Sleep(1) "
              "polling",
              GetLastError());
          CloseHandle(timer_);
          timer_ = nullptr;
          unavailable_ = true;
          return false;
        }
        WaitForSingleObject(timer_, INFINITE);
      }
    }
    uint64_t spin_start = chrono::Clock::QueryHostTickCount();
    uint64_t host_time = spin_start;
    while (host_time < host_deadline) {
      YieldProcessor();
      host_time = chrono::Clock::QueryHostTickCount();
    }
    spun_us = uint32_t((host_time - spin_start) * 1000000 / host_frequency);
    return true;
  }

 private:
  HANDLE timer_ = nullptr;
  bool unavailable_ = false;
};

}  // namespace
#endif  // REX_PLATFORM_WIN32

GraphicsSystem::GraphicsSystem() : vsync_worker_running_(false) {}

GraphicsSystem::~GraphicsSystem() = default;

X_STATUS GraphicsSystem::SetupPresentation(ui::WindowedAppContext* app_context) {
  if (presenter_) {
    return X_STATUS_SUCCESS;
  }

  if (guest_gpu_only_) {
    // wwe2k14 patch (Session 101, lane 233): a guest-GPU-only system builds no provider, so it has nothing to present with.
    REXGPU_ERROR("SetupPresentation refused: guest_gpu_only (no graphics provider)");
    return X_STATUS_UNSUCCESSFUL;
  }

  if (!provider_) {
    CreateProvider(true);
    if (!provider_) {
      REXGPU_ERROR("Unable to create graphics provider");
      return X_STATUS_UNSUCCESSFUL;
    }
    provider_supports_presentation_ = true;
  } else if (!provider_supports_presentation_) {
    // A prior SetupGuestGpu built a headless provider; backends like Vulkan
    // need swapchain support baked in at provider creation time.
    REXGPU_ERROR("SetupPresentation called after headless SetupGuestGpu; call order is reversed");
    return X_STATUS_UNSUCCESSFUL;
  }

  app_context_ = app_context;
  auto loss_cb = [this](bool is_responsible, bool statically_from_ui_thread) {
    OnHostGpuLossFromAnyThread(is_responsible);
  };
  if (app_context_) {
    // Presenter creation must happen on the UI thread.
    app_context_->CallInUIThreadSynchronous(
        [this, loss_cb]() { presenter_ = provider_->CreatePresenter(loss_cb); });
  } else {
    // Offscreen path (e.g. capturing guest output without a window).
    presenter_ = provider_->CreatePresenter(loss_cb);
  }

  if (!presenter_) {
    REXGPU_ERROR("Unable to create presenter");
    return X_STATUS_UNSUCCESSFUL;
  }
  return X_STATUS_SUCCESS;
}

X_STATUS GraphicsSystem::SetupGuestGpu(runtime::FunctionDispatcher* function_dispatcher,
                                       system::KernelState* kernel_state) {
  memory_ = function_dispatcher->memory();
  function_dispatcher_ = function_dispatcher;
  kernel_state_ = kernel_state;

  if (guest_gpu_only_) {
    // wwe2k14 patch (Session 101, lane 233, LEAVE_THE_COMMAND_STREAM.md 3b): no provider, no command processor. Only the MMIO range and
    // the VSync thread below; every command_processor_ use in this file is null-guarded.
    REXGPU_INFO("[gpu] guest_gpu_only: no graphics provider, no command processor (MMIO + VSync only)");
  } else {
  // Headless path: no one set up presentation, so build a no-presentation
  // provider just for the command processor.
  if (!provider_) {
    CreateProvider(false);
    provider_supports_presentation_ = false;
  }

  // Create command processor. This will spin up a thread to process all
  // incoming ringbuffer packets.
  command_processor_ = CreateCommandProcessor();
  if (!command_processor_->Initialize()) {
    REXGPU_ERROR("Unable to initialize command processor");
    return X_STATUS_UNSUCCESSFUL;
  }
  command_processor_->SetDesiredSwapPostEffect(ParseSwapPostEffect(REXCVAR_GET(swap_post_effect)));
  }

  // Register GPU MMIO handlers
  // GPU registers are at 0x7FC80000-0x7FCFFFFF
  memory_->AddVirtualMappedRange(0x7FC80000,  // base address
                                 0xFFFF0000,  // mask
                                 0x0000FFFF,  // size (64KB)
                                 this,        // context (GraphicsSystem*)
                                 reinterpret_cast<runtime::MMIOReadCallback>(ReadRegisterThunk),
                                 reinterpret_cast<runtime::MMIOWriteCallback>(WriteRegisterThunk));

  if (guest_vblank_ours_) {
    // wwe2k14 patch (Session 101, lane 260, LEAVE_THE_COMMAND_STREAM.md 3e): the app ticks the guest vblank (src/owner_timing.h); no VSync thread, so nothing
    // here dispatches interrupt source 0. Shutdown() already handles an absent thread.
    REXGPU_INFO("[gpu] guest_vblank_ours: no VSync thread (the app dispatches the guest vblank)");
    return X_STATUS_SUCCESS;
  }

  // Guest vblank timer based on the configured guest video mode.
  vsync_worker_running_ = true;
  vsync_worker_thread_ = system::object_ref<system::XHostThread>(
      new system::XHostThread(kernel_state_, 128 * 1024, 0, [this]() {
        system::X_VIDEO_MODE video_mode;
        kernel::xboxkrnl::VdQueryVideoMode(&video_mode);
        double refresh_rate_hz = std::max(1.0, double(float(video_mode.refresh_rate)));
        uint64_t guest_tick_frequency = chrono::Clock::guest_tick_frequency();
        uint64_t vsync_interval_ticks =
            std::max(uint64_t(1), uint64_t(double(guest_tick_frequency) / refresh_rate_hz));
        uint64_t no_vsync_interval_ticks = std::max(uint64_t(1), guest_tick_frequency / 1000);
        uint64_t last_frame_time = chrono::Clock::QueryGuestTickCount();
        // wwe2k14 patch (Session 6, O1): how the wait before the next vblank went (timeline b/c),
        // and stats of the precise timer (logged only while it is in use).
        uint32_t wait_precise = 0;
        uint32_t wait_spun_us = 0;
        bool precise_active = false;
        uint32_t stats_vblanks = 0, stats_catch_up = 0, stats_late_500us = 0;
        uint64_t stats_late_sum_us = 0, stats_spin_sum_us = 0;
        uint32_t stats_late_max_us = 0;
#if REX_PLATFORM_WIN32
        PreciseVblankWaiter precise_waiter;
        CommandCoreKeeper core_keeper;  // wwe2k14 patch (Session 95, round 7): gpu_cmd_keep_core
        uint32_t core_keeper_vblanks = 0;
#endif
        uint64_t det_timeouts = 0;
        bool noring_warned = false;  // wwe2k14 patch (Session 101, lane 233)
        while (vsync_worker_running_) {
          // wwe2k14 patch (Session 15): det_frames: the next vblank only once the guest is quiescent (all guest
          // threads blocked, the CP idle or waiting); with vsync on, also not before the 60 Hz deadline.
          // wwe2k14 patch (Session 15): det_serial: a vblank only when nothing is runnable (idle); the vblank's
          // interrupt then runs, and the scheduler picks who goes next.
          if (det::SerialEnabled()) {
            if (!det::SerialWaitIdle(0)) {
              ++det_timeouts;
              det::NoteVblankTimeout();  // wwe2k14 patch (Session 77): the CSV marks it (not only the log)
              if (det_timeouts <= 20 || det_timeouts % 100 == 0) {
                REXGPU_WARN("[det] serial: vblank {} fired while a guest thread still runs #{}", det::Vblanks() + 1,
                            det_timeouts);
              }
            }
            if (REXCVAR_GET(vsync)) {
              while (vsync_worker_running_ &&
                     chrono::Clock::QueryGuestTickCount() - last_frame_time < vsync_interval_ticks) {
                rex::thread::Sleep(std::chrono::milliseconds(1));
              }
            }
            last_frame_time = chrono::Clock::QueryGuestTickCount();
            MarkVblank(0, 0, 0);
            det::NoteVblank();
            det::RunVblankHooks();
            det::RunAudioForVblank();
            det::SerialAfterVblank();
            continue;
          }
          if (det::Enabled()) {
            if (!det::WaitQuiescent(0, 0)) {
              ++det_timeouts;
              det::NoteVblankTimeout();  // wwe2k14 patch (Session 77): the CSV marks it (not only the log)
              if (det_timeouts <= 20 || det_timeouts % 100 == 0) {
                REXGPU_WARN("[det] vblank {} fired without quiescence ({} guest threads running: host tids {}) #{}",
                            det::Vblanks() + 1, det::RunningGuestThreads(), det::DescribeRunning(), det_timeouts);
              }
            }
            if (REXCVAR_GET(vsync)) {
              while (vsync_worker_running_ &&
                     chrono::Clock::QueryGuestTickCount() - last_frame_time < vsync_interval_ticks) {
                rex::thread::Sleep(std::chrono::milliseconds(1));
              }
            }
            last_frame_time = chrono::Clock::QueryGuestTickCount();
            // At a quiescent point: the frame thread waiting at its barrier is released (with a vblank only if none
            // fired since its previous barrier); otherwise everyone waits on something only a vblank can give.
            const bool barrier = det::BarrierWaiting();
            if (!barrier || det::VblankDueAtBarrier()) {
              MarkVblank(0, 0, 0);
              det::NoteVblank();
            }
            if (barrier) det::ReleaseBarrier();
            continue;
          }
          uint64_t current_time = chrono::Clock::QueryGuestTickCount();
          uint64_t interval_ticks =
              REXCVAR_GET(vsync) ? vsync_interval_ticks : no_vsync_interval_ticks;
          // wwe2k14 patch (Session 14, S3 slow-motion experiment): a guest vblank rate override
          // (Hz, 0 = the video mode's rate), read every loop like vsync, only while vsync is on.
          if (const double hz = REXCVAR_GET(gpu_vblank_rate_override_hz);
              hz > 0.0 && REXCVAR_GET(vsync)) {
            interval_ticks = std::max(uint64_t(1), uint64_t(double(guest_tick_frequency) / hz));
          }
          // wwe2k14 patch (Session 15): the vsync-off rate (gpu_vblank_unthrottled_hz).
          const double unthrottled_hz = REXCVAR_GET(vsync) ? 0.0 : REXCVAR_GET(gpu_vblank_unthrottled_hz);
          if (unthrottled_hz > 0.0) {
            interval_ticks =
                std::max(uint64_t(1), uint64_t(double(guest_tick_frequency) / unthrottled_hz));
          }
          bool first_in_batch = true;
          while (current_time - last_frame_time >= interval_ticks) {
            // wwe2k14 patch (Session 6, O1): lateness = guest time from this vblank's deadline to
            // now. The first vblank of a batch uses current_time, read just above; catch-up ones
            // re-read the clock, but only while the timeline records (the value is unused
            // otherwise).
            uint64_t deadline = last_frame_time + interval_ticks;
            uint64_t fired_time = current_time;
            if (!first_in_batch && perf::timeline::Enabled()) {
              fired_time = chrono::Clock::QueryGuestTickCount();
            }
            uint32_t late_us =
                uint32_t(std::min(fired_time - deadline, guest_tick_frequency * 4000) * 1000000 /
                         guest_tick_frequency);
            if (wait_precise) {
              if (first_in_batch) {
                ++stats_vblanks;
                stats_late_sum_us += late_us;
                stats_late_max_us = std::max(stats_late_max_us, late_us);
                stats_late_500us += late_us >= 500 ? 1 : 0;
                stats_spin_sum_us += wait_spun_us;
              } else {
                ++stats_catch_up;
              }
            }
            MarkVblank(late_us, wait_precise, first_in_batch ? wait_spun_us : 0);
            first_in_batch = false;
            last_frame_time += interval_ticks;
#if REX_PLATFORM_WIN32
            ++core_keeper_vblanks;
#endif
          }
#if REX_PLATFORM_WIN32
          // wwe2k14 patch (Session 95, round 7): right after the vblanks, the furthest point from the next deadline.
          if (core_keeper_vblanks >= 60) {
            core_keeper_vblanks = 0;
            core_keeper.Look(kernel_state_);
          }
#endif
          if (guest_gpu_only_ && !noring_warned && REXCVAR_GET(gpu_d3d_noring_shared) != 0) {
            // wwe2k14 patch (Session 101, lane 233): the no-ring queue's only consumer is the command processor.
            noring_warned = true;
            REXGPU_WARN("[gpu] guest_gpu_only with gpu_d3d_noring_shared set: nothing consumes the queue; fences / flips are never written unless the app answers them");
          }
          if (stats_vblanks >= 1800) {
            REXGPU_INFO(
                "[s6][O1] precise vblank timer, last {} vblanks: late mean {} us, max {} us, {} at "
                ">=500 us late, {} catch-up vblanks, spin mean {} us",
                stats_vblanks, stats_late_sum_us / stats_vblanks, stats_late_max_us,
                stats_late_500us, stats_catch_up, stats_spin_sum_us / stats_vblanks);
            stats_vblanks = stats_catch_up = stats_late_500us = stats_late_max_us = 0;
            stats_late_sum_us = stats_spin_sum_us = 0;
          }
          // wwe2k14 patch (Session 6, O1): wait for the next deadline on the precise timer when
          // enabled. Only for the vsync interval: with vsync off the "vblank" runs at 1 kHz and the
          // spin would take a whole core. Both waits leave last_frame_time alone, so switching
          // between them keeps the same grid.
          bool waited = false;
#if REX_PLATFORM_WIN32
          if (((REXCVAR_GET(gpu_vblank_precise_timer) && REXCVAR_GET(vsync)) ||
               unthrottled_hz > 0.0) &&
              precise_waiter.Available()) {
            waited = precise_waiter.WaitUntil(last_frame_time + interval_ticks,
                                              REXCVAR_GET(gpu_vblank_precise_timer_spin_us),
                                              wait_spun_us);
          }
#endif
          if (waited != precise_active) {
            precise_active = waited;
            REXGPU_INFO(
                "[s6][O1] Guest vblank timer: {}",
                waited ? "precise (high-resolution waitable timer + spin)" : "Sleep(1) polling");
          }
          wait_precise = waited ? 1 : 0;
          if (!waited) {
            wait_spun_us = 0;
            rex::thread::Sleep(std::chrono::milliseconds(1));
          }
        }
        return 0;
      }));
  // TODO: set_can_debugger_suspend not yet ported
  // vsync_worker_thread_->set_can_debugger_suspend(true);
  vsync_worker_thread_->set_name("GPU VSync");
  vsync_worker_thread_->Create();

  return X_STATUS_SUCCESS;
}

void GraphicsSystem::Shutdown() {
  if (command_processor_) {
    command_processor_->Shutdown();
    command_processor_.reset();
  }

  if (vsync_worker_thread_) {
    vsync_worker_running_ = false;
    vsync_worker_thread_->Wait(0, 0, 0, nullptr);
    vsync_worker_thread_.reset();
  }

  if (presenter_) {
    if (app_context_) {
      app_context_->CallInUIThreadSynchronous([this]() { presenter_.reset(); });
    }
    // If there's no app context (thus the presenter is owned by the thread that
    // initialized the GraphicsSystem) or can't be queueing UI thread calls
    // anymore, shutdown anyway.
    presenter_.reset();
  }

  provider_.reset();
}

void GraphicsSystem::OnHostGpuLossFromAnyThread([[maybe_unused]] bool is_responsible) {
  // TODO(Triang3l): Somehow gain exclusive ownership of the Provider (may be
  // used by the command processor, the presenter, and possibly anything else,
  // it's considered free-threaded, except for lifetime management which will be
  // involved in this case) and reset it so a new host GPU API device is
  // created. Then ask the command processor to reset itself in its thread, and
  // ask the UI thread to reset the Presenter (the UI thread manages its
  // lifetime - but if there's no WindowedAppContext, either don't reset it as
  // in this case there's no user who needs uninterrupted gameplay, or somehow
  // protect it with a mutex so any thread can be considered a UI thread and
  // reset).
  if (host_gpu_loss_reported_.test_and_set(std::memory_order_relaxed)) {
    return;
  }
  rex::FatalError("Graphics device lost (probably due to an internal error)");
}

uint32_t GraphicsSystem::ReadRegisterThunk(void* ppc_context, GraphicsSystem* gs, uint32_t addr) {
  return gs->ReadRegister(addr);
}

void GraphicsSystem::WriteRegisterThunk(void* ppc_context, GraphicsSystem* gs, uint32_t addr,
                                        uint32_t value) {
  gs->WriteRegister(addr, value);
}

uint32_t GraphicsSystem::ReadRegister(uint32_t addr) {
  uint32_t r = (addr & 0xFFFF) / 4;
  rex::graphics::free_counters::BumpIndex(rex::graphics::free_counters::g_mmio_reads, r);  // (Session 92 M5a, log only)

  switch (r) {
    case 0x0F00:  // RB_EDRAM_TIMING
      return 0x08100748;
    case 0x0F01:  // RB_BC_CONTROL
      return 0x0000200E;
    case 0x194C: {  // R500_D1MODE_V_COUNTER
      system::X_VIDEO_MODE video_mode;
      kernel::xboxkrnl::VdQueryVideoMode(&video_mode);
      return std::min(uint32_t(video_mode.display_height), uint32_t(0x0FFF));
    }
    case 0x1951:    // interrupt status
      return 1;     // vblank
    case 0x1961: {  // AVIVO_D1MODE_VIEWPORT_SIZE
      // Maximum [width(0x0FFF), height(0x0FFF)].
      system::X_VIDEO_MODE video_mode;
      kernel::xboxkrnl::VdQueryVideoMode(&video_mode);
      uint32_t viewport_width = std::min(uint32_t(video_mode.display_width), uint32_t(0x0FFF));
      uint32_t viewport_height = std::min(uint32_t(video_mode.display_height), uint32_t(0x0FFF));
      return (viewport_width << 16) | viewport_height;
    }
    default:
      if (!register_file_.GetRegisterInfo(r)) {
        REXGPU_DEBUG("GPU: Read from unknown register ({:04X})", r);
      }
  }

  assert_true(r < RegisterFile::kRegisterCount);
  if (r >= 0x2000) CommandProcessor::NfNoteHighRead(r, "the guest (MMIO)");  // wwe2k14 patch (Session 79): census
  return register_file_.values[r];
}

void GraphicsSystem::WriteRegister(uint32_t addr, uint32_t value) {
  uint32_t r = (addr & 0xFFFF) / 4;

  switch (r) {
    case 0x01C5:  // CP_RB_WPTR
      if (command_processor_) {  // wwe2k14 patch (Session 101, lane 233): none when guest_gpu_only
        command_processor_->UpdateWritePointer(value);
      }
      break;
    case 0x1844:  // AVIVO_D1GRPH_PRIMARY_SURFACE_ADDRESS
      break;
    default:
      REXGPU_WARN("Unknown GPU register {:04X} write: {:08X}", r, value);
      break;
  }

  assert_true(r < RegisterFile::kRegisterCount);
  register_file_.values[r] = value;
}

void GraphicsSystem::InitializeRingBuffer(uint32_t ptr, uint32_t size_log2) {
  if (!command_processor_) return;  // wwe2k14 patch (Session 101, lane 233): guest_gpu_only
  command_processor_->InitializeRingBuffer(ptr, size_log2);
}

void GraphicsSystem::EnableReadPointerWriteBack(uint32_t ptr, uint32_t block_size_log2) {
  if (!command_processor_) return;  // wwe2k14 patch (Session 101, lane 233): guest_gpu_only
  command_processor_->EnableReadPointerWriteBack(ptr, block_size_log2);
}

void GraphicsSystem::SetInterruptCallback(uint32_t callback, uint32_t user_data) {
  // wwe2k14 patch (Session 95, RACE R3): the vsync thread is already running here; on ARM it must never see the callback before its data.
  interrupt_callback_data_.store(user_data, std::memory_order_relaxed);
  interrupt_callback_.store(callback, std::memory_order_release);
  REXGPU_INFO("SetInterruptCallback({:08X}, {:08X})", callback, user_data);
}

void GraphicsSystem::DispatchInterruptCallback(uint32_t source, uint32_t cpu) {
  const uint32_t interrupt_callback = interrupt_callback_.load(std::memory_order_acquire);  // (Session 95, RACE R3: pairs with the release above)
  if (!interrupt_callback) {
    return;
  }

  auto thread = system::XThread::GetCurrentThread();
  assert_not_null(thread);

  // Pick a CPU, if needed. We're going to guess 2. Because.
  if (cpu == 0xFFFFFFFF) {
    cpu = 2;
  }
  thread->SetActiveCpu(cpu);

  // REXGPU_INFO("Dispatching GPU interrupt at {:08X} w/ mode {} on cpu {}",
  //          interrupt_callback, source, cpu);

  uint64_t args[] = {source, interrupt_callback_data_.load(std::memory_order_relaxed)};
  perf::timeline::Scope timeline_scope("intr_b", "intr_e", source, cpu);
  function_dispatcher_->ExecuteInterrupt(thread->thread_state(), interrupt_callback, args,
                                         rex::countof(args));
}

void GraphicsSystem::MarkVblank(uint32_t late_us, uint32_t precise_timer, uint32_t spin_us) {
  // TODO: Enable profiling once ported
  // SCOPE_profile_cpu_f("gpu");

  // Increment vblank counter (so the game sees us making progress).
  if (command_processor_) {
    command_processor_->increment_counter();
  }
  // wwe2k14 patch (Session 6, O1): a/b/c as documented in graphics_system.h (were all 0 before).
  perf::timeline::Record("vblank", late_us, precise_timer, spin_us);

  // TODO(benvanik): we shouldn't need to do the dispatch here, but there's
  //     something wrong and the CP will block waiting for code that
  //     needs to be run in the interrupt.
  DispatchInterruptCallback(0, 2);
  // wwe2k14 patch (Session 92, M5b): the guest's vblank handler has run (the flip word is cleared): wake the pacing wait.
  D3dNoringVblankSignal();
}

void GraphicsSystem::ClearCaches() {
  if (!command_processor_) return;  // wwe2k14 patch (Session 101, lane 233): guest_gpu_only
  command_processor_->CallInThread([&]() { command_processor_->ClearCaches(); });
}

void GraphicsSystem::InvalidateGpuMemory() {
  if (!command_processor_) return;  // wwe2k14 patch (Session 101, lane 233): guest_gpu_only
  command_processor_->CallInThread([&]() { command_processor_->InvalidateGpuMemory(); });
}

void GraphicsSystem::InitializeShaderStorage(const std::filesystem::path& cache_root,
                                             uint32_t title_id, bool blocking) {
  if (!REXCVAR_GET(store_shaders)) {
    return;
  }
  if (!command_processor_) return;  // wwe2k14 patch (Session 101, lane 233): guest_gpu_only (no shader storage without a backend)
  if (blocking) {
    if (command_processor_->is_paused()) {
      // Safe to run on any thread while the command processor is paused, no
      // race condition.
      command_processor_->InitializeShaderStorage(cache_root, title_id, true);
    } else {
      rex::thread::Fence fence;
      command_processor_->CallInThread([this, cache_root, title_id, &fence]() {
        command_processor_->InitializeShaderStorage(cache_root, title_id, true);
        fence.Signal();
      });
      fence.Wait();
    }
  } else {
    command_processor_->CallInThread([this, cache_root, title_id]() {
      command_processor_->InitializeShaderStorage(cache_root, title_id, false);
    });
  }
}

void GraphicsSystem::Pause() {
  paused_ = true;
  if (command_processor_) command_processor_->Pause();  // wwe2k14 patch (Session 101, lane 233): none when guest_gpu_only
}

void GraphicsSystem::Resume() {
  paused_ = false;
  if (command_processor_) command_processor_->Resume();  // wwe2k14 patch (Session 101, lane 233): none when guest_gpu_only
}

bool GraphicsSystem::Save(::rex::stream::ByteStream* stream) {
  stream->Write<uint32_t>(interrupt_callback_.load(std::memory_order_relaxed));
  stream->Write<uint32_t>(interrupt_callback_data_.load(std::memory_order_relaxed));
  return command_processor_ ? command_processor_->Save(stream) : true;  // wwe2k14 patch (Session 101, lane 233)
}

bool GraphicsSystem::Restore(::rex::stream::ByteStream* stream) {
  const uint32_t interrupt_callback = stream->Read<uint32_t>();
  interrupt_callback_data_.store(stream->Read<uint32_t>(), std::memory_order_relaxed);
  interrupt_callback_.store(interrupt_callback, std::memory_order_release);
  return command_processor_ ? command_processor_->Restore(stream) : true;  // wwe2k14 patch (Session 101, lane 233)
}

}  // namespace rex::graphics
