/**
 * wwe2k14 patch (Session 15, determinism): deterministic frames mode. See include/rex/det.h.
 */
#include <rex/cvar.h>
#include <rex/det.h>
#include <rex/logging.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#if defined(_WIN32)
#include "platform_win.h"
#endif

#if defined(_MSC_VER) || defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

REXCVAR_DEFINE_BOOL(det_frames, false, "GPU",
                    "wwe2k14 (Session 15): deterministic frames: a vblank only once all guest threads are blocked and "
                    "the command processor waits; guest time derived from the vblank count; timed guest waits end at "
                    "vblanks. Set at startup");
REXCVAR_DEFINE_INT32(det_frames_settle_us, 200, "GPU",
                     "det_frames: how long the guest must stay quiescent before the next vblank (microseconds)");
REXCVAR_DEFINE_INT32(det_frames_timeout_ms, 250, "GPU",
                     "det_frames: longest wait for quiescence before a vblank is fired anyway (logged)");
// wwe2k14 patch (Session 27.5): the virtual time one det vblank stands for. 60 = 1/60 s (Session 15, unchanged). The
// app keeps it at the guest's own update rate (one det vblank per frame at a step rate above 60), so video, audio and
// timed waits follow game time in replays and slow-motion re-shoots. Read at each det vblank; a change rebases.
REXCVAR_DEFINE_INT32(det_frames_time_hz, 60, "GPU",
                     "det_frames: virtual time per det vblank = 1/N s (default 60); the guest timebase, system time, "
                     "uptime, timed waits and the audio pumps follow it. Read at each det vblank");
// wwe2k14 patch (Session 77, cross-device tester check): include/rex/det.h.
REXCVAR_DEFINE_BOOL(det_time_constant, false, "GPU",
                    "det_frames: the virtual clocks start from constant bases (timebase 1 s, uptime 1000 ms, system "
                    "time 2013-10-29 00:00 UTC), not the host's uptime and date; guest callbacks on host threads read "
                    "them too (include/rex/det.h)");
REXCVAR_DEFINE_BOOL(det_fs_constant, false, "GPU",
                    "Constant file times, and a disc image's attributes / volume size as a read-only folder's, so a "
                    "folder dump, a disc image and any machine's save folder answer the same (include/rex/det.h)");

namespace rex::det {
namespace {
std::atomic<int32_t> g_running{0};
std::atomic<bool> g_cp_waiting{true};
std::atomic<uint64_t> g_vblanks{0};
std::mutex g_mutex;
std::condition_variable g_cv;
thread_local bool t_guest = false;
thread_local int32_t t_wait_depth = 0;
thread_local uint64_t t_read_vblank = ~0ull;
thread_local uint64_t t_reads = 0;
std::atomic<bool> g_base_ticks_set{false}, g_base_sys_set{false};
// Per guest thread: host thread id and whether it is running (for the timeout log).
constexpr int kSlots = 128;
std::atomic<uint32_t> g_slot_tid[kSlots];
std::atomic<int32_t> g_slot_running[kSlots];
std::atomic<int32_t> g_slot_count{0};
thread_local int32_t t_slot = -1;
inline uint32_t HostTid();
std::atomic<uint64_t> g_base_ticks{0}, g_base_sys{0};

// wwe2k14 patch (Session 27.5, det_frames_time_hz): virtual time as segments of constant rate. Until the first change
// (g_time_rebased false) every reader uses the Session 15 formulas unchanged, so a run that stays at 60 is
// bit-identical. A segment starts at det vblank vb0 with the accumulated values at vb0 and lasts while hz holds. The
// vblank thread rebases (NoteVblank) while readers on guest threads read: a sequence lock keeps them consistent.
std::atomic<bool> g_time_rebased{false};
std::atomic<uint32_t> g_time_seq{0};
std::atomic<uint64_t> g_seg_vb0{0}, g_seg_ticks{0}, g_seg_ms{0}, g_seg_sys{0}, g_seg_us{0};
std::atomic<int32_t> g_seg_hz{60};
std::atomic<uint64_t> g_tick_freq{0};  // guest timebase frequency (from the first VirtualGuestTicks call)

struct Segment {
  uint64_t vb0, ticks, ms, sys, us;
  int32_t hz;
};
Segment ReadSegment() {
  Segment s;
  for (;;) {
    const uint32_t a = g_time_seq.load(std::memory_order_acquire);
    if (a & 1) {
      std::this_thread::yield();
      continue;
    }
    s = {g_seg_vb0.load(std::memory_order_relaxed), g_seg_ticks.load(std::memory_order_relaxed),
         g_seg_ms.load(std::memory_order_relaxed),  g_seg_sys.load(std::memory_order_relaxed),
         g_seg_us.load(std::memory_order_relaxed),  g_seg_hz.load(std::memory_order_relaxed)};
    std::atomic_thread_fence(std::memory_order_acquire);
    if (g_time_seq.load(std::memory_order_relaxed) == a) return s;
  }
}
uint64_t SysPerVblank(int32_t hz) { return hz == 60 ? 166667ull : 10000000ull / uint64_t(hz); }
// Values at det vblank vb in segment s (vb >= s.vb0).
uint64_t SegTicks(const Segment& s, uint64_t vb, uint64_t freq) { return s.ticks + (vb - s.vb0) * (freq / s.hz); }
uint64_t SegMs(const Segment& s, uint64_t vb) { return s.ms + (vb - s.vb0) * 1000 / uint64_t(s.hz); }
uint64_t SegSys(const Segment& s, uint64_t vb) { return s.sys + (vb - s.vb0) * SysPerVblank(s.hz); }
uint64_t SegUs(const Segment& s, uint64_t vb) { return s.us + (vb - s.vb0) * 1000000 / uint64_t(s.hz); }

// Vblank thread, before a det vblank is counted: start a new segment if det_frames_time_hz changed.
void MaybeRebase() {
  int32_t hz = REXCVAR_GET(det_frames_time_hz);
  if (hz < 1) hz = 60;
  if (hz > 100000) hz = 100000;
  const int32_t old_hz = g_seg_hz.load(std::memory_order_relaxed);
  if (hz == old_hz) return;
  const uint64_t vb = g_vblanks.load(std::memory_order_acquire);
  Segment s = g_time_rebased.load(std::memory_order_acquire)
                  ? ReadSegment()
                  : Segment{0, 0, 0, 0, 0, 60};  // the Session 15 formulas are segment 0 at 60
  const uint64_t freq = g_tick_freq.load(std::memory_order_acquire);
  const Segment n{vb, SegTicks(s, vb, freq), SegMs(s, vb), SegSys(s, vb), SegUs(s, vb), hz};
  g_time_seq.fetch_add(1, std::memory_order_acq_rel);  // odd: writing
  g_seg_vb0.store(n.vb0, std::memory_order_relaxed);
  g_seg_ticks.store(n.ticks, std::memory_order_relaxed);
  g_seg_ms.store(n.ms, std::memory_order_relaxed);
  g_seg_sys.store(n.sys, std::memory_order_relaxed);
  g_seg_us.store(n.us, std::memory_order_relaxed);
  g_seg_hz.store(n.hz, std::memory_order_relaxed);
  g_time_seq.fetch_add(1, std::memory_order_release);  // even: done
  g_time_rebased.store(true, std::memory_order_release);
  REXLOG_INFO("[det] virtual time: {} Hz per det vblank from vblank {} (was {} Hz; virtual {:.3f} s)", hz, vb, old_hz,
              double(n.us) / 1e6);
}

inline uint32_t HostTid() {
#if defined(_WIN32)
  return uint32_t(GetCurrentThreadId());  // the [tNNNN] id of the log lines
#else
  return uint32_t(std::hash<std::thread::id>{}(std::this_thread::get_id()));
#endif
}

inline void Pause() {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(_M_X64)
  _mm_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield");  // wwe2k14 patch (Session 77): the spin-wait hint on ARM64 (was nothing)
#endif
}
}  // namespace

bool Enabled() { return REXCVAR_GET(det_frames); }

void GuestThreadStarted() {
  if (t_guest) return;
  t_guest = true;
  t_wait_depth = 0;
  if (t_slot < 0) {
    const int32_t i = g_slot_count.fetch_add(1);
    if (i < kSlots) {
      t_slot = i;
      g_slot_tid[i].store(HostTid());
    }
  }
  if (t_slot >= 0) g_slot_running[t_slot].store(1);
  g_running.fetch_add(1, std::memory_order_acq_rel);
}

void GuestThreadExiting() {
  if (!t_guest) return;
  if (t_wait_depth == 0) g_running.fetch_sub(1, std::memory_order_acq_rel);
  if (t_slot >= 0) g_slot_running[t_slot].store(0);
  t_guest = false;
  t_wait_depth = 0;
}

bool IsGuestThread() { return t_guest; }

// wwe2k14 patch (Session 77, cross-device tester check): det_time_constant / det_fs_constant (include/rex/det.h).
namespace {
thread_local int32_t t_callback_depth = 0;
}  // namespace
bool TimeConstant() { return Enabled() && REXCVAR_GET(det_time_constant); }
uint64_t ConstantSystemTimeBase() { return 130274784000000000ull; }  // 2013-10-29 00:00:00 UTC as a FILETIME
bool ReadsVirtualTime() { return t_guest || (t_callback_depth > 0 && TimeConstant()); }
GuestCallbackScope::GuestCallbackScope() { ++t_callback_depth; }
GuestCallbackScope::~GuestCallbackScope() { --t_callback_depth; }
bool FsConstant() { return REXCVAR_GET(det_fs_constant); }

void EnterWait() {
  if (!t_guest) return;
  if (t_wait_depth++ == 0) {
    if (t_slot >= 0) g_slot_running[t_slot].store(0);
    g_running.fetch_sub(1, std::memory_order_acq_rel);
  }
}

void LeaveWait() {
  if (!t_guest || t_wait_depth <= 0) return;
  if (--t_wait_depth == 0) {
    if (t_slot >= 0) g_slot_running[t_slot].store(1);
    g_running.fetch_add(1, std::memory_order_acq_rel);
  }
}

void SetCpWaiting(bool waiting) { g_cp_waiting.store(waiting, std::memory_order_release); }
bool CpWaiting() { return g_cp_waiting.load(std::memory_order_acquire); }

uint64_t Vblanks() { return g_vblanks.load(std::memory_order_acquire); }

void NoteVblank() {
  MaybeRebase();  // wwe2k14 patch (Session 27.5): det_frames_time_hz
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_vblanks.fetch_add(1, std::memory_order_acq_rel);
  }
  g_cv.notify_all();
}

// wwe2k14 patch (Session 77, cross-device): timed-out vblanks and the effective renderer, for the det hash CSV.
namespace {
std::atomic<uint64_t> g_vblank_timeouts{0};
std::mutex g_renderer_mutex;
std::string g_renderer_info;
}  // namespace

void NoteVblankTimeout() { g_vblank_timeouts.fetch_add(1, std::memory_order_acq_rel); }
uint64_t VblankTimeouts() { return g_vblank_timeouts.load(std::memory_order_acquire); }

void SetRendererInfo(const std::string& info) {
  std::lock_guard<std::mutex> lock(g_renderer_mutex);
  g_renderer_info = info;
}
std::string RendererInfo() {
  std::lock_guard<std::mutex> lock(g_renderer_mutex);
  return g_renderer_info;
}

void WaitVblanks(uint64_t target) {
  WaitScope scope;
  std::unique_lock<std::mutex> lock(g_mutex);
  g_cv.wait(lock, [&] { return g_vblanks.load(std::memory_order_acquire) >= target || !Enabled(); });
}

uint64_t VblanksForMs(uint64_t ms) {
  const uint64_t hz = uint64_t(TimeHz());  // wwe2k14 patch (Session 27.5): 60 unless det_frames_time_hz changed
  return std::max<uint64_t>(1, (ms * hz + 999) / 1000);
}

int32_t TimeHz() { return g_time_rebased.load(std::memory_order_acquire) ? g_seg_hz.load(std::memory_order_acquire) : 60; }

uint64_t VirtualMicros() {
  const uint64_t vb = g_vblanks.load(std::memory_order_acquire);
  if (!g_time_rebased.load(std::memory_order_acquire)) return vb * 1000000 / 60;
  const Segment s = ReadSegment();
  return SegUs(s, std::max(vb, s.vb0));
}

bool WaitQuiescent(uint32_t settle_us, uint32_t timeout_ms) {
  if (settle_us == 0) settle_us = uint32_t(std::max(1, REXCVAR_GET(det_frames_settle_us)));
  if (timeout_ms == 0) timeout_ms = uint32_t(std::max(1, REXCVAR_GET(det_frames_timeout_ms)));
  using clock = std::chrono::steady_clock;
  const auto start = clock::now();
  auto stable_since = clock::time_point::min();
  uint32_t spins = 0;
  for (;;) {
    const auto now = clock::now();
    const bool quiet =
        g_running.load(std::memory_order_acquire) <= 0 && g_cp_waiting.load(std::memory_order_acquire);
    if (quiet) {
      if (stable_since == clock::time_point::min()) {
        stable_since = now;
      } else if (now - stable_since >= std::chrono::microseconds(settle_us)) {
        return true;
      }
    } else {
      stable_since = clock::time_point::min();
    }
    if (now - start >= std::chrono::milliseconds(timeout_ms)) return false;
    if (++spins % 64 == 0) {
      std::this_thread::yield();
    } else {
      Pause();
    }
  }
}

// Frame barrier (see det.h). Guarded by g_mutex.
namespace {
bool g_barrier_waiting = false;
uint64_t g_barrier_release = 0;
uint64_t g_vblanks_at_last_barrier = ~0ull;
}  // namespace

void FrameBarrier() {
  WaitScope scope;
  std::unique_lock<std::mutex> lock(g_mutex);
  g_barrier_waiting = true;
  const uint64_t release = g_barrier_release;
  g_cv.wait(lock, [&] { return g_barrier_release != release || !Enabled(); });
}

bool BarrierWaiting() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_barrier_waiting;
}

bool VblankDueAtBarrier() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_vblanks.load(std::memory_order_acquire) == g_vblanks_at_last_barrier;
}

void ReleaseBarrier() {
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_barrier_waiting) return;
    g_barrier_waiting = false;
    g_vblanks_at_last_barrier = g_vblanks.load(std::memory_order_acquire);
    ++g_barrier_release;
  }
  g_cv.notify_all();
}

int32_t RunningGuestThreads() { return g_running.load(std::memory_order_acquire); }

std::string DescribeRunning() {
  std::string out;
  const int32_t n = std::min(g_slot_count.load(), kSlots);
  for (int32_t i = 0; i < n; ++i) {
    if (g_slot_running[i].load()) {
      if (!out.empty()) out += ",";
      out += std::to_string(g_slot_tid[i].load());
    }
  }
  return out + (g_cp_waiting.load() ? " (cp waiting)" : " (cp busy)");
}

uint64_t VirtualGuestTicks(uint64_t real_ticks, uint64_t guest_tick_frequency) {
  if (!g_base_ticks_set.load(std::memory_order_acquire)) {
    bool expected = false;
    // (Session 77: det_time_constant - 1 s of guest ticks, not the host's tick count at the first read.)
    if (g_base_ticks_set.compare_exchange_strong(expected, true))
      g_base_ticks.store(TimeConstant() ? guest_tick_frequency : real_ticks);
  }
  const uint64_t vb = g_vblanks.load(std::memory_order_acquire);
  if (t_read_vblank != vb) {
    t_read_vblank = vb;
    t_reads = 0;
  }
  ++t_reads;
  const uint64_t per_frame = guest_tick_frequency / 60;
  const uint64_t step = std::max<uint64_t>(1, per_frame / 100000);
  // (Session 77, review: det_time_constant - guest code called on a host thread (an interrupt, an audio callback)
  // reads half a frame later than the frame's start, so after the guest threads' reads of that frame, not before.)
  const uint64_t callback_lead = (!t_guest && t_callback_depth > 0 && TimeConstant()) ? per_frame / 2 : 0;
  if (!g_tick_freq.load(std::memory_order_relaxed)) g_tick_freq.store(guest_tick_frequency, std::memory_order_release);
  if (g_time_rebased.load(std::memory_order_acquire)) {  // wwe2k14 patch (Session 27.5): det_frames_time_hz
    const Segment s = ReadSegment();
    return g_base_ticks.load(std::memory_order_relaxed) + SegTicks(s, std::max(vb, s.vb0), guest_tick_frequency) +
           callback_lead + t_reads * step;
  }
  return g_base_ticks.load(std::memory_order_relaxed) + vb * per_frame + callback_lead + t_reads * step;
}

uint32_t VirtualUptimeMillis(uint32_t real_ms) {
  static std::atomic<bool> set{false};
  static std::atomic<uint32_t> base{0};
  if (!set.load(std::memory_order_acquire)) {
    bool expected = false;
    // (Session 77: det_time_constant - 1000 ms, not the host's uptime at the first read.)
    if (set.compare_exchange_strong(expected, true)) base.store(TimeConstant() ? 1000u : real_ms);
  }
  const uint64_t vb = g_vblanks.load(std::memory_order_acquire);
  if (g_time_rebased.load(std::memory_order_acquire)) {  // wwe2k14 patch (Session 27.5): det_frames_time_hz
    const Segment s = ReadSegment();
    return base.load(std::memory_order_relaxed) + uint32_t(SegMs(s, std::max(vb, s.vb0)));
  }
  return base.load(std::memory_order_relaxed) + uint32_t(vb * 50 / 3);
}

namespace {
std::mutex g_hook_mu;
std::vector<std::function<void()>> g_vblank_hooks;
}  // namespace

void AddVblankHook(std::function<void()> hook) {
  std::lock_guard<std::mutex> lock(g_hook_mu);
  g_vblank_hooks.push_back(std::move(hook));
}

void RunVblankHooks() {
  std::lock_guard<std::mutex> lock(g_hook_mu);
  for (auto& h : g_vblank_hooks) h();
}

uint64_t VirtualSystemTimeOffset(uint64_t real_offset) {
  if (!g_base_sys_set.load(std::memory_order_acquire)) {
    bool expected = false;
    // (Session 77: det_time_constant - 0: the system time is ConstantSystemTimeBase() + virtual time, clock.cpp.)
    if (g_base_sys_set.compare_exchange_strong(expected, true)) g_base_sys.store(TimeConstant() ? 0 : real_offset);
  }
  const uint64_t vb = g_vblanks.load(std::memory_order_acquire);
  if (g_time_rebased.load(std::memory_order_acquire)) {  // wwe2k14 patch (Session 27.5): det_frames_time_hz
    const Segment s = ReadSegment();
    return g_base_sys.load(std::memory_order_relaxed) + SegSys(s, std::max(vb, s.vb0));
  }
  return g_base_sys.load(std::memory_order_relaxed) + vb * 166667ull;
}

}  // namespace rex::det
