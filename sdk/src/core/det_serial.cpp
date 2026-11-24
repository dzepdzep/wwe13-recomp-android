/**
 * wwe2k14 patch (Session 15, determinism): det_serial, a deterministic cooperative scheduler. See include/rex/det.h.
 */
#include <rex/cvar.h>
#include <rex/det.h>
#include <rex/logging.h>
#include <rex/thread.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include "platform_win.h"
#include <winternl.h>
#endif

#if defined(_MSC_VER) || defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

REXCVAR_DEFINE_BOOL(det_serial, false, "GPU",
                    "wwe2k14 (Session 15): with det_frames, run guest threads one at a time (a baton), switching only "
                    "when one blocks, in a fixed order; GPU interrupts and other host-side guest code run between "
                    "turns. Set at startup");
REXCVAR_DEFINE_INT32(det_serial_vblank_timeout_ms, 10000, "GPU",
                     "det_serial: longest wait for idle before a vblank is fired anyway (breaks determinism; logged)");
REXCVAR_DEFINE_STRING(det_serial_trace, "", "GPU",
                      "det_serial: folder for a scheduler trace (det_serial_<pid>.txt: one line per det vblank with the "
                      "decision count, a hash of the picks and the first picks; x = an external request, - = idle)");
REXCVAR_DEFINE_INT32(det_serial_stall_ms, 3000, "GPU",
                     "det_serial: log when one guest thread keeps the baton this long (a spin wait on another thread?)");
REXCVAR_DEFINE_BOOL(det_serial_stats, false, "GPU",
                    "wwe2k14 (Session 77): det_serial timing, logged every 600 det vblanks: where the time between "
                    "vblanks goes (guest turns, drain waits, external requests, picks, wake-up delay, idle)");
REXCVAR_DEFINE_BOOL(det_serial_trace_detail, false, "GPU",
                    "wwe2k14 (Session 77): det_serial_trace also notes, before each thread pick, what the command "
                    "processor was waiting on, and each GPU interrupt request's ring position (no 240-character cut)");

namespace rex::det {
namespace {
struct Slot {
  int state = 0;  // 1 runnable (not run yet), 2 running, 3 blocked, 4 exited, 5 created suspended
  std::function<bool()> ready;
  uint64_t deadline = ~0ull;
  bool go = false;
  std::condition_variable cv;
  uint32_t tid = 0;
  bool guest = true;
};
struct External {
  bool granted = false, done = false;
  std::condition_variable cv;
};
std::mutex s_mu;
std::vector<std::unique_ptr<Slot>> s_slots;
std::deque<std::shared_ptr<External>> s_ext;  // shared: the requester may return first
int s_holder = -1;  // slot running guest code, -1 = none
int s_last = -1;    // last slot that ran (round-robin start)
bool s_external_running = false;
bool s_idle = false;  // nothing runnable: the vblank thread's turn
bool s_guest_started = false;  // a guest thread has taken the first turn (no vblanks before)
std::condition_variable s_idle_cv;
std::atomic<uint64_t> s_cp_progress{0};
std::atomic<int64_t> s_holder_since_ms{0};
thread_local int t_sslot = -1;

// Scheduler trace (det_serial_trace), written under s_mu.
FILE* s_trace = nullptr;
bool s_trace_tried = false;
uint64_t s_tr_count = 0, s_tr_hash = 1469598103934665603ull;
std::string s_tr_picks;
std::function<std::string()> s_cp_describe;  // Session 77 (det_serial_trace_detail)
std::mutex s_tag_mu;                          // TraceTag comes from the CP thread without s_mu
// Session 77 (det_serial_stats): sums in nanoseconds, under s_mu. A hand-off: the holder calls ScheduleLocked (t0),
// the drain wait ends (t1), the external requests are done (t2), the next slot is woken (t3), and that slot runs again
// (its wait returns: the wake-up delay = then - t3). A turn: from a slot resuming to its next ScheduleLocked.
struct DetStats {
  uint64_t handoffs = 0, externals = 0, wakes = 0, idles = 0;
  uint64_t drain_ns = 0, ext_ns = 0, pick_ns = 0, wake_ns = 0, turn_ns = 0;
  uint64_t vb0 = 0;
  std::chrono::steady_clock::time_point wall0{};
};
DetStats s_st;
bool s_st_on = false;
std::chrono::steady_clock::time_point s_woken_at{};  // t3 of the last wake (the slot that was woken)
std::chrono::steady_clock::time_point s_turn_since{};  // when the running slot resumed
inline uint64_t Ns(std::chrono::steady_clock::duration d) {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
}
std::string s_tags;
std::atomic<bool> s_trace_open{false};
void TraceNote(int code) {  // slot, -1 idle, -2 external
  if (!s_trace_tried) {
    s_trace_tried = true;
    const std::string dir = REXCVAR_GET(det_serial_trace);
    if (!dir.empty()) {
#if defined(_WIN32)
      const std::string path = dir + "\\det_serial_" + std::to_string(GetCurrentProcessId()) + ".txt";
#else
      const std::string path = dir + "/det_serial.txt";
#endif
      s_trace = std::fopen(path.c_str(), "w");
      s_trace_open.store(s_trace != nullptr, std::memory_order_release);
      REXLOG_INFO("[det] serial trace -> {} ({})", path, s_trace ? "open" : "failed");
    }
  }
  if (!s_trace) return;
  ++s_tr_count;
  s_tr_hash = (s_tr_hash ^ uint64_t(code + 3)) * 1099511628211ull;
  // Session 77: detail mode (not part of the hash): the CP's wait before a thread pick, and the tags (interrupt ring
  // positions) noted since the previous decision, before an external.
  static const bool detail = REXCVAR_GET(det_serial_trace_detail);
  if (detail) {
    {
      std::lock_guard<std::mutex> tl(s_tag_mu);
      if (!s_tags.empty()) {
        s_tr_picks += s_tags;
        s_tags.clear();
      }
    }
    if (code != -2 && s_cp_describe) s_tr_picks += "[" + s_cp_describe() + "]";
  }
  if (detail || s_tr_picks.size() < 240) {
    s_tr_picks += code == -1 ? "-" : code == -2 ? "x" : std::to_string(code);
    s_tr_picks += ' ';
  }
}
void TraceFlush() {
  if (!s_trace) return;
  std::fprintf(s_trace, "%llu %llu %016llx %s\n", (unsigned long long)Vblanks(), (unsigned long long)s_tr_count,
               (unsigned long long)s_tr_hash, s_tr_picks.c_str());
  if (Vblanks() % 60 == 0) std::fflush(s_trace);
  s_tr_count = 0;
  s_tr_hash = 1469598103934665603ull;
  s_tr_picks.clear();
}

inline void Pause() {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(_M_X64)
  _mm_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield");  // wwe2k14 patch (Session 77): the spin-wait hint on ARM64 (was nothing)
#endif
}

inline uint32_t HostTid() {
#if defined(_WIN32)
  return uint32_t(GetCurrentThreadId());
#else
  return uint32_t(std::hash<std::thread::id>{}(std::this_thread::get_id()));
#endif
}

int64_t NowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Other host workers that change guest memory on their own (the XMA decoder): drained when all return true.
std::mutex s_drain_mu;
std::vector<std::pair<int, std::function<bool()>>> s_drains;  // Session 77: (id, check), removable
int s_next_drain_id = 0;
int g_last_undrained = -2;  // -1 = the command processor, n = drain check id n (for the stall log)
bool AllDrained() {
  if (!CpWaiting()) {
    g_last_undrained = -1;
    return false;
  }
  std::lock_guard<std::mutex> lock(s_drain_mu);
  for (const auto& d : s_drains) {
    if (!d.second()) {
      g_last_undrained = d.first;
      return false;
    }
  }
  return true;
}

// Wait (without the lock) until the command processor (and the other registered host workers) have drained, so
// whatever they do with the work submitted so far (fences written, interrupts requested, audio decoded) is done
// before a scheduling decision.
void WaitCpDrained(std::unique_lock<std::mutex>& lock) {
  if (AllDrained()) return;
  lock.unlock();
  uint32_t spins = 0;
  auto since = std::chrono::steady_clock::now();
  while (!AllDrained()) {
    if (std::chrono::steady_clock::now() - since > std::chrono::seconds(3)) {
      REXLOG_WARN("[det] serial: still waiting for a drain after 3 s ({})",
                  g_last_undrained == -1 ? std::string("command processor busy")
                                         : "drain check " + std::to_string(g_last_undrained));
      since = std::chrono::steady_clock::now();
    }
    if (++spins % 64 == 0) {
      std::this_thread::yield();
    } else {
      Pause();
    }
  }
  lock.lock();
}

// With the lock held: run pending external requests one at a time (arrival order), then give the baton to the next
// ready slot after s_last (round-robin), or go idle. Called by whoever gives the baton up.
void ScheduleLocked(std::unique_lock<std::mutex>& lock) {
  const bool st = s_st_on;
  std::chrono::steady_clock::time_point t0, t1;
  if (st) {
    t0 = std::chrono::steady_clock::now();
    if (s_turn_since.time_since_epoch().count()) s_st.turn_ns += Ns(t0 - s_turn_since);
    s_turn_since = {};
    ++s_st.handoffs;
  }
  bool drained_once = false;
  for (;;) {
    WaitCpDrained(lock);
    if (st && !drained_once) {
      drained_once = true;
      t1 = std::chrono::steady_clock::now();
      s_st.drain_ns += Ns(t1 - t0);
    }
    if (s_ext.empty()) break;
    std::shared_ptr<External> e = s_ext.front();
    s_ext.pop_front();
    TraceNote(-2);
    s_external_running = true;
    e->granted = true;
    e->cv.notify_all();
    e->cv.wait(lock, [&] { return e->done; });
    s_external_running = false;
    if (st) ++s_st.externals;
  }
  std::chrono::steady_clock::time_point t2;
  if (st) {
    t2 = std::chrono::steady_clock::now();
    s_st.ext_ns += Ns(t2 - t1);
  }
  const int n = int(s_slots.size());
  const uint64_t vb = Vblanks();
  for (int k = 1; k <= n; ++k) {
    const int i = ((s_last < 0 ? -1 : s_last) + k) % n;
    Slot& s = *s_slots[i];
    bool pick = false;
    if (s.state == 1) {
      pick = true;
    } else if (s.state == 3) {
      pick = vb >= s.deadline || (s.ready && s.ready());
    }
    if (pick) {
      TraceNote(i);
      s.state = 2;
      s.go = true;
      s_holder = i;
      s_last = i;
      s_idle = false;
      s_holder_since_ms.store(NowMs());
      if (st) {
        s_woken_at = std::chrono::steady_clock::now();
        s_st.pick_ns += Ns(s_woken_at - t2);
        ++s_st.wakes;
      }
      s.cv.notify_all();
      return;
    }
  }
  if (st) {
    s_st.pick_ns += Ns(std::chrono::steady_clock::now() - t2);
    ++s_st.idles;
  }
  TraceNote(-1);
  s_holder = -1;
  s_idle = true;
  s_idle_cv.notify_all();
}
}  // namespace

bool SerialEnabled() { return Enabled() && REXCVAR_GET(det_serial); }

int SerialRegisterThread(bool runnable, bool guest) {
  std::unique_lock<std::mutex> lock(s_mu);
  s_slots.push_back(std::make_unique<Slot>());
  s_slots.back()->state = runnable ? 1 : 5;
  s_slots.back()->guest = guest;
  return int(s_slots.size()) - 1;
}

void SerialMarkRunnable(int slot) {
  if (slot < 0) return;
  std::unique_lock<std::mutex> lock(s_mu);
  if (s_slots[slot]->state == 5) s_slots[slot]->state = 1;
}

void SerialThreadStart(int slot) {
  if (slot < 0) return;
  std::unique_lock<std::mutex> lock(s_mu);
  t_sslot = slot;
  Slot& s = *s_slots[slot];
  s.tid = HostTid();
  // The first guest thread of the run (nobody holds the baton and nothing ran yet): take it. Host workers (the
  // audio worker) wait for the scheduler instead, whenever their host thread happens to start.
  if (s.guest && !s_guest_started && s_holder < 0 && s_last < 0 && !s_idle && !s_external_running &&
      s.state == 1) {
    s_guest_started = true;
    s_idle_cv.notify_all();
    s.state = 2;
    s.go = true;
    s_holder = slot;
    s_last = slot;
    s_holder_since_ms.store(NowMs());
  }
  s.cv.wait(lock, [&] { return s.go; });
  s.go = false;
  if (s_st_on) {  // Session 77 (det_serial_stats): as in SerialBlock
    const auto now = std::chrono::steady_clock::now();
    if (s_woken_at.time_since_epoch().count()) s_st.wake_ns += Ns(now - s_woken_at);
    s_woken_at = {};
    s_turn_since = now;
  }
}

void SerialThreadExit() {
  if (t_sslot < 0) return;
  std::unique_lock<std::mutex> lock(s_mu);
  s_slots[t_sslot]->state = 4;
  if (s_holder == t_sslot) ScheduleLocked(lock);
  t_sslot = -1;
}

bool SerialIsGuestThread() { return t_sslot >= 0; }

void SerialBlock(std::function<bool()> ready, uint64_t deadline_vblank) {
  if (t_sslot < 0) return;
  std::unique_lock<std::mutex> lock(s_mu);
  Slot& s = *s_slots[t_sslot];
  s.state = 3;
  s.ready = std::move(ready);
  s.deadline = deadline_vblank;
  if (s_holder == t_sslot) ScheduleLocked(lock);
  s.cv.wait(lock, [&] { return s.go; });
  s.go = false;
  if (s_st_on) {  // Session 77 (det_serial_stats)
    const auto now = std::chrono::steady_clock::now();
    if (s_woken_at.time_since_epoch().count()) s_st.wake_ns += Ns(now - s_woken_at);
    s_woken_at = {};
    s_turn_since = now;
  }
  s.ready = nullptr;
  s.deadline = ~0ull;
}

void SerialRunExternal(const std::function<void()>& fn, bool cp_request) {
  auto e = std::make_shared<External>();
  {
    std::unique_lock<std::mutex> lock(s_mu);
    s_ext.push_back(e);
    if (cp_request) SetCpWaiting(true);
    s_idle_cv.notify_all();  // if idle, the vblank thread grants it
    e->cv.wait(lock, [&] { return e->granted; });
  }
  fn();
  if (cp_request) SetCpWaiting(false);
  {
    std::unique_lock<std::mutex> lock(s_mu);
    e->done = true;
    e->cv.notify_all();
  }
}

bool SerialWaitIdle(uint32_t timeout_ms) {
  if (timeout_ms == 0) timeout_ms = uint32_t(std::max(1, REXCVAR_GET(det_serial_vblank_timeout_ms)));
  std::unique_lock<std::mutex> lock(s_mu);
  // No vblanks before the first guest thread runs (their number would depend on host start-up time).
  while (!s_guest_started) s_idle_cv.wait_for(lock, std::chrono::milliseconds(100));
  const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  for (;;) {
    if (s_idle && s_holder < 0 && !s_external_running) {
      if (!s_ext.empty()) {
        ScheduleLocked(lock);  // external requests while idle: the vblank thread runs the decision
        continue;
      }
      WaitCpDrained(lock);
      if (s_idle && s_holder < 0 && s_ext.empty()) return true;
      continue;
    }
    if (s_holder >= 0) {
      const int64_t held = NowMs() - s_holder_since_ms.load();
      if (held > REXCVAR_GET(det_serial_stall_ms)) {
        REXLOG_WARN("[det] serial: slot {} (host tid {}) has held the baton for {} ms", s_holder,
                    s_slots[s_holder]->tid, held);
        s_holder_since_ms.store(NowMs());
      }
    }
    if (s_idle_cv.wait_until(lock, until) == std::cv_status::timeout) return false;
  }
}

void SerialAfterVblank() {
  std::unique_lock<std::mutex> lock(s_mu);
  TraceFlush();
  // Session 77 (det_serial_stats): read once per vblank (a setting change mid-run starts a fresh window).
  if (REXCVAR_GET(det_serial_stats) != s_st_on) {
    s_st_on = REXCVAR_GET(det_serial_stats);
    s_st = DetStats{};
    s_st.vb0 = Vblanks();
    s_st.wall0 = std::chrono::steady_clock::now();
  }
  if (s_st_on && Vblanks() - s_st.vb0 >= 600) {
    const double wall = double(Ns(std::chrono::steady_clock::now() - s_st.wall0));
    const double vbs = double(Vblanks() - s_st.vb0);
    auto pct = [&](uint64_t ns) { return 100.0 * double(ns) / wall; };
    auto per = [&](uint64_t ns) { return double(ns) / vbs / 1000.0; };  // us per vblank
    const uint64_t acc = s_st.turn_ns + s_st.drain_ns + s_st.ext_ns + s_st.pick_ns + s_st.wake_ns;
    REXLOG_INFO("[det] serial stats vblanks {}-{}: {:.2f} ms per vblank; per vblank {:.1f} hand-offs, {:.1f} external, "
                "{:.1f} idle; guest turns {:.0f} us ({:.0f}%), drain waits {:.0f} us ({:.0f}%), external requests "
                "{:.0f} us ({:.0f}%), picks {:.0f} us ({:.0f}%), wake-up delay {:.0f} us ({:.0f}%; {:.1f} us per "
                "wake), other (vblank thread, idle) {:.0f}%",
                s_st.vb0, Vblanks(), wall / vbs / 1e6, double(s_st.handoffs) / vbs, double(s_st.externals) / vbs,
                double(s_st.idles) / vbs, per(s_st.turn_ns), pct(s_st.turn_ns), per(s_st.drain_ns),
                pct(s_st.drain_ns), per(s_st.ext_ns), pct(s_st.ext_ns), per(s_st.pick_ns), pct(s_st.pick_ns),
                per(s_st.wake_ns), pct(s_st.wake_ns),
                s_st.wakes ? double(s_st.wake_ns) / double(s_st.wakes) / 1000.0 : 0.0,
                100.0 - pct(acc));
    s_st = DetStats{};
    s_st.vb0 = Vblanks();
    s_st.wall0 = std::chrono::steady_clock::now();
  }
  if (s_holder < 0) ScheduleLocked(lock);
}

int AddDrainCheck(std::function<bool()> drained) {
  std::lock_guard<std::mutex> lock(s_drain_mu);
  const int id = s_next_drain_id++;
  s_drains.emplace_back(id, std::move(drained));
  return id;
}

void RemoveDrainCheck(int id) {
  std::lock_guard<std::mutex> lock(s_drain_mu);
  s_drains.erase(std::remove_if(s_drains.begin(), s_drains.end(), [id](const auto& d) { return d.first == id; }),
                 s_drains.end());
}

// wwe2k14 patch (Session 77): exited slots (state 4).
bool SerialSlotExitedLocked(int slot) {
  return slot >= 0 && size_t(slot) < s_slots.size() && s_slots[slot]->state == 4;
}
bool SerialSlotExited(int slot) {
  std::unique_lock<std::mutex> lock(s_mu);
  return SerialSlotExitedLocked(slot);
}
void SerialThreadKilled(int slot) {
  if (slot < 0) return;
  std::unique_lock<std::mutex> lock(s_mu);
  if (size_t(slot) >= s_slots.size()) return;
  Slot& s = *s_slots[slot];
  s.state = 4;
  s.ready = nullptr;  // its predicate may point at objects the killed thread's unwind frees
  if (s_holder == slot) ScheduleLocked(lock);
}

// wwe2k14 patch (Session 77): det_serial_trace_detail.
bool TraceDetail() {
  static const bool on = !REXCVAR_GET(det_serial_trace).empty() && REXCVAR_GET(det_serial_trace_detail);
  return on;
}
void SetCpDescribe(std::function<std::string()> describe) {
  std::unique_lock<std::mutex> lock(s_mu);
  s_cp_describe = std::move(describe);
}
void TraceTag(const std::string& tag) {
  // Only once the trace file is open: TraceNote, which drains the tags, returns early without a file.
  if (!TraceDetail() || !s_trace_open.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> tl(s_tag_mu);
  s_tags += tag;
  s_tags += ' ';
}

// Audio pumps: the audio worker runs its client callbacks when the vblank thread asks, at idle.
namespace {
std::mutex a_mu;
std::condition_variable a_cv;
uint64_t a_requested = 0, a_done = 0, a_extra = 0, a_acc = 0;
uint64_t a_hz = 60;  // wwe2k14 patch (Session 27.5): the rate a_acc is counted in
bool a_worker = false;
}  // namespace

void AudioSetWorker(bool present) {
  std::lock_guard<std::mutex> lock(a_mu);
  a_worker = present;
  a_cv.notify_all();
}

void AudioAddPumps(uint64_t n) {
  std::lock_guard<std::mutex> lock(a_mu);
  a_extra += n;
}

bool AudioWaitPump(uint32_t timeout_ms) {
  std::unique_lock<std::mutex> lock(a_mu);
  return a_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [] { return a_requested > a_done; });
}

void AudioPumpDone() {
  std::lock_guard<std::mutex> lock(a_mu);
  ++a_done;
  a_cv.notify_all();
}

void RunAudioForVblank() {
  std::unique_lock<std::mutex> lock(a_mu);
  if (!a_worker) return;
  // 256-sample frames at 48 kHz: 3.125 per 60 Hz vblank; plus the queue fill a client gets when it registers.
  // The audio worker is a scheduler participant: its slot is ready while pumps are pending.
  // wwe2k14 patch (Session 27.5): 187.5 / hz per det vblank at det_frames_time_hz = hz, counted in 1 / (1000 hz) of
  // a pump (at 60 exactly the Session 15 sequence: 187500 / 60000 = 3125 / 1000); rescaled when the rate changes.
  const uint64_t hz = uint64_t(TimeHz());
  if (hz != a_hz) {
    a_acc = a_acc * hz / a_hz;
    a_hz = hz;
  }
  a_acc += 187500;
  const uint64_t n = a_acc / (1000 * hz) + a_extra;
  a_acc %= 1000 * hz;
  a_extra = 0;
  a_requested += n;
}

bool AudioPumpsPending() {
  std::lock_guard<std::mutex> lock(a_mu);
  return a_requested > a_done;
}

void NoteCpProgress() { s_cp_progress.fetch_add(1, std::memory_order_acq_rel); }
uint64_t CpProgress() { return s_cp_progress.load(std::memory_order_acquire); }

#if defined(_WIN32)
namespace {
typedef LONG(NTAPI* NtQueryFn)(HANDLE, int, PVOID, ULONG, PULONG);
struct NtFns {
  NtQueryFn query_object = nullptr, query_event = nullptr, query_semaphore = nullptr, query_mutant = nullptr,
            query_timer = nullptr;
  NtFns() {
    HMODULE nt = GetModuleHandleW(L"ntdll.dll");
    query_object = reinterpret_cast<NtQueryFn>(GetProcAddress(nt, "NtQueryObject"));
    query_event = reinterpret_cast<NtQueryFn>(GetProcAddress(nt, "NtQueryEvent"));
    query_semaphore = reinterpret_cast<NtQueryFn>(GetProcAddress(nt, "NtQuerySemaphore"));
    query_mutant = reinterpret_cast<NtQueryFn>(GetProcAddress(nt, "NtQueryMutant"));
    query_timer = reinterpret_cast<NtQueryFn>(GetProcAddress(nt, "NtQueryTimer"));
  }
};
const NtFns& Nt() {
  static NtFns fns;
  return fns;
}
}  // namespace
#endif

bool HandleSignaled(void* native_handle) {
#if defined(_WIN32)
  HANDLE h = HANDLE(native_handle);
  const NtFns& nt = Nt();
  alignas(8) uint8_t buf[512];
  ULONG len = 0;
  // ObjectTypeInformation (2) starts with the type name (UNICODE_STRING).
  if (nt.query_object && nt.query_object(h, 2, buf, sizeof(buf), &len) >= 0) {
    auto* name = reinterpret_cast<UNICODE_STRING*>(buf);
    const std::wstring type(name->Buffer, name->Length / sizeof(wchar_t));
    if (type == L"Event") {
      struct {
        int type;
        LONG state;
      } info{};
      if (nt.query_event(h, 0, &info, sizeof(info), &len) >= 0) return info.state != 0;
    } else if (type == L"Semaphore") {
      struct {
        LONG count;
        LONG max;
      } info{};
      if (nt.query_semaphore(h, 0, &info, sizeof(info), &len) >= 0) return info.count > 0;
    } else if (type == L"Mutant") {
      struct {
        LONG count;
        BOOLEAN owned_by_caller;
        BOOLEAN abandoned;
      } info{};
      if (nt.query_mutant(h, 0, &info, sizeof(info), &len) >= 0) return info.count > 0;
    } else if (type == L"Timer") {
      struct {
        LARGE_INTEGER remaining;
        BOOLEAN state;
      } info{};
      if (nt.query_timer(h, 0, &info, sizeof(info), &len) >= 0) return info.state != 0;
    }
  }
  // Threads, processes and anything else: a zero wait does not consume their signal.
  return WaitForSingleObject(h, 0) == WAIT_OBJECT_0;
#else
  (void)native_handle;
  return false;
#endif
}

bool HandleSignaled(rex::thread::WaitHandle* wait_handle, std::thread::id waiter) {
  if (!wait_handle) return false;
#if defined(_WIN32)
  (void)waiter;
  return HandleSignaled(wait_handle->native_handle());
#else
  const int peek = rex::thread::PeekSignaled(wait_handle, waiter);
  if (peek < 0) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true))
      REXLOG_WARN("[det] serial: a wait handle without a readiness peek; its waits wake only at their deadline");
  }
  return peek == 1;
#endif
}

}  // namespace rex::det
