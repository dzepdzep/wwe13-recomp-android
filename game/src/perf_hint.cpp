// WWE13 (Android): ADPF performance hints for the frame's threads.
//
// Once the spin-waits were gone (SDL UI loop, fence polls, the D3D wait loops) the big cores ran lightly loaded and the
// WALT governor dropped them to ~0.8-2.0 GHz between bursts: a heavy frame then started on a slow core. A hint session
// (APerformanceHint, API 33) over the render thread, the command thread and the submit thread, with a 16.67 ms target
// and each frame's busy time reported, lets the system raise the clocks only as much as the frame needs.
// Off by default: no gain (modes 1/2), and mode 3 (the game's busy threads too, boosted) measured -0.35 fps in an in-run
// A/B. WWE13_PERF_HINT=1 (busy time), 2 (whole interval), 3 (+ game threads, over-target reports).
#include <dirent.h>
#include <sched.h>
#include <thread>
#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <rex/logging.h>

extern std::atomic<uint32_t> g_wwe13_present_interval;
namespace wwe13_perf {
namespace {
struct Api {
  void* (*get_manager)() = nullptr;
  void* (*create_session)(void*, const int32_t*, size_t, int64_t) = nullptr;
  int (*report)(void*, int64_t) = nullptr;
  int (*update_target)(void*, int64_t) = nullptr;
};

thread_local int64_t t_wait_ns = 0;
void* g_session = nullptr;
Api g_api;
int64_t g_last_swap_ns = 0;
uint32_t g_frames = 0;
bool g_failed = false;

int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int Mode() {
  static const int mode = [] {
    const char* v = std::getenv("WWE13_PERF_HINT");
    return v ? std::atoi(v) : 0;  // (off by default: no measurable gain, 1 = busy time, 2 = whole interval)
  }();
  return mode;
}
bool Enabled() { return Mode() != 0; }

std::vector<int32_t> FindThreads() {
  std::vector<int32_t> tids{int32_t(gettid())};
  std::vector<std::pair<unsigned long long, int32_t>> xthreads;
  DIR* dir = opendir("/proc/self/task");
  if (!dir) return tids;
  while (dirent* e = readdir(dir)) {
    if (e->d_name[0] == '.') continue;
    char path[64];
    std::snprintf(path, sizeof(path), "/proc/self/task/%s/comm", e->d_name);
    FILE* f = std::fopen(path, "r");
    if (!f) continue;
    char name[32] = {};
    std::fgets(name, sizeof(name), f);
    std::fclose(f);
    if (std::strncmp(name, "GPU Commands", 12) == 0 || std::strncmp(name, "GPU Submit", 10) == 0) {
      tids.push_back(std::atoi(e->d_name));
    } else if (Mode() >= 3 && std::strncmp(name, "XThread", 7) == 0) {
      // mode 3: the game's busy threads too (CPU time so far from /proc/self/task/<tid>/stat, fields 14 + 15)
      std::snprintf(path, sizeof(path), "/proc/self/task/%s/stat", e->d_name);
      if (FILE* st = std::fopen(path, "r")) {
        char buf[512] = {};
        std::fgets(buf, sizeof(buf), st);
        std::fclose(st);
        const char* q = std::strrchr(buf, ')');
        unsigned long long ut = 0, stt = 0;
        if (q && std::sscanf(q + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %llu %llu", &ut, &stt) == 2) {
          xthreads.push_back({ut + stt, std::atoi(e->d_name)});
        }
      }
    }
  }
  closedir(dir);
  if (!xthreads.empty()) {
    unsigned long long top = 0;
    for (auto& x : xthreads) top = x.first > top ? x.first : top;
    for (auto& x : xthreads) {
      if (x.first * 10 >= top && x.second != tids[0]) tids.push_back(x.second);
    }
  }
  return tids;
}

bool CreateSession() {
  void* lib = dlopen("libandroid.so", RTLD_NOW);
  if (!lib) return false;
  g_api.get_manager = reinterpret_cast<void* (*)()>(dlsym(lib, "APerformanceHint_getManager"));
  g_api.create_session =
      reinterpret_cast<void* (*)(void*, const int32_t*, size_t, int64_t)>(dlsym(lib, "APerformanceHint_createSession"));
  g_api.report = reinterpret_cast<int (*)(void*, int64_t)>(dlsym(lib, "APerformanceHint_reportActualWorkDuration"));
  g_api.update_target =
      reinterpret_cast<int (*)(void*, int64_t)>(dlsym(lib, "APerformanceHint_updateTargetWorkDuration"));
  if (!g_api.get_manager || !g_api.create_session || !g_api.report) return false;
  void* manager = g_api.get_manager();
  if (!manager) return false;
  const std::vector<int32_t> tids = FindThreads();
  g_session = g_api.create_session(manager, tids.data(), tids.size(), 16666667);
  std::string list;
  for (int32_t t : tids) list += " " + std::to_string(t);
  REXLOG_INFO("[wwe13-perfhint] session {} for threads{}", g_session ? "created" : "NOT created", list);
  return g_session != nullptr;
}
}  // namespace

// Time the render thread spent waiting (the D3D wait loops): not part of the frame's work.
void AddWait(int64_t ns) { t_wait_ns += ns; }

// A/B within one run (WWE13_AB_PERIOD_S=<s>): the experiment flips between A (phase 0) and B (phase 1) every <s>
// seconds, and the frames of each phase are counted - the demo matches are random, so separate runs differ by more
// than the effects being measured.
int AbPeriod() {
  static const int period = [] {
    const char* v = std::getenv("WWE13_AB_PERIOD_S");
    return v ? std::atoi(v) : 0;
  }();
  return period;
}
// The phase for the experiment named `what` (WWE13_AB_WHAT): -1 when that one is not being compared.
int AbPhase(const char* what) {
  if (AbPeriod() <= 0) return -1;
  if (what) {
    const char* w = std::getenv("WWE13_AB_WHAT");
    if (!w || std::strcmp(w, what) != 0) return -1;
  }
  return int((NowNs() / 1000000000) / AbPeriod()) & 1;
}

namespace {
struct AbStats {
  int64_t ns[2] = {};
  uint64_t frames[2] = {}, late[2] = {};
  int64_t last = 0, last_log = 0;
} g_ab;
void AbOnSwap() {
  const int phase = AbPhase(nullptr);
  if (phase < 0) return;
  const int64_t now = NowNs();
  if (g_ab.last && g_wwe13_present_interval.load(std::memory_order_relaxed) <= 1) {
    const int64_t d = now - g_ab.last;
    if (d < 100000000) {
      g_ab.ns[phase] += d;
      ++g_ab.frames[phase];
      if (d > 17500000) ++g_ab.late[phase];
    }
  }
  g_ab.last = now;
  if (!g_ab.last_log) g_ab.last_log = now;
  if (now - g_ab.last_log > 30000000000ll) {
    g_ab.last_log = now;
    auto fps = [](int i) { return g_ab.ns[i] ? double(g_ab.frames[i]) * 1e9 / double(g_ab.ns[i]) : 0.0; };
    REXLOG_INFO("[wwe13-ab] A {:.2f} fps ({} frames, {} late) | B {:.2f} fps ({} frames, {} late)", fps(0), g_ab.frames[0],
                g_ab.late[0], fps(1), g_ab.frames[1], g_ab.late[1]);
  }
}
}  // namespace

// Big cores only (default; WWE13_BIG_CORES=0 = off; in-run A/B with WWE13_AB_WHAT=affinity: B = big cores, A = all):
// every thread of the process is kept on CPUs 3-7 (Snapdragon 8 Gen 2: 3-6 gold, 7 prime), so a job the frame waits for
// never lands on a little A510 core. 6-man tag match, in-run A/B: 52.8 -> 58.5 fps, late frames halved.
namespace {
void ApplyAffinity(bool big) {
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int c = big ? 3 : 0; c < 8; ++c) CPU_SET(c, &set);
  if (DIR* dir = opendir("/proc/self/task")) {
    while (dirent* e = readdir(dir)) {
      if (e->d_name[0] == '.') continue;
      sched_setaffinity(std::atoi(e->d_name), sizeof(set), &set);
    }
    closedir(dir);
  }
}
void AffinityThreadStart() {
  static bool started = false;
  if (started) return;
  started = true;
  const char* big_env = std::getenv("WWE13_BIG_CORES");
  const bool always = !(big_env && std::strcmp(big_env, "0") == 0);  // (default on; WWE13_BIG_CORES=0 = off)
  const bool ab = AbPhase("affinity") >= 0;
  if (!always && !ab) return;
  std::thread([always = always && !ab] {
    int last = -1;
    int ticks = 0;
    for (;;) {
      const int want = always ? 1 : AbPhase("affinity");
      // (re-applied every 5 s too: threads created since)
      if (want != last || ++ticks >= 10) {
        ApplyAffinity(want == 1);
        last = want;
        ticks = 0;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }).detach();
}
}  // namespace

// Once per presented frame, on the render thread.
void OnSwap() {
  AffinityThreadStart();
  AbOnSwap();
  if (!Enabled() || g_failed) return;
  const int64_t now = NowNs();
  // (a few seconds in: the command and submit threads exist by then)
  if (!g_session) {
    if (++g_frames < (Mode() >= 3 ? 2500u : 300u)) {
      g_last_swap_ns = now;
      t_wait_ns = 0;
      return;
    }
    if (!CreateSession()) {
      g_failed = true;
      REXLOG_INFO("[wwe13-perfhint] not available");
      return;
    }
  }
  const int64_t interval = now - g_last_swap_ns;
  int64_t busy = interval - t_wait_ns;
  // A frame that missed its vblank: report the whole interval (the work did not fit).
  if (interval > 17500000 || Mode() == 2) busy = interval;  // (mode 2: always the whole interval - clocks held up)
  if (Mode() >= 3) busy = AbPhase("hint") == 0 ? 4000000 : 20000000;  // (mode 3: B / no A/B = over target: boost; A = relaxed)
  if (busy < 1000000) busy = 1000000;
  if (interval < 200000000) g_api.report(g_session, busy);
  g_last_swap_ns = now;
  t_wait_ns = 0;
}
}  // namespace wwe13_perf
