// wwe2k14 patch (Session 95, item N1; read_N.md section 3): the GPU job pool. See include\rex\graphics\job_pool.h.
//
// The deques are Chase-Lev (Le, Pop, Cohen, Zappa Nardelli 2013, the C11 version): the owner pushes and takes at the bottom, thieves take at the
// top with a CAS. An element is the index + 1 of a Job in the owner's ring (0 = empty); the Job is copied BEFORE the thief's CAS, so a lost CAS
// discards the copy. The ring is twice the deque, so an index in a live deque is never rewritten before it is taken.
#include <rex/graphics/job_pool.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__) || defined(__ANDROID__)
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

#include <rex/logging.h>
#include <rex/thread.h>

#if defined(_MSC_VER) || defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define REX_JOBS_PAUSE() _mm_pause()
#else
#define REX_JOBS_PAUSE() std::this_thread::yield()
#endif

namespace rex::graphics::jobs {

namespace {
constexpr uint32_t kSlots = kMaxWorkers + kMaxExternal;
constexpr uint32_t kUnset = 0xFFFFFFFEu;
constexpr uint32_t kAbort = 0xFFFFFFFFu;
thread_local uint32_t t_slot = kUnset;

uint64_t NowNs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
}  // namespace

struct Participant {
  std::atomic<int64_t> top{0}, bottom{0};
  std::unique_ptr<std::atomic<uint32_t>[]> buf;
  uint32_t cap = 0;  // power of two
  std::unique_ptr<Job[]> ring;
  uint32_t ring_mask = 0;
  uint32_t next = 0;  // owner only
  std::atomic<uint64_t> busy_ns{0};
  std::atomic<bool> ready{false};

  void Init(uint32_t deque_cap) {
    cap = deque_cap;
    buf.reset(new std::atomic<uint32_t>[cap]);
    for (uint32_t i = 0; i < cap; ++i) buf[i].store(0, std::memory_order_relaxed);
    ring.reset(new Job[size_t(cap) * 2]);
    ring_mask = cap * 2 - 1;
    ready.store(true, std::memory_order_release);
  }
  // owner
  bool Push(const Job& job) {
    const int64_t b = bottom.load(std::memory_order_relaxed);
    const int64_t t = top.load(std::memory_order_acquire);
    if (b - t >= int64_t(cap)) return false;
    const uint32_t idx = next++ & ring_mask;
    ring[idx] = job;
    buf[uint64_t(b) & (cap - 1)].store(idx + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    bottom.store(b + 1, std::memory_order_relaxed);
    return true;
  }
  // owner: true = *out filled
  bool Take(Job* out) {
    int64_t b = bottom.load(std::memory_order_relaxed) - 1;
    bottom.store(b, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    int64_t t = top.load(std::memory_order_relaxed);
    if (t > b) {  // empty
      bottom.store(b + 1, std::memory_order_relaxed);
      return false;
    }
    const uint32_t v = buf[uint64_t(b) & (cap - 1)].load(std::memory_order_relaxed);
    if (t == b) {  // the last element: race with the thieves
      const bool won = top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed);
      bottom.store(b + 1, std::memory_order_relaxed);
      if (!won) return false;
    }
    *out = ring[(v - 1) & ring_mask];
    return true;
  }
  // thief: 1 = taken, 0 = empty, kAbort = lost a race
  uint32_t Steal(Job* out) {
    int64_t t = top.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const int64_t b = bottom.load(std::memory_order_acquire);
    if (t >= b) return 0;
    const uint32_t v = buf[uint64_t(t) & (cap - 1)].load(std::memory_order_relaxed);
    if (!v) return kAbort;
    const Job copy = ring[(v - 1) & ring_mask];
    if (!top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) return kAbort;
    *out = copy;
    return 1;
  }
};

struct Pool::Impl {
  Participant parts[kSlots];
  std::mutex mu;
  std::condition_variable cv;
  std::atomic<uint32_t> sleepers{0};
  std::atomic<int64_t> queued{0};
  std::atomic<uint32_t> rr{0};
  std::atomic<uint32_t> ext_next{0};
  std::atomic<bool> sampling{false};
  std::condition_variable park_cv;  // parked workers (above the live count) wait here, not counted in sleepers

  struct KindRec {
    char name[24] = {};
    std::atomic<uint64_t> jobs{0}, items{0}, ns{0}, sampled_items{0};
    std::atomic<uint32_t> ema{0};
  };
  KindRec kinds[kMaxKinds];
  std::atomic<uint32_t> kinds_n{0};

  std::atomic<uint64_t> steals{0}, jobs_run{0}, inline_runs{0}, submit_full{0};
  uint64_t window_start_ns = 0;
};

// ---- sizing
uint32_t Pool::LogicalProcessors() {
  uint32_t n = 0;
#if defined(_WIN32)
  DWORD_PTR process_mask = 0, system_mask = 0;
  if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) && process_mask) {
    n = uint32_t(std::popcount(uint64_t(process_mask)));
  }
#elif defined(__linux__) || defined(__ANDROID__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(getpid(), sizeof(set), &set) == 0) n = uint32_t(CPU_COUNT(&set));
#endif
  if (!n) n = std::thread::hardware_concurrency();
  return std::max<uint32_t>(n, 1);
}

uint32_t Pool::AutoWorkers(uint32_t lp) {
  const int32_t w = int32_t(lp) - 7;  // the game's main thread, 2 guest workers, the command processor, the submit thread, audio, the scout
  return uint32_t(std::clamp<int32_t>(w, 0, 14));
}

Pool& Pool::Get() {
  static Pool* pool = new Pool();  // (never destroyed: see the header)
  return *pool;
}

Pool::Pool() : impl_(new Impl()) { impl_->window_start_ns = NowNs(); }

// ---- kinds
uint32_t Pool::RegisterKind(const char* name) {
  Impl& p = *impl_;
  std::lock_guard<std::mutex> lock(p.mu);
  const uint32_t n = p.kinds_n.load();
  for (uint32_t i = 0; i < n; ++i) {
    if (std::strncmp(p.kinds[i].name, name, sizeof(p.kinds[i].name) - 1) == 0) return i;
  }
  if (n >= kMaxKinds) return kMaxKinds - 1;
  std::snprintf(p.kinds[n].name, sizeof(p.kinds[n].name), "%s", name);
  p.kinds_n.store(n + 1);
  return n;
}

uint32_t Pool::Batch(uint32_t kind) const {
  if (kind >= kMaxKinds) return 1;
  const uint32_t ema = impl_->kinds[kind].ema.load(std::memory_order_relaxed);
  if (!ema) return 8;  // not measured yet
  return std::clamp<uint32_t>(50000u / ema, 1u, 64u);
}

// ---- running
namespace {
inline void Finish(const Job& j) {
  if (j.counter) j.counter->pending.fetch_sub(1, std::memory_order_acq_rel);
}
}  // namespace

// Runs one job with the pool's bookkeeping. busy: a worker's busy-time accumulator (nullptr for the others).
static void RunCounted(Pool::Impl& p, const Job& j, std::atomic<uint64_t>* busy) {
  p.jobs_run.fetch_add(1, std::memory_order_relaxed);
  const bool sample = p.sampling.load(std::memory_order_relaxed);
  const bool timed = sample || busy;
  const uint64_t t0 = timed ? NowNs() : 0;
  if (j.fn) j.fn(j.ctx, j.first, j.count);
  Finish(j);
  if (j.kind < kMaxKinds) {
    Pool::Impl::KindRec& k = p.kinds[j.kind];
    k.jobs.fetch_add(1, std::memory_order_relaxed);
    k.items.fetch_add(j.count, std::memory_order_relaxed);
    if (timed) {
      const uint64_t ns = NowNs() - t0;
      if (busy) busy->fetch_add(ns, std::memory_order_relaxed);
      if (sample && j.count) {
        k.ns.fetch_add(ns, std::memory_order_relaxed);
        k.sampled_items.fetch_add(j.count, std::memory_order_relaxed);
        const uint32_t per_item = uint32_t(std::min<uint64_t>(ns / j.count, 0x7FFFFFFFu));
        const uint32_t old = k.ema.load(std::memory_order_relaxed);
        k.ema.store(old ? (old * 7 + per_item) / 8 : std::max<uint32_t>(per_item, 1), std::memory_order_relaxed);
      }
    }
  } else if (busy) {
    busy->fetch_add(NowNs() - t0, std::memory_order_relaxed);
  }
}

void Pool::RunInline(const Job& job) {
  impl_->inline_runs.fetch_add(1, std::memory_order_relaxed);
  RunCounted(*impl_, job, nullptr);
}

// Steals from any other ready participant, starting at a rotating one.
static bool StealAny(Pool::Impl& p, uint32_t self, Job* out) {
  const uint32_t start = p.rr.fetch_add(1, std::memory_order_relaxed);
  for (uint32_t n = 0; n < kSlots; ++n) {
    const uint32_t slot = (start + n) % kSlots;
    if (slot == self) continue;
    Participant& part = p.parts[slot];
    if (!part.ready.load(std::memory_order_acquire)) continue;
    for (uint32_t tries = 0; tries < 3; ++tries) {
      const uint32_t r = part.Steal(out);
      if (r == 1) {
        p.queued.fetch_sub(1, std::memory_order_relaxed);
        p.steals.fetch_add(1, std::memory_order_relaxed);
        return true;
      }
      if (r == 0) break;
    }
  }
  return false;
}

bool Pool::Submit(uint32_t who, const Job& job) {
  Impl& p = *impl_;
  if (who >= kSlots || !active_.load(std::memory_order_relaxed)) return false;
  Participant& me = p.parts[who];
  if (!me.ready.load(std::memory_order_acquire)) return false;
  if (job.counter) job.counter->pending.fetch_add(1, std::memory_order_relaxed);
  if (!me.Push(job)) {
    if (job.counter) job.counter->pending.fetch_sub(1, std::memory_order_relaxed);
    p.submit_full.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  p.queued.fetch_add(1, std::memory_order_seq_cst);
  if (p.sleepers.load(std::memory_order_seq_cst)) {
    std::lock_guard<std::mutex> lock(p.mu);
    p.cv.notify_one();
  }
  return true;
}

void Pool::Wait(uint32_t who, Counter& counter) {
  Impl& p = *impl_;
  while (counter.pending.load(std::memory_order_acquire) > 0) {
    Job j;
    if (who < kSlots && p.parts[who].ready.load(std::memory_order_acquire) && p.parts[who].Take(&j)) {
      p.queued.fetch_sub(1, std::memory_order_relaxed);
      RunCounted(p, j, nullptr);
    } else if (StealAny(p, who, &j)) {
      RunCounted(p, j, nullptr);
    } else {
      REX_JOBS_PAUSE();
    }
  }
}

uint32_t Pool::RegisterThread() {
  if (t_slot != kUnset) return t_slot;
  Impl& p = *impl_;
  const uint32_t n = p.ext_next.fetch_add(1);
  if (n >= kMaxExternal) {
    t_slot = kInvalid;
    return t_slot;
  }
  const uint32_t slot = kMaxWorkers + n;
  p.parts[slot].Init(4096);
  t_slot = slot;
  return slot;
}

// ---- the workers
namespace {
void WorkerMain(Pool::Impl* pp, std::atomic<uint32_t>* active, uint32_t id) {
  Pool::Impl& p = *pp;
  char name[24];
  std::snprintf(name, sizeof(name), "GPU Jobs %u", id);
  rex::thread::set_current_thread_name(name);
  ConfigureWorkerThread();
  Participant& me = p.parts[id];
  for (;;) {
    if (id >= active->load(std::memory_order_acquire)) {  // parked: above the live worker count
      std::unique_lock<std::mutex> lock(p.mu);
      p.park_cv.wait(lock, [&] { return id < active->load(std::memory_order_acquire); });
      continue;
    }
    Job j;
    if (StealAny(p, id, &j)) {
      RunCounted(p, j, &me.busy_ns);
      continue;
    }
    // idle: spin a while (>= 4 workers; an SMT sibling is not given the core below that), then sleep
    const uint32_t spin_us = active->load(std::memory_order_relaxed) >= 4 ? 50u : 0u;
    bool got = false;
    if (spin_us) {
      const uint64_t until = NowNs() + uint64_t(spin_us) * 1000u;
      while (NowNs() < until) {
        REX_JOBS_PAUSE();
        if (p.queued.load(std::memory_order_relaxed) > 0 && StealAny(p, id, &j)) {
          RunCounted(p, j, &me.busy_ns);
          got = true;
          break;
        }
      }
    }
    if (got) continue;
    std::unique_lock<std::mutex> lock(p.mu);
    p.sleepers.fetch_add(1, std::memory_order_seq_cst);
    if (p.queued.load(std::memory_order_seq_cst) <= 0 && id < active->load(std::memory_order_acquire)) p.cv.wait_for(lock, std::chrono::milliseconds(50));
    p.sleepers.fetch_sub(1, std::memory_order_seq_cst);
  }
}
}  // namespace

void Pool::Configure(int32_t forced) {
  static const uint32_t auto_workers = AutoWorkers(LogicalProcessors());  // (the affinity mask is read once)
  const uint32_t want = forced < 0 ? 0u : forced > 0 ? std::min<uint32_t>(uint32_t(forced), kMaxWorkers) : auto_workers;
  if (want == active_.load(std::memory_order_relaxed) && made_.load(std::memory_order_relaxed) >= want) return;
  Impl& p = *impl_;
  {
    std::lock_guard<std::mutex> lock(p.mu);
    while (made_.load() < want) {
      const uint32_t id = made_.load();
      p.parts[id].Init(256);
      std::thread(WorkerMain, impl_, &active_, id).detach();
      made_.store(id + 1);
    }
    active_.store(want, std::memory_order_release);
  }
  p.cv.notify_all();
  p.park_cv.notify_all();
}

void Pool::SetSampling(bool on) { impl_->sampling.store(on, std::memory_order_relaxed); }

void ConfigureWorkerThread() {
#if defined(_WIN32)
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
  DWORD_PTR process_mask = 0, system_mask = 0;
  if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) && process_mask) SetThreadAffinityMask(GetCurrentThread(), process_mask);
#elif defined(__linux__) || defined(__ANDROID__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(getpid(), sizeof(set), &set) == 0) sched_setaffinity(0, sizeof(set), &set);
  setpriority(PRIO_PROCESS, 0, 5);
#endif
}

void Pool::TakeStats(PoolStats& out) {
  Impl& p = *impl_;
  out = PoolStats();
  const uint32_t n = p.kinds_n.load();
  out.kinds = n;
  for (uint32_t i = 0; i < n; ++i) {
    Impl::KindRec& k = p.kinds[i];
    out.kind[i].name = k.name;
    out.kind[i].jobs = k.jobs.exchange(0);
    out.kind[i].items = k.items.exchange(0);
    out.kind[i].ns = k.ns.exchange(0);
    out.kind[i].sampled_items = k.sampled_items.exchange(0);
    out.kind[i].ema_ns_per_item = k.ema.load();
    out.kind[i].batch = Batch(i);
  }
  out.steals = p.steals.exchange(0);
  out.jobs_run = p.jobs_run.exchange(0);
  out.inline_runs = p.inline_runs.exchange(0);
  out.submit_full = p.submit_full.exchange(0);
  for (uint32_t w = 0; w < kMaxWorkers; ++w) out.worker_busy_ns[w] = p.parts[w].busy_ns.exchange(0);
  out.workers_active = active_.load();
  out.workers_made = made_.load();
  const uint64_t now = NowNs();
  out.window_ns = now - p.window_start_ns;
  p.window_start_ns = now;
}

}  // namespace rex::graphics::jobs
