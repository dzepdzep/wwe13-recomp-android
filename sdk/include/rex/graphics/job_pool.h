// wwe2k14 patch (Session 95, item N1; read_N.md section 3): the GPU job pool.
//
// A small work-stealing pool for PURE jobs (a job reads only what it was handed and writes only into its own result): one Chase-Lev deque per
// participant (the workers, and the external threads that register: the scout, the command processor, the submit thread), a fixed ring of 4096
// per deque, stealing round robin, a condition variable for sleep. Nothing here waits on anything but its own work: Submit never blocks (false =
// full or no workers: the caller runs the job itself), Wait(counter) HELPS (runs jobs) instead of sleeping.
//
// Sizing (Pool::AutoWorkers): the logical processors of the PROCESS affinity mask (std::thread::hardware_concurrency ignores the mask), minus 6
// (the game's main thread, 2 guest workers, the command processor, the submit thread, audio), clamped to 0..14. 0 workers = every job runs inline
// on its caller. `gpu_jobs_workers` (live) parks workers above the count: one route measures a scaling curve with in-run arms.
//
// Job cost: an EMA of ns per item for each kind, sampled on 1 job in 8; Batch(kind) = clamp(50 us / ema, 1, 64) items per job.
//
// The pool object is made once, on first use, and never destroyed (its threads are not joined: process exit ends them).
#pragma once

#include <atomic>
#include <cstdint>

namespace rex::graphics::jobs {

struct Counter {
  std::atomic<int32_t> pending{0};  // add before Submit, the pool subtracts when the job has run
};

using JobFn = void (*)(void* ctx, uint32_t first, uint32_t count);

struct Job {
  JobFn fn = nullptr;
  void* ctx = nullptr;
  uint32_t first = 0, count = 0;
  Counter* counter = nullptr;  // nullptr = fire and forget
  uint32_t kind = 0;           // Pool::RegisterKind
};

inline constexpr uint32_t kMaxKinds = 8;
inline constexpr uint32_t kMaxWorkers = 24;
inline constexpr uint32_t kMaxExternal = 8;
inline constexpr uint32_t kInvalid = 0xFFFFFFFFu;

struct PoolStats {
  struct Kind {
    const char* name = nullptr;
    uint64_t jobs = 0, items = 0, ns = 0;  // ns: summed over the sampled jobs only
    uint64_t sampled_items = 0;
    uint32_t ema_ns_per_item = 0, batch = 1;
  };
  Kind kind[kMaxKinds];
  uint32_t kinds = 0;
  uint64_t steals = 0, jobs_run = 0, inline_runs = 0, submit_full = 0;
  uint64_t worker_busy_ns[kMaxWorkers] = {};
  uint32_t workers_active = 0, workers_made = 0;
  uint64_t window_ns = 0;
};

class Pool {
 public:
  static Pool& Get();

  // The logical processors this process may run on (affinity mask), at least 1.
  static uint32_t LogicalProcessors();
  // clamp(lp - 7, 0, 14).
  static uint32_t AutoWorkers(uint32_t lp);

  // Sets the number of ACTIVE workers: forced > 0 = that many (up to kMaxWorkers), 0 = AutoWorkers. Threads are made when first needed and
  // parked when above the count. Cheap when nothing changed. Thread-safe (the command processor calls it once a frame).
  // forced < 0 = 0 workers (the W=0 arm).
  void Configure(int32_t forced);
  uint32_t Active() const { return active_.load(std::memory_order_relaxed); }
  uint32_t Made() const { return made_.load(std::memory_order_relaxed); }
  uint32_t SpinMicros() const { return Active() >= 4 ? 50u : 0u; }

  // Names a job kind (up to kMaxKinds, idempotent by name); the EMA / batch size / counters are per kind.
  uint32_t RegisterKind(const char* name);
  uint32_t Batch(uint32_t kind) const;
  // Job timing for the EMA runs only in frames the owner marks (1 frame in 8), as the lean_sample timers do.
  void SetSampling(bool on);

  // The calling thread's participant slot (made on the first call; kInvalid when the external slots are used up).
  uint32_t RegisterThread();

  // Pushes the job to `who`'s own deque and wakes a worker. False = no active worker, or the deque is full: the caller runs the job (RunInline).
  bool Submit(uint32_t who, const Job& job);
  // Runs the job on the calling thread (stats and the counter handled as for a worker's run).
  void RunInline(const Job& job);
  // Runs jobs (its own deque first, then steals) until the counter is 0.
  void Wait(uint32_t who, Counter& counter);

  void TakeStats(PoolStats& out);  // resets the window

  struct Impl;  // (public only so the .cpp's helpers can name it)

 private:
  Pool();
  Impl* impl_;
  std::atomic<uint32_t> active_{0};
  std::atomic<uint32_t> made_{0};
};

// The calling thread: below normal priority, allowed on every processor of the process mask (a thread made by a pinned caller inherits the pin).
void ConfigureWorkerThread();

}  // namespace rex::graphics::jobs
