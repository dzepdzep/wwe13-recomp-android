/**
 * @file        logging/async_sink.h
 * @brief       Session 101 lane 302: a sink wrapper that moves the (slow, blocking) disk write off the calling thread.
 *
 * Why: the SDK logger flushes on every info line (LogConfig::flush_level = info), so each log call was a synchronous WriteFile on
 * the game / render / timing thread. When C: stalled (47-62 s in run 7202) the det-baton swap thread, the vblank thread and the
 * owner-timing thread all blocked inside their log calls. Header only so the offline test (tools\tests\log_async_test.cpp)
 * compiles it on its own.
 *
 * Contract:
 *   - log() copies the message into a queue under a mutex held for the copy only (never across I/O). One writer thread
 *     writes the batch to the inner sink and flushes it. flush() on the wrapper is a no-op (the writer flushes after every
 *     batch, so a line is on disk one write after it was logged); it never blocks the caller.
 *   - error / critical are NEVER dropped. info and below are dropped when the queue holds max_queue messages (the disk is
 *     stalled); warn is dropped only above a large cap (warn_cap, a warn storm during a long stall); the writer then logs one
 *     "[log] async sink dropped N lines" warn line when it catches up.
 *   - A CRITICAL line blocks its caller in Drain(1000) after the enqueue: critical is what precedes abort() / _Exit(), and
 *     nothing else would flush it (flush() is a no-op).
 *   - Drain(timeout) is the blocking path (crash handlers, FlushLogging, exit): waits until everything queued so far is
 *     written AND flushed, or the timeout (so a dead disk cannot hang a crash handler).
 *   - The writer thread is detached and owns the shared state, so process exit / sink destruction never joins a thread
 *     stuck in a write.
 */
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <spdlog/details/log_msg_buffer.h>
#include <spdlog/sinks/sink.h>

namespace rex {

/** Counters of the async log sink, read by the app's [perf][sys] line. */
struct LogWriteStats {
  uint64_t lines_queued = 0;      // total lines accepted
  uint64_t lines_dropped = 0;     // total info-and-below lines dropped on a full queue
  uint64_t queue_depth = 0;       // lines waiting right now
  uint64_t worst_write_us = 0;    // slowest single inner write / batch flush since the last take
  uint64_t worst_caller_us = 0;   // slowest log() call on a calling thread since the last take (mutex + copy)
};

struct AsyncSinkOptions {
  size_t max_queue = 65536;   // info-and-below dropped at this depth
  size_t warn_cap = 262144;   // warn dropped at this depth (error / critical never are)
};

class AsyncSink final : public spdlog::sinks::sink {
 public:
  using Options = AsyncSinkOptions;

  explicit AsyncSink(std::shared_ptr<spdlog::sinks::sink> inner) : AsyncSink(std::move(inner), AsyncSinkOptions()) {}
  AsyncSink(std::shared_ptr<spdlog::sinks::sink> inner, const AsyncSinkOptions& opt)
      : st_(std::make_shared<State>()) {
    st_->inner = std::move(inner);
    st_->opt = opt;
    auto st = st_;
    std::thread([st] { Run(st); }).detach();
  }

  // Non-blocking and lock free: it can run in a DLL's process-detach after ExitProcess killed the writer (a thread killed inside st_->mu
  // would hang a locking destructor). Whoever needs the queue written calls Drain() first (ShutdownLogging does).
  ~AsyncSink() override {
    st_->stop.store(true, std::memory_order_release);
    st_->cv.notify_all();
  }

  void log(const spdlog::details::log_msg& msg) override {
    const auto t0 = std::chrono::steady_clock::now();
    bool wake = false;
    {
      std::lock_guard<std::mutex> lk(st_->mu);
      if ((msg.level < spdlog::level::warn && st_->q.size() >= st_->opt.max_queue) ||
          (msg.level == spdlog::level::warn && st_->q.size() >= st_->opt.warn_cap)) {
        ++st_->dropped;
        return;
      }
      wake = st_->q.empty();
      st_->q.emplace_back(msg);
      ++st_->enq;
    }
    if (wake) st_->cv.notify_one();
    if (msg.level >= spdlog::level::critical) Drain(1000);  // before abort() / _Exit(): see the contract above
    const uint64_t us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    uint64_t cur = st_->worst_caller_us.load(std::memory_order_relaxed);
    while (us > cur && !st_->worst_caller_us.compare_exchange_weak(cur, us, std::memory_order_relaxed)) {}
  }

  // Never blocks: the writer thread flushes the inner sink after every batch.
  void flush() override {}

  void set_pattern(const std::string& p) override { st_->inner->set_pattern(p); }
  void set_formatter(std::unique_ptr<spdlog::formatter> f) override { st_->inner->set_formatter(std::move(f)); }

  /** Blocks until every line queued before this call is written and flushed; false on timeout. */
  bool Drain(unsigned timeout_ms) {
    std::unique_lock<std::mutex> lk(st_->mu);
    const uint64_t target = st_->enq;
    if (st_->done >= target) return true;
    st_->cv.notify_all();
    return st_->idle_cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return st_->done >= target; });
  }

  LogWriteStats TakeStats() {
    LogWriteStats s;
    std::lock_guard<std::mutex> lk(st_->mu);
    s.lines_queued = st_->enq;
    s.lines_dropped = st_->dropped;
    s.queue_depth = st_->enq - st_->done;
    s.worst_write_us = st_->worst_write_us.exchange(0);
    s.worst_caller_us = st_->worst_caller_us.exchange(0);
    return s;
  }

 private:
  struct State {
    std::shared_ptr<spdlog::sinks::sink> inner;
    Options opt;
    std::mutex mu;
    std::condition_variable cv, idle_cv;
    std::deque<spdlog::details::log_msg_buffer> q;
    uint64_t enq = 0, done = 0, dropped = 0, dropped_reported = 0;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> worst_write_us{0}, worst_caller_us{0};
  };

  static void Run(std::shared_ptr<State> st) {
    std::deque<spdlog::details::log_msg_buffer> batch;
    for (;;) {
      uint64_t n = 0, drop_now = 0;
      {
        std::unique_lock<std::mutex> lk(st->mu);
        st->cv.wait(lk, [&] { return !st->q.empty() || st->stop.load(std::memory_order_acquire) || st->dropped != st->dropped_reported; });
        if (st->q.empty() && st->dropped == st->dropped_reported) {
          if (st->stop.load(std::memory_order_acquire)) return;
          continue;
        }
        batch.swap(st->q);
        n = batch.size();
        drop_now = st->dropped - st->dropped_reported;
        st->dropped_reported = st->dropped;
      }
      auto timed = [&](auto&& fn) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const uint64_t us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        uint64_t cur = st->worst_write_us.load(std::memory_order_relaxed);
        while (us > cur && !st->worst_write_us.compare_exchange_weak(cur, us, std::memory_order_relaxed)) {}
      };
      // (one try per line: a rotation rename failure on one line must not drop the rest of the batch)
      for (auto& m : batch) {
        try {
          timed([&] { st->inner->log(m); });
        } catch (...) {
        }
      }
      try {
        if (drop_now) {
          std::string text = "[log] async sink dropped " + std::to_string(drop_now) + " lines (info and below, or warn in a storm; disk write stalled, queue full)";
          spdlog::details::log_msg dm(spdlog::source_loc{}, "core", spdlog::level::warn, text);
          timed([&] { st->inner->log(dm); });
        }
        timed([&] { st->inner->flush(); });
      } catch (...) {
      }
      batch.clear();
      {
        std::lock_guard<std::mutex> lk(st->mu);
        st->done += n;
      }
      st->idle_cv.notify_all();
    }
  }

  std::shared_ptr<State> st_;
};

}  // namespace rex
