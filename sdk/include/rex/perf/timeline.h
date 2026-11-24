/**
 * @file        perf/timeline.h
 * @brief       Event timeline for frame-pacing investigations (wwe2k14 patch)
 *
 * One CSV line per event: time since Start() in ns, host thread id, event name, and up to three values.
 * Off (one relaxed atomic load per call site) unless Start() was called. Exported from rexruntime so the
 * GPU plugin and the app record into the same file with the same clock (std::chrono::steady_clock).
 */
#pragma once

#include <cstdint>

namespace rex::perf::timeline {

// Opens `path` (overwrites) and starts recording. Returns false if the file can't be opened.
bool Start(const char* path);
// Flushes and closes the file.
void Stop();
bool Enabled();
// `event` must be a string literal (only the pointer is kept until the line is written).
void Record(const char* event, uint32_t a = 0, uint32_t b = 0, uint32_t c = 0);

// Records `begin` now and `end` when the scope exits (if recording was on at construction).
class Scope {
 public:
  Scope(const char* begin, const char* end, uint32_t a = 0, uint32_t b = 0, uint32_t c = 0)
      : end_(Enabled() ? end : nullptr), a_(a), b_(b), c_(c) {
    if (end_) Record(begin, a, b, c);
  }
  ~Scope() {
    if (end_) Record(end_, a_, b_, c_);
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

 private:
  const char* end_;
  uint32_t a_, b_, c_;
};

}  // namespace rex::perf::timeline
