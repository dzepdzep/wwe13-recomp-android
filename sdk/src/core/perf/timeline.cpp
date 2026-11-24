/**
 * @file        perf/timeline.cpp
 * @brief       Event timeline for frame-pacing investigations (wwe2k14 patch); see perf/timeline.h
 */
#include <rex/perf/processor.h>
#include <rex/perf/timeline.h>
#include <rex/thread.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include "../platform_win.h"
#else
#include <pthread.h>
#if defined(__linux__)
#include <sched.h>
#endif
#endif

namespace rex::perf {

uint32_t CurrentProcessorNumber() {
#ifdef _WIN32
  PROCESSOR_NUMBER number;
  GetCurrentProcessorNumberEx(&number);
  return uint32_t(number.Group) * 64 + uint32_t(number.Number);
#elif defined(__linux__)
  int cpu = sched_getcpu();
  return cpu >= 0 ? uint32_t(cpu) : 0;
#else
  return 0;
#endif
}

namespace timeline {
namespace {

// wwe2k14 patch (Session 6, COUNTERS): Record() used to take one global mutex, fprintf under it and fflush about
// once a second inside it, on whichever thread was recording. Now the line is formatted on the recording thread
// outside any lock and appended to an in-memory buffer under a short lock (a memcpy, no syscalls); a writer
// thread started by Start() writes and flushes the buffer every ~250 ms, and Stop() writes the rest. The CSV
// format is unchanged.

std::atomic<bool> g_enabled{false};
// steady_clock time of Start(), in ns since the clock's epoch. Written before g_enabled is set (release).
std::atomic<int64_t> g_start_ns{0};

// Serializes Start() and Stop().
std::mutex g_control_mutex;

// Recorded lines not yet handed to the writer. Held only for an append or a buffer swap.
std::mutex g_buffer_mutex;
std::vector<char> g_buffer;
bool g_accepting = false;

// Held by whoever writes to the file: the writer thread, or Stop() after joining it.
std::mutex g_io_mutex;
std::FILE* g_file = nullptr;
// The writer's half of the double buffer: swapped with g_buffer, written, cleared (keeping its capacity).
std::vector<char> g_write_buffer;

std::mutex g_wake_mutex;
std::condition_variable g_wake_cv;
bool g_writer_stop = false;
// Heap-allocated and intentionally leaked if Stop() is never called: a joinable std::thread destroyed during
// static destruction would call std::terminate.
std::thread* g_writer = nullptr;

constexpr size_t kBufferReserve = size_t(4) << 20;
constexpr auto kWriteInterval = std::chrono::milliseconds(250);
// Longest event name formatted on the stack; longer ones take a slow path (event names are short literals).
constexpr size_t kMaxFastEventLength = 128;

uint32_t HostThreadId() {
#ifdef _WIN32
  return uint32_t(GetCurrentThreadId());
#else
  return uint32_t(uintptr_t(pthread_self()));
#endif
}

int64_t SteadyNowNs() {
  return int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now().time_since_epoch())
                     .count());
}

char* PutDecimal(char* out, uint64_t value) {
  char digits[20];
  size_t count = 0;
  do {
    digits[count++] = char('0' + value % 10);
    value /= 10;
  } while (value);
  while (count) {
    *out++ = digits[--count];
  }
  return out;
}

// Same output as printf "0x%X".
char* PutHex(char* out, uint32_t value) {
  static const char kHexDigits[] = "0123456789ABCDEF";
  *out++ = '0';
  *out++ = 'x';
  char digits[8];
  size_t count = 0;
  do {
    digits[count++] = kHexDigits[value & 0xF];
    value >>= 4;
  } while (value);
  while (count) {
    *out++ = digits[--count];
  }
  return out;
}

// Formats "%lld,%u,%s,0x%X,0x%X,0x%X\n" (the format the timeline has always used) into `out`, which must hold
// at least 72 + event_length bytes. Returns the length.
size_t FormatLine(char* out, long long t, uint32_t thread, const char* event, size_t event_length, uint32_t a,
                  uint32_t b, uint32_t c) {
  char* p = out;
  if (t < 0) {
    *p++ = '-';
    p = PutDecimal(p, uint64_t(0) - uint64_t(t));
  } else {
    p = PutDecimal(p, uint64_t(t));
  }
  *p++ = ',';
  p = PutDecimal(p, thread);
  *p++ = ',';
  std::memcpy(p, event, event_length);
  p += event_length;
  *p++ = ',';
  p = PutHex(p, a);
  *p++ = ',';
  p = PutHex(p, b);
  *p++ = ',';
  p = PutHex(p, c);
  *p++ = '\n';
  return size_t(p - out);
}

void Append(const char* line, size_t length) {
  std::lock_guard<std::mutex> lock(g_buffer_mutex);
  if (!g_accepting) {
    return;
  }
  g_buffer.insert(g_buffer.end(), line, line + length);
}

// Takes everything recorded so far and writes it to the file. With `close`, stops accepting new lines in the
// same step. Call with g_io_mutex held.
void WriteBufferedLocked(bool flush, bool close) {
  {
    std::lock_guard<std::mutex> lock(g_buffer_mutex);
    // g_write_buffer is empty here, so the recorders continue into its (already reserved) memory.
    g_buffer.swap(g_write_buffer);
    if (close) {
      g_accepting = false;
    }
  }
  if (g_file && !g_write_buffer.empty()) {
    std::fwrite(g_write_buffer.data(), 1, g_write_buffer.size(), g_file);
  }
  g_write_buffer.clear();
  if (g_file && flush) {
    std::fflush(g_file);
  }
}

void WriterThreadMain() {
  rex::thread::set_current_thread_name("Timeline Writer");
  std::unique_lock<std::mutex> wake_lock(g_wake_mutex);
  for (;;) {
    g_wake_cv.wait_for(wake_lock, kWriteInterval, [] { return g_writer_stop; });
    const bool stop = g_writer_stop;
    wake_lock.unlock();
    {
      std::lock_guard<std::mutex> io_lock(g_io_mutex);
      // Flush so a crash loses little and the file can be read while the game runs.
      WriteBufferedLocked(true, false);
    }
    if (stop) {
      return;
    }
    wake_lock.lock();
  }
}

// If the process exits without Stop() (the app doesn't call it), write what is still buffered. rexruntime is a
// DLL, so on Windows this runs during process detach, after the other threads were terminated, possibly in the
// middle of a write: only proceed if the locks can be taken. Defined after everything it uses, so it's destroyed
// first.
struct ExitFlush {
  ~ExitFlush() {
    g_enabled.store(false, std::memory_order_release);
#ifndef _WIN32
    // Other threads are still running during static destruction here: let the writer finish first.
    if (g_writer) {
      {
        std::lock_guard<std::mutex> wake_lock(g_wake_mutex);
        g_writer_stop = true;
      }
      g_wake_cv.notify_all();
      if (g_writer->joinable()) {
        g_writer->join();
      }
    }
#endif
    if (!g_io_mutex.try_lock()) {
      return;
    }
    if (g_file && g_buffer_mutex.try_lock()) {
      g_accepting = false;
      if (!g_buffer.empty()) {
        std::fwrite(g_buffer.data(), 1, g_buffer.size(), g_file);
        g_buffer.clear();
      }
      g_buffer_mutex.unlock();
      std::fflush(g_file);
    }
    g_io_mutex.unlock();
  }
};
ExitFlush g_exit_flush;

}  // namespace

bool Start(const char* path) {
  std::lock_guard<std::mutex> control_lock(g_control_mutex);
  if (g_file) {
    return true;
  }
  std::FILE* file = std::fopen(path, "w");
  if (!file) {
    return false;
  }
  std::setvbuf(file, nullptr, _IOFBF, size_t(1) << 20);
  std::fputs("t_ns,thread,event,a,b,c\n", file);
  {
    std::lock_guard<std::mutex> io_lock(g_io_mutex);
    g_file = file;
    g_write_buffer.clear();
    g_write_buffer.reserve(kBufferReserve);
  }
  {
    std::lock_guard<std::mutex> buffer_lock(g_buffer_mutex);
    g_buffer.clear();
    g_buffer.reserve(kBufferReserve);
    g_accepting = true;
  }
  {
    std::lock_guard<std::mutex> wake_lock(g_wake_mutex);
    g_writer_stop = false;
  }
  g_writer = new std::thread(WriterThreadMain);
  g_start_ns.store(SteadyNowNs(), std::memory_order_relaxed);
  g_enabled.store(true, std::memory_order_release);
  return true;
}

void Stop() {
  std::lock_guard<std::mutex> control_lock(g_control_mutex);
  g_enabled.store(false, std::memory_order_release);
  if (g_writer) {
    {
      std::lock_guard<std::mutex> wake_lock(g_wake_mutex);
      g_writer_stop = true;
    }
    g_wake_cv.notify_all();
    g_writer->join();
    delete g_writer;
    g_writer = nullptr;
  }
  std::lock_guard<std::mutex> io_lock(g_io_mutex);
  if (!g_file) {
    return;
  }
  // Lines recorded after the writer's last pass, then no more.
  WriteBufferedLocked(false, true);
  std::fclose(g_file);
  g_file = nullptr;
}

bool Enabled() { return g_enabled.load(std::memory_order_relaxed); }

void Record(const char* event, uint32_t a, uint32_t b, uint32_t c) {
  if (!g_enabled.load(std::memory_order_acquire)) {
    return;
  }
  const int64_t now_ns = SteadyNowNs();
  const uint32_t thread = HostThreadId();
  const long long t = (long long)(now_ns - g_start_ns.load(std::memory_order_relaxed));
  const size_t event_length = std::strlen(event);
  if (event_length <= kMaxFastEventLength) {
    char line[kMaxFastEventLength + 80];
    Append(line, FormatLine(line, t, thread, event, event_length, a, b, c));
    return;
  }
  std::string line(event_length + 80, '\0');
  line.resize(FormatLine(line.data(), t, thread, event, event_length, a, b, c));
  Append(line.data(), line.size());
}

}  // namespace timeline
}  // namespace rex::perf
