/**
 * @file        perf/gpu_gap.cpp
 * @brief       What the GPU does between two frames (wwe2k14 patch, Session 95, item T); see perf/gpu_gap.h
 */
#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/perf/gpu_gap.h>

#include <algorithm>
#include <mutex>
#include <vector>

REXCVAR_DEFINE_BOOL(present_gpu_gap_stats, false, "UI/Presenter",
                    "Session 95: measurement only. GPU timestamps around each presenter paint and each submission, "
                    "and every 600 frames a [pace][gpu-gap] line: GPU time of the frames, of the paints, and the "
                    "time between them on the GPU's own clock");

namespace rex::perf::gpu_gap {
namespace {

struct Span {
  uint64_t begin, end;
  bool paint;
};

constexpr uint32_t kReportFrames = 600;
// The newest spans wait for the next report: the paints are read back later than the submissions around them.
constexpr size_t kKeepSpans = 32;
constexpr size_t kMaxSpans = 16384;

std::mutex g_mutex;
std::vector<Span> g_spans;
double g_ns_per_tick = 0.0;
uint64_t g_mask = UINT64_MAX;
uint32_t g_swaps = 0;
uint64_t g_frames_reported = 0;

// b - a in ticks, 0 if b is not after a (bottom-of-pipe of one against top-of-pipe of the next can cross).
uint64_t Delta(uint64_t a, uint64_t b) {
  const uint64_t d = (b - a) & g_mask;
  return d > (g_mask >> 1) ? 0 : d;
}

void Report() {
  const uint64_t first_frame = g_frames_reported;
  const uint32_t swaps = g_swaps;
  g_frames_reported += swaps;
  g_swaps = 0;
  if (g_ns_per_tick <= 0.0 || g_spans.size() <= kKeepSpans + 1) {
    g_spans.clear();
    return;
  }
  std::sort(g_spans.begin(), g_spans.end(), [](const Span& a, const Span& b) { return a.begin < b.begin; });
  const size_t count = g_spans.size() - kKeepSpans;
  uint64_t busy_submission = 0, busy_paint = 0;
  uint64_t sub_to_paint = 0, paint_to_sub = 0, sub_to_sub = 0, paint_to_paint = 0;
  uint32_t subs = 0, paints = 0, n_sub_to_paint = 0, n_paint_to_sub = 0, n_sub_to_sub = 0;
  for (size_t i = 0; i < count; ++i) {
    const Span& s = g_spans[i];
    (s.paint ? busy_paint : busy_submission) += Delta(s.begin, s.end);
    ++(s.paint ? paints : subs);
    if (!i) {
      continue;
    }
    const Span& p = g_spans[i - 1];
    const uint64_t gap = Delta(p.end, s.begin);
    if (p.paint) {
      if (s.paint) {
        paint_to_paint += gap;
      } else {
        paint_to_sub += gap;
        ++n_paint_to_sub;
      }
    } else if (s.paint) {
      sub_to_paint += gap;
      ++n_sub_to_paint;
    } else {
      sub_to_sub += gap;
      ++n_sub_to_sub;
    }
  }
  const uint64_t wall = Delta(g_spans[0].begin, g_spans[count - 1].end);
  const uint64_t gaps = sub_to_paint + paint_to_sub + sub_to_sub + paint_to_paint;
  const auto ms = [](uint64_t ticks, uint32_t n) { return n ? double(ticks) * g_ns_per_tick / 1.0e6 / n : 0.0; };
  REXLOG_INFO(
      "[pace][gpu-gap] frames {}-{}: {} submissions {:.3f} ms each, {} paints {:.3f} ms each; between them, on the GPU "
      "clock: submission>paint {:.3f} ms (x{}), paint>submission {:.3f} ms (x{}), submission>submission {:.3f} ms (x{}); "
      "wall {:.3f} ms a submission, not covered by a span {:.1f}%",
      first_frame, first_frame + swaps - 1, subs, ms(busy_submission, subs), paints, ms(busy_paint, paints),
      ms(sub_to_paint, n_sub_to_paint), n_sub_to_paint, ms(paint_to_sub, n_paint_to_sub), n_paint_to_sub,
      ms(sub_to_sub, n_sub_to_sub), n_sub_to_sub, ms(wall, subs), wall ? 100.0 * double(gaps) / double(wall) : 0.0);
  g_spans.erase(g_spans.begin(), g_spans.begin() + count);
}

void Add(uint64_t begin_tick, uint64_t end_tick, bool paint) {
  if (g_spans.size() >= kMaxSpans) {
    g_spans.clear();
  }
  g_spans.push_back({begin_tick, end_tick, paint});
}

}  // namespace

bool Enabled() { return REXCVAR_GET(present_gpu_gap_stats); }

void SetTicks(double ns_per_tick, uint64_t tick_mask) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_ns_per_tick = ns_per_tick;
  g_mask = tick_mask;
}

void NoteSubmission(uint64_t begin_tick, uint64_t end_tick, bool is_swap) {
  std::lock_guard<std::mutex> lock(g_mutex);
  Add(begin_tick, end_tick, false);
  if (is_swap && ++g_swaps >= kReportFrames) {
    Report();
  }
}

void NotePaint(uint64_t begin_tick, uint64_t end_tick) {
  std::lock_guard<std::mutex> lock(g_mutex);
  Add(begin_tick, end_tick, true);
}

}  // namespace rex::perf::gpu_gap
