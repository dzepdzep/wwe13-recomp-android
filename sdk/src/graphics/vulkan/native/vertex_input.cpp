// wwe2k14 patch (Session 95, Lane B2): the attribute draws' state. See vertex_input.h.

#include "vertex_input.h"

#include <string>

#include <rex/cvar.h>
#include <rex/logging.h>

REXCVAR_DECLARE(bool, gpu_native_index_no_restart);  // (vertex_mirror.cpp)
REXCVAR_DECLARE(bool, gpu_native_index_verify);      // (same)
REXCVAR_DECLARE(bool, gpu_native_index_stitch);      // (same)
REXCVAR_DECLARE(bool, gpu_native_crowd_input);       // Session 95 (P7): own_crowd.cpp

namespace rex::graphics::vulkan::native {

namespace {
const char* const kRefusalNames[kViRefusalCount] = {"not-typed",   "not-owned-only", "crowd",       "merge",       "xfb",         "fast-clear",  "instance-world", "view-blocks",
                                                    "vs-expand",   "memexport",      "ps-fetch",    "line-loop",   "translated-off", "no-form",  "variant-refused", "attributes",
                                                    "fetch-type",  "fetch-size",     "fetch-endian", "fetch-range", "pool-none",   "pool-full",   "pool-too-big",   "gpu-written",
                                                    "index-range", "identity",       "reset-values", "converted",  "builtin-dma", "no-index-identity", "crowd-mesh"};
}  // namespace

ViForm& VertexInputState::Form(VkShaderModule module, uint64_t modification, uint32_t ext, bool* fresh) {
  uint64_t h = uint64_t(reinterpret_cast<uintptr_t>(module)) * 0x9E3779B97F4A7C15ull;
  h ^= modification + 0x7F4A7C159E3779B9ull + (h << 6) + (h >> 2);
  h ^= uint64_t(ext) * 0xC2B2AE3D27D4EB4Full;
  const auto [it, inserted] = forms_.try_emplace(h);
  *fresh = inserted;
  return it->second;
}

void VertexInputState::FrameTick(uint64_t frame, IndexPool& pool, uint64_t window_changes) {
  const bool quit = quit_req_.load(std::memory_order_acquire) && !quit_ack_.load(std::memory_order_relaxed);
  if (frame == frame_ && !quit) return;
  frame_ = frame;
  struct AckAtExit {
    std::atomic<bool>* ack;
    ~AckAtExit() { if (ack) ack->store(true, std::memory_order_release); }
  } ack_at_exit{quit ? &quit_ack_ : nullptr};
  no_restart_ = REXCVAR_GET(gpu_native_index_no_restart);
  index_verify_ = REXCVAR_GET(gpu_native_index_verify);
  stitch_ = REXCVAR_GET(gpu_native_index_stitch);
  crowd_input_ = REXCVAR_GET(gpu_native_crowd_input);
  // (The storage window's own line resets its count every 600 frames of its own.)
  window_changes_ += window_changes >= window_changes_last_ ? window_changes - window_changes_last_ : window_changes;
  window_changes_last_ = window_changes;
  if (!window_first_frame_) window_first_frame_ = frame;
  if (!quit && frame - window_first_frame_ < 600) return;
  const Counters& n = counters_;
  uint64_t storage = 0;
  std::string reasons;
  for (uint32_t i = 0; i < kViRefusalCount; ++i) {
    storage += n.refused[i];
    if (n.refused[i]) reasons += fmt::format("{}{} {}", reasons.empty() ? "" : ", ", kRefusalNames[i], n.refused[i]);
  }
  const uint64_t attribute = n.ours + n.translated + n.ring;
  // Session 95 (P7): the crowd's own part, at the end of the line (nothing when the switch is off, so the line reads as before).
  std::string crowd;
  if (crowd_input_) {
    std::string why;
    for (uint32_t i = 0; i < kViRefusalCount; ++i) {
      if (n.crowd_why[i]) why += fmt::format("{}{} {}", why.empty() ? "" : ", ", kRefusalNames[i], n.crowd_why[i]);
    }
    crowd = fmt::format("; crowd {{first members state only {}, own runs on attributes {} ({} members), own runs on storage {} {{{}}}, member draws on attributes {}}}", n.crowd_state, n.crowd_runs,
                        n.crowd_run_members, n.refused[kViCrowd], why.empty() ? "none" : why, n.crowd_member_draws);
  }
  if (attribute || storage || n.crowd_state) {
    const IndexPool::Counters& p = pool.counters();
    const uint64_t calls = pool.callback_calls();
    const uint64_t calls_window = calls - callback_calls_last_;  // (callback_calls() is monotonic)
    callback_calls_last_ = calls;
    REXGPU_INFO("[vinput] frames {}-{}: attribute draws {} (ours {}, translated {}, ring {}), storage draws {} {{{}}}, pipeline pending {}, planned but fell back {}; attribute draws with a range uploaded this submission {}; "
                "index pool {{hits {}, converts {}, marks {}, callback calls {}, peak bytes {}, with reset values {}}}; window changes {}; stitched {} ({} guest elements to {}){}",
                window_first_frame_, frame, attribute, n.ours, n.translated, n.ring, storage, reasons.empty() ? "none" : reasons, n.pending, n.fell_back, n.uploaded, p.hits, p.uploads,
                p.invalidations + p.gpu_invalidations, calls_window, p.peak_bytes, n.with_reset, window_changes_, n.stitched, n.stitched_in, n.stitched_out, crowd);
  }
  counters_ = Counters{};
  window_changes_ = 0;
  window_first_frame_ = frame;
}

}  // namespace rex::graphics::vulkan::native
