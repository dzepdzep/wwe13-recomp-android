#include <algorithm>
#include <cstdio>
#include <vector>
// Native renderer cvar presets. WWE13_NATIVE=0 keeps the plain translated path.
// Every cvar can also be set from env as REX_CVAR_<name>=value via launch options.
#include <cstdlib>
#include <cstring>
#include <rex/cvar.h>
#include <rex/logging.h>

namespace wwe13 {
void ApplyNativeRendererSettings() {
  const char* n = std::getenv("WWE13_NATIVE");
  const int level = n ? std::atoi(n) : 1;
  // The game builds some wrestlers' attires on the GPU (a 2048-wide copy, then a 1024-wide one) and compresses them on
  // the CPU; not read back, the body is black with coloured stripes (HBK in Attitude Era). The latch reads those
  // one-off copies back. Any renderer level.
  if (rex::cvar::GetFlagSource("readback_resolve_attire_latch") == rex::cvar::Source::kDefault)
    rex::cvar::SetFlagByName("readback_resolve_attire_latch", "120");
  if (level <= 0) { REXLOG_INFO("[wwe13] native renderer presets off"); return; }
  // The native renderer ("truth frames" + owned frame, from WWE 2K14's NativeRendererFlags and its command-processor
  // defaults S6/S7/S22/S23), plus the D3D layer (src/k14). wwe2k14_d3d_copy stays off: its clear records leave the
  // stencil unknown to the truth frames and every frame fell back. Each only if left at default.
  struct Kv { const char* k; const char* v; };
  static const Kv kFlags[] = {
      {"gpu_native", "2"},
      {"gpu_fast_clear", "true"},
      {"gpu_native_truth", "true"},
      {"gpu_native_half2_skip_gpr_draws", "true"},
      {"gpu_native_half2", "2"},
      {"gpu_native_lean_draws", "true"},
      {"gpu_native_trust_geometry_pages", "true"},
      {"gpu_native_truth_skip_writeback", "true"},
      {"vulkan_merge_adjacent_barriers", "true"},
      {"gpu_native_truth_direct_resolve_scaled", "true"},
      {"gpu_native_truth_fill_direct", "true"},
      {"vulkan_texture_load_direct_unscaled", "true"},
      {"vulkan_texture_load_direct_unscaled_any_page", "true"},
      {"gpu_native_truth_direct_resolve_scaled_full", "true"},
      {"gpu_native_truth_direct_resolve_scaled_texture", "true"},
      {"gpu_native_truth_defer_dest_use", "true"},
      {"gpu_native_truth_texture_fill_scaled", "true"},
      {"gpu_native_truth_texture_fill_scaled_kinds", "32"},
      {"gpu_native_truth_batch_resolves", "true"},
      {"gpu_native_truth_batch_rename", "true"},
      {"gpu_native_truth_lazy_pass_deps", "true"},
      {"gpu_native_truth_lazy_pass_style", "1"},
      {"gpu_native_truth_rename_depth", "true"},
      {"vulkan_texture_sample_written_early", "true"},
      {"gpu_native_lean_benign_writes", "true"},
      {"gpu_native_truth_skip_writeback_scaled", "true"},
      {"gpu_native_truth_defer_dest_use_scaled", "true"},
      {"gpu_native_truth_batch_fills", "true"},
      {"gpu_native_truth_direct_resolve_scaled_full_texture", "true"},
      {"gpu_native_truth_direct_resolve_full_1x", "true"},
      {"vulkan_texture_native_storage_small", "true"},
      {"gpu_native_truth_fill_direct_small", "true"},
      {"gpu_native_truth_fill_direct_depth", "true"},
      {"gpu_native_truth_direct_resolve_depth", "true"},
      {"gpu_native_truth_direct_resolve_scaled_small", "true"},
      {"vulkan_uniform_pool_device_local", "true"},
      {"vulkan_device_local_upload_min_heap_mb", "128"},
      {"gpu_native_truth_host_rt", "true"},
      {"gpu_native_truth_pass_dont_care", "true"},
      {"gpu_native_truth_color_none", "true"},
      {"gpu_native_vertex_endian_spec", "true"},
      {"gpu_native_ps_texture_spec", "true"},
      {"gpu_native_ps_mul_mode", "2"},
      {"gpu_native_ps_implicit_lod", "true"},
      {"gpu_native_fma", "3"},
      {"gpu_native_truth_drop_dead_resolves", "1"},
      {"gpu_native_vs_mul_mode", "1"},
      {"gpu_native_vs_mul_fetch_finite", "true"},
      {"gpu_native_ps_epilogue_spec", "true"},
      {"gpu_native_shader_swap", "true"},
      {"gpu_native_truth_skip_restores", "true"},
      {"gpu_native_truth_skip_repeat_resolves", "true"},
      {"gpu_native_truth_fill_rename_check", "true"},
      {"gpu_scene_crowd_cull", "2"},
      {"gpu_scene_crowd_thin", "100"},
      {"readback_resolve_batch", "true"},
      {"readback_resolve_batch_min_bytes", "16385"},
      {"vulkan_require_vertex_pipeline_stores_and_atomics", "false"},
      {"vulkan_require_fill_mode_non_solid", "false"},
      {"vulkan_require_geometry_shader", "false"},
      {"vulkan_repeat_draw_fast", "true"},
      {"vulkan_repeat_draw_merge", "true"},
      {"vulkan_repeat_draw_merge_shaders", "183CC64A4C4E96AC,5344BF064B17B7C2"},
      {"wwe2k14_d3d_layer", "true"},
      {"wwe2k14_d3d_copy", "false"},
      {"wwe2k14_d3d_inline", "true"},
      {"gpu_wait_reg_mem_spin", "true"},
      {"vulkan_async_submit", "true"},
      {"vulkan_submit_on_wait", "true"},
      {"vulkan_deferred_descriptor_writes", "true"},
      {"gpu_type0_bulk_writes", "true"},
      {"gpu_type0_block_split", "true"},
      {"gpu_request_ranges_fast", "true"},
      {"vulkan_reuse_texture_bindings", "true"},
      {"vulkan_sampler_param_cache", "true"},
      {"vulkan_constants_dynamic_offsets", "true"},
      {"gpu_rt_cache_skip_unchanged", "true"},
      {"vulkan_viewport_info_cache", "true"},
      {"vulkan_vertex_residency_cache_off", "true"},
      {"vulkan_async_swap", "true"},
      {"vulkan_pipeline_no_submit_wait", "true"},
      {"gpu_request_textures_skip_unchanged", "true"},
      {"gpu_upload_skip_unchanged", "true"},
      {"vulkan_texture_load_direct", "true"},
      {"gpu_native_lean_multi_base", "true"},
      {"gpu_native_lean_state_cache", "true"},
      {"gpu_make_valid_skip_arm", "true"},
      {"vulkan_float_constant_runs", "true"},
      {"gpu_native_owned_frame", "true"},
      {"gpu_native_owned_only", "true"},
      {"gpu_native_owned_frame_tiles_memo", "true"},
      {"gpu_d3d_noring", "1"},
      {"wwe2k14_d3d_noring", "3"},
      {"gpu_skip_noop_draws", "true"},
      {"vulkan_upload_hoist", "2"},
      {"vulkan_texture_load_eager", "true"},
      {"vulkan_texture_load_direct_depth", "true"},
  };
  int set = 0;
  for (auto& kv : kFlags) {
    if (rex::cvar::GetFlagSource(kv.k) == rex::cvar::Source::kDefault) {
      rex::cvar::SetFlagByName(kv.k, kv.v);
      ++set;
    }
  }
  if (!std::getenv("WWE13_FRAME_QUEUE")) setenv("WWE13_FRAME_QUEUE", "2", 1);
  REXLOG_INFO("[wwe13] native renderer: {} of {} settings applied", set, std::size(kFlags));
  REXLOG_INFO("[wwe13] native renderer presets level {}", level);
}
}  // namespace wwe13

// Frame stats (WWE13_FRAME_STATS=1): every 3 s, presents, GPU busy per frame and where the command thread waited
// for the GPU (rex/perf/frame_events.h).
#include <chrono>
#include <thread>
#include <string>
#include <fmt/format.h>
#include <rex/perf/frame_events.h>
#include <rex/perf/timeline.h>
namespace wwe13 {
void StartFrameStats() {
  const char* v = std::getenv("WWE13_FRAME_STATS");
  if (!v || std::strcmp(v, "1") != 0) return;
  auto& c = rex::perf::frame_events::Get();
  c.measure_gpu.store(true, std::memory_order_relaxed);
  // WWE13_CVAR_FILE=<file> (testing): polled every second; each line "name=value" is applied with SetFlagByName, then the
  // file is emptied. "timeline=<s>" writes a <s>-second timeline to logs/timeline_live.csv.
  if (const char* cf = std::getenv("WWE13_CVAR_FILE"); cf && *cf) {
    std::string path(cf);
    std::thread([path] {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        FILE* f = std::fopen(path.c_str(), "r");
        if (!f) continue;
        std::vector<std::string> lines;
        char buf[512];
        while (std::fgets(buf, sizeof(buf), f)) lines.emplace_back(buf);
        std::fclose(f);
        if (lines.empty()) continue;
        if (FILE* w = std::fopen(path.c_str(), "w")) std::fclose(w);
        for (std::string line : lines) {
          while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ')) line.pop_back();
          const size_t eq = line.find('=');
          if (eq == std::string::npos) continue;
          const std::string k = line.substr(0, eq), v = line.substr(eq + 1);
          if (k == "timeline") {
            const int secs = std::max(1, std::atoi(v.c_str()));
            rex::perf::timeline::Start("logs/timeline_live.csv");
            std::this_thread::sleep_for(std::chrono::seconds(secs));
            rex::perf::timeline::Stop();
            REXLOG_INFO("[wwe13-cvar] timeline_live.csv written ({} s)", secs);
            continue;
          }
          const bool ok = rex::cvar::SetFlagByName(k, v);
          REXLOG_INFO("[wwe13-cvar] {} = {} ({})", k, v, ok ? "set" : "unknown");
        }
      }
    }).detach();
  }
  if (const char* tl = std::getenv("WWE13_TIMELINE"); tl && *tl) {
    const int at = std::atoi(tl);
    // WWE13_TIMELINE_EVERY=<s>: then again every <s> seconds into logs/timeline_<n>.csv (12 at most).
    const char* every_env = std::getenv("WWE13_TIMELINE_EVERY");
    const int every = every_env ? std::atoi(every_env) : 0;
    std::thread([at, every] {
      std::this_thread::sleep_for(std::chrono::seconds(at > 0 ? at : 70));
      for (int n = 0; n < (every > 0 ? 12 : 1); ++n) {
        const std::string path = n == 0 ? std::string("logs/timeline.csv") : fmt::format("logs/timeline_{}.csv", n);
        rex::perf::timeline::Start(path.c_str());
        std::this_thread::sleep_for(std::chrono::seconds(2));
        rex::perf::timeline::Stop();
        REXLOG_INFO("[wwe13-stats] timeline written to {}", path);
        if (every > 2) std::this_thread::sleep_for(std::chrono::seconds(every - 2));
      }
    }).detach();
  }
  std::thread([] {
    auto& c = rex::perf::frame_events::Get();
    uint64_t last_presents = 0, last_wait = 0, last_busy = 0, last_by[rex::perf::frame_events::kWaitSiteCount] = {};
    for (;;) {
      std::this_thread::sleep_for(std::chrono::seconds(3));
      const uint64_t p = c.presents.load(), w = c.gpu_wait_us.load(), b = c.gpu_busy_us_total.load();
      const double n = double(p - last_presents) ? double(p - last_presents) : 1.0;
      std::string by;
      for (uint32_t i = 0; i < rex::perf::frame_events::kWaitSiteCount; ++i) {
        const uint64_t x = c.gpu_wait_us_by[i].load();
        by += fmt::format(" {}={:.2f}", rex::perf::frame_events::kWaitSiteNames[i], (x - last_by[i]) / 1000.0 / n);
        last_by[i] = x;
      }
      REXLOG_INFO("[wwe13-stats] fps={:.1f} gpu_frame={:.2f}ms gpu_busy/frame={:.2f}ms cp_wait/frame={:.2f}ms ({} ) pipelines={} textures={}",
                  (p - last_presents) / 3.0, c.gpu_frame_us.load() / 1000.0, (b - last_busy) / 1000.0 / n,
                  (w - last_wait) / 1000.0 / n, by, c.pipelines.load(), c.textures.load());
      last_presents = p; last_wait = w; last_busy = b;
    }
  }).detach();
}
}  // namespace wwe13
