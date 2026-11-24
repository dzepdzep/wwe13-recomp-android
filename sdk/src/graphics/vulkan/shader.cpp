/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <rex/assert.h>
#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/hash.h>  // XXH3 (Session 70: the optimised SPIR-V store)
#include "spirv_opt_store.h"
#include <rex/perf/frame_events.h>  // Session 73 (what a frame did)
#include "../pipeline/shader/spirv_relaxed_precision.h"  // Session 73, gpu_native_ps_relaxed_precision
#include <rex/graphics/vulkan/shader.h>
#include <rex/logging.h>
#include <rex/ui/vulkan/provider.h>

#if REX_HAS_SPIRV_OPT
#include <spirv-tools/optimizer.hpp>
#endif

// wwe2k14 patch (Session 61): SPIRV-Tools' performance passes (spirv-opt -O) on every translated shader before its
// module is made. Session 60 measured them on dumped shaders: vertex shaders -32%, interlock pixel shaders -12.5% in
// size, 39 / 245 ms a shader on this PC - a test switch for an integrated-graphics A/B, not a default.
REXCVAR_DEFINE_BOOL(vulkan_spirv_optimize, false, "GPU",
                    "Session 61: run SPIRV-Tools' performance passes (spirv-opt -O) on each translated shader before "
                    "its Vulkan module is made (slower translation; for GPU A/Bs). The unoptimised code is used if "
                    "the optimiser fails");
REXCVAR_DEFINE_BOOL(vulkan_spirv_optimize_store, true, "GPU",
                    "Session 70: make each Vulkan shader module from the optimised code the precompile stored "
                    "(<cache>/shaders/shareable/TITLE.spvopt) when there is one (exact: the same passes as "
                    "vulkan_spirv_optimize; no cost in the game); false = always the translator's own code");
REXCVAR_DEFINE_BOOL(vulkan_spirv_optimize_collect, false, "GPU",
                    "Session 70 (set by the game's precompile): every module made from code not stored yet is "
                    "optimised on a pool of threads and stored for the next runs (spirv_opt_store.h)");
// wwe2k14 patch (Session 80, HD 520 arc): the relaxed pixel-shader maths of gpu_native_ps_relaxed_precision as real
// 16-bit floats (SPIRV-Tools' convert-relaxed-to-half), for drivers that may ignore the decoration: Intel's Gen9 runs
// fp16 at twice the fp32 rate, and on the Surface's UHD 620 the scene pass is 21-38x the 4090's. The code is made SSA
// first (the performance passes), so the translator's register variables do not convert back and forth. Needs the
// device's shaderFloat16; the characters' shaders stay full precision as with the decoration.
REXCVAR_DEFINE_BOOL(gpu_native_ps_half, false, "GPU/Native",
                    "Session 80: with gpu_native_ps_relaxed_precision, the relaxed pixel-shader maths converted to "
                    "16-bit floats (needs shaderFloat16; looks: judged by eye). Applies to modules made after it is "
                    "set");

namespace rex::graphics::vulkan {

namespace {
#if REX_HAS_SPIRV_OPT
std::atomic<uint64_t> g_spirv_opt_count{0}, g_spirv_opt_us{0}, g_spirv_opt_in{0}, g_spirv_opt_out{0};
std::atomic<uint32_t> g_spirv_opt_failed{0};

// The optimised words of a module, or empty if the optimiser failed (the caller keeps the original).
std::vector<uint32_t> OptimizeSpirv(const uint32_t* words, size_t count, uint64_t hash) {
  std::vector<uint32_t> out;
  if (count < 5) return out;
  // The target environment from the module's SPIR-V version (the translator's: 1.0 or the 1.3+ of newer devices).
  const uint32_t version = words[1];
  const spv_target_env env = version >= 0x10600   ? SPV_ENV_VULKAN_1_3
                             : version >= 0x10400 ? SPV_ENV_VULKAN_1_2
                             : version >= 0x10300 ? SPV_ENV_VULKAN_1_1
                                                  : SPV_ENV_VULKAN_1_0;
  const auto t0 = std::chrono::steady_clock::now();
  spvtools::Optimizer optimizer(env);
  optimizer.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char*) {});
  optimizer.RegisterPerformancePasses();
  spvtools::OptimizerOptions options;
  options.set_run_validator(false);  // (the translator's own output: validated by the driver as before)
  if (!optimizer.Run(words, count, &out, options) || out.size() < 5) {
    out.clear();
    if (g_spirv_opt_failed.fetch_add(1, std::memory_order_relaxed) < 8) {
      REXGPU_WARN("[spirv_opt] shader {:016X}: the optimiser failed, the unoptimised code is used", hash);
    }
    return out;
  }
  const uint64_t us = uint64_t(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
  const uint64_t n = g_spirv_opt_count.fetch_add(1, std::memory_order_relaxed) + 1;
  const uint64_t total_us = g_spirv_opt_us.fetch_add(us, std::memory_order_relaxed) + us;
  const uint64_t in = g_spirv_opt_in.fetch_add(count, std::memory_order_relaxed) + count;
  const uint64_t o = g_spirv_opt_out.fetch_add(out.size(), std::memory_order_relaxed) + out.size();
  if ((n & 63) == 1) {
    REXGPU_INFO("[spirv_opt] {} shaders optimised: {:.1f}% of their size, {:.1f} ms each on average", n,
                100.0 * double(o) / double(in), double(total_us) / 1000.0 / double(n));
  }
  return out;
}

// wwe2k14 patch (Session 80, gpu_native_ps_half): the RelaxedPrecision maths as 16-bit floats, then the clean-up of the
// conversions it inserts. Empty if it failed (the relaxed code is used).
std::atomic<uint32_t> g_half_count{0}, g_half_failed{0};
std::vector<uint32_t> ConvertRelaxedToHalf(const uint32_t* words, size_t count, uint64_t hash) {
  std::vector<uint32_t> out;
  if (count < 5) return out;
  const uint32_t version = words[1];
  const spv_target_env env = version >= 0x10600   ? SPV_ENV_VULKAN_1_3
                             : version >= 0x10400 ? SPV_ENV_VULKAN_1_2
                             : version >= 0x10300 ? SPV_ENV_VULKAN_1_1
                                                  : SPV_ENV_VULKAN_1_0;
  spvtools::Optimizer optimizer(env);
  optimizer.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char*) {});
  optimizer.RegisterPass(spvtools::CreateConvertRelaxedToHalfPass());
  optimizer.RegisterPass(spvtools::CreateSimplificationPass());
  optimizer.RegisterPass(spvtools::CreateRedundancyEliminationPass());
  optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
  spvtools::OptimizerOptions options;
  options.set_run_validator(false);
  if (!optimizer.Run(words, count, &out, options) || out.size() < 5) {
    out.clear();
    if (g_half_failed.fetch_add(1, std::memory_order_relaxed) < 8) {
      REXGPU_WARN("[ps_half] shader {:016X}: the half conversion failed, the relaxed code is used", hash);
    }
    return out;
  }
  const uint32_t n = g_half_count.fetch_add(1, std::memory_order_relaxed) + 1;
  if ((n & 63) == 1) {
    REXGPU_INFO("[ps_half] {} pixel shaders with 16-bit maths ({} failed)", n,
                g_half_failed.load(std::memory_order_relaxed));
  }
  return out;
}
#endif
}  // namespace

// wwe2k14 patch (Session 70): the precompile's optimised SPIR-V (spirv_opt_store.h).
namespace spirv_opt_store {
namespace {
constexpr uint32_t kMagic = 0x4F505358;  // 'XSPO'
constexpr uint32_t kVersion = 1;
struct Header {
  uint32_t magic, version;
};
struct RecordHead {
  uint64_t key, check;
  uint32_t count;
};
std::mutex g_mutex;
std::condition_variable g_cv;
std::unordered_map<uint64_t, std::vector<uint32_t>> g_map;
std::unordered_set<uint64_t> g_queued;
std::deque<std::pair<uint64_t, std::vector<uint32_t>>> g_jobs;
std::vector<std::thread> g_pool;
bool g_quit = false;
uint32_t g_running = 0, g_collected = 0;
FILE* g_file = nullptr;

#if REX_HAS_SPIRV_OPT
void Worker() {
  std::unique_lock<std::mutex> lock(g_mutex);
  for (;;) {
    g_cv.wait(lock, [] { return g_quit || !g_jobs.empty(); });
    if (g_quit) break;
    auto job = std::move(g_jobs.front());
    g_jobs.pop_front();
    ++g_running;
    lock.unlock();
    std::vector<uint32_t> out = OptimizeSpirv(job.second.data(), job.second.size(), job.first);
    lock.lock();
    if (!out.empty() && !g_map.count(job.first)) {
      if (g_file) {
        const RecordHead head = {job.first, XXH3_64bits(out.data(), out.size() * 4), uint32_t(out.size())};
        std::fwrite(&head, sizeof(head), 1, g_file);
        std::fwrite(out.data(), 4, out.size(), g_file);
        std::fflush(g_file);
      }
      g_map.emplace(job.first, std::move(out));
      ++g_collected;
    }
    --g_running;
    g_cv.notify_all();
  }
}
#endif

void CloseLocked(std::unique_lock<std::mutex>& lock) {
  g_quit = true;
  g_jobs.clear();
  g_cv.notify_all();
  std::vector<std::thread> pool = std::move(g_pool);
  g_pool.clear();
  lock.unlock();
  for (std::thread& t : pool) t.join();
  lock.lock();
  g_quit = false;
  g_queued.clear();
  if (g_file) std::fclose(g_file), g_file = nullptr;
}
}  // namespace

void Open(const std::filesystem::path& file) {
  std::unique_lock<std::mutex> lock(g_mutex);
  CloseLocked(lock);
  g_map.clear();
  g_collected = 0;
  // Records until the end or the first bad one (a torn write: the file is cut there).
  uint64_t valid = sizeof(Header);
  bool header_ok = false;
  if (FILE* f = rex::filesystem::OpenFile(file, "rb")) {
    Header h = {};
    header_ok = std::fread(&h, sizeof(h), 1, f) == 1 && h.magic == kMagic && h.version == kVersion;
    RecordHead r;
    while (header_ok && std::fread(&r, sizeof(r), 1, f) == 1 && r.count && r.count < (1u << 24)) {
      std::vector<uint32_t> words(r.count);
      if (std::fread(words.data(), 4, words.size(), f) != words.size() ||
          XXH3_64bits(words.data(), words.size() * 4) != r.check) {
        break;
      }
      valid += sizeof(r) + words.size() * 4;
      g_map.emplace(r.key, std::move(words));
    }
    std::fclose(f);
  }
  g_file = rex::filesystem::OpenFile(file, header_ok ? "r+b" : "wb");
  if (g_file) {
    if (header_ok) {
      rex::filesystem::TruncateStdioFile(g_file, valid);
      std::fseek(g_file, 0, SEEK_END);
    } else {
      const Header h = {kMagic, kVersion};
      std::fwrite(&h, sizeof(h), 1, g_file);
      std::fflush(g_file);
    }
  }
  REXGPU_INFO("[spirv_opt] store {}: {} optimised shaders", rex::path_to_utf8(file), g_map.size());
}

void Close() {
  std::unique_lock<std::mutex> lock(g_mutex);
  CloseLocked(lock);
}

bool Find(uint64_t key, std::vector<uint32_t>& out) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_map.find(key);
  if (it == g_map.end()) return false;
  out = it->second;
  return true;
}

void Collect(uint64_t key, const uint32_t* words, size_t count) {
#if REX_HAS_SPIRV_OPT
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_file || g_map.count(key) || !g_queued.insert(key).second) return;
  g_jobs.emplace_back(key, std::vector<uint32_t>(words, words + count));
  if (g_pool.empty()) {
    const uint32_t n = std::max(1u, std::thread::hardware_concurrency());
    for (uint32_t i = 0; i < n; ++i) g_pool.emplace_back(Worker);
  }
  g_cv.notify_one();
#else
  (void)key, (void)words, (void)count;
#endif
}

uint32_t Pending() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return uint32_t(g_jobs.size()) + g_running;
}

uint32_t Collected() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_collected;
}
}  // namespace spirv_opt_store

VulkanShader::VulkanTranslation::~VulkanTranslation() {
  if (shader_module_) {
    const ui::vulkan::VulkanDevice* const vulkan_device =
        static_cast<const VulkanShader&>(shader()).vulkan_device_;
    vulkan_device->functions().vkDestroyShaderModule(vulkan_device->device(), shader_module_,
                                                     nullptr);
  }
}

VkShaderModule VulkanShader::VulkanTranslation::GetOrCreateShaderModule() {
  if (!is_valid()) {
    return VK_NULL_HANDLE;
  }
  if (shader_module_ != VK_NULL_HANDLE) {
    return shader_module_;
  }
  const ui::vulkan::VulkanDevice* const vulkan_device =
      static_cast<const VulkanShader&>(shader()).vulkan_device_;
  VkShaderModuleCreateInfo shader_module_create_info;
  shader_module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_module_create_info.pNext = nullptr;
  shader_module_create_info.flags = 0;
  shader_module_create_info.codeSize = translated_binary().size();
  shader_module_create_info.pCode = reinterpret_cast<const uint32_t*>(translated_binary().data());
#if REX_HAS_SPIRV_OPT
  // wwe2k14 patch (Session 61, vulkan_spirv_optimize): the module from the optimised copy (translated_binary(), and
  // with it --dump_shaders, stays the translator's output).
  std::vector<uint32_t> optimized;
  // wwe2k14 patch (Session 70): the optimised code the precompile stored for this very translator output, if any
  // (vulkan_spirv_optimize_store); in the precompile (vulkan_spirv_optimize_collect) code not stored yet is optimised
  // on its pool for the next runs - this module stays the translator's own.
  const bool use_store = REXCVAR_GET(vulkan_spirv_optimize_store), collect = REXCVAR_GET(vulkan_spirv_optimize_collect);
  if ((use_store || collect || REXCVAR_GET(vulkan_spirv_optimize)) && translated_binary().size() % 4 == 0) {
    const uint32_t* words = reinterpret_cast<const uint32_t*>(translated_binary().data());
    const size_t count = translated_binary().size() / 4;
    const uint64_t key = XXH3_64bits(words, count * 4);
    const bool hit = use_store && spirv_opt_store::Find(key, optimized);
    if (use_store) {  // (how much of a run's code came from the store: every 128 modules)
      static std::atomic<uint32_t> hits{0}, misses{0};
      const uint32_t h = hits.fetch_add(hit ? 1 : 0) + (hit ? 1 : 0), m = misses.fetch_add(hit ? 0 : 1) + (hit ? 0 : 1);
      if (((h + m) & 127) == 0) REXGPU_INFO("[spirv_opt] modules from the store: {} of {}", h, h + m);
    }
    if (!hit) {
      if (REXCVAR_GET(vulkan_spirv_optimize)) {
        optimized = OptimizeSpirv(words, count, shader().ucode_data_hash());
      } else if (collect) {
        spirv_opt_store::Collect(key, words, count);
      }
    }
    if (!optimized.empty()) {
      shader_module_create_info.codeSize = optimized.size() * sizeof(uint32_t);
      shader_module_create_info.pCode = optimized.data();
    }
  }
#endif
  // wwe2k14 patch (Session 73, gpu_native_ps_relaxed_precision): pixel shaders' float maths RelaxedPrecision (mediump)
  // for phone GPUs, on the code the module is made from (the translator's, the optimised store's, or NATIVE's shader
  // swap replacement); translated_binary() and --dump_shaders stay as they were. Off = this code is never touched.
  std::vector<uint32_t> relaxed;
#if REX_HAS_SPIRV_OPT
  // wwe2k14 patch (Session 80, gpu_native_ps_half): SSA code first (the store's or the performance passes'), so the
  // half conversion works on values rather than the translator's register variables.
  std::vector<uint32_t> ssa, half;
  const bool want_half = shader().type() == xenos::ShaderType::kPixel && REXCVAR_GET(gpu_native_ps_half) &&
                         vulkan_device->properties().shaderFloat16 && shader_module_create_info.codeSize % 4 == 0;
  if (want_half && optimized.empty()) {
    ssa = OptimizeSpirv(shader_module_create_info.pCode, shader_module_create_info.codeSize / 4,
                        shader().ucode_data_hash());
    if (!ssa.empty()) {
      shader_module_create_info.codeSize = ssa.size() * sizeof(uint32_t);
      shader_module_create_info.pCode = ssa.data();
    }
  }
#endif
  if (shader().type() == xenos::ShaderType::kPixel && shader_module_create_info.codeSize % 4 == 0 &&
      spirv_relaxed_precision::ApplyIfEnabled(shader_module_create_info.pCode, shader_module_create_info.codeSize / 4,
                                              shader().ucode_data_hash(), modification(), relaxed)) {
    shader_module_create_info.codeSize = relaxed.size() * sizeof(uint32_t);
    shader_module_create_info.pCode = relaxed.data();
#if REX_HAS_SPIRV_OPT
    if (want_half) {
      half = ConvertRelaxedToHalf(relaxed.data(), relaxed.size(), shader().ucode_data_hash());
      if (!half.empty()) {
        shader_module_create_info.codeSize = half.size() * sizeof(uint32_t);
        shader_module_create_info.pCode = half.data();
      }
    }
#endif
  }
  rex::perf::frame_events::Get().shaders.fetch_add(1, std::memory_order_relaxed);  // Session 73
  if (vulkan_device->functions().vkCreateShaderModule(vulkan_device->device(),
                                                      &shader_module_create_info, nullptr,
                                                      &shader_module_) != VK_SUCCESS) {
    REXGPU_ERROR(
        "VulkanShader::VulkanTranslation: Failed to create a Vulkan shader "
        "module for shader {:016X} modification {:016X}",
        shader().ucode_data_hash(), modification());
    MakeInvalid();
    return VK_NULL_HANDLE;
  }
  return shader_module_;
}

VulkanShader::VulkanShader(const ui::vulkan::VulkanDevice* const vulkan_device,
                           const xenos::ShaderType shader_type, const uint64_t ucode_data_hash,
                           const uint32_t* const ucode_dwords, const size_t ucode_dword_count,
                           const std::endian ucode_source_endian)
    : SpirvShader(shader_type, ucode_data_hash, ucode_dwords, ucode_dword_count,
                  ucode_source_endian),
      vulkan_device_(vulkan_device) {
  assert_not_null(vulkan_device);
}

Shader::Translation* VulkanShader::CreateTranslationInstance(uint64_t modification) {
  return new VulkanTranslation(*this, modification);
}

}  // namespace rex::graphics::vulkan
