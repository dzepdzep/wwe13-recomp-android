/**
 * @file        graphics/vulkan/disk_pipeline_cache.cpp
 * @brief       A VkPipelineCache kept on disk (wwe2k14 Android port); see the header
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#include "disk_pipeline_cache.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/string.h>
#include <rex/ui/vulkan/device.h>

#if REX_PLATFORM_ANDROID
#define REX_DISK_PIPELINE_CACHE_DEFAULT true
#else
#define REX_DISK_PIPELINE_CACHE_DEFAULT false
#endif
REXCVAR_DEFINE_BOOL(vulkan_disk_pipeline_cache, REX_DISK_PIPELINE_CACHE_DEFAULT, "GPU/Vulkan",
                    "wwe2k14: keep a VkPipelineCache in the shader storage folder (TITLE.vkpc), for drivers without "
                    "their own shader cache (Adreno). Default on on Android, off elsewhere")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace rex::graphics::vulkan::disk_pipeline_cache {

namespace {
std::mutex g_mutex;  // Open / Close / the save itself
const ui::vulkan::VulkanDevice* g_device = nullptr;
VkPipelineCache g_cache = VK_NULL_HANDLE;
std::filesystem::path g_file;
std::atomic<uint32_t> g_created{0};      // created with the cache since the last save
std::atomic<bool> g_saving{false};
std::atomic<int64_t> g_last_save_ms{0};  // steady clock

int64_t NowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Caller holds g_mutex.
void SaveLocked() {
  if (g_cache == VK_NULL_HANDLE || !g_device) return;
  const auto& dfn = g_device->functions();
  size_t size = 0;
  if (dfn.vkGetPipelineCacheData(g_device->device(), g_cache, &size, nullptr) != VK_SUCCESS || !size) return;
  std::vector<uint8_t> data(size);
  if (dfn.vkGetPipelineCacheData(g_device->device(), g_cache, &size, data.data()) != VK_SUCCESS) return;
  // Whole file or nothing: written to .part, then renamed over the old one.
  std::filesystem::path part = g_file;
  part += ".part";
  FILE* f = std::fopen(rex::path_to_utf8(part).c_str(), "wb");
  if (!f) return;
  const bool ok = std::fwrite(data.data(), 1, size, f) == size;
  std::fclose(f);
  std::error_code ec;
  if (ok) std::filesystem::rename(part, g_file, ec);
  REXGPU_INFO("[vkpc] pipeline cache saved: {} KB ({})", size / 1024, ok && !ec ? "ok" : "FAILED");
}
}  // namespace

void Open(const ui::vulkan::VulkanDevice* device, const std::filesystem::path& file) {
  Close();
  if (!REXCVAR_GET(vulkan_disk_pipeline_cache) || !device) return;
  std::vector<uint8_t> data;
  if (FILE* f = std::fopen(rex::path_to_utf8(file).c_str(), "rb")) {
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n > 0) {
      data.resize(size_t(n));
      if (std::fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
    }
    std::fclose(f);
  }
  // Session 95 (PLAIN): the 32-byte header (VkPipelineCacheHeaderVersionOne: size, version, vendor, device, pipelineCacheUUID) must
  // be this device's and this driver's. After a driver update or on another phone the data starts empty instead of being handed to
  // a driver that may accept it and miscompile or crash.
  if (!data.empty()) {
    VkPhysicalDeviceProperties props = {};
    device->vulkan_instance()->functions().vkGetPhysicalDeviceProperties(device->physical_device(), &props);
    uint32_t header[4] = {};
    const bool has_header = data.size() >= 32;
    if (has_header) std::memcpy(header, data.data(), sizeof(header));
    const bool same = has_header && header[0] >= 32 && header[1] == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
                      header[2] == props.vendorID && header[3] == props.deviceID &&
                      std::memcmp(data.data() + 16, props.pipelineCacheUUID, VK_UUID_SIZE) == 0;
    if (!same) {
      REXGPU_INFO("[vkpc] stale: {} KB on disk is not this device's / driver's (header {}, vendor 0x{:04X} device 0x{:04X} vs 0x{:04X} 0x{:04X}): starting empty",
                  data.size() / 1024, has_header ? "read" : "missing", has_header ? header[2] : 0, has_header ? header[3] : 0,
                  props.vendorID, props.deviceID);
      data.clear();
    }
  }
  const auto& dfn = device->functions();
  VkPipelineCacheCreateInfo info = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  info.initialDataSize = data.size();
  info.pInitialData = data.empty() ? nullptr : data.data();
  VkPipelineCache cache = VK_NULL_HANDLE;
  VkResult r = dfn.vkCreatePipelineCache(device->device(), &info, nullptr, &cache);
  if (r != VK_SUCCESS && !data.empty()) {  // unusable data: start empty
    info.initialDataSize = 0;
    info.pInitialData = nullptr;
    r = dfn.vkCreatePipelineCache(device->device(), &info, nullptr, &cache);
  }
  if (r != VK_SUCCESS) {
    REXGPU_WARN("[vkpc] vkCreatePipelineCache failed ({}): pipelines are not cached on disk", int(r));
    return;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  g_device = device;
  g_cache = cache;
  g_file = file;
  g_created = 0;
  g_last_save_ms = NowMs();
  REXGPU_INFO("[vkpc] pipeline cache {}: {} KB loaded", rex::path_to_utf8(file), data.size() / 1024);
}

VkPipelineCache Get() { return g_cache; }

void NoteCreated() {
  if (g_cache == VK_NULL_HANDLE) return;
  const uint32_t created = ++g_created;
  // At most every 5 s, and only with something new; on a worker so the creating thread does not wait.
  if (NowMs() - g_last_save_ms.load() < 5000 || created < 1 || g_saving.exchange(true)) return;
  g_last_save_ms = NowMs();
  g_created = 0;
  std::thread([] {
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      SaveLocked();
    }
    g_saving = false;
  }).detach();
}

void Close() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_cache == VK_NULL_HANDLE) return;
  SaveLocked();
  g_device->functions().vkDestroyPipelineCache(g_device->device(), g_cache, nullptr);
  g_cache = VK_NULL_HANDLE;
  g_device = nullptr;
}

}  // namespace rex::graphics::vulkan::disk_pipeline_cache
