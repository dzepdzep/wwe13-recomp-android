/**
 * @file        ui/vulkan/fsr.cpp
 * @brief       AMD FSR 3 upscaling for the Vulkan backend (wwe2k14 patch, Session 33); see fsr.h
 */

#include <rex/ui/vulkan/fsr.h>

#include <rex/logging.h>

#if defined(REX_HAS_FFX_API) && REX_HAS_FFX_API

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_upscale.h>
#include <ffx_api/vk/ffx_api_vk.h>

namespace rex::ui::vulkan::fsr {
namespace {

PfnFfxCreateContext p_create = nullptr;
PfnFfxDestroyContext p_destroy = nullptr;
PfnFfxQuery p_query = nullptr;
PfnFfxDispatch p_dispatch = nullptr;

std::atomic<bool> g_available{false};
bool g_initialized = false;
// Session 37: set once (any thread may read it after g_failed is seen true).
std::atomic<bool> g_failed{false};
std::atomic<bool> g_fail_reason_set{false};
char g_fail_reason[256] = {};

void MarkFailed(const std::string& reason) {
  bool expected = false;
  if (!g_fail_reason_set.compare_exchange_strong(expected, true)) return;
  std::snprintf(g_fail_reason, sizeof(g_fail_reason), "%s", reason.c_str());
  g_failed.store(true, std::memory_order_release);
  REXLOG_WARN("[fsr] FSR turned off for this run: {} (the picture falls back to the plain upscale, no jitter)", reason);
}
VkDevice g_device = VK_NULL_HANDLE;
VkPhysicalDevice g_physical_device = VK_NULL_HANDLE;
PFN_vkGetDeviceProcAddr g_get_device_proc_addr = nullptr;

// The context (submission thread only). Recreated when the sizes change; the old one is destroyed a few evaluations
// later, when the GPU is done with it.
ffxContext g_context = nullptr;
uint32_t g_render_w = 0, g_render_h = 0, g_out_w = 0, g_out_h = 0;
struct Retired {
  ffxContext context;
  uint32_t evaluations_left;
};
std::vector<Retired> g_retired;
std::chrono::steady_clock::time_point g_last_time{};
uint32_t g_evaluations = 0;

// Some drivers expose Vulkan 1.1+ core entry points but not the KHR aliases parts of the FidelityFX backend ask for
// (the same shim as the presenter's).
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddrCompat(VkDevice device, const char* name) {
  if (!name || !g_get_device_proc_addr) return nullptr;
  if (PFN_vkVoidFunction f = g_get_device_proc_addr(device, name)) return f;
  static const char* const kAliases[][2] = {{"vkGetBufferMemoryRequirements2KHR", "vkGetBufferMemoryRequirements2"},
                                            {"vkGetImageMemoryRequirements2KHR", "vkGetImageMemoryRequirements2"},
                                            {"vkBindBufferMemory2KHR", "vkBindBufferMemory2"},
                                            {"vkBindImageMemory2KHR", "vkBindImageMemory2"}};
  for (const auto& a : kAliases)
    if (!std::strcmp(name, a[0])) return g_get_device_proc_addr(device, a[1]);
  return nullptr;
}

FfxApiResource ToResource(const streamline::Image& i, uint32_t width, uint32_t height, uint32_t usage,
                          uint32_t state) {
  VkImageCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.flags = i.flags;
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = i.format;
  info.extent = {width, height, 1};
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = i.usage;
  return ffxApiGetResourceVK(reinterpret_cast<void*>(i.image), ffxApiGetImageResourceDescriptionVK(i.image, info, usage),
                             state);
}

void TickRetired() {
  for (auto it = g_retired.begin(); it != g_retired.end();) {
    if (it->evaluations_left-- == 0) {
      p_destroy(&it->context, nullptr);
      it = g_retired.erase(it);
    } else {
      ++it;
    }
  }
}

bool EnsureContext(uint32_t rw, uint32_t rh, uint32_t ow, uint32_t oh, std::string* reason) {
  if (g_context && g_render_w == rw && g_render_h == rh && g_out_w == ow && g_out_h == oh) return true;
  if (g_context) g_retired.push_back({g_context, 8}), g_context = nullptr;
  ffxCreateContextDescUpscale desc = {};
  desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
  desc.flags = FFX_UPSCALE_ENABLE_DEPTH_INVERTED;  // the game's colour is 8-bit gamma (LDR); reversed depth
  desc.maxRenderSize = {rw, rh};
  desc.maxUpscaleSize = {ow, oh};
  ffxCreateBackendVKDesc backend = {};
  backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
  backend.vkDevice = g_device;
  backend.vkPhysicalDevice = g_physical_device;
  backend.vkDeviceProcAddr = GetDeviceProcAddrCompat;
  desc.header.pNext = &backend.header;
  ffxContext context = nullptr;
  const ffxReturnCode_t r = p_create(&context, &desc.header, nullptr);
  if (r != FFX_API_RETURN_OK || !context) {
    if (reason) *reason = "ffxCreateContext failed (" + std::to_string(r) + ")";
    return false;
  }
  g_context = context;
  g_render_w = rw, g_render_h = rh, g_out_w = ow, g_out_h = oh;
  ffxQueryGetProviderVersion version = {};
  version.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
  const bool named = p_query(&g_context, &version.header) == FFX_API_RETURN_OK && version.versionName;
  REXLOG_INFO("[fsr] context for {}x{} -> {}x{} ({})", rw, rh, ow, oh,
              named ? version.versionName : "provider version unknown");
  return true;
}

}  // namespace

bool Initialize(VkInstance instance, PFN_vkGetInstanceProcAddr get_instance_proc_addr, VkPhysicalDevice physical_device,
                VkDevice device) {
  if (g_initialized) return g_available.load();
  g_initialized = true;
  wchar_t exe[MAX_PATH] = {};
  GetModuleFileNameW(nullptr, exe, MAX_PATH);
  const std::filesystem::path dll = std::filesystem::path(exe).parent_path() / L"amd_fidelityfx_vk.dll";
  HMODULE module = LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!module) {
    const DWORD error = GetLastError();
    REXLOG_WARN("[fsr] {} not found or not loadable (error {}): FSR not available", dll.string(), error);
    MarkFailed("amd_fidelityfx_vk.dll could not be loaded (error " + std::to_string(error) + ")");
    return false;
  }
  p_create = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(module, "ffxCreateContext"));
  p_destroy = reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(module, "ffxDestroyContext"));
  p_query = reinterpret_cast<PfnFfxQuery>(GetProcAddress(module, "ffxQuery"));
  p_dispatch = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(module, "ffxDispatch"));
  g_get_device_proc_addr = get_instance_proc_addr
                               ? reinterpret_cast<PFN_vkGetDeviceProcAddr>(
                                     get_instance_proc_addr(instance, "vkGetDeviceProcAddr"))
                               : nullptr;
  if (!p_create || !p_destroy || !p_query || !p_dispatch || !g_get_device_proc_addr || !device || !physical_device) {
    REXLOG_WARN("[fsr] amd_fidelityfx_vk.dll lacks the ffx-api entry points (or no device): FSR not available");
    MarkFailed("amd_fidelityfx_vk.dll lacks the ffx-api entry points");
    return false;
  }
  g_device = device, g_physical_device = physical_device;
  g_available = true;
  REXLOG_INFO("[fsr] AMD FidelityFX runtime loaded ({})", dll.string());
  return true;
}

bool Available() { return g_available.load() && !g_failed.load(std::memory_order_acquire); }

bool Failed() { return g_failed.load(std::memory_order_acquire); }

void Fail(const std::string& reason) { MarkFailed(reason); }

std::string FailReason() { return Failed() ? std::string(g_fail_reason) : std::string(); }

bool Evaluate(VkCommandBuffer command_buffer, const streamline::DlaaFrame& f, float sharpness, std::string* reason) {
  if (!Available()) {
    if (reason) *reason = Failed() ? "FSR turned off: " + FailReason() : std::string("FSR not available");
    return false;
  }
  const bool upscale = f.out_width && f.out_height;
  const uint32_t ow = upscale ? f.out_width : f.width, oh = upscale ? f.out_height : f.height;
  TickRetired();
  // Session 37: a failed context is not retried every frame (each attempt can take long on a weak GPU).
  if (!EnsureContext(f.width, f.height, ow, oh, reason)) {
    MarkFailed(reason ? *reason : std::string("ffxCreateContext failed"));
    return false;
  }
  const auto now = std::chrono::steady_clock::now();
  float dt_ms = g_evaluations ? std::chrono::duration<float, std::milli>(now - g_last_time).count() : 16.667f;
  dt_ms = std::clamp(dt_ms, 1.0f, 100.0f);
  g_last_time = now;
  ++g_evaluations;
  ffxDispatchDescUpscale d = {};
  d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
  d.commandList = command_buffer;
  d.color = ToResource(f.color_in, f.width, f.height, FFX_API_RESOURCE_USAGE_READ_ONLY,
                       FFX_API_RESOURCE_STATE_COMPUTE_READ);
  d.depth = ToResource(f.depth, f.width, f.height, FFX_API_RESOURCE_USAGE_READ_ONLY,
                       FFX_API_RESOURCE_STATE_COMPUTE_READ);
  d.motionVectors = ToResource(f.motion, f.width, f.height, FFX_API_RESOURCE_USAGE_READ_ONLY,
                               FFX_API_RESOURCE_STATE_COMPUTE_READ);
  d.output = ToResource(f.color_out, ow, oh, FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
  d.exposure = {};
  d.reactive = {};
  d.transparencyAndComposition = {};
  d.jitterOffset = {f.jitter[0], f.jitter[1]};
  // Our vectors are host pixels (current -> previous); mvec_scale turns them into the normalized units FSR wants too.
  d.motionVectorScale = {f.mvec_scale[0], f.mvec_scale[1]};
  d.renderSize = {f.width, f.height};
  d.upscaleSize = {ow, oh};
  d.enableSharpening = sharpness > 0.0f;
  d.sharpness = std::clamp(sharpness, 0.0f, 1.0f);
  d.frameTimeDelta = dt_ms;
  d.preExposure = 1.0f;
  d.reset = f.reset;
  d.cameraNear = f.near_z;
  d.cameraFar = f.far_z;
  d.cameraFovAngleVertical = f.vertical_fov_radians;
  d.viewSpaceToMetersFactor = 0.01f;  // the game's units are centimetres (unconfirmed; only affects FSR's heuristics)
  d.flags = 0;
  const ffxReturnCode_t r = p_dispatch(&g_context, &d.header);
  if (r != FFX_API_RETURN_OK) {
    if (reason) *reason = "ffxDispatch failed (" + std::to_string(r) + ")";
    MarkFailed(reason ? *reason : std::string("ffxDispatch failed"));
    return false;
  }
  return true;
}

}  // namespace rex::ui::vulkan::fsr

#else  // built without the ffx-api headers

namespace rex::ui::vulkan::fsr {
bool Initialize(VkInstance, PFN_vkGetInstanceProcAddr, VkPhysicalDevice, VkDevice) {
  REXLOG_WARN("[fsr] built without the FidelityFX ffx-api headers: FSR not available");
  return false;
}
bool Available() { return false; }
bool Failed() { return true; }
void Fail(const std::string&) {}
std::string FailReason() { return "built without FidelityFX"; }
bool Evaluate(VkCommandBuffer, const streamline::DlaaFrame&, float, std::string* reason) {
  if (reason) *reason = "built without FidelityFX";
  return false;
}
}  // namespace rex::ui::vulkan::fsr

#endif
