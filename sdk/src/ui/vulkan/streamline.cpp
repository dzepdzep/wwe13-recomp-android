/**
 * @file        ui/vulkan/streamline.cpp
 * @brief       NVIDIA Streamline (DLSS / DLAA) loader and DLAA evaluation (wwe2k14 patch, Session 18). See
 *              rex/ui/vulkan/streamline.h.
 */
#include <rex/ui/vulkan/streamline.h>

#include <rex/cvar.h>
#include <rex/logging.h>

#include <atomic>
#include <cstring>
#include <filesystem>
#include <mutex>

REXCVAR_DEFINE_STRING(vulkan_streamline_dir, "", "GPU/Vulkan",
                      "DLAA (Session 18): folder with the user's own NVIDIA Streamline runtime (sl.interposer.dll, "
                      "sl.common.dll, sl.dlss.dll, nvngx_dlss.dll). Empty = off (plain Vulkan)");
REXCVAR_DEFINE_BOOL(vulkan_streamline_log_verbose, false, "GPU/Vulkan", "DLAA: Streamline's verbose log");

#if defined(REX_HAS_STREAMLINE) && defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#pragma comment(lib, "wintrust.lib")
#pragma comment(lib, "crypt32.lib")

#include <sl.h>
#include <sl_dlss.h>
#include <sl_security.h>

namespace rex::ui::vulkan::streamline {

namespace {
HMODULE g_module = nullptr;
bool g_active = false;
int g_dlss_state = -1;  // -1 unknown, 0 no, 1 yes
std::mutex g_mutex;
PFun_slInit* p_slInit = nullptr;
PFun_slShutdown* p_slShutdown = nullptr;
PFun_slIsFeatureSupported* p_slIsFeatureSupported = nullptr;
PFun_slGetFeatureFunction* p_slGetFeatureFunction = nullptr;
PFun_slGetNewFrameToken* p_slGetNewFrameToken = nullptr;
PFun_slSetConstants* p_slSetConstants = nullptr;
PFun_slSetTagForFrame* p_slSetTagForFrame = nullptr;
PFun_slEvaluateFeature* p_slEvaluateFeature = nullptr;
PFun_slDLSSSetOptions* p_slDLSSSetOptions = nullptr;
std::wstring g_dir;
VkPhysicalDevice g_physical_device = VK_NULL_HANDLE;
uint64_t g_evaluations = 0;

void LogCallback(sl::LogType type, const char* msg) {
  std::string m(msg ? msg : "");
  while (!m.empty() && (m.back() == '\n' || m.back() == '\r')) m.pop_back();
  switch (type) {
    case sl::LogType::eError: REXLOG_ERROR("[streamline] {}", m); break;
    case sl::LogType::eWarn: REXLOG_WARN("[streamline] {}", m); break;
    default: REXLOG_INFO("[streamline] {}", m); break;
  }
}

template <typename T>
bool Get(T*& out, const char* name) {
  out = reinterpret_cast<T*>(GetProcAddress(g_module, name));
  if (!out) REXLOG_ERROR("[dlaa] Streamline: {} not found in sl.interposer.dll", name);
  return out != nullptr;
}

void ToFloat4x4(sl::float4x4& out, const float m[16]) { std::memcpy(&out, m, 64); }

sl::Resource ToResource(const Image& i, uint32_t w, uint32_t h) {
  sl::Resource r(sl::ResourceType::eTex2d, reinterpret_cast<void*>(i.image), reinterpret_cast<void*>(i.memory),
                 reinterpret_cast<void*>(i.view), uint32_t(i.layout));
  r.width = w, r.height = h, r.nativeFormat = uint32_t(i.format), r.mipLevels = 1, r.arrayLayers = 1;
  r.flags = i.flags, r.usage = i.usage;
  return r;
}
}  // namespace

PFN_vkGetInstanceProcAddr InitializeBeforeInstance() {
  std::lock_guard<std::mutex> lock(g_mutex);
  const std::string dir_setting = REXCVAR_GET(vulkan_streamline_dir);
  if (dir_setting.empty() || g_module) return nullptr;
  std::error_code ec;
  const std::filesystem::path dir = std::filesystem::absolute(std::filesystem::u8path(dir_setting), ec);
  const std::filesystem::path interposer = dir / "sl.interposer.dll";
  if (!std::filesystem::is_regular_file(interposer, ec)) {
    REXLOG_WARN("[dlaa] Streamline: {} not found: DLAA not available", interposer.string());
    return nullptr;
  }
  // Always the secure load (ProgrammingGuide 2.1.1): a module without NVIDIA's signature is not loaded.
  if (!sl::security::verifyEmbeddedSignature(interposer.wstring().c_str())) {
    REXLOG_ERROR("[dlaa] Streamline: {} is not signed by NVIDIA: not loaded", interposer.string());
    return nullptr;
  }
  g_module = LoadLibraryW(interposer.wstring().c_str());
  if (!g_module) {
    REXLOG_ERROR("[dlaa] Streamline: LoadLibrary({}) failed ({})", interposer.string(), GetLastError());
    return nullptr;
  }
  bool ok = Get(p_slInit, "slInit") && Get(p_slShutdown, "slShutdown") &&
            Get(p_slIsFeatureSupported, "slIsFeatureSupported") && Get(p_slGetFeatureFunction, "slGetFeatureFunction") &&
            Get(p_slGetNewFrameToken, "slGetNewFrameToken") && Get(p_slSetConstants, "slSetConstants") &&
            Get(p_slSetTagForFrame, "slSetTagForFrame") && Get(p_slEvaluateFeature, "slEvaluateFeature");
  auto proxy = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(g_module, "vkGetInstanceProcAddr"));
  if (!ok || !proxy) {
    REXLOG_ERROR("[dlaa] Streamline: the interposer lacks functions: not used");
    FreeLibrary(g_module);
    g_module = nullptr;
    return nullptr;
  }
  g_dir = dir.wstring();
  static const wchar_t* paths[1];
  paths[0] = g_dir.c_str();
  static const sl::Feature features[] = {sl::kFeatureDLSS};
  sl::Preferences pref{};
  pref.showConsole = false;
  pref.logLevel = REXCVAR_GET(vulkan_streamline_log_verbose) ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
  pref.pathsToPlugins = paths;
  pref.numPathsToPlugins = 1;
  pref.logMessageCallback = &LogCallback;
  // Manual hooking (we create the device ourselves). No over-the-air updates: nothing is ever downloaded.
  pref.flags = sl::PreferenceFlags::eDisableCLStateTracking | sl::PreferenceFlags::eUseManualHooking |
               sl::PreferenceFlags::eUseFrameBasedResourceTagging;
  pref.featuresToLoad = features;
  pref.numFeaturesToLoad = 1;
  pref.engine = sl::EngineType::eCustom;
  pref.engineVersion = "rexglue-wwe2k14";
  pref.projectId = "4e1c3b7a-2f5d-4c86-9a1e-7b3d0f6c2a91";  // our own id for this local integration
  pref.renderAPI = sl::RenderAPI::eVulkan;
  const sl::Result r = p_slInit(pref, sl::kSDKVersion);
  if (r != sl::Result::eOk) {
    REXLOG_ERROR("[dlaa] Streamline: slInit failed ({}): not used", int(r));
    FreeLibrary(g_module);
    g_module = nullptr;
    return nullptr;
  }
  g_active = true;
  REXLOG_INFO("[dlaa] Streamline loaded from {} (manual hooking, DLSS); Vulkan goes through its proxies",
              dir.string());
  return proxy;
}

bool Active() { return g_active; }

bool DlssAvailable() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_active) return false;
  if (g_dlss_state >= 0) return g_dlss_state == 1;
  sl::AdapterInfo adapter{};
  adapter.vkPhysicalDevice = reinterpret_cast<void*>(g_physical_device);
  const sl::Result r = p_slIsFeatureSupported(sl::kFeatureDLSS, adapter);
  void* fn = nullptr;
  const sl::Result rf =
      r == sl::Result::eOk ? p_slGetFeatureFunction(sl::kFeatureDLSS, "slDLSSSetOptions", fn) : r;
  p_slDLSSSetOptions = reinterpret_cast<PFun_slDLSSSetOptions*>(fn);
  g_dlss_state = (r == sl::Result::eOk && rf == sl::Result::eOk && p_slDLSSSetOptions) ? 1 : 0;
  REXLOG_INFO("[dlaa] Streamline: DLSS {} (slIsFeatureSupported {}, slDLSSSetOptions {})",
              g_dlss_state ? "available" : "NOT available", int(r), int(rf));
  return g_dlss_state == 1;
}

void Shutdown() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_active && p_slShutdown) p_slShutdown();
  g_active = false;
}

bool EvaluateDlaa(VkCommandBuffer command_buffer, const DlaaFrame& f, std::string* reason) {
  return EvaluateDlss(command_buffer, f, reason);
}

bool EvaluateDlss(VkCommandBuffer command_buffer, const DlaaFrame& f, std::string* reason) {
  if (!g_active || g_dlss_state != 1) {
    if (reason) *reason = "DLSS not available";
    return false;
  }
  sl::FrameToken* token = nullptr;
  const uint32_t index = f.frame_index;
  if (p_slGetNewFrameToken(token, &index) != sl::Result::eOk || !token) {
    if (reason) *reason = "slGetNewFrameToken failed";
    return false;
  }
  const sl::ViewportHandle viewport(0);
  // Session 24: DLSS Super Resolution modes; 0 = DLAA (Session 18).
  static const sl::DLSSMode kModes[] = {sl::DLSSMode::eDLAA,        sl::DLSSMode::eMaxPerformance,
                                        sl::DLSSMode::eBalanced,    sl::DLSSMode::eMaxQuality,
                                        sl::DLSSMode::eUltraPerformance, sl::DLSSMode::eUltraQuality};
  const bool upscale = f.mode > 0 && f.mode < 6 && f.out_width && f.out_height;
  const uint32_t out_w = upscale ? f.out_width : f.width, out_h = upscale ? f.out_height : f.height;
  sl::DLSSOptions options{};
  options.mode = upscale ? kModes[f.mode] : sl::DLSSMode::eDLAA;
  options.outputWidth = out_w, options.outputHeight = out_h;
  options.colorBuffersHDR = sl::Boolean::eFalse;  // the game's colour is 8-bit (DLAA: before its own post)
  if (f.preset > 0 && f.preset < int(sl::DLSSPreset::eCount)) {
    const auto p = sl::DLSSPreset(f.preset);
    options.dlaaPreset = options.qualityPreset = options.balancedPreset = p;
    options.performancePreset = options.ultraPerformancePreset = options.ultraQualityPreset = p;
  }
  sl::Result r = p_slDLSSSetOptions(viewport, options);
  if (r != sl::Result::eOk) {
    if (reason) *reason = "slDLSSSetOptions failed (" + std::to_string(int(r)) + ")";
    return false;
  }
  sl::Constants c{};
  ToFloat4x4(c.cameraViewToClip, f.view_to_clip);
  ToFloat4x4(c.clipToCameraView, f.clip_to_view);
  float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  ToFloat4x4(c.clipToLensClip, identity);
  ToFloat4x4(c.clipToPrevClip, f.clip_to_prev_clip);
  ToFloat4x4(c.prevClipToClip, f.prev_clip_to_clip);
  c.jitterOffset = {f.jitter[0], f.jitter[1]};
  c.mvecScale = {f.mvec_scale[0], f.mvec_scale[1]};
  c.cameraPinholeOffset = {0, 0};
  c.cameraPos = {f.camera_pos[0], f.camera_pos[1], f.camera_pos[2]};
  c.cameraUp = {f.camera_up[0], f.camera_up[1], f.camera_up[2]};
  c.cameraRight = {f.camera_right[0], f.camera_right[1], f.camera_right[2]};
  c.cameraFwd = {f.camera_fwd[0], f.camera_fwd[1], f.camera_fwd[2]};
  c.cameraNear = f.near_z, c.cameraFar = f.far_z;
  c.cameraFOV = f.vertical_fov_radians, c.cameraAspectRatio = f.aspect;
  c.depthInverted = f.depth_inverted ? sl::Boolean::eTrue : sl::Boolean::eFalse;
  c.cameraMotionIncluded = sl::Boolean::eTrue;
  c.motionVectors3D = sl::Boolean::eFalse;
  c.reset = f.reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
  r = p_slSetConstants(c, *token, viewport);
  if (r != sl::Result::eOk) {
    if (reason) *reason = "slSetConstants failed (" + std::to_string(int(r)) + ")";
    return false;
  }
  sl::Resource color_in = ToResource(f.color_in, f.width, f.height);
  sl::Resource depth = ToResource(f.depth, f.width, f.height);
  sl::Resource motion = ToResource(f.motion, f.width, f.height);
  sl::Resource color_out = ToResource(f.color_out, out_w, out_h);
  sl::Resource bias = ToResource(f.bias, f.width, f.height);
  const sl::Extent extent{0, 0, f.width, f.height};
  const sl::Extent out_extent{0, 0, out_w, out_h};
  sl::ResourceTag tags[5] = {
      sl::ResourceTag(&color_in, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eOnlyValidNow, &extent),
      sl::ResourceTag(&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eOnlyValidNow, &extent),
      sl::ResourceTag(&motion, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eOnlyValidNow, &extent),
      sl::ResourceTag(&color_out, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eOnlyValidNow,
                      &out_extent),
      sl::ResourceTag(&bias, sl::kBufferTypeBiasCurrentColorHint, sl::ResourceLifecycle::eOnlyValidNow, &extent)};
  auto* cb = reinterpret_cast<sl::CommandBuffer*>(command_buffer);
  r = p_slSetTagForFrame(*token, viewport, tags, f.has_bias ? 5 : 4, cb);
  if (r != sl::Result::eOk) {
    if (reason) *reason = "slSetTagForFrame failed (" + std::to_string(int(r)) + ")";
    return false;
  }
  const sl::BaseStructure* inputs[] = {&viewport};
  r = p_slEvaluateFeature(sl::kFeatureDLSS, *token, inputs, 1, cb);
  if (r != sl::Result::eOk) {
    if (reason) *reason = "slEvaluateFeature failed (" + std::to_string(int(r)) + ")";
    return false;
  }
  ++g_evaluations;
  return true;
}

// The device's physical device (vulkan_device.cpp, after creation).
void SetPhysicalDevice(VkPhysicalDevice physical_device) { g_physical_device = physical_device; }

}  // namespace rex::ui::vulkan::streamline

#else  // no Streamline headers at build time

namespace rex::ui::vulkan::streamline {
PFN_vkGetInstanceProcAddr InitializeBeforeInstance() {
  if (!REXCVAR_GET(vulkan_streamline_dir).empty())
    REXLOG_WARN("[dlaa] vulkan_streamline_dir is set, but this build has no Streamline support (headers not found "
                "at build time)");
  return nullptr;
}
bool Active() { return false; }
bool DlssAvailable() { return false; }
void Shutdown() {}
bool EvaluateDlaa(VkCommandBuffer, const DlaaFrame&, std::string* reason) {
  if (reason) *reason = "built without Streamline";
  return false;
}
bool EvaluateDlss(VkCommandBuffer, const DlaaFrame&, std::string* reason) {
  if (reason) *reason = "built without Streamline";
  return false;
}
void SetPhysicalDevice(VkPhysicalDevice) {}
}  // namespace rex::ui::vulkan::streamline

#endif
