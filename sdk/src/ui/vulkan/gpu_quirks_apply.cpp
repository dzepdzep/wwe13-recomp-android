// wwe2k14 patch (Session 93, idea 6): the driver quirks table, device side. Fills a DeviceInfo from the real device,
// evaluates the table (gpu_quirks.cpp), logs every decision and applies the ones whose cvar is still at its default.
// Called from VulkanDevice::CreateIfSupported before the device features are chosen.

#include <algorithm>
#include <string>
#include <vector>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/ui/vulkan/device.h>
#include <rex/ui/vulkan/gpu_fatal.h>

#include "gpu_quirks.h"
#include "gpu_quirks_apply.h"

REXCVAR_DEFINE_BOOL(gpu_quirks, true, "UI/Vulkan",
                    "Session 93: the driver quirks table. false = detect the GPU and log what the rules would do, but "
                    "apply nothing (for A/B checks)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(gpu_quirks_mobile_no_gs, false, "UI/Vulkan",
                    "Session 93: on phone-class GPUs expand rectangle lists / point sprites in the vertex shader and "
                    "convert quad lists instead of using geometry shaders (the three vulkan_force_* switches, plus "
                    "gpu_native_vs_expand so TRUE NATIVE's typed draws take them). Off until the Session 94 milestone check")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(gpu_quirks_tiler_classic_passes, false, "UI/Vulkan",
                    "Session 94 (matrix 3 rule 4): on every Qualcomm / Arm / Imagination GPU use classic render passes "
                    "(vulkan_dynamic_rendering=false) - tilers merge subpasses and keep the tile memory better with them. "
                    "Off: only the sourced Arm erratum 3786926 versions switch dynamic rendering off. An A/B candidate, unmeasured")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(gpu_quirks_vertex_input_phone, false, "UI/Vulkan",
                    "Session 95 (P8 part 1): on phone-class GPUs set gpu_native_vertex_input=2 and gpu_native_index_no_restart=true (vertices and indices as real "
                    "attributes, so the 128 MiB storage window stops mattering). The master switch: off here, turned on by the game's TRUE NATIVE list on Android builds; never applied to an iGPU class "
                    "(the Surface) until a paired thermal-matched A/B says so. A switch set by hand still wins over the rule")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(gpu_quirks_advisory, false, "UI/Vulkan",
                    "Session 93: also apply the quirks rules marked advisory (sourced from other emulators, not "
                    "measured on this renderer), e.g. one pipeline compile thread on Qualcomm")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_STRING(gpu_quirks_disable, "", "UI/Vulkan",
                      "Session 93: comma separated quirk rule names to skip (the names in the [quirks] log lines)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_INT32(gpu_quirks_eds_allowed, 31, "UI/Vulkan",
                     "Session 93: extended dynamic state the driver is trusted with, bits: 1 EDS1, 2 EDS2, 4 EDS3, 8 "
                     "vertex input dynamic state, 16 EDS3 colour blend. Set by the quirks table; nothing in the SDK "
                     "reads it yet (idea 7)")
    .range(0, 31)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_STRING(gpu_class, "", "UI/Vulkan",
                      "Session 93: READ-ONLY, set at device creation: phone | igpu | desktop | cpu. The low-end tier "
                      "reads it with rex::cvar::GetFlagByName(\"gpu_class\")")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_STRING(gpu_plain_mode, "auto", "UI/Vulkan",
                      "Session 95 (PLAIN): the boring flag set - gpu_owned_transient_attachments, gpu_owned_image_format_list, "
                      "gpu_owned_tiler_passes, gpu_owned_cover_dont_care and present_display_timing off, gpu_d3d_pace_gate off, TRUE NATIVE's Session 95 speed list at its defaults. auto = only on a "
                      "phone-class GPU of no known family (known: Qualcomm stock, Turnip, Arm, Imagination) or a virtual GPU on Android, on = always (the "
                      "launcher's repair ladder, test runs), off = never. A switch given on the command line / config / environment wins")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_INT32(gpu_fatal_exit, -1, "UI/Vulkan",
                     "Session 95 (PLAIN): a required format with no fallback (gpu_quirks_required_missing) or no truth frames under TRUE "
                     "NATIVE writes 'FATAL: GPU lacks ...' and ends the run (logs/exit_request.txt, _Exit(3)) instead of drawing black. "
                     "-1 = on Android only, 0 = never, 1 = always (a PC test)")
    .range(-1, 1)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(vulkan_robust_buffer_access, true, "UI/Vulkan",
                    "Session 93: enable robustBufferAccess when the device supports it (kept on: Arm errata 3779191 / "
                    "3781554 make SSBO reads speculative). Off only for a perf A/B")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_STRING(vulkan_quirk_robust_buffer_access, "auto", "UI/Vulkan",
                      "Session 94: robustBufferAccess rule. auto = vulkan_robust_buffer_access (on everywhere, today's "
                      "behaviour; Arm must keep it on: errata 3779191 / 3781554), on, off, off_non_arm = on on Arm, off "
                      "on every other vendor (for the milestone A/B)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_STRING(gpu_quirks_required_missing, "", "UI/Vulkan",
                      "Session 94: READ-ONLY, set at device creation: comma list of the required format features this "
                      "device lacks that have NO fallback (FORMAT:FEATURE, e.g. R8G8B8A8_UNORM:COLOR_BLEND). Empty = "
                      "nothing missing. The phone report reads it with rex::cvar::GetFlagByName")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(vulkan_allow_custom_border_color, true, "UI/Vulkan",
                    "Session 93: use VK_EXT_custom_border_color when the device has it (the quirks table turns it off "
                    "on Qualcomm / Turnip, where it is broken); off = the texture cache's standard-colour fallback")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_STRING(vulkan_emulate_gpu, "", "UI/Vulkan",
                      "Diagnostic (Session 93): make the quirks table see another GPU, \"vendorhex:driverver:name[:type]\", "
                      "e.g. \"5143:512.800.55:Adreno (TM) 830\" or \"13B5:r49p1:Mali-G715:integrated\" (driverver: a raw "
                      "driverVersion number, rNNpM for Arm, or dotted). The rules apply for real (cvars set, features "
                      "dropped); the device itself is unchanged")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace rex::ui::vulkan {

namespace {

using namespace quirks;

// "Still at its default" = no user-level value: not set by config / environment / command line / the player (rule-set
// values from an earlier attempt were reset at the top of ApplyGpuQuirks and are never "explicit").
bool IsDefault(const std::string& name) { return !rex::cvar::HasExplicitValue(name); }

// Rule values are "auto" flags: not saved by the settings overlay, outranked by any user value, reset on a retry.
bool SetCvar(const std::string& name, const std::string& value) {
  rex::cvar::testing::ScopedLifecycleOverride override_lifecycle;  // init-only flags, set after FinalizeInit
  return rex::cvar::SetAutoFlag(name, value);
}

// Session 95 (PLAIN): force a flag to its boring value. A value from the command line, config or environment is a user's and
// stays; a value set from code (source kRuntime: the game's defaults, d3d_owned) is overridden. Not SetAutoFlag: that is undone by
// the next device attempt's ResetAutoFlags and would hand the flag back to the code that set it.
bool ForcePlain(const char* name, const char* value) {
  const bool user = rex::cvar::HasExplicitValue(name) && rex::cvar::GetFlagSource(name) != rex::cvar::Source::kRuntime;
  const std::string old = rex::cvar::GetFlagByName(name);
  if (user) {
    REXLOG_INFO("[quirks] plain: {} stays {} (set explicitly)", name, old);
    return false;
  }
  if (old == value) {
    REXLOG_INFO("[quirks] plain: {} already {}", name, value);
    return true;
  }
  rex::cvar::testing::ScopedLifecycleOverride override_lifecycle;  // init-only flags, set after FinalizeInit
  if (!rex::cvar::SetFlagByName(name, value)) {
    REXLOG_WARN("[quirks] plain: {} {} to {} FAILED (flag not registered yet?)", name, old, value);
    return false;
  }
  REXLOG_INFO("[quirks] plain: {} {} to {}", name, old, value);
  return true;
}

bool Listed(const std::string& list, const std::string& rule) {
  size_t start = 0;
  while (start <= list.size()) {
    size_t end = list.find(',', start);
    if (end == std::string::npos) end = list.size();
    std::string item = list.substr(start, end - start);
    item.erase(0, item.find_first_not_of(" \t"));
    item.erase(item.find_last_not_of(" \t") + 1);
    if (item == rule) return true;
    start = end + 1;
  }
  return false;
}

// Session 94 (idea 13): the formats / features TRUE NATIVE's own renderer needs, one row per (format, feature bits,
// user). Checked once at device creation through the instance function table, which vulkan_emulate_formats wraps, so an
// emulated GPU is what this sees. Optimal tiling throughout. The phone launcher keeps a small copy of the core of this
// list (game repo, src\launcher_mobile\phone_check.h, CheckVulkan): keep both in step.
constexpr VkFormatFeatureFlags kFSampled = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
constexpr VkFormatFeatureFlags kFLinear = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
constexpr VkFormatFeatureFlags kFColor = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
constexpr VkFormatFeatureFlags kFBlend = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
constexpr VkFormatFeatureFlags kFDepth = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
constexpr VkFormatFeatureFlags kFStorage = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
constexpr VkFormatFeatureFlags kFBlitSrc = VK_FORMAT_FEATURE_BLIT_SRC_BIT;
constexpr VkFormatFeatureFlags kFBlitDst = VK_FORMAT_FEATURE_BLIT_DST_BIT;
constexpr VkFormatFeatureFlags kFXferSrc = VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
constexpr VkFormatFeatureFlags kFXferDst = VK_FORMAT_FEATURE_TRANSFER_DST_BIT;

struct RequiredRow {
  VkFormat format;
  const char* format_name;
  VkFormatFeatureFlags need;
  const char* user;      // file:function that needs it
  const char* fallback;  // nullptr = NONE
  int alt_row;           // >= 0: the fallback only exists when that row is fully supported
  int unless_row;        // >= 0: the row is irrelevant (not reported) when that row is fully supported
  int only_row = -1;     // >= 0: the row is relevant only when that row is fully supported
};

enum RequiredIds { kRowD32S8 = 0, kRowD24S8 = 1 };

const RequiredRow kRequiredRows[] = {
    // 0, 1: the depth / stencil format (post_ds_format_: D32S8 first, else D24S8; EnsurePostObjects' rule is DEPTH|SAMPLED only).
    {VK_FORMAT_D32_SFLOAT_S8_UINT, "D32_SFLOAT_S8_UINT", kFDepth | kFSampled,
     "native_renderer.cpp:EnsurePostObjects (post_ds_format_: owned depth, scratch depth, truth depth)",
     "D24_UNORM_S8_UINT", kRowD24S8, -1},
    {VK_FORMAT_D24_UNORM_S8_UINT, "D24_UNORM_S8_UINT", kFDepth | kFSampled,
     "native_renderer.cpp:EnsurePostObjects (post_ds_format_ when D32S8 is missing)", nullptr, -1, kRowD32S8},
    // The selected depth format's other needs (D32S8 when row 0 passes, else the D24S8 twin).
    {VK_FORMAT_D32_SFLOAT_S8_UINT, "D32_SFLOAT_S8_UINT", kFXferSrc | kFXferDst,
     "owned_steps.cpp:OwnedEnsureDepth (depth image usage TRANSFER_SRC / DST)", nullptr, -1, -1, kRowD32S8},
    {VK_FORMAT_D24_UNORM_S8_UINT, "D24_UNORM_S8_UINT", kFXferSrc | kFXferDst,
     "owned_steps.cpp:OwnedEnsureDepth (depth image usage TRANSFER_SRC / DST)", nullptr, -1, kRowD32S8},
    {VK_FORMAT_D32_SFLOAT_S8_UINT, "D32_SFLOAT_S8_UINT", kFBlitSrc | kFBlitDst,
     "native_truth.cpp:EnsureTruthDrawTwins (truth_depth_blit_; AMD Windows has none)",
     "the depth resample pass (EnsureTruthDepthResampleObjects)", -1, -1, kRowD32S8},
    {VK_FORMAT_D32_SFLOAT_S8_UINT, "D32_SFLOAT_S8_UINT", kFLinear,
     "owned_scene.cpp (owned_depth_direct_ok_ on post_ds_format_, gpu_native_owned_depth_direct)",
     "the depth copy goes through the conversion pass", -1, -1, kRowD32S8},
    {VK_FORMAT_D24_UNORM_S8_UINT, "D24_UNORM_S8_UINT", kFLinear,
     "owned_scene.cpp (owned_depth_direct_ok_ on post_ds_format_, gpu_native_owned_depth_direct)",
     "the depth copy goes through the conversion pass", -1, kRowD32S8},
    // RGBA8: the truth / owned class 1 colour image (OwnedFormat(1), kTruthColorFormats[1]), post class 1.
    {VK_FORMAT_R8G8B8A8_UNORM, "R8G8B8A8_UNORM", kFColor | kFSampled | kFLinear | kFXferSrc | kFXferDst,
     "owned_steps.cpp:OwnedMakeColour / native_truth.cpp:1421 (resample passes, kTruthColorFormats[1])", nullptr, -1, -1},
    {VK_FORMAT_R8G8B8A8_UNORM, "R8G8B8A8_UNORM", kFBlend,
     "native_renderer.cpp:EnsurePostObjects post_class_ok_[1] (blended 8-bit draws)", nullptr, -1, -1},
    {VK_FORMAT_R8G8B8A8_SNORM, "R8G8B8A8_SNORM", kFSampled,
     "NativeRenderer::OwnedView (owned_steps.cpp:5952, :6143; SNORM view of an RGBA8 image)", nullptr, -1, -1},
    // RGBA16_SNORM: class 2 (the packed 16-bit values).
    {VK_FORMAT_R16G16B16A16_SNORM, "R16G16B16A16_SNORM", kFColor | kFSampled | kFXferSrc | kFXferDst,
     "owned_steps.cpp:OwnedMakeColour (kTruthColorFormats[2])", nullptr, -1, -1},
    {VK_FORMAT_R16G16B16A16_SNORM, "R16G16B16A16_SNORM", kFLinear,
     "native_truth.cpp:TruthDrawRatio (resample passes, :1421)", "no resample: the passes draw at the resolution scale", -1, -1},
    {VK_FORMAT_R16G16B16A16_SNORM, "R16G16B16A16_SNORM", kFBlend,
     "native_truth.cpp:EnsureTruthObjects (truth_class2_blend_: a 16-bit blended draw)",
     "blended 16-bit draws are refused (TruthReason) and leave the typed path", -1, -1},
    // The post target of the old (EDRAM buffer) path.
    {VK_FORMAT_R32G32B32A32_SFLOAT, "R32G32B32A32_SFLOAT", kFColor | kFSampled,
     "native_renderer.cpp:EnsurePostObjects (color_features, always required)", nullptr, -1, -1},
    {VK_FORMAT_R32G32B32A32_SFLOAT, "R32G32B32A32_SFLOAT", kFBlend,
     "native_renderer.cpp:EnsurePostObjects (float_blend_needed = !M2dEdramLess)",
     "not needed without an EDRAM buffer, i.e. under TRUE NATIVE (0ba44dd); with one the post stays emulated", -1, -1},
    {VK_FORMAT_R16G16B16A16_UNORM, "R16G16B16A16_UNORM", kFColor | kFSampled,
     "native_renderer.cpp:kColorFormat (the old EDRAM-buffer scene target)", "that path is off (EDRAM buffer only)", -1, -1},
    {VK_FORMAT_D32_SFLOAT, "D32_SFLOAT", kFDepth | kFSampled,
     "native_renderer.cpp:kDepthFormat (the old EDRAM-buffer scene depth)", "that path is off (EDRAM buffer only)", -1, -1},
    // Page images (OwnedPageImageFor, owned_steps.cpp:4191-4199): the texture's own format sampled, the UINT twin stored.
    {VK_FORMAT_R16G16B16A16_UNORM, "R16G16B16A16_UNORM", kFSampled,
     "owned_steps.cpp:OwnedPageImageFor (16-bit page image)", "no page image: the copy goes through memory", -1, -1},
    {VK_FORMAT_R16G16_UNORM, "R16G16_UNORM", kFSampled,
     "owned_steps.cpp:OwnedPageImageFor (k_16_16 page image)", "no page image: the copy goes through memory", -1, -1},
    {VK_FORMAT_R32_SFLOAT, "R32_SFLOAT", kFSampled,
     "owned_steps.cpp:OwnedPageImageFor (32-bit depth page image)", "no page image: the copy goes through memory", -1, -1},
    {VK_FORMAT_R16G16B16A16_UNORM, "R16G16B16A16_UNORM", kFLinear,
     "texture_cache.cpp (16-bit textures filtered)", "k_10_11_11 decodes around it (texture_cache.cpp:4837); other uses: none confirmed", -1, -1},
    {VK_FORMAT_R16G16_UNORM, "R16G16_UNORM", kFLinear, "texture_cache.cpp (k_16_16 textures filtered)", nullptr, -1, -1},
    {VK_FORMAT_R32_SFLOAT, "R32_SFLOAT", kFLinear, "texture_cache.cpp (32-bit float / depth textures filtered)", nullptr, -1, -1},
    {VK_FORMAT_R16G16B16A16_UINT, "R16G16B16A16_UINT", kFStorage,
     "owned_steps.cpp:OwnedPageImageFor (the fill's rgba16ui storage view)", "no page image: the copy goes through memory", -1, -1},
    {VK_FORMAT_R32_UINT, "R32_UINT", kFStorage,
     "owned_steps.cpp:OwnedPageImageFor (the fill's r32ui storage view); texture_cache.cpp native storage",
     "no page image: the copy goes through memory", -1, -1},
    {VK_FORMAT_R16G16B16A16_SNORM, "R16G16B16A16_SNORM", kFSampled,
     "NativeRenderer::OwnedView (owned_steps.cpp:6086 / :6142; signed 16-bit sampler view)", nullptr, -1, -1},
    {VK_FORMAT_R16G16_SNORM, "R16G16_SNORM", kFSampled, "NativeRenderer::OwnedView (owned_steps.cpp:6086; signed pair sampler view)",
     nullptr, -1, -1},
    // Texture cache host formats that have a decode / transcode fallback (texture_cache.cpp:4868-4892).
    {VK_FORMAT_BC1_RGBA_UNORM_BLOCK, "BC1_RGBA_UNORM_BLOCK", kFSampled | kFLinear, "texture_cache.cpp (DXT1)",
     "decompress to RGBA8, or ASTC 4x4 with vulkan_texture_transcode", -1, -1},
    {VK_FORMAT_BC2_UNORM_BLOCK, "BC2_UNORM_BLOCK", kFSampled | kFLinear, "texture_cache.cpp (DXT3)",
     "decompress to RGBA8", -1, -1},
    {VK_FORMAT_BC3_UNORM_BLOCK, "BC3_UNORM_BLOCK", kFSampled | kFLinear, "texture_cache.cpp (DXT5)",
     "decompress to RGBA8", -1, -1},
    {VK_FORMAT_BC4_UNORM_BLOCK, "BC4_UNORM_BLOCK", kFSampled | kFLinear, "texture_cache.cpp (DXT5A)",
     "decompress to R8, or EAC R11 with vulkan_texture_transcode_eac", -1, -1},
    {VK_FORMAT_BC5_UNORM_BLOCK, "BC5_UNORM_BLOCK", kFSampled | kFLinear, "texture_cache.cpp (DXN)",
     "decompress to RG8, or EAC RG11 with vulkan_texture_transcode_eac", -1, -1},
    {VK_FORMAT_R16G16B16A16_SFLOAT, "R16G16B16A16_SFLOAT", kFSampled | kFLinear,
     "texture_cache.cpp (k_16_16_16_16_FLOAT, rgba16_float_filterable)", "filtered in the shader", -1, -1},
};
constexpr int kRequiredRowCount = int(sizeof(kRequiredRows) / sizeof(kRequiredRows[0]));

std::string FeatureNames(VkFormatFeatureFlags bits) {
  static const struct {
    VkFormatFeatureFlags bit;
    const char* name;
  } kNames[] = {{kFSampled, "SAMPLED"},   {kFLinear, "SAMPLED_LINEAR"}, {kFColor, "COLOR_ATTACHMENT"},
                {kFBlend, "COLOR_BLEND"}, {kFDepth, "DEPTH_STENCIL"},   {kFStorage, "STORAGE_IMAGE"},
                {kFBlitSrc, "BLIT_SRC"},  {kFBlitDst, "BLIT_DST"},      {kFXferSrc, "TRANSFER_SRC"},
                {kFXferDst, "TRANSFER_DST"}};
  std::string out;
  for (const auto& n : kNames) {
    if (bits & n.bit) out += std::string(out.empty() ? "" : "|") + n.name;
  }
  return out;
}

// One line per missing row, a summary, and the read-only cvar. `features_of` answers through the instance function
// table, so vulkan_emulate_formats (which wraps it) is what is seen here.
template <typename FeaturesOf>
void CheckRequiredFormats(const FeaturesOf& features_of, bool emulate_no_bc, bool transfer_bits) {
  bool ok[kRequiredRowCount];
  VkFormatFeatureFlags lacking[kRequiredRowCount];
  for (int i = 0; i < kRequiredRowCount; ++i) {
    const RequiredRow& r = kRequiredRows[i];
    VkFormatFeatureFlags have = features_of(r.format);
    if (emulate_no_bc && r.format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && r.format <= VK_FORMAT_BC5_SNORM_BLOCK) have = 0;
    lacking[i] = (transfer_bits ? r.need : (r.need & ~(kFXferSrc | kFXferDst))) & ~have;
    ok[i] = lacking[i] == 0;
  }
  int missing = 0, no_fallback = 0, with_fallback = 0;
  std::string none_list;
  std::string fatal_list;  // Session 95 review: rows lacking more than linear filtering (a Mali without R16G16 filtering plays today)
  for (int i = 0; i < kRequiredRowCount; ++i) {
    const RequiredRow& r = kRequiredRows[i];
    if (ok[i]) continue;
    if (r.unless_row >= 0 && ok[r.unless_row]) continue;
    if (r.only_row >= 0 && !ok[r.only_row]) continue;
    ++missing;
    const char* fallback = (r.alt_row >= 0 && !ok[r.alt_row]) ? nullptr : r.fallback;
    const std::string feature = FeatureNames(lacking[i]);
    if (fallback) {
      ++with_fallback;
      REXLOG_INFO("[quirks] optional missing {} {} for {} -> {}", r.format_name, feature, r.user, fallback);
    } else {
      REXLOG_WARN("[quirks] required MISSING {} {} for {} -> NONE", r.format_name, feature, r.user);
      ++no_fallback;
      none_list += std::string(none_list.empty() ? "" : ",") + r.format_name + ":" + feature;
      if (lacking[i] & ~kFLinear) fatal_list += std::string(fatal_list.empty() ? "" : ",") + r.format_name + ":" + feature;
    }
  }
  REXLOG_INFO("[quirks] required: {} rows checked, {} missing ({} with no fallback, {} with a fallback)",
              kRequiredRowCount, missing, no_fallback, with_fallback);
  {
    rex::cvar::testing::ScopedLifecycleOverride override_lifecycle;
    rex::cvar::SetAutoFlag("gpu_quirks_required_missing", none_list);
  }
  if (!none_list.empty()) {
    REXLOG_ERROR("[quirks] required formats with no fallback are missing: {} (gpu_quirks_required_missing)", none_list);
  }
  if (!fatal_list.empty()) {
    GpuFatalExit(fatal_list);  // Session 95 (PLAIN): a named end instead of a black frame (Android; gpu_fatal_exit); linear-filter-only gaps just log
  }
}

}  // namespace

GpuQuirksOutcome ApplyGpuQuirks(const VulkanInstance::Functions& ifn, VkPhysicalDevice physical_device,
                                const VkPhysicalDeviceProperties& properties,
                                const VkPhysicalDeviceDriverProperties* driver_properties,
                                bool geometry_shader) {
  // A retried device creation (vulkan_provider.cpp:80 / :134) starts clean: values an earlier attempt set are undone.
  rex::cvar::ResetAutoFlags();

  DeviceInfo info;
  info.vendor_id = properties.vendorID;
  info.device_id = properties.deviceID;
  info.device_type = uint32_t(properties.deviceType);
  info.driver_version = properties.driverVersion;
  info.api_version = properties.apiVersion;
  info.device_name = properties.deviceName;
  if (driver_properties) {
    info.driver_id = uint32_t(driver_properties->driverID);
    info.driver_name = driver_properties->driverName;
    info.driver_info = driver_properties->driverInfo;
  }
#if REX_PLATFORM_ANDROID
  info.is_android = true;
#endif
  info.max_storage_buffer_range = properties.limits.maxStorageBufferRange;
  info.max_per_stage_storage_buffers = properties.limits.maxPerStageDescriptorStorageBuffers;
  info.max_dynamic_storage_buffers = properties.limits.maxDescriptorSetStorageBuffersDynamic;
  info.min_storage_buffer_offset_alignment = properties.limits.minStorageBufferOffsetAlignment;
  info.max_push_constants = properties.limits.maxPushConstantsSize;
  info.geometry_shader = geometry_shader;
  info.residency_non_resident_strict = properties.sparseProperties.residencyNonResidentStrict != VK_FALSE;

  VkPhysicalDeviceMemoryProperties memory = {};
  ifn.vkGetPhysicalDeviceMemoryProperties(physical_device, &memory);
  for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
    if (memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      info.device_local_bytes = std::max<uint64_t>(info.device_local_bytes, memory.memoryHeaps[i].size);
    }
  }
  for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
    const VkMemoryPropertyFlags f = memory.memoryTypes[i].propertyFlags;
    if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
      info.device_local_host_visible = true;
    }
  }

  const auto features_of = [&](VkFormat format) {
    VkFormatProperties fp = {};
    ifn.vkGetPhysicalDeviceFormatProperties(physical_device, format, &fp);
    return fp.optimalTilingFeatures;
  };
  const VkFormatFeatureFlags depth_need =
      VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
  const VkFormatFeatureFlags filter_need =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
  info.formats.known = true;
  info.formats.d24s8 = (features_of(VK_FORMAT_D24_UNORM_S8_UINT) & depth_need) == depth_need;
  info.formats.d32s8 = (features_of(VK_FORMAT_D32_SFLOAT_S8_UINT) & depth_need) == depth_need;
  info.formats.rg16_unorm_linear = (features_of(VK_FORMAT_R16G16_UNORM) & filter_need) == filter_need;
  info.formats.rgba16_unorm_linear = (features_of(VK_FORMAT_R16G16B16A16_UNORM) & filter_need) == filter_need;
  info.formats.r32f_linear = (features_of(VK_FORMAT_R32_SFLOAT) & filter_need) == filter_need;
  info.formats.bc1_sampled = (features_of(VK_FORMAT_BC1_RGBA_UNORM_BLOCK) & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;

  // Session 95 (Lane D): the real device, whatever vulkan_emulate_gpu makes the table see (the header line shows the emulated one); logged before the format check so a phone that stops there still has it in its report.
  REXLOG_INFO("[quirks] real device: {}, Vulkan {}.{}, driver {} {} (version 0x{:X}), residencyNonResidentStrict={}",
              properties.deviceName, VK_API_VERSION_MAJOR(properties.apiVersion), VK_API_VERSION_MINOR(properties.apiVersion),
              driver_properties ? driver_properties->driverName : "unknown", driver_properties ? driver_properties->driverInfo : "",
              properties.driverVersion, properties.sparseProperties.residencyNonResidentStrict ? "true" : "false");

  CheckRequiredFormats(features_of, [] {
    const std::string v = rex::cvar::GetFlagByName("vulkan_emulate_no_bc");
    return v == "true" || v == "1";
  }(), VK_API_VERSION_MAJOR(properties.apiVersion) > 1 || VK_API_VERSION_MINOR(properties.apiVersion) >= 1);

  const std::string emulate = REXCVAR_GET(vulkan_emulate_gpu);
  if (!emulate.empty()) {
    std::string error;
    if (ParseEmulateGpu(emulate, info, error)) {
      REXLOG_WARN("[quirks] vulkan_emulate_gpu: the table sees '{}' (the real device is '{}')", emulate,
                  properties.deviceName);
    } else {
      REXLOG_ERROR("[quirks] vulkan_emulate_gpu '{}' ignored: {}", emulate, error);
    }
  }

  Options options;
  options.mobile_no_gs = REXCVAR_GET(gpu_quirks_mobile_no_gs);
  options.tiler_classic_passes = REXCVAR_GET(gpu_quirks_tiler_classic_passes);
  options.plain_mode = REXCVAR_GET(gpu_plain_mode);
  options.vertex_input_phone = REXCVAR_GET(gpu_quirks_vertex_input_phone);
  const Result result = Evaluate(info, options);
  const bool master = REXCVAR_GET(gpu_quirks);
  const bool advisory_on = REXCVAR_GET(gpu_quirks_advisory);
  const std::string disabled = REXCVAR_GET(gpu_quirks_disable);

  REXLOG_INFO("{}", HeaderLine(info, result));
  REXLOG_INFO("[quirks] master gpu_quirks={}, gpu_quirks_advisory={}, gpu_quirks_mobile_no_gs={}", master, advisory_on,
              options.mobile_no_gs);
  {
    rex::cvar::testing::ScopedLifecycleOverride override_lifecycle;
    rex::cvar::SetAutoFlag("gpu_class", GpuClassName(result.cls));
  }
  REXLOG_INFO("[quirks] gpu_class={} (read-only cvar for the tier)", GpuClassName(result.cls));
  for (const Decision& d : result.decisions) {
    if (d.kind == Decision::kInfo) {
      REXLOG_INFO("[quirks] {}: {}", d.rule, d.reason);
      continue;
    }
    const char* why_not = nullptr;
    if (!master) why_not = "gpu_quirks=false";
    else if (Listed(disabled, d.rule)) why_not = "listed in gpu_quirks_disable";
    else if (d.advisory && !advisory_on) why_not = "advisory rule, gpu_quirks_advisory=false";
    else if (!IsDefault(d.cvar)) why_not = "switch set explicitly";
    if (why_not) {
      REXLOG_INFO("[quirks] {}: {}={} NOT APPLIED ({}) ({})", d.rule, d.cvar, d.value, why_not, d.reason);
      continue;
    }
    if (SetCvar(d.cvar, d.value)) {
      REXLOG_INFO("[quirks] {}: {}={} ({})", d.rule, d.cvar, d.value, d.reason);
    } else {
      REXLOG_WARN("[quirks] {}: {}={} could not be set ({})", d.rule, d.cvar, d.value, d.reason);
    }
  }

  // Session 95 (PLAIN): unknown GPUs start on boring paths. The flags below are all read after this point (images and passes at
  // first use, the pace gate per frame); present_display_timing is also read at the extension list earlier in device creation
  // (the extension is then requested and not used: harmless).
  if (!master) {
    REXLOG_INFO("[quirks] plain: not applied (gpu_quirks=false) ({})", result.plain_why);
  } else if (result.plain) {
    REXLOG_INFO("[quirks] plain: ON ({})", result.plain_why);
    static const struct {
      const char* name;
      const char* value;
    } kPlain[] = {
        {"gpu_owned_transient_attachments", "false"},
        {"gpu_owned_image_format_list", "false"},
        {"gpu_owned_tiler_passes", "false"},
        {"gpu_owned_cover_dont_care", "false"},
        {"present_display_timing", "false"},
        {"gpu_d3d_pace_gate", "false"},
        {"gpu_native_vertex_input", "0"},          // (review s95-all: the boring set has no attribute draws, whatever the phone rule said)
        {"gpu_native_index_no_restart", "false"},
        {"present_keep_paint_pipeline", "false"},  // (Session 95: a phone default now, android_pace.h; no real phone has run it)
        // Session 95 (Jarrod's order): the rest of TRUE NATIVE's speed list, on phones since then (the game's d3d_owned.h, the lean set in wwe2k14_app.h), back at each flag's default.
        // Left out: kernel_timestamp_tick_ms, read when the kernel module is made, before the device (runtime.cpp: kernel_init, then the GPU's Setup); this code cannot reach it.
        {"present_every_frame", "false"},
        {"present_render_pass_clear", "true"},
        {"gpu_d3d_wall", "0"},
        {"vulkan_float_constant_runs", "false"},
        {"gpu_native_skin_once", "false"},
        {"gpu_native_skin_with_input", "false"},
        {"gpu_native_skin_bind_lean", "false"},
        {"gpu_native_index_stitch", "false"},
        {"gpu_native_crowd_input", "false"},
        {"wwe2k14_nf_lean", "false"},
        {"wwe2k14_nf_reg_save", "false"},
        {"gpu_d3d_draw_record", "0"},
        {"gpu_d3d_draw_slot", "0"},
        {"gpu_owned_restore_skip", "0"},
        {"gpu_native_post_lite_near", "false"},
        {"gpu_native_wrestler_pow", "false"},
        {"gpu_d3d_inline_merge", "false"},
        {"gpu_native_readback_quiet_frames", "0"},
        {"gpu_tex_bind_memo", "0"},
        {"vulkan_hoist_skin_overlap", "false"},
        {"gpu_tex_planes_unwatched", "false"},
        {"gpu_tex_planes_nohash", "false"},
        {"vulkan_upload_hoist_merge", "false"},
        {"vulkan_hoist_tight_barrier", "false"},
        {"kernel_cs_spin_pause", "0"},
        {"present_ui_idle_wait_ms", "2"},
        {"wwe2k14_low_power_spin_us", "100"},
        {"wwe2k14_low_power_sleep_us", "250"},
        {"wwe2k14_load_turbo_vblank_hz", "16000"},
    };
    for (const auto& p : kPlain) ForcePlain(p.name, p.value);
  } else {
    REXLOG_INFO("[quirks] plain: off ({})", result.plain_why);
  }
  // Session 95: the game's TRUE NATIVE list turns vertex input on for every PC from code. A phone-class GPU in a PC build (an Adreno laptop, a faked phone) stays behind the master switch all the same;
  // a value given by hand stays. A real phone's build turns the master switch on with the list (the game's d3d_owned.h), so this undoes nothing there unless the switch is given as false.
  if (master && result.cls == GpuClass::kPhone && !options.vertex_input_phone) {
    for (const auto& [name, value] : {std::pair<const char*, const char*>{"gpu_native_vertex_input", "0"}, {"gpu_native_index_no_restart", "false"}}) {
      if (rex::cvar::GetFlagSource(name) == rex::cvar::Source::kRuntime && rex::cvar::GetFlagByName(name) != value) ForcePlain(name, value);
    }
  }

  GpuQuirksOutcome outcome;
  outcome.robust_buffer_access = REXCVAR_GET(vulkan_robust_buffer_access);
  {
    // Session 94: vulkan_quirk_robust_buffer_access. auto = the switch above (today); off_non_arm = on only on Arm.
    const std::string rule = REXCVAR_GET(vulkan_quirk_robust_buffer_access);
    if (rule == "on") {
      outcome.robust_buffer_access = true;
    } else if (rule == "off") {
      outcome.robust_buffer_access = false;
    } else if (rule == "off_non_arm") {
      outcome.robust_buffer_access = properties.vendorID == kVendorArm;  // the REAL vendor (info.vendor_id may be emulated)
    } else if (rule != "auto") {
      REXLOG_WARN("[quirks] vulkan_quirk_robust_buffer_access '{}' unknown (auto | on | off | off_non_arm): auto used", rule);
    }
    if (rule != "auto") {
      REXLOG_INFO("[quirks] robust_buffer_access rule '{}' -> robustBufferAccess={} (real vendor 0x{:04X})", rule,
                  outcome.robust_buffer_access, properties.vendorID);
    }
  }
  outcome.allow_custom_border_color = REXCVAR_GET(vulkan_allow_custom_border_color);
  REXLOG_INFO("[quirks] robust_buffer_access: final robustBufferAccess={} (cvar vulkan_robust_buffer_access={}, rule "
              "vulkan_quirk_robust_buffer_access={}; kept on by default: Arm errata 3779191 / 3781554 [S13])",
              outcome.robust_buffer_access, REXCVAR_GET(vulkan_robust_buffer_access),
              REXCVAR_GET(vulkan_quirk_robust_buffer_access));
  REXLOG_INFO("[quirks] custom_border_color: vulkan_allow_custom_border_color={}", outcome.allow_custom_border_color);
  return outcome;
}

}  // namespace rex::ui::vulkan
