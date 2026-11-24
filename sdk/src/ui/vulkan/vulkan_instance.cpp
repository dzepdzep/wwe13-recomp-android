/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/ui/vulkan/streamline.h>  // wwe2k14 patch (Session 18, DLAA)
#include <rex/ui/vulkan/instance.h>
#include <rex/ui/vulkan/presenter.h>

#if REX_PLATFORM_MAC
#include "vulkan_moltenvk.h"
#endif

REXCVAR_DEFINE_BOOL(vulkan_log_debug_messages, true, "UI/Vulkan", "Log Vulkan debug messages");

// wwe2k14 patch (Session 73, phone GPUs on the PC): formats a weaker phone lacks. "FORMAT[:MASK]" pairs, comma
// separated: FORMAT a VkFormat number, MASK the VkFormatFeatureFlags taken away from its optimal-tiling (and buffer)
// features (hex, default all). E.g. "129" = D24_UNORM_S8_UINT gone; "131-146" = every BC format gone (a range).
REXCVAR_DEFINE_STRING(vulkan_emulate_formats, "", "UI/Vulkan",
                      "Diagnostic (Session 73): format features reported as missing, \"FORMAT[:MASK]\" pairs "
                      "(VkFormat numbers, ranges A-B; MASK hex VkFormatFeatureFlags, default all)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace rex {
namespace ui {
namespace vulkan {

namespace {

// Session 73: vulkan_emulate_formats.
PFN_vkGetPhysicalDeviceFormatProperties g_real_get_format_properties = nullptr;
std::vector<std::pair<uint32_t, uint32_t>> g_emulate_format_ranges;  // (first, last)
std::vector<uint32_t> g_emulate_format_masks;

VKAPI_ATTR void VKAPI_CALL EmulateGetFormatProperties(VkPhysicalDevice physical_device, VkFormat format,
                                                      VkFormatProperties* properties) {
  g_real_get_format_properties(physical_device, format, properties);
  for (size_t i = 0; i < g_emulate_format_ranges.size(); ++i) {
    if (uint32_t(format) >= g_emulate_format_ranges[i].first && uint32_t(format) <= g_emulate_format_ranges[i].second) {
      properties->optimalTilingFeatures &= ~g_emulate_format_masks[i];
      properties->linearTilingFeatures &= ~g_emulate_format_masks[i];
      properties->bufferFeatures &= ~g_emulate_format_masks[i];
    }
  }
}
VulkanLoaderOverride g_loader_override = nullptr;
}  // namespace

void SetVulkanLoaderOverride(VulkanLoaderOverride override_fn) { g_loader_override = override_fn; }

std::unique_ptr<VulkanInstance> VulkanInstance::Create(const bool with_surface,
                                                       const bool try_enable_validation) {
  std::unique_ptr<VulkanInstance> vulkan_instance(new VulkanInstance());

  // Load the RenderDoc API if connected.

  vulkan_instance->renderdoc_api_ = RenderDocAPI::CreateIfConnected();

  // Load the loader library.

  Functions& ifn = vulkan_instance->functions_;

  bool functions_loaded = true;
  bool loader_loaded = false;
#if REX_PLATFORM_MAC
  const MacOSVulkanRuntimePaths macos_runtime_paths = DetectMacOSVulkanRuntimePaths();
  ConfigureMacOSVulkanEnvironment(macos_runtime_paths);

  std::vector<std::filesystem::path> loader_candidates = macos_runtime_paths.loader_candidates;
  loader_candidates.emplace_back(platform::lib_names::kVulkanLoader);
  loader_candidates.emplace_back("libvulkan.dylib");
  loader_candidates.emplace_back("libMoltenVK.dylib");

  std::vector<std::string> attempted_loader_paths;
  attempted_loader_paths.reserve(loader_candidates.size());
  for (const std::filesystem::path& candidate : loader_candidates) {
    if (candidate.empty()) {
      continue;
    }
    attempted_loader_paths.push_back(candidate.string());
    if (vulkan_instance->loader_.Load(candidate)) {
      loader_loaded = true;
      REXLOG_INFO("Loaded Vulkan runtime from {}", candidate.string());
      break;
    }
  }
  if (!loader_loaded) {
    REXLOG_ERROR("Failed to load a Vulkan runtime on macOS");
    for (const std::string& attempted_loader_path : attempted_loader_paths) {
      REXLOG_ERROR("* tried {}", attempted_loader_path);
    }
    return nullptr;
  }
#else
  // wwe2k14 patch (Session 72, Android drivers): the app's own Vulkan library first (a custom driver).
  if (g_loader_override) {
    if (void* handle = g_loader_override()) {
      vulkan_instance->loader_.Adopt(handle);
      loader_loaded = true;
      REXLOG_INFO("Vulkan: using the library the app supplied (custom driver)");
    }
  }
  if (!loader_loaded)
    loader_loaded = vulkan_instance->loader_.Load(platform::lib_names::kVulkanLoader);
  if (!loader_loaded) {
    REXLOG_ERROR("Failed to load {}", platform::lib_names::kVulkanLoader);
    return nullptr;
  }
#endif

#define XE_VULKAN_LOAD_LOADER_FUNCTION(name) \
  functions_loaded &= (ifn.name = vulkan_instance->loader_.GetSymbol<PFN_##name>(#name)) != nullptr;
  XE_VULKAN_LOAD_LOADER_FUNCTION(vkGetInstanceProcAddr);
  XE_VULKAN_LOAD_LOADER_FUNCTION(vkDestroyInstance);
#undef XE_VULKAN_LOAD_LOADER_FUNCTION
  // wwe2k14 patch (Session 18, DLAA): with vulkan_streamline_dir, everything goes through Streamline's proxies
  // (manual hooking: it adds its extensions / queues at instance and device creation and hooks the swapchain).
  if (PFN_vkGetInstanceProcAddr streamline_gipa = streamline::InitializeBeforeInstance()) {
    ifn.vkGetInstanceProcAddr = streamline_gipa;
  }
  if (!functions_loaded) {
    REXLOG_ERROR("Failed to get Vulkan loader function pointers");
    return nullptr;
  }

  // Load global functions.

  functions_loaded &= (ifn.vkCreateInstance = PFN_vkCreateInstance(
                           ifn.vkGetInstanceProcAddr(nullptr, "vkCreateInstance"))) != nullptr;
  functions_loaded &= (ifn.vkEnumerateInstanceExtensionProperties =
                           PFN_vkEnumerateInstanceExtensionProperties(ifn.vkGetInstanceProcAddr(
                               nullptr, "vkEnumerateInstanceExtensionProperties"))) != nullptr;
  functions_loaded &=
      (ifn.vkEnumerateInstanceLayerProperties = PFN_vkEnumerateInstanceLayerProperties(
           ifn.vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceLayerProperties"))) != nullptr;
  if (!functions_loaded) {
    REXLOG_ERROR(
        "Failed to get Vulkan global function pointers via "
        "vkGetInstanceProcAddr");
    return nullptr;
  }
  // Available since Vulkan 1.1. If this is nullptr, it's a Vulkan 1.0 instance.
  ifn.vkEnumerateInstanceVersion = PFN_vkEnumerateInstanceVersion(
      ifn.vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));

  // Get the API version.

  if (ifn.vkEnumerateInstanceVersion) {
    ifn.vkEnumerateInstanceVersion(&vulkan_instance->api_version_);
  }

  // Enable extensions and layers.

  // Name pointers from `requested_extensions` will be used in the enabled
  // extensions vector.
  std::unordered_map<std::string, bool*> requested_extensions;
  if (vulkan_instance->api_version_ >= VK_MAKE_API_VERSION(0, 1, 1, 0)) {
    vulkan_instance->extensions_.ext_1_1_KHR_get_physical_device_properties2 = true;
  } else {
    // #60.
    requested_extensions.emplace(
        "VK_KHR_get_physical_device_properties2",
        &vulkan_instance->extensions_.ext_1_1_KHR_get_physical_device_properties2);
  }
  // #129.
  requested_extensions.emplace("VK_EXT_debug_utils",
                               &vulkan_instance->extensions_.ext_EXT_debug_utils);
  // #395.
  requested_extensions.emplace("VK_KHR_portability_enumeration",
                               &vulkan_instance->extensions_.ext_KHR_portability_enumeration);
  if (with_surface) {
    // #1.
    requested_extensions.emplace("VK_KHR_surface", &vulkan_instance->extensions_.ext_KHR_surface);
#ifdef VK_USE_PLATFORM_XCB_KHR
    // #6.
    requested_extensions.emplace("VK_KHR_xcb_surface",
                                 &vulkan_instance->extensions_.ext_KHR_xcb_surface);
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    // #7.
    requested_extensions.emplace("VK_KHR_wayland_surface",
                                 &vulkan_instance->extensions_.ext_KHR_wayland_surface);
#endif
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    // #9.
    requested_extensions.emplace("VK_KHR_android_surface",
                                 &vulkan_instance->extensions_.ext_KHR_android_surface);
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
    // #10.
    requested_extensions.emplace("VK_KHR_win32_surface",
                                 &vulkan_instance->extensions_.ext_KHR_win32_surface);
#endif
#ifdef VK_USE_PLATFORM_METAL_EXT
    // #217.
    requested_extensions.emplace("VK_EXT_metal_surface",
                                 &vulkan_instance->extensions_.ext_EXT_metal_surface);
#endif
  }

  std::vector<const char*> enabled_extensions;

  std::vector<VkExtensionProperties> supported_implementation_extensions;
  while (true) {
    uint32_t supported_implementation_extension_count = 0;
    const VkResult get_supported_implementation_extension_count_result =
        ifn.vkEnumerateInstanceExtensionProperties(
            nullptr, &supported_implementation_extension_count, nullptr);
    if (get_supported_implementation_extension_count_result != VK_SUCCESS &&
        get_supported_implementation_extension_count_result != VK_INCOMPLETE) {
      REXLOG_WARN("Failed to get the Vulkan instance extension count");
      return nullptr;
    }
    if (supported_implementation_extension_count) {
      supported_implementation_extensions.resize(supported_implementation_extension_count);
      const VkResult get_supported_implementation_extensions_result =
          ifn.vkEnumerateInstanceExtensionProperties(nullptr,
                                                     &supported_implementation_extension_count,
                                                     supported_implementation_extensions.data());
      if (get_supported_implementation_extensions_result == VK_INCOMPLETE) {
        continue;
      }
      if (get_supported_implementation_extensions_result != VK_SUCCESS) {
        REXLOG_WARN("Failed to get the Vulkan instance extensions");
        return nullptr;
      }
    }
    supported_implementation_extensions.resize(supported_implementation_extension_count);
    break;
  }

  for (const VkExtensionProperties& supported_extension : supported_implementation_extensions) {
    const auto requested_extension_it =
        requested_extensions.find(supported_extension.extensionName);
    if (requested_extension_it == requested_extensions.cend()) {
      continue;
    }
    assert_not_null(requested_extension_it->second);
    if (!*requested_extension_it->second) {
      enabled_extensions.emplace_back(requested_extension_it->first.c_str());
      *requested_extension_it->second = true;
    }
  }

  // If enabled layers are not present, will disable all extensions provided by
  // the layers by truncating the enabled extension vector to this size.
  const size_t enabled_implementation_extension_count = enabled_extensions.size();
  std::vector<bool*> enabled_layer_extension_enablement_bools;

  // Name pointers from `requested_layers` will be used in the enabled layer
  // vector.
  std::unordered_map<std::string, bool*> requested_layers;
  bool layer_khronos_validation = false;
  if (try_enable_validation) {
    requested_layers.emplace("VK_LAYER_KHRONOS_validation", &layer_khronos_validation);
  }

  std::vector<const char*> enabled_layers;

  if (!requested_layers.empty()) {
    std::vector<VkLayerProperties> available_layers;
    // "The list of available layers may change at any time due to actions
    // outside of the Vulkan implementation"
    while (true) {
      available_layers.clear();
      uint32_t available_layer_count = 0;
      const VkResult get_available_layer_count_result =
          ifn.vkEnumerateInstanceLayerProperties(&available_layer_count, nullptr);
      if (get_available_layer_count_result != VK_SUCCESS &&
          get_available_layer_count_result != VK_INCOMPLETE) {
        break;
      }
      if (available_layer_count) {
        available_layers.resize(available_layer_count);
        const VkResult get_available_layers_result =
            ifn.vkEnumerateInstanceLayerProperties(&available_layer_count, available_layers.data());
        if (get_available_layers_result == VK_INCOMPLETE) {
          // New layers were added.
          continue;
        }
        if (get_available_layers_result != VK_SUCCESS) {
          available_layers.clear();
          break;
        }
        // In case the second enumeration returned fewer layers.
        available_layers.resize(available_layer_count);
      }
      break;
    }

    if (!available_layers.empty()) {
      std::vector<VkExtensionProperties> supported_layer_extensions;

      for (const VkLayerProperties& available_layer : available_layers) {
        auto requested_layer_it = requested_layers.find(available_layer.layerName);
        if (requested_layer_it == requested_layers.cend()) {
          continue;
        }

        bool got_layer_extensions = true;
        // "Because the list of available layers may change externally between
        // calls to vkEnumerateInstanceExtensionProperties, two calls may
        // retrieve different results if a pLayerName is available in one call
        // but not in another."
        while (true) {
          uint32_t supported_layer_extension_count = 0;
          const VkResult get_supported_layer_extension_count_result =
              ifn.vkEnumerateInstanceExtensionProperties(nullptr, &supported_layer_extension_count,
                                                         nullptr);
          if (get_supported_layer_extension_count_result != VK_SUCCESS &&
              get_supported_layer_extension_count_result != VK_INCOMPLETE) {
            got_layer_extensions = false;
            break;
          }
          if (supported_layer_extension_count) {
            supported_layer_extensions.resize(supported_layer_extension_count);
            const VkResult get_supported_layer_extensions_result =
                ifn.vkEnumerateInstanceExtensionProperties(available_layer.layerName,
                                                           &supported_layer_extension_count,
                                                           supported_layer_extensions.data());
            if (get_supported_layer_extensions_result == VK_INCOMPLETE) {
              continue;
            }
            if (get_supported_layer_extensions_result != VK_SUCCESS) {
              got_layer_extensions = false;
              break;
            }
          }
          supported_layer_extensions.resize(supported_layer_extension_count);
          break;
        }
        if (!got_layer_extensions) {
          // The layer was possibly removed.
          continue;
        }

        for (const VkExtensionProperties& supported_extension : supported_layer_extensions) {
          const auto requested_extension_it =
              requested_extensions.find(supported_extension.extensionName);
          if (requested_extension_it == requested_extensions.cend()) {
            continue;
          }
          assert_not_null(requested_extension_it->second);
          // Don't add the extension to the enabled vector multiple times if
          // provided by the implementation itself or by another layer.
          if (!*requested_extension_it->second) {
            enabled_extensions.emplace_back(requested_extension_it->first.c_str());
            enabled_layer_extension_enablement_bools.push_back(requested_layer_it->second);
            *requested_extension_it->second = true;
          }
        }

        assert_not_null(requested_layer_it->second);
        if (!*requested_layer_it->second) {
          enabled_layers.emplace_back(requested_layer_it->first.c_str());
          *requested_layer_it->second = true;
        }
      }
    }
  }

  // Create the instance.

  VkApplicationInfo application_info;
  application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  application_info.pNext = nullptr;
  application_info.pApplicationName = "Xenia";
  application_info.applicationVersion = 1;
  application_info.pEngineName = nullptr;
  application_info.engineVersion = 0;
  // "The patch version number specified in apiVersion is ignored when creating
  // an instance object."
  // "Vulkan 1.0 implementations were required to return
  // VK_ERROR_INCOMPATIBLE_DRIVER if apiVersion was larger than 1.0."
  application_info.apiVersion = vulkan_instance->api_version_ >= VK_MAKE_API_VERSION(0, 1, 1, 0)
                                    ? VulkanDevice::kHighestUsedApiMinorVersion
                                    : VK_MAKE_API_VERSION(0, 1, 0, 0);

  VkInstanceCreateInfo instance_create_info;
  instance_create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_create_info.pNext = nullptr;
  instance_create_info.flags = 0;
  // VK_KHR_get_physical_device_properties2 is needed to get the portability
  // subset features.
  if (vulkan_instance->extensions_.ext_KHR_portability_enumeration &&
      vulkan_instance->extensions_.ext_1_1_KHR_get_physical_device_properties2) {
    instance_create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  }
  instance_create_info.pApplicationInfo = &application_info;
  instance_create_info.enabledLayerCount = uint32_t(enabled_layers.size());
  instance_create_info.ppEnabledLayerNames = enabled_layers.data();
  instance_create_info.enabledExtensionCount = uint32_t(enabled_extensions.size());
  instance_create_info.ppEnabledExtensionNames = enabled_extensions.data();
  VkResult instance_create_result =
      ifn.vkCreateInstance(&instance_create_info, nullptr, &vulkan_instance->instance_);

  if (instance_create_result == VK_ERROR_LAYER_NOT_PRESENT ||
      instance_create_result == VK_ERROR_EXTENSION_NOT_PRESENT) {
    // A layer was possibly removed. Try without layers.
    for (bool* const extension_enablement : enabled_layer_extension_enablement_bools) {
      *extension_enablement = false;
    }
    for (const std::pair<std::string, bool*>& requested_layer : requested_layers) {
      *requested_layer.second = false;
    }
    instance_create_info.enabledLayerCount = 0;
    instance_create_info.enabledExtensionCount = uint32_t(enabled_implementation_extension_count);
    instance_create_result =
        ifn.vkCreateInstance(&instance_create_info, nullptr, &vulkan_instance->instance_);
  }

  if (instance_create_result != VK_SUCCESS) {
    REXLOG_ERROR("Failed to create a Vulkan instance: {}",
                 vk::to_string(vk::Result(instance_create_result)));
    return nullptr;
  }

  // Load instance functions.

#define XE_UI_VULKAN_FUNCTION(name)                                                            \
  functions_loaded &=                                                                          \
      (ifn.name = PFN_##name(ifn.vkGetInstanceProcAddr(vulkan_instance->instance_, #name))) != \
      nullptr;

  // Vulkan 1.0.
#include <rex/ui/vulkan/functions/instance_1_0.inc>

  // Extensions promoted to a Vulkan version supported by the instance.
#define XE_UI_VULKAN_FUNCTION_PROMOTED(extension_name, core_name)                 \
  functions_loaded &= (ifn.core_name = PFN_##core_name(ifn.vkGetInstanceProcAddr( \
                           vulkan_instance->instance_, #core_name))) != nullptr;
  if (vulkan_instance->api_version_ >= VK_MAKE_API_VERSION(0, 1, 1, 0)) {
#include <rex/ui/vulkan/functions/instance_1_1_khr_get_physical_device_properties2.inc>
  }
#undef XE_UI_VULKAN_FUNCTION_PROMOTED

  // Non-promoted extensions, and extensions promoted to a Vulkan version not
  // supported by the instance.
#define XE_UI_VULKAN_FUNCTION_PROMOTED(extension_name, core_name)                 \
  functions_loaded &= (ifn.core_name = PFN_##core_name(ifn.vkGetInstanceProcAddr( \
                           vulkan_instance->instance_, #extension_name))) != nullptr;
  if (vulkan_instance->api_version_ < VK_MAKE_API_VERSION(0, 1, 1, 0)) {
    if (vulkan_instance->extensions_.ext_1_1_KHR_get_physical_device_properties2) {
#include <rex/ui/vulkan/functions/instance_1_1_khr_get_physical_device_properties2.inc>
    }
  }
#ifdef VK_USE_PLATFORM_XCB_KHR
  if (vulkan_instance->extensions_.ext_KHR_xcb_surface) {
#include <rex/ui/vulkan/functions/instance_khr_xcb_surface.inc>
  }
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
  if (vulkan_instance->extensions_.ext_KHR_wayland_surface) {
#include <rex/ui/vulkan/functions/instance_khr_wayland_surface.inc>
  }
#endif
#ifdef VK_USE_PLATFORM_ANDROID_KHR
  if (vulkan_instance->extensions_.ext_KHR_android_surface) {
#include <rex/ui/vulkan/functions/instance_khr_android_surface.inc>
  }
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
  if (vulkan_instance->extensions_.ext_KHR_win32_surface) {
#include <rex/ui/vulkan/functions/instance_khr_win32_surface.inc>
  }
#endif
#ifdef VK_USE_PLATFORM_METAL_EXT
  if (vulkan_instance->extensions_.ext_EXT_metal_surface) {
#include <rex/ui/vulkan/functions/instance_ext_metal_surface.inc>
  }
#endif
  if (vulkan_instance->extensions_.ext_KHR_surface) {
#include <rex/ui/vulkan/functions/instance_khr_surface.inc>
  }
  if (vulkan_instance->extensions_.ext_EXT_debug_utils) {
#include <rex/ui/vulkan/functions/instance_ext_debug_utils.inc>
  }
#undef XE_UI_VULKAN_FUNCTION_PROMOTED

#undef XE_UI_VULKAN_FUNCTION

  // wwe2k14 patch (Session 73): vulkan_emulate_formats.
  if (const std::string list = REXCVAR_GET(vulkan_emulate_formats); !list.empty() && ifn.vkGetPhysicalDeviceFormatProperties) {
    std::stringstream ss(list);
    std::string item;
    while (std::getline(ss, item, ',')) {
      if (item.empty()) continue;
      const size_t colon = item.find(':');
      const std::string fmt = item.substr(0, colon);
      const uint32_t mask = colon == std::string::npos ? UINT32_MAX : uint32_t(std::stoul(item.substr(colon + 1), nullptr, 16));
      const size_t dash = fmt.find('-');
      const uint32_t first = uint32_t(std::stoul(fmt.substr(0, dash)));
      const uint32_t last = dash == std::string::npos ? first : uint32_t(std::stoul(fmt.substr(dash + 1)));
      g_emulate_format_ranges.emplace_back(first, last);
      g_emulate_format_masks.push_back(mask);
      REXLOG_WARN("Vulkan: emulating formats {}-{} without features 0x{:X} (vulkan_emulate_formats)", first, last, mask);
    }
    g_real_get_format_properties = ifn.vkGetPhysicalDeviceFormatProperties;
    ifn.vkGetPhysicalDeviceFormatProperties = EmulateGetFormatProperties;
  }

  if (!functions_loaded) {
    REXLOG_ERROR("Failed to get all Vulkan instance function pointers");
    return nullptr;
  }

  // Check whether a surface can be created.

  if (with_surface &&
      !VulkanPresenter::GetSurfaceTypesSupportedByInstance(vulkan_instance->extensions_)) {
    REXLOG_ERROR("The Vulkan instance doesn't support surface types used by Xenia");
    return nullptr;
  }

  // Log instance properties.

  REXLOG_INFO("Vulkan instance API version {}.{}.{}. Enabled layers and extensions:",
              VK_VERSION_MAJOR(vulkan_instance->api_version_),
              VK_VERSION_MINOR(vulkan_instance->api_version_),
              VK_VERSION_PATCH(vulkan_instance->api_version_));
  for (uint32_t enabled_layer_index = 0;
       enabled_layer_index < instance_create_info.enabledLayerCount; ++enabled_layer_index) {
    REXLOG_INFO("* {}", instance_create_info.ppEnabledLayerNames[enabled_layer_index]);
  }
  for (uint32_t enabled_extension_index = 0;
       enabled_extension_index < instance_create_info.enabledExtensionCount;
       ++enabled_extension_index) {
    REXLOG_INFO("* {}", instance_create_info.ppEnabledExtensionNames[enabled_extension_index]);
  }

  // Create the debug messenger if requested and available.

  if (vulkan_instance->extensions_.ext_EXT_debug_utils && REXCVAR_GET(vulkan_log_debug_messages)) {
    VkDebugUtilsMessengerCreateInfoEXT debug_utils_messenger_create_info = {
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    auto gpu_logger = rex::GetLogger(rex::log::gpu());
    if (gpu_logger) {
      if (gpu_logger->should_log(spdlog::level::debug)) {
        debug_utils_messenger_create_info.messageSeverity |=
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT;
      }
      if (gpu_logger->should_log(spdlog::level::info)) {
        debug_utils_messenger_create_info.messageSeverity |=
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
      }
      if (gpu_logger->should_log(spdlog::level::warn)) {
        debug_utils_messenger_create_info.messageSeverity |=
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
      }
      if (gpu_logger->should_log(spdlog::level::err)) {
        debug_utils_messenger_create_info.messageSeverity |=
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      }
    }
    // VUID-VkDebugUtilsMessengerCreateInfoEXT-messageSeverity-requiredbitmask:
    // "messageSeverity must not be 0"
    if (debug_utils_messenger_create_info.messageSeverity) {
      debug_utils_messenger_create_info.messageType =
          VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      debug_utils_messenger_create_info.pfnUserCallback = DebugUtilsMessengerCallback;
      debug_utils_messenger_create_info.pUserData = vulkan_instance.get();
      const VkResult debug_utils_messenger_create_result = ifn.vkCreateDebugUtilsMessengerEXT(
          vulkan_instance->instance_, &debug_utils_messenger_create_info, nullptr,
          &vulkan_instance->debug_utils_messenger_);
      if (debug_utils_messenger_create_result != VK_SUCCESS) {
        REXLOG_WARN("Failed to create the Vulkan debug utils messenger: {}",
                    vk::to_string(vk::Result(debug_utils_messenger_create_result)));
      }
    }
  }

  return vulkan_instance;
}

VulkanInstance::~VulkanInstance() {
  if (instance_) {
    if (debug_utils_messenger_ != VK_NULL_HANDLE) {
      functions_.vkDestroyDebugUtilsMessengerEXT(instance_, debug_utils_messenger_, nullptr);
    }

    functions_.vkDestroyInstance(instance_, nullptr);
  }

  loader_.Close();
}

void VulkanInstance::EnumeratePhysicalDevices(
    std::vector<VkPhysicalDevice>& physical_devices_out) const {
  physical_devices_out.clear();
  while (true) {
    uint32_t physical_device_count = 0;
    const VkResult get_physical_device_count_result =
        functions_.vkEnumeratePhysicalDevices(instance_, &physical_device_count, nullptr);
    if ((get_physical_device_count_result != VK_SUCCESS &&
         get_physical_device_count_result != VK_INCOMPLETE) ||
        !physical_device_count) {
      return;
    }
    physical_devices_out.resize(physical_device_count);
    const VkResult get_physical_devices_result = functions_.vkEnumeratePhysicalDevices(
        instance_, &physical_device_count, physical_devices_out.data());
    if (get_physical_devices_result == VK_INCOMPLETE) {
      continue;
    }
    physical_devices_out.resize(get_physical_devices_result == VK_SUCCESS ? physical_device_count
                                                                          : 0);
    return;
  }
}

VkBool32 VulkanInstance::DebugUtilsMessengerCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_types,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data, [[maybe_unused]] void* user_data) {
  // wwe2k14 patch (Session 72 part 8): a message the driver repeats is logged 5 times, then once at each power of ten
  // with its count. A tester's Adreno driver sent one performance note (VKDBGUTILWARN003, ID 0, no name) ~900 times a
  // second: 7 MB of log in 27 s, which rotated the run's start (GPU, driver) out of the kept files and cost frames.
  uint64_t repeats = 0;
  {
    constexpr uint64_t kLoggedInFull = 5;
    static std::mutex repeat_mutex;
    static std::unordered_map<uint64_t, uint64_t> repeat_counts;
    const std::string_view text = callback_data->pMessage ? callback_data->pMessage : "";
    const uint64_t key = std::hash<std::string_view>{}(text) ^
                         (uint64_t(uint32_t(callback_data->messageIdNumber)) << 32) ^ uint64_t(message_severity);
    {
      std::lock_guard<std::mutex> lock(repeat_mutex);
      if (repeat_counts.size() > 4096) repeat_counts.clear();  // many different messages: start counting again
      repeats = ++repeat_counts[key];
    }
    if (repeats > kLoggedInFull) {
      uint64_t p = 10;
      while (p < repeats) p *= 10;
      if (p != repeats) return VK_FALSE;
    }
  }
  std::ostringstream log_str;
  if (repeats > 5) log_str << "[repeated " << repeats << " times] ";

  log_str << "Vulkan " << vk::to_string(vk::DebugUtilsMessageSeverityFlagBitsEXT(message_severity))
          << " (" << vk::to_string(vk::DebugUtilsMessageTypeFlagsEXT(message_types)) << ", ID "
          << callback_data->messageIdNumber;
  if (callback_data->pMessageIdName) {
    log_str << ": " << callback_data->pMessageIdName;
  }
  log_str << ')';

  if (callback_data->pMessage) {
    log_str << ": " << callback_data->pMessage;
  }

  bool annotations_begun = false;
  const auto begin_annotation = [&log_str, &annotations_begun]() {
    log_str << (annotations_begun ? ", " : " (");
    annotations_begun = true;
  };

  for (uint32_t queue_label_index = 0; queue_label_index < callback_data->queueLabelCount;
       ++queue_label_index) {
    begin_annotation();
    log_str << "queue label " << queue_label_index << ": "
            << callback_data->pQueueLabels[queue_label_index].pLabelName;
  }

  for (uint32_t cmd_buf_label_index = 0; cmd_buf_label_index < callback_data->cmdBufLabelCount;
       ++cmd_buf_label_index) {
    begin_annotation();
    log_str << "command buffer label " << cmd_buf_label_index << ": "
            << callback_data->pCmdBufLabels[cmd_buf_label_index].pLabelName;
  }

  for (uint32_t object_index = 0; object_index < callback_data->objectCount; ++object_index) {
    begin_annotation();
    const VkDebugUtilsObjectNameInfoEXT& object_info = callback_data->pObjects[object_index];
    // Lowercase hexadecimal digits in the handle to match the default Vulkan
    // debug utils messenger.
    log_str << "object " << object_index << ": "
            << vk::to_string(vk::ObjectType(object_info.objectType)) << " 0x" << std::hex
            << object_info.objectHandle << std::dec;
    if (object_info.pObjectName) {
      log_str << " '" << object_info.pObjectName << '\'';
    }
  }

  if (annotations_begun) {
    log_str << ')';
  }

  auto msg = log_str.str();
  if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    REXGPU_ERROR("{}", msg);
  } else if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    REXGPU_WARN("{}", msg);
  } else if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    REXGPU_INFO("{}", msg);
  } else {
    REXGPU_DEBUG("{}", msg);
  }

  return VK_FALSE;
}

}  // namespace vulkan
}  // namespace ui
}  // namespace rex
