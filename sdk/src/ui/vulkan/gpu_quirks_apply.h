// wwe2k14 patch (Session 93, idea 6): see gpu_quirks_apply.cpp.
#pragma once

#include <rex/ui/vulkan/instance.h>

namespace rex::ui::vulkan {

struct GpuQuirksOutcome {
  bool robust_buffer_access = true;       // vulkan_robust_buffer_access: enable robustBufferAccess if supported
  bool allow_custom_border_color = true;  // vulkan_allow_custom_border_color: enable the custom border colour extension
};

// Detects the GPU, evaluates the quirks table, logs the decisions, sets the cvars whose switch is still at its
// default. Runs once per device creation, before the device features are chosen.
GpuQuirksOutcome ApplyGpuQuirks(const VulkanInstance::Functions& ifn, VkPhysicalDevice physical_device,
                                const VkPhysicalDeviceProperties& properties,
                                const VkPhysicalDeviceDriverProperties* driver_properties, bool geometry_shader);

}  // namespace rex::ui::vulkan
