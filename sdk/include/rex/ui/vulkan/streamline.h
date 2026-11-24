/**
 * @file        ui/vulkan/streamline.h
 * @brief       NVIDIA Streamline (DLSS / DLAA) for the Vulkan backend (wwe2k14 patch, Session 18, DLAA step 3)
 *
 * Optional, default off. With vulkan_streamline_dir set to a folder holding the user's own Streamline runtime
 * (sl.interposer.dll, sl.common.dll, sl.dlss.dll, nvngx_dlss.dll: the signed production DLLs of the Streamline SDK
 * release), InitializeBeforeInstance verifies sl.interposer.dll's signature, loads it, calls slInit (manual hooking,
 * DLSS only) and returns the interposer's vkGetInstanceProcAddr: the Vulkan instance, the device and every function
 * are then resolved through Streamline's proxies (it adds its extensions / queues and hooks the swapchain and
 * present), as ProgrammingGuideManualHooking.md describes for an engine that creates its own device. Without the
 * folder (or if anything fails) nothing changes: plain Vulkan. Built without the Streamline headers
 * (thirdparty/streamline/sdk/include next to the SDK source), everything here is a stub that reports "not built".
 *
 * Nothing from NVIDIA is part of this project: the headers come from the user's own download at build time and the
 * DLLs from the folder the user points at.
 */
#pragma once

#include <cstdint>
#include <string>

#include <rex/ui/vulkan/api.h>  // ReXGlue's Vulkan include (its macros first)

namespace rex::ui::vulkan::streamline {

// Before vkCreateInstance: the interposer's vkGetInstanceProcAddr, or nullptr (not configured, or it failed: logged).
PFN_vkGetInstanceProcAddr InitializeBeforeInstance();
// slInit succeeded (the instance and device go through Streamline).
bool Active();
// After the device exists: whether DLSS is supported on it (cached; the reason is logged once).
bool DlssAvailable();
void Shutdown();
// vulkan_device.cpp: the physical device of the created device (for the DLSS support check).
void SetPhysicalDevice(VkPhysicalDevice physical_device);

struct Image {
  VkImage image = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
  VkImageCreateFlags flags = 0;
  VkImageUsageFlags usage = 0;
};

// One DLAA evaluation (all images width x height; matrices row-major, row vectors, no jitter).
struct DlaaFrame {
  uint32_t frame_index = 0;
  uint32_t width = 0, height = 0;
  Image color_in, depth, motion, color_out;
  float view_to_clip[16], clip_to_view[16], clip_to_prev_clip[16], prev_clip_to_clip[16];
  float jitter[2] = {};      // host pixels
  float mvec_scale[2] = {};  // multiplies the motion vectors (pixels) into normalized units
  float near_z = 0, far_z = 0, vertical_fov_radians = 0, aspect = 1;
  float camera_pos[3] = {}, camera_up[3] = {}, camera_right[3] = {}, camera_fwd[3] = {};
  bool depth_inverted = true;
  bool reset = false;
  // Session 24 (DLSS Super Resolution): 0 = DLAA (output = input size); 1 performance, 2 balanced, 3 quality,
  // 4 ultra performance, 5 ultra quality: color_out is out_width x out_height, the other inputs width x height.
  int mode = 0;
  int preset = 0;  // sl::DLSSPreset for every mode (0 = default)
  uint32_t out_width = 0, out_height = 0;
  // Optional (Session 24 rung 2a): a bias-current-colour mask (width x height, 1 = trust this frame's colour).
  Image bias;
  bool has_bias = false;
};

// Records the DLAA evaluation into `command_buffer` (a real one, at execution time). False + `reason` on failure.
bool EvaluateDlaa(VkCommandBuffer command_buffer, const DlaaFrame& frame, std::string* reason);
// Session 24: DLAA or DLSS Super Resolution (frame.mode), same contract.
bool EvaluateDlss(VkCommandBuffer command_buffer, const DlaaFrame& frame, std::string* reason);

}  // namespace rex::ui::vulkan::streamline
