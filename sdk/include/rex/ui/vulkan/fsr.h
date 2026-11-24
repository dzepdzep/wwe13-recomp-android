/**
 * @file        ui/vulkan/fsr.h
 * @brief       AMD FSR 3 upscaling for the Vulkan backend (wwe2k14 patch, Session 33)
 *
 * Optional, default off (gpu_dlss_fsr). The same frame DLSS gets at the swap (streamline::DlaaFrame: colour without
 * the HUD, depth, camera + character motion vectors, jitter) goes through AMD's FSR 3 upscaler instead, on any
 * graphics card. AMD's runtime (amd_fidelityfx_vk.dll, built from the MIT-licensed FidelityFX SDK's ffx-api) is
 * loaded at run time from the exe's folder; without it (or if anything fails) FSR reports itself unavailable and the
 * swap falls back to the plain upscale. Built without the ffx-api headers (thirdparty/FidelityFX-SDK next to the SDK
 * source), everything here is a stub that reports "not built". Upscaling only: no frame generation.
 */
#pragma once

#include <cstdint>
#include <string>

#include <rex/ui/vulkan/api.h>
#include <rex/ui/vulkan/streamline.h>  // DlaaFrame, Image

namespace rex::ui::vulkan::fsr {

// Once, on the command processor's thread: loads the runtime and remembers the device. True if FSR can run.
bool Initialize(VkInstance instance, PFN_vkGetInstanceProcAddr get_instance_proc_addr, VkPhysicalDevice physical_device,
                VkDevice device);
// Initialize succeeded (cached) and FSR has not failed since.
bool Available();
// Session 37: FSR is off for the rest of the run: the runtime could not load, or its context or a dispatch failed
// (a playtester's Intel UHD 620 shook and hung at a loading screen with FSR on). Nothing is retried after that, and
// FailReason() says why. Any thread.
bool Failed();
std::string FailReason();
// Turns FSR off for the rest of the run, as a failure would (gpu_fsr_test_fail: the fallback without a weak GPU).
void Fail(const std::string& reason);
// Records the FSR dispatch into `command_buffer` (a real one, at execution time, on the submission thread). The
// inputs (colour, depth, motion) must be in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, the output in GENERAL; FSR
// leaves them so. frame.mode / out size as for DLSS (out = in for native). False + `reason` on failure.
bool Evaluate(VkCommandBuffer command_buffer, const streamline::DlaaFrame& frame, float sharpness,
              std::string* reason);

}  // namespace rex::ui::vulkan::fsr
