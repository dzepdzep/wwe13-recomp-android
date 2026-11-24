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

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <rex/perf/frame_events.h>  // wwe2k14 patch (Session 95)
#include <rex/perf/gpu_gap.h>  // wwe2k14 patch (Session 95, item T)
#include <rex/perf/hang_beats.h>  // wwe2k14 patch (Session 95, HANG)
#include <rex/perf/timeline.h>  // wwe2k14 patch (Session 19, LATENCY)
#include <rex/assert.h>
#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/platform.h>
#include <rex/ui/vulkan/presenter.h>
#include <rex/ui/vulkan/util.h>

// wwe2k14 patch (Session 92): log a swapchain-creation failure once per failure episode.
static std::atomic<bool> g_swapchain_failure_logged{false};

// wwe2k14 patch (Session 95, item T, present_gpu_gap_stats): GPU timestamps around each paint's draw command buffer, in
// a query pool of their own (measurement only). File-private, so presenter.h keeps its layout; one presenter paints at
// a time. A slot is read back when its turn comes again, kSlots paints later, without waiting.
namespace {
struct PaintTimestamps {
  static constexpr uint32_t kSlots = 16;
  VkQueryPool pool = VK_NULL_HANDLE;
  bool unsupported = false;
  uint32_t next = 0;
  bool submitted[kSlots] = {};
  double ns_per_tick = 0.0;
  uint64_t mask = 0;
};
PaintTimestamps g_paint_timestamps;

// wwe2k14 patch (Session 95, present_acquire_fence): the swapchain image is acquired with a VkFence the paint thread
// polls instead of a semaphore the paint submission waits on (that wait sits on the same queue as the game's frames).
// File-static like g_paint_timestamps; touched only by the thread that paints (a job would own this struct).
struct AcquireFenceState {
  VkFence fence = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  bool pending = false;  // an image is acquired (index below) and not yet used by a submission
  bool ready = false;    // the fence was seen signalled: the image is usable
  uint32_t image_index = 0;
  std::chrono::steady_clock::time_point acquired_at;
  bool logged = false;
  uint32_t paints = 0;
  uint32_t skipped = 0;
  uint32_t skipped_total = 0;
  double longest_ms = 0.0;
};
AcquireFenceState g_acquire_fence;
}  // namespace

#if defined(REX_HAS_FIDELITYFX_RUNTIME) && REX_HAS_FIDELITYFX_RUNTIME
#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_upscale.h>
#include <ffx_api/vk/ffx_api_vk.h>
#endif

#if REX_PLATFORM_ANDROID
#include <rex/ui/surface_android.h>
#endif
#if REX_PLATFORM_GNU_LINUX
#include <rex/ui/surface_gnulinux.h>
#endif
#if REX_PLATFORM_WIN32
#include <rex/ui/surface_win.h>
#endif
#if REX_PLATFORM_MAC
#include <rex/ui/surface_mac.h>
#endif

REXCVAR_DEFINE_BOOL(present_render_pass_clear, true, "UI/Presenter",
                    "Clear render pass during presentation");

REXCVAR_DEFINE_BOOL(vulkan_allow_present_mode_immediate, true, "UI/Vulkan",
                    "Allow immediate present mode (no vsync)");

REXCVAR_DEFINE_BOOL(vulkan_allow_present_mode_mailbox, true, "UI/Vulkan",
                    "Allow mailbox present mode (triple buffering)");

REXCVAR_DEFINE_BOOL(vulkan_allow_present_mode_fifo_relaxed, true, "UI/Vulkan",
                    "Allow FIFO relaxed present mode");

// wwe2k14 patch (Session 93, idea 14): present_display_timing is defined in vulkan_device.cpp (it decides the extension).
REXCVAR_DECLARE(bool, present_display_timing);
REXCVAR_DECLARE(bool, vulkan_validation_enabled);  // (review s95-all: PrepareForSwapchainRetirement)
// The game's frame rate for the pacing (the game publishes it when it changes; 0 = 60). 60 on a 120 Hz panel = 2 cycles.
REXCVAR_DEFINE_INT32(present_target_fps, 0, "UI/Vulkan",
                     "The frame rate present_display_timing paces to (0 = 60); the game sets it as its frame rate changes");

// wwe2k14 patch (Session 94, idea 9): Android only. Create the swapchain with the display's own transform and draw the
// final pass already rotated, so the phone's compositor does not spend a GPU pass rotating every frame to landscape.
REXCVAR_DEFINE_BOOL(present_pre_rotate, false, "UI/Vulkan",
                    "Session 94: Android only. Create the swapchain with the display's current transform (ROTATE_90/180/270) and "
                    "draw the final pass rotated, so the compositor does not rotate each frame. The presenter's UI drawers "
                    "(ImGui overlays) are skipped while it is rotating. Does nothing on other platforms.");
REXCVAR_DEFINE_BOOL(present_acquire_fence, false, "UI/Vulkan",
                    "Session 95: acquire the swapchain image with a fence the paint thread polls, instead of a semaphore the "
                    "paint submission waits on (that wait is on the same queue as the game's frames, so a late image from "
                    "the compositor stalls them). A paint whose image is not ready is skipped (the newest guest frame stays "
                    "in the mailbox). Off = the semaphore path, unchanged. Logs [pace] present_acquire_fence counters.");
// wwe2k14 patch (Session 95, spike): PaintContext::GuestOutputPaintPipeline::swapchain_format was never written (upstream
// too), so its check against the render pass format held on every paint: each paint waited on the CPU for the paint before
// it to finish on the GPU, destroyed the final pass's pipeline and built it again. In a match the wait was nothing (the GPU
// had finished); with the GPU behind, the paint before sat on the queue behind the frames already there, so frames waited
// here 16-40 ms with the paint not yet submitted, and the paints reached the queue late and bunched: Surface Bryan run 200,
// 13.5 s of the 33 s spent in this step, the frame gaps of 34-40 ms followed by 2 ms.
REXCVAR_DEFINE_BOOL(present_keep_paint_pipeline, false, "UI/Vulkan",
                    "Session 95: note the format the final paint pipeline was built for, so a paint does not wait for the "
                    "paint before it and rebuild the pipeline every time. Off = the old rebuild on every paint.");
REXCVAR_DEFINE_BOOL(present_pre_rotate_invert, false, "UI/Vulkan",
                    "Session 94: swap the 90 and 270 degree directions of present_pre_rotate's picture rotation (if the picture "
                    "comes out upside down in landscape)");

namespace rex {
namespace ui {
namespace vulkan {

namespace {
// wwe2k14 patch (Session 94, present_pre_rotate): the display transform of the current swapchain (turns of 90 degrees
// clockwise; 0 = not pre-rotating), set where the swapchain is created and read by the painting.
constexpr size_t kPrerotEffectCapacity = 16;
std::atomic<uint32_t> g_prerot_turns{0};
// The pipeline variants of the final pass that were built for pre-rotation (per effect), see CreateGuestOutputPaintPipeline.
std::atomic<uint32_t> g_prerot_logged_transform{UINT32_MAX};
VkShaderModule g_prerot_vs = VK_NULL_HANDLE;
VkShaderModule g_prerot_fs[kPrerotEffectCapacity] = {};
// What each effect swapchain pipeline was built for: rotation map (0 = the stock pipeline) and logical size.
uint32_t g_prerot_built_map[kPrerotEffectCapacity] = {};
uint32_t g_prerot_built_width[kPrerotEffectCapacity] = {};
uint32_t g_prerot_built_height[kPrerotEffectCapacity] = {};
// Set when the rotating shaders / pipeline could not be built: the session falls back to an unrotated swapchain.
std::atomic<bool> g_prerot_failed{false};
// The picture rotation (see the rotating vertex shaders) for the display transform turns of 90 degrees clockwise:
// ROTATE_90 rotates clip space by +90 degrees (the Android / Khronos pre-rotation recipe), i.e. map 3.
uint32_t PrerotMapOfTurns(uint32_t turns) {
  if (!(turns & 1) || REXCVAR_GET(present_pre_rotate_invert)) return turns;
  return 4 - turns;
}

// wwe2k14 patch (Session 93, log only; the Adreno 720 stock-driver freeze): a host acquire / present that took 250 ms or more (64 lines
// at most). desired_ns = the VK_GOOGLE_display_timing desired present time given (0 = none), shown relative to the call's start.
void SlowHostPresentCallNote(const char* what, std::chrono::steady_clock::time_point t0, int result, bool fifo, uint64_t desired_ns) {
  const auto t1 = std::chrono::steady_clock::now();
  const int64_t ms = int64_t(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
  if (ms < 250) return;
  static std::atomic<uint32_t> logged{0};
  if (logged.fetch_add(1, std::memory_order_relaxed) >= 64) return;
  const int64_t t0_ns = int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(t0.time_since_epoch()).count());
  REXLOG_WARN("[pace][present][slow] {} took {} ms (VkResult {}, FIFO {}, present_display_timing {}, desired present time {})", what, ms,
              result, fifo ? "yes" : "no", REXCVAR_GET(present_display_timing) ? "on" : "off",
              desired_ns ? fmt::format("{:+.1f} ms from the call's start", double(int64_t(desired_ns) - t0_ns) / 1e6)
                         : std::string("none"));
}
}  // namespace

#if defined(REX_HAS_FIDELITYFX_RUNTIME) && REX_HAS_FIDELITYFX_RUNTIME
namespace {

std::atomic<PFN_vkGetDeviceProcAddr> g_ffx_vk_get_device_proc_addr{nullptr};

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL FfxVkGetDeviceProcAddrCompat(VkDevice device,
                                                                      const char* p_name) {
  if (!p_name) {
    return nullptr;
  }

  PFN_vkGetDeviceProcAddr get_device_proc_addr =
      g_ffx_vk_get_device_proc_addr.load(std::memory_order_relaxed);
  if (!get_device_proc_addr) {
    return nullptr;
  }

  PFN_vkVoidFunction proc = get_device_proc_addr(device, p_name);
  if (proc) {
    return proc;
  }

  // Some drivers expose Vulkan 1.1+ core entry points, but not the legacy KHR
  // aliases expected by parts of FidelityFX VK backend initialization.
  if (!std::strcmp(p_name, "vkGetBufferMemoryRequirements2KHR")) {
    return get_device_proc_addr(device, "vkGetBufferMemoryRequirements2");
  }
  if (!std::strcmp(p_name, "vkGetImageMemoryRequirements2KHR")) {
    return get_device_proc_addr(device, "vkGetImageMemoryRequirements2");
  }
  if (!std::strcmp(p_name, "vkBindBufferMemory2KHR")) {
    return get_device_proc_addr(device, "vkBindBufferMemory2");
  }
  if (!std::strcmp(p_name, "vkBindImageMemory2KHR")) {
    return get_device_proc_addr(device, "vkBindImageMemory2");
  }

  return nullptr;
}

}  // namespace
#endif  // defined(REX_HAS_FIDELITYFX_RUNTIME) && REX_HAS_FIDELITYFX_RUNTIME

// Generated with `xb buildshaders`.
namespace shaders {
#include "../shaders/vulkan_spirv/guest_output_bilinear_dither_ps.h"
#include "../shaders/vulkan_spirv/guest_output_bilinear_ps.h"
// wwe2k14 patch (Session 73, SGSR): Snapdragon GSR 1 (Qualcomm, BSD-3-Clause; source, license and port notes in
// ../shaders/guest_output_sgsr.ps.glsl; header from tools\perf\gen_spirv_header.py --stage frag).
#include "../shaders/vulkan_spirv/guest_output_sgsr_ps.h"
#if defined(REX_HAS_FIDELITYFX_SDK)
#include "../shaders/vulkan_spirv/guest_output_ffx_cas_resample_dither_ps.h"
#include "../shaders/vulkan_spirv/guest_output_ffx_cas_resample_ps.h"
#include "../shaders/vulkan_spirv/guest_output_ffx_cas_sharpen_dither_ps.h"
#include "../shaders/vulkan_spirv/guest_output_ffx_cas_sharpen_ps.h"
#include "../shaders/vulkan_spirv/guest_output_ffx_fsr_easu_ps.h"
#include "../shaders/vulkan_spirv/guest_output_ffx_fsr_rcas_dither_ps.h"
#include "../shaders/vulkan_spirv/guest_output_ffx_fsr_rcas_ps.h"
#endif
#include "../shaders/vulkan_spirv/guest_output_triangle_strip_rect_vs.h"
// wwe2k14 patch (Session 94, present_pre_rotate): the final-pass vertex shader that rotates by the display transform
// (source ../shaders/guest_output_triangle_strip_rect_rot_vs.vert.glsl; header from tools\perf\gen_spirv_header.py).
#include "../shaders/vulkan_spirv/guest_output_triangle_strip_rect_rot_vs.h"
}  // namespace shaders

// wwe2k14 patch (Session 94, present_pre_rotate): the rotation (turns of 90 degrees, 0 = none) the UI drawn into the
// swapchain image must apply, the same mapping as the final guest output pass's.
uint32_t GetPrerotMapForUi() {
  return PrerotMapOfTurns(g_prerot_turns.load());
}

VulkanPresenter::PaintContext::Submission::~Submission() {
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  if (draw_command_pool_ != VK_NULL_HANDLE) {
    dfn.vkDestroyCommandPool(device, draw_command_pool_, nullptr);
  }

  if (present_semaphore_ != VK_NULL_HANDLE) {
    dfn.vkDestroySemaphore(device, present_semaphore_, nullptr);
  }
  if (acquire_semaphore_ != VK_NULL_HANDLE) {
    dfn.vkDestroySemaphore(device, acquire_semaphore_, nullptr);
  }
}

bool VulkanPresenter::PaintContext::Submission::Initialize() {
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  VkSemaphoreCreateInfo semaphore_create_info;
  semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphore_create_info.pNext = nullptr;
  semaphore_create_info.flags = 0;
  if (dfn.vkCreateSemaphore(device, &semaphore_create_info, nullptr, &acquire_semaphore_) !=
      VK_SUCCESS) {
    REXLOG_ERROR(
        "VulkanPresenter: Failed to create a swapchain image acquisition "
        "semaphore");
    return false;
  }
  if (dfn.vkCreateSemaphore(device, &semaphore_create_info, nullptr, &present_semaphore_) !=
      VK_SUCCESS) {
    REXLOG_ERROR(
        "VulkanPresenter: Failed to create a swapchain image presentation "
        "semaphore");
    return false;
  }

  VkCommandPoolCreateInfo command_pool_create_info;
  command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  command_pool_create_info.pNext = nullptr;
  command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  command_pool_create_info.queueFamilyIndex = vulkan_device_->queue_family_graphics_compute();
  if (dfn.vkCreateCommandPool(device, &command_pool_create_info, nullptr, &draw_command_pool_) !=
      VK_SUCCESS) {
    REXLOG_ERROR(
        "VulkanPresenter: Failed to create a command pool for drawing to a "
        "swapchain");
    return false;
  }
  VkCommandBufferAllocateInfo command_buffer_allocate_info;
  command_buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  command_buffer_allocate_info.pNext = nullptr;
  command_buffer_allocate_info.commandPool = draw_command_pool_;
  command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_buffer_allocate_info.commandBufferCount = 1;
  if (dfn.vkAllocateCommandBuffers(device, &command_buffer_allocate_info, &draw_command_buffer_) !=
      VK_SUCCESS) {
    REXLOG_ERROR(
        "VulkanPresenter: Failed to allocate a command buffer for drawing to a "
        "swapchain");
    return false;
  }

  return true;
}

VulkanPresenter::~VulkanPresenter() {
  // Destroy the swapchain after its images are not used for drawing anymore.
  // This is a confusing part in Vulkan, as vkQueuePresentKHR doesn't signal a
  // fence clearly indicating when it's safe to destroy a swapchain, so we
  // assume that its lifetime is tracked internally in the WSI. This is also
  // done before destroying the semaphore awaited by vkQueuePresentKHR, hoping
  // that it will prevent the destruction during the semaphore wait in
  // vkQueuePresentKHR execution (or between the vkQueueSubmit semaphore signal
  // and the vkQueuePresentKHR semaphore wait).
  // This will await completion of all paint submissions also.
  paint_context_.DestroySwapchainAndVulkanSurface();

  if (g_acquire_fence.fence != VK_NULL_HANDLE) {  // wwe2k14 patch (Session 95, present_acquire_fence)
    vulkan_device_->functions().vkDestroyFence(vulkan_device_->device(), g_acquire_fence.fence, nullptr);
    g_acquire_fence = AcquireFenceState();
  }
  if (g_paint_timestamps.pool != VK_NULL_HANDLE) {  // wwe2k14 patch (Session 95, item T); the paints have completed
    vulkan_device_->functions().vkDestroyQueryPool(vulkan_device_->device(), g_paint_timestamps.pool, nullptr);
    g_paint_timestamps = PaintTimestamps();
  }

  // Await completion of the usage of everything before destroying anything
  // (paint submission completion already awaited).
  // From most likely the latest to most likely the earliest to be signaled, so
  // just one sleep will likely be needed.
  ui_submission_tracker_.Shutdown();
  guest_output_image_refresher_submission_tracker_.Shutdown();
#if defined(REX_HAS_FIDELITYFX_RUNTIME) && REX_HAS_FIDELITYFX_RUNTIME
  DestroyTemporalUpscalerContext();
#endif

  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  if (paint_context_.swapchain_render_pass != VK_NULL_HANDLE) {
    dfn.vkDestroyRenderPass(device, paint_context_.swapchain_render_pass, nullptr);
  }
  for (VkSemaphore semaphore : paint_context_.image_present_semaphores) {  // wwe2k14 patch (Session 73)
    dfn.vkDestroySemaphore(device, semaphore, nullptr);
  }

  for (const PaintContext::UISetupCommandBuffer& ui_setup_command_buffer :
       paint_context_.ui_setup_command_buffers) {
    dfn.vkDestroyCommandPool(device, ui_setup_command_buffer.command_pool, nullptr);
  }

  for (VkFramebuffer& framebuffer : paint_context_.guest_output_intermediate_framebuffers) {
    util::DestroyAndNullHandle(dfn.vkDestroyFramebuffer, device, framebuffer);
  }
  util::DestroyAndNullHandle(dfn.vkDestroyDescriptorPool, device,
                             paint_context_.guest_output_descriptor_pool);
  for (PaintContext::GuestOutputPaintPipeline& guest_output_paint_pipeline :
       paint_context_.guest_output_paint_pipelines) {
    util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                               guest_output_paint_pipeline.swapchain_pipeline);
    util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                               guest_output_paint_pipeline.intermediate_pipeline);
  }

  util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device,
                             guest_output_intermediate_render_pass_);
  for (VkShaderModule& shader_module : guest_output_paint_fs_) {
    util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, shader_module);
  }
  util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, guest_output_paint_vs_);
  util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, g_prerot_vs);
  for (VkShaderModule& shader_module : g_prerot_fs) {
    util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, shader_module);
  }
  for (VkPipelineLayout& pipeline_layout : guest_output_paint_pipeline_layouts_) {
    util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device, pipeline_layout);
  }
  util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device,
                             guest_output_paint_image_descriptor_set_layout_);
}

#if defined(REX_HAS_FIDELITYFX_RUNTIME) && REX_HAS_FIDELITYFX_RUNTIME
void VulkanPresenter::DestroyTemporalUpscalerContext() {
  if (!temporal_upscaler_context_) {
    return;
  }
  ffxContext* context = reinterpret_cast<ffxContext*>(&temporal_upscaler_context_);
  ffxDestroyContext(context, nullptr);
  temporal_upscaler_context_ = nullptr;
  temporal_upscaler_max_render_width_ = 0;
  temporal_upscaler_max_render_height_ = 0;
  temporal_upscaler_max_output_width_ = 0;
  temporal_upscaler_max_output_height_ = 0;
}

bool VulkanPresenter::EnsureTemporalUpscalerContext(uint32_t render_width, uint32_t render_height,
                                                    uint32_t output_width, uint32_t output_height) {
  if (!temporal_upscaler_context_ || temporal_upscaler_max_render_width_ != render_width ||
      temporal_upscaler_max_render_height_ != render_height ||
      temporal_upscaler_max_output_width_ != output_width ||
      temporal_upscaler_max_output_height_ != output_height) {
    DestroyTemporalUpscalerContext();

    ffxCreateContextDescUpscale create_desc = {};
    create_desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create_desc.header.pNext = nullptr;
    create_desc.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
    create_desc.maxRenderSize.width = render_width;
    create_desc.maxRenderSize.height = render_height;
    create_desc.maxUpscaleSize.width = output_width;
    create_desc.maxUpscaleSize.height = output_height;
    create_desc.fpMessage = nullptr;

    ffxCreateBackendVKDesc backend_desc = {};
    backend_desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend_desc.header.pNext = nullptr;
    backend_desc.vkDevice = vulkan_device_->device();
    backend_desc.vkPhysicalDevice = vulkan_device_->physical_device();
    PFN_vkGetDeviceProcAddr vk_device_proc_addr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
        vulkan_device_->vulkan_instance()->functions().vkGetInstanceProcAddr(
            vulkan_device_->vulkan_instance()->instance(), "vkGetDeviceProcAddr"));
    if (!vk_device_proc_addr) {
      REXLOG_WARN("VulkanPresenter: vkGetDeviceProcAddr is unavailable for FidelityFX");
      return false;
    }
    g_ffx_vk_get_device_proc_addr.store(vk_device_proc_addr, std::memory_order_relaxed);
    backend_desc.vkDeviceProcAddr = FfxVkGetDeviceProcAddrCompat;

    create_desc.header.pNext = &backend_desc.header;

    ffxContext* context = reinterpret_cast<ffxContext*>(&temporal_upscaler_context_);
    if (ffxCreateContext(context, &create_desc.header, nullptr) != FFX_API_RETURN_OK) {
      REXLOG_WARN(
          "VulkanPresenter: Failed to create FidelityFX temporal upscaler "
          "context");
      temporal_upscaler_context_ = nullptr;
      return false;
    }

    temporal_upscaler_max_render_width_ = render_width;
    temporal_upscaler_max_render_height_ = render_height;
    temporal_upscaler_max_output_width_ = output_width;
    temporal_upscaler_max_output_height_ = output_height;
    temporal_upscaler_provider_logged_ = false;
  }

  if (!temporal_upscaler_provider_logged_) {
    ffxQueryGetProviderVersion provider_version = {};
    provider_version.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    provider_version.header.pNext = nullptr;
    ffxContext* context = reinterpret_cast<ffxContext*>(&temporal_upscaler_context_);
    if (ffxQuery(context, &provider_version.header) == FFX_API_RETURN_OK &&
        provider_version.versionName) {
      REXLOG_INFO("VulkanPresenter: FidelityFX upscaler provider {}", provider_version.versionName);
    }
    temporal_upscaler_provider_logged_ = true;
  }

  return temporal_upscaler_context_ != nullptr;
}

bool VulkanPresenter::DispatchTemporalUpscaler(VkCommandBuffer command_buffer, VkImage input_image,
                                               uint32_t input_width, uint32_t input_height,
                                               VkImage output_image, uint32_t output_width,
                                               uint32_t output_height,
                                               const GuestOutputPaintConfig& config) {
  if (command_buffer == VK_NULL_HANDLE || input_image == VK_NULL_HANDLE ||
      output_image == VK_NULL_HANDLE || !input_width || !input_height || !output_width ||
      !output_height) {
    return false;
  }
  if (!EnsureTemporalUpscalerContext(input_width, input_height, output_width, output_height)) {
    return false;
  }

  VkImageCreateInfo source_image_create_info = {};
  source_image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  source_image_create_info.imageType = VK_IMAGE_TYPE_2D;
  source_image_create_info.format = kGuestOutputFormat;
  source_image_create_info.extent.width = input_width;
  source_image_create_info.extent.height = input_height;
  source_image_create_info.extent.depth = 1;
  source_image_create_info.mipLevels = 1;
  source_image_create_info.arrayLayers = 1;
  source_image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
  source_image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  source_image_create_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  source_image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  source_image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  FfxApiResourceDescription source_desc = ffxApiGetImageResourceDescriptionVK(
      input_image, source_image_create_info, FFX_API_RESOURCE_USAGE_READ_ONLY);
  FfxApiResource source = ffxApiGetResourceVK(reinterpret_cast<void*>(input_image), source_desc,
                                              FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
  FfxApiResource depth =
      ffxApiGetResourceVK(reinterpret_cast<void*>(input_image),
                          ffxApiGetImageResourceDescriptionVK(input_image, source_image_create_info,
                                                              FFX_API_RESOURCE_USAGE_DEPTHTARGET),
                          FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
  FfxApiResource motion_vectors = source;

  VkImageCreateInfo output_image_create_info = source_image_create_info;
  output_image_create_info.extent.width = output_width;
  output_image_create_info.extent.height = output_height;
  FfxApiResourceDescription output_desc = ffxApiGetImageResourceDescriptionVK(
      output_image, output_image_create_info,
      FFX_API_RESOURCE_USAGE_UAV | FFX_API_RESOURCE_USAGE_READ_ONLY);
  FfxApiResource output = ffxApiGetResourceVK(reinterpret_cast<void*>(output_image), output_desc,
                                              FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);

  ffxDispatchDescUpscale dispatch_desc = {};
  dispatch_desc.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
  dispatch_desc.header.pNext = nullptr;
  dispatch_desc.commandList = command_buffer;
  dispatch_desc.color = source;
  dispatch_desc.depth = depth;
  dispatch_desc.motionVectors = motion_vectors;
  dispatch_desc.exposure = {};
  dispatch_desc.reactive = {};
  dispatch_desc.transparencyAndComposition = {};
  dispatch_desc.output = output;
  dispatch_desc.jitterOffset.x = 0.0f;
  dispatch_desc.jitterOffset.y = 0.0f;
  dispatch_desc.motionVectorScale.x = float(input_width);
  dispatch_desc.motionVectorScale.y = float(input_height);
  dispatch_desc.renderSize.width = input_width;
  dispatch_desc.renderSize.height = input_height;
  dispatch_desc.upscaleSize.width = output_width;
  dispatch_desc.upscaleSize.height = output_height;
  dispatch_desc.enableSharpening = true;
  dispatch_desc.sharpness = std::clamp(1.0f - config.GetFsrSharpnessReduction() * 0.5f, 0.0f, 1.0f);
  dispatch_desc.reset = true;
  dispatch_desc.frameTimeDelta = 16.666f;
  dispatch_desc.preExposure = 1.0f;
  dispatch_desc.cameraNear = 0.1f;
  dispatch_desc.cameraFar = 1000.0f;
  dispatch_desc.cameraFovAngleVertical = 1.0472f;
  dispatch_desc.viewSpaceToMetersFactor = 1.0f;
  dispatch_desc.flags = 0;

  ffxContext* context = reinterpret_cast<ffxContext*>(&temporal_upscaler_context_);
  if (ffxDispatch(context, &dispatch_desc.header) != FFX_API_RETURN_OK) {
    return false;
  }
  return true;
}
#endif

Surface::TypeFlags VulkanPresenter::GetSurfaceTypesSupportedByInstance(
    const VulkanInstance::Extensions& instance_extensions) {
  if (!instance_extensions.ext_KHR_surface) {
    return 0;
  }
  Surface::TypeFlags type_flags = 0;
#if REX_PLATFORM_ANDROID
  if (instance_extensions.ext_KHR_android_surface) {
    type_flags |= Surface::kTypeFlag_AndroidNativeWindow;
  }
#endif
#if REX_PLATFORM_GNU_LINUX
  if (instance_extensions.ext_KHR_wayland_surface) {
    type_flags |= Surface::kTypeFlag_WaylandSurface;
  }
  if (instance_extensions.ext_KHR_xcb_surface) {
    type_flags |= Surface::kTypeFlag_XcbWindow;
  }
#endif
#if REX_PLATFORM_WIN32
  if (instance_extensions.ext_KHR_win32_surface) {
    type_flags |= Surface::kTypeFlag_Win32Hwnd;
  }
#endif
#if REX_PLATFORM_MAC
  if (instance_extensions.ext_EXT_metal_surface) {
    type_flags |= Surface::kTypeFlag_CAMetalLayer;
  }
#endif
  return type_flags;
}

Surface::TypeFlags VulkanPresenter::GetSupportedSurfaceTypes() const {
  if (!vulkan_device_->extensions().ext_KHR_swapchain) {
    return 0;
  }
  return GetSurfaceTypesSupportedByInstance(vulkan_device_->vulkan_instance()->extensions());
}

bool VulkanPresenter::CaptureGuestOutput(RawImage& image_out) {
  std::shared_ptr<GuestOutputImage> guest_output_image;
  GuestOutputProperties capture_properties;  // wwe2k14 patch (Session 19, LATENCY)
  // wwe2k14 patch (Session 6, DRAIN): see GuestOutputImageInstance::deferred_submission_order.
  std::shared_ptr<VulkanQueueSubmitOrder> guest_output_deferred_submission_order;
  uint64_t guest_output_deferred_submission_ticket = 0;
  {
    uint32_t guest_output_mailbox_index;
    std::unique_lock<std::mutex> guest_output_consumer_lock(
        ConsumeGuestOutput(guest_output_mailbox_index, &capture_properties, nullptr));
    // wwe2k14 patch (Session 19, LATENCY): a = refresh sequence number of the image captured.
    perf::timeline::Record("gocapture", uint32_t(capture_properties.refresh_seq));
    if (guest_output_mailbox_index != UINT32_MAX) {
      assert_true(guest_output_images_[guest_output_mailbox_index].ever_successfully_refreshed);
      guest_output_image = guest_output_images_[guest_output_mailbox_index].image;
      guest_output_deferred_submission_order =
          guest_output_images_[guest_output_mailbox_index].deferred_submission_order;
      guest_output_deferred_submission_ticket =
          guest_output_images_[guest_output_mailbox_index].deferred_submission_ticket;
    }
    // Incremented the reference count of the guest output image - safe to leave
    // the consumer critical section now.
  }
  if (!guest_output_image) {
    return false;
  }
  // wwe2k14 patch (Session 6, DRAIN): the copy below relies on queue submission order to come after the
  // refresher's writes.
  if (guest_output_deferred_submission_order) {
    guest_output_deferred_submission_order->AwaitSubmitted(guest_output_deferred_submission_ticket);
  }

  VkExtent2D image_extent = guest_output_image->extent();
  size_t pixel_count = size_t(image_extent.width) * image_extent.height;
  VkDeviceSize buffer_size = VkDeviceSize(sizeof(uint32_t) * pixel_count);
  VkBuffer buffer;
  VkDeviceMemory buffer_memory;
  if (!util::CreateDedicatedAllocationBuffer(
          vulkan_device_, buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          util::MemoryPurpose::kReadback, buffer, buffer_memory)) {
    REXLOG_ERROR("VulkanPresenter: Failed to create the guest output capture buffer");
    return false;
  }

  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  {
    VkCommandPoolCreateInfo command_pool_create_info;
    command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_create_info.pNext = nullptr;
    command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    command_pool_create_info.queueFamilyIndex = vulkan_device_->queue_family_graphics_compute();
    VkCommandPool command_pool;
    if (dfn.vkCreateCommandPool(device, &command_pool_create_info, nullptr, &command_pool) !=
        VK_SUCCESS) {
      REXLOG_ERROR(
          "VulkanPresenter: Failed to create the guest output capturing "
          "command pool");
      dfn.vkDestroyBuffer(device, buffer, nullptr);
      dfn.vkFreeMemory(device, buffer_memory, nullptr);
      return false;
    }

    VkCommandBufferAllocateInfo command_buffer_allocate_info;
    command_buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_allocate_info.pNext = nullptr;
    command_buffer_allocate_info.commandPool = command_pool;
    command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_allocate_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer;
    if (dfn.vkAllocateCommandBuffers(device, &command_buffer_allocate_info, &command_buffer) !=
        VK_SUCCESS) {
      REXLOG_ERROR(
          "VulkanPresenter: Failed to allocate the guest output capturing "
          "command buffer");
      dfn.vkDestroyCommandPool(device, command_pool, nullptr);
      dfn.vkDestroyBuffer(device, buffer, nullptr);
      dfn.vkFreeMemory(device, buffer_memory, nullptr);
      return false;
    }

    VkCommandBufferBeginInfo command_buffer_begin_info;
    command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    command_buffer_begin_info.pNext = nullptr;
    command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    command_buffer_begin_info.pInheritanceInfo = nullptr;
    if (dfn.vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info) != VK_SUCCESS) {
      REXLOG_ERROR(
          "VulkanPresenter: Failed to begin recording the guest output "
          "capturing command buffer");
      dfn.vkDestroyCommandPool(device, command_pool, nullptr);
      dfn.vkDestroyBuffer(device, buffer, nullptr);
      dfn.vkFreeMemory(device, buffer_memory, nullptr);
      return false;
    }

    VkImageMemoryBarrier image_memory_barrier;
    image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_memory_barrier.pNext = nullptr;
    image_memory_barrier.srcAccessMask = kGuestOutputInternalAccessMask;
    image_memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    image_memory_barrier.oldLayout = kGuestOutputInternalLayout;
    image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    image_memory_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_memory_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_memory_barrier.image = guest_output_image->image();
    image_memory_barrier.subresourceRange = util::InitializeSubresourceRange();
    dfn.vkCmdPipelineBarrier(command_buffer, kGuestOutputInternalStageMask,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &image_memory_barrier);

    VkBufferImageCopy buffer_image_copy = {};
    buffer_image_copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    buffer_image_copy.imageSubresource.layerCount = 1;
    buffer_image_copy.imageExtent.width = image_extent.width;
    buffer_image_copy.imageExtent.height = image_extent.height;
    buffer_image_copy.imageExtent.depth = 1;
    dfn.vkCmdCopyImageToBuffer(command_buffer, guest_output_image->image(),
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &buffer_image_copy);

    // A fence doesn't guarantee host visibility and availability.
    VkBufferMemoryBarrier buffer_memory_barrier;
    buffer_memory_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    buffer_memory_barrier.pNext = nullptr;
    buffer_memory_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    buffer_memory_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    buffer_memory_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    buffer_memory_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    buffer_memory_barrier.buffer = buffer;
    buffer_memory_barrier.offset = 0;
    buffer_memory_barrier.size = VK_WHOLE_SIZE;
    std::swap(image_memory_barrier.srcAccessMask, image_memory_barrier.dstAccessMask);
    std::swap(image_memory_barrier.oldLayout, image_memory_barrier.newLayout);
    dfn.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT | kGuestOutputInternalStageMask, 0, 0,
                             nullptr, 1, &buffer_memory_barrier, 1, &image_memory_barrier);

    if (dfn.vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
      REXLOG_ERROR(
          "VulkanPresenter: Failed to end recording the guest output capturing "
          "command buffer");
      dfn.vkDestroyCommandPool(device, command_pool, nullptr);
      dfn.vkDestroyBuffer(device, buffer, nullptr);
      dfn.vkFreeMemory(device, buffer_memory, nullptr);
      return false;
    }

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    VulkanSubmissionTracker submission_tracker(vulkan_device_);
    {
      VulkanSubmissionTracker::FenceAcquisition fence_acqusition(
          submission_tracker.AcquireFenceToAdvanceSubmission());
      if (!fence_acqusition.fence()) {
        REXLOG_ERROR(
            "VulkanPresenter: Failed to acquire a fence for guest output "
            "capturing");
        fence_acqusition.SubmissionFailedOrDropped();
        dfn.vkDestroyCommandPool(device, command_pool, nullptr);
        dfn.vkDestroyBuffer(device, buffer, nullptr);
        dfn.vkFreeMemory(device, buffer_memory, nullptr);
        return false;
      }
      VkResult submit_result;
      {
        const VulkanDevice::Queue::Acquisition queue_acquisition =
            vulkan_device_->AcquireQueue(vulkan_device_->queue_family_graphics_compute(), 0);
        submit_result =
            dfn.vkQueueSubmit(queue_acquisition.queue(), 1, &submit_info, fence_acqusition.fence());
      }
      if (submit_result != VK_SUCCESS) {
        REXLOG_ERROR(
            "VulkanPresenter: Failed to submit the guest output capturing "
            "command buffer");
        fence_acqusition.SubmissionFailedOrDropped();
        dfn.vkDestroyCommandPool(device, command_pool, nullptr);
        dfn.vkDestroyBuffer(device, buffer, nullptr);
        dfn.vkFreeMemory(device, buffer_memory, nullptr);
        return false;
      }
    }
    if (!submission_tracker.AwaitAllSubmissionsCompletion()) {
      REXLOG_ERROR("VulkanPresenter: Failed to await the guest output capturing fence");
      dfn.vkDestroyCommandPool(device, command_pool, nullptr);
      dfn.vkDestroyBuffer(device, buffer, nullptr);
      dfn.vkFreeMemory(device, buffer_memory, nullptr);
      return false;
    }

    dfn.vkDestroyCommandPool(device, command_pool, nullptr);
  }

  // Don't need the buffer anymore, just its memory.
  dfn.vkDestroyBuffer(device, buffer, nullptr);

  void* mapping;
  if (dfn.vkMapMemory(device, buffer_memory, 0, VK_WHOLE_SIZE, 0, &mapping) != VK_SUCCESS) {
    REXLOG_ERROR("VulkanPresenter: Failed to map the guest output capture memory");
    dfn.vkFreeMemory(device, buffer_memory, nullptr);
    return false;
  }

  {  // wwe2k14 patch (Session 95, HANG): the readback type can be cached but not coherent; invalidate before reading.
    VkMappedMemoryRange range = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, buffer_memory, 0, VK_WHOLE_SIZE};
    dfn.vkInvalidateMappedMemoryRanges(device, 1, &range);
  }

  image_out.width = image_extent.width;
  image_out.height = image_extent.height;
  image_out.stride = sizeof(uint32_t) * image_extent.width;
  image_out.data.resize(size_t(buffer_size));
  uint32_t* image_out_pixels = reinterpret_cast<uint32_t*>(image_out.data.data());
  for (size_t i = 0; i < pixel_count; ++i) {
    image_out_pixels[i] = Packed10bpcRGBTo8bpcBytes(reinterpret_cast<const uint32_t*>(mapping)[i]);
  }

  // Unmapping will be done by freeing.
  dfn.vkFreeMemory(device, buffer_memory, nullptr);

  return true;
}

VkCommandBuffer VulkanPresenter::AcquireUISetupCommandBufferFromUIThread() {
  if (paint_context_.ui_setup_command_buffer_current_index != SIZE_MAX) {
    return paint_context_
        .ui_setup_command_buffers[paint_context_.ui_setup_command_buffer_current_index]
        .command_buffer;
  }

  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  VkCommandBufferBeginInfo command_buffer_begin_info;
  command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  command_buffer_begin_info.pNext = nullptr;
  command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  command_buffer_begin_info.pInheritanceInfo = nullptr;

  // Try to reuse an existing command buffer.
  if (!paint_context_.ui_setup_command_buffers.empty()) {
    uint64_t submission_index_completed = ui_submission_tracker_.UpdateAndGetCompletedSubmission();
    for (size_t i = 0; i < paint_context_.ui_setup_command_buffers.size(); ++i) {
      PaintContext::UISetupCommandBuffer& ui_setup_command_buffer =
          paint_context_.ui_setup_command_buffers[i];
      if (ui_setup_command_buffer.last_usage_submission_index > submission_index_completed) {
        continue;
      }
      if (dfn.vkResetCommandPool(device, ui_setup_command_buffer.command_pool, 0) != VK_SUCCESS) {
        REXLOG_ERROR("VulkanPresenter: Failed to reset a UI setup command pool");
        return VK_NULL_HANDLE;
      }
      if (dfn.vkBeginCommandBuffer(ui_setup_command_buffer.command_buffer,
                                   &command_buffer_begin_info) != VK_SUCCESS) {
        REXLOG_ERROR(
            "VulkanPresenter: Failed to begin UI setup command buffer "
            "recording");
        return VK_NULL_HANDLE;
      }
      paint_context_.ui_setup_command_buffer_current_index = i;
      ui_setup_command_buffer.last_usage_submission_index =
          ui_submission_tracker_.GetCurrentSubmission();
      return ui_setup_command_buffer.command_buffer;
    }
  }

  // Create a new command buffer.
  VkCommandPoolCreateInfo command_pool_create_info;
  command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  command_pool_create_info.pNext = nullptr;
  command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  command_pool_create_info.queueFamilyIndex = vulkan_device_->queue_family_graphics_compute();
  VkCommandPool new_command_pool;
  if (dfn.vkCreateCommandPool(device, &command_pool_create_info, nullptr, &new_command_pool) !=
      VK_SUCCESS) {
    REXLOG_ERROR("VulkanPresenter: Failed to create a UI setup command pool");
    return VK_NULL_HANDLE;
  }
  VkCommandBufferAllocateInfo command_buffer_allocate_info;
  command_buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  command_buffer_allocate_info.pNext = nullptr;
  command_buffer_allocate_info.commandPool = new_command_pool;
  command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_buffer_allocate_info.commandBufferCount = 1;
  VkCommandBuffer new_command_buffer;
  if (dfn.vkAllocateCommandBuffers(device, &command_buffer_allocate_info, &new_command_buffer) !=
      VK_SUCCESS) {
    REXLOG_ERROR("VulkanPresenter: Failed to allocate a UI setup command buffer");
    dfn.vkDestroyCommandPool(device, new_command_pool, nullptr);
    return VK_NULL_HANDLE;
  }
  if (dfn.vkBeginCommandBuffer(new_command_buffer, &command_buffer_begin_info) != VK_SUCCESS) {
    REXLOG_ERROR("VulkanPresenter: Failed to begin UI setup command buffer recording");
    dfn.vkDestroyCommandPool(device, new_command_pool, nullptr);
    return VK_NULL_HANDLE;
  }
  paint_context_.ui_setup_command_buffer_current_index =
      paint_context_.ui_setup_command_buffers.size();
  paint_context_.ui_setup_command_buffers.emplace_back(
      new_command_pool, new_command_buffer, ui_submission_tracker_.GetCurrentSubmission());
  return new_command_buffer;
}

Presenter::SurfacePaintConnectResult
VulkanPresenter::ConnectOrReconnectPaintingToSurfaceFromUIThread(Surface& new_surface,
                                                                 uint32_t new_surface_width,
                                                                 uint32_t new_surface_height,
                                                                 bool was_paintable,
                                                                 bool& is_vsync_implicit_out) {
  const VulkanInstance* const vulkan_instance = vulkan_device_->vulkan_instance();
  const VulkanInstance::Functions& ifn = vulkan_instance->functions();
  const VkInstance instance = vulkan_instance->instance();
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  VkFormat new_swapchain_format;

  // ConnectOrReconnectToSurfaceFromUIThread may be called only for the
  // ui::Surface of the current swapchain or when the old swapchain and
  // VkSurface have, if the ui::Surface is the same, try using the existing
  // VkSurface and creating the swapchain smoothly from the existing one - if
  // this doesn't succeed, start from scratch.
  // The retirement or destruction of the swapchain here will also cause
  // awaiting completion of the usage of the swapchain and the surface on the
  // GPU.
  if (paint_context_.vulkan_surface != VK_NULL_HANDLE) {
    VkSwapchainKHR old_swapchain = paint_context_.PrepareForSwapchainRetirement();
    bool surface_unusable;
    paint_context_.swapchain = PaintContext::CreateSwapchainForVulkanSurface(
        vulkan_device_, paint_context_.vulkan_surface, new_surface_width, new_surface_height,
        old_swapchain, paint_context_.present_queue_family, new_swapchain_format,
        paint_context_.swapchain_extent, paint_context_.swapchain_is_fifo, surface_unusable);
    // Destroy the old swapchain that may be retired now.
    if (old_swapchain != VK_NULL_HANDLE) {
      dfn.vkDestroySwapchainKHR(device, old_swapchain, nullptr);
    }
    if (paint_context_.swapchain == VK_NULL_HANDLE) {
      // Couldn't create the swapchain for the existing surface - start over.
      paint_context_.DestroySwapchainAndVulkanSurface();
    }
  }

  // If failed to create the swapchain for the previous surface, recreate the
  // surface and create the new swapchain.
  if (paint_context_.swapchain == VK_NULL_HANDLE) {
    // DestroySwapchainAndVulkanSurface should have been called previously.
    assert_true(paint_context_.vulkan_surface == VK_NULL_HANDLE);
    Surface::TypeIndex surface_type = new_surface.GetType();
    // Check if the surface type is supported according to the Vulkan
    // extensions.
    if (!(GetSupportedSurfaceTypes() & (Surface::TypeFlags(1) << surface_type))) {
      REXLOG_ERROR(
          "VulkanPresenter: Tried to create a Vulkan surface for an "
          "unsupported Xenia surface type");
      return SurfacePaintConnectResult::kFailureSurfaceUnusable;
    }
    VkResult vulkan_surface_create_result = VK_ERROR_UNKNOWN;
    switch (surface_type) {
#if REX_PLATFORM_ANDROID
      case Surface::kTypeIndex_AndroidNativeWindow: {
        auto& android_native_window_surface =
            static_cast<const AndroidNativeWindowSurface&>(new_surface);
        VkAndroidSurfaceCreateInfoKHR surface_create_info;
        surface_create_info.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
        surface_create_info.pNext = nullptr;
        surface_create_info.flags = 0;
        surface_create_info.window = android_native_window_surface.window();
        vulkan_surface_create_result = ifn.vkCreateAndroidSurfaceKHR(
            instance, &surface_create_info, nullptr, &paint_context_.vulkan_surface);
      } break;
#endif
#if REX_PLATFORM_GNU_LINUX
      case Surface::kTypeIndex_XcbWindow: {
        auto& xcb_window_surface = static_cast<const XcbWindowSurface&>(new_surface);
        VkXcbSurfaceCreateInfoKHR surface_create_info;
        surface_create_info.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
        surface_create_info.pNext = nullptr;
        surface_create_info.flags = 0;
        surface_create_info.connection = xcb_window_surface.connection();
        surface_create_info.window = xcb_window_surface.window();
        vulkan_surface_create_result = ifn.vkCreateXcbSurfaceKHR(
            instance, &surface_create_info, nullptr, &paint_context_.vulkan_surface);
      } break;
      case Surface::kTypeIndex_WaylandSurface: {
        auto& wayland_surface = static_cast<const WaylandSurface&>(new_surface);
        VkWaylandSurfaceCreateInfoKHR surface_create_info;
        surface_create_info.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
        surface_create_info.pNext = nullptr;
        surface_create_info.flags = 0;
        surface_create_info.display = wayland_surface.display();
        surface_create_info.surface = wayland_surface.surface();
        vulkan_surface_create_result = ifn.vkCreateWaylandSurfaceKHR(
            instance, &surface_create_info, nullptr, &paint_context_.vulkan_surface);
      } break;
#endif
#if REX_PLATFORM_WIN32
      case Surface::kTypeIndex_Win32Hwnd: {
        auto& win32_hwnd_surface = static_cast<const Win32HwndSurface&>(new_surface);
        VkWin32SurfaceCreateInfoKHR surface_create_info;
        surface_create_info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        surface_create_info.pNext = nullptr;
        surface_create_info.flags = 0;
        surface_create_info.hinstance = win32_hwnd_surface.hinstance();
        surface_create_info.hwnd = win32_hwnd_surface.hwnd();
        vulkan_surface_create_result = ifn.vkCreateWin32SurfaceKHR(
            instance, &surface_create_info, nullptr, &paint_context_.vulkan_surface);
      } break;
#endif
#if REX_PLATFORM_MAC
      case Surface::kTypeIndex_CAMetalLayer: {
        auto& metal_surface = static_cast<const CAMetalLayerSurface&>(new_surface);
        VkMetalSurfaceCreateInfoEXT surface_create_info;
        surface_create_info.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
        surface_create_info.pNext = nullptr;
        surface_create_info.flags = 0;
        surface_create_info.pLayer = metal_surface.layer();
        vulkan_surface_create_result = ifn.vkCreateMetalSurfaceEXT(
            instance, &surface_create_info, nullptr, &paint_context_.vulkan_surface);
      } break;
#endif
      default:
        assert_unhandled_case(surface_type);
        REXLOG_ERROR(
            "VulkanPresenter: Tried to create a Vulkan surface for an "
            "unknown Xenia surface type");
        return SurfacePaintConnectResult::kFailureSurfaceUnusable;
    }
    if (vulkan_surface_create_result != VK_SUCCESS) {
      REXLOG_ERROR("VulkanPresenter: Failed to create a Vulkan surface");
      return SurfacePaintConnectResult::kFailure;
    }
    bool surface_unusable;
    paint_context_.swapchain = PaintContext::CreateSwapchainForVulkanSurface(
        vulkan_device_, paint_context_.vulkan_surface, new_surface_width, new_surface_height,
        VK_NULL_HANDLE, paint_context_.present_queue_family, new_swapchain_format,
        paint_context_.swapchain_extent, paint_context_.swapchain_is_fifo, surface_unusable);
    if (paint_context_.swapchain == VK_NULL_HANDLE) {
      // Failed to create the swapchain for the new Vulkan surface - destroy the
      // Vulkan surface.
      ifn.vkDestroySurfaceKHR(instance, paint_context_.vulkan_surface, nullptr);
      paint_context_.vulkan_surface = VK_NULL_HANDLE;
      return surface_unusable ? SurfacePaintConnectResult::kFailureSurfaceUnusable
                              : SurfacePaintConnectResult::kFailure;
    }
    // Successfully attached (at least for now).
  }

  // From now on, in case of failure,
  // paint_context_.DestroySwapchainAndVulkanSurface must be called before
  // returning.

  // Update the render pass to the new format.
  if (paint_context_.swapchain_render_pass_format != new_swapchain_format) {
    util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device,
                               paint_context_.swapchain_render_pass);
    paint_context_.swapchain_render_pass_format = new_swapchain_format;
  }
  if (paint_context_.swapchain_render_pass == VK_NULL_HANDLE) {
    VkAttachmentDescription render_pass_attachment;
    render_pass_attachment.flags = 0;
    render_pass_attachment.format = new_swapchain_format;
    render_pass_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    render_pass_attachment.loadOp = REXCVAR_GET(present_render_pass_clear)
                                        ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                        : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    render_pass_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    render_pass_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    render_pass_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    render_pass_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    render_pass_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference render_pass_color_attachment;
    render_pass_color_attachment.attachment = 0;
    render_pass_color_attachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkSubpassDescription render_pass_subpass = {};
    render_pass_subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    render_pass_subpass.colorAttachmentCount = 1;
    render_pass_subpass.pColorAttachments = &render_pass_color_attachment;
    VkSubpassDependency render_pass_dependencies[2];
    render_pass_dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    render_pass_dependencies[0].dstSubpass = 0;
    // srcStageMask is the semaphore wait stage.
    render_pass_dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    render_pass_dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    render_pass_dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    // The main target can be used for UI drawing at any moment, which requires
    // blending, so VK_ACCESS_COLOR_ATTACHMENT_READ_BIT is also included.
    render_pass_dependencies[0].dstAccessMask =
        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    render_pass_dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    render_pass_dependencies[1].srcSubpass = 0;
    render_pass_dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    render_pass_dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    // Semaphores are signaled at VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT.
    render_pass_dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    render_pass_dependencies[1].srcAccessMask =
        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    render_pass_dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    render_pass_dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    VkRenderPassCreateInfo render_pass_create_info;
    render_pass_create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_create_info.pNext = nullptr;
    render_pass_create_info.flags = 0;
    render_pass_create_info.attachmentCount = 1;
    render_pass_create_info.pAttachments = &render_pass_attachment;
    render_pass_create_info.subpassCount = 1;
    render_pass_create_info.pSubpasses = &render_pass_subpass;
    render_pass_create_info.dependencyCount = uint32_t(rex::countof(render_pass_dependencies));
    render_pass_create_info.pDependencies = render_pass_dependencies;
    VkRenderPass new_render_pass;
    if (dfn.vkCreateRenderPass(device, &render_pass_create_info, nullptr, &new_render_pass) !=
        VK_SUCCESS) {
      REXLOG_ERROR(
          "VulkanPresenter: Failed to create the render pass for drawing to "
          "swapchain images");
      paint_context_.DestroySwapchainAndVulkanSurface();
      return SurfacePaintConnectResult::kFailure;
    }
    paint_context_.swapchain_render_pass = new_render_pass;
    paint_context_.swapchain_render_pass_format = new_swapchain_format;
    paint_context_.swapchain_render_pass_clear_load_op =
        render_pass_attachment.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
  }

  // Get the swapchain images.
  paint_context_.swapchain_images.clear();
  VkResult swapchain_images_get_result;
  for (;;) {
    uint32_t swapchain_image_count = uint32_t(paint_context_.swapchain_images.size());
    bool swapchain_images_were_empty = !swapchain_image_count;
    swapchain_images_get_result = dfn.vkGetSwapchainImagesKHR(
        device, paint_context_.swapchain, &swapchain_image_count,
        swapchain_images_were_empty ? nullptr : paint_context_.swapchain_images.data());
    // If the original swapchain image count was 0 (first call), SUCCESS is
    // returned, not INCOMPLETE.
    if (swapchain_images_get_result == VK_SUCCESS || swapchain_images_get_result == VK_INCOMPLETE) {
      paint_context_.swapchain_images.resize(swapchain_image_count);
      if (swapchain_images_get_result == VK_SUCCESS &&
          (!swapchain_images_were_empty || !swapchain_image_count)) {
        break;
      }
    } else {
      break;
    }
  }
  if (swapchain_images_get_result != VK_SUCCESS) {
    REXLOG_ERROR("VulkanPresenter: Failed to get swapchain images");
    paint_context_.DestroySwapchainAndVulkanSurface();
    return SurfacePaintConnectResult::kFailure;
  }

  // Create the image views and the framebuffers.
  assert_true(paint_context_.swapchain_framebuffers.empty());
  paint_context_.swapchain_framebuffers.reserve(paint_context_.swapchain_images.size());
  VkImageViewCreateInfo image_view_create_info;
  image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  image_view_create_info.pNext = nullptr;
  image_view_create_info.flags = 0;
  image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  image_view_create_info.format = new_swapchain_format;
  image_view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  image_view_create_info.subresourceRange.baseMipLevel = 0;
  image_view_create_info.subresourceRange.levelCount = 1;
  image_view_create_info.subresourceRange.baseArrayLayer = 0;
  image_view_create_info.subresourceRange.layerCount = 1;
  VkImageView image_view;
  VkFramebufferCreateInfo framebuffer_create_info;
  framebuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebuffer_create_info.pNext = nullptr;
  framebuffer_create_info.flags = 0;
  framebuffer_create_info.renderPass = paint_context_.swapchain_render_pass;
  framebuffer_create_info.attachmentCount = 1;
  framebuffer_create_info.pAttachments = &image_view;
  framebuffer_create_info.width = paint_context_.swapchain_extent.width;
  framebuffer_create_info.height = paint_context_.swapchain_extent.height;
  framebuffer_create_info.layers = 1;
  for (VkImage image : paint_context_.swapchain_images) {
    image_view_create_info.image = image;
    if (dfn.vkCreateImageView(device, &image_view_create_info, nullptr, &image_view) !=
        VK_SUCCESS) {
      REXLOG_ERROR("VulkanPresenter: Failed to create a swapchain image view");
      paint_context_.DestroySwapchainAndVulkanSurface();
      return SurfacePaintConnectResult::kFailure;
    }
    VkFramebuffer framebuffer;
    if (dfn.vkCreateFramebuffer(device, &framebuffer_create_info, nullptr, &framebuffer) !=
        VK_SUCCESS) {
      REXLOG_ERROR("VulkanPresenter: Failed to create a swapchain framebuffer");
      dfn.vkDestroyImageView(device, image_view, nullptr);
      paint_context_.DestroySwapchainAndVulkanSurface();
      return SurfacePaintConnectResult::kFailure;
    }
    paint_context_.swapchain_framebuffers.emplace_back(image_view, framebuffer);
  }

  is_vsync_implicit_out = paint_context_.swapchain_is_fifo;
  return SurfacePaintConnectResult::kSuccess;
}

void VulkanPresenter::DisconnectPaintingFromSurfaceFromUIThreadImpl() {
  paint_context_.DestroySwapchainAndVulkanSurface();
}

bool VulkanPresenter::RefreshGuestOutputImpl(
    uint32_t mailbox_index, uint32_t frontbuffer_width, uint32_t frontbuffer_height,
    std::function<bool(GuestOutputRefreshContext& context)> refresher, bool& is_8bpc_out_ref) {
  assert_not_zero(frontbuffer_width);
  assert_not_zero(frontbuffer_height);
  VkExtent2D max_framebuffer_extent = util::GetMax2DFramebufferExtent(vulkan_device_->properties());
  if (frontbuffer_width > max_framebuffer_extent.width ||
      frontbuffer_height > max_framebuffer_extent.height) {
    // Writing the guest output isn't supposed to rescale, and a guest texture
    // exceeding the maximum size won't be loadable anyway.
    return false;
  }

  GuestOutputImageInstance& image_instance = guest_output_images_[mailbox_index];
  if (image_instance.image && (image_instance.image->extent().width != frontbuffer_width ||
                               image_instance.image->extent().height != frontbuffer_height)) {
    guest_output_image_refresher_submission_tracker_.AwaitSubmissionCompletion(
        image_instance.last_refresher_submission);
    image_instance.image.reset();
  }
  if (!image_instance.image) {
    std::unique_ptr<GuestOutputImage> new_image =
        GuestOutputImage::Create(vulkan_device_, frontbuffer_width, frontbuffer_height);
    if (!new_image) {
      return false;
    }
    image_instance.SetToNewImage(std::move(new_image), guest_output_image_next_version_++);
  }

  VulkanGuestOutputRefreshContext context(is_8bpc_out_ref, image_instance.image->image(),
                                          image_instance.image->view(), image_instance.version,
                                          image_instance.ever_successfully_refreshed);
  bool refresher_succeeded = refresher(context);
  if (refresher_succeeded) {
    image_instance.ever_successfully_refreshed = true;
  }
  // Even if the refresher has returned false, it still might have submitted
  // some commands referencing the image. It's better to put an excessive
  // signal and wait slightly longer, for nothing important, while shutting down
  // than to destroy the image while it's still in use.
  image_instance.last_refresher_submission =
      guest_output_image_refresher_submission_tracker_.GetCurrentSubmission();
  // wwe2k14 patch (Session 6, DRAIN): if the refresher left its submission to another thread, consumers of
  // this image must wait until it is on the queue (see PaintAndPresentImpl and CaptureGuestOutput). Published
  // to them together with the image by the mailbox in Presenter::RefreshGuestOutput.
  image_instance.deferred_submission_order = context.deferred_submission_order();
  image_instance.deferred_submission_ticket = context.deferred_submission_ticket();
  // No need to make the refresher signal the fence by itself - signal it here
  // instead to have more control:
  // "Fence signal operations that are defined by vkQueueSubmit additionally
  //  include in the first synchronization scope all commands that occur earlier
  //  in submission order."
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  {
    VulkanSubmissionTracker::FenceAcquisition fence_acqusition(
        guest_output_image_refresher_submission_tracker_.AcquireFenceToAdvanceSubmission());
    // wwe2k14 patch (Session 6, DRAIN): the signal must follow the refresher's submission, which may not be on
    // the queue yet - let the thread submitting it signal the fence right after it instead of waiting here.
    // If it is already on the queue, signal here as usual.
    if (!image_instance.deferred_submission_order || fence_acqusition.fence() == VK_NULL_HANDLE ||
        !image_instance.deferred_submission_order->SignalFenceAfter(
            image_instance.deferred_submission_ticket, fence_acqusition.fence())) {
      const VulkanDevice::Queue::Acquisition queue_acquisition =
          vulkan_device_->AcquireQueue(vulkan_device_->queue_family_graphics_compute(), 0);
      if (dfn.vkQueueSubmit(queue_acquisition.queue(), 0, nullptr, fence_acqusition.fence()) !=
          VK_SUCCESS) {
        fence_acqusition.SubmissionSucceededSignalFailed();
      }
    } else {
      // wwe2k14 patch (Session 73): the submit thread signals it; not queried until it has.
      guest_output_image_refresher_submission_tracker_.NoteFenceSignalHandedOver(
          fence_acqusition.fence(), image_instance.deferred_submission_order);
    }
  }

  return refresher_succeeded;
}

VkSwapchainKHR VulkanPresenter::PaintContext::CreateSwapchainForVulkanSurface(
    const VulkanDevice* vulkan_device, VkSurfaceKHR surface, uint32_t width, uint32_t height,
    VkSwapchainKHR old_swapchain, uint32_t& present_queue_family_out, VkFormat& image_format_out,
    VkExtent2D& image_extent_out, bool& is_fifo_out, bool& ui_surface_unusable_out) {
  ui_surface_unusable_out = false;

  const VulkanInstance::Functions& ifn = vulkan_device->vulkan_instance()->functions();
  const VkPhysicalDevice physical_device = vulkan_device->physical_device();
  const VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  // Get surface capabilities.
  VkSurfaceCapabilitiesKHR surface_capabilities;
  if (ifn.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface,
                                                    &surface_capabilities) != VK_SUCCESS) {
    REXLOG_ERROR("VulkanPresenter: Failed to get Vulkan surface capabilities");
    // Some strange error, try again later.
    return VK_NULL_HANDLE;
  }

  // First, check if the surface is not zero-area because in this case, the rest
  // of the fields in theory may not be informative as the surface doesn't need
  // to go to presentation anyway, thus there's no need to return real
  // information, and ui_surface_unusable_out (from which it may not be possible
  // to recover on certain platforms at all) may be set to true spuriously if
  // any checks set it to true. VkSurfaceKHR may have zero-area bounds in some
  // window state cases on Windows, for example. Also clamp the requested size
  // to the maximum supported by the physical device as long as minImageExtent
  // in the instance's WSI allows that (if not, there's no way to satisfy both
  // requirements - the maximum 2D framebuffer size on the specific physical
  // device, and the minimum swap chain size on the whole instance - fail to
  // create until the surface becomes smaller).
  // wwe2k14 patch (Session 94, present_pre_rotate): follow the display's transform (ROTATE_90/180/270 only; mirrored
  // ones stay as before). The swapchain extent is then in the panel's native orientation: for 90 / 270 the surface's
  // own current extent when it has one, else the requested size swapped.
  uint32_t prerot_turns = 0;
  VkSurfaceTransformFlagBitsKHR prerot_transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  if (REXCVAR_GET(present_pre_rotate) && !g_prerot_failed.load()) {
#if REX_PLATFORM_ANDROID
    const VkSurfaceTransformFlagBitsKHR current = surface_capabilities.currentTransform;
    if ((current == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR ||
         current == VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR ||
         current == VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR) &&
        (surface_capabilities.supportedTransforms & current)) {
      prerot_transform = current;
      prerot_turns = current == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR    ? 1
                     : current == VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR ? 2
                                                                          : 3;
      if (prerot_turns & 1) {
        if (surface_capabilities.currentExtent.width != UINT32_MAX &&
            surface_capabilities.currentExtent.height != UINT32_MAX) {
          // currentExtent is in the current (window) orientation; the image is in the panel native one.
          width = surface_capabilities.currentExtent.height;
          height = surface_capabilities.currentExtent.width;
        } else {
          std::swap(width, height);
        }
      }
    }
#else
    static std::atomic<bool> logged_not_android{false};
    if (!logged_not_android.exchange(true)) {
      REXLOG_INFO("[pace] pre-rotate: present_pre_rotate does nothing on this platform (Android only)");
    }
#endif
  }
  VkExtent2D max_framebuffer_extent = util::GetMax2DFramebufferExtent(vulkan_device->properties());
  VkExtent2D image_extent;
  image_extent.width = std::min(std::max(std::min(width, max_framebuffer_extent.width),
                                         surface_capabilities.minImageExtent.width),
                                surface_capabilities.maxImageExtent.width);
  image_extent.height = std::min(std::max(std::min(height, max_framebuffer_extent.height),
                                          surface_capabilities.minImageExtent.height),
                                 surface_capabilities.maxImageExtent.height);
  if (!image_extent.width || !image_extent.height ||
      image_extent.width > max_framebuffer_extent.width ||
      image_extent.height > max_framebuffer_extent.height) {
    if (!g_swapchain_failure_logged.exchange(true)) REXLOG_INFO("VulkanPresenter: no swapchain created: image extent {}x{} unusable (requested {}x{}, max framebuffer {}x{})",
                image_extent.width, image_extent.height, width, height, max_framebuffer_extent.width,
                max_framebuffer_extent.height);
    return VK_NULL_HANDLE;
  }

  // Get the queue family for presentation.
  uint32_t queue_family_index_present = UINT32_MAX;
  const std::vector<VulkanDevice::QueueFamily>& queue_families = vulkan_device->queue_families();
  VkBool32 queue_family_present_supported;
  // First try the graphics and compute queue, prefer it to avoid the concurrent
  // image sharing mode.
  uint32_t queue_family_index_graphics_compute = vulkan_device->queue_family_graphics_compute();
  const VulkanDevice::QueueFamily& queue_family_graphics_compute =
      queue_families[queue_family_index_graphics_compute];
  if (queue_family_graphics_compute.may_support_presentation &&
      !queue_family_graphics_compute.queues.empty() &&
      ifn.vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, queue_family_index_graphics_compute,
                                               surface,
                                               &queue_family_present_supported) == VK_SUCCESS &&
      queue_family_present_supported) {
    queue_family_index_present = queue_family_index_graphics_compute;
  } else {
    for (uint32_t i = 0; i < uint32_t(queue_families.size()); ++i) {
      const VulkanDevice::QueueFamily& queue_family = queue_families[i];
      if (!queue_family.queues.empty() && queue_family.may_support_presentation &&
          ifn.vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, i, surface,
                                                   &queue_family_present_supported) == VK_SUCCESS &&
          queue_family_present_supported) {
        queue_family_index_present = i;
        break;
      }
    }
  }
  if (queue_family_index_present == UINT32_MAX) {
    // Not unusable though - may become presentable if the window (with the same
    // surface) moved to a different display, for instance.
    if (!g_swapchain_failure_logged.exchange(true)) REXLOG_INFO("VulkanPresenter: no swapchain created: no queue family can present to the surface ({}x{})", width,
                height);
    return VK_NULL_HANDLE;
  }

  // TODO(Triang3l): Support transforms.
  if (!prerot_turns && !(surface_capabilities.supportedTransforms &
        (VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR | VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR))) {
    REXLOG_ERROR(
        "VulkanPresenter: The surface doesn't support identity or "
        "window-system-controlled transform");
    return VK_NULL_HANDLE;
  }

  // Get the surface format.
  std::vector<VkSurfaceFormatKHR> surface_formats;
  VkResult surface_formats_get_result;
  for (;;) {
    uint32_t surface_format_count = uint32_t(surface_formats.size());
    bool surface_formats_were_empty = !surface_format_count;
    surface_formats_get_result = ifn.vkGetPhysicalDeviceSurfaceFormatsKHR(
        physical_device, surface, &surface_format_count,
        surface_formats_were_empty ? nullptr : surface_formats.data());
    // If the original presentation mode count was 0 (first call), SUCCESS is
    // returned, not INCOMPLETE.
    if (surface_formats_get_result == VK_SUCCESS || surface_formats_get_result == VK_INCOMPLETE) {
      surface_formats.resize(surface_format_count);
      if (surface_formats_get_result == VK_SUCCESS &&
          (!surface_formats_were_empty || !surface_format_count)) {
        break;
      }
    } else {
      break;
    }
  }
  if (surface_formats_get_result != VK_SUCCESS) {
    // Assuming any format in case of an error (or as some fallback in case of
    // specification violation).
    surface_formats.clear();
  }
#if REX_PLATFORM_ANDROID
  // Android uses R8G8B8A8.
  static const VkFormat kFormat8888Primary = VK_FORMAT_R8G8B8A8_UNORM;
  static const VkFormat kFormat8888Secondary = VK_FORMAT_B8G8R8A8_UNORM;
#else
  // GNU/Linux X11 and Windows DWM use B8G8R8A8.
  static const VkFormat kFormat8888Primary = VK_FORMAT_B8G8R8A8_UNORM;
  static const VkFormat kFormat8888Secondary = VK_FORMAT_R8G8B8A8_UNORM;
#endif
  VkSurfaceFormatKHR image_format;
  if (surface_formats.empty() ||
      (surface_formats.size() == 1 || surface_formats[0].format == VK_FORMAT_UNDEFINED)) {
    // Can choose any format if the implementation specifies only UNDEFINED.
    image_format.format = kFormat8888Primary;
    image_format.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  } else {
    // Pick the sRGB 8888 format preferred by the OS, fall back to any sRGB
    // 8888, then to 8888 with an unknown color space, and then to the first
    // sRGB available, and then to the first.
    auto format_8888_primary_it = surface_formats.cend();
    auto format_8888_secondary_it = surface_formats.cend();
    auto any_non_8888_srgb_it = surface_formats.cend();
    for (auto it = surface_formats.cbegin(); it != surface_formats.cend(); ++it) {
      if (it->format != kFormat8888Primary && it->format != kFormat8888Secondary) {
        if (it->colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR &&
            any_non_8888_srgb_it == surface_formats.cend()) {
          any_non_8888_srgb_it = it;
        }
        continue;
      }
      auto& preferred_8888_it =
          it->format == kFormat8888Primary ? format_8888_primary_it : format_8888_secondary_it;
      if (preferred_8888_it == surface_formats.cend()) {
        // First primary or secondary 8888 encounter.
        preferred_8888_it = it;
        continue;
      }
      // Is this a better primary or secondary 8888, that is, this is sRGB,
      // while the previous encounter was not?
      if (it->colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR &&
          preferred_8888_it->colorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        preferred_8888_it = it;
      }
    }
    if (format_8888_primary_it != surface_formats.cend() &&
        format_8888_secondary_it != surface_formats.cend()) {
      // Both the primary and the secondary 8888 formats are available - prefer
      // sRGB, if both are sRGB or not, prefer the primary format.
      if (format_8888_primary_it->colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR ||
          format_8888_secondary_it->colorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        image_format = *format_8888_primary_it;
      } else {
        image_format = *format_8888_secondary_it;
      }
    } else if (format_8888_primary_it != surface_formats.cend()) {
      // Only primary 8888.
      image_format = *format_8888_primary_it;
    } else if (format_8888_secondary_it != surface_formats.cend()) {
      // Only secondary 8888.
      image_format = *format_8888_secondary_it;
    } else if (any_non_8888_srgb_it != surface_formats.cend()) {
      // No 8888, but some sRGB format is available.
      image_format = *any_non_8888_srgb_it;
    } else {
      // Just pick any format.
      image_format = surface_formats.front();
    }
  }

  // Get presentation modes.
  std::vector<VkPresentModeKHR> present_modes;
  VkResult present_modes_get_result;
  for (;;) {
    uint32_t present_mode_count = uint32_t(present_modes.size());
    bool present_modes_were_empty = !present_mode_count;
    present_modes_get_result = ifn.vkGetPhysicalDeviceSurfacePresentModesKHR(
        physical_device, surface, &present_mode_count,
        present_modes_were_empty ? nullptr : present_modes.data());
    // If the original presentation mode count was 0 (first call), SUCCESS is
    // returned, not INCOMPLETE.
    if (present_modes_get_result == VK_SUCCESS || present_modes_get_result == VK_INCOMPLETE) {
      present_modes.resize(present_mode_count);
      if (present_modes_get_result == VK_SUCCESS &&
          (!present_modes_were_empty || !present_mode_count)) {
        break;
      }
    } else {
      break;
    }
  }
  if (present_modes_get_result != VK_SUCCESS) {
    // Assuming FIFO only (required everywhere) in case of an error.
    present_modes.clear();
  }

  // Create the swapchain.
  VkSwapchainCreateInfoKHR swapchain_create_info;
  swapchain_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  swapchain_create_info.pNext = nullptr;
  swapchain_create_info.flags = 0;
  swapchain_create_info.surface = surface;
  swapchain_create_info.minImageCount =
      std::max(kSubmissionCount, surface_capabilities.minImageCount);
  if (surface_capabilities.maxImageCount) {
    swapchain_create_info.minImageCount =
        std::min(swapchain_create_info.minImageCount, surface_capabilities.maxImageCount);
  }
  swapchain_create_info.imageFormat = image_format.format;
  swapchain_create_info.imageColorSpace = image_format.colorSpace;
  swapchain_create_info.imageExtent = image_extent;
  swapchain_create_info.imageArrayLayers = 1;
  swapchain_create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  uint32_t swapchain_queue_family_indices[2];
  if (queue_family_index_graphics_compute != queue_family_index_present) {
    // Using concurrent sharing mode to avoid an explicit ownership transfer
    // before presenting, which would require an additional command buffer
    // submission with the acquire barrier and a semaphore between the graphics
    // queue and the present queue. Different queues are an extremely rare case,
    // and Xenia only uses the swapchain for the final guest output and the
    // internal UI, so keeping framebuffer compression is not worth the
    // additional submission complexity and possibly latency.
    swapchain_queue_family_indices[0] = queue_family_index_graphics_compute;
    swapchain_queue_family_indices[1] = queue_family_index_present;
    swapchain_create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    swapchain_create_info.queueFamilyIndexCount = 2;
    swapchain_create_info.pQueueFamilyIndices = swapchain_queue_family_indices;
  } else {
    swapchain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchain_create_info.queueFamilyIndexCount = 0;
    swapchain_create_info.pQueueFamilyIndices = nullptr;
  }
  swapchain_create_info.preTransform =
      prerot_turns ? prerot_transform
      : (surface_capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
          ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
          : VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR;
  // Prefer opaque to avoid blending in the window system, or let that be
  // specified via the window system if it can't be forced. As a last resort,
  // just pick any - guest output will write alpha of 1 anyway.
  if (surface_capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) {
    swapchain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  } else if (surface_capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) {
    swapchain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
  } else {
    uint32_t composite_alpha_shift;
    if (!rex::bit_scan_forward(uint32_t(surface_capabilities.supportedCompositeAlpha),
                               &composite_alpha_shift)) {
      // Against the Vulkan specification, but breaks the logic here.
      REXLOG_ERROR(
          "VulkanPresenter: The surface doesn't support any composite alpha "
          "mode");
      return VK_NULL_HANDLE;
    }
    swapchain_create_info.compositeAlpha =
        VkCompositeAlphaFlagBitsKHR(uint32_t(1) << composite_alpha_shift);
  }
  // As presentation is usually controlled by the GPU command processor, it's
  // better to use modes that allow as quick acquisition as possible to avoid
  // interfering with GPU command processing, and also to allow tearing so
  // variable refresh rate may be used where it's available.
  // Note: If the priorities here are changes, update the cvar descriptions.
  if (vulkan_device->extensions().ext_GOOGLE_display_timing && REXCVAR_GET(present_display_timing)) {  // Session 95 review: plain turns pacing off; FIFO without pacing was a new path
    // wwe2k14 patch (Session 93): paced presents (present_display_timing) need the queue, as Android's Frame Pacing
    // library uses it: a desired present time means nothing in the modes that replace or skip frames.
    swapchain_create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  } else if (REXCVAR_GET(vulkan_allow_present_mode_immediate) &&
      std::find(present_modes.cbegin(), present_modes.cend(), VK_PRESENT_MODE_IMMEDIATE_KHR) !=
          present_modes.cend()) {
    // Allowing tearing to reduce latency, and possibly variable refresh rate
    // (though on Windows with borderless fullscreen, GDI copying is used
    // instead of independent flip, so it's not supported there).
    swapchain_create_info.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
  } else if (REXCVAR_GET(vulkan_allow_present_mode_mailbox) &&
             std::find(present_modes.cbegin(), present_modes.cend(), VK_PRESENT_MODE_MAILBOX_KHR) !=
                 present_modes.cend()) {
    // Allowing dropping frames to reduce latency, but no tearing.
    swapchain_create_info.presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
  } else if (REXCVAR_GET(vulkan_allow_present_mode_fifo_relaxed) &&
             std::find(present_modes.cbegin(), present_modes.cend(),
                       VK_PRESENT_MODE_FIFO_RELAXED_KHR) != present_modes.cend()) {
    // Limiting the frame rate, but lets too long frames cause tearing not to
    // make the latency even worse.
    swapchain_create_info.presentMode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
  } else {
    // Highest latency (but always guaranteed to be available).
    swapchain_create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  }
  swapchain_create_info.clipped = VK_TRUE;
  swapchain_create_info.oldSwapchain = old_swapchain;
  VkSwapchainKHR swapchain;
  if (dfn.vkCreateSwapchainKHR(device, &swapchain_create_info, nullptr, &swapchain) != VK_SUCCESS) {
    REXLOG_ERROR("VulkanPresenter: Failed to create a swapchain");
    return VK_NULL_HANDLE;
  }
  g_swapchain_failure_logged.store(false);
  // wwe2k14 patch (Session 94, present_pre_rotate): publish the transform for the painting, log when it changes.
  g_prerot_turns.store(prerot_turns);
  if (REXCVAR_GET(present_pre_rotate)) {
    const uint32_t logged_key = (prerot_turns << 24) ^ (image_extent.width << 12) ^ image_extent.height;
    if (g_prerot_logged_transform.exchange(logged_key) != logged_key) {
      REXLOG_INFO("[pace] pre-rotate: transform {}, swapchain {}x{}", uint32_t(surface_capabilities.currentTransform),
                  image_extent.width, image_extent.height);
    }
  }
  REXLOG_INFO(
      "VulkanPresenter: Created {}x{} swapchain with format {}, color space "
      "{}, presentation mode {}",
      swapchain_create_info.imageExtent.width, swapchain_create_info.imageExtent.height,
      uint32_t(swapchain_create_info.imageFormat), uint32_t(swapchain_create_info.imageColorSpace),
      uint32_t(swapchain_create_info.presentMode));

  present_queue_family_out = queue_family_index_present;
  image_format_out = swapchain_create_info.imageFormat;
  image_extent_out = swapchain_create_info.imageExtent;
  is_fifo_out = swapchain_create_info.presentMode == VK_PRESENT_MODE_FIFO_KHR ||
                swapchain_create_info.presentMode == VK_PRESENT_MODE_FIFO_RELAXED_KHR;
  return swapchain;
}

VkSwapchainKHR VulkanPresenter::PaintContext::PrepareForSwapchainRetirement() {
  const VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  if (swapchain != VK_NULL_HANDLE) {
    submission_tracker.AwaitAllSubmissionsCompletion();
    // wwe2k14 patch (Session 95, LIFE, VUID-vkDestroySwapchainKHR-swapchain-01282): the fences above cover the paint submissions, but
    // vkQueuePresentKHR takes no fence, so the last present can still be on the queue when the swapchain goes. The entry is looked up here
    // (not in VulkanDevice::Functions) so no header the game exe includes changes; this runs only at a resize or a surface change.
    // Review (s95-all): the wait holds the queue through a drain of all queued game work, so it is Android's (surfaces come and go there), and a PC run with the
    // validation layer's, where the VUID is read. A player's PC keeps the old timing at a resize or a fullscreen toggle.
#if REX_PLATFORM_ANDROID
    const bool wait_for_present = true;
#else
    const bool wait_for_present = REXCVAR_GET(vulkan_validation_enabled);
#endif
    if (wait_for_present && present_queue_family != UINT32_MAX) {
      const auto queue_wait_idle = reinterpret_cast<PFN_vkQueueWaitIdle>(
          vulkan_device->vulkan_instance()->functions().vkGetDeviceProcAddr(device, "vkQueueWaitIdle"));
      if (queue_wait_idle) {
        const VulkanDevice::Queue::Acquisition queue_acquisition = vulkan_device->AcquireQueue(present_queue_family, 0);
        VkResult idle_result = queue_wait_idle(queue_acquisition.queue());
#if REX_PLATFORM_ANDROID
        // Adreno returns VK_TIMEOUT early from infinite fence waits (see the submission tracker); retry so the present is really done (bounded).
        for (int retry = 0; idle_result == VK_TIMEOUT && retry < 100; ++retry) {
          idle_result = queue_wait_idle(queue_acquisition.queue());
        }
#endif
        if (idle_result != VK_SUCCESS) {
          REXLOG_WARN("VulkanPresenter: vkQueueWaitIdle before retiring the swapchain gave {}", int(idle_result));
        }
      }
    }
  }
  for (const SwapchainFramebuffer& framebuffer : swapchain_framebuffers) {
    dfn.vkDestroyFramebuffer(device, framebuffer.framebuffer, nullptr);
    dfn.vkDestroyImageView(device, framebuffer.image_view, nullptr);
  }
  swapchain_framebuffers.clear();
  swapchain_images.clear();
  swapchain_extent.width = 0;
  swapchain_extent.height = 0;
  // wwe2k14 patch (Session 95, present_acquire_fence): an image acquired from the retiring swapchain is dropped with it.
  if (g_acquire_fence.fence != VK_NULL_HANDLE && g_acquire_fence.pending && !g_acquire_fence.ready) {
    // Bounded wait so the fence can be reused; if it never signals, abandon it (leaked) for a fresh one.
    if (vulkan_device->functions().vkWaitForFences(vulkan_device->device(), 1, &g_acquire_fence.fence, VK_TRUE,
                                                   100000000ull) != VK_SUCCESS) {
      g_acquire_fence.fence = VK_NULL_HANDLE;
    }
  }
  g_acquire_fence.pending = false;
  g_acquire_fence.ready = false;
  g_acquire_fence.swapchain = VK_NULL_HANDLE;
  // The old swapchain must be destroyed externally.
  VkSwapchainKHR old_swapchain = swapchain;
  swapchain = nullptr;
  return old_swapchain;
}

void VulkanPresenter::PaintContext::DestroySwapchainAndVulkanSurface() {
  VkSwapchainKHR old_swapchain = PrepareForSwapchainRetirement();
  if (old_swapchain != VK_NULL_HANDLE) {
    vulkan_device->functions().vkDestroySwapchainKHR(vulkan_device->device(), old_swapchain,
                                                     nullptr);
  }
  present_queue_family = UINT32_MAX;
  if (vulkan_surface != VK_NULL_HANDLE) {
    const VulkanInstance* vulkan_instance = vulkan_device->vulkan_instance();
    vulkan_instance->functions().vkDestroySurfaceKHR(vulkan_instance->instance(), vulkan_surface,
                                                     nullptr);
    vulkan_surface = VK_NULL_HANDLE;
  }
}

VulkanPresenter::GuestOutputImage::~GuestOutputImage() {
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  if (view_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImageView(device, view_, nullptr);
  }
  if (image_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImage(device, image_, nullptr);
  }
  if (memory_ != VK_NULL_HANDLE) {
    dfn.vkFreeMemory(device, memory_, nullptr);
  }
}

bool VulkanPresenter::GuestOutputImage::Initialize() {
  VkImageCreateInfo image_create_info;
  image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_create_info.pNext = nullptr;
  image_create_info.flags = 0;
  image_create_info.imageType = VK_IMAGE_TYPE_2D;
  image_create_info.format = kGuestOutputFormat;
  image_create_info.extent.width = extent_.width;
  image_create_info.extent.height = extent_.height;
  image_create_info.extent.depth = 1;
  image_create_info.mipLevels = 1;
  image_create_info.arrayLayers = 1;
  image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_create_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_create_info.queueFamilyIndexCount = 0;
  image_create_info.pQueueFamilyIndices = nullptr;
  image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (!ui::vulkan::util::CreateDedicatedAllocationImage(
          vulkan_device_, image_create_info, ui::vulkan::util::MemoryPurpose::kDeviceLocal, image_,
          memory_)) {
    REXLOG_ERROR("VulkanPresenter: Failed to create a guest output image");
    return false;
  }

  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  VkImageViewCreateInfo image_view_create_info;
  image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  image_view_create_info.pNext = nullptr;
  image_view_create_info.flags = 0;
  image_view_create_info.image = image_;
  image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  image_view_create_info.format = kGuestOutputFormat;
  image_view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  image_view_create_info.subresourceRange.baseMipLevel = 0;
  image_view_create_info.subresourceRange.levelCount = 1;
  image_view_create_info.subresourceRange.baseArrayLayer = 0;
  image_view_create_info.subresourceRange.layerCount = 1;
  if (dfn.vkCreateImageView(device, &image_view_create_info, nullptr, &view_) != VK_SUCCESS) {
    REXLOG_ERROR("VulkanPresenter: Failed to create a guest output image view");
    return false;
  }

  return true;
}

Presenter::PaintResult VulkanPresenter::PaintAndPresentImpl(bool execute_ui_drawers) {
  // wwe2k14 patch (Session 95, HANG): whichever thread paints (the UI thread or the guest output thread, see
  // GetDesiredPaintModeFromUIThread), the guard sees it go in and whether it came out.
  auto& hang_beats = rex::perf::hang_beats::Get();
  hang_beats.paints.fetch_add(1, std::memory_order_relaxed);
  hang_beats.in_paint.fetch_add(1, std::memory_order_relaxed);
  struct InPaint {
    std::atomic<uint32_t>& flag;
    ~InPaint() { flag.fetch_sub(1, std::memory_order_relaxed); }
  } in_paint{hang_beats.in_paint};
  rex::perf::hang_beats::TestBlock(rex::perf::hang_beats::kTestPaint);  // --wwe2k14_test_hang=paint

  // wwe2k14 patch (Session 95, present_acquire_fence): get the swapchain image with a fence, never blocking; a paint
  // whose image is not ready yet is skipped here (nothing recorded, nothing waited on; the guest output mailbox keeps the
  // newest frame and the next paint tries again).
  // Paced presents (present_display_timing) hold FIFO images on purpose, so a late image is normal there and skipping
  // would unpace them: the fence mode stays off. A pending image keeps the mode on until it is used.
  const bool acquire_fence_requested = REXCVAR_GET(present_acquire_fence);
  const bool acquire_fence_blocked = acquire_fence_requested && REXCVAR_GET(present_display_timing);
  static bool acquire_fence_blocked_logged = false;
  if (acquire_fence_blocked && !acquire_fence_blocked_logged) {
    acquire_fence_blocked_logged = true;
    REXLOG_WARN("[pace] present_acquire_fence ignored: present_display_timing is on (paced presents hold images on "
                "purpose)");
  }
  const bool acquire_fence_mode = (acquire_fence_requested && !acquire_fence_blocked) || g_acquire_fence.pending;
  if (acquire_fence_mode) {
    AcquireFenceState& af = g_acquire_fence;
    const VulkanDevice::Functions& afn = vulkan_device_->functions();
    const VkDevice adevice = vulkan_device_->device();
    if (!af.logged) {
      af.logged = true;
      REXLOG_INFO("[pace] present_acquire_fence on: swapchain images are acquired with a polled fence, paints with a late "
                  "image are skipped, the paint submission waits on no acquire semaphore");
    }
    if (++af.paints >= 600) {
      REXLOG_INFO("[pace] present_acquire_fence: {} paints skipped of 600 ({} in all), longest acquire-to-ready {:.2f} ms",
                  af.skipped, af.skipped_total, af.longest_ms);
      af.paints = 0;
      af.skipped = 0;
      af.longest_ms = 0.0;
    }
    if (af.fence == VK_NULL_HANDLE) {
      VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      if (afn.vkCreateFence(adevice, &fence_info, nullptr, &af.fence) != VK_SUCCESS) {
        REXLOG_ERROR("VulkanPresenter: Failed to create the swapchain image acquisition fence");
        return PaintResult::kNotPresented;
      }
    }
    if (af.pending && af.swapchain != paint_context_.swapchain) {
      af.pending = false;
      af.ready = false;
    }
    if (!af.pending) {
      afn.vkResetFences(adevice, 1, &af.fence);
      uint32_t acquired_index = 0;
      perf::timeline::Record("pacq_begin");  // where the paint thread can block
      const VkResult fence_acquire_result = afn.vkAcquireNextImageKHR(adevice, paint_context_.swapchain, 0, VK_NULL_HANDLE,
                                                                      af.fence, &acquired_index);
      perf::timeline::Record("pacq_end", acquired_index, uint32_t(fence_acquire_result));
      switch (fence_acquire_result) {
        case VK_SUCCESS:
        case VK_SUBOPTIMAL_KHR:
          af.pending = true;
          af.ready = false;
          af.swapchain = paint_context_.swapchain;
          af.image_index = acquired_index;
          af.acquired_at = std::chrono::steady_clock::now();
          break;
        case VK_NOT_READY:
        case VK_TIMEOUT:
          ++af.skipped;
          ++af.skipped_total;
          return PaintResult::kNotPresentedImageLate;
        case VK_ERROR_DEVICE_LOST:
          REXLOG_ERROR("VulkanPresenter: Failed to acquire the swapchain image as the device has been lost");
          return PaintResult::kGpuLostResponsible;
        case VK_ERROR_OUT_OF_DATE_KHR:
        case VK_ERROR_SURFACE_LOST_KHR:
        case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
          REXLOG_INFO("VulkanPresenter: Presentation to the swapchain image has been dropped as the swapchain or the "
                      "surface has become outdated (acquire)");
          return PaintResult::kNotPresentedConnectionOutdated;
        default:
          REXLOG_ERROR("VulkanPresenter: Failed to acquire the swapchain image");
          return PaintResult::kNotPresented;
      }
    }
    if (!af.ready) {
      const VkResult fence_status = afn.vkGetFenceStatus(adevice, af.fence);
      if (fence_status == VK_ERROR_DEVICE_LOST) {
        REXLOG_ERROR("VulkanPresenter: The device has been lost (swapchain image acquisition fence)");
        return PaintResult::kGpuLostResponsible;
      }
      if (fence_status != VK_SUCCESS) {
        ++af.skipped;
        ++af.skipped_total;
        return PaintResult::kNotPresentedImageLate;
      }
      af.ready = true;
      const double waited_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - af.acquired_at).count();
      if (waited_ms > af.longest_ms) {
        af.longest_ms = waited_ms;
      }
      perf::timeline::Record("pacq_ready", af.image_index, uint32_t(waited_ms * 1000.0 + 0.5));
    }
  }
  // Begin the submission in place of the one not currently potentially used on
  // the GPU.
  uint64_t current_paint_submission_index =
      paint_context_.submission_tracker.GetCurrentSubmission();
  uint64_t paint_submission_count = uint64_t(paint_context_.submissions.size());
  if (current_paint_submission_index >= paint_submission_count) {
    // wwe2k14 patch (Session 95): timeline events around the wait for an older paint's fence.
    perf::timeline::Record("pfence_begin", uint32_t(current_paint_submission_index));
    paint_context_.submission_tracker.AwaitSubmissionCompletion(current_paint_submission_index -
                                                                paint_submission_count);
    perf::timeline::Record("pfence_end", uint32_t(current_paint_submission_index));
  }
  const PaintContext::Submission& paint_submission =
      *paint_context_.submissions[current_paint_submission_index % paint_submission_count];

  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  VkCommandPool draw_command_pool = paint_submission.draw_command_pool();
  if (dfn.vkResetCommandPool(device, draw_command_pool, 0) != VK_SUCCESS) {
    REXLOG_ERROR(
        "VulkanPresenter: Failed to reset a command buffer for drawing to the "
        "swapchain");
    return PaintResult::kNotPresented;
  }

  VkCommandBuffer draw_command_buffer = paint_submission.draw_command_buffer();
  VkCommandBufferBeginInfo command_buffer_begin_info;
  command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  command_buffer_begin_info.pNext = nullptr;
  command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  command_buffer_begin_info.pInheritanceInfo = nullptr;
  if (dfn.vkBeginCommandBuffer(draw_command_buffer, &command_buffer_begin_info) != VK_SUCCESS) {
    REXLOG_ERROR(
        "VulkanPresenter: Failed to being recording the command buffer for "
        "drawing to the swapchain");
    return PaintResult::kNotPresented;
  }
  // vkResetCommandPool resets from both initial and recording states, still
  // safe to return early from this function in case of an error.

  // wwe2k14 patch (Session 95, item T): the paint's begin timestamp; the slot counts only once the submit has succeeded.
  uint32_t paint_timestamp_slot = UINT32_MAX;
  if (rex::perf::gpu_gap::Enabled()) {
    PaintTimestamps& pt = g_paint_timestamps;
    if (pt.pool == VK_NULL_HANDLE && !pt.unsupported) {
      const VulkanInstance::Functions& ifn = vulkan_device_->vulkan_instance()->functions();
      uint32_t family_count = 0;
      ifn.vkGetPhysicalDeviceQueueFamilyProperties(vulkan_device_->physical_device(), &family_count, nullptr);
      std::vector<VkQueueFamilyProperties> families(family_count);
      ifn.vkGetPhysicalDeviceQueueFamilyProperties(vulkan_device_->physical_device(), &family_count, families.data());
      const uint32_t family = vulkan_device_->queue_family_graphics_compute();
      const uint32_t valid_bits = family < family_count ? families[family].timestampValidBits : 0;
      VkPhysicalDeviceProperties device_properties;
      ifn.vkGetPhysicalDeviceProperties(vulkan_device_->physical_device(), &device_properties);
      VkQueryPoolCreateInfo pool_info = {};
      pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
      pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
      pool_info.queryCount = PaintTimestamps::kSlots * 2;
      if (valid_bits && device_properties.limits.timestampPeriod > 0.0f &&
          dfn.vkCreateQueryPool(device, &pool_info, nullptr, &pt.pool) == VK_SUCCESS) {
        pt.ns_per_tick = double(device_properties.limits.timestampPeriod);
        pt.mask = valid_bits >= 64 ? UINT64_MAX : (uint64_t(1) << valid_bits) - 1;
        rex::perf::gpu_gap::SetTicks(pt.ns_per_tick, pt.mask);
        REXLOG_INFO("[pace][gpu-gap] on: paint timestamps, {} valid bits, {} ns per tick, swapchain {}x{}", valid_bits,
                    device_properties.limits.timestampPeriod, paint_context_.swapchain_extent.width,
                    paint_context_.swapchain_extent.height);
      } else {
        REXLOG_WARN("[pace][gpu-gap] paint timestamps unavailable ({} valid bits)", valid_bits);
        pt.pool = VK_NULL_HANDLE;
        pt.unsupported = true;
      }
    }
    if (pt.pool != VK_NULL_HANDLE) {
      const uint32_t slot = pt.next;
      pt.next = (pt.next + 1) % PaintTimestamps::kSlots;
      if (pt.submitted[slot]) {
        pt.submitted[slot] = false;
        uint64_t ticks[2];
        if (dfn.vkGetQueryPoolResults(device, pt.pool, slot * 2, 2, sizeof(ticks), ticks, sizeof(uint64_t),
                                      VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
          rex::perf::gpu_gap::NotePaint(ticks[0], ticks[1]);
          perf::timeline::Record("ppaint_ts", slot,
                                 uint32_t(double((ticks[1] - ticks[0]) & pt.mask) * pt.ns_per_tick / 1000.0 + 0.5));
        }
      }
      dfn.vkCmdResetQueryPool(draw_command_buffer, pt.pool, slot * 2, 2);
      dfn.vkCmdWriteTimestamp(draw_command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pt.pool, slot * 2);
      paint_timestamp_slot = slot;
    }
  }

  VkSemaphore acquire_semaphore = paint_submission.acquire_semaphore();
  uint32_t swapchain_image_index = 0;
  VkResult acquire_result = VK_SUCCESS;
  if (acquire_fence_mode) {
    // Acquired and seen ready above; the paint submission waits on no acquire semaphore.
    swapchain_image_index = g_acquire_fence.image_index;
    acquire_semaphore = VK_NULL_HANDLE;
  } else {
    perf::timeline::Record("pacq_begin");  // wwe2k14 patch (Session 95): where the paint thread blocks
    // wwe2k14 patch (Session 95, HANG): bounded on Android, where a stuck acquire would hold the paint thread for good. Some
    // drivers may ignore a finite timeout: the guard (src/hang_guard.h in the game) stays the backstop.
#if REX_PLATFORM_ANDROID
    constexpr uint64_t kAcquireTimeoutNs = 2000000000ull;
#else
    constexpr uint64_t kAcquireTimeoutNs = UINT64_MAX;
#endif
    // wwe2k14 patch (Session 93, log only): the call is marked in progress (Presenter::HostPresentCallMs) and a slow one is logged.
    Presenter::NoteHostPresentCall("vkAcquireNextImageKHR");
    const auto acquire_t0 = std::chrono::steady_clock::now();
    acquire_result = dfn.vkAcquireNextImageKHR(device, paint_context_.swapchain, kAcquireTimeoutNs, acquire_semaphore,
                                               VK_NULL_HANDLE, &swapchain_image_index);
    Presenter::NoteHostPresentCall(nullptr);
    SlowHostPresentCallNote("vkAcquireNextImageKHR", acquire_t0, int(acquire_result), paint_context_.swapchain_is_fifo, 0);
    perf::timeline::Record("pacq_end", swapchain_image_index, uint32_t(acquire_result));
  }
  switch (acquire_result) {
    case VK_SUCCESS:
    case VK_SUBOPTIMAL_KHR:
      break;
    case VK_ERROR_DEVICE_LOST:
      REXLOG_ERROR(
          "VulkanPresenter: Failed to acquire the swapchain image as the "
          "device has been lost");
      return PaintResult::kGpuLostResponsible;
    case VK_ERROR_OUT_OF_DATE_KHR:
    case VK_ERROR_SURFACE_LOST_KHR:
    case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
      // Not an error, reporting just as info (may normally occur while resizing
      // on some platforms).
      REXLOG_INFO(
          "VulkanPresenter: Presentation to the swapchain image has been "
          "dropped as the swapchain or the surface has become outdated (acquire)");
      return PaintResult::kNotPresentedConnectionOutdated;
    case VK_TIMEOUT:
    case VK_NOT_READY: {
      // wwe2k14 patch (Session 95, HANG): no image, the semaphore untouched: this present is skipped, the next paint tries again.
      const uint64_t timeouts = hang_beats.acquire_timeouts.fetch_add(1, std::memory_order_relaxed) + 1;
      if (timeouts <= 16) {
        REXLOG_WARN("[pace][present] the swapchain gave no image in time (VkResult {}): present skipped ({} so far)",
                    int(acquire_result), timeouts);
      }
      return PaintResult::kNotPresented;
    }
    default:
      REXLOG_ERROR("VulkanPresenter: Failed to acquire the swapchain image");
      return PaintResult::kNotPresented;
  }

  // Non-zero extents needed for both the viewport (width must not be zero) and
  // the guest output rectangle.
  assert_not_zero(paint_context_.swapchain_extent.width);
  assert_not_zero(paint_context_.swapchain_extent.height);

  // wwe2k14 patch (Session 94, present_pre_rotate): the picture is laid out in the window's (logical) orientation and
  // rotated by the final pass into the swapchain image (native orientation, extent swapped for 90 / 270).
  const uint32_t prerot_turns = g_prerot_turns.load();
  const uint32_t logical_width = (prerot_turns & 1) ? paint_context_.swapchain_extent.height
                                                    : paint_context_.swapchain_extent.width;
  const uint32_t logical_height = (prerot_turns & 1) ? paint_context_.swapchain_extent.width
                                                     : paint_context_.swapchain_extent.height;
  const uint32_t prerot_map = PrerotMapOfTurns(prerot_turns);
  bool prerot_failed_now = false;

  bool swapchain_image_clear_needed = true;
  VkClearAttachment swapchain_image_clear_attachment;
  swapchain_image_clear_attachment.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  swapchain_image_clear_attachment.colorAttachment = 0;
  swapchain_image_clear_attachment.clearValue.color.float32[0] = 0.0f;
  swapchain_image_clear_attachment.clearValue.color.float32[1] = 0.0f;
  swapchain_image_clear_attachment.clearValue.color.float32[2] = 0.0f;
  swapchain_image_clear_attachment.clearValue.color.float32[3] = 1.0f;

  VkRenderPassBeginInfo swapchain_render_pass_begin_info;
  swapchain_render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  swapchain_render_pass_begin_info.pNext = nullptr;
  swapchain_render_pass_begin_info.renderPass = paint_context_.swapchain_render_pass;
  swapchain_render_pass_begin_info.framebuffer =
      paint_context_.swapchain_framebuffers[swapchain_image_index].framebuffer;
  swapchain_render_pass_begin_info.renderArea.offset.x = 0;
  swapchain_render_pass_begin_info.renderArea.offset.y = 0;
  swapchain_render_pass_begin_info.renderArea.extent = paint_context_.swapchain_extent;
  swapchain_render_pass_begin_info.clearValueCount = 0;
  swapchain_render_pass_begin_info.pClearValues = nullptr;
  if (paint_context_.swapchain_render_pass_clear_load_op) {
    swapchain_render_pass_begin_info.clearValueCount = 1;
    swapchain_render_pass_begin_info.pClearValues = &swapchain_image_clear_attachment.clearValue;
    swapchain_image_clear_needed = false;
  }

  bool swapchain_image_pass_begun = false;

  GuestOutputProperties guest_output_properties;
  GuestOutputPaintConfig guest_output_paint_config;
  std::shared_ptr<GuestOutputImage> guest_output_image;
  // wwe2k14 patch (Session 6, DRAIN): see GuestOutputImageInstance::deferred_submission_order. Awaited right
  // before the vkQueueSubmit below, so recording overlaps the wait.
  std::shared_ptr<VulkanQueueSubmitOrder> guest_output_deferred_submission_order;
  uint64_t guest_output_deferred_submission_ticket = 0;
  {
    uint32_t guest_output_mailbox_index;
    std::unique_lock<std::mutex> guest_output_consumer_lock(ConsumeGuestOutput(
        guest_output_mailbox_index, &guest_output_properties, &guest_output_paint_config));
    if (guest_output_mailbox_index != UINT32_MAX) {
      assert_true(guest_output_images_[guest_output_mailbox_index].ever_successfully_refreshed);
      guest_output_image = guest_output_images_[guest_output_mailbox_index].image;
      guest_output_deferred_submission_order =
          guest_output_images_[guest_output_mailbox_index].deferred_submission_order;
      guest_output_deferred_submission_ticket =
          guest_output_images_[guest_output_mailbox_index].deferred_submission_ticket;
    }
    // Incremented the reference count of the guest output image - safe to leave
    // the consumer critical section now as everything here either will be using
    // the new reference or is exclusively owned by main target painting (and
    // multiple threads can't paint the main target at the same time).
  }
  // wwe2k14 patch (Session 95): the steps between taking a frame and presenting it, on the timeline: pcons, prec, pfa, psub,
  // ppres (the Surface's paints spend 30+ ms here under GPU load; which call is not known yet).
  perf::timeline::Record("pcons", uint32_t(guest_output_properties.refresh_seq));
  PaintGateNoteConsumed(guest_output_properties.refresh_seq);

  if (guest_output_image) {
    VkExtent2D max_framebuffer_extent =
        util::GetMax2DFramebufferExtent(vulkan_device_->properties());
    GuestOutputPaintFlow guest_output_flow = GetGuestOutputPaintFlow(
        guest_output_properties, logical_width, logical_height, max_framebuffer_extent.width,
        max_framebuffer_extent.height, guest_output_paint_config);
    if (guest_output_flow.effect_count) {
      // Store the main target reference to the guest output image so it's not
      // destroyed while it's still potentially in use by main target painting
      // queued on the GPU.
      size_t guest_output_image_paint_ref_index = SIZE_MAX;
      size_t guest_output_image_paint_ref_new_index = SIZE_MAX;
      // Try to find the existing reference to the same image, or an already
      // released (or a taken, but never actually used) slot.
      for (size_t i = 0; i < paint_context_.guest_output_image_paint_refs.size(); ++i) {
        const std::pair<uint64_t, std::shared_ptr<GuestOutputImage>>& guest_output_image_paint_ref =
            paint_context_.guest_output_image_paint_refs[i];
        if (guest_output_image_paint_ref.second == guest_output_image) {
          guest_output_image_paint_ref_index = i;
          break;
        }
        if (guest_output_image_paint_ref_new_index == SIZE_MAX &&
            (!guest_output_image_paint_ref.second || !guest_output_image_paint_ref.first)) {
          guest_output_image_paint_ref_new_index = i;
        }
      }
      if (guest_output_image_paint_ref_index == SIZE_MAX) {
        // New image - store the reference and create the descriptors.
        if (guest_output_image_paint_ref_new_index == SIZE_MAX) {
          // Replace the earliest used reference.
          guest_output_image_paint_ref_new_index = 0;
          for (size_t i = 1; i < paint_context_.guest_output_image_paint_refs.size(); ++i) {
            if (paint_context_.guest_output_image_paint_refs[i].first <
                paint_context_.guest_output_image_paint_refs[guest_output_image_paint_ref_new_index]
                    .first) {
              guest_output_image_paint_ref_new_index = i;
            }
          }
          // Await the completion of the usage of the old guest output image and
          // its descriptors.
          perf::timeline::Record("pm_refwait");  // wwe2k14 patch (Session 95, spike): the steps inside pcons .. prec
          paint_context_.submission_tracker.AwaitSubmissionCompletion(
              paint_context_.guest_output_image_paint_refs[guest_output_image_paint_ref_new_index]
                  .first);
        }
        guest_output_image_paint_ref_index = guest_output_image_paint_ref_new_index;
        // The actual submission index will be set if the image is actually
        // used, not dropped due to some error.
        paint_context_.guest_output_image_paint_refs[guest_output_image_paint_ref_index] =
            std::make_pair(uint64_t(0), guest_output_image);
        // Create the descriptors of the new image.
        VkDescriptorImageInfo guest_output_image_descriptor_image_info;
        guest_output_image_descriptor_image_info.sampler = VK_NULL_HANDLE;
        guest_output_image_descriptor_image_info.imageView = guest_output_image->view();
        guest_output_image_descriptor_image_info.imageLayout =
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet guest_output_image_descriptor_write;
        guest_output_image_descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        guest_output_image_descriptor_write.pNext = nullptr;
        guest_output_image_descriptor_write.dstSet =
            paint_context_.guest_output_descriptor_sets
                [PaintContext::kGuestOutputDescriptorSetGuestOutput0Sampled +
                 guest_output_image_paint_ref_index];
        guest_output_image_descriptor_write.dstBinding = 0;
        guest_output_image_descriptor_write.dstArrayElement = 0;
        guest_output_image_descriptor_write.descriptorCount = 1;
        guest_output_image_descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        guest_output_image_descriptor_write.pImageInfo = &guest_output_image_descriptor_image_info;
        guest_output_image_descriptor_write.pBufferInfo = nullptr;
        guest_output_image_descriptor_write.pTexelBufferView = nullptr;
        dfn.vkUpdateDescriptorSets(device, 1, &guest_output_image_descriptor_write, 0, nullptr);
      }

      perf::timeline::Record("pm_refs");  // wwe2k14 patch (Session 95, spike)
      // Make sure intermediate textures of the needed size are available, and
      // unneeded intermediate textures are destroyed.
      for (size_t i = 0; i < kMaxGuestOutputPaintEffects - 1; ++i) {
        std::pair<uint32_t, uint32_t> intermediate_needed_size(0, 0);
        if (i + 1 < guest_output_flow.effect_count) {
          intermediate_needed_size = guest_output_flow.effect_output_sizes[i];
        }
        std::unique_ptr<GuestOutputImage>& intermediate_image_ptr_ref =
            paint_context_.guest_output_intermediate_images[i];
        VkExtent2D intermediate_current_extent(
            intermediate_image_ptr_ref ? intermediate_image_ptr_ref->extent() : VkExtent2D{});
        if (intermediate_current_extent.width != intermediate_needed_size.first ||
            intermediate_current_extent.height != intermediate_needed_size.second) {
          if (intermediate_needed_size.first && intermediate_needed_size.second) {
            // Need to replace immediately as a new image with the requested
            // size is needed.
            if (intermediate_image_ptr_ref) {
              paint_context_.submission_tracker.AwaitSubmissionCompletion(
                  paint_context_.guest_output_intermediate_image_last_submission);
              intermediate_image_ptr_ref.reset();
              util::DestroyAndNullHandle(dfn.vkDestroyFramebuffer, device,
                                         paint_context_.guest_output_intermediate_framebuffers[i]);
            }
            // Image.
            intermediate_image_ptr_ref = GuestOutputImage::Create(
                vulkan_device_, intermediate_needed_size.first, intermediate_needed_size.second);
            if (!intermediate_image_ptr_ref) {
              // Don't display the guest output, and don't try to create more
              // intermediate textures (only destroy them).
              guest_output_flow.effect_count = 0;
              continue;
            }
            // Framebuffer.
            VkImageView intermediate_framebuffer_attachment = intermediate_image_ptr_ref->view();
            VkFramebufferCreateInfo intermediate_framebuffer_create_info;
            intermediate_framebuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            intermediate_framebuffer_create_info.pNext = nullptr;
            intermediate_framebuffer_create_info.flags = 0;
            intermediate_framebuffer_create_info.renderPass =
                guest_output_intermediate_render_pass_;
            intermediate_framebuffer_create_info.attachmentCount = 1;
            intermediate_framebuffer_create_info.pAttachments =
                &intermediate_framebuffer_attachment;
            intermediate_framebuffer_create_info.width = intermediate_needed_size.first;
            intermediate_framebuffer_create_info.height = intermediate_needed_size.second;
            intermediate_framebuffer_create_info.layers = 1;
            if (dfn.vkCreateFramebuffer(
                    device, &intermediate_framebuffer_create_info, nullptr,
                    &paint_context_.guest_output_intermediate_framebuffers[i]) != VK_SUCCESS) {
              REXLOG_ERROR(
                  "VulkanPresenter: Failed to create a guest output "
                  "intermediate framebuffer");
              // Don't display the guest output, and don't try to create more
              // intermediate textures (only destroy them).
              guest_output_flow.effect_count = 0;
              continue;
            }
            // Descriptors.
            VkDescriptorImageInfo intermediate_descriptor_image_info;
            intermediate_descriptor_image_info.sampler = VK_NULL_HANDLE;
            intermediate_descriptor_image_info.imageView = intermediate_image_ptr_ref->view();
            intermediate_descriptor_image_info.imageLayout =
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkWriteDescriptorSet intermediate_descriptor_write;
            intermediate_descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            intermediate_descriptor_write.pNext = nullptr;
            intermediate_descriptor_write.dstSet =
                paint_context_.guest_output_descriptor_sets
                    [PaintContext ::kGuestOutputDescriptorSetIntermediate0Sampled + i];
            intermediate_descriptor_write.dstBinding = 0;
            intermediate_descriptor_write.dstArrayElement = 0;
            intermediate_descriptor_write.descriptorCount = 1;
            intermediate_descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            intermediate_descriptor_write.pImageInfo = &intermediate_descriptor_image_info;
            intermediate_descriptor_write.pBufferInfo = nullptr;
            intermediate_descriptor_write.pTexelBufferView = nullptr;
            dfn.vkUpdateDescriptorSets(device, 1, &intermediate_descriptor_write, 0, nullptr);
          } else {
            // Was previously needed, but not anymore - destroy when possible.
            if (intermediate_image_ptr_ref &&
                paint_context_.submission_tracker.UpdateAndGetCompletedSubmission() >=
                    paint_context_.guest_output_intermediate_image_last_submission) {
              intermediate_image_ptr_ref.reset();
              util::DestroyAndNullHandle(dfn.vkDestroyFramebuffer, device,
                                         paint_context_.guest_output_intermediate_framebuffers[i]);
            }
          }
        }
      }

      if (guest_output_flow.effect_count) {
        // Check if all the intermediate effects are supported by the
        // implementation.
        for (size_t i = 0; i + 1 < guest_output_flow.effect_count; ++i) {
          if (paint_context_.guest_output_paint_pipelines[size_t(guest_output_flow.effects[i])]
                  .intermediate_pipeline == VK_NULL_HANDLE) {
            guest_output_flow.effect_count = 0;
            break;
          }
        }
        if (guest_output_flow.effect_count) {
          // Ensure the pipeline exists for the final effect drawing to the
          // swapchain, for the render pass with the up-to-date image format.
          GuestOutputPaintEffect swapchain_effect =
              guest_output_flow.effects[guest_output_flow.effect_count - 1];
          PaintContext::GuestOutputPaintPipeline& swapchain_effect_pipeline =
              paint_context_.guest_output_paint_pipelines[size_t(swapchain_effect)];
          if (swapchain_effect_pipeline.swapchain_pipeline != VK_NULL_HANDLE &&
              (swapchain_effect_pipeline.swapchain_format !=
                   paint_context_.swapchain_render_pass_format ||
               (size_t(swapchain_effect) < kPrerotEffectCapacity &&
                (g_prerot_built_map[size_t(swapchain_effect)] != prerot_map ||
                 (prerot_map && (g_prerot_built_width[size_t(swapchain_effect)] != logical_width ||
                                 g_prerot_built_height[size_t(swapchain_effect)] != logical_height)))))) {
            paint_context_.submission_tracker.AwaitSubmissionCompletion(
                paint_context_.guest_output_image_paint_last_submission);
            util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                                       swapchain_effect_pipeline.swapchain_pipeline);
          }
          if (swapchain_effect_pipeline.swapchain_pipeline == VK_NULL_HANDLE) {
            assert_true(CanGuestOutputPaintEffectBeFinal(swapchain_effect));
            assert_true(guest_output_paint_fs_[size_t(swapchain_effect)] != VK_NULL_HANDLE);
            swapchain_effect_pipeline.swapchain_pipeline = CreateGuestOutputPaintPipeline(
                swapchain_effect, paint_context_.swapchain_render_pass);
            if (swapchain_effect_pipeline.swapchain_pipeline == VK_NULL_HANDLE) {
              guest_output_flow.effect_count = 0;
              if (prerot_turns) {
                // Log once, then the swapchain is recreated unrotated (the frame returns "suboptimal" below).
                if (!g_prerot_failed.exchange(true)) {
                  REXLOG_ERROR("[pace] pre-rotate: the rotating pipeline could not be built; falling back to an unrotated swapchain");
                }
                prerot_failed_now = true;
              }
            } else if (REXCVAR_GET(present_keep_paint_pipeline)) {  // wwe2k14 patch (Session 95, spike)
              swapchain_effect_pipeline.swapchain_format = paint_context_.swapchain_render_pass_format;
              static std::atomic<uint32_t> keep_builds{0};
              const uint32_t builds = ++keep_builds;
              if (builds == 1 || !(builds & (builds - 1))) {  // 1, 2, 4, 8...: a pipeline built again and again would show here
                REXLOG_INFO("[pace] present_keep_paint_pipeline on: final paint pipeline built {} time(s), format {}", builds,
                            int(paint_context_.swapchain_render_pass_format));
              }
            }
          }
        }
      }

      perf::timeline::Record("pm_draw", uint32_t(guest_output_flow.effect_count));  // wwe2k14 patch (Session 95, spike)
      if (guest_output_flow.effect_count) {
        // Actually draw the guest output.
        paint_context_.guest_output_image_paint_refs[guest_output_image_paint_ref_index].first =
            current_paint_submission_index;
        paint_context_.guest_output_image_paint_last_submission = current_paint_submission_index;
        VkViewport guest_output_viewport;
        guest_output_viewport.x = 0.0f;
        guest_output_viewport.y = 0.0f;
        guest_output_viewport.minDepth = 0.0f;
        guest_output_viewport.maxDepth = 1.0f;
        VkRect2D guest_output_scissor;
        guest_output_scissor.offset.x = 0;
        guest_output_scissor.offset.y = 0;
        if (guest_output_flow.effect_count > 1) {
          paint_context_.guest_output_intermediate_image_last_submission =
              current_paint_submission_index;
        }
        [[maybe_unused]] bool temporal_effect_selected = false;
#if defined(REX_HAS_FIDELITYFX_RUNTIME) && REX_HAS_FIDELITYFX_RUNTIME
        temporal_effect_selected =
            guest_output_paint_config.GetEffect() == GuestOutputPaintConfig::Effect::kFsr2 ||
            guest_output_paint_config.GetEffect() == GuestOutputPaintConfig::Effect::kFsr3;
#endif
        for (size_t i = 0; i < guest_output_flow.effect_count; ++i) {
          bool is_final_effect = i + 1 >= guest_output_flow.effect_count;
          GuestOutputPaintEffect effect = guest_output_flow.effects[i];

          std::pair<uint32_t, uint32_t> effect_rect_size = guest_output_flow.effect_output_sizes[i];
          if (is_final_effect && guest_output_flow.output_x == 0 &&
              guest_output_flow.output_y == 0 &&
              effect_rect_size.first < logical_width && effect_rect_size.second < logical_height) {
            static std::atomic<bool> logged_swapchain_expand = false;
            if (!logged_swapchain_expand.exchange(true)) {
              REXLOG_WARN(
                  "VulkanPresenter: expanding final guest output rect from "
                  "{}x{} to swapchain {}x{} to avoid top-left-only output",
                  effect_rect_size.first, effect_rect_size.second, logical_width, logical_height);
            }
            effect_rect_size = std::make_pair(logical_width, logical_height);
            // Keep all final-pass data coherent - shader constants and
            // letterbox clears are derived from guest_output_flow.
            guest_output_flow.effect_output_sizes[i] = effect_rect_size;
            guest_output_flow.output_x = 0;
            guest_output_flow.output_y = 0;
            guest_output_flow.letterbox_clear_rectangle_count = 0;
          }
#if defined(REX_HAS_FIDELITYFX_RUNTIME) && REX_HAS_FIDELITYFX_RUNTIME
          bool use_temporal_upscaler =
              temporal_effect_selected && effect == GuestOutputPaintEffect::kFsrEasu;
          if (use_temporal_upscaler && !is_final_effect) {
            VkImage source_image =
                i ? paint_context_.guest_output_intermediate_images[i - 1]->image()
                  : guest_output_image->image();
            VkImage output_image = paint_context_.guest_output_intermediate_images[i]->image();

            VkImageMemoryBarrier barrier_to_uav = {};
            barrier_to_uav.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier_to_uav.srcAccessMask = 0;
            barrier_to_uav.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier_to_uav.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier_to_uav.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier_to_uav.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier_to_uav.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier_to_uav.image = output_image;
            barrier_to_uav.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier_to_uav.subresourceRange.baseMipLevel = 0;
            barrier_to_uav.subresourceRange.levelCount = 1;
            barrier_to_uav.subresourceRange.baseArrayLayer = 0;
            barrier_to_uav.subresourceRange.layerCount = 1;
            dfn.vkCmdPipelineBarrier(draw_command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                                     nullptr, 1, &barrier_to_uav);

            uint32_t effect_input_width = 0, effect_input_height = 0;
            guest_output_flow.GetEffectInputSize(i, effect_input_width, effect_input_height);
            if (DispatchTemporalUpscaler(draw_command_buffer, source_image, effect_input_width,
                                         effect_input_height, output_image, effect_rect_size.first,
                                         effect_rect_size.second, guest_output_paint_config)) {
              VkImageMemoryBarrier barrier_to_shader_read = {};
              barrier_to_shader_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
              barrier_to_shader_read.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
              barrier_to_shader_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
              barrier_to_shader_read.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
              barrier_to_shader_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
              barrier_to_shader_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
              barrier_to_shader_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
              barrier_to_shader_read.image = output_image;
              barrier_to_shader_read.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
              barrier_to_shader_read.subresourceRange.baseMipLevel = 0;
              barrier_to_shader_read.subresourceRange.levelCount = 1;
              barrier_to_shader_read.subresourceRange.baseArrayLayer = 0;
              barrier_to_shader_read.subresourceRange.layerCount = 1;
              dfn.vkCmdPipelineBarrier(
                  draw_command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                  0, nullptr, 0, nullptr, 1, &barrier_to_shader_read);
              continue;
            }
          }
#endif

          int32_t effect_rect_x, effect_rect_y;
          if (is_final_effect) {
            effect_rect_x = guest_output_flow.output_x;
            effect_rect_y = guest_output_flow.output_y;
            perf::timeline::Record("pm_rp0");  // wwe2k14 patch (Session 95, spike)
            dfn.vkCmdBeginRenderPass(draw_command_buffer, &swapchain_render_pass_begin_info,
                                     VK_SUBPASS_CONTENTS_INLINE);
            perf::timeline::Record("pm_rp1");
            swapchain_image_pass_begun = true;
            guest_output_viewport.width = float(paint_context_.swapchain_extent.width);
            guest_output_viewport.height = float(paint_context_.swapchain_extent.height);
            guest_output_scissor.extent.width = paint_context_.swapchain_extent.width;
            guest_output_scissor.extent.height = paint_context_.swapchain_extent.height;
          } else {
            effect_rect_x = 0;
            effect_rect_y = 0;
            VkRenderPassBeginInfo intermediate_render_pass_begin_info;
            intermediate_render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            intermediate_render_pass_begin_info.pNext = nullptr;
            intermediate_render_pass_begin_info.renderPass = guest_output_intermediate_render_pass_;
            intermediate_render_pass_begin_info.framebuffer =
                paint_context_.guest_output_intermediate_framebuffers[i];
            intermediate_render_pass_begin_info.renderArea.offset.x = 0;
            intermediate_render_pass_begin_info.renderArea.offset.y = 0;
            intermediate_render_pass_begin_info.renderArea.extent.width = effect_rect_size.first;
            intermediate_render_pass_begin_info.renderArea.extent.height = effect_rect_size.second;
            intermediate_render_pass_begin_info.clearValueCount = 0;
            intermediate_render_pass_begin_info.pClearValues = nullptr;
            dfn.vkCmdBeginRenderPass(draw_command_buffer, &intermediate_render_pass_begin_info,
                                     VK_SUBPASS_CONTENTS_INLINE);
            guest_output_viewport.width = float(effect_rect_size.first);
            guest_output_viewport.height = float(effect_rect_size.second);
            guest_output_scissor.extent.width = effect_rect_size.first;
            guest_output_scissor.extent.height = effect_rect_size.second;
          }
          dfn.vkCmdSetViewport(draw_command_buffer, 0, 1, &guest_output_viewport);
          dfn.vkCmdSetScissor(draw_command_buffer, 0, 1, &guest_output_scissor);

          const PaintContext::GuestOutputPaintPipeline& effect_pipeline =
              paint_context_.guest_output_paint_pipelines[size_t(effect)];
          VkPipeline effect_vulkan_pipeline = is_final_effect
                                                  ? effect_pipeline.swapchain_pipeline
                                                  : effect_pipeline.intermediate_pipeline;
          assert_true(effect_vulkan_pipeline != VK_NULL_HANDLE);
          dfn.vkCmdBindPipeline(draw_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                effect_vulkan_pipeline);

          GuestOutputPaintPipelineLayoutIndex guest_output_paint_pipeline_layout_index =
              GetGuestOutputPaintPipelineLayoutIndex(effect);
          VkPipelineLayout guest_output_paint_pipeline_layout =
              guest_output_paint_pipeline_layouts_[guest_output_paint_pipeline_layout_index];

          PaintContext::GuestOutputDescriptorSet effect_src_descriptor_set;
          if (i) {
            effect_src_descriptor_set = PaintContext::GuestOutputDescriptorSet(
                PaintContext::kGuestOutputDescriptorSetIntermediate0Sampled + (i - 1));
          } else {
            effect_src_descriptor_set = PaintContext::GuestOutputDescriptorSet(
                PaintContext::kGuestOutputDescriptorSetGuestOutput0Sampled +
                guest_output_image_paint_ref_index);
          }
          dfn.vkCmdBindDescriptorSets(
              draw_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
              guest_output_paint_pipeline_layout, 0, 1,
              &paint_context_.guest_output_descriptor_sets[effect_src_descriptor_set], 0, nullptr);

          GuestOutputPaintRectangleConstants effect_rect_constants;
          // The rectangle is in the logical viewport's NDC (the physical one is swapped when pre-rotating).
          const bool effect_prerot = is_final_effect && prerot_turns != 0;
          float effect_x_to_ndc = 2.0f / (effect_prerot ? float(logical_width) : guest_output_viewport.width);
          float effect_y_to_ndc = 2.0f / (effect_prerot ? float(logical_height) : guest_output_viewport.height);
          effect_rect_constants.x = -1.0f + float(effect_rect_x) * effect_x_to_ndc;
          effect_rect_constants.y = -1.0f + float(effect_rect_y) * effect_y_to_ndc;
          effect_rect_constants.width = float(effect_rect_size.first) * effect_x_to_ndc;
          effect_rect_constants.height = float(effect_rect_size.second) * effect_y_to_ndc;
          dfn.vkCmdPushConstants(draw_command_buffer, guest_output_paint_pipeline_layout,
                                 VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(effect_rect_constants),
                                 &effect_rect_constants);

          uint32_t effect_constants_size = 0;
          union {
            BilinearConstants bilinear;
            SgsrConstants sgsr;
#if defined(REX_HAS_FIDELITYFX_SDK)
            CasSharpenConstants cas_sharpen;
            CasResampleConstants cas_resample;
            FsrEasuConstants fsr_easu;
            FsrRcasConstants fsr_rcas;
#endif
          } effect_constants;
          switch (guest_output_paint_pipeline_layout_index) {
            case kGuestOutputPaintPipelineLayoutIndexBilinear: {
              effect_constants_size = sizeof(effect_constants.bilinear);
              effect_constants.bilinear.Initialize(guest_output_flow, i);
            } break;
            case kGuestOutputPaintPipelineLayoutIndexSgsr: {
              effect_constants_size = sizeof(effect_constants.sgsr);
              effect_constants.sgsr.Initialize(guest_output_flow, i, guest_output_paint_config);
            } break;
#if defined(REX_HAS_FIDELITYFX_SDK)
            case kGuestOutputPaintPipelineLayoutIndexCasSharpen: {
              effect_constants_size = sizeof(effect_constants.cas_sharpen);
              effect_constants.cas_sharpen.Initialize(guest_output_flow, i,
                                                      guest_output_paint_config);
            } break;
            case kGuestOutputPaintPipelineLayoutIndexCasResample: {
              effect_constants_size = sizeof(effect_constants.cas_resample);
              effect_constants.cas_resample.Initialize(guest_output_flow, i,
                                                       guest_output_paint_config);
            } break;
            case kGuestOutputPaintPipelineLayoutIndexFsrEasu: {
              effect_constants_size = sizeof(effect_constants.fsr_easu);
              effect_constants.fsr_easu.Initialize(guest_output_flow, i);
            } break;
            case kGuestOutputPaintPipelineLayoutIndexFsrRcas: {
              effect_constants_size = sizeof(effect_constants.fsr_rcas);
              effect_constants.fsr_rcas.Initialize(guest_output_flow, i, guest_output_paint_config);
            } break;
#endif
            default:
              break;
          }
          if (effect_constants_size) {
            dfn.vkCmdPushConstants(draw_command_buffer, guest_output_paint_pipeline_layout,
                                   VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(effect_rect_constants),
                                   effect_constants_size, &effect_constants);
          }

          dfn.vkCmdDraw(draw_command_buffer, 4, 1, 0, 0);
          perf::timeline::Record("pm_dr");  // wwe2k14 patch (Session 95, spike)

          if (is_final_effect) {
            // Clear the letterbox around the guest output if the guest output
            // doesn't cover the entire swapchain image.
            if (swapchain_image_clear_needed && guest_output_flow.letterbox_clear_rectangle_count) {
              VkClearRect
                  letterbox_clear_vulkan_rectangles[GuestOutputPaintFlow::kMaxClearRectangles];
              for (size_t i = 0; i < guest_output_flow.letterbox_clear_rectangle_count; ++i) {
                VkClearRect& letterbox_clear_vulkan_rectangle =
                    letterbox_clear_vulkan_rectangles[i];
                const GuestOutputPaintFlow::ClearRectangle& letterbox_clear_rectangle =
                    guest_output_flow.letterbox_clear_rectangles[i];
                // Logical rectangle to image space (same mapping as the vertex shader's).
                int32_t clear_x = int32_t(letterbox_clear_rectangle.x);
                int32_t clear_y = int32_t(letterbox_clear_rectangle.y);
                int32_t clear_w = int32_t(letterbox_clear_rectangle.width);
                int32_t clear_h = int32_t(letterbox_clear_rectangle.height);
                if (prerot_turns) {
                  const int32_t logical_w = int32_t(logical_width);
                  const int32_t logical_h = int32_t(logical_height);
                  if (prerot_map == 1) {
                    const int32_t x = clear_x, w = clear_w;
                    clear_x = clear_y;
                    clear_y = logical_w - (x + w);
                    std::swap(clear_w, clear_h);
                  } else if (prerot_map == 2) {
                    clear_x = logical_w - (clear_x + clear_w);
                    clear_y = logical_h - (clear_y + clear_h);
                  } else {
                    const int32_t y = clear_y, h = clear_h;
                    clear_y = clear_x;
                    clear_x = logical_h - (y + h);
                    std::swap(clear_w, clear_h);
                  }
                }
                letterbox_clear_vulkan_rectangle.rect.offset.x = clear_x;
                letterbox_clear_vulkan_rectangle.rect.offset.y = clear_y;
                letterbox_clear_vulkan_rectangle.rect.extent.width = uint32_t(clear_w);
                letterbox_clear_vulkan_rectangle.rect.extent.height = uint32_t(clear_h);
                letterbox_clear_vulkan_rectangle.baseArrayLayer = 0;
                letterbox_clear_vulkan_rectangle.layerCount = 1;
              }
              dfn.vkCmdClearAttachments(draw_command_buffer, 1, &swapchain_image_clear_attachment,
                                        uint32_t(guest_output_flow.letterbox_clear_rectangle_count),
                                        letterbox_clear_vulkan_rectangles);
            }
            swapchain_image_clear_needed = false;
          } else {
            // Still need the swapchain pass to be open for UI drawing.
            dfn.vkCmdEndRenderPass(draw_command_buffer);
          }
        }
      }
    }
  }

  perf::timeline::Record("pm_drawn");  // wwe2k14 patch (Session 95, spike)
  // Release main target guest output image references that aren't needed
  // anymore (this is done after various potential guest-output-related main
  // target submission tracker waits so the completed submission value is the
  // most actual).
  uint64_t completed_paint_submission =
      paint_context_.submission_tracker.UpdateAndGetCompletedSubmission();
  for (std::pair<uint64_t, std::shared_ptr<GuestOutputImage>>& guest_output_image_paint_ref :
       paint_context_.guest_output_image_paint_refs) {
    if (!guest_output_image_paint_ref.second ||
        guest_output_image_paint_ref.second == guest_output_image) {
      continue;
    }
    if (completed_paint_submission >= guest_output_image_paint_ref.first) {
      guest_output_image_paint_ref.second.reset();
    }
  }

  perf::timeline::Record("pm_rel");  // wwe2k14 patch (Session 95, spike)
  // If hasn't presented the guest output, begin the pass to clear and, if
  // needed, to draw the UI.
  if (!swapchain_image_pass_begun) {
    dfn.vkCmdBeginRenderPass(draw_command_buffer, &swapchain_render_pass_begin_info,
                             VK_SUBPASS_CONTENTS_INLINE);
  }
  if (swapchain_image_clear_needed) {
    VkClearRect swapchain_image_clear_rectangle;
    swapchain_image_clear_rectangle.rect.offset.x = 0;
    swapchain_image_clear_rectangle.rect.offset.y = 0;
    swapchain_image_clear_rectangle.rect.extent = paint_context_.swapchain_extent;
    swapchain_image_clear_rectangle.baseArrayLayer = 0;
    swapchain_image_clear_rectangle.layerCount = 1;
    dfn.vkCmdClearAttachments(draw_command_buffer, 1, &swapchain_image_clear_attachment, 1,
                              &swapchain_image_clear_rectangle);
    swapchain_image_clear_needed = false;
  }

  if (execute_ui_drawers) {
    // Draw the UI (in the logical orientation: the immediate drawer rotates it into the image, see GetPrerotMapForUi).
    VulkanUIDrawContext ui_draw_context(
        *this, logical_width, logical_height,
        draw_command_buffer, ui_submission_tracker_.GetCurrentSubmission(),
        ui_submission_tracker_.UpdateAndGetCompletedSubmission(),
        paint_context_.swapchain_render_pass, paint_context_.swapchain_render_pass_format);
    perf::timeline::Record("pm_ui0");  // wwe2k14 patch (Session 95, spike)
    ExecuteUIDrawersFromUIThread(ui_draw_context);
    perf::timeline::Record("pm_ui1");
  }

  perf::timeline::Record("pm_endrp");  // wwe2k14 patch (Session 95, spike)
  dfn.vkCmdEndRenderPass(draw_command_buffer);

  if (paint_timestamp_slot != UINT32_MAX) {  // wwe2k14 patch (Session 95, item T)
    dfn.vkCmdWriteTimestamp(draw_command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_paint_timestamps.pool,
                            paint_timestamp_slot * 2 + 1);
  }

  perf::timeline::Record("pm_endcb");  // wwe2k14 patch (Session 95, spike)
  dfn.vkEndCommandBuffer(draw_command_buffer);
  perf::timeline::Record("prec");  // wwe2k14 patch (Session 95)

  VkPipelineStageFlags acquire_semaphore_wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkCommandBuffer command_buffers[2];
  uint32_t command_buffer_count = 0;
  // UI setup command buffers must be accessed only if execute_ui_drawers is not
  // null, to identify the UI thread.
  size_t ui_setup_command_buffer_index =
      execute_ui_drawers ? paint_context_.ui_setup_command_buffer_current_index : SIZE_MAX;
  if (ui_setup_command_buffer_index != SIZE_MAX) {
    PaintContext::UISetupCommandBuffer& ui_setup_command_buffer =
        paint_context_.ui_setup_command_buffers[ui_setup_command_buffer_index];
    dfn.vkEndCommandBuffer(ui_setup_command_buffer.command_buffer);
    command_buffers[command_buffer_count++] = ui_setup_command_buffer.command_buffer;
    // Release the current UI setup command buffer regardless of submission
    // result. Failed submissions (if the UI submission index wasn't incremented
    // since the previous draw) should be handled by UI drawers themselves by
    // retrying all the failed work if needed.
    paint_context_.ui_setup_command_buffer_current_index = SIZE_MAX;
  }
  command_buffers[command_buffer_count++] = draw_command_buffer;
  VkSemaphore present_semaphore = paint_submission.present_semaphore();
  // wwe2k14 patch (Session 73): one per swapchain image (PaintContext::image_present_semaphores).
  while (paint_context_.image_present_semaphores.size() <= swapchain_image_index) {
    VkSemaphoreCreateInfo semaphore_create_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore semaphore = VK_NULL_HANDLE;
    if (dfn.vkCreateSemaphore(vulkan_device_->device(), &semaphore_create_info, nullptr, &semaphore) != VK_SUCCESS) {
      break;
    }
    paint_context_.image_present_semaphores.push_back(semaphore);
  }
  if (swapchain_image_index < paint_context_.image_present_semaphores.size()) {
    present_semaphore = paint_context_.image_present_semaphores[swapchain_image_index];
  }
  VkSubmitInfo submit_info;
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.pNext = nullptr;
  submit_info.waitSemaphoreCount = acquire_semaphore != VK_NULL_HANDLE ? 1 : 0;
  submit_info.pWaitSemaphores = &acquire_semaphore;
  submit_info.pWaitDstStageMask = &acquire_semaphore_wait_stage;
  submit_info.commandBufferCount = command_buffer_count;
  submit_info.pCommandBuffers = command_buffers;
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = &present_semaphore;
  // wwe2k14 patch (Session 6, DRAIN): reading the guest output image relies on queue submission order to come
  // after the refresher's writes - wait until its submission is on the queue (not with the queue acquired).
  if (guest_output_deferred_submission_order) {
    guest_output_deferred_submission_order->AwaitSubmitted(guest_output_deferred_submission_ticket);
  }
  {
    VulkanSubmissionTracker::FenceAcquisition fence_acqusition(
        paint_context_.submission_tracker.AcquireFenceToAdvanceSubmission());
    // Also update the submission tracker giving submission indices to UI draw
    // callbacks if submission is successful.
    VulkanSubmissionTracker::FenceAcquisition ui_fence_acquisition;
    if (execute_ui_drawers) {
      ui_fence_acquisition = ui_submission_tracker_.AcquireFenceToAdvanceSubmission();
    }
    VkResult submit_result;
    perf::timeline::Record("pfa");  // wwe2k14 patch (Session 95): the fences are in hand
    {
      const VulkanDevice::Queue::Acquisition queue_acquisition =
          vulkan_device_->AcquireQueue(vulkan_device_->queue_family_graphics_compute(), 0);
      submit_result =
          dfn.vkQueueSubmit(queue_acquisition.queue(), 1, &submit_info, fence_acqusition.fence());
      if (ui_fence_acquisition.fence() != VK_NULL_HANDLE && submit_result == VK_SUCCESS) {
        if (dfn.vkQueueSubmit(queue_acquisition.queue(), 0, nullptr,
                              ui_fence_acquisition.fence()) != VK_SUCCESS) {
          ui_fence_acquisition.SubmissionSucceededSignalFailed();
        }
      }
    }
    if (acquire_fence_mode) {  // wwe2k14 patch (Session 95): the image is used up (or dropped with the swapchain)
      g_acquire_fence.pending = false;
      g_acquire_fence.ready = false;
    }
    if (submit_result != VK_SUCCESS) {
      REXLOG_ERROR("VulkanPresenter: Failed to submit command buffers");
      fence_acqusition.SubmissionFailedOrDropped();
      ui_fence_acquisition.SubmissionFailedOrDropped();
      if (ui_setup_command_buffer_index != SIZE_MAX) {
        // If failed to submit, make the UI setup command buffer available for
        // immediate reuse, as the completed submission index won't be updated
        // to the current index, and failing submissions with setup command
        // buffer over and over will result in never reusing the setup command
        // buffers.
        paint_context_.ui_setup_command_buffers[ui_setup_command_buffer_index]
            .last_usage_submission_index = 0;
      }
      // The image is in an acquired state - but now, it will be in it forever.
      // To avoid that, recreate the swapchain - don't return just
      // kNotPresented.
      return PaintResult::kNotPresentedConnectionOutdated;
    }
    if (paint_timestamp_slot != UINT32_MAX) {  // wwe2k14 patch (Session 95, item T)
      g_paint_timestamps.submitted[paint_timestamp_slot] = true;
    }
  }

  VkPresentInfoKHR present_info;
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.pNext = nullptr;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = &present_semaphore;
  present_info.swapchainCount = 1;
  present_info.pSwapchains = &paint_context_.swapchain;
  present_info.pImageIndices = &swapchain_image_index;
  present_info.pResults = nullptr;
  // wwe2k14 patch (Session 93): VK_GOOGLE_display_timing, only with present_display_timing and a device that has it.
  VkPresentTimeGOOGLE display_timing_time;
  VkPresentTimesInfoGOOGLE display_timing_times;
  if (REXCVAR_GET(present_display_timing)) {
    DisplayTimingPrepare(present_info, display_timing_time, display_timing_times);
  }
  VkResult present_result;
  perf::timeline::Record("psub");  // wwe2k14 patch (Session 95): the paint's vkQueueSubmit has returned
  // wwe2k14 patch (Session 93, log only): as the acquire above; the queue acquisition (shared with the submit thread) is inside the timing.
  Presenter::NoteHostPresentCall("vkQueuePresentKHR");
  const auto present_t0 = std::chrono::steady_clock::now();
  {
    const VulkanDevice::Queue::Acquisition queue_acquisition =
        vulkan_device_->AcquireQueue(paint_context_.present_queue_family, 0);
    perf::timeline::Record("ppres");  // wwe2k14 patch (Session 95): the queue is ours, the present call starts
    present_result = dfn.vkQueuePresentKHR(queue_acquisition.queue(), &present_info);
  }
  // wwe2k14 patch (Session 95, HANG): a picture really went out.
  if (present_result == VK_SUCCESS || present_result == VK_SUBOPTIMAL_KHR) {
    hang_beats.presents.fetch_add(1, std::memory_order_relaxed);
  }
  Presenter::NoteHostPresentCall(nullptr);
  SlowHostPresentCallNote("vkQueuePresentKHR", present_t0, int(present_result), paint_context_.swapchain_is_fifo,
                          present_info.pNext == &display_timing_times ? display_timing_time.desiredPresentTime : 0);
  // wwe2k14 patch (Session 19, LATENCY): a = refresh sequence number of the guest image presented, b = result.
  perf::timeline::Record("hpresent", uint32_t(guest_output_properties.refresh_seq), uint32_t(present_result));
  // wwe2k14 patch (Session 21, ONCE): the guest frame now on screen (present_guest_frames_once).
  NotePresentedGuestFrame(guest_output_properties.refresh_seq);
  static uint64_t last_counted_refresh_seq = UINT64_MAX;
  if ((present_result == VK_SUCCESS || present_result == VK_SUBOPTIMAL_KHR) && guest_output_image &&
      guest_output_properties.refresh_seq != last_counted_refresh_seq) {
    last_counted_refresh_seq = guest_output_properties.refresh_seq;
    // wwe2k14 patch (Session 95): the frames that reached a present, next to the swaps (frames_*.csv: presented,
    // present_gap_ms). Only the thread that paints writes these (one at a time).
    static std::chrono::steady_clock::time_point last_present;
    const auto present_now = std::chrono::steady_clock::now();
    auto& frame_events = rex::perf::frame_events::Get();
    if (last_present != std::chrono::steady_clock::time_point{}) {
      frame_events.present_gap_us.store(
          uint32_t(std::chrono::duration_cast<std::chrono::microseconds>(present_now - last_present).count()),
          std::memory_order_relaxed);
    }
    last_present = present_now;
    frame_events.presents.fetch_add(1, std::memory_order_release);
  }
  switch (present_result) {
    case VK_SUCCESS:
      return (prerot_failed_now || (g_prerot_failed.load() && prerot_turns)) ? PaintResult::kPresentedSuboptimal
                                                                              : PaintResult::kPresented;
    case VK_SUBOPTIMAL_KHR:
      return PaintResult::kPresentedSuboptimal;
    case VK_ERROR_DEVICE_LOST:
      REXLOG_ERROR(
          "VulkanPresenter: Failed to present the swapchain image as the "
          "device has been lost");
      return PaintResult::kGpuLostResponsible;
    case VK_ERROR_OUT_OF_DATE_KHR:
    case VK_ERROR_SURFACE_LOST_KHR:
    case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
      // Not an error, reporting just as info (may normally occur while resizing
      // on some platforms).
      REXLOG_INFO(
          "VulkanPresenter: Presentation to the swapchain image has been "
          "dropped as the swapchain or the surface has become outdated (present)");
      // Note that the semaphore wait (followed by reset) has been enqueued,
      // however, this should have no effect on anything here likely.
      return PaintResult::kNotPresentedConnectionOutdated;
    default:
      REXLOG_ERROR("VulkanPresenter: Failed to present the swapchain image");
      // The image is in an acquired state - but now, it will be in it forever.
      // To avoid that, recreate the swapchain - don't return just
      // kNotPresented.
      return PaintResult::kNotPresentedConnectionOutdated;
  }
}

// wwe2k14 patch (Session 93, idea 14): the present's desired time, as Android's Frame Pacing library does it. The first
// present (and any after a stall) is unpaced; each later one asks for "the newest known actual present time + N refresh
// cycles - half a cycle" (the presentation engine shows it at the first vsync not before that), N = the display's refresh
// rate / the game's frame rate. Actual times come back a few frames late (vkGetPastPresentationTimingGOOGLE): between them
// the cycles counter carries the anchor forward.
bool VulkanPresenter::DisplayTimingPrepare(VkPresentInfoKHR& present_info, VkPresentTimeGOOGLE& time,
                                           VkPresentTimesInfoGOOGLE& times) {
  DisplayTiming& dt = display_timing_;
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  const VkSwapchainKHR swapchain = paint_context_.swapchain;
  if (!vulkan_device_->extensions().ext_GOOGLE_display_timing || swapchain == VK_NULL_HANDLE) {
    if (!dt.logged_start) {
      dt.logged_start = true;
      REXLOG_INFO("[pace] display timing: ext present=no (VK_GOOGLE_display_timing missing on this device): presents unchanged");
    }
    return false;
  }
  const auto now_ns = []() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count());
  };
  const uint64_t query_now = now_ns();
  if (dt.swapchain != swapchain || query_now >= dt.refresh_query_ns) {
    // A new swapchain, or about once a second (the panel's rate can change underneath: ANativeWindow_setFrameRate): the
    // refresh duration again; a change re-anchors. The old swapchain's presents' times are not this one's.
    const bool new_swapchain = dt.swapchain != swapchain;
    dt.swapchain = swapchain;
    dt.refresh_query_ns = query_now + 1000000000ull;
    uint64_t refresh_ns = 0;
    VkRefreshCycleDurationGOOGLE refresh;
    if (dfn.vkGetRefreshCycleDurationGOOGLE(device, swapchain, &refresh) == VK_SUCCESS && refresh.refreshDuration) {
      refresh_ns = refresh.refreshDuration;
    }
    if (new_swapchain || refresh_ns != dt.refresh_ns) {
      dt.anchor_id = 0;
      dt.anchor_min_id = dt.next_id;
      dt.refresh_ns = refresh_ns;
      dt.logged_cycles = -1;
    }
  }
  if (!dt.refresh_ns) {
    if (!dt.logged_start) {
      dt.logged_start = true;
      REXLOG_INFO("[pace] display timing: ext present=yes but the refresh duration query failed: presents unchanged");
    }
    return false;
  }
  const double refresh_ms = double(dt.refresh_ns) / 1e6;
  // Past presents' actual times.
  for (int rounds = 0; rounds < 4; ++rounds) {
    VkPastPresentationTimingGOOGLE past[16];
    uint32_t count = uint32_t(rex::countof(past));
    const VkResult past_result = dfn.vkGetPastPresentationTimingGOOGLE(device, swapchain, &count, past);
    if (past_result != VK_SUCCESS && past_result != VK_INCOMPLETE) {
      break;
    }
    for (uint32_t i = 0; i < count; ++i) {
      if (!past[i].actualPresentTime) {
        continue;
      }
      const DisplayTiming::Slot& slot = dt.ring[past[i].presentID % rex::countof(dt.ring)];
      if (slot.id != past[i].presentID) {
        continue;
      }
      if (slot.ideal_ns) {
        const double error_ms = (double(past[i].actualPresentTime) - double(slot.ideal_ns)) / 1e6;
        ++dt.window_paced;
        dt.window_error_ms += error_ms;
        if (error_ms > refresh_ms * 0.5) {
          ++dt.window_late;
        }
      }
      if (past[i].presentID >= dt.anchor_min_id && past[i].presentID > dt.anchor_id) {
        dt.anchor_id = past[i].presentID;
        dt.anchor_actual_ns = past[i].actualPresentTime;
        dt.anchor_cum = slot.cum;
      }
    }
    if (past_result != VK_INCOMPLETE) {
      break;
    }
  }  // N, from the display's rate and the game's.
  int32_t fps = REXCVAR_GET(present_target_fps);
  if (fps < 10 || fps > 240) {
    fps = 60;
  }
  const double ratio = (1e9 / double(dt.refresh_ns)) / double(fps);
  int cycles = int(std::lround(ratio));
  if (cycles < 1 || cycles > 16 || std::fabs(ratio - double(cycles)) > 0.1) {
    cycles = 0;  // 90 Hz for 60 fps is 1.5 cycles: no whole number, so no pacing
  }
  if (cycles != dt.logged_cycles) {
    dt.logged_cycles = cycles;
    dt.logged_start = true;
    if (cycles) {
      REXLOG_INFO("[pace] display timing: ext present=yes refresh={:.2f} ms cycles per frame={} (game {} fps)", refresh_ms,
                  cycles, fps);
    } else {
      REXLOG_INFO("[pace] display timing: ext present=yes refresh={:.2f} ms cycles per frame=none (game {} fps is not a "
                  "whole number of cycles): presents unpaced", refresh_ms, fps);
    }
  }
  const uint32_t id = dt.next_id++;
  uint64_t ideal = 0, desired = 0;
  dt.cum += uint64_t(cycles);
  if (cycles && dt.anchor_id) {
    ideal = dt.anchor_actual_ns + (dt.cum - dt.anchor_cum) * dt.refresh_ns;
    const uint64_t now = now_ns();
    if (ideal + 2 * dt.refresh_ns < now || ideal > now + uint64_t(cycles) * 8 * dt.refresh_ns) {
      // A stall (a load, a pause) or a drifted anchor: present at once and take the next actual time as the anchor.
      ideal = 0;
      dt.anchor_id = 0;
      dt.anchor_min_id = id;
      ++dt.window_resets;
    } else {
      desired = ideal - dt.refresh_ns / 2;
    }
  }
  DisplayTiming::Slot& slot = dt.ring[id % rex::countof(dt.ring)];
  slot.id = id;
  slot.ideal_ns = ideal;
  slot.cum = dt.cum;
  time.presentID = id;
  time.desiredPresentTime = desired;
  times.sType = VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE;
  times.pNext = nullptr;
  times.swapchainCount = 1;
  times.pTimes = &time;
  present_info.pNext = &times;
  ++dt.total_presents;
  if (++dt.window_presents >= 600) {
    REXLOG_INFO("[pace] display timing: presents {} (total {}), paced {}, late {}, resets {}, mean present error {:+.2f} ms",
                dt.window_presents, dt.total_presents, dt.window_paced, dt.window_late, dt.window_resets,
                dt.window_paced ? dt.window_error_ms / double(dt.window_paced) : 0.0);
    dt.window_presents = dt.window_paced = dt.window_late = dt.window_resets = 0;
    dt.window_error_ms = 0.0;
  }
  return true;
}

bool VulkanPresenter::InitializeSurfaceIndependent() {
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  VkDescriptorSetLayoutBinding guest_output_image_sampler_bindings[2];
  guest_output_image_sampler_bindings[0].binding = 0;
  guest_output_image_sampler_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  guest_output_image_sampler_bindings[0].descriptorCount = 1;
  guest_output_image_sampler_bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  guest_output_image_sampler_bindings[0].pImmutableSamplers = nullptr;
  const VkSampler sampler_linear_clamp =
      ui_samplers_->samplers()[UISamplers::kSamplerIndexLinearClampToEdge];
  guest_output_image_sampler_bindings[1].binding = 1;
  guest_output_image_sampler_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
  guest_output_image_sampler_bindings[1].descriptorCount = 1;
  guest_output_image_sampler_bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  guest_output_image_sampler_bindings[1].pImmutableSamplers = &sampler_linear_clamp;
  VkDescriptorSetLayoutCreateInfo guest_output_paint_image_descriptor_set_layout_create_info;
  guest_output_paint_image_descriptor_set_layout_create_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  guest_output_paint_image_descriptor_set_layout_create_info.pNext = nullptr;
  guest_output_paint_image_descriptor_set_layout_create_info.flags = 0;
  guest_output_paint_image_descriptor_set_layout_create_info.bindingCount =
      uint32_t(rex::countof(guest_output_image_sampler_bindings));
  guest_output_paint_image_descriptor_set_layout_create_info.pBindings =
      guest_output_image_sampler_bindings;
  if (dfn.vkCreateDescriptorSetLayout(
          device, &guest_output_paint_image_descriptor_set_layout_create_info, nullptr,
          &guest_output_paint_image_descriptor_set_layout_) != VK_SUCCESS) {
    REXLOG_ERROR(
        "VulkanPresenter: Failed to create the guest output image descriptor "
        "set layout");
    return false;
  }

  VkPushConstantRange guest_output_paint_push_constant_ranges[2];
  VkPushConstantRange& guest_output_paint_push_constant_range_rect =
      guest_output_paint_push_constant_ranges[0];
  guest_output_paint_push_constant_range_rect.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  guest_output_paint_push_constant_range_rect.offset = 0;
  guest_output_paint_push_constant_range_rect.size = sizeof(GuestOutputPaintRectangleConstants);
  VkPushConstantRange& guest_output_paint_push_constant_range_ffx =
      guest_output_paint_push_constant_ranges[1];
  guest_output_paint_push_constant_range_ffx.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  guest_output_paint_push_constant_range_ffx.offset =
      guest_output_paint_push_constant_ranges[0].offset +
      guest_output_paint_push_constant_ranges[0].size;
  VkPipelineLayoutCreateInfo guest_output_paint_pipeline_layout_create_info;
  guest_output_paint_pipeline_layout_create_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  guest_output_paint_pipeline_layout_create_info.pNext = nullptr;
  guest_output_paint_pipeline_layout_create_info.flags = 0;
  guest_output_paint_pipeline_layout_create_info.setLayoutCount = 1;
  guest_output_paint_pipeline_layout_create_info.pSetLayouts =
      &guest_output_paint_image_descriptor_set_layout_;
  guest_output_paint_pipeline_layout_create_info.pPushConstantRanges =
      guest_output_paint_push_constant_ranges;
  for (size_t i = 0; i < size_t(kGuestOutputPaintPipelineLayoutCount); ++i) {
    switch (GuestOutputPaintPipelineLayoutIndex(i)) {
      case kGuestOutputPaintPipelineLayoutIndexBilinear:
        guest_output_paint_push_constant_range_ffx.size = sizeof(BilinearConstants);
        break;
      case kGuestOutputPaintPipelineLayoutIndexSgsr:
        guest_output_paint_push_constant_range_ffx.size = sizeof(SgsrConstants);
        break;
#if defined(REX_HAS_FIDELITYFX_SDK)
      case kGuestOutputPaintPipelineLayoutIndexCasSharpen:
        guest_output_paint_push_constant_range_ffx.size = sizeof(CasSharpenConstants);
        break;
      case kGuestOutputPaintPipelineLayoutIndexCasResample:
        guest_output_paint_push_constant_range_ffx.size = sizeof(CasResampleConstants);
        break;
      case kGuestOutputPaintPipelineLayoutIndexFsrEasu:
        guest_output_paint_push_constant_range_ffx.size = sizeof(FsrEasuConstants);
        break;
      case kGuestOutputPaintPipelineLayoutIndexFsrRcas:
        guest_output_paint_push_constant_range_ffx.size = sizeof(FsrRcasConstants);
        break;
#endif
      default:
        assert_unhandled_case(GuestOutputPaintPipelineLayoutIndex(i));
        continue;
    }
    guest_output_paint_pipeline_layout_create_info.pushConstantRangeCount =
        1 + uint32_t(guest_output_paint_push_constant_range_ffx.size != 0);
    if (dfn.vkCreatePipelineLayout(device, &guest_output_paint_pipeline_layout_create_info, nullptr,
                                   &guest_output_paint_pipeline_layouts_[i]) != VK_SUCCESS) {
      REXLOG_ERROR(
          "VulkanPresenter: Failed to create a guest output presentation "
          "pipeline layout with {} bytes of push constants",
          guest_output_paint_push_constant_range_rect.size +
              guest_output_paint_push_constant_range_ffx.size);
      return false;
    }
  }

  VkShaderModuleCreateInfo shader_module_create_info;
  shader_module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_module_create_info.pNext = nullptr;
  shader_module_create_info.flags = 0;
  shader_module_create_info.codeSize = sizeof(shaders::guest_output_triangle_strip_rect_vs);
  shader_module_create_info.pCode = shaders::guest_output_triangle_strip_rect_vs;
  if (dfn.vkCreateShaderModule(device, &shader_module_create_info, nullptr,
                               &guest_output_paint_vs_) != VK_SUCCESS) {
    REXLOG_ERROR(
        "VulkanPresenter: Failed to create the guest output presentation "
        "vertex shader module");
    return false;
  }
  for (size_t i = 0; i < size_t(GuestOutputPaintEffect::kCount); ++i) {
    GuestOutputPaintEffect guest_output_paint_effect = GuestOutputPaintEffect(i);
    switch (guest_output_paint_effect) {
      case GuestOutputPaintEffect::kBilinear:
        shader_module_create_info.codeSize = sizeof(shaders::guest_output_bilinear_ps);
        shader_module_create_info.pCode = shaders::guest_output_bilinear_ps;
        break;
      case GuestOutputPaintEffect::kBilinearDither:
        shader_module_create_info.codeSize = sizeof(shaders::guest_output_bilinear_dither_ps);
        shader_module_create_info.pCode = shaders::guest_output_bilinear_dither_ps;
        break;
      case GuestOutputPaintEffect::kSgsr:
        shader_module_create_info.codeSize = sizeof(shaders::guest_output_sgsr_ps);
        shader_module_create_info.pCode = shaders::guest_output_sgsr_ps;
        break;
#if defined(REX_HAS_FIDELITYFX_SDK)
      case GuestOutputPaintEffect::kCasSharpen:
        shader_module_create_info.codeSize = sizeof(shaders::guest_output_ffx_cas_sharpen_ps);
        shader_module_create_info.pCode = shaders::guest_output_ffx_cas_sharpen_ps;
        break;
      case GuestOutputPaintEffect::kCasSharpenDither:
        shader_module_create_info.codeSize =
            sizeof(shaders::guest_output_ffx_cas_sharpen_dither_ps);
        shader_module_create_info.pCode = shaders::guest_output_ffx_cas_sharpen_dither_ps;
        break;
      case GuestOutputPaintEffect::kCasResample:
        shader_module_create_info.codeSize = sizeof(shaders::guest_output_ffx_cas_resample_ps);
        shader_module_create_info.pCode = shaders::guest_output_ffx_cas_resample_ps;
        break;
      case GuestOutputPaintEffect::kCasResampleDither:
        shader_module_create_info.codeSize =
            sizeof(shaders::guest_output_ffx_cas_resample_dither_ps);
        shader_module_create_info.pCode = shaders::guest_output_ffx_cas_resample_dither_ps;
        break;
      case GuestOutputPaintEffect::kFsrEasu:
        shader_module_create_info.codeSize = sizeof(shaders::guest_output_ffx_fsr_easu_ps);
        shader_module_create_info.pCode = shaders::guest_output_ffx_fsr_easu_ps;
        break;
      case GuestOutputPaintEffect::kFsrRcas:
        shader_module_create_info.codeSize = sizeof(shaders::guest_output_ffx_fsr_rcas_ps);
        shader_module_create_info.pCode = shaders::guest_output_ffx_fsr_rcas_ps;
        break;
      case GuestOutputPaintEffect::kFsrRcasDither:
        shader_module_create_info.codeSize = sizeof(shaders::guest_output_ffx_fsr_rcas_dither_ps);
        shader_module_create_info.pCode = shaders::guest_output_ffx_fsr_rcas_dither_ps;
        break;
#endif
      default:
        // Not supported by this implementation.
        continue;
    }
    if (dfn.vkCreateShaderModule(device, &shader_module_create_info, nullptr,
                                 &guest_output_paint_fs_[i]) != VK_SUCCESS) {
      REXLOG_ERROR(
          "VulkanPresenter: Failed to create the guest output painting shader "
          "module for effect {}",
          i);
      return false;
    }
  }

  VkAttachmentDescription intermediate_render_pass_attachment;
  intermediate_render_pass_attachment.flags = 0;
  intermediate_render_pass_attachment.format = kGuestOutputFormat;
  intermediate_render_pass_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  intermediate_render_pass_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  intermediate_render_pass_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  intermediate_render_pass_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  intermediate_render_pass_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  intermediate_render_pass_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  intermediate_render_pass_attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  VkAttachmentReference intermediate_render_pass_color_attachment;
  intermediate_render_pass_color_attachment.attachment = 0;
  intermediate_render_pass_color_attachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  VkSubpassDescription intermediate_render_pass_subpass = {};
  intermediate_render_pass_subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  intermediate_render_pass_subpass.colorAttachmentCount = 1;
  intermediate_render_pass_subpass.pColorAttachments = &intermediate_render_pass_color_attachment;
  VkSubpassDependency intermediate_render_pass_dependencies[2];
  intermediate_render_pass_dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  intermediate_render_pass_dependencies[0].dstSubpass = 0;
  intermediate_render_pass_dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  intermediate_render_pass_dependencies[0].dstStageMask =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  intermediate_render_pass_dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  intermediate_render_pass_dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  intermediate_render_pass_dependencies[0].dependencyFlags = 0;
  intermediate_render_pass_dependencies[1].srcSubpass = 0;
  intermediate_render_pass_dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  intermediate_render_pass_dependencies[1].srcStageMask =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  intermediate_render_pass_dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  intermediate_render_pass_dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  intermediate_render_pass_dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  intermediate_render_pass_dependencies[1].dependencyFlags = 0;
  VkRenderPassCreateInfo intermediate_render_pass_create_info;
  intermediate_render_pass_create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  intermediate_render_pass_create_info.pNext = nullptr;
  intermediate_render_pass_create_info.flags = 0;
  intermediate_render_pass_create_info.attachmentCount = 1;
  intermediate_render_pass_create_info.pAttachments = &intermediate_render_pass_attachment;
  intermediate_render_pass_create_info.subpassCount = 1;
  intermediate_render_pass_create_info.pSubpasses = &intermediate_render_pass_subpass;
  intermediate_render_pass_create_info.dependencyCount =
      uint32_t(rex::countof(intermediate_render_pass_dependencies));
  intermediate_render_pass_create_info.pDependencies = intermediate_render_pass_dependencies;
  if (dfn.vkCreateRenderPass(device, &intermediate_render_pass_create_info, nullptr,
                             &guest_output_intermediate_render_pass_) != VK_SUCCESS) {
    REXLOG_ERROR(
        "VulkanPresenter: Failed to create the guest output intermediate image "
        "render pass");
    return false;
  }

  // Initialize connection-independent parts of the painting context.

  for (size_t i = 0; i < paint_context_.submissions.size(); ++i) {
    paint_context_.submissions[i] = PaintContext::Submission::Create(vulkan_device_);
    if (!paint_context_.submissions[i]) {
      return false;
    }
  }

  // Guest output paint pipelines drawing to intermediate images, not depending
  // on runtime state unlike ones drawing to the swapchain images as those
  // depend on the swapchain format.
  for (size_t i = 0; i < size_t(GuestOutputPaintEffect::kCount); ++i) {
    if (!CanGuestOutputPaintEffectBeIntermediate(GuestOutputPaintEffect(i)) ||
        guest_output_paint_fs_[i] == VK_NULL_HANDLE) {
      continue;
    }
    VkPipeline guest_output_paint_intermediate_pipeline = CreateGuestOutputPaintPipeline(
        GuestOutputPaintEffect(i), guest_output_intermediate_render_pass_);
    if (guest_output_paint_intermediate_pipeline == VK_NULL_HANDLE) {
      return false;
    }
    paint_context_.guest_output_paint_pipelines[i].intermediate_pipeline =
        guest_output_paint_intermediate_pipeline;
  }

  // Guest output painting descriptor sets.
  VkDescriptorPoolSize guest_output_paint_descriptor_pool_sizes[2];
  guest_output_paint_descriptor_pool_sizes[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  guest_output_paint_descriptor_pool_sizes[0].descriptorCount =
      PaintContext::kGuestOutputDescriptorSetCount;
  // Required even when using immutable samplers, otherwise failing to allocate
  // descriptor sets (tested on AMD Software: Adrenalin Edition 22.3.2 on
  // Windows 10 on AMD Radeon RX Vega 10 with Vulkan validation enabled).
  guest_output_paint_descriptor_pool_sizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLER;
  guest_output_paint_descriptor_pool_sizes[1].descriptorCount =
      PaintContext::kGuestOutputDescriptorSetCount;
  VkDescriptorPoolCreateInfo guest_output_paint_descriptor_pool_create_info;
  guest_output_paint_descriptor_pool_create_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  guest_output_paint_descriptor_pool_create_info.pNext = nullptr;
  guest_output_paint_descriptor_pool_create_info.flags = 0;
  guest_output_paint_descriptor_pool_create_info.maxSets =
      PaintContext::kGuestOutputDescriptorSetCount;
  guest_output_paint_descriptor_pool_create_info.poolSizeCount =
      uint32_t(rex::countof(guest_output_paint_descriptor_pool_sizes));
  guest_output_paint_descriptor_pool_create_info.pPoolSizes =
      guest_output_paint_descriptor_pool_sizes;
  if (dfn.vkCreateDescriptorPool(device, &guest_output_paint_descriptor_pool_create_info, nullptr,
                                 &paint_context_.guest_output_descriptor_pool) != VK_SUCCESS) {
    REXLOG_ERROR(
        "VulkanPresenter: Failed to create the guest output painting "
        "descriptor pool");
    return false;
  }
  VkDescriptorSetLayout
      guest_output_paint_descriptor_set_layouts[PaintContext::kGuestOutputDescriptorSetCount];
  std::fill(guest_output_paint_descriptor_set_layouts,
            guest_output_paint_descriptor_set_layouts +
                rex::countof(guest_output_paint_descriptor_set_layouts),
            guest_output_paint_image_descriptor_set_layout_);
  VkDescriptorSetAllocateInfo guest_output_paint_descriptor_set_allocate_info;
  guest_output_paint_descriptor_set_allocate_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  guest_output_paint_descriptor_set_allocate_info.pNext = nullptr;
  guest_output_paint_descriptor_set_allocate_info.descriptorPool =
      paint_context_.guest_output_descriptor_pool;
  guest_output_paint_descriptor_set_allocate_info.descriptorSetCount =
      PaintContext::kGuestOutputDescriptorSetCount;
  guest_output_paint_descriptor_set_allocate_info.pSetLayouts =
      guest_output_paint_descriptor_set_layouts;
  if (dfn.vkAllocateDescriptorSets(device, &guest_output_paint_descriptor_set_allocate_info,
                                   paint_context_.guest_output_descriptor_sets) != VK_SUCCESS) {
    REXLOG_ERROR(
        "VulkanPresenter: Failed to allocate the guest output painting "
        "descriptor sets");
    return false;
  }

  return InitializeCommonSurfaceIndependent();
}

VkPipeline VulkanPresenter::CreateGuestOutputPaintPipeline(GuestOutputPaintEffect effect,
                                                           VkRenderPass render_pass) {
  VkPipelineShaderStageCreateInfo stages[2] = {};
  for (uint32_t i = 0; i < 2; ++i) {
    VkPipelineShaderStageCreateInfo& stage = stages[i];
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = i ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_VERTEX_BIT;
    stage.pName = "main";
  }
  stages[0].module = guest_output_paint_vs_;
  stages[1].module = guest_output_paint_fs_[size_t(effect)];
  assert_true(stages[1].module != VK_NULL_HANDLE);
  // wwe2k14 patch (Session 94, present_pre_rotate): the final pass of a pre-rotated swapchain uses the rotating vertex
  // shader and the pixel shader reading the logical pixel coordinate from it in place of gl_FragCoord.
  const bool prerot_pipeline = render_pass == paint_context_.swapchain_render_pass &&
                               g_prerot_turns.load() != 0 && size_t(effect) < kPrerotEffectCapacity;
  const uint32_t prerot_map = PrerotMapOfTurns(g_prerot_turns.load());
  const uint32_t prerot_lw = (g_prerot_turns.load() & 1) ? paint_context_.swapchain_extent.height
                                                         : paint_context_.swapchain_extent.width;
  const uint32_t prerot_lh = (g_prerot_turns.load() & 1) ? paint_context_.swapchain_extent.width
                                                         : paint_context_.swapchain_extent.height;
  // Rotation and logical size reach the vertex shader as specialization constants (no push constant layout change).
  const uint32_t prerot_spec_data[3] = {prerot_map, std::bit_cast<uint32_t>(float(prerot_lw)),
                                        std::bit_cast<uint32_t>(float(prerot_lh))};
  const VkSpecializationMapEntry prerot_spec_entries[3] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}};
  VkSpecializationInfo prerot_spec_info = {3, prerot_spec_entries, sizeof(prerot_spec_data),
                                           prerot_spec_data};
  if (prerot_pipeline) {
    const VulkanDevice::Functions& prerot_dfn = vulkan_device_->functions();
    const VkDevice prerot_device = vulkan_device_->device();
    bool prerot_ok = true;
    if (g_prerot_vs == VK_NULL_HANDLE) {
      VkShaderModuleCreateInfo module_info = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      module_info.codeSize = sizeof(shaders::guest_output_triangle_strip_rect_rot_vs);
      module_info.pCode = shaders::guest_output_triangle_strip_rect_rot_vs;
      prerot_ok = prerot_dfn.vkCreateShaderModule(prerot_device, &module_info, nullptr, &g_prerot_vs) ==
                  VK_SUCCESS;
    }
    if (prerot_ok && g_prerot_fs[size_t(effect)] == VK_NULL_HANDLE) {
      // The effect's own SPIR-V (already built into the stock module), with its gl_FragCoord decoration (BuiltIn
      // FragCoord) turned into Location 0, where the rotating vertex shader writes the logical coordinate.
      const uint32_t* code = nullptr;
      size_t code_bytes = 0;
      switch (effect) {
        case GuestOutputPaintEffect::kBilinear:
          code = shaders::guest_output_bilinear_ps;
          code_bytes = sizeof(shaders::guest_output_bilinear_ps);
          break;
        case GuestOutputPaintEffect::kBilinearDither:
          code = shaders::guest_output_bilinear_dither_ps;
          code_bytes = sizeof(shaders::guest_output_bilinear_dither_ps);
          break;
        case GuestOutputPaintEffect::kSgsr:
          code = shaders::guest_output_sgsr_ps;
          code_bytes = sizeof(shaders::guest_output_sgsr_ps);
          break;
#if defined(REX_HAS_FIDELITYFX_SDK)
        case GuestOutputPaintEffect::kCasSharpen:
          code = shaders::guest_output_ffx_cas_sharpen_ps;
          code_bytes = sizeof(shaders::guest_output_ffx_cas_sharpen_ps);
          break;
        case GuestOutputPaintEffect::kCasSharpenDither:
          code = shaders::guest_output_ffx_cas_sharpen_dither_ps;
          code_bytes = sizeof(shaders::guest_output_ffx_cas_sharpen_dither_ps);
          break;
        case GuestOutputPaintEffect::kCasResample:
          code = shaders::guest_output_ffx_cas_resample_ps;
          code_bytes = sizeof(shaders::guest_output_ffx_cas_resample_ps);
          break;
        case GuestOutputPaintEffect::kCasResampleDither:
          code = shaders::guest_output_ffx_cas_resample_dither_ps;
          code_bytes = sizeof(shaders::guest_output_ffx_cas_resample_dither_ps);
          break;
        case GuestOutputPaintEffect::kFsrEasu:
          code = shaders::guest_output_ffx_fsr_easu_ps;
          code_bytes = sizeof(shaders::guest_output_ffx_fsr_easu_ps);
          break;
        case GuestOutputPaintEffect::kFsrRcas:
          code = shaders::guest_output_ffx_fsr_rcas_ps;
          code_bytes = sizeof(shaders::guest_output_ffx_fsr_rcas_ps);
          break;
        case GuestOutputPaintEffect::kFsrRcasDither:
          code = shaders::guest_output_ffx_fsr_rcas_dither_ps;
          code_bytes = sizeof(shaders::guest_output_ffx_fsr_rcas_dither_ps);
          break;
#endif
        default:
          break;
      }
      prerot_ok = code != nullptr;
      if (prerot_ok) {
        std::vector<uint32_t> patched(code, code + code_bytes / sizeof(uint32_t));
        size_t patched_count = 0;
        for (size_t w = 5; w < patched.size();) {
          const uint32_t length = patched[w] >> 16;
          if (!length || w + length > patched.size()) break;
          // OpDecorate (71) <id> BuiltIn (11) FragCoord (15)  ->  OpDecorate <id> Location (30) 0
          if ((patched[w] & 0xFFFFu) == 71 && length == 4 && patched[w + 2] == 11 && patched[w + 3] == 15) {
            patched[w + 2] = 30;
            patched[w + 3] = 0;
            ++patched_count;
          }
          w += length;
        }
        prerot_ok = patched_count == 1;
        if (prerot_ok) {
          VkShaderModuleCreateInfo module_info = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
          module_info.codeSize = patched.size() * sizeof(uint32_t);
          module_info.pCode = patched.data();
          prerot_ok = prerot_dfn.vkCreateShaderModule(prerot_device, &module_info, nullptr,
                                                      &g_prerot_fs[size_t(effect)]) == VK_SUCCESS;
        }
      }
    }
    if (prerot_ok) {
      stages[0].module = g_prerot_vs;
      stages[0].pSpecializationInfo = &prerot_spec_info;
      stages[1].module = g_prerot_fs[size_t(effect)];
    } else {
      if (!g_prerot_failed.exchange(true)) {
        REXLOG_ERROR("[pace] pre-rotate: could not build the rotating shaders for effect {}; falling back to an unrotated swapchain", size_t(effect));
      }
      return VK_NULL_HANDLE;
    }
  }
  if (render_pass == paint_context_.swapchain_render_pass && size_t(effect) < kPrerotEffectCapacity) {
    g_prerot_built_map[size_t(effect)] = prerot_pipeline ? prerot_map : 0;
    g_prerot_built_width[size_t(effect)] = prerot_pipeline ? prerot_lw : 0;
    g_prerot_built_height[size_t(effect)] = prerot_pipeline ? prerot_lh : 0;
  }

  VkPipelineVertexInputStateCreateInfo vertex_input_state = {};
  vertex_input_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineInputAssemblyStateCreateInfo input_assembly_state = {};
  input_assembly_state.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

  VkPipelineViewportStateCreateInfo viewport_state = {};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rasterization_state = {};
  rasterization_state.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterization_state.polygonMode = VK_POLYGON_MODE_FILL;
  rasterization_state.cullMode = VK_CULL_MODE_NONE;
  rasterization_state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterization_state.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo multisample_state = {};
  multisample_state.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample_state.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState color_blend_attachment_state = {};
  color_blend_attachment_state.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                                VK_COLOR_COMPONENT_G_BIT |
                                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo color_blend_state = {};
  color_blend_state.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blend_state.attachmentCount = 1;
  color_blend_state.pAttachments = &color_blend_attachment_state;

  static const VkDynamicState kPipelineDynamicStates[] = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
  };
  VkPipelineDynamicStateCreateInfo dynamic_state = {};
  dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_state.dynamicStateCount = uint32_t(rex::countof(kPipelineDynamicStates));
  dynamic_state.pDynamicStates = kPipelineDynamicStates;

  VkGraphicsPipelineCreateInfo pipeline_create_info;
  pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_create_info.pNext = nullptr;
  pipeline_create_info.flags = 0;
  pipeline_create_info.stageCount = uint32_t(rex::countof(stages));
  pipeline_create_info.pStages = stages;
  pipeline_create_info.pVertexInputState = &vertex_input_state;
  pipeline_create_info.pInputAssemblyState = &input_assembly_state;
  pipeline_create_info.pTessellationState = nullptr;
  pipeline_create_info.pViewportState = &viewport_state;
  pipeline_create_info.pRasterizationState = &rasterization_state;
  pipeline_create_info.pMultisampleState = &multisample_state;
  pipeline_create_info.pDepthStencilState = nullptr;
  pipeline_create_info.pColorBlendState = &color_blend_state;
  pipeline_create_info.pDynamicState = &dynamic_state;
  pipeline_create_info.layout =
      guest_output_paint_pipeline_layouts_[GetGuestOutputPaintPipelineLayoutIndex(effect)];
  pipeline_create_info.renderPass = render_pass;
  pipeline_create_info.subpass = 0;
  pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
  pipeline_create_info.basePipelineIndex = -1;

  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  VkPipeline pipeline;
  if (dfn.vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr,
                                    &pipeline) != VK_SUCCESS) {
    REXLOG_ERROR(
        "VulkanPresenter: Failed to create the guest output painting pipeline "
        "for effect {}",
        size_t(effect));
    return VK_NULL_HANDLE;
  }
  return pipeline;
}

}  // namespace vulkan
}  // namespace ui
}  // namespace rex
