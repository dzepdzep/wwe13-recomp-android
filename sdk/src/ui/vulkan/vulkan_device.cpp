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

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <rex/assert.h>
#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/string.h>
#include <rex/ui/vulkan/streamline.h>  // wwe2k14 patch (Session 18, DLAA)
#include <rex/ui/vulkan/device.h>
#include "gpu_quirks_apply.h"  // wwe2k14 patch (Session 93, driver quirks)

#if REX_PLATFORM_WIN32  // wwe2k14 patch (Session 55): vulkan_debug_check_buffer_offsets (the calling module)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

// wwe2k14 patch (Session 94, idea 1 phase 1A): rendering decoupled from the simulation - multiview plumbing. Defined here (the device
// enables the feature only when it is on); read by the native renderer (native_renderer.cpp).
REXCVAR_DEFINE_INT32(gpu_d3d_decouple, 0, "GPU/Native",
                     "Session 94 (idea 1, phase 1A; docs/native/d3d/S93_DECOUPLE_DESIGN.md): 0 off (default); bit 0 (1) = the owned passes render K = 2 views "
                     "(Vulkan multiview: layered owned images, render passes with a view mask); bit 2 (4) = A/A proof mode, both views use the same constants "
                     "(layer 0 = today's picture, the in-between view 1 uses the same constants) and a per-frame compare counts differing pixels; bit 3 (8) = phase 1B: every vertex float constant buffer carries one 256-vec4 block per view (view 0 = the tick's own constants, view 1 = the draws of OUR vertex shaders blended with the previous tick: matrices lerped, the camera by value), our vertex shaders run their multiview variants, the repeat / merge paths are off, and while the game's frame word equals gpu_d3d_decouple_word the presenter shows view 1 then view 0 half a tick later; bit 4 (16) = with bit 3, no two-step present (blend only); bit 5 (32) = Session 95 proof mode: the swap shows ONLY view 1 (the in-between), one present a tick, so shots, clips and the screenshot see it. 5 = on + A/A, 9 = on + blend + two-step present. Needs M2dEdramLess (TRUE NATIVE) - the truth / post targets are 1-layer. TRUE NATIVE (owned-only) only; logs "
                     "[native] decouple: multiview <yes|no>. Init only")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_INT32(gpu_d3d_decouple_device, 0, "GPU/Native",
                     "Session 94 (read-only result, set by the device): bit 0 = multiview was enabled, bit 1 = multiviewGeometryShader too");

REXCVAR_DEFINE_BOOL(vulkan_require_fragment_stores_and_atomics, true, "UI/Vulkan",
                    "Deprecated and ignored for parity; fragmentStoresAndAtomics is always "
                    "required for Vulkan GPU emulation")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
// wwe2k14 patch (Session 73): no longer ignored. Every Mali GPU lacks vertex shader stores (vulkan.gpuinfo.org, stock
// drivers), so they were refused outright; false lets such a GPU run, and a vertex shader's memory export is left out
// (logged once per shader by the command processor).
REXCVAR_DEFINE_BOOL(vulkan_require_vertex_pipeline_stores_and_atomics, true, "UI/Vulkan",
                    "Refuse a GPU without vertexPipelineStoresAndAtomics (Session 73: false lets it run; vertex shader "
                    "memory export is then left out)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
// Apple Silicon / MoltenVK does not expose geometryShader or fillModeNonSolid;
// default both to false on macOS so the device can still be selected.
REXCVAR_DEFINE_BOOL(vulkan_require_geometry_shader, !REX_PLATFORM_MAC, "UI/Vulkan",
                    "Require geometryShader support for Vulkan GPU emulation (disable to allow "
                    "fallback primitive emulation paths)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(vulkan_require_fill_mode_non_solid, !REX_PLATFORM_MAC, "UI/Vulkan",
                    "Require fillModeNonSolid support for Vulkan GPU emulation (disable to "
                    "allow fallback to solid fill for line/point polygon modes)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(vulkan_gpu_crash_diagnostics, false, "UI/Vulkan",
                    "On device loss, log VK_EXT_device_fault info and the last draws reached "
                    "(VK_NV_device_diagnostic_checkpoints) when the driver supports them")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

// wwe2k14 patch (Session 93, idea 14): VK_GOOGLE_display_timing for present pacing on Android (vulkan_presenter.cpp).
// Off by default: the extension is not requested and the present is unchanged.
REXCVAR_DEFINE_BOOL(present_display_timing, false, "UI/Vulkan",
                    "Pace presents to the display's refresh with VK_GOOGLE_display_timing where the device has it "
                    "(desiredPresentTime = last actual present + N refresh cycles for the game's frame rate; Android)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

// wwe2k14 patch (Session 24, OBJMV): VK_EXT_transform_feedback for the DLSS object motion vectors (capturing the
// character draws' positions). Off by default: the device is created as before.
REXCVAR_DEFINE_BOOL(vulkan_transform_feedback, false, "UI/Vulkan",
                    "Enable VK_EXT_transform_feedback if supported (DLSS object motion vectors, gpu_dlss_objmv)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

// wwe2k14 patch (Session 93, idea 7): VK_EXT_extended_dynamic_state for the guest pipelines (cull mode, front face, depth and
// stencil state dynamic: fewer pipeline variants). Off by default: not requested, the pipeline keys are unchanged. Also needs
// gpu_quirks_eds_allowed bit 1 (the quirks table) at the point of use (pipeline_cache.cpp).
REXCVAR_DEFINE_BOOL(vulkan_extended_dynamic_state, false, "UI/Vulkan",
                    "Session 93: make the guest pipelines' cull mode, front face, depth and stencil state dynamic "
                    "(VK_EXT_extended_dynamic_state, where the device has it and the quirks table allows it)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

// wwe2k14 patch (Session 55, the red blocks on Intel laptops): this PC's limits are not every player's. The HD 520 (and
// other older Intel graphics) needs storage buffer and texel buffer offsets in multiples of 64 bytes, where the 4090
// takes 16; a binding that breaks the device's rule is undefined behaviour there (the data lands elsewhere), and
// nothing in the code reads minStorageBufferOffsetAlignment. Diagnostic: check every buffer descriptor write and texel
// buffer view against the alignment given here and log the ones that break it (with the calling code's address).
REXCVAR_DEFINE_INT32(vulkan_debug_check_buffer_offsets, 0, "UI/Vulkan",
                     "Diagnostic (Session 55): log buffer descriptor writes and texel buffer views whose offset is not a "
                     "multiple of this many bytes (64 = Intel HD / Iris Xe's storage and texel buffer alignment); 0 = off")
    .range(0, 4096)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_STRING(vulkan_device_report_path, "", "UI/Vulkan",
                      "Session 95 (the diagnostic run): write a JSON report of this device (properties, every limit and "
                      "feature, extensions, memory heaps, queue families, format support) to this file once the device "
                      "is created; empty = off. Logs [diag] device report written")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

// wwe2k14 patch (Session 73, phone GPUs on the PC): the testers' phones take code paths that neither the 4090 nor
// Jarrod's Adreno 830 ever take (both have every feature and 2 GB storage buffers). These make this GPU report a
// weaker phone's limits / features / extensions to the rest of the code, so those paths run here. Diagnostic only.
REXCVAR_DEFINE_STRING(vulkan_emulate_limits, "", "UI/Vulkan",
                      "Diagnostic (Session 73): lower device limits, e.g. "
                      "\"maxStorageBufferRange=134217728,maxPerStageDescriptorStorageBuffers=24\"")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_INT32(vulkan_emulate_api_minor, -1, "UI/Vulkan",
                     "Diagnostic (Session 73): report the device as Vulkan 1.N at most (e.g. 1 for the Vulkan 1.1 drivers "
                     "many older phones ship); -1 = as it is")
    .range(-1, 4)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_STRING(vulkan_emulate_disable, "", "UI/Vulkan",
                      "Diagnostic (Session 73): features / extensions reported as missing, comma separated, e.g. "
                      "\"geometryShader,shaderCullDistance,VK_EXT_shader_stencil_export\"")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace rex {
namespace ui {
namespace vulkan {

namespace {

// wwe2k14 patch (Session 95, the diagnostic run): device.json, written once after device creation, so it holds what the
// driver the game really runs on reports (a custom driver's truth, not the launcher's). Everything is read with
// plain Vulkan queries; nothing is created. The reader (tools\native\diag_read.py) turns it into a profiles_*.tsv row.
void JsonEsc(std::string& out, const char* s) {
  out += '"';
  for (; s && *s; ++s) {
    const unsigned char c = (unsigned char)*s;
    if (c == '"' || c == '\\') out += '\\', out += char(c);
    else if (c < 0x20) out += ' ';
    else out += char(c);
  }
  out += '"';
}
void JsonKV(std::string& out, const char* key, uint64_t v) {
  out += '"', out += key, out += "\":", out += std::to_string(v), out += ',';
}
void JsonKVF(std::string& out, const char* key, double v) {
  char b[48];
  std::snprintf(b, sizeof(b), "%.9g", v);
  out += '"', out += key, out += "\":", out += b, out += ',';
}

void WriteDeviceReport(const VulkanInstance* const vulkan_instance, const VkPhysicalDevice physical_device,
                       const VkPhysicalDeviceProperties& properties,
                       const std::vector<const char*>& enabled_extensions, bool with_swapchain) {
  const std::string path = REXCVAR_GET(vulkan_device_report_path);
  if (path.empty()) return;
  const VulkanInstance::Functions& ifn = vulkan_instance->functions();
  std::string j = "{\"schema\":1,";
  j += "\"deviceName\":", JsonEsc(j, properties.deviceName), j += ',';
  JsonKV(j, "apiVersion", properties.apiVersion);
  JsonKV(j, "driverVersion", properties.driverVersion);
  JsonKV(j, "vendorID", properties.vendorID);
  JsonKV(j, "deviceID", properties.deviceID);
  JsonKV(j, "deviceType", uint32_t(properties.deviceType));
  JsonKV(j, "instanceApiVersion", vulkan_instance->api_version());
  JsonKV(j, "withSwapchain", with_swapchain ? 1 : 0);
  // Limits: every number (VkPhysicalDeviceLimits), by name.
  j += "\"limits\":{";
  const VkPhysicalDeviceLimits& l = properties.limits;
#define L(name) JsonKV(j, #name, uint64_t(l.name));
#define LF(name) JsonKVF(j, #name, double(l.name));
  L(maxImageDimension1D) L(maxImageDimension2D) L(maxImageDimension3D) L(maxImageDimensionCube)
  L(maxImageArrayLayers) L(maxTexelBufferElements) L(maxUniformBufferRange) L(maxStorageBufferRange)
  L(maxPushConstantsSize) L(maxMemoryAllocationCount) L(maxSamplerAllocationCount) L(bufferImageGranularity)
  L(maxBoundDescriptorSets) L(maxPerStageDescriptorSamplers) L(maxPerStageDescriptorUniformBuffers)
  L(maxPerStageDescriptorStorageBuffers) L(maxPerStageDescriptorSampledImages) L(maxPerStageDescriptorStorageImages)
  L(maxPerStageResources) L(maxDescriptorSetSamplers) L(maxDescriptorSetUniformBuffers)
  L(maxDescriptorSetUniformBuffersDynamic) L(maxDescriptorSetStorageBuffers) L(maxDescriptorSetStorageBuffersDynamic)
  L(maxDescriptorSetSampledImages) L(maxDescriptorSetStorageImages) L(maxVertexInputAttributes)
  L(maxVertexInputBindings) L(maxVertexInputAttributeOffset) L(maxVertexInputBindingStride)
  L(maxVertexOutputComponents) L(maxFragmentInputComponents) L(maxFragmentOutputAttachments)
  L(maxFragmentDualSrcAttachments) L(maxFragmentCombinedOutputResources) L(maxComputeSharedMemorySize)
  L(maxComputeWorkGroupInvocations) L(maxDrawIndexedIndexValue) L(maxDrawIndirectCount) L(maxViewports)
  L(maxFramebufferWidth) L(maxFramebufferHeight) L(maxFramebufferLayers) L(framebufferColorSampleCounts)
  L(framebufferDepthSampleCounts) L(maxColorAttachments) L(sampledImageColorSampleCounts)
  L(sampledImageDepthSampleCounts) L(storageImageSampleCounts) L(maxSampleMaskWords) L(timestampComputeAndGraphics)
  L(maxClipDistances) L(maxCullDistances) L(maxCombinedClipAndCullDistances)
  L(minMemoryMapAlignment) L(minTexelBufferOffsetAlignment) L(minUniformBufferOffsetAlignment)
  L(minStorageBufferOffsetAlignment) L(optimalBufferCopyOffsetAlignment) L(optimalBufferCopyRowPitchAlignment)
  L(nonCoherentAtomSize) L(minTexelOffset) L(maxTexelOffset) L(subPixelPrecisionBits)
  L(viewportSubPixelBits)
  LF(maxSamplerLodBias) LF(maxSamplerAnisotropy) LF(pointSizeGranularity) LF(lineWidthGranularity) LF(timestampPeriod)
#undef L
#undef LF
  j += "\"_\":0},";
  // Features: the whole Vulkan 1.0 set, true ones by name.
  VkPhysicalDeviceFeatures f = {};
  ifn.vkGetPhysicalDeviceFeatures(physical_device, &f);
  j += "\"features\":{";
#define FT(name) JsonKV(j, #name, f.name ? 1 : 0);
  FT(robustBufferAccess) FT(fullDrawIndexUint32) FT(imageCubeArray) FT(independentBlend) FT(geometryShader)
  FT(tessellationShader) FT(sampleRateShading) FT(dualSrcBlend) FT(logicOp) FT(multiDrawIndirect)
  FT(drawIndirectFirstInstance) FT(depthClamp) FT(depthBiasClamp) FT(fillModeNonSolid) FT(depthBounds)
  FT(wideLines) FT(largePoints) FT(alphaToOne) FT(multiViewport) FT(samplerAnisotropy)
  FT(textureCompressionETC2) FT(textureCompressionASTC_LDR) FT(textureCompressionBC) FT(occlusionQueryPrecise)
  FT(pipelineStatisticsQuery) FT(vertexPipelineStoresAndAtomics) FT(fragmentStoresAndAtomics)
  FT(shaderTessellationAndGeometryPointSize) FT(shaderImageGatherExtended) FT(shaderStorageImageExtendedFormats)
  FT(shaderStorageImageMultisample) FT(shaderStorageImageReadWithoutFormat)
  FT(shaderStorageImageWriteWithoutFormat) FT(shaderUniformBufferArrayDynamicIndexing)
  FT(shaderSampledImageArrayDynamicIndexing) FT(shaderStorageBufferArrayDynamicIndexing)
  FT(shaderStorageImageArrayDynamicIndexing) FT(shaderClipDistance) FT(shaderCullDistance) FT(shaderFloat64)
  FT(shaderInt64) FT(shaderInt16) FT(shaderResourceResidency) FT(shaderResourceMinLod) FT(sparseBinding)
  FT(variableMultisampleRate) FT(inheritedQueries)
#undef FT
  j += "\"_\":0},";
  // The Vulkan 1.1 / 1.2 / 1.3 feature and property structs and the driver / float control / subgroup chains, when
  // the instance can ask (properties2).
  if (ifn.vkGetPhysicalDeviceFeatures2 && ifn.vkGetPhysicalDeviceProperties2) {
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    const uint32_t api = properties.apiVersion;
    void** tail = &f2.pNext;
    if (api >= VK_MAKE_API_VERSION(0, 1, 2, 0)) *tail = &f11, tail = &f11.pNext, *tail = &f12, tail = &f12.pNext;
    if (api >= VK_MAKE_API_VERSION(0, 1, 3, 0)) *tail = &f13, tail = &f13.pNext;
    ifn.vkGetPhysicalDeviceFeatures2(physical_device, &f2);
    j += "\"features11\":{";
#define FT(s, name) JsonKV(j, #name, (s).name ? 1 : 0);
    if (api >= VK_MAKE_API_VERSION(0, 1, 2, 0)) {
      FT(f11, storageBuffer16BitAccess) FT(f11, uniformAndStorageBuffer16BitAccess) FT(f11, storagePushConstant16)
      FT(f11, multiview) FT(f11, multiviewGeometryShader) FT(f11, variablePointers)
      FT(f11, variablePointersStorageBuffer) FT(f11, shaderDrawParameters)
    }
    j += "\"_\":0},\"features12\":{";
    if (api >= VK_MAKE_API_VERSION(0, 1, 2, 0)) {
      FT(f12, samplerMirrorClampToEdge) FT(f12, drawIndirectCount) FT(f12, storageBuffer8BitAccess)
      FT(f12, shaderBufferInt64Atomics) FT(f12, shaderFloat16) FT(f12, shaderInt8)
      FT(f12, descriptorIndexing) FT(f12, scalarBlockLayout) FT(f12, uniformBufferStandardLayout)
      FT(f12, shaderSubgroupExtendedTypes) FT(f12, hostQueryReset) FT(f12, timelineSemaphore)
      FT(f12, bufferDeviceAddress) FT(f12, vulkanMemoryModel) FT(f12, shaderOutputViewportIndex)
      FT(f12, shaderOutputLayer) FT(f12, subgroupBroadcastDynamicId)
    }
    j += "\"_\":0},\"features13\":{";
    if (api >= VK_MAKE_API_VERSION(0, 1, 3, 0)) {
      FT(f13, robustImageAccess) FT(f13, inlineUniformBlock) FT(f13, pipelineCreationCacheControl)
      FT(f13, synchronization2) FT(f13, dynamicRendering) FT(f13, shaderDemoteToHelperInvocation)
      FT(f13, maintenance4)
    }
#undef FT
    j += "\"_\":0},";
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceFloatControlsProperties fc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
    VkPhysicalDeviceSubgroupProperties sg = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    const bool v12 = api >= VK_MAKE_API_VERSION(0, 1, 2, 0);
    p2.pNext = &sg;
    if (v12) sg.pNext = &drv, drv.pNext = &fc;
    ifn.vkGetPhysicalDeviceProperties2(physical_device, &p2);
    j += "\"subgroup\":{";
    JsonKV(j, "subgroupSize", sg.subgroupSize);
    JsonKV(j, "supportedStages", sg.supportedStages);
    JsonKV(j, "supportedOperations", sg.supportedOperations);
    JsonKV(j, "quadOperationsInAllStages", sg.quadOperationsInAllStages);
    j += "\"_\":0},";
    if (v12) {
      j += "\"driver\":{";
      JsonKV(j, "driverID", uint32_t(drv.driverID));
      j += "\"driverName\":", JsonEsc(j, drv.driverName), j += ",\"driverInfo\":", JsonEsc(j, drv.driverInfo), j += ',';
      JsonKV(j, "conformance_major", drv.conformanceVersion.major);
      JsonKV(j, "conformance_minor", drv.conformanceVersion.minor);
      j += "\"_\":0},\"floatControls\":{";
      JsonKV(j, "denormBehaviorIndependence", uint32_t(fc.denormBehaviorIndependence));
      JsonKV(j, "roundingModeIndependence", uint32_t(fc.roundingModeIndependence));
      JsonKV(j, "shaderSignedZeroInfNanPreserveFloat16", fc.shaderSignedZeroInfNanPreserveFloat16);
      JsonKV(j, "shaderSignedZeroInfNanPreserveFloat32", fc.shaderSignedZeroInfNanPreserveFloat32);
      JsonKV(j, "shaderDenormPreserveFloat16", fc.shaderDenormPreserveFloat16);
      JsonKV(j, "shaderDenormPreserveFloat32", fc.shaderDenormPreserveFloat32);
      JsonKV(j, "shaderDenormFlushToZeroFloat16", fc.shaderDenormFlushToZeroFloat16);
      JsonKV(j, "shaderDenormFlushToZeroFloat32", fc.shaderDenormFlushToZeroFloat32);
      JsonKV(j, "shaderRoundingModeRTEFloat16", fc.shaderRoundingModeRTEFloat16);
      JsonKV(j, "shaderRoundingModeRTEFloat32", fc.shaderRoundingModeRTEFloat32);
      JsonKV(j, "shaderRoundingModeRTZFloat32", fc.shaderRoundingModeRTZFloat32);
      j += "\"_\":0},";
    }
  }
  // Extensions: every supported one with its spec version; the enabled ones listed apart.
  {
    uint32_t n = 0;
    ifn.vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> ext(n);
    if (n) ifn.vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &n, ext.data());
    j += "\"extensions\":[";
    for (uint32_t i = 0; i < n; ++i) {
      j += i ? ",[" : "[";
      JsonEsc(j, ext[i].extensionName);
      j += ',' + std::to_string(ext[i].specVersion) + ']';
    }
    j += "],\"enabledExtensions\":[";
    for (size_t i = 0; i < enabled_extensions.size(); ++i) {
      if (i) j += ',';
      JsonEsc(j, enabled_extensions[i]);
    }
    j += "],";
  }
  // Memory heaps and types.
  {
    VkPhysicalDeviceMemoryProperties m = {};
    ifn.vkGetPhysicalDeviceMemoryProperties(physical_device, &m);
    j += "\"memoryHeaps\":[";
    for (uint32_t i = 0; i < m.memoryHeapCount; ++i)
      j += (i ? "," : "") + std::string("[") + std::to_string(m.memoryHeaps[i].size) + ',' +
           std::to_string(m.memoryHeaps[i].flags) + ']';
    j += "],\"memoryTypes\":[";
    for (uint32_t i = 0; i < m.memoryTypeCount; ++i)
      j += (i ? "," : "") + std::string("[") + std::to_string(m.memoryTypes[i].heapIndex) + ',' +
           std::to_string(m.memoryTypes[i].propertyFlags) + ']';
    j += "],";
  }
  // Queue families: flags, count, timestampValidBits, minImageTransferGranularity.
  {
    uint32_t n = 0;
    ifn.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &n, nullptr);
    std::vector<VkQueueFamilyProperties> q(n);
    if (n) ifn.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &n, q.data());
    j += "\"queueFamilies\":[";
    for (uint32_t i = 0; i < n; ++i)
      j += (i ? "," : "") + std::string("[") + std::to_string(q[i].queueFlags) + ',' + std::to_string(q[i].queueCount) +
           ',' + std::to_string(q[i].timestampValidBits) + ']';
    j += "],";
  }
  // Format support: VkFormat 1..184 (every core format incl. BC / ETC2 / ASTC), the three feature masks in hex
  // (linear, optimal, buffer); formats with nothing set are left out.
  {
    j += "\"formats\":{";
    bool first = true;
    for (uint32_t fmt = 1; fmt <= 184; ++fmt) {
      VkFormatProperties fp = {};
      ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VkFormat(fmt), &fp);
      if (!fp.linearTilingFeatures && !fp.optimalTilingFeatures && !fp.bufferFeatures) continue;
      char b[96];
      std::snprintf(b, sizeof(b), "%s\"%u\":[\"%x\",\"%x\",\"%x\"]", first ? "" : ",", fmt, fp.linearTilingFeatures,
                    fp.optimalTilingFeatures, fp.bufferFeatures);
      j += b;
      first = false;
    }
    j += "},";
  }
  // The emulation switches in force (a fake device row is a different truth from the real one).
  j += "\"emulate\":{\"gpu\":";
  JsonEsc(j, rex::cvar::GetFlagByName("vulkan_emulate_gpu").c_str());
  j += ",\"limits\":";
  JsonEsc(j, REXCVAR_GET(vulkan_emulate_limits).c_str());
  j += ",\"disable\":";
  JsonEsc(j, REXCVAR_GET(vulkan_emulate_disable).c_str());
  j += "},\"surface\":\"not captured here (needs the surface: see system.txt / the log for the present modes)\"}\n";
  const std::string part = path + ".part";
  if (FILE* fp = std::fopen(part.c_str(), "wb")) {
    std::fwrite(j.data(), 1, j.size(), fp);
    std::fclose(fp);
    std::remove(path.c_str());
    if (std::rename(part.c_str(), path.c_str()) == 0) {
      REXLOG_INFO("[diag] device report written: {} ({} bytes)", path, j.size());
      return;
    }
  }
  REXLOG_WARN("[diag] device report could not be written to {}", path);
}

// Session 73: the comma separated names of vulkan_emulate_disable / the name=value pairs of vulkan_emulate_limits.
std::vector<std::string> EmulateSplit(const std::string& list) {
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= list.size()) {
    size_t end = list.find(',', start);
    if (end == std::string::npos) end = list.size();
    std::string item = list.substr(start, end - start);
    item.erase(0, item.find_first_not_of(" \t"));
    item.erase(item.find_last_not_of(" \t") + 1);
    if (!item.empty()) out.push_back(item);
    start = end + 1;
  }
  return out;
}

bool EmulateDisabled(const char* name) {
  static const std::vector<std::string> names = EmulateSplit(REXCVAR_GET(vulkan_emulate_disable));
  for (const std::string& n : names) {
    if (n == name) return true;
  }
  return false;
}
// wwe2k14 patch (Session 55): vulkan_debug_check_buffer_offsets.
PFN_vkUpdateDescriptorSets g_real_update_descriptor_sets = nullptr;
PFN_vkCreateBufferView g_real_create_buffer_view = nullptr;
VkDeviceSize g_check_alignment = 0;
std::atomic<uint64_t> g_bad_offsets{0}, g_checked_offsets{0};
#if REX_PLATFORM_WIN32
#define CallerAddress() _ReturnAddress()
#else
#define CallerAddress() __builtin_return_address(0)
#endif

uintptr_t CallerRva(void* address, const char*& module_name) {
#if REX_PLATFORM_WIN32
  static thread_local char name[MAX_PATH];
  HMODULE module = nullptr;
  module_name = "?";
  if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          static_cast<LPCSTR>(address), &module)) {
    return uintptr_t(address);
  }
  if (GetModuleFileNameA(module, name, MAX_PATH)) {
    const char* slash = std::strrchr(name, '\\');
    module_name = slash ? slash + 1 : name;
  }
  return uintptr_t(address) - uintptr_t(module);
#else
  module_name = "?";
  return uintptr_t(address);
#endif
}

void NoteBadOffset(void* caller, const char* what, VkDeviceSize offset, VkDeviceSize range, uint32_t binding) {
  const uint64_t n = g_bad_offsets.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n > 64 && (n & (n - 1))) return;  // the first 64, then at powers of two
  const char* module_name;
  const uintptr_t rva = CallerRva(caller, module_name);
  REXLOG_WARN("[vulkan][offsets] #{} (of {} checked): {} offset {} (0x{:X}, {} past a multiple of {}) range {} binding {} "
              "from {}+0x{:X}", n, g_checked_offsets.load(std::memory_order_relaxed), what, offset, offset,
              offset % g_check_alignment, g_check_alignment, range == VK_WHOLE_SIZE ? int64_t(-1) : int64_t(range),
              binding, module_name, rva);
}

VKAPI_ATTR void VKAPI_CALL CheckedUpdateDescriptorSets(VkDevice device, uint32_t write_count,
                                                       const VkWriteDescriptorSet* writes, uint32_t copy_count,
                                                       const VkCopyDescriptorSet* copies) {
  for (uint32_t i = 0; i < write_count; ++i) {
    const VkWriteDescriptorSet& w = writes[i];
    const bool storage = w.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    const bool uniform = w.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    if ((!storage && !uniform) || !w.pBufferInfo) continue;
    for (uint32_t j = 0; j < w.descriptorCount; ++j) {
      g_checked_offsets.fetch_add(1, std::memory_order_relaxed);
      if (w.pBufferInfo[j].offset % g_check_alignment) {
        NoteBadOffset(CallerAddress(), storage ? "storage buffer" : "uniform buffer", w.pBufferInfo[j].offset,
                      w.pBufferInfo[j].range, w.dstBinding);
      }
    }
  }
  g_real_update_descriptor_sets(device, write_count, writes, copy_count, copies);
}

VKAPI_ATTR VkResult VKAPI_CALL CheckedCreateBufferView(VkDevice device, const VkBufferViewCreateInfo* info,
                                                       const VkAllocationCallbacks* allocator, VkBufferView* view) {
  g_checked_offsets.fetch_add(1, std::memory_order_relaxed);
  if (info && info->offset % g_check_alignment) {
    NoteBadOffset(CallerAddress(), "texel buffer view", info->offset, info->range, 0);
  }
  return g_real_create_buffer_view(device, info, allocator, view);
}
}  // namespace

// wwe2k14 patch (Session 94, idea 1 phase 1A): whether the device was created with multiview enabled (a global, not a Features
// member, so the public header's layout stays as it was). Read by native_renderer.cpp.
// (published to the GPU library through the cvar gpu_d3d_decouple_device: bit 0 multiview, bit 1 multiviewGeometryShader - a cvar crosses the DLL border, a plain global does not)
bool g_device_multiview = false;
bool g_device_multiview_geometry_shader = false;  // (multiviewGeometryShader: the emulated pipelines' geometry shaders - rectangles, point sprites - may run in a multiview pass)
bool g_device_multiview_tessellation = false;  // (multiviewTessellationShader: bit 2 (4) of gpu_d3d_decouple_device)

template <typename Structure, VkStructureType StructureType>
struct VulkanFeatures {
  Structure supported = {StructureType};
  Structure enabled = {StructureType};

  void Link(VkPhysicalDeviceFeatures2& supported_features_2,
            VkDeviceCreateInfo& device_create_info) {
    supported.pNext = supported_features_2.pNext;
    supported_features_2.pNext = &supported;
    enabled.pNext = const_cast<void*>(device_create_info.pNext);
    device_create_info.pNext = &enabled;
  }
};

std::unique_ptr<VulkanDevice> VulkanDevice::CreateIfSupported(
    const VulkanInstance* const vulkan_instance, const VkPhysicalDevice physical_device,
    const bool with_gpu_emulation, const bool with_swapchain) {
  assert_not_null(vulkan_instance);
  assert_not_null(physical_device);

  const VulkanInstance::Functions& ifn = vulkan_instance->functions();

  // Get supported Vulkan 1.0 properties and features.

  VkPhysicalDeviceProperties properties = {};
  ifn.vkGetPhysicalDeviceProperties(physical_device, &properties);

  // wwe2k14 patch (Session 73): vulkan_emulate_api_minor (an older phone driver's Vulkan version).
  if (const int32_t minor = REXCVAR_GET(vulkan_emulate_api_minor);
      minor >= 0 && VK_API_VERSION_MINOR(properties.apiVersion) > uint32_t(minor)) {
    REXLOG_WARN("Vulkan: emulating Vulkan 1.{} (the device has 1.{})", minor, VK_API_VERSION_MINOR(properties.apiVersion));
    properties.apiVersion = VK_MAKE_API_VERSION(0, 1, uint32_t(minor), VK_API_VERSION_PATCH(properties.apiVersion));
  }
  const uint32_t unclamped_api_version = properties.apiVersion;
  if (vulkan_instance->api_version() < VK_MAKE_API_VERSION(0, 1, 1, 0)) {
    // From the VkApplicationInfo specification:
    //
    // "The Khronos validation layers will treat apiVersion as the highest API
    // version the application targets, and will validate API usage against the
    // minimum of that version and the implementation version (instance or
    // device, depending on context). If an application tries to use
    // functionality from a greater version than this, a validation error will
    // be triggered."
    //
    // "Vulkan 1.0 implementations were required to return
    // VK_ERROR_INCOMPATIBLE_DRIVER if apiVersion was larger than 1.0."
    properties.apiVersion =
        VK_MAKE_API_VERSION(0, 1, 0, VK_API_VERSION_PATCH(properties.apiVersion));
  }

  VkPhysicalDeviceFeatures supported_features = {};
  ifn.vkGetPhysicalDeviceFeatures(physical_device, &supported_features);

  if (with_gpu_emulation) {
    if (!supported_features.independentBlend) {
      // Not trivial to work around:
      // - Affects not only the blend equation, but also the color write mask.
      // - Can't reuse the blend state of the first attachment for all because
      //   some attachments may have a format that doesn't support blending.
      // - Not possible to split the draw into per-attachment draws because of
      //   depth / stencil.
      // Not supported only on the proprietary driver for the Qualcomm
      // Adreno 4xx, where the driver is largely experimental and doesn't expose
      // a lot of the functionality available in the hardware.
      REXLOG_WARN(
          "Vulkan device '{}' doesn't support the independentBlend feature "
          "required for GPU emulation",
          properties.deviceName);
      return nullptr;
    }
    if (!supported_features.fragmentStoresAndAtomics) {
      REXLOG_WARN(
          "Vulkan device '{}' doesn't support fragmentStoresAndAtomics, which "
          "is required for Vulkan GPU emulation parity",
          properties.deviceName);
      return nullptr;
    }
    if (!supported_features.vertexPipelineStoresAndAtomics &&
        REXCVAR_GET(vulkan_require_vertex_pipeline_stores_and_atomics)) {  // wwe2k14 patch (Session 73)
      REXLOG_WARN(
          "Vulkan device '{}' doesn't support vertexPipelineStoresAndAtomics, which "
          "is required for Vulkan GPU emulation parity",
          properties.deviceName);
      return nullptr;
    }
    if (REXCVAR_GET(vulkan_require_geometry_shader) && !supported_features.geometryShader) {
      REXLOG_WARN(
          "Vulkan device '{}' doesn't support geometryShader, but "
          "vulkan_require_geometry_shader=true",
          properties.deviceName);
      return nullptr;
    }
    if (REXCVAR_GET(vulkan_require_fill_mode_non_solid) && !supported_features.fillModeNonSolid) {
      REXLOG_WARN(
          "Vulkan device '{}' doesn't support fillModeNonSolid, but "
          "vulkan_require_fill_mode_non_solid=true",
          properties.deviceName);
      return nullptr;
    }
  }

  // Enable needed extensions.

  std::unique_ptr<VulkanDevice> device(new VulkanDevice(vulkan_instance, physical_device));

  const bool get_physical_device_properties2_supported =
      vulkan_instance->extensions().ext_1_1_KHR_get_physical_device_properties2;

  // Name pointers from `requested_extensions` will be used in the enabled
  // extensions vector.
  std::unordered_map<std::string, bool*> requested_extensions;

  const auto request_promoted_extension = [&](const char* const name, uint32_t const major,
                                              uint32_t const minor, bool* const supported_ptr) {
    assert_not_null(supported_ptr);
    if (properties.apiVersion >= VK_MAKE_API_VERSION(0, major, minor, 0)) {
      *supported_ptr = true;
    } else {
      requested_extensions.emplace(name, supported_ptr);
    }
  };

#define XE_UI_VULKAN_STRUCT_EXTENSION(name) \
  requested_extensions.emplace("VK_" #name, &device->extensions_.ext_##name);
#define XE_UI_VULKAN_LOCAL_EXTENSION(name) requested_extensions.emplace("VK_" #name, &ext_##name);
#define XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(name, major, minor) \
  request_promoted_extension("VK_" #name, major, minor,            \
                             &device->extensions_.ext_##major##_##minor##_##name);
#define XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION(name, major, minor) \
  request_promoted_extension("VK_" #name, major, minor, &ext_##major##_##minor##_##name);

  bool ext_KHR_portability_subset = false;
  bool ext_1_2_KHR_driver_properties = false;
  if (get_physical_device_properties2_supported) {
    // #164. Must be enabled according to the specification if the physical
    // device is a portability subset one.
    XE_UI_VULKAN_LOCAL_EXTENSION(KHR_portability_subset)
    // #197
    XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION(KHR_driver_properties, 1, 2)
  }

  // Used by the Vulkan Memory Allocator and potentially by Xenia.
  // #128.
  XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(KHR_dedicated_allocation, 1, 1)
  // #147. Also must be enabled for VK_KHR_dedicated_allocation and
  // VK_KHR_sampler_ycbcr_conversion.
  XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(KHR_get_memory_requirements2, 1, 1)
  // #158. Also must be enabled for VK_KHR_sampler_ycbcr_conversion.
  XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(KHR_bind_memory2, 1, 1)
  if (get_physical_device_properties2_supported) {
    // #238.
    XE_UI_VULKAN_STRUCT_EXTENSION(EXT_memory_budget)
  }
  if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0)) {
    // #414.
    XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(KHR_maintenance4, 1, 3)
    // #55.
    XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(KHR_dynamic_rendering, 1, 3)
  }

  if (with_swapchain) {
    // #2.
    XE_UI_VULKAN_STRUCT_EXTENSION(KHR_swapchain)
    if (REXCVAR_GET(present_display_timing)) {
      // #93 (wwe2k14 patch, Session 93).
      XE_UI_VULKAN_STRUCT_EXTENSION(GOOGLE_display_timing)
    }
  }

  bool ext_1_2_KHR_sampler_mirror_clamp_to_edge = false;
  bool ext_1_1_KHR_maintenance1 = false;
  bool ext_1_2_KHR_shader_float_controls = false;
  bool ext_EXT_fragment_shader_interlock = false;
  bool ext_1_3_EXT_shader_demote_to_helper_invocation = false;
  bool ext_1_3_KHR_dynamic_rendering = false;
  bool ext_EXT_non_seamless_cube_map = false;
  if (with_gpu_emulation) {
    // #15.
    XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION(KHR_sampler_mirror_clamp_to_edge, 1, 2)
    // #70. Must be enabled for VK_KHR_sampler_ycbcr_conversion.
    XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION(KHR_maintenance1, 1, 1)
    // #141.
    XE_UI_VULKAN_STRUCT_EXTENSION(EXT_shader_stencil_export)
    // #148.
    XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(KHR_image_format_list, 1, 2)
    if (get_physical_device_properties2_supported) {
      // #157.
      XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(KHR_sampler_ycbcr_conversion, 1, 1)
      // #198. Also must be enabled for VK_KHR_spirv_1_4.
      XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION(KHR_shader_float_controls, 1, 2)
      // #252.
      XE_UI_VULKAN_LOCAL_EXTENSION(EXT_fragment_shader_interlock)
      // #55.
      XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION(KHR_dynamic_rendering, 1, 3)
      // #277.
      XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION(EXT_shader_demote_to_helper_invocation, 1, 3)
      // #423.
      XE_UI_VULKAN_LOCAL_EXTENSION(EXT_non_seamless_cube_map)
      // Required for non-black YCbCr border color parity with D3D12.
      XE_UI_VULKAN_STRUCT_EXTENSION(EXT_custom_border_color)
      // Required for true null descriptors in bindless texture bindings.
      XE_UI_VULKAN_STRUCT_EXTENSION(EXT_robustness2)
      // #401 (wwe2k14 patch, Session 73): attachments kept without being loaded / stored (tile GPUs).
      XE_UI_VULKAN_STRUCT_EXTENSION(EXT_load_store_op_none)
      if (REXCVAR_GET(vulkan_transform_feedback)) {
        // #29 (wwe2k14 patch, Session 24).
        XE_UI_VULKAN_STRUCT_EXTENSION(EXT_transform_feedback)
      }
      if (REXCVAR_GET(vulkan_extended_dynamic_state)) {
        // #268 (wwe2k14 patch, Session 93, idea 7).
        XE_UI_VULKAN_STRUCT_EXTENSION(EXT_extended_dynamic_state)
      }
      if (REXCVAR_GET(vulkan_gpu_crash_diagnostics)) {
        // #207.
        XE_UI_VULKAN_STRUCT_EXTENSION(NV_device_diagnostic_checkpoints)
        // #342.
        XE_UI_VULKAN_STRUCT_EXTENSION(EXT_device_fault)
      }
    }
    if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0)) {
      // #237.
      XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(KHR_spirv_1_4, 1, 2)
    }
  }

#undef XE_UI_VULKAN_STRUCT_EXTENSION
#undef XE_UI_VULKAN_LOCAL_EXTENSION
#undef XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION
#undef XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION

  std::vector<const char*> enabled_extensions;
  {
    uint32_t supported_extension_count = 0;
    const VkResult get_supported_extension_count_result = ifn.vkEnumerateDeviceExtensionProperties(
        physical_device, nullptr, &supported_extension_count, nullptr);
    if (get_supported_extension_count_result != VK_SUCCESS &&
        get_supported_extension_count_result != VK_INCOMPLETE) {
      REXLOG_WARN("Failed to get the Vulkan device '{}' extension count", properties.deviceName);
      return nullptr;
    }
    if (supported_extension_count) {
      std::vector<VkExtensionProperties> supported_extensions(supported_extension_count);
      if (ifn.vkEnumerateDeviceExtensionProperties(physical_device, nullptr,
                                                   &supported_extension_count,
                                                   supported_extensions.data()) != VK_SUCCESS) {
        REXLOG_WARN("Failed to get the Vulkan device '{}' extensions", properties.deviceName);
        return nullptr;
      }
      assert_true(supported_extension_count == supported_extensions.size());
      for (const VkExtensionProperties& supported_extension : supported_extensions) {
        const auto requested_extension_it =
            requested_extensions.find(supported_extension.extensionName);
        if (requested_extension_it == requested_extensions.cend()) {
          continue;
        }
        assert_not_null(requested_extension_it->second);
        if (EmulateDisabled(supported_extension.extensionName)) {  // wwe2k14 patch (Session 73)
          REXLOG_WARN("Vulkan: emulating a missing {} (vulkan_emulate_disable)", supported_extension.extensionName);
          continue;
        }
        if (!*requested_extension_it->second) {
          enabled_extensions.emplace_back(requested_extension_it->first.c_str());
          *requested_extension_it->second = true;
        }
      }
    }
  }

  if (with_swapchain && !device->extensions_.ext_KHR_swapchain) {
    REXLOG_WARN("Vulkan device '{}' doesn't support swapchains", properties.deviceName);
    return nullptr;
  }

  VkDeviceCreateInfo device_create_info = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};

  device_create_info.enabledExtensionCount = uint32_t(enabled_extensions.size());
  device_create_info.ppEnabledExtensionNames = enabled_extensions.data();

  // Get supported Vulkan 1.1+ and extension properties and features.
  //
  // The property and feature structures are initialized to zero or to the
  // minimum / maximum requirements for the simplicity of handling unavailable
  // VK_KHR_get_physical_device_properties2.

  VkPhysicalDeviceProperties2 properties_2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};

  VkPhysicalDeviceFeatures2 supported_features_2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};

  VulkanFeatures<VkPhysicalDeviceVulkan12Features,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES>
      features_1_2;
  VulkanFeatures<VkPhysicalDeviceVulkan13Features,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES>
      features_1_3;
  VulkanFeatures<VkPhysicalDevicePortabilitySubsetFeaturesKHR,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR>
      features_KHR_portability_subset;
  VkPhysicalDeviceDriverPropertiesKHR properties_1_2_KHR_driver_properties = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
  VkPhysicalDeviceFloatControlsProperties properties_1_2_KHR_shader_float_controls = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
  // wwe2k14 patch (Session 94, idea 1 phase 1A): core 1.1 multiview, only when gpu_d3d_decouple is on.
  VulkanFeatures<VkPhysicalDeviceMultiviewFeatures, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES>
      features_multiview;
  VulkanFeatures<VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT>
      features_EXT_fragment_shader_interlock;
  VulkanFeatures<VkPhysicalDeviceDynamicRenderingFeaturesKHR,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR>
      features_1_3_KHR_dynamic_rendering;
  VulkanFeatures<VkPhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES_EXT>
      features_1_3_EXT_shader_demote_to_helper_invocation;
  VulkanFeatures<VkPhysicalDeviceNonSeamlessCubeMapFeaturesEXT,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NON_SEAMLESS_CUBE_MAP_FEATURES_EXT>
      features_EXT_non_seamless_cube_map;
  VulkanFeatures<VkPhysicalDeviceCustomBorderColorFeaturesEXT,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT>
      features_EXT_custom_border_color;
  VkPhysicalDeviceCustomBorderColorPropertiesEXT properties_EXT_custom_border_color = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_PROPERTIES_EXT};
  VulkanFeatures<VkPhysicalDeviceRobustness2FeaturesEXT,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT>
      features_EXT_robustness2;
  VulkanFeatures<VkPhysicalDeviceTransformFeedbackFeaturesEXT,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT>
      features_EXT_transform_feedback;
  VulkanFeatures<VkPhysicalDeviceExtendedDynamicStateFeaturesEXT,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT>
      features_EXT_extended_dynamic_state;
  VulkanFeatures<VkPhysicalDeviceFaultFeaturesEXT,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT>
      features_EXT_device_fault;

  if (get_physical_device_properties2_supported) {
    if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 2, 0)) {
      features_1_2.Link(supported_features_2, device_create_info);
    }
    if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 3, 0)) {
      features_1_3.Link(supported_features_2, device_create_info);
    } else {
      if (ext_1_3_KHR_dynamic_rendering) {
        features_1_3_KHR_dynamic_rendering.Link(supported_features_2, device_create_info);
      }
      if (ext_1_3_EXT_shader_demote_to_helper_invocation) {
        features_1_3_EXT_shader_demote_to_helper_invocation.Link(supported_features_2,
                                                                 device_create_info);
      }
    }
    if (ext_KHR_portability_subset) {
      features_KHR_portability_subset.Link(supported_features_2, device_create_info);
    }
    if (ext_1_2_KHR_driver_properties) {
      properties_1_2_KHR_driver_properties.pNext = properties_2.pNext;
      properties_2.pNext = &properties_1_2_KHR_driver_properties;
    }
    if (ext_1_2_KHR_shader_float_controls) {
      properties_1_2_KHR_shader_float_controls.pNext = properties_2.pNext;
      properties_2.pNext = &properties_1_2_KHR_shader_float_controls;
    }
    if (ext_EXT_fragment_shader_interlock) {
      features_EXT_fragment_shader_interlock.Link(supported_features_2, device_create_info);
    }
    if (ext_EXT_non_seamless_cube_map) {
      features_EXT_non_seamless_cube_map.Link(supported_features_2, device_create_info);
    }
    if (device->extensions_.ext_EXT_custom_border_color) {
      features_EXT_custom_border_color.Link(supported_features_2, device_create_info);
      properties_EXT_custom_border_color.pNext = properties_2.pNext;
      properties_2.pNext = &properties_EXT_custom_border_color;
    }
    if (device->extensions_.ext_EXT_robustness2) {
      features_EXT_robustness2.Link(supported_features_2, device_create_info);
    }
    if (device->extensions_.ext_EXT_transform_feedback) {
      features_EXT_transform_feedback.Link(supported_features_2, device_create_info);
    }
    if (device->extensions_.ext_EXT_extended_dynamic_state) {  // wwe2k14 patch (Session 93, idea 7)
      features_EXT_extended_dynamic_state.Link(supported_features_2, device_create_info);
    }
    if (device->extensions_.ext_EXT_device_fault) {
      features_EXT_device_fault.Link(supported_features_2, device_create_info);
    }
    if (REXCVAR_GET(gpu_d3d_decouple) != 0 && properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0)) {
      features_multiview.Link(supported_features_2, device_create_info);
    }
    ifn.vkGetPhysicalDeviceProperties2(physical_device, &properties_2);
    ifn.vkGetPhysicalDeviceFeatures2(physical_device, &supported_features_2);
  }

  uint32_t queue_family_count = 0;
  ifn.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  ifn.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count,
                                               queue_families.data());

  device->queue_families_.resize(queue_family_count);

  uint32_t first_queue_family_graphics_compute_sparse_binding = UINT32_MAX;
  uint32_t first_queue_family_graphics_compute = UINT32_MAX;
  uint32_t first_queue_family_sparse_binding = UINT32_MAX;
  bool has_presentation_queue_family = false;

  for (uint32_t queue_family_index = 0; queue_family_index < queue_family_count;
       ++queue_family_index) {
    QueueFamily& queue_family = device->queue_families_[queue_family_index];
    const VkQueueFamilyProperties& queue_family_properties = queue_families[queue_family_index];

    const VkQueueFlags queue_unsupported_flags = ~queue_family_properties.queueFlags;

    if (!(queue_unsupported_flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
      first_queue_family_graphics_compute =
          std::min(queue_family_index, first_queue_family_graphics_compute);
    }

    if (with_gpu_emulation && supported_features.sparseBinding &&
        !(queue_unsupported_flags & VK_QUEUE_SPARSE_BINDING_BIT)) {
      first_queue_family_sparse_binding =
          std::min(queue_family_index, first_queue_family_sparse_binding);
      if (!(queue_unsupported_flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
        first_queue_family_graphics_compute_sparse_binding =
            std::min(queue_family_index, first_queue_family_graphics_compute_sparse_binding);
      }
    }

    if (with_swapchain) {
#if REX_PLATFORM_WIN32
      queue_family.may_support_presentation =
          vulkan_instance->extensions().ext_KHR_win32_surface &&
          ifn.vkGetPhysicalDeviceWin32PresentationSupportKHR(physical_device, queue_family_index);
#else
      queue_family.may_support_presentation = true;
#endif
      if (queue_family.may_support_presentation) {
        queue_family.queues.resize(std::max(size_t(1), queue_family.queues.size()));
        has_presentation_queue_family = true;
      }
    }
  }

  if (first_queue_family_graphics_compute == UINT32_MAX) {
    // Not valid according to the Vulkan specification, but for safety.
    REXLOG_WARN(
        "Vulkan device '{}' doesn't provide a graphics and compute queue "
        "family",
        properties.deviceName);
    return nullptr;
  }

  if (with_swapchain && !has_presentation_queue_family) {
    REXLOG_WARN(
        "Vulkan device '{}' doesn't provide a queue family that supports "
        "presentation",
        properties.deviceName);
    return nullptr;
  }

  // Get the queues to create.

  if (first_queue_family_sparse_binding == UINT32_MAX) {
    // Not valid not to provide a sparse binding queue if the sparseBinding
    // feature is supported according to the Vulkan specification, but for
    // safety and simplicity.
    supported_features.sparseBinding = VK_FALSE;
  }
  if (!supported_features.sparseBinding) {
    supported_features.sparseResidencyBuffer = VK_FALSE;
    supported_features.sparseResidencyImage2D = VK_FALSE;
    supported_features.sparseResidencyImage3D = VK_FALSE;
    supported_features.sparseResidency2Samples = VK_FALSE;
    supported_features.sparseResidency4Samples = VK_FALSE;
    supported_features.sparseResidency8Samples = VK_FALSE;
    supported_features.sparseResidency16Samples = VK_FALSE;
    supported_features.sparseResidencyAliased = VK_FALSE;
  }

  // Prefer using one queue for everything whenever possible for simplicity.
  // TODO(Triang3l): Research if separate queues for purposes like composition,
  // swapchain image presentation, and sparse binding, may be beneficial.

  if (first_queue_family_graphics_compute_sparse_binding != UINT32_MAX) {
    device->queue_family_graphics_compute_ = first_queue_family_graphics_compute_sparse_binding;
    device->queue_family_sparse_binding_ = first_queue_family_graphics_compute_sparse_binding;
  } else {
    device->queue_family_graphics_compute_ = first_queue_family_graphics_compute;
    device->queue_family_sparse_binding_ = first_queue_family_sparse_binding;
  }

  device->queue_families_[device->queue_family_graphics_compute_].queues.resize(std::max(
      size_t(1), device->queue_families_[device->queue_family_graphics_compute_].queues.size()));
  if (device->queue_family_sparse_binding_ != UINT32_MAX) {
    device->queue_families_[device->queue_family_sparse_binding_].queues.resize(std::max(
        size_t(1), device->queue_families_[device->queue_family_sparse_binding_].queues.size()));
  }

  size_t max_enabled_queues_per_family = 0;
  for (const QueueFamily& queue_family : device->queue_families_) {
    max_enabled_queues_per_family =
        std::max(queue_family.queues.size(), max_enabled_queues_per_family);
  }
  const std::vector<float> queue_priorities(max_enabled_queues_per_family, 1.0f);
  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  for (size_t queue_family_index = 0; queue_family_index < device->queue_families_.size();
       ++queue_family_index) {
    const QueueFamily& queue_family = device->queue_families_[queue_family_index];
    if (queue_family.queues.empty()) {
      continue;
    }
    VkDeviceQueueCreateInfo& queue_create_info = queue_create_infos.emplace_back();
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.pNext = nullptr;
    queue_create_info.flags = 0;
    queue_create_info.queueFamilyIndex = uint32_t(queue_family_index);
    queue_create_info.queueCount = uint32_t(queue_family.queues.size());
    queue_create_info.pQueuePriorities = queue_priorities.data();
  }
  device_create_info.queueCreateInfoCount = uint32_t(queue_create_infos.size());
  device_create_info.pQueueCreateInfos = queue_create_infos.data();

  // Enable needed features and copy the properties.
  //
  // Enabling only actually used features because drivers may take more optimal
  // paths when certain features are disabled. Also, in VK_EXT_shader_object,
  // the state that the application must set for the draw depends on which
  // features are enabled.

  device->properties_.apiVersion = properties.apiVersion;
  device->properties_.driverVersion = properties.driverVersion;
  device->properties_.vendorID = properties.vendorID;
  device->properties_.deviceID = properties.deviceID;
  rex::string::copy_truncating(device->properties_.deviceName, properties.deviceName,
                               sizeof(device->properties_.deviceName));

  REXLOG_INFO(
      "Vulkan device '{}': API {}.{}.{}, vendor 0x{:04X}, device 0x{:04X}, "
      "driver version 0x{:X}",
      properties.deviceName, VK_VERSION_MAJOR(properties.apiVersion),
      VK_VERSION_MINOR(properties.apiVersion), VK_VERSION_PATCH(properties.apiVersion),
      properties.vendorID, properties.deviceID, properties.driverVersion);
  if (unclamped_api_version != properties.apiVersion) {
    REXLOG_INFO(
        "Device supports Vulkan API {}.{}.{}, but the used version is limited "
        "by the instance",
        VK_VERSION_MAJOR(unclamped_api_version), VK_VERSION_MINOR(unclamped_api_version),
        VK_VERSION_PATCH(unclamped_api_version));
  }

  REXLOG_INFO("Enabled Vulkan device extensions:");
  for (uint32_t enabled_extension_index = 0;
       enabled_extension_index < device_create_info.enabledExtensionCount;
       ++enabled_extension_index) {
    REXLOG_INFO("* {}", device_create_info.ppEnabledExtensionNames[enabled_extension_index]);
  }

  // wwe2k14 patch (Session 93, idea 6): the driver quirks table (detect the GPU + driver, log, set the cvars whose
  // switch is still at its default). Before the features are chosen below.
  const GpuQuirksOutcome quirks =
      ApplyGpuQuirks(ifn, physical_device, properties,
                     ext_1_2_KHR_driver_properties ? &properties_1_2_KHR_driver_properties : nullptr,
                     supported_features.geometryShader != VK_FALSE && !EmulateDisabled("geometryShader"));

  REXLOG_INFO("Vulkan device properties and enabled features:");

  VkPhysicalDeviceFeatures enabled_features = {};
  device_create_info.pEnabledFeatures = &enabled_features;

#define XE_UI_VULKAN_LIMIT(name)                     \
  device->properties_.name = properties.limits.name; \
  REXLOG_INFO("* " #name ": {}", properties.limits.name);
#define XE_UI_VULKAN_ENUM_LIMIT(name, type)          \
  device->properties_.name = properties.limits.name; \
  REXLOG_INFO("* " #name ": {}", vk::to_string(vk::type(properties.limits.name)));
#define XE_UI_VULKAN_FEATURE(name)                    \
  enabled_features.name = supported_features.name;    \
  device->properties_.name = supported_features.name; \
  if (supported_features.name) {                      \
    REXLOG_INFO("* " #name);                          \
  }
#define XE_UI_VULKAN_PROPERTY_2(structure, name) \
  device->properties_.name = structure.name;     \
  REXLOG_INFO("* " #name ": {}", structure.name);
#define XE_UI_VULKAN_ENUM_PROPERTY_2(structure, name, type) \
  device->properties_.name = structure.name;                \
  REXLOG_INFO("* " #name ": {}", vk::to_string(vk::type(structure.name)));
#define XE_UI_VULKAN_FEATURE_2(structure, name)        \
  structure.enabled.name = structure.supported.name;   \
  device->properties_.name = structure.supported.name; \
  if (structure.supported.name) {                      \
    REXLOG_INFO("* " #name);                           \
  }
#define XE_UI_VULKAN_FEATURE_IMPLIED(name) \
  device->properties_.name = true;         \
  REXLOG_INFO("* " #name);

  if (ext_1_2_KHR_driver_properties) {
    XE_UI_VULKAN_ENUM_PROPERTY_2(properties_1_2_KHR_driver_properties, driverID, DriverId);
    REXLOG_INFO("* driverName: {}", properties_1_2_KHR_driver_properties.driverName);
    if (properties_1_2_KHR_driver_properties.driverInfo[0]) {
      REXLOG_INFO("* driverInfo: {}", properties_1_2_KHR_driver_properties.driverInfo);
    }
    REXLOG_INFO("* conformanceVersion: {}.{}.{}.{}",
                properties_1_2_KHR_driver_properties.conformanceVersion.major,
                properties_1_2_KHR_driver_properties.conformanceVersion.minor,
                properties_1_2_KHR_driver_properties.conformanceVersion.subminor,
                properties_1_2_KHR_driver_properties.conformanceVersion.patch);
  }

  XE_UI_VULKAN_LIMIT(maxImageDimension2D)
  XE_UI_VULKAN_LIMIT(maxImageDimension3D)
  XE_UI_VULKAN_LIMIT(maxImageDimensionCube)
  XE_UI_VULKAN_LIMIT(maxImageArrayLayers)
  XE_UI_VULKAN_LIMIT(maxStorageBufferRange)
  XE_UI_VULKAN_LIMIT(maxSamplerAllocationCount)
  XE_UI_VULKAN_LIMIT(maxPerStageDescriptorSamplers)
  XE_UI_VULKAN_LIMIT(maxPerStageDescriptorStorageBuffers)
  XE_UI_VULKAN_LIMIT(maxPerStageDescriptorSampledImages)
  XE_UI_VULKAN_LIMIT(maxPerStageResources)
  XE_UI_VULKAN_LIMIT(maxVertexOutputComponents)
  XE_UI_VULKAN_LIMIT(maxTessellationEvaluationOutputComponents)
  XE_UI_VULKAN_LIMIT(maxGeometryInputComponents)
  XE_UI_VULKAN_LIMIT(maxGeometryOutputComponents)
  XE_UI_VULKAN_LIMIT(maxFragmentInputComponents)
  XE_UI_VULKAN_LIMIT(maxFragmentCombinedOutputResources)
  XE_UI_VULKAN_LIMIT(maxSamplerAnisotropy)
  XE_UI_VULKAN_LIMIT(maxViewportDimensions[0])
  XE_UI_VULKAN_LIMIT(maxViewportDimensions[1])
  XE_UI_VULKAN_LIMIT(minUniformBufferOffsetAlignment)
  XE_UI_VULKAN_LIMIT(minStorageBufferOffsetAlignment)
  XE_UI_VULKAN_LIMIT(maxFramebufferWidth)
  XE_UI_VULKAN_LIMIT(maxFramebufferHeight)
  XE_UI_VULKAN_ENUM_LIMIT(framebufferColorSampleCounts, SampleCountFlags)
  XE_UI_VULKAN_ENUM_LIMIT(framebufferDepthSampleCounts, SampleCountFlags)
  XE_UI_VULKAN_ENUM_LIMIT(framebufferStencilSampleCounts, SampleCountFlags)
  XE_UI_VULKAN_ENUM_LIMIT(framebufferNoAttachmentsSampleCounts, SampleCountFlags)
  XE_UI_VULKAN_ENUM_LIMIT(sampledImageColorSampleCounts, SampleCountFlags)
  XE_UI_VULKAN_ENUM_LIMIT(sampledImageIntegerSampleCounts, SampleCountFlags)
  XE_UI_VULKAN_ENUM_LIMIT(sampledImageDepthSampleCounts, SampleCountFlags)
  XE_UI_VULKAN_ENUM_LIMIT(sampledImageStencilSampleCounts, SampleCountFlags)
  XE_UI_VULKAN_LIMIT(standardSampleLocations)
  XE_UI_VULKAN_LIMIT(optimalBufferCopyOffsetAlignment)
  XE_UI_VULKAN_LIMIT(optimalBufferCopyRowPitchAlignment)
  XE_UI_VULKAN_LIMIT(nonCoherentAtomSize)

  if (with_gpu_emulation) {
    if (quirks.robust_buffer_access) {  // wwe2k14 patch (Session 93): vulkan_robust_buffer_access
      XE_UI_VULKAN_FEATURE(robustBufferAccess)
    } else {
      REXLOG_INFO("* robustBufferAccess left off (vulkan_robust_buffer_access / vulkan_quirk_robust_buffer_access: see the [quirks] robust_buffer_access line)");
    }
    XE_UI_VULKAN_FEATURE(fullDrawIndexUint32)
    XE_UI_VULKAN_FEATURE(independentBlend)
    XE_UI_VULKAN_FEATURE(geometryShader)
    XE_UI_VULKAN_FEATURE(tessellationShader)
    XE_UI_VULKAN_FEATURE(sampleRateShading)
    XE_UI_VULKAN_FEATURE(depthClamp)
    XE_UI_VULKAN_FEATURE(fillModeNonSolid)
    XE_UI_VULKAN_FEATURE(samplerAnisotropy)
    XE_UI_VULKAN_FEATURE(occlusionQueryPrecise)
    XE_UI_VULKAN_FEATURE(vertexPipelineStoresAndAtomics)
    XE_UI_VULKAN_FEATURE(fragmentStoresAndAtomics)
    XE_UI_VULKAN_FEATURE(shaderStorageImageExtendedFormats)  // wwe2k14 patch (Session 59)
    XE_UI_VULKAN_FEATURE(shaderClipDistance)
    XE_UI_VULKAN_FEATURE(shaderCullDistance)
    XE_UI_VULKAN_FEATURE(sparseBinding)
    XE_UI_VULKAN_FEATURE(sparseResidencyBuffer)
  }

  if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 2, 0)) {
    if (with_gpu_emulation) {
      XE_UI_VULKAN_FEATURE_2(features_1_2, samplerMirrorClampToEdge);
      // wwe2k14 patch (Session 80, HD 520 arc): 16-bit float maths in shaders, for gpu_native_ps_half (Intel's Gen9
      // runs it at twice the 32-bit rate; the HD 520 / UHD 620 Windows drivers report it). A capability only: no shader
      // uses it unless that cvar converts one.
      XE_UI_VULKAN_FEATURE_2(features_1_2, shaderFloat16);
    }
  } else {
    if (ext_1_2_KHR_sampler_mirror_clamp_to_edge) {
      XE_UI_VULKAN_FEATURE_IMPLIED(samplerMirrorClampToEdge)
    }
  }

  if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 3, 0)) {
    if (with_gpu_emulation) {
      XE_UI_VULKAN_FEATURE_2(features_1_3, shaderDemoteToHelperInvocation);
      XE_UI_VULKAN_FEATURE_2(features_1_3, dynamicRendering);
    }
  } else {
    if (ext_1_3_KHR_dynamic_rendering) {
      if (with_gpu_emulation) {
        XE_UI_VULKAN_FEATURE_2(features_1_3_KHR_dynamic_rendering, dynamicRendering);
      }
    }
    if (ext_1_3_EXT_shader_demote_to_helper_invocation) {
      if (with_gpu_emulation) {
        XE_UI_VULKAN_FEATURE_2(features_1_3_EXT_shader_demote_to_helper_invocation,
                               shaderDemoteToHelperInvocation);
      }
    }
  }

  if (ext_KHR_portability_subset) {
    if (with_gpu_emulation) {
      XE_UI_VULKAN_FEATURE_2(features_KHR_portability_subset, constantAlphaColorBlendFactors)
      XE_UI_VULKAN_FEATURE_2(features_KHR_portability_subset, imageViewFormatReinterpretation)
      XE_UI_VULKAN_FEATURE_2(features_KHR_portability_subset, imageViewFormatSwizzle)
      XE_UI_VULKAN_FEATURE_2(features_KHR_portability_subset, pointPolygons)
      XE_UI_VULKAN_FEATURE_2(features_KHR_portability_subset, separateStencilMaskRef)
      XE_UI_VULKAN_FEATURE_2(features_KHR_portability_subset,
                             shaderSampleRateInterpolationFunctions)
    }
  } else {
    // Not a portability subset device.
    XE_UI_VULKAN_FEATURE_IMPLIED(constantAlphaColorBlendFactors)
    XE_UI_VULKAN_FEATURE_IMPLIED(imageViewFormatReinterpretation)
    XE_UI_VULKAN_FEATURE_IMPLIED(imageViewFormatSwizzle)
    XE_UI_VULKAN_FEATURE_IMPLIED(pointPolygons)
    XE_UI_VULKAN_FEATURE_IMPLIED(separateStencilMaskRef)
    XE_UI_VULKAN_FEATURE_IMPLIED(shaderSampleRateInterpolationFunctions)
  }

  if (ext_1_2_KHR_shader_float_controls) {
    XE_UI_VULKAN_PROPERTY_2(properties_1_2_KHR_shader_float_controls,
                            shaderSignedZeroInfNanPreserveFloat32);
    XE_UI_VULKAN_PROPERTY_2(properties_1_2_KHR_shader_float_controls,
                            shaderDenormFlushToZeroFloat32);
    XE_UI_VULKAN_PROPERTY_2(properties_1_2_KHR_shader_float_controls, shaderRoundingModeRTEFloat32);
  }

  if (ext_EXT_fragment_shader_interlock) {
    if (with_gpu_emulation) {
      XE_UI_VULKAN_FEATURE_2(features_EXT_fragment_shader_interlock, fragmentShaderSampleInterlock)
      XE_UI_VULKAN_FEATURE_2(features_EXT_fragment_shader_interlock, fragmentShaderPixelInterlock)
    }
  }

  if (ext_EXT_non_seamless_cube_map) {
    if (with_gpu_emulation) {
      XE_UI_VULKAN_FEATURE_2(features_EXT_non_seamless_cube_map, nonSeamlessCubeMap)
    }
  }

  if (device->extensions_.ext_EXT_custom_border_color) {
    if (with_gpu_emulation) {
      if (quirks.allow_custom_border_color) {  // wwe2k14 patch (Session 93): vulkan_allow_custom_border_color
        XE_UI_VULKAN_FEATURE_2(features_EXT_custom_border_color, customBorderColors)
        XE_UI_VULKAN_FEATURE_2(features_EXT_custom_border_color, customBorderColorWithoutFormat)
      } else {
        REXLOG_INFO("* customBorderColors left off (vulkan_allow_custom_border_color=false)");
      }
    }
    XE_UI_VULKAN_PROPERTY_2(properties_EXT_custom_border_color, maxCustomBorderColorSamplers)
  }

  if (device->extensions_.ext_EXT_robustness2) {
    if (with_gpu_emulation) {
      XE_UI_VULKAN_FEATURE_2(features_EXT_robustness2, nullDescriptor)
    }
  }

  if (device->extensions_.ext_EXT_transform_feedback) {  // wwe2k14 patch (Session 24)
    if (with_gpu_emulation) {
      XE_UI_VULKAN_FEATURE_2(features_EXT_transform_feedback, transformFeedback)
    }
  }

  if (device->extensions_.ext_EXT_extended_dynamic_state) {  // wwe2k14 patch (Session 93, idea 7)
    if (with_gpu_emulation) {
      XE_UI_VULKAN_FEATURE_2(features_EXT_extended_dynamic_state, extendedDynamicState)
    }
  }

  // wwe2k14 patch (Session 94, idea 1 phase 1A): multiview, when asked for and supported.
  g_device_multiview = false;
  g_device_multiview_geometry_shader = false;
  g_device_multiview_tessellation = false;
  REXCVAR_SET(gpu_d3d_decouple_device, 0);  // (review: a stale answer of an earlier device attempt never stays)
  if (REXCVAR_GET(gpu_d3d_decouple) != 0 && properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0) &&
      features_multiview.supported.multiview) {
    features_multiview.enabled.multiview = VK_TRUE;
    g_device_multiview = true;
    g_device_multiview_geometry_shader = features_multiview.supported.multiviewGeometryShader != VK_FALSE;
    features_multiview.enabled.multiviewGeometryShader = features_multiview.supported.multiviewGeometryShader;
    g_device_multiview_tessellation = features_multiview.supported.multiviewTessellationShader != VK_FALSE;
    features_multiview.enabled.multiviewTessellationShader = features_multiview.supported.multiviewTessellationShader;
    // (the flags are published below, once vkCreateDevice has succeeded)
    REXLOG_INFO("* multiview (gpu_d3d_decouple), multiviewGeometryShader {}", g_device_multiview_geometry_shader ? "yes" : "no");
  }

  if (device->extensions_.ext_EXT_device_fault) {
    // Fault info only; the vendor binary dump isn't consumed.
    features_EXT_device_fault.enabled.deviceFault = features_EXT_device_fault.supported.deviceFault;
    if (features_EXT_device_fault.supported.deviceFault) {
      REXLOG_INFO("* deviceFault");
    } else {
      device->extensions_.ext_EXT_device_fault = false;
    }
  }

  // wwe2k14 patch (Session 73): a weaker phone's limits and missing features, for the rest of the code (the device is
  // still created with what it has; nothing reads the enabled set back).
  {
    Properties& p = device->properties_;
#define XE_EMU_LIMIT(name)                                                             \
  if (key == #name) {                                                                  \
    const uint64_t v = std::stoull(value, nullptr, 0);                                 \
    REXLOG_WARN("Vulkan: emulating " #name " {} (was {})", v, uint64_t(p.name));       \
    p.name = decltype(p.name)(v);                                                      \
    continue;                                                                          \
  }
    for (const std::string& pair : EmulateSplit(REXCVAR_GET(vulkan_emulate_limits))) {
      const size_t eq = pair.find('=');
      if (eq == std::string::npos) continue;
      const std::string key = pair.substr(0, eq), value = pair.substr(eq + 1);
      XE_EMU_LIMIT(maxImageDimension2D)
      XE_EMU_LIMIT(maxImageDimension3D)
      XE_EMU_LIMIT(maxImageDimensionCube)
      XE_EMU_LIMIT(maxImageArrayLayers)
      XE_EMU_LIMIT(maxStorageBufferRange)
      XE_EMU_LIMIT(maxSamplerAllocationCount)
      XE_EMU_LIMIT(maxPerStageDescriptorSamplers)
      XE_EMU_LIMIT(maxPerStageDescriptorStorageBuffers)
      XE_EMU_LIMIT(maxPerStageDescriptorSampledImages)
      XE_EMU_LIMIT(maxPerStageResources)
      XE_EMU_LIMIT(maxVertexOutputComponents)
      XE_EMU_LIMIT(maxFragmentInputComponents)
      XE_EMU_LIMIT(maxFragmentCombinedOutputResources)
      XE_EMU_LIMIT(minUniformBufferOffsetAlignment)
      XE_EMU_LIMIT(minStorageBufferOffsetAlignment)
      XE_EMU_LIMIT(maxFramebufferWidth)
      XE_EMU_LIMIT(maxFramebufferHeight)
      XE_EMU_LIMIT(nonCoherentAtomSize)
      XE_EMU_LIMIT(maxCustomBorderColorSamplers)
      REXLOG_ERROR("Vulkan: vulkan_emulate_limits: unknown limit {}", key);
    }
#undef XE_EMU_LIMIT
#define XE_EMU_FEATURE(name)                                         \
  if (p.name && EmulateDisabled(#name)) {                            \
    REXLOG_WARN("Vulkan: emulating a missing " #name);              \
    p.name = false;                                                  \
  }
    XE_EMU_FEATURE(robustBufferAccess)
    XE_EMU_FEATURE(fullDrawIndexUint32)
    XE_EMU_FEATURE(independentBlend)
    XE_EMU_FEATURE(geometryShader)
    XE_EMU_FEATURE(tessellationShader)
    XE_EMU_FEATURE(sampleRateShading)
    XE_EMU_FEATURE(depthClamp)
    XE_EMU_FEATURE(fillModeNonSolid)
    XE_EMU_FEATURE(samplerAnisotropy)
    XE_EMU_FEATURE(occlusionQueryPrecise)
    XE_EMU_FEATURE(vertexPipelineStoresAndAtomics)
    XE_EMU_FEATURE(shaderStorageImageExtendedFormats)
    XE_EMU_FEATURE(shaderClipDistance)
    XE_EMU_FEATURE(shaderCullDistance)
    XE_EMU_FEATURE(sparseBinding)
    XE_EMU_FEATURE(sparseResidencyBuffer)
    XE_EMU_FEATURE(samplerMirrorClampToEdge)
    XE_EMU_FEATURE(uniformBufferStandardLayout)
    XE_EMU_FEATURE(scalarBlockLayout)
    XE_EMU_FEATURE(shaderFloat16)
    XE_EMU_FEATURE(shaderSignedZeroInfNanPreserveFloat32)
    XE_EMU_FEATURE(shaderDenormFlushToZeroFloat32)
    XE_EMU_FEATURE(shaderRoundingModeRTEFloat32)
    XE_EMU_FEATURE(fragmentShaderSampleInterlock)
    XE_EMU_FEATURE(fragmentShaderPixelInterlock)
    XE_EMU_FEATURE(shaderDemoteToHelperInvocation)
    XE_EMU_FEATURE(dynamicRendering)
    XE_EMU_FEATURE(nonSeamlessCubeMap)
    XE_EMU_FEATURE(customBorderColors)
    XE_EMU_FEATURE(customBorderColorWithoutFormat)
    XE_EMU_FEATURE(nullDescriptor)
#undef XE_EMU_FEATURE
  }

#undef XE_UI_VULKAN_LIMIT
#undef XE_UI_VULKAN_ENUM_LIMIT
#undef XE_UI_VULKAN_FEATURE
#undef XE_UI_VULKAN_PROPERTY_2
#undef XE_UI_VULKAN_ENUM_PROPERTY_2
#undef XE_UI_VULKAN_FEATURE_2

  // Create the device.

  const VkResult device_create_result =
      ifn.vkCreateDevice(physical_device, &device_create_info, nullptr, &device->device_);
  if (device_create_result != VK_SUCCESS) {
    REXLOG_ERROR(
        "Failed to create a Vulkan logical device from the physical device "
        "'{}': {}",
        properties.deviceName, vk::to_string(vk::Result(device_create_result)));
    return nullptr;
  }

  streamline::SetPhysicalDevice(physical_device);  // wwe2k14 patch (Session 18, DLAA)
  if (g_device_multiview) {  // wwe2k14 patch (Session 94 review): published only after the device exists
    REXCVAR_SET(gpu_d3d_decouple_device, 1 | (g_device_multiview_geometry_shader ? 2 : 0) | (g_device_multiview_tessellation ? 4 : 0));
  }

  // Load device functions.

  bool functions_loaded = true;

  Functions& dfn = device->functions_;

#define XE_UI_VULKAN_FUNCTION(name) \
  functions_loaded &=               \
      (dfn.name = PFN_##name(ifn.vkGetDeviceProcAddr(device->device_, #name))) != nullptr;

  // Vulkan 1.0.
#include <rex/ui/vulkan/functions/device_1_0.inc>

  // Extensions promoted to a Vulkan version supported by the device.
#define XE_UI_VULKAN_FUNCTION_PROMOTED(extension_name, core_name)                                \
  functions_loaded &=                                                                            \
      (dfn.core_name = PFN_##core_name(ifn.vkGetDeviceProcAddr(device->device_, #core_name))) != \
      nullptr;
  if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0)) {
#include <rex/ui/vulkan/functions/device_1_1_khr_bind_memory2.inc>
#include <rex/ui/vulkan/functions/device_1_1_khr_get_memory_requirements2.inc>
  }
  if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 3, 0)) {
#include <rex/ui/vulkan/functions/device_1_3_khr_dynamic_rendering.inc>
#include <rex/ui/vulkan/functions/device_1_3_khr_maintenance4.inc>
  }
#undef XE_UI_VULKAN_FUNCTION_PROMOTED

  // Non-promoted extensions, and extensions promoted to a Vulkan version not
  // supported by the device.
#define XE_UI_VULKAN_FUNCTION_PROMOTED(extension_name, core_name) \
  functions_loaded &= (dfn.core_name = PFN_##core_name(           \
                           ifn.vkGetDeviceProcAddr(device->device_, #extension_name))) != nullptr;
  if (properties.apiVersion < VK_MAKE_API_VERSION(0, 1, 1, 0)) {
    if (device->extensions_.ext_1_1_KHR_get_memory_requirements2) {
#include <rex/ui/vulkan/functions/device_1_1_khr_get_memory_requirements2.inc>
    }
    if (device->extensions_.ext_1_1_KHR_bind_memory2) {
#include <rex/ui/vulkan/functions/device_1_1_khr_bind_memory2.inc>
    }
  }
  if (properties.apiVersion < VK_MAKE_API_VERSION(0, 1, 3, 0)) {
    if (device->extensions_.ext_1_3_KHR_dynamic_rendering) {
#include <rex/ui/vulkan/functions/device_1_3_khr_dynamic_rendering.inc>
    }
    if (device->extensions_.ext_1_3_KHR_maintenance4) {
#include <rex/ui/vulkan/functions/device_1_3_khr_maintenance4.inc>
    }
  }
  if (device->extensions_.ext_KHR_swapchain) {
#include <rex/ui/vulkan/functions/device_khr_swapchain.inc>
  }
  if (device->extensions_.ext_NV_device_diagnostic_checkpoints) {
#include <rex/ui/vulkan/functions/device_nv_device_diagnostic_checkpoints.inc>
  }
  if (device->extensions_.ext_EXT_device_fault) {
#include <rex/ui/vulkan/functions/device_ext_device_fault.inc>
  }
  if (device->extensions_.ext_EXT_transform_feedback) {  // wwe2k14 patch (Session 24)
#include <rex/ui/vulkan/functions/device_ext_transform_feedback.inc>
  }
  if (device->extensions_.ext_EXT_extended_dynamic_state) {  // wwe2k14 patch (Session 93, idea 7)
#include <rex/ui/vulkan/functions/device_ext_extended_dynamic_state.inc>
  }
  if (device->extensions_.ext_GOOGLE_display_timing) {  // wwe2k14 patch (Session 93)
#include <rex/ui/vulkan/functions/device_google_display_timing.inc>
  }
#undef XE_UI_VULKAN_FUNCTION_PROMOTED

#undef XE_UI_VULKAN_FUNCTION

  if (!functions_loaded) {
    REXLOG_ERROR("Failed to get all Vulkan device function pointers for '{}'",
                 properties.deviceName);
    return nullptr;
  }

  // wwe2k14 patch (Session 55): vulkan_debug_check_buffer_offsets (one device per process).
  if (const int32_t align = REXCVAR_GET(vulkan_debug_check_buffer_offsets); align > 0 && !g_real_update_descriptor_sets) {
    g_check_alignment = VkDeviceSize(align);
    g_real_update_descriptor_sets = dfn.vkUpdateDescriptorSets;
    g_real_create_buffer_view = dfn.vkCreateBufferView;
    dfn.vkUpdateDescriptorSets = CheckedUpdateDescriptorSets;
    dfn.vkCreateBufferView = CheckedCreateBufferView;
    REXLOG_INFO("[vulkan][offsets] checking buffer descriptor and texel buffer view offsets against {} bytes (this "
                "device: storage {}, uniform {}, texel {})", align, properties.limits.minStorageBufferOffsetAlignment,
                properties.limits.minUniformBufferOffsetAlignment, properties.limits.minTexelBufferOffsetAlignment);
  }

  WriteDeviceReport(vulkan_instance, physical_device, properties, enabled_extensions, with_swapchain);  // Session 95

  // Get the queues.

  for (size_t queue_family_index = 0; queue_family_index < device->queue_families_.size();
       ++queue_family_index) {
    QueueFamily& queue_family = device->queue_families_[queue_family_index];
    for (size_t queue_index = 0; queue_index < queue_family.queues.size(); ++queue_index) {
      VkQueue queue;
      dfn.vkGetDeviceQueue(device->device_, uint32_t(queue_family_index), uint32_t(queue_index),
                           &queue);
      queue_family.queues[queue_index] = std::make_unique<Queue>(queue);
    }
  }

  // Get the memory types.

  VkPhysicalDeviceMemoryProperties memory_properties;
  ifn.vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
  for (uint32_t memory_type_index = 0; memory_type_index < memory_properties.memoryTypeCount;
       ++memory_type_index) {
    const uint32_t memory_type_bit = uint32_t(1) << memory_type_index;
    const VkMemoryPropertyFlags memory_type_flags =
        memory_properties.memoryTypes[memory_type_index].propertyFlags;
    if (memory_type_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
      device->memory_types_.device_local |= memory_type_bit;
    }
    if (memory_type_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
      device->memory_types_.host_visible |= memory_type_bit;
    }
    if (memory_type_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
      device->memory_types_.host_coherent |= memory_type_bit;
    }
    if (memory_type_flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) {
      device->memory_types_.host_cached |= memory_type_bit;
    }
  }

  return device;
}

VulkanDevice::~VulkanDevice() {
  streamline::Shutdown();  // wwe2k14 patch (Session 18, DLAA): before the device goes (no-op when unused)
  if (device_) {
    vulkan_instance_->functions().vkDestroyDevice(device_, nullptr);
  }
}

VulkanDevice::VulkanDevice(const VulkanInstance* const vulkan_instance,
                           const VkPhysicalDevice physical_device)
    : vulkan_instance_(vulkan_instance), physical_device_(physical_device) {
  assert_not_null(vulkan_instance);
  assert_not_null(physical_device);
}

}  // namespace vulkan
}  // namespace ui
}  // namespace rex
