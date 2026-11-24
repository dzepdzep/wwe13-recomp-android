// wwe2k14 patch (Session 92, M2b): the reasons a draw is refused / a truth frame ends early, shared by native_truth.cpp and owned_entry.cpp.
// An internal header of native\ - the game exe does not include it.
#pragma once
#include <cstdint>

namespace rex::graphics::vulkan::native {

// Reasons a truth frame ends early (TruthReason, TruthFallback).
enum TruthReasonCode : uint32_t {
  kTruthOk = 0,
  kTruthEmulated,       // an emulated draw follows (a draw nobody claimed: captured positions, a repeat, ...)
  kTruthClearingResolve,
  kTruthMemexport,
  kTruthPrimitive,
  kTruthMsaa,           // an MSAA draw that is not a fast clear
  kTruthOtherTargets,
  kTruthColorFormat,
  kTruthStencil,        // a stencil test (not a fast clear's)
  kTruthOcclusion,
  kTruthNoShader,
  kTruthNoPipeline,
  kTruthBlend16,        // blending into a 16-bit image whose format cannot blend
  kTruthImages,         // too many images
  kTruthStaging,        // the staging buffer could not be made
  kTruthDebug,          // gpu_native_truth_debug_draws
  kTruthReasonCount,
};

}  // namespace rex::graphics::vulkan::native
