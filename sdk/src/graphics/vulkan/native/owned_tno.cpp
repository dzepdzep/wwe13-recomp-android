// wwe2k14 patch (Session 92, M2d, Part 2): the REX_TRUE_NATIVE_ONLY proof build. With the option ON the truth path (native_truth.cpp: recorder, stores, staging, twins, deferred
// copies, materialise, write-out, replay) is not compiled. These are its entry points that code which stays compiled still names (the command processor's Native* wrappers
// of the stock resolve / draw paths, the owned frame's replay and materialise branches, the frame boundary). Every one is DEAD under the owned-only latch (which the proof build
// is: TRUE NATIVE never leaves it) and is defined here with the "no truth path" meaning - nothing recorded, nothing served, nothing redundant - or, when the answer is not
// neutral, as a marker that logs once. Nothing in this file runs in the normal build (the whole file is empty there): the proof is that the list is this short.
// The proof build cannot run the default NATIVE renderer.

#include <rex/graphics/vulkan/native/native_renderer.h>

#if REX_TRUE_NATIVE_ONLY

#include <rex/graphics/vulkan/command_processor.h>
#include <rex/logging.h>

namespace rex::graphics::vulkan::native {

namespace {
// A truth function a dead branch still names was reached: loud, a few times.
void Reached(const char* what) {
  static uint32_t logged = 0;
  if (++logged <= 12) REXGPU_ERROR("[native][owned-only] REX_TRUE_NATIVE_ONLY: {} was reached - dead under the latch, not compiled in this build (nothing done)", what);
}
}  // namespace

// ---- the command processor's resolve / draw wrappers (the stock paths: dead under the latch, where OwnedTakeResolve takes every copy and no draw reaches the stock pipelines)
void NativeRenderer::AliasNoteLoad(uint32_t, uint32_t) {}
void NativeRenderer::AliasNotePath(uint32_t) {}
void NativeRenderer::AliasNoteSampled(void*, bool) {}
float NativeRenderer::TruthDrawRatio() { return 1.0f; }
void NativeRenderer::TruthAfterResolveMarked(const draw_util::ResolveInfo&) {}
void NativeRenderer::TruthBeforeResolveWrite(const draw_util::ResolveInfo&) {}
void NativeRenderer::TruthResolveOverlaps(const draw_util::ResolveInfo&, void*, const int32_t*, int32_t) {}
bool NativeRenderer::TruthFillOnly(const draw_util::ResolveInfo&, const draw_util::ResolveCopyShaderConstants&) { return false; }
void NativeRenderer::TruthFillTexture(const draw_util::ResolveInfo&) {}
void NativeRenderer::TruthMaterialize(uint32_t, uint32_t, uint32_t) {}
void NativeRenderer::TruthMaterializeAll(uint32_t) {}
void NativeRenderer::TruthMaterializeScaled(uint32_t, uint32_t) {}
void NativeRenderer::TruthNoteWriteBack(bool, uint32_t) {}
bool NativeRenderer::TruthResolveDirect(const draw_util::ResolveInfo&, const draw_util::ResolveCopyShaderConstants&, VkDescriptorSet, uint32_t, uint32_t) { return false; }
bool NativeRenderer::TruthResolveFull1x(const draw_util::ResolveInfo&, uint32_t, uint32_t, uint32_t, VkDescriptorSet) { return false; }
bool NativeRenderer::TruthResolveRedundant(const draw_util::ResolveInfo&) { return false; }
bool NativeRenderer::TruthResolveScaledFull(const draw_util::ResolveInfo&, VkDescriptorSet) { return false; }
VkDescriptorSet NativeRenderer::TruthResolveSource(const draw_util::ResolveInfo&) { return VK_NULL_HANDLE; }
bool NativeRenderer::TruthRestoreRedundantEarly(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
VkImageView NativeRenderer::TruthServeView(const void*, uint32_t, uint32_t, VkFormat, const VkComponentMapping&, const ServeTargets&) { return VK_NULL_HANDLE; }
VkDescriptorSet NativeRenderer::TruthStageSurface(uint32_t, uint32_t, uint32_t) { return VK_NULL_HANDLE; }
void NativeRenderer::TruthStampSubmission() {}
void NativeRenderer::TruthTimer(uint32_t, bool) {}

// ---- the frame boundary and the shared bookkeeping
void NativeRenderer::AliasOnFrameEnd() {}
void NativeRenderer::AliasStart(int32_t) {}
void NativeRenderer::EndTruthPass() {}  // (no truth pass is ever open)
void NativeRenderer::LogTruthCensus() {}
void NativeRenderer::LogTruthPasses(uint64_t, const FrameSlot&, const std::vector<uint64_t>&, double) {}
void NativeRenderer::RetireTruthAll() {}
void NativeRenderer::TruthDropRecords() {}
void NativeRenderer::TruthFlushDeferredSlow(uint32_t, const DeferConcern&) {}
void NativeRenderer::TruthOnFrameEnd(bool, bool) {}
void NativeRenderer::TruthStorageOpen(const std::filesystem::path&, uint32_t) {}
void NativeRenderer::TruthWriteOut() {}

// ---- reached only through the latch-off branches of the owned frame (replay, materialise, truth-image claims): markers
bool NativeRenderer::ClaimTruth(Draw&) { Reached("ClaimTruth"); return false; }
bool NativeRenderer::EnsureTruthImage(int16_t, uint32_t) { Reached("EnsureTruthImage"); return false; }
bool NativeRenderer::EnsureTruthObjects() { return true; }  // (the frame boundary asks; the proof build has no truth objects to make)
void NativeRenderer::TruthFallback(uint32_t, const Draw*) { Reached("TruthFallback"); }
int16_t NativeRenderer::TruthFindImage(uint32_t, uint32_t, uint32_t, uint32_t) { Reached("TruthFindImage"); return -1; }
void NativeRenderer::TruthRecordDraw(const Draw&) { Reached("TruthRecordDraw"); }
void NativeRenderer::TruthSetOwners(const std::bitset<2048>&, int16_t, const std::bitset<2048>&, int16_t) { Reached("TruthSetOwners"); }
void NativeRenderer::TruthTileRects(const TruthImage&, const std::bitset<2048>&, std::vector<std::array<int32_t, 4>>&) const { Reached("TruthTileRects"); }

}  // namespace rex::graphics::vulkan::native

#endif  // REX_TRUE_NATIVE_ONLY
