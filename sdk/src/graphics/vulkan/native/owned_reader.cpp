// wwe2k14 patch (Session 82, the owned match frame, P0): the extension point for the readers of an owned page the base
// cases (OwnedView: size, format, byte order, scale and class as the copy wrote it) refuse - depth views, mip ranges,
// k_4_4_4_4 views. A later piece defines it here, in this file. Docs: docs\native\OWNED_FRAME_DESIGN.md section 3.5.

#include <rex/graphics/vulkan/native/native_renderer.h>

namespace rex::graphics::vulkan::native {

VkImageView NativeRenderer::OwnedReaderExtra(const OwnedReaderContext& ctx) {
  // (No other reader yet: null = the base cases' refusal stands.)
  (void)ctx;
  return VK_NULL_HANDLE;
}

}  // namespace rex::graphics::vulkan::native
