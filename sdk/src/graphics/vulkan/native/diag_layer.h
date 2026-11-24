// wwe2k14 patch (Session 95, the diagnostic run; out\s95\diag_design.md sections 3-5): the GPU side of the report.
//
// gpu_diag_layer=<folder> (set by the launchers' DIAGNOSTIC RUN for each arm, read once in SetupContext) puts thunks into the device's
// function table, so images, views, framebuffers, render passes, shader modules and pipelines are known by handle whichever renderer
// made them. Nothing else happens until the game asks for one frame (gpu_diag_capture="<seq>:<scene>:<mode>", on a frozen frame):
//   mode 1 sheet:    at each pass end the pass's target is blitted into a cell of an atlas (one mechanism for emulated, NATIVE and TRUE
//                    NATIVE: the hook is in ExecuteStream, not in the draw paths), read back once at the frame's end, stats + hashes per
//                    cell in sheet_<scene>.json, the pixels in sheet_<scene>.rgba (the game turns them into a JPEG: diag_jpeg.h).
//   mode 2 truth:    the sampled draws: Bytes (the fetch ranges as the GPU's shared memory holds them at the pass end against the guest
//                    bytes at the draw) and Coverage (an occlusion query of our own around the real draw), and the CPU's positions
//                    (ShaderInterpreter on the same inputs): truth_<scene>.json.
//   mode 4 numbers:  the sampled draws run again after their pass with the vertex module patched (diag_spirv.h) into a 256 x 256 uint target:
//                    the real positions, bit-exact, against the interpreter's; no stores, so it works on Mali-G57.
//   mode 8 solo:     the same draws with their real module, one silhouette each.
//   mode 16 pick:    this frame's draws choose the samples for later truth / numbers frames.
// Everything but the pass-end work is read-only. Off: one bool test per replayed stream, one per draw at the record sites.
//
// Draw ordering: the frame's boundaries and the sampled draws travel in the stream itself (DeferredCommandBuffer::kDiagMark), so the replay
// thread (vulkan_async_submit) sees them in order with no shared state to race on.
#pragma once

#include <atomic>
#include <cstdint>

#include <rex/ui/vulkan/api.h>

namespace rex::graphics {
class Shader;
}

namespace rex::graphics::vulkan {
class VulkanCommandProcessor;
namespace native {
struct Draw;
}

namespace diag {

extern bool g_latched;                 // gpu_diag_layer was set (thunks in)
extern bool g_armed;                   // command thread: the frame being recorded is captured (the draw sites' one test)
extern std::atomic<bool> g_observing;  // a capture is in flight: ExecuteStream takes its observing body (its one test)

enum MarkKind : uint32_t { kMarkFrameOpen = 1, kMarkFrameClose, kMarkNextDraw, kMarkPrevDraw };
constexpr uint32_t kAuxFamilyMask = 0xF, kAuxSampled = 0x10, kAuxQuery = 0x20;

// ---- command thread
void Install(VulkanCommandProcessor& cp);    // SetupContext, before anything is created
void Shutdown();                             // ShutdownContext: waits (bounded) for the detached workers; the capture objects are freed with the device
void OnFrameOpen(VulkanCommandProcessor& cp);   // BeginSubmission, opening a frame
void OnFrameClose(VulkanCommandProcessor& cp);  // EndSubmission, closing a frame, after the last render pass ended

// What a draw site knows. floats = 256 vec4 of the vertex bank, bool_loop = 8 + 32 words, fetch = 32 x 6 words.
struct DrawInputs {
  native::Draw* draw = nullptr;  // diag_id / diag_aux are written
  const Shader* vertex_shader = nullptr;
  const uint32_t* floats = nullptr;
  const uint32_t* ps_floats = nullptr;  // (Session 95: 256 vec4 of the pixel bank, for the skin material's rows in the report)
  const uint32_t* bool_loop = nullptr;
  const uint32_t* fetch = nullptr;
  uint32_t vs_const = 0;
  bool indexed = false;
  uint32_t guest_index_base = 0, index_count = 0;
  bool index_32bit = false;
  uint32_t index_endian = 0;
  bool inline_draw = false;  // an inline ring draw: its vertices are not in the shared memory
};
void NoteDraw(VulkanCommandProcessor& cp, const DrawInputs& in);
void NoteRepeat(native::Draw& draw);  // a repeat of the base draw: counted, its coverage queried if a wrestler's, never sampled
// A draw with no guest inputs here (our crowd's instanced call): light, numbers only. Returns the mark's id, 0 = none; aux out.
uint32_t NoteOwnDraw(uint32_t family, uint32_t vertex_count, uint32_t& aux);

// ---- the replay thread (DeferredCommandBuffer::ExecuteStreamT<true>)
void StreamBegin(VkCommandBuffer cb);
void StreamEnd(VkCommandBuffer cb);
void Mark(VkCommandBuffer cb, uint32_t kind, uint32_t id, uint32_t aux);
void BeginPass(VkCommandBuffer cb, VkRenderPass render_pass, VkFramebuffer framebuffer);
void BeginDynamicPass(VkCommandBuffer cb, const VkRenderingInfo& info);
void EndPass(VkCommandBuffer cb);
void BindPipeline(VkPipelineBindPoint point, VkPipeline pipeline);
void BindSets(VkPipelineBindPoint point, VkPipelineLayout layout, uint32_t first, uint32_t count, const VkDescriptorSet* sets,
              uint32_t dynamic_count, const uint32_t* dynamic);
void BindIndexBuffer(VkBuffer buffer, VkDeviceSize offset, VkIndexType type);
void BindVertexBuffers(uint32_t first, uint32_t count, const VkBuffer* buffers, const VkDeviceSize* offsets);
void PushConstants(VkPipelineLayout layout, VkShaderStageFlags stages, uint32_t offset, uint32_t size, const void* data);
void SetViewport(uint32_t first, uint32_t count, const VkViewport* viewports);
void SetScissor(uint32_t first, uint32_t count, const VkRect2D* scissors);
void GameQuery(bool begin);
void BeforeDraw(VkCommandBuffer cb, bool indexed, uint32_t count, uint32_t instances, uint32_t first, int32_t vertex_offset,
                uint32_t first_instance);
void AfterDraw(VkCommandBuffer cb);

}  // namespace diag
}  // namespace rex::graphics::vulkan
