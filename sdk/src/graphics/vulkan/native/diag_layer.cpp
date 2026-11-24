// wwe2k14 patch (Session 95, the diagnostic run): see diag_layer.h and out\s95\diag_design.md sections 3-5.
//
// Three threads touch this file:
//   the command thread (Install, OnFrameOpen / Close, NoteDraw): arms a frame, notes its draws, snapshots the sampled draws' guest inputs;
//   the replay thread (the hooks of ExecuteStreamT<true>; the submit thread with vulkan_async_submit): shadows what is bound, records
//     the pass-end work into the frame's own command buffers;
//   a worker (one at a time): waits for the frame's event, reads the results back, runs the interpreter, writes the files.
// A capture (struct Capture) is handed from one to the next through the stream's marks: nothing is read while another thread writes it,
// except the snapshots (under the mutex).
#include "diag_layer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <rex/cvar.h>
#include <rex/graphics/pipeline/shader/interpreter.h>
#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/native/native_renderer.h>
#include <rex/graphics/xenos.h>
#include <rex/hash.h>
#include <rex/logging.h>
#include <rex/memory.h>
#include <rex/ui/vulkan/instance.h>
#include <rex/ui/vulkan/util.h>

#include <fmt/format.h>

#include "diag_spirv.h"
#include "shader_swap.h"

REXCVAR_DEFINE_STRING(gpu_diag_layer, "", "GPU/Native",
                      "Session 95 (the diagnostic run): the folder for the per-pass sheets and the vertex truth; set = the layer's thunks go "
                      "into the device's function table at start. Empty = off (the default). Set by the launchers' DIAGNOSTIC RUN")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_STRING(gpu_diag_capture, "", "GPU/Native",
                      "Session 95 (the diagnostic run): <seq>:<scene>:<mode>, one frame to observe (mode bits: 1 sheet, 2 truth, 4 numbers, 8 solo, "
                      "16 pick the samples); a new value arms the next frame. Written by the game on a frozen frame");
REXCVAR_DEFINE_INT32(gpu_diag_capture_seq, 0, "GPU/Native",
                     "Session 95 (the diagnostic run): written after gpu_diag_capture: a new number is what arms the frame (the text is read only then, "
                     "while the game waits for gpu_diag_capture_done, so nothing reads it while it changes)");
REXCVAR_DEFINE_INT32(gpu_diag_capture_done, 0, "GPU/Native",
                     "Session 95 (the diagnostic run, read-out only): the last <seq> whose files are written");

namespace rex::graphics::vulkan {

// The friend of VulkanCommandProcessor (command_processor.h): a friend declaration only, no member, no layout change.
struct DiagAccess {
  using VCP = VulkanCommandProcessor;
  static const SpirvShaderTranslator::SystemConstants& Sys(const VCP& cp) { return cp.system_constants_; }
  static VkBuffer SharedBuffer(const VCP& cp) { return cp.shared_memory_->buffer(); }
  static memory::Memory* Memory(const VCP& cp) { return cp.memory_; }
  static bool WindowOn(const VCP& cp) { return cp.window_on_; }
  static uint32_t WindowCur(const VCP& cp) { return cp.window_cur_; }
  static uint32_t WindowSize(const VCP& cp) { return cp.window_size_; }
  static uint64_t Frame(const VCP& cp) { return cp.frame_current_; }
};

namespace diag {

bool g_latched = false;
bool g_armed = false;
std::atomic<bool> g_observing{false};

namespace {

using Fn = ui::vulkan::VulkanDevice::Functions;

constexpr uint32_t kModeSheet = 1, kModeTruth = 2, kModeNumbers = 4, kModeSolo = 8, kModePick = 16;
constexpr uint32_t kCellW = 160, kCellH = 90, kCellCols = 16, kCellRows = 8, kMaxCells = kCellCols * kCellRows;
constexpr uint32_t kAtlasW = kCellW * kCellCols, kAtlasH = kCellH * kCellRows;
constexpr uint32_t kMaxTimestamps = 512, kMaxQueries = 512;
constexpr uint32_t kMaxSamples = 16;  // the numbers / solo slots: 12 by family + 4 translated
constexpr uint32_t kBytesRangeMax = 64 * 1024, kBytesRangesPerDraw = 4, kBytesStaging = kMaxSamples * kBytesRangeMax * kBytesRangesPerDraw;
constexpr uint32_t kCapturePixels = kCaptureSize * kCaptureSize;
constexpr uint32_t kNumbersSlotBytes = kCapturePixels * 20;  // positions (uvec4) then ids (uint)
constexpr uint32_t kSoloCols = 4, kSoloSlotBytes = kCellW * kCellH * 4;
constexpr uint32_t kCpuVertices = 2000, kWorstRows = 16;
#ifdef __ANDROID__
constexpr size_t kSpirvKeepMax = size_t(48) << 20;  // a phone: an out-of-memory kill would read as a game crash
#else
constexpr size_t kSpirvKeepMax = size_t(192) << 20;
#endif
constexpr const char* kFamilyNames[] = {"bg", "prop", "particle", "wrestler", "crowd", "shadow", "blur", "other", "translated"};
constexpr uint32_t kFamilyTranslated = 8, kFamilyCount = 9, kFamilyWrestler = 3, kFamilyCrowd = 4;

template <class H>
uint64_t K(H handle) {
  return (uint64_t)handle;
}

// ---- what the thunks know, by handle
struct ImageInfo {
  VkFormat format = VK_FORMAT_UNDEFINED;
  uint32_t width = 0, height = 0, serial = 0;
  VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
  VkImageUsageFlags usage = 0;
};
struct ViewInfo {
  VkImage image = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  uint32_t mip = 0, layer = 0;
  VkImageAspectFlags aspect = 0;
};
struct AttachmentInfo {
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
  VkImageLayout final_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkAttachmentStoreOp store = VK_ATTACHMENT_STORE_OP_STORE;
};
struct RenderPassInfo {
  std::vector<AttachmentInfo> attachments;
  std::vector<int32_t> color, resolve;  // the last subpass's references (-1 unused)
  int32_t depth = -1;
  bool multiview = false;
};
struct FramebufferInfo {
  std::vector<VkImageView> views;
  uint32_t width = 0, height = 0;
};
struct ModuleInfo {
  uint64_t hash = 0;
  std::shared_ptr<const std::vector<uint32_t>> words;
};
struct SpecInfo {
  std::vector<VkSpecializationMapEntry> entries;
  std::vector<uint8_t> data;
};
struct PipelineInfo {
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkShaderStageFlags stages = 0;
  uint64_t vertex_hash = 0, fragment_hash = 0;
  std::shared_ptr<const std::vector<uint32_t>> vertex_words;
  std::string entry = "main";
  std::shared_ptr<SpecInfo> spec;
  std::vector<VkVertexInputBindingDescription> bindings;
  std::vector<VkVertexInputAttributeDescription> attributes;
  VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  bool restart = false;
  std::vector<VkDynamicState> dynamics;
};
// The debug pipelines of a sampled draw's pipeline.
struct Debug {
  VkPipeline numbers = VK_NULL_HANDLE, solo = VK_NULL_HANDLE;
  std::string why_numbers, why_solo;
};

// ---- the frame
struct BindCall {
  VkPipelineLayout layout = VK_NULL_HANDLE;
  uint32_t first = 0, count = 0, dynamic_count = 0;
  std::array<VkDescriptorSet, 8> sets{};
  std::array<uint32_t, 16> dynamic{};
};
struct PushRec {
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkShaderStageFlags stages = 0;
  uint32_t offset = 0, size = 0;
  std::array<uint8_t, 256> data{};
};
// Everything a draw needs bound, as the command buffer had it.
struct Bound {
  VkPipeline pipeline = VK_NULL_HANDLE;
  std::vector<BindCall> calls;  // in order; a call none of whose sets is current any more is dropped
  std::array<int32_t, 8> call_of_set{-1, -1, -1, -1, -1, -1, -1, -1};
  VkBuffer index_buffer = VK_NULL_HANDLE;
  VkDeviceSize index_offset = 0;
  VkIndexType index_type = VK_INDEX_TYPE_UINT16;
  std::array<VkBuffer, 4> vertex_buffers{};
  std::array<VkDeviceSize, 4> vertex_offsets{};
  std::vector<PushRec> pushes;
  bool has_viewport = false, has_scissor = false;
  VkViewport viewport{};
  VkRect2D scissor{};
};
struct DrawCall {
  bool indexed = false;
  uint32_t count = 0, instances = 1, first = 0, first_instance = 0;
  int32_t vertex_offset = 0;
};

struct Source {  // what a pass's picture is taken from
  VkImage image = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  uint32_t mip = 0, layer = 0, width = 0, height = 0, samples = 1;
};
struct PassRec {
  uint32_t index = 0, draws = 0, target = 0, colors = 0;
  uint32_t family[kFamilyCount] = {};
  uint32_t first_id = 0;
  uint64_t vk_vs = 0, vk_fs = 0;
  const char* kind = "unknown";  // color, msaa, depth, int, noblit, notstored, transient, overflow, nodraws, unknown
  Source source;
  bool multiview = false, dynamic = false;
  int32_t cell = -1, timestamp = -1;
};
struct Light {
  uint32_t id = 0, count = 0, nth = 0;
  uint64_t vs = 0, ps = 0;
  uint8_t family = kFamilyTranslated;
  bool memexport = false, repeat = false;
};
struct Range {
  uint32_t address = 0, size = 0, staging = 0, fetch = 0;
  std::vector<uint8_t> guest;
};
struct Snapshot {
  uint32_t id = 0;
  const uint32_t* ucode = nullptr;
  bool can_interpret = false, inline_draw = false, attr = false, memexport = false, window_on = false;  // (attr, Session 95: an attribute draw: gl_VertexIndex already is the guest index)
  uint32_t window = 0, window_size = 0;
  std::array<uint32_t, 1024> floats{};
  std::array<uint32_t, 16> ps_rows{};  // (Session 95: c190, c234, c246 and c218 of the pixel bank)
  std::array<uint32_t, 40> bool_loop{};
  std::array<uint32_t, 192> fetch{};
  uint32_t vs_const = 0;
  uint32_t flags = 0, endian = 0, load_address = 0, index_min = 0, index_max = 0;
  int32_t base = 0;
  float ndc_scale[3] = {}, ndc_offset[3] = {};
  std::vector<Range> ranges;
  std::vector<uint32_t> cpu_indices;  // guest indices (post swap, offset and clamp) to run on the CPU
  uint64_t input_hash = 0;
  uint32_t host_vs_type = 0;
  bool gpu_written = false;  // a fetched range holds data the GPU wrote (resolve / memexport): the guest memory is not what the shader read
};
struct Run {  // a draw this frame looks at
  uint32_t id = 0, aux = 0, pass = 0;
  int32_t query = -1, numbers_slot = -1, solo_slot = -1;
  bool have_state = false, debug_done = false;
  Bound state;
  DrawCall call;
  uint32_t target_w = 0, target_h = 0;
  const char* skip_query = nullptr;
  std::string skip_numbers, skip_solo;
};
struct Capture {
  uint32_t seq = 0, mode = 0;
  std::string scene;
  uint64_t frame = 0;
  std::chrono::steady_clock::time_point start;
  // command thread
  std::vector<Light> draws;
  std::unordered_map<uint64_t, uint32_t> occurrences;
  std::unordered_set<uint64_t> picks;
  uint32_t picks_found = 0, bytes_used = 0;
  // under G::m (the replay thread looks a sampled draw up)
  std::unordered_map<uint32_t, std::unique_ptr<Snapshot>> snapshots;
  // replay thread
  std::vector<PassRec> passes;
  std::vector<Run> runs;
  std::unordered_map<uint32_t, VkPipeline> pipelines;  // id -> as drawn
  uint32_t cells = 0, timestamps = 0, queries = 0, numbers_slots = 0, solo_slots = 0, unmarked_draws = 0, overflow = 0;
};

struct G {
  VulkanCommandProcessor* cp = nullptr;
  const ui::vulkan::VulkanDevice* dev = nullptr;
  VkDevice device = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  Fn real;
  PFN_vkGetPhysicalDeviceFormatProperties format_properties = nullptr;
  PFN_vkCreateEvent create_event = nullptr;
  PFN_vkGetEventStatus event_status = nullptr;
  PFN_vkResetEvent reset_event = nullptr;
  PFN_vkCmdSetEvent cmd_set_event = nullptr;
  PFN_vkCmdResolveImage cmd_resolve = nullptr;
  std::filesystem::path dir;
  float timestamp_period = 0.0f;
  uint32_t timestamp_bits = 0;
  bool precise = false;
  std::atomic<bool> stop{false};
  std::atomic<int> workers{0};  // detached workers alive (Finish, BuildDebug): Shutdown waits for them
  uint32_t max_locations = 16;

  std::mutex m;  // the maps below + Capture::snapshots
  std::unordered_map<uint64_t, ImageInfo> images;
  std::unordered_map<uint64_t, ViewInfo> views;
  std::unordered_map<uint64_t, RenderPassInfo> render_passes;
  std::unordered_map<uint64_t, FramebufferInfo> framebuffers;
  std::unordered_map<uint64_t, ModuleInfo> modules;
  std::unordered_map<uint64_t, std::shared_ptr<PipelineInfo>> pipelines;
  std::unordered_map<uint64_t, Debug> debug;
  std::atomic<size_t> spirv_kept{0};  // words alive; the shared_ptr deleter subtracts, so words a pipeline still holds still count
  uint32_t spirv_dropped = 0, image_serial = 0, images_usage_added = 0;
  std::unordered_map<uint32_t, VkFormatProperties> format_cache;

  // capture objects (made at the first capture)
  bool ready = false;
  VkImage atlas = VK_NULL_HANDLE;
  VkDeviceMemory atlas_memory = VK_NULL_HANDLE, atlas_buffer_memory = VK_NULL_HANDLE, bytes_memory = VK_NULL_HANDLE;
  VkBuffer atlas_buffer = VK_NULL_HANDLE, bytes_buffer = VK_NULL_HANDLE;
  void* atlas_map = nullptr;
  void* bytes_map = nullptr;
  VkQueryPool timestamp_pool = VK_NULL_HANDLE, occlusion_pool = VK_NULL_HANDLE;
  VkEvent event = VK_NULL_HANDLE;
  std::map<std::tuple<uint32_t, uint32_t, uint32_t>, VkImage> resolve_images;
  // numbers / solo (made when first asked for)
  bool debug_ready = false;
  std::string debug_why;
  VkRenderPass numbers_pass = VK_NULL_HANDLE, solo_pass = VK_NULL_HANDLE;
  VkFramebuffer numbers_fb = VK_NULL_HANDLE, solo_fb = VK_NULL_HANDLE;
  VkImage numbers_pos = VK_NULL_HANDLE, numbers_id = VK_NULL_HANDLE, solo_image = VK_NULL_HANDLE;
  VkBuffer numbers_buffer = VK_NULL_HANDLE, solo_buffer = VK_NULL_HANDLE;
  void* numbers_map = nullptr;
  void* solo_map = nullptr;
  VkDeviceMemory numbers_buffer_memory = VK_NULL_HANDLE, solo_buffer_memory = VK_NULL_HANDLE;
  VkShaderModule solo_fs = VK_NULL_HANDLE;
  std::map<uint32_t, VkShaderModule> capture_fs;

  // the capture cycle (command thread, but `busy` is cleared by the worker)
  std::atomic<bool> busy{false};
  std::atomic<bool> prep_done{false};
  bool prep = false;
  std::string pending_request;
  std::chrono::steady_clock::time_point pending_at;  // when the debug pipelines were asked for: an old request is dropped (it would arm a live frame, after the game gave up)
  int32_t parity = -1;                                // the frame parity of the first capture: later ones arm on the same (LOW phones change the frame's shape every other frame)
  int32_t last_seq = 0;
  std::atomic<Capture*> cap{nullptr};
  std::vector<uint64_t> pick_keys;                 // written by the worker, read at arming (busy is false then)
  std::vector<VkPipeline> pick_pipelines;
  uint32_t captures = 0;
};
G* g = nullptr;

// ---- the thunks (any thread)
VkFormatProperties FormatProps(VkFormat format) {
  std::lock_guard<std::mutex> l(g->m);
  const auto it = g->format_cache.find(uint32_t(format));
  if (it != g->format_cache.end()) return it->second;
  VkFormatProperties p{};
  g->format_properties(g->physical, format, &p);
  g->format_cache[uint32_t(format)] = p;
  return p;
}

VKAPI_ATTR VkResult VKAPI_CALL ThunkCreateImage(VkDevice device, const VkImageCreateInfo* info, const VkAllocationCallbacks* allocator, VkImage* out) {
  VkImageCreateInfo copy = *info;
  bool added = false;
  // A colour target must be a blit source for the sheet. Almost all are already (truth and owned targets); post targets and the emulated
  // render targets are not. Never on a transient attachment (it has no memory of its own to read).
  if ((copy.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) && !(copy.usage & (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)) &&
      (FormatProps(copy.format).optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT)) {
    copy.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    added = true;
  }
  const VkResult r = g->real.vkCreateImage(device, &copy, allocator, out);
  if (r == VK_SUCCESS) {
    std::lock_guard<std::mutex> l(g->m);
    ImageInfo i;
    i.format = copy.format;
    i.width = copy.extent.width;
    i.height = copy.extent.height;
    i.samples = copy.samples;
    i.usage = copy.usage;
    i.serial = ++g->image_serial;
    g->images[K(*out)] = i;
    g->images_usage_added += added;
  }
  return r;
}
VKAPI_ATTR void VKAPI_CALL ThunkDestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* allocator) {
  {
    std::lock_guard<std::mutex> l(g->m);
    g->images.erase(K(image));
  }
  g->real.vkDestroyImage(device, image, allocator);
}
VKAPI_ATTR VkResult VKAPI_CALL ThunkCreateImageView(VkDevice device, const VkImageViewCreateInfo* info, const VkAllocationCallbacks* allocator, VkImageView* out) {
  const VkResult r = g->real.vkCreateImageView(device, info, allocator, out);
  if (r == VK_SUCCESS) {
    std::lock_guard<std::mutex> l(g->m);
    ViewInfo v;
    v.image = info->image;
    v.format = info->format;
    v.mip = info->subresourceRange.baseMipLevel;
    v.layer = info->subresourceRange.baseArrayLayer;
    v.aspect = info->subresourceRange.aspectMask;
    g->views[K(*out)] = v;
  }
  return r;
}
VKAPI_ATTR void VKAPI_CALL ThunkDestroyImageView(VkDevice device, VkImageView view, const VkAllocationCallbacks* allocator) {
  {
    std::lock_guard<std::mutex> l(g->m);
    g->views.erase(K(view));
  }
  g->real.vkDestroyImageView(device, view, allocator);
}
VKAPI_ATTR VkResult VKAPI_CALL ThunkCreateRenderPass(VkDevice device, const VkRenderPassCreateInfo* info, const VkAllocationCallbacks* allocator, VkRenderPass* out) {
  const VkResult r = g->real.vkCreateRenderPass(device, info, allocator, out);
  if (r == VK_SUCCESS) {
    RenderPassInfo p;
    for (uint32_t i = 0; i < info->attachmentCount; ++i) {
      const VkAttachmentDescription& a = info->pAttachments[i];
      p.attachments.push_back({a.format, a.samples, a.finalLayout, a.storeOp});
    }
    if (info->subpassCount) {
      const VkSubpassDescription& s = info->pSubpasses[info->subpassCount - 1];
      for (uint32_t i = 0; i < s.colorAttachmentCount; ++i) {
        const uint32_t a = s.pColorAttachments[i].attachment;
        p.color.push_back(a == VK_ATTACHMENT_UNUSED ? -1 : int32_t(a));
        const uint32_t b = s.pResolveAttachments ? s.pResolveAttachments[i].attachment : VK_ATTACHMENT_UNUSED;
        p.resolve.push_back(b == VK_ATTACHMENT_UNUSED ? -1 : int32_t(b));
      }
      if (s.pDepthStencilAttachment && s.pDepthStencilAttachment->attachment != VK_ATTACHMENT_UNUSED) p.depth = int32_t(s.pDepthStencilAttachment->attachment);
    }
    for (const VkBaseInStructure* n = static_cast<const VkBaseInStructure*>(info->pNext); n; n = n->pNext) {
      if (n->sType == VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO) p.multiview = true;
    }
    std::lock_guard<std::mutex> l(g->m);
    g->render_passes[K(*out)] = std::move(p);
  }
  return r;
}
VKAPI_ATTR void VKAPI_CALL ThunkDestroyRenderPass(VkDevice device, VkRenderPass pass, const VkAllocationCallbacks* allocator) {
  {
    std::lock_guard<std::mutex> l(g->m);
    g->render_passes.erase(K(pass));
  }
  g->real.vkDestroyRenderPass(device, pass, allocator);
}
VKAPI_ATTR VkResult VKAPI_CALL ThunkCreateFramebuffer(VkDevice device, const VkFramebufferCreateInfo* info, const VkAllocationCallbacks* allocator, VkFramebuffer* out) {
  const VkResult r = g->real.vkCreateFramebuffer(device, info, allocator, out);
  if (r == VK_SUCCESS) {
    FramebufferInfo f;
    f.width = info->width;
    f.height = info->height;
    if (info->pAttachments) f.views.assign(info->pAttachments, info->pAttachments + info->attachmentCount);
    std::lock_guard<std::mutex> l(g->m);
    g->framebuffers[K(*out)] = std::move(f);
  }
  return r;
}
VKAPI_ATTR void VKAPI_CALL ThunkDestroyFramebuffer(VkDevice device, VkFramebuffer framebuffer, const VkAllocationCallbacks* allocator) {
  {
    std::lock_guard<std::mutex> l(g->m);
    g->framebuffers.erase(K(framebuffer));
  }
  g->real.vkDestroyFramebuffer(device, framebuffer, allocator);
}
VKAPI_ATTR VkResult VKAPI_CALL ThunkCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* info, const VkAllocationCallbacks* allocator, VkShaderModule* out) {
  const VkResult r = g->real.vkCreateShaderModule(device, info, allocator, out);
  if (r == VK_SUCCESS) {
    ModuleInfo m;
    m.hash = XXH3_64bits(info->pCode, info->codeSize);
    std::lock_guard<std::mutex> l(g->m);
    if (g->spirv_kept.load() + info->codeSize <= kSpirvKeepMax) {
      const size_t bytes = info->codeSize;
      m.words = std::shared_ptr<const std::vector<uint32_t>>(new std::vector<uint32_t>(info->pCode, info->pCode + info->codeSize / 4),
                                                             [bytes](const std::vector<uint32_t>* v) {
                                                               if (g) g->spirv_kept -= std::min<size_t>(g->spirv_kept.load(), bytes);
                                                               delete v;
                                                             });
      g->spirv_kept += bytes;
    } else {
      ++g->spirv_dropped;
    }
    g->modules[K(*out)] = std::move(m);
  }
  return r;
}
VKAPI_ATTR void VKAPI_CALL ThunkDestroyShaderModule(VkDevice device, VkShaderModule module, const VkAllocationCallbacks* allocator) {
  {
    std::lock_guard<std::mutex> l(g->m);
    const auto it = g->modules.find(K(module));
    if (it != g->modules.end()) {
      g->modules.erase(it);
    }
  }
  g->real.vkDestroyShaderModule(device, module, allocator);
}
VKAPI_ATTR VkResult VKAPI_CALL ThunkCreateGraphicsPipelines(VkDevice device, VkPipelineCache cache, uint32_t count, const VkGraphicsPipelineCreateInfo* infos,
                                                            const VkAllocationCallbacks* allocator, VkPipeline* out) {
  const VkResult r = g->real.vkCreateGraphicsPipelines(device, cache, count, infos, allocator, out);
  if (r != VK_SUCCESS && r != VK_PIPELINE_COMPILE_REQUIRED) return r;
  std::lock_guard<std::mutex> l(g->m);
  for (uint32_t i = 0; i < count; ++i) {
    if (out[i] == VK_NULL_HANDLE) continue;
    const VkGraphicsPipelineCreateInfo& c = infos[i];
    auto p = std::make_shared<PipelineInfo>();
    p->layout = c.layout;
    for (uint32_t s = 0; s < c.stageCount; ++s) {
      const VkPipelineShaderStageCreateInfo& st = c.pStages[s];
      p->stages |= st.stage;
      const auto m = g->modules.find(K(st.module));
      if (st.stage == VK_SHADER_STAGE_VERTEX_BIT) {
        if (m != g->modules.end()) {
          p->vertex_hash = m->second.hash;
          p->vertex_words = m->second.words;
        }
        p->entry = st.pName ? st.pName : "main";
        if (st.pSpecializationInfo) {
          p->spec = std::make_shared<SpecInfo>();
          const VkSpecializationInfo& si = *st.pSpecializationInfo;
          p->spec->entries.assign(si.pMapEntries, si.pMapEntries + si.mapEntryCount);
          const uint8_t* d = static_cast<const uint8_t*>(si.pData);
          p->spec->data.assign(d, d + si.dataSize);
        }
      } else if (st.stage == VK_SHADER_STAGE_FRAGMENT_BIT && m != g->modules.end()) {
        p->fragment_hash = m->second.hash;
        if (m->second.words) {  // the fragment stage's words are never needed: the memory goes
          m->second.words.reset();
        }
      }
    }
    if (c.pVertexInputState) {
      const VkPipelineVertexInputStateCreateInfo& v = *c.pVertexInputState;
      if (v.vertexBindingDescriptionCount) p->bindings.assign(v.pVertexBindingDescriptions, v.pVertexBindingDescriptions + v.vertexBindingDescriptionCount);
      if (v.vertexAttributeDescriptionCount) p->attributes.assign(v.pVertexAttributeDescriptions, v.pVertexAttributeDescriptions + v.vertexAttributeDescriptionCount);
    }
    if (c.pInputAssemblyState) {
      p->topology = c.pInputAssemblyState->topology;
      p->restart = c.pInputAssemblyState->primitiveRestartEnable != VK_FALSE;
    }
    if (c.pDynamicState && c.pDynamicState->dynamicStateCount) {
      p->dynamics.assign(c.pDynamicState->pDynamicStates, c.pDynamicState->pDynamicStates + c.pDynamicState->dynamicStateCount);
    }
    g->pipelines[K(out[i])] = std::move(p);
  }
  return r;
}
VKAPI_ATTR void VKAPI_CALL ThunkDestroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* allocator) {
  {
    std::lock_guard<std::mutex> l(g->m);
    g->pipelines.erase(K(pipeline));
  }
  g->real.vkDestroyPipeline(device, pipeline, allocator);
}

// ---- small helpers
bool IsDepthFormat(VkFormat f) { return f >= VK_FORMAT_D16_UNORM && f <= VK_FORMAT_D32_SFLOAT_S8_UINT; }
bool IsIntegerFormat(VkFormat f) {
  switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_B8G8R8A8_UINT: case VK_FORMAT_B8G8R8A8_SINT: case VK_FORMAT_A8B8G8R8_UINT_PACK32:
    case VK_FORMAT_A8B8G8R8_SINT_PACK32: case VK_FORMAT_A2R10G10B10_UINT_PACK32: case VK_FORMAT_A2B10G10R10_UINT_PACK32:
    case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT:
      return true;
    default:
      return false;
  }
}
std::string Hex64(uint64_t v) {
  char b[20];
  std::snprintf(b, sizeof(b), "%016llx", (unsigned long long)v);
  return b;
}
std::string Json(const std::string& s) {
  std::string o = "\"";
  for (const unsigned char c : s) {
    if (c == '"' || c == '\\') o += '\\', o += char(c);
    else if (c < 0x20) o += ' ';
    else o += char(c);
  }
  return o + "\"";
}
std::string F(double v) {  // a float for json: non-finite values have no literal
  if (!std::isfinite(v)) return "null";
  char b[40];
  std::snprintf(b, sizeof(b), "%.7g", v);
  return b;
}
std::string SafeScene(const std::string& scene) {
  std::string o;
  for (const char c : scene) o += (std::isalnum((unsigned char)c) || c == '_') ? c : '_';
  return o;
}
void WriteAtomic(const std::string& name, const void* data, size_t size) {
  std::error_code ec;
  std::filesystem::create_directories(g->dir, ec);
  const std::filesystem::path p = g->dir / name;
  std::filesystem::path part = p;
  part += ".part";
  {
    std::ofstream f(part, std::ios::binary | std::ios::trunc);
    f.write(static_cast<const char*>(data), std::streamsize(size));
  }
  std::filesystem::rename(part, p, ec);
}
bool MakeBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buffer, VkDeviceMemory& memory, void** map) {
  if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(g->dev, size, usage, ui::vulkan::util::MemoryPurpose::kReadback, buffer, memory)) return false;
  return g->real.vkMapMemory(g->device, memory, 0, VK_WHOLE_SIZE, 0, map) == VK_SUCCESS;
}
bool MakeImage(VkFormat format, uint32_t w, uint32_t h, VkImageUsageFlags usage, VkImage& image, VkDeviceMemory& memory) {
  VkImageCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = format;
  info.extent = {w, h, 1};
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = usage;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  return ui::vulkan::util::CreateDedicatedAllocationImage(g->dev, info, ui::vulkan::util::MemoryPurpose::kDeviceLocal, image, memory);
}
void Invalidate(VkDeviceMemory memory) {
  VkMappedMemoryRange r{};
  r.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  r.memory = memory;
  r.size = VK_WHOLE_SIZE;
  g->real.vkInvalidateMappedMemoryRanges(g->device, 1, &r);
}

bool EnsureCaptureObjects() {
  if (g->ready) return true;
  if (!MakeImage(VK_FORMAT_R8G8B8A8_UNORM, kAtlasW, kAtlasH, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, g->atlas, g->atlas_memory) ||
      !MakeBuffer(VkDeviceSize(kAtlasW) * kAtlasH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, g->atlas_buffer, g->atlas_buffer_memory, &g->atlas_map) ||
      !MakeBuffer(kBytesStaging, VK_BUFFER_USAGE_TRANSFER_DST_BIT, g->bytes_buffer, g->bytes_memory, &g->bytes_map)) {
    return false;
  }
  VkQueryPoolCreateInfo q{};
  q.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  q.queryType = VK_QUERY_TYPE_TIMESTAMP;
  q.queryCount = kMaxTimestamps;
  if (g->timestamp_bits && g->real.vkCreateQueryPool(g->device, &q, nullptr, &g->timestamp_pool) != VK_SUCCESS) g->timestamp_pool = VK_NULL_HANDLE;
  q.queryType = VK_QUERY_TYPE_OCCLUSION;
  q.queryCount = kMaxQueries;
  if (g->real.vkCreateQueryPool(g->device, &q, nullptr, &g->occlusion_pool) != VK_SUCCESS) g->occlusion_pool = VK_NULL_HANDLE;
  VkEventCreateInfo e{};
  e.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
  if (g->create_event(g->device, &e, nullptr, &g->event) != VK_SUCCESS) return false;
  g->ready = true;
  return true;
}

VkRenderPass MakePass(std::initializer_list<VkFormat> formats) {
  std::vector<VkAttachmentDescription> a;
  std::vector<VkAttachmentReference> refs;
  for (const VkFormat f : formats) {
    VkAttachmentDescription d{};
    d.format = f;
    d.samples = VK_SAMPLE_COUNT_1_BIT;
    d.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    d.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    d.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    d.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    d.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    d.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    refs.push_back({uint32_t(a.size()), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    a.push_back(d);
  }
  VkSubpassDescription s{};
  s.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  s.colorAttachmentCount = uint32_t(refs.size());
  s.pColorAttachments = refs.data();
  VkRenderPassCreateInfo c{};
  c.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  c.attachmentCount = uint32_t(a.size());
  c.pAttachments = a.data();
  c.subpassCount = 1;
  c.pSubpasses = &s;
  VkRenderPass pass = VK_NULL_HANDLE;
  g->real.vkCreateRenderPass(g->device, &c, nullptr, &pass);
  return pass;
}
VkImageView MakeView(VkImage image, VkFormat format) {
  VkImageViewCreateInfo v{};
  v.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  v.image = image;
  v.viewType = VK_IMAGE_VIEW_TYPE_2D;
  v.format = format;
  v.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageView view = VK_NULL_HANDLE;
  g->real.vkCreateImageView(g->device, &v, nullptr, &view);
  return view;
}
VkShaderModule MakeModule(const std::vector<uint32_t>& words) {
  VkShaderModuleCreateInfo c{};
  c.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  c.codeSize = words.size() * 4;
  c.pCode = words.data();
  VkShaderModule m = VK_NULL_HANDLE;
  g->real.vkCreateShaderModule(g->device, &c, nullptr, &m);
  return m;
}

bool EnsureDebugObjects() {
  if (g->debug_ready) return true;
  const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
  if (!(FormatProps(VK_FORMAT_R32G32B32A32_UINT).optimalTilingFeatures & need) || !(FormatProps(VK_FORMAT_R32_UINT).optimalTilingFeatures & need)) {
    g->debug_why = "the capture target's uint formats cannot be colour attachments here";
    return false;
  }
  VkDeviceMemory m0, m1, m2;
  const VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  if (!MakeImage(VK_FORMAT_R32G32B32A32_UINT, kCaptureSize, kCaptureSize, usage, g->numbers_pos, m0) ||
      !MakeImage(VK_FORMAT_R32_UINT, kCaptureSize, kCaptureSize, usage, g->numbers_id, m1) ||
      !MakeImage(VK_FORMAT_R8G8B8A8_UNORM, kCellW, kCellH, usage, g->solo_image, m2) ||
      !MakeBuffer(VkDeviceSize(kNumbersSlotBytes) * kMaxSamples, VK_BUFFER_USAGE_TRANSFER_DST_BIT, g->numbers_buffer, g->numbers_buffer_memory, &g->numbers_map) ||
      !MakeBuffer(VkDeviceSize(kSoloSlotBytes) * kMaxSamples, VK_BUFFER_USAGE_TRANSFER_DST_BIT, g->solo_buffer, g->solo_buffer_memory, &g->solo_map)) {
    g->debug_why = "the capture target could not be made";
    return false;
  }
  g->numbers_pass = MakePass({VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32_UINT});
  g->solo_pass = MakePass({VK_FORMAT_R8G8B8A8_UNORM});
  const VkImageView views[2] = {MakeView(g->numbers_pos, VK_FORMAT_R32G32B32A32_UINT), MakeView(g->numbers_id, VK_FORMAT_R32_UINT)};
  const VkImageView solo_view = MakeView(g->solo_image, VK_FORMAT_R8G8B8A8_UNORM);
  if (!g->numbers_pass || !g->solo_pass || !views[0] || !views[1] || !solo_view) {
    g->debug_why = "the capture pass could not be made";
    return false;
  }
  VkFramebufferCreateInfo f{};
  f.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  f.renderPass = g->numbers_pass;
  f.attachmentCount = 2;
  f.pAttachments = views;
  f.width = f.height = kCaptureSize;
  f.layers = 1;
  if (g->real.vkCreateFramebuffer(g->device, &f, nullptr, &g->numbers_fb) != VK_SUCCESS) {
    g->debug_why = "the capture framebuffer could not be made";
    return false;
  }
  f.renderPass = g->solo_pass;
  f.attachmentCount = 1;
  f.pAttachments = &solo_view;
  f.width = kCellW;
  f.height = kCellH;
  if (g->real.vkCreateFramebuffer(g->device, &f, nullptr, &g->solo_fb) != VK_SUCCESS) {
    g->debug_why = "the solo framebuffer could not be made";
    return false;
  }
  g->solo_fs = MakeModule(SoloFragmentModule());
  g->debug_ready = g->solo_fs != VK_NULL_HANDLE;
  return g->debug_ready;
}

// A debug pipeline: the source pipeline's layout, vertex input and dynamic state list (so binding it disturbs nothing the source would not),
// our pass, the given modules.
VkPipeline MakeDebugPipeline(const PipelineInfo& p, VkShaderModule vs, VkShaderModule fs, VkRenderPass pass, uint32_t colors, VkPrimitiveTopology topology, bool restart) {
  VkSpecializationInfo spec{};
  if (p.spec) {
    spec.mapEntryCount = uint32_t(p.spec->entries.size());
    spec.pMapEntries = p.spec->entries.data();
    spec.dataSize = p.spec->data.size();
    spec.pData = p.spec->data.data();
  }
  VkPipelineShaderStageCreateInfo stages[2] = {};
  stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vs;
  stages[0].pName = p.entry.c_str();
  stages[0].pSpecializationInfo = p.spec ? &spec : nullptr;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = fs;
  stages[1].pName = "main";
  VkPipelineVertexInputStateCreateInfo vi{};
  vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vi.vertexBindingDescriptionCount = uint32_t(p.bindings.size());
  vi.pVertexBindingDescriptions = p.bindings.data();
  vi.vertexAttributeDescriptionCount = uint32_t(p.attributes.size());
  vi.pVertexAttributeDescriptions = p.attributes.data();
  VkPipelineInputAssemblyStateCreateInfo ia{};
  ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  ia.topology = topology;
  ia.primitiveRestartEnable = restart;
  VkPipelineViewportStateCreateInfo vp{};
  vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  vp.viewportCount = vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{};
  rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode = VK_CULL_MODE_NONE;
  rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rs.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo ms{};
  ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineColorBlendAttachmentState blend[2] = {};
  blend[0].colorWriteMask = blend[1].colorWriteMask = 0xF;
  VkPipelineColorBlendStateCreateInfo cb{};
  cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  cb.attachmentCount = colors;
  cb.pAttachments = blend;
  std::vector<VkDynamicState> dynamics = p.dynamics;
  for (const VkDynamicState d : {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR}) {
    if (std::find(dynamics.begin(), dynamics.end(), d) == dynamics.end()) dynamics.push_back(d);
  }
  VkPipelineDynamicStateCreateInfo dy{};
  dy.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dy.dynamicStateCount = uint32_t(dynamics.size());
  dy.pDynamicStates = dynamics.data();
  VkGraphicsPipelineCreateInfo c{};
  c.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  c.stageCount = 2;
  c.pStages = stages;
  c.pVertexInputState = &vi;
  c.pInputAssemblyState = &ia;
  c.pViewportState = &vp;
  c.pRasterizationState = &rs;
  c.pMultisampleState = &ms;
  c.pColorBlendState = &cb;
  c.pDynamicState = &dy;
  c.layout = p.layout;
  c.renderPass = pass;
  VkPipeline out = VK_NULL_HANDLE;
  if (g->real.vkCreateGraphicsPipelines(g->device, VK_NULL_HANDLE, 1, &c, nullptr, &out) != VK_SUCCESS) return VK_NULL_HANDLE;
  return out;
}

// The worker: the debug pipelines of the picked draws (before a numbers frame is armed; a new pipeline can crash a fragile driver, so this
// runs last in an arm, and the game has written its stage marker first).
void BuildDebug(std::vector<VkPipeline> sources) {
  const bool objects = EnsureDebugObjects();
  uint32_t made = 0, refused = 0;
  for (const VkPipeline source : sources) {
    std::shared_ptr<PipelineInfo> info;
    {
      std::lock_guard<std::mutex> l(g->m);
      if (g->debug.count(K(source))) continue;
      const auto it = g->pipelines.find(K(source));
      if (it != g->pipelines.end()) info = it->second;
    }
    Debug d;
    if (!objects) {
      d.why_numbers = d.why_solo = g->debug_why;
    } else if (!info) {
      d.why_numbers = d.why_solo = "the pipeline was not seen";
    } else if (info->stages & (VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)) {
      d.why_numbers = d.why_solo = "a geometry or tessellation stage";
    } else if (!info->vertex_words) {
      d.why_numbers = d.why_solo = "its vertex module's SPIR-V was not kept";
    } else {
      PositionPatch patch;
      std::string why;
      if (PatchPositionCapture(info->vertex_words->data(), info->vertex_words->size(), g->max_locations, patch, &why)) {
        VkShaderModule fs;
        const auto it = g->capture_fs.find(patch.location);
        if (it != g->capture_fs.end()) fs = it->second;
        else g->capture_fs[patch.location] = fs = MakeModule(CaptureFragmentModule(patch.location));
        const VkShaderModule vs = MakeModule(patch.words);
        d.numbers = vs && fs ? MakeDebugPipeline(*info, vs, fs, g->numbers_pass, 2, VK_PRIMITIVE_TOPOLOGY_POINT_LIST, false) : VK_NULL_HANDLE;
        if (!d.numbers) d.why_numbers = "the patched pipeline could not be made";
        if (vs) g->real.vkDestroyShaderModule(g->device, vs, nullptr);
      } else {
        d.why_numbers = why;
      }
      const VkShaderModule plain = MakeModule(*info->vertex_words);
      d.solo = plain ? MakeDebugPipeline(*info, plain, g->solo_fs, g->solo_pass, 1, info->topology, info->restart) : VK_NULL_HANDLE;
      if (!d.solo) d.why_solo = "the solo pipeline could not be made";
      if (plain) g->real.vkDestroyShaderModule(g->device, plain, nullptr);
    }
    made += d.numbers != VK_NULL_HANDLE;
    refused += d.numbers == VK_NULL_HANDLE;
    std::lock_guard<std::mutex> l(g->m);
    g->debug[K(source)] = std::move(d);
  }
  REXGPU_INFO("[diag] debug pipelines: {} made, {} refused of {} sampled pipelines", made, refused, sources.size());
  g->prep_done = true;
}

// ---- the replay thread's state
struct Tracker {
  VkCommandBuffer cb = VK_NULL_HANDLE;
  Bound bound;
  DrawCall last_call;
  bool in_pass = false, game_query = false, query_open = false, setup_done = false, close_pending = false;
  Capture* cap = nullptr;
  uint32_t pending_id = 0, pending_aux = 0, marked_draw = 0;
};
Tracker t;

void Setup(VkCommandBuffer cb) {
  if (!t.cap || t.setup_done) return;
  t.setup_done = true;
  const Fn& f = g->real;
  if (g->timestamp_pool) f.vkCmdResetQueryPool(cb, g->timestamp_pool, 0, kMaxTimestamps);
  if (g->occlusion_pool) f.vkCmdResetQueryPool(cb, g->occlusion_pool, 0, kMaxQueries);
  if (t.cap->mode & kModeSheet) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = g->atlas;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    f.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkClearColorValue clear{};  // an empty cell is dark magenta: never mistaken for a black target
    clear.float32[0] = 0.25f, clear.float32[1] = 0.0f, clear.float32[2] = 0.25f, clear.float32[3] = 1.0f;
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    f.vkCmdClearColorImage(cb, g->atlas, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.oldLayout = b.newLayout;
    f.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
  }
}

void WriteTimestamp(VkCommandBuffer cb, PassRec& p, bool end) {
  if (p.timestamp < 0 || !g->timestamp_pool) return;
  g->real.vkCmdWriteTimestamp(cb, end ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g->timestamp_pool, uint32_t(p.timestamp) + (end ? 1 : 0));
}

// The pass's target, from the render pass's attachments and the framebuffer's views.
void ChooseSource(PassRec& p, VkRenderPass render_pass, VkFramebuffer framebuffer) {
  std::lock_guard<std::mutex> l(g->m);
  const auto rp = g->render_passes.find(K(render_pass));
  const auto fb = g->framebuffers.find(K(framebuffer));
  if (rp == g->render_passes.end() || fb == g->framebuffers.end()) return;
  p.multiview = rp->second.multiview;
  const RenderPassInfo& r = rp->second;
  int32_t attachment = -1, resolve = -1;
  p.colors = 0;
  for (size_t i = 0; i < r.color.size(); ++i) {
    if (r.color[i] < 0) continue;
    ++p.colors;
    if (attachment < 0) {
      attachment = r.color[i];
      resolve = i < r.resolve.size() ? r.resolve[i] : -1;
    }
  }
  if (attachment < 0) {
    p.kind = r.depth >= 0 ? "depth" : "none";
    return;
  }
  const int32_t use = resolve >= 0 ? resolve : attachment;
  if (size_t(use) >= fb->second.views.size() || size_t(use) >= r.attachments.size()) return;
  const auto view = g->views.find(K(fb->second.views[size_t(use)]));
  if (view == g->views.end()) return;
  const auto image = g->images.find(K(view->second.image));
  if (image == g->images.end()) return;
  const AttachmentInfo& a = r.attachments[size_t(use)];
  p.source = {view->second.image, image->second.format, a.final_layout, view->second.mip, view->second.layer, fb->second.width, fb->second.height, uint32_t(image->second.samples)};
  p.target = image->second.serial;
  p.kind = (image->second.usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) ? "transient" : a.store == VK_ATTACHMENT_STORE_OP_STORE ? "color" : "notstored";
}

void BlitCell(VkCommandBuffer cb, PassRec& p) {
  const Fn& f = g->real;
  if (std::strcmp(p.kind, "color")) return;  // (depth and integer targets: a labelled cell with no picture in v1)
  if (!p.draws) {
    p.kind = "nodraws";
    return;
  }
  const VkFormat format = p.source.format;
  if (IsDepthFormat(format)) {
    p.kind = "depth";
    return;
  }
  if (IsIntegerFormat(format)) {
    p.kind = "int";
    return;
  }
  const VkFormatProperties props = FormatProps(format);
  if (!(props.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT)) {
    p.kind = "noblit";
    return;
  }
  {
    std::lock_guard<std::mutex> l(g->m);
    const auto i = g->images.find(K(p.source.image));
    if (i == g->images.end() || !(i->second.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
      p.kind = "noblit";
      return;
    }
  }
  if (t.cap->cells >= kMaxCells) {
    ++t.cap->overflow;
    p.kind = "overflow";
    return;
  }
  p.cell = int32_t(t.cap->cells++);
  VkImageMemoryBarrier b{};
  b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  b.oldLayout = p.source.layout;
  b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = p.source.image;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, p.source.mip, 1, p.source.layer, 1};
  f.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
  VkImage source = p.source.image;
  uint32_t source_mip = p.source.mip, source_layer = p.source.layer;
  if (p.source.samples > 1) {  // a target with samples: resolved into a temporary first
    p.kind = "msaa";
    const auto key = std::make_tuple(uint32_t(format), p.source.width, p.source.height);
    VkImage temp = VK_NULL_HANDLE;
    const auto it = g->resolve_images.find(key);
    if (it != g->resolve_images.end()) {
      temp = it->second;
    } else {
      VkDeviceMemory memory;
      if (MakeImage(format, p.source.width, p.source.height, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, temp, memory)) g->resolve_images[key] = temp;
    }
    if (temp) {
      VkImageMemoryBarrier tb{};
      tb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      tb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      tb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      tb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      tb.srcQueueFamilyIndex = tb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      tb.image = temp;
      tb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      f.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &tb);
      VkImageResolve r{};
      r.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, p.source.mip, p.source.layer, 1};
      r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      r.extent = {p.source.width, p.source.height, 1};
      g->cmd_resolve(cb, p.source.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, temp, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
      tb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      tb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      tb.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      tb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      f.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &tb);
      source = temp;
      source_mip = source_layer = 0;
    } else {
      source = VK_NULL_HANDLE;
    }
  }
  if (source) {
    const uint32_t col = uint32_t(p.cell) % kCellCols, row = uint32_t(p.cell) / kCellCols;
    VkImageBlit region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, source_mip, source_layer, 1};
    region.srcOffsets[1] = {int32_t(p.source.width), int32_t(p.source.height), 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstOffsets[0] = {int32_t(col * kCellW), int32_t(row * kCellH), 0};
    region.dstOffsets[1] = {int32_t((col + 1) * kCellW), int32_t((row + 1) * kCellH), 1};
    // LINEAR only where the format says so (Mali has no filtering on RG16 / RGBA16 UNORM).
    const VkFilter filter = (props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    f.vkCmdBlitImage(cb, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g->atlas, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, filter);
  }
  b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  b.newLayout = p.source.layout;
  f.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// Bytes: at the pass end the buffers are as the pass's draws saw them (uploads end the render pass).
void CopyBytes(VkCommandBuffer cb, uint32_t pass) {
  Capture& c = *t.cap;
  std::vector<VkBufferCopy> regions;
  {
    std::lock_guard<std::mutex> l(g->m);
    for (const Run& r : c.runs) {
      if (r.pass != pass || !(r.aux & kAuxSampled)) continue;
      const auto s = c.snapshots.find(r.id);
      if (s == c.snapshots.end()) continue;
      for (const Range& range : s->second->ranges) regions.push_back({range.address, range.staging, range.size});
    }
  }
  if (regions.empty()) return;
  VkMemoryBarrier b{};
  b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  g->real.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
  g->real.vkCmdCopyBuffer(cb, DiagAccess::SharedBuffer(*g->cp), g->bytes_buffer, uint32_t(regions.size()), regions.data());
}

void BindState(VkCommandBuffer cb, const Bound& s, VkPipeline pipeline) {
  const Fn& f = g->real;
  if (pipeline) f.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
  for (const BindCall& c : s.calls) {
    f.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, c.layout, c.first, c.count, c.sets.data(), c.dynamic_count, c.dynamic_count ? c.dynamic.data() : nullptr);
  }
  for (uint32_t i = 0; i < s.vertex_buffers.size(); ++i) {
    if (s.vertex_buffers[i]) f.vkCmdBindVertexBuffers(cb, i, 1, &s.vertex_buffers[i], &s.vertex_offsets[i]);
  }
  if (s.index_buffer) f.vkCmdBindIndexBuffer(cb, s.index_buffer, s.index_offset, s.index_type);
  for (const PushRec& p : s.pushes) f.vkCmdPushConstants(cb, p.layout, p.stages, p.offset, p.size, p.data.data());
}

void DrawOnce(VkCommandBuffer cb, const DrawCall& c) {
  if (c.indexed) g->real.vkCmdDrawIndexed(cb, c.count, 1, c.first, c.vertex_offset, c.first_instance);
  else g->real.vkCmdDraw(cb, c.count, 1, c.first, c.first_instance);
}

void RunNumbers(VkCommandBuffer cb, Run& r, VkPipeline pipeline) {
  const Fn& f = g->real;
  Capture& c = *t.cap;
  if (c.numbers_slots >= kMaxSamples) {
    r.skip_numbers = "no free slot";
    return;
  }
  r.numbers_slot = int32_t(c.numbers_slots++);
  VkClearValue clear[2];
  std::memset(clear, 0xFF, sizeof(clear));  // every word all ones: "no vertex here"
  VkRenderPassBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  bi.renderPass = g->numbers_pass;
  bi.framebuffer = g->numbers_fb;
  bi.renderArea.extent = {kCaptureSize, kCaptureSize};
  bi.clearValueCount = 2;
  bi.pClearValues = clear;
  f.vkCmdBeginRenderPass(cb, &bi, VK_SUBPASS_CONTENTS_INLINE);
  BindState(cb, r.state, pipeline);
  const VkViewport vp{0.0f, 0.0f, float(kCaptureSize), float(kCaptureSize), 0.0f, 1.0f};
  const VkRect2D sc{{0, 0}, {kCaptureSize, kCaptureSize}};
  f.vkCmdSetViewport(cb, 0, 1, &vp);
  f.vkCmdSetScissor(cb, 0, 1, &sc);
  DrawOnce(cb, r.call);
  f.vkCmdEndRenderPass(cb);
  VkMemoryBarrier b{};
  b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;  // (ALL_COMMANDS: it chains with the pass's own final layout transition)
  b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  f.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = {kCaptureSize, kCaptureSize, 1};
  region.bufferOffset = VkDeviceSize(kNumbersSlotBytes) * uint32_t(r.numbers_slot);
  f.vkCmdCopyImageToBuffer(cb, g->numbers_pos, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g->numbers_buffer, 1, &region);
  region.bufferOffset += VkDeviceSize(kCapturePixels) * 16;
  f.vkCmdCopyImageToBuffer(cb, g->numbers_id, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g->numbers_buffer, 1, &region);
  b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  b.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  f.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}

void RunSolo(VkCommandBuffer cb, Run& r, VkPipeline pipeline) {
  const Fn& f = g->real;
  Capture& c = *t.cap;
  if (c.solo_slots >= kMaxSamples || !r.state.has_viewport || !r.target_w || !r.target_h) {
    r.skip_solo = "no free slot, or no viewport known";
    return;
  }
  r.solo_slot = int32_t(c.solo_slots++);
  VkClearValue clear{};
  clear.color.float32[3] = 1.0f;
  VkRenderPassBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  bi.renderPass = g->solo_pass;
  bi.framebuffer = g->solo_fb;
  bi.renderArea.extent = {kCellW, kCellH};
  bi.clearValueCount = 1;
  bi.pClearValues = &clear;
  f.vkCmdBeginRenderPass(cb, &bi, VK_SUBPASS_CONTENTS_INLINE);
  BindState(cb, r.state, pipeline);
  // the draw's own viewport, scaled from its target to the cell
  const float sx = float(kCellW) / float(r.target_w), sy = float(kCellH) / float(r.target_h);
  VkViewport vp = r.state.viewport;
  vp.x *= sx, vp.width *= sx, vp.y *= sy, vp.height *= sy;
  const VkRect2D sc{{0, 0}, {kCellW, kCellH}};
  f.vkCmdSetViewport(cb, 0, 1, &vp);
  f.vkCmdSetScissor(cb, 0, 1, &sc);
  DrawOnce(cb, r.call);
  f.vkCmdEndRenderPass(cb);
  VkMemoryBarrier b{};
  b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;  // (ALL_COMMANDS: it chains with the pass's own final layout transition)
  b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  f.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = {kCellW, kCellH, 1};
  region.bufferOffset = VkDeviceSize(kSoloSlotBytes) * uint32_t(r.solo_slot);
  f.vkCmdCopyImageToBuffer(cb, g->solo_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g->solo_buffer, 1, &region);
  b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  b.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  f.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}

// The debug draws of this pass's samples (and of earlier passes that had to wait), then everything bound put back: the frame's later
// commands may rely on what was bound (their records are deduplicated).
void RunDebug(VkCommandBuffer cb, uint32_t pass) {
  Capture& c = *t.cap;
  if (t.game_query) return;  // our draws would count in the game's query: at the next pass end
  bool any = false;
  for (Run& r : c.runs) {
    if (!(r.aux & kAuxSampled) || r.debug_done || r.pass > pass || !r.have_state) continue;
    r.debug_done = true;
    if (r.pass < c.passes.size() && c.passes[r.pass].multiview) {
      r.skip_numbers = r.skip_solo = "a multiview pass";
      continue;
    }
    Debug d;
    {
      std::lock_guard<std::mutex> l(g->m);
      const auto it = g->debug.find(K(r.state.pipeline));
      if (it != g->debug.end()) d = it->second;
      else d.why_numbers = d.why_solo = "no debug pipeline for this pipeline";
    }
    if (c.mode & kModeNumbers) {
      if (d.numbers) RunNumbers(cb, r, d.numbers), any = true;
      else r.skip_numbers = d.why_numbers;
    }
    if (c.mode & kModeSolo) {
      if (d.solo) RunSolo(cb, r, d.solo), any = true;
      else r.skip_solo = d.why_solo;
    }
  }
  if (any) {
    BindState(cb, t.bound, t.bound.pipeline);
    if (t.bound.has_viewport) g->real.vkCmdSetViewport(cb, 0, 1, &t.bound.viewport);
    if (t.bound.has_scissor) g->real.vkCmdSetScissor(cb, 0, 1, &t.bound.scissor);
  }
}

void Finish(Capture* c);

void Close(VkCommandBuffer cb) {
  const Fn& f = g->real;
  Capture* c = t.cap;
  Setup(cb);
  if (c->mode & (kModeNumbers | kModeSolo)) RunDebug(cb, UINT32_MAX);
  if (c->mode & kModeSheet) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = g->atlas;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    f.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {kAtlasW, kAtlasH, 1};
    f.vkCmdCopyImageToBuffer(cb, g->atlas, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g->atlas_buffer, 1, &region);
  }
  VkMemoryBarrier b{};
  b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  f.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
  g->cmd_set_event(cb, g->event, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  t.cap = nullptr;
  t.close_pending = false;
  t.cb = VK_NULL_HANDLE;  // (command buffers are reused: the next capture's stream must not inherit this one's bound state)
  ++g->workers;
  std::thread(Finish, c).detach();
}

// ---- the worker's half of a capture
struct Stats {
  uint32_t min[3], max[3];
  double mean[3], black;
  uint64_t hash, dhash;
};
Stats CellStats(const uint8_t* atlas, uint32_t col, uint32_t row) {
  Stats s{};
  for (int i = 0; i < 3; ++i) s.min[i] = 255;
  uint64_t sum[3] = {}, black = 0;
  std::array<uint8_t, kCellW * kCellH * 4> pixels;
  for (uint32_t y = 0; y < kCellH; ++y) {
    const uint8_t* line = atlas + (size_t(row * kCellH + y) * kAtlasW + size_t(col) * kCellW) * 4;
    std::memcpy(pixels.data() + size_t(y) * kCellW * 4, line, kCellW * 4);
    for (uint32_t x = 0; x < kCellW; ++x) {
      const uint8_t* p = line + x * 4;
      for (int i = 0; i < 3; ++i) {
        s.min[i] = std::min<uint32_t>(s.min[i], p[i]);
        s.max[i] = std::max<uint32_t>(s.max[i], p[i]);
        sum[i] += p[i];
      }
      black += p[0] <= 8 && p[1] <= 8 && p[2] <= 8;
    }
  }
  const double n = double(kCellW) * kCellH;
  for (int i = 0; i < 3; ++i) s.mean[i] = double(sum[i]) / n;
  s.black = double(black) / n;
  s.hash = XXH3_64bits(pixels.data(), pixels.size());
  double lum[8][9];
  for (uint32_t y = 0; y < 8; ++y) {
    for (uint32_t x = 0; x < 9; ++x) {
      const uint32_t x0 = kCellW * x / 9, x1 = kCellW * (x + 1) / 9, y0 = kCellH * y / 8, y1 = kCellH * (y + 1) / 8;
      double total = 0;
      for (uint32_t yy = y0; yy < y1; ++yy) {
        for (uint32_t xx = x0; xx < x1; ++xx) {
          const uint8_t* p = pixels.data() + (size_t(yy) * kCellW + xx) * 4;
          total += 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
        }
      }
      lum[y][x] = total / double((x1 - x0) * (y1 - y0));
    }
  }
  for (uint32_t y = 0; y < 8; ++y) {
    for (uint32_t x = 0; x < 8; ++x) s.dhash = (s.dhash << 1) | (lum[y][x] > lum[y][x + 1] ? 1u : 0u);
  }
  return s;
}

uint64_t QueryResult(VkQueryPool pool, uint32_t index, bool* ok) {
  uint64_t v = 0;
  const VkResult r = g->real.vkGetQueryPoolResults(g->device, pool, index, 1, sizeof(v), &v, sizeof(v), VK_QUERY_RESULT_64_BIT);
  *ok = r == VK_SUCCESS;
  return v;
}

void WriteSheet(Capture& c) {
  const uint8_t* atlas = static_cast<const uint8_t*>(g->atlas_map);
  Invalidate(g->atlas_buffer_memory);
  std::unordered_map<uint32_t, const Light*> by_id;
  for (const Light& l : c.draws) by_id[l.id] = &l;
  std::string j = fmt::format("{{\"scene\":{},\"seq\":{},\"frame\":{},\"mode\":{},\"atlas\":[{},{}],\"cell\":[{},{}],\"cols\":{},\"passes\":{},\"cells_used\":{},\"overflow\":{},"
                              "\"draws_marked\":{},\"images_usage_added\":{},\"timestamp_period\":{},\"cells\":[",
                              Json(c.scene), c.seq, c.frame, c.mode, kAtlasW, kAtlasH, kCellW, kCellH, kCellCols, c.passes.size(), c.cells, c.overflow, c.draws.size(),
                              g->images_usage_added, F(g->timestamp_period));
  bool first = true;
  for (const PassRec& p : c.passes) {
    std::string fam;
    for (uint32_t i = 0; i < kFamilyCount; ++i) {
      if (p.family[i]) fam += fmt::format("{}\"{}\":{}", fam.empty() ? "" : ",", kFamilyNames[i], p.family[i]);
    }
    double ms = -1.0;
    if (p.timestamp >= 0 && g->timestamp_pool) {
      bool ok0 = false, ok1 = false;
      const uint64_t a = QueryResult(g->timestamp_pool, uint32_t(p.timestamp), &ok0), b = QueryResult(g->timestamp_pool, uint32_t(p.timestamp) + 1, &ok1);
      if (ok0 && ok1 && b >= a) ms = double(b - a) * g->timestamp_period / 1e6;
    }
    const Light* opening = nullptr;
    if (p.first_id) {
      const auto it = by_id.find(p.first_id);
      if (it != by_id.end()) opening = it->second;
    }
    j += fmt::format("{}{{\"pass\":{},\"kind\":\"{}\",\"cell\":{},\"target\":{},\"w\":{},\"h\":{},\"fmt\":{},\"samples\":{},\"colors\":{},\"draws\":{},\"family\":{{{}}},\"multiview\":{},"
                     "\"vs\":\"{}\",\"ps\":\"{}\",\"vk_vs\":\"{}\",\"vk_fs\":\"{}\",\"gpu_ms\":{}",
                     first ? "" : ",", p.index, p.kind, p.cell, p.target, p.source.width, p.source.height, uint32_t(p.source.format), p.source.samples, p.colors, p.draws, fam,
                     p.multiview ? "true" : "false", opening ? Hex64(opening->vs) : "", opening ? Hex64(opening->ps) : "", Hex64(p.vk_vs), Hex64(p.vk_fs), F(ms));
    first = false;
    if (p.cell >= 0) {
      const Stats s = CellStats(atlas, uint32_t(p.cell) % kCellCols, uint32_t(p.cell) / kCellCols);
      j += fmt::format(",\"min\":[{},{},{}],\"max\":[{},{},{}],\"mean\":[{},{},{}],\"black\":{},\"xxh\":\"{}\",\"dhash\":\"{}\"", s.min[0], s.min[1], s.min[2], s.max[0], s.max[1], s.max[2],
                       F(s.mean[0]), F(s.mean[1]), F(s.mean[2]), F(s.black), Hex64(s.hash), Hex64(s.dhash));
    }
    j += "}";
  }
  j += "]}\n";
  const std::string name = SafeScene(c.scene);
  // only the cells in use: the atlas's rows beyond them are empty
  const uint32_t rows = std::max(1u, (c.cells + kCellCols - 1) / kCellCols);
  WriteAtomic("sheet_" + name + ".rgba", atlas, size_t(kAtlasW) * kCellH * rows * 4);
  WriteAtomic("sheet_" + name + ".json", j.data(), j.size());
}

// The samples for the following truth / numbers frames. Same rule in every arm: by family, largest vertex count first, plus translated draws.
void PickSamples(Capture& c) {
  static constexpr std::pair<uint32_t, uint32_t> kQuota[] = {{kFamilyWrestler, 4}, {kFamilyCrowd, 2}, {0, 2}, {1, 2}, {2, 2}, {kFamilyTranslated, 4}};
  std::vector<const Light*> sorted;
  for (const Light& l : c.draws) {
    if (!l.memexport && !l.repeat && l.count >= 12) sorted.push_back(&l);
  }
  std::stable_sort(sorted.begin(), sorted.end(), [](const Light* a, const Light* b) { return a->count > b->count; });
  std::vector<uint64_t> keys;
  std::vector<VkPipeline> pipelines;
  for (const auto& [family, quota] : kQuota) {
    uint32_t taken = 0;
    std::unordered_set<uint64_t> shapes;
    for (const Light* l : sorted) {
      if (l->family != family || taken >= quota) continue;
      const auto shape = (l->vs * 31 + l->ps) * 31 + l->count;
      if (!shapes.insert(shape).second) continue;  // the same mesh again adds nothing
      const auto it = c.pipelines.find(l->id);
      if (it == c.pipelines.end()) continue;
      const uint64_t words[4] = {l->vs, l->ps, l->count, l->nth};
      keys.push_back(XXH3_64bits(words, sizeof(words)));
      pipelines.push_back(it->second);
      ++taken;
    }
  }
  g->pick_keys = std::move(keys);
  g->pick_pipelines = std::move(pipelines);
  REXGPU_INFO("[diag] picked {} sampled draws of {} draws (scene {})", g->pick_keys.size(), c.draws.size(), c.scene);
}

struct PositionSink : ShaderInterpreter::ExportSink {
  void Export(ucode::ExportRegister export_register, const float* value, uint32_t value_mask) override {
    if (export_register == ucode::ExportRegister::kVSPointSizeEdgeFlagKillVertex) {
      if (value_mask & 0b100) {  // the translator's kill test: the Z component's bits, sign ignored
        uint32_t bits;
        std::memcpy(&bits, &value[2], 4);
        killed = (bits & 0x7FFFFFFFu) != 0;
      }
      return;
    }
    if (export_register != ucode::ExportRegister::kVSPosition) return;
    for (uint32_t i = 0; i < 4; ++i) {
      if (value_mask & (1u << i)) position[i] = value[i];
    }
    written |= value_mask;
  }
  float position[4] = {};
  uint32_t written = 0;
  bool killed = false;
};
using SysFlags = SpirvShaderTranslator;

// gl_VertexIndex as the shader saw it -> the guest index, as the translated shader's prologue makes it (spirv_translator.cpp).
uint32_t GuestIndex(const Snapshot& s, memory::Memory* memory, uint32_t raw) {
  if (s.attr) return raw;  // (Session 95: the pool's host-order index plus vertexOffset, its identity proven before the draw)
  uint32_t index = raw;
  if (s.flags & SysFlags::kSysFlag_VertexIndexLoad) {
    const uint8_t* p = memory->TranslatePhysical<const uint8_t*>(s.load_address + raw * 4);
    if (p) std::memcpy(&index, p, 4);
  }
  index = xenos::GpuSwap(index, xenos::Endian(s.endian));
  index = (index + uint32_t(s.base)) & xenos::kVertexIndexMask;
  return std::clamp(index, s.index_min, s.index_max);
}
// The guest position through the translated shader's epilogue: clip space on the host.
void HostPosition(const Snapshot& s, const float in[4], float out[4]) {
  const float w = (s.flags & SysFlags::kSysFlag_WNotReciprocal) ? in[3] : 1.0f / in[3];
  float x = in[0], y = in[1], z = in[2];
  if (s.flags & SysFlags::kSysFlag_XYDividedByW) x *= w, y *= w;
  if (s.flags & SysFlags::kSysFlag_ZDividedByW) z *= w;
  out[0] = x * s.ndc_scale[0] + s.ndc_offset[0] * w;
  out[1] = y * s.ndc_scale[1] + s.ndc_offset[1] * w;
  out[2] = z * s.ndc_scale[2] + s.ndc_offset[2] * w;
  out[3] = w;
}

struct Cpu {
  std::unique_ptr<RegisterFile> regs = std::make_unique<RegisterFile>();
  ShaderInterpreter interp;
  PositionSink sink;
  ShaderInterpreter::ConstantSource source;
  uint32_t skipped_oob = 0, skipped_killed = 0;  // rows the CPU cannot stand for: a fetch outside the fetch constant reads 0 here, real memory on the GPU; a killed vertex
  explicit Cpu(memory::Memory* memory) : interp(*regs, *memory) {}
  void Begin(const Snapshot& s) {
    ShaderInterpreter::SetExactPackedFetch(true);  // (Session 95: it stands for what the shader read; the worker thread is the diagnostic's own)
    skipped_oob = skipped_killed = 0;
    source.floats = s.floats.data();
    source.bool_loop = s.bool_loop.data();
    source.fetch = s.fetch.data();
    source.vs_const = s.vs_const;
    interp.SetRegisterSource(nullptr);
    interp.SetConstantSource(&source);
    interp.SetShader(xenos::ShaderType::kVertex, s.ucode);
    interp.SetExportSink(&sink);
    interp.set_stop_after_steps(UINT32_MAX);
  }
  bool Run(const Snapshot& s, uint32_t guest_index, float out[4]) {
    std::memset(interp.temp_registers(), 0, sizeof(float) * 4 * xenos::kMaxShaderTempRegisters);
    interp.temp_registers()[0] = float(guest_index);
    sink.written = 0;
    sink.killed = false;
    interp.reset_fetches_out_of_range();
    interp.Execute();
    if (interp.fetches_out_of_range()) {
      ++skipped_oob;
      return false;
    }
    if (sink.killed) {  // (the translator makes w NaN for such a vertex: not comparable)
      ++skipped_killed;
      return false;
    }
    if (!(sink.written & 0b1000) && !(sink.written & 0b0011)) return false;
    HostPosition(s, sink.position, out);
    return true;
  }
};

// The live guest memory against the snapshot: the freeze stops the scene, not the renderer, so a dynamic vertex range may have been rewritten since the draw.
bool GuestMoved(const Snapshot& s) {
  memory::Memory* const memory = DiagAccess::Memory(*g->cp);
  for (const Range& r : s.ranges) {
    const uint8_t* now = memory->TranslatePhysical<const uint8_t*>(r.address);
    if (!now || std::memcmp(now, r.guest.data(), r.size)) return true;
  }
  return false;
}
// Why the CPU may not stand for what the shader read (empty: it may).
const char* CpuRefusal(const Snapshot& s) {
  if (s.gpu_written) return "a vertex range holds GPU-written data (resolve / memexport): not in the guest memory";
  if (GuestMoved(s)) return "guest_moved: a vertex range was rewritten after the draw";
  return "";
}

// Session 95 (the purple Savage): the constants the body material's colour comes from, as raw words (a game value, not an asset): vs c145, c237, c238, c252-c255; ps c190, c234, c246, c218; texture fetch 14.
std::string MaterialJson(const Snapshot* s) {
  if (!s) return "null";
  const auto words = [](const uint32_t* w, uint32_t n) {
    std::string r;
    for (uint32_t i = 0; i < n; ++i) r += fmt::format("{}\"{:08X}\"", i ? "," : "", w[i]);
    return r;
  };
  return fmt::format("{{\"c145\":[{}],\"c237\":[{}],\"c238\":[{}],\"c252\":[{}],\"ps190\":[{}],\"ps234\":[{}],\"ps246\":[{}],\"ps218\":[{}],\"fetch14\":[{}],\"bool0\":\"{:08X}\"}}", words(s->floats.data() + 4 * 145, 4),
                     words(s->floats.data() + 4 * 237, 4), words(s->floats.data() + 4 * 238, 4), words(s->floats.data() + 4 * 252, 16), words(s->ps_rows.data(), 4), words(s->ps_rows.data() + 4, 4),
                     words(s->ps_rows.data() + 8, 4), words(s->ps_rows.data() + 12, 4), words(s->fetch.data() + 6 * 14, 6), s->bool_loop[0]);
}

std::string BoxJson(const double b[4]) { return b[0] > b[2] ? "null" : fmt::format("[{},{},{},{}]", F(b[0]), F(b[1]), F(b[2]), F(b[3])); }
void Grow(double b[4], double x, double y) {
  b[0] = std::min(b[0], x), b[1] = std::min(b[1], y), b[2] = std::max(b[2], x), b[3] = std::max(b[3], y);
}

// Layer 1 and the CPU's own view of the draw (the truth frame).
std::string BytesJson(const Snapshot* s) {
  if (!s) return "null";
  if (s->inline_draw) return "{\"skipped\":\"inline ring draw: its vertices are not in the shared memory\"}";
  if (s->gpu_written) return "{\"skipped\":\"gpu_written: a vertex range holds GPU-written data, the guest bytes are not what the shader read\"}";
  uint64_t bytes = 0;
  uint32_t diff_ranges = 0;
  int64_t first_diff = -1;
  std::string list;
  for (const Range& r : s->ranges) {
    bytes += r.size;
    const uint8_t* gpu = static_cast<const uint8_t*>(g->bytes_map) + r.staging;
    uint32_t differing = 0;
    int64_t first = -1;
    for (uint32_t i = 0; i < r.size; i += 4) {
      if (std::memcmp(gpu + i, r.guest.data() + i, 4)) {
        ++differing;
        if (first < 0) first = i;
      }
    }
    if (differing) {
      ++diff_ranges;
      if (first_diff < 0) first_diff = first;
    }
    list += fmt::format("{}{{\"fetch\":{},\"address\":\"{:08X}\",\"size\":{},\"diff_dwords\":{},\"first_diff\":{}}}", list.empty() ? "" : ",", r.fetch, r.address, r.size, differing, first);
  }
  return fmt::format("{{\"ranges\":{},\"bytes\":{},\"diff_ranges\":{},\"first_diff\":{},\"window\":{},\"window_on\":{},\"list\":[{}]}}", s->ranges.size(), bytes, diff_ranges, first_diff, s->window,
                     s->window_on ? "true" : "false", list);
}

std::string CpuJson(Cpu& cpu, const Snapshot* s, const Run& r) {
  if (!s) return "null";
  if (!s->can_interpret) return "{\"refused\":\"the vertex shader fetches textures\"}";
  if (s->inline_draw) return "{\"refused\":\"inline ring draw\"}";
  if (s->host_vs_type) return "{\"refused\":\"not a plain vertex draw\"}";
  if (const char* why = CpuRefusal(*s); *why) return std::string("{\"refused\":") + Json(why) + "}";
  cpu.Begin(*s);
  uint32_t n = 0, bad = 0, behind = 0;
  double box[4] = {1e30, 1e30, -1e30, -1e30}, clamped[4] = {1e30, 1e30, -1e30, -1e30};
  for (const uint32_t index : s->cpu_indices) {
    float p[4];
    if (!cpu.Run(*s, index, p)) continue;
    ++n;
    if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]) || !std::isfinite(p[3])) {
      ++bad;
      continue;
    }
    if (p[3] <= 0.0f) {
      ++behind;
      continue;
    }
    const double x = double(p[0]) / p[3], y = double(p[1]) / p[3];
    Grow(box, x, y);
    Grow(clamped, std::clamp(x, -1.0, 1.0), std::clamp(y, -1.0, 1.0));
  }
  const double area = clamped[0] > clamped[2] ? 0.0 : (clamped[2] - clamped[0]) * (clamped[3] - clamped[1]) / 4.0;
  return fmt::format("{{\"n\":{},\"nonfinite\":{},\"behind\":{},\"skipped_oob\":{},\"skipped_killed\":{},\"box\":{},\"on_screen\":{},\"area_share\":{},\"area_px\":{}}}", n, bad, behind, cpu.skipped_oob,
                     cpu.skipped_killed, BoxJson(box), area > 0.0 ? "true" : "false", F(area), F(area * double(r.target_w) * double(r.target_h)));
}

// Layer 3: what came out of the real module, against the interpreter.
std::string NumbersJson(Cpu& cpu, const Snapshot* s, const Run& r) {
  if (r.numbers_slot < 0) return "{\"skipped\":" + Json(r.skip_numbers.empty() ? "not run" : r.skip_numbers) + "}";
  const uint8_t* slot = static_cast<const uint8_t*>(g->numbers_map) + size_t(kNumbersSlotBytes) * uint32_t(r.numbers_slot);
  const uint32_t* pos = reinterpret_cast<const uint32_t*>(slot);
  const uint32_t* ids = reinterpret_cast<const uint32_t*>(slot + size_t(kCapturePixels) * 16);
  std::vector<uint32_t> written;
  for (uint32_t i = 0; i < kCapturePixels; ++i) {
    if (ids[i] != 0xFFFFFFFFu) written.push_back(i);
  }
  const uint32_t step = std::max<uint32_t>(1, uint32_t(written.size()) / kCpuVertices);
  const char* refusal = s ? CpuRefusal(*s) : "";
  const bool compare = s && s->can_interpret && !s->inline_draw && !s->host_vs_type && r.call.instances == 1 && !*refusal;
  const char* no_compare = !s ? "no guest snapshot (our own draw)" : *refusal ? refusal : !s->can_interpret ? "the vertex shader fetches textures" : s->inline_draw ? "inline ring draw" : s->host_vs_type ? "not a plain vertex draw" : r.call.instances != 1 ? "instanced" : "";
  if (compare) cpu.Begin(*s);
  struct Row {
    double err;
    uint32_t raw, guest;
    float gpu[4], cpu[4];
  };
  std::vector<Row> rows;
  std::vector<double> errors, xs, ys, zs;
  uint32_t gpu_bad = 0, cpu_bad = 0, compared = 0, gpu_bad_cpu_ok = 0;
  double box[4] = {1e30, 1e30, -1e30, -1e30}, box_cpu[4] = {1e30, 1e30, -1e30, -1e30};
  for (size_t i = 0; i < written.size(); ++i) {
    const uint32_t pixel = written[i];
    float gp[4];
    std::memcpy(gp, pos + size_t(pixel) * 4, 16);
    const bool finite = std::isfinite(gp[0]) && std::isfinite(gp[1]) && std::isfinite(gp[2]) && std::isfinite(gp[3]);
    if (!finite) ++gpu_bad;
    else if (gp[3] > 0.0f) {
      const double x = double(gp[0]) / gp[3], y = double(gp[1]) / gp[3], z = double(gp[2]) / gp[3];
      Grow(box, x, y);
      if (gp[3] > 1e-2f) xs.push_back(x), ys.push_back(y), zs.push_back(z);  // (a vertex a hair in front of the camera plane is huge on purpose: not a spike)
    }
    if (!compare || i % step) continue;
    Row row{};
    row.raw = ids[pixel];
    row.guest = GuestIndex(*s, DiagAccess::Memory(*g->cp), row.raw);
    std::memcpy(row.gpu, gp, 16);
    if (!cpu.Run(*s, row.guest, row.cpu)) continue;
    ++compared;
    const bool cfin = std::isfinite(row.cpu[0]) && std::isfinite(row.cpu[1]) && std::isfinite(row.cpu[2]) && std::isfinite(row.cpu[3]);
    if (!cfin) {
      ++cpu_bad;
      row.err = finite ? 1e30 : 0.0;
    } else if (!finite) {
      ++gpu_bad_cpu_ok;
      row.err = 1e30;
    } else {
      const double scale = std::max(1.0, std::fabs(double(row.cpu[3])));
      for (int k = 0; k < 4; ++k) row.err = std::max(row.err, std::fabs(double(row.gpu[k]) - double(row.cpu[k])) / scale);
      if (row.cpu[3] > 0.0f) Grow(box_cpu, double(row.cpu[0]) / row.cpu[3], double(row.cpu[1]) / row.cpu[3]);
    }
    errors.push_back(row.err);
    rows.push_back(row);
  }
  // Spikes need no CPU truth: positions far outside the draw's own box.
  uint32_t spikes = 0;
  double radius = 0.0;
  if (xs.size() >= 8) {
    const auto median = [](std::vector<double> v) {
      std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
      return v[v.size() / 2];
    };
    const double mx = median(xs), my = median(ys), mz = median(zs);
    std::vector<double> d(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) d[i] = std::sqrt((xs[i] - mx) * (xs[i] - mx) + (ys[i] - my) * (ys[i] - my) + (zs[i] - mz) * (zs[i] - mz));
    std::vector<double> sorted = d;
    std::sort(sorted.begin(), sorted.end());
    radius = sorted[size_t(double(sorted.size() - 1) * 0.95)];
    for (const double v : d) spikes += v > radius * 8.0 + 0.25;
  }
  std::sort(errors.begin(), errors.end());
  std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.err > b.err; });
  std::string worst;
  for (size_t i = 0; i < rows.size() && i < kWorstRows; ++i) {
    const Row& w = rows[i];
    worst += fmt::format("{}{{\"raw\":{},\"guest\":{},\"err\":{},\"gpu\":[{},{},{},{}],\"cpu\":[{},{},{},{}]}}", worst.empty() ? "" : ",", w.raw, w.guest, F(w.err), F(w.gpu[0]), F(w.gpu[1]), F(w.gpu[2]),
                         F(w.gpu[3]), F(w.cpu[0]), F(w.cpu[1]), F(w.cpu[2]), F(w.cpu[3]));
  }
  uint32_t outliers = 0;
  for (const double e : errors) outliers += e > 1e-2;
  return fmt::format("{{\"written\":{},\"gpu_nonfinite\":{},\"gpu_nonfinite_cpu_finite\":{},\"skipped_oob\":{},\"skipped_killed\":{},\"box\":{},\"spikes\":{},\"radius95\":{},\"compared\":{},\"no_compare\":{},\"cpu_nonfinite\":{},\"max_err\":{},\"median_err\":{},\"outliers\":{},"
                     "\"box_cpu\":{},\"worst\":[{}]}}",
                     written.size(), gpu_bad, gpu_bad_cpu_ok, compare ? cpu.skipped_oob : 0, compare ? cpu.skipped_killed : 0, BoxJson(box), spikes, F(radius), compared, Json(no_compare), cpu_bad, errors.empty() ? "null" : F(errors.back()),
                     errors.empty() ? "null" : F(errors[errors.size() / 2]), outliers, BoxJson(box_cpu), worst);
}

void WriteVertex(Capture& c) {
  Invalidate(g->bytes_memory);
  if (g->numbers_buffer_memory) Invalidate(g->numbers_buffer_memory);
  if (g->solo_buffer_memory) Invalidate(g->solo_buffer_memory);
  Cpu cpu(DiagAccess::Memory(*g->cp));
  std::unordered_map<uint32_t, const Light*> by_id;
  for (const Light& l : c.draws) by_id[l.id] = &l;
  const bool truth = c.mode & kModeTruth;
  std::string rows, per_pass;
  std::map<std::pair<uint32_t, uint32_t>, std::pair<uint64_t, uint64_t>> family_cov;  // (pass, family) -> draws, samples
  std::vector<uint8_t> solo_sheet;
  const uint32_t solo_rows = (c.solo_slots + kSoloCols - 1) / kSoloCols;
  if (c.solo_slots) solo_sheet.assign(size_t(kCellW) * kSoloCols * kCellH * solo_rows * 4, 0);
  for (const Run& r : c.runs) {
    const Light* l = nullptr;
    const auto it = by_id.find(r.id);
    if (it != by_id.end()) l = it->second;
    const Snapshot* s = nullptr;
    {
      std::lock_guard<std::mutex> lock(g->m);
      const auto sn = c.snapshots.find(r.id);
      if (sn != c.snapshots.end()) s = sn->second.get();
    }
    const uint32_t family = r.aux & kAuxFamilyMask;
    uint64_t samples = 0;
    bool samples_ok = false;
    if (r.query >= 0 && g->occlusion_pool) samples = QueryResult(g->occlusion_pool, uint32_t(r.query), &samples_ok);
    if (samples_ok) {
      auto& cell = family_cov[{r.pass, family}];
      ++cell.first;
      cell.second += samples;
    }
    if (!(r.aux & kAuxSampled)) continue;
    std::string solo = "null";
    if (r.solo_slot >= 0) {
      const uint8_t* src = static_cast<const uint8_t*>(g->solo_map) + size_t(kSoloSlotBytes) * uint32_t(r.solo_slot);
      uint32_t lit = 0;
      const uint32_t col = uint32_t(r.solo_slot) % kSoloCols, row = uint32_t(r.solo_slot) / kSoloCols;
      for (uint32_t y = 0; y < kCellH; ++y) {
        const uint8_t* line = src + size_t(y) * kCellW * 4;
        std::memcpy(solo_sheet.data() + (size_t(row * kCellH + y) * kCellW * kSoloCols + size_t(col) * kCellW) * 4, line, kCellW * 4);
        for (uint32_t x = 0; x < kCellW; ++x) lit += line[x * 4] > 127;
      }
      solo = fmt::format("{{\"cell\":{},\"px\":{}}}", r.solo_slot, lit);
    } else if (c.mode & kModeSolo) {
      solo = "{\"skipped\":" + Json(r.skip_solo.empty() ? "not run" : r.skip_solo) + "}";
    }
    rows += fmt::format("{}{{\"id\":{},\"family\":\"{}\",\"vs\":\"{}\",\"ps\":\"{}\",\"nth\":{},\"pass\":{},\"vertex_count\":{},\"instances\":{},\"indexed\":{},\"target\":[{},{}],\"input_hash\":\"{}\","
                        "\"coverage\":{{\"samples\":{},\"ran\":{},\"skipped\":{}}}",
                        rows.empty() ? "" : ",", r.id, kFamilyNames[std::min(family, kFamilyCount - 1)], l ? Hex64(l->vs) : "", l ? Hex64(l->ps) : "", l ? l->nth : 0, r.pass, r.call.count, r.call.instances,
                        r.call.indexed ? "true" : "false", r.target_w, r.target_h, s ? Hex64(s->input_hash) : "", samples, samples_ok ? "true" : "false", r.skip_query ? Json(r.skip_query) : "null");
    if (truth) rows += ",\"bytes\":" + BytesJson(s) + ",\"cpu\":" + CpuJson(cpu, s, r);
    if (l && l->family == kFamilyWrestler) rows += ",\"material\":" + MaterialJson(s);
    if (c.mode & kModeNumbers) rows += ",\"numbers\":" + NumbersJson(cpu, s, r);
    if (c.mode & kModeSolo) rows += ",\"solo\":" + solo;
    rows += "}";
  }
  cpu.interp.SetExportSink(nullptr);
  cpu.interp.SetConstantSource(nullptr);
  for (const auto& [key, value] : family_cov) {
    per_pass += fmt::format("{}{{\"pass\":{},\"family\":\"{}\",\"draws\":{},\"samples\":{}}}", per_pass.empty() ? "" : ",", key.first, kFamilyNames[std::min(key.second, kFamilyCount - 1)], value.first, value.second);
  }
  const std::string j = fmt::format("{{\"scene\":{},\"seq\":{},\"frame\":{},\"mode\":{},\"occlusion\":\"{}\",\"picks\":{},\"picks_found\":{},\"family_coverage\":[{}],\"draws\":[{}]}}\n", Json(c.scene), c.seq, c.frame, c.mode,
                                    !g->occlusion_pool ? "none" : g->precise ? "precise" : "yes-no", c.picks.size(), c.picks_found, per_pass, rows);
  const std::string name = SafeScene(c.scene);
  WriteAtomic((truth ? "truth_" : "vertex_") + name + ".json", j.data(), j.size());
  if (!solo_sheet.empty()) WriteAtomic("solo_" + name + ".rgba", solo_sheet.data(), solo_sheet.size());  // (640 wide; the game makes it a JPEG)
}

void Finish(Capture* c) {
  const auto start = std::chrono::steady_clock::now();
  bool signalled = false;
  while (!g->stop.load() && std::chrono::steady_clock::now() - start < std::chrono::seconds(8)) {
    if (g->event_status(g->device, g->event) == VK_EVENT_SET) {
      signalled = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  if (signalled) {
    if (c->mode & kModeSheet) WriteSheet(*c);
    if (c->mode & kModePick) PickSamples(*c);
    if (c->mode & (kModeTruth | kModeNumbers | kModeSolo)) WriteVertex(*c);
  }
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - c->start).count();
  REXGPU_INFO("[diag] capture {} scene {} mode {}: {} passes, {} cells, {} marked draws, {} looked at, {} ms{}", c->seq, c->scene, c->mode, c->passes.size(), c->cells, c->draws.size(), c->runs.size(), ms,
              signalled ? "" : " (the frame's event never came: nothing written)");
  const uint32_t seq = c->seq;
  delete c;
  g_observing = false;
  g->busy = false;
  REXCVAR_SET(gpu_diag_capture_done, int32_t(seq));
  --g->workers;
}

void Arm(VulkanCommandProcessor& cp, uint32_t seq, const std::string& scene, uint32_t mode) {
  if (!EnsureCaptureObjects()) {
    REXGPU_WARN("[diag] the capture objects could not be made: capture {} dropped", seq);
    REXCVAR_SET(gpu_diag_capture_done, int32_t(seq));
    return;
  }
  auto* c = new Capture;
  c->seq = seq;
  c->scene = scene;
  c->mode = mode;
  c->frame = DiagAccess::Frame(cp);
  if (g->parity < 0) g->parity = int32_t(c->frame & 1);
  c->start = std::chrono::steady_clock::now();
  if (mode & (kModeTruth | kModeNumbers | kModeSolo)) c->picks.insert(g->pick_keys.begin(), g->pick_keys.end());
  g->reset_event(g->device, g->event);
  g->busy = true;
  g->cap = c;
  g_observing = true;
  g_armed = true;
  cp.deferred_command_buffer().CmdDiagMark(kMarkFrameOpen, seq, 0);
}

}  // namespace

// ---- command thread
void Install(VulkanCommandProcessor& cp) {
  const std::string dir = REXCVAR_GET(gpu_diag_layer);
  if (dir.empty()) return;
  g = new G;
  g->cp = &cp;
  g->dev = cp.GetVulkanDevice();
  g->device = g->dev->device();
  g->physical = g->dev->physical_device();
  g->real = g->dev->functions();
  g->dir = std::filesystem::u8path(dir);
  const auto& ifn = g->dev->vulkan_instance()->functions();
  g->format_properties = ifn.vkGetPhysicalDeviceFormatProperties;
  // Not in the device's table (nothing else uses them): the core functions, looked up here.
  const auto load = [&](const char* name) { return ifn.vkGetDeviceProcAddr(g->device, name); };
  g->create_event = reinterpret_cast<PFN_vkCreateEvent>(load("vkCreateEvent"));
  g->event_status = reinterpret_cast<PFN_vkGetEventStatus>(load("vkGetEventStatus"));
  g->reset_event = reinterpret_cast<PFN_vkResetEvent>(load("vkResetEvent"));
  g->cmd_set_event = reinterpret_cast<PFN_vkCmdSetEvent>(load("vkCmdSetEvent"));
  g->cmd_resolve = reinterpret_cast<PFN_vkCmdResolveImage>(load("vkCmdResolveImage"));
  if (!g->create_event || !g->event_status || !g->reset_event || !g->cmd_set_event || !g->cmd_resolve) {
    REXGPU_WARN("[diag] the event functions are missing: the layer stays off");
    return;
  }
  VkPhysicalDeviceProperties props{};
  ifn.vkGetPhysicalDeviceProperties(g->physical, &props);
  g->timestamp_period = props.limits.timestampPeriod;
  uint32_t families = 0;
  ifn.vkGetPhysicalDeviceQueueFamilyProperties(g->physical, &families, nullptr);
  std::vector<VkQueueFamilyProperties> family_props(families);
  ifn.vkGetPhysicalDeviceQueueFamilyProperties(g->physical, &families, family_props.data());
  const uint32_t family = g->dev->queue_family_graphics_compute();
  g->timestamp_bits = family < families ? family_props[family].timestampValidBits : 0;
  const auto& dp = g->dev->properties();
  g->precise = dp.occlusionQueryPrecise;
  g->max_locations = std::min(dp.maxVertexOutputComponents, dp.maxFragmentInputComponents) / 4;
  Fn& table = const_cast<Fn&>(g->dev->functions());  // (read by reference everywhere; the pointer stores are the only change, before any object exists)
  table.vkCreateImage = ThunkCreateImage;
  table.vkDestroyImage = ThunkDestroyImage;
  table.vkCreateImageView = ThunkCreateImageView;
  table.vkDestroyImageView = ThunkDestroyImageView;
  table.vkCreateRenderPass = ThunkCreateRenderPass;
  table.vkDestroyRenderPass = ThunkDestroyRenderPass;
  table.vkCreateFramebuffer = ThunkCreateFramebuffer;
  table.vkDestroyFramebuffer = ThunkDestroyFramebuffer;
  table.vkCreateShaderModule = ThunkCreateShaderModule;
  table.vkDestroyShaderModule = ThunkDestroyShaderModule;
  table.vkCreateGraphicsPipelines = ThunkCreateGraphicsPipelines;
  table.vkDestroyPipeline = ThunkDestroyPipeline;
  g_latched = true;
  REXGPU_INFO("[diag] layer on: thunks in, results to {} (occlusion {}, locations {}, timestamp period {})", dir, g->precise ? "precise" : "yes-no", g->max_locations, g->timestamp_period);
}

void Shutdown() {
  if (!g) return;
  g->stop = true;
  g_armed = false;
  g_observing = false;
  // The workers use the device and the guest memory: the context goes down only after they are out (a bounded wait; Finish gives up its own wait at once on `stop`).
  const auto start = std::chrono::steady_clock::now();
  while (g->workers.load() > 0 && std::chrono::steady_clock::now() - start < std::chrono::seconds(10)) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  if (g->workers.load() > 0) REXGPU_WARN("[diag] shutdown: {} worker(s) still running after 10 s", g->workers.load());
}

void OnFrameOpen(VulkanCommandProcessor& cp) {
  if (g->busy.load()) return;
  if (g->parity >= 0 && int32_t(DiagAccess::Frame(cp) & 1) != g->parity) return;
  std::string request;
  if (g->prep) {
    if (!g->prep_done.load()) return;
    g->prep = false;
    request = g->pending_request;
    if (std::chrono::steady_clock::now() - g->pending_at > std::chrono::seconds(35)) {
      uint32_t old_seq = 0;
      std::sscanf(request.c_str(), "%u", &old_seq);
      REXGPU_WARN("[diag] the debug pipelines came too late: request '{}' dropped", request);
      g->prep_done = false;
      REXCVAR_SET(gpu_diag_capture_done, int32_t(old_seq));
      return;
    }
  } else {
    const int32_t seq_now = REXCVAR_GET(gpu_diag_capture_seq);
    if (seq_now == g->last_seq) return;
    g->last_seq = seq_now;
    request = REXCVAR_GET(gpu_diag_capture);
  }
  uint32_t seq = 0, mode = 0;
  char scene[64] = {};
  if (std::sscanf(request.c_str(), "%u:%63[^:]:%u", &seq, scene, &mode) != 3 || !seq || !mode) {
    REXGPU_WARN("[diag] bad capture request '{}'", request);
    return;
  }
  if ((mode & (kModeNumbers | kModeSolo)) && !g->prep_done.exchange(false)) {
    // the debug pipelines first, off this thread; the capture is armed at a later frame
    g->prep = true;
    g->pending_request = request;
    g->pending_at = std::chrono::steady_clock::now();
    ++g->workers;
    std::thread([pipelines = g->pick_pipelines] {
      BuildDebug(pipelines);
      --g->workers;
    }).detach();
    return;
  }
  ++g->captures;
  Arm(cp, seq, scene, mode);
}

void OnFrameClose(VulkanCommandProcessor& cp) {
  if (!g_armed) return;
  g_armed = false;
  cp.deferred_command_buffer().CmdDiagMark(kMarkFrameClose, 0, 0);
}

static Light LightOf(Capture& c, const native::Draw& draw) {
  Light l;
  l.id = uint32_t(c.draws.size()) + 1;
  l.vs = draw.vertex_shader_hash;
  l.ps = draw.pixel_shader_hash;
  l.count = draw.vertex_count;
  l.memexport = draw.memexport;
  uint32_t family = l.ps ? shader_swap::FamilyOf(l.ps) : 0xFFu;
  if (family == 0xFFu) family = shader_swap::FamilyOf(l.vs);
  l.family = uint8_t(family == 0xFFu ? kFamilyTranslated : family);
  return l;
}

void NoteRepeat(native::Draw& draw) {
  Capture* c = g->cap.load(std::memory_order_relaxed);
  if (!c) return;
  Light l = LightOf(*c, draw);
  l.repeat = true;  // (its constants are its base's with new offsets: counted and queried, never sampled)
  c->draws.push_back(l);
  draw.diag_id = l.id;
  draw.diag_aux = uint32_t(l.family) | (((c->mode & kModeTruth) && l.family == kFamilyWrestler) ? kAuxQuery : 0u);
}

void NoteDraw(VulkanCommandProcessor& cp, const DrawInputs& in) {
  Capture* c = g->cap.load(std::memory_order_relaxed);
  native::Draw& draw = *in.draw;
  if (!c) return;
  Light l = LightOf(*c, draw);
  const uint64_t shape[3] = {l.vs, l.ps, l.count};
  l.nth = c->occurrences[XXH3_64bits(shape, sizeof(shape))]++;
  const uint64_t words[4] = {l.vs, l.ps, l.count, l.nth};
  // (Review s95-all: a skin once draw is an attribute draw of its own (vi 12, not stored): its gl_VertexIndex is the element minus the job's min, not a guest index, so the vertex truth leaves it out.)
  const bool skin_draw = draw.attribute && draw.no_store && draw.vi == 12;
  const bool sampled = !skin_draw && !c->picks.empty() && c->picks.count(XXH3_64bits(words, sizeof(words)));
  const bool query = (c->mode & kModeTruth) && (sampled || l.family == kFamilyWrestler);
  c->draws.push_back(l);
  draw.diag_id = l.id;
  draw.diag_aux = uint32_t(l.family) | (sampled ? kAuxSampled : 0u) | (query ? kAuxQuery : 0u);
  if (!sampled) return;
  ++c->picks_found;
  auto s = std::make_unique<Snapshot>();
  s->id = l.id;
  const Shader& vs = *in.vertex_shader;
  s->ucode = vs.ucode_dwords();
  s->can_interpret = ShaderInterpreter::CanInterpretShader(vs);
  s->inline_draw = in.inline_draw;
  s->attr = draw.attribute;
  s->memexport = draw.memexport;
  s->host_vs_type = draw.host_vs_type;
  s->window_on = DiagAccess::WindowOn(cp);
  s->window = DiagAccess::WindowCur(cp);
  s->window_size = DiagAccess::WindowSize(cp);
  std::memcpy(s->floats.data(), in.floats, sizeof(uint32_t) * s->floats.size());
  std::memcpy(s->bool_loop.data(), in.bool_loop, sizeof(uint32_t) * s->bool_loop.size());
  if (in.ps_floats) {
    const uint32_t rows[4] = {190, 234, 246, 218};
    for (uint32_t k = 0; k < 4; ++k) std::memcpy(s->ps_rows.data() + 4 * k, in.ps_floats + 4 * rows[k], 16);
  }
  std::memcpy(s->fetch.data(), in.fetch, sizeof(uint32_t) * s->fetch.size());
  s->vs_const = in.vs_const;
  const auto& sys = DiagAccess::Sys(cp);
  s->flags = sys.flags;
  s->endian = uint32_t(sys.vertex_index_endian);
  s->load_address = sys.vertex_index_load_address;
  s->base = sys.vertex_base_index;
  s->index_min = sys.vertex_index_min;
  s->index_max = sys.vertex_index_max;
  std::memcpy(s->ndc_scale, sys.ndc_scale, sizeof(s->ndc_scale));
  std::memcpy(s->ndc_offset, sys.ndc_offset, sizeof(s->ndc_offset));
  memory::Memory* const memory = DiagAccess::Memory(cp);
  // The fetch ranges and their guest bytes as they are now (Bytes).
  if (!in.inline_draw) {  // (Session 95: a direct-bound attribute draw reads these same bytes; a ring draw's are not in the shared memory)
    for (const Shader::VertexBinding& b : vs.vertex_bindings()) {
      if (s->ranges.size() >= kBytesRangesPerDraw) break;
      xenos::xe_gpu_vertex_fetch_t f;
      std::memcpy(&f, s->fetch.data() + size_t(b.fetch_constant) * 2, sizeof(f));
      const uint32_t address = uint32_t(f.address) << 2, size = std::min<uint32_t>(uint32_t(f.size) << 2, kBytesRangeMax);
      if (!size || c->bytes_used + size > kBytesStaging) continue;
      if (std::any_of(s->ranges.begin(), s->ranges.end(), [&](const Range& r) { return r.address == address; })) continue;
      const uint8_t* guest = memory->TranslatePhysical<const uint8_t*>(address);
      if (!guest) continue;
      if (cp.NativeSharedMemory() && cp.NativeSharedMemory()->IsRangeGpuWritten(address, size)) s->gpu_written = true;
      Range r;
      r.address = address;
      r.size = size;
      r.fetch = b.fetch_constant;
      r.staging = c->bytes_used;
      r.guest.assign(guest, guest + size);
      c->bytes_used += (size + 15u) & ~15u;
      s->ranges.push_back(std::move(r));
    }
  }
  // The vertices the CPU runs: the draw's own indices, strided (a clean sample of what it draws).
  const uint32_t n = in.indexed ? in.index_count : draw.vertex_count;
  const uint32_t step = std::max<uint32_t>(1, n / kCpuVertices);
  const uint8_t* indices = in.indexed ? memory->TranslatePhysical<const uint8_t*>(in.guest_index_base & 0x1FFFFFFFu) : nullptr;
  xenos::Endian endian = xenos::Endian(in.index_endian);
  if (!in.index_32bit && endian != xenos::Endian::kNone && endian != xenos::Endian::k8in16) {  // (as the primitive processor reads 16-bit indices)
    endian = endian == xenos::Endian::k8in32 ? xenos::Endian::k8in16 : xenos::Endian::kNone;
  }
  for (uint32_t i = 0; i < n; i += step) {
    uint32_t index = i;
    if (in.indexed) {
      if (!indices) break;
      if (in.index_32bit) {
        std::memcpy(&index, indices + size_t(i) * 4, 4);
        index = xenos::GpuSwap(index, endian);
        if ((index & xenos::kVertexIndexMask) == xenos::kVertexIndexMask) continue;
      } else {
        uint16_t h;
        std::memcpy(&h, indices + size_t(i) * 2, 2);
        index = xenos::GpuSwap(h, endian);
        if (index == 0xFFFFu) continue;
      }
    }
    if (s->attr) {  // (Session 95: what gl_VertexIndex is on this draw: the element plus vertexOffset, or the first vertex plus i; the plan proved the mask and the clamp change nothing)
      s->cpu_indices.push_back(index + uint32_t(s->base));
      continue;
    }
    index = (index + uint32_t(s->base)) & xenos::kVertexIndexMask;
    s->cpu_indices.push_back(std::clamp(index, s->index_min, s->index_max));
  }
  // The guest inputs in one number: across devices only the replay arm can expect it to match.
  uint64_t h = XXH3_64bits(s->floats.data(), sizeof(uint32_t) * s->floats.size());
  h ^= XXH3_64bits(s->fetch.data(), sizeof(uint32_t) * s->fetch.size()) * 3;
  for (const Range& r : s->ranges) h = h * 31 + XXH3_64bits(r.guest.data(), r.guest.size());
  s->input_hash = h;
  std::lock_guard<std::mutex> lock(g->m);
  c->snapshots[l.id] = std::move(s);
}

uint32_t NoteOwnDraw(uint32_t family, uint32_t vertex_count, uint32_t& aux) {
  Capture* c = g->cap.load(std::memory_order_relaxed);
  if (!c) return 0;
  Light l;
  l.id = uint32_t(c->draws.size()) + 1;
  l.count = vertex_count;
  l.family = uint8_t(family);
  const uint64_t shape[3] = {0, 0, l.count};
  l.nth = c->occurrences[XXH3_64bits(shape, sizeof(shape))]++;
  const uint64_t words[4] = {0, 0, l.count, l.nth};
  const bool sampled = !c->picks.empty() && c->picks.count(XXH3_64bits(words, sizeof(words)));
  if (sampled) ++c->picks_found;
  c->draws.push_back(l);
  aux = family | (sampled ? kAuxSampled : 0u) | ((sampled && (c->mode & kModeTruth)) ? kAuxQuery : 0u);
  return l.id;
}

// ---- the replay thread
void StreamBegin(VkCommandBuffer cb) {
  if (t.cb != cb) {  // a new command buffer starts with nothing bound
    t.cb = cb;
    t.bound = Bound{};
    t.in_pass = false;
    t.game_query = false;
    if (t.cap && t.close_pending) Close(cb);
  }
}
void StreamEnd(VkCommandBuffer cb) {
  if (t.cap && t.close_pending && !t.in_pass) Close(cb);
}

void Mark(VkCommandBuffer cb, uint32_t kind, uint32_t id, uint32_t aux) {
  switch (kind) {
    case kMarkFrameOpen:
      t.cap = g->cap.load();
      t.setup_done = false;
      t.close_pending = false;
      if (!t.in_pass) Setup(cb);
      break;
    case kMarkFrameClose:
      if (!t.cap) break;
      g->cap = nullptr;
      if (t.in_pass) t.close_pending = true;
      else Close(cb);
      break;
    case kMarkNextDraw:
      t.pending_id = id;
      t.pending_aux = aux;
      break;
    case kMarkPrevDraw: {  // the emulated draw was recorded before the renderer heard of it: no query, the rest as usual
      if (!t.cap || !t.in_pass || t.cap->passes.empty()) break;
      Capture& c = *t.cap;
      PassRec& p = c.passes.back();
      const uint32_t family = aux & kAuxFamilyMask;
      if (family < kFamilyCount) ++p.family[family];
      if (!p.first_id) p.first_id = id;
      c.pipelines[id] = t.bound.pipeline;
      if ((aux & kAuxSampled) && (c.mode & (kModeTruth | kModeNumbers | kModeSolo))) {
        Run r;
        r.id = id;
        r.aux = aux;
        r.pass = p.index;
        r.skip_query = "the draw was recorded before its mark";
        r.state = t.bound;
        r.call = t.last_call;
        r.have_state = true;
        r.target_w = p.source.width;
        r.target_h = p.source.height;
        c.runs.push_back(std::move(r));
      }
    } break;
    default:
      break;
  }
}

void BeginPass(VkCommandBuffer cb, VkRenderPass render_pass, VkFramebuffer framebuffer) {
  t.in_pass = true;
  if (!t.cap) return;
  Setup(cb);
  Capture& c = *t.cap;
  PassRec p;
  p.index = uint32_t(c.passes.size());
  ChooseSource(p, render_pass, framebuffer);
  if (c.timestamps + 2 <= kMaxTimestamps) {
    p.timestamp = int32_t(c.timestamps);
    c.timestamps += 2;
  }
  WriteTimestamp(cb, p, false);
  c.passes.push_back(p);
}

void BeginDynamicPass(VkCommandBuffer cb, const VkRenderingInfo& info) {
  t.in_pass = true;
  if (!t.cap) return;
  Setup(cb);
  Capture& c = *t.cap;
  PassRec p;
  p.index = uint32_t(c.passes.size());
  p.dynamic = true;
  p.multiview = info.viewMask != 0;
  p.colors = info.colorAttachmentCount;
  if (info.colorAttachmentCount && info.pColorAttachments[0].imageView) {
    const VkRenderingAttachmentInfo& a = info.pColorAttachments[0];
    const bool resolved = a.resolveImageView != VK_NULL_HANDLE;
    std::lock_guard<std::mutex> l(g->m);
    const auto view = g->views.find(K(resolved ? a.resolveImageView : a.imageView));
    const auto image = view != g->views.end() ? g->images.find(K(view->second.image)) : g->images.end();
    if (image != g->images.end()) {
      p.source = {view->second.image, image->second.format, resolved ? a.resolveImageLayout : a.imageLayout, view->second.mip, view->second.layer, info.renderArea.extent.width, info.renderArea.extent.height,
                  uint32_t(image->second.samples)};
      p.target = image->second.serial;
      p.kind = (image->second.usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) ? "transient" : a.storeOp == VK_ATTACHMENT_STORE_OP_STORE ? "color" : "notstored";
    }
  } else {
    p.kind = info.pDepthAttachment ? "depth" : "none";
  }
  if (c.timestamps + 2 <= kMaxTimestamps) {
    p.timestamp = int32_t(c.timestamps);
    c.timestamps += 2;
  }
  WriteTimestamp(cb, p, false);
  c.passes.push_back(p);
}

void EndPass(VkCommandBuffer cb) {
  t.in_pass = false;
  t.pending_id = 0;
  if (!t.cap || t.cap->passes.empty()) return;
  Capture& c = *t.cap;
  PassRec& p = c.passes.back();
  WriteTimestamp(cb, p, true);
  if (c.mode & kModeSheet) BlitCell(cb, p);
  if (c.mode & kModeTruth) CopyBytes(cb, p.index);
  if (c.mode & (kModeNumbers | kModeSolo)) RunDebug(cb, p.index);
  if (t.close_pending) Close(cb);
}

void BindPipeline(VkPipelineBindPoint point, VkPipeline pipeline) {
  if (point == VK_PIPELINE_BIND_POINT_GRAPHICS) t.bound.pipeline = pipeline;
}

void BindSets(VkPipelineBindPoint point, VkPipelineLayout layout, uint32_t first, uint32_t count, const VkDescriptorSet* sets, uint32_t dynamic_count, const uint32_t* dynamic) {
  if (point != VK_PIPELINE_BIND_POINT_GRAPHICS || count > 8 || dynamic_count > 16 || first + count > 8) return;
  Bound& b = t.bound;
  BindCall call;
  call.layout = layout;
  call.first = first;
  call.count = count;
  call.dynamic_count = dynamic_count;
  std::copy(sets, sets + count, call.sets.begin());
  if (dynamic_count) std::copy(dynamic, dynamic + dynamic_count, call.dynamic.begin());
  b.calls.push_back(call);
  const int32_t index = int32_t(b.calls.size()) - 1;
  for (uint32_t i = 0; i < count; ++i) b.call_of_set[first + i] = index;
  // Calls none of whose sets is current any more are gone (the list stays at eight at most).
  if (b.calls.size() > 8) {
    std::vector<BindCall> kept;
    std::array<int32_t, 8> remap{-1, -1, -1, -1, -1, -1, -1, -1};
    std::array<int32_t, 16> new_index;
    new_index.fill(-1);
    for (int32_t i = 0; i < int32_t(b.calls.size()); ++i) {
      if (std::find(b.call_of_set.begin(), b.call_of_set.end(), i) == b.call_of_set.end()) continue;
      new_index[size_t(i)] = int32_t(kept.size());
      kept.push_back(b.calls[size_t(i)]);
    }
    for (size_t s = 0; s < 8; ++s) remap[s] = b.call_of_set[s] >= 0 ? new_index[size_t(b.call_of_set[s])] : -1;
    b.call_of_set = remap;
    b.calls = std::move(kept);
  }
}

void BindIndexBuffer(VkBuffer buffer, VkDeviceSize offset, VkIndexType type) {
  t.bound.index_buffer = buffer;
  t.bound.index_offset = offset;
  t.bound.index_type = type;
}
void BindVertexBuffers(uint32_t first, uint32_t count, const VkBuffer* buffers, const VkDeviceSize* offsets) {
  for (uint32_t i = 0; i < count && first + i < t.bound.vertex_buffers.size(); ++i) {
    t.bound.vertex_buffers[first + i] = buffers[i];
    t.bound.vertex_offsets[first + i] = offsets[i];
  }
}
void PushConstants(VkPipelineLayout layout, VkShaderStageFlags stages, uint32_t offset, uint32_t size, const void* data) {
  if (size > 256) return;
  for (PushRec& p : t.bound.pushes) {
    if (p.offset == offset && p.size == size && p.stages == stages) {
      p.layout = layout;
      std::memcpy(p.data.data(), data, size);
      return;
    }
  }
  if (t.bound.pushes.size() >= 16) t.bound.pushes.erase(t.bound.pushes.begin());
  PushRec p;
  p.layout = layout;
  p.stages = stages;
  p.offset = offset;
  p.size = size;
  std::memcpy(p.data.data(), data, size);
  t.bound.pushes.push_back(p);
}
void SetViewport(uint32_t first, uint32_t count, const VkViewport* viewports) {
  if (first == 0 && count) {
    t.bound.viewport = viewports[0];
    t.bound.has_viewport = true;
  }
}
void SetScissor(uint32_t first, uint32_t count, const VkRect2D* scissors) {
  if (first == 0 && count) {
    t.bound.scissor = scissors[0];
    t.bound.has_scissor = true;
  }
}
void GameQuery(bool begin) { t.game_query = begin; }

void BeforeDraw(VkCommandBuffer cb, bool indexed, uint32_t count, uint32_t instances, uint32_t first, int32_t vertex_offset, uint32_t first_instance) {
  t.last_call = {indexed, count, instances, first, first_instance, vertex_offset};
  if (!t.cap) return;
  Capture& c = *t.cap;
  const uint32_t id = t.pending_id, aux = t.pending_aux;
  t.pending_id = 0;
  PassRec* p = t.in_pass && !c.passes.empty() ? &c.passes.back() : nullptr;
  if (p) {
    ++p->draws;
    if (!p->vk_vs && t.bound.pipeline) {
      std::lock_guard<std::mutex> l(g->m);
      const auto it = g->pipelines.find(K(t.bound.pipeline));
      if (it != g->pipelines.end()) {
        p->vk_vs = it->second->vertex_hash;
        p->vk_fs = it->second->fragment_hash;
      }
    }
  }
  if (!id) {
    ++c.unmarked_draws;
    return;
  }
  if (p) {
    const uint32_t family = aux & kAuxFamilyMask;
    if (family < kFamilyCount) ++p->family[family];
    if (!p->first_id) p->first_id = id;
  }
  c.pipelines[id] = t.bound.pipeline;
  if (!(aux & (kAuxSampled | kAuxQuery)) || !(c.mode & (kModeTruth | kModeNumbers | kModeSolo)) || !p) return;
  Run r;
  r.id = id;
  r.aux = aux;
  r.pass = p->index;
  r.target_w = p->source.width;
  r.target_h = p->source.height;
  if ((aux & kAuxQuery) && (c.mode & kModeTruth)) {
    if (t.game_query) r.skip_query = "the game's own query is running";
    else if (p->multiview) r.skip_query = "a multiview pass";
    else if (!g->occlusion_pool) r.skip_query = "no query pool";
    else if (c.queries >= kMaxQueries) r.skip_query = "out of queries";
    else {
      r.query = int32_t(c.queries++);
      g->real.vkCmdBeginQuery(cb, g->occlusion_pool, uint32_t(r.query), g->precise ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
      t.query_open = true;
    }
  }
  if (aux & kAuxSampled) {
    r.state = t.bound;
    r.call = t.last_call;
    r.have_state = true;
  }
  c.runs.push_back(std::move(r));
}

void AfterDraw(VkCommandBuffer cb) {
  if (!t.query_open) return;
  t.query_open = false;
  const Capture& c = *t.cap;
  g->real.vkCmdEndQuery(cb, g->occlusion_pool, uint32_t(c.queries - 1));
}

}  // namespace diag
}  // namespace rex::graphics::vulkan
