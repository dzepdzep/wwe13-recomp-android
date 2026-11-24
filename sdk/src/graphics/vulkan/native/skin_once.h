// wwe2k14 patch (Session 95, gpu_native_skin_once, default off): each wrestler vertex skinned once a frame, into a real vertex buffer every later pass reads.
// The wrestlers' meshes are drawn by 4-6 passes a frame (shadow maps, rim light map, main pass), and every pass ran the whole blend in its vertex shader (docs\native\WRESTLER_DRAWS_MAP.md: 80k index
// vertices became ~300k). Here a typed draw of a listed shader (shader_swap::SkinOnceRow) names a JOB: (vertex range, palette bytes, bool constants b1-b3). The first draw of a job in a frame adds it to
// the submission's list; every other draw with the same key (the rim map and the main pass share palettes; ~70% of the shadow draws too) finds it. The draw itself binds the job's rows as a vertex
// buffer and takes the shader's _so variant (an attribute draw of its own: Draw::vb[0] / vi = 12). EndSubmission records ONE compute dispatch for the submission's jobs at the head of the
// submission, after the hoisted uploads (Record): no render pass is ever broken for it. A job's source is the vertex mirror pool (host-endian vertices, kept right by VertexMirror's own watches).
// A main-pass draw brings what only it knows (the blend form, b4, the c218 normal scale): written into the job while it is still pending, else the draw gets a job of its own.
// Every refusal leaves the draw exactly as it was (the plain skinning shader); counted by reason in the 600-frame line.
// State: everything is in this object, used from the command thread only. A job that runs a draw would own the per-draw Request; the job table and the rings would stay shared behind one owner.
#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <unordered_map>

#include <rex/ui/vulkan/api.h>

namespace rex::memory {
class Memory;
}
namespace rex::graphics::vulkan {
class DeferredCommandBuffer;
class VulkanCommandProcessor;
namespace native {
// (Session 95, r6gamma) true: the next Record leaves out its own COMPUTE -> VERTEX_INPUT barrier (the hoist stream closing barrier covers the write).
void SkinOnceMergeBarrier(bool on);


class SkinOnce {
 public:
  enum Role : uint32_t { kRoleShadow, kRoleRim, kRoleMain, kRoleCount };
  enum Refused : uint32_t { kRefSpecial, kRefBaseExt, kRefNotFresh, kRefIndex, kRefFetch, kRefVariant, kRefMirror, kRefInit, kRefParams, kRefOut, kRefPending, kRefClaim, kRefCount };

  explicit SkinOnce(VulkanCommandProcessor& cp);
  ~SkinOnce();

  struct Request {
    rex::memory::Memory* mem = nullptr;
    VkBuffer mirror_buffer = VK_NULL_HANDLE;
    VkDeviceSize mirror_offset = 0;  // the mirror's place of the fetch constant's address
    uint32_t address = 0;            // the same, guest physical bytes
    uint32_t stride_words = 0;
    uint32_t first = 0, count = 0;   // the vertices the draw's indices name (the fetch constant's numbering)
    const uint32_t* vs_float = nullptr;  // the draw's 1024-word float bank
    uint32_t bools = 0;              // the bool constants' first word
    bool main = false;               // a main-pass draw: `style` and `scale_bits` count, and b4
    uint32_t style = 0;              // CH_SKIN_STYLE
    uint32_t scale_bits = 0;         // the float bits of c218.w + 1
  };
  // Once per listed draw, before Get: a new frame starts a new job table and closes the frame's output space.
  void Tick(uint64_t frame, uint64_t submission, uint64_t completed);
  // True: the job's rows start at *offset (bytes) of buffer(), one 48-byte vertex per row group, vertex `first` first.
  bool Get(const Request& q, VkDeviceSize* offset) {
    const bool ok = GetJob(q, offset);
    if (ok) window_.draw_vertices += q.count;
    return ok;
  }
  VkBuffer buffer() const { return out_buffer_; }
  bool Pending() const { return chunks_ != 0; }
  // The submission's jobs into its hoist stream (after the stream's closing barrier). `submission` = the one being ended.
  void Record(DeferredCommandBuffer& hoist, uint64_t submission);

  void Refuse(Refused why) { ++window_.refused[why]; }
  void NoteTaken(Role role, uint32_t index_count) {
    ++window_.draws[role];
    window_.index_vertices[role] += index_count;
  }

 private:
  bool GetJob(const Request& q, VkDeviceSize* offset);
  static constexpr uint32_t kSlots = 8;                      // params: one slot per submission in flight
  static constexpr uint32_t kSlotRows = 65536;               // 1 MiB of 16-byte rows
  static constexpr uint32_t kOutVertices = 32 * 1024 * 1024 / 48;  // the output ring
  static constexpr uint32_t kChunk = 64;                     // the compute shader's local size

  struct Job {
    uint32_t out_vertex = 0;     // the first vertex in the output ring
    uint32_t row = 0;            // its two rows in params
    uint32_t flags = 0;
    uint32_t scale_bits = 0;
    uint32_t palette = 0;
    uint64_t serial = 0;         // the Record it goes out with
    bool main_set = false;
  };
  struct Segment {
    uint32_t start, end;         // output vertices
    uint64_t submission;         // UINT64_MAX while its frame is open
  };

  bool EnsureInit(VkBuffer mirror_buffer);
  uint32_t PaletteRows(const Request& q, uint32_t bones);
  bool AllocateOut(uint32_t vertices, uint32_t* at);
  bool OpenSlot();
  void WriteJobFlags(const Job& j);
  bool MakeJob(const Request& q, uint64_t key, uint32_t palette_rows, VkDeviceSize* offset);
  void Summary(uint64_t frame);

  VulkanCommandProcessor& cp_;
  bool init_failed_ = false, ready_ = false;
  VkBuffer out_buffer_ = VK_NULL_HANDLE, params_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory out_memory_ = VK_NULL_HANDLE, params_memory_ = VK_NULL_HANDLE;
  uint32_t* params_map_ = nullptr;  // uint32 view of the 16-byte rows
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
  VkDescriptorSet set_ = VK_NULL_HANDLE;
  VkShaderModule module_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;

  uint64_t frame_ = UINT64_MAX, submission_ = 0, completed_ = 0;
  std::unordered_map<uint64_t, Job> jobs_;                 // this frame's
  std::unordered_map<uint64_t, uint32_t> scanned_;         // palette rows of a vertex range (by its mirror place)
  // The open slot: rows from the bottom (jobs, palettes), chunks from the top.
  int32_t slot_ = -1;
  uint32_t slot_rows_ = 0, chunks_ = 0;
  uint64_t serial_ = 1;
  std::array<uint64_t, kSlots> slot_submission_{};
  // The output ring.
  uint32_t out_head_ = 0;
  std::deque<Segment> segments_;

  struct Window {
    uint64_t draws[kRoleCount] = {}, index_vertices[kRoleCount] = {}, refused[kRefCount] = {};
    uint64_t draw_vertices = 0, jobs = 0, job_vertices = 0, found = 0, own_main = 0, palette_rows = 0;
  } window_;
  uint64_t window_first_frame_ = 0;
  bool logged_first_ = false;
};

}  // namespace native
}  // namespace rex::graphics::vulkan
