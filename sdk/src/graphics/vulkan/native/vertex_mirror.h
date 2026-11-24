// wwe2k14 patch (Session 95, item D phases 0-1: docs\native\d3d\S93_MIRROR_DESIGN.md; the mirror draw and its switch are gone: this stays as skin once's vertex source, SkinMirror()): the memory mirror of our arena draws' vertex
// and index data. A plain host-endian copy of a draw's one vertex buffer and of its DMA index buffer in one pool (16 MiB, host-visible mapped memory, device-local where the
// device has such a type), so the typed draw can bind them as ordinary vertex / index buffers and run our vertex-input (_vi) shader variant: the draw then touches no
// storage-buffer range and requests no residency.
//   - Entries: a vertex buffer is keyed by the END of its guest range (a fetch constant's address + size; stable across stream offsets) and grows its START downward when a draw
//     names more; an index buffer is keyed by the 64 KiB guest page of its start (+ element width and byte order, + one of 4 slots) and grows to cover the draws that name it - but never
//     across a gap of more than 4 KiB (a draw further away gets a slot of its own). A grown entry is made again. An index draw's min / max is scanned once per distinct (start, end)
//     of the entry (a small memo cleared when the entry is made again), so the identity checks use that draw's own range, not the shared bytes' whole range.
//   - The pool is a ring: a new entry takes the next free space, skipping (never waiting for) space whose last use is not completed yet and evicting what lies there otherwise; an entry that
//     cannot be placed is a counted "pool full" fallback (the draw takes the storage-buffer path). A stale or grown entry is never overwritten in place: it is made again in new space, so a
//     submission in flight keeps the bytes it was recorded with.
//   - Sync: (1) one Memory invalidation callback (the primitive processor's pattern) over the guest pages each entry was read from, enabled BEFORE the bytes are read; (2) one SharedMemory
//     global watch, which also fires for a GPU write (a resolve / memexport into the range). Both only mark overlapping entries stale (no Vulkan calls, no guest-memory writes). A write that lands
//     while an entry is being made marks that entry stale at once (used this once, made again next time). A CPU write is seen by (1) alone (the global watch is only taken for GPU writes,
//     so one fault marks once). A stale entry is first re-checked: the watch is armed again and the hash of its guest bytes compared; unchanged bytes keep the entry and its space.
//     (3) A hash of each entry's guest bytes is compared every 240 guest frames (staggered per entry) as a safety net, and on every use in verify mode (gpu_native_vertex_mirror_verify);
//     the pool is never read back.
// State: everything is in this object (a job that runs a draw would own its own copy of the per-draw inputs; the table is behind one mutex and the callbacks only take it).
#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <rex/ui/vulkan/api.h>

namespace rex::memory {
class Memory;
}
namespace rex::graphics {
class SharedMemory;
}
namespace rex::graphics::vulkan {
class VulkanCommandProcessor;
namespace native {

class VertexMirror {
 public:
  explicit VertexMirror(VulkanCommandProcessor& cp);
  ~VertexMirror();

  struct Ctx {
    rex::memory::Memory* mem = nullptr;
    rex::graphics::SharedMemory* shared_memory = nullptr;
    uint64_t submission = 0, completed = 0, frame = 0;
    bool verify = false;
  };
  // Why a lookup failed (the counters of the 600-frame line).
  enum Why : uint32_t { kOk, kNoPool, kPoolFull, kTooBig, kGpuWritten, kBadRange, kWhyCount };

  // The pool buffer (VK_NULL_HANDLE until the first successful call).
  VkBuffer buffer() const { return buffer_; }
  // The vertex data [address, address + size) (bytes, guest physical), 8-in-32 swapped into host order; *offset = the place of `address` in the pool.
  Why Vertex(const Ctx& c, uint32_t address, uint32_t size, VkDeviceSize* offset);
  // The DMA index data: `count` elements of 16 / 32 bits at guest physical `base`, put in host order the way the vertex shader's index step does (8-in-16 swap when endian is 1 or 2, then
  // 16-bit halves swapped when it is 2 or 3), as the same element width; offset = the place of `base` in the pool. min / max = the smallest / largest value among THIS draw's
  // `count` elements (scanned from the guest bytes once per distinct draw range of the entry); a 16-bit element that does not fit 16 bits after the step is refused.
  struct Index {
    VkDeviceSize offset = 0;
    uint32_t min = 0, max = 0;
  };
  // reset = the draw has primitive restart on: the restart value (0xFFFF / 0xFFFFFFFF, the same bits in every byte order) is left out of min / max - the hardware restarts on it in the mirror's
  // buffer exactly as on the guest's, and it is never a vertex.
  Why IndexData(const Ctx& c, uint32_t base, uint32_t count, bool is16, uint32_t endian, bool reset, Index* out);

  // The typed draw's fallbacks (counters().fb).
  enum Fallback : uint32_t { kFbNoVariant, kFbInstancedOrMultiview, kFbMemexport, kFbPsFetch, kFbFetchType, kFbIndexIdentity, kFbReset, kFbPool, kFbPipelinePending, kFbGpuWritten, kFbNoPipeline, kFbOther,
                          kFbNotOurs, kFbOwnedOnly, kFbRing, kFbRestored, kFbBaseExt, kFbCrowd, kFbSpecialBase, kFbCount };
  // Counters (the typed draw adds the fallbacks it decides itself).
  struct Counters {
    uint64_t mirrored = 0, uploads = 0, upload_bytes = 0, invalidations = 0, gpu_invalidations = 0, refreshes = 0, refresh_mismatch = 0, verify_mismatch = 0, evictions = 0, skipped_busy = 0,
             grown = 0, revalidated = 0, why[kWhyCount] = {};
    uint64_t fb[kFbCount] = {};  // typed-draw fallbacks by reason (Fallback)
    uint64_t hits = 0;
  };
  Counters& counters() { return counters_; }
  uint32_t entries() const;
  // The 600-guest-frame summary line (call once a guest frame).
  void FrameEnd(uint64_t frame);

  // The vertex-input variant modules, by (plain module, modification, ext) of the base draw (made once; cleared with the caches). The pipeline is made by the claim, not here.
  struct PipeMemo {
    VkShaderModule module = VK_NULL_HANDLE;
    bool failed = false;  // no variant module (refused / not translatable): never retried
  };
  static uint64_t MemoKey(VkShaderModule module, uint64_t modification, uint32_t ext) {
    uint64_t h = uint64_t(reinterpret_cast<uintptr_t>(module)) * 0x9E3779B97F4A7C15ull;
    h ^= modification + 0x7F4A7C159E3779B9ull + (h << 6) + (h >> 2);
    h ^= uint64_t(ext) * 0xC2B2AE3D27D4EB4Full;
    return h;
  }
  std::unordered_map<uint64_t, PipeMemo>& pipe_memo() { return pipe_memo_; }
  void ClearPipeMemo() { pipe_memo_.clear(); }

 private:
  struct Key {
    uint32_t kind, a, b, c;  // c: the slot (index entries: up to 4 per page, apart by more than 4 KiB)
    bool operator==(const Key& o) const { return kind == o.kind && a == o.a && b == o.b && c == o.c; }
  };
  struct KeyHash {
    size_t operator()(const Key& k) const { return size_t((uint64_t(k.a) * 0x9E3779B97F4A7C15ull) ^ (uint64_t(k.b) << 29) ^ (uint64_t(k.c) << 37) ^ (uint64_t(k.kind) << 61)); }
  };
  struct Result {
    VkDeviceSize off = 0;
    uint32_t min = 0, max = 0;
  };
  struct RangeMemo {
    uint32_t start, end, min, max;
    bool reset;
  };
  struct Entry {
    VkDeviceSize off = 0;
    uint32_t guest_start = 0, guest_end = 0;  // the guest bytes the entry was read from [start, end), watched
    std::vector<RangeMemo> memo;              // an index entry: the min / max of each distinct draw range scanned (cleared when the entry is made again)
    uint64_t hash = 0;                        // of the guest bytes
    uint64_t last_submission = 0;
    uint64_t next_check = 0;
    bool stale = false;
  };
  struct Occupant {  // one placed allocation (by pool offset), live until swept by a later allocation
    VkDeviceSize len = 0;
    Key key{};
    uint64_t last_submission = 0;
    bool dead = false;  // replaced by a newer entry of the same key: the table no longer points here
  };
  bool EnsurePool();
  // Takes `len` bytes (aligned 16) at the head of the ring, skipping space still in use by an uncompleted submission; false when there is none. mutex_ held.
  bool Allocate(VkDeviceSize len, uint64_t completed, VkDeviceSize* off);
  void Flush(VkDeviceSize off, VkDeviceSize len);
  void Watch(const Ctx& c, uint32_t start, uint32_t end);
  void MarkOverlapping(uint32_t start, uint32_t end, bool by_gpu);  // mutex_ held
  static std::pair<uint32_t, uint32_t> InvalidationThunk(void* ctx, uint32_t start, uint32_t length, bool exact);
  static void GlobalWatchThunk(const std::unique_lock<std::recursive_mutex>& lock, void* ctx, uint32_t first, uint32_t last, bool by_gpu);
  std::pair<uint32_t, uint32_t> Invalidated(uint32_t start, uint32_t length);
  // The converted bytes of an entry's guest range into `dst` (vertex: bswap32 each dword; index: the shader's index step), returning an index's min / max.
  static void ConvertVertex(rex::memory::Memory* mem, uint32_t start, uint32_t end, uint32_t* dst);
  static bool ConvertIndex(rex::memory::Memory* mem, uint32_t start, uint32_t end, bool is16, uint32_t endian, uint8_t* dst);
  static bool ScanIndex(rex::memory::Memory* mem, uint32_t start, uint32_t end, bool is16, uint32_t endian, bool reset, uint32_t* mn, uint32_t* mx);
  Why Place(const Ctx& c, Key key, uint32_t req_start, uint32_t req_end, bool is16, uint32_t endian, bool reset, Result* out);

  VulkanCommandProcessor& cp_;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  uint8_t* map_ = nullptr;
  VkDeviceSize size_ = 0;
  VkDeviceSize head_ = 0;
  bool pool_failed_ = false;
  bool device_local_ = false;
  void* callback_handle_ = nullptr;
  rex::memory::Memory* callback_memory_ = nullptr;
  void* watch_handle_ = nullptr;
  rex::graphics::SharedMemory* watch_shared_memory_ = nullptr;
  mutable std::mutex mutex_;
  std::unordered_map<Key, Entry, KeyHash> table_;
  std::map<VkDeviceSize, Occupant> occupants_;
  // The range being read right now (between the watch and the entry's insertion): a write into it marks the new entry stale.
  struct InProgress {
    bool active = false, hit = false;
    uint32_t start = 0, end = 0;
  } in_progress_;
  Counters counters_;
  std::unordered_map<uint64_t, PipeMemo> pipe_memo_;
  uint64_t window_first_frame_ = 0;
};

}  // namespace native
}  // namespace rex::graphics::vulkan
