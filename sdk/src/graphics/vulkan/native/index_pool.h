// wwe2k14 patch (Session 95, Lane B1 of out\s95\android_structural.md; was vertex_mirror.h, item D phases 0-1): the INDEX pool. A plain host-order copy of a draw's DMA index buffer in one pool
// (gpu_native_index_pool_mb MiB, default 16; host-visible mapped memory, device-local where the device has such a type; usage INDEX only), so the typed draw can bind it as an ordinary index
// buffer. The vertex half of the old mirror is gone: vertices are read by the vertex-input path (Lane B2) straight from the shared memory buffer.
//   - Entries: keyed by the 64 KiB guest page of the start (+ element width and byte order, + one of 4 slots); they grow to cover the draws that name them, never across a gap of more than
//     4 KiB (a draw further away gets a slot of its own). A grown entry is made again. A draw's min / max (and whether it holds a reset value) is scanned once per distinct (start, end) of the
//     entry (a small memo of 64, cleared when the entry is made again), so the identity checks use that draw's own range, not the shared bytes' whole range.
//   - Hot table: a direct-mapped table of 1024 slots keyed (base, count, is16, endian, reset) in front of all that. A hit takes the mutex, checks one global epoch, stamps the use on the entry
//     and its occupant and returns: no hash find, no scan of the memo, no map find. The epoch is bumped on every mark, eviction and remake, so a slot never outlives what it points at.
//     A hit on a slot whose entry is due its hash check (frame >= next_check), or with verify on, falls through to Place, so the hash net still sees hot entries.
//   - Stitched entries (Session 95, P2b; key kind 2): a triangle strip with reset values under gpu_native_index_no_restart gets an entry of its own for exactly that (start, bytes), holding one strip
//     without reset values (index_stitch.h), so its own element count. Same watch, hash check, stale rule, eviction and hot table as the plain entries; never shared by offset, never grown.
//   - The pool is a ring: a new entry takes the next free space, skipping (never waiting for) space whose last use is not completed yet and evicting what lies there otherwise; an entry that
//     cannot be placed is a counted "pool full" fallback (the draw takes the storage-buffer path). A stale or grown entry is never overwritten in place: it is made again in new space, so a
//     submission in flight keeps the bytes it was recorded with.
//   - Sync: (1) one Memory invalidation callback (the primitive processor's pattern) over the guest pages each entry was read from, enabled BEFORE the bytes are read; (2) one SharedMemory
//     global watch, which also fires for a GPU write (a resolve / memexport into the range). Both only mark overlapping entries stale (no Vulkan calls, no guest-memory writes). A write that lands
//     while an entry is being made marks that entry stale at once (used this once, made again next time). A CPU write is seen by (1) alone (the global watch is only taken for GPU writes,
//     so one fault marks once). A stale entry is first re-checked: the watch is armed again and the hash of its guest bytes compared; unchanged bytes keep the entry and its space.
//     (3) A hash of each entry's guest bytes is compared every 240 guest frames (staggered per entry) as a safety net, and on every use in verify mode (gpu_native_index_verify or
//     gpu_native_vertex_mirror_verify); the pool is never read back.
// State: everything is in this object (a job that runs a draw would own its own copy of the per-draw inputs; the table and the hot slots are behind one mutex and the callbacks only take it).
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

class IndexPool {
 public:
  explicit IndexPool(VulkanCommandProcessor& cp);
  ~IndexPool();

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
  // The DMA index data: `count` elements of 16 / 32 bits at guest physical `base`, put in host order the way the vertex shader's index step does (8-in-16 swap when endian is 1 or 2, then
  // 16-bit halves swapped when it is 2 or 3), as the same element width; offset = the place of `base` in the pool. min / max = the smallest / largest value among THIS draw's
  // `count` elements (scanned from the guest bytes once per distinct draw range of the entry); a 16-bit element that does not fit 16 bits after the step is refused.
  struct Index {
    VkDeviceSize offset = 0;
    uint32_t min = 0, max = 0;
    bool has_reset = false;  // (Session 95, P0) reset on and this draw's elements hold the restart value
    uint32_t count = 0;      // (Session 95, P2b) a stitched entry: the elements to draw (0 from a plain one: the draw's own count)
  };
  // reset = the draw has primitive restart on: the restart value (0xFFFF / 0xFFFFFFFF, the same bits in every byte order) is left out of min / max - the hardware restarts on it in the mirror's
  // buffer exactly as on the guest's, and it is never a vertex.
  // stitch = (P2b) the same elements as one triangle strip without reset values, in an entry of its own (index_stitch.h); min / max are those of the guest elements, as above.
  Why Find(const Ctx& c, uint32_t base, uint32_t count, bool is16, uint32_t endian, bool reset, Index* out, bool stitch = false);
  // Session 95 (gpu_d3d_draw_slot): a hot slot's answer kept by a draw that repeats. Good while the epoch stands, its entry is not marked and its hash check is not due: Touch then does what a hot hit does for the
  // draw (the entry and its space stay in use by this submission) without the lookup. A held answer is only made from a hot slot that gives the plan's own min, max and reset; false = Find answers.
  struct HeldIndex {
    uint64_t epoch = 0;
    void* entry = nullptr;
    void* occupant = nullptr;
    VkDeviceSize offset = 0;
  };
  bool Hold(uint32_t base, uint32_t count, bool is16, uint32_t endian, bool reset, bool stitch, uint32_t min, uint32_t max, bool has_reset, uint32_t stitched_count, HeldIndex* out);
  bool Touch(const HeldIndex& held, uint64_t submission, uint64_t frame);
  bool HeldGood(const HeldIndex& held, uint64_t frame);
  // (Session 95, the [vinput] line) every call of the two callbacks so far that can mark an entry.
  uint64_t callback_calls() const { return calls_total_.load(std::memory_order_relaxed); }  // monotonic (the census line resets calls_cpu_ / calls_gpu_, not this)

  // The typed draw's fallbacks (counters().fb).
  enum Fallback : uint32_t { kFbNoVariant, kFbInstancedOrMultiview, kFbMemexport, kFbPsFetch, kFbFetchType, kFbIndexIdentity, kFbReset, kFbPool, kFbPipelinePending, kFbGpuWritten, kFbNoPipeline, kFbOther,
                          kFbNotOurs, kFbOwnedOnly, kFbRing, kFbRestored, kFbBaseExt, kFbCrowd, kFbSpecialBase, kFbCount };
  // Counters (the typed draw adds the fallbacks it decides itself).
  struct Counters {
    uint64_t mirrored = 0, uploads = 0, upload_bytes = 0, invalidations = 0, gpu_invalidations = 0, refreshes = 0, refresh_mismatch = 0, verify_mismatch = 0, evictions = 0, skipped_busy = 0,
             grown = 0, revalidated = 0, why[kWhyCount] = {};
    uint64_t fb[kFbCount] = {};  // typed-draw fallbacks by reason (Fallback)
    uint64_t hits = 0;
    uint64_t hot_mismatch = 0;   // (verify mode) a hot slot answer that differs from what Place found
    uint64_t hot_hits = 0;       // (Session 95, B1) of `hits`, the ones the hot table answered
    uint64_t peak_bytes = 0;     // (Session 95, P0) the most live entry bytes in the pool this window
    uint64_t touched_bytes = 0;  // (Session 95, P0) the bytes of the entries this window used, each once
  };
  Counters& counters() { return counters_; }
  uint32_t entries() const;
  // The 600-guest-frame summary line (call once a guest frame).
  void FrameEnd(uint64_t frame);

  // Session 95 (P0: gpu_native_vinput_census, out\s95\android_structural.md): log-only counts of the draws the vertex-input design has to serve, per 600 guest frames ([vinput] census lines).
  // The typed draw fills them (owned_front_own.cpp) and calls Find dry on every non-crowd DMA draw, using nothing from it: no binding, no shader, no draw changes.
  struct Census {
    enum Class : uint32_t { kOurs, kTranslated, kRing, kCrowd, kClassCount };
    uint64_t typed = 0, not_typed = 0, cls[kClassCount] = {}, ring_ours = 0;
    uint64_t index[5] = {}, crowd_index[5] = {};  // by PrimitiveProcessor::ProcessedIndexBufferType
    uint64_t restart = 0, base_nonzero = 0, clamp_trivial = 0;  // of the non-crowd draws
    uint64_t dma_asked = 0, dma_found = 0, dma_restart = 0, dma_reset = 0, dma_reset_tri_strip = 0, dma_reset_line_strip = 0, dma_identity = 0, plain_identity = 0;
    uint64_t fetch_type = 0, fetch_size = 0, fetch_endian = 0, fetch_range = 0;  // non-crowd draws with a stream the vertex buffer binding would refuse (first reason)
    uint64_t memexport = 0, ps_fetch = 0, vs_expand = 0, line_loop = 0;          // non-crowd draws that stay on storage whatever else holds
    std::unordered_map<uint64_t, uint64_t> not_ours;                             // draws of each shader that is not ours (by guest hash): the translated list
  };
  Census& census() { return census_; }

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
    uint32_t kind, a, b, c;  // kind 1 (index): a = the 64 KiB page of the start, b = is16 | endian << 1, c = the slot (up to 4 per page, apart by more than 4 KiB). kind 2 (stitched): a = the start, b the same, c = the bytes
    bool operator==(const Key& o) const { return kind == o.kind && a == o.a && b == o.b && c == o.c; }
  };
  struct KeyHash {
    size_t operator()(const Key& k) const { return size_t((uint64_t(k.a) * 0x9E3779B97F4A7C15ull) ^ (uint64_t(k.b) << 29) ^ (uint64_t(k.c) << 37) ^ (uint64_t(k.kind) << 61)); }
  };
  struct Result {
    VkDeviceSize off = 0;
    uint32_t min = 0, max = 0;
    bool has_reset = false;
    uint32_t count = 0;
  };
  struct Occupant;
  struct RangeMemo {
    uint32_t start, end, min, max;
    bool reset, has_reset;
  };
  struct Entry {
    VkDeviceSize off = 0;
    uint32_t guest_start = 0, guest_end = 0;  // the guest bytes the entry was read from [start, end), watched
    std::vector<RangeMemo> memo;              // an index entry: the min / max of each distinct draw range scanned (cleared when the entry is made again)
    uint64_t hash = 0;                        // of the guest bytes
    uint64_t last_submission = 0;
    uint64_t next_check = 0;
    uint64_t touched_window = 0;  // (Session 95, P0) the census window that last counted this entry's bytes
    uint32_t stitched_count = 0;  // (Session 95, P2b) a stitched entry: its elements
    bool stale = false;
  };
  struct Occupant {  // one placed allocation (by pool offset), live until swept by a later allocation
    VkDeviceSize len = 0;
    Key key{};
    uint64_t last_submission = 0;
    bool dead = false;  // replaced by a newer entry of the same key: the table no longer points here
  };
  // One hot-table slot: valid while `epoch` equals epoch_ (bumped on every eviction, sweep and remake - a mark sets the entry's own stale flag instead - so `entry` / `occupant` never dangle while it matches).
  struct HotSlot {
    uint64_t epoch = 0;  // 0 = empty
    uint32_t base = 0, count = 0, flags = 0;  // flags: is16 | endian << 1 | reset << 3 | stitch << 4
    VkDeviceSize offset = 0;
    uint32_t min = 0, max = 0;
    bool has_reset = false;
    uint32_t stitched_count = 0;
    Entry* entry = nullptr;
    Occupant* occupant = nullptr;
  };
  static constexpr uint32_t kHotSlots = 8192;  // (Session 95, r3cmd: 1024 slots left 25-45% of the requests of a frame to Place, by conflicts)
  bool EnsurePool();
  // Takes `len` bytes (aligned 16) at the head of the ring, skipping space still in use by an uncompleted submission; false when there is none. mutex_ held.
  bool Allocate(VkDeviceSize len, uint64_t completed, VkDeviceSize* off);
  void Flush(VkDeviceSize off, VkDeviceSize len);
  void Watch(const Ctx& c, uint32_t start, uint32_t end);
  void MarkOverlapping(uint32_t start, uint32_t end, bool by_gpu);  // mutex_ held
  static std::pair<uint32_t, uint32_t> InvalidationThunk(void* ctx, uint32_t start, uint32_t length, bool exact);
  static void GlobalWatchThunk(const std::unique_lock<std::recursive_mutex>& lock, void* ctx, uint32_t first, uint32_t last, bool by_gpu);
  std::pair<uint32_t, uint32_t> Invalidated(uint32_t start, uint32_t length);
  // The converted bytes of an entry's guest range into `dst` (the shader's index step).
  static bool ConvertIndex(rex::memory::Memory* mem, uint32_t start, uint32_t end, bool is16, uint32_t endian, uint8_t* dst);
  // (P2b) One strip without reset values from the guest elements [start, end) into stitch16_ / stitch32_; *bytes and *count = its size; false = nothing to draw from or a 16-bit element that does not fit.
  bool StitchIndex(rex::memory::Memory* mem, uint32_t start, uint32_t end, bool is16, uint32_t endian, uint32_t* bytes, uint32_t* count);
  static bool ScanIndex(rex::memory::Memory* mem, uint32_t start, uint32_t end, bool is16, uint32_t endian, bool reset, uint32_t* mn, uint32_t* mx, bool* has_reset);
  void CensusLine(uint64_t frame);  // (Session 95, P0) the [vinput] census lines of the window ending at `frame`
  Why Place(const Ctx& c, Key key, uint32_t req_start, uint32_t req_end, bool is16, uint32_t endian, bool reset, bool stitch, Result* out);

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
  // (Session 95) gpu_native_index_pool_buckets: Entry* by the 64 KiB guest pages each covers (mutex_ held); empty until the first entry when on.
  static constexpr uint64_t kBucketWalkMaxPages = 64;
  std::vector<std::vector<Entry*>> buckets_;
  const bool census_on_, buckets_on_;
  std::atomic<bool> quit_req_{false}, quit_ack_{false};  // (Session 95) rex::RegisterQuitFlush: the quit closes the last window
  uint64_t mark_ns_ = 0, mark_scanned_ = 0, bucket_misses_ = 0;  // (census only, mutex_ held)
  void BucketAdd(Entry* en);
  void BucketRemove(Entry* en);
  std::vector<HotSlot> hot_;  // kHotSlots, direct-mapped (mutex_ held)
  uint64_t epoch_ = 1;         // bumped on every eviction, sweep and remake (mutex_ held); a mark sets the entry's stale flag
  VkDeviceSize max_entry_bytes_ = 4u << 20;
  std::map<VkDeviceSize, Occupant> occupants_;
  // The range being read right now (between the watch and the entry's insertion): a write into it marks the new entry stale.
  struct InProgress {
    bool active = false, hit = false;
    uint32_t start = 0, end = 0;
  } in_progress_;
  Counters counters_;
  Census census_;
  uint64_t live_bytes_ = 0, window_id_ = 1, last_frame_ = 0;
  // (Session 95, P0) every call of the two callbacks, whatever they mark: cpu = Invalidated, gpu = the global watch by a GPU write, watch_cpu = the global watch by a CPU write (it returns at once).
  std::atomic<uint64_t> calls_cpu_{0}, calls_gpu_{0}, calls_watch_cpu_{0}, calls_total_{0};
  std::unordered_map<uint64_t, PipeMemo> pipe_memo_;
  // (P2b) StitchIndex's scratch: the elements in host order, then the strip. Only the thread that calls Find touches them.
  std::vector<uint16_t> stitch_in16_, stitch16_;
  std::vector<uint32_t> stitch_in32_, stitch32_;
  bool stitch_logged_ = false;
  uint64_t window_first_frame_ = 0;
};

}  // namespace native
}  // namespace rex::graphics::vulkan
