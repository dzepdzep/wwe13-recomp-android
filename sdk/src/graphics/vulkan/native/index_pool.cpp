// wwe2k14 patch (Session 95, Lane B1; was vertex_mirror.cpp): the index pool. See index_pool.h.

#include "index_pool.h"
#include "index_stitch.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <utility>
#include <cstring>
#include <vector>

#include <rex/cvar.h>
#include <rex/graphics/shared_memory.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/hash.h>
#include <rex/logging.h>
#include <rex/system/xmemory.h>
#include <rex/ui/vulkan/util.h>

REXCVAR_DEFINE_BOOL(gpu_native_vertex_mirror_verify, false, "GPU/Native",
                    "Session 95 (item D; for skin once's vertex source): every use of a mirror entry compares a hash of its guest bytes with the one it was made from (a mismatch is counted in the "
                    "[mirror] line and the entry is made again). Diagnostic; off = the entries are compared every 240 guest frames (staggered) only");

REXCVAR_DEFINE_BOOL(gpu_native_index_no_restart, false, "GPU/Native",
                    "Session 95 (out\\s95\\android_structural.md section 3.2): attribute draws run with primitive restart off when their indices hold no reset value; a draw that holds one "
                    "stays on storage, counted. Meant on for class phone, off elsewhere");
REXCVAR_DEFINE_BOOL(gpu_native_index_stitch, false, "GPU/Native",
                    "Session 95 (P2b, out\\s95\\android_structural.md; needs gpu_native_vertex_input and gpu_native_index_no_restart): a triangle strip whose indices hold reset values is drawn as an attribute "
                    "draw from a pool entry of its own, one strip joined by zero-area triangles, instead of staying on storage. Off = those draws stay on storage (reset-values)");
REXCVAR_DEFINE_BOOL(gpu_native_index_verify, false, "GPU/Native",
                    "Session 95 (out\\s95\\android_structural.md section 3.3): every use of an index pool entry hashes its guest bytes, and the pool's hot table is off. Diagnostic");
REXCVAR_DEFINE_INT32(gpu_native_index_pool_mb, 16, "GPU/Native",
                     "Session 95 (read at start; out\\s95\\android_structural.md section 3.3): the index pool's size in MiB. 1 = the small-pool check")
    .range(1, 256)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(gpu_native_index_pool_buckets, false, "GPU/Native",
                    "Session 95 (P0 decision table: callback calls above 1000 a window): the index pool finds the entries a write touches through 64 KiB page buckets instead of walking every entry "
                    "(a write spanning more than 64 pages still walks them all). Marks the same entries; with gpu_native_vinput_census a linear walk after each bucketed mark counts any entry the buckets missed "
                    "(bucket misses, must read 0). Read when the pool is built (first draw). Off = identical")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(gpu_native_vinput_census, false, "GPU/Native",
                    "Session 95 (P0 of the vertex-input work, out\\s95\\android_structural.md; read at start; log only): counts per 600 guest frames what the vertex-input design has to serve ([vinput] census "
                    "lines): typed draws by class and index type, restart, index base, the VGT clamp, reset values in DMA indices, the mirror pool's callback calls and peak bytes, the top shaders that are "
                    "not ours. Every non-crowd DMA draw's indices go through the mirror pool dry (an entry is made and watched, nothing is bound). Changes no draw. Off = identical")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DECLARE(int32_t, gpu_d3d_draw_record);  // (Session 95, r3cmd)

namespace rex::graphics::vulkan::native {

namespace {
constexpr uint32_t kMaxEntryBytesCap = 4u << 20;    // a single entry at most; a bigger buffer is not ours to pool (also at most a quarter of the pool)
constexpr uint64_t kRefreshFrames = 240;            // the safety net: the guest bytes' hash is compared this often
constexpr uint32_t kGuestMax = 0x20000000u;         // the physical space
constexpr uint32_t kAllocateTries = 4096;           // busy occupants stepped over before a placement gives up (Session 95: 64 refused 5 crowd runs with 1.7 of 16 MiB in use, the GPU behind; each step is one map lookup, and only a lagging GPU takes many)

inline uint32_t Bswap32(uint32_t v) { return __builtin_bswap32(v); }

// The vertex shader's index step (xenos_vs.glsl XeGuestVertexIndex) on a raw element: 8-in-16 swap for endian 1 / 2, then the 16-bit halves swapped for 2 / 3.
inline uint32_t IndexStep(uint32_t v, uint32_t endian) {
  if (endian == 1u || endian == 2u) v = ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu);
  if (endian == 2u || endian == 3u) v = (v >> 16) | (v << 16);
  return v;
}
inline uint32_t HotFlags(bool is16, uint32_t endian, bool reset, bool stitch) { return (is16 ? 1u : 0u) | ((endian & 3u) << 1) | (reset ? 8u : 0u) | (stitch ? 16u : 0u); }
inline uint32_t HotIndex(uint32_t base, uint32_t count, uint32_t flags) {
  const uint64_t h = (uint64_t(base) * 0x9E3779B97F4A7C15ull) ^ (uint64_t(count) * 0xC2B2AE3D27D4EB4Full) ^ (uint64_t(flags) * 0x165667B19E3779F9ull);
  return uint32_t(h >> 40) & 8191u;
}
const char* const kFallbackNames[IndexPool::kFbCount] = {"no-variant", "instanced/multiview/special", "memexport", "ps-vertex-fetch", "fetch-type", "index-identity", "reset",
                                                            "pool-full", "pipeline-pending", "gpu-written", "no-pipeline", "other", "not-ours", "not-owned-only", "inline-ring", "restored", "base-not-swapped", "crowd", "special-base"};
}  // namespace

IndexPool::IndexPool(VulkanCommandProcessor& cp) : cp_(cp), census_on_(REXCVAR_GET(gpu_native_vinput_census)), buckets_on_(REXCVAR_GET(gpu_native_index_pool_buckets)) { rex::RegisterQuitFlush(&quit_req_, &quit_ack_); }

IndexPool::~IndexPool() {
  rex::UnregisterQuitFlush(&quit_req_);
  if (REXCVAR_GET(gpu_native_vinput_census) && last_frame_) CensusLine(last_frame_);  // (the last window, cut short by the quit)
  if (watch_handle_ && watch_shared_memory_) watch_shared_memory_->UnregisterGlobalWatch(watch_handle_);
  if (callback_handle_ && callback_memory_) callback_memory_->UnregisterPhysicalMemoryInvalidationCallback(callback_handle_);
  const ui::vulkan::VulkanDevice* device = cp_.GetVulkanDevice();
  if (!device) return;
  const ui::vulkan::VulkanDevice::Functions& dfn = device->functions();
  if (map_) dfn.vkUnmapMemory(device->device(), memory_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device->device(), buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device->device(), memory_);
}

bool IndexPool::EnsurePool() {
  if (map_) return true;
  if (pool_failed_) return false;
  pool_failed_ = true;
  const VkDeviceSize pool_bytes = VkDeviceSize(uint32_t(std::clamp<int32_t>(REXCVAR_GET(gpu_native_index_pool_mb), 1, 256))) << 20;
  max_entry_bytes_ = std::min<VkDeviceSize>(kMaxEntryBytesCap, pool_bytes / 4);
  hot_.assign(kHotSlots, HotSlot{});
  const ui::vulkan::VulkanDevice* device = cp_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = device->functions();
  const VkBufferUsageFlags usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  // Preferred: host-visible AND device-local memory (resizable BAR / UMA): the GPU reads the vertices from its own memory. Else any upload memory.
  bool made = false;
  {
    VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = pool_bytes;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    if (dfn.vkCreateBuffer(device->device(), &info, nullptr, &buffer) == VK_SUCCESS) {
      VkMemoryRequirements req;
      dfn.vkGetBufferMemoryRequirements(device->device(), buffer, &req);
      const uint32_t types = req.memoryTypeBits & device->memory_types().device_local & device->memory_types().host_visible;
      VkMemoryAllocateInfo alloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      alloc.allocationSize = req.size;
      VkDeviceMemory memory = VK_NULL_HANDLE;
      if (types) {
        alloc.memoryTypeIndex = uint32_t(std::countr_zero(types));
        if (dfn.vkAllocateMemory(device->device(), &alloc, nullptr, &memory) == VK_SUCCESS) {
          if (dfn.vkBindBufferMemory(device->device(), buffer, memory, 0) == VK_SUCCESS) {
            buffer_ = buffer;
            memory_ = memory;
            device_local_ = true;
            made = true;
          } else {
            dfn.vkFreeMemory(device->device(), memory, nullptr);
          }
        }
      }
      if (!made) dfn.vkDestroyBuffer(device->device(), buffer, nullptr);
    }
  }
  if (!made && !ui::vulkan::util::CreateDedicatedAllocationBuffer(device, pool_bytes, usage, ui::vulkan::util::MemoryPurpose::kUpload, buffer_, memory_)) {
    REXGPU_WARN("[mirror] the {} MiB index pool could not be created: indexed attribute draws stay off", pool_bytes >> 20);
    return false;
  }
  void* mapped = nullptr;
  if (dfn.vkMapMemory(device->device(), memory_, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS || !mapped) {
    REXGPU_WARN("[mirror] the index pool could not be mapped: indexed attribute draws stay off");
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device->device(), buffer_);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device->device(), memory_);
    return false;
  }
  map_ = static_cast<uint8_t*>(mapped);
  size_ = pool_bytes;
  pool_failed_ = false;
  REXGPU_INFO("[mirror] index pool made: {} MiB host-visible{} (index only), ring placement, {} hot slots, hash check every {} guest frames", pool_bytes >> 20,
              device_local_ ? " and device-local" : "", kHotSlots, kRefreshFrames);
  return true;
}

void IndexPool::Flush(VkDeviceSize off, VkDeviceSize len) {
  const ui::vulkan::VulkanDevice* device = cp_.GetVulkanDevice();
  const VkDeviceSize atom = std::max<VkDeviceSize>(device->properties().nonCoherentAtomSize, 1);
  VkMappedMemoryRange range = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, memory_, 0, VK_WHOLE_SIZE};
  range.offset = off / atom * atom;
  const VkDeviceSize end = (off + len + atom - 1) / atom * atom;
  range.size = end <= size_ ? end - range.offset : VK_WHOLE_SIZE;
  device->functions().vkFlushMappedMemoryRanges(device->device(), 1, &range);
}

bool IndexPool::Allocate(VkDeviceSize len, uint64_t completed, VkDeviceSize* off) {
  len = (len + 15) & ~VkDeviceSize(15);
  if (len > size_) return false;
  VkDeviceSize at = head_;
  bool wrapped = false;
  for (uint32_t attempt = 0; attempt < kAllocateTries; ++attempt) {
    if (at + len > size_) {
      if (wrapped) {
        head_ = 0;
        return false;
      }
      wrapped = true;
      at = 0;
    }
    const VkDeviceSize end = at + len;
    auto it = occupants_.upper_bound(at);
    if (it != occupants_.begin()) {
      auto prev = std::prev(it);
      if (prev->first + prev->second.len > at) it = prev;
    }
    // Busy space (a submission not completed uses it) is stepped over, never waited for.
    const Occupant* busy = nullptr;
    VkDeviceSize busy_end = 0;
    for (auto v = it; v != occupants_.end() && v->first < end; ++v) {
      if (v->second.last_submission && v->second.last_submission > completed) {
        busy = &v->second;
        busy_end = v->first + v->second.len;
        break;
      }
    }
    if (busy) {
      ++counters_.skipped_busy;
      at = busy_end;
      continue;
    }
    for (auto v = it; v != occupants_.end() && v->first < end;) {
      ++epoch_;  // (hot slots may point at this occupant or its entry)
      if (!v->second.dead) {
        auto t = table_.find(v->second.key);
        if (t != table_.end() && t->second.off == v->first) {
          BucketRemove(&t->second);
          table_.erase(t);
        }
        ++counters_.evictions;
        live_bytes_ -= v->second.len;
      }
      v = occupants_.erase(v);
    }
    head_ = end;
    *off = at;
    return true;
  }
  if (at < size_) head_ = at;  // (past the busy space it stepped over: the next try does not walk it again)
  return false;
}

void IndexPool::Watch(const Ctx& c, uint32_t start, uint32_t end) {
  if (!callback_handle_) {
    callback_memory_ = c.mem;
    callback_handle_ = c.mem->RegisterPhysicalMemoryInvalidationCallback(InvalidationThunk, this);
  }
  if (!watch_handle_ && c.shared_memory) {
    watch_shared_memory_ = c.shared_memory;
    watch_handle_ = c.shared_memory->RegisterGlobalWatch(GlobalWatchThunk, this);
  }
  c.mem->EnablePhysicalMemoryAccessCallbacks(start, end - start, true, false);
}

void IndexPool::MarkOverlapping(uint32_t start, uint32_t end, bool by_gpu) {
  if (in_progress_.active && uint64_t(in_progress_.start) < end && in_progress_.end > uint64_t(start)) in_progress_.hit = true;
  std::chrono::steady_clock::time_point t0;
  if (census_on_) t0 = std::chrono::steady_clock::now();
  const auto mark = [&](Entry& en) {
    if (en.stale) return;
    if (uint64_t(en.guest_start) < end && en.guest_end > uint64_t(start)) {
      en.stale = true;
      // (Session 95, r3cmd, gpu_d3d_draw_record: a hot slot looks at its entry's mark, so a mark leaves the others answering: every watch that fired sent the whole table to Place, ~40% of a frame's requests.)
      if (!REXCVAR_GET(gpu_d3d_draw_record)) ++epoch_;
      ++(by_gpu ? counters_.gpu_invalidations : counters_.invalidations);
    }
  };
  const uint64_t first_page = start >> 16, last_page = end > start ? (uint64_t(end) - 1) >> 16 : first_page;
  if (buckets_on_ && last_page - first_page < kBucketWalkMaxPages) {
    uint64_t scanned = 0;
    for (uint64_t p = first_page; p <= last_page && p < buckets_.size(); ++p) {
      for (Entry* en : buckets_[p]) {
        ++scanned;
        mark(*en);
      }
    }
    if (census_on_) {
      mark_scanned_ += scanned;
      mark_ns_ += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());  // (the bucketed walk only; the cross-check below is not timed)
      // The cross-check: whatever a linear walk would still mark is a bucket miss (and is marked, so the run stays right).
      for (auto& kv : table_) {
        Entry& en = kv.second;
        if (!en.stale && uint64_t(en.guest_start) < end && en.guest_end > uint64_t(start)) {
          ++bucket_misses_;
          mark(en);
        }
      }
    }
  } else {
    for (auto& kv : table_) mark(kv.second);
    if (census_on_) mark_scanned_ += table_.size();
  }
  if (census_on_ && !(buckets_on_ && last_page - first_page < kBucketWalkMaxPages))
    mark_ns_ += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
}

void IndexPool::BucketAdd(Entry* en) {
  if (!buckets_on_) return;
  if (buckets_.empty()) buckets_.resize(kGuestMax >> 16);
  for (uint32_t p = en->guest_start >> 16, e = (en->guest_end - 1) >> 16; p <= e && p < buckets_.size(); ++p) buckets_[p].push_back(en);
}

void IndexPool::BucketRemove(Entry* en) {
  if (!buckets_on_ || buckets_.empty() || en->guest_end <= en->guest_start) return;
  for (uint32_t p = en->guest_start >> 16, e = (en->guest_end - 1) >> 16; p <= e && p < buckets_.size(); ++p) {
    auto& b = buckets_[p];
    for (size_t i = 0; i < b.size(); ++i) {
      if (b[i] == en) {
        b[i] = b.back();
        b.pop_back();
        break;
      }
    }
  }
}

std::pair<uint32_t, uint32_t> IndexPool::InvalidationThunk(void* ctx, uint32_t start, uint32_t length, bool) {
  return static_cast<IndexPool*>(ctx)->Invalidated(start, length);
}

std::pair<uint32_t, uint32_t> IndexPool::Invalidated(uint32_t start, uint32_t length) {
  calls_cpu_.fetch_add(1, std::memory_order_relaxed);
  calls_total_.fetch_add(1, std::memory_order_relaxed);
  if (!length) return {0, UINT32_MAX};
  // Widened to 64 KiB and ALWAYS returned: the pages the system stops watching after this call may be larger than the write, and every entry on them is marked here (entries keep their watch
  // by being made again, which enables it again).
  const uint64_t s = uint64_t(start) & ~uint64_t(0xFFFF);
  const uint64_t e = (uint64_t(start) + length + 0xFFFF) & ~uint64_t(0xFFFF);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    MarkOverlapping(uint32_t(s), uint32_t(std::min<uint64_t>(e, UINT32_MAX)), false);
  }
  return {uint32_t(s), uint32_t(e - s)};
}

void IndexPool::GlobalWatchThunk(const std::unique_lock<std::recursive_mutex>&, void* ctx, uint32_t first, uint32_t last, bool by_gpu) {
  IndexPool* self = static_cast<IndexPool*>(ctx);
  (by_gpu ? self->calls_gpu_ : self->calls_watch_cpu_).fetch_add(1, std::memory_order_relaxed);
  if (by_gpu) self->calls_total_.fetch_add(1, std::memory_order_relaxed);
  if (!by_gpu) return;  // (a CPU write is seen by the Memory callback: one mark per fault)
  std::lock_guard<std::mutex> lock(self->mutex_);
  self->MarkOverlapping(first, last == UINT32_MAX ? last : last + 1, by_gpu);
}

bool IndexPool::ConvertIndex(rex::memory::Memory* mem, uint32_t start, uint32_t end, bool is16, uint32_t endian, uint8_t* dst) {
  if (is16) {
    const uint32_t count = (end - start) >> 1;
    const uint16_t* src = mem->TranslatePhysical<const uint16_t*>(start);
    uint16_t* out = reinterpret_cast<uint16_t*>(dst);
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t v = IndexStep(src[i], endian);
      if (v > 0xFFFFu) return false;
      out[i] = uint16_t(v);
    }
  } else {
    const uint32_t count = (end - start) >> 2;
    const uint32_t* src = mem->TranslatePhysical<const uint32_t*>(start);
    uint32_t* out = reinterpret_cast<uint32_t*>(dst);
    for (uint32_t i = 0; i < count; ++i) out[i] = IndexStep(src[i], endian);
  }
  return end > start;
}

// (P2b) The restart the draw would have had: 0xFFFF / 0xFFFFFFFF after the index step, the same compare as ScanIndex.
bool IndexPool::StitchIndex(rex::memory::Memory* mem, uint32_t start, uint32_t end, bool is16, uint32_t endian, uint32_t* bytes, uint32_t* count) {
  if (is16) {
    stitch_in16_.resize((end - start) >> 1);
    if (!ConvertIndex(mem, start, end, true, endian, reinterpret_cast<uint8_t*>(stitch_in16_.data()))) return false;
    if (!StitchTriangleStrip<uint16_t>(stitch_in16_.data(), uint32_t(stitch_in16_.size()), 0xFFFFu, stitch16_)) return false;
    *count = uint32_t(stitch16_.size());
    *bytes = *count * 2u;
  } else {
    stitch_in32_.resize((end - start) >> 2);
    if (!ConvertIndex(mem, start, end, false, endian, reinterpret_cast<uint8_t*>(stitch_in32_.data()))) return false;
    if (!StitchTriangleStrip<uint32_t>(stitch_in32_.data(), uint32_t(stitch_in32_.size()), 0xFFFFFFFFu, stitch32_)) return false;
    *count = uint32_t(stitch32_.size());
    *bytes = *count * 4u;
  }
  return true;
}

// The smallest / largest index among the elements [start, end) of the guest buffer, after the shader's index step.
bool IndexPool::ScanIndex(rex::memory::Memory* mem, uint32_t start, uint32_t end, bool is16, uint32_t endian, bool reset, uint32_t* mn, uint32_t* mx, bool* has_reset) {
  uint32_t lo = 0xFFFFFFFFu, hi = 0, resets = 0;
  if (is16) {
    const uint16_t* src = mem->TranslatePhysical<const uint16_t*>(start);
    for (uint32_t i = 0, n = (end - start) >> 1; i < n; ++i) {
      const uint32_t v = IndexStep(src[i], endian);
      if (reset && v == 0xFFFFu) {
        ++resets;
        continue;
      }
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
  } else {
    const uint32_t* src = mem->TranslatePhysical<const uint32_t*>(start);
    for (uint32_t i = 0, n = (end - start) >> 2; i < n; ++i) {
      const uint32_t v = IndexStep(src[i], endian);
      if (reset && v == 0xFFFFFFFFu) {
        ++resets;
        continue;
      }
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
  }
  *mn = lo;
  *mx = hi;
  *has_reset = resets != 0;
  return lo != 0xFFFFFFFFu;
}

// One index entry, found (and checked) or made. key.kind is 1 (a = the 64 KiB page of its start, b = is16 | endian << 1, c = the slot chosen here), or 2 with stitch (the whole key made by Find: exact range, no slot).
IndexPool::Why IndexPool::Place(const Ctx& c, Key key, uint32_t req_start, uint32_t req_end, bool is16, uint32_t endian, bool reset, bool stitch, Result* out) {
  if (!EnsurePool()) return kNoPool;
  if (req_end <= req_start || req_end - req_start > max_entry_bytes_) return kTooBig;
  if (uint64_t(req_end) > kGuestMax) return kBadRange;
  const bool is_index = true;  // (the vertex half of the old mirror is gone)
  uint32_t start = req_start, end = req_end;
  bool revalidate = false;
  // An index draw's own min / max (the entry's bytes are shared): memoized per distinct range of the entry. mutex_ held.
  const auto give = [&](Entry& e) {
    out->off = e.off + (req_start - e.guest_start);
    out->count = e.stitched_count;
    if (is_index) {
      bool found_memo = false;
      for (const RangeMemo& m : e.memo) {
        if (m.start == req_start && m.end == req_end && m.reset == reset) {
          out->min = m.min;
          out->max = m.max;
          out->has_reset = m.has_reset;
          found_memo = true;
          break;
        }
      }
      if (!found_memo) {
        uint32_t mn = 0, mx = 0;
        bool has_reset = false;
        if (!ScanIndex(c.mem, req_start, req_end, is16, endian, reset, &mn, &mx, &has_reset)) return kBadRange;
        if (e.memo.size() < 64) e.memo.push_back({req_start, req_end, mn, mx, reset, has_reset});
        out->min = mn;
        out->max = mx;
        out->has_reset = has_reset;
      }
    }
    // The hot slot for this exact request (not for a stale entry; verify mode fills them too, only to compare, see Find).
    if (!e.stale && !hot_.empty()) {
      const uint32_t flags = HotFlags(is16, endian, reset, stitch);
      const uint32_t base_key = req_start, count_key = req_end - req_start;
      auto o = occupants_.find(e.off);
      if (o != occupants_.end()) {
        HotSlot& hs = hot_[HotIndex(base_key, count_key, flags)];
        hs.epoch = epoch_;
        hs.base = base_key;
        hs.count = count_key;
        hs.flags = flags;
        hs.offset = out->off;
        hs.min = out->min;
        hs.max = out->max;
        hs.has_reset = out->has_reset;
        hs.stitched_count = e.stitched_count;
        hs.entry = &e;
        hs.occupant = &o->second;
      }
    }
    return kOk;
  };
  const auto count_touch = [&](Entry& e) {
    if (e.touched_window == window_id_) return;
    e.touched_window = window_id_;
    counters_.touched_bytes += e.guest_end - e.guest_start;
  };
  const auto touch = [&](Entry& e) {
    e.last_submission = c.submission;
    auto o = occupants_.find(e.off);
    if (o != occupants_.end()) o->second.last_submission = c.submission;
    count_touch(e);
  };
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_index && !stitch) {
      // The slot: the entry of this page and kind whose range is within 4 KiB of the request, else a free slot of its own (4 at most; the first when all are used).
      int near_slot = -1, free_slot = -1;
      for (uint32_t sl = 0; sl < 4; ++sl) {
        key.c = sl;
        auto f = table_.find(key);
        if (f == table_.end()) {
          if (free_slot < 0) free_slot = int(sl);
          continue;
        }
        const Entry& e = f->second;
        const uint32_t gap = req_start >= e.guest_end ? req_start - e.guest_end : (e.guest_start >= req_end ? e.guest_start - req_end : 0);
        if (gap <= 4096) {
          near_slot = int(sl);
          break;
        }
      }
      key.c = uint32_t(near_slot >= 0 ? near_slot : free_slot >= 0 ? free_slot : 0);
    }
    auto found = table_.find(key);
    if (found != table_.end()) {
      Entry& e = found->second;
      const bool covered = req_start >= e.guest_start && req_end <= e.guest_end;
      bool changed = false;
      if (covered && !e.stale && (c.verify || c.frame >= e.next_check)) {
        // The guest bytes against the hash they were converted from (the pool is not read).
        const uint64_t h = XXH3_64bits(c.mem->TranslatePhysical<const uint8_t*>(e.guest_start), e.guest_end - e.guest_start);
        if (c.verify) {
          if (h != e.hash) ++counters_.verify_mismatch;
        } else {
          ++counters_.refreshes;
          if (h != e.hash) ++counters_.refresh_mismatch;
        }
        e.next_check = c.frame + kRefreshFrames;
        if (h != e.hash) {
          e.stale = true;
          ++epoch_;  // (hot slots on this entry must stop answering at once)
          changed = true;
        }
      }
      if (covered && !e.stale) {
        const Why w = give(e);
        if (w != kOk) return w;
        touch(e);
        ++counters_.hits;
        return kOk;
      }
      // Made again, or re-checked: the union of what it held and what is asked for (a vertex buffer grows its start downward, an index page both ways, within the gap rule above).
      if (!e.stale) ++counters_.grown;
      revalidate = covered && !changed;  // (stale only by a mark: the bytes may be the same)
      start = std::min(start, e.guest_start);
      end = std::max(end, e.guest_end);
      if (end - start > max_entry_bytes_) return kTooBig;
    }
  }
  if (c.shared_memory->IsRangeGpuWritten(start, end - start)) return kGpuWritten;
  // The watch first, the bytes after; a write into the range from here until the entry is inserted marks it stale.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    in_progress_.active = true;
    in_progress_.hit = false;
    in_progress_.start = start;
    in_progress_.end = end;
  }
  Watch(c, start, end);
  const uint64_t hash = XXH3_64bits(c.mem->TranslatePhysical<const uint8_t*>(start), end - start);
  std::lock_guard<std::mutex> lock(mutex_);
  if (revalidate) {
    auto f = table_.find(key);
    if (f != table_.end() && f->second.hash == hash && f->second.guest_start == start && f->second.guest_end == end && !in_progress_.hit) {
      Entry& e = f->second;
      e.stale = false;
      e.next_check = c.frame + kRefreshFrames;
      in_progress_.active = false;
      ++counters_.revalidated;
      const Why w = give(e);
      if (w != kOk) return w;
      touch(e);
      return kOk;
    }
  }
  const auto finish = [&](Why why) {
    in_progress_.active = false;
    return why;
  };
  // What goes in the pool: the guest elements as they are, or the stitched strip (made first: its size is the allocation's).
  uint32_t bytes = end - start, stitched = 0;
  if (stitch) {
    if (!StitchIndex(c.mem, start, end, is16, endian, &bytes, &stitched)) return finish(kBadRange);
    if (bytes > max_entry_bytes_) return finish(kTooBig);
  }
  VkDeviceSize off = 0;
  if (!Allocate(bytes, c.completed, &off)) return finish(kPoolFull);
  // The old entry of this key (it may be gone: the allocation can have swept it): its space stays until a later allocation sweeps it (a submission in flight may still read it).
  {
    auto old = table_.find(key);
    if (old != table_.end()) {
      auto o = occupants_.find(old->second.off);
      if (o != occupants_.end()) {
        o->second.dead = true;
        live_bytes_ -= o->second.len;
      }
      BucketRemove(&old->second);
      table_.erase(old);
      ++epoch_;
    }
  }
  Entry e;
  e.off = off;
  e.guest_start = start;
  e.guest_end = end;
  e.last_submission = c.submission;
  e.stitched_count = stitched;
  if (stitch) {
    std::memcpy(map_ + off, is16 ? static_cast<const void*>(stitch16_.data()) : static_cast<const void*>(stitch32_.data()), bytes);
    if (!stitch_logged_) {
      stitch_logged_ = true;
      REXGPU_INFO("[vinput] stitched the first triangle strip with reset values: {} guest elements of {} bits at 0x{:08X} became {} elements, primitive restart off", (end - start) / (is16 ? 2u : 4u), is16 ? 16 : 32, start,
                  stitched);
    }
  } else if (!ConvertIndex(c.mem, start, end, is16, endian, map_ + off)) {
    head_ = off;  // (the space is given back to the ring: nothing in it is valid)
    return finish(kBadRange);
  }
  e.hash = XXH3_64bits(c.mem->TranslatePhysical<const uint8_t*>(start), end - start);
  Flush(off, bytes);
  e.stale = in_progress_.hit;  // (a write came in while the bytes were read: used this once, made again next time)
  in_progress_.active = false;
  // Staggered: entries made together are not all hash-checked on the same frame.
  e.next_check = c.frame + kRefreshFrames / 2 + (uint64_t(KeyHash()(key)) >> 7) % kRefreshFrames;
  Occupant occ;
  occ.len = (VkDeviceSize(bytes) + 15) & ~VkDeviceSize(15);
  occ.key = key;
  occ.last_submission = c.submission;
  occupants_[off] = occ;
  live_bytes_ += occ.len;
  counters_.peak_bytes = std::max(counters_.peak_bytes, live_bytes_);
  Entry& placed = table_[key] = e;
  BucketAdd(&placed);
  count_touch(placed);
  ++counters_.uploads;
  counters_.upload_bytes += bytes;
  const Why w = give(placed);
  return w;
}

IndexPool::Why IndexPool::Find(const Ctx& c, uint32_t base, uint32_t count, bool is16, uint32_t endian, bool reset, Index* out, bool stitch) {
  const uint32_t isz = is16 ? 2u : 4u;
  const uint32_t start = base & ~(isz - 1);
  const bool verify = c.verify || REXCVAR_GET(gpu_native_index_verify);
  Ctx cv = c;
  cv.verify = verify;
  if (!verify) {
    // The hot table: one direct-mapped slot per (start, bytes, width, order, reset, stitched). A hit that is not due its hash check answers at once.
    const uint32_t flags = HotFlags(is16, endian, reset, stitch);
    const uint32_t bytes = count * isz;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!hot_.empty()) {
      const HotSlot& hs = hot_[HotIndex(start, bytes, flags)];
      if (hs.epoch == epoch_ && hs.base == start && hs.count == bytes && hs.flags == flags && !hs.entry->stale && c.frame < hs.entry->next_check) {
        Entry& en = *hs.entry;
        en.last_submission = c.submission;
        hs.occupant->last_submission = c.submission;
        if (en.touched_window != window_id_) {
          en.touched_window = window_id_;
          counters_.touched_bytes += en.guest_end - en.guest_start;
        }
        ++counters_.hits;
        ++counters_.hot_hits;
        out->offset = hs.offset;
        out->min = hs.min;
        out->max = hs.max;
        out->has_reset = hs.has_reset;
        out->count = hs.stitched_count;
        return kOk;
      }
    }
  }
  // Verify mode: the hot table does not answer, but a slot that would have answered is compared with what Place finds (hot_mismatch).
  HotSlot snap{};
  bool have_snap = false;
  if (verify) {
    const uint32_t flags = HotFlags(is16, endian, reset, stitch);
    const uint32_t bytes = count * isz;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!hot_.empty()) {
      const HotSlot& hs = hot_[HotIndex(start, bytes, flags)];
      if (hs.epoch == epoch_ && hs.base == start && hs.count == bytes && hs.flags == flags && !hs.entry->stale && c.frame < hs.entry->next_check) {
        snap = hs;
        have_snap = true;
      }
    }
  }
  Result e;
  const uint32_t order = uint32_t(is16 ? 1u : 0u) | (endian << 1);
  const Key key = stitch ? Key{2, start, order, count * isz} : Key{1, start >> 16, order, 0};
  const Why w = Place(cv, key, start, start + count * isz, is16, endian, reset, stitch, &e);
  if (w == kOk) {
    out->offset = e.off;
    out->min = e.min;
    out->max = e.max;
    out->has_reset = e.has_reset;
    out->count = e.count;
    if (have_snap && (snap.offset != e.off || snap.min != e.min || snap.max != e.max || snap.has_reset != e.has_reset || snap.stitched_count != e.count)) ++counters_.hot_mismatch;
  } else {
    ++counters_.why[w];
  }
  return w;
}

// Session 95 (gpu_d3d_draw_slot): see index_pool.h.
bool IndexPool::Hold(uint32_t base, uint32_t count, bool is16, uint32_t endian, bool reset, bool stitch, uint32_t min, uint32_t max, bool has_reset, uint32_t stitched_count, HeldIndex* out) {
  const uint32_t isz = is16 ? 2u : 4u;
  const uint32_t start = base & ~(isz - 1);
  const uint32_t flags = HotFlags(is16, endian, reset, stitch);
  const uint32_t bytes = count * isz;
  *out = HeldIndex();
  std::lock_guard<std::mutex> lock(mutex_);
  if (hot_.empty()) return false;
  const HotSlot& hs = hot_[HotIndex(start, bytes, flags)];
  if (hs.epoch != epoch_ || hs.base != start || hs.count != bytes || hs.flags != flags || hs.entry->stale || hs.min != min || hs.max != max || hs.has_reset != has_reset ||
      (stitch && hs.stitched_count != stitched_count)) {
    return false;
  }
  out->epoch = hs.epoch;
  out->entry = hs.entry;
  out->occupant = hs.occupant;
  out->offset = hs.offset;
  return true;
}

bool IndexPool::Touch(const HeldIndex& held, uint64_t submission, uint64_t frame) {
  if (REXCVAR_GET(gpu_native_index_verify)) return false;
  std::lock_guard<std::mutex> lock(mutex_);
  if (!held.entry || held.epoch != epoch_) return false;
  Entry& en = *static_cast<Entry*>(held.entry);
  if (en.stale || frame >= en.next_check) return false;
  en.last_submission = submission;
  static_cast<Occupant*>(held.occupant)->last_submission = submission;
  if (en.touched_window != window_id_) {
    en.touched_window = window_id_;
    counters_.touched_bytes += en.guest_end - en.guest_start;
  }
  ++counters_.hits;
  ++counters_.hot_hits;
  return true;
}

bool IndexPool::HeldGood(const HeldIndex& held, uint64_t frame) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!held.entry || held.epoch != epoch_) return false;
  const Entry& en = *static_cast<const Entry*>(held.entry);
  return !en.stale && frame < en.next_check;
}

uint32_t IndexPool::entries() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return uint32_t(table_.size());
}

void IndexPool::FrameEnd(uint64_t frame) {
  last_frame_ = frame;
  if (!window_first_frame_) window_first_frame_ = frame;
  // Session 95: the quit (RunQuitFlushes) closes the window it cut short, once; the ack goes after the lines are written.
  const bool quit = quit_req_.load(std::memory_order_acquire) && !quit_ack_.load(std::memory_order_relaxed);
  if (!quit && frame - window_first_frame_ < 600) return;
  struct AckAtExit {
    std::atomic<bool>* ack;
    ~AckAtExit() { if (ack) ack->store(true, std::memory_order_release); }
  } ack_at_exit{quit ? &quit_ack_ : nullptr};
  const Counters& n = counters_;
  std::string fb, why;
  uint64_t any_fb = 0;
  for (uint32_t i = 0; i < kFbCount; ++i) {
    any_fb += n.fb[i];
    if (n.fb[i]) fb += fmt::format("{}{} {}", fb.empty() ? "" : ", ", kFallbackNames[i], n.fb[i]);
  }
  if (n.mirrored || n.uploads || n.hits || any_fb) {
    static const char* const kWhy[kWhyCount] = {"ok", "no-pool", "pool-full", "too-big", "gpu-written", "bad-range"};
    for (uint32_t i = 1; i < kWhyCount; ++i) {
      if (n.why[i]) why += fmt::format("{}{} {}", why.empty() ? "" : ", ", kWhy[i], n.why[i]);
    }
    REXGPU_INFO("[mirror] frames {}-{}: mirrored {} draws; fallbacks {{{}}}; entries {}, hits {} (hot {}), grown {}, uploads {} ({} KiB), invalidations cpu {} gpu {}, hash checks {} (changed {}), rechecked unchanged {}, verify mismatches {}, hot mismatches {}, "
                "evictions {}, busy steps {}; refused {{{}}}",
                window_first_frame_, frame, n.mirrored, fb.empty() ? "none" : fb, entries(), n.hits, n.hot_hits, n.grown, n.uploads, n.upload_bytes >> 10, n.invalidations, n.gpu_invalidations, n.refreshes,
                n.refresh_mismatch, n.revalidated, n.verify_mismatch, n.hot_mismatch, n.evictions, n.skipped_busy, why.empty() ? "none" : why);
  }
  if (REXCVAR_GET(gpu_native_vinput_census)) CensusLine(frame);
  counters_ = Counters{};
  counters_.peak_bytes = live_bytes_;
  ++window_id_;
  window_first_frame_ = frame;
}

// Session 95 (P0): every number of the decision table in out\s95\android_structural.md, one window a line.
void IndexPool::CensusLine(uint64_t frame) {
  const Census& k = census_;
  const Counters& n = counters_;
  const uint64_t draws = k.typed + k.not_typed;
  if (!draws) return;
  static const char* const kWhy[kWhyCount] = {"ok", "no-pool", "pool-full", "too-big", "gpu-written", "bad-range"};
  std::string why;
  for (uint32_t i = 1; i < kWhyCount; ++i) {
    if (n.why[i]) why += fmt::format("{}{} {}", why.empty() ? "" : ", ", kWhy[i], n.why[i]);
  }
  const auto types = [](const uint64_t* t) { return fmt::format("none {}, dma {}, converted {}, builtin-auto {}, builtin-dma {}", t[0], t[1], t[2], t[3], t[4]); };
  const uint64_t calls_cpu = calls_cpu_.exchange(0, std::memory_order_relaxed), calls_gpu = calls_gpu_.exchange(0, std::memory_order_relaxed),
                 calls_watch_cpu = calls_watch_cpu_.exchange(0, std::memory_order_relaxed);
  uint64_t mark_ns = 0, mark_scanned = 0, bucket_misses = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    mark_ns = std::exchange(mark_ns_, 0);
    mark_scanned = std::exchange(mark_scanned_, 0);
    bucket_misses = std::exchange(bucket_misses_, 0);
  }
  REXGPU_INFO("[vinput] census frames {}-{}: mark walk {{buckets {}, ns {}, entries scanned {}, bucket misses {}}} (per callback call: {} ns, {} entries)", window_first_frame_, frame, buckets_on_ ? "on" : "off", mark_ns,
              mark_scanned, bucket_misses, (calls_cpu + calls_gpu) ? mark_ns / (calls_cpu + calls_gpu) : 0, (calls_cpu + calls_gpu) ? mark_scanned / (calls_cpu + calls_gpu) : 0);
  REXGPU_INFO("[vinput] census frames {}-{}: draws {} (typed {}, not typed {}); class {{ours {}, translated {}, ring {} (ours {}), crowd {}}}; index {{{}}}; crowd index {{{}}}; restart on {}, base nonzero {}, clamp trivial {}; "
              "dma asked {}: found {}, restart on {}, reset values {} (triangle strip {}, line strip {}), identity refused {}, refused {{{}}}; no-index identity refused {}; "
              "stream refused {{type {}, size {}, endian {}, range {}}}; storage always {{memexport {}, ps-fetch {}, vs-expand {}, line-loop {}}}; "
              "pool {{entries {}, hits {}, converts {} ({} KiB), marks cpu {} gpu {}, callback calls cpu {} gpu {} watch-cpu {}, peak bytes {}, window bytes {}, evictions {}, busy steps {}}}",
              window_first_frame_, frame, draws, k.typed, k.not_typed, k.cls[Census::kOurs], k.cls[Census::kTranslated], k.cls[Census::kRing], k.ring_ours, k.cls[Census::kCrowd], types(k.index), types(k.crowd_index),
              k.restart, k.base_nonzero, k.clamp_trivial, k.dma_asked, k.dma_found, k.dma_restart, k.dma_reset, k.dma_reset_tri_strip, k.dma_reset_line_strip, k.dma_identity, why.empty() ? "none" : why,
              k.plain_identity, k.fetch_type, k.fetch_size, k.fetch_endian, k.fetch_range, k.memexport, k.ps_fetch, k.vs_expand, k.line_loop, entries(), n.hits, n.uploads, n.upload_bytes >> 10, n.invalidations,
              n.gpu_invalidations, calls_cpu, calls_gpu, calls_watch_cpu, n.peak_bytes, n.touched_bytes, n.evictions, n.skipped_busy);
  if (!k.not_ours.empty()) {
    std::vector<std::pair<uint64_t, uint64_t>> top;
    top.reserve(k.not_ours.size());
    for (const auto& kv : k.not_ours) top.emplace_back(kv.second, kv.first);
    const size_t shown = std::min<size_t>(top.size(), 12);
    std::partial_sort(top.begin(), top.begin() + shown, top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::string text;
    for (size_t i = 0; i < shown; ++i) text += fmt::format("{}{:016X} {}", i ? ", " : "", top[i].second, top[i].first);
    REXGPU_INFO("[vinput] census frames {}-{}: {} shaders not ours, top: {}", window_first_frame_, frame, top.size(), text);
  }
  census_ = Census{};
}

}  // namespace rex::graphics::vulkan::native
