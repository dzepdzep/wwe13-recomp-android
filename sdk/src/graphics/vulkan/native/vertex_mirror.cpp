// wwe2k14 patch (Session 95, s95-all): since gpu_native_vertex_input this pool serves gpu_native_skin_once only (its host-order vertices and indices); typed draws no longer take a mirror draw.
// wwe2k14 patch (Session 95, item D phases 0-1): the vertex / index mirror pool. See vertex_mirror.h.

#include "vertex_mirror.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <vector>

#include <rex/cvar.h>
#include <rex/graphics/shared_memory.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/hash.h>
#include <rex/logging.h>
#include <rex/system/xmemory.h>
#include <rex/ui/vulkan/util.h>

REXCVAR_DECLARE(bool, gpu_native_skin_once);  // (Session 95: defined in skin_once.cpp)

REXCVAR_DECLARE(bool, gpu_native_vertex_mirror_verify);  // (Session 95: defined in index_pool.cpp)

namespace rex::graphics::vulkan::native {

namespace {
constexpr VkDeviceSize kPoolBytes = 16u << 20;      // 4x the measured union of an arena frame (phase 0 census: 3.8-4.1 MiB)
constexpr uint32_t kMaxEntryBytes = 4u << 20;       // a single entry; a bigger buffer is not ours to mirror
constexpr uint64_t kRefreshFrames = 240;            // the safety net: the guest bytes' hash is compared this often
constexpr uint32_t kGuestMax = 0x20000000u;         // the physical space
constexpr uint32_t kAllocateTries = 64;             // busy occupants stepped over before a placement gives up

inline uint32_t Bswap32(uint32_t v) { return __builtin_bswap32(v); }

// The vertex shader's index step (xenos_vs.glsl XeGuestVertexIndex) on a raw element: 8-in-16 swap for endian 1 / 2, then the 16-bit halves swapped for 2 / 3.
inline uint32_t IndexStep(uint32_t v, uint32_t endian) {
  if (endian == 1u || endian == 2u) v = ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu);
  if (endian == 2u || endian == 3u) v = (v >> 16) | (v << 16);
  return v;
}
const char* const kFallbackNames[VertexMirror::kFbCount] = {"no-variant", "instanced/multiview/special", "memexport", "ps-vertex-fetch", "fetch-type", "index-identity", "reset",
                                                            "pool-full", "pipeline-pending", "gpu-written", "no-pipeline", "other", "not-ours", "not-owned-only", "inline-ring", "restored", "base-not-swapped", "crowd", "special-base"};
}  // namespace

VertexMirror::VertexMirror(VulkanCommandProcessor& cp) : cp_(cp) {}

VertexMirror::~VertexMirror() {
  if (watch_handle_ && watch_shared_memory_) watch_shared_memory_->UnregisterGlobalWatch(watch_handle_);
  if (callback_handle_ && callback_memory_) callback_memory_->UnregisterPhysicalMemoryInvalidationCallback(callback_handle_);
  const ui::vulkan::VulkanDevice* device = cp_.GetVulkanDevice();
  if (!device) return;
  const ui::vulkan::VulkanDevice::Functions& dfn = device->functions();
  if (map_) dfn.vkUnmapMemory(device->device(), memory_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device->device(), buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device->device(), memory_);
}

bool VertexMirror::EnsurePool() {
  if (map_) return true;
  if (pool_failed_) return false;
  pool_failed_ = true;
  const ui::vulkan::VulkanDevice* device = cp_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = device->functions();
  const VkBufferUsageFlags usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                   (REXCVAR_GET(gpu_native_skin_once) ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT : VkBufferUsageFlags(0));  // (Session 95: skin_once.comp reads the pool)
  // Preferred: host-visible AND device-local memory (resizable BAR / UMA): the GPU reads the vertices from its own memory. Else any upload memory.
  bool made = false;
  {
    VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = kPoolBytes;
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
  if (!made && !ui::vulkan::util::CreateDedicatedAllocationBuffer(device, kPoolBytes, usage, ui::vulkan::util::MemoryPurpose::kUpload, buffer_, memory_)) {
    REXGPU_WARN("[mirror] the {} MiB pool could not be created: the mirror stays off", kPoolBytes >> 20);
    return false;
  }
  void* mapped = nullptr;
  if (dfn.vkMapMemory(device->device(), memory_, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS || !mapped) {
    REXGPU_WARN("[mirror] the pool could not be mapped: the mirror stays off");
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device->device(), buffer_);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device->device(), memory_);
    return false;
  }
  map_ = static_cast<uint8_t*>(mapped);
  size_ = kPoolBytes;
  pool_failed_ = false;
  REXGPU_INFO("[mirror] pool made: {} MiB host-visible{} (vertex + index), ring placement, hash check every {} guest frames", kPoolBytes >> 20, device_local_ ? " and device-local" : "",
              kRefreshFrames);
  return true;
}

void VertexMirror::Flush(VkDeviceSize off, VkDeviceSize len) {
  const ui::vulkan::VulkanDevice* device = cp_.GetVulkanDevice();
  const VkDeviceSize atom = std::max<VkDeviceSize>(device->properties().nonCoherentAtomSize, 1);
  VkMappedMemoryRange range = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, memory_, 0, VK_WHOLE_SIZE};
  range.offset = off / atom * atom;
  const VkDeviceSize end = (off + len + atom - 1) / atom * atom;
  range.size = end <= size_ ? end - range.offset : VK_WHOLE_SIZE;
  device->functions().vkFlushMappedMemoryRanges(device->device(), 1, &range);
}

bool VertexMirror::Allocate(VkDeviceSize len, uint64_t completed, VkDeviceSize* off) {
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
      if (!v->second.dead) {
        auto t = table_.find(v->second.key);
        if (t != table_.end() && t->second.off == v->first) table_.erase(t);
        ++counters_.evictions;
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

void VertexMirror::Watch(const Ctx& c, uint32_t start, uint32_t end) {
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

void VertexMirror::MarkOverlapping(uint32_t start, uint32_t end, bool by_gpu) {
  if (in_progress_.active && uint64_t(in_progress_.start) < end && in_progress_.end > uint64_t(start)) in_progress_.hit = true;
  for (auto& kv : table_) {
    Entry& en = kv.second;
    if (en.stale) continue;
    if (uint64_t(en.guest_start) < end && en.guest_end > uint64_t(start)) {
      en.stale = true;
      ++(by_gpu ? counters_.gpu_invalidations : counters_.invalidations);
    }
  }
}

std::pair<uint32_t, uint32_t> VertexMirror::InvalidationThunk(void* ctx, uint32_t start, uint32_t length, bool) {
  return static_cast<VertexMirror*>(ctx)->Invalidated(start, length);
}

std::pair<uint32_t, uint32_t> VertexMirror::Invalidated(uint32_t start, uint32_t length) {
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

void VertexMirror::GlobalWatchThunk(const std::unique_lock<std::recursive_mutex>&, void* ctx, uint32_t first, uint32_t last, bool by_gpu) {
  if (!by_gpu) return;  // (a CPU write is seen by the Memory callback: one mark per fault)
  VertexMirror* self = static_cast<VertexMirror*>(ctx);
  std::lock_guard<std::mutex> lock(self->mutex_);
  self->MarkOverlapping(first, last == UINT32_MAX ? last : last + 1, by_gpu);
}

void VertexMirror::ConvertVertex(rex::memory::Memory* mem, uint32_t start, uint32_t end, uint32_t* dst) {
  const uint32_t* src = mem->TranslatePhysical<const uint32_t*>(start);
  const uint32_t n = (end - start) >> 2;
  for (uint32_t i = 0; i < n; ++i) dst[i] = Bswap32(src[i]);
}

bool VertexMirror::ConvertIndex(rex::memory::Memory* mem, uint32_t start, uint32_t end, bool is16, uint32_t endian, uint8_t* dst) {
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

// The smallest / largest index among the elements [start, end) of the guest buffer, after the shader's index step.
bool VertexMirror::ScanIndex(rex::memory::Memory* mem, uint32_t start, uint32_t end, bool is16, uint32_t endian, bool reset, uint32_t* mn, uint32_t* mx) {
  uint32_t lo = 0xFFFFFFFFu, hi = 0;
  if (is16) {
    const uint16_t* src = mem->TranslatePhysical<const uint16_t*>(start);
    for (uint32_t i = 0, n = (end - start) >> 1; i < n; ++i) {
      const uint32_t v = IndexStep(src[i], endian);
      if (reset && v == 0xFFFFu) continue;
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
  } else {
    const uint32_t* src = mem->TranslatePhysical<const uint32_t*>(start);
    for (uint32_t i = 0, n = (end - start) >> 2; i < n; ++i) {
      const uint32_t v = IndexStep(src[i], endian);
      if (reset && v == 0xFFFFFFFFu) continue;
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
  }
  *mn = lo;
  *mx = hi;
  return lo != 0xFFFFFFFFu;
}

// One entry, found (and checked) or made. key.kind 0 = vertex (a = the END of the guest range), 1 = index (a = the 64 KiB page of its start, b = is16 | endian << 1, c = the slot chosen here).
VertexMirror::Why VertexMirror::Place(const Ctx& c, Key key, uint32_t req_start, uint32_t req_end, bool is16, uint32_t endian, bool reset, Result* out) {
  if (!EnsurePool()) return kNoPool;
  if (req_end <= req_start || req_end - req_start > kMaxEntryBytes) return kTooBig;
  if (uint64_t(req_end) > kGuestMax) return kBadRange;
  const bool is_index = key.kind == 1;
  uint32_t start = req_start, end = req_end;
  bool revalidate = false;
  // An index draw's own min / max (the entry's bytes are shared): memoized per distinct range of the entry. mutex_ held.
  const auto give = [&](Entry& e) {
    out->off = e.off + (req_start - e.guest_start);
    if (is_index) {
      bool found_memo = false;
      for (const RangeMemo& m : e.memo) {
        if (m.start == req_start && m.end == req_end && m.reset == reset) {
          out->min = m.min;
          out->max = m.max;
          found_memo = true;
          break;
        }
      }
      if (!found_memo) {
        uint32_t mn = 0, mx = 0;
        if (!ScanIndex(c.mem, req_start, req_end, is16, endian, reset, &mn, &mx)) return kBadRange;
        if (e.memo.size() < 64) e.memo.push_back({req_start, req_end, mn, mx, reset});
        out->min = mn;
        out->max = mx;
      }
    }
    return kOk;
  };
  const auto touch = [&](Entry& e) {
    e.last_submission = c.submission;
    auto o = occupants_.find(e.off);
    if (o != occupants_.end()) o->second.last_submission = c.submission;
  };
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_index) {
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
      if (end - start > kMaxEntryBytes) return kTooBig;
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
  in_progress_.hit = false;  // (the range is watched now; the read below starts after this point)
  const auto finish = [&](Why why) {
    in_progress_.active = false;
    return why;
  };
  VkDeviceSize off = 0;
  if (!Allocate(end - start, c.completed, &off)) return finish(kPoolFull);
  // The old entry of this key (it may be gone: the allocation can have swept it): its space stays until a later allocation sweeps it (a submission in flight may still read it).
  {
    auto old = table_.find(key);
    if (old != table_.end()) {
      auto o = occupants_.find(old->second.off);
      if (o != occupants_.end()) o->second.dead = true;
      table_.erase(old);
    }
  }
  Entry e;
  e.off = off;
  e.guest_start = start;
  e.guest_end = end;
  e.last_submission = c.submission;
  if (!is_index) {
    ConvertVertex(c.mem, start, end, reinterpret_cast<uint32_t*>(map_ + off));
  } else if (!ConvertIndex(c.mem, start, end, is16, endian, map_ + off)) {
    head_ = off;  // (the space is given back to the ring: nothing in it is valid)
    return finish(kBadRange);
  }
  e.hash = XXH3_64bits(c.mem->TranslatePhysical<const uint8_t*>(start), end - start);
  Flush(off, end - start);
  e.stale = in_progress_.hit;  // (a write came in while the bytes were read: used this once, made again next time)
  in_progress_.active = false;
  // Staggered: entries made together are not all hash-checked on the same frame.
  e.next_check = c.frame + kRefreshFrames / 2 + (uint64_t(KeyHash()(key)) >> 7) % kRefreshFrames;
  Occupant occ;
  occ.len = (VkDeviceSize(end - start) + 15) & ~VkDeviceSize(15);
  occ.key = key;
  occ.last_submission = c.submission;
  occupants_[off] = occ;
  Entry& placed = table_[key] = e;
  ++counters_.uploads;
  counters_.upload_bytes += end - start;
  const Why w = give(placed);
  return w;
}

VertexMirror::Why VertexMirror::Vertex(const Ctx& c, uint32_t address, uint32_t size, VkDeviceSize* offset) {
  Result e;
  const Why w = Place(c, Key{0, address + size, 0, 0}, address, address + size, false, 0, false, &e);
  if (w == kOk) *offset = e.off;
  else ++counters_.why[w];
  return w;
}

VertexMirror::Why VertexMirror::IndexData(const Ctx& c, uint32_t base, uint32_t count, bool is16, uint32_t endian, bool reset, Index* out) {
  const uint32_t isz = is16 ? 2u : 4u;
  const uint32_t start = base & ~(isz - 1);
  Result e;
  const Why w = Place(c, Key{1, start >> 16, uint32_t(is16 ? 1u : 0u) | (endian << 1), 0}, start, start + count * isz, is16, endian, reset, &e);
  if (w == kOk) {
    out->offset = e.off;
    out->min = e.min;
    out->max = e.max;
  } else {
    ++counters_.why[w];
  }
  return w;
}

uint32_t VertexMirror::entries() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return uint32_t(table_.size());
}

void VertexMirror::FrameEnd(uint64_t frame) {
  if (!window_first_frame_) window_first_frame_ = frame;
  if (frame - window_first_frame_ < 600) return;
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
    REXGPU_INFO("[mirror] frames {}-{}: mirrored {} draws; fallbacks {{{}}}; entries {}, hits {}, grown {}, uploads {} ({} KiB), invalidations cpu {} gpu {}, hash checks {} (changed {}), rechecked unchanged {}, verify mismatches {}, "
                "evictions {}, busy steps {}; refused {{{}}}",
                window_first_frame_, frame, n.mirrored, fb.empty() ? "none" : fb, entries(), n.hits, n.grown, n.uploads, n.upload_bytes >> 10, n.invalidations, n.gpu_invalidations, n.refreshes,
                n.refresh_mismatch, n.revalidated, n.verify_mismatch, n.evictions, n.skipped_busy, why.empty() ? "none" : why);
  }
  counters_ = Counters{};
  window_first_frame_ = frame;
}

}  // namespace rex::graphics::vulkan::native
