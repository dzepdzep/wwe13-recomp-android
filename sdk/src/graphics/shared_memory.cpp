/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <bit>  // wwe2k14 patch (Session 50, LEAN)
#include <atomic>
#include <chrono>
#include <cstring>
#include <utility>
#include <mutex>  // wwe2k14 patch (Session 19, PARHASH)
#include <thread>  // wwe2k14 patch (Session 19, PARHASH)
#include <functional>  // wwe2k14 patch (Session 19, PARHASH)
#include <condition_variable>  // wwe2k14 patch (Session 19, PARHASH)

#include <rex/assert.h>
#include <rex/bit.h>
#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/graphics/shared_memory.h>
#include <rex/hash.h>
#include <rex/logging.h>
#include <rex/thread.h>  // wwe2k14 patch (Session 19, PARHASH)
#include <rex/math.h>
#include <rex/memory.h>

// wwe2k14 patch (Session 6, CP2): RequestRanges runs for the index buffer of every draw and for every vertex
// buffer whose fetch address changed. In heavy WWE 2K14 entrance shots it was ~6.6% of the command processor
// thread (stacks_067): the std::vector allocated per call ~2.3%, the process-wide global critical region ~1.1%
// (partly waiting for guest threads) and EnsureHostGpuMemoryAllocated ~0.6%, against ~2.3% of real uploading.
REXCVAR_DEFINE_BOOL(gpu_request_ranges_fast, false, "GPU",
                    "SharedMemory::RequestRanges without a heap allocation per call, returning without "
                    "the global critical region when every requested page is already valid and its host "
                    "GPU memory is allocated. Read per call (toggle-safe).");
REXCVAR_DEFINE_BOOL(gpu_request_ranges_fast_verify, false, "GPU",
                    "With gpu_request_ranges_fast: when the lock-free check says a request needs nothing, "
                    "still run the original locked path (so the result is the original one) and count and "
                    "log disagreements ([s6][CP2]).");

// wwe2k14 patch (Session 87, step 4): MakeRangeValid watches the pages it makes valid (PhysicalHeap::
// EnableAccessCallbacks in the three physical views: a lock, a page loop each, a VirtualProtect for a page not yet
// watched) - 5.6% of Bryan's command thread on the Surface (stacks 055). A page whose watch fired since it was last
// made valid (system_page_flags_watch_fired_: set by every invalidation, which is also every unwatch - the physical
// heap's trigger path - and at the start, before any page was made valid) was unwatched; one whose bit is clear was
// watched by its last MakeRangeValid and has not been unwatched since, so watching it again changes nothing: the
// original loop passes over watched pages. About half of the calls of Bryan's entrance (the frame-end clear's
// re-validation of unchanged pages, resolves of the same destinations every frame) are over such pages only.
REXCVAR_DEFINE_BOOL(gpu_make_valid_skip_arm, false, "GPU",
                    "Session 87: SharedMemory::MakeRangeValid does not watch a range again when no page of it had its "
                    "watch fire since it was last made valid (already watched, as the original loop finds). A page the "
                    "guest made writable behind the watch's back (a protection change to read-write of a page that "
                    "was read-only when made valid) stays unwatched until its next trigger. Read per call "
                    "(toggle-safe)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_make_valid_skip_arm_verify, false, "GPU",
                    "Session 87 (diagnostic): with gpu_make_valid_skip_arm, the watch is still made and the pages it "
                    "newly watched are counted (and logged, [s87][UPSKIP]): the skip must have left none. The skip "
                    "then saves nothing")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 95, HOTPAGES): CPU audit rows 1 / 1b. Per match frame 219 pages were watched and 216 fired:
// every one is a guest access violation, an unwatch (VirtualProtect) and, at the next MakeRangeValid, a watch again
// (VirtualProtect), about 168 calls and 7+ faults a frame. A page that fired in each of the last N frame ends is
// "hot": MakeRangeValid no longer watches it and no longer marks it valid, so every request finds it not valid and
// uploads it (no hash: it always changes). No watch means no fault and no VirtualProtect. Safety case, in short:
//  * Every request of a hot page copies the guest bytes as they are at that request, which is what the original
//    does after any write; the original only skips the copy when no write came in since the last request, so the
//    cost of a hot page is extra copies of unchanged bytes, never stale ones. The repeat epoch moves at each request of a hot page, since no fault moves it.
//  * A page with a texture watch (WatchMemoryRange) is never hot: its watch needs the physical protection.
//    Registering a watch on a hot page makes it not hot and watches it at once.
//  * A page GPU-written (resolve) is watched as before: hot only applies to CPU uploads (written_by_gpu false).
//  * Probation: every gpu_hot_pages_probation frame ends all hot pages are watched again, so a page that stopped
//    changing is valid again within that many frames; pages that still fire become hot again after N frames.
//  * The set is process-wide (one SharedMemory), maintained under the global critical region.
REXCVAR_DEFINE_INT32(gpu_hot_pages_frames, 0, "GPU",
                     "Session 95 (HOTPAGES): N > 0 = a page whose guest write watch fired in each of the last N frame "
                     "ends stays unwatched and is uploaded at every request (no hash), saving the fault and the two "
                     "VirtualProtect calls per frame. 0 = off. Read at start (not hot-reload). Needs "
                     "clear_memory_page_state (the frame end)");
REXCVAR_DEFINE_INT32(gpu_hot_pages_probation, 120, "GPU",
                     "Session 95 (HOTPAGES): frame ends between full resets of the hot set (every hot page is watched "
                     "again and has to earn it again)");
// wwe2k14 patch (Session 95, r6tex): the movie planes are most of the command thread's texture time in an entrance: 13 loads a frame, about 500 pages not valid a frame, each load paying a re-arm of the
// memory watch (12 VirtualProtect a frame on the command thread, about 18 us each on the PC; the [tex] split: 950 of the 1200 ticks of "ranges" are neither the copy nor the validity scan) and 31 access
// violations a frame on the decoder. The plane flush (Changed) already says when a plane is done. So a plane's pages are declared hot at each Changed: no watch, never valid, with the difference that a
// declared page keeps its hash: what is requested is still hashed and only what differs is copied. The texture itself takes no watch from then on (Texture::SetEngineFlushed): only Changed outdates it.
// How it ends: the declaration lasts until the next probation reset (gpu_hot_pages_probation) and is renewed by the next Changed, so a movie that stopped loses it within that many frames; a texture
// watch registered on such a page by anything else (the memory reused) ends it at once, as for any hot page. The texture's own no-watch ends with its object: it is listed as watch-dropped, so a Destroy
// or a redefinition outdates it and its next load watches again (D3dTexProcessChanged). A live watch on a declared page keeps working: a watch is only placed on an armed page, a declaration disarms nothing,
// and it fires once on the next write. A plane is engine-dynamic, linear k_8, no mips (D3dTexProcessChanged checks all three; other engine-dynamic objects exist, 52 live in a match, and are left alone).
REXCVAR_DEFINE_BOOL(gpu_tex_planes_unwatched, false, "GPU",
                    "Session 95 (r6tex): a movie plane's pages go unwatched once the plane flush has announced it (no VirtualProtect, no access violation on the decoder); its texture is reloaded by the "
                    "announcement alone. Needs clear_memory_page_state and gpu_d3d_textures 2. Read at start. Logs [tex] planes unwatched");
REXCVAR_DEFINE_BOOL(gpu_tex_planes_nohash, false, "GPU",
                    "Session 95 (r7 lane D): with gpu_tex_planes_unwatched, a movie plane's pages are not hashed before the upload (the plane flush already announced the change); they go straight to the upload. "
                    "Can only upload more than needed. Read per request. Logs [tex] planes nohash");
namespace {
struct HotPages {
  bool enabled = false;
  bool any_hot = false;
  std::vector<uint64_t> fired_frame;  // pages whose watch fired since the last frame end
  std::vector<uint64_t> hot;
  std::vector<uint64_t> engine;       // (Session 95, r6tex) hot pages declared by DeclareEnginePages: they keep their hash
  uint64_t engine_declared = 0;
  uint64_t nohash_pages = 0;  // (Session 95, r7 lane D) plane pages that skipped the pre-upload hash
  std::vector<uint64_t> tex;          // pages that ever had a WatchMemoryRange since the last ClearCache
  std::vector<uint64_t> uploaded;     // hot pages left not valid by MakeRangeValid since the last frame end
  std::vector<uint8_t> streak;
  std::vector<uint32_t> active;       // pages with streak > 0 that are not hot
  uint32_t frames = 0, frames_since_probation = 0, frames_logged = 0;
  uint64_t became_hot = 0, resets = 0, hot_now = 0, make_valid_hot_pages = 0, make_valid_calls_hot = 0,
           tex_unhot = 0, fired_total = 0, fired_tex = 0, streak2 = 0, streak_lost = 0;
  void Init(uint32_t blocks) {
    enabled = REXCVAR_GET(gpu_hot_pages_frames) > 0 || REXCVAR_GET(gpu_tex_planes_unwatched);
    any_hot = false;
    if (!enabled) {
      return;
    }
    fired_frame.assign(blocks, 0);
    hot.assign(blocks, 0);
    engine.assign(blocks, 0);
    uploaded.assign(blocks, 0);
    tex.assign(blocks, 0);
    streak.assign(size_t(blocks) * 64, 0);
    active.clear();
    frames = frames_since_probation = frames_logged = 0;
    nohash_pages = 0;
    became_hot = resets = hot_now = make_valid_hot_pages = make_valid_calls_hot = tex_unhot = fired_total = fired_tex = streak2 = streak_lost = 0;
  }
  bool WasUploaded(uint32_t page) const { return (uploaded[page >> 6] >> (page & 63)) & 1; }
  bool IsHot(uint32_t page) const {
    return any_hot && (hot[page >> 6] >> (page & 63)) & 1;
  }
  bool IsEngine(uint32_t page) const { return (engine[page >> 6] >> (page & 63)) & 1; }
  void ResetHot() {
    std::fill(hot.begin(), hot.end(), uint64_t(0));
    std::fill(engine.begin(), engine.end(), uint64_t(0));
    std::fill(streak.begin(), streak.end(), uint8_t(0));
    active.clear();
    any_hot = false;
    hot_now = 0;
  }
};
HotPages g_hot;
}  // namespace

namespace rex::memory {
uint64_t PhysicalPagesWatchedTotal();  // xmemory.cpp (rexruntime exports all symbols)
}
// wwe2k14 patch (Session 7b, UPSKIP): with clear_memory_page_state (our default) the frame end clears the valid
// bit of every page the GPU hasn't written, so every page a frame uses is uploaded again the next time it's
// requested: ~1,000 mid-frame uploads per heavy frame, each ending the render pass with a whole-buffer barrier
// (session7b notes: ~83-85% of them re-upload unchanged bytes; with clear_memory_page_state=false, run 141, GPU
// time per heavy 1440p60 frame fell 18 -> 10.6 ms). With this on, the bytes last copied to each page of the GPU
// buffer are hashed (XXH3-64 of the exact snapshot uploaded), and the hash is dropped when anything else writes
// that page of the buffer (resolve, memory export: RangeWrittenByGpu / MakeRangeValid(written_by_gpu) and
// VulkanSharedMemory::Use with a write usage). A requested page that is not valid but still has its hash is made
// valid and watched exactly as UploadRanges does before its copy, then its guest bytes are hashed; if the hash is
// the same, the GPU buffer already holds these bytes and the copy is skipped, otherwise it is uploaded as before.
// clear_memory_page_state stays on: the decision compares bytes, it doesn't trust the write watch.
REXCVAR_DEFINE_BOOL(gpu_upload_skip_unchanged, false, "GPU",
                    "Skip shared memory uploads of pages whose guest bytes hash (XXH3-64) the same as the bytes "
                    "last uploaded to that page, when nothing else has written that page of the GPU buffer since; "
                    "the page is made valid and watched as by an upload. Per-frame counters: cpf_up / cpf_up2 "
                    "timeline events, [s7][UPSKIP] log. Read per request (toggle-safe)");
REXCVAR_DEFINE_BOOL(gpu_upload_skip_unchanged_stats, false, "GPU",
                    "Hash shared memory uploads as gpu_upload_skip_unchanged does without skipping anything, to "
                    "count (cpf_up c) uploaded pages whose bytes changed while only the frame-end clear "
                    "(clear_memory_page_state) had invalidated them - the pages clear_memory_page_state=false "
                    "would have left stale. Read per request (toggle-safe)");
// wwe2k14 patch (Session 50, LEAN): see SharedMemory::geometry_tracking().
REXCVAR_DEFINE_BOOL(gpu_native_trust_geometry_pages, false, "GPU/Native",
                    "Session 50: the frame-end clear (clear_memory_page_state) keeps valid the pages only vertex / "
                    "index fetches have requested - their write watches stay armed - instead of every frame hashing "
                    "and comparing them again. Off by default; takes effect at the next frame end");
REXCVAR_DEFINE_BOOL(gpu_native_trust_geometry_pages_check, false, "GPU/Native",
                    "Session 50 (diagnostic): classify pages as gpu_native_trust_geometry_pages does, keep the clear as "
                    "it is, and count the geometry-only pages that changed while only the clear had invalidated them "
                    "([s7][UPSKIP]: what the trust would have left stale). Needs the upload hashing "
                    "(gpu_upload_skip_unchanged or _stats)");
REXCVAR_DEFINE_INT32(gpu_native_trust_geometry_pages_refresh, 30, "GPU/Native",
                     "Session 50 (review): with gpu_native_trust_geometry_pages, the full frame-end clear every N "
                     "frames anyway (0 = never): a page a CPU write slipped past is stale at most N frames, and the "
                     "stale count still sees the kept pages");
// wwe2k14 patch (Session 19, PARHASH): the skip check's page hashing (XXH3 per 4 KB page, ~1,400 pages / ~0.4 ms of
// the command processor per frame at the 1440p preset, run 1435) on worker threads.
REXCVAR_DEFINE_INT32(gpu_upload_skip_parallel_hash, 0, "GPU",
                     "wwe2k14 (Session 19): with gpu_upload_skip_unchanged (not _verify), N > 0 = hash a request's "
                     "skip-check pages on N worker threads plus the command processor when it has at least "
                     "gpu_upload_skip_parallel_min_pages of them; 0 = off. Same result; logs [s19][PARHASH]");
REXCVAR_DEFINE_INT32(gpu_upload_skip_parallel_min_pages, 64, "GPU",
                     "wwe2k14 (Session 19): smallest request (pages to hash) that gpu_upload_skip_parallel_hash "
                     "spreads over its workers (smaller ones are hashed on the command processor)");
REXCVAR_DEFINE_BOOL(gpu_upload_skip_unchanged_verify, false, "GPU",
                    "With gpu_upload_skip_unchanged: upload the pages it would skip anyway (rendering as with it "
                    "off), and for up to 256 of them per frame (round robin) first copy the page out of the GPU "
                    "buffer at that point of the command stream and compare it with the hashed guest bytes once "
                    "the submission has completed; logs [s7][UPSKIP] verify mismatches (slow, for checking only)");

namespace {
// wwe2k14 patch (Session 7b, UPSKIP): gpu_upload_skip_unchanged_verify readbacks per frame.
constexpr uint32_t kUploadSkipVerifyPagesPerFrame = 256;
}  // namespace

// wwe2k14 patch (Session 7, PROTECT): defined in src/system/xmemory.cpp (rexruntime).
REXCVAR_DECLARE(bool, gpu_protect_batch);
// wwe2k14 patch (Session 49, RAM TIES): defined in src/system/xmemory.cpp (rexruntime).
REXCVAR_DECLARE(bool, gpu_ram_stats);

REXCVAR_DECLARE(uint32_t, gpu_d3d_wall);  // (Session 95: d3d_noring.cpp)

namespace rex::graphics {

SharedMemory::SharedMemory(memory::Memory& memory) : memory_(memory) {
  page_size_log2_ = rex::log2_ceil(uint32_t(rex::memory::page_size()));
}

SharedMemory::~SharedMemory() {
  ShutdownCommon();
}

void SharedMemory::InitializeCommon() {
  num_system_page_flags_ = ((kBufferSize >> page_size_log2_) + 63) / 64;
  system_page_flags_valid_.assign(num_system_page_flags_, 0);
  system_page_flags_valid_and_gpu_written_.assign(num_system_page_flags_, 0);
  // wwe2k14 patch (Session 7b, UPSKIP): never made valid yet, so not "invalidated only by the frame-end clear".
  system_page_flags_watch_fired_.assign(num_system_page_flags_, UINT64_MAX);
  g_hot.Init(num_system_page_flags_);  // wwe2k14 patch (Session 95, HOTPAGES)

  memory_invalidation_callback_handle_ =
      memory_.RegisterPhysicalMemoryInvalidationCallback(MemoryInvalidationCallbackThunk, this);
}

void SharedMemory::InitializeSparseHostGpuMemory(uint32_t granularity_log2) {
  assert_true(granularity_log2 <= kBufferSizeLog2);
  assert_true(host_gpu_memory_sparse_granularity_log2_ == UINT32_MAX);
  host_gpu_memory_sparse_granularity_log2_ = granularity_log2;
  host_gpu_memory_sparse_allocated_.resize(
      size_t(1) << (std::max(kBufferSizeLog2 - granularity_log2, uint32_t(6)) - 6));
}

void SharedMemory::ShutdownCommon() {
  FireWatches(0, (kBufferSize - 1) >> page_size_log2_, false);
  assert_true(global_watches_.empty());
  // No watches now, so no references to the pools accessible by guest threads -
  // safe not to enter the global critical region.
  watch_node_first_free_ = nullptr;
  watch_node_current_pool_allocated_ = 0;
  for (WatchNode* pool : watch_node_pools_) {
    delete[] pool;
  }
  watch_node_pools_.clear();
  watch_range_first_free_ = nullptr;
  watch_range_current_pool_allocated_ = 0;
  for (WatchRange* pool : watch_range_pools_) {
    delete[] pool;
  }
  watch_range_pools_.clear();

  if (memory_invalidation_callback_handle_ != nullptr) {
    memory_.UnregisterPhysicalMemoryInvalidationCallback(memory_invalidation_callback_handle_);
    memory_invalidation_callback_handle_ = nullptr;
  }

  if (host_gpu_memory_sparse_used_bytes_) {
    host_gpu_memory_sparse_used_bytes_ = 0;
    COUNT_profile_set("gpu/shared_memory/host_gpu_memory_sparse_used_mb", 0);
  }
  if (host_gpu_memory_sparse_allocations_) {
    host_gpu_memory_sparse_allocations_ = 0;
    COUNT_profile_set("gpu/shared_memory/host_gpu_memory_sparse_allocations", 0);
  }
  host_gpu_memory_sparse_allocated_.clear();
  host_gpu_memory_sparse_allocated_.shrink_to_fit();
  host_gpu_memory_sparse_granularity_log2_ = UINT32_MAX;

  system_page_flags_valid_.clear();
  system_page_flags_valid_.shrink_to_fit();
  system_page_flags_valid_and_gpu_written_.clear();
  system_page_flags_valid_and_gpu_written_.shrink_to_fit();
  num_system_page_flags_ = 0;
  g_hot = HotPages();  // wwe2k14 patch (Session 95, HOTPAGES)

  // wwe2k14 patch (Session 7b, UPSKIP): the GPU buffer goes away with the implementation's shutdown.
  system_page_flags_watch_fired_.clear();
  system_page_flags_watch_fired_.shrink_to_fit();
  upload_hashes_.clear();
  upload_hashes_.shrink_to_fit();
  upload_hash_valid_.clear();
  upload_hash_valid_.shrink_to_fit();
  upload_clear_only_.clear();
  upload_clear_only_.shrink_to_fit();
  upload_hash_tracking_ = false;
  upload_hash_tracking_previous_ = false;
  upload_skip_ = false;
  upload_skip_verify_ = false;
}

void SharedMemory::InvalidateAllPages() {
  auto global_lock = global_critical_region_.Acquire();
  BumpRepeatEpoch();  // wwe2k14 patch (Session 23, REPEAT)

  std::fill(system_page_flags_valid_.begin(), system_page_flags_valid_.end(), uint64_t(0));
  std::fill(system_page_flags_valid_and_gpu_written_.begin(),
            system_page_flags_valid_and_gpu_written_.end(), uint64_t(0));
  // wwe2k14 patch (Session 7b, UPSKIP): an invalidation clear_memory_page_state=false would make too. The GPU
  // buffer is not touched, so the upload hashes stay.
  std::fill(system_page_flags_watch_fired_.begin(), system_page_flags_watch_fired_.end(), UINT64_MAX);
}

void SharedMemory::SetSystemPageBlocksValidWithGpuDataWritten(bool frame_end) {
  auto global_lock = global_critical_region_.Acquire();
  BumpRepeatEpoch();  // wwe2k14 patch (Session 23, REPEAT)

  // Pages that are valid only because the CPU uploaded them lose their valid
  // bit here, so the next frame re-reads them from guest memory.
  // wwe2k14 patch (Session 50, LEAN: gpu_native_trust_geometry_pages): except the valid geometry-only pages (still
  // watched: a CPU write invalidates them as before). Review: only at the frame end; never after lost uploads (their
  // copies may not be in the GPU buffer: the frame-end clear is their repair); and a full clear every
  // gpu_native_trust_geometry_pages_refresh frames, so a page a write slipped past (RAM_TIES.md section 7) is stale at
  // most that long and the stale count ([s7][UPSKIP]) still sees the kept pages.
  const uint64_t lost = GetLostUploadCount();
  const bool lost_changed = lost != geometry_lost_seen_;
  if (frame_end) geometry_lost_seen_ = lost;
  const uint32_t refresh = uint32_t(std::max(0, REXCVAR_GET(gpu_native_trust_geometry_pages_refresh)));
  const bool refresh_now = frame_end && refresh && ++geometry_trust_frames_ >= refresh;
  if (refresh_now) geometry_trust_frames_ = 0;
  if (frame_end && geometry_trust_ && !lost_changed && !refresh_now &&
      geometry_pages_.size() == system_page_flags_valid_.size()) {
    uint64_t kept = 0;
    for (size_t i = 0; i < system_page_flags_valid_.size(); ++i) {
      const uint64_t gpu = system_page_flags_valid_and_gpu_written_[i];
      const uint64_t keep = system_page_flags_valid_[i] & geometry_pages_[i] & ~nongeometry_pages_[i] & ~gpu;
      system_page_flags_valid_[i] = gpu | keep;
      kept += uint64_t(std::popcount(keep));
    }
    geometry_kept_pages_ = kept;
  } else {
    system_page_flags_valid_ = system_page_flags_valid_and_gpu_written_;
    geometry_kept_pages_ = 0;
  }
  upload_skip_window_.geometry_kept_pages += geometry_kept_pages_;
  if (!frame_end) return;
  if (g_hot.enabled) {
    UpdateHotPagesAtFrameEnd();
  }
  // The settings for the next frame (the classification starts over whenever it was off).
  const bool trust = REXCVAR_GET(gpu_native_trust_geometry_pages);
  const bool tracking = trust || REXCVAR_GET(gpu_native_trust_geometry_pages_check);
  if (tracking && !geometry_tracking_) {
    geometry_pages_.assign(num_system_page_flags_, 0);
    nongeometry_pages_.assign(num_system_page_flags_, 0);
  }
  geometry_tracking_ = tracking;
  geometry_trust_ = trust;
}

// wwe2k14 patch (Session 95, HOTPAGES): global critical region held (the caller took it).
void SharedMemory::DeclareEnginePages(uint32_t start, uint32_t length) {
  HotPages& h = g_hot;
  if (!h.enabled || !length || start >= kBufferSize) return;
  length = std::min(length, kBufferSize - start);
  const uint32_t page_last = (start + length - 1) >> page_size_log2_;
  for (uint32_t page = start >> page_size_log2_; page <= page_last; ++page) {
    const uint64_t bit = uint64_t(1) << (page & 63);
    if (!(h.hot[page >> 6] & bit)) {
      h.hot[page >> 6] |= bit;
      ++h.hot_now;
      ++h.engine_declared;
    }
    h.engine[page >> 6] |= bit;
  }
  if (!h.any_hot) {
    h.any_hot = true;
    static bool logged = false;
    if (!logged) {
      logged = true;
      REXGPU_INFO("[tex] planes unwatched: the first plane's pages go unwatched (0x{:08X}, {} bytes)", start, length);
    }
  }
}

void SharedMemory::UpdateHotPagesAtFrameEnd() {
  HotPages& h = g_hot;
  // wwe2k14 patch (Session 95, r6tex): only the declared pages: nothing learned, the probation reset as ever.
  if (REXCVAR_GET(gpu_hot_pages_frames) <= 0) {
    std::fill(h.uploaded.begin(), h.uploaded.end(), uint64_t(0));
    std::fill(h.fired_frame.begin(), h.fired_frame.end(), uint64_t(0));
    if (++h.frames - h.frames_logged >= 600) {
      REXGPU_INFO("[tex] planes unwatched: frames {}-{}: {} pages hot now, {} declared, {} requests left {} pages unwatched and not valid, {} un-hotted by a texture watch", h.frames_logged, h.frames, h.hot_now,
                  h.engine_declared, h.make_valid_calls_hot, h.make_valid_hot_pages, h.tex_unhot);
      if (REXCVAR_GET(gpu_tex_planes_nohash)) {
        REXGPU_INFO("[tex] planes nohash: frames {}-{}: {} plane pages skipped the hash", h.frames_logged, h.frames, h.nohash_pages);
      }
      h.nohash_pages = 0;
      h.frames_logged = h.frames;
      h.engine_declared = h.make_valid_calls_hot = h.make_valid_hot_pages = h.tex_unhot = 0;
    }
    const uint32_t probation = uint32_t(std::max(0, REXCVAR_GET(gpu_hot_pages_probation)));
    if (probation && ++h.frames_since_probation >= probation) {
      h.frames_since_probation = 0;
      h.ResetHot();
    }
    return;
  }
  const uint32_t need = uint32_t(std::max(1, REXCVAR_GET(gpu_hot_pages_frames)));
  ++h.frames;
  std::fill(h.uploaded.begin(), h.uploaded.end(), uint64_t(0));
  // Pages that were counting: fired again this frame -> one more, else back to zero.
  size_t keep = 0;
  for (size_t k = 0; k < h.active.size(); ++k) {
    const uint32_t page = h.active[k];
    uint64_t& word = h.fired_frame[page >> 6];
    const uint64_t bit = uint64_t(1) << (page & 63);
    if (word & bit) {
      word &= ~bit;
      if (h.streak[page] < 255) ++h.streak[page];
      if (h.streak[page] == 2) ++h.streak2;
      if (h.streak[page] >= need && !(h.tex[page >> 6] & bit)) {
        h.hot[page >> 6] |= bit;
        h.streak[page] = 0;
        ++h.became_hot;
        ++h.hot_now;
        h.any_hot = true;
        continue;
      }
      h.active[keep++] = page;
    } else {
      h.streak[page] = 0;
      ++h.streak_lost;
    }
  }
  h.active.resize(keep);
  // Pages that fired this frame and were not counting (or are already hot, which only means a watch that was armed
  // when the page became hot fired once more).
  for (size_t i = 0; i < h.fired_frame.size(); ++i) {
    uint64_t bits = h.fired_frame[i];
    if (!bits) continue;
    h.fired_frame[i] = 0;
    bits &= ~h.hot[i];
    while (bits) {
      const uint32_t page = uint32_t(i * 64 + std::countr_zero(bits));
      bits &= bits - 1;
      ++h.fired_total;
      if (h.tex[i] >> (page & 63) & 1) {  // a texture watch: never hot
        ++h.fired_tex;
        continue;
      }
      if (need <= 1) {
        h.hot[i] |= uint64_t(1) << (page & 63);
        ++h.became_hot;
        ++h.hot_now;
        h.any_hot = true;
      } else {
        h.streak[page] = 1;
        h.active.push_back(page);
      }
    }
  }
  if (h.frames - h.frames_logged >= 600) {  // logged before the probation reset below, so "hot pages now" is real
    REXGPU_INFO("[s95][HOTPAGES] frames {}-{}: hot pages now {} (became hot {}, probation resets {}, watches fired {}), "
                "MakeRangeValid calls with hot pages {} left {} page(s) unwatched and not valid, {} hot pages "
                "un-hotted by a texture watch; pages that fired {} (of them with a texture watch {}), reached 2 frames {}, "
                "streaks that ended {}",
                h.frames_logged, h.frames, h.hot_now, h.became_hot, h.resets, h.fired_total,
                h.make_valid_calls_hot, h.make_valid_hot_pages, h.tex_unhot, h.fired_total, h.fired_tex, h.streak2,
                h.streak_lost);
    h.frames_logged = h.frames;
    // The counts above are per window (since the last line).
    h.became_hot = h.resets = h.make_valid_hot_pages = h.make_valid_calls_hot = h.tex_unhot = h.fired_total =
        h.fired_tex = h.streak2 = h.streak_lost = 0;
  }
  const uint32_t probation = uint32_t(std::max(0, REXCVAR_GET(gpu_hot_pages_probation)));
  if (probation && ++h.frames_since_probation >= probation) {
    h.frames_since_probation = 0;
    if (h.hot_now) ++h.resets;
    h.ResetHot();
  }
}

void SharedMemory::NoteGeometryRanges(const std::pair<uint32_t, uint32_t>* ranges, size_t count) {
  // wwe2k14 patch (Session 50, LEAN): command processor thread (the frame-end clear reads these on the same thread).
  std::vector<uint64_t>& bits =
      (ram_reader_ == RamReader::kVertex || ram_reader_ == RamReader::kIndex) ? geometry_pages_ : nongeometry_pages_;
  for (size_t r = 0; r < count; ++r) {
    const uint32_t start = ranges[r].first, length = ranges[r].second;
    if (!length || start >= kBufferSize) continue;
    const uint32_t page_first = start >> page_size_log2_;
    const uint32_t page_last = (start + std::min(length, kBufferSize - start) - 1) >> page_size_log2_;
    for (uint32_t i = page_first >> 6; i <= (page_last >> 6) && i < bits.size(); ++i) {
      uint64_t mask = UINT64_MAX;
      if (i == (page_first >> 6)) mask &= ~((uint64_t(1) << (page_first & 63)) - 1);
      if (i == (page_last >> 6) && (page_last & 63) != 63) mask &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
      bits[i] |= mask;
    }
  }
}

void SharedMemory::ClearCache() {
  // Keeping GPU-written data, so "invalidated by GPU".
  FireWatches(0, (kBufferSize - 1) >> page_size_log2_, true);
  // No watches now, so no references to the pools accessible by guest threads -
  // safe not to enter the global critical region.
  watch_node_first_free_ = nullptr;
  watch_node_current_pool_allocated_ = 0;
  for (WatchNode* pool : watch_node_pools_) {
    delete[] pool;
  }
  watch_node_pools_.clear();
  watch_range_first_free_ = nullptr;
  watch_range_current_pool_allocated_ = 0;
  for (WatchRange* pool : watch_range_pools_) {
    delete[] pool;
  }
  watch_range_pools_.clear();
  {
    // wwe2k14 patch (Session 7b, UPSKIP): this clear happens with clear_memory_page_state=false too, unlike the
    // frame-end one. The GPU buffer is not touched, so the upload hashes stay.
    auto global_lock = global_critical_region_.Acquire();
    std::fill(system_page_flags_watch_fired_.begin(), system_page_flags_watch_fired_.end(), UINT64_MAX);
    if (g_hot.enabled) {  // wwe2k14 patch (Session 95, HOTPAGES): no watches left, so none to keep
      g_hot.ResetHot();
      std::fill(g_hot.tex.begin(), g_hot.tex.end(), uint64_t(0));
    }
  }
  SetSystemPageBlocksValidWithGpuDataWritten();
}

SharedMemory::GlobalWatchHandle SharedMemory::RegisterGlobalWatch(GlobalWatchCallback callback,
                                                                  void* callback_context) {
  GlobalWatch* watch = new GlobalWatch;
  watch->callback = callback;
  watch->callback_context = callback_context;

  auto global_lock = global_critical_region_.Acquire();
  global_watches_.push_back(watch);

  return reinterpret_cast<GlobalWatchHandle>(watch);
}

void SharedMemory::UnregisterGlobalWatch(GlobalWatchHandle handle) {
  auto watch = reinterpret_cast<GlobalWatch*>(handle);

  {
    auto global_lock = global_critical_region_.Acquire();
    auto it = std::find(global_watches_.begin(), global_watches_.end(), watch);
    assert_false(it == global_watches_.end());
    if (it != global_watches_.end()) {
      global_watches_.erase(it);
    }
  }

  delete watch;
}

SharedMemory::WatchHandle SharedMemory::WatchMemoryRange(uint32_t start, uint32_t length,
                                                         WatchCallback callback,
                                                         void* callback_context,
                                                         void* callback_data,
                                                         uint64_t callback_argument) {
  if (length == 0 || start >= kBufferSize) {
    return nullptr;
  }
  length = std::min(length, kBufferSize - start);
  uint32_t watch_page_first = start >> page_size_log2_;
  uint32_t watch_page_last = (start + length - 1) >> page_size_log2_;
  uint32_t bucket_first = watch_page_first << page_size_log2_ >> kWatchBucketSizeLog2;
  uint32_t bucket_last = watch_page_last << page_size_log2_ >> kWatchBucketSizeLog2;

  auto global_lock = global_critical_region_.Acquire();

  // wwe2k14 patch (Session 95, HOTPAGES): a texture watch needs the physical watch of its pages.
  if (g_hot.enabled) {
    bool was_hot = false;
    for (uint32_t i = watch_page_first >> 6; i <= (watch_page_last >> 6); ++i) {
      uint64_t mask = UINT64_MAX;
      if (i == (watch_page_first >> 6)) mask &= ~((uint64_t(1) << (watch_page_first & 63)) - 1);
      if (i == (watch_page_last >> 6) && (watch_page_last & 63) != 63) {
        mask &= (uint64_t(1) << ((watch_page_last & 63) + 1)) - 1;
      }
      g_hot.tex[i] |= mask;
      if (g_hot.hot[i] & mask) {
        was_hot = true;
        g_hot.tex_unhot += uint64_t(std::popcount(g_hot.hot[i] & mask));
        g_hot.hot_now -= uint64_t(std::popcount(g_hot.hot[i] & mask));
        g_hot.hot[i] &= ~mask;
      }
    }
    if (was_hot && memory_invalidation_callback_handle_) {
      memory().EnablePhysicalMemoryAccessCallbacks(watch_page_first << page_size_log2_,
                                                   (watch_page_last - watch_page_first + 1) << page_size_log2_, true,
                                                   false);
    }
  }

  // Allocate the range.
  WatchRange* range = watch_range_first_free_;
  if (range != nullptr) {
    watch_range_first_free_ = range->next_free;
  } else {
    if (watch_range_pools_.empty() || watch_range_current_pool_allocated_ >= kWatchRangePoolSize) {
      watch_range_pools_.push_back(new WatchRange[kWatchRangePoolSize]);
      watch_range_current_pool_allocated_ = 0;
    }
    range = &(watch_range_pools_.back()[watch_range_current_pool_allocated_++]);
  }
  range->callback = callback;
  range->callback_context = callback_context;
  range->callback_data = callback_data;
  range->callback_argument = callback_argument;
  range->page_first = watch_page_first;
  range->page_last = watch_page_last;

  // Allocate and link the nodes.
  WatchNode* node_previous = nullptr;
  for (uint32_t i = bucket_first; i <= bucket_last; ++i) {
    WatchNode* node = watch_node_first_free_;
    if (node != nullptr) {
      watch_node_first_free_ = node->next_free;
    } else {
      if (watch_node_pools_.empty() || watch_node_current_pool_allocated_ >= kWatchNodePoolSize) {
        watch_node_pools_.push_back(new WatchNode[kWatchNodePoolSize]);
        watch_node_current_pool_allocated_ = 0;
      }
      node = &(watch_node_pools_.back()[watch_node_current_pool_allocated_++]);
    }
    node->range = range;
    node->range_node_next = nullptr;
    if (node_previous != nullptr) {
      node_previous->range_node_next = node;
    } else {
      range->node_first = node;
    }
    node_previous = node;
    node->bucket_node_previous = nullptr;
    node->bucket_node_next = watch_buckets_[i];
    if (watch_buckets_[i] != nullptr) {
      watch_buckets_[i]->bucket_node_previous = node;
    }
    watch_buckets_[i] = node;
  }

  return reinterpret_cast<WatchHandle>(range);
}

void SharedMemory::UnwatchMemoryRange(WatchHandle handle) {
  auto global_lock = global_critical_region_.Acquire();
  UnlinkWatchRange(reinterpret_cast<WatchRange*>(handle));
}

void SharedMemory::FireWatches(uint32_t page_first, uint32_t page_last, bool invalidated_by_gpu) {
  uint32_t address_first = page_first << page_size_log2_;
  uint32_t address_last = (page_last << page_size_log2_) + ((1 << page_size_log2_) - 1);
  uint32_t bucket_first = address_first >> kWatchBucketSizeLog2;
  uint32_t bucket_last = address_last >> kWatchBucketSizeLog2;

  auto global_lock = global_critical_region_.Acquire();
  BumpRepeatEpoch();  // wwe2k14 patch (Session 23, REPEAT)

  // Fire global watches.
  for (const auto global_watch : global_watches_) {
    global_watch->callback(global_lock, global_watch->callback_context, address_first, address_last,
                           invalidated_by_gpu);
  }

  // Fire per-range watches.
  for (uint32_t i = bucket_first; i <= bucket_last; ++i) {
    WatchNode* node = watch_buckets_[i];
    while (node != nullptr) {
      WatchRange* range = node->range;
      // Store the next node now since when the callback is triggered, the links
      // will be broken.
      node = node->bucket_node_next;
      if (page_first <= range->page_last && page_last >= range->page_first) {
        range->callback(global_lock, range->callback_context, range->callback_data,
                        range->callback_argument, invalidated_by_gpu);
        UnlinkWatchRange(range);
      }
    }
  }
}

void SharedMemory::RangeWrittenByGpu(uint32_t start, uint32_t length) {
  if (length == 0 || start >= kBufferSize) {
    return;
  }
  length = std::min(length, kBufferSize - start);
  uint32_t end = start + length - 1;
  uint32_t page_first = start >> page_size_log2_;
  uint32_t page_last = end >> page_size_log2_;

  // wwe2k14 patch (Session 49, RAM TIES): a memory export overwrote resolve output (a resolve notes its pages again
  // after this, NoteResolveWritten).
  if (ram_reader_ == RamReader::kMemexport && !resolve_written_.empty()) {
    auto global_lock = global_critical_region_.Acquire();
    RamDropPages(start, length, ram_stats_.gpu_overwrite_pages);
  }

  // Trigger modification callbacks so, for instance, resolved data is loaded to
  // the texture.
  FireWatches(page_first, page_last, true);

  // Mark the range as valid (so pages are not reuploaded until modified by the
  // CPU) and watch it so the CPU can reuse it and this will be caught.
  MakeRangeValid(start, length, true);
}

void SharedMemory::MarkRangeGpuWrittenQuiet(uint32_t start, uint32_t length) {
  MakeRangeValid(start, length, true);
}

bool SharedMemory::AllocateSparseHostGpuMemoryRange(uint32_t offset_allocations,
                                                    uint32_t length_allocations) {
  assert_always(
      "Sparse host GPU memory allocation has been initialized, but the "
      "implementation doesn't provide AllocateSparseHostGpuMemoryRange");
  return false;
}

void SharedMemory::MakeRangeValid(uint32_t start, uint32_t length, bool written_by_gpu) {
  if (length == 0 || start >= kBufferSize) {
    return;
  }
  length = std::min(length, kBufferSize - start);
  uint32_t last = start + length - 1;
  uint32_t valid_page_first = start >> page_size_log2_;
  uint32_t valid_page_last = last >> page_size_log2_;
  uint32_t valid_block_first = valid_page_first >> 6;
  uint32_t valid_block_last = valid_page_last >> 6;

  // wwe2k14 patch (Session 7b, UPSKIP): GPU-written data (resolve, memory export) replaced what was last
  // uploaded to these pages of the GPU buffer.
  if (written_by_gpu) {
    InvalidateUploadHashes(start, length);
  }

  std::vector<std::pair<uint32_t, uint32_t>> watch_runs;  // wwe2k14 patch (Session 95, HOTPAGES)
  bool watch_runs_filtered = false;  // true: watch only watch_runs (which may be empty when every page is hot)
  auto do_watch = [&]() {
    if (!memory_invalidation_callback_handle_) {
      return;
    }
    if (!watch_runs_filtered) {
      memory().EnablePhysicalMemoryAccessCallbacks(
          valid_page_first << page_size_log2_,
          (valid_page_last - valid_page_first + 1) << page_size_log2_, true, false);
      return;
    }
    for (const std::pair<uint32_t, uint32_t>& run : watch_runs) {
      memory().EnablePhysicalMemoryAccessCallbacks(run.first << page_size_log2_, run.second << page_size_log2_, true,
                                                   false);
    }
  };
  {
    auto global_lock = global_critical_region_.Acquire();

    bool any_watch_fired = false;  // wwe2k14 patch (Session 87, gpu_make_valid_skip_arm)
    // wwe2k14 patch (Session 95, HOTPAGES): hot pages of this range (CPU uploads only) are neither marked valid nor
    // watched; watch_runs lists the pages to watch when there are any.
    const bool use_hot = g_hot.any_hot && !written_by_gpu;
    bool range_has_hot = false;
    for (uint32_t i = valid_block_first; i <= valid_block_last; ++i) {
      uint64_t valid_bits = UINT64_MAX;
      if (i == valid_block_first) {
        valid_bits &= ~((uint64_t(1) << (valid_page_first & 63)) - 1);
      }
      if (i == valid_block_last && (valid_page_last & 63) != 63) {
        valid_bits &= (uint64_t(1) << ((valid_page_last & 63) + 1)) - 1;
      }
      any_watch_fired |= (system_page_flags_watch_fired_[i] & valid_bits) != 0;
      uint64_t valid_set_bits = valid_bits;
      if (use_hot) {
        const uint64_t hot_bits = g_hot.hot[i] & valid_bits;
        if (hot_bits) {
          range_has_hot = true;
          valid_set_bits &= ~hot_bits;
          g_hot.uploaded[i] |= hot_bits;
          g_hot.make_valid_hot_pages += uint64_t(std::popcount(hot_bits));
        }
      }
      system_page_flags_valid_[i] |= valid_set_bits;
      // wwe2k14 patch (Session 49, RAM TIES): an upload replaced resolve output.
      if (!written_by_gpu && !resolve_written_.empty() && (resolve_written_[i] & valid_bits)) {
        ram_stats_.upload_pages += rex::bit_count(resolve_written_[i] & valid_bits);
        resolve_written_[i] &= ~valid_bits;
      }
      uint64_t& gpu_written = system_page_flags_valid_and_gpu_written_[i];
      gpu_written = written_by_gpu ? (gpu_written | valid_bits) : (gpu_written & ~valid_bits);
      // wwe2k14 patch (Session 7b, UPSKIP): valid (and watched) again from here.
      system_page_flags_watch_fired_[i] &= ~valid_set_bits;
    }
    if (range_has_hot) {
      ++g_hot.make_valid_calls_hot;
      watch_runs_filtered = true;
      // Hot pages get no fault, so nothing else moves the epoch that the repeat-draw and vertex-stamp fast paths
      // trust when the guest writes them: move it here, so a later draw asks for the range again.
      BumpRepeatEpoch();
      watch_runs.clear();
      uint32_t run_start = UINT32_MAX;
      for (uint32_t page = valid_page_first; page <= valid_page_last + 1; ++page) {
        const bool hot = page <= valid_page_last && g_hot.IsHot(page);
        if (!hot && page <= valid_page_last) {
          if (run_start == UINT32_MAX) run_start = page;
        } else if (run_start != UINT32_MAX) {
          watch_runs.emplace_back(run_start, page - run_start);
          run_start = UINT32_MAX;
        }
      }
    }

    // wwe2k14 patch (Session 87, step 4, gpu_make_valid_skip_arm): see the cvar. Counted either way.
    ++upload_skip_window_.make_valid_calls;
    if (!any_watch_fired) {
      ++upload_skip_window_.make_valid_unfired;
      if (REXCVAR_GET(gpu_make_valid_skip_arm)) {
        if (!REXCVAR_GET(gpu_make_valid_skip_arm_verify)) {
          return;
        }
        // Verify: watched as before, counting the pages that call newly watched - the skip must have left none.
        const uint64_t watched_before = rex::memory::PhysicalPagesWatchedTotal();
        do_watch();
        const uint64_t newly_watched = rex::memory::PhysicalPagesWatchedTotal() - watched_before;
        if (newly_watched) {
          upload_skip_window_.make_valid_skip_wrong += newly_watched;
          static uint32_t logged = 0;
          if (logged < 8) {
            ++logged;
            REXGPU_WARN("[s87][UPSKIP] gpu_make_valid_skip_arm would have left {} page(s) unwatched (pages {:X}-{:X}, "
                        "written by gpu {})", newly_watched, valid_page_first, valid_page_last, written_by_gpu);
          }
        }
        return;
      }
    }

    // wwe2k14 patch (Session 7, PROTECT): with gpu_protect_batch, watch the pages without leaving the global
    // critical region (Memory::EnablePhysicalMemoryAccessCallbacks and each physical view enter it again,
    // only a count for the owning thread): one acquisition instead of four. The same calls in the same
    // order. The only difference: a guest write access violation on these pages (possible where they are
    // still watched from an earlier upload) can no longer be handled between marking them valid and
    // watching them, only before both (this call then watches them again: the retried write lands before
    // the new watch, so before the caller's copy, or traps on it) or after both (it clears the valid bits,
    // so they are uploaded again on their next use) - both outcomes the original has too.
    if (REXCVAR_GET(gpu_protect_batch)) {
      do_watch();
      return;
    }
  }

  do_watch();
}

void SharedMemory::UnlinkWatchRange(WatchRange* range) {
  uint32_t bucket = range->page_first << page_size_log2_ >> kWatchBucketSizeLog2;
  WatchNode* node = range->node_first;
  while (node != nullptr) {
    WatchNode* node_next = node->range_node_next;
    if (node->bucket_node_previous != nullptr) {
      node->bucket_node_previous->bucket_node_next = node->bucket_node_next;
    } else {
      watch_buckets_[bucket] = node->bucket_node_next;
    }
    if (node->bucket_node_next != nullptr) {
      node->bucket_node_next->bucket_node_previous = node->bucket_node_previous;
    }
    node->next_free = watch_node_first_free_;
    watch_node_first_free_ = node;
    node = node_next;
    ++bucket;
  }
  range->next_free = watch_range_first_free_;
  watch_range_first_free_ = range;
}

// wwe2k14 patch (Session 49, RAM TIES): resolve-written pages in a range. Global critical region held.
uint32_t SharedMemory::RamOverlapPages(uint32_t start, uint32_t length) const {
  if (!length || start >= kBufferSize) return 0;
  length = std::min(length, kBufferSize - start);
  const uint32_t page_first = start >> page_size_log2_, page_last = (start + length - 1) >> page_size_log2_;
  uint32_t pages = 0;
  for (uint32_t b = page_first >> 6; b <= page_last >> 6; ++b) {
    uint64_t bits = resolve_written_[b];
    if (!bits) continue;
    if (b == page_first >> 6) bits &= ~((uint64_t(1) << (page_first & 63)) - 1);
    if (b == page_last >> 6 && (page_last & 63) != 63) bits &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
    pages += rex::bit_count(bits);
  }
  return pages;
}

void SharedMemory::RamDropPages(uint32_t start, uint32_t length, uint64_t& counter) {
  if (!length || start >= kBufferSize || resolve_written_.empty()) return;
  length = std::min(length, kBufferSize - start);
  const uint32_t page_first = start >> page_size_log2_, page_last = (start + length - 1) >> page_size_log2_;
  for (uint32_t b = page_first >> 6; b <= page_last >> 6; ++b) {
    uint64_t bits = UINT64_MAX;
    if (b == page_first >> 6) bits &= ~((uint64_t(1) << (page_first & 63)) - 1);
    if (b == page_last >> 6 && (page_last & 63) != 63) bits &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
    counter += rex::bit_count(resolve_written_[b] & bits);
    resolve_written_[b] &= ~bits;
  }
}

void SharedMemory::NoteResolveWritten(uint32_t start, uint32_t length) {
  if (!REXCVAR_GET(gpu_ram_stats) || !length || start >= kBufferSize) return;
  length = std::min(length, kBufferSize - start);
  auto global_lock = global_critical_region_.Acquire();
  if (resolve_written_.empty()) resolve_written_.assign(num_system_page_flags_, 0);
  const uint32_t page_first = start >> page_size_log2_, page_last = (start + length - 1) >> page_size_log2_;
  for (uint32_t b = page_first >> 6; b <= page_last >> 6; ++b) {
    uint64_t bits = UINT64_MAX;
    if (b == page_first >> 6) bits &= ~((uint64_t(1) << (page_first & 63)) - 1);
    if (b == page_last >> 6 && (page_last & 63) != 63) bits &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
    resolve_written_[b] |= bits;
  }
  ++ram_stats_.noted_calls;
  ram_stats_.noted_pages += page_last - page_first + 1;
}

void SharedMemory::ReportRamStats(uint64_t frame) {
  if (resolve_written_.empty()) return;
  auto stats_lock = global_critical_region_.Acquire();  // (review: the invalidation callback counts under it)
  RamStats& s = ram_stats_;
  if (s.window_first_frame == UINT64_MAX) {
    s.window_first_frame = frame;
    return;
  }
  if (frame - s.window_first_frame < 300) return;
  static const char* const kNames[] = {"other", "vertex", "index", "texture", "memexport", "resolve dest"};
  const double frames = double(frame - s.window_first_frame);
  std::string readers;
  for (size_t i = 0; i < size_t(RamReader::kCount); ++i) {
    if (!s.requests[i]) continue;
    readers += fmt::format("{}{} {:.1f} of {:.1f} ({:.1f} pages)", readers.empty() ? "" : ", ", kNames[i],
                           s.reads[i] / frames, s.requests[i] / frames, s.read_pages[i] / frames);
  }
  uint64_t pages_now = 0;
  for (uint64_t b : resolve_written_) pages_now += rex::bit_count(b);
  REXGPU_INFO("[s49][ram] frames {}-{}: resolves noted per frame {:.1f} ({:.1f} pages); requests overlapping "
              "resolve-written pages per frame, of all requests by caller: {}; resolve pages dropped per frame by CPU "
              "writes {:.2f} in {:.2f} faults, by uploads {:.2f}, by memexport {:.2f}; resolve-written pages now {}",
              s.window_first_frame, frame, s.noted_calls / frames, s.noted_pages / frames,
              readers.empty() ? "none" : readers, s.cpu_pages / frames, s.cpu_invalidations / frames,
              s.upload_pages / frames, s.gpu_overwrite_pages / frames, pages_now);
  RamStats next;
  next.window_first_frame = frame;
  next.logged = s.logged;
  s = next;
}

void SharedMemory::RamCountRequest(const std::pair<uint32_t, uint32_t>* ranges, size_t count) {
  if (!resolve_written_.empty() && ranges) {
    const size_t reader = size_t(ram_reader_);
    ++ram_stats_.requests[reader];
    if (ram_reader_ != RamReader::kResolveDest) {
      auto global_lock = global_critical_region_.Acquire();
      uint32_t pages = 0;
      for (size_t i = 0; i < count; ++i) pages += RamOverlapPages(ranges[i].first, ranges[i].second);
      if (pages) {
        ++ram_stats_.reads[reader];
        ram_stats_.read_pages[reader] += pages;
        if (ram_stats_.logged < 32) {
          ++ram_stats_.logged;
          REXGPU_INFO("[s49][ram] read of resolve output {}: reader {} range {:08X}+{:X} ({} of {} ranges), {} "
                      "resolve-written pages", ram_stats_.logged, reader, ranges[0].first, ranges[0].second,
                      count ? 1 : 0, count, pages);
        }
      }
    }
  }
}

// wwe2k14 patch (Session 49, NATIVE TRUTH skip write-back): see the header.
void SharedMemory::SetRangeSkipped(uint32_t start, uint32_t length, bool skipped) {
  if (!length || start >= kBufferSize) return;
  if (skipped_.empty()) {
    if (!skipped) return;
    skipped_.assign(num_system_page_flags_, 0);
  }
  length = std::min(length, kBufferSize - start);
  const uint32_t page_first = start >> page_size_log2_, page_last = (start + length - 1) >> page_size_log2_;
  for (uint32_t b = page_first >> 6; b <= page_last >> 6; ++b) {
    uint64_t bits = UINT64_MAX;
    if (b == page_first >> 6) bits &= ~((uint64_t(1) << (page_first & 63)) - 1);
    if (b == page_last >> 6 && (page_last & 63) != 63) bits &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
    const uint64_t before = skipped_[b];
    skipped_[b] = skipped ? (before | bits) : (before & ~bits);
    skipped_pages_ += rex::bit_count(skipped_[b]);
    skipped_pages_ -= rex::bit_count(before);
  }
}

void SharedMemory::CheckSkippedRanges(std::pair<uint32_t, uint32_t> range) {
  if (materializing_ || !materialize_callback_ || !range.second || range.first >= kBufferSize) return;
  // (The resolve's own destination is decided by TruthResolveOverlaps, after this request - unless the request is
  // about to upload skipped pages a CPU write invalidated: their records go first, then the upload, as without them.)
  const bool dest = ram_reader_ == RamReader::kResolveDest;
  const uint32_t length = std::min(range.second, kBufferSize - range.first);
  const uint32_t page_first = range.first >> page_size_log2_,
                 page_last = (range.first + length - 1) >> page_size_log2_;
  for (uint32_t b = page_first >> 6; b <= page_last >> 6; ++b) {
    uint64_t bits = skipped_[b];
    if (!bits) continue;
    if (b == page_first >> 6) bits &= ~((uint64_t(1) << (page_first & 63)) - 1);
    if (b == page_last >> 6 && (page_last & 63) != 63) bits &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
    if (dest) bits &= ~system_page_flags_valid_[b];  // (review: invalid ones only)
    if (!bits) continue;
    materializing_ = true;
    materialize_callback_(materialize_context_, range.first, length, ram_reader_);
    materializing_ = false;
    return;
  }
}

bool SharedMemory::RequestRanges(const std::pair<uint32_t, uint32_t>* ranges, size_t count) {
  // wwe2k14 patch (Session 49, NATIVE TRUTH skip write-back): resolve output only a texture holds is written into the
  // GPU copy before anything reads it (also before this request's own uploads: those come after it in the stream).
  if (skipped_pages_ && ranges) {
    for (size_t i = 0; i < count; ++i) CheckSkippedRanges(ranges[i]);
  }
  RamCountRequest(ranges, count);  // wwe2k14 patch (Session 49, RAM TIES: gpu_ram_stats): who requests resolve output
  if (geometry_tracking_ && ranges) NoteGeometryRanges(ranges, count);  // wwe2k14 patch (Session 50, LEAN)
  // wwe2k14 patch (Session 21, HOIST): every request, after its own uploads were decided.
  // wwe2k14 patch (Session 95, gpu_d3d_wall bit 8): the range stamp of the typed draw (owned_front_own.cpp) by range instead of by fetch slot, for every single-range request (vertex streams and index
  // buffers: ~1,700 a frame, 92-94% all valid, the same meshes in the shadow and scene passes): a range found all valid under the page-validity epoch still is while the epoch stands (every clear of a
  // valid bit bumps it first). Only the validity scan is skipped: the skipped-page check, the counters and OnRangesRequested (the hoist's access marks) run for every request. A stamp is only made from
  // an all-valid answer of the fast path, so a range that uploaded - or holds a page kept not valid - asks again. Command thread only, as this path.
  bool result;
  if (ranges && count == 1 && (REXCVAR_GET(gpu_d3d_wall) & 8u)) {
    struct Stamp {
      uint32_t start = 0, length = 0;
      uint64_t epoch = UINT64_MAX;
    };
    static Stamp stamps[4096];
    const uint32_t start = ranges[0].first, length = ranges[0].second;
    Stamp& stamp = stamps[(start * 0x9E3779B1u ^ length * 0x85EBCA77u) >> 20];
    const uint64_t epoch = repeat_epoch();
    if (stamp.epoch == epoch && stamp.start == start && stamp.length == length) {
      result = true;
    } else {
      const uint64_t hits_before = request_ranges_fast_stats_.hits;
      result = RequestRangesInner(ranges, count);
      if (result && request_ranges_fast_stats_.hits != hits_before) {
        stamp.start = start;
        stamp.length = length;
        stamp.epoch = epoch;
      }
    }
  } else {
    result = RequestRangesInner(ranges, count);
  }
  if (ranges && count) {
    OnRangesRequested(ranges, count);
  }
  return result;
}

bool SharedMemory::RequestRangesInner(const std::pair<uint32_t, uint32_t>* ranges, size_t count) {
  ++request_stats_.calls;  // wwe2k14 patch (Session 6, COUNTERS)
  if (ranges == nullptr || !count) {
    return true;
  }

  // wwe2k14 patch (Session 6, CP2): see RequestRangesFast.
  if (REXCVAR_GET(gpu_request_ranges_fast)) {
    return RequestRangesFast(ranges, count);
  }

  // Some texture or buffer is empty, for example - safe to draw in this case.
  std::vector<std::pair<uint32_t, uint32_t>> merged_ranges;
  merged_ranges.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    uint32_t start = ranges[i].first;
    uint32_t length = ranges[i].second;
    if (!length) {
      continue;
    }
    if (start > kBufferSize || (kBufferSize - start) < length) {
      return false;
    }
    merged_ranges.emplace_back(start, length);
  }
  if (merged_ranges.empty()) {
    return true;
  }

  SCOPE_profile_cpu_f("gpu");

  merged_ranges.resize(SortAndMergeRanges(merged_ranges.data(), merged_ranges.size()));
  return RequestMergedRanges(merged_ranges.data(), merged_ranges.size(), count);
}

size_t SharedMemory::SortAndMergeRanges(std::pair<uint32_t, uint32_t>* ranges, size_t count) {
  // wwe2k14 patch (Session 6, CP2): moved out of RequestRanges unchanged
  // (count must be at least 1), returns the number of merged ranges.
  std::sort(ranges, ranges + count,
            [](const std::pair<uint32_t, uint32_t>& a, const std::pair<uint32_t, uint32_t>& b) {
              return a.first < b.first;
            });
  size_t merged_write = 0;
  for (size_t i = 1; i < count; ++i) {
    std::pair<uint32_t, uint32_t>& range_previous = ranges[merged_write];
    const std::pair<uint32_t, uint32_t>& range_current = ranges[i];
    uint64_t previous_end = uint64_t(range_previous.first) + uint64_t(range_previous.second);
    uint64_t current_start = uint64_t(range_current.first);
    if (current_start <= previous_end) {
      uint64_t current_end = current_start + uint64_t(range_current.second);
      if (current_end > previous_end) {
        range_previous.second = uint32_t(current_end - uint64_t(range_previous.first));
      }
    } else {
      ranges[++merged_write] = range_current;
    }
  }
  return merged_write + 1;
}

bool SharedMemory::RequestMergedRanges(const std::pair<uint32_t, uint32_t>* merged_ranges,
                                       size_t merged_count, size_t request_count) {
  // wwe2k14 patch (Session 6, CP2): the rest of the original RequestRanges,
  // unchanged except for taking the merged ranges as a pointer and a count.
  for (size_t range_index = 0; range_index < merged_count; ++range_index) {
    const std::pair<uint32_t, uint32_t>& range = merged_ranges[range_index];
    if (!EnsureHostGpuMemoryAllocated(range.first, range.second)) {
      return false;
    }
  }

  // wwe2k14 patch (Session 7b, UPSKIP): the settings for this request and its UploadRanges call.
  LatchUploadSkipSettings();

  upload_ranges_.clear();
  auto append_upload_range = [this](uint32_t page_start, uint32_t page_count) {
    if (!page_count) {
      return;
    }
    if (!upload_ranges_.empty()) {
      std::pair<uint32_t, uint32_t>& last_upload_range = upload_ranges_.back();
      if (last_upload_range.first + last_upload_range.second == page_start) {
        last_upload_range.second += page_count;
        return;
      }
    }
    upload_ranges_.emplace_back(page_start, page_count);
  };
  {
    auto global_lock = global_critical_region_.Acquire();
    for (size_t range_index = 0; range_index < merged_count; ++range_index) {
      const std::pair<uint32_t, uint32_t>& range = merged_ranges[range_index];
      uint32_t page_first = range.first >> page_size_log2_;
      uint32_t page_last = (range.first + range.second - 1) >> page_size_log2_;
      uint32_t block_first = page_first >> 6;
      uint32_t block_last = page_last >> 6;
      uint32_t range_start = UINT32_MAX;
      for (uint32_t i = block_first; i <= block_last; ++i) {
        uint64_t block_valid = system_page_flags_valid_[i];
        // Consider pages in the block outside the requested range valid.
        if (i == block_first) {
          uint64_t block_before = (uint64_t(1) << (page_first & 63)) - 1;
          block_valid |= block_before;
        }
        if (i == block_last && (page_last & 63) != 63) {
          uint64_t block_inside = (uint64_t(1) << ((page_last & 63) + 1)) - 1;
          block_valid |= ~block_inside;
        }

        while (true) {
          uint32_t block_page;
          if (range_start == UINT32_MAX) {
            // Check if need to open a new range.
            if (!rex::bit_scan_forward(~block_valid, &block_page)) {
              break;
            }
            range_start = (i << 6) + block_page;
          } else {
            // Check if need to close the range.
            // Ignore the valid pages before the beginning of the range.
            uint64_t block_valid_from_start = block_valid;
            if (i == (range_start >> 6)) {
              block_valid_from_start &= ~((uint64_t(1) << (range_start & 63)) - 1);
            }
            if (!rex::bit_scan_forward(block_valid_from_start, &block_page)) {
              break;
            }
            append_upload_range(range_start, (i << 6) + block_page - range_start);
            // In the next iteration within this block, consider this range
            // valid since it has been queued for upload.
            block_valid |= (uint64_t(1) << block_page) - 1;
            range_start = UINT32_MAX;
          }
        }
      }
      if (range_start != UINT32_MAX) {
        append_upload_range(range_start, page_last + 1 - range_start);
      }
    }
    // wwe2k14 patch (Session 7b, UPSKIP): for cpf_up c, whether a watch fired for these pages since they were
    // last made valid - read under the global critical region, as the scan is.
    if (upload_hash_tracking_ && !upload_ranges_.empty()) {
      CaptureUploadClearOnly();
    }
  }

  COUNT_profile_set("gpu/shared_memory/request_ranges_count", uint32_t(request_count));
  COUNT_profile_set("gpu/shared_memory/request_ranges_merged_count", uint32_t(merged_count));
  COUNT_profile_set("gpu/shared_memory/request_ranges_upload_count",
                    uint32_t(upload_ranges_.size()));

  if (upload_ranges_.empty()) {
    return true;
  }

  // wwe2k14 patch (Session 7b, UPSKIP): pages whose bytes are already in the GPU buffer are made valid here
  // without a copy; the rest is uploaded as before. upload_ranges_ keeps every page found not valid (what
  // gpu_request_ranges_fast's statistics and verify read).
  const std::vector<std::pair<uint32_t, uint32_t>>* ranges_to_upload = &upload_ranges_;
  if (upload_skip_) {
    SkipUnchangedUploadPages();
    if (upload_ranges_after_skip_.empty()) {
      return true;
    }
    ranges_to_upload = &upload_ranges_after_skip_;
  }

  // wwe2k14 patch (Session 6, COUNTERS): what goes to the upload.
  ++request_stats_.upload_calls;
  uint32_t upload_pages = 0;
  for (const std::pair<uint32_t, uint32_t>& upload_range : *ranges_to_upload) {
    request_stats_.upload_bytes += uint64_t(upload_range.second) << page_size_log2_;
    upload_pages += upload_range.second;
  }
  // wwe2k14 patch (Session 7b, UPSKIP): cpf_up a, cpf_up2 a.
  ++upload_skip_frame_stats_.upload_calls;
  upload_skip_frame_stats_.pages_uploaded += upload_pages;
  upload_hash_next_page_ = 0;
  if (!UploadRanges(*ranges_to_upload)) {
    // wwe2k14 patch (Session 7b, UPSKIP review): without skipping, UploadRanges only makes a chunk valid once it
    // has upload buffer space for it, so on failure the rest stays not valid and is retried on its next request.
    if (upload_skip_) {
      InvalidateFailedUploadPages();
    }
    return false;
  }
  return true;
}

void SharedMemory::InvalidateFailedUploadPages() {
  // wwe2k14 patch (Session 7b, UPSKIP review): SkipUnchangedUploadPages made every hashed page it checked valid
  // before comparing, and left the ones whose bytes changed for UploadRanges to copy. UploadRanges has failed
  // part-way (no upload buffer space), so some of them may be valid without their copy. Every page of the list
  // is made not valid again: the ones UploadRanges did copy are only uploaded (or skipped by their new hash)
  // again on their next request, the ones it didn't are retried, as without skipping. They were all made valid
  // with written_by_gpu false, so no GPU-written page loses its bit. Marked as watch-fired (an invalidation
  // clear_memory_page_state=false would make too). Their watches stay armed: a watch firing on a page that is
  // not valid only invalidates it again.
  if (upload_ranges_after_skip_.empty()) {
    return;
  }
  auto global_lock = global_critical_region_.Acquire();
  BumpRepeatEpoch();  // wwe2k14 patch (Session 23, REPEAT)
  for (const std::pair<uint32_t, uint32_t>& upload_range : upload_ranges_after_skip_) {
    uint32_t page_first = upload_range.first;
    uint32_t page_last = upload_range.first + upload_range.second - 1;
    uint32_t block_first = page_first >> 6;
    uint32_t block_last = page_last >> 6;
    for (uint32_t i = block_first; i <= block_last; ++i) {
      uint64_t mask = UINT64_MAX;
      if (i == block_first) {
        mask &= ~((uint64_t(1) << (page_first & 63)) - 1);
      }
      if (i == block_last && (page_last & 63) != 63) {
        mask &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
      }
      // The lock-free gpu_request_ranges_fast check reads these blocks on this thread only, so no atomic store
      // is needed for it; guest threads change them under the global critical region, held here.
      system_page_flags_valid_[i] &= ~mask;
      system_page_flags_valid_and_gpu_written_[i] &= ~mask;
      system_page_flags_watch_fired_[i] |= mask;
    }
  }
  REXGPU_ERROR(
      "[s7][UPSKIP] an upload failed with gpu_upload_skip_unchanged on - its {} range(s) made not valid again",
      upload_ranges_after_skip_.size());
}

void SharedMemory::LatchUploadSkipSettings() {
  // wwe2k14 patch (Session 7b, UPSKIP). Read once per request; nothing here depends on the previous request's
  // settings except the hashes, and uploads made while hashing was off didn't record any, so every hash is
  // dropped when hashing starts (again).
  const bool skip = REXCVAR_GET(gpu_upload_skip_unchanged);
  const bool tracking = skip || REXCVAR_GET(gpu_upload_skip_unchanged_stats);
  if (tracking && !upload_hash_tracking_previous_) {
    size_t page_count = size_t(kBufferSize >> page_size_log2_);
    if (upload_hashes_.size() != page_count) {
      // 8 bytes per page (1 MB with 4 KB pages), allocated the first time it's turned on.
      upload_hashes_.assign(page_count, 0);
      upload_hash_valid_.assign(num_system_page_flags_, 0);
      upload_clear_only_.assign(num_system_page_flags_, 0);
      upload_hash_scratch_.resize(size_t(1) << page_size_log2_);
    } else {
      std::fill(upload_hash_valid_.begin(), upload_hash_valid_.end(), uint64_t(0));
    }
    // wwe2k14 patch (Session 7b, UPSKIP review): no hash is left to drop for earlier losses.
    upload_hashes_lost_seen_ = GetLostUploadCount();
  } else if (tracking) {
    // wwe2k14 patch (Session 7b, UPSKIP review): copies recorded earlier may never have reached the GPU buffer
    // (Vulkan: SUBMITFIX dropped the commands of an async submission after its recording or ~1 s of submit
    // retries failed). Their hashes would say the buffer holds bytes it doesn't, and with skipping that would
    // last until the guest changes those bytes, where without it the next frame's re-upload repairs them. So
    // every hash is dropped, and with skipping on, the pages valid only through uploads (or skips relying on
    // them) are made not valid, as the frame-end clear does, so all of them are uploaded again on their next
    // request (stats only: uploads as without hashing, nothing else changes). Before this request's scan. Draws
    // recorded between the loss and here may have read stale data (also without skipping, for the lost copies;
    // with skipping, also for pages skipped meanwhile, in the submissions recorded before the submit thread
    // gave up on that one - repaired from here on).
    const uint64_t lost = GetLostUploadCount();
    if (lost != upload_hashes_lost_seen_) {
      upload_hashes_lost_seen_ = lost;
      std::fill(upload_hash_valid_.begin(), upload_hash_valid_.end(), uint64_t(0));
      if (skip) {
        SetSystemPageBlocksValidWithGpuDataWritten();
      }
      REXGPU_ERROR(
          "[s7][UPSKIP] recorded uploads were lost (dropped submission commands, {} so far) - every upload hash "
          "dropped{}",
          lost, skip ? " and the uploaded pages made not valid" : "");
    }
  }
  upload_hash_tracking_previous_ = tracking;
  upload_hash_tracking_ = tracking;
  upload_skip_ = skip;
  upload_skip_verify_ = skip && REXCVAR_GET(gpu_upload_skip_unchanged_verify);
}

void SharedMemory::CaptureUploadClearOnly() {
  // wwe2k14 patch (Session 7b, UPSKIP): global critical region held by the caller (the watch-fired bits are set
  // by guest threads in MemoryInvalidationCallback).
  for (const std::pair<uint32_t, uint32_t>& upload_range : upload_ranges_) {
    uint32_t page_first = upload_range.first;
    uint32_t page_last = upload_range.first + upload_range.second - 1;
    uint32_t block_first = page_first >> 6;
    uint32_t block_last = page_last >> 6;
    for (uint32_t i = block_first; i <= block_last; ++i) {
      uint64_t mask = UINT64_MAX;
      if (i == block_first) {
        mask &= ~((uint64_t(1) << (page_first & 63)) - 1);
      }
      if (i == block_last && (page_last & 63) != 63) {
        mask &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
      }
      upload_clear_only_[i] =
          (upload_clear_only_[i] & ~mask) | (~system_page_flags_watch_fired_[i] & mask);
    }
  }
}

namespace {
// wwe2k14 patch (Session 19, PARHASH): a small pool for one job at a time from one caller (the command processor).
// The job word holds the generation (high 32 bits) and the next index (low 32; 0xFFFFFFFF = closed). A worker counts
// itself busy before reading the job word, so the caller, which closes the job and then waits for busy == 0, never
// returns (and never rewrites fn_ / count_) while a worker might still read them. Workers spin ~50 us for the next
// job before sleeping, since a frame makes dozens of requests.
class ParhashPool {
 public:
  ~ParhashPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (std::thread& t : threads_) {
      t.join();
    }
  }
  void Run(uint32_t workers, uint32_t count, const std::function<void(uint32_t)>& fn) {
    while (threads_.size() < workers) {
      const uint32_t id = uint32_t(threads_.size());
      threads_.emplace_back([this, id] { WorkerMain(id); });
    }
    fn_ = &fn;
    count_ = count;
    done_.store(0, std::memory_order_relaxed);
    const uint64_t gen = ++gen_;
    job_.store(gen << 32);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      wake_gen_ = gen;
    }
    cv_.notify_all();
    Work(gen);
    while (done_.load(std::memory_order_acquire) < count) {
      std::this_thread::yield();
    }
    job_.store((gen << 32) | 0xFFFFFFFFull);
    while (busy_.load()) {
      std::this_thread::yield();
    }
  }

 private:
  static constexpr uint32_t kChunk = 16;  // pages per claim (64 KB)
  void Work(uint64_t gen) {
    for (;;) {
      uint64_t job = job_.load();
      if ((job >> 32) != gen) {
        return;
      }
      const uint32_t index = uint32_t(job);
      if (index == 0xFFFFFFFFu || index >= count_) {
        return;
      }
      const uint32_t end = std::min(index + kChunk, count_);
      if (!job_.compare_exchange_weak(job, (gen << 32) | end)) {
        continue;
      }
      for (uint32_t i = index; i < end; ++i) {
        (*fn_)(i);
      }
      done_.fetch_add(end - index, std::memory_order_release);
    }
  }
  void WorkerMain(uint32_t id) {
    rex::thread::set_current_thread_name(fmt::format("GPU Upload Hash {}", id));
    uint64_t seen = 0;
    for (;;) {
      uint64_t gen = 0;
      const auto spin_start = std::chrono::steady_clock::now();
      for (uint32_t spin = 0;; ++spin) {
        const uint64_t job_gen = job_.load(std::memory_order_relaxed) >> 32;
        if (job_gen != seen) {
          gen = job_gen;
          break;
        }
        if ((spin & 63) == 63 &&
            std::chrono::steady_clock::now() - spin_start > std::chrono::microseconds(50)) {
          break;
        }
        std::this_thread::yield();
      }
      if (!gen) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return stop_ || wake_gen_ != seen; });
        if (stop_) {
          return;
        }
        gen = wake_gen_;
      }
      seen = gen;
      busy_.fetch_add(1);
      Work(gen);
      busy_.fetch_sub(1);
    }
  }
  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_ = false;
  uint64_t wake_gen_ = 0;
  uint64_t gen_ = 0;
  std::atomic<uint64_t> job_{0};
  std::atomic<uint32_t> done_{0};
  std::atomic<uint32_t> busy_{0};
  const std::function<void(uint32_t)>* fn_ = nullptr;
  uint32_t count_ = 0;
};

ParhashPool& GetParhashPool() {
  static ParhashPool pool;
  return pool;
}
}  // namespace

void SharedMemory::SkipUnchangedUploadPagesParallel(uint32_t workers, uint32_t min_pages) {
  // wwe2k14 patch (Session 19, PARHASH): see the declaration. Mirrors SkipUnchangedUploadPages without verify.
  upload_ranges_after_skip_.clear();
  auto append = [this](uint32_t page_start, uint32_t page_count) {
    if (!page_count) {
      return;
    }
    if (!upload_ranges_after_skip_.empty()) {
      std::pair<uint32_t, uint32_t>& last_range = upload_ranges_after_skip_.back();
      if (last_range.first + last_range.second == page_start) {
        last_range.second += page_count;
        return;
      }
    }
    upload_ranges_after_skip_.emplace_back(page_start, page_count);
  };
  const uint32_t page_size = uint32_t(1) << page_size_log2_;
  const uint64_t* hash_valid = upload_hash_valid_.data();
  // wwe2k14 patch (Session 95, r7 lane D, gpu_tex_planes_nohash): a declared plane page is not hashed at all (it goes to the upload as a page without a hash).
  const bool planes_nohash = g_hot.enabled && REXCVAR_GET(gpu_tex_planes_nohash) && REXCVAR_GET(gpu_tex_planes_unwatched);
  uint32_t nohash_pages = 0;
  uint32_t last_nohash_page = ~0u;
  auto has_hash = [hash_valid, planes_nohash, &nohash_pages, &last_nohash_page](uint32_t page) -> bool {
    if (!((hash_valid[page >> 6] >> (page & 63)) & 1)) return false;
    // Only a hot, declared page not yet uploaded this frame: a repeat request hashes against the hash written at the copy (a stale engine bit on a watched page does nothing).
    if (planes_nohash && g_hot.IsHot(page) && g_hot.IsEngine(page) && !g_hot.WasUploaded(page)) {
      if (page != last_nohash_page) {
        last_nohash_page = page;
        ++nohash_pages;
      }
      return false;
    }
    // wwe2k14 patch (Session 95, HOTPAGES): no hash on a page that always changes.
    // A hot page is hashed only when it was already uploaded since the last frame end (a second request).
    return !g_hot.IsHot(page) || g_hot.WasUploaded(page) || g_hot.IsEngine(page);
  };
  UploadSkipStats& stats = upload_skip_frame_stats_;
  UploadSkipWindow& window = upload_skip_window_;
  // Pass 1: the runs, in order; hashed runs made valid now (before their pages are hashed, as in the serial scan).
  parhash_segments_.clear();
  parhash_pages_.clear();
  uint32_t processed_end = 0;
  for (const std::pair<uint32_t, uint32_t>& upload_range : upload_ranges_) {
    const uint32_t range_end = upload_range.first + upload_range.second;
    uint32_t page = std::max(upload_range.first, processed_end);
    processed_end = std::max(processed_end, range_end);
    while (page < range_end) {
      uint32_t hashed_first = page;
      while (hashed_first < range_end && !has_hash(hashed_first)) {
        ++hashed_first;
      }
      if (hashed_first > page) {
        parhash_segments_.push_back({page, hashed_first - page, false});
      }
      if (hashed_first >= range_end) {
        break;
      }
      uint32_t hashed_end = hashed_first + 1;
      while (hashed_end < range_end && has_hash(hashed_end)) {
        ++hashed_end;
      }
      MakeRangeValid(hashed_first << page_size_log2_, (hashed_end - hashed_first) << page_size_log2_,
                     false);
      ++window.skip_make_valid_calls;
      parhash_segments_.push_back({hashed_first, hashed_end - hashed_first, true});
      for (uint32_t hashed_page = hashed_first; hashed_page < hashed_end; ++hashed_page) {
        parhash_pages_.push_back(hashed_page);
      }
      page = hashed_end;
    }
  }
  // Pass 2: the hashes.
  const uint32_t count = uint32_t(parhash_pages_.size());
  parhash_match_.resize(count);
  const auto hash_start = std::chrono::steady_clock::now();
  if (count) {
    const uint32_t* pages = parhash_pages_.data();
    uint8_t* match = parhash_match_.data();
    const uint64_t* hashes = upload_hashes_.data();
    memory::Memory& mem = memory();
    const uint32_t log2 = page_size_log2_;
    std::function<void(uint32_t)> hash_one = [pages, match, hashes, &mem, log2, page_size](uint32_t i) {
      const uint32_t p = pages[i];
      match[i] = XXH3_64bits(mem.TranslatePhysical(p << log2), page_size) == hashes[p];
    };
    ParhashWindow& pw = parhash_window_;
    ++pw.requests;
    pw.pages += count;
    if (count >= min_pages) {
      GetParhashPool().Run(workers, count, hash_one);
      ++pw.parallel_requests;
      pw.parallel_pages += count;
    } else {
      for (uint32_t i = 0; i < count; ++i) {
        hash_one(i);
      }
    }
  }
  const uint64_t hash_ns = uint64_t(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - hash_start).count());
  stats.hash_ns += hash_ns;
  parhash_window_.hash_ns += hash_ns;
  stats.bytes_hashed += uint64_t(count) << page_size_log2_;
  window.pages_hash_checked += count;
  // Pass 3: what still needs a copy, in the original order.
  uint32_t index = 0;
  for (const ParhashSegment& segment : parhash_segments_) {
    if (!segment.hashed) {
      append(segment.first, segment.count);
      continue;
    }
    for (uint32_t k = 0; k < segment.count; ++k, ++index) {
      if (parhash_match_[index]) {
        ++stats.pages_skipped;
      } else {
        append(segment.first + k, 1);
      }
    }
  }
  g_hot.nohash_pages += nohash_pages;
}

void SharedMemory::SkipUnchangedUploadPages() {
  // wwe2k14 patch (Session 19, PARHASH): gpu_upload_skip_parallel_hash (never with verify).
  if (!upload_skip_verify_) {
    const int32_t parallel = REXCVAR_GET(gpu_upload_skip_parallel_hash);
    if (parallel > 0) {
      SkipUnchangedUploadPagesParallel(uint32_t(std::min(parallel, 31)),
                                       uint32_t(std::max(REXCVAR_GET(gpu_upload_skip_parallel_min_pages), 1)));
      return;
    }
  }
  // wwe2k14 patch (Session 7b, UPSKIP): gpu_upload_skip_unchanged. For each page of upload_ranges_ (not valid,
  // about to be uploaded) whose upload hash is valid - the GPU buffer holds exactly the bytes last uploaded to it,
  // nothing else has written it since - do what UploadRanges does to the page before its copy (MakeRangeValid:
  // valid, not GPU-written, watched, so a guest write from now on invalidates it again), then hash the guest
  // bytes at that point, where UploadRanges would read them for the copy. Equal hash: the buffer already holds
  // these bytes, nothing to copy. Otherwise, and for pages without a hash, the page goes to the upload as before
  // (UploadRanges makes it valid again before copying, which changes nothing if no write came in between, or
  // re-validates it after one, as it would without this). The pages left are merged into ascending ranges as
  // the scan makes them.
  upload_ranges_after_skip_.clear();
  auto append = [this](uint32_t page_start, uint32_t page_count) {
    if (!page_count) {
      return;
    }
    if (!upload_ranges_after_skip_.empty()) {
      std::pair<uint32_t, uint32_t>& last_range = upload_ranges_after_skip_.back();
      if (last_range.first + last_range.second == page_start) {
        last_range.second += page_count;
        return;
      }
    }
    upload_ranges_after_skip_.emplace_back(page_start, page_count);
  };
  const uint32_t page_size = uint32_t(1) << page_size_log2_;
  const uint64_t* hash_valid = upload_hash_valid_.data();
  // wwe2k14 patch (Session 95, r7 lane D, gpu_tex_planes_nohash): a declared plane page is not hashed at all (it goes to the upload as a page without a hash).
  const bool planes_nohash = g_hot.enabled && REXCVAR_GET(gpu_tex_planes_nohash) && REXCVAR_GET(gpu_tex_planes_unwatched);
  uint32_t nohash_pages = 0;
  uint32_t last_nohash_page = ~0u;
  auto has_hash = [hash_valid, planes_nohash, &nohash_pages, &last_nohash_page](uint32_t page) -> bool {
    if (!((hash_valid[page >> 6] >> (page & 63)) & 1)) return false;
    // Only a hot, declared page not yet uploaded this frame: a repeat request hashes against the hash written at the copy (a stale engine bit on a watched page does nothing).
    if (planes_nohash && g_hot.IsHot(page) && g_hot.IsEngine(page) && !g_hot.WasUploaded(page)) {
      if (page != last_nohash_page) {
        last_nohash_page = page;
        ++nohash_pages;
      }
      return false;
    }
    // wwe2k14 patch (Session 95, HOTPAGES): no hash on a page that always changes.
    // A hot page is hashed only when it was already uploaded since the last frame end (a second request).
    return !g_hot.IsHot(page) || g_hot.WasUploaded(page) || g_hot.IsEngine(page);
  };
  uint8_t* snapshot = upload_hash_scratch_.data();
  const bool verify = upload_skip_verify_;
  UploadSkipStats& stats = upload_skip_frame_stats_;
  UploadSkipWindow& window = upload_skip_window_;
  // wwe2k14 patch (Session 7b, UPSKIP review): upload_ranges_ is in ascending page order except that a page can
  // be listed twice (the boundary page of two merged byte ranges sharing it). Each page is checked and kept for
  // the upload once, so it is copied once - without this it would be copied twice with overlapping destination
  // regions (and, should a guest write land between the two snapshots, the GPU buffer could end up with either
  // while the hash is of the second). Pages from processed_end on are not processed yet.
  uint32_t processed_end = 0;
  for (const std::pair<uint32_t, uint32_t>& upload_range : upload_ranges_) {
    const uint32_t range_end = upload_range.first + upload_range.second;
    uint32_t page = std::max(upload_range.first, processed_end);
    processed_end = std::max(processed_end, range_end);
    while (page < range_end) {
      uint32_t hashed_first = page;
      while (hashed_first < range_end && !has_hash(hashed_first)) {
        ++hashed_first;
      }
      append(page, hashed_first - page);
      window.pages_no_hash += hashed_first - page;  // wwe2k14 patch (Session 87, diagnostic)
      if (hashed_first >= range_end) {
        break;
      }
      uint32_t hashed_end = hashed_first + 1;
      while (hashed_end < range_end && has_hash(hashed_end)) {
        ++hashed_end;
      }
      MakeRangeValid(hashed_first << page_size_log2_, (hashed_end - hashed_first) << page_size_log2_,
                     false);
      ++window.skip_make_valid_calls;
      const auto hash_start = std::chrono::steady_clock::now();
      for (uint32_t hashed_page = hashed_first; hashed_page < hashed_end; ++hashed_page) {
        const uint8_t* guest = memory().TranslatePhysical(hashed_page << page_size_log2_);
        const uint8_t* hashed_bytes = guest;
        if (verify) {
          // Keep the exact bytes hashed for the comparison with the GPU buffer.
          std::memcpy(snapshot, guest, page_size);
          hashed_bytes = snapshot;
        }
        const bool page_differs = XXH3_64bits(hashed_bytes, page_size) != upload_hashes_[hashed_page];
        {
          // wwe2k14 patch (Session 87, step 4, diagnostic): a watch fired on the page since it was last valid, or only
          // the frame-end clear made it not valid; and whether its bytes still equal the last upload's.
          const bool clear_only = (upload_clear_only_[hashed_page >> 6] >> (hashed_page & 63)) & 1;
          ++(clear_only ? (page_differs ? window.pages_clear_changed : window.pages_clear_same)
                        : (page_differs ? window.pages_fired_changed : window.pages_fired_same));
        }
        if (page_differs) {
          append(hashed_page, 1);
          continue;
        }
        if (!verify) {
          ++stats.pages_skipped;
          continue;
        }
        // Verify: uploaded anyway; up to kUploadSkipVerifyPagesPerFrame per frame first get a readback of the
        // GPU buffer's page as it is here, before this upload.
        ++window.pages_matched_verify;
        uint32_t candidate = upload_skip_verify_candidates_++;
        if (candidate >= upload_skip_verify_offset_ &&
            upload_skip_verify_recorded_frame_ < kUploadSkipVerifyPagesPerFrame &&
            RecordUploadSkipVerify(hashed_page, snapshot)) {
          ++upload_skip_verify_recorded_frame_;
        }
        append(hashed_page, 1);
      }
      stats.hash_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - hash_start)
                                    .count());
      stats.bytes_hashed += uint64_t(hashed_end - hashed_first) << page_size_log2_;
      window.pages_hash_checked += hashed_end - hashed_first;
      page = hashed_end;
    }
  }
  g_hot.nohash_pages += nohash_pages;
}

void SharedMemory::CopyAndHashUploadPages(uint8_t* dest, uint32_t page_first, uint32_t size) {
  // wwe2k14 patch (Session 7b, UPSKIP): the UploadRanges copy of one chunk (after its MakeRangeValid), through a
  // one-page snapshot: guest -> snapshot, hash the snapshot, snapshot -> upload buffer. The bytes copied are
  // those of a read of guest memory at the same point as the plain memcpy's, and the hash is of exactly those
  // bytes, even if a guest thread writes the page meanwhile (that write invalidates the page through its watch,
  // so it is requested again; the hash stays that of what the GPU buffer gets).
  const uint32_t page_size = uint32_t(1) << page_size_log2_;
  const uint8_t* source = memory().TranslatePhysical(page_first << page_size_log2_);
  uint8_t* snapshot = upload_hash_scratch_.data();
  UploadSkipStats& stats = upload_skip_frame_stats_;
  UploadSkipWindow& window = upload_skip_window_;
  const auto hash_start = std::chrono::steady_clock::now();
  uint32_t page = page_first;
  uint32_t offset = 0;
  for (; size - offset >= page_size; offset += page_size, ++page) {
    std::memcpy(snapshot, source + offset, page_size);
    const uint64_t hash = XXH3_64bits(snapshot, page_size);
    std::memcpy(dest + offset, snapshot, page_size);
    const uint64_t page_bit = uint64_t(1) << (page & 63);
    uint64_t& hash_valid_block = upload_hash_valid_[page >> 6];
    if (page < upload_hash_next_page_) {
      // wwe2k14 patch (Session 7b, UPSKIP review): copied a second time in this call (a page listed twice in
      // upload_ranges_ - only without skipping, which uploads each page once). The two copies' destination
      // regions overlap, so if the snapshots differ (a guest write in between) the buffer may hold either:
      // no hash, so the page is uploaded again on its next request, as without hashing.
      hash_valid_block &= ~page_bit;
      continue;
    }
    upload_hash_next_page_ = page + 1;
    // cpf_up c: the GPU buffer held the previous upload's bytes, no watch fired since the page was last made
    // valid (so with clear_memory_page_state=false it would have stayed valid, read from the buffer), and the
    // guest bytes differ from them.
    if ((hash_valid_block & page_bit) && (upload_clear_only_[page >> 6] & page_bit) &&
        upload_hashes_[page] != hash) {
      ++stats.pages_c1_stale;
      if (IsGeometryOnlyPage(page)) ++stats.pages_c1_stale_geometry;  // wwe2k14 patch (Session 50, LEAN)
      if (window.c1_example_count < uint32_t(rex::countof(window.c1_examples))) {
        window.c1_examples[window.c1_example_count++] = page << page_size_log2_;
      }
    }
    upload_hashes_[page] = hash;
    hash_valid_block |= page_bit;
  }
  if (offset < size) {
    // Not a whole page (UploadRanges copies whole pages, so not expected): copied as without hashing, and the
    // page has no hash until it's uploaded whole.
    std::memcpy(dest + offset, source + offset, size - offset);
    upload_hash_valid_[page >> 6] &= ~(uint64_t(1) << (page & 63));
  }
  stats.hash_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - hash_start)
                                .count());
  stats.bytes_hashed += offset;
}

void SharedMemory::InvalidateUploadHashes(uint32_t start, uint32_t length) {
  // wwe2k14 patch (Session 7b, UPSKIP). Maintained whenever the hashes exist (also while hashing is off, though
  // turning it on drops them all anyway).
  if (upload_hash_valid_.empty() || !length || start >= kBufferSize) {
    return;
  }
  length = std::min(length, kBufferSize - start);
  uint32_t page_first = start >> page_size_log2_;
  uint32_t page_last = (start + length - 1) >> page_size_log2_;
  uint32_t block_first = page_first >> 6;
  uint32_t block_last = page_last >> 6;
  for (uint32_t i = block_first; i <= block_last; ++i) {
    uint64_t mask = UINT64_MAX;
    if (i == block_first) {
      mask &= ~((uint64_t(1) << (page_first & 63)) - 1);
    }
    if (i == block_last && (page_last & 63) != 63) {
      mask &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
    }
    upload_hash_valid_[i] &= ~mask;
  }
}

SharedMemory::UploadSkipStats SharedMemory::TakeUploadSkipStats(uint64_t frame) {
  // wwe2k14 patch (Session 7b, UPSKIP).
  UploadSkipStats stats = upload_skip_frame_stats_;
  upload_skip_frame_stats_ = UploadSkipStats();
  // Verify round robin: the next frame starts recording after the would-be-skipped pages recorded this frame
  // (from the first one again once that passes this frame's count).
  if (uint64_t(upload_skip_verify_offset_) + kUploadSkipVerifyPagesPerFrame <
      upload_skip_verify_candidates_) {
    upload_skip_verify_offset_ += kUploadSkipVerifyPagesPerFrame;
  } else {
    upload_skip_verify_offset_ = 0;
  }
  upload_skip_verify_candidates_ = 0;
  upload_skip_verify_recorded_frame_ = 0;

  UploadSkipWindow& window = upload_skip_window_;
  window.upload_calls += stats.upload_calls;
  window.pages_uploaded += stats.pages_uploaded;
  window.pages_skipped += stats.pages_skipped;
  window.pages_c1_stale += stats.pages_c1_stale;
  window.c1_stale_total += stats.pages_c1_stale;
  window.pages_c1_stale_geometry += stats.pages_c1_stale_geometry;  // wwe2k14 patch (Session 50, LEAN)
  window.c1_stale_geometry_total += stats.pages_c1_stale_geometry;
  window.bytes_hashed += stats.bytes_hashed;
  window.hash_ns += stats.hash_ns;
  if (window.window_first_frame == UINT64_MAX) {
    window.window_first_frame = frame;
    window.verify_at_start = upload_skip_verify_stats_;
    return stats;
  }
  if (frame - window.window_first_frame < 300) {
    return stats;
  }
  const UploadSkipVerifyStats& verify_now = upload_skip_verify_stats_;
  const UploadSkipVerifyStats& verify_start = window.verify_at_start;
  // Only when hashing ran in this window (nothing is hashed with every setting off).
  if (window.bytes_hashed || verify_now.recorded != verify_start.recorded ||
      verify_now.checked != verify_start.checked) {
    uint64_t frames = std::max(frame - window.window_first_frame, uint64_t(1));
    std::string c1_examples;
    for (uint32_t i = 0; i < window.c1_example_count; ++i) {
      c1_examples += fmt::format("{}0x{:08X}", i ? ", " : "", window.c1_examples[i]);
    }
    REXGPU_INFO(
        "[s7][UPSKIP] frames {}-{} (skip {}, stats {}, verify {}): {} upload calls ({:.1f}/frame), {} pages "
        "uploaded ({:.1f}/frame), {} pages skipped ({:.1f}/frame) of {} not-valid pages with a hash checked, {} "
        "matched but uploaded for verify; {} uploaded pages changed while only the frame-end clear had "
        "invalidated them (stale with clear_memory_page_state=false; {} in total){}{}; hashed {} MB in {:.2f} "
        "ms ({:.3f} ms/frame); {} MakeRangeValid calls for skip checks; verify: {} readbacks recorded, {} "
        "dropped, {} compared, {} mismatches ({} in total)",
        window.window_first_frame, frame, REXCVAR_GET(gpu_upload_skip_unchanged),
        REXCVAR_GET(gpu_upload_skip_unchanged_stats), REXCVAR_GET(gpu_upload_skip_unchanged_verify),
        window.upload_calls, double(window.upload_calls) / double(frames), window.pages_uploaded,
        double(window.pages_uploaded) / double(frames), window.pages_skipped,
        double(window.pages_skipped) / double(frames), window.pages_hash_checked,
        window.pages_matched_verify, window.pages_c1_stale, window.c1_stale_total,
        window.c1_example_count ? ", e.g. " : "", c1_examples, window.bytes_hashed >> 20,
        double(window.hash_ns) / 1.0e6, double(window.hash_ns) / 1.0e6 / double(frames),
        window.skip_make_valid_calls, verify_now.recorded - verify_start.recorded,
        verify_now.dropped - verify_start.dropped, verify_now.checked - verify_start.checked,
        verify_now.mismatches - verify_start.mismatches, verify_now.mismatches);
    REXGPU_INFO("[s87][UPSKIP] frames {}-{}, skip-scan pages per frame: watch fired + same bytes {:.1f}, fired + changed "
                "{:.1f}, only the frame-end clear + same {:.1f}, clear + changed {:.1f}, no hash yet {:.1f}; "
                "MakeRangeValid calls {:.1f} a frame, {:.1f} of them over pages none of which had a watch fire "
                "(gpu_make_valid_skip_arm {}; verify: {} pages a skipped call would have left unwatched)",
                window.window_first_frame, frame, double(window.pages_fired_same) / double(frames),
                double(window.pages_fired_changed) / double(frames), double(window.pages_clear_same) / double(frames),
                double(window.pages_clear_changed) / double(frames), double(window.pages_no_hash) / double(frames),
                double(window.make_valid_calls) / double(frames), double(window.make_valid_unfired) / double(frames),
                REXCVAR_GET(gpu_make_valid_skip_arm), window.make_valid_skip_wrong);
    // wwe2k14 patch (Session 50, LEAN; review: its own line, after the one tools parse).
    if (geometry_tracking_ || window.geometry_kept_pages || window.c1_stale_geometry_total) {
      REXGPU_INFO("[s7][UPSKIP] geometry: stale geometry-only pages {} ({} in total; trust {}, check {}), kept valid "
                  "by the frame-end clear {:.1f}/frame",
                  window.pages_c1_stale_geometry, window.c1_stale_geometry_total, geometry_trust_,
                  geometry_tracking_ && !geometry_trust_, double(window.geometry_kept_pages) / double(frames));
    }
  }
  // wwe2k14 patch (Session 19, PARHASH): one line per window while the parallel scan ran.
  if (parhash_window_.requests) {
    const ParhashWindow& pw = parhash_window_;
    uint64_t frames = std::max(frame - window.window_first_frame, uint64_t(1));
    REXGPU_INFO(
        "[s19][PARHASH] frames {}-{} (workers {}, min pages {}): {} requests ({:.1f}/frame), {} on the workers; "
        "{} pages hashed ({:.1f}/frame), {} of them on the workers; hashing {:.3f} ms/frame",
        window.window_first_frame, frame, REXCVAR_GET(gpu_upload_skip_parallel_hash),
        REXCVAR_GET(gpu_upload_skip_parallel_min_pages), pw.requests, double(pw.requests) / double(frames),
        pw.parallel_requests, pw.pages, double(pw.pages) / double(frames), pw.parallel_pages,
        double(pw.hash_ns) / 1.0e6 / double(frames));
    parhash_window_ = ParhashWindow();
  }
  UploadSkipWindow next_window;
  next_window.window_first_frame = frame;
  next_window.c1_stale_total = window.c1_stale_total;
  next_window.c1_stale_geometry_total = window.c1_stale_geometry_total;  // wwe2k14 patch (Session 50, LEAN)
  next_window.verify_at_start = upload_skip_verify_stats_;
  window = next_window;
  return stats;
}

bool SharedMemory::RequestRangesFast(const std::pair<uint32_t, uint32_t>* ranges, size_t count) {
  // wwe2k14 patch (Session 6, CP2): gpu_request_ranges_fast.
  // - If every page of every requested range is already valid and its host GPU
  //   memory is allocated, the original path would allocate nothing, find
  //   nothing to upload and return true - return true without the global
  //   critical region and without merging. Other threads only ever clear valid
  //   bits (MemoryInvalidationCallback, under the global critical region), and
  //   a guest CPU write is preceded by that invalidation (the access violation
  //   is handled before the write is retried), which happens before the
  //   guest's write pointer kick that this thread acquired before seeing the
  //   draw - so a valid page seen here can only be stale if the guest writes
  //   it without synchronizing with the GPU, which races in the original as
  //   well. Everything that sets valid bits (uploads, resolves, memexport,
  //   frame-end clearing) runs on this thread.
  // - Otherwise the original locked path, a single range without a vector,
  //   several in a reused member vector (every caller is on the command
  //   processor thread, as for upload_ranges_).
  // Read per call, nothing carries over between calls, so it can be toggled at
  // any point.
  bool verify = REXCVAR_GET(gpu_request_ranges_fast_verify);
  // Before the lock-free loads (acquire): if an invalidation's count is seen
  // here, its cleared bits are seen by the loads below.
  uint64_t invalidation_count_before =
      verify ? invalidation_count_.load(std::memory_order_acquire) : 0;

  // Same validation as the original path: empty ranges are skipped, a range
  // outside the buffer fails the request before anything is done.
  size_t non_empty_count = 0;
  size_t non_empty_last = 0;
  uint64_t request_bytes = 0;
  bool all_valid = true;
  for (size_t i = 0; i < count; ++i) {
    uint32_t start = ranges[i].first;
    uint32_t length = ranges[i].second;
    if (!length) {
      continue;
    }
    if (start > kBufferSize || (kBufferSize - start) < length) {
      return false;
    }
    ++non_empty_count;
    non_empty_last = i;
    request_bytes += length;
    if (all_valid) {
      all_valid = IsRangeValidAndAllocated(start, length);
    }
  }
  if (!non_empty_count) {
    return true;
  }

  RequestRangesFastStats& stats = request_ranges_fast_stats_;
  ++stats.requests;
  if (non_empty_count == 1) {
    ++stats.single_range_requests;
  }
  if (all_valid) {
    ++stats.hits;
    stats.hit_bytes += request_bytes;
    if (!verify) {
      return true;
    }
  }

  SCOPE_profile_cpu_f("gpu");

  uint32_t sparse_allocations_before = host_gpu_memory_sparse_allocations_;
  bool result;
  if (non_empty_count == 1) {
    // One range - nothing to sort or merge.
    result = RequestMergedRanges(&ranges[non_empty_last], 1, count);
  } else {
    request_ranges_scratch_.clear();
    for (size_t i = 0; i < count; ++i) {
      if (ranges[i].second) {
        request_ranges_scratch_.push_back(ranges[i]);
      }
    }
    size_t merged_count =
        SortAndMergeRanges(request_ranges_scratch_.data(), request_ranges_scratch_.size());
    result = RequestMergedRanges(request_ranges_scratch_.data(), merged_count, count);
  }

  if (!all_valid) {
    if (result) {
      for (const std::pair<uint32_t, uint32_t>& upload_range : upload_ranges_) {
        stats.miss_upload_pages += upload_range.second;
      }
    }
    return result;
  }

  // Verify: the lock-free check said there is nothing to do, the original path
  // has just run - did it allocate or upload anything (or fail)?
  ++stats.verify_checks;
  bool allocated = host_gpu_memory_sparse_allocations_ != sparse_allocations_before;
  // upload_ranges_ is only stale if RequestMergedRanges failed in
  // EnsureHostGpuMemoryAllocated, which already counts as a mismatch.
  bool uploaded = result && !upload_ranges_.empty();
  if (!result || allocated || uploaded) {
    // After the locked scan (the global critical region synchronizes with the
    // invalidation): a changed count means a guest invalidation ran between
    // the lock-free check and the scan - the same outcome the original path
    // has when the guest thread takes the lock first, not an error in the
    // check.
    bool raced = invalidation_count_.load(std::memory_order_acquire) != invalidation_count_before;
    ++stats.verify_mismatches;
    ++stats.verify_mismatches_total;
    if (raced) {
      ++stats.verify_mismatches_raced;
    }
    if (stats.verify_mismatches_logged < 16) {
      ++stats.verify_mismatches_logged;
      const std::pair<uint32_t, uint32_t>& last_range = ranges[non_empty_last];
      std::pair<uint32_t, uint32_t> first_upload =
          (result && !upload_ranges_.empty()) ? upload_ranges_.front()
                                              : std::pair<uint32_t, uint32_t>(0, 0);
      REXGPU_WARN(
          "[s6][CP2] verify mismatch {}: lock-free check saw {} range(s) ({} bytes, last "
          "{:08X}+{:X}) all valid and allocated, but the locked path returned {}, uploaded {} "
          "range(s) (first page {:X} x{}), sparse allocations {} -> {}{}",
          stats.verify_mismatches_total, non_empty_count, request_bytes, last_range.first,
          last_range.second, result, result ? upload_ranges_.size() : size_t(0),
          first_upload.first, first_upload.second, sparse_allocations_before,
          host_gpu_memory_sparse_allocations_,
          raced ? " - raced with a guest invalidation (benign)" : " - NOT explained by a race");
    }
  }
  return result;
}

bool SharedMemory::IsRangeValidAndAllocated(uint32_t start, uint32_t length) {
  // wwe2k14 patch (Session 6, CP2): the lock-free check of gpu_request_ranges_fast. The caller has checked
  // that length != 0 and start + length <= kBufferSize.
  uint32_t page_first = start >> page_size_log2_;
  uint32_t page_last = (start + length - 1) >> page_size_log2_;
  // Whether bits first...last are all set, loading 64-bit blocks with load_block.
  auto all_bits_set = [](uint32_t first, uint32_t last, auto&& load_block) -> bool {
    uint32_t block_first = first >> 6;
    uint32_t block_last = last >> 6;
    for (uint32_t i = block_first; i <= block_last; ++i) {
      uint64_t mask = UINT64_MAX;
      if (i == block_first) {
        mask &= ~((uint64_t(1) << (first & 63)) - 1);
      }
      if (i == block_last && (last & 63) != 63) {
        mask &= (uint64_t(1) << ((last & 63) + 1)) - 1;
      }
      if ((load_block(i) & mask) != mask) {
        return false;
      }
    }
    return true;
  };
  // Sparse host GPU memory: EnsureHostGpuMemoryAllocated (same allocation
  // index math) would have nothing to allocate. The bitmap is only accessed on
  // this thread. Checked here rather than assumed from validity, because a
  // scaled resolve (MarkRangeAsResolved) makes pages valid without requesting
  // them.
  if (host_gpu_memory_sparse_granularity_log2_ != UINT32_MAX) {
    const uint64_t* allocated = host_gpu_memory_sparse_allocated_.data();
    if (!all_bits_set(page_first << page_size_log2_ >> host_gpu_memory_sparse_granularity_log2_,
                      page_last << page_size_log2_ >> host_gpu_memory_sparse_granularity_log2_,
                      [allocated](uint32_t i) { return allocated[i]; })) {
      return false;
    }
  }
  // Valid pages, without the global critical region: other threads only clear
  // bits here (MemoryInvalidationCallback, with atomic stores).
  uint64_t* valid = system_page_flags_valid_.data();
  return all_bits_set(page_first, page_last, [valid](uint32_t i) {
    return std::atomic_ref<uint64_t>(valid[i]).load(std::memory_order_relaxed);
  });
}

bool SharedMemory::IsRangeGpuWritten(uint32_t start, uint32_t length) {
  // wwe2k14 patch (Session 7, EXTENTS-SHADOW).
  if (!length) {
    return false;
  }
  if (start > kBufferSize || (kBufferSize - start) < length) {
    return true;
  }
  uint32_t page_first = start >> page_size_log2_;
  uint32_t page_last = (start + length - 1) >> page_size_log2_;
  uint32_t block_first = page_first >> 6;
  uint32_t block_last = page_last >> 6;
  auto global_lock = global_critical_region_.Acquire();
  for (uint32_t i = block_first; i <= block_last; ++i) {
    uint64_t mask = UINT64_MAX;
    if (i == block_first) {
      mask &= ~((uint64_t(1) << (page_first & 63)) - 1);
    }
    if (i == block_last && (page_last & 63) != 63) {
      mask &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
    }
    if (system_page_flags_valid_and_gpu_written_[i] & mask) {
      return true;
    }
  }
  return false;
}

void SharedMemory::ReportRequestRangesFastStats(uint64_t frame) {
  // wwe2k14 patch (Session 6, CP2).
  RequestRangesFastStats& stats = request_ranges_fast_stats_;
  if (!stats.requests) {
    return;
  }
  if (stats.window_first_frame == UINT64_MAX) {
    stats.window_first_frame = frame;
    return;
  }
  if (frame - stats.window_first_frame < 300) {
    return;
  }
  REXGPU_INFO(
      "[s6][CP2] frames {}-{}: {} requests ({} single-range), {} all valid ({:.1f}%, {} KB), {} "
      "took the locked path (uploaded {} KB); verify: {} checked, {} mismatches ({} raced with a "
      "guest invalidation), {} mismatches in total",
      stats.window_first_frame, frame, stats.requests, stats.single_range_requests, stats.hits,
      double(stats.hits) * 100.0 / double(stats.requests), stats.hit_bytes >> 10,
      stats.requests - stats.hits, (stats.miss_upload_pages << page_size_log2_) >> 10,
      stats.verify_checks, stats.verify_mismatches, stats.verify_mismatches_raced,
      stats.verify_mismatches_total);
  RequestRangesFastStats next_stats;
  next_stats.verify_mismatches_total = stats.verify_mismatches_total;
  next_stats.verify_mismatches_logged = stats.verify_mismatches_logged;
  stats = next_stats;
}

bool SharedMemory::RequestRange(uint32_t start, uint32_t length) {
  std::pair<uint32_t, uint32_t> range(start, length);
  return RequestRanges(&range, 1);
}

std::pair<uint32_t, uint32_t> SharedMemory::MemoryInvalidationCallbackThunk(
    void* context_ptr, uint32_t physical_address_start, uint32_t length, bool exact_range) {
  return reinterpret_cast<SharedMemory*>(context_ptr)
      ->MemoryInvalidationCallback(physical_address_start, length, exact_range);
}

std::pair<uint32_t, uint32_t> SharedMemory::MemoryInvalidationCallback(
    uint32_t physical_address_start, uint32_t length, bool exact_range) {
  if (length == 0 || physical_address_start >= kBufferSize) {
    return std::make_pair(uint32_t(0), UINT32_MAX);
  }
  length = std::min(length, kBufferSize - physical_address_start);
  uint32_t physical_address_last = physical_address_start + (length - 1);

  uint32_t page_first = physical_address_start >> page_size_log2_;
  uint32_t page_last = physical_address_last >> page_size_log2_;
  uint32_t block_first = page_first >> 6;
  uint32_t block_last = page_last >> 6;

  auto global_lock = global_critical_region_.Acquire();
  BumpRepeatEpoch();  // wwe2k14 patch (Session 23, REPEAT): before the valid bits change (FireWatches bumps again)

  const uint32_t written_page_first = page_first, written_page_last = page_last;  // wwe2k14 patch (Session 95, HOTPAGES)
  if (!exact_range) {
    // Check if a somewhat wider range (up to 256 KB with 4 KB pages) can be
    // invalidated - if no GPU-written data nearby that was not intended to be
    // invalidated since it's not in sync with CPU memory and can't be
    // reuploaded. It's a lot cheaper to upload some excess data than to catch
    // access violations - with 4 KB callbacks, 58410824 (being a
    // software-rendered game) runs at 4 FPS on Intel Core i7-3770, with 64 KB,
    // the CPU game code takes 3 ms to run per frame, but with 256 KB, it's
    // 0.7 ms.
    if (page_first & 63) {
      uint64_t gpu_written_start = system_page_flags_valid_and_gpu_written_[block_first];
      gpu_written_start &= (uint64_t(1) << (page_first & 63)) - 1;
      page_first = (page_first & ~uint32_t(63)) + (64 - rex::lzcnt(gpu_written_start));
    }
    if ((page_last & 63) != 63) {
      uint64_t gpu_written_end = system_page_flags_valid_and_gpu_written_[block_last];
      gpu_written_end &= ~((uint64_t(1) << ((page_last & 63) + 1)) - 1);
      page_last =
          (page_last & ~uint32_t(63)) + (std::max(rex::tzcnt(gpu_written_end), uint8_t(1)) - 1);
    }
  }

  uint32_t ram_dropped = 0;  // wwe2k14 patch (Session 49, RAM TIES)
  for (uint32_t i = block_first; i <= block_last; ++i) {
    uint64_t invalidate_bits = UINT64_MAX;
    if (i == block_first) {
      invalidate_bits &= ~((uint64_t(1) << (page_first & 63)) - 1);
    }
    if (i == block_last && (page_last & 63) != 63) {
      invalidate_bits &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
    }
    // wwe2k14 patch (Session 6, CP2): an atomic store of the same value, since
    // gpu_request_ranges_fast reads these blocks on the command processor
    // thread without the global critical region (all writers hold it, so a
    // separate load and store is enough).
    std::atomic_ref<uint64_t> valid_block(system_page_flags_valid_[i]);
    valid_block.store(valid_block.load(std::memory_order_relaxed) & ~invalidate_bits,
                      std::memory_order_relaxed);
    system_page_flags_valid_and_gpu_written_[i] &= ~invalidate_bits;
    // wwe2k14 patch (Session 49, RAM TIES): a CPU write into (or next to) resolve output.
    if (!resolve_written_.empty() && (resolve_written_[i] & invalidate_bits)) {
      ram_dropped += rex::bit_count(resolve_written_[i] & invalidate_bits);
      resolve_written_[i] &= ~invalidate_bits;
    }
    // wwe2k14 patch (Session 7b, UPSKIP): invalidated by a watch (as with clear_memory_page_state=false too).
    system_page_flags_watch_fired_[i] |= invalidate_bits;
    if (g_hot.enabled) {  // wwe2k14 patch (Session 95, HOTPAGES)
      // Only the pages the guest wrote, not the widened range (a hot page must not pull its 64-page block hot).
      uint64_t written_bits = UINT64_MAX;
      if (i == (written_page_first >> 6)) written_bits &= ~((uint64_t(1) << (written_page_first & 63)) - 1);
      if (i == (written_page_last >> 6) && (written_page_last & 63) != 63) {
        written_bits &= (uint64_t(1) << ((written_page_last & 63) + 1)) - 1;
      }
      g_hot.fired_frame[i] |= invalidate_bits & written_bits;
    }
  }
  // wwe2k14 patch (Session 6, CP2): for gpu_request_ranges_fast_verify, after
  // the bits (release), under the global critical region.
  invalidation_count_.store(invalidation_count_.load(std::memory_order_relaxed) + 1,
                            std::memory_order_release);
  if (ram_dropped) {  // wwe2k14 patch (Session 49, RAM TIES)
    ++ram_stats_.cpu_invalidations;
    ram_stats_.cpu_pages += ram_dropped;
  }

  FireWatches(page_first, page_last, false);

  return std::make_pair(page_first << page_size_log2_, (page_last - page_first + 1)
                                                           << page_size_log2_);
}

bool SharedMemory::EnsureHostGpuMemoryAllocated(uint32_t start, uint32_t length) {
  if (host_gpu_memory_sparse_granularity_log2_ == UINT32_MAX) {
    return true;
  }
  if (!length) {
    return true;
  }
  if (start > kBufferSize || (kBufferSize - start) < length) {
    return false;
  }
  uint32_t page_first = start >> page_size_log2_;
  uint32_t page_last = (start + length - 1) >> page_size_log2_;
  uint32_t allocation_first =
      page_first << page_size_log2_ >> host_gpu_memory_sparse_granularity_log2_;
  uint32_t allocation_last =
      page_last << page_size_log2_ >> host_gpu_memory_sparse_granularity_log2_;
  while (true) {
    std::pair<size_t, size_t> allocation_range =
        rex::bit::GetNextRangeUnset(host_gpu_memory_sparse_allocated_.data(), allocation_first,
                                    allocation_last - allocation_first + 1);
    if (!allocation_range.second) {
      break;
    }
    if (!AllocateSparseHostGpuMemoryRange(uint32_t(allocation_range.first),
                                          uint32_t(allocation_range.second))) {
      return false;
    }
    rex::bit::SetRange(host_gpu_memory_sparse_allocated_.data(), allocation_range.first,
                       allocation_range.second);
    ++host_gpu_memory_sparse_allocations_;
    COUNT_profile_set("gpu/shared_memory/host_gpu_memory_sparse_allocations",
                      host_gpu_memory_sparse_allocations_);
    host_gpu_memory_sparse_used_bytes_ += uint32_t(allocation_range.second)
                                          << host_gpu_memory_sparse_granularity_log2_;
    COUNT_profile_set("gpu/shared_memory/host_gpu_memory_sparse_used_mb",
                      (host_gpu_memory_sparse_used_bytes_ + ((1 << 20) - 1)) >> 20);
    allocation_first = uint32_t(allocation_range.first + allocation_range.second);
  }
  return true;
}

}  // namespace rex::graphics
