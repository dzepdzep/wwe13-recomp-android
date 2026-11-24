#pragma once
/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <atomic>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

#include <rex/memory.h>
#include <rex/thread/mutex.h>

namespace rex::graphics {

// Manages memory for unconverted textures, resolve targets, vertex and index
// buffers that can be accessed from shaders with Xenon physical addresses, with
// system page size granularity.
class SharedMemory {
 public:
  static constexpr uint32_t kBufferSizeLog2 = 29;
  static constexpr uint32_t kBufferSize = 1 << kBufferSizeLog2;

  virtual ~SharedMemory();
  // Call in the implementation-specific ClearCache.
  virtual void ClearCache();
  // wwe2k14 patch (Session 50, review T1): frame_end = the command processor's frame-end clear, the only caller the
  // geometry-page trust applies to (a lost-upload repair or a cache clear always un-validates everything).
  void SetSystemPageBlocksValidWithGpuDataWritten(bool frame_end = false);
  void InvalidateAllPages();

  // wwe2k14 patch (Session 23, REPEAT): changes whenever any page may have stopped being valid or any watch fired
  // (FireWatches - CPU writes through MemoryInvalidationCallback, GPU writes, cache clears -, InvalidateAllPages,
  // SetSystemPageBlocksValidWithGpuDataWritten, a failed skip-checked upload). Equal values at two moments mean
  // no shared memory page and no texture watch was invalidated in between. Readable from any thread.
  uint64_t repeat_epoch() const { return repeat_epoch_.load(std::memory_order_acquire); }

  typedef void (*GlobalWatchCallback)(const std::unique_lock<std::recursive_mutex>& global_lock,
                                      void* context, uint32_t address_first, uint32_t address_last,
                                      bool invalidated_by_gpu);
  typedef void* GlobalWatchHandle;
  // Registers a callback invoked when something is invalidated in the GPU
  // memory copy by the CPU or (if triggered explicitly - such as by a resolve)
  // by the GPU. It will be fired for writes to pages previously requested, but
  // may also be fired regardless of whether it was used by GPU emulation - for
  // example, if the game changes protection level of a memory range containing
  // the watched range.
  //
  // The callback is called within the global critical region.
  GlobalWatchHandle RegisterGlobalWatch(GlobalWatchCallback callback, void* callback_context);
  void UnregisterGlobalWatch(GlobalWatchHandle handle);
  typedef void (*WatchCallback)(const std::unique_lock<std::recursive_mutex>& global_lock,
                                void* context, void* data, uint64_t argument,
                                bool invalidated_by_gpu);
  typedef void* WatchHandle;
  // Registers a callback invoked when the specified memory range is invalidated
  // in the GPU memory copy by the CPU or (if triggered explicitly - such as by
  // a resolve) by the GPU. It will be fired for writes to pages previously
  // requested, but may also be fired regardless of whether it was used by GPU
  // emulation - for example, if the game changes protection level of a memory
  // range containing the watched range.
  //
  // Generally the context is the subsystem pointer (for example, the texture
  // cache), the data is the object (such as a texture), and the argument is
  // additional subsystem/object-specific data (such as whether the range
  // belongs to the base mip level or to the rest of the mips).
  //
  // Called with the global critical region locked. Do NOT watch or unwatch
  // ranges from within it! The watch for the callback is cancelled after the
  // callback - the handle becomes invalid.
  WatchHandle WatchMemoryRange(uint32_t start, uint32_t length, WatchCallback callback,
                               void* callback_context, void* callback_data,
                               uint64_t callback_argument);
  // Unregisters previously registered watched memory range.
  void UnwatchMemoryRange(WatchHandle handle);
  // wwe2k14 patch (Session 95, r6tex, gpu_tex_planes_unwatched): pages whose changes the engine announces: no watch; hashed at every request (the first request of a frame unhashed under gpu_tex_planes_nohash). Global critical region held.
  void DeclareEnginePages(uint32_t start, uint32_t length);

  // Checks if the range has been updated, uploads new data if needed and
  // ensures the host GPU memory backing the range are resident. Returns true if
  // the range has been fully updated and is usable.
  bool RequestRanges(const std::pair<uint32_t, uint32_t>* ranges, size_t count);
  // wwe2k14 patch (Session 21, HOIST): called after every RequestRanges (whatever it found or uploaded, also on
  // failure) with the ranges it was given: the GPU work recorded next may read them. Command processor thread.
  virtual void OnRangesRequested(const std::pair<uint32_t, uint32_t>* ranges, size_t count) {}
  bool RequestRange(uint32_t start, uint32_t length);

  // wwe2k14 patch (Session 49, RAM TIES: gpu_ram_stats): a census of the pages resolves wrote into (until a CPU write
  // invalidates them, an upload replaces them, or another GPU write - memexport - does) and of who requests them
  // afterwards: the caller of the next RequestRanges names itself with a RamReaderScope. Command processor thread
  // (the invalidation side under the global critical region). Nothing is recorded without gpu_ram_stats.
  enum class RamReader : uint32_t { kOther, kVertex, kIndex, kTexture, kMemexport, kResolveDest, kCount };
  struct RamReaderScope {
    SharedMemory& memory;
    RamReader previous;
    RamReaderScope(SharedMemory& m, RamReader reader) : memory(m), previous(m.ram_reader_) { m.ram_reader_ = reader; }
    ~RamReaderScope() { memory.ram_reader_ = previous; }
  };
  // wwe2k14 patch (Session 50, LEAN: gpu_native_trust_geometry_pages): while it or its _check is on, each request is
  // classified by its RamReaderScope: pages only vertex / index fetches have requested (geometry-only) and pages
  // anything else requested (textures, memexport, resolve destinations, unnamed callers). With the trust setting the
  // frame-end clear (SetSystemPageBlocksValidWithGpuDataWritten) keeps the valid geometry-only pages valid - their
  // write watches stay armed, so a CPU write still invalidates them - instead of every frame requesting, hashing and
  // comparing them again. _check: the clear as before, with the stale count ([s7][UPSKIP]: uploaded pages that
  // changed while only the clear had invalidated them) split for the geometry-only pages - the evidence that their
  // watches hold. Both latched at each frame end. Command processor thread (the clear under the global lock).
  bool geometry_tracking() const { return geometry_tracking_; }
  uint64_t geometry_kept_pages() const { return geometry_kept_pages_; }  // kept valid by the last frame-end clear
  void NoteResolveWritten(uint32_t start, uint32_t length);
  void ReportRamStats(uint64_t frame);
  // Counts a request (RequestRanges calls it; also callers that skip RequestRanges for a range they know is valid).
  void RamCountRequest(const std::pair<uint32_t, uint32_t>* ranges, size_t count);
  bool ram_stats_on() const { return !resolve_written_.empty(); }

  // wwe2k14 patch (Session 49, NATIVE TRUTH skip write-back): pages whose GPU copy is behind a texture's (a truth
  // resolve filled its texture and did not write guest memory). Any request overlapping them (RequestRanges, or
  // CheckSkipped from a caller that reads without one) first calls the materialise callback with that range, which
  // writes the pages from the texture and clears them (SetRangeSkipped false). Command processor thread only.
  typedef void (*MaterializeCallback)(void* context, uint32_t start, uint32_t length, RamReader reader);
  void SetMaterializeCallback(MaterializeCallback callback, void* context) {
    materialize_callback_ = callback;
    materialize_context_ = context;
  }
  void SetRangeSkipped(uint32_t start, uint32_t length, bool skipped);
  bool AnySkipped() const { return skipped_pages_ != 0; }
  void CheckSkipped(uint32_t start, uint32_t length) {
    if (skipped_pages_) CheckSkippedRanges(std::pair<uint32_t, uint32_t>(start, length));
  }

  // wwe2k14 patch (Session 6, COUNTERS): RequestRanges statistics for the per-frame timeline counters, taken
  // (and reset) by the command processor at each frame boundary. RequestRanges runs only on the command
  // processor thread (it already relies on that for upload_ranges_), so plain counters are enough.
  struct RequestStats {
    uint32_t calls = 0;          // RequestRanges calls (RequestRange included)
    uint32_t upload_calls = 0;   // of those, the ones that called UploadRanges
    uint64_t upload_bytes = 0;   // bytes (whole pages) passed to UploadRanges
  };
  RequestStats TakeRequestStats() {
    RequestStats stats = request_stats_;
    request_stats_ = RequestStats();
    return stats;
  }
  // wwe2k14 patch (Session 6, CP2): logs the gpu_request_ranges_fast counters
  // (hit rate, verify mismatches) about every 300 frames. Call from the command
  // processor thread with the current frame number, for example once per
  // submission. Returns right away if nothing has been counted.
  void ReportRequestRangesFastStats(uint64_t frame);

  // wwe2k14 patch (Session 6, RESID): whether RequestRange(start, length) would find nothing to allocate or
  // upload right now (every page valid, host GPU memory allocated), checked without the global critical
  // region (the gpu_request_ranges_fast check). An empty range is up to date, a range outside the buffer is
  // not (RequestRange fails for it). Read-only. Command processor thread only, like RequestRange.
  bool IsRangeUpToDate(uint32_t start, uint32_t length) {
    if (!length) {
      return true;
    }
    if (start > kBufferSize || (kBufferSize - start) < length) {
      return false;
    }
    return IsRangeValidAndAllocated(start, length);
  }

  // Marks the range and, if not exact_range, potentially its surroundings
  // (to up to the first GPU-written page, as an access violation exception
  // count optimization) as modified by the CPU, also invalidating GPU-written
  // pages directly in the range.
  std::pair<uint32_t, uint32_t> MemoryInvalidationCallback(uint32_t physical_address_start,
                                                           uint32_t length, bool exact_range);

  // Marks the range as containing GPU-generated data (such as resolves),
  // triggering modification callbacks, making it valid (so pages are not
  // copied from the main memory until they're modified by the CPU) and
  // protecting it. Before writing anything from the GPU side, RequestRange must
  // be called, to make sure, if the GPU writes don't overwrite *everything* in
  // the pages they touch, the CPU data is properly loaded to the unmodified
  // regions in those pages.
  void RangeWrittenByGpu(uint32_t start, uint32_t length);
  // wwe2k14 patch (Session 82, the owned frame): the range marked valid and GPU-written (watched, no watch fired, no
  // upload) - a copy left out of it: the CPU-write widening (MemoryInvalidationCallback) then stops at these pages.
  void MarkRangeGpuWrittenQuiet(uint32_t start, uint32_t length);

  // wwe2k14 patch (Session 7, EXTENTS-SHADOW): whether any page of the range holds data the GPU wrote (a resolve
  // or memexport) that guest memory may not have: valid and GPU-written pages (RangeWrittenByGpu, until the CPU
  // writes the page again). A range outside the buffer counts as written. Read-only, takes the global critical
  // region. For CPU work that reads guest memory standing in for the host copy (screen extents estimates).
  bool IsRangeGpuWritten(uint32_t start, uint32_t length);

  // wwe2k14 patch (Session 7b, UPSKIP): gpu_upload_skip_unchanged (see shared_memory.cpp). Per-frame counters for
  // the cpf_up / cpf_up2 timeline events, taken (and reset) by the command processor at each frame boundary; this
  // also adds them to the [s7][UPSKIP] summary logged about every 300 frames (only when hashing was on) and
  // advances the verify round robin. Command processor thread only.
  struct UploadSkipStats {
    uint32_t upload_calls = 0;    // UploadRanges calls (each records one or more copies)
    uint32_t pages_uploaded = 0;  // pages passed to UploadRanges
    uint32_t pages_skipped = 0;   // pages made valid without a copy (bytes equal to the last upload)
    uint32_t pages_c1_stale = 0;  // uploaded pages whose bytes changed while only the frame-end clear
                                  // invalidated them (what clear_memory_page_state=false would show stale)
    uint32_t pages_c1_stale_geometry = 0;  // wwe2k14 (Session 50): of those, geometry-only pages
    uint64_t bytes_hashed = 0;    // bytes hashed (skip checks and hashed uploads)
    uint64_t hash_ns = 0;         // time in the hashing loops
  };
  UploadSkipStats TakeUploadSkipStats(uint64_t frame);

 protected:
  void ResetUploadHashNextPage() { upload_hash_next_page_ = 0; }  // wwe2k14 patch (Session 94, DIRECT)
  SharedMemory(memory::Memory& memory);
  // Call in implementation-specific initialization.
  void InitializeCommon();
  void InitializeSparseHostGpuMemory(uint32_t granularity_log2);
  // Call last in implementation-specific shutdown, also callable from the
  // destructor.
  void ShutdownCommon();

  // Sparse allocations are 4 MB, so not too many of them are allocated, but
  // also not to waste too much memory for padding (with 16 MB there's too
  // much).
  static constexpr uint32_t kHostGpuMemoryOptimalSparseAllocationLog2 = 22;
  static_assert(kHostGpuMemoryOptimalSparseAllocationLog2 <= kBufferSizeLog2);

  memory::Memory& memory() const { return memory_; }

  uint32_t page_size_log2() const { return page_size_log2_; }

  uint32_t host_gpu_memory_sparse_granularity_log2() const {
    return host_gpu_memory_sparse_granularity_log2_;
  }

  // Allocations in the host buffer are aligned the same way as in the guest
  // physical memory (for instance, if an allocation is 64 KB, it can represent
  // 0-64 KB, 64-128 KB, 128-192 KB in the guest memory, and so on, but not
  // something like 16-80 KB. This is assumed by the rules for texture data
  // access in the texture cache.
  virtual bool AllocateSparseHostGpuMemoryRange(uint32_t offset_allocations,
                                                uint32_t length_allocations);

  // Mark the memory range as updated and protect it.
  void MakeRangeValid(uint32_t start, uint32_t length, bool written_by_gpu);

  // Uploads a range of host pages - only called if host GPU sparse memory
  // allocation succeeded if needed. While uploading, MakeRangeValid must be
  // called for each successfully uploaded range as early as possible, before
  // the memcpy, to make sure invalidation that happened during the CPU -> GPU
  // memcpy isn't missed (upload_page_ranges is in pages because of this -
  // MakeRangeValid has page granularity). upload_page_ranges are sorted in
  // ascending address order, so front and back can be used to determine the
  // overall bounds of pages to be uploaded.
  // wwe2k14 patch (Session 21, HOIST): RequestRanges before the OnRangesRequested call.
  bool RequestRangesInner(const std::pair<uint32_t, uint32_t>* ranges, size_t count);

  virtual bool UploadRanges(
      const std::vector<std::pair<uint32_t, uint32_t>>& upload_page_ranges) = 0;

  // wwe2k14 patch (Session 7b, UPSKIP): gpu_upload_skip_unchanged.
  // Latched for each RequestRanges call: whether UploadRanges must copy each chunk with CopyAndHashUploadPages.
  bool upload_hash_tracking() const { return upload_hash_tracking_; }
  // In place of the UploadRanges memcpy, after its MakeRangeValid: copies size bytes of guest memory from page
  // page_first on into dest through a one-page snapshot, hashing each snapshot, so the recorded hash is of
  // exactly the bytes that go to the GPU buffer.
  void CopyAndHashUploadPages(uint8_t* dest, uint32_t page_first, uint32_t size);
  // A GPU write other than an upload (resolve, memory export) to the range: its pages no longer hold the bytes
  // last uploaded to them, so they can't be skipped until uploaded again. Command processor thread.
  void InvalidateUploadHashes(uint32_t start, uint32_t length);
  // gpu_upload_skip_unchanged_verify: record a copy of the page from the host GPU buffer at this point of the
  // command stream, to be compared with bytes (the guest bytes that were hashed, page-sized) once the submission
  // has completed. Returns whether it was recorded.
  virtual bool RecordUploadSkipVerify(uint32_t page, const uint8_t* bytes) { return false; }
  struct UploadSkipVerifyStats {
    uint64_t recorded = 0;    // readback copies recorded
    uint64_t dropped = 0;     // not recorded (readback ring full, or no readback buffer)
    uint64_t checked = 0;     // compared after the submission completed
    uint64_t mismatches = 0;  // GPU bytes != hashed guest bytes
  };
  UploadSkipVerifyStats& upload_skip_verify_stats() { return upload_skip_verify_stats_; }
  uint64_t upload_skip_verify_mismatches_logged_ = 0;
  // A count that grows whenever copies already recorded (UploadRanges) may never reach the GPU buffer (Vulkan:
  // an async submission whose commands were dropped - SUBMITFIX). Read on the command processor thread at each
  // locked request while hashing is on; when it has changed, every upload hash is dropped and the pages valid
  // only through uploads are invalidated (see LatchUploadSkipSettings). Any thread may increase it.
  virtual uint64_t GetLostUploadCount() const { return 0; }

 private:
  memory::Memory& memory_;

  // Log2 of invalidation granularity (the system page size, but the dependency
  // on it is not hard - the access callback takes a range as an argument, and
  // touched pages of the buffer of this size will be invalidated).
  uint32_t page_size_log2_;

  bool EnsureHostGpuMemoryAllocated(uint32_t start, uint32_t length);
  uint32_t host_gpu_memory_sparse_granularity_log2_ = UINT32_MAX;
  std::vector<uint64_t> host_gpu_memory_sparse_allocated_;
  uint32_t host_gpu_memory_sparse_allocations_ = 0;
  uint32_t host_gpu_memory_sparse_used_bytes_ = 0;

  void* memory_invalidation_callback_handle_ = nullptr;
  void* memory_data_provider_handle_ = nullptr;

  // Ranges that need to be uploaded, generated by GetRangesToUpload (a
  // persistently allocated vector).
  std::vector<std::pair<uint32_t, uint32_t>> upload_ranges_;
  RequestStats request_stats_;  // wwe2k14 patch (Session 6, COUNTERS)

  // wwe2k14 patch (Session 6, CP2): RequestRanges split into the merge (shared
  // by the original and the gpu_request_ranges_fast paths) and the locked scan
  // + upload of already sorted and merged ranges, plus the lock-free "all pages
  // valid and host memory allocated" check of the fast path.
  static size_t SortAndMergeRanges(std::pair<uint32_t, uint32_t>* ranges, size_t count);
  bool RequestMergedRanges(const std::pair<uint32_t, uint32_t>* merged_ranges,
                           size_t merged_count, size_t request_count);
  bool RequestRangesFast(const std::pair<uint32_t, uint32_t>* ranges, size_t count);
  bool IsRangeValidAndAllocated(uint32_t start, uint32_t length);
  // Like upload_ranges_, only used on the command processor thread (every
  // RequestRanges caller is on it), so it can be a persistent member.
  std::vector<std::pair<uint32_t, uint32_t>> request_ranges_scratch_;
  struct RequestRangesFastStats {
    uint64_t window_first_frame = UINT64_MAX;
    uint64_t requests = 0;
    uint64_t single_range_requests = 0;
    uint64_t hits = 0;
    uint64_t hit_bytes = 0;
    uint64_t miss_upload_pages = 0;
    uint64_t verify_checks = 0;
    uint64_t verify_mismatches = 0;
    uint64_t verify_mismatches_raced = 0;
    uint64_t verify_mismatches_total = 0;
    uint64_t verify_mismatches_logged = 0;
  };
  // Command processor thread only.
  RequestRangesFastStats request_ranges_fast_stats_;
  // wwe2k14 patch (Session 49, RAM TIES): see NoteResolveWritten. One bit per page; empty until gpu_ram_stats is on.
  RamReader ram_reader_ = RamReader::kOther;
  std::vector<uint64_t> resolve_written_;
  struct RamStats {
    uint64_t window_first_frame = UINT64_MAX;
    uint64_t noted_calls = 0, noted_pages = 0;
    uint64_t reads[size_t(RamReader::kCount)] = {};       // requests overlapping resolve-written pages
    uint64_t read_pages[size_t(RamReader::kCount)] = {};  // their overlapping pages
    uint64_t requests[size_t(RamReader::kCount)] = {};    // all requests by caller
    uint64_t cpu_invalidations = 0, cpu_pages = 0;        // CPU writes (watch faults) that dropped resolve pages
    uint64_t upload_pages = 0, gpu_overwrite_pages = 0;   // dropped by an upload / a non-resolve GPU write
    uint32_t logged = 0;
  } ram_stats_;
  uint32_t RamOverlapPages(uint32_t start, uint32_t length) const;
  void RamDropPages(uint32_t start, uint32_t length, uint64_t& counter);
  // wwe2k14 patch (Session 49, NATIVE TRUTH skip write-back): see SetRangeSkipped. One bit per page.
  std::vector<uint64_t> skipped_;
  uint64_t skipped_pages_ = 0;
  MaterializeCallback materialize_callback_ = nullptr;
  void* materialize_context_ = nullptr;
  bool materializing_ = false;
  void CheckSkippedRanges(std::pair<uint32_t, uint32_t> range);
  // Number of MemoryInvalidationCallback calls (written under the global
  // critical region, read without it by the verify path to tell a real
  // mismatch from a race with a guest invalidation).
  std::atomic<uint64_t> invalidation_count_{0};
  // wwe2k14 patch (Session 23, REPEAT): see repeat_epoch(). Bumped before the change is made visible (under the
  // global critical region where the change is).
  std::atomic<uint64_t> repeat_epoch_{0};
  void BumpRepeatEpoch() { repeat_epoch_.fetch_add(1, std::memory_order_acq_rel); }

  // wwe2k14 patch (Session 7b, UPSKIP): gpu_upload_skip_unchanged state. Command processor thread only (every
  // RequestRanges, RangeWrittenByGpu and UploadRanges caller is on it), except where noted.
  void LatchUploadSkipSettings();
  // Copies ~watch-fired into upload_clear_only_ for the pages of upload_ranges_. Global critical region held.
  void CaptureUploadClearOnly();
  // Hashes the hashed pages of upload_ranges_ and fills upload_ranges_after_skip_ with what still needs a copy.
  void SkipUnchangedUploadPages();
  // wwe2k14 patch (Session 19, PARHASH): gpu_upload_skip_parallel_hash. The same result as SkipUnchangedUploadPages
  // (no verify): every hashed run is made valid first (as before, each page before its hash), the hashes of the
  // request are computed on worker threads plus this thread when there are at least min_pages of them, then the
  // pages left are appended in the original order.
  void SkipUnchangedUploadPagesParallel(uint32_t workers, uint32_t min_pages);
  struct ParhashSegment {
    uint32_t first;
    uint32_t count;
    bool hashed;
  };
  std::vector<ParhashSegment> parhash_segments_;
  std::vector<uint32_t> parhash_pages_;
  std::vector<uint8_t> parhash_match_;
  struct ParhashWindow {
    uint64_t requests = 0;           // requests that went through the parallel scan
    uint64_t parallel_requests = 0;  // of those, hashed on the workers (>= min_pages)
    uint64_t pages = 0;
    uint64_t parallel_pages = 0;
    uint64_t hash_ns = 0;            // wall time of the hashing (all requests of the scan)
  };
  ParhashWindow parhash_window_;
  // UploadRanges failed (no upload buffer space): makes the pages of upload_ranges_after_skip_ not valid again,
  // since SkipUnchangedUploadPages made the hashed ones among them valid before UploadRanges could copy them.
  void InvalidateFailedUploadPages();
  // CopyAndHashUploadPages: the page after the last one hashed in the current UploadRanges call. upload_ranges_
  // can list a page twice (the boundary page of two merged ranges in the same page, not merged since their
  // bytes are disjoint); pages come in ascending order otherwise, so a page below this is a repeat.
  uint32_t upload_hash_next_page_ = 0;
  uint64_t upload_hashes_lost_seen_ = 0;  // GetLostUploadCount at the last request with hashing on
  bool upload_hash_tracking_ = false;           // hash uploads (skip or stats on), latched per request
  bool upload_hash_tracking_previous_ = false;  // at the previous request (off -> on drops every hash)
  bool upload_skip_ = false;                    // gpu_upload_skip_unchanged, latched per request
  bool upload_skip_verify_ = false;             // gpu_upload_skip_unchanged_verify (with skip), per request
  // Per page: XXH3 of the bytes last copied to the page of the GPU buffer, valid while the page's bit in
  // upload_hash_valid_ is set (cleared by GPU writes other than uploads). Allocated when first turned on.
  std::vector<uint64_t> upload_hashes_;
  std::vector<uint64_t> upload_hash_valid_;
  // Per page of upload_ranges_: no watch fired since the page was last made valid (captured in the scan).
  std::vector<uint64_t> upload_clear_only_;
  std::vector<uint8_t> upload_hash_scratch_;  // one page
  std::vector<std::pair<uint32_t, uint32_t>> upload_ranges_after_skip_;
  UploadSkipStats upload_skip_frame_stats_;
  UploadSkipVerifyStats upload_skip_verify_stats_;
  // Verify round robin: would-be-skipped pages seen this frame, readbacks recorded this frame, and the index of
  // the first would-be-skipped page of the frame to record (moves on by the per-frame cap every frame).
  uint32_t upload_skip_verify_candidates_ = 0;
  uint32_t upload_skip_verify_recorded_frame_ = 0;
  uint32_t upload_skip_verify_offset_ = 0;
  struct UploadSkipWindow {
    uint64_t window_first_frame = UINT64_MAX;
    uint64_t upload_calls = 0;
    uint64_t pages_uploaded = 0;
    uint64_t pages_skipped = 0;
    uint64_t pages_matched_verify = 0;  // would-be-skipped pages uploaded anyway (verify)
    uint64_t pages_c1_stale = 0;
    uint64_t pages_hash_checked = 0;    // invalid pages with a hash, checked against guest memory
    uint64_t skip_make_valid_calls = 0;
    // wwe2k14 patch (Session 87, step 4, diagnostic): the skip scan's pages by whether a watch fired since they were
    // last valid (fired) or only the frame-end clear invalidated them (clear), and whether their bytes still equal the
    // last upload's (same) or not (changed); pages with no hash yet.
    uint64_t pages_fired_same = 0, pages_fired_changed = 0, pages_clear_same = 0, pages_clear_changed = 0;
    uint64_t pages_no_hash = 0;
    uint64_t make_valid_calls = 0;    // MakeRangeValid calls that reached the watch (Session 87)
    uint64_t make_valid_skip_wrong = 0;  // verify: pages a skipped call would have newly watched
    uint64_t make_valid_unfired = 0;  // ... of them over pages none of which had a watch fire since they were last valid
    uint64_t bytes_hashed = 0;
    uint64_t hash_ns = 0;
    uint64_t c1_stale_total = 0;
    uint64_t pages_c1_stale_geometry = 0;  // wwe2k14 (Session 50)
    uint64_t c1_stale_geometry_total = 0;
    uint64_t geometry_kept_pages = 0;      // kept valid by the frame-end clears of the window
    uint32_t c1_examples[4] = {};
    uint32_t c1_example_count = 0;
    UploadSkipVerifyStats verify_at_start;
  };
  UploadSkipWindow upload_skip_window_;

  // Mutex between the guest memory subsystem and the command processor, to be
  // locked when checking or updating validity of pages/ranges and when firing
  // watches.
  rex::thread::global_critical_region global_critical_region_;

  // ***************************************************************************
  // Things below should be fully protected by global_critical_region.
  // ***************************************************************************

  // Pages whose contents in the buffer are in sync with guest memory.
  std::vector<uint64_t> system_page_flags_valid_;
  // Subset of valid pages containing data written by the GPU.
  std::vector<uint64_t> system_page_flags_valid_and_gpu_written_;
  // wwe2k14 patch (Session 7b, UPSKIP): pages a guest write watch (MemoryInvalidationCallback), InvalidateAllPages
  // or ClearCache invalidated since they were last made valid (MakeRangeValid clears them). The frame-end clear
  // (clear_memory_page_state) does not set them: a page that is invalid with its bit clear would still be valid
  // with clear_memory_page_state=false. Maintained in every mode (an OR / AND per 64-page block).
  std::vector<uint64_t> system_page_flags_watch_fired_;
  uint32_t num_system_page_flags_ = 0;
  // wwe2k14 patch (Session 50, LEAN): see geometry_tracking(). Requested by vertex / index fetches / by anything else.
  std::vector<uint64_t> geometry_pages_;
  std::vector<uint64_t> nongeometry_pages_;
  bool geometry_tracking_ = false;
  bool geometry_trust_ = false;
  uint64_t geometry_kept_pages_ = 0;
  uint64_t geometry_lost_seen_ = 0;      // GetLostUploadCount() at the last frame-end clear (review T1)
  uint32_t geometry_trust_frames_ = 0;   // frame-end clears since the last full one (review T2)
  void NoteGeometryRanges(const std::pair<uint32_t, uint32_t>* ranges, size_t count);
  void UpdateHotPagesAtFrameEnd();  // wwe2k14 patch (Session 95, HOTPAGES)
  bool IsGeometryOnlyPage(uint32_t page) const {
    return geometry_tracking_ && page < geometry_pages_.size() * 64 &&
           ((geometry_pages_[page >> 6] & ~nongeometry_pages_[page >> 6]) >> (page & 63) & 1);
  }

  static std::pair<uint32_t, uint32_t> MemoryInvalidationCallbackThunk(
      void* context_ptr, uint32_t physical_address_start, uint32_t length, bool exact_range);

  struct GlobalWatch {
    GlobalWatchCallback callback;
    void* callback_context;
  };
  std::vector<GlobalWatch*> global_watches_;
  struct WatchNode;
  // Watched range placed by other GPU subsystems.
  struct WatchRange {
    union {
      struct {
        WatchCallback callback;
        void* callback_context;
        void* callback_data;
        uint64_t callback_argument;
        WatchNode* node_first;
        uint32_t page_first;
        uint32_t page_last;
      };
      WatchRange* next_free;
    };
  };
  // Node for faster checking of watches when pages have been written to - all
  // 512 MB are split into smaller equally sized buckets, and then ranges are
  // linearly checked.
  struct WatchNode {
    union {
      struct {
        WatchRange* range;
        // Link to another node of this watched range in the next bucket.
        WatchNode* range_node_next;
        // Links to nodes belonging to other watched ranges in the bucket.
        WatchNode* bucket_node_previous;
        WatchNode* bucket_node_next;
      };
      WatchNode* next_free;
    };
  };
  static constexpr uint32_t kWatchBucketSizeLog2 = 22;
  static constexpr uint32_t kWatchBucketCount = 1 << (kBufferSizeLog2 - kWatchBucketSizeLog2);
  WatchNode* watch_buckets_[kWatchBucketCount] = {};
  // Allocation from pools - taking new WatchRanges and WatchNodes from the free
  // list, and if there are none, creating a pool if the current one is fully
  // used, and linearly allocating from the current pool.
  static constexpr uint32_t kWatchRangePoolSize = 8192;
  static constexpr uint32_t kWatchNodePoolSize = 8192;
  std::vector<WatchRange*> watch_range_pools_;
  std::vector<WatchNode*> watch_node_pools_;
  uint32_t watch_range_current_pool_allocated_ = 0;
  uint32_t watch_node_current_pool_allocated_ = 0;
  WatchRange* watch_range_first_free_ = nullptr;
  WatchNode* watch_node_first_free_ = nullptr;
  // Triggers the watches (global and per-range), removing triggered range
  // watches.
  void FireWatches(uint32_t page_first, uint32_t page_last, bool invalidated_by_gpu);
  // Unlinks and frees the range and its nodes. Call this in the global critical
  // region.
  void UnlinkWatchRange(WatchRange* range);
};

}  // namespace rex::graphics
