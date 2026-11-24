// wwe2k14 patch (Session 87, D3D layer stage 3; docs\native\d3d\S3_CONTRACT.md in the game repository): the record's
// register image - the state of a record-fed draw lives here, not in the emulated register file.
//
// Mechanism (include\rex\graphics\d3d_feed.h has the contract between the base processor and this file):
//  - A RUN item of the D3D token (first register, n values) is stored into g_f.image in one copy per class piece. The
//    notifications the register writes would have made are made from the record's blocks (Notify): the repeat / lean
//    dirty classes (RepeatNoteRegisters, whole piece, when a value changed), the constant buffers marked out of date
//    (floats only when a used constant is inside the piece, as WriteRegistersFromMem does), the texture cache's fetch
//    constants written and the vertex buffers' residency. The register file is NOT written: the ranges are marked pending.
//  - A draw (BeginDraw) points the derivations' register source at the image: VulkanCommandProcessor::DrawRegs() (system
//    constants, the constants upload, the repeat path, cover / bounds rectangles), the texture cache's register_file(),
//    the pipeline cache's key reads, and d3d_direct.cpp feeds PrimitiveProcessor::Process from the draw item.
//  - Flush copies the pending ranges into the register file (no notifications: they were made at the run). It runs before
//    any packet that is not the token (the check in CommandProcessor::ExecutePacket), before a draw is handed back to
//    the stock path, and before a run the image does not model.
//  - After a foreign packet the image is stale (the packet may have written the register file): the next run or draw
//    re-copies the ranges the derivations read (render state 0x2000-0x23FF, constants and fetch constants 0x4000-0x492F).
//  - gpu_d3d_own_input_verify: the runs go the stock way AND into the image; after each draw the image is compared with the
//    register file over those ranges (a difference is logged, counted, and the image re-copied). The draw is the stock
//    one: the verify answers "would the image have fed the same values", with no picture at risk.
//
// What is NOT modelled (the stock path takes the run, after a flush): any register outside the ranges above except
// 0x5000-0x5002, 0x5C8, 0xE00, 0xE40 (the record's trailing flush-fetch writes and the library's misc flush).
#include "d3d_feed.h"
#include <rex/graphics/d3d_tex.h>

#include <rex/graphics/vulkan/native/native_renderer.h>

#include <rex/graphics/d3d_feed.h>
#include <rex/graphics/d3d_noring.h>  // (Session 92, M4f-C) the Registers item's bin op numbers
#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/registers.h>
#include <rex/cvar.h>
#include <rex/graphics/free_counters.h>
#include <rex/logging.h>
#include <rex/memory/ring_buffer.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

REXCVAR_DEFINE_BOOL(gpu_d3d_own_input, false, "GPU/Native",
                    "Session 87 (D3D layer stage 3, experimental, needs gpu_d3d_direct): a record draw's constants, system "
                    "constants and primitive processing are built from the record's register image - the token's runs are "
                    "stored into it in one copy and the emulated register file is not written for a taken draw (it is "
                    "brought up to date once when something else needs it). A draw it cannot take goes the F4 way (counted, "
                    "[d3d][feed]).");
REXCVAR_DEFINE_BOOL(gpu_d3d_own_input_verify, false, "GPU/Native",
                    "Session 87 (D3D layer stage 3): the runs are applied the stock way AND into the record's register "
                    "image, the draws are the stock ones, and after each draw the image is compared with the register file "
                    "(mismatches logged and counted). Wins over gpu_d3d_own_input.");
REXCVAR_DEFINE_BOOL(gpu_d3d_own_full, false, "GPU/Native",
                    "Session 87 (D3D layer stage 3, needs gpu_d3d_own_input): a record draw that misses the lean template "
                    "(no base, state written, other shaders, a refused repeat) takes the full derivation (IssueDraw / the stock "
                    "lean path) reading the record's register image - the pipeline, render target, viewport, system "
                    "constants, bindings and primitive processing all come from the image - instead of being handed back to "
                    "the register file. The register file is written only lazily, for a later non-record consumer "
                    "(counted). Crowd first members and copy-mode draws are still handed back, by name.");
REXCVAR_DEFINE_BOOL(gpu_d3d_own_copy, false, "GPU/Native",
                    "Session 87 (D3D layer stage 3, needs gpu_d3d_own_input and gpu_d3d_own_full): the copy-family token "
                    "(clear / resolve), the inline-draw token and the ring's type-0 register packets run on the record's register "
                    "image - their register writes go into it, their draw and copy derivations read it (IssueCopy and the "
                    "render target cache's resolve through the register sources) - and the register file is not brought up to "
                    "date for them. Packets that touch no register (the bin-mask packets, INVALIDATE_STATE) and type-0 "
                    "packets wholly below 0x2000 leave the image alone.");
REXCVAR_DEFINE_BOOL(gpu_d3d_own_crowd, false, "GPU/Native",
                    "Session 87 (D3D layer stage 3, needs gpu_d3d_own_input and gpu_d3d_own_full): a crowd run's first member is "
                    "described from the record like any draw and the crowd code (SceneCrowdPreTake / AfterDraw, "
                    "AppendCrowdInstances, OwnCrowdDraw, the cull) reads the record's register image; the c76-c78 it writes for "
                    "each replayed member go into the image (pending for the register file, counted), not into the register "
                    "file.");
REXCVAR_DEFINE_INT32(gpu_d3d_own_input_abi, 1, "GPU/Native",
                     "Session 87 (D3D layer stage 3): capability marker - this build has the record-fed draw. Read-only, do "
                     "not set")
    .range(1, 1);

REXCVAR_DECLARE(bool, gpu_d3d_direct);
REXCVAR_DECLARE(bool, gpu_det_skip_host_draws);
REXCVAR_DECLARE(bool, gpu_native_frame_draw);  // (Session 92, M4f-C: CutBinOp)

namespace rex::graphics::vulkan {

namespace {

// The ranges the derivations read (and the image is copied from the register file over).
constexpr uint32_t kStateLo = 0x2000, kStateHi = 0x2400;  // render state
constexpr uint32_t kConstLo = 0x4000, kConstHi = 0x4930;  // floats, fetch constants, bool / loop
constexpr uint32_t kFloatEnd = 0x4800;                    // floats 0x4000-0x47FF
constexpr uint32_t kFetchEnd = 0x48C0;                    // fetch constants 0x4800-0x48BF
constexpr uint32_t kBoolLo = 0x4900, kBoolHi = 0x4928;    // bool / loop 0x4900-0x4927
constexpr uint32_t kRegisterCount = uint32_t(RegisterFile::kRegisterCount);

struct Feed {
  std::unique_ptr<RegisterFile> image;
  uint64_t dirty[(kRegisterCount + 63) / 64] = {};  // registers written into the image and not yet into the register file
  uint64_t next_print = 4096;
};
Feed g_f;
D3dFeedCensus g_c;

bool g_registered = [] {
  D3dFeedHandlers& h = D3dFeedHandlerSlots();
  h.run = &D3dFeedAccess::Run;
  h.flush = &D3dFeedAccess::Flush;
  h.foreign = &D3dFeedAccess::Foreign;
  h.scoped_write = &D3dFeedAccess::ScopedWrite;
  h.scope_wanted = &D3dFeedAccess::ScopeWanted;
  h.scope_begin = &D3dFeedAccess::ScopeBegin;
  h.reg_rmw = &D3dFeedAccess::RegRmw;
  h.scope_end = &D3dFeedAccess::ScopeEnd;
  h.image_values = &D3dFeedAccess::ImageValues;
  h.image_dirty = &D3dFeedAccess::ImageDirty;
  return true;
}();

bool Modelled(uint32_t first, uint32_t n) {
  // (Review 4: a record's count is the game's own word - a 32-bit wrap must not pass the range tests.)
  if (first >= kRegisterCount || n == 0 || n > kRegisterCount) return false;
  const uint32_t end = first + n;
  if (first >= kStateLo && end <= kStateHi) return true;
  if (first >= kConstLo && end <= kConstHi) return true;
  if (first >= 0x5000 && end <= 0x5003) return true;  // SHADER_CONSTANT_FLUSH_FETCH_0..2 (only stored)
  return n == 1 && (first == 0x5C8 || first == 0xE00 || first == 0xE40);
}

void MarkDirty(uint32_t first, uint32_t count) {
  uint32_t r = first;
  const uint32_t end = first + count;
  while (r < end) {
    const uint32_t bit = r & 63;
    const uint32_t run = std::min(64 - bit, end - r);
    const uint64_t mask = run == 64 ? ~uint64_t(0) : ((uint64_t(1) << run) - 1) << bit;
    g_f.dirty[r >> 6] |= mask;
    r += run;
  }
  D3dFeedState() |= kD3dFeedPending;
}

bool DirtyBit(uint32_t r) { return (g_f.dirty[r >> 6] >> (r & 63)) & 1; }

// The image's pending registers into the register file; returns the registers copied.
uint32_t CopyPending(RegisterFile& file) {
  uint32_t copied = 0;
  uint32_t r = 0;
  while (r < kRegisterCount) {
    const uint64_t word = g_f.dirty[r >> 6] >> (r & 63);
    if (!word) {
      r = ((r >> 6) + 1) << 6;
      continue;
    }
    if (!(word & 1)) {
      r += uint32_t(std::countr_zero(word));
      continue;
    }
    const uint32_t start = r;
    while (r < kRegisterCount && DirtyBit(r)) ++r;
    std::memcpy(file.values + start, g_f.image->values + start, size_t(r - start) * sizeof(uint32_t));
    copied += r - start;
  }
  std::memset(g_f.dirty, 0, sizeof(g_f.dirty));
  return copied;
}

void EnsureImage() {
  if (!g_f.image) g_f.image = std::make_unique<RegisterFile>();
}

// The image from the register file over the ranges the derivations read. The register file must be current (no pending).
void Baseline(RegisterFile& file) {
  EnsureImage();
  std::memcpy(g_f.image->values + kStateLo, file.values + kStateLo, size_t(kStateHi - kStateLo) * sizeof(uint32_t));
  std::memcpy(g_f.image->values + kConstLo, file.values + kConstLo, size_t(kConstHi - kConstLo) * sizeof(uint32_t));
  std::memset(g_f.dirty, 0, sizeof(g_f.dirty));
  D3dFeedState() = kD3dFeedValid;
  ++g_c.rebaselines;
}


}  // namespace


// Stores `n` values for registers first.. into the image; with `notify`, the notifications; with `track`, pending.
void D3dFeedAccess::Apply(VCP& cp, uint32_t first, uint32_t n, const uint32_t* values, bool notify, bool track) {
  uint32_t* const image = g_f.image->values;
  rex::graphics::free_counters::Bump(rex::graphics::free_counters::g_c.image_stores, n);  // (Session 92 M0: log-only counter)
  uint32_t i = 0;
  while (i < n) {
    const uint32_t reg = first + i;
    uint32_t end = first + n;
    Piece piece = Piece::kOther;
    if (reg >= kConstLo && reg < kFloatEnd) {
      piece = Piece::kFloat;
      end = std::min(end, kFloatEnd);
    } else if (reg >= kFloatEnd && reg < kFetchEnd) {
      piece = Piece::kFetch;
      end = std::min(end, kFetchEnd);
    } else if (reg >= kBoolLo && reg < kBoolHi) {
      piece = Piece::kBoolLoop;
      end = std::min(end, kBoolHi);
    } else {
      // The next class boundary above reg, if any.
      for (const uint32_t boundary : {kConstLo, kFloatEnd, kFetchEnd, kBoolLo, kBoolHi}) {
        if (boundary > reg) {
          end = std::min(end, boundary);
          break;
        }
      }
    }
    const uint32_t len = end - reg;
    const bool changed = std::memcmp(image + reg, values + i, size_t(len) * sizeof(uint32_t)) != 0;
    std::memcpy(image + reg, values + i, size_t(len) * sizeof(uint32_t));
    if (track) MarkDirty(reg, len);
    if (notify) Notify(cp, piece, reg, len, changed);
    i += len;
  }
}

D3dFeedCensus& D3dFeedAccess::Census() { return g_c; }

D3dFeedAccess::Mode D3dFeedAccess::ModeNow(VCP& cp) {
  // (Session 92, M4f-C) THE CUT: the feed is off for the session (no image, D3dFeedState() 0, the cvars gpu_d3d_own_* are not looked at): the record decoder
  // executes the queue's items on its own typed state (owned_record.cpp). The flag is set where the queue attaches (d3d_noring.cpp Shared()).
  if (rex::graphics::free_counters::g_cut.active) return Mode::kOff;
  if (!REXCVAR_GET(gpu_d3d_direct)) return Mode::kOff;
  const bool own = REXCVAR_GET(gpu_d3d_own_input), verify = REXCVAR_GET(gpu_d3d_own_input_verify);
  if (!own && !verify) return Mode::kOff;
  // The modes whose per-draw work needs the register file as the stock path keeps it (the native frame's two sides, the
  // frame-anatomy rows, the extents shadow, the det skip) or that have no native renderer to draw into: the stock way.
  // (The native frame: the drawing side (kDraw) is fed - its register file, DrawRegisterFile(), is the one the image
  // lags; the answers side keeps its stock handling and never touches the image.)
  if (cp.nf_role_ == VCP::NfRole::kAnswers || cp.anatomy_active_ || cp.D3dDirectExtentsActive() ||
      REXCVAR_GET(gpu_det_skip_host_draws) || !cp.native_ || !cp.native_->enabled()) {
    return Mode::kOff;
  }
  return verify ? Mode::kVerify : Mode::kOwn;
}

bool D3dFeedAccess::CrowdMatch(VCP& cp) {
  // The crowd code (SceneCrowdPreTake / SceneCrowdAfterDraw) acts on a draw only when the draw's c76-c78 equal a queued
  // run's first member (among the oldest 32), or while a verify run is active; the queue itself is non-empty nearly
  // always in a crowd scene (run 007: 99% of the draws), so "a run is queued" is not the test. The same match, from the
  // image (the register file lags it). The queue is moved into the local deque as both stock functions do first.
  if (cp.scene_crowd_verify_active_) return true;
  if (cp.scene_crowd_queued_.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lock(cp.scene_crowd_mutex_);
    for (auto& r : cp.scene_crowd_queue_) cp.scene_crowd_local_.push_back(std::move(r));
    cp.scene_crowd_queue_.clear();
    cp.scene_crowd_queued_.store(0, std::memory_order_relaxed);
  }
  if (cp.scene_crowd_local_.empty() || !g_f.image) return false;
  constexpr uint32_t kCrowdRegister = 0x4000 + 4 * 76;  // c76-c78 (command_processor.cpp kSceneCrowdRegister)
  const uint32_t* now = g_f.image->values + kCrowdRegister;
  const size_t look = std::min<size_t>(cp.scene_crowd_local_.size(), 32);
  for (size_t i = 0; i < look; ++i) {
    const uint32_t* first = cp.scene_crowd_local_[i].first;
    if (first[0] == now[0] && std::equal(now, now + 12, first)) return true;
  }
  return false;
}

void D3dFeedAccess::Flush(CommandProcessor& base) {
  if (!(D3dFeedState() & kD3dFeedPending)) return;
  VCP& cp = static_cast<VCP&>(base);
  const uint32_t copied = CopyPending(*cp.register_file_);
  D3dFeedState() &= ~kD3dFeedPending;
  ++g_c.flushes;
  g_c.flushed_registers += copied;
}

void D3dFeedAccess::Foreign(CommandProcessor& base) {
  const uint32_t kind = D3dFeedLastPacket() & 1023;
  if (D3dFeedState() & kD3dFeedValid) {
    ++g_c.foreign;
    ++g_c.foreign_by_kind[kind];
  }
  if (D3dFeedState() & kD3dFeedPending) {
    // (Census: where the packets that still make the register file catch up come from - the frames they occur in, and the first
    // few with their guest address and the indirect buffer that called them, to name the writer.)
    VCP& cp = static_cast<VCP&>(base);
    const uint32_t frame = uint32_t(cp.frame_current_);
    if (!g_c.flush_by_kind[kind]) g_c.flush_first_frame[kind] = frame;
    g_c.flush_last_frame[kind] = frame;
    // A no-ring Registers item (type 2 opcode 9) that brought the register file up to date: why the image scope was not wanted for it,
    // and what the item is (diagnostic, the first 16).
    if (kind == ((2u << 8) | 9u)) {
      static uint32_t reg_item_logged = 0;
      if (reg_item_logged++ < 16) {
        const char* why = "the scope was wanted (?)";
        if (!REXCVAR_GET(gpu_d3d_direct)) why = "gpu_d3d_direct off";
        else if (!REXCVAR_GET(gpu_d3d_own_input) && !REXCVAR_GET(gpu_d3d_own_input_verify)) why = "own_input off";
        else if (cp.nf_role_ == VCP::NfRole::kAnswers) why = "the answers side";
        else if (cp.anatomy_active_) why = "frame anatomy active";
        else if (cp.D3dDirectExtentsActive()) why = "the extents shadow is active";
        else if (REXCVAR_GET(gpu_det_skip_host_draws)) why = "gpu_det_skip_host_draws";
        else if (!cp.native_) why = "no native renderer";
        else if (!cp.native_->enabled()) why = "the native renderer is not enabled this frame";
        else if (REXCVAR_GET(gpu_d3d_own_input_verify)) why = "verify mode";
        else if (!REXCVAR_GET(gpu_d3d_own_copy)) why = "gpu_d3d_own_copy off";
        else if (!REXCVAR_GET(gpu_d3d_own_full)) why = "gpu_d3d_own_full off";
        const uint32_t info = D3dFeedItemInfo();
        REXGPU_INFO("[d3d][feed] a Registers item made the register file catch up: frame {}, scope not taken because {}; item: first op {} register 0x{:04X}, "
                    "{} op(s); an image scope is open: {}",
                    frame, why, info >> 24, info & 0xFFFF, (info >> 16) & 0xFF, D3dFeedDrawRegs() ? "yes" : "no");
      }
    }
    if (kind != 0x364 && g_c.flush_logged < 12 && g_c.flush_by_kind[kind] < 3) {
      ++g_c.flush_logged;
      REXGPU_INFO("[d3d][feed] a packet makes the register file catch up: type {} opcode 0x{:02X} at guest 0x{:08X} (called from the "
                  "indirect buffer packet at 0x{:08X}), frame {}",
                  kind >> 8, kind & 0xFF, cp.half2_packet_at_, cp.half2_ib_caller_, frame);
    }
    Flush(base);
    ++g_c.flush_by_packet;
    ++g_c.flush_by_kind[kind];
  }
  D3dFeedState() = 0;  // (stale: the packet may write the register file)
}

// gpu_d3d_own_crowd: a register write inside a record-fed draw (the crowd code's c76-c78 of each member it replays) lands in the
// image as a run of one register would (the notifications, pending for the register file).
bool D3dFeedAccess::ScopedWrite(CommandProcessor& base, uint32_t index, uint32_t value) {
  if (!D3dFeedDrawRegs() || !g_f.image || !Modelled(index, 1)) return false;
  Apply(static_cast<VCP&>(base), index, 1, &value, true, true);
  ++g_c.scoped_writes;
  return true;
}

uint32_t* D3dFeedAccess::ImageValues() { return g_f.image ? g_f.image->values : nullptr; }

void D3dFeedAccess::ImageDirty(uint32_t first, uint32_t count) {
  MarkDirty(first, count);
  g_c.scoped_group_registers += count;
}

bool D3dFeedAccess::Run(CommandProcessor& base, rex::memory::RingBuffer* reader, uint32_t first, uint32_t n) {
  if (!D3dFeedState() && !REXCVAR_GET(gpu_d3d_own_input) && !REXCVAR_GET(gpu_d3d_own_input_verify)) {
    return false;  // (the default: nothing to look at)
  }
  VCP& cp = static_cast<VCP&>(base);
  D3dFeedCensus& c = g_c;
  // The answers side (native frame) applies the token the stock way to the real register file; the image is the
  // drawing side's, which applies the same token once when it catches up.
  if (cp.nf_role_ == VCP::NfRole::kAnswers) return false;
  const Mode mode = ModeNow(cp);
  if (mode == Mode::kOff || !Modelled(first, n)) {
    if (mode != Mode::kOff) ++c.runs_unmodelled;
    ++c.runs_stock;
    D3dFeedLastPacket() = 0x3FE;  // (Session 90: a type-0 range the image does not take - the census key)
    Foreign(base);  // (the register file current, the image stale)
    return false;
  }
  RegisterFile& file = *cp.register_file_;
  if (mode == Mode::kVerify && (D3dFeedState() & kD3dFeedPending)) Flush(base);  // (switched from own to verify)
  if (!(D3dFeedState() & kD3dFeedValid)) Baseline(file);

  static thread_local std::vector<uint32_t> values;
  values.resize(n);
  const size_t saved = reader->read_offset();
  for (uint32_t i = 0; i < n; ++i) values[i] = reader->ReadAndSwap<uint32_t>();
  if (mode == Mode::kVerify) {
    // The stock path writes the register file and notes everything; the image only follows.
    reader->set_read_offset(saved);
    Apply(cp, first, n, values.data(), false, false);
    return false;
  }
  cp.repeat_benign_ = true;  // (the F4 handler's latch: the record's shader loads / trailing flush registers are benign)
  // The bookkeeping ExecutePacketType0Body does before the writes.
  if (cp.half2_ && cp.nf_role_ != VCP::NfRole::kAnswers) cp.Half2NoteWrite(first, n, 0, cp.half2_packet_at_, 0);
  if (cp.frame_counters_enabled_) cp.CountType0Packet(first, n, false);
  Apply(cp, first, n, values.data(), true, true);
  ++c.runs_fed;
  c.registers_fed += n;
  return true;
}

void D3dFeedAccess::BeginDraw(VCP& cp, const D3dFeedItem& item, Mode mode) {
  if (!(D3dFeedState() & kD3dFeedValid)) {
    if (D3dFeedState() & kD3dFeedPending) Flush(cp);
    Baseline(*cp.register_file_);
  }
  RegisterFile& image = *g_f.image;
  const bool own = mode == Mode::kOwn;
  // The draw's words, as DRAW_INDX's registers (not noted: the draw packet's own registers are not state; the backend
  // compares the draw parameters itself).
  image.values[XE_GPU_REG_VGT_DRAW_INITIATOR] = item.initiator;
  image.values[XE_GPU_REG_VGT_DMA_BASE] = item.dma_base;
  image.values[XE_GPU_REG_VGT_DMA_SIZE] = item.dma_size;
  if (own) {
    MarkDirty(XE_GPU_REG_VGT_DMA_BASE, 3);  // (0x21FA-0x21FC: base, size, initiator)
    D3dFeedDrawRegs() = &image;
    if (cp.texture_cache_) cp.texture_cache_->SetRegisterSource(&image);
    if (cp.pipeline_cache_) cp.pipeline_cache_->SetRegisterSource(&image);
    if (cp.render_target_cache_) cp.render_target_cache_->SetRegisterSource(&image);
    if (cp.primitive_processor_) cp.primitive_processor_->SetRegisterSource(&image);
  }
}

// gpu_d3d_own_copy. The mode: the copy-family / inline / type-0 packets run on the image while the draw path does (kOwn).
bool D3dFeedAccess::ScopeWanted(CommandProcessor& base) {
  if (!REXCVAR_GET(gpu_d3d_own_copy) || !REXCVAR_GET(gpu_d3d_own_full)) return false;
  return ModeNow(static_cast<VCP&>(base)) == Mode::kOwn;
}

bool D3dFeedAccess::ScopeBegin(CommandProcessor& base) {
  VCP& cp = static_cast<VCP&>(base);
  if (D3dFeedDrawRegs() || !ScopeWanted(base)) return false;  // (a scope is open already, or the mode is off)
  if (!(D3dFeedState() & kD3dFeedValid)) {
    if (D3dFeedState() & kD3dFeedPending) Flush(cp);
    Baseline(*cp.register_file_);
  }
  RegisterFile& image = *g_f.image;
  D3dFeedDrawRegs() = &image;
  if (cp.texture_cache_) cp.texture_cache_->SetRegisterSource(&image);
  if (cp.pipeline_cache_) cp.pipeline_cache_->SetRegisterSource(&image);
  if (cp.render_target_cache_) cp.render_target_cache_->SetRegisterSource(&image);
  if (cp.primitive_processor_) cp.primitive_processor_->SetRegisterSource(&image);
  ++g_c.copy_scopes;
  return true;
}

// Session 90 (contract R): a REG_RMW packet on the image. Both the target and (when the packet reads registers) the source registers must be
// ones the image models; the old value is the image's, the new one goes in as a single-register run would.
bool D3dFeedAccess::RegRmw(CommandProcessor& base, uint32_t rmw_info, uint32_t and_mask, uint32_t or_mask) {
  if (!ScopeWanted(base)) return false;
  const uint32_t target = rmw_info & 0x1FFF;
  // Only registers the baseline copies into the image (0x2000-0x23FF, 0x4000-0x492F) hold real values there; Modelled() also accepts
  // 0x5C8 / 0xE00 / 0xE40, which the baseline never copies, so a REG_RMW on those would compute on a stale image value.
  // Every reachable REG_RMW targets such a register (the 13-bit target field cannot reach 0x4000+), so in practice this declines
  // them all: R's REG_RMW-on-image never applies and frame 88's catch-up stays (reg_rmw_fed stays 0).
  const auto baselined = [](uint32_t r) { return (r >= kStateLo && r < kStateHi) || (r >= kConstLo && r < kConstHi); };
  if (!baselined(target) || !Modelled(target, 1)) return false;
  if (((rmw_info >> 31) & 1) && !(baselined(and_mask & 0x1FFF) && Modelled(and_mask & 0x1FFF, 1))) return false;
  if (((rmw_info >> 30) & 1) && !(baselined(or_mask & 0x1FFF) && Modelled(or_mask & 0x1FFF, 1))) return false;
  const bool opened = ScopeBegin(base);
  if (!D3dFeedDrawRegs() || !g_f.image) return false;
  const uint32_t* v = g_f.image->values;
  uint32_t value = v[target];
  value &= ((rmw_info >> 31) & 1) ? v[and_mask & 0x1FFF] : and_mask;
  value |= ((rmw_info >> 30) & 1) ? v[or_mask & 0x1FFF] : or_mask;
  Apply(static_cast<VCP&>(base), target, 1, &value, true, true);
  ++g_c.reg_rmw_fed;
  if (opened) EndDraw(static_cast<VCP&>(base));
  return true;
}

void D3dFeedAccess::ScopeEnd(CommandProcessor& base) { EndDraw(static_cast<VCP&>(base)); }

void D3dFeedAccess::EndDraw(VCP& cp) {
  if (!D3dFeedDrawRegs()) return;
  D3dFeedDrawRegs() = nullptr;
  if (cp.texture_cache_) cp.texture_cache_->SetRegisterSource(nullptr);
  if (cp.pipeline_cache_) cp.pipeline_cache_->SetRegisterSource(nullptr);
  if (cp.render_target_cache_) cp.render_target_cache_->SetRegisterSource(nullptr);
  if (cp.primitive_processor_) cp.primitive_processor_->SetRegisterSource(nullptr);
}

void D3dFeedAccess::VerifyDraw(VCP& cp) {
  if (!(D3dFeedState() & kD3dFeedValid)) return;  // (a foreign packet in between: nothing to compare)
  D3dFeedCensus& c = g_c;
  ++c.verify_checked;
  const RegisterFile& file = *cp.register_file_;
  RegisterFile& image = *g_f.image;
  uint32_t bad = 0;
  const auto compare = [&](uint32_t lo, uint32_t hi) {
    for (uint32_t r = lo; r < hi; ++r) {
      if (file.values[r] == image.values[r]) continue;
      ++bad;
      if (c.verify_logged < 24) {
        ++c.verify_logged;
        REXGPU_WARN("[d3d][feed] verify: register 0x{:04X} of the image is 0x{:08X}, the register file's 0x{:08X} (frame {})", r,
                    image.values[r], file.values[r], cp.frame_current_);
      }
      image.values[r] = file.values[r];  // (back in step)
    }
  };
  compare(kStateLo, kStateHi);
  compare(kConstLo, kConstHi);
  if (bad) {
    ++c.verify_mismatches;
    c.verify_registers += bad;
  }
}

void D3dFeedAccess::Print(VCP& cp) {
  D3dFeedCensus& c = g_c;
  if (c.runs_fed + c.runs_stock + c.draws_direct + c.verify_checked + c.draws_seen == 0) return;
  {
    const uint64_t fed = c.draws_direct + c.draws_full_image;
    const uint64_t back = c.fb_crowd + c.fb_copy_mode + c.fb_pass + c.fb_template + c.fb_mode;
    REXGPU_INFO(
        "[d3d][feed] FED FROM RECORD frame {}: {} of {} draw items ({:.1f}%) = {} by the lean template / cache / repeat + {} by "
        "the full derivation reading the image ({} IssueDraw, {} stock lean); register-file round trips (handed back): {} = "
        "crowd first member {}, copy mode {}, special mode {}, template did not hold (gpu_d3d_own_full off) {}, mode {}; "
        "register file brought up to date {} times for them and later consumers",
        cp.frame_current_, fed, c.draws_seen, c.draws_seen ? 100.0 * double(fed) / double(c.draws_seen) : 0.0,
        c.draws_direct, c.draws_full_image, c.full_image_template, c.full_image_pass, back, c.fb_crowd, c.fb_copy_mode,
        c.fb_pass, c.fb_template, c.fb_mode, c.flushes);
    REXGPU_INFO("[d3d][feed] crowd on the image (gpu_d3d_own_crowd): {} crowd runs' first members fed from the record; {} single "
                "c76-c78 register writes and {} registers of member groups stored into the image (pending for the register file)",
                c.draws_crowd_image, c.scoped_writes, c.scoped_group_registers);
    {
      // Session 90 (K2): the crowd code's c76-c78 restore, the catch-ups it removed and the ones left, and why our crowd program declined.
      const D3dFeedCrowdState& cs = D3dFeedCrowd();
      std::string why;
      for (uint32_t r = 0; r < 16; ++r) {
        if (cs.decline_by_reason[r]) why += fmt::format("{}{}x{}", why.empty() ? "" : " ", r, cs.decline_by_reason[r]);
      }
      REXGPU_INFO("[d3d][feed] crowd constants restored after a run the game drew (c76-c78 put back to what the game set): {} runs in the "
                  "image, {} in the register file, {} not restored; crowd catch-ups skipped because restored {}; still made: crowd run's "
                  "first member with the register file as source {}, draw handed back {}, writes not restored {}; our crowd program "
                  "declined {} runs (reason x count: {}); REG_RMW packets applied on the image {}",
                  cs.restored_image, cs.restored_regfile, cs.unrestored_runs, cs.skipped_restored, cs.catchup_crowd_hit,
                  cs.catchup_scope_ended, cs.catchup_unrestored, cs.declined_runs, why.empty() ? "none" : why, c.reg_rmw_fed);
    }
    // The packets that make the register file catch up (foreign packets seen with a valid image, and the ones that flushed).
    std::array<std::pair<uint64_t, uint32_t>, 1024> kinds;
    for (uint32_t k = 0; k < 1024; ++k) kinds[k] = {c.flush_by_kind[k], k};
    std::partial_sort(kinds.begin(), kinds.begin() + 8, kinds.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::string text;
    for (int i = 0; i < 8 && kinds[i].first; ++i) {
      text += fmt::format("{}type {} opcode 0x{:02X}: {} flushes of {} seen (frames {}-{})", i ? "; " : "", kinds[i].second >> 8,
                          kinds[i].second & 0xFF, kinds[i].first, c.foreign_by_kind[kinds[i].second],
                          c.flush_first_frame[kinds[i].second], c.flush_last_frame[kinds[i].second]);
    }
    REXGPU_INFO("[d3d][feed] register-file flushes by packet ({} in all, {} seen with a valid image): {}", c.flush_by_packet,
                c.foreign, text);
  }
  REXGPU_INFO(
      "[d3d][feed] frame {}: {} draws fed directly from the record's image; handed back to F4's path: {} crowd, {} special "
      "mode, {} template did not hold, {} mode; runs: {} stored in the image ({} registers, register file not written), "
      "{} written the stock way ({} not modelled); image re-copied {} times (foreign packets seen with a valid image {}); "
      "register file brought up to date {} times ({} registers; {} by a packet, {} by a handed-back draw); verify: {} "
      "draws compared, {} with a mismatch ({} registers); of {} draw items seen with the feed on, {} were a crowd run's first "
      "member",
      cp.frame_current_, c.draws_direct, c.fb_crowd, c.fb_pass, c.fb_template, c.fb_mode, c.runs_fed, c.registers_fed,
      c.runs_stock, c.runs_unmodelled, c.rebaselines, c.foreign, c.flushes, c.flushed_registers, c.flush_by_packet,
      c.flush_by_draw, c.verify_checked, c.verify_mismatches, c.verify_registers, c.draws_seen, c.crowd_hits);
}

}  // namespace rex::graphics::vulkan
