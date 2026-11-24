// wwe2k14 patch (Session 95, item 4: frame graph lite): the walk over a submission's recorded stream, at its end.
//
// Design: out\s95\read_AG.md. What is built here (every piece behind a bit of gpu_owned_graph, default 0 = no walk, nothing changes):
//   bit 1  (G0) the event log and the dry line: the stream is scanned once at EndSubmission (DeferredCommandBuffer::ScanStream); what the
//          proofs below would do is counted and printed as a [native][graph] line per window. No behaviour change.
//   bit 8  (G3) dead passes and dead copies, by REACHABILITY inside the stream: a render pass instance (and an image copy / blit / clear
//          into an owned image) whose output is never read after it and is either overwritten in the window or held by nothing at the
//          window's end is turned off (NopRenderPass with the state commands kept / NopImageCommandAt); every barrier stays, so the
//          layout bookkeeping stays true.
//   bit 16 (round 7) a read by a dead pass is not a read: every reader of an owned image is noted in the stream (OwnedGraphNoteRead, beside the tripwire; the read of a draw is stamped where the
//          draw is recorded, inside its pass: OwnedGraphReadsLand). A segment that is live only because of a transition to a read layout, whose noted reads all sit in passes that go, is dead too
//          (until nothing changes; a pass that cannot go - refused, kept by a shared segment - keeps what it reads). A read by nothing known (a fill, a write-back, the swap, a draw that was not
//          recorded in a pass of ours, a pass that is not a pass image's) keeps the segment live, and so does a transition with no noted read at all. Needs bit 1 or 8.
// (G4, predicted snapshots out, was built and removed: a continuing draw that covers a committed pass's box already ends the pass - OwnedTakeDrawImpl.)
// G1 (exact source stage masks) and G2 (depth store by reachability) are not built (the Surface barrier A/B decides G1).
//
// The reading of a stream (hard rule 6: images, passes and holders - nothing of the 360):
//  * A RESOURCE is an owned image's colour or its own depth (the shared scratch depths are never read: not tracked).
//  * A SEGMENT of a resource runs from a point where its old contents are discarded to the next such point: an image barrier from
//    UNDEFINED, or a pass begin whose load op for it is not LOAD (CLEAR / DONT_CARE). Its WRITERS are the pass instances begun over
//    it and the copies / blits / clears into it.
//  * A segment is LIVE when, after its first writer, something may read it: a copy / blit / image-to-buffer with it as source (named in
//    the stream), a barrier that moves it to a layout something can read it in (shader read, transfer source, general: a read of an
//    image in an attachment layout needs such a transition - the layouts are the owned code's own, a render pass of ours has the same
//    initial and final layout), or it is the LAST segment of the window and something holds it (OwnedImageHeld / OwnedDepthHeld: a page,
//    a pending / parked pass ...). Everything else is dead. No hook at the sites that sample or fill: a missed reader cannot happen
//    unless it reads an image in an attachment layout.
//  * Anything not named (a dispatch, a host callback) is not a writer or a reader here; a window with a host callback is not changed.
//
// For the job system: the walk reads this submission's stream and the renderer's holder lists on the command thread and patches only
// begins / ends / copy commands of that stream; its state (OwnedGraphState) is its own and is passed around by pointer. The noted reads are that state too; a job
// recording draws would own its own list, in its stream's offsets.

#include <rex/graphics/vulkan/native/native_renderer.h>

#include <algorithm>
#include <chrono>
#include <cstring>

#include <rex/cvar.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/deferred_command_buffer.h>
#include <rex/logging.h>

REXCVAR_DECLARE(bool, gpu_native_owned_frame_continue_copy);  // (owned_steps.cpp)

REXCVAR_DEFINE_UINT32(gpu_owned_graph, 0, "GPU/Native",
                      "Session 95 (frame graph lite): bit 1 = scan every submission's recorded stream and print the [native][graph] line (dry: nothing changes); "
                      "bit 8 = G3: render pass instances and image copies / clears into owned images that nothing can read afterwards (no reader in the stream, "
                      "overwritten in the window or held by nothing at its end) are turned off, barriers kept (a segment's passes all or none); "
                      "bit 16 (Session 95, round 7, with bit 1 or 8) = a read by a pass that goes too is not a read: a segment live only through a transition to a read layout, whose noted reads are all in dead passes, is dead. 0 = off, byte-identical");

namespace rex::graphics::vulkan::native {

struct OwnedGraphState {
  using Event = DeferredCommandBuffer::GraphEvent;
  struct Seg {
    uint32_t res = 0;
    bool has_writer = false, read_real = false, read_transition = false, last = false, held = false;
    bool read_unknown = false;  // (bit 16: a noted read that is not a known pass's)
    int32_t last_reader = -1;   // (bit 16: the instance of the last entry in seg_reads for this segment)
    bool Live() const { return read_real || read_transition || (last && held); }
    bool LiveIfTransitionsIgnored() const { return read_real || (last && held); }
  };
  struct Inst {
    uint32_t begin_args = 0, end_next = 0;
    int32_t cseg = -1, dseg = -1;  // segments of its colour / its own depth (-1: none)
    bool closed = false;
  };
  struct Writer {
    uint32_t at = 0;
    uint32_t seg = 0;
  };
  // Scratch, reused by every walk.
  std::vector<Event> ev;
  std::unordered_map<uint64_t, uint32_t> image_res;   // VkImage -> res (image index * 2 + 1 for a depth)
  std::unordered_map<uint64_t, uint32_t> fb_info;     // framebuffer -> (image index << 1) | has own depth
  std::unordered_map<uint64_t, uint32_t> rp_variant;  // render pass -> kOwnedPass*
  std::vector<Seg> segs;
  std::vector<Inst> insts;
  std::vector<Writer> writers;
  std::vector<uint32_t> cur;  // per res: the current segment (UINT32_MAX none)
  // Round 7 (bit 16): this submission's noted reads, in stream order, and what the walk makes of them.
  enum ReadKind : uint8_t { kReadAsked, kReadLanded, kReadDirect };  // (asked: a view handed to a draw not recorded yet - left so, nobody known reads it; landed: in the draw's pass; direct: not a draw of a pass)
  struct Read {
    uint32_t at = 0;
    int16_t image = 0;
    ReadKind kind = kReadDirect;
  };
  std::vector<Read> reads;
  size_t reads_open = 0;                // (reads from here on are the current draw's request)
  std::vector<int16_t> last_draw_reads, read_scratch;  // (what the last recorded draw read: its repeats read it too, they ask for nothing)
  std::vector<uint64_t> read_pass;      // per image: the begin it was last noted in (one entry per image and pass)
  std::vector<std::pair<uint32_t, int32_t>> seg_reads;  // (segment, instance)
  std::vector<uint8_t> forgive_keep, forgive_drop, forgive_dry, forgive_seen, forgive_blocked;
  // The window's counters (printed and reset by OwnedGraphSummary).
  enum Bail { kBailOverflow, kBailReadback, kBailLayers, kBailWriteBack, kBailHostCallback, kBails };
  uint64_t walks = 0, events = 0, barriers = 0, dispatches = 0, unknown_begins = 0, host_callbacks = 0;
  uint64_t segments = 0, segments_dead = 0, segments_dead_if_ignored = 0;
  uint64_t instances = 0, instances_unclosed = 0, inst_dead = 0, inst_dead_killed = 0, inst_dead_unheld = 0, inst_colour_dead_depth_live = 0;
  uint64_t inst_dropped = 0, inst_refused = 0;
  uint64_t copies_dead = 0, copies_dropped = 0;
  uint64_t reads_forgiven = 0, segments_freed = 0;
  uint64_t bail[kBails] = {};
  uint64_t walk_ns = 0;
  uint64_t snapshots = 0, continue_copies = 0, predicted_snapshots = 0, snapshots_continued = 0, tripwire = 0;
  uint64_t first_logged_frame = UINT64_MAX;
};

namespace {
constexpr uint32_t kNone = UINT32_MAX;
constexpr uint32_t kBitDry = 1, kBitDead = 8, kBitReads = 16;

bool ReadLayout(int32_t layout) {
  // A transition to anything but an attachment layout, a transfer destination or "contents discarded" makes the image readable.
  switch (VkImageLayout(layout)) {
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
    case VK_IMAGE_LAYOUT_UNDEFINED:
      return false;
    default:
      return true;
  }
}
}  // namespace

void NativeRenderer::OwnedGraphCount(uint32_t what) {
  if (!REXCVAR_GET(gpu_owned_graph)) return;
  if (!graph_state_) graph_state_ = std::make_shared<OwnedGraphState>();
  OwnedGraphState& g = *graph_state_;
  switch (what) {
    case kGraphCountSnapshot: ++g.snapshots; break;
    case kGraphCountContinueCopy: ++g.continue_copies; break;
    case kGraphCountPredictedSnapshot: ++g.predicted_snapshots; break;
    case kGraphCountSnapshotContinued: ++g.snapshots_continued; break;
    default: break;
  }
}

bool NativeRenderer::OwnedDepthHeld(int16_t image) const {
  // (Mirrors OwnedImageHeld for the depth: the only routes to a pass image's depth are a pass that keeps it - pending, parked, restorable or a surface's
  // newest record, unless the record's depth is gone - a direct depth page, a dirty depth entry and the write-back.)
  const auto keeps = [&](const OwnedPass& q) { return q.image == image && q.depth && !q.rec_depth_gone; };
  if ((owned_pending_.valid && keeps(owned_pending_)) || std::any_of(owned_parked_.begin(), owned_parked_.end(), keeps) ||
      std::any_of(owned_restorable_.begin(), owned_restorable_.end(), keeps) || std::any_of(owned_newest_.begin(), owned_newest_.end(), keeps)) {
    return true;
  }
  if (std::any_of(owned_pages_.begin(), owned_pages_.end(), [&](const OwnedPage& p) { return p.image == image && p.depth_direct; })) return true;
  if (std::any_of(owned_dirty_.begin(), owned_dirty_.end(), [&](const OwnedDirty& d) {
        return d.image == image && (d.aspects & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT));
      })) {
    return true;
  }
  return owned_wb_.armed && owned_wb_.image == image;
}

void NativeRenderer::OwnedGraphTripwire(int16_t image) {
  if (image < 0 || size_t(image) >= owned_images_.size() || !owned_images_[image].graph_dropped) return;
  if (!graph_state_) return;
  OwnedGraphState& g = *graph_state_;
  if (++g.tripwire <= 8) {
    REXGPU_ERROR("[native][graph] TRIPWIRE: frame {} - image {} was read, copied or filled after its last segment was dropped (gpu_owned_graph bit 8)", frame_, image);
  }
}

void NativeRenderer::OwnedGraphNoteRead(int16_t image, bool by_draw) {
  const uint32_t bits = REXCVAR_GET(gpu_owned_graph);
  if (!(bits & kBitReads) || !(bits & (kBitDry | kBitDead)) || image < 0) return;
  if (!graph_state_) graph_state_ = std::make_shared<OwnedGraphState>();
  OwnedGraphState::Read r;
  r.at = uint32_t(command_processor_.deferred_command_buffer().stream_offset());
  r.image = image;
  r.kind = by_draw ? OwnedGraphState::kReadAsked : OwnedGraphState::kReadDirect;
  graph_state_->reads.push_back(r);
}

void NativeRenderer::OwnedGraphReadsLand(const Draw& draw) {
  // (OwnedRecord, the pass begun and open, the draw not written yet. A flushed clear recorded inside another draw's OwnedTakeDraw samples nothing of that draw's: not it.)
  if (!graph_state_ || owned_flushing_held_ || draw.fast_clear || !(REXCVAR_GET(gpu_owned_graph) & kBitReads)) return;
  OwnedGraphState& g = *graph_state_;
  const DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const uint32_t at = uint32_t(cb.stream_offset());
  const uint64_t pass = uint64_t(cb.last_begin_render_pass_offset());
  const auto land = [&](int16_t image) {
    if (g.read_pass.size() <= size_t(image)) g.read_pass.resize(size_t(image) + 1, 0);
    if (g.read_pass[image] == pass) return;
    g.read_pass[image] = pass;
    OwnedGraphState::Read r;
    r.at = at;
    r.image = image;
    r.kind = OwnedGraphState::kReadLanded;
    g.reads.push_back(r);
  };
  if (!draw.repeat) {  // (a repeat asked for nothing: what is open is not its own)
    g.read_scratch.clear();
    size_t kept = g.reads_open;
    for (size_t i = g.reads_open; i < g.reads.size(); ++i) {
      if (g.reads[i].kind == OwnedGraphState::kReadAsked) g.read_scratch.push_back(g.reads[i].image);
      else g.reads[kept++] = g.reads[i];  // (a fill recorded on the way here: it stays in stream order, ahead of the draw)
    }
    g.reads.resize(kept);
    g.last_draw_reads.swap(g.read_scratch);
  }
  for (const int16_t image : g.last_draw_reads) land(image);
  if (!draw.repeat) g.reads_open = g.reads.size();
}

void NativeRenderer::OwnedGraphReadsClose() {
  if (!graph_state_) return;
  OwnedGraphState& g = *graph_state_;
  // (A draw that asked and was never recorded in a pass leaves its asked reads open: it is not the base of the repeats that follow, so an older draw's list does not stand for it.)
  for (size_t i = g.reads_open; i < g.reads.size(); ++i) {
    if (g.reads[i].kind == OwnedGraphState::kReadAsked) {
      g.last_draw_reads.clear();
      break;
    }
  }
  g.reads_open = g.reads.size();
}

void NativeRenderer::OwnedGraphFinalize() {
  const uint32_t bits = REXCVAR_GET(gpu_owned_graph);
  // (This submission's notes go with it, however the walk ends. The last draw's reads stay: a repeat may come in the next stream.)
  struct ReadsDone {
    OwnedGraphState* s;
    ~ReadsDone() {
      if (!s) return;
      s->reads.clear();
      s->reads_open = 0;
      s->read_pass.clear();
    }
  } reads_done{graph_state_.get()};
  if (!(bits & (kBitDry | kBitDead))) return;
  if (!graph_state_) graph_state_ = std::make_shared<OwnedGraphState>();
  OwnedGraphState& g = *graph_state_;
  using Event = OwnedGraphState::Event;
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const auto t0 = std::chrono::steady_clock::now();
  cb.FlushHeldCommands();  // (the held barrier is written: its offset in the stream is known)
  g.ev.clear();
  ++g.walks;
  if (!cb.ScanStream(g.ev, 8192)) {
    ++g.bail[OwnedGraphState::kBailOverflow];
    return;
  }
  if (g.ev.empty()) return;
  g.events += g.ev.size();
  // The owned images' names in the stream.
  g.image_res.clear();
  g.fb_info.clear();
  g.rp_variant.clear();
  for (size_t i = 0; i < owned_images_.size(); ++i) {
    const OwnedImage& o = owned_images_[i];
    if (!o.width) continue;
    if (o.image != VK_NULL_HANDLE) g.image_res[reinterpret_cast<uint64_t>(o.image)] = uint32_t(i) * 2;
    if (o.depth != VK_NULL_HANDLE) g.image_res[reinterpret_cast<uint64_t>(o.depth)] = uint32_t(i) * 2 + 1;
    if (o.framebuffer != VK_NULL_HANDLE) g.fb_info[reinterpret_cast<uint64_t>(o.framebuffer)] = uint32_t(i) << 1;
    if (o.framebuffer_depth != VK_NULL_HANDLE) g.fb_info[reinterpret_cast<uint64_t>(o.framebuffer_depth)] = (uint32_t(i) << 1) | 1;
  }
  for (uint32_t v = 0; v < kOwnedPasses; ++v) {
    for (uint32_t k = 0; k < kPostTargetClasses; ++k) {
      if (owned_render_passes_[v][k] != VK_NULL_HANDLE) g.rp_variant[reinterpret_cast<uint64_t>(owned_render_passes_[v][k])] = v;
    }
  }
  g.segs.clear();
  g.insts.clear();
  g.writers.clear();
  g.cur.assign(owned_images_.size() * 2, kNone);
  const auto lookup = [&](uint64_t image, uint32_t& res) {
    const auto it = g.image_res.find(image);
    if (it == g.image_res.end()) return false;
    res = it->second;
    return true;
  };
  // A segment starts where the old contents are discarded: a fresh one unless the current one has no writer yet.
  const auto kill = [&](uint32_t res) {
    const uint32_t c = g.cur[res];
    if (c != kNone && !g.segs[c].has_writer) return;
    g.segs.emplace_back();
    g.segs.back().res = res;
    g.cur[res] = uint32_t(g.segs.size() - 1);
  };
  const auto current = [&](uint32_t res) {
    if (g.cur[res] == kNone) {
      g.segs.emplace_back();
      g.segs.back().res = res;
      g.cur[res] = uint32_t(g.segs.size() - 1);
    }
    return g.cur[res];
  };
  const auto read_real = [&](uint32_t res) {
    const uint32_t c = g.cur[res];
    if (c != kNone && g.segs[c].has_writer) g.segs[c].read_real = true;
  };
  int32_t open = -1;
  uint32_t walk_host_callbacks = 0;  // (this walk's: a host callback may touch any image, the window is left as recorded)
  // Bit 16: each noted read goes to the segments its image is in at that point of the stream and to the pass open there; a read outside a known pass is nobody's known.
  const bool reads_on = (bits & kBitReads) != 0 && OwnedOnly();  // (outside owned-only frames replays and repeats ask for nothing, so their reads are not noted: no forgiving there)
  size_t next_read = 0;
  if (reads_on) {
    g.seg_reads.clear();
    const auto by_offset = [](const OwnedGraphState::Read& a, const OwnedGraphState::Read& b) { return a.at < b.at; };
    if (!std::is_sorted(g.reads.begin(), g.reads.end(), by_offset)) std::stable_sort(g.reads.begin(), g.reads.end(), by_offset);
  }
  const auto take_reads = [&](uint32_t upto) {
    for (; next_read < g.reads.size() && g.reads[next_read].at <= upto; ++next_read) {
      const OwnedGraphState::Read& r = g.reads[next_read];
      if (size_t(r.image) >= owned_images_.size()) continue;
      for (uint32_t res = uint32_t(r.image) * 2; res <= uint32_t(r.image) * 2 + 1; ++res) {
        const uint32_t c = g.cur[res];
        if (c == kNone || !g.segs[c].has_writer) continue;
        OwnedGraphState::Seg& s = g.segs[c];
        if (r.kind != OwnedGraphState::kReadLanded || open < 0) {
          s.read_unknown = true;
        } else if (s.last_reader != open) {
          s.last_reader = open;
          g.seg_reads.emplace_back(c, open);
        }
      }
    }
  };
  for (size_t i = 0; i < g.ev.size(); ++i) {
    const Event& e = g.ev[i];
    uint32_t res = 0, res2 = 0;
    if (reads_on) take_reads(e.at);
    switch (e.kind) {
      case Event::kBarrierImage:
        ++g.barriers;
        if (!lookup(e.a, res)) break;
        if (e.old_layout == int32_t(VK_IMAGE_LAYOUT_UNDEFINED)) {
          kill(res);
        } else if (ReadLayout(e.new_layout)) {
          const uint32_t c = g.cur[res];
          if (c != kNone && g.segs[c].has_writer) g.segs[c].read_transition = true;
        }
        break;
      case Event::kBegin: {
        open = -1;
        const auto fb = g.fb_info.find(e.b);
        if (e.b == 0 || fb == g.fb_info.end()) {
          ++g.unknown_begins;  // (a post / page / feedback pass: its attachments are not owned pass images)
          break;
        }
        const uint32_t image_index = fb->second >> 1;
        const bool own_depth = (fb->second & 1) != 0;
        const auto rv = g.rp_variant.find(e.a);
        const uint32_t variant = rv == g.rp_variant.end() ? uint32_t(kOwnedPassResume) : rv->second;
        const bool colour_load = variant == kOwnedPassResume || variant == kOwnedPassResumeNoDepth || rv == g.rp_variant.end();
        const bool depth_load = variant == kOwnedPassResume || rv == g.rp_variant.end();
        OwnedGraphState::Inst inst;
        inst.begin_args = e.args;
        const uint32_t cres = image_index * 2;
        if (!colour_load) kill(cres);
        inst.cseg = int32_t(current(cres));
        g.segs[inst.cseg].has_writer = true;
        if (own_depth) {
          if (!depth_load) kill(cres + 1);
          inst.dseg = int32_t(current(cres + 1));
          g.segs[inst.dseg].has_writer = true;
        }
        g.insts.push_back(inst);
        open = int32_t(g.insts.size() - 1);
      } break;
      case Event::kEnd:
        if (open >= 0) {
          g.insts[open].end_next = e.next;
          g.insts[open].closed = true;
          open = -1;
        }
        break;
      case Event::kCopyImage:
      case Event::kBlit:
        if (lookup(e.a, res)) read_real(res);
        if (lookup(e.b, res2)) {
          const uint32_t c = current(res2);
          g.segs[c].has_writer = true;
          g.writers.push_back({e.at, c});
        }
        break;
      case Event::kClearColor:
      case Event::kClearDepth:
        if (lookup(e.b, res)) {
          const uint32_t c = current(res);
          g.segs[c].has_writer = true;
          g.writers.push_back({e.at, c});
        }
        break;
      case Event::kImageToBuffer:
        if (lookup(e.a, res)) read_real(res);
        break;
      case Event::kBufferToImage:
        if (lookup(e.b, res)) g.segs[current(res)].has_writer = true;  // (a writer, never turned off: not recorded in `writers`)
        break;
      case Event::kDispatch: ++g.dispatches; break;
      case Event::kHostCallback: ++walk_host_callbacks; break;
      case Event::kQuery: break;
    }
  }
  if (reads_on) take_reads(UINT32_MAX);
  // The window's end: the last segment of each resource is held or not.
  for (uint32_t res = 0; res < g.cur.size(); ++res) {
    if (g.cur[res] == kNone) continue;
    OwnedGraphState::Seg& s = g.segs[g.cur[res]];
    s.last = true;
    const int16_t index = int16_t(res >> 1);
    if (res & 1) {
      s.held = OwnedDepthHeld(index);
    } else {
      s.held = OwnedImageHeld(index) || index == decouple_swap_image_ || (owned_strip_batch_.valid && owned_strip_batch_.image == index) ||
               (owned_wb_.armed && owned_wb_.image == index);
    }
  }
  // Bit 16: a read by a pass that goes is not a read. Which passes go is decided as below (dead colour and depth, not refused, no segment kept by another instance), so a pass that stays keeps what
  // it reads; each segment freed may free the passes that wrote it and what those read: until nothing changes. The decision below then runs once on the outcome, as it always did.
  uint64_t forgiven_now = 0, freed_now = 0;  // (counted into the window only when the frame's drops are applied)
  if (reads_on && !g.seg_reads.empty()) {
    const auto freeable = [](const OwnedGraphState::Seg& s) { return s.has_writer && s.read_transition && !s.read_real && !s.read_unknown && !(s.last && s.held); };
    g.forgive_dry.assign(g.insts.size(), 0);  // (per instance: 0 not asked, 1 NopRenderPass would take it, 2 it would refuse)
    for (bool changed = true; changed;) {
      changed = false;
      g.forgive_seen.assign(g.segs.size(), 0);
      bool any = false;
      for (const auto& sr : g.seg_reads) {
        if (!freeable(g.segs[sr.first])) continue;
        g.forgive_seen[sr.first] = 1;
        any = true;
      }
      if (!any) break;
      std::vector<uint8_t>& keep_s = g.forgive_keep;
      std::vector<uint8_t>& drop = g.forgive_drop;
      keep_s.assign(g.segs.size(), 0);
      drop.assign(g.insts.size(), 0);
      const auto keep_inst = [&](const OwnedGraphState::Inst& inst) {
        keep_s[inst.cseg] = 1;
        if (inst.dseg >= 0) keep_s[inst.dseg] = 1;
      };
      for (size_t i = 0; i < g.insts.size(); ++i) {
        const OwnedGraphState::Inst& inst = g.insts[i];
        if (inst.closed && !g.segs[inst.cseg].Live() && (inst.dseg < 0 || !g.segs[inst.dseg].Live())) {
          if (!g.forgive_dry[i]) g.forgive_dry[i] = cb.NopRenderPass(inst.begin_args, inst.end_next, true, true) ? 1 : 2;  // (dry check: a pass with a query in it is refused here)
          if (g.forgive_dry[i] == 1) {
            drop[i] = 1;
            continue;
          }
        }
        keep_inst(inst);
      }
      for (bool spread = true; spread;) {
        spread = false;
        for (size_t i = 0; i < g.insts.size(); ++i) {
          const OwnedGraphState::Inst& inst = g.insts[i];
          if (drop[i] && (keep_s[inst.cseg] || (inst.dseg >= 0 && keep_s[inst.dseg]))) {
            drop[i] = 0;
            keep_inst(inst);
            spread = true;
          }
        }
      }
      g.forgive_blocked.assign(g.segs.size(), 0);
      for (const auto& sr : g.seg_reads) {
        if (!drop[sr.second]) g.forgive_blocked[sr.first] = 1;
      }
      for (const auto& sr : g.seg_reads) {
        if (!g.forgive_seen[sr.first] || g.forgive_blocked[sr.first]) continue;
        ++forgiven_now;
        OwnedGraphState::Seg& s = g.segs[sr.first];
        if (!s.read_transition) continue;  // (its other readers: counted, the segment is freed once)
        s.read_transition = false;
        ++freed_now;
        changed = true;
        static bool logged_freed = false;
        if (!logged_freed) {
          logged_freed = true;
          REXGPU_INFO("[native][graph] first segment freed by a dead reader (frame {}): image {}{}, the pass that read it begins at stream offset {}", frame_, s.res >> 1, (s.res & 1) ? " (its depth)" : "",
                      g.insts[sr.second].begin_args);
        }
      }
    }
  }
  // The proofs.
  for (const OwnedGraphState::Seg& s : g.segs) {
    if (!s.has_writer) continue;
    ++g.segments;
    if (!s.Live()) ++g.segments_dead;
    if (!s.LiveIfTransitionsIgnored()) ++g.segments_dead_if_ignored;
  }
  g.host_callbacks += walk_host_callbacks;
  uint32_t bail_reason = OwnedGraphState::kBails;
  if (command_processor_.NativeReadbackHolds()) bail_reason = OwnedGraphState::kBailReadback;
  else if (OwnedLayers() > 1) bail_reason = OwnedGraphState::kBailLayers;
  else if (owned_wb_planned_ || owned_wb_.armed) bail_reason = OwnedGraphState::kBailWriteBack;
  else if (walk_host_callbacks) bail_reason = OwnedGraphState::kBailHostCallback;
  const bool apply = (bits & kBitDead) != 0 && bail_reason == OwnedGraphState::kBails;
  if ((bits & kBitDead) && !apply) ++g.bail[bail_reason];
  if (apply) {
    g.reads_forgiven += forgiven_now;
    g.segments_freed += freed_now;
  }
  // Instances of one segment are dropped all or none: a pass over a segment resumes (LOADs) what the one before it drew, and a segment's depth is shared by
  // every instance over it. An instance is a candidate when its colour and its own depth are dead; a segment with any instance that is not a candidate, or
  // that NopRenderPass would refuse (checked first, nothing changed), keeps all of its instances and its copies.
  std::vector<uint8_t> keep(g.segs.size(), 0);
  std::vector<uint8_t> cand(g.insts.size(), 0);
  for (size_t i = 0; i < g.insts.size(); ++i) {
    const OwnedGraphState::Inst& inst = g.insts[i];
    ++g.instances;
    if (!inst.closed) {
      ++g.instances_unclosed;
      keep[inst.cseg] = 1;
      if (inst.dseg >= 0) keep[inst.dseg] = 1;
      continue;
    }
    const OwnedGraphState::Seg& c = g.segs[inst.cseg];
    const bool cdead = !c.Live();
    const bool ddead = inst.dseg < 0 || !g.segs[inst.dseg].Live();
    if (cdead && !ddead) ++g.inst_colour_dead_depth_live;
    if (cdead && ddead) {
      cand[i] = 1;
      ++g.inst_dead;
      if (c.last) ++g.inst_dead_unheld; else ++g.inst_dead_killed;
    } else {
      keep[inst.cseg] = 1;
      if (inst.dseg >= 0) keep[inst.dseg] = 1;
    }
  }
  if (apply) {
    for (size_t i = 0; i < g.insts.size(); ++i) {
      const OwnedGraphState::Inst& inst = g.insts[i];
      if (cand[i] && !cb.NopRenderPass(inst.begin_args, inst.end_next, true, true)) {  // (dry check)
        cand[i] = 0;
        keep[inst.cseg] = 1;
        if (inst.dseg >= 0) keep[inst.dseg] = 1;
        ++g.inst_refused;
      }
    }
    // (A segment kept by one instance keeps its others, whose depth or colour segment may then be shared with a third: until nothing changes.)
    for (bool changed = true; changed;) {
      changed = false;
      for (size_t i = 0; i < g.insts.size(); ++i) {
        const OwnedGraphState::Inst& inst = g.insts[i];
        if (!cand[i]) continue;
        if (keep[inst.cseg] || (inst.dseg >= 0 && keep[inst.dseg])) {
          cand[i] = 0;
          keep[inst.cseg] = 1;
          if (inst.dseg >= 0) keep[inst.dseg] = 1;
          changed = true;
        }
      }
    }
    for (size_t i = 0; i < g.insts.size(); ++i) {
      const OwnedGraphState::Inst& inst = g.insts[i];
      if (!cand[i]) continue;
      if (cb.NopRenderPass(inst.begin_args, inst.end_next, true)) {
        ++g.inst_dropped;
        if (g.segs[inst.cseg].last) owned_images_[g.segs[inst.cseg].res >> 1].graph_dropped = true;  // (the tripwire: nothing may read it from now on)
      } else {
        ++g.inst_refused;  // (not expected after the dry check)
      }
    }
  }
  for (const OwnedGraphState::Writer& w : g.writers) {
    if (g.segs[w.seg].Live()) continue;
    ++g.copies_dead;
    if (apply && !keep[w.seg] && cb.NopImageCommandAt(w.at)) {
      ++g.copies_dropped;
      if (g.segs[w.seg].last) owned_images_[g.segs[w.seg].res >> 1].graph_dropped = true;
    }
  }
  g.walk_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
  static bool logged = false;
  if (!logged) {
    logged = true;
    REXGPU_INFO("[native][graph] first walk (frame {}): switch {}, {} events, {} image barriers, {} pass instances, {} segments, {} dispatches, {} unknown begins, {} owned images, apply {}",
                frame_, bits, g.ev.size(), g.barriers, g.insts.size(), g.segs.size(), g.dispatches, g.unknown_begins, owned_images_.size(), apply ? "yes" : "no");
  }
}

void NativeRenderer::OwnedGraphSummary(uint64_t first_frame, uint64_t last_frame) {
  if (!graph_state_) return;
  OwnedGraphState& g = *graph_state_;
  const uint32_t bits = REXCVAR_GET(gpu_owned_graph);
  const double frames = double(last_frame >= first_frame ? last_frame - first_frame + 1 : 1);
  if (bits) {
    const double walks = double(std::max<uint64_t>(g.walks, 1));
    REXGPU_INFO("[native][graph] frames {}-{}: switch {} ({}); walks/frame {:.2f}, events/walk {:.0f}, image barriers/frame {:.1f}, dispatches/frame {:.1f}, unknown begins/frame {:.1f}, host callbacks {}; "
                "pass instances/frame {:.1f} (unclosed {}): dead proven {:.2f}/frame (overwritten in the window {:.2f}, held by nothing {:.2f}; colour dead but depth live {:.2f}), "
                "dropped {:.2f}/frame, drop refused {}; image copies / clears dead {:.2f}/frame, dropped {:.2f}; segments/frame {:.1f}: dead {:.2f}, dead if a transition to a read layout were not a read {:.2f}; "
                "bails (overflow {}, readback {}, layers {}, write-back {}, host callback {}); "
                "snapshots {:.2f}/frame (predicted {:.2f}, of those the pass went on {:.2f}, mispredicted {:.2f}), continue copies {:.2f}/frame; TRIPWIRE hits {} (must be 0); walk {:.1f} us/frame ({:.1f} us per walk){}",
                first_frame, last_frame, bits, (bits & kBitDead) ? "applied" : "dry", double(g.walks) / frames, double(g.events) / walks, double(g.barriers) / frames, double(g.dispatches) / frames,
                double(g.unknown_begins) / frames, g.host_callbacks, double(g.instances) / frames, g.instances_unclosed, double(g.inst_dead) / frames,
                double(g.inst_dead_killed) / frames, double(g.inst_dead_unheld) / frames, double(g.inst_colour_dead_depth_live) / frames, double(g.inst_dropped) / frames, g.inst_refused,
                double(g.copies_dead) / frames, double(g.copies_dropped) / frames, double(g.segments) / frames, double(g.segments_dead) / frames, double(g.segments_dead_if_ignored) / frames,
                g.bail[0], g.bail[1], g.bail[2], g.bail[3], g.bail[4], double(g.snapshots) / frames, double(g.predicted_snapshots) / frames, double(g.snapshots_continued) / frames,
                double(g.predicted_snapshots > g.snapshots_continued ? g.predicted_snapshots - g.snapshots_continued : 0) / frames, double(g.continue_copies) / frames, g.tripwire,
                double(g.walk_ns) / frames / 1000.0, double(g.walk_ns) / walks / 1000.0,
                (bits & kBitReads) ? fmt::format("; reads forgiven {:.2f}/frame, segments freed {:.2f}/frame", double(g.reads_forgiven) / frames, double(g.segments_freed) / frames) : std::string());
  }
  const uint64_t keep_first = g.first_logged_frame;
  // (A window can end inside a submission: its noted reads are not the window's.)
  std::vector<OwnedGraphState::Read> keep_reads;
  std::vector<int16_t> keep_last_draw;
  std::vector<uint64_t> keep_read_pass;
  keep_reads.swap(g.reads);
  keep_last_draw.swap(g.last_draw_reads);
  keep_read_pass.swap(g.read_pass);
  const size_t keep_open = g.reads_open;
  g = OwnedGraphState();  // (the scratch containers are rebuilt by the next walk)
  g.first_logged_frame = keep_first;
  g.reads.swap(keep_reads);
  g.last_draw_reads.swap(keep_last_draw);
  g.read_pass.swap(keep_read_pass);
  g.reads_open = keep_open;
}

}  // namespace rex::graphics::vulkan::native
