// wwe2k14 patch (Session 87, D3D layer STAGE 4, textures in our own form): see include\rex\graphics\d3d_tex.h.
#include <rex/graphics/d3d_tex.h>

#include <algorithm>
#include <cstring>
#include <string>

#include <rex/cvar.h>
#include <rex/logging.h>

REXCVAR_DEFINE_INT32(gpu_d3d_tex_abi, rex::graphics::d3d_tex::kAbi, "GPU/Native",
                     "Session 87 (D3D layer stage 4, textures): the object-table ops this build consumes (Registers item ops 16-22, "
                     "include\\rex\\graphics\\d3d_tex.h). Read-only marker, do not set")
    .range(1, 1);
REXCVAR_DEFINE_INT32(gpu_d3d_textures, 0, "GPU/Native",
                     "Session 87 (D3D layer stage 4, experimental): 0 = off (the table is not kept); 1 = keep the object table and "
                     "count/verify only (the fetch constant decides the bind); 2 = a slot with a Bind id is bound by its object's "
                     "identity (the game's --wwe2k14_d3d_textures=2 sets it)")
    .range(0, 2);
REXCVAR_DEFINE_BOOL(gpu_d3d_textures_verify, true, "GPU/Native",
                    "Session 87 (D3D layer stage 4): with gpu_d3d_textures 2, also derive the key from the fetch constant and compare "
                    "(a disagreement is counted, logged, and the fetch constant's key is used)");
REXCVAR_DEFINE_BOOL(gpu_d3d_textures_drop_watch, false, "GPU/Native",
                    "Session 87 (D3D layer stage 4): with gpu_d3d_textures 2, a registered file-loaded static texture would drop its guest memory watch once "
                    "its first upload is done. DO NOT TURN ON: the evidence is in - with the watch kept, statics' watches fire (987 in the integrator's "
                    "entrance run, mostly while MISC02_INGAME.PAC loads, and 23 / 51 / 7 in the scenes), so the memory of a static is rewritten after "
                    "its load and dropping the watch shows stale textures. Statics keep their watch permanently; the switch stays for experiments");
REXCVAR_DECLARE(int32_t, gpu_d3d_layer_log_frames);

namespace rex::graphics::d3d_tex {

Table& Table::Get() {
  static Table table;
  return table;
}

bool Enabled() { return REXCVAR_GET(gpu_d3d_textures) >= 1; }
bool UseIds() { return REXCVAR_GET(gpu_d3d_textures) >= 2; }
bool VerifyKeys() { return REXCVAR_GET(gpu_d3d_textures_verify); }
bool DropWatches() { return REXCVAR_GET(gpu_d3d_textures_drop_watch) && UseIds(); }

namespace {
std::atomic<uint64_t> g_objects_generation{1};  // (Session 95)
uint64_t g_objects_taken = 0;
}  // namespace
uint64_t ObjectsGeneration() { return g_objects_generation.load(std::memory_order_relaxed); }
bool TakeObjectsChanged() {
  const uint64_t now = ObjectsGeneration();
  if (now == g_objects_taken) return false;
  g_objects_taken = now;
  return true;
}

void Table::Define(uint32_t part, uint32_t id, uint32_t a, uint32_t b, uint32_t c) {
  g_active.store(true, std::memory_order_relaxed);
  g_objects_generation.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(mu);
  Entry& e = entries[id];
  if (part == 0) {  // a (re)definition starts here: the old key and texture are dropped
    const bool had = e.id != 0;
    if (had && e.texture) RetireTexture(e.texture, e.kind);
    uint64_t binds = e.binds;
    bool loaded = e.loaded;
    e = Entry{};
    e.binds = binds;
    e.loaded = had ? loaded : false;
    e.id = id;
    e.kind = (a & 0xFF) < kKindCount ? (a & 0xFF) : kUnknown;
    e.words[0] = b;
    e.have_mask = 1;
    census.defines.fetch_add(1, std::memory_order_relaxed);
    (void)c;
    return;
  }
  if (e.id == 0) return;  // a part without its start (cannot happen: the parts travel in one item)
  if (part == 1) {
    e.words[1] = a;
    e.words[2] = b;
  } else if (part == 2) {
    e.words[3] = a;
    e.words[4] = b;
  } else {
    e.words[5] = a;
  }
  e.have_mask |= 1u << part;
}

bool Table::Find(uint32_t id, Entry& out) {
  std::lock_guard<std::mutex> lock(mu);
  const auto it = entries.find(id);
  if (it == entries.end() || it->second.have_mask != 0xF) return false;
  out = it->second;
  return true;
}

void Table::StoreKey(uint32_t id, const uint8_t key16[16], uint32_t w4, bool valid) {
  std::lock_guard<std::mutex> lock(mu);
  const auto it = entries.find(id);
  if (it == entries.end()) return;
  std::memcpy(it->second.key, key16, 16);
  it->second.key_known = true;
  it->second.key_valid = valid;
  it->second.key_w4 = w4;
}

void Table::StoreTexture(uint32_t id, void* texture, const uint8_t key16[16], uint64_t texture_epoch, uint64_t marks) {
  std::lock_guard<std::mutex> lock(mu);
  const auto it = entries.find(id);
  if (it == entries.end()) return;
  Entry& e = it->second;
  if (e.texture && e.texture != texture) RetireTexture(e.texture, e.kind);
  e.texture = texture;
  e.texture_epoch = texture_epoch;
  e.texture_marks = marks;
  std::memcpy(e.key_requested, key16, 16);
  if (texture) texture_to_id[texture] = id;
}

void Table::NoteLoad(const void* texture) {
  std::lock_guard<std::mutex> lock(mu);
  const auto it = texture_to_id.find(texture);
  if (it == texture_to_id.end()) return;
  const auto e = entries.find(it->second);
  if (e == entries.end()) return;
  if (!e->second.loaded) {
    e->second.loaded = true;
    census.first_binds[e->second.kind].fetch_add(1, std::memory_order_relaxed);
  } else {
    census.reuploads[e->second.kind].fetch_add(1, std::memory_order_relaxed);
  }
  if (e->second.kind == kFileStatic && DropWatches()) {  // the static's contents are final once loaded (trace: no flush, no later write)
    drop_pending_.emplace_back(texture, e->second.kind);
    drop_any.store(true, std::memory_order_relaxed);
  }
}

void Table::NoteWatchFire(const void* texture) {
  std::lock_guard<std::mutex> lock(mu);
  const auto it = texture_to_id.find(texture);
  if (it == texture_to_id.end()) {
    census.watch_fires_unregistered.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  const auto e = entries.find(it->second);
  const uint32_t kind = e != entries.end() ? e->second.kind : kUnknown;
  census.watch_fires[kind].fetch_add(1, std::memory_order_relaxed);
  if (e != entries.end() && e->second.loaded && (kind == kFileStatic || kind == kRenderTarget)) {
    // "Watch fired on registered static": a log line for the first few (the aim is 0 over the bench).
    static std::atomic<uint32_t> logged{0};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 40) {
      REXGPU_WARN("[d3d][tex] watch fired on registered {} object {} (texture {:p}): re-upload (counted)", kKindNames[kind], it->second,
                  texture);
    }
  }
}

void Table::RetireTexture(const void* texture, uint32_t kind) {
  if (dropped_textures_.erase(texture)) {  // nothing would notice the memory being reused: the image is outdated from now on
    // (Session 95, r6tex review: kind unknown: this is the object's end, not a Changed of its kind: D3dTexProcessChanged tells them apart by it.)
    outdate_pending_.emplace_back(texture, uint32_t(kUnknown));
    changed_any.store(true, std::memory_order_relaxed);
    census.destroyed_outdated.fetch_add(1, std::memory_order_relaxed);
  }
  texture_to_id.erase(texture);
}

void Table::EndToken() {
  for (uint32_t s = 0; s < 32; ++s) {
    if (pending_[s]) {
      census.pending_left.fetch_add(1, std::memory_order_relaxed);
      pending_[s] = 0;
    }
  }
}

void EndTokenScope() { Table::Get().EndToken(); }

void Table::ForgetTexture(const void* texture) {
  std::lock_guard<std::mutex> lock(mu);
  dropped_textures_.erase(texture);
  const auto it = texture_to_id.find(texture);
  if (it != texture_to_id.end()) {
    const auto e = entries.find(it->second);
    if (e != entries.end() && e->second.texture == texture) e->second.texture = nullptr;
    texture_to_id.erase(it);
  }
  for (auto* list : {&drop_pending_, &outdate_pending_}) {
    list->erase(std::remove_if(list->begin(), list->end(), [&](const auto& p) { return p.first == texture; }), list->end());
  }
  BumpEpoch();
}

void Table::TakeChanged(std::vector<std::pair<const void*, uint32_t>>& out) {
  std::lock_guard<std::mutex> lock(mu);
  for (auto& [id, e] : entries) {
    if (!e.changed_pending) continue;
    e.changed_pending = false;
    if (e.texture && e.loaded) out.emplace_back(e.texture, e.kind);
  }
  for (const auto& p : outdate_pending_) out.push_back(p);
  outdate_pending_.clear();
}

uint32_t Table::Recall(uint32_t slot, const uint32_t words[6], bool& never) const {
  never = true;
  if (slot >= 32 || !last_id_[slot]) return 0;
  never = false;
  return std::memcmp(last_words_[slot], words, sizeof(last_words_[slot])) == 0 ? last_id_[slot] : 0;
}

void Table::Remember(uint32_t slot, uint32_t id, const uint32_t words[6]) {
  if (slot >= 32) return;
  last_id_[slot] = id;
  std::memcpy(last_words_[slot], words, sizeof(last_words_[slot]));
}

void Table::TakeDrops(std::vector<std::pair<const void*, uint32_t>>& out) {
  std::lock_guard<std::mutex> lock(mu);
  out.swap(drop_pending_);
  drop_pending_.clear();
  drop_any.store(false, std::memory_order_relaxed);
}

void Table::RequeueDrop(const void* texture, uint32_t kind) {
  std::lock_guard<std::mutex> lock(mu);
  drop_pending_.emplace_back(texture, kind);
  drop_any.store(true, std::memory_order_relaxed);
}

void Table::MarkWatchDropped(const void* texture) {
  std::lock_guard<std::mutex> lock(mu);
  const auto it = texture_to_id.find(texture);
  if (it == texture_to_id.end()) return;
  dropped_textures_.insert(texture);
  census.watches_dropped.fetch_add(1, std::memory_order_relaxed);
}

void Table::Destroy(uint32_t id) {
  g_objects_generation.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(mu);
  const auto it = entries.find(id);
  if (it == entries.end()) {
    census.unknown_destroy.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  if (it->second.texture) RetireTexture(it->second.texture, it->second.kind);
  entries.erase(it);
  census.destroys.fetch_add(1, std::memory_order_relaxed);
}

void Table::Changed(uint32_t id) {
  std::lock_guard<std::mutex> lock(mu);
  const auto it = entries.find(id);
  if (it == entries.end()) {
    census.unknown_changed.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  if (it->second.kind == kEngineDynamic) census.plane_changed.fetch_add(1, std::memory_order_relaxed);
  it->second.changed_pending = true;
  changed_any.store(true, std::memory_order_relaxed);
  census.changed.fetch_add(1, std::memory_order_relaxed);
}

void Table::Bind(uint32_t slot, uint32_t id) {
  if (slot >= 32) return;
  g_active.store(true, std::memory_order_relaxed);
  pending_[slot] = id;
  census.bind_items.fetch_add(1, std::memory_order_relaxed);
}

void Table::OnFetchWritten(uint32_t first, uint32_t last, const uint32_t* fetch_block) {
  if (first > last) {
    const uint32_t t = first;
    first = last;
    last = t;
  }
  if (first > 31) return;
  if (last > 31) last = 31;
  for (uint32_t s = first; s <= last; ++s) {
    if (pending_[s]) {
      census.binds_consumed.fetch_add(1, std::memory_order_relaxed);
    } else {
      census.unbound_writes[g_writer][s >= 26 ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
      // The first 24 texture-type writes with no Bind (and up to 8 more per writer kind): the packet that made them, its indirect
      // buffer caller (0: the ring), the writer kind, the slot and its six dwords - the engine function that writes the slot is found
      // from the packet's guest address.
      const uint32_t* d = fetch_block ? fetch_block + 6 * s : nullptr;
      if (s < 26 && (!d || (d[0] & 3) == 2)) {
        const bool first24 = census.unbound_logged.fetch_add(1, std::memory_order_relaxed) < 24;
        const bool per_writer = census.unbound_logged_w[g_writer].fetch_add(1, std::memory_order_relaxed) < 8;
        if (first24 || per_writer) {
          if (d) {
            REXGPU_INFO("[d3d][tex] fetch constant write without a Bind: slot {} by {} (bound id was {}); packet {:08X}, indirect buffer caller "
                        "{:08X}; dwords {:08X} {:08X} {:08X} {:08X} {:08X} {:08X}",
                        s, kWriterNames[g_writer], bound_[s], g_packet_at, g_ib_caller, d[0], d[1], d[2], d[3], d[4], d[5]);
          } else {
            REXGPU_INFO("[d3d][tex] fetch constant write without a Bind: slot {} by {} (bound id was {}); packet {:08X}, indirect buffer caller {:08X}",
                        s, kWriterNames[g_writer], bound_[s], g_packet_at, g_ib_caller);
          }
        }
      }
    }
    last_writer_[s] = g_writer;
    last_had_bind_[s] = pending_[s] != 0;
    bound_[s] = pending_[s];
    pending_[s] = 0;
  }
}

void LogCensus(uint32_t frame) {
  static uint32_t window_start = 0;
  const int32_t every = REXCVAR_GET(gpu_d3d_layer_log_frames);
  if (!Enabled() || every <= 0 || frame - window_start < uint32_t(every)) return;
  Table& t = Table::Get();
  Census& c = t.census;
  auto take = [](std::atomic<uint64_t>& a) { return a.exchange(0, std::memory_order_relaxed); };
  const uint64_t by_id = take(c.by_id);
  uint64_t fb_total = 0;
  std::string fb;
  for (uint32_t i = 0; i < kFbCount; ++i) {
    const uint64_t n = take(c.fallback[i]);
    fb_total += n;
    if (n) fb += (fb.empty() ? "" : ", ") + std::string(kFallbackNames[i]) + " " + std::to_string(n);
  }
  std::string kinds, firsts, reup, wf;
  for (uint32_t k = 0; k < kKindCount; ++k) {
    const std::string sep = k ? ", " : "";
    kinds += sep + kKindNames[k] + " " + std::to_string(take(c.by_id_kind[k]));
    firsts += sep + kKindNames[k] + " " + std::to_string(take(c.first_binds[k]));
    reup += sep + kKindNames[k] + " " + std::to_string(take(c.reuploads[k]));
    wf += sep + kKindNames[k] + " " + std::to_string(take(c.watch_fires[k]));
  }
  size_t live = 0;
  {
    std::lock_guard<std::mutex> lock(t.mu);
    live = t.entries.size();
  }
  const uint64_t defines = take(c.defines), destroys = take(c.destroys), changed = take(c.changed), marked = take(c.changed_marked),
                 bind_items = take(c.bind_items), consumed = take(c.binds_consumed);
  std::string by_slot;
  for (uint32_t s = 0; s < 32; ++s) {
    const uint64_t n = take(c.fallback_slot[s]);
    if (n) by_slot += (by_slot.empty() ? "" : ", ") + std::to_string(s) + " x" + std::to_string(n);
  }
  if (!by_slot.empty()) REXLOG_INFO("[d3d][tex] fetch fallbacks by fetch slot: {}", by_slot);
  {
    std::string writers;
    for (uint32_t w = 0; w < kWriterCount; ++w) {
      const uint64_t tex = take(c.unbound_writes[w][0]), vtx = take(c.unbound_writes[w][1]);
      if (tex || vtx) writers += (writers.empty() ? "" : "; ") + std::string(kWriterNames[w]) + ": texture slots " + std::to_string(tex) + ", vertex slots " + std::to_string(vtx);
    }
    if (!writers.empty()) REXLOG_INFO("[d3d][tex] fetch constant writes that consumed no Bind, by writer: {}", writers);
  }
  {
    std::string nw;
    for (uint32_t w = 0; w < kWriterCount; ++w) {
      const uint64_t n = take(c.no_id_by_writer[w]);
      if (n) nw += (nw.empty() ? "" : "; ") + std::string(kWriterNames[w]) + " " + std::to_string(n);
    }
    if (!nw.empty()) REXLOG_INFO("[d3d][tex] no_id fallbacks (a texture fetch constant bound by no object) by the slot's last writer: {}", nw);
    const uint64_t recalled = take(c.recalled), never = take(c.no_id_never), fresh = take(c.no_id_new_content), dropped = take(c.watches_dropped),
                   outdated = take(c.destroyed_outdated), marks = take(c.shortcut_marks_miss);
    const uint64_t planes = take(c.plane_changed);
    const uint64_t left = take(c.pending_left), smis = take(c.shortcut_mismatch), dfail = take(c.drop_failed);
    {  // (always printed while the census runs: no line must not be read as zero)
      REXLOG_INFO("[d3d][tex] Binds a token ended without consuming (dropped) {}; remembered image differs from FindOrCreateTexture's (verify) {}; "
                  "watch drops that failed {}",
                  left, smis, dfail);
    }
    if (recalled || never || fresh || dropped || outdated || marks || planes) {
      REXLOG_INFO("[d3d][tex] no_id named: rewritten with the same dwords as the slot's last by-id bind (id kept) {}; never bound by id {}; other dwords "
                  "(a writer that sent no Bind) {}; static watches dropped {}; destroyed watch-less statics marked outdated {}; shortcut refused "
                  "after a scaled-resolve marks change {}; movie-plane objects marked changed by the plane flush {}",
                  recalled, never, fresh, dropped, outdated, marks, planes);
    }
  }
  if (by_id || fb_total || defines || destroys || bind_items) {
    // (Three lines: the log buffer cuts a line at about 730 characters.)
    REXLOG_INFO("[d3d][tex] frames {}-{} (mode {}{}): object table {} entries; defines {}, destroys {} (unknown {}), changed {} (marked {}, unknown {}); "
                "bind items {} consumed {}",
                window_start, frame, REXCVAR_GET(gpu_d3d_textures), REXCVAR_GET(gpu_d3d_textures_verify) ? ", verify" : "", live, defines,
                destroys, take(c.unknown_destroy), changed, marked, take(c.unknown_changed), bind_items, consumed);
    REXLOG_INFO("[d3d][tex] frames {}-{}: texture binds by id {} ({}); fetch fallbacks {}{}; key mismatches (verify) logged {}; "
                "object->texture hits {} / finds {}",
                window_start, frame, by_id, kinds, fb_total, fb_total ? ": " + fb : "", take(c.mismatch_logged), take(c.texture_hits),
                take(c.texture_finds));
    REXLOG_INFO("[d3d][tex] frames {}-{}: first binds (uploads) ({}); re-uploads from engine flushes ({}); memory watch fired on a registered "
                "object's texture ({}), on an unregistered one {}",
                window_start, frame, firsts, reup, wf, take(c.watch_fires_unregistered));
  }
  window_start = frame;
}

}  // namespace rex::graphics::d3d_tex
