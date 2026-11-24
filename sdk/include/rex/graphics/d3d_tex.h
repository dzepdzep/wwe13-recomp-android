// wwe2k14 patch (Session 87, D3D layer STAGE 4, textures in our own form; docs\native\d3d\S4_TEXTURES_CONTRACT.md in the game
// repository): the id -> texture-object identity table the game's registry (src\d3d_textures.h) feeds through the no-ring queue's
// Registers item, and the per-slot "bound object" state the texture cache reads in RequestTextures.
//
// SDK-internal: no game-exe header includes this file (the exe has its own copy of the op numbers, tied by the int capability cvar
// gpu_d3d_tex_abi). Nothing here changes the layout of a public SDK class.
//
// The ops (quads {op, a, b, c} in a Registers item; the consumer applies them once, on the drawing side):
//   16 Define0 {id, kind | flags << 8, word0}   17 Define1 {id, word1, word2}   18 Define2 {id, word3, word4}   19 Define3 {id, word5, 0}
//   20 Destroy {id}   21 Changed {id}   22 Bind {slot, id}
// Define carries an object's six header dwords as SetTexture would put them into a slot record (the image identity: base / mip
// address in GPU form, size, format, pitch, tiled; the sampler / view bits are the slot's, kept in the fetch constant).
// Bind says: the NEXT write of slot `slot`'s fetch constant comes from object `id`; the write consumes it (OnFetchWritten), so a slot
// written without a Bind (a fallback writer, the present, a ring packet) has no object.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rex::graphics::d3d_tex {

inline constexpr int32_t kAbi = 1;
inline constexpr uint32_t kOpDefine0 = 16, kOpDefine1 = 17, kOpDefine2 = 18, kOpDefine3 = 19, kOpDestroy = 20, kOpChanged = 21,
                          kOpBind = 22;

// Set by the first op the consumer applies: the fetch-constant notification (TextureCache::TextureFetchConstantsWritten) calls
// into the table only after that.
inline std::atomic<bool> g_active{false};

// Who is writing fetch constants right now (set by the packet handlers; the census of writes with no Bind names them).
enum Writer : uint32_t { kWRing = 0, kWToken, kWInline, kWCopy, kWRegsItem, kWFeed, kWriterCount };
inline constexpr const char* kWriterNames[kWriterCount] = {"ring packet / other", "record token 0x7C", "inline token 0x7E", "copy token 0x7D",
                                                           "Registers item", "stage 3 feed"};
inline thread_local Writer g_writer = kWRing;
void EndTokenScope();  // d3d_tex.cpp: the end of a token that applied fetch writes (Table::EndToken)
// The packet being executed (guest physical address of its header word) and the INDIRECT_BUFFER packet that called it (0: the ring),
// set by CommandProcessor::ExecutePacket while the table is active: the census of fetch writes with no Bind names them.
inline thread_local uint32_t g_packet_at = 0, g_ib_caller = 0;
struct WriterScope {
  Writer prev, mine;
  bool ends_binds;  // false for the answers side of the native frame (it runs the same token; only the drawing side owns the Binds)
  explicit WriterScope(Writer w, bool ends_binds_at_end = true) : prev(g_writer), mine(w), ends_binds(ends_binds_at_end) { g_writer = w; }
  ~WriterScope() {
    g_writer = prev;
    // A Bind belongs to the token (draw) it was pushed for: whatever that token did not consume is dropped at its end and counted.
    if (ends_binds && (mine == kWToken || mine == kWInline || mine == kWCopy) && g_active.load(std::memory_order_relaxed)) EndTokenScope();
  }
  WriterScope(const WriterScope&) = delete;
  WriterScope& operator=(const WriterScope&) = delete;
};

enum Kind : uint32_t { kUnknown = 0, kRenderTarget = 1, kFileStatic = 2, kEngineDynamic = 3, kKindCount };
inline constexpr const char* kKindNames[kKindCount] = {"unknown", "render_target", "file_static", "engine_dynamic"};

// Why a draw's texture slot was bound from its fetch constant and not by its object.
enum Fallback : uint32_t {
  kFbNoId = 0,      // the slot was written without a Bind (no object known for it)
  kFbUnknownId,     // the bound id is not in the table (retired, or never defined)
  kFbKeyMismatch,   // the object's key and the fetch constant's key disagree (the fetch constant wins)
  kFbBadObject,     // the object's words give no valid key
  kFbNotTexture,    // the fetch constant is not a texture (the old path's own invalid handling)
  kFbCount
};
inline constexpr const char* kFallbackNames[kFbCount] = {"no_id", "unknown_id", "key_mismatch", "bad_object", "not_texture"};

struct Entry {
  uint32_t id = 0;
  uint32_t kind = kUnknown;
  uint32_t words[6] = {};
  uint32_t have_mask = 0;    // bit k: Define part k seen
  bool key_known = false;    // key computed (valid or not)
  bool key_valid = false;
  alignas(8) uint8_t key[16] = {};  // a TextureCache::TextureKey (memcpy'd; the cache checks sizeof)
  uint32_t key_w4 = 0;       // the slot's word 4 (view bits) the key was derived with
  uint8_t key_requested[16] = {};  // the key FindOrCreateTexture was asked for when `texture` was found
  void* texture = nullptr;   // the host image found for the key last time (valid while texture_epoch == the table's epoch)
  uint64_t texture_epoch = 0;
  uint64_t texture_marks = 0;   // the scaled-resolve marks version `texture` was found under (a tiled texture at a scaled resolution)
  uint64_t binds = 0;
  bool loaded = false;       // its texture has been requested to load once (the first bind)
  bool changed_pending = false;
};

struct Census {
  std::atomic<uint64_t> defines{0}, destroys{0}, changed{0}, changed_marked{0}, bind_items{0}, binds_consumed{0};
  std::atomic<uint64_t> by_id{0}, by_id_kind[kKindCount] = {};
  std::atomic<uint64_t> fallback[kFbCount] = {};
  std::atomic<uint64_t> fallback_slot[32] = {};      // the same fallbacks by fetch slot
  std::atomic<uint64_t> fallback_logged{0};          // detail lines written (the first few)
  std::atomic<uint64_t> mismatch_logged{0};
  std::atomic<uint64_t> first_binds[kKindCount] = {};     // the first bind of an object (its upload)
  std::atomic<uint64_t> reuploads[kKindCount] = {};       // an object's texture marked outdated by the engine's flush
  std::atomic<uint64_t> watch_fires[kKindCount] = {};     // memory watch fired on a registered object's texture
  std::atomic<uint64_t> watch_fires_unregistered{0};
  std::atomic<uint64_t> texture_hits{0}, texture_finds{0};
  std::atomic<uint64_t> unknown_changed{0}, unknown_destroy{0};
  // no_id fallbacks, named: recalled = the slot was rewritten with the SAME six dwords the last by-id bind had (a replay / re-write with no Bind: the
  // id is kept); never = the slot has not been bound by id before; new_content = other dwords than the last by-id bind's
  std::atomic<uint64_t> recalled{0}, no_id_never{0}, no_id_new_content{0};
  std::atomic<uint64_t> watches_dropped{0}, destroyed_outdated{0}, shortcut_marks_miss{0};
  std::atomic<uint64_t> pending_left{0};       // Binds a token ended without consuming (dropped)
  std::atomic<uint64_t> shortcut_mismatch{0};  // verify: the remembered image is not what FindOrCreateTexture(key) returns now
  std::atomic<uint64_t> drop_failed{0};        // watch drops that failed (not retried)
  // Fetch constant writes that consumed no Bind, by who wrote them and whether the slot is a texture slot (< 26) or a vertex one.
  std::atomic<uint64_t> unbound_writes[kWriterCount][2] = {};
  std::atomic<uint64_t> unbound_logged{0};
  std::atomic<uint32_t> unbound_logged_w[kWriterCount] = {};  // log lines per writer kind
  std::atomic<uint64_t> no_id_by_writer[kWriterCount] = {};   // the fetch fallbacks (a texture-type fetch constant, no object) by the slot's last writer
  std::atomic<uint64_t> plane_changed{0};   // Changed ops that came from the movie planes' flush
};

class Table {
 public:
  static Table& Get();

  // The consumer's ops (command thread).
  void Define(uint32_t part, uint32_t id, uint32_t a, uint32_t b, uint32_t c);
  void Destroy(uint32_t id);
  void Changed(uint32_t id);
  void Bind(uint32_t slot, uint32_t id);
  // A fetch constant range was written (TextureCache::TextureFetchConstantsWritten): bound[slot] = pending[slot], pending cleared.
  // fetch_block (optional): dword 0 of fetch constant 0 in the register file the write went to (for the log line of a write with no Bind).
  void OnFetchWritten(uint32_t first, uint32_t last, const uint32_t* fetch_block = nullptr);

  // The texture cache's side. All of these take mu; none is called with the global critical region held except ForgetTexture
  // (a texture's destructor) and NoteWatchFire (the watch callback), which only touch the table.
  // Entry for id, copied out (false = unknown).
  bool Find(uint32_t id, Entry& out);
  // Stores what the cache derived/found for the entry.
  void StoreKey(uint32_t id, const uint8_t key16[16], uint32_t w4, bool valid);
  void StoreTexture(uint32_t id, void* texture, const uint8_t key16[16], uint64_t epoch, uint64_t marks);
  // The slot's last by-id bind (cache thread): a rewrite with the same six dwords and no Bind keeps its object.
  uint32_t Recall(uint32_t slot, const uint32_t words[6], bool& never) const;
  void Remember(uint32_t slot, uint32_t id, const uint32_t words[6]);
  // Statics whose first upload happened: their memory watch can go once the load is done (TakeDrops / the cache).
  void TakeDrops(std::vector<std::pair<const void*, uint32_t>>& out);
  void RequeueDrop(const void* texture, uint32_t kind);
  void MarkWatchDropped(const void* texture);
  // The texture was queued for a load: first load of the object = a first bind (upload); later = a re-upload.
  void NoteLoad(const void* texture);
  void NoteWatchFire(const void* texture);
  // A token that applied fetch writes ended: Binds it left pending are dropped and counted (a Bind belongs to its draw).
  void EndToken();
  void ForgetTexture(const void* texture);
  // Textures of the entries the engine flushed since the last call (changed_pending), cleared.
  void TakeChanged(std::vector<std::pair<const void*, uint32_t>>& out);

  // The slot's last writer, for naming a no_id fallback.
  Writer LastWriter(uint32_t slot) const { return slot < 32 ? last_writer_[slot] : kWRing; }
  bool LastHadBind(uint32_t slot) const { return slot < 32 && last_had_bind_[slot]; }
  uint32_t BoundId(uint32_t slot) const { return slot < 32 ? bound_[slot] : 0; }
  // A bump of this makes every slot's cached "object" decision stale (cleared by OnFetchWritten per slot; read by the cache).
  uint64_t epoch() const { return epoch_.load(std::memory_order_relaxed); }
  void BumpEpoch() { epoch_.fetch_add(1, std::memory_order_relaxed); }

  std::mutex mu;
  std::unordered_map<uint32_t, Entry> entries;   // guarded by mu
  // texture (opaque host-image pointer) -> object id and kind, for the watch-fire census (any thread)
  std::unordered_map<const void*, uint32_t> texture_to_id;  // guarded by mu
  Census census;
  std::atomic<bool> changed_any{false};   // a Changed op (or a destroyed watch-less static) is waiting for the cache (TakeChanged)
  std::atomic<bool> drop_any{false};      // a first upload of a static is waiting for its watch to go (TakeDrops)
  // Bumped whenever the scaled-resolve marks of the texture cache change (a range newly marked, or cleared by a CPU write): the answer of
  // FindOrCreateTexture for a tiled texture at a scaled draw resolution is a function of the key AND of these marks.
  std::atomic<uint64_t> marks_version{1};

 private:
  uint32_t pending_[32] = {};
  uint32_t last_id_[32] = {};
  Writer last_writer_[32] = {};     // who wrote each slot last (OnFetchWritten), with or without a Bind
  bool last_had_bind_[32] = {};
  uint32_t last_words_[32][6] = {};
  std::vector<std::pair<const void*, uint32_t>> drop_pending_;   // guarded by mu
  std::vector<std::pair<const void*, uint32_t>> outdate_pending_;   // guarded by mu (retired watch-less statics)
  std::unordered_set<const void*> dropped_textures_;                // guarded by mu: host images whose memory watch was dropped
  // The entry's image goes away (replaced, redefined, destroyed): a watch-less one is outdated, the id map forgets it. Lock held.
  void RetireTexture(const void* texture, uint32_t kind);
  uint32_t bound_[32] = {};
  std::atomic<uint64_t> epoch_{1};
};

bool Enabled();         // cvar gpu_d3d_textures >= 1
bool UseIds();          // cvar gpu_d3d_textures >= 2
bool VerifyKeys();      // cvar gpu_d3d_textures_verify
bool DropWatches();     // cvar gpu_d3d_textures_drop_watch
void LogCensus(uint32_t frame);   // the [d3d][tex] line every gpu_d3d_layer_log_frames frames
// (Session 95, gpu_d3d_wall bit 4) Whether an object was defined or destroyed since the last call (command thread).
bool TakeObjectsChanged();
// (Session 95, gpu_d3d_wall bit 16) Counts the Defines and Destroys: what a key kept from an object's words is valid for.
uint64_t ObjectsGeneration();

}  // namespace rex::graphics::d3d_tex
