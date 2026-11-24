// Heap diagnostics: the game carves its memory into fixed heaps at boot (sub_82B17448, called from sub_82743BE0 with
// one region and a table of heap sizes) and allocates from the current thread's heap (sub_82B17018, sub_82B184C0).
// A heap that runs out returns 0, and the game writes through the null pointer (match load with ported moves).
// Logs the heap layout once at boot and every failed allocation (heap number, size, caller).
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <fmt/format.h>
#include <unordered_map>
#include <map>
#include <mutex>
#include <vector>

#include <rex/logging.h>
#include <rex/ppc/context.h>
#include "rex_compat.h"

namespace {
constexpr uint32_t kHeapCount = 0x835B5124;  // (lis -31909 = 0x835B, 20772)
constexpr uint32_t kHeapTable = 0x835B5120;  // (20768): 36-byte records, [0] blocks start, [4] end

void LogHeaps(uint8_t* base, const char* why) {
  const uint32_t n = PPC_LOAD_U32(kHeapCount);
  const uint32_t t = PPC_LOAD_U32(kHeapTable);
  REXLOG_INFO("[wwe13-heap] {}: {} heaps, table 0x{:08X}", why, n, t);
  for (uint32_t i = 0; i < n && i < 64 && t; ++i) {
    const uint32_t s = PPC_LOAD_U32(t + i * 36), e = PPC_LOAD_U32(t + i * 36 + 4);
    // free 16-byte units: walk the free list ([8] next index; [0] low 28 bits = end index of the block)
    uint64_t free_units = 0, largest = 0;
    uint32_t idx = s ? PPC_LOAD_U32(s + 8) : 0, guard = 0;
    const uint32_t units = (e - s) >> 4;
    while (s && idx && idx < units && guard++ < 100000) {
      const uint32_t b = s + idx * 16;
      const uint32_t end = PPC_LOAD_U32(b) & 0x0FFFFFFF;
      const uint32_t sz = end > idx ? end - idx : 0;
      free_units += sz;
      if (sz > largest) largest = sz;
      idx = PPC_LOAD_U32(b + 8);
    }
    REXLOG_INFO("[wwe13-heap]   heap {:2}: 0x{:08X}-0x{:08X} ({:7} KB), free ~{} KB, largest ~{} KB", i, s, e,
                (e - s) >> 10, (free_units * 16) >> 10, (largest * 16) >> 10);
  }
}

}  // namespace

// Heap 12 holds a match's motion groups (mot0X.pac ZMVM entries, read whole: SPE0 is 9.6 MB). Moves ported from WWE 2K14
// make the groups bigger than its 11 MB: loads failed and were retried (stalls in entrances). Heaps 34-39 (1.2 MB each)
// are per-superstar; the ported moves overfilled one after the entrances (crash). They get more from the last heap (the
// region left after the table, 10.9 MB, unused in a match) (heap 4 is left alone: the attires built at match load use 2 MB blocks of it and the Move-Set editor an 8 MB one; menus take at
// most 2 MB of it at a time). Heap 43 gets 7 MB (3 MB from heap 11, about 2.5 MB used in menus and matches, heap 20 being full in Attitude Era; 4 MB from heap 54, at least 16 MB free in every mode tested; the region after it is only 401 KB, so the sizes must add up): a Royal Rumble entrant needs a 4.6 MB block there, and Create-a-Moveset
// > Special Moves loads five motion sets there and the mod's are up to 11% bigger. WWE13_HEAP_GROW="<heap>:<KB>[,...]" (KB may be negative) overrides; "" = none.
namespace {
void GrowHeaps(uint8_t* base, uint32_t table) {
  const char* env = std::getenv("WWE13_HEAP_GROW");
  std::string spec = env ? env : "12:7424,34:512,35:512,36:512,37:512,38:512,39:512,43:7168,11:-3072,54:-4096";
  size_t p = 0;
  while (p < spec.size()) {
    size_t q = spec.find(',', p);
    if (q == std::string::npos) q = spec.size();
    const std::string item = spec.substr(p, q - p);
    p = q + 1;
    const size_t c = item.find(':');
    if (c == std::string::npos) continue;
    const uint32_t heap = uint32_t(std::atoi(item.c_str()));
    const int32_t kb = std::atoi(item.c_str() + c + 1);
    if (heap > 54 || !kb) continue;
    const uint32_t old = PPC_LOAD_U32(table + heap * 4);
    if (old == 0xFFFFFFFFu) continue;
    if (kb < 0 && uint32_t(-kb) >= old) continue;
    PPC_STORE_U32(table + heap * 4, uint32_t(int32_t(old) + kb));
    REXLOG_INFO("[wwe13-heap] heap {}: {} KB -> {} KB", heap, old, int32_t(old) + kb);
  }
}
}  // namespace

extern "C" PPC_FUNC(__imp__sub_82B17448);
extern "C" PPC_FUNC(sub_82B17448) {
  const uint32_t desc = ctx.r3.u32, count = ctx.r4.u32;
  GrowHeaps(base, PPC_LOAD_U32(desc + 8));
  REXLOG_INFO("[wwe13-heap] init: region 0x{:08X}-0x{:08X}, size table 0x{:08X}, {} heaps", PPC_LOAD_U32(desc),
              PPC_LOAD_U32(desc + 4), PPC_LOAD_U32(desc + 8), count);
  {
    std::string sizes;
    const uint32_t t = PPC_LOAD_U32(desc + 8);
    for (uint32_t i = 0; t && i < 64; ++i) {
      const uint32_t v = PPC_LOAD_U32(t + i * 4);
      sizes += std::to_string(int32_t(v)) + " ";
      if (v == 0xFFFFFFFFu) break;
    }
    REXLOG_INFO("[wwe13-heap] size table (KB): {}", sizes);
  }
  __imp__sub_82B17448(ctx, base);
  LogHeaps(base, "after init");
}

namespace {
struct Took { uint32_t units; uint32_t lr[4]; };
std::mutex g_took_mu;
std::map<uint32_t, Took> g_took;
}  // namespace
extern "C" PPC_FUNC(__imp__sub_82B16960);  // the calling thread's heap number
namespace {
void Failed(PPCContext& ctx, uint8_t* base, uint32_t bytes, uint64_t lr, const char* fn) {
  static uint32_t seen[64] = {};  // (each failing size once: the game retries the motion loads)
  static int nseen = 0;
  for (int i = 0; i < nseen; ++i) {
    if (seen[i] == bytes) return;
  }
  if (nseen >= 64) return;
  seen[nseen++] = bytes;
  PPCContext saved = ctx;
  __imp__sub_82B16960(ctx, base);
  const uint32_t heap = ctx.r3.u32;
  ctx = saved;
  REXLOG_ERROR("[wwe13-heap] {}: {} bytes failed in heap {} (caller 0x{:08X})", fn, bytes, heap, uint32_t(lr));
  if (heap == 12) {  // who holds heap 12: the biggest live blocks and their callers
    std::vector<std::pair<uint32_t, std::string>> v;
    {
      std::lock_guard<std::mutex> lk(g_took_mu);
      const uint32_t t = PPC_LOAD_U32(kHeapTable);
      const uint32_t s = t ? PPC_LOAD_U32(t + 12 * 36) : 0, e = t ? PPC_LOAD_U32(t + 12 * 36 + 4) : 0;
      uint64_t total = 0;
      for (auto& [a, k] : g_took) {
        if (a < s || a >= e) continue;
        total += k.units * 16ull;
        v.emplace_back(k.units * 16, fmt::format("{} KB at 0x{:08X} by {:08X} < {:08X} < {:08X} < {:08X}", k.units / 64, a,
                                                 k.lr[0], k.lr[1], k.lr[2], k.lr[3]));
      }
      std::sort(v.begin(), v.end(), [](auto& x, auto& y) { return x.first > y.first; });
      REXLOG_INFO("[wwe13-heap] heap 12 holds {} recorded blocks, {} KB:", v.size(), total >> 10);
    }
    for (size_t i = 0; i < v.size() && i < 20; ++i) REXLOG_INFO("[wwe13-heap]   {}", v[i].second);
  }
  LogHeaps(base, "at the failure");
}
}  // namespace

// Motion arenas sized from the game's fixed table (sub_828F3460 with r5 != 0: table 0x820536C8, e.g. kind 36 =
// 4271664 bytes for the Special Moves screen of Create-a-Moveset). The mod's motions for such a set are bigger, the
// load runs past the end into heap 43's next block and the game crashed in the allocator. These arenas get
// WWE13_STATIC_ARENA_PAD percent more (default 14, at most 768 KB; kind 17 was seen using 3% more, kind 36 11%; 0 = only the flat 256 KB). The tail is filled with a marker and
// checked now and then: the furthest write past the game's size is logged, to size the pad.
thread_local uint32_t g_static_arena_extra = 0;
struct ArenaWatch { uint32_t ptr, size, extra, high; };
std::mutex g_arena_mu;
std::vector<ArenaWatch> g_arena_watch;
constexpr uint32_t kArenaMark = 0x5AFE0A1Du;
void ArenaWatchAdd(uint8_t* base, uint32_t ptr, uint32_t size, uint32_t extra) {
  for (uint32_t o = size & ~3u; o + 4 <= size + extra; o += 4) PPC_STORE_U32(ptr + o, kArenaMark);
  std::lock_guard<std::mutex> lk(g_arena_mu);
  g_arena_watch.push_back({ptr, size, extra, 0});
  static int logged = 0;
  if (logged++ < 40) REXLOG_INFO("[wwe13-heap] motion arena at {:08X}: {} + {} bytes", ptr, size, extra);
}
void ArenaWatchScan(uint8_t* base, uint32_t only_ptr) {
  std::lock_guard<std::mutex> lk(g_arena_mu);
  for (size_t i = 0; i < g_arena_watch.size();) {
    ArenaWatch& w = g_arena_watch[i];
    if (!only_ptr || w.ptr == only_ptr) {
      uint32_t hi = 0;
      for (uint32_t o = (w.size + w.extra - 4) & ~3u; o >= (w.size & ~3u); o -= 4)
        if (PPC_LOAD_U32(w.ptr + o) != kArenaMark) { hi = o + 4 - w.size; break; }
      if (hi > w.high || (only_ptr && !w.high)) {
        w.high = hi;
        REXLOG_WARN("[wwe13-heap] motion arena {:08X} ({} bytes by the game's table): used {} bytes past it (pad {})",
                    w.ptr, w.size, hi, w.extra);
      }
    }
    if (only_ptr && w.ptr == only_ptr) { g_arena_watch.erase(g_arena_watch.begin() + i); continue; }
    ++i;
  }
}
extern "C" void wwe13_arena_watch_tick(uint8_t* base) {
  static uint32_t n = 0;
  if ((++n & 4095) == 0 && !g_arena_watch.empty()) ArenaWatchScan(base, 0);
}

bool CheckHeap(uint8_t* base, uint32_t heap, uint32_t caller);
void RecordBlock(uint8_t* base, uint32_t got, uint32_t units, uint32_t lr, uint32_t sp);
extern "C" PPC_FUNC(__imp__sub_82B17018);
extern "C" PPC_FUNC(sub_82B173A0);
extern "C" PPC_FUNC(sub_82B17018) {
  {  // the motion buffer's other allocation path (0x82379494, size in bytes): padded like the one in sub_82B184C0
    static const bool pad = [] { const char* e = std::getenv("WWE13_MOTION_BUFFER_PAD"); return !e || *e != '0'; }();
    bool in43 = false;
    if (pad && ctx.r3.u32 >= 0x40000) {
      PPCContext c = ctx;
      __imp__sub_82B16960(c, base);
      in43 = c.r3.u32 == 43;
    }
    if (pad && ctx.r3.u32 && (uint32_t(ctx.lr) == 0x82379494u || in43)) {
      const uint32_t b = ctx.r3.u32;
      // (an arena sized from the game's fixed table, see sub_828F3460 below, asks for more than the flat 256 KB)
      uint32_t want = 0x40000;
      if (uint32_t(ctx.lr) == 0x82379494u && g_static_arena_extra > want) want = g_static_arena_extra;
      g_static_arena_extra = 0;
      static int logged = 0;
      // Tries: the padded size in the current heap (43); then, for a motion arena, the padded size in heap 12 (the
      // pac cache: free in the menus; the Special Moves screen of Create-a-Moveset loads more motion sets than heap
      // 43 holds once they are the mod's sizes); then +256 KB; then the game's size.
      const bool arena = uint32_t(ctx.lr) == 0x82379494u;
      struct Try { uint32_t extra; uint32_t heap; };
      // (heap 12 only with WWE13_STATIC_ARENA_HEAP12=1: leaving the editor for the main menu froze once with an
      // arena there)
      static const bool h12 = [] { const char* e = std::getenv("WWE13_STATIC_ARENA_HEAP12"); return e && *e == '1'; }();
      const Try tries[] = {{want, 0}, {want, arena && h12 ? 12u : 0u}, {0x40000u, 0}};
      for (const Try& t : tries) {
        if (t.heap == 0 && &t == &tries[1]) continue;
        if (&t == &tries[2] && want == 0x40000u) break;
        ctx.r3.u64 = b + t.extra;
        if (logged++ < 10) REXLOG_INFO("[wwe13-heap] motion buffer {} bytes padded to {} (byte path)", b, ctx.r3.u32);
        PPCContext c = ctx;
        uint32_t was = 0;
        if (t.heap) {
          PPCContext h = ctx;
          __imp__sub_82B16960(h, base);
          was = h.r3.u32;
          h.r3.u64 = t.heap;
          sub_82B173A0(h, base);
        }
        __imp__sub_82B17018(c, base);
        if (t.heap) {
          PPCContext h = ctx;
          h.r3.u64 = was;
          sub_82B173A0(h, base);
          static int logged12 = 0;
          if (logged12++ < 20)
            REXLOG_WARN("[wwe13-heap] motion arena {} bytes: heap 43 full, heap {}: {}", ctx.r3.u32, t.heap,
                        c.r3.u32 ? fmt::format("{:08X}", c.r3.u32) : std::string("no room either"));
        }
        if (c.r3.u32) {
          RecordBlock(base, c.r3.u32, (ctx.r3.u32 + 15) / 16, uint32_t(ctx.lr), ctx.r1.u32);
          if (t.extra > 0x40000) ArenaWatchAdd(base, c.r3.u32, b, t.extra);
          ctx.r3.u64 = c.r3.u64;
          return;
        }
      }
      ctx.r3.u64 = b;
    }
  }
  const uint32_t bytes = ctx.r3.u32;
  const uint64_t lr = ctx.lr;
  {
    PPCContext c = ctx;
    __imp__sub_82B16960(c, base);
    if (!CheckHeap(base, c.r3.u32, uint32_t(lr))) {
      ctx.r3.u64 = 0;
      return;
    }
  }
  const uint32_t sp0 = ctx.r1.u32;
  __imp__sub_82B17018(ctx, base);
  if (ctx.r3.u32) RecordBlock(base, ctx.r3.u32, (bytes + 15) / 16, uint32_t(lr), sp0);
  if (bytes && ctx.r3.u32 == 0) Failed(ctx, base, bytes, lr, "alloc");
}

extern "C" PPC_FUNC(__imp__sub_82B184C0);
// Heap 43 watch (Create-a-Moveset preview: heap 43's blocks get overwritten). Records who took each heap-43 block
// (guest call chain) and walks the block chain every 32 allocations there; at the first break it logs the block
// before it and who took that one (the likely overflow). WWE13_HEAP43_WATCH=0 turns it off.
bool CheckHeap(uint8_t* base, uint32_t heap, uint32_t caller);

bool g_watch_done = false;
void RecordBlock(uint8_t* base, uint32_t got, uint32_t units, uint32_t lr, uint32_t sp) {
  if (got < 16) return;
  const uint8_t heap = PPC_LOAD_U32(got - 4) & 0xff;
  if (heap != 43 && heap != 54 && heap != 12) return;
  Took t{units, {lr, 0, 0, 0}};
  for (int k = 1; k < 4 && sp; ++k) {
    sp = PPC_LOAD_U32(sp);
    if (!sp || sp < 0x70000000) break;
    t.lr[k] = PPC_LOAD_U32(sp - 8);
  }
  std::lock_guard<std::mutex> lk(g_took_mu);
  g_took[got] = t;
}
void Heap43Check(uint8_t* base, const char* when, uint32_t H) {
  const uint32_t t = PPC_LOAD_U32(kHeapTable);
  if (!t || PPC_LOAD_U32(kHeapCount) <= H) return;
  const uint32_t s = PPC_LOAD_U32(t + H * 36), e = PPC_LOAD_U32(t + H * 36 + 4);
  if (!s || e <= s) return;
  const uint32_t units = (e - s) >> 4;
  uint32_t idx = 0, prev = 0, steps = 0;
  while (idx < units) {
    const uint32_t next = PPC_LOAD_U32(s + idx * 16) & 0x0FFFFFFF;
    if (next <= idx || next > units || ++steps > units) {
      g_watch_done = true;
      auto rec = [&](uint32_t i) {
        // the record whose pointer lies in block i (aligned allocations return a pointer inside the block)
        const uint32_t blk = s + i * 16, nxt = s + (PPC_LOAD_U32(blk) & 0x0FFFFFFF) * 16;
        auto it = g_took.lower_bound(blk + 16);
        if (it == g_took.end() || it->first >= std::max(nxt, blk + 32)) return std::string("(no record)");
        return fmt::format("{} bytes by {:08X} < {:08X} < {:08X} < {:08X}", it->second.units * 16, it->second.lr[0],
                           it->second.lr[1], it->second.lr[2], it->second.lr[3]);
      };
      const uint32_t b = s + idx * 16, pb = s + prev * 16;
      REXLOG_WARN("[wwe13-heap] heap {} block chain broken at block {:#x} (0x{:08X}: {:08X} {:08X} {:08X} {:08X}), "
                  "seen {}", H, idx, b, PPC_LOAD_U32(b), PPC_LOAD_U32(b + 4), PPC_LOAD_U32(b + 8), PPC_LOAD_U32(b + 12), when);
      REXLOG_WARN("[wwe13-heap]   block before: {:#x} at 0x{:08X} ({:08X} {:08X} {:08X} {:08X}): {}", prev, pb,
                  PPC_LOAD_U32(pb), PPC_LOAD_U32(pb + 4), PPC_LOAD_U32(pb + 8), PPC_LOAD_U32(pb + 12), rec(prev));
      REXLOG_WARN("[wwe13-heap]   broken block's own record: {}", rec(idx));
      // the nearest recorded blocks below the break (an overflow runs upward)
      int shown = 0;
      for (uint32_t back = 16; back <= 0x40000 && shown < 6; back += 16) {
        if (b < s + back) break;
        auto it = g_took.find(b - back + 16);
        if (it != g_took.end() && b - back + 16 + it->second.units * 16 > b - 64) {
          REXLOG_WARN("[wwe13-heap]   block at 0x{:08X} reaches 0x{:08X}: {}", b - back, b - back + 16 + it->second.units * 16,
                      rec((b - back - s) >> 4));
          ++shown;
        }
      }
      return;
    }
    prev = idx;
    idx = next;
  }
}

// sub_828F3460(this, kind, fixed): the size of a motion arena; fixed != 0 takes it from the game's table.
extern "C" PPC_FUNC(__imp__sub_828F3460);
extern "C" PPC_FUNC(sub_828F3460) {
  const uint32_t kind = ctx.r4.u32, fixed = ctx.r5.u32;
  __imp__sub_828F3460(ctx, base);
  static const uint32_t pct = [] {
    const char* e = std::getenv("WWE13_STATIC_ARENA_PAD");
    return uint32_t(e && *e ? std::atoi(e) : 14);
  }();
  static const int only = [] {  // (WWE13_STATIC_ARENA_ONLY=<kind>: pad just that kind, for measuring)
    const char* e = std::getenv("WWE13_STATIC_ARENA_ONLY");
    return e && *e ? std::atoi(e) : -1;
  }();
  if (fixed && pct && ctx.r3.u32 >= 0x40000 && (only < 0 || uint32_t(only) == kind)) {
    // (+pct%, at least 256 KB, at most 768 KB: measured use past the game's size was 0-483 KB)
    g_static_arena_extra =
        std::min<uint32_t>(0xC0000u, std::max<uint32_t>(0x40000u, uint32_t(uint64_t(ctx.r3.u32) * pct / 100)));
    static int logged = 0;
    if (logged++ < 20)
      REXLOG_INFO("[wwe13-heap] motion arena kind {}: {} bytes by the game's table, +{}", kind, ctx.r3.u32,
                  g_static_arena_extra);
  }
}
// sub_82B17970(ptr, ...): frees a block; a watched arena gets its last check first.
extern "C" PPC_FUNC(__imp__sub_82B17970);
extern "C" PPC_FUNC(sub_82B17970) {
  if (ctx.r3.u32 && !g_arena_watch.empty()) ArenaWatchScan(base, ctx.r3.u32);
  __imp__sub_82B17970(ctx, base);
}
extern "C" PPC_FUNC(sub_82B184C0) {
  // Motion bank buffers (allocated from 0x823794A8: Create-a-Moveset / preview loads): the game sizes them from its
  // own figures, and the mod's bigger banks (mot06 BASE bank 45: 1984322 bytes in a 1962816-byte buffer) ran past
  // the end into the next block's header (heap 43 broke). They get 256 KB more (the original size if that fails).
  // WWE13_MOTION_BUFFER_PAD=0: off.
  {
    static const bool pad = [] { const char* e = std::getenv("WWE13_MOTION_BUFFER_PAD"); return !e || *e != '0'; }();
    if (uint32_t(ctx.lr) == 0x823794A8u) g_static_arena_extra = 0;  // (that arena takes the unit path below)
    if (pad && ctx.r3.u32 && uint32_t(ctx.lr) == 0x823794A8u) {
      const uint32_t u = ctx.r3.u32;
      ctx.r3.u64 = u + 0x4000;  // (+256 KB: the mod's banks are at most 300 KB bigger)
      static int logged = 0;
      if (logged++ < 10)
        REXLOG_INFO("[wwe13-heap] motion buffer {} bytes padded to {}", u * 16, ctx.r3.u32 * 16);
      PPCContext c = ctx;
      __imp__sub_82B184C0(c, base);  // (padded; when the heap has no room for it, the original size)
      if (c.r3.u32) {
        RecordBlock(base, c.r3.u32, ctx.r3.u32, uint32_t(ctx.lr), ctx.r1.u32);
        ctx.r3.u64 = c.r3.u64;
        return;
      }
      ctx.r3.u64 = u;
    }
  }
  const uint32_t units = ctx.r3.u32;
  const uint64_t lr = ctx.lr;
  static const bool watch = [] { const char* e = std::getenv("WWE13_HEAP43_WATCH"); return !e || *e != '0'; }();
  Took took{units, {uint32_t(lr), 0, 0, 0}};
  if (watch) {
    uint32_t sp = ctx.r1.u32;
    for (int k = 1; k < 4 && sp; ++k) {
      sp = PPC_LOAD_U32(sp);
      if (!sp || sp < 0x70000000) break;
      took.lr[k] = PPC_LOAD_U32(sp - 8);
    }
  }
  {
    PPCContext c = ctx;
    __imp__sub_82B16960(c, base);
    if (!CheckHeap(base, c.r3.u32, uint32_t(lr))) {
      ctx.r3.u64 = 0;
      return;
    }
  }
  const uint32_t sp184 = ctx.r1.u32;
  __imp__sub_82B184C0(ctx, base);
  if (watch && ctx.r3.u32) RecordBlock(base, ctx.r3.u32, units, uint32_t(lr), sp184);
  if (watch && ctx.r3.u32 && !g_watch_done) {
    const uint32_t got = ctx.r3.u32;
    PPCContext saved = ctx;
    __imp__sub_82B16960(ctx, base);
    const uint32_t heap = ctx.r3.u32;
    ctx = saved;
    static std::atomic<uint32_t> n{0};
    if ((++n & 15) == 0) {
      std::lock_guard<std::mutex> lk(g_took_mu);
      if (!g_watch_done) Heap43Check(base, "after an allocation", 43);
    }
  }
  if (units && ctx.r3.u32 == 0) {
    Failed(ctx, base, units * 16, lr, "alloc16");
  } else if (units >= (1u << 16)) {  // 1 MB or more: which big blocks a match takes
    const uint32_t got = ctx.r3.u32;
    PPCContext saved = ctx;
    __imp__sub_82B16960(ctx, base);
    const uint32_t heap = ctx.r3.u32;
    ctx = saved;
    REXLOG_INFO("[wwe13-heap] big alloc: {} KB in heap {} at 0x{:08X} (caller 0x{:08X})", units >> 6, heap, got,
                uint32_t(lr));
  }
}

// sub_82B18630(a, b, align): allocates a + b bytes (the big blocks go through here); logs who asks for 1 MB or more.
extern "C" PPC_FUNC(__imp__sub_82B18630);
extern "C" PPC_FUNC(sub_82B18630) {
  const uint32_t a = ctx.r3.u32, b = ctx.r4.u32;
  const uint64_t lr = ctx.lr;
  PPCContext in = ctx;
  __imp__sub_82B18630(ctx, base);
  // The pac cache (sub_82B3CBA0, heap 12) retries a failed load forever, evicting what it can. In Create-an-Entrance
  // its blocks plus the editor's own fill heap 12 and a 2-3 MB pac never fits: the game hung on the loading screen.
  // After 64 failures in a row for the same request the block is taken from another heap with room (free in the
  // menus); the cache frees it there later (blocks carry their heap). WWE13_CACHE_HEAP_FALLBACK=0: off.
  if (uint32_t(lr) == 0x82B3CC08u) {
    static const bool fb = [] { const char* e = std::getenv("WWE13_CACHE_HEAP_FALLBACK"); return !e || *e != '0'; }();
    thread_local uint32_t last_a = 0, last_b = 0, fails = 0;
    if (ctx.r3.u32) {
      fails = 0;
    } else if (fb) {
      if (a == last_a && b == last_b) ++fails; else last_a = a, last_b = b, fails = 1;
      if (fails == 64 || fails == 1000 || fails == 100000)
        REXLOG_WARN("[wwe13-heap] pac cache: {} failed loads in a row of {} + {} bytes", fails, a, b);
      if (fails >= 64) {
        PPCContext h = in;
        __imp__sub_82B16960(h, base);
        const uint32_t was = h.r3.u32;
        for (uint32_t heap : {4u, 20u, 11u, 15u, 18u}) {
          PPCContext s2 = in;
          s2.r3.u64 = heap;
          sub_82B173A0(s2, base);
          PPCContext c = in;
          __imp__sub_82B18630(c, base);
          PPCContext r2 = in;
          r2.r3.u64 = was;
          sub_82B173A0(r2, base);
          if (fails == 64) REXLOG_WARN("[wwe13-heap] pac cache: heap {} -> {:08X}", heap, c.r3.u32);
          if (c.r3.u32) {
            static int logged = 0;
            if (logged++ < 20)
              REXLOG_WARN("[wwe13-heap] pac cache: {} KB never fit in heap {} - taken from heap {} at {:08X}",
                          (a + b) >> 10, was, heap, c.r3.u32);
            ctx.r3.u64 = c.r3.u64;
            fails = 0;
            break;
          }
        }
      }
    }
  }
  if (a + b >= (1u << 20)) {
    REXLOG_INFO("[wwe13-heap] big block {} KB -> 0x{:08X} for 0x{:08X}", (a + b) >> 10, ctx.r3.u32, uint32_t(lr));
  }
}

// sub_821FA640(obj, pach, size): reads a move database container (m3.pac MDAT/PRM: parts 0-9). Logs its part table.
extern "C" PPC_FUNC(__imp__sub_821FA640);
extern "C" PPC_FUNC(sub_821FA640) {
  const uint32_t data = ctx.r4.u32, size = ctx.r5.u32;
  const uint64_t lr = ctx.lr;
  if (data) {
    std::string parts;
    const uint32_t magic = PPC_LOAD_U32(data);
    const uint32_t n = __builtin_bswap32(PPC_LOAD_U32(data + 4));
    for (uint32_t i = 0; i < n && i < 16; ++i) {
      const uint32_t id = __builtin_bswap32(PPC_LOAD_U32(data + 8 + 12 * i));
      const uint32_t off = __builtin_bswap32(PPC_LOAD_U32(data + 12 + 12 * i));
      const uint32_t sz = __builtin_bswap32(PPC_LOAD_U32(data + 16 + 12 * i));
      const uint32_t at = data + 8 + 12 * n + off;
      parts += std::to_string(id) + ":" + std::to_string(off) + "+" + std::to_string(sz) + "[" +
               std::to_string(PPC_LOAD_U32(at) == 0 ? 0 : 1) + "] ";
      if (id == 7) {
        REXLOG_INFO("[wwe13-heap] PRM part 7 at 0x{:08X}: {:08X} {:08X}", at, PPC_LOAD_U32(at), PPC_LOAD_U32(at + 4));
      }
    }
    REXLOG_INFO("[wwe13-heap] PRM container 0x{:08X} size {} magic {:08X}, {} parts: {}(caller 0x{:08X})", data, size,
                magic, n, parts, uint32_t(lr));
  }
  __imp__sub_821FA640(ctx, base);
}

// Start of one of the game's heaps (0 before they are made).
uint32_t wwe13_heap_start(uint8_t* base, uint32_t heap) {
  const uint32_t t = PPC_LOAD_U32(kHeapTable);
  if (!t || heap >= PPC_LOAD_U32(kHeapCount)) return 0;
  return PPC_LOAD_U32(t + heap * 36);
}

// sub_82B17218(p): frees p into heap (*(p-4) & 0xff). It walks the heap's free list (block index order, [8] = next
// index) to insert the block; a list that loops makes that walk spin forever (the Move-Set editor froze in it: one
// guest thread at 100% in __imp__sub_82B17218, every other thread waiting). Before freeing, the list is walked with a
// step limit: a block already on the list (double free) or a list that loops is logged with the caller and the free is
// skipped (the block leaks instead of the game hanging). WWE13_HEAP_FREE_CHECK=0 turns the check off.
extern "C" PPC_FUNC(__imp__sub_82B17218);
extern "C" PPC_FUNC(sub_82B17218) {
  if (ctx.r3.u32 >= 16 && ((PPC_LOAD_U32(ctx.r3.u32 - 4) & 0xff) == 43 || (PPC_LOAD_U32(ctx.r3.u32 - 4) & 0xff) == 54 ||
                            (PPC_LOAD_U32(ctx.r3.u32 - 4) & 0xff) == 12)) {
    std::lock_guard<std::mutex> lk(g_took_mu);
    g_took.erase(ctx.r3.u32);
  }
  static const bool on = [] { const char* e = std::getenv("WWE13_HEAP_FREE_CHECK"); return !e || *e != '0'; }();
  const uint32_t p = ctx.r3.u32;
  if (on && p >= 16) {
    const uint32_t heap = PPC_LOAD_U32(p - 4) & 0xff;
    const uint32_t t = PPC_LOAD_U32(kHeapTable);
    if (t && heap < PPC_LOAD_U32(kHeapCount)) {
      const uint32_t s = PPC_LOAD_U32(t + heap * 36), e = PPC_LOAD_U32(t + heap * 36 + 4);
      const uint32_t blk = p - 16;
      const char* bad = nullptr;
      if (blk < s || blk >= e) {
        bad = "pointer outside its heap";
      } else {
        const uint32_t me = (blk - s) >> 4, units = (e - s) >> 4;
        uint32_t idx = PPC_LOAD_U32(s + 8), prev = 0, steps = 0;
        while (idx && idx < units) {
          if (idx == me) { bad = "block is already free (double free)"; break; }
          if (prev && idx <= prev) { bad = "free list loops"; break; }
          if (++steps > units) { bad = "free list too long"; break; }
          prev = idx;
          idx = PPC_LOAD_U32(s + idx * 16 + 8);
        }
      }
      if (bad) {
        static int logged = 0;
        if (logged++ < 64)
          REXLOG_ERROR("[wwe13-heap] free 0x{:08X} heap {}: {} - skipped (caller 0x{:08X})", p, heap, bad,
                       uint32_t(ctx.lr));
        return;
      }
    }
  }
  __imp__sub_82B17218(ctx, base);
}

// sub_82B182F8 (allocate from the current thread's heap, top-down best fit): walks the heap's free list ([8] next
// free index, [4] previous, block 0 = list head, last free block's [8] = the heap's unit count) with no step limit.
// In Create-a-Moveset a list that loops hung the game thread there. Before each call the list is checked with a step
// limit; a list that loops or leaves the heap is rebuilt from the block chain ([0] low 28 bits = next block's index;
// a block with [8] = 0 is in use). WWE13_HEAP_ALLOC_CHECK=0 turns this off.
extern "C" PPC_FUNC(sub_82B16960);
// Checks the free list of heap `heap` (step limit, every free block's size inside the heap) and repairs what it
// can: a free block whose size field leaves the heap gets it back from the real next block, a list that loops or
// leaves the heap is rebuilt from the block chain (damaged chain headers repaired the same way). false: the heap is
// beyond repair (the allocation is refused). Runs before every allocation (WWE13_HEAP_ALLOC_CHECK=0: off).
bool CheckHeap(uint8_t* base, uint32_t heap, uint32_t caller) {
  // Heap 54 is laid out differently on purpose (its last free block reaches into other memory the game hands out from
  // it, 0x2C.../0x2E...): "repairing" it broke the move picker's text. It is left alone.
  if (heap == 54) return true;
  static const bool on = [] { const char* e = std::getenv("WWE13_HEAP_ALLOC_CHECK"); return !e || *e != '0'; }();
  if (on) {
    const uint32_t t = PPC_LOAD_U32(kHeapTable);
    if (t && heap < PPC_LOAD_U32(kHeapCount)) {
      const uint32_t s = PPC_LOAD_U32(t + heap * 36), e = PPC_LOAD_U32(t + heap * 36 + 4);
      const uint32_t units = (e - s) >> 4;
      const char* bad = nullptr;
      if (s && e > s) {
        // forward: from the head to the end marker
        uint32_t idx = PPC_LOAD_U32(s + 8), last = 0, steps = 0;
        while (idx != units) {
          if (idx == 0 || idx > units) { bad = "next link leaves the heap"; break; }
          if (++steps > units) { bad = "next links loop"; break; }
          // A free block's size (header [0], low 28 bits = next block's index) past the heap end makes the
          // allocator hand out addresses outside the heap (heap 54 in Create-a-Moveset: 0x2Cxxxxxx-0x2Exxxxxx,
          // other memory got overwritten). Repaired from the real next block: the header whose [4] (previous
          // block) names this one.
          const uint32_t b = s + idx * 16, end = PPC_LOAD_U32(b) & 0x0FFFFFFF;
          if (end <= idx || end > units) {
            uint32_t fix = 0;
            for (uint32_t j = idx + 1; j < units && j < idx + 0x100000; ++j) {
              const uint32_t h = s + j * 16;
              const uint32_t jn = PPC_LOAD_U32(h) & 0x0FFFFFFF;
              if (PPC_LOAD_U32(h + 4) == idx && (PPC_LOAD_U32(h + 12) & 0xff) == heap && jn > j && jn <= units) {
                fix = j;
                break;
              }
            }
            if (!fix && PPC_LOAD_U32(b + 8) == units) fix = units;  // (the last block)
            static int logged = 0;
            if (logged++ < 20)
              REXLOG_WARN("[wwe13-heap] heap {} free block {:#x} (0x{:08X}) has size field {:08X}: {}", heap, idx, b,
                          PPC_LOAD_U32(b), fix ? fmt::format("repaired, next block {:#x}", fix) : std::string("not repairable"));
            if (fix) {
              PPC_STORE_U32(b, (PPC_LOAD_U32(b) & 0xF0000000u) | fix);
            } else {
              bad = "a free block's size leaves the heap";
              break;
            }
          }
          last = idx;
          idx = PPC_LOAD_U32(s + idx * 16 + 8);
        }
        // backward: from the last free block to the head
        if (!bad) {
          steps = 0;
          idx = last;
          while (idx) {
            if (idx >= units) { bad = "previous link leaves the heap"; break; }
            if (++steps > units) { bad = "previous links loop"; break; }
            idx = PPC_LOAD_U32(s + idx * 16 + 4);
          }
        }
      }
      if (bad) {
        // rebuild from the block chain
        std::vector<uint32_t> free_blocks;
        uint32_t idx = 0, steps = 0;
        bool chain_ok = true;
        while (idx < units) {
          uint32_t next = PPC_LOAD_U32(s + idx * 16) & 0x0FFFFFFF;
          if (next <= idx || next > units) {  // (a damaged header: the real next block names this one as previous)
            uint32_t fix = 0;
            for (uint32_t j = idx + 1; j < units && j < idx + 0x100000; ++j) {
              const uint32_t h = s + j * 16;
              const uint32_t jn = PPC_LOAD_U32(h) & 0x0FFFFFFF;
              if (PPC_LOAD_U32(h + 4) == idx && (PPC_LOAD_U32(h + 12) & 0xff) == heap && jn > j && jn <= units) {
                fix = j;
                break;
              }
            }
            static int logged2 = 0;
            if (logged2++ < 20)
              REXLOG_WARN("[wwe13-heap] heap {} block {:#x} header {:08X}: {}", heap, idx, PPC_LOAD_U32(s + idx * 16),
                          fix ? fmt::format("repaired, next block {:#x}", fix) : std::string("not repairable"));
            if (!fix) { chain_ok = false; break; }
            PPC_STORE_U32(s + idx * 16, (PPC_LOAD_U32(s + idx * 16) & 0xF0000000u) | fix);
            next = fix;
          }
          if (++steps > units) { chain_ok = false; break; }
          if (idx && PPC_LOAD_U32(s + idx * 16 + 8) != 0) free_blocks.push_back(idx);
          idx = next;
        }
        static int logged = 0;
        if (logged++ < 20)
          REXLOG_WARN("[wwe13-heap] heap {} free list broken ({}), caller {:08X}; {} ({} free blocks)", heap, bad,
                      caller, chain_ok ? "rebuilt from the block chain" : "block chain broken too - allocation refused",
                      free_blocks.size());
        if (!chain_ok) return false;
        PPC_STORE_U32(s + 8, free_blocks.empty() ? units : free_blocks[0]);
        for (size_t i = 0; i < free_blocks.size(); ++i) {
          const uint32_t b = s + free_blocks[i] * 16;
          PPC_STORE_U32(b + 4, i ? free_blocks[i - 1] : 0);
          PPC_STORE_U32(b + 8, i + 1 < free_blocks.size() ? free_blocks[i + 1] : units);
        }
      }
    }
  }
  return true;
}

extern "C" PPC_FUNC(__imp__sub_82B182F8);
extern "C" void rex_trap_set_phys(uint32_t phys, uint32_t len);
extern "C" void rex_trap_rearm();
extern "C" PPC_FUNC(sub_82B182F8) {
  {  // write trap on the heap-54 block header that gets overwritten in Create-a-Moveset (diagnostic; WWE13_TRAP=1: on)
    static const bool trap = [] { const char* e = std::getenv("WWE13_TRAP"); return e && *e == '1'; }();
    static bool set = false;
    if (trap) {
      if (!set && PPC_LOAD_U32(kHeapTable)) {
        set = true;
        REXLOG_INFO("[wwe13-heap] write trap on the heap-54 header at 0xA3E2CA60");
        rex_trap_set_phys(0x03E2CA60u, 16);
      }
      rex_trap_rearm();
    }
  }
  {
    PPCContext c = ctx;
    sub_82B16960(c, base);
    if (!CheckHeap(base, c.r3.u32, uint32_t(ctx.lr))) {
      ctx.r3.u64 = 0;
      return;
    }
  }
  {
    static const bool watch = [] { const char* e = std::getenv("WWE13_HEAP43_WATCH"); return !e || *e != '0'; }();
    Took took{ctx.r3.u32, {0, 0, 0, 0}};
    uint32_t sp = ctx.r1.u32;
    if (watch)  // (callers of the allocation wrapper: skip this frame's own link)
      for (int k = 0; k < 4 && sp; ++k) {
        sp = PPC_LOAD_U32(sp);
        if (!sp || sp < 0x70000000) break;
        took.lr[k] = PPC_LOAD_U32(sp - 8);
      }
    __imp__sub_82B182F8(ctx, base);
    if (watch && ctx.r3.u32) {
      const uint32_t got = ctx.r3.u32;
      const uint8_t heap = PPC_LOAD_U32(got - 4) & 0xff;
      if (heap == 43 || heap == 54 || heap == 12) {
        std::lock_guard<std::mutex> lk(g_took_mu);
        g_took[got] = took;
      }
    }
  }
}

// sub_82AA4C70(queue, &cmd): runs one queued render command (type = cmd[0], 0-101). The Create-a-Moveset editor
// crashes in it on a command holding a bad pointer; the last command (address and first 12 words) goes into the
// crash log.
void wwe13_training_tick(uint8_t* base);  // training.cpp
extern "C" void wwe13_arena_watch_tick(uint8_t* base);
extern "C" uint32_t wwe13_last_cmd[16] = {};
extern "C" uint32_t wwe13_cmd_ring[8][6] = {};
extern "C" uint32_t wwe13_cmd_ring_pos = 0;
extern "C" uint8_t* wwe13_guest_base = nullptr;
extern "C" PPC_FUNC(__imp__sub_82AA4C70);
extern "C" PPC_FUNC(sub_82AA4C70) {
  wwe13_training_tick(base);
  wwe13_arena_watch_tick(base);
  const uint32_t c = PPC_LOAD_U32(ctx.r4.u32);
  wwe13_guest_base = base;
  wwe13_last_cmd[0] = c;
  wwe13_last_cmd[1] = uint32_t(ctx.lr);
  if (c >= 0x10000)
    for (int i = 0; i < 12; ++i) wwe13_last_cmd[2 + i] = PPC_LOAD_U32(c + 4 * i);
  uint32_t* r = wwe13_cmd_ring[wwe13_cmd_ring_pos++ & 7];
  r[0] = c;
  for (int i = 0; i < 5; ++i) r[1 + i] = c >= 0x10000 ? PPC_LOAD_U32(c + 4 * i) : 0;
  // Commands 54-57 read through a pointer at +16 (54/55 also store through +20). In Create-a-Moveset a display list
  // is sometimes run half-written: a type-56 command whose pointer slot still holds the old list's RETURN (79),
  // and the game read guest 0x4F. Such a command is skipped (the caller steps over it by its length).
  // WWE13_CMD_GUARD=0: off.
  static const bool guard = [] { const char* e = std::getenv("WWE13_CMD_GUARD"); return !e || *e != '0'; }();
  auto bad_ptr_cmd = [&](uint32_t a) {
    const uint32_t type = PPC_LOAD_U32(a);
    if (type < 54 || type > 57) return false;
    const uint32_t p16 = PPC_LOAD_U32(a + 16), p20 = PPC_LOAD_U32(a + 20);
    return p16 < 0x10000 || ((type == 54 || type == 55) && p20 < 0x10000);
  };
  if (guard && c >= 0x10000 && PPC_LOAD_U32(c) == 78) {
    // CALL: the dispatcher switches to the target list and runs its first command itself (not through this hook).
    const uint32_t t = PPC_LOAD_U32(c + 16);
    {
      static int seen = 0;
      if (seen++ < 400 && t >= 0x10000)
        REXLOG_INFO("[wwe13-cmd] call 0x{:08X} -> 0x{:08X}: {:08X} {:08X} {:08X} {:08X} | {:08X} {:08X} {:08X} {:08X}", c, t,
                    PPC_LOAD_U32(t), PPC_LOAD_U32(t + 4), PPC_LOAD_U32(t + 8), PPC_LOAD_U32(t + 12), PPC_LOAD_U32(t + 16),
                    PPC_LOAD_U32(t + 20), PPC_LOAD_U32(t + 24), PPC_LOAD_U32(t + 28));
    }
    if (t < 0x10000 || bad_ptr_cmd(t)) {
      static int logged = 0;
      if (logged++ < 20)
        REXLOG_WARN("[wwe13-cmd] display-list call at 0x{:08X} to 0x{:08X}: first command {} has a bad pointer "
                    "({:08X}): call skipped", c, t, t >= 0x10000 ? PPC_LOAD_U32(t) : 0, t >= 0x10000 ? PPC_LOAD_U32(t + 16) : 0);
      return;
    }
  }
  if (guard && c >= 0x10000) {
    const uint32_t type = PPC_LOAD_U32(c);
    if (type >= 54 && type <= 57) {
      const uint32_t p16 = PPC_LOAD_U32(c + 16), p20 = PPC_LOAD_U32(c + 20);
      if (bad_ptr_cmd(c)) {
        static int logged = 0;
        if (logged++ < 20)
          REXLOG_WARN("[wwe13-cmd] render command {} at 0x{:08X} has a bad pointer ({:08X} {:08X}): skipped", type, c,
                      p16, p20);
        return;
      }
    }
  }
  __imp__sub_82AA4C70(ctx, base);
}

// sub_82B222F0 (UTF-8 character count) / sub_82B223B8 (find character): the UI text renderer calls them per
// character. On a string with no terminator (Create-a-Moveset move picker) the game hung. A string with no 0 byte in
// its first 64 KB is logged and treated as empty. WWE13_STRING_GUARD=0: off.
namespace {
bool StringUnterminated(uint8_t* base, uint32_t p) {
  if (p < 0x10000) return false;
  for (uint32_t i = 0; i < 0x10000; ++i)
    if (PPC_LOAD_U8(p + i) == 0) return false;
  static int logged = 0;
  if (logged++ < 10) {
    char t[33];
    for (int i = 0; i < 32; ++i) {
      const uint8_t ch = PPC_LOAD_U8(p + i);
      t[i] = ch >= 32 && ch < 127 ? char(ch) : '.';
    }
    t[32] = 0;
    REXLOG_WARN("[wwe13-text] string at 0x{:08X} has no end in 64 KB: \"{}\" - treated as empty", p, t);
  }
  return true;
}
// The helpers step through the text by a per-byte length table (0x833D2F80: UTF-8 sequence length). A byte whose
// entry is 0 (e.g. a Latin-1 character in a ported move name) never advances, and each call then spins 33 million
// times: the move picker hung. Such bytes are replaced with '?' in place.
void StringSanitize(uint8_t* base, uint32_t p) {
  if (p < 0x10000) return;
  // walk it as the game does: step by the length of each sequence's lead byte
  uint32_t i = 0;
  for (int n = 0; n < 0x10000; ++n) {
    const uint8_t ch = PPC_LOAD_U8(p + i);
    if (!ch) return;
    const uint32_t step = PPC_LOAD_U32(0x833D2F80u + ch * 4u);
    if (step == 0 || step > 4) {
      static int logged = 0;
      if (logged++ < 20) {
        char t[49];
        int k = 0;
        for (; k < 48; ++k) {
          const uint8_t c2 = PPC_LOAD_U8(p + k);
          if (!c2) break;
          t[k] = c2 >= 32 && c2 < 127 ? char(c2) : '?';
        }
        t[k] = 0;
        REXLOG_WARN("[wwe13-text] byte {:02X} at 0x{:08X} stalls the text walk (\"{}\"): replaced with '?'", ch, p + i, t);
      }
      PPC_STORE_U8(p + i, '?');
      ++i;
      continue;
    }
    i += step;
  }
}
bool StringGuardOn() {
  static const bool on = [] { const char* e = std::getenv("WWE13_STRING_GUARD"); return !e || *e != '0'; }();
  return on;
}
}  // namespace
extern "C" PPC_FUNC(__imp__sub_82B222F0);
extern "C" PPC_FUNC(sub_82B222F0) {
  {  // sample: which string is measured over and over (a text layout loop that does not advance hangs the game)
    static std::atomic<uint64_t> n{0};
    const uint64_t k = ++n;
    if ((k & 0xFFFFF) == 0 && k < (40ull << 20)) {
      const uint32_t p = ctx.r3.u32;
      char t[65];
      int i = 0;
      for (; i < 64 && p >= 0x10000; ++i) {
        const uint8_t ch = PPC_LOAD_U8(p + i);
        if (!ch) break;
        t[i] = ch >= 32 && ch < 127 ? char(ch) : '?';
      }
      t[i] = 0;
      REXLOG_WARN("[wwe13-text] {}M string measures; now 0x{:08X} from {:08X}: \"{}\"", k >> 20, p, uint32_t(ctx.lr), t);
    }
  }
  if (StringGuardOn()) StringSanitize(base, ctx.r3.u32);
  if (StringGuardOn() && StringUnterminated(base, ctx.r3.u32)) {
    ctx.r3.u64 = 0;
    return;
  }
  __imp__sub_82B222F0(ctx, base);
}
extern "C" PPC_FUNC(__imp__sub_82B223B8);
extern "C" PPC_FUNC(sub_82B223B8) {
  if (StringGuardOn()) StringSanitize(base, ctx.r3.u32);
  if (StringGuardOn() && StringUnterminated(base, ctx.r3.u32)) {
    ctx.r3.u64 = uint64_t(int64_t(-1));
    return;
  }
  __imp__sub_82B223B8(ctx, base);
}

// sub_82B16C98(heap, stats): walks a heap's whole block chain ([0] low 28 bits = the next block's index, the address
// computed in 32 bits) to count free and used space. sub_82B17988 calls it for heaps 44 and up when the game lays its
// heaps out again (leaving Creations for the main menu). After Create-a-Moveset heap 54's last free block is far bigger
// than the heap (see CheckHeap): start + index * 16 wraps past 4 GB to a low address and the walk never ends - the game
// froze on the loading screen. The walk is done here with indices (a block past the end ends it); the stats are the
// game's: [0] heap bytes, [4] largest free, [8] heap - used, [12] used, [16] used blocks, [20] free, [24] free blocks,
// [28] the heap's 7-character name. WWE13_HEAP_WALK_GUARD=0: off.
extern "C" PPC_FUNC(__imp__sub_82B16C98);
extern "C" PPC_FUNC(sub_82B16C98) {
  static const bool on = [] { const char* e = std::getenv("WWE13_HEAP_WALK_GUARD"); return !e || *e != '0'; }();
  const uint32_t heap = ctx.r3.u32, stats = ctx.r4.u32;
  const uint32_t t = PPC_LOAD_U32(kHeapTable);
  if (on && t && heap < PPC_LOAD_U32(kHeapCount)) {
    const uint32_t rec = t + heap * 36;
    const uint32_t s = PPC_LOAD_U32(rec), e = PPC_LOAD_U32(rec + 4);
    if (s && e > s) {
      const uint32_t units = (e - s) >> 4;
      bool wraps = false;
      for (uint32_t idx = 0, steps = 0; idx < units && steps <= units; ++steps) {
        const uint32_t next = PPC_LOAD_U32(s + idx * 16) & 0x0FFFFFFF;
        if (next <= idx || next > units) { wraps = true; break; }
        idx = next;
      }
      if (wraps) {
        uint32_t largest = 0, used = 0, used_n = 0, fr = 0, free_n = 0;
        for (uint32_t idx = 0, steps = 0; idx < units && steps <= units; ++steps) {
          uint32_t next = PPC_LOAD_U32(s + idx * 16) & 0x0FFFFFFF;
          if (next <= idx || next > units) next = units;
          const uint32_t size = next - idx - 1;
          if (PPC_LOAD_U32(s + idx * 16 + 8) == 0) {
            used += size, ++used_n;
          } else {
            largest = std::max(largest, size), fr += size, ++free_n;
          }
          idx = next;
        }
        static int logged = 0;
        if (logged++ < 10)
          REXLOG_WARN("[wwe13-heap] heap {} walk: a block reaches past the heap end; counted up to the end", heap);
        PPC_STORE_U32(stats + 0, e - s);
        PPC_STORE_U32(stats + 4, largest * 16);
        PPC_STORE_U32(stats + 8, (e - s) - used * 16);
        PPC_STORE_U32(stats + 12, used * 16);
        PPC_STORE_U32(stats + 16, used_n);
        PPC_STORE_U32(stats + 20, fr * 16);
        PPC_STORE_U32(stats + 24, free_n);
        PPC_STORE_U32(stats + 28, 0);
        PPC_STORE_U32(stats + 32, 0);
        for (uint32_t i = 0; i < 7; ++i) PPC_STORE_U8(stats + 28 + i, PPC_LOAD_U8(rec + 28 + i));
        ctx.r3.u64 = 1;
        return;
      }
    }
  }
  __imp__sub_82B16C98(ctx, base);
}

// sub_82B1EEC8(size, heap): the pac loader's buffer for a file it reads whole (2 KB-rounded). In Create-an-Entrance
// a read asked for 0 bytes; the pac cache (sub_82B3CBA0) then retried "allocate 0 bytes" forever and the game hung
// on the loading screen. A 0-byte read gets one 2 KB sector. WWE13_ZERO_READ_FIX=0: off.
extern "C" PPC_FUNC(__imp__sub_82B1EEC8);
extern "C" PPC_FUNC(sub_82B1EEC8) {
  // (diagnostic, off by default: the 0-byte read came from a save holding creations made with the mod's superstars
  // while the mod was off - with a fresh save the editor loads. Logging only unless WWE13_ZERO_READ_FIX=1.)
  static const bool on = [] { const char* e = std::getenv("WWE13_ZERO_READ_FIX"); return e && *e == '1'; }();
  if (ctx.r3.u32 == 0) {
    static int logged = 0;
    if (logged++ < 20) {
      uint32_t lr2 = 0, sp = ctx.r1.u32;
      if (sp) { sp = PPC_LOAD_U32(sp); if (sp) lr2 = PPC_LOAD_U32(sp - 8); }
      REXLOG_WARN("[wwe13-heap] pac loader: a 0-byte read (from {:08X} < {:08X}) gets 2 KB", uint32_t(ctx.lr), lr2);
      const uint32_t req = PPC_LOAD_U32(0x835C51E4u);
      if (req >= 0x10000) {
        std::string hex, txt;
        for (uint32_t i = 0; i < 256; i += 4) {
          const uint32_t w = PPC_LOAD_U32(req + i);
          hex += fmt::format("{:08X} ", w);
          if (w >= 0x40000000 && w < 0xC0000000) {  // a pointer: show text there
            std::string t;
            for (uint32_t k = 0; k < 64; ++k) { const uint8_t c = PPC_LOAD_U8(w + k); if (c < 32 || c > 126) break; t += char(c); }
            if (t.size() >= 4) txt += fmt::format(" [+{:x}]={}", i, t);
          }
        }
        std::string inl;
        for (uint32_t k = 0; k < 256; ++k) { const uint8_t c = PPC_LOAD_U8(req + k); inl += (c >= 32 && c <= 126) ? char(c) : '.'; }
        REXLOG_WARN("[wwe13-heap] pac loader request {:08X}: {}", req, hex);
        REXLOG_WARN("[wwe13-heap] pac loader request text: {} |{}|", txt, inl);
      }
    }
    if (on) ctx.r3.u64 = 2048;
  }
  __imp__sub_82B1EEC8(ctx, base);
}

// sub_82B17988(first heap, layout, layouts, ...): lays heaps out again for a mode (a match lays heaps 28-43 out from
// its own layout: [0]/[4] region, [8] size table in KB). Logged here to size heap 43 for matches.
extern "C" PPC_FUNC(__imp__sub_82B17988);
extern "C" PPC_FUNC(sub_82B17988) {
  const uint32_t first = ctx.r3.u32, layout = ctx.r4.u32, layouts = ctx.r5.u32, flag = ctx.r6.u32;
  const uint32_t d = layouts + layout * 12;
  const uint32_t s = PPC_LOAD_U32(d), e = PPC_LOAD_U32(d + 4), t = PPC_LOAD_U32(d + 8);
  std::string sizes;
  for (uint32_t i = 0; t && i < 64; ++i) {
    const uint32_t v = PPC_LOAD_U32(t + i * 4);
    sizes += std::to_string(int32_t(v)) + " ";
    if (v == 0xFFFFFFFFu) break;
  }
  REXLOG_INFO("[wwe13-heap] relayout: first heap {}, layout {} of 0x{:08X} (flag {}): 0x{:08X}-0x{:08X}, table 0x{:08X}: {}",
              first, layout, layouts, flag, s, e, t, sizes);
  // Creations (first heap 13) gives heap 43 4352 KB and heap 53 30720 KB; Create-a-Superstar filled heap 43 (a 232 KB
  // load failed -> crash) while heap 53 stayed empty. Move 4 MB from 53 to 43 once per table. WWE13_RELAYOUT_43=0: off.
  static const bool fix43 = [] { const char* v = std::getenv("WWE13_RELAYOUT_43"); return !v || *v != '0'; }();
  static std::vector<uint32_t> patched;
  if (fix43 && t && first <= 43 && std::find(patched.begin(), patched.end(), t) == patched.end()) {
    const uint32_t a = t + (43 - first) * 4, b = t + (53 - first) * 4;
    const uint32_t v43 = PPC_LOAD_U32(a), v53 = PPC_LOAD_U32(b);
    bool ok = v43 && v43 != 0xFFFFFFFFu && v53 != 0xFFFFFFFFu && v53 >= 16384;
    for (uint32_t i = 0; ok && i <= 53 - first; ++i) ok = PPC_LOAD_U32(t + i * 4) != 0xFFFFFFFFu;
    if (ok) {
      PPC_STORE_U32(a, v43 + 4096);
      PPC_STORE_U32(b, v53 - 4096);
      patched.push_back(t);
      REXLOG_INFO("[wwe13-heap] relayout table 0x{:08X}: heap 43 {} -> {} KB, heap 53 {} -> {} KB", t, v43, v43 + 4096, v53,
                  v53 - 4096);
    }
  }
  __imp__sub_82B17988(ctx, base);
  LogHeaps(base, "after relayout");
}
