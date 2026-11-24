// Roster mods: new wrestlers in the game's list of selectable superstars.
//
// The game keeps the superstar ids it lets you pick (100-499; 0-49 are created wrestlers, 50-99 DLC) in a fixed table in
// the executable's data: 100 u16 ids at 0x833D2AE0, the 85 real ones followed by padding (358 repeated). Every
// "is this id available" check (sub_82A67B00, sub_82A67C70, sub_82A68530, sub_82A695B0) scans those 100 entries, so an
// id outside the table (a ported wrestler in a free slot like 360) is refused.
//
// A mod folder (REX_MOD_OVERLAYS, from mods/enabled.txt) may hold roster.txt: one id per line (# comments). Those ids
// go to the FRONT of the table, the original ids follow, and the padding at the end absorbs the shift. Patched once, at
// the first availability check. WWE13_ROSTER_ADD=<id,id,...> adds ids too (testing).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

#include <rex/logging.h>
#include <rex/ppc/context.h>
#include "rex_compat.h"

namespace {
constexpr uint32_t kRosterTable = 0x833D2AE0;
constexpr int kRosterSlots = 100;

// A new wrestler's runtime character record: the game keeps the roster in the save (records of 320 bytes at
// *(0x83489690), index = u16[500] at 0x83387408), so an id the save never had is all zeros. roster.txt line
// "<id>|<copy record of id>|<full name>|<short name>[|<DSBD id, hex>]" fills it (in memory) from an existing wrestler's record.
struct NewRecord {
  uint16_t id = 0, from = 0, dsbd = 0, titantron = 0, music = 0, entrance = 0;
  std::string name, short_name;
  std::vector<std::pair<uint16_t, uint16_t>> profile;  // (offset, u16) over the copied profile: moveset etc.
  std::vector<std::pair<uint16_t, uint8_t>> record;    // (offset, byte) over the copied record: attributes etc.
};
std::vector<NewRecord> g_new_records;

void ParseRecordLine(const std::string& line) {
  std::vector<std::string> f;
  size_t s = 0;
  for (;;) {
    size_t p = line.find('|', s);
    f.push_back(line.substr(s, p == std::string::npos ? std::string::npos : p - s));
    if (p == std::string::npos) break;
    s = p + 1;
  }
  if (f.size() < 4) return;
  NewRecord r;
  r.id = uint16_t(std::atoi(f[0].c_str()));
  r.from = uint16_t(std::atoi(f[1].c_str()));
  r.name = f[2];
  r.short_name = f[3];
  while (!r.short_name.empty() && (r.short_name.back() == '\r' || r.short_name.back() == ' ')) r.short_name.pop_back();
  if (f.size() > 4) r.dsbd = uint16_t(std::strtoul(f[4].c_str(), nullptr, 16));
  if (f.size() > 5) r.titantron = uint16_t(std::atoi(f[5].c_str()));  // (movies/titantron/<n>.bik2)
  if (f.size() > 6) r.music = uint16_t(std::atoi(f[6].c_str()));      // (Wwise event Play_MUS_<n>_a_b, Music.pck)
  if (f.size() > 7) r.entrance = uint16_t(std::atoi(f[7].c_str()));   // (pac/evt/Entrance*.pac EVP/EVZ <n>)
  if (f.size() > 8) {  // "off:value,…" (hex): profile u16 fields (the moveset, 0x000-0x23f)
    const char* p = f[8].c_str();
    while (*p) {
      char* e = nullptr;
      const unsigned long off = std::strtoul(p, &e, 16);
      if (!e || *e != ':') break;
      const unsigned long v = std::strtoul(e + 1, &e, 16);
      if (off + 2 <= 0x240) r.profile.emplace_back(uint16_t(off), uint16_t(v));
      p = e;
      while (*p == ',' || *p == ' ') ++p;
      if (*p == '\r' || *p == '\n') break;
    }
  }
  if (f.size() > 9) {  // "off:byte,…" (hex): runtime record bytes (attributes, weight, …)
    const char* p = f[9].c_str();
    while (*p) {
      char* e = nullptr;
      const unsigned long off = std::strtoul(p, &e, 16);
      if (!e || *e != ':') break;
      const unsigned long v = std::strtoul(e + 1, &e, 16);
      if (off < 0x34 || (off >= 0x10e && off < 320)) r.record.emplace_back(uint16_t(off), uint8_t(v));  // (not names/ids)
      p = e;
      while (*p == ',' || *p == ' ') ++p;
      if (*p == '\r' || *p == '\n') break;
    }
  }
  if (r.id >= 100 && r.id < 500 && r.from >= 100 && r.from < 500) g_new_records.push_back(r);
}

void ParseIds(const std::string& text, std::vector<uint16_t>* out) {
  // record lines first (they also add their id)
  size_t ls = 0;
  while (ls < text.size()) {
    size_t le = text.find('\n', ls);
    if (le == std::string::npos) le = text.size();
    const std::string line = text.substr(ls, le - ls);
    if (line.find('|') != std::string::npos && line[0] != '#') ParseRecordLine(line);
    ls = le + 1;
  }
  size_t i = 0;
  while (i < text.size()) {
    if (text[i] == '#') {
      while (i < text.size() && text[i] != '\n') ++i;
      continue;
    }
    if (text[i] == '|') {  // (rest of a record line)
      while (i < text.size() && text[i] != '\n') ++i;
      continue;
    }
    if (text[i] >= '0' && text[i] <= '9') {
      unsigned v = 0;
      while (i < text.size() && text[i] >= '0' && text[i] <= '9') v = v * 10 + unsigned(text[i++] - '0');
      if (v >= 100 && v < 500) out->push_back(uint16_t(v));
      continue;
    }
    ++i;
  }
}

std::vector<uint16_t> WantedIds() {
  std::vector<uint16_t> ids;
  if (const char* overlays = std::getenv("REX_MOD_OVERLAYS")) {
    std::string rest(overlays);
    size_t start = 0;
    while (start <= rest.size()) {
      size_t sep = rest.find(';', start);
      if (sep == std::string::npos) sep = rest.size();
      const std::string folder = rest.substr(start, sep - start);
      start = sep + 1;
      if (folder.empty()) continue;
      if (FILE* f = std::fopen((folder + "/roster.txt").c_str(), "rb")) {
        std::string text;
        char buf[512];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
        std::fclose(f);
        ParseIds(text, &ids);
      }
    }
  }
  if (const char* extra = std::getenv("WWE13_ROSTER_ADD")) ParseIds(extra, &ids);
  return ids;
}

uint32_t RecordAddr(uint8_t* base, uint32_t id) {
  const uint32_t idx = PPC_LOAD_U16(0x83387408 + id * 2);
  const uint32_t table = PPC_LOAD_U32(0x83489690);
  if (idx == 512 || !table) return 0;
  return table + idx * 320;
}

// Profile records (entrance, moves, …): the save holds one per id, 1176 bytes = chEtc CHAR/PRO without its last 4 bytes
// (u16 big-endian; 0x240/0x24c: two entrance slots [music, titantron, entrance motion, ?, ?]). An id the save never had
// is all zeros (no entrance). In memory they are an array indexed by id (found once: the address P where the profiles
// of 100, 101 and 322 start with 4 zero bytes and name their own id as entrance music, 0x244).
constexpr uint32_t kProfileSize = 1176;
uint32_t g_profile0 = 0;
uint32_t ProfileAddr(uint8_t* base, uint32_t id) {
  static int searches = 0;
  if (!g_profile0 && searches < 20) {
    ++searches;
    // (the executable's image incl. its data/bss – always mapped)
    auto own = [&](uint32_t rec, uint32_t id) {  // (entrance slot 1: entrance, titantron, music = the id itself)
      return PPC_LOAD_U16(rec + 0x240) == id && PPC_LOAD_U16(rec + 0x242) == id && PPC_LOAD_U16(rec + 0x244) == id;
    };
    for (uint32_t a = 0x82000000 + 322 * kProfileSize; a < 0x836C0000 - kProfileSize; a += 2) {
      if (!own(a, 322)) continue;
      const uint32_t p0 = a - 322 * kProfileSize;
      if (own(p0 + 100 * kProfileSize, 100) && own(p0 + 101 * kProfileSize, 101)) {
        g_profile0 = p0;
        break;
      }
    }
    if (g_profile0 || searches == 20) REXLOG_INFO("[wwe13-roster] profiles at {:08x}", g_profile0);
  }
  if (!g_profile0 || id >= 500) return 0;
  return g_profile0 + id * kProfileSize;
}
void EnsureProfile(uint8_t* base, const NewRecord& r) {
  const uint32_t dst = ProfileAddr(base, r.id), src = ProfileAddr(base, r.from);
  if (!dst || !src) return;
  if (PPC_LOAD_U16(dst + 0x240) != 0 || PPC_LOAD_U16(dst + 0x244) != 0) {
    // Filled already, or the save has one: a save made with an older mod build holds the copied wrestler's moves
    // and entrance (Del Rio's finishers on the ported stars). A field that still equals the copied wrestler's value
    // gets the mod's value; a field the player changed in the game is left alone.
    int fixed = 0;
    // The entrance (music, titantron, entrance motion) always follows roster.txt: a mod update that gives a ported
    // star his own entrance replaces what an older build put in the save.
    auto fix = [&](uint32_t off, uint16_t v, bool always = false) {
      if (!v) return;
      const uint16_t cur = PPC_LOAD_U16(dst + off);
      if (cur != v && (always || cur == PPC_LOAD_U16(src + off))) {
        PPC_STORE_U16(dst + off, v);
        ++fixed;
      }
    };
    for (uint32_t slot : {0x240u, 0x24cu}) {
      fix(slot, r.music, true);
      fix(slot + 2, r.titantron, true);
      fix(slot + 4, r.entrance, true);
    }
    for (const auto& [off, v] : r.profile) fix(off, v);
    static int logged = 0;
    if (fixed && logged++ < 16)
      REXLOG_INFO("[wwe13-roster] profile {}: {} fields from the copied {} replaced by the mod's", r.id, fixed, r.from);
    return;
  }
  for (uint32_t i = 0; i < kProfileSize; i += 4)
    if (PPC_LOAD_U32(dst + i) != 0) {  // (not an empty profile: wrong address – leave it)
      static int skipped = 0;
      if (skipped++ < 3)
        REXLOG_WARN("[wwe13-roster] profile {} at {:08x} not empty (+{:x} = {:08x}): left alone", r.id, dst, i,
                    PPC_LOAD_U32(dst + i));
      return;
    }
  for (uint32_t i = 0; i < kProfileSize; ++i) PPC_STORE_U8(dst + i, PPC_LOAD_U8(src + i));
  for (uint32_t slot : {0x240u, 0x24cu}) {
    if (r.titantron) PPC_STORE_U16(dst + slot + 2, r.titantron);
    if (r.music) PPC_STORE_U16(dst + slot, r.music);  // (slot: music, titantron, entrance motion, …)
    if (r.entrance) PPC_STORE_U16(dst + slot + 4, r.entrance);
  }
  for (const auto& [off, v] : r.profile) PPC_STORE_U16(dst + off, v);
  static int logged = 0;
  if (logged++ < 8)
    REXLOG_INFO("[wwe13-roster] profile {} filled from {} at {:08x} (entrance {} {} {} {} {})", r.id, r.from, dst,
                PPC_LOAD_U16(dst + 0x240), PPC_LOAD_U16(dst + 0x242), PPC_LOAD_U16(dst + 0x244),
                PPC_LOAD_U16(dst + 0x246), PPC_LOAD_U16(dst + 0x248));
}

// Runtime record = the chEtc DAT record without its 4-byte header (u16 fields big-endian): 0x34 id, 0x36 full name
// (68), 0x7a full name (92), 0xd6 short name (38), 0xfc type, 0xfe-0x105 DSBD ids, 0x106 id, 0x10a grid sort key,
// 0x10c id.
void EnsureRecords(uint8_t* base) {
  // Runs on every record lookup (Universe does thousands a frame: 1 FPS there). The full pass (byte fixes, profiles)
  // runs at most every 250 ms; in between only a quick check that every new record is still there (a save reload
  // zeroes them; then the full pass runs at once).
  {
    static std::chrono::steady_clock::time_point last{};
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::milliseconds(250)) {
      bool ok = true;
      for (const NewRecord& r : g_new_records) {
        const uint32_t dst = RecordAddr(base, r.id), src = RecordAddr(base, r.from);
        if (!dst || !src || PPC_LOAD_U8(src + 0x36) == 0) continue;
        if (PPC_LOAD_U16(dst + 0x34) != r.id || PPC_LOAD_U8(dst + 0x36) == 0) { ok = false; break; }
      }
      if (ok) return;
    }
    last = now;
  }
  int filled = 0;
  for (const NewRecord& r : g_new_records) {
    const uint32_t dst = RecordAddr(base, r.id), src = RecordAddr(base, r.from);
    if (!dst || !src) continue;
    if (PPC_LOAD_U8(src + 0x36) != 0) EnsureProfile(base, r);
    if (PPC_LOAD_U16(dst + 0x34) == r.id && PPC_LOAD_U8(dst + 0x36) != 0) {  // (already there)
      // A record a save made with an older mod build: a byte roster.txt sets (attributes, brand at 0x1c-0x1f) that
      // still equals the copied wrestler's value gets the mod's value (all ported stars showed Del Rio's SmackDown).
      // (checked on every call: the save is read after the first fill and brings its own copy of the record)
      if (PPC_LOAD_U8(src + 0x36) != 0) {
        int n = 0;
        for (const auto& [off, v] : r.record) {
          const uint8_t cur = PPC_LOAD_U8(dst + off);
          if (cur != v && cur == PPC_LOAD_U8(src + off)) {
            PPC_STORE_U8(dst + off, v);
            ++n;
          }
        }
        static std::vector<uint16_t> said;
        bool was = false;
        for (uint16_t u : said) was |= u == r.id;
        if (n && !was) {
          said.push_back(r.id);
          REXLOG_INFO("[wwe13-roster] record {} ({}): {} byte(s) updated from roster.txt", r.id, r.name, n);
        }
      }
      continue;
    }
    if (PPC_LOAD_U8(src + 0x36) == 0) continue;  // (save not loaded yet)
    for (uint32_t i = 0; i < 320; ++i) PPC_STORE_U8(dst + i, PPC_LOAD_U8(src + i));
    auto put_str = [&](uint32_t off, uint32_t len, const std::string& v) {
      for (uint32_t i = 0; i < len; ++i) PPC_STORE_U8(dst + off + i, i < v.size() ? uint8_t(v[i]) : 0);
    };
    put_str(0x36, 68, r.name);
    put_str(0x7a, 92, r.name);
    put_str(0xd6, 38, r.short_name);
    PPC_STORE_U16(dst + 0x34, r.id);
    PPC_STORE_U16(dst + 0x106, r.id);
    PPC_STORE_U16(dst + 0x10c, r.id);
    // DSBD ids (5th field, hex: the first of the id's 4, as in chEtc CHAR/DSBD); default: the copied wrestler's
    if (r.dsbd)
      for (uint32_t k = 0; k < 4; ++k) PPC_STORE_U16(dst + 0xfe + k * 2, uint16_t(r.dsbd + k));
    for (const auto& [off, v] : r.record) PPC_STORE_U8(dst + off, v);
    ++filled;
    static int logged = 0;
    if (logged++ < 8) REXLOG_INFO("[wwe13-roster] record {} filled from {} ({})", r.id, r.from, r.name);
  }
  if (!filled) return;
  // The select grid is in sort-key order (0x10a: 1 = first; 0x04xx = DLC / created). The new wrestlers take keys
  // 1..n and every other superstar moves n places down. (Done exactly when the records were (re)filled: the save
  // holds the old keys then. After the game saves, the save has the new records and keys, and nothing is redone.)
  const uint32_t n = uint32_t(g_new_records.size());
  for (uint32_t id = 50; id < 500; ++id) {
    const uint32_t rec = RecordAddr(base, id);
    if (!rec) continue;
    bool is_new = false;
    for (const NewRecord& r : g_new_records) is_new |= r.id == id;
    if (is_new) continue;
    const uint16_t key = PPC_LOAD_U16(rec + 0x10a);
    if (key >= 1 && key < 0x400) PPC_STORE_U16(rec + 0x10a, uint16_t(key + n));
  }
  for (uint32_t k = 0; k < n; ++k) {
    const uint32_t rec = RecordAddr(base, g_new_records[k].id);
    if (rec) PPC_STORE_U16(rec + 0x10a, uint16_t(1 + k));
  }
}

std::atomic<uint32_t> g_pending_row{0};
// WWE13_ROSTER_SCAN=<file>: while that file exists, guest memory is searched every 3 s for the select grid's id order
// (Del Rio 322, Big Show 125, Booker T 338 as consecutive u32 / u16, big-endian) and the hits are logged (finding the
// select grid's cell array).
void StartScanner(uint8_t* base) {
  static bool started = false;
  const char* trigger = std::getenv("WWE13_ROSTER_SCAN");
  if (started || !trigger) return;
  started = true;
  std::string path(trigger);
  std::thread([base, path] {
    for (;;) {
      std::this_thread::sleep_for(std::chrono::seconds(3));
      FILE* f = std::fopen(path.c_str(), "rb");
      if (!f) continue;
      std::fclose(f);
      int hits = 0;
      const uint8_t p32[12] = {0, 0, 1, 0x42, 0, 0, 0, 125, 0, 0, 1, 0x52};
      const uint8_t p16[6] = {1, 0x42, 0, 125, 1, 0x52};
      for (uint64_t a = 0xA4000000; a < 0xA6000000 && hits < 40; a += 4) {
        const uint8_t* m = base + a;
        if (m[3] == 0x42 && std::memcmp(m, p32, 12) == 0) {
          std::string around;
          for (int k = -24; k < 72; ++k) around += " " + std::to_string(PPC_LOAD_U32(uint32_t(a) + k * 4));
          REXLOG_INFO("[wwe13-roster] scan: u32 order at {:08x}; from -96:{}", uint32_t(a), around);
          ++hits;
        }
      }
      for (uint64_t a = 0xA4000000; a < 0xA6000000 && hits < 80; a += 2) {
        const uint8_t* m = base + a;
        if (m[1] == 0x42 && std::memcmp(m, p16, 6) == 0) {
          REXLOG_INFO("[wwe13-roster] scan: u16 order at {:08x}", uint32_t(a));
          ++hits;
        }
      }
      REXLOG_INFO("[wwe13-roster] scan done ({} hits)", hits);
      // experiment: put 360 in front of row 1 (the 22nd id drops off)
      for (uint64_t a = 0xA4000000; a < 0xA6000000; a += 4) {
        const uint8_t* m = base + a;
        if (m[3] == 0x42 && std::memcmp(m, p32, 12) == 0) {
          g_pending_row.store(uint32_t(a));  // (written on a guest thread: host-thread writes to watched pages fault)
        }
      }
      std::remove(path.c_str());
    }
  }).detach();
}

void EnsureRoster(uint8_t* base) {
  StartScanner(base);
  EnsureRecords(base);
  static bool done = false;
  if (done) return;
  done = true;
  const std::vector<uint16_t> add = WantedIds();
  if (add.empty()) return;
  std::vector<uint16_t> table;
  for (int i = 0; i < kRosterSlots; ++i) table.push_back(PPC_LOAD_U16(kRosterTable + i * 2));
  // the original ids, without the padding repeats at the end and without ids the mod lists itself
  std::vector<uint16_t> original;
  for (uint16_t id : table) {
    bool seen = false;
    for (uint16_t o : original) seen |= o == id;
    for (uint16_t a : add) seen |= a == id;
    if (!seen) original.push_back(id);
  }
  // The original ids always stay (dropping them would hide real superstars); mod ids fill the padding after them. A mod
  // id that does not fit is still offered: the grid rows are patched directly, and for ids 100-499 the availability
  // checks only need the table when the id's record is missing (EnsureRecords fills it).
  std::vector<uint16_t> result = original;
  int skipped = 0;
  for (uint16_t a : add) {
    bool dup = false;
    for (uint16_t r : result) dup |= r == a;
    if (dup) continue;
    if (int(result.size()) < kRosterSlots - 1) result.push_back(a);  // (one padding entry kept)
    else ++skipped;
  }
  if (skipped)
    REXLOG_INFO("[wwe13-roster] {} mod ids past the table's {} slots (offered through their records)", skipped,
                kRosterSlots);
  const uint16_t pad = original.empty() ? result.back() : original.back();
  while (int(result.size()) < kRosterSlots) result.push_back(pad);
  for (int i = 0; i < kRosterSlots; ++i) PPC_STORE_U16(kRosterTable + i * 2, result[i]);
  std::string list;
  for (uint16_t a : add) list += " " + std::to_string(a);
  REXLOG_INFO("[wwe13-roster] mod ids:{} ({} original ids kept)", list, original.size());
}
// WWE13_ROSTER_TRACE=1: each (function, id, result) once in the log, for ids from the mods (finding what refuses them).
bool Trace() {
  static const bool on = [] {
    const char* v = std::getenv("WWE13_ROSTER_TRACE");
    return v && *v == '1';
  }();
  return on;
}
void TraceCall(const char* fn, uint32_t id, uint32_t arg2, uint32_t result) {
  static std::vector<uint64_t> seen;
  if (!Trace()) return;
  const uint64_t key = (uint64_t(uintptr_t(fn) & 0xFFFF) << 48) | (uint64_t(id & 0xFFFF) << 32) | (arg2 << 16) | (result & 0xFFFF);
  for (uint64_t k : seen) if (k == key) return;
  seen.push_back(key);
  REXLOG_INFO("[wwe13-roster] {}({}, {}) = {}", fn, id, arg2, result);
}
}  // namespace

// A move record (misc MOVS/WAZE, in memory: 3 x u64 category masks, name, ..., u16 owner at +128, id at +152) whose
// owner is a superstar is offered in Create-a-Moveset only when the game counts that superstar as available
// (sub_828E2AA0(?, record) -> 1). The game's checks know nothing of the mod's superstars, so their own moves
// (Dusty's Bionic Elbow Stamp etc., owner 388) were left out of the lists, and a slot holding one showed NONE.
// A move owned by a superstar from the mods counts as available. WWE13_MOD_MOVES_AVAILABLE=0: off.
bool wwe13_is_mod_star(uint32_t id) {
  for (const NewRecord& r : g_new_records)
    if (r.id == id) return true;
  return false;
}
extern "C" PPC_FUNC(__imp__sub_828E2AA0);
extern "C" PPC_FUNC(sub_828E2AA0) {
  const uint32_t rec = ctx.r4.u32;
  __imp__sub_828E2AA0(ctx, base);
  static const bool on = [] { const char* e = std::getenv("WWE13_MOD_MOVES_AVAILABLE"); return !e || *e != '0'; }();
  if (!on || ctx.r3.u32 || !rec) return;
  const uint32_t owner = PPC_LOAD_U16(rec + 128);
  if (owner >= 100 && owner < 500 && wwe13_is_mod_star(owner)) {
    ctx.r3.u64 = 1;
    static int logged = 0;
    if (logged++ < 20) REXLOG_INFO("[wwe13-roster] move {} (owner {}) offered: the owner is a mod superstar",
                                   PPC_LOAD_U16(rec + 152), owner);
  }
}

#define ROSTER_TRACED(name, fn)                                          \
  extern "C" PPC_FUNC(__imp__##fn);                                       \
  extern "C" PPC_FUNC(fn) {                                               \
    const uint32_t id = ctx.r3.u32, a2 = ctx.r4.u32;                      \
    __imp__##fn(ctx, base);                                               \
    if (id >= 340 && id < 400) TraceCall(name, id, a2 & 0xFFFF, ctx.r3.u32); \
  }
ROSTER_TRACED("in_roster_list_has_record", sub_82295960)

// WWE13_ROSTER_TRACE: who reads a superstar's profile (sub_82721588: id -> profile; the moves at 0x22-0x28), once
// per (id, caller), for Del Rio (322) and the mod ids 340-399.
extern "C" PPC_FUNC(__imp__sub_82721588);
extern "C" PPC_FUNC(sub_82721588) {
  const uint32_t id = ctx.r3.u32;
  const uint32_t lr = uint32_t(ctx.lr);
  __imp__sub_82721588(ctx, base);
  if (!Trace() || !(id == 322 || (id >= 340 && id < 400))) return;
  static std::vector<uint64_t> seen;
  const uint64_t key = (uint64_t(id) << 32) | lr;
  for (uint64_t k : seen) if (k == key) return;
  if (seen.size() > 400) return;
  seen.push_back(key);
  const uint32_t p = ctx.r3.u32 ? ctx.r3.u32 - 4 : 0;  // (returns the profile after its 4-byte header)
  REXLOG_INFO("[wwe13-roster] profile({}) = {:08x} for {:08x}: moves {:04x} {:04x} {:04x} {:04x}", id, p, lr,
              p ? PPC_LOAD_U16(p + 0x22) : 0, p ? PPC_LOAD_U16(p + 0x24) : 0, p ? PPC_LOAD_U16(p + 0x26) : 0,
              p ? PPC_LOAD_U16(p + 0x28) : 0);
}

// The game's character records (sub_827216A8): id -> index (u16[500] at 0x83387408), records of 320 bytes at the
// pointer in 0x83489690. WWE13_ROSTER_TRACE: dumps the records of 322 and 360 once.
void DumpRecord(uint8_t* base, uint32_t id) {
  const uint32_t idx = PPC_LOAD_U16(0x83387408 + id * 2);
  const uint32_t table = PPC_LOAD_U32(0x83489690);
  if (idx == 512 || !table) return;
  const uint32_t rec = table + idx * 320;
  std::string hex;
  char b[4];
  for (uint32_t i = 0; i < 320; ++i) {
    std::snprintf(b, sizeof(b), "%02x", unsigned(PPC_LOAD_U8(rec + i)));
    hex += b;
  }
  REXLOG_INFO("[wwe13-roster] record {} (index {}, {:08x}): {}", id, idx, rec, hex);
}
// (sub_82288448, the unlock check: hooked in unlock_all.cpp)
ROSTER_TRACED("record_flag297", sub_82207960)

extern std::atomic<uint32_t> g_last_lists;
extern "C" PPC_FUNC(__imp__sub_82775870);
extern "C" PPC_FUNC(sub_82775870) {
  const uint32_t lists = g_last_lists.load();
  const uint32_t before = lists ? PPC_LOAD_U32(lists + 0x162F4) : 0;
  __imp__sub_82775870(ctx, base);
  if (Trace() && lists)
    REXLOG_INFO("[wwe13-roster] sub_82775870: row1[0] {} -> {}", before, PPC_LOAD_U32(lists + 0x162F4));
}

// Once per presented frame, on a guest thread (present_interval.cpp sub_82B9A548).
std::atomic<uint32_t> g_last_lists{0};
void PatchGridRows(uint8_t* base, uint32_t row1);
namespace {
uint32_t FindRow(uint8_t* base, uint32_t lo, uint32_t hi);
}
// After the select menu built its candidate lists (sub_82839468 with mode 2), the grid rows appear within a few
// seconds and their cells are created a frame or more later: the rows are searched for once per frame in that window
// and patched as soon as they exist (the heap address differs between runs).
std::atomic<int> g_row_search_frames{1 << 30};  // (searching from the start: the rows exist before the lists calls)

// The select grid's rows (found by the memory scan): u32 ids, row 1 at lists + 0x162F4 (lists = the r4 of the last
// sub_82839468 call), 22 cells (the 22nd = 323, not selectable), then [0xFFFF0000][0][2 ids][0][54] and row 2 (22
// cells, -1 = empty). The game fills them from its own filter, which drops ids it does not know; the mods' ids are put
// in front here, the modern superstars shifted after them (row 1 -> row 2).
// Mods pages: page 0 = the game's own grid; page k >= 1 = the mod superstars (k-1)*43 ... in the two modern rows
// (row 1: 21 cells, its 22nd cell is the game's unselectable placeholder; row 2: 22 cells; -1 = empty cell).
// R3 (right stick click) on the select screen flips to the next page; the screen is rebuilt by backing out and in.
std::atomic<int> g_page{0};
std::atomic<uint32_t> g_select_seen_frame{0};
std::atomic<uint32_t> g_frame_no{0};
std::atomic<int> g_hot_frames{0};  // (after a page switch: the new grid is searched for every frame)
constexpr int kPageCells = 43;

int PageCount() { return int((g_new_records.size() + kPageCells - 1) / kPageCells); }

void PatchGridRows(uint8_t* base, uint32_t row1) {
  if (g_new_records.empty() || PPC_LOAD_U32(row1 - 4) != 54) return;
  const uint32_t first = PPC_LOAD_U32(row1);
  if (first < 100 || first >= 500) return;
  const uint32_t row2 = row1 + (22 + 6) * 4;
  if (PPC_LOAD_U32(row2 - 4) != 54) return;
  g_select_seen_frame.store(g_frame_no.load());
  const int page = g_page.load();
  // (the game's own rows are kept the first time they are seen: page 0 puts them back after a switch in place)
  static std::vector<uint32_t> stock;
  static uint32_t stock_row = 0;
  if (page <= 0 && stock_row != row1) {
    bool mods = false;
    for (int k = 0; k < 21 && !mods; ++k)
      for (const auto& r : g_new_records) mods |= PPC_LOAD_U32(row1 + k * 4) == r.id;
    if (!mods) {
      stock.assign(kPageCells, 0xFFFFFFFF);
      for (int k = 0; k < 21; ++k) stock[k] = PPC_LOAD_U32(row1 + k * 4);
      for (int k = 0; k < 22; ++k) stock[21 + k] = PPC_LOAD_U32(row2 + k * 4);
      stock_row = row1;
    }
  }
  if (page <= 0 && (stock.empty() || stock_row != row1)) return;  // (the game's own grid)
  std::vector<uint32_t> want(kPageCells, 0xFFFFFFFF);
  if (page <= 0) {
    want = stock;
  } else {
    for (int k = 0; k < kPageCells; ++k) {
      const size_t i = size_t(page - 1) * kPageCells + size_t(k);
      if (i < g_new_records.size()) want[k] = g_new_records[i].id;
    }
  }
  bool same = true;
  for (int k = 0; k < 21 && same; ++k) same = PPC_LOAD_U32(row1 + k * 4) == want[k];
  for (int k = 0; k < 22 && same; ++k) same = PPC_LOAD_U32(row2 + k * 4) == want[21 + k];
  if (same) return;
  for (int k = 0; k < 21; ++k) PPC_STORE_U32(row1 + k * 4, want[k]);
  for (int k = 0; k < 22; ++k) PPC_STORE_U32(row2 + k * 4, want[21 + k]);
  REXLOG_INFO("[wwe13-roster] grid rows at {:08x}: mods page {} of {}", row1, page, PageCount());
}

extern "C" void rex_input_inject(const char* lines);
extern "C" void (*rex_input_filter)(uint32_t player, uint16_t* buttons);

// After a page switch: if the select screen has not come back (an A was lost while the match menu was still coming
// back, which left the player on the match menu), A is pressed again (twice at most).
std::atomic<uint32_t> g_switch_check_frame{0};
std::atomic<bool> g_refresh_pending{false};
std::atomic<uint32_t> g_select_screen{0};  // (the select screen object, from sub_82773FD8)
std::atomic<int> g_switch_retries{0};

namespace {
void PadFilter(uint32_t player, uint16_t* buttons) {
  static uint16_t prev[4] = {};
  if (player > 3) return;
  // RT (0x0800 from the input driver): the next page; LT (0x0400): the previous one
  const uint16_t now = *buttons;
  const uint16_t edge = uint16_t(now & ~prev[player]);
  prev[player] = now;
  if (!(now & 0x0C00)) return;
  // only on the select screen (its rows were seen in the last half second)
  if (g_new_records.empty() || g_frame_no.load() - g_select_seen_frame.load() > 30) return;
  *buttons = uint16_t(now & ~0x0C00);
  if (!(edge & 0x0C00)) return;
  const int step = (edge & 0x0800) ? 1 : -1;
  // (one switch at a time: R3 during the back-out-and-in would add its presses to the running ones)
  static uint32_t busy_until = 0;
  if (g_frame_no.load() < busy_until) return;
  busy_until = g_frame_no.load() + 20;
  const int pages = PageCount();
  g_page.store((g_page.load() + step + pages + 1) % (pages + 1));
  g_hot_frames.store(600);
  REXLOG_INFO("[wwe13-roster] select screen: page {} ({})", g_page.load(), g_page.load() ? "mods" : "WWE roster");
  // B (release the controller), B (back to the match menu), A (select screen again, the grid is rebuilt with the new
  // page), A (take the controller again)
  // (the gaps are generous: an A pressed while the match menu is still coming back is lost, which left the player on
  // the match menu - seen on the Odin 2 Portal)
  // (no backing out: the select screen's own update rebuilds its lists and grid cells - sub_82773A78 below)
  g_refresh_pending.store(true);
}
}  // namespace

uint32_t wwe13_heap_start(uint8_t* base, uint32_t heap);  // heap_diag.cpp
void wwe13_unlock_select_build(bool begin);                // unlock_all.cpp

// The game's frames (flips) so far: the FPS counter (mods_hint.cpp).
uint32_t wwe13_game_frames() { return g_frame_no.load(); }

void wwe13_roster_frame(uint8_t* base) {
  const uint32_t fno = g_frame_no.fetch_add(1) + 1;
  if (const uint32_t check = g_switch_check_frame.load(); check && fno >= check) {
    if (fno - g_select_seen_frame.load() <= 30 || g_switch_retries.load() <= 0) {
      g_switch_check_frame.store(0);
    } else {
      g_switch_retries.fetch_sub(1);
      REXLOG_INFO("[wwe13-roster] select screen did not come back after the page switch: A again");
      rex_input_inject("1000 150\n0 4000\n1000 150\n");
      g_switch_check_frame.store(fno + 60 * 6);
    }
  }
  static bool filter_set = false;
  if (!filter_set) {
    filter_set = true;
    rex_input_filter = PadFilter;
  }
  const uint32_t a = g_pending_row.exchange(0);
  if (a) {
    for (int k = 21; k > 0; --k) PPC_STORE_U32(a + k * 4, PPC_LOAD_U32(a + (k - 1) * 4));
    PPC_STORE_U32(a, 360);
    REXLOG_INFO("[wwe13-roster] row patched at {:08x}", a);
  }
  if (const uint32_t lists = g_last_lists.load()) PatchGridRows(base, lists + 0x162F4);
  // (cost: the address where the rows were found last is checked every frame; the menu heap range is searched
  // every 12th frame, ~120 K loads per frame on average)
  static uint32_t frame = 0, last_row = 0;
  if (!g_new_records.empty()) {
    if (last_row && FindRow(base, last_row - 4, last_row) == last_row) PatchGridRows(base, last_row);
    const bool hot = g_hot_frames.load() > 0;
    if (hot) g_hot_frames.fetch_sub(1);
    if (hot || ++frame % 12 == 0) {
      // (every grid in the range: an old one may still be in memory below the new one)
      // (the range is inside the game's heap 54, 0x35BA00-0x4DBA00 past its start: 0xA4900000-0xA4A80000 with the
      // game's own heap sizes; heap_diag.cpp's bigger heaps 0 and 12 move it down)
      const uint32_t h54 = wwe13_heap_start(base, 54);
      const uint32_t lo = h54 ? h54 + 0x35BA00 : 0xA4900000, hi = h54 ? h54 + 0x4DBA00 : 0xA4A80000;
      uint32_t from = lo;
      for (int n = 0; n < 8; ++n) {
        const uint32_t row = FindRow(base, from, hi);
        if (!row) break;
        last_row = row;
        PatchGridRows(base, row);
        from = row + 0x800;  // (past this grid: its other rows match the pattern too)
      }
    }
  }
}

// Called at boot (display setup, res_probe.cpp sub_82AA62D8): the roster table is patched before the game reads it
// for anything (menu assets are loaded per roster id early on).
void wwe13_roster_early(uint8_t* base) { EnsureRoster(base); }

// The record getter: the save reload zeroes the new records again, so they are re-filled before any record lookup.
extern "C" PPC_FUNC(__imp__sub_827216A8);
extern "C" PPC_FUNC(sub_827216A8) {
  if (!g_new_records.empty()) EnsureRecords(base);
  if (Trace() && (ctx.r3.u32 == 322 || ctx.r3.u32 == 360)) {  // who looks up records (finding the select grid builder)
    static std::vector<uint64_t> callers;
    const uint64_t key = (uint64_t(ctx.r3.u32) << 32) | uint32_t(ctx.lr);
    bool seen = false;
    for (uint64_t c : callers) seen |= c == key;
    if (!seen && callers.size() < 400) {
      callers.push_back(key);
      REXLOG_INFO("[wwe13-roster] record {} looked up from {:08x}", ctx.r3.u32, uint32_t(ctx.lr));
    }
  }
  __imp__sub_827216A8(ctx, base);
}

// Select screen candidate lists (sub_82839468, r4 = 5 lists of [u32 count][500 ids][500 flags], 4004 bytes apart).
// WWE13_ROSTER_TRACE: logs each list's ids after it is built.
extern "C" PPC_FUNC(__imp__sub_82839468);
extern "C" PPC_FUNC(sub_82839468) {
  const uint32_t lists = ctx.r4.u32, mode = ctx.r5.u32;
  const uint32_t before = PPC_LOAD_U32(lists + 0x162F4);
  PatchGridRows(base, lists + 0x162F4);  // (the rows already exist when the select screen calls this)
  wwe13_unlock_select_build(true);  // (unlock_all.cpp: locked superstars offered while the lists are built)
  __imp__sub_82839468(ctx, base);
  wwe13_unlock_select_build(false);
  g_last_lists.store(lists);
  if (mode == 2 && g_row_search_frames.load() < 600) g_row_search_frames.store(600);
  if (Trace())
    REXLOG_INFO("[wwe13-roster] sub_82839468 lists {:08x} mode {}: row1[0] {} -> {}", lists, mode, before,
                PPC_LOAD_U32(lists + 0x162F4));
  if (!Trace()) return;
  for (uint32_t l = 0; l < 5; ++l) {
    const uint32_t at = lists + l * 4004;
    const uint32_t n = PPC_LOAD_U32(at);
    if (n == 0 || n > 500) continue;
    std::string ids;
    for (uint32_t i = 0; i < n; ++i) ids += " " + std::to_string(PPC_LOAD_U32(at + 4 + i * 4));
    REXLOG_INFO("[wwe13-roster] select list {} (mode {}, {:08x}): {} ids:{}", l, mode, at, n, ids);
  }
}

extern "C" PPC_FUNC(__imp__sub_82A67B00);
extern "C" PPC_FUNC(sub_82A67B00) {
  EnsureRoster(base);
  const uint32_t id = ctx.r3.u32, a2 = ctx.r4.u32;
  static bool dumped = false;
  if (Trace() && !dumped && id == 360) {
    dumped = true;
    DumpRecord(base, 322);
    DumpRecord(base, 360);
  }
  __imp__sub_82A67B00(ctx, base);
  if (id >= 340 && id < 400) TraceCall("available_82A67B00", id, a2 & 0xFFFF, ctx.r3.u32);
}
extern "C" PPC_FUNC(__imp__sub_82A67C70);
extern "C" PPC_FUNC(sub_82A67C70) {
  EnsureRoster(base);
  __imp__sub_82A67C70(ctx, base);
}
extern "C" PPC_FUNC(__imp__sub_82A68530);
extern "C" PPC_FUNC(sub_82A68530) {
  EnsureRoster(base);
  __imp__sub_82A68530(ctx, base);
}
extern "C" PPC_FUNC(__imp__sub_82A695B0);
extern "C" PPC_FUNC(sub_82A695B0) {
  EnsureRoster(base);
  __imp__sub_82A695B0(ctx, base);
}

// WWE13_ROSTER_TRACE: which select-scene call writes the grid rows (watching row 1's first cell at its usual address).
namespace {
// The first grid row in [lo, hi): [0x2004xxxx][x][0][54] followed by a superstar id.
uint32_t FindRow(uint8_t* base, uint32_t lo, uint32_t hi) {
  for (uint32_t a = lo; a < hi; a += 4) {
    if (PPC_LOAD_U32(a) != 54) continue;
    if ((PPC_LOAD_U32(a - 12) >> 16) != 0x2004) continue;
    const uint32_t id = PPC_LOAD_U32(a + 4);
    if (id >= 100 && id < 500) return a + 4;
  }
  return 0;
}
void RowWatch(uint8_t* base, const char* name, uint32_t before) {
  const uint32_t after = FindRow(base, 0xA4900000, 0xA4A80000);
  if (Trace() && before != after)
    REXLOG_INFO("[wwe13-roster] {}: grid row {:08x} -> {:08x} (first {})", name, before, after,
                after ? PPC_LOAD_U32(after) : 0);
}
}  // namespace
#define ROW_WATCHED(fn)                                              \
  extern "C" PPC_FUNC(__imp__##fn);                                  \
  extern "C" PPC_FUNC(fn) {                                          \
    const uint32_t before = Trace() ? FindRow(base, 0xA4900000, 0xA4A80000) : 0; \
    __imp__##fn(ctx, base);                                          \
    if (Trace()) RowWatch(base, #fn, before);                        \
  }
// (the select screen's setup: its cells ask whether each superstar is unlocked - unlock_all.cpp)
extern "C" PPC_FUNC(__imp__sub_82773FD8);
extern "C" PPC_FUNC(sub_82773FD8) {
  const uint32_t before = Trace() ? FindRow(base, 0xA4900000, 0xA4A80000) : 0;
  g_select_screen.store(ctx.r3.u32);  // (vtable 0x8203BDC8)
  wwe13_unlock_select_build(true);
  __imp__sub_82773FD8(ctx, base);
  wwe13_unlock_select_build(false);
  if (Trace()) RowWatch(base, "sub_82773FD8", before);
}
ROW_WATCHED(sub_82777400)
ROW_WATCHED(sub_82776610)
ROW_WATCHED(sub_8276F188)
ROW_WATCHED(sub_8276E5A8)
ROW_WATCHED(sub_82835FD8)
ROW_WATCHED(sub_82837ED0)
ROW_WATCHED(sub_82839C08)
ROW_WATCHED(sub_82792C70)
ROW_WATCHED(sub_82792C58)
ROW_WATCHED(sub_82792BD8)
ROW_WATCHED(sub_82795C38)
ROW_WATCHED(sub_82777F80)

// WWE13_ROSTER_TRACE: the entrance music the game asks Wwise for (sub_829D52F8 formats "@%s-MUS_%04d_%1d_%1d").
extern "C" PPC_FUNC(__imp__sub_829D52F8);
extern "C" PPC_FUNC(sub_829D52F8) {
  if (Trace())
    REXLOG_INFO("[wwe13-roster] music event: MUS_{:04d}_{}_{} (lr {:08x})", ctx.r4.u32, ctx.r5.u32, ctx.r6.u32,
                uint32_t(ctx.lr));
  __imp__sub_829D52F8(ctx, base);
}

// For the on-screen hint (mods_hint.cpp): the select screen is up and a mod adds superstars.
bool wwe13_select_visible_frames(uint32_t frames) { return g_select_seen_frame.load() && g_frame_no.load() - g_select_seen_frame.load() <= frames; }

bool wwe13_roster_hint(int* page, int* pages) {
  if (g_new_records.empty() || g_frame_no.load() - g_select_seen_frame.load() > 30) return false;
  *page = g_page.load();
  *pages = PageCount();
  return true;
}

// The select screen (vtable 0x8203BDC8): [4] sub_82777F80 enter = sub_82773FD8 (its lists: the rows PatchGridRows pages) and
// sub_82776610(grid at +56776, the cells built from the lists); [6] sub_82773A78 its update, on the game's UI thread. A
// page switch runs those two again from the update instead of leaving and re-entering the screen, so the players' picks
// and cursors stay.
extern "C" PPC_FUNC(__imp__sub_82776610);
void wwe13_select_refresh(PPCContext& ctx, uint8_t* base, const char* from) {
  const uint32_t screen = g_select_screen.load();
  if (!screen || ctx.r3.u32 != screen || !g_refresh_pending.exchange(false)) return;
  const uint32_t grid = PPC_LOAD_U32(screen + 56776);
  REXLOG_INFO("[wwe13-roster] page {}: rebuilding the select grid in place from {} (screen {:08x}, grid {:08x})",
              g_page.load(), from, screen, grid);
  g_hot_frames.store(120);
  PPCContext saved = ctx;
  sub_82773FD8(ctx, base);
  if (const uint32_t lists = g_last_lists.load()) PatchGridRows(base, lists + 0x162F4);  // (the rows of the new page)
  {
    const uint32_t h54 = wwe13_heap_start(base, 54);
    uint32_t lo = h54 ? h54 + 0x35BA00 : 0xA4900000;
    const uint32_t hi = h54 ? h54 + 0x4DBA00 : 0xA4A80000;
    for (int n = 0; n < 8; ++n) {
      const uint32_t row = FindRow(base, lo, hi);
      if (!row) break;
      PatchGridRows(base, row);
      lo = row + 0x800;
    }
  }
  if (grid) {
    ctx = saved;
    ctx.r3.u64 = grid;
    PPC_STORE_U32(grid + 0x10FA0, 0);  // (its "cells built" flag: sub_82776610 returns at once when set - as sub_82777400 does)
    __imp__sub_82776610(ctx, base);
  }
  ctx = saved;
}

// The select screen's per-frame method (vtable [1], sub_8276DC40, r3 = the screen) runs the page switch.
extern "C" PPC_FUNC(__imp__sub_8276DC40);
extern "C" PPC_FUNC(sub_8276DC40) {
  wwe13_select_refresh(ctx, base, "sub_8276DC40");
  __imp__sub_8276DC40(ctx, base);
}
