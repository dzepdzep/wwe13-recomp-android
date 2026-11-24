// k14port: ports the WWE 2K14 superstars WWE '13 doesn't have into a WWE '13 mod folder, on the device, from the
// player's own discs: their WWE '13 game files (the folder the game runs from) and their WWE 2K14 disc image (ISO).
// Nothing of either game is in this tool. Same steps as the PC tools (tools/pac, tools/sound, tools/video):
// models + select pictures + names, titantrons (Bink 2 KB2f -> KB2d), entrance music (Wwise), entrances (EVP/EVZ),
// movesets and ratings (roster.txt fields read by roster_mods.cpp).
//
//   k14port --w13 <WWE '13 folder> --iso <WWE 2K14 .iso> --out <mods/k14 folder> [--no-titantrons] [--ids 360,361]
// Progress lines on stdout: "progress <0-100> <text>"; the last line "done <n>" or "error <text>".
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <set>

#include "k14port_io.h"

extern "C" int kb2d_main(int argc, char** argv);

namespace {
// The 2K14 superstars WWE '13 lacks (the disc's own ids; 374 has no ch pac on the disc).
const int kIds[] = {109, 119, 130, 136, 138, 144, 160, 172, 193, 196, 197, 198, 215, 228, 229, 230, 249, 250, 253,
                    278, 290, 298, 334, 336, 353, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371};
constexpr int kCopy = 322;  // the WWE '13 superstar whose records the new ones start from
// Parts taken from the WWE '13 template attire: shared table, a PACH 2K14 lacks, the YANM animation, DUMY.
const uint32_t kKeepW13[] = {0x32, 0x64, 0x6e, 0x9c40};

std::string g_w13, g_out;
std::unique_ptr<Iso> g_iso;

void progress(int pct, const std::string& text) { say("progress %d %s", pct, text.c_str()); }

std::string w13(const std::string& rel) {
  // the game folder's file, whatever the case of its names (Android storage is case sensitive)
  std::string path = g_w13;
  size_t a = 0;
  while (a < rel.size()) {
    size_t b = rel.find('/', a);
    if (b == std::string::npos) b = rel.size();
    const std::string part = rel.substr(a, b - a);
    std::string hit = part;
    for (const auto& n : list_dir(path)) {
      if (n.size() == part.size() && std::equal(n.begin(), n.end(), part.begin(), [](char x, char y) {
            return std::tolower(uint8_t(x)) == std::tolower(uint8_t(y));
          })) {
        hit = n;
        break;
      }
    }
    path += "/" + hit;
    a = b + 1;
  }
  return path;
}
Bytes w13_pac(const std::string& rel) { return load_pac(read_file(w13(rel))); }
Bytes k14_pac(const std::string& rel) {
  Bytes raw = g_iso->file(rel);
  if (raw.empty()) fail("the 2K14 disc has no " + rel);
  return load_pac(raw);
}

std::map<uint32_t, Bytes> char_table(const Bytes& etc, const char* name) {
  const auto es = epk8_entries(etc);
  std::map<uint32_t, Bytes> out;
  for (auto& [pid, b] : pach_parse(pac_get(etc, es, "CHAR", name))) out[pid] = std::move(b);
  return out;
}

std::string ascii_upper(const std::string& utf8) {
  // UTF-8 -> ASCII capitals (WWE '13's name fonts): accented Latin letters lose their accents
  std::string o;
  for (size_t i = 0; i < utf8.size(); ++i) {
    uint8_t c = uint8_t(utf8[i]);
    if (c < 0x80) {
      o += char(std::toupper(c));
      continue;
    }
    if ((c & 0xe0) == 0xc0 && i + 1 < utf8.size()) {
      const unsigned cp = ((c & 0x1f) << 6) | (uint8_t(utf8[i + 1]) & 0x3f);
      ++i;
      static const char* map = "AAAAAAACEEEEIIIIDNOOOOOxOUUUUYPsaaaaaaaceeeeiiiidnooooo/ouuuuypy";
      if (cp >= 0xc0 && cp <= 0xff) {
        const char m = map[cp - 0xc0];
        if (std::isalpha(uint8_t(m))) o += char(std::toupper(uint8_t(m)));
      }
      continue;
    }
    while (i + 1 < utf8.size() && (uint8_t(utf8[i + 1]) & 0xc0) == 0x80) ++i;
  }
  return o;
}

// ------------------------------------------------------------------------------------------------ names
std::map<int, std::pair<std::string, std::string>> k14_names(const std::map<uint32_t, Bytes>& dat14,
                                                              const std::vector<int>& ids) {
  std::map<int, std::pair<uint16_t, uint16_t>> want;
  std::set<uint32_t> sids;
  for (int c : ids) {
    auto it = dat14.find(uint32_t(c));
    if (it == dat14.end()) continue;
    const Bytes r = unpack_any(it->second);
    if (r.size() < 14) continue;
    want[c] = {le16(&r[10]), le16(&r[12])};
    sids.insert(le16(&r[10]));
    sids.insert(le16(&r[12]));
  }
  std::map<uint32_t, std::string> found;
  const Bytes sp = k14_pac("pac/string.pac");
  for (const auto& e : epk8_entries(sp)) {
    if (e.group != "SDB" || e.name.size() < 4 || e.name.substr(2) != "00") continue;  // (English)
    for (auto& [pid, b] : pach_parse(sub(sp, e.off, e.size))) {
      Bytes u;
      try {
        u = unpack_any(b);
      } catch (...) {
        continue;
      }
      if (u.size() < 8) continue;
      const uint32_t cnt = le32(&u[4]);
      if (!(cnt > 0 && cnt < 100000) || 8 + 12 * size_t(cnt) > u.size()) continue;
      for (uint32_t j = 0; j < cnt; ++j) {
        const uint32_t off = le32(&u[8 + 12 * j]), ln = le32(&u[12 + 12 * j]), sid = le32(&u[16 + 12 * j]);
        if (!sids.count(sid) || found.count(sid) || off + ln > u.size()) continue;
        std::string s(reinterpret_cast<const char*>(&u[off]), ln);
        s = s.substr(0, s.find('\0'));
        found[sid] = s;
      }
    }
  }
  std::map<int, std::pair<std::string, std::string>> out;
  for (auto& [c, ss] : want)
    if (found.count(ss.first) && found.count(ss.second))
      out[c] = {ascii_upper(found[ss.first]), ascii_upper(found[ss.second])};
  return out;
}

// ------------------------------------------------------------------------------------------------ models
Bytes build_ch(const Bytes& k, const Bytes& t, int cid) {
  const auto tes = epk8_entries(t);
  std::map<uint32_t, Bytes> tmpl;
  for (const auto& e : tes)
    if (e.group == "EMD") {
      for (auto& [pid, b] : pach_parse(sub(t, e.off, e.size))) tmpl[pid] = b;
      break;
    }
  PacGroup g{"EMD", 0x11, {}};
  for (const auto& e : epk8_entries(k)) {
    if (e.group != "EMD") continue;
    std::map<uint32_t, Bytes> parts;
    for (auto& [pid, b] : pach_parse(sub(k, e.off, e.size))) parts[pid] = b;
    for (uint32_t pid : kKeepW13)
      if (tmpl.count(pid)) parts[pid] = tmpl[pid];
    std::vector<std::pair<uint32_t, Bytes>> ps;
    for (auto& [pid, b] : parts) ps.emplace_back(pid, trim_bpe(b));
    const std::string name = e.name.size() == 8 ? "000" + std::to_string(cid) + e.name.substr(6) : e.name;
    g.ents.emplace_back(name, pach_build(ps, 4));
  }
  if (g.ents.empty()) fail("ch" + std::to_string(cid) + ".pac: no attires");
  return epk8_write({g});
}

Bytes chetc_with(const Bytes& d, const std::map<int, std::pair<std::string, std::string>>& names) {
  std::vector<std::pair<std::pair<std::string, std::string>, Bytes>> adds;
  for (const auto& e : epk8_entries(d)) {
    if (e.group != "CHAR" || (e.name != "DAT" && e.name != "PRO")) continue;
    std::map<uint32_t, Bytes> parts;
    for (auto& [pid, b] : pach_parse(sub(d, e.off, e.size))) parts[pid] = b;
    const Bytes src = unpack_any(parts.at(kCopy));
    for (const auto& [cid, nm] : names) {
      Bytes rec = src;
      if (e.name == "DAT") {
        for (int j = 0; j < 4; ++j) put_le16(&rec[0x102 + 2 * j], uint16_t(cid * 10 + 1 + j));
        for (size_t off : {0x38u, 0x10au, 0x110u}) put_le16(&rec[off], uint16_t(cid));
        auto put = [&](size_t off, size_t len, const std::string& v) {
          for (size_t i = 0; i < len; ++i) rec[off + i] = i < v.size() ? uint8_t(v[i]) : 0;
        };
        put(0x3a, 0x44, nm.first);
        put(0x7e, 0x5c, nm.first);
        put(0xda, 0x26, nm.second);
      }
      parts[uint32_t(cid)] = bpe_compress_wrap(rec);
    }
    std::vector<std::pair<uint32_t, Bytes>> ps;
    for (auto& [pid, b] : parts) ps.emplace_back(pid, magic(b, "BPE ") ? trim_bpe(b) : b);
    adds.push_back({{e.group, e.name}, pach_build(ps, 4)});
  }
  return epk8_add(d, adds);
}

Bytes menu_with(const Bytes& w, const Bytes& k, const std::vector<int>& ids) {
  const auto wes = epk8_entries(w), kes = epk8_entries(k);
  std::vector<std::pair<std::pair<std::string, std::string>, Bytes>> adds;
  for (int i : ids) {
    char name[8];
    std::snprintf(name, sizeof name, "%04d", i);
    for (const char* g : {"SSFA", "SSFB", "SSFC"}) {
      Bytes b = pac_get(k, kes, g, name);
      if (!b.empty()) adds.push_back({{g, name}, trim_bpe(b)});
    }
  }
  auto icons_of = [](const Bytes& d, const std::vector<PacEntry>& es) {
    std::map<uint32_t, Bytes> m;
    const auto outer = pach_parse(pac_get(d, es, "MENU", "SSFD"));
    if (outer.empty()) return m;
    for (auto& [pid, b] : pach_parse(unpack_any(outer[0].second))) m[pid] = b;
    return m;
  };
  auto icons = icons_of(w, wes);
  const auto k_icons = icons_of(k, kes);
  for (int i : ids)
    if (k_icons.count(uint32_t(i))) icons[uint32_t(i)] = k_icons.at(uint32_t(i));
  std::vector<std::pair<uint32_t, Bytes>> ps(icons.begin(), icons.end());
  const Bytes inner = pach_build(ps, 16);
  adds.push_back({{"MENU", "SSFD"}, pach_build({{0, bpe_compress_wrap(inner)}}, 4)});
  return epk8_add(w, adds);
}

// ------------------------------------------------------------------------------------------------ profiles
std::vector<uint16_t> u16s(const Bytes& b, size_t n) {
  std::vector<uint16_t> v(n);
  for (size_t i = 0; i < n && 2 * i + 1 < b.size(); ++i) v[i] = le16(&b[2 * i]);
  return v;
}

// moveset: WWE '13 profile u16 field -> 2K14 field, learned from the superstars both games have (port_moves.py)
std::vector<std::pair<int, int>> learn_move_map(const std::map<uint32_t, Bytes>& p13,
                                                const std::map<uint32_t, Bytes>& p14, std::set<uint16_t>* known) {
  std::map<uint32_t, std::vector<uint16_t>> W, K;
  for (auto& [i, b] : p13) {
    const Bytes u = unpack_any(b);
    if (u.size() >= 1180) W[i] = u16s(u, 590);
  }
  for (auto& [i, b] : p14) {
    const Bytes u = unpack_any(b);
    if (u.size() >= 1276) K[i] = u16s(u, 638);
  }
  std::vector<uint32_t> both;
  for (auto& [i, w] : W)
    if (i >= 100 && i < 400 && K.count(i) && w[0x120] == i && K[i][0x120] == i) both.push_back(i);
  std::vector<std::pair<int, int>> fmap;
  for (int wi = 0; wi < 0x120; ++wi) {
    int best = -1, best_n = -1;
    for (int ki = 0; ki < 0x130; ++ki) {
      int n = 0;
      for (uint32_t i : both)
        if (W[i][wi] == K[i][ki] && W[i][wi]) ++n;
      if (n > best_n) best_n = n, best = ki;
    }
    if (best_n >= 20) fmap.emplace_back(wi, best);
  }
  for (auto& [i, w] : W)
    for (int j = 0; j < 0x120; ++j) known->insert(w[j]);
  return fmap;
}

// ------------------------------------------------------------------------------------------------ titantrons
std::map<int, int> port_titantrons(const std::map<uint32_t, Bytes>& pro14, const std::vector<int>& ids) {
  std::set<int> used;
  for (const auto& n : list_dir(w13("movies/titantron")))
    if (n.size() > 5 && n.substr(n.size() - 5) == ".bik2") used.insert(std::atoi(n.c_str()));
  const std::string tdir = g_out + "/movies/titantron";
  make_dirs(tdir);
  std::map<int, int> num_of_tid, result;
  size_t k = 0;
  for (int cid : ids) {
    ++k;
    auto it = pro14.find(uint32_t(cid));
    if (it == pro14.end()) continue;
    const Bytes r = unpack_any(it->second);
    const int tid = le16(&r[0x242]);
    if (tid == 0 || tid == 255 || tid == 0xffff) continue;
    if (!num_of_tid.count(tid)) {
      const Bytes src = g_iso->file("movies/titantron/" + std::to_string(tid) + ".bik2");
      if (src.empty()) continue;
      int n = cid;
      std::set<int> taken = used;
      for (auto& [a, b] : num_of_tid) taken.insert(b);
      while (taken.count(n)) n += 1000;
      const std::string tmp = g_out + "/.tron_in.bik2", dst = tdir + "/" + std::to_string(n) + ".bik2";
      write_file(tmp, src);
      progress(int(40 + 25 * k / ids.size()), "titantron for " + std::to_string(cid));
      std::string a0 = "kb2d_enc", a1 = tmp, a2 = dst;
      char* argv[] = {a0.data(), a1.data(), a2.data(), nullptr};
      const int rc = kb2d_main(3, argv);
      std::remove(tmp.c_str());
      if (rc) continue;
      num_of_tid[tid] = n;
    }
    result[cid] = num_of_tid[tid];
  }
  return result;
}

// ------------------------------------------------------------------------------------------------ music (Wwise)
uint32_t fnv(const std::string& s) {
  uint32_t h = 2166136261u;
  for (char c : s) {
    h *= 16777619u;
    h ^= uint8_t(std::tolower(uint8_t(c)));
  }
  return h;
}
std::map<std::string, std::pair<size_t, size_t>> bnk_chunks(const Bytes& d) {
  std::map<std::string, std::pair<size_t, size_t>> out;
  for (size_t p = 0; p + 8 <= d.size();) {
    const size_t ln = be32(&d[p + 4]);
    out[std::string(reinterpret_cast<const char*>(&d[p]), 4)] = {p + 8, ln};
    p += 8 + ln;
  }
  return out;
}
struct HObj {
  int type;
  size_t pos, len;  // pos = the object's type byte; body at pos + 5
};
std::map<uint32_t, HObj> hirc(const Bytes& d) {
  auto c = bnk_chunks(d);
  std::map<uint32_t, HObj> objs;
  if (!c.count("HIRC")) return objs;
  size_t q = c["HIRC"].first;
  const uint32_t n = be32(&d[q]);
  q += 4;
  for (uint32_t i = 0; i < n && q + 9 <= d.size(); ++i) {
    const size_t ln = be32(&d[q + 1]);
    objs[be32(&d[q + 5])] = {d[q], q, ln};
    q += 5 + ln;
  }
  return objs;
}
struct Pck {
  uint32_t ver;
  Bytes head;  // the whole header
  size_t ls, es;
  struct Ent {
    uint32_t id, blk, size, start, lang;
  };
  std::vector<Ent> banks, streams;
};
Pck read_pck_head(const Bytes& h) {
  if (!magic(h, "AKPK")) fail("not a Wwise package");
  Pck p;
  p.head = h;
  p.ver = be32(&h[8]);
  p.ls = be32(&h[12]);
  const size_t bs = be32(&h[16]), ss = be32(&h[20]);
  p.es = be32(&h[24]);
  size_t q = 28 + p.ls;
  for (auto* v : {&p.banks, &p.streams}) {
    const uint32_t n = be32(&h[q]);
    for (uint32_t i = 0; i < n; ++i) {
      const uint8_t* e = &h[q + 4 + 20 * i];
      v->push_back({be32(e), be32(e + 4), be32(e + 8), be32(e + 12), be32(e + 16)});
    }
    q += (v == &p.banks) ? bs : ss;
  }
  return p;
}

std::map<int, int> port_music(const std::map<uint32_t, Bytes>& pro14, const std::vector<int>& ids) {
  const uint32_t EV = 0xf4757ccf, MU = 0x9a4847cb, EV14 = 0xe85dda21;
  const std::string src = w13("sound/Music.pck"), dst = g_out + "/sound/Music.pck";
  // the WWE '13 package's header and its two banks
  FILE* f = std::fopen(src.c_str(), "rb");
  if (!f) fail("can't read " + src);
  Bytes h(8);
  if (std::fread(h.data(), 1, 8, f) != 8) fail("Music.pck");
  h.resize(8 + be32(&h[4]));
  if (std::fread(h.data() + 8, 1, h.size() - 8, f) != h.size() - 8) fail("Music.pck");
  Pck pk = read_pck_head(h);
  auto read_at = [&](uint64_t off, size_t n) {
    Bytes b(n);
    fseeko(f, off_t(off), SEEK_SET);
    b.resize(std::fread(b.data(), 1, n, f));
    return b;
  };
  Bytes ev13, mu13;
  Pck::Ent mu_ent{};
  for (auto& e : pk.banks) {
    if (e.id == EV) ev13 = read_at(uint64_t(e.start) * e.blk, e.size);
    if (e.id == MU) mu13 = read_at(uint64_t(e.start) * e.blk, e.size), mu_ent = e;
  }
  std::fclose(f);
  if (ev13.empty() || mu13.empty()) fail("Music.pck: entrance music banks not found");
  auto E13 = hirc(ev13), M13 = hirc(mu13);
  // 2K14's structure bank and media table
  auto iso_pck = [&](const char* path, uint64_t* base) {
    uint64_t o;
    uint32_t s;
    if (!g_iso->find(path, &o, &s)) fail(std::string("the 2K14 disc has no ") + path);
    *base = o;
    Bytes hh = g_iso->read(o, 8);
    hh = g_iso->read(o, 8 + be32(&hh[4]));
    return read_pck_head(hh);
  };
  uint64_t es_base, m_base;
  Pck es14 = iso_pck("sound/Game_Sound_E_S.pck", &es_base), m14 = iso_pck("sound/Game_Sound_M.pck", &m_base);
  Bytes ev14;
  for (auto& e : es14.banks)
    if (e.id == EV14) ev14 = g_iso->read(es_base + uint64_t(e.start) * e.blk, e.size);
  if (ev14.empty()) fail("2K14 music bank not found");
  auto H14 = hirc(ev14);
  std::map<uint32_t, Pck::Ent> m14s;
  for (auto& e : m14.streams) m14s[e.id] = e;
  // event Play_MUS_<n>_0_0 -> play action -> container -> its sound
  auto sound_of = [](const Bytes& evb, std::map<uint32_t, HObj>& E, const Bytes& mub, std::map<uint32_t, HObj>& M,
                     int music) -> uint32_t {
    char nm[32];
    std::snprintf(nm, sizeof nm, "Play_MUS_%04d_0_0", music);
    auto it = E.find(fnv(nm));
    if (it == E.end() || it->second.len < 12) return 0;
    const uint32_t aid = be32(&evb[it->second.pos + 5 + 8]);
    auto at = E.find(aid);
    if (at == E.end() || at->second.len < 10) return 0;
    const uint32_t tgt = be32(&evb[at->second.pos + 5 + 6]);
    auto rt = M.find(tgt);
    if (rt == M.end()) return 0;
    const uint8_t* rb = &mub[rt->second.pos + 5];
    for (size_t o = 0; o + 4 <= rt->second.len; ++o) {
      auto st = M.find(be32(rb + o));
      if (st != M.end() && st->second.type == 2) return st->first;
    }
    return 0;
  };
  auto carrier_ok = [&](int n) -> uint32_t {
    const uint32_t sid = sound_of(ev13, E13, mu13, M13, n);
    if (!sid) return 0;
    return be32(&mu13[M13[sid].pos + 5 + 8]) == 2 ? sid : 0;
  };
  // which 2K14 music each superstar has (its own number when its profile names a group's: 902 3MB, 903 Shield ...)
  std::map<int, int> music_of;
  for (int cid : ids) {
    auto it = pro14.find(uint32_t(cid));
    if (it == pro14.end()) continue;
    const Bytes r = unpack_any(it->second);
    int m = le16(&r[0x240]);
    if (m == 0 || m == 255 || m == 0xffff) continue;
    if (!sound_of(ev14, H14, ev14, H14, m)) m = cid;
    if (sound_of(ev14, H14, ev14, H14, m)) music_of[cid] = m;
  }
  std::vector<int> songs;
  for (auto& [c, m] : music_of)
    if (std::find(songs.begin(), songs.end(), m) == songs.end()) songs.push_back(m);
  std::sort(songs.begin(), songs.end());
  std::vector<int> free;
  for (int n = 519; n < 579; ++n)
    if (carrier_ok(n)) free.push_back(n);
  std::map<uint32_t, uint32_t> change;  // WWE '13 sound -> 2K14 media
  std::map<int, int> carrier_of_song;
  std::vector<uint32_t> new_media;
  std::set<uint32_t> used_sounds;
  for (int m : songs) {
    while (!free.empty() && used_sounds.count(carrier_ok(free.front()))) free.erase(free.begin());
    if (free.empty()) break;
    const int n = free.front();
    free.erase(free.begin());
    const uint32_t sid13 = carrier_ok(n), sid14 = sound_of(ev14, H14, ev14, H14, m);
    if (!sid13 || !sid14) continue;
    const uint32_t media = be32(&ev14[H14[sid14].pos + 5 + 12]);
    if (!m14s.count(media)) continue;
    used_sounds.insert(sid13);
    change[sid13] = media;
    carrier_of_song[m] = n;
    new_media.push_back(media);
  }
  // the media bank: the carriers' sounds stream the 2K14 media from the start, their prefetch parts dropped
  std::vector<std::pair<int, Bytes>> objs;
  {
    std::vector<std::pair<size_t, uint32_t>> order;
    for (auto& [id, o] : M13) order.emplace_back(o.pos, id);
    std::sort(order.begin(), order.end());
    for (auto& [pos, id] : order) objs.emplace_back(M13[id].type, sub(mu13, pos + 5, M13[id].len));
  }
  std::set<uint32_t> drop;
  for (auto& [t, b] : objs) {
    const uint32_t oid = be32(&b[0]);
    auto it = change.find(oid);
    if (it == change.end()) continue;
    drop.insert(be32(&b[12]));
    Bytes nb(b.begin(), b.begin() + 8);
    app_be32(nb, 1);
    app_be32(nb, it->second);
    app_be32(nb, it->second);
    nb.insert(nb.end(), b.begin() + 28, b.end());  // (streamed: no offset / in-memory size fields)
    b = nb;
  }
  auto c = bnk_chunks(mu13);
  const size_t s0 = c["DIDX"].first, ln0 = c["DIDX"].second, ds0 = c["DATA"].first;
  Bytes didx, data;
  for (size_t i = 0; i < ln0 / 12; ++i) {
    const uint32_t mid = be32(&mu13[s0 + 12 * i]), off = be32(&mu13[s0 + 12 * i + 4]), size = be32(&mu13[s0 + 12 * i + 8]);
    if (drop.count(mid)) continue;
    pad_to(data, 16);
    app_be32(didx, mid);
    app_be32(didx, uint32_t(data.size()));
    app_be32(didx, size);
    app(data, mu13.data() + ds0 + off, size);
  }
  Bytes newmu;
  for (size_t p = 0; p + 8 <= mu13.size();) {
    const std::string tag(reinterpret_cast<const char*>(&mu13[p]), 4);
    const size_t ln = be32(&mu13[p + 4]);
    Bytes body = sub(mu13, p + 8, ln);
    if (tag == "HIRC") {
      body.clear();
      app_be32(body, uint32_t(objs.size()));
      for (auto& [t, b] : objs) {
        body.push_back(uint8_t(t));
        app_be32(body, uint32_t(b.size()));
        app(body, b);
      }
    } else if (tag == "DIDX")
      body = didx;
    else if (tag == "DATA")
      body = data;
    app(newmu, reinterpret_cast<const uint8_t*>(tag.data()), 4);
    app_be32(newmu, uint32_t(body.size()));
    app(newmu, body);
    p += 8 + ln;
  }
  const size_t room = (mu_ent.size + mu_ent.blk - 1) / mu_ent.blk * mu_ent.blk;
  if (newmu.size() > room) fail("music bank grew past its room");
  // the new package: WWE '13's, the media bank rewritten in place, the 2K14 media appended
  progress(70, "copying the music package");
  if (!copy_file(src, dst)) fail("can't write " + dst + " (disk full?)");
  FILE* o = std::fopen(dst.c_str(), "r+b");
  if (!o) fail("can't write " + dst);
  fseeko(o, off_t(uint64_t(mu_ent.start) * mu_ent.blk), SEEK_SET);
  Bytes padded = newmu;
  if (padded.size() < mu_ent.size) padded.resize(room, 0);
  std::fwrite(padded.data(), 1, padded.size(), o);
  for (auto& e : pk.banks)
    if (e.id == MU) e.size = uint32_t(newmu.size());
  fseeko(o, 0, SEEK_END);
  uint64_t end = uint64_t(ftello(o));
  const uint32_t blk = 2048;
  size_t k = 0;
  for (uint32_t media : new_media) {
    const auto& se = m14s[media];
    const Bytes b = g_iso->read(m_base + uint64_t(se.start) * se.blk, se.size);
    const uint64_t pos = (end + blk - 1) / blk;
    fseeko(o, off_t(pos * blk), SEEK_SET);
    std::fwrite(b.data(), 1, b.size(), o);
    end = pos * blk + b.size();
    pk.streams.push_back({media, blk, uint32_t(b.size()), uint32_t(pos), 0});
    progress(int(70 + 10 * ++k / std::max<size_t>(1, new_media.size())), "music");
  }
  std::sort(pk.streams.begin(), pk.streams.end(), [](auto& a, auto& b) { return a.id < b.id; });
  std::sort(pk.banks.begin(), pk.banks.end(), [](auto& a, auto& b) { return a.id < b.id; });
  Bytes bl, sl;
  app_be32(bl, uint32_t(pk.banks.size()));
  for (auto& e : pk.banks) for (uint32_t v : {e.id, e.blk, e.size, e.start, e.lang}) app_be32(bl, v);
  app_be32(sl, uint32_t(pk.streams.size()));
  for (auto& e : pk.streams) for (uint32_t v : {e.id, e.blk, e.size, e.start, e.lang}) app_be32(sl, v);
  const size_t first_data = [&] {
    uint64_t m = UINT64_MAX;
    for (auto* v : {&pk.banks, &pk.streams})
      for (auto& e : *v)
        if (std::find(new_media.begin(), new_media.end(), e.id) == new_media.end())
          m = std::min<uint64_t>(m, uint64_t(e.start) * e.blk);
    return size_t(m);
  }();
  Bytes head = {'A', 'K', 'P', 'K'};
  app_be32(head, uint32_t(20 + pk.ls + bl.size() + sl.size() + pk.es));
  for (uint32_t v : {pk.ver, uint32_t(pk.ls), uint32_t(bl.size()), uint32_t(sl.size()), uint32_t(pk.es)}) app_be32(head, v);
  app(head, pk.head.data() + 28, pk.ls);
  app(head, bl);
  app(head, sl);
  const size_t ext_at = 28 + pk.ls + be32(&pk.head[16]) + be32(&pk.head[20]);
  app(head, pk.head.data() + ext_at, pk.es);
  if (head.size() > first_data) fail("music package header too big");
  head.resize(first_data, 0);
  fseeko(o, 0, SEEK_SET);
  std::fwrite(head.data(), 1, head.size(), o);
  std::fclose(o);
  std::map<int, int> out;
  for (auto& [cid, m] : music_of)
    if (carrier_of_song.count(m)) out[cid] = carrier_of_song[m];
  return out;
}

// ------------------------------------------------------------------------------------------------ entrances
Bytes fix_evz(const Bytes& blob) {
  // 2K14 names its motions "25000<id>_A00_def"; WWE '13 only knows the 5-character form and hangs on longer ones
  std::vector<std::pair<uint32_t, Bytes>> parts = pach_parse(unpack_any(blob));
  for (auto& [pid, b] : parts) {
    if (b.size() < 0x6c) continue;
    size_t n = 0;
    while (n < 32 && b[0x4c + n]) ++n;
    if (n > 5 && b[0x4c] == '2' && b[0x4d] == '5' && b[0x4e] == '0') std::fill(b.begin() + 0x4c + 5, b.begin() + 0x6c, 0);
  }
  return bpe_store_wrap(pach_build(parts, 1));
}

std::map<int, int> port_entrances(const std::map<uint32_t, Bytes>& pro14, const std::vector<int>& ids,
                                  std::vector<std::string>* changed_pacs) {
  std::map<int, int> want;  // superstar -> entrance id
  std::set<int> need;
  for (int cid : ids) {
    auto it = pro14.find(uint32_t(cid));
    if (it == pro14.end()) continue;
    const Bytes r = unpack_any(it->second);
    const int e = le16(&r[0x244]);
    if (e >= 100 && e < 1000) want[cid] = e, need.insert(e);
  }
  std::set<int> have;
  for (int nn = 0; nn < 10; ++nn) {
    char name[32];
    std::snprintf(name, sizeof name, "pac/evt/Entrance%02d.pac", nn);
    const std::string wpath = w13(name);
    if (!file_exists(wpath)) continue;
    progress(82 + nn, std::string("entrances ") + name);
    Bytes wd = load_pac(read_file(wpath));
    std::set<std::string> wnames;
    for (const auto& e : epk8_entries(wd)) {
      wnames.insert(e.name);
      if (e.group == "EVP" && e.name.size() == 3) have.insert(std::atoi(e.name.c_str()));
    }
    Bytes kraw = g_iso->file(name);
    if (kraw.empty()) continue;
    const Bytes kd = load_pac(kraw);
    std::vector<std::pair<std::pair<std::string, std::string>, Bytes>> adds;
    std::set<int> got;
    for (const auto& e : epk8_entries(kd)) {
      if ((e.group != "EVP" && e.group != "EVZ") || wnames.count(e.name)) continue;
      if (e.name.size() != 3 && e.name.size() != 5) continue;
      const int id = std::atoi(e.name.substr(e.name.size() - 3).c_str());
      if (!need.count(id) || have.count(id)) continue;
      Bytes blob = sub(kd, e.off, e.size);
      adds.push_back({{e.group, e.name}, e.group == "EVZ" ? fix_evz(blob) : blob});
      got.insert(id);
    }
    if (adds.empty()) continue;
    const std::string out = g_out + "/" + name;
    write_file(out, epk8_add(wd, adds));
    changed_pacs->push_back(name);
    for (int id : got) have.insert(id);
  }
  std::map<int, int> out;
  for (auto& [cid, e] : want)
    if (have.count(e)) out[cid] = e;
  return out;
}

std::string hexs(int v) {
  char b[16];
  std::snprintf(b, sizeof b, "%x", v);
  return b;
}

int run(int argc, char** argv) {
  std::string iso;
  bool trons = true;
  std::vector<int> ids(std::begin(kIds), std::end(kIds));
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--w13" && i + 1 < argc) g_w13 = argv[++i];
    else if (a == "--iso" && i + 1 < argc) iso = argv[++i];
    else if (a == "--out" && i + 1 < argc) g_out = argv[++i];
    else if (a == "--no-titantrons") trons = false;
    else if (a == "--ids" && i + 1 < argc) {
      ids.clear();
      for (const char* p = argv[++i]; *p;) {
        ids.push_back(int(std::strtol(p, const_cast<char**>(&p), 10)));
        while (*p == ',') ++p;
      }
    }
  }
  if (g_w13.empty() || iso.empty() || g_out.empty()) fail("usage: k14port --w13 <dir> --iso <2K14 iso> --out <dir>");
  progress(0, "opening the WWE 2K14 disc");
  g_iso = std::make_unique<Iso>(iso);
  if (!g_iso->has("pac/ch/chEtc.pac") || !g_iso->has("sound/Game_Sound_M.pck"))
    fail("this disc image isn't WWE 2K14");
  make_dirs(g_out + "/pac/ch");
  make_dirs(g_out + "/pac/menu");
  progress(2, "reading superstar data");
  const Bytes etc14 = k14_pac("pac/ch/chEtc.pac");
  const auto dat14 = char_table(etc14, "DAT"), pro14 = char_table(etc14, "PRO");
  const Bytes etc13 = w13_pac("pac/ch/chEtc.pac");
  const auto dat13 = char_table(etc13, "DAT"), pro13 = char_table(etc13, "PRO");
  auto names = k14_names(dat14, ids);
  // models
  const Bytes tmpl = w13_pac("pac/ch/ch" + std::to_string(kCopy) + ".pac");
  std::vector<int> done;
  std::map<int, std::pair<std::string, std::string>> cid_names;
  for (size_t k = 0; k < ids.size(); ++k) {
    const int cid = ids[k];
    const std::string rel = "pac/ch/ch" + std::to_string(cid) + ".pac";
    Bytes raw = g_iso->file(rel);
    if (raw.empty()) continue;
    progress(int(3 + 30 * k / ids.size()), "superstar " + std::to_string(cid) + " " +
                                               (names.count(cid) ? names[cid].first : std::string()));
    write_file(g_out + "/" + rel, build_ch(load_pac(raw), tmpl, cid));
    cid_names[cid] = names.count(cid) ? names[cid]
                                      : std::make_pair("SUPERSTAR " + std::to_string(cid), "SUPERSTAR " + std::to_string(cid));
    done.push_back(cid);
  }
  if (done.empty()) fail("no superstars found on the disc");
  progress(34, "superstar records");
  const Bytes chetc = chetc_with(etc13, cid_names);
  write_file(g_out + "/pac/ch/chEtc.pac", chetc);
  progress(36, "select screen pictures");
  const Bytes menu = menu_with(w13_pac("pac/menu/menuHD.pac"), k14_pac("pac/menu/menuHD.pac"), done);
  write_file(g_out + "/pac/menu/menuHD.pac", menu);
  // titantrons, music, entrances
  std::map<int, int> trons_of, music_of, ent_of;
  if (trons) trons_of = port_titantrons(pro14, done);
  progress(66, "entrance music");
  music_of = port_music(pro14, done);
  std::vector<std::string> evt_pacs;
  ent_of = port_entrances(pro14, done, &evt_pacs);
  // plist360 (+ 4x3): chEtc patched, new pacs added, menuHD and the entrance pacs regenerated
  progress(93, "pac lists");
  for (const char* suffix : {"", "_4x3"}) {
    const std::string dfile = w13(std::string("plist360") + suffix + ".def"),
                      afile = w13(std::string("plist360") + suffix + ".arc");
    const std::string odef = g_out + "/plist360" + suffix + ".def", oarc = g_out + "/plist360" + suffix + ".arc";
    if (!copy_file(dfile, odef)) fail("can't copy " + dfile);
    const std::string tmp = oarc + ".tmp";
    plist_patch(odef, afile, "pac\\ch\\chEtc.pac", chetc, tmp);
    for (int cid : done) {
      const std::string pac = "pac\\ch\\ch" + std::to_string(cid) + ".pac";
      const Bytes built = read_file(g_out + "/pac/ch/ch" + std::to_string(cid) + ".pac");
      if (plist_lists(odef, pac))
        plist_replace(odef, tmp, pac, built, tmp);
      else
        plist_add(odef, tmp, pac, built, odef, tmp);
    }
    if (plist_lists(odef, "pac\\menu\\menuHD.pac")) plist_replace(odef, tmp, "pac\\menu\\menuHD.pac", menu, tmp);
    for (const auto& rel : evt_pacs) {
      std::string pac = rel;
      std::replace(pac.begin(), pac.end(), '/', '\\');
      if (plist_lists(odef, pac)) plist_replace(odef, tmp, pac, read_file(g_out + "/" + rel), tmp);
    }
    std::rename(tmp.c_str(), oarc.c_str());
  }
  // movesets and ratings
  progress(96, "movesets and ratings");
  std::set<uint16_t> known;
  const auto fmap = learn_move_map(pro13, pro14, &known);
  std::map<int, std::set<uint8_t>> seen;  // ratings: the values WWE '13 uses per byte
  std::vector<std::pair<int, int>> stat_bytes;
  for (int wo = 0x04; wo < 0x20; ++wo) stat_bytes.emplace_back(wo, 0x24 + (wo - 0x04) % 14);
  for (int i = 0; i < 5; ++i) stat_bytes.emplace_back(0x112 + i, 0x38 + i);
  for (auto& [i, b] : dat13) {
    const Bytes u = unpack_any(b);
    for (auto& [wo, ko] : stat_bytes)
      if (size_t(wo) < u.size()) seen[wo].insert(u[wo]);
  }
  std::string roster =
      "# ported from WWE 2K14 on this device: id|copy record of|full name|short name|DSBD id (hex)|titantron|music "
      "(carrier no.)|entrance|moveset (profile off:value, hex)|attributes (record off:byte, hex)\n";
  for (int cid : done) {
    std::string line = std::to_string(cid) + "|" + std::to_string(kCopy) + "|" + cid_names[cid].first + "|" +
                       cid_names[cid].second + "|" + hexs(cid * 10 + 1) + "|";
    if (trons_of.count(cid)) line += std::to_string(trons_of[cid]);
    line += "|";
    if (music_of.count(cid)) line += std::to_string(music_of[cid]);
    line += "|";
    if (ent_of.count(cid)) line += std::to_string(ent_of[cid]);
    line += "|";
    std::string moves;
    if (pro14.count(uint32_t(cid))) {
      const Bytes u = unpack_any(pro14.at(uint32_t(cid)));
      if (u.size() >= 1276) {
        const auto K = u16s(u, 638);
        for (auto& [wi, ki] : fmap) {
          const uint16_t v = K[size_t(ki)];
          if (v == 0 || known.count(v)) moves += (moves.empty() ? "" : ",") + hexs(wi * 2) + ":" + hexs(v);
        }
      }
    }
    line += moves + "|";
    std::string stats;
    if (dat14.count(uint32_t(cid))) {
      const Bytes d = unpack_any(dat14.at(uint32_t(cid)));
      for (auto& [wo, ko] : stat_bytes)
        if (size_t(ko) < d.size() && seen[wo].count(d[ko])) stats += (stats.empty() ? "" : ",") + hexs(wo - 4) + ":" + hexs(d[ko]);
      if (d.size() > 0x18) {
        const unsigned wt = le16(&d[0x16]);
        if (wt > 0 && wt < 0x4000) stats += std::string(stats.empty() ? "" : ",") + "30:" + hexs(int(wt >> 8)) + ",31:" + hexs(int(wt & 0xff));
      }
    }
    line += stats;
    roster += line + "\n";
  }
  write_file(g_out + "/roster.txt", Bytes(roster.begin(), roster.end()));
  progress(100, "done");
  say("done %zu", done.size());
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    say("error %s", e.what());
    return 1;
  }
}
