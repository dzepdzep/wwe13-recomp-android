// k14port: shared helpers. Formats as found for the PC tools (tools/pac, tools/sound): see their notes.
#include "k14port_io.h"

#include <dirent.h>
#include <sys/stat.h>
#include <zlib.h>

#include <algorithm>
#include <cstdarg>
#include <cstring>
#include <set>

extern "C" {
#include "mspack.h"
#include "lzx.h"
extern void (*lzx_blk_hook)(unsigned);
}

void fail(const std::string& why) { throw std::runtime_error(why); }

void say(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::vprintf(fmt, ap);
  va_end(ap);
  std::printf("\n");
  std::fflush(stdout);
}

// ------------------------------------------------------------------------------------------------ files
Bytes read_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) fail("can't read " + path);
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  Bytes b(size_t(n > 0 ? n : 0));
  if (n > 0 && std::fread(b.data(), 1, b.size(), f) != b.size()) {
    std::fclose(f);
    fail("can't read " + path);
  }
  std::fclose(f);
  return b;
}
bool file_exists(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0;
}
void make_dirs(const std::string& path) {
  std::string cur;
  for (size_t i = 0; i <= path.size(); ++i) {
    if (i == path.size() || path[i] == '/') {
      if (!cur.empty()) ::mkdir(cur.c_str(), 0775);
    }
    if (i < path.size()) cur += path[i];
  }
}
void write_file(const std::string& path, const Bytes& b) {
  const size_t slash = path.rfind('/');
  if (slash != std::string::npos) make_dirs(path.substr(0, slash));
  const std::string tmp = path + ".part";
  FILE* f = std::fopen(tmp.c_str(), "wb");
  if (!f) fail("can't write " + path);
  if (!b.empty() && std::fwrite(b.data(), 1, b.size(), f) != b.size()) {
    std::fclose(f);
    fail("can't write " + path + " (disk full?)");
  }
  std::fclose(f);
  std::rename(tmp.c_str(), path.c_str());
}
std::vector<std::string> list_dir(const std::string& path) {
  std::vector<std::string> out;
  if (DIR* d = ::opendir(path.c_str())) {
    while (dirent* e = ::readdir(d))
      if (e->d_name[0] != '.') out.push_back(e->d_name);
    ::closedir(d);
  }
  return out;
}
bool copy_file(const std::string& from, const std::string& to) {
  FILE* a = std::fopen(from.c_str(), "rb");
  if (!a) return false;
  const size_t slash = to.rfind('/');
  if (slash != std::string::npos) make_dirs(to.substr(0, slash));
  FILE* b = std::fopen(to.c_str(), "wb");
  if (!b) {
    std::fclose(a);
    return false;
  }
  std::vector<char> buf(1 << 20);
  bool ok = true;
  for (size_t n; (n = std::fread(buf.data(), 1, buf.size(), a)) > 0;)
    if (std::fwrite(buf.data(), 1, n, b) != n) {
      ok = false;
      break;
    }
  std::fclose(a);
  std::fclose(b);
  return ok;
}

// ------------------------------------------------------------------------------------------------ BPE
Bytes bpe_expand(const uint8_t* src, size_t n, size_t out_size) {
  Bytes out;
  out.reserve(out_size);
  size_t p = 0;
  uint8_t left[256], right[256], stack[256];
  while (p < n && out.size() < out_size) {
    for (int i = 0; i < 256; ++i) left[i] = uint8_t(i), right[i] = 0;
    int c = 0;
    while (c < 256) {
      if (p >= n) fail("BPE: truncated table");
      int count = src[p++];
      if (count > 127) {
        c += count - 127;
        count = 0;
      }
      if (c == 256) break;
      for (int k = 0; k <= count && c < 256; ++k) {
        left[c] = src[p++];
        if (left[c] != c) right[c] = src[p++];
        ++c;
      }
    }
    if (p + 2 > n) fail("BPE: truncated block");
    const size_t size = size_t(src[p] | src[p + 1] << 8);  // (the games' parts: u16 LE block sizes)
    p += 2;
    int sp = 0;
    size_t i = 0;
    while (i < size || sp) {
      uint8_t b;
      if (sp)
        b = stack[--sp];
      else {
        if (p >= n) fail("BPE: truncated data");
        b = src[p++];
        ++i;
      }
      if (left[b] == b)
        out.push_back(b);
      else {
        if (sp > 250) fail("BPE: bad pair");
        stack[sp++] = right[b];
        stack[sp++] = left[b];
      }
    }
  }
  return out;
}

static Bytes bpe_header(const Bytes& body, size_t size) {
  Bytes o = {'B', 'P', 'E', ' '};
  app_le32(o, 0x100);
  app_le32(o, uint32_t(body.size()));
  app_le32(o, uint32_t(size));
  app(o, body);
  return o;
}

Bytes bpe_store_wrap(const Bytes& data) {
  // The game's BPE reader has a small fixed buffer: stored blocks stay at 2 KB.
  Bytes body;
  for (size_t p = 0; p < data.size();) {
    const size_t n = std::min<size_t>(2048, data.size() - p);
    body.push_back(0xff);
    body.push_back(0x80);
    body.push_back(0xfe);
    body.push_back(uint8_t(n));
    body.push_back(uint8_t(n >> 8));
    app(body, data.data() + p, n);
    p += n;
  }
  return bpe_header(body, data.size());
}

static Bytes bpe_block(const uint8_t* data, size_t n) {
  uint8_t left[256], right[256];
  bool ever[256] = {};
  for (int i = 0; i < 256; ++i) left[i] = uint8_t(i), right[i] = 0;
  Bytes buf(data, data + n);
  for (uint8_t b : buf) ever[b] = true;
  std::vector<uint32_t> counts(65536);
  for (;;) {
    std::fill(counts.begin(), counts.end(), 0);
    for (size_t i = 0; i + 1 < buf.size(); ++i) ++counts[buf[i] << 8 | buf[i + 1]];
    int code = -1;
    for (int c = 0; c < 256; ++c)
      if (!ever[c] && left[c] == c) {
        code = c;
        break;
      }
    if (code < 0 || buf.size() < 2) break;
    // the most frequent pair (first one on ties, as the PC tool's dict order: first seen)
    uint32_t best_n = 0;
    int best = -1;
    {
      std::vector<uint8_t> seen(65536);
      for (size_t i = 0; i + 1 < buf.size(); ++i) {
        const int k = buf[i] << 8 | buf[i + 1];
        if (seen[k]) continue;
        seen[k] = 1;
        if (counts[k] > best_n) best_n = counts[k], best = k;
      }
    }
    if (best < 0 || best_n < 3) break;
    const uint8_t a = uint8_t(best >> 8), b = uint8_t(best);
    Bytes out;
    out.reserve(buf.size());
    for (size_t i = 0; i < buf.size();) {
      if (i + 1 < buf.size() && buf[i] == a && buf[i + 1] == b) {
        out.push_back(uint8_t(code));
        i += 2;
      } else
        out.push_back(buf[i++]);
    }
    buf.swap(out);
    left[code] = a;
    right[code] = b;
    ever[code] = true;
  }
  Bytes tab;
  auto entry = [&](int k) {
    tab.push_back(left[k]);
    if (left[k] != k) tab.push_back(right[k]);
  };
  for (int c = 0; c < 256;) {
    if (left[c] == c) {
      int r = 0;
      while (c + r < 256 && left[c + r] == c + r && r < 128) ++r;
      tab.push_back(uint8_t(r + 127));
      c += r;
      if (c == 256) break;
      entry(c);
      ++c;
    } else {
      int k = 0;
      while (c + k < 256 && left[c + k] != c + k && k < 128) ++k;
      tab.push_back(uint8_t(k - 1));
      for (int j = c; j < c + k; ++j) entry(j);
      c += k;
    }
  }
  tab.push_back(uint8_t(buf.size()));
  tab.push_back(uint8_t(buf.size() >> 8));
  app(tab, buf);
  return tab;
}

Bytes bpe_compress_wrap(const Bytes& data) {
  Bytes body;
  for (size_t p = 0; p < data.size();) {
    bool seen[256] = {};
    int distinct = 0;
    size_t q = p;
    while (q < data.size() && q - p < 4096) {
      if (!seen[data[q]]) {
        if (distinct >= 230) break;
        seen[data[q]] = true;
        ++distinct;
      }
      ++q;
    }
    app(body, bpe_block(data.data() + p, q - p));
    p = q;
  }
  return bpe_header(body, data.size());
}

Bytes trim_bpe(const Bytes& b) {
  if (magic(b, "BPE ") && b.size() >= 16) return sub(b, 0, 16 + le32(&b[8]));
  return b;
}

Bytes zlib_inflate(const uint8_t* p, size_t n, size_t hint) {
  Bytes out(hint ? hint : n * 4);
  z_stream z{};
  if (inflateInit(&z) != Z_OK) fail("zlib");
  z.next_in = const_cast<uint8_t*>(p);
  z.avail_in = uInt(n);
  size_t got = 0;
  for (;;) {
    if (got == out.size()) out.resize(out.size() * 2 + 1024);
    z.next_out = out.data() + got;
    z.avail_out = uInt(out.size() - got);
    const int r = inflate(&z, Z_NO_FLUSH);
    got = out.size() - z.avail_out;
    if (r == Z_STREAM_END) break;
    if (r != Z_OK && r != Z_BUF_ERROR) {
      inflateEnd(&z);
      fail("zlib data error");
    }
    if (r == Z_BUF_ERROR && z.avail_in == 0) break;
  }
  inflateEnd(&z);
  out.resize(got);
  return out;
}

Bytes unpack_any(const Bytes& b) {
  if (magic(b, "BPE ") && b.size() >= 16) {
    const size_t csz = le32(&b[8]), usz = le32(&b[12]);
    if (16 + csz > b.size()) fail("BPE part cut short");
    return bpe_expand(b.data() + 16, csz, usz);
  }
  if (magic(b, "ZLIB") && b.size() >= 16) return zlib_inflate(b.data() + 16, b.size() - 16, le32(&b[12]));
  return b;
}

// ------------------------------------------------------------------------------------------------ LZX / XCTD
namespace {
struct MFile {
  const uint8_t* in;
  uint8_t* out;
  size_t len, pos;
};
mspack_file* m_open(mspack_system*, const char*, int) { return nullptr; }
void m_close(mspack_file*) {}
int m_read(mspack_file* f, void* b, int n) {
  auto* m = reinterpret_cast<MFile*>(f);
  size_t r = m->len - m->pos;
  if (size_t(n) < r) r = size_t(n);
  std::memcpy(b, m->in + m->pos, r);
  m->pos += r;
  return int(r);
}
int m_write(mspack_file* f, void* b, int n) {
  auto* m = reinterpret_cast<MFile*>(f);
  size_t r = m->len - m->pos;
  if (size_t(n) < r) r = size_t(n);
  std::memcpy(m->out + m->pos, b, r);
  m->pos += r;
  return int(r);
}
int m_seek(mspack_file*, off_t, int) { return 0; }
off_t m_tell(mspack_file* f) { return off_t(reinterpret_cast<MFile*>(f)->pos); }
void m_msg(mspack_file*, const char*, ...) {}
void* m_alloc(mspack_system*, size_t n) { return std::malloc(n); }
void m_free(void* p) { std::free(p); }
void m_copy(void* a, void* b, size_t n) { std::memmove(b, a, n); }
mspack_system g_sys = {m_open, m_close, m_read, m_write, m_seek, m_tell, m_msg, m_alloc, m_free, m_copy, nullptr};

std::vector<unsigned>* g_blks = nullptr;
void blk_hook(unsigned v) {
  if (g_blks) g_blks->push_back(v);
}

// decodes concatenated chunks (window 2^17, 32 KB frames); returns bytes produced, *err set on an error
size_t lzx_dec(const Bytes& in, uint8_t* out, size_t outlen, bool* err, std::vector<unsigned>* blks) {
  MFile src{in.data(), nullptr, in.size(), 0}, dst{nullptr, out, outlen, 0};
  g_blks = blks;
  lzx_blk_hook = blk_hook;
  lzxd_stream* s = lzxd_init(&g_sys, reinterpret_cast<mspack_file*>(&src), reinterpret_cast<mspack_file*>(&dst), 17,
                             0, 0x8000, off_t(outlen), 0);
  if (!s) {
    *err = true;
    return 0;
  }
  const int r = lzxd_decompress(s, off_t(outlen));
  lzxd_free(s);
  g_blks = nullptr;
  *err = r != 0;
  return dst.pos;
}

Bytes joined(const std::vector<std::pair<size_t, size_t>>& ch, const Bytes& d, size_t a, size_t b) {
  Bytes o;
  for (size_t k = a; k < b && k < ch.size(); ++k) app(o, d.data() + ch[k].first, ch[k].second);
  o.resize(o.size() + 64, 0);
  return o;
}
}  // namespace

static bool walk_ends(const Bytes& d, size_t start) {
  const size_t L = d.size();
  size_t p = start;
  while (p + 2 <= L) {
    const unsigned c = be16(&d[p]);
    if (c == 0 && p > start && p + 4 <= L && be16(&d[p + 2]) > 0 && be16(&d[p + 2]) <= 0x9000) {
      p += 2;  // (big files: a zero u16 before some chunks - skipped)
      continue;
    }
    if (c == 0 || c > 0x9000) break;
    p += 2 + c;
  }
  return p == L;
}

static std::vector<std::pair<size_t, size_t>> chunks_from(const Bytes& d, size_t p, size_t max = SIZE_MAX) {
  std::vector<std::pair<size_t, size_t>> ch;
  while (p + 2 <= d.size() && ch.size() < max) {
    const size_t c = be16(&d[p]);
    p += 2;
    if (!c) continue;
    ch.push_back({p, c});
    p += c;
  }
  return ch;
}

Bytes load_pac(const Bytes& d) {
  if (!(d.size() > 16 && be32(d.data()) == 0x0FF512ED)) return d;
  // the chunk walk start: ends exactly at EOF and its first LZX block is a plausible 32 KB frame
  size_t start = 0;
  for (size_t s = 16; s < 8192 && !start; ++s) {
    if (!walk_ends(d, s)) continue;
    auto ch = chunks_from(d, s, 4);
    std::vector<unsigned> blks;
    Bytes out(4 * 32768);
    bool err;
    lzx_dec(joined(ch, d, 0, ch.size()), out.data(), out.size(), &err, &blks);
    if (!blks.empty() && blks[0] > 0 && blks[0] <= 32768) start = s;
  }
  if (!start) fail("compressed pac: no chunk layout found");
  const auto ch = chunks_from(d, start);
  Bytes out;
  std::vector<uint8_t> buf;
  for (size_t i = 0; i < ch.size();) {
    // the LZX block lengths from chunk i on
    std::vector<unsigned> blks;
    {
      const size_t lim = 512 * 32768;
      buf.resize(lim);
      bool err;
      lzx_dec(joined(ch, d, i, i + 512), buf.data(), lim, &err, &blks);
    }
    std::vector<size_t> cums;
    size_t c = 0;
    for (unsigned v : blks) {
      if (v == 0 || v > (1u << 25)) break;
      c += v;
      cums.push_back(c);
    }
    std::sort(cums.begin(), cums.end());
    cums.erase(std::unique(cums.begin(), cums.end()), cums.end());
    bool found = false;
    for (auto it = cums.rbegin(); it != cums.rend(); ++it) {
      const size_t n = *it, j = i + (n + 32767) / 32768;
      if (j > ch.size()) continue;
      buf.assign(n, 0);
      bool err;
      const size_t got = lzx_dec(joined(ch, d, i, j), buf.data(), n, &err, nullptr);
      if (err || got != n) continue;
      if (j < ch.size()) {
        std::vector<unsigned> nb;
        Bytes probe(32768);
        bool e2;
        lzx_dec(joined(ch, d, j, j + 1), probe.data(), probe.size(), &e2, &nb);
        if (nb.empty()) continue;
      }
      out.insert(out.end(), buf.begin(), buf.begin() + long(n));
      i = j;
      found = true;
      break;
    }
    if (!found) fail("compressed pac: no group at chunk " + std::to_string(i));
  }
  return out;
}

// ------------------------------------------------------------------------------------------------ EPK8 / PACH
static std::string trim(std::string s) {
  while (!s.empty() && (s.back() == ' ' || s.back() == '\0')) s.pop_back();
  size_t a = 0;
  while (a < s.size() && (s[a] == ' ' || s[a] == '\0')) ++a;
  return s.substr(a);
}

std::vector<PacEntry> epk8_entries(const Bytes& d) {
  if (!(magic(d, "EPK8") || magic(d, "EPAC"))) fail("not a pac (EPK8/EPAC)");
  std::vector<PacEntry> out;
  size_t p = 0x800;
  while (p + 12 <= 0x4000 && p + 12 <= d.size()) {
    bool any = false;
    for (int k = 0; k < 4; ++k) any |= d[p + k] != 0;
    if (!any) break;
    const std::string group = trim(std::string(reinterpret_cast<const char*>(&d[p]), 4));
    const unsigned count = le16(&d[p + 4]), kind = le16(&d[p + 6]);
    p += 12;
    const int gi = int(p);
    const size_t nlen = (kind & 1) ? 8 : 4;
    const size_t n = count * 4 / (nlen + 8);
    for (size_t k = 0; k < n; ++k) {
      const std::string name = trim(std::string(reinterpret_cast<const char*>(&d[p]), nlen));
      p += nlen;
      const uint32_t off = le32(&d[p]), size = le32(&d[p + 4]);
      p += 8;
      out.push_back({group, name, 0x4000 + size_t(off) * 0x800, size_t(size) * 0x100, int(kind), gi});
    }
  }
  return out;
}

const PacEntry* pac_find(const std::vector<PacEntry>& es, const std::string& g, const std::string& n) {
  for (const auto& e : es)
    if (e.group == g && e.name == n) return &e;
  return nullptr;
}
Bytes pac_get(const Bytes& d, const std::vector<PacEntry>& es, const std::string& g, const std::string& n) {
  const PacEntry* e = pac_find(es, g, n);
  return e ? sub(d, e->off, e->size) : Bytes();
}

static void put_name(Bytes& o, const std::string& n, size_t len) {
  for (size_t i = 0; i < len; ++i) o.push_back(i < n.size() ? uint8_t(n[i]) : uint8_t(' '));
}

Bytes epk8_write(const std::vector<PacGroup>& groups) {
  Bytes data, dir;
  for (const auto& g : groups) {
    const size_t nlen = (g.kind & 1) ? 8 : 4;
    Bytes body;
    for (const auto& [name, blob] : g.ents) {
      pad_to(data, 0x800);
      const size_t off = data.size();
      app(data, blob);
      put_name(body, name, nlen);
      app_le32(body, uint32_t(off / 0x800));
      app_le32(body, uint32_t((blob.size() + 0xff) / 0x100));
    }
    put_name(dir, g.tag, 4);
    app_le16(dir, uint16_t(body.size() / 4));
    app_le16(dir, uint16_t(g.kind));
    app_le32(dir, 0);
    app(dir, body);
  }
  pad_to(data, 0x800);
  Bytes out(0x4000, 0);
  std::memcpy(out.data(), "EPK8", 4);
  put_le32(&out[4], 44);
  put_le32(&out[8], uint32_t(data.size()));
  put_le32(&out[12], 7);
  if (0x800 + dir.size() > 0x4000) fail("pac directory too large");
  std::memcpy(&out[0x800], dir.data(), dir.size());
  app(out, data);
  return out;
}

Bytes epk8_add(const Bytes& src, const std::vector<std::pair<std::pair<std::string, std::string>, Bytes>>& adds) {
  Bytes d = src;
  if (!magic(d, "EPK8")) fail("epk8_add: not EPK8");
  struct G {
    std::string tag;
    int kind;
    std::vector<std::tuple<std::string, uint32_t, uint32_t>> ents;
  };
  std::vector<G> groups;
  size_t p = 0x800;
  while (p + 12 <= 0x4000) {
    bool any = false;
    for (int k = 0; k < 4; ++k) any |= d[p + k] != 0;
    if (!any) break;
    G g;
    g.tag = std::string(reinterpret_cast<const char*>(&d[p]), 4);
    const unsigned cnt = le16(&d[p + 4]);
    g.kind = le16(&d[p + 6]);
    p += 12;
    const size_t nlen = (g.kind & 1) ? 8 : 4;
    for (size_t k = 0; k < cnt * 4 / (nlen + 8); ++k) {
      g.ents.emplace_back(std::string(reinterpret_cast<const char*>(&d[p]), nlen), le32(&d[p + nlen]),
                          le32(&d[p + nlen + 4]));
      p += nlen + 8;
    }
    groups.push_back(std::move(g));
  }
  for (const auto& [key, blob] : adds) {
    G* g = nullptr;
    for (auto& x : groups)
      if (trim(x.tag) == key.first) g = &x;
    if (!g) fail("epk8_add: no group " + key.first);
    const size_t nlen = (g->kind & 1) ? 8 : 4;
    std::string name = key.second;
    name.resize(nlen, ' ');
    pad_to(d, 0x800);
    const uint32_t off = uint32_t((d.size() - 0x4000) / 0x800);
    app(d, blob);
    g->ents.erase(std::remove_if(g->ents.begin(), g->ents.end(),
                                 [&](const auto& e) { return std::get<0>(e) == name; }),
                  g->ents.end());
    g->ents.emplace_back(name, off, uint32_t((blob.size() + 0xff) / 0x100));
    std::sort(g->ents.begin(), g->ents.end());
  }
  pad_to(d, 0x800);
  Bytes dir;
  for (const auto& g : groups) {
    Bytes body;
    for (const auto& [n, o, s] : g.ents) {
      app(body, reinterpret_cast<const uint8_t*>(n.data()), n.size());
      app_le32(body, o);
      app_le32(body, s);
    }
    app(dir, reinterpret_cast<const uint8_t*>(g.tag.data()), 4);
    app_le16(dir, uint16_t(body.size() / 4));
    app_le16(dir, uint16_t(g.kind));
    app_le32(dir, 0);
    app(dir, body);
  }
  if (0x800 + dir.size() > 0x4000) fail("pac directory too large");
  std::fill(d.begin() + 0x800, d.begin() + 0x4000, 0);
  std::memcpy(&d[0x800], dir.data(), dir.size());
  put_le32(&d[8], uint32_t(d.size() - 0x4000));
  return d;
}

std::vector<std::pair<uint32_t, Bytes>> pach_parse(const Bytes& d) {
  if (!magic(d, "PACH")) fail("not a PACH");
  const uint32_t n = le32(&d[4]);
  const size_t ds = 8 + 12 * size_t(n);
  std::vector<std::pair<uint32_t, Bytes>> out;
  for (uint32_t j = 0; j < n; ++j) {
    const uint8_t* e = &d[8 + 12 * j];
    out.emplace_back(le32(e), sub(d, ds + le32(e + 4), le32(e + 8)));
  }
  return out;
}

Bytes pach_build(const std::vector<std::pair<uint32_t, Bytes>>& parts, size_t align) {
  const size_t ds = 8 + 12 * parts.size();
  Bytes hdr = {'P', 'A', 'C', 'H'}, body;
  app_le32(hdr, uint32_t(parts.size()));
  std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> ents;
  for (const auto& [pid, b] : parts) {
    while ((ds + body.size()) % align) body.push_back(0);
    ents.emplace_back(pid, uint32_t(body.size()), uint32_t(b.size()));
    app(body, b);
  }
  for (const auto& [pid, o, s] : ents) {
    app_le32(hdr, pid);
    app_le32(hdr, o);
    app_le32(hdr, s);
  }
  app(hdr, body);
  return hdr;
}

// ------------------------------------------------------------------------------------------------ plist360
static std::vector<std::string> def_lines(const std::string& def) {
  const Bytes b = read_file(def);
  std::vector<std::string> out;
  std::string cur;
  for (size_t i = 0; i < b.size(); ++i) {
    if (b[i] == '\r' && i + 1 < b.size() && b[i + 1] == '\n') {
      out.push_back(cur);
      cur.clear();
      ++i;
    } else
      cur += char(b[i]);
  }
  out.push_back(cur);
  return out;
}
static std::string lower(std::string s) {
  for (auto& c : s) c = char(std::tolower(uint8_t(c)));
  return s;
}
bool plist_lists(const std::string& def, const std::string& pac) {
  for (const auto& l : def_lines(def))
    if (lower(l) == lower(pac)) return true;
  return false;
}
static int def_index(const std::string& def, const std::string& pac) {
  const auto lines = def_lines(def);
  for (size_t i = 0; i < lines.size(); ++i)
    if (lower(lines[i]) == lower(pac)) return int(i) - 1;
  fail("plist: " + pac + " not listed");
}
static size_t find_bytes(const Bytes& a, const Bytes& pat, size_t from) {
  if (pat.empty() || a.size() < pat.size()) return SIZE_MAX;
  for (size_t i = from; i + pat.size() <= a.size(); ++i)
    if (std::memcmp(&a[i], pat.data(), pat.size()) == 0) return i;
  return SIZE_MAX;
}
static Bytes plist_record(int idx, const Bytes& new_pac) {
  const Bytes d = load_pac(new_pac);
  std::map<std::tuple<int, std::string, int>, std::vector<std::tuple<std::string, uint32_t, uint32_t>>> groups;
  for (const auto& e : epk8_entries(d))
    groups[{e.gi, e.group, e.kind}].emplace_back(e.name, uint32_t((e.off - 0x4000) / 0x800), uint32_t(e.size / 0x100));
  Bytes rec = {0xff, 0xff};
  app_be16(rec, uint16_t(idx));
  for (const auto& [k, ents] : groups) {
    const size_t nlen = (std::get<2>(k) & 1) ? 8 : 4;
    put_name(rec, std::get<1>(k), 4);
    app_be16(rec, uint16_t((ents.size() * (nlen + 8) / 4) << 4));
    app_be16(rec, uint16_t(nlen == 8 ? 0x1000 : 0));
    app_be32(rec, 0);
    for (const auto& [n, o, s] : ents) {
      put_name(rec, n, nlen);
      app_be32(rec, o);
      app_be32(rec, s);
    }
  }
  return rec;
}
void plist_patch(const std::string& def, const std::string& arc_in, const std::string& pac, const Bytes& new_pac,
                 const std::string& arc_out) {
  const int idx = def_index(def, pac);
  Bytes a = read_file(arc_in);
  Bytes key = {0xff, 0xff};
  app_be16(key, uint16_t(idx));
  const size_t rec = find_bytes(a, key, 0);
  if (rec == SIZE_MAX) fail("plist: record not found for " + pac);
  const size_t nxt = find_bytes(a, {0xff, 0xff}, rec + 4);
  const size_t end = nxt == SIZE_MAX ? a.size() : nxt;
  std::map<std::pair<std::string, std::string>, std::pair<uint32_t, uint32_t>> now;
  const Bytes d = load_pac(new_pac);
  for (const auto& e : epk8_entries(d))
    now[{e.group, e.name}] = {uint32_t((e.off - 0x4000) / 0x800), uint32_t(e.size / 0x100)};
  size_t p = rec + 4;
  while (p < end) {
    const std::string tag = trim(std::string(reinterpret_cast<const char*>(&a[p]), 4));
    const unsigned cnt = be16(&a[p + 4]), kind = be16(&a[p + 6]);
    p += 12;
    const size_t nlen = (kind & 0x1000) ? 8 : 4;
    for (size_t k = 0; k < (cnt >> 4) * 4 / (nlen + 8); ++k) {
      const std::string name = trim(std::string(reinterpret_cast<const char*>(&a[p]), nlen));
      auto it = now.find({tag, name});
      if (it == now.end()) fail("plist patch: " + tag + "/" + name + " missing from the new pac");
      put_be32(&a[p + nlen], it->second.first);
      put_be32(&a[p + nlen + 4], it->second.second);
      p += nlen + 8;
    }
  }
  write_file(arc_out, a);
}
void plist_add(const std::string& def_in, const std::string& arc_in, const std::string& pac, const Bytes& new_pac,
               const std::string& def_out, const std::string& arc_out) {
  auto lines = def_lines(def_in);
  while (!lines.empty() && lines.back().empty()) lines.pop_back();
  const int idx = int(lines.size()) - 1;
  lines.push_back(pac);
  std::string t;
  for (const auto& l : lines) t += l + "\r\n";
  write_file(def_out, Bytes(t.begin(), t.end()));
  const Bytes rec = plist_record(idx, new_pac);
  Bytes a = read_file(arc_in);
  if (a.size() < 4 || be32(&a[a.size() - 4]) != 0xffffffff) fail("plist: no end marker");
  a.insert(a.end() - 4, rec.begin(), rec.end());
  write_file(arc_out, a);
}
void plist_replace(const std::string& def, const std::string& arc_in, const std::string& pac, const Bytes& new_pac,
                   const std::string& arc_out) {
  const int idx = def_index(def, pac);
  Bytes a = read_file(arc_in);
  Bytes k1 = {0xff, 0xff}, k2 = {0xff, 0xff};
  app_be16(k1, uint16_t(idx));
  app_be16(k2, uint16_t(idx + 1));
  const size_t start = find_bytes(a, k1, 0);
  if (start == SIZE_MAX) fail("plist: record not found for " + pac);
  size_t end = find_bytes(a, k2, start + 4);
  if (end == SIZE_MAX) end = a.size() - 4;
  const Bytes rec = plist_record(idx, new_pac);
  Bytes o(a.begin(), a.begin() + long(start));
  app(o, rec);
  o.insert(o.end(), a.begin() + long(end), a.end());
  write_file(arc_out, o);
}

// ------------------------------------------------------------------------------------------------ ISO (XDVDFS)
Iso::Iso(const std::string& path) {
  f_ = std::fopen(path.c_str(), "rb");
  if (!f_) fail("can't open " + path);
  for (uint64_t base : {uint64_t(0x2080000), uint64_t(0xFD90000), uint64_t(0x18300000), uint64_t(0)}) {
    Bytes h = read(base + 32 * 2048, 28);
    if (h.size() == 28 && std::memcmp(h.data(), "MICROSOFT*XBOX*MEDIA", 20) == 0) {
      base_ = base;
      root_sector_ = le32(&h[20]);
      root_size_ = le32(&h[24]);
      return;
    }
  }
  fail(path + " is not an Xbox 360 game disc image (ISO)");
}
Iso::~Iso() {
  if (f_) std::fclose(f_);
}
Bytes Iso::read(uint64_t off, size_t n) {
  Bytes b(n);
#if defined(_WIN32)
  _fseeki64(f_, int64_t(off), SEEK_SET);
#else
  fseeko(f_, off_t(off), SEEK_SET);
#endif
  b.resize(std::fread(b.data(), 1, n, f_));
  return b;
}
std::vector<Iso::Node> Iso::entries(uint32_t sector, uint32_t size) {
  const Bytes d = read(base_ + uint64_t(sector) * 2048, size);
  std::vector<Node> res;
  std::set<size_t> seen;
  std::vector<size_t> stack = {0};
  while (!stack.empty()) {
    const size_t o = stack.back();
    stack.pop_back();
    if (seen.count(o) || o * 4 + 14 > d.size()) continue;
    seen.insert(o);
    const uint16_t l = le16(&d[o * 4]), r = le16(&d[o * 4 + 2]);
    if (l == 0xffff) continue;
    Node nd;
    nd.sector = le32(&d[o * 4 + 4]);
    nd.size = le32(&d[o * 4 + 8]);
    nd.attr = d[o * 4 + 12];
    const size_t nl = d[o * 4 + 13];
    if (o * 4 + 14 + nl > d.size()) continue;
    nd.name.assign(reinterpret_cast<const char*>(&d[o * 4 + 14]), nl);
    res.push_back(nd);
    if (l) stack.push_back(l);
    if (r) stack.push_back(r);
  }
  return res;
}
bool Iso::find(const std::string& path, uint64_t* off, uint32_t* size) {
  uint32_t sec = root_sector_, sz = root_size_;
  size_t a = 0;
  while (a < path.size() && path[a] == '/') ++a;
  while (a < path.size()) {
    size_t b = path.find('/', a);
    if (b == std::string::npos) b = path.size();
    const std::string part = lower(path.substr(a, b - a));
    bool hit = false;
    for (const auto& e : entries(sec, sz))
      if (lower(e.name) == part) {
        sec = e.sector;
        sz = e.size;
        hit = true;
        break;
      }
    if (!hit) return false;
    a = b + 1;
  }
  *off = base_ + uint64_t(sec) * 2048;
  *size = sz;
  return true;
}
Bytes Iso::file(const std::string& path) {
  uint64_t o;
  uint32_t s;
  if (!find(path, &o, &s)) return {};
  return read(o, s);
}
std::vector<std::string> Iso::list(const std::string& dir) {
  uint64_t o;
  uint32_t s;
  std::vector<std::string> out;
  if (!find(dir, &o, &s)) return out;
  for (const auto& e : entries(uint32_t((o - base_) / 2048), s)) out.push_back(e.name);
  return out;
}
