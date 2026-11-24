// k14port: shared helpers (bytes, files, BPE, zlib, LZX/XCTD, EPK8/PACH, ISO reader).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Bytes = std::vector<uint8_t>;

[[noreturn]] void fail(const std::string& why);
void say(const char* fmt, ...);  // progress / log line (stdout, flushed)

inline uint16_t le16(const uint8_t* p) { return uint16_t(p[0] | p[1] << 8); }
inline uint32_t le32(const uint8_t* p) { return uint32_t(p[0] | p[1] << 8 | p[2] << 16 | uint32_t(p[3]) << 24); }
inline uint16_t be16(const uint8_t* p) { return uint16_t(p[0] << 8 | p[1]); }
inline uint32_t be32(const uint8_t* p) { return uint32_t(uint32_t(p[0]) << 24 | p[1] << 16 | p[2] << 8 | p[3]); }
inline void put_le16(uint8_t* p, uint16_t v) { p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); }
inline void put_le32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i)); }
inline void put_be16(uint8_t* p, uint16_t v) { p[0] = uint8_t(v >> 8); p[1] = uint8_t(v); }
inline void put_be32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = uint8_t(v >> (24 - 8 * i)); }
inline void app_le32(Bytes& b, uint32_t v) { uint8_t t[4]; put_le32(t, v); b.insert(b.end(), t, t + 4); }
inline void app_be32(Bytes& b, uint32_t v) { uint8_t t[4]; put_be32(t, v); b.insert(b.end(), t, t + 4); }
inline void app_le16(Bytes& b, uint16_t v) { b.push_back(uint8_t(v)); b.push_back(uint8_t(v >> 8)); }
inline void app_be16(Bytes& b, uint16_t v) { b.push_back(uint8_t(v >> 8)); b.push_back(uint8_t(v)); }
inline void app(Bytes& b, const Bytes& x) { b.insert(b.end(), x.begin(), x.end()); }
inline void app(Bytes& b, const uint8_t* p, size_t n) { b.insert(b.end(), p, p + n); }
inline void pad_to(Bytes& b, size_t align) { while (b.size() % align) b.push_back(0); }
inline Bytes sub(const Bytes& b, size_t off, size_t n) {
  if (off > b.size()) return {};
  if (n > b.size() - off) n = b.size() - off;
  return Bytes(b.begin() + long(off), b.begin() + long(off + n));
}
inline bool magic(const Bytes& b, const char* m) { return b.size() >= 4 && std::memcmp(b.data(), m, 4) == 0; }

Bytes read_file(const std::string& path);
bool file_exists(const std::string& path);
void write_file(const std::string& path, const Bytes& b);
void make_dirs(const std::string& path);
std::vector<std::string> list_dir(const std::string& path);
bool copy_file(const std::string& from, const std::string& to);

// BPE (Yuke's / Gage): parts "BPE " + u32 0x100 + u32 packed + u32 size + blocks (u16 LE block sizes)
Bytes bpe_expand(const uint8_t* src, size_t n, size_t out_size);
Bytes bpe_store_wrap(const Bytes& data);     // valid stream, no pairs (every byte literal), 2 KB blocks
Bytes bpe_compress_wrap(const Bytes& data);  // real byte-pair compression (the game's own sizes)
Bytes trim_bpe(const Bytes& b);              // a "BPE " part cut to its header + packed size
Bytes unpack_any(const Bytes& b);            // BPE / ZLIB part -> bytes (others as they are)
Bytes zlib_inflate(const uint8_t* p, size_t n, size_t hint);

// XCTD (0x0FF512ED, LZX) -> raw EPK8; other data as it is
Bytes load_pac(const Bytes& raw);

// EPK8 / EPAC directory
struct PacEntry {
  std::string group, name;
  size_t off, size;
  int kind, gi;
};
std::vector<PacEntry> epk8_entries(const Bytes& d);
const PacEntry* pac_find(const std::vector<PacEntry>& es, const std::string& g, const std::string& n);
Bytes pac_get(const Bytes& d, const std::vector<PacEntry>& es, const std::string& g, const std::string& n);
struct PacGroup {
  std::string tag;
  int kind;
  std::vector<std::pair<std::string, Bytes>> ents;
};
Bytes epk8_write(const std::vector<PacGroup>& groups);
// entries added / replaced (data appended, directory rewritten): epkadd.py
Bytes epk8_add(const Bytes& d, const std::vector<std::pair<std::pair<std::string, std::string>, Bytes>>& adds);

// PACH
std::vector<std::pair<uint32_t, Bytes>> pach_parse(const Bytes& d);
Bytes pach_build(const std::vector<std::pair<uint32_t, Bytes>>& parts, size_t align);

// plist360 (the game's cache of pac directories)
void plist_patch(const std::string& def, const std::string& arc_in, const std::string& pac, const Bytes& new_pac,
                 const std::string& arc_out);
void plist_add(const std::string& def_in, const std::string& arc_in, const std::string& pac, const Bytes& new_pac,
               const std::string& def_out, const std::string& arc_out);
void plist_replace(const std::string& def, const std::string& arc_in, const std::string& pac, const Bytes& new_pac,
                   const std::string& arc_out);
bool plist_lists(const std::string& def, const std::string& pac);

// Xbox 360 game disc (XDVDFS) image
class Iso {
 public:
  explicit Iso(const std::string& path);
  ~Iso();
  bool find(const std::string& path, uint64_t* off, uint32_t* size);
  Bytes read(uint64_t off, size_t n);
  Bytes file(const std::string& path);  // empty when missing
  bool has(const std::string& path) {
    uint64_t o;
    uint32_t s;
    return find(path, &o, &s);
  }
  std::vector<std::string> list(const std::string& dir);

 private:
  struct Node {
    std::string name;
    uint32_t sector, size;
    uint8_t attr;
  };
  std::vector<Node> entries(uint32_t sector, uint32_t size);
  FILE* f_ = nullptr;
  uint64_t base_ = 0;
  uint32_t root_sector_ = 0, root_size_ = 0;
};
