// Extracts an Xbox 360 XDVDFS disc image (XGD2/XGD3) to a folder. Resumable: skips files already complete.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
static FILE* g_iso;
static uint64_t g_base;
static uint64_t g_total;
static bool ReadAt(uint64_t off, void* buf, size_t n) {
  if (fseeko(g_iso, off, SEEK_SET)) return false;
  return fread(buf, 1, n, g_iso) == n;
}
static void MkDir(const std::string& p) { mkdir(p.c_str(), 0775); }
static bool Walk(uint32_t sec, uint32_t size, const std::string& path) {
  MkDir(path);
  std::vector<uint8_t> d(size);
  if (!ReadAt(g_base + uint64_t(sec) * 2048, d.data(), size)) return false;
  std::vector<uint32_t> stack{0}, seen;
  while (!stack.empty()) {
    uint32_t o = stack.back(); stack.pop_back();
    if (o * 4 + 14 > size) continue;
    bool dup = false; for (auto s : seen) if (s == o) { dup = true; break; }
    if (dup) continue; seen.push_back(o);
    const uint8_t* e = &d[o * 4];
    uint16_t l, r; uint32_t s, sz; memcpy(&l, e, 2); memcpy(&r, e + 2, 2); memcpy(&s, e + 4, 4); memcpy(&sz, e + 8, 4);
    uint8_t attr = e[12], n = e[13];
    if (l == 0xFFFF) continue;
    std::string name(reinterpret_cast<const char*>(e + 14), n);
    std::string p = path + "/" + name;
    if (attr & 0x10) {
      if (sz) { if (!Walk(s, sz, p)) return false; } else MkDir(p);
    } else {
      struct stat st;
      if (stat(p.c_str(), &st) == 0 && uint64_t(st.st_size) == sz) { /* done */ }
      else {
        std::string part = p + ".part";
        FILE* out = fopen(part.c_str(), "wb");
        if (!out) { fprintf(stderr, "open %s: %s\n", part.c_str(), strerror(errno)); return false; }
        static std::vector<char> buf(16 << 20);
        uint64_t left = sz, off = g_base + uint64_t(s) * 2048;
        while (left) {
          size_t c = left < buf.size() ? size_t(left) : buf.size();
          if (!ReadAt(off, buf.data(), c) || fwrite(buf.data(), 1, c, out) != c) { fclose(out); return false; }
          left -= c; off += c;
        }
        fclose(out);
        rename(part.c_str(), p.c_str());
        g_total += sz;
        printf("%s %u\n", p.c_str(), sz); fflush(stdout);
      }
    }
    if (l) stack.push_back(l);
    if (r) stack.push_back(r);
  }
  return true;
}
int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s image.iso outdir\n", argv[0]); return 2; }
  g_iso = fopen(argv[1], "rb");
  if (!g_iso) { perror("iso"); return 1; }
  const uint64_t bases[] = {0x2080000, 0xFD90000, 0x18300000, 0};
  char magic[20]; bool found = false;
  for (uint64_t b : bases) { if (ReadAt(b + 32 * 2048, magic, 20) && !memcmp(magic, "MICROSOFT*XBOX*MEDIA", 20)) { g_base = b; found = true; break; } }
  if (!found) { fprintf(stderr, "not an Xbox disc image\n"); return 1; }
  uint32_t root[2]; ReadAt(g_base + 32 * 2048 + 20, root, 8);
  if (!Walk(root[0], root[1], argv[2])) { fprintf(stderr, "failed\n"); return 1; }
  FILE* f = fopen((std::string(argv[2]) + "/.extract_done").c_str(), "w"); if (f) fclose(f);
  printf("done %llu bytes\n", (unsigned long long)g_total);
  return 0;
}
