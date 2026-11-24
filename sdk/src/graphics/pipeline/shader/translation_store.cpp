/**
 * @file        graphics/pipeline/shader/translation_store.cpp
 * @brief       The translation store (wwe2k14 Session 92, M6); see translation_store.h
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#include "translation_store.h"

#include <rex/graphics/pipeline/shader/spirv_translator.h>  // (Session 95 review fix) the native ext bits

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <tuple>
#include <unordered_map>

#include <rex/filesystem.h>
#include <rex/hash.h>
#include <rex/logging.h>
#include <rex/platform.h>

#if REX_PLATFORM_WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <share.h>
#else
#include <dlfcn.h>
#include <sys/file.h>
#endif

REXCVAR_DEFINE_BOOL(gpu_m6_store, false, "GPU/Shader",
                    "Session 92 (M6, translate before play): serve the shader translator's output from the per-player "
                    "translation store (TITLE.vk.xtc beside the shader storage; made by the game's precompile mode from the "
                    "player's own files) and store what it still translates. TRUE NATIVE turns it on (src\\d3d_owned.h)");
REXCVAR_DEFINE_STRING(gpu_m6_store_path, "", "GPU/Shader",
                      "Session 92 (M6): the translation store file instead of <cache>/shaders/shareable/TITLE.vk.xtc "
                      "(the check's store; read when the shader storage opens)");
REXCVAR_DEFINE_BOOL(gpu_m6_store_append, true, "GPU/Shader",
                    "Session 92 (M6): write the translations the store did not hold (and the variants to the .xvar list); "
                    "false = read only (the check's parallel lanes)");
REXCVAR_DEFINE_BOOL(gpu_m6_store_verify, false, "GPU/Shader",
                    "Session 92 (M6, diagnostic): a key the store holds is translated anyway and compared with the stored "
                    "output (code and bindings); differences are counted and logged [m6][verify]");
REXCVAR_DEFINE_STRING(gpu_m6_xvar_path, "", "GPU/Shader",
                      "Session 92 (M6): the variant list file instead of <cache>/shaders/shareable/TITLE.vk.xvar");

namespace rex::graphics::translation_store {

namespace {

constexpr uint32_t kStoreMagic = 0x43545358;  // 'XSTC'
constexpr uint32_t kStoreFormat = 1;
constexpr uint32_t kVarMagic = 0x52415658;  // 'XVAR'
constexpr uint32_t kVarFormat = 1;
constexpr uint32_t kMaxExtras = 1u << 16;
constexpr uint32_t kMaxBinary = 1u << 26;

struct StoreHeader {
  uint32_t magic, format;
  uint64_t module_id;
};
static_assert(sizeof(StoreHeader) == 16);

struct RecordHead {
  uint64_t key;    // XXH3 of the key fields
  uint64_t check;  // XXH3 of every byte of the record after this field
  uint64_t hash;
  uint64_t modification;
  uint64_t config;
  uint32_t ucode_dwords;
  uint32_t native_ext;
  uint32_t extras_size;
  uint32_t binary_size;
  uint8_t is_pixel;
  uint8_t valid;
  uint16_t pad0;
  uint32_t pad1;
};
static_assert(sizeof(RecordHead) == 64);

struct VarHeader {
  uint32_t magic, format, modification_version, record_size;
};
static_assert(sizeof(VarHeader) == 16);

#pragma pack(push, 1)
struct VarRecord {  // piece C's layout: [0..8) check, [8] native, [9] type, [10..12) zero, [12..20) hash, [20..28) mod, [28..32) ext, [32..40) scale
  uint64_t check;  // XXH3 of the 32 bytes after it
  uint8_t native, pixel;
  uint16_t pad;
  uint64_t hash;
  uint64_t modification;
  uint32_t native_ext;
  float scale_x, scale_y;
};
#pragma pack(pop)
static_assert(sizeof(VarRecord) == 40);

// What the store keeps in memory per translation: the key and where the record is in the file (read on a hit), or the entry
// itself when there is no file to read it from (a read-only run's own translations, or no writable file).
struct Held {
  Key key;
  uint64_t offset = 0;
  uint32_t extras_size = 0, binary_size = 0;
  bool valid = false;
  bool in_memory = false;
  Entry mem;
};

std::shared_mutex g_map_mutex;
std::unordered_map<uint64_t, Held> g_map;
std::mutex g_file_mutex;  // g_file / g_xvar_file writes
FILE* g_file = nullptr;       // the writer (append mode), write-deny
FILE* g_read = nullptr;       // the reader of the same file (a hit reads its record from here)
std::mutex g_read_mutex;
uint64_t g_file_end = 0;      // the file's valid length (under g_file_mutex)
FILE* g_xvar_file = nullptr;
std::atomic<bool> g_open{false};
std::atomic<bool> g_append{false};
std::atomic<uint32_t> g_generation{0};
std::atomic<uint64_t> g_hits{0}, g_misses{0}, g_stored{0}, g_collisions{0}, g_guard_skips{0}, g_verify_hits{0},
    g_verify_diffs{0};
std::mutex g_log_mutex;
std::set<uint64_t> g_miss_logged;  // key hashes logged once
std::set<std::tuple<bool, bool, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t>> g_xvar_seen;
std::vector<Variant> g_loaded_variants;
uint32_t g_var_version = 0;
bool g_xvar_ok = false;  // the .xvar file has a good header of this build's modification version
uint64_t g_module_id = 0;
bool g_module_id_known = false;
thread_local int t_precompile_depth = 0;
thread_local int t_outcome = 0;  // 0 none, 1 the last translation on this thread was a store hit, 2 it was translated

// Opens a file the store writes: no other process may write it (a second game on the same cache). nullptr when it is taken.
FILE* OpenWriter(const std::filesystem::path& path, bool create_new) {
#if REX_PLATFORM_WIN32
  return _wfsopen(path.c_str(), create_new ? L"wb" : L"r+b", _SH_DENYWR);
#else
  FILE* f = rex::filesystem::OpenFile(path, create_new ? "wb" : "r+b");
  if (f && flock(fileno(f), LOCK_EX | LOCK_NB) != 0) {
    std::fclose(f);
    return nullptr;
  }
  return f;
#endif
}
// Opens a file for reading while a writer may hold it.
FILE* OpenReader(const std::filesystem::path& path) {
#if REX_PLATFORM_WIN32
  return _wfsopen(path.c_str(), L"rb", _SH_DENYNO);
#else
  return rex::filesystem::OpenFile(path, "rb");
#endif
}

uint64_t KeyHash(const Key& k) {
  struct {
    uint8_t is_pixel;
    uint8_t pad[3];
    uint32_t ucode_dwords;
    uint64_t hash, modification, config;
    uint32_t native_ext, pad2;
  } packed = {};
  packed.is_pixel = k.is_pixel;
  packed.ucode_dwords = k.ucode_dwords;
  packed.hash = k.hash;
  packed.modification = k.modification;
  packed.config = k.config;
  packed.native_ext = k.native_ext;
  return XXH3_64bits(&packed, sizeof(packed));
}

// The XXH3 of the file of the module this code is in (the GPU module). ~7 MB, hashed once.
bool ComputeModuleId(uint64_t& out) {
  if (g_module_id_known) {
    out = g_module_id;
    return true;
  }
  std::filesystem::path path;
#if REX_PLATFORM_WIN32
  HMODULE module = nullptr;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(&ComputeModuleId), &module)) {
    return false;
  }
  std::wstring buffer(32768, L'\0');
  const DWORD length = GetModuleFileNameW(module, buffer.data(), DWORD(buffer.size()));
  if (!length || length >= buffer.size()) return false;
  buffer.resize(length);
  path = std::filesystem::path(buffer);
#else
  Dl_info info{};
  if (!dladdr(reinterpret_cast<const void*>(&ComputeModuleId), &info) || !info.dli_fname) return false;
  path = std::filesystem::path(info.dli_fname);
#endif
  FILE* f = rex::filesystem::OpenFile(path, "rb");
  if (!f) return false;
  XXH3_state_t* state = XXH3_createState();
  XXH3_64bits_reset(state);
  std::vector<uint8_t> chunk(1u << 20);
  uint64_t total = 0;
  for (;;) {
    const size_t n = std::fread(chunk.data(), 1, chunk.size(), f);
    if (!n) break;
    XXH3_64bits_update(state, chunk.data(), n);
    total += n;
  }
  std::fclose(f);
  const uint64_t id = XXH3_64bits_digest(state);
  XXH3_freeState(state);
  if (!total) return false;
  g_module_id = id;
  g_module_id_known = true;
  out = id;
  REXGPU_INFO("[m6][store] module {} ({} bytes), id {:016X}", rex::path_to_utf8(path), total, id);
  return true;
}

void SerializeRecord(const Key& key, const Entry& entry, std::vector<uint8_t>& out) {
  RecordHead head = {};
  head.key = KeyHash(key);
  head.hash = key.hash;
  head.modification = key.modification;
  head.config = key.config;
  head.ucode_dwords = key.ucode_dwords;
  head.native_ext = key.native_ext;
  head.extras_size = uint32_t(entry.extras.size());
  head.binary_size = uint32_t(entry.binary.size());
  head.is_pixel = key.is_pixel;
  head.valid = entry.valid ? 1 : 0;
  out.resize(sizeof(head) + entry.extras.size() + entry.binary.size());
  std::memcpy(out.data(), &head, sizeof(head));
  if (!entry.extras.empty()) std::memcpy(out.data() + sizeof(head), entry.extras.data(), entry.extras.size());
  if (!entry.binary.empty()) std::memcpy(out.data() + sizeof(head) + entry.extras.size(), entry.binary.data(), entry.binary.size());
  const uint64_t check = XXH3_64bits(out.data() + 16, out.size() - 16);
  std::memcpy(out.data() + 8, &check, 8);
}

void CloseLocked() {
  g_open.store(false, std::memory_order_release);
  g_append.store(false, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(g_file_mutex);
    if (g_file) std::fclose(g_file), g_file = nullptr;
    if (g_xvar_file) std::fclose(g_xvar_file), g_xvar_file = nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(g_read_mutex);
    if (g_read) std::fclose(g_read), g_read = nullptr;
  }
  {
    std::unique_lock<std::shared_mutex> lock(g_map_mutex);
    g_map.clear();
  }
}

void WriteVarRecord(const Variant& v) {
  VarRecord r = {};
  r.hash = v.hash;
  r.modification = v.modification;
  r.native_ext = v.native_ext;
  r.scale_x = v.scale_x;
  r.scale_y = v.scale_y;
  r.native = v.native ? 1 : 0;
  r.pixel = v.pixel ? 1 : 0;
  r.check = XXH3_64bits(reinterpret_cast<const uint8_t*>(&r) + 8, sizeof(r) - 8);
  std::fwrite(&r, sizeof(r), 1, g_xvar_file);
  std::fflush(g_xvar_file);
}

std::tuple<bool, bool, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t> VarTuple(const Variant& v) {
  uint32_t sx, sy;
  std::memcpy(&sx, &v.scale_x, 4);
  std::memcpy(&sy, &v.scale_y, 4);
  return {v.native, v.pixel, v.hash, v.modification, v.native_ext, sx, sy};
}

}  // namespace

PrecompileScope::PrecompileScope() { ++t_precompile_depth; }
PrecompileScope::~PrecompileScope() { --t_precompile_depth; }

bool Open(const std::filesystem::path& default_store, const std::filesystem::path& default_xvar, uint32_t modification_version) {
  Close();
  if (!REXCVAR_GET(gpu_m6_store)) return false;
  uint64_t module_id = 0;
  if (!ComputeModuleId(module_id)) {
    REXGPU_ERROR("[m6][store] the GPU module's file could not be read: the translation store stays off");
    return false;
  }
  bool append = REXCVAR_GET(gpu_m6_store_append);
  const std::filesystem::path store_path =
      REXCVAR_GET(gpu_m6_store_path).empty() ? default_store : std::filesystem::path(REXCVAR_GET(gpu_m6_store_path));
  const std::filesystem::path xvar_path =
      REXCVAR_GET(gpu_m6_xvar_path).empty() ? default_xvar : std::filesystem::path(REXCVAR_GET(gpu_m6_xvar_path));
  std::filesystem::path parent = store_path.parent_path();
  if (!parent.empty() && append) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
  }
  size_t loaded = 0, torn = 0;
  bool header_ok = false, stale_module = false, existed = false;
  uint64_t valid_bytes = sizeof(StoreHeader);
  {
    std::unique_lock<std::shared_mutex> lock(g_map_mutex);
    g_map.clear();
    if (FILE* f = OpenReader(store_path)) {
      existed = true;
      StoreHeader h = {};
      header_ok = std::fread(&h, sizeof(h), 1, f) == 1 && h.magic == kStoreMagic && h.format == kStoreFormat;
      if (header_ok && h.module_id != module_id) {
        header_ok = false;
        stale_module = true;
      }
      std::vector<uint8_t> buffer;
      while (header_ok) {
        RecordHead head;
        if (std::fread(&head, sizeof(head), 1, f) != 1) break;
        if (head.extras_size > kMaxExtras || head.binary_size > kMaxBinary) {
          ++torn;
          break;
        }
        buffer.resize(sizeof(head) + head.extras_size + head.binary_size);
        std::memcpy(buffer.data(), &head, sizeof(head));
        if (head.extras_size + head.binary_size &&
            std::fread(buffer.data() + sizeof(head), 1, head.extras_size + head.binary_size, f) != head.extras_size + head.binary_size) {
          ++torn;
          break;
        }
        if (XXH3_64bits(buffer.data() + 16, buffer.size() - 16) != head.check) {
          ++torn;
          break;
        }
        Held held;
        held.offset = valid_bytes;
        held.extras_size = head.extras_size;
        held.binary_size = head.binary_size;
        held.valid = head.valid != 0;
        held.key.is_pixel = head.is_pixel;
        held.key.ucode_dwords = head.ucode_dwords;
        held.key.hash = head.hash;
        held.key.modification = head.modification;
        held.key.native_ext = head.native_ext;
        held.key.config = head.config;
        if (KeyHash(held.key) != head.key) {
          ++torn;
          break;
        }
        valid_bytes += buffer.size();
        ++loaded;
        g_map.emplace(head.key, std::move(held));
      }
      std::fclose(f);
    }
  }
  {
    std::lock_guard<std::mutex> lock(g_file_mutex);
    if (append) {
      g_file = OpenWriter(store_path, !header_ok);
      if (!g_file) {
        REXGPU_ERROR("[m6][store] {} is held by another process (or could not be opened for writing): append off, read only",
                     rex::path_to_utf8(store_path));
      }
      if (g_file) {
        if (header_ok) {
          rex::filesystem::TruncateStdioFile(g_file, valid_bytes);
          std::fseek(g_file, 0, SEEK_END);
        } else {
          const StoreHeader h = {kStoreMagic, kStoreFormat, module_id};
          std::fwrite(&h, sizeof(h), 1, g_file);
          std::fflush(g_file);
          valid_bytes = sizeof(h);
        }
        g_file_end = valid_bytes;
      }
    }
  }
  // The reader of the same file: a hit reads its record from here.
  if (header_ok || g_file) {
    std::lock_guard<std::mutex> lock(g_read_mutex);
    g_read = OpenReader(store_path);
    if (!g_read) {
      std::unique_lock<std::shared_mutex> map_lock(g_map_mutex);
      g_map.clear();  // (cannot be read back: the entries are dropped)
    }
  }
  // The variant list.
  g_var_version = modification_version;
  g_loaded_variants.clear();
  g_xvar_seen.clear();
  g_xvar_ok = false;
  size_t var_count = 0;
  bool var_stale = false;
  uint64_t var_valid_bytes = sizeof(VarHeader);
  if (FILE* f = OpenReader(xvar_path)) {
    VarHeader h = {};
    const bool header = std::fread(&h, sizeof(h), 1, f) == 1 && h.magic == kVarMagic && h.format == kVarFormat && h.record_size == sizeof(VarRecord);
    if (header && h.modification_version == modification_version) {
      g_xvar_ok = true;
      VarRecord r;
      while (std::fread(&r, sizeof(r), 1, f) == 1) {
        if (XXH3_64bits(reinterpret_cast<const uint8_t*>(&r) + 8, sizeof(r) - 8) != r.check) break;
        Variant v;
        v.native = r.native != 0;
        v.pixel = r.pixel != 0;
        v.hash = r.hash;
        v.modification = r.modification;
        v.native_ext = r.native_ext;
        v.scale_x = r.scale_x;
        v.scale_y = r.scale_y;
        if (g_xvar_seen.insert(VarTuple(v)).second) g_loaded_variants.push_back(v);
        var_valid_bytes += sizeof(r);
        ++var_count;
      }
    } else if (header) {
      var_stale = true;
    }
    std::fclose(f);
  }
  if (append) {
    std::lock_guard<std::mutex> lock(g_file_mutex);
    g_xvar_file = OpenWriter(xvar_path, !g_xvar_ok);
    if (!g_xvar_file) {
      REXGPU_ERROR("[m6][store] {} is held by another process (or could not be opened for writing): its variants are not appended",
                   rex::path_to_utf8(xvar_path));
    }
    if (g_xvar_file) {
      if (g_xvar_ok) {
        rex::filesystem::TruncateStdioFile(g_xvar_file, var_valid_bytes);
        std::fseek(g_xvar_file, 0, SEEK_END);
      } else {
        const VarHeader h = {kVarMagic, kVarFormat, modification_version, uint32_t(sizeof(VarRecord))};
        std::fwrite(&h, sizeof(h), 1, g_xvar_file);
        std::fflush(g_xvar_file);
        g_xvar_ok = true;
      }
    }
  }
  g_hits = g_misses = g_stored = g_collisions = g_guard_skips = g_verify_hits = g_verify_diffs = 0;
  {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_miss_logged.clear();
  }
  append = append && g_file != nullptr;
  g_append.store(append, std::memory_order_relaxed);
  g_generation.fetch_add(1, std::memory_order_relaxed);
  g_open.store(true, std::memory_order_release);
  REXGPU_INFO("[m6][store] {}: {} translations{}{}{}; variants {}: {} ({}); append {}, verify {}", rex::path_to_utf8(store_path), loaded,
              existed ? "" : " (new file)", stale_module ? " (made by another GPU module: dropped)" : "",
              torn ? " (a torn tail cut)" : "", rex::path_to_utf8(xvar_path), g_loaded_variants.size(),
              var_stale ? "another modification version: ignored" : (var_count ? "read" : "none"), append ? "on" : "off (read only)",
              REXCVAR_GET(gpu_m6_store_verify) ? "on" : "off");
  return true;
}

void Close() {
  if (g_open.load(std::memory_order_acquire) || g_file || g_xvar_file) {
    const Stats s = GetStats();
    REXGPU_INFO("[m6][store] closed: hits {}, misses {}, stored {}, guard skips {}, collisions {}, verify hits {} differ {}", s.hits,
                s.misses, s.stored, s.guard_skips, s.collisions, s.verify_hits, s.verify_diffs);
  }
  CloseLocked();
}

bool IsOpen() { return g_open.load(std::memory_order_acquire); }

bool Find(const Key& key, Entry& out) {
  Held held;
  {
    std::shared_lock<std::shared_mutex> lock(g_map_mutex);
    auto it = g_map.find(KeyHash(key));
    if (it == g_map.end() || !(it->second.key == key)) return false;
    if (it->second.in_memory) {
      out = it->second.mem;
      return true;
    }
    held.offset = it->second.offset;
    held.extras_size = it->second.extras_size;
    held.binary_size = it->second.binary_size;
    held.valid = it->second.valid;
  }
  // The record from the file (the SPIR-V of every translation is not held in memory).
  std::vector<uint8_t> buffer(sizeof(RecordHead) + held.extras_size + held.binary_size);
  {
    std::lock_guard<std::mutex> lock(g_read_mutex);
    if (!g_read || !rex::filesystem::Seek(g_read, int64_t(held.offset), SEEK_SET) ||
        std::fread(buffer.data(), 1, buffer.size(), g_read) != buffer.size()) {
      return false;
    }
  }
  RecordHead head;
  std::memcpy(&head, buffer.data(), sizeof(head));
  if (head.extras_size != held.extras_size || head.binary_size != held.binary_size ||
      XXH3_64bits(buffer.data() + 16, buffer.size() - 16) != head.check) {
    return false;  // (never expected: counted as a miss, translated)
  }
  out.valid = held.valid;
  out.extras.assign(buffer.begin() + sizeof(head), buffer.begin() + sizeof(head) + held.extras_size);
  out.binary.assign(buffer.begin() + sizeof(head) + held.extras_size, buffer.end());
  return true;
}

void Add(const Key& key, const Entry& entry) {
  if (!IsOpen()) return;
  if (entry.extras.size() > kMaxExtras || entry.binary.size() > kMaxBinary) return;
  const uint64_t h = KeyHash(key);
  std::lock_guard<std::mutex> file_lock(g_file_mutex);  // (one writer at a time; the file and the map change together)
  {
    std::shared_lock<std::shared_mutex> lock(g_map_mutex);
    auto it = g_map.find(h);
    if (it != g_map.end()) {
      if (!(it->second.key == key)) g_collisions.fetch_add(1, std::memory_order_relaxed);
      return;  // held already (or a collision: the first one stays)
    }
  }
  Held held;
  held.key = key;
  held.valid = entry.valid;
  held.extras_size = uint32_t(entry.extras.size());
  held.binary_size = uint32_t(entry.binary.size());
  if (g_append.load(std::memory_order_relaxed) && g_file) {
    std::vector<uint8_t> bytes;
    SerializeRecord(key, entry, bytes);
    held.offset = g_file_end;
    if (std::fwrite(bytes.data(), 1, bytes.size(), g_file) == bytes.size() && std::fflush(g_file) == 0) {
      g_file_end += bytes.size();
    } else {
      held.in_memory = true;
      held.mem = entry;
      rex::filesystem::TruncateStdioFile(g_file, g_file_end);
      std::fseek(g_file, 0, SEEK_END);
    }
  } else {
    held.in_memory = true;
    held.mem = entry;
  }
  {
    std::unique_lock<std::shared_mutex> lock(g_map_mutex);
    g_map.emplace(h, std::move(held));
  }
  g_stored.fetch_add(1, std::memory_order_relaxed);
}

void ResetOutcome() { t_outcome = 0; }
int LastOutcome() { return t_outcome; }

void NoteMiss(const Key& key, const Variant& variant) {
  t_outcome = 2;
  g_misses.fetch_add(1, std::memory_order_relaxed);
  const bool in_precompile = t_precompile_depth > 0;
  // The variant list: every translated key a later precompile (at another configuration) should make too.
  // (Session 95 review fix) An attribute form the precompile derives from a V line is no line of its own: the list stays as it was (a boot at switch 0 would queue it and refuse it).
  const bool derived_form = in_precompile && !variant.pixel && (variant.native_ext & SpirvShaderTranslator::kNativeExtVsInput) && !(variant.native_ext & SpirvShaderTranslator::kNativeExtVsSwap);
  if (key.ucode_dwords && g_append.load(std::memory_order_relaxed) && !derived_form) {
    std::lock_guard<std::mutex> lock(g_file_mutex);
    if (g_xvar_file && g_xvar_ok && g_xvar_seen.insert(VarTuple(variant)).second) WriteVarRecord(variant);
  }
  if (in_precompile) return;
  std::lock_guard<std::mutex> lock(g_log_mutex);
  if (!g_miss_logged.insert(KeyHash(key)).second) return;
  REXGPU_INFO("[m6][miss] {} {} {:016X} mod {:016X} ext {:08X} scale {:.4f}x{:.4f}", variant.native ? "nat" : "emu",
              variant.pixel ? "PS" : "VS", variant.hash, variant.modification, variant.native_ext, variant.scale_x, variant.scale_y);
}

void NoteHit() {
  t_outcome = 1;
  g_hits.fetch_add(1, std::memory_order_relaxed);
}

void NoteVerify(const Key& key, bool differs, const char* what) {
  t_outcome = 1;
  g_verify_hits.fetch_add(1, std::memory_order_relaxed);
  if (!differs) return;
  const uint64_t n = g_verify_diffs.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n <= 32) {
    REXGPU_ERROR("[m6][verify] {} {:016X} mod {:016X} ext {:08X} config {:016X}: the translation differs from the stored one ({})",
                 key.is_pixel ? "PS" : "VS", key.hash, key.modification, key.native_ext, key.config, what);
  }
}

void NoteGuardSkip() { g_guard_skips.fetch_add(1, std::memory_order_relaxed); }

const std::vector<Variant>& LoadedVariants() { return g_loaded_variants; }
uint32_t Generation() { return g_generation.load(std::memory_order_relaxed); }

Stats GetStats() {
  Stats s;
  s.hits = g_hits.load(std::memory_order_relaxed);
  s.misses = g_misses.load(std::memory_order_relaxed);
  s.stored = g_stored.load(std::memory_order_relaxed);
  s.collisions = g_collisions.load(std::memory_order_relaxed);
  s.guard_skips = g_guard_skips.load(std::memory_order_relaxed);
  s.verify_hits = g_verify_hits.load(std::memory_order_relaxed);
  s.verify_diffs = g_verify_diffs.load(std::memory_order_relaxed);
  std::shared_lock<std::shared_mutex> lock(g_map_mutex);
  s.entries = g_map.size();
  return s;
}

}  // namespace rex::graphics::translation_store
