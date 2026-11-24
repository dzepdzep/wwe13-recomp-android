/**
 * @file        graphics/pipeline/shader/translation_store.h
 * @brief       The translation store (wwe2k14 Session 92, M6: translate before play)
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 *
 * Every output of ShaderTranslator::TranslateAnalyzedShader (the SPIR-V and the shader bindings the translation sets up), kept on
 * the player's machine in <cache>/shaders/shareable/TITLE.vk.xtc and served back to the translator's hook instead of running the
 * translator. Made from the player's own game files by the game's precompile mode, in their own folder: never shipped, never
 * committed (CLAUDE.md rule 1).
 *
 * .xtc file: header {'XSTC', format 1, module id (u64)}; the module id is the XXH3 of the file of the GPU module that holds the
 * translator (rexgpu-xenos.dll / the plugin .so): a store made by another build is dropped. Records (RecordHead, 64 bytes, then
 * extras, then the code): {key, check = XXH3 of everything after it, the key's fields, valid, sizes}. Loaded until the end or the
 * first bad record (a torn tail is cut). Appended with an fflush per record under a mutex.
 *
 * .xvar file (the variant list the precompile translates, written by the game's shader-list import from the list's V lines and
 * by the store itself for every key it misses): header {'XVAR', format 1, Modification::kVersion, record size 40}; records
 * {[0..8) check = XXH3 of bytes [8..40), [8] native, [9] type (1 = PS), [10..12) zero, [12..20) hash, [20..28) modification,
 * [28..32) native_ext, [32..36) scale_x, [36..40) scale_y}, little endian.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <rex/cvar.h>

REXCVAR_DECLARE(bool, gpu_m6_store);
REXCVAR_DECLARE(std::string, gpu_m6_store_path);
REXCVAR_DECLARE(bool, gpu_m6_store_append);
REXCVAR_DECLARE(bool, gpu_m6_store_verify);
REXCVAR_DECLARE(std::string, gpu_m6_xvar_path);

namespace rex::graphics::translation_store {

// What a translation is keyed by (the translator instance's own configuration is `config`).
struct Key {
  uint8_t is_pixel = 0;
  uint32_t ucode_dwords = 0;
  uint64_t hash = 0;
  uint64_t modification = 0;
  uint32_t native_ext = 0;
  uint64_t config = 0;
  bool operator==(const Key& o) const {
    return is_pixel == o.is_pixel && ucode_dwords == o.ucode_dwords && hash == o.hash && modification == o.modification &&
           native_ext == o.native_ext && config == o.config;
  }
};

// The stored output: validity, the translator's shader-level binding lists (opaque to the store) and the translated code.
struct Entry {
  bool valid = false;
  std::vector<uint8_t> extras;
  std::vector<uint8_t> binary;
};

// A translation variant as the census names it (the .xvar record).
struct Variant {
  bool native = false;  // nat (NATIVE's translator instance / ext) or emu
  bool pixel = false;
  uint64_t hash = 0;
  uint64_t modification = 0;
  uint32_t native_ext = 0;
  float scale_x = 0.0f, scale_y = 0.0f;
};

// Opens (or re-opens) the store and the variant list. `default_store` / `default_xvar` are used unless gpu_m6_store_path /
// gpu_m6_xvar_path name others. False (store stays off) when the module cannot be identified. Reads the variant list into memory.
bool Open(const std::filesystem::path& default_store, const std::filesystem::path& default_xvar, uint32_t modification_version);
void Close();
// Cheap: the store is open and in use.
bool IsOpen();

bool Find(const Key& key, Entry& out);
// Keeps the entry (memory always; the file when gpu_m6_store_append). A key already held is left alone.
void Add(const Key& key, const Entry& entry);

// The translator ran for a key the store did not hold (outside the precompile: logged once per key, [m6][miss]).
void NoteMiss(const Key& key, const Variant& variant);
void NoteHit();
// What the last TranslateAnalyzedShader on this thread did with the store: 0 nothing (no store), 1 served (a hit, or a verified
// hit), 2 translated (a miss). ResetOutcome() is called by the hook before it looks.
void ResetOutcome();
int LastOutcome();
// The verify mode (gpu_m6_store_verify): a held key was translated anyway; `differs` says the output was not the stored one.
void NoteVerify(const Key& key, bool differs, const char* what);
// A shader whose analysis was the registration's: neither looked up nor stored.
void NoteGuardSkip();

// The precompile's scope on this thread: misses inside it are the precompile's own translations, not play's misses.
struct PrecompileScope {
  PrecompileScope();
  ~PrecompileScope();
};

// The variants the .xvar held at Open (the precompile's queue); Generation() changes with every Open.
const std::vector<Variant>& LoadedVariants();
uint32_t Generation();

struct Stats {
  uint64_t hits = 0, misses = 0, stored = 0, collisions = 0, guard_skips = 0, verify_hits = 0, verify_diffs = 0;
  uint64_t entries = 0;
};
Stats GetStats();

}  // namespace rex::graphics::translation_store
