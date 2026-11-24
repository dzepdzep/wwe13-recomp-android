// wwe2k14 patch (Session 72, gpu_native_shader_swap): see shader_swap.h.

#include "shader_swap.h"
#include "sampler_overrides.h"  // Session 87 (sampler filter parity)

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <bit>
#include <unordered_map>
#include <unordered_set>

#include <fmt/format.h>
#include <rex/cvar.h>
#include <rex/graphics/pipeline/shader/spirv.h>
#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include <rex/logging.h>

REXCVAR_DECLARE(bool, gpu_native_skin_once);  // (Session 95: defined in skin_once.cpp)
REXCVAR_DECLARE(bool, gpu_native_skin_with_input);  // (Session 95, r3gpu; skin_once.cpp)

REXCVAR_DEFINE_BOOL(gpu_native_shader_swap, false, "GPU/Native",
                    "Session 72: NATIVE's draws use the title's own fragment shaders in place of the guest pixel "
                    "shaders it registered (CommandProcessor::RegisterNativeShaderSwap; the same passes, textures and "
                    "constants); false = the translated guest shaders. Live: the next draws make their own pipelines")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_epilogue_spec_swap, false, "GPU/Native",
                    "Session 95 (B1): our own pixel replacements get kSpecEpilogueIdle = 1 when the draw's colour epilogue is idle (no alpha test, exponent bias 1.0, no gamma), "
                    "so their XeOutputColor compiles the epilogue out (no discard). Read when a module is built (set at start); default off")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);
REXCVAR_DEFINE_STRING(gpu_native_shader_ablate, "", "GPU/Native",
                      "Session 95 (B2, diagnostic, pictures wrong): a comma list of replacement families drawn flat grey instead (bg, prop, particle, wrestler, crowd, "
                      "shadow, blur, other, all); the GPU ms a family saves in a pass = the most its pixel shaders could save. Read once a frame; live (tools/native/shader_rank.py)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_STRING(gpu_native_shader_swap_skip, "", "GPU/Native",
                      "Session 80 (in-run A/Bs of our replacement shaders): a comma-separated list of guest pixel "
                      "shader hashes (hex, up to 128) whose registered replacement is left out - their draws use the "
                      "translation; read once a frame (empty = none)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// wwe2k14 patch (Session 87, stage 4): the vertex side - our own vertex shaders in place of the guest vertex shaders it
// registered (CommandProcessor::RegisterNativeVertexSwap; the same vertex fetch, constants and interpolators).
REXCVAR_DEFINE_BOOL(gpu_native_vs_swap, false, "GPU/Native",
                    "Session 87 (stage 4): NATIVE's draws use the title's own vertex shaders in place of the guest vertex "
                    "shaders it registered (the same vertex data, constants and interpolators); false = the translated "
                    "guest shaders. Live: the next full draws make their own pipelines")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_STRING(gpu_native_vs_swap_skip, "", "GPU/Native",
                      "Session 87 (in-run A/Bs of our vertex shaders): a comma-separated list of guest vertex shader "
                      "hashes (hex, up to 128) whose registered replacement is left out; read once a frame (empty = none)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(gpu_native_instance_world, false, "GPU/Native",
                    "Session 94 (idea 4; read at start, needs gpu_native_vs_swap and vulkan_repeat_draw_merge): back-to-back draws of one of OUR arena vertex "
                    "shaders that differ only in their vertex constants (the world matrix c76-c78) become one instanced draw: the repeat merge "
                    "(RepeatDrawFast / the typed repeat) is allowed for them, and their instancing variant (src/scene, *_inst) reads each instance's constant "
                    "block one stride apart. Draw order is kept (only consecutive draws). [native] world instancing in the log. Off by default");

REXCVAR_DEFINE_INT32(gpu_native_vertex_input, 0, "GPU/Native",
                     "Session 95 (read at start; out\\s95\\android_structural.md; replaces gpu_native_vertex_mirror): which typed draws read their vertices as vertex attributes, the shared memory and the "
                     "inline ring bound directly as vertex buffers, indices from the index pool. [vinput] in the log. 0 off (identical), 1 our vertex shaders, 2 ours plus the translated ones")
    .range(0, 2)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace rex::graphics::vulkan::shader_swap {

namespace {
struct Entry {
  uint64_t hash = 0;
  std::vector<uint32_t> words;
  // (Session 92, vertex swaps) the replacement's code for the shared memory split in 2 / 4 storage buffers (empty: none).
  std::vector<uint32_t> words_split[2];
  // (Session 94, gpu_native_instance_world) the instancing variant: every float constant read at register + gl_InstanceIndex * stride (the
  // stride a specialization constant, SpecId 0, patched in BuildVertex), for the merged draws of vulkan_repeat_draw_merge (empty: none).
  std::vector<uint32_t> words_inst;
  // (Session 94, gpu_d3d_decouple bit 3) the multiview variant: every float constant read at register + gl_ViewIndex * XE_VSTRIDE (SpecId 1, patched to 256 in BuildVertex; empty: none).
  std::vector<uint32_t> words_mv;
  // (Session 95, gpu_native_vertex_input) the vertex-input variant: the vertex from vertex inputs (empty: none).
  std::vector<uint32_t> words_vi;
  std::vector<uint32_t> words_so;  // (Session 95, r3gpu) the skin-once variant (RegisterVertexSkinOnce)
  uint64_t float_bitmap[4] = {};
  std::string name;
  uint32_t flags = 0;
  // (Session 87, vertex swaps) the vertex stream the replacement reads: the translation's single vertex binding must be
  // in this fetch constant (0-95) with this stride in dwords.
  uint32_t stream_fetch_constant = 95;
  uint32_t stream_stride_words = 8;
  // (Session 90, Adreno link) a pixel replacement's input locations as an interpolator mask, contiguous 0..N-1 (set at Register).
  uint32_t read_mask = 0;
  uint32_t family = 7;  // (Session 95) FamilyOfName(name)
};
uint32_t SpirvLocationMask(const uint32_t* words, size_t count, uint32_t storage);
uint32_t ContiguousMask(uint32_t mask) {
  if (!mask) return 0;
  const uint32_t bound = 32 - uint32_t(std::countl_zero(mask));
  return bound >= 16 ? 0xFFFFu : ((1u << bound) - 1u);
}
// Copy on write: readers take the current table without a lock; replaced tables are kept (a handful a run).
struct Table {
  std::vector<std::shared_ptr<const Entry>> entries;
};
std::atomic<uint64_t> g_translation_generation{1};
std::mutex g_mutex;
std::atomic<const Table*> g_table{nullptr};
std::vector<std::unique_ptr<Table>> g_tables;
std::unordered_set<uint64_t> g_logged;
std::unordered_set<uint64_t> g_logged_not_taken;  // (Session 80)
// (Session 80) gpu_native_shader_swap_skip, parsed at the guest swap; read by any thread.
std::atomic<uint64_t> g_skips[128];  // (Session 80: 16 cut the arena A/Bs short)
std::atomic<uint32_t> g_skip_count{0};
std::atomic<uint32_t> g_ablate_mask{0};  // (Session 95) bit per family
std::atomic<bool> g_have_flat{false};  // the flat module (kAblationHash) is registered
std::atomic<uint32_t> g_idle_patched{0}, g_modules_built{0};  // (B1) modules made / made with kXeEpilogueIdle = 1
constexpr const char* kFamilyNames[] = {"bg", "prop", "particle", "wrestler", "crowd", "shadow", "blur", "other"};
uint32_t FamilyOfName(const std::string& n) {
  const auto has = [&](const char* s) { return n.find(s) != std::string::npos; };
  if (has("wrestler")) return 3;
  if (has("crowd")) return 4;
  if (has("arena particle")) return 2;
  if (has("arena background") || has("arena floor")) return 0;
  if (has("arena")) return 1;
  if (has("shadow")) return 5;
  if (has("blur") || has("downscale") || has("bloom") || has("glare") || has("dof") || has("sprite") || has("movie")) return 6;
  return 7;
}

std::shared_ptr<const Entry> Find(uint64_t hash) {
  const Table* table = g_table.load(std::memory_order_acquire);
  if (!table) return nullptr;
  for (const auto& e : table->entries) {
    if (e->hash == hash) return e;
  }
  return nullptr;
}

// Once per shader and reason.
bool Refuse(uint64_t hash, const char* name, const char* why) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_logged.insert(hash).second) {
    REXGPU_WARN("[native][swap] pixel shader {:016X} ('{}'): not swapped - {}", hash, name, why);
  }
  return false;
}
}  // namespace

uint64_t TranslationGeneration() { return g_translation_generation.load(std::memory_order_relaxed); }
void NoteTranslationChange() { g_translation_generation.fetch_add(1, std::memory_order_relaxed); }

bool Register(uint64_t guest_hash, const uint32_t* words, size_t count, const uint8_t* float_registers,
              size_t float_register_count, const char* name, uint32_t flags) {
  if (!words || count < 5 || words[0] != 0x07230203u) {
    REXGPU_WARN("[native][swap] '{}': not SPIR-V, not registered", name ? name : "");
    return false;
  }
  auto entry = std::make_shared<Entry>();
  entry->hash = guest_hash;
  entry->words.assign(words, words + count);
  for (size_t i = 0; i < float_register_count; ++i) {
    entry->float_bitmap[float_registers[i] >> 6] |= uint64_t(1) << (float_registers[i] & 63);
  }
  entry->name = name ? name : "";
  entry->flags = flags;
  entry->family = FamilyOfName(entry->name);
  if (guest_hash == kAblationHash) g_have_flat.store(true, std::memory_order_relaxed);
  entry->read_mask = ContiguousMask(SpirvLocationMask(entry->words.data(), entry->words.size(), 1));
  std::lock_guard<std::mutex> lock(g_mutex);
  auto table = std::make_unique<Table>();
  if (const Table* old = g_table.load(std::memory_order_relaxed)) {
    for (const auto& e : old->entries) {
      if (e->hash != guest_hash) table->entries.push_back(e);
    }
  }
  table->entries.push_back(entry);
  g_table.store(table.get(), std::memory_order_release);
  g_tables.push_back(std::move(table));
  g_logged.erase(guest_hash);
  g_logged_not_taken.erase(guest_hash);
  NoteTranslationChange();
  REXGPU_INFO("[native][swap] pixel shader {:016X}: '{}' registered ({} words, {} float constants{})", guest_hash,
              entry->name, count, float_register_count, (flags & kFlagOwnEpilogue) ? ", its own epilogue" : "");
  return true;
}

bool Has(uint64_t guest_hash, uint32_t* flags) {
  const Table* table = g_table.load(std::memory_order_acquire);
  if (!table) return false;
  for (const auto& e : table->entries) {
    if (e->hash == guest_hash) {
      if (flags) *flags = e->flags;
      return true;
    }
  }
  return false;
}

void RefreshVertexSkips();
void RefreshSkips() {
  RefreshVertexSkips();  // (Session 87)
  // Copied under the cvar registry lock (as gpu_debug_skip_vertex_shader).
  const std::string text = rex::cvar::Query<std::string>("gpu_native_shader_swap_skip");
  uint32_t n = 0;
  bool changed = false;
  for (size_t pos = 0; pos < text.size() && n < std::size(g_skips);) {
    const size_t end = std::min(text.find(',', pos), text.size());
    const uint64_t h = std::strtoull(text.substr(pos, end - pos).c_str(), nullptr, 16);
    if (h) {
      if (g_skips[n].load(std::memory_order_relaxed) != h) changed = true;
      g_skips[n++].store(h, std::memory_order_relaxed);
    }
    pos = end + 1;
  }
  {
    const std::string ab = rex::cvar::Query<std::string>("gpu_native_shader_ablate");
    uint32_t mask = 0;
    for (size_t pos = 0; pos < ab.size();) {
      const size_t end = std::min(ab.find(',', pos), ab.size());
      const std::string w = ab.substr(pos, end - pos);
      if (w == "all") mask = 0xFFu;
      for (uint32_t f = 0; f < 8; ++f) if (w == kFamilyNames[f]) mask |= 1u << f;
      pos = end + 1;
    }
    if (mask != g_ablate_mask.exchange(mask, std::memory_order_relaxed)) {
      changed = true;
      REXGPU_INFO("[native][ablate] families flat: mask {:#04x} (\"{}\")", mask, ab);
    }
  }
  if (n != g_skip_count.load(std::memory_order_relaxed)) changed = true;
  g_skip_count.store(n, std::memory_order_release);
  if (changed) NoteTranslationChange();
}

bool Skipped(uint64_t guest_hash) {
  const uint32_t n = g_skip_count.load(std::memory_order_acquire);
  for (uint32_t i = 0; i < n; ++i) {
    if (g_skips[i].load(std::memory_order_relaxed) == guest_hash) return true;
  }
  return false;
}

uint32_t AblateExt(uint64_t guest_hash) {
  const uint32_t mask = g_ablate_mask.load(std::memory_order_relaxed);
  if (!mask) return 0;
  if (!g_have_flat.load(std::memory_order_relaxed)) return 0;  // (latched at Register of kAblationHash)
  const Table* table = g_table.load(std::memory_order_acquire);
  if (!table) return 0;
  for (const auto& e : table->entries) {
    if (e->hash == guest_hash) return (mask >> e->family) & 1u ? uint32_t(SpirvShaderTranslator::kNativeExtAblate) : 0u;
  }
  return 0;
}

bool SwapStale(uint64_t guest_hash, uint32_t recorded_pixel_ext) {
  const bool recorded = (recorded_pixel_ext & SpirvShaderTranslator::kNativeExtSwap) != 0;
  uint32_t flags = 0;
  if (!REXCVAR_GET(gpu_native_shader_swap) || !Has(guest_hash, &flags) || Skipped(guest_hash)) return recorded;
  if (recorded) return ((recorded_pixel_ext & SpirvShaderTranslator::kNativeExtAblate) != 0) != (AblateExt(guest_hash) != 0);
  // Not recorded with the replacement although it is on: stale where the draw would take it now (an idle epilogue, or
  // a replacement with its own; a draw with alpha to coverage is then rebuilt each time - none of ours has one).
  return (recorded_pixel_ext & SpirvShaderTranslator::kNativeExtEpilogueIdle) || (flags & kFlagOwnEpilogue);
}

bool Takes(uint64_t guest_hash, bool epilogue_idle, bool alpha_to_coverage) {
  uint32_t flags = 0;
  if (!Has(guest_hash, &flags) || Skipped(guest_hash)) return false;
  if (epilogue_idle || ((flags & kFlagOwnEpilogue) && !alpha_to_coverage)) return true;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_logged_not_taken.insert(guest_hash).second) {
    REXGPU_INFO("[native][swap] pixel shader {:016X}: a draw keeps the translation - {}", guest_hash,
                (flags & kFlagOwnEpilogue) ? "alpha to coverage"
                                           : "its epilogue is not idle (register the replacement with its own)");
  }
  return false;
}

// Session 82 (gpu_native_owned_alpha_learn): before the single OpStore of colour output location 0, a test of the stored
// vector's alpha against 254.5 / 255 and an OpKill when it is at least that - the SPIR-V is cut at the store, a selection
// (header block: the compare and the conditional branch; a kill block; the rest in a new block that goes on with the store).
static bool PatchAlphaProbe(std::vector<uint32_t>& words) {
  if (words.size() < 6) return false;
  uint32_t out_var = 0, float_type = 0, bool_type = 0, location_target = 0;
  std::vector<uint32_t> location0_targets;
  size_t first_function = 0;
  for (size_t at = 5; at < words.size();) {
    const uint32_t wc = words[at] >> 16, op = words[at] & 0xFFFF;
    if (!wc || at + wc > words.size()) return false;
    if (op == 54 && !first_function) first_function = at;  // OpFunction
    if (op == 71 && wc == 4 && words[at + 2] == 30 && words[at + 3] == 0) location0_targets.push_back(words[at + 1]);
    if (op == 22 && wc == 3 && words[at + 2] == 32 && !float_type) float_type = words[at + 1];
    if (op == 20 && wc == 2 && !bool_type) bool_type = words[at + 1];
    if (op == 59 && wc >= 4 && words[at + 3] == 3 /* Output */) {
      for (uint32_t t : location0_targets) {
        if (t == words[at + 2]) out_var = words[at + 2];
      }
    }
    at += wc;
  }
  (void)location_target;
  if (!out_var || !float_type || !first_function) return false;
  // The single store of the output variable.
  size_t store_at = 0;
  uint32_t value_id = 0;
  uint32_t stores = 0, current_label = 0, store_label = 0;
  for (size_t at = first_function; at < words.size();) {
    const uint32_t wc = words[at] >> 16, op = words[at] & 0xFFFF;
    // (The probe splits the store's block: a phi after it names the block it was in; an access chain into the output is a store
    // we do not see. Neither is patched.)
    if (op == 248 && wc == 2) current_label = words[at + 1];  // OpLabel
    if ((op == 65 || op == 66) && wc >= 4 && words[at + 3] == out_var) return false;  // OpAccessChain / InBounds
    if (op == 62 && wc >= 3 && words[at + 1] == out_var) {
      ++stores;
      store_at = at;
      store_label = current_label;
      value_id = words[at + 2];
    }
    at += wc;
  }
  if (stores != 1 || !store_label) return false;
  // (The store's block is split by the probe - its tail moves to a new block; a phi that names the block as a predecessor would
  // name the wrong one. Refuse those only.)
  for (size_t at = first_function; at < words.size();) {
    const uint32_t wc = words[at] >> 16, op = words[at] & 0xFFFF;
    if (op == 245) {  // OpPhi %type %id (%value %parent)...
      for (size_t k = 4; k < wc; k += 2) {
        if (words[at + k] == store_label) return false;
      }
    }
    at += wc;
  }
  uint32_t bound = words[3];
  const uint32_t constant_id = bound++, bool_id = bool_type ? bool_type : bound++, alpha_id = bound++, opaque_id = bound++,
                 one_id = bound++, ge_id = bound++, le_id = bound++,
                 kill_label = bound++, cont_label = bound++;
  words[3] = bound;
  // The function-body replacement of the store.
  std::vector<uint32_t> body;
  body.push_back((5u << 16) | 81u);  // OpCompositeExtract %float %alpha %value 3
  body.push_back(float_type), body.push_back(alpha_id), body.push_back(value_id), body.push_back(3);
  body.push_back((5u << 16) | 190u);  // OpFOrdGreaterThanEqual %bool %ge %alpha %constant (254.5/255)
  body.push_back(bool_id), body.push_back(ge_id), body.push_back(alpha_id), body.push_back(constant_id);
  body.push_back((5u << 16) | 188u);  // OpFOrdLessThanEqual %bool %le %alpha %one (a float target's alpha above 1 is not 1)
  body.push_back(bool_id), body.push_back(le_id), body.push_back(alpha_id), body.push_back(one_id);
  body.push_back((5u << 16) | 167u);  // OpLogicalAnd %bool %opaque %ge %le
  body.push_back(bool_id), body.push_back(opaque_id), body.push_back(ge_id), body.push_back(le_id);
  body.push_back((3u << 16) | 247u);  // OpSelectionMerge %cont None
  body.push_back(cont_label), body.push_back(0);
  body.push_back((4u << 16) | 250u);  // OpBranchConditional %opaque %kill %cont
  body.push_back(opaque_id), body.push_back(kill_label), body.push_back(cont_label);
  body.push_back((2u << 16) | 248u);  // %kill = OpLabel
  body.push_back(kill_label);
  body.push_back((1u << 16) | 252u);  // OpKill
  body.push_back((2u << 16) | 248u);  // %cont = OpLabel
  body.push_back(cont_label);
  // The store itself stays after them.
  words.insert(words.begin() + store_at, body.begin(), body.end());
  // The module-level additions go just before the first function: the bool type when the module has none, the constant.
  std::vector<uint32_t> globals;
  if (!bool_type) {
    globals.push_back((2u << 16) | 20u), globals.push_back(bool_id);
  }
  union {
    float f;
    uint32_t u;
  } c;
  c.f = 254.5f / 255.0f;
  globals.push_back((4u << 16) | 43u);  // OpConstant %float %constant 254.5/255
  globals.push_back(float_type), globals.push_back(constant_id), globals.push_back(c.u);
  c.f = 1.0f;
  globals.push_back((4u << 16) | 43u);  // OpConstant %float %one 1.0
  globals.push_back(float_type), globals.push_back(one_id), globals.push_back(c.u);
  words.insert(words.begin() + first_function, globals.begin(), globals.end());
  return true;
}

bool& PixelSamplersReplacement() {
  static bool value = false;
  return value;
}

bool Build(const SpirvShader& shader, std::vector<uint8_t>& out, uint32_t color_quantize, bool alpha_probe,
           bool replacement_bindings, bool epilogue_idle, bool ablate) {
  const uint64_t hash = shader.ucode_data_hash();
  const std::shared_ptr<const Entry> entry = Find(hash);
  if (!entry) return false;
  const char* name = entry->name.c_str();
  const Shader::ConstantRegisterMap& constants = shader.constant_register_map();
  if (constants.float_dynamic_addressing) return Refuse(hash, name, "the guest shader indexes its constants");
  for (int i = 0; i < 4 && !(entry->flags & kFlagAnyConstants); ++i) {
    if (constants.float_bitmap[i] != entry->float_bitmap[i]) {
      return Refuse(hash, name, "the guest shader uses other float constants than the replacement was written for");
    }
  }
  const auto& textures = shader.TextureBindingsFor(replacement_bindings);
  const auto& samplers = shader.SamplerBindingsFor(replacement_bindings);
  // (Session 95, gpu_d3d_decouple phase 2) with the views on, a pixel replacement flagged kFlagLayerTrue reads its inputs at the view's layer: only its
  // multiview variant (RegisterPixelMultiview) will do; none = refused, never a silent layer 0. (A refused draw keeps whatever the guest shader's translation does here: its own fetch reads the
  // view's layer, spirv_translator_fetch.cpp; under replacement bindings the draw is not taken by the swap at all.)
  const bool layer_true = (entry->flags & kFlagLayerTrue) != 0 && SpirvShaderTranslator::DecoupleViews() != 0;
  if (layer_true && entry->words_mv.empty()) {
    return Refuse(hash, name, "gpu_d3d_decouple is on and the replacement has no layer-true (multiview) variant (it would read layer 0 in both views)");
  }
  std::vector<uint32_t> words = layer_true ? entry->words_mv : entry->words;
  if (ablate) {  // (Session 95, B2) the flat module registered under kAblationHash (no bindings to remap)
    if (const std::shared_ptr<const Entry> flat = Find(kAblationHash)) words = flat->words;
  }
  // The canonical set-3 bindings to the translation's: images first (in the translation's order), then samplers.
  const auto remap = [&](uint32_t value, uint32_t& binding) {
    if (value >= kSampler && value < kSampler + 32) {
      for (size_t j = 0; j < samplers.size(); ++j) {
        if (samplers[j].fetch_constant == value - kSampler) {
          binding = uint32_t(textures.size() + j);
          return true;
        }
      }
      return false;
    }
    const bool is_signed = value >= kImageSigned && value < kImageSigned + 32;
    const bool is_cube = value >= kImageCube && value < kImageCube + 32;  // (Session 80)
    if (!is_signed && !is_cube && !(value >= kImage && value < kImage + 32)) return true;  // (not ours: as it is)
    const uint32_t fc = value - (is_signed ? kImageSigned : is_cube ? kImageCube : kImage);
    const xenos::FetchOpDimension dimension = is_cube ? xenos::FetchOpDimension::kCube : xenos::FetchOpDimension::k2D;
    for (size_t i = 0; i < textures.size(); ++i) {
      if (textures[i].fetch_constant == fc && textures[i].dimension == dimension &&
          bool(textures[i].is_signed) == is_signed) {
        binding = uint32_t(i);
        return true;
      }
    }
    return false;
  };
  uint32_t quantize_id = 0;  // (Session 80) the result id decorated SpecId kSpecColorQuantize, if any
  uint32_t idle_id = 0;  // (Session 95, B1) the result id decorated SpecId kSpecEpilogueIdle, if any
  for (size_t at = 5; at < words.size();) {
    const uint32_t word_count = words[at] >> 16, opcode = words[at] & 0xFFFF;
    if (!word_count || at + word_count > words.size()) return Refuse(hash, name, "the replacement's code is cut");
    // OpDecorate %target Binding N
    if (opcode == 71 && word_count == 4 && words[at + 2] == 33) {
      uint32_t binding = words[at + 3];
      if (!remap(words[at + 3], binding)) {
        return Refuse(hash, name, "it samples a texture or sampler the guest shader does not have");
      }
      words[at + 3] = binding;
    }
    // OpDecorate %target SpecId kSpecColorQuantize
    if (opcode == 71 && word_count == 4 && words[at + 2] == 1 && words[at + 3] == kSpecColorQuantize) {
      quantize_id = words[at + 1];
    }
    if (opcode == 71 && word_count == 4 && words[at + 2] == 1 && words[at + 3] == kSpecEpilogueIdle) idle_id = words[at + 1];
    if (opcode == 50 && word_count == 4 && idle_id && words[at + 2] == idle_id) {
      const bool on = epilogue_idle && REXCVAR_GET(gpu_native_epilogue_spec_swap);
      words[at + 3] = on ? 1u : 0u;
      const uint32_t built = g_modules_built.fetch_add(1) + 1, idle = on ? g_idle_patched.fetch_add(1) + 1 : g_idle_patched.load();
      if (on && (idle == 1 || idle == 10 || idle == 100 || idle == 1000)) {
        REXGPU_INFO("[native][swap] epilogue spec: {} of {} replacement modules made so far have kXeEpilogueIdle = 1 (gpu_native_epilogue_spec_swap)", idle, built);
      }
    }
    // OpSpecConstant %type %id value (decorations come before the constants in a module)
    if (opcode == 50 && word_count == 4 && quantize_id && words[at + 2] == quantize_id) {
      words[at + 3] = color_quantize;
    }
    at += word_count;
  }
  if (alpha_probe && !PatchAlphaProbe(words)) return Refuse(hash, name, "no single colour store to put the alpha probe before");
  out.resize(words.size() * sizeof(uint32_t));
  std::memcpy(out.data(), words.data(), out.size());
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_logged.insert(hash ^ 0x5357415050454421ull).second) {
      REXGPU_INFO("[native][swap] pixel shader {:016X}: '{}' in place of the translation ({} textures, {} samplers)",
                  hash, name, textures.size(), samplers.size());
    }
  }
  return true;
}

// ---- Session 87, stage 4 (gpu_native_vs_swap): our own vertex shaders (see shader_swap.h) --------------------------------

namespace {
std::atomic<const Table*> g_vtable{nullptr};
std::vector<std::unique_ptr<Table>> g_vtables;
std::unordered_set<uint64_t> g_vlogged;
std::atomic<uint64_t> g_vskips[128];
std::atomic<uint32_t> g_vskip_count{0};

std::shared_ptr<const Entry> FindVertex(uint64_t hash) {
  const Table* table = g_vtable.load(std::memory_order_acquire);
  if (!table) return nullptr;
  for (const auto& e : table->entries) {
    if (e->hash == hash) return e;
  }
  return nullptr;
}

bool VertexSkipped(uint64_t hash) {
  const uint32_t n = g_vskip_count.load(std::memory_order_acquire);
  for (uint32_t i = 0; i < n; ++i) {
    if (g_vskips[i].load(std::memory_order_relaxed) == hash) return true;
  }
  return false;
}

// Once per shader and reason.
bool RefuseVertex(uint64_t hash, const char* name, const char* why) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_vlogged.insert(hash).second) {
    REXGPU_WARN("[native][swap] vertex shader {:016X} ('{}'): not swapped - {}", hash, name, why);
  }
  return false;
}
}  // namespace

// Session 95 (the diagnostic run): the family of a registered pixel or vertex shader; 0xFF = none.
uint32_t FamilyOf(uint64_t guest_hash) {
  if (const std::shared_ptr<const Entry> e = Find(guest_hash)) return e->family;
  if (const std::shared_ptr<const Entry> e = FindVertex(guest_hash)) return FamilyOfName(e->name);
  return 0xFFu;
}

bool RegisterVertex(uint64_t guest_hash, const uint32_t* words, size_t count, const uint8_t* float_registers,
                    size_t float_register_count, const char* name, uint32_t flags, uint32_t stream_stride_words,
                    uint32_t stream_fetch_constant) {
  if (!words || count < 5 || words[0] != 0x07230203u) {
    REXGPU_WARN("[native][swap] '{}': not SPIR-V, not registered", name ? name : "");
    return false;
  }
  auto entry = std::make_shared<Entry>();
  entry->hash = guest_hash;
  entry->words.assign(words, words + count);
  for (size_t i = 0; i < float_register_count; ++i) {
    entry->float_bitmap[float_registers[i] >> 6] |= uint64_t(1) << (float_registers[i] & 63);
  }
  entry->name = name ? name : "";
  entry->flags = flags;
  entry->stream_stride_words = stream_stride_words;
  entry->stream_fetch_constant = stream_fetch_constant;
  std::lock_guard<std::mutex> lock(g_mutex);
  auto table = std::make_unique<Table>();
  if (const Table* old = g_vtable.load(std::memory_order_relaxed)) {
    for (const auto& e : old->entries) {
      if (e->hash != guest_hash) table->entries.push_back(e);
    }
  }
  table->entries.push_back(entry);
  g_vtable.store(table.get(), std::memory_order_release);
  g_vtables.push_back(std::move(table));
  g_vlogged.erase(guest_hash);
  g_vlogged.erase(guest_hash ^ 0x5357415050454421ull);
  NoteTranslationChange();
  REXGPU_INFO("[native][swap] vertex shader {:016X}: '{}' registered ({} words, {} float constants)", guest_hash,
              entry->name, count, float_register_count);
  return true;
}

void RefreshVertexSkips() {
  const std::string text = rex::cvar::Query<std::string>("gpu_native_vs_swap_skip");
  uint32_t n = 0;
  bool changed = false;
  for (size_t pos = 0; pos < text.size() && n < std::size(g_vskips);) {
    const size_t end = std::min(text.find(',', pos), text.size());
    const uint64_t h = std::strtoull(text.substr(pos, end - pos).c_str(), nullptr, 16);
    if (h) {
      if (g_vskips[n].load(std::memory_order_relaxed) != h) changed = true;
      g_vskips[n++].store(h, std::memory_order_relaxed);
    }
    pos = end + 1;
  }
  if (n != g_vskip_count.load(std::memory_order_relaxed)) changed = true;
  g_vskip_count.store(n, std::memory_order_release);
  if (changed) NoteTranslationChange();
}

bool VertexTakes(uint64_t guest_hash) {
  if (!REXCVAR_GET(gpu_native_vs_swap)) return false;
  const Table* table = g_vtable.load(std::memory_order_acquire);
  if (!table) return false;
  for (const auto& e : table->entries) {
    if (e->hash == guest_hash) return !VertexSkipped(guest_hash);
  }
  return false;
}

// Session 93 (gpu_d3d_cull_ours): the vertex shaders whose clip position is exactly (position dwords 0-3 with w = 1) x (c76-c78) x (c56-c59): the arena meshes' plain
// materials (src\scenerena_vs_bg*.vert except the bump shader, which builds its position from c255 compares, and arena_vs_prop.vert). Read from our own sources
// (AvClip(AvWorld(position), one)); not skinned (no dynamic constants), one stream of 8 dwords a vertex.
bool VertexBoxCullable(uint64_t guest_hash, uint32_t* stream_fetch_constant, std::string* why) {
  const auto no = [&](const std::string& reason) {
    if (why) *why = reason;
    return false;
  };
  if (!REXCVAR_GET(gpu_native_vs_swap)) return no("not ours (gpu_native_vs_swap off)");
  const std::shared_ptr<const Entry> e = FindVertex(guest_hash);
  if (!e) return no("no entry");
  if (why) *why = "'" + e->name + "'";
  if (VertexSkipped(guest_hash)) return no("not ours (skipped) '" + e->name + "'");
  if (e->flags & kFlagDynamicConstants) return no("dynamic constants '" + e->name + "'");
  if (e->stream_stride_words != 8) return no("stride " + std::to_string(e->stream_stride_words) + " '" + e->name + "'");
  const bool bg = e->name.rfind("yBG_", 0) == 0 && e->name.find("BumpMap") == std::string::npos;
  if (!bg && e->name != "yProp_Default") return no("name '" + e->name + "'");
  for (uint32_t r : {56u, 57u, 58u, 59u, 76u, 77u, 78u}) {
    if (!(e->float_bitmap[r >> 6] & (uint64_t(1) << (r & 63)))) return no("row c" + std::to_string(r) + " missing '" + e->name + "'");
  }
  if (stream_fetch_constant) *stream_fetch_constant = e->stream_fetch_constant;
  if (why) *why = "'" + e->name + "'";
  return true;
}

bool VertexSwapStale(uint64_t guest_hash, uint32_t recorded_vertex_ext) {
  const bool recorded = (recorded_vertex_ext & SpirvShaderTranslator::kNativeExtVsSwap) != 0;
  return recorded != VertexTakes(guest_hash);
}

bool& VertexSamplersReplacement() {
  static bool value = false;
  return value;
}

namespace {
std::atomic<uint32_t> g_shared_memory_bindings{1};
}  // namespace

void SetSharedMemoryBindingCount(uint32_t count) { g_shared_memory_bindings.store(count, std::memory_order_relaxed); }

bool RegisterVertexSplit(uint64_t guest_hash, uint32_t binding_count, const uint32_t* words, size_t count) {
  if ((binding_count != 2 && binding_count != 4) || !words || count < 5 || words[0] != 0x07230203u) return false;
  std::lock_guard<std::mutex> lock(g_mutex);
  const Table* old = g_vtable.load(std::memory_order_relaxed);
  if (!old) return false;
  auto table = std::make_unique<Table>();
  bool found = false;
  for (const auto& e : old->entries) {
    if (e->hash != guest_hash) {
      table->entries.push_back(e);
      continue;
    }
    auto copy = std::make_shared<Entry>(*e);
    copy->words_split[binding_count == 2 ? 0 : 1].assign(words, words + count);
    table->entries.push_back(std::move(copy));
    found = true;
  }
  if (!found) return false;
  g_vtable.store(table.get(), std::memory_order_release);
  g_vtables.push_back(std::move(table));
  NoteTranslationChange();
  return true;
}

// Session 94 (gpu_native_instance_world): the instancing variant of a registered vertex shader (call after RegisterVertex for the same hash).
bool RegisterVertexInstanced(uint64_t guest_hash, const uint32_t* words, size_t count) {
  if (!words || count < 5 || words[0] != 0x07230203u) return false;
  std::lock_guard<std::mutex> lock(g_mutex);
  const Table* old = g_vtable.load(std::memory_order_relaxed);
  if (!old) return false;
  auto table = std::make_unique<Table>();
  bool found = false;
  for (const auto& e : old->entries) {
    if (e->hash != guest_hash) {
      table->entries.push_back(e);
      continue;
    }
    auto copy = std::make_shared<Entry>(*e);
    copy->words_inst.assign(words, words + count);
    table->entries.push_back(std::move(copy));
    found = true;
  }
  if (!found) return false;
  g_vtable.store(table.get(), std::memory_order_release);
  g_vtables.push_back(std::move(table));
  NoteTranslationChange();
  return true;
}

namespace {
std::atomic<bool> g_instance_world_latched{false};
}  // namespace

void LatchInstanceWorld() {
  const bool on = REXCVAR_GET(gpu_native_instance_world);
  g_instance_world_latched.store(on, std::memory_order_relaxed);
  if (on) REXGPU_INFO("[native] world instancing on (needs gpu_native_vs_swap and vulkan_repeat_draw_merge; instancing variants registered for the arena vertex shaders)");
}
bool InstanceWorldLatched() { return g_instance_world_latched.load(std::memory_order_relaxed); }

bool VertexInstanceable(uint64_t guest_hash) {
  if (!InstanceWorldLatched() || !REXCVAR_GET(gpu_native_vs_swap)) return false;
  if (g_shared_memory_bindings.load(std::memory_order_relaxed) != 1) return false;  // (no instancing variant for a split shared memory: a refused translation would leave the draw unclaimed)
  const std::shared_ptr<const Entry> e = FindVertex(guest_hash);
  return e && !e->words_inst.empty() && !(e->flags & kFlagDynamicConstants) && !VertexSkipped(guest_hash);
}

namespace {
struct WorldCounts {
  uint64_t first_frame = 0, absorbed = 0, runs = 0, stat[kWsCount] = {}, miss[8] = {}, refused[32] = {};
  uint32_t max_batch = 0;
} g_world;
void WorldPrint(uint64_t frame) {
  WorldCounts& w = g_world;
  if (!w.first_frame) w.first_frame = frame;
  if (frame - w.first_frame < 600) return;
  static const char* const kGroups[8] = {"adjacency/serial", "render state", "ps floats/bool loop/fetch", "externals", "shaders/draw words", "mesh (index range)", "occlusion/jitter", "no previous base"};
  std::string miss, refused;
  for (uint32_t i = 0; i < 8; ++i) {
    if (w.miss[i]) miss += fmt::format("{}{} {}", miss.empty() ? "" : ", ", kGroups[i], w.miss[i]);
  }
  for (uint32_t i = 0; i < 32; ++i) {
    if (w.refused[i]) refused += fmt::format("{}reason{} {}", refused.empty() ? "" : ", ", i, w.refused[i]);
  }
  REXGPU_INFO("[native] world instancing: {} draws -> {} instanced draws (max batch {}) in frames {}-{}; typed repeat: pairs checked {}, candidates {} (first differing group of the others: {}), "
              "no base {}, tried {}, refused by RepeatDrawFast {} ({}), drawn apart {}",
              w.absorbed, w.runs, w.max_batch, w.first_frame, frame, w.stat[kWsChecked], w.stat[kWsCandidate], miss.empty() ? "none" : miss, w.stat[kWsNoBase], w.stat[kWsTried],
              w.stat[kWsRefused], refused.empty() ? "none" : refused, w.stat[kWsDrawnApart]);
  const uint64_t first = frame;
  w = WorldCounts{};
  w.first_frame = first;
}
}  // namespace

// Called by the command processor each time a merged draw was added to a run (n = the run's instances now).
void NoteWorldInstance(uint64_t frame, uint32_t n, uint64_t guest_hash) {
  if (!VertexInstanceable(guest_hash)) return;
  WorldCounts& w = g_world;
  if (n == 2) {  // (the run's first draw plus this one)
    ++w.runs;
    w.absorbed += 2;
  } else {
    ++w.absorbed;
  }
  w.max_batch = std::max(w.max_batch, n);
  WorldPrint(frame);
}

void NoteWorldStat(uint64_t frame, uint32_t what, uint32_t detail) {
  if (!InstanceWorldLatched() || what >= kWsCount) return;
  WorldCounts& w = g_world;
  ++w.stat[what];
  if (what == kWsMiss && detail < 8) ++w.miss[detail];
  if (what == kWsNoBase) ++w.miss[7];
  if (what == kWsRefused) ++w.refused[detail < 32 ? detail : 31];
  WorldPrint(frame);
}

// Session 95 (gpu_d3d_decouple phase 2): the layer-true variant of a registered PIXEL replacement (flag kFlagLayerTrue; tools/shaders/build_multiview_ps.py): the same
// bindings and constants, its array inputs read at layer min(gl_ViewIndex, layers - 1). Call after Register for the same hash.
bool RegisterPixelMultiview(uint64_t guest_hash, const uint32_t* words, size_t count) {
  if (!words || count < 5 || words[0] != 0x07230203u) return false;
  std::lock_guard<std::mutex> lock(g_mutex);
  const Table* old = g_table.load(std::memory_order_relaxed);
  if (!old) return false;
  auto table = std::make_unique<Table>();
  bool found = false;
  for (const auto& e : old->entries) {
    if (e->hash != guest_hash) {
      table->entries.push_back(e);
      continue;
    }
    auto copy = std::make_shared<Entry>(*e);
    copy->words_mv.assign(words, words + count);
    table->entries.push_back(std::move(copy));
    found = true;
  }
  if (!found) return false;
  g_table.store(table.get(), std::memory_order_release);
  g_tables.push_back(std::move(table));
  NoteTranslationChange();
  return true;
}

// Session 95 (gpu_native_vertex_input): the vertex-input variant of a registered vertex shader (call after RegisterVertex for the same hash).
static bool RegisterVertexSlot(uint64_t guest_hash, const uint32_t* words, size_t count, std::vector<uint32_t> Entry::*slot) {
  if (!words || count < 5 || words[0] != 0x07230203u) return false;
  std::lock_guard<std::mutex> lock(g_mutex);
  const Table* old = g_vtable.load(std::memory_order_relaxed);
  if (!old) return false;
  auto table = std::make_unique<Table>();
  bool found = false;
  for (const auto& e : old->entries) {
    if (e->hash != guest_hash) {
      table->entries.push_back(e);
      continue;
    }
    auto copy = std::make_shared<Entry>(*e);
    ((*copy).*slot).assign(words, words + count);
    table->entries.push_back(std::move(copy));
    found = true;
  }
  if (!found) return false;
  g_vtable.store(table.get(), std::memory_order_release);
  g_vtables.push_back(std::move(table));
  NoteTranslationChange();
  return true;
}

namespace {
std::atomic<int32_t> g_vertex_input_level{0};
}  // namespace

bool RegisterVertexMirror(uint64_t guest_hash, const uint32_t* words, size_t count) { return RegisterVertexSlot(guest_hash, words, count, &Entry::words_vi); }
bool RegisterVertexSkinOnce(uint64_t guest_hash, const uint32_t* words, size_t count) { return RegisterVertexSlot(guest_hash, words, count, &Entry::words_so); }

void LatchVertexMirror() {
  const int32_t level = REXCVAR_GET(gpu_native_vertex_input);
  g_vertex_input_level.store(level, std::memory_order_relaxed);
  if (level) REXGPU_INFO("[vinput] vertex input on, level {} ({}; needs the typed draw and, for our shaders, gpu_native_vs_swap)", level, level == 1 ? "our shaders" : "our shaders and the translated ones");
}
int32_t VertexInputLevel() { return g_vertex_input_level.load(std::memory_order_relaxed); }
bool VertexMirrorLatched() { return VertexInputLevel() != 0; }

bool VertexMirrorable(uint64_t guest_hash, uint32_t* stream_fetch_constant, uint32_t* stride_words) {
  if (!VertexMirrorLatched() || !REXCVAR_GET(gpu_native_vs_swap)) return false;
  const std::shared_ptr<const Entry> e = FindVertex(guest_hash);
  if (!e || e->words_vi.empty() || (e->flags & kFlagAnyConstants) || VertexSkipped(guest_hash)) return false;
  if (stream_fetch_constant) *stream_fetch_constant = e->stream_fetch_constant;
  if (stride_words) *stride_words = e->stream_stride_words;
  return true;
}

// Session 95 (gpu_native_skin_once): the listed shaders, written at registration (before the first draw) and only read after.
namespace {
std::unordered_map<uint64_t, uint32_t> g_skin_rows;
std::vector<uint32_t> g_skin_compute;
std::atomic<bool> g_skin_once_latched{false};
}  // namespace

bool RegisterSkinOnceRow(uint64_t guest_hash, uint32_t role) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_skin_rows[guest_hash] = role;
  return true;
}
bool RegisterSkinOnceCompute(const uint32_t* words, size_t count) {
  if (!words || count < 5 || words[0] != 0x07230203u) return false;
  std::lock_guard<std::mutex> lock(g_mutex);
  g_skin_compute.assign(words, words + count);
  return true;
}
const uint32_t* SkinOnceCompute(size_t* count) {
  *count = g_skin_compute.size();
  return g_skin_compute.empty() ? nullptr : g_skin_compute.data();
}
void LatchSkinOnce() {
  bool on = REXCVAR_GET(gpu_native_skin_once);
  if (on && VertexInputLevel()) {
    if (REXCVAR_GET(gpu_native_skin_with_input)) {  // (Session 95, r3gpu: skin once writes the rows, the listed shaders' _so variants read them as attributes; every draw it refuses is an attribute draw as before)
      REXGPU_INFO("[skin1] skin once with vertex input on: listed draws read the skinned rows, the others their _vi form");
    } else {
      REXGPU_WARN("[skin1] gpu_native_skin_once left off: gpu_native_vertex_input is on (gpu_native_skin_with_input joins them)");
      on = false;
    }
  }
  g_skin_once_latched.store(on, std::memory_order_relaxed);
  if (on) REXGPU_INFO("[skin1] skin once on (needs gpu_native_vs_swap and the typed draw; the game registers its shaders next)");
}
bool SkinOnceLatched() { return g_skin_once_latched.load(std::memory_order_relaxed); }
bool SkinOnceRow(uint64_t guest_hash, uint32_t* role, uint32_t* stride_words) {
  const auto it = g_skin_rows.find(guest_hash);
  if (it == g_skin_rows.end() || !REXCVAR_GET(gpu_native_vs_swap)) return false;
  const std::shared_ptr<const Entry> e = FindVertex(guest_hash);
  if (!e || e->words_so.empty() || VertexSkipped(guest_hash)) return false;
  *role = it->second;
  *stride_words = e->stream_stride_words;
  return true;
}

// Session 94 (gpu_d3d_decouple bit 3): the multiview variant of a registered vertex shader (call after RegisterVertex for the same hash).
bool RegisterVertexMultiview(uint64_t guest_hash, const uint32_t* words, size_t count) {
  if (!words || count < 5 || words[0] != 0x07230203u) return false;
  std::lock_guard<std::mutex> lock(g_mutex);
  const Table* old = g_vtable.load(std::memory_order_relaxed);
  if (!old) return false;
  auto table = std::make_unique<Table>();
  bool found = false;
  for (const auto& e : old->entries) {
    if (e->hash != guest_hash) {
      table->entries.push_back(e);
      continue;
    }
    auto copy = std::make_shared<Entry>(*e);
    copy->words_mv.assign(words, words + count);
    table->entries.push_back(std::move(copy));
    found = true;
  }
  if (!found) return false;
  g_vtable.store(table.get(), std::memory_order_release);
  g_vtables.push_back(std::move(table));
  NoteTranslationChange();
  return true;
}

// Session 94 (gpu_d3d_decouple phase 1B): which registers of a taken vertex replacement are matrices our GLSL names (the blend set; see shader_swap.h).
bool VertexBlendRegisters(uint64_t guest_hash, uint64_t mask[4]) {
  mask[0] = mask[1] = mask[2] = mask[3] = 0;
  if (!REXCVAR_GET(gpu_native_vs_swap)) return false;
  const std::shared_ptr<const Entry> e = FindVertex(guest_hash);
  if (!e || VertexSkipped(guest_hash) || (e->flags & kFlagAnyConstants)) return false;
  const auto set = [&](uint32_t from, uint32_t to) {
    for (uint32_t r = from; r <= to; ++r) mask[r >> 6] |= uint64_t(1) << (r & 63);
  };
  if (e->flags & kFlagDynamicConstants) {
    set(56, 59);   // the view-projection
    set(76, 161);  // the bone palette: c76 + a (3 rows a bone); c162 and up are named lights / literals in the skinned shaders
  } else {
    for (uint32_t r : {56u, 57u, 58u, 59u, 76u, 77u, 78u}) {
      if (e->float_bitmap[r >> 6] & (uint64_t(1) << (r & 63))) mask[r >> 6] |= uint64_t(1) << (r & 63);
    }
  }
  return mask[0] || mask[1] || mask[2] || mask[3];
}


bool BuildVertex(const SpirvShader& shader, uint64_t modification_value, std::vector<uint8_t>& out, bool replacement_bindings,
                 bool dry_run, bool mirror, bool skin) {
  const uint64_t hash = shader.ucode_data_hash();
  const std::shared_ptr<const Entry> entry = FindVertex(hash);
  if (!entry) return false;
  const char* name = entry->name.c_str();
  const SpirvShaderTranslator::Modification modification(modification_value);
  const auto& v = modification.vertex;
  if (v.host_vertex_shader_type != Shader::HostVertexShaderType::kVertex) {
    return RefuseVertex(hash, name, "not a plain vertex draw (tessellation / rectangle / point expansion)");
  }
  // (These refusals are also what makes the replacement's gl_PerVertex block, {gl_Position} only, the translation's own: the
  // translator adds ClipDistance / CullDistance members only for user clip planes or a vertex kill, and the point size is a
  // separate varying - so the SDK's geometry stage after the vertex shader (rectangle lists, point expansion) matches it.)
  if (v.output_point_parameters || v.user_clip_plane_count || v.user_clip_plane_cull || v.vertex_kill_and ||
      v.xfb_position) {
    return RefuseVertex(hash, name,
                        "the translation has user clip planes, a point size, a vertex kill or a transform-feedback capture "
                        "(draws like that keep the game's shader)");
  }
  // Session 94 (gpu_native_instance_world): merged constants (vulkan_repeat_draw_merge) are taken by the instancing variant only.
  const bool instanced = v.instanced_float_constants_stride != 0;
  if (instanced && entry->words_inst.empty()) {
    return RefuseVertex(hash, name, "the translation has merged constants and the replacement has no instancing variant (draws like that keep the game's shader)");
  }
  // The fetch's byte order is fixed in the replacement: 8-in-32 for every vertex fetch constant (the baked endian modes of
  // gpu_native_vertex_endian_spec).
  if (!v.vertex_fetch_endian_spec) {
    return RefuseVertex(hash, name, "the vertex fetch endian modes are not baked (gpu_native_vertex_endian_spec)");
  }
  // (Session 92) The GPU's shared memory bindings: the replacement must read them as the translator would (its code for that
  // count), or the game's translated shader stays.
  const uint32_t shared_bindings = g_shared_memory_bindings.load(std::memory_order_relaxed);
  const std::vector<uint32_t>* code = &entry->words;
  const bool multiview = SpirvShaderTranslator::DecoupleViewConstantBlocks();
  if (mirror && (instanced || multiview)) {
    return RefuseVertex(hash, name, "the vertex mirror (_vi variant) does not combine with world instancing or the multiview decoupling");
  }
  if (mirror && ((skin ? entry->words_so : entry->words_vi).empty() || !(skin ? SkinOnceLatched() : VertexMirrorLatched()))) {
    return RefuseVertex(hash, name, "the vertex mirror asked for and the replacement has no vertex-input variant (or the switch is off)");
  }
  if (multiview && instanced) {
    return RefuseVertex(hash, name, "gpu_d3d_decouple bit 3 and world instancing together (the instance offsets would be lost; the view blocks are the vertex constants' layout)");
  }
  if (mirror) {  // (Session 95: the _vi code reads no shared memory, so the binding count of the shared memory does not matter)
    code = skin ? &entry->words_so : &entry->words_vi;
  } else
  if (multiview) {  // (Session 94: the view blocks - the plain code reads block 0 only, both views would draw the same constants)
    if (entry->words_mv.empty()) return RefuseVertex(hash, name, "gpu_d3d_decouple bit 3 and the replacement has no multiview variant (the game's translated shader stays)");
    if (shared_bindings != 1) return RefuseVertex(hash, name, "gpu_d3d_decouple bit 3 and the shared memory is bound as several storage buffers (no multiview variant for that)");
    code = &entry->words_mv;
  } else
  if (instanced) {
    if (shared_bindings != 1) {
      return RefuseVertex(hash, name, "merged constants and the shared memory is bound as several storage buffers (no instancing variant for that)");
    }
    code = &entry->words_inst;
  } else if (shared_bindings != 1) {
    code = (shared_bindings == 2 || shared_bindings == 4) ? &entry->words_split[shared_bindings == 2 ? 0 : 1] : nullptr;
    if (!code || code->empty()) {
      return RefuseVertex(hash, name, "the shared memory is bound as several storage buffers on this GPU and the replacement "
                                      "has no code for that");
    }
  }
  const size_t bindings = shader.vertex_bindings().size();
  // The replacement reads the one vertex stream it was registered for (its fetch constant and stride; the offsets and
  // formats of the attributes inside are its own business, one hash being one vertex declaration).
  if (bindings != 1 || shader.vertex_bindings()[0].fetch_constant != entry->stream_fetch_constant ||
      shader.vertex_bindings()[0].stride_words != entry->stream_stride_words) {
    return RefuseVertex(hash, name, "its vertex fetch is not the one stream (fetch constant, stride) the replacement reads");
  }
  for (size_t i = 0; i < bindings && i < 7; ++i) {
    if (((v.vertex_fetch_endians >> (2 * i)) & 3u) != uint32_t(xenos::Endian::k8in32)) {
      return RefuseVertex(hash, name, "a vertex fetch constant is not 8-in-32");
    }
  }
  const Shader::ConstantRegisterMap& constants = shader.constant_register_map();
  // (kFlagDynamicConstants: the replacement indexes the whole 256-register float block like a skinned guest shader - the
  // translation then holds all 256 registers by number, not packed - and requires exactly such a translation.)
  if (bool(constants.float_dynamic_addressing) != bool(entry->flags & kFlagDynamicConstants)) {
    return RefuseVertex(hash, name, (entry->flags & kFlagDynamicConstants)
                                        ? "the replacement indexes its constants and the guest shader does not"
                                        : "the guest shader indexes its constants");
  }
  for (int i = 0; i < 4 && !(entry->flags & (kFlagAnyConstants | kFlagDynamicConstants)); ++i) {
    if (constants.float_bitmap[i] != entry->float_bitmap[i]) {
      return RefuseVertex(hash, name,
                          "the guest shader uses other float constants than the replacement was written for");
    }
  }
  if (dry_run) return true;  // (VertexFits: every refusal above)
  const auto& textures = shader.TextureBindingsFor(replacement_bindings);
  const auto& samplers = shader.SamplerBindingsFor(replacement_bindings);
  std::vector<uint32_t> words = *code;
  if (multiview) {
    // Session 94: the view stride (vec4s between two views' constant blocks) is the specialization constant with SpecId 1 (int): its default is patched to 256.
    uint32_t spec_target = 0;
    for (size_t at = 5; at < words.size();) {
      const uint32_t word_count = words[at] >> 16, opcode = words[at] & 0xFFFF;
      if (!word_count || at + word_count > words.size()) return RefuseVertex(hash, name, "the replacement's code is cut");
      if (opcode == 71 && word_count == 4 && words[at + 2] == 1 && words[at + 3] == 1) spec_target = words[at + 1];  // OpDecorate SpecId 1
      at += word_count;
    }
    bool patched = false;
    for (size_t at = 5; spec_target && at < words.size();) {
      const uint32_t word_count = words[at] >> 16, opcode = words[at] & 0xFFFF;
      if (!word_count) break;
      if (opcode == 50 && word_count == 4 && words[at + 2] == spec_target) {  // OpSpecConstant %type %id value
        words[at + 3] = 256;
        patched = true;
        break;
      }
      at += word_count;
    }
    if (!patched) return RefuseVertex(hash, name, "the multiview variant has no view stride specialization constant");
  }
  if (instanced) {
    // The stride (vec4s between two instances' constant blocks) is the specialization constant with SpecId 0: its default is patched.
    uint32_t spec_target = 0;
    for (size_t at = 5; at < words.size();) {
      const uint32_t word_count = words[at] >> 16, opcode = words[at] & 0xFFFF;
      if (!word_count || at + word_count > words.size()) return RefuseVertex(hash, name, "the replacement's code is cut");
      if (opcode == 71 && word_count == 4 && words[at + 2] == 1 && words[at + 3] == 0) spec_target = words[at + 1];  // OpDecorate SpecId 0
      at += word_count;
    }
    bool patched = false;
    for (size_t at = 5; spec_target && at < words.size();) {
      const uint32_t word_count = words[at] >> 16, opcode = words[at] & 0xFFFF;
      if (!word_count) break;
      if (opcode == 50 && word_count == 4 && words[at + 2] == spec_target) {  // OpSpecConstant %type %id value
        words[at + 3] = v.instanced_float_constants_stride;
        patched = true;
        break;
      }
      at += word_count;
    }
    if (!patched) return RefuseVertex(hash, name, "the instancing variant has no stride specialization constant");
  }
  const auto remap = [&](uint32_t value, uint32_t& binding) {
    if (value >= kSampler && value < kSampler + 32) {
      for (size_t j = 0; j < samplers.size(); ++j) {
        if (samplers[j].fetch_constant == value - kSampler) {
          binding = uint32_t(textures.size() + j);
          return true;
        }
      }
      return false;
    }
    const bool is_signed = value >= kImageSigned && value < kImageSigned + 32;
    const bool is_cube = value >= kImageCube && value < kImageCube + 32;  // (Session 87: as the pixel remap)
    if (!is_signed && !is_cube && !(value >= kImage && value < kImage + 32)) return true;  // (not ours: as it is)
    const uint32_t fc = value - (is_signed ? kImageSigned : is_cube ? kImageCube : kImage);
    const xenos::FetchOpDimension dimension = is_cube ? xenos::FetchOpDimension::kCube : xenos::FetchOpDimension::k2D;
    for (size_t i = 0; i < textures.size(); ++i) {
      if (textures[i].fetch_constant == fc && textures[i].dimension == dimension &&
          bool(textures[i].is_signed) == is_signed) {
        binding = uint32_t(i);
        return true;
      }
    }
    return false;
  };
  for (size_t at = 5; at < words.size();) {
    const uint32_t word_count = words[at] >> 16, opcode = words[at] & 0xFFFF;
    if (!word_count || at + word_count > words.size()) return RefuseVertex(hash, name, "the replacement's code is cut");
    // OpDecorate %target Binding N
    if (opcode == 71 && word_count == 4 && words[at + 2] == 33) {
      uint32_t binding = words[at + 3];
      if (!remap(words[at + 3], binding)) {
        return RefuseVertex(hash, name, "it samples a texture or sampler the guest shader does not have");
      }
      words[at + 3] = binding;
    }
    at += word_count;
  }
  out.resize(words.size() * sizeof(uint32_t));
  std::memcpy(out.data(), words.data(), out.size());
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_vlogged.insert(hash ^ 0x5357415050454421ull).second) {
      REXGPU_INFO("[native][swap] vertex shader {:016X}: '{}' in place of the translation ({} textures, {} samplers)",
                  hash, name, textures.size(), samplers.size());
    }
  }
  return true;
}


// ---- Session 87, stage 4 (gpu_native_swap_skip_translation): the analysis from the registration -------------------------
namespace {
// Reads the replacement's SPIR-V and the registration into `out` (see DescribePixel).
bool DescribeEntry(const Entry& entry, bool vertex, Shader::RegisteredAnalysis& out) {
  if (entry.flags & kFlagAnyConstants) return false;  // (the guest shader's real float registers are not known)
  const std::vector<uint32_t>& words = entry.words;
  if (words.size() < 6) return false;
  struct Var {
    uint32_t id = 0, storage = 0xFFFFFFFFu, binding = 0xFFFFFFFFu, set = 0xFFFFFFFFu, location = 0xFFFFFFFFu,
             builtin = 0xFFFFFFFFu, type = 0;  // type: the pointer type
  };
  std::vector<Var> vars;
  const auto var_of = [&](uint32_t id) -> Var& {
    for (Var& v : vars) {
      if (v.id == id) return v;
    }
    vars.emplace_back().id = id;
    return vars.back();
  };
  // Types, for the number of locations a variable takes (an input array or struct takes one per element / member).
  struct Type {
    uint32_t id = 0, opcode = 0, a = 0, b = 0;  // pointer: a = pointee; array: a = element, b = length id; matrix: a = column type, b = count
    std::vector<uint32_t> members;               // struct
  };
  std::vector<Type> types;
  // Access chains and loads (the bool / loop block's bitmaps are read from the chains that start at it, through copies and
  // chains of chains) and the constants they index with; `used` = the ids something reads through or passes on.
  struct Chain {
    uint32_t result, base;
    bool ptr_form;                  // OpPtrAccessChain: the first index is an element of the pointer, not a member
    std::vector<uint32_t> indices;
  };
  std::vector<Chain> chains;
  std::vector<uint32_t> loaded;                          // ids a load reads through directly
  std::vector<uint32_t> used;                            // ids loaded, passed to a call, or the base of a chain
  std::vector<std::pair<uint32_t, uint32_t>> aliases;    // OpCopyObject: (result, source)
  std::vector<std::pair<uint32_t, uint32_t>> constants;  // (result id, value) of OpConstant
  for (size_t at = 5; at < words.size();) {
    const uint32_t word_count = words[at] >> 16, opcode = words[at] & 0xFFFF;
    if (!word_count || at + word_count > words.size()) return false;
    if (opcode == 71 && word_count >= 4) {  // OpDecorate %target decoration literal
      Var& v = var_of(words[at + 1]);
      switch (words[at + 2]) {
        case 33: v.binding = words[at + 3]; break;   // Binding
        case 34: v.set = words[at + 3]; break;       // DescriptorSet
        case 30: v.location = words[at + 3]; break;  // Location
        case 11: v.builtin = words[at + 3]; break;   // BuiltIn
        default: break;
      }
    } else if (opcode == 59 && word_count >= 4) {  // OpVariable %type %id storage
      Var& v = var_of(words[at + 2]);
      v.storage = words[at + 3];
      v.type = words[at + 1];
    } else if (opcode == 32 && word_count >= 4) {  // OpTypePointer %id storage %type
      Type t;
      t.id = words[at + 1], t.opcode = opcode, t.a = words[at + 3];
      types.push_back(std::move(t));
    } else if (opcode == 28 && word_count >= 4) {  // OpTypeArray %id %elem %length
      Type t;
      t.id = words[at + 1], t.opcode = opcode, t.a = words[at + 2], t.b = words[at + 3];
      types.push_back(std::move(t));
    } else if (opcode == 24 && word_count >= 4) {  // OpTypeMatrix %id %column %count
      Type t;
      t.id = words[at + 1], t.opcode = opcode, t.a = words[at + 2], t.b = words[at + 3];
      types.push_back(std::move(t));
    } else if (opcode == 30 && word_count >= 2) {  // OpTypeStruct %id %member...
      Type t;
      t.id = words[at + 1], t.opcode = opcode;
      for (uint32_t k = 2; k < word_count; ++k) t.members.push_back(words[at + k]);
      types.push_back(std::move(t));
    } else if ((opcode == 65 || opcode == 66 || opcode == 67 || opcode == 68) && word_count >= 4) {
      // OpAccessChain / OpInBoundsAccessChain %type %result %base %index...; OpPtrAccessChain / OpInBoundsPtrAccessChain
      // %type %result %base %element %index...
      Chain chain;
      chain.result = words[at + 2];
      chain.base = words[at + 3];
      chain.ptr_form = opcode == 67 || opcode == 68;
      for (uint32_t k = 4; k < word_count; ++k) chain.indices.push_back(words[at + k]);
      used.push_back(chain.base);
      chains.push_back(std::move(chain));
    } else if (opcode == 61 && word_count >= 4) {  // OpLoad %type %result %pointer
      loaded.push_back(words[at + 3]);
      used.push_back(words[at + 3]);
    } else if (opcode == 57 && word_count >= 4) {  // OpFunctionCall %type %result %function %argument...
      for (uint32_t k = 4; k < word_count; ++k) used.push_back(words[at + k]);
    } else if (opcode == 83 && word_count >= 4) {  // OpCopyObject %type %result %source
      aliases.emplace_back(words[at + 2], words[at + 3]);
      used.push_back(words[at + 3]);
    } else if (opcode == 43 && word_count >= 4) {  // OpConstant %type %result value
      constants.emplace_back(words[at + 2], words[at + 3]);
    }
    at += word_count;
  }
  const auto constant_of = [&](uint32_t id, uint32_t& value) {
    for (const auto& c : constants) {
      if (c.first == id) {
        value = c.second;
        return true;
      }
    }
    return false;
  };
  const auto type_of = [&](uint32_t id) -> const Type* {
    for (const Type& t : types) {
      if (t.id == id) return &t;
    }
    return nullptr;
  };
  // Locations a type takes (scalars and vectors one, a matrix its columns, an array its length times its element, a struct the sum).
  const std::function<uint32_t(uint32_t, int)> locations_of = [&](uint32_t id, int depth) -> uint32_t {
    const Type* t = type_of(id);
    if (!t || depth > 8) return 1;
    switch (t->opcode) {
      case 32: return locations_of(t->a, depth + 1);
      case 28: {
        uint32_t length = 1;
        if (!constant_of(t->b, length)) length = 1;
        return std::max<uint32_t>(1, length) * locations_of(t->a, depth + 1);
      }
      case 24: return std::max<uint32_t>(1, t->b) * locations_of(t->a, depth + 1);
      case 30: {
        uint32_t total = 0;
        for (const uint32_t member : t->members) total += locations_of(member, depth + 1);
        return std::max<uint32_t>(1, total);
      }
      default: return 1;
    }
  };
  const auto resolve = [&](uint32_t id) {
    for (int guard = 0; guard < 8; ++guard) {
      bool moved = false;
      for (const auto& a : aliases) {
        if (a.first == id) {
          id = a.second;
          moved = true;
          break;
        }
      }
      if (!moved) break;
    }
    return id;
  };
  const auto is_used = [&](uint32_t id) {
    for (const uint32_t u : used) {
      if (resolve(u) == id) return true;
    }
    return false;
  };
  out = Shader::RegisteredAnalysis();
  const bool dynamic = (entry.flags & kFlagDynamicConstants) != 0;
  out.float_dynamic_addressing = dynamic;
  for (int i = 0; i < 4; ++i) out.float_bitmap[i] = entry.float_bitmap[i];
  if (vertex) {
    out.stream_fetch_constant = entry.stream_fetch_constant;
    out.stream_stride_words = entry.stream_stride_words;
  }
  out.kills_pixels = !vertex && (entry.flags & kFlagKills) != 0;  // (the guest shader's kill: registered, see kFlagKills)
  // The descriptor set of the textures and samplers (shader_swap.h: set 3 for a pixel shader, set 2 for a vertex shader).
  const uint32_t texture_set = vertex ? 2u : 3u;
  struct Tex {
    uint32_t fc, kind;
  };
  std::vector<Tex> textures;
  std::vector<uint32_t> samplers;
  for (const Var& v : vars) {
    if (v.set == 1 && v.binding == 3) {
      // The bool / loop constants block (member 0: uvec4[2] = 8 words of bools; member 1: uvec4[8] = 32 loop constants). Which
      // words the replacement reads comes from the constant indices of the access chains that start at it (through copies and
      // chains of chains); an index it computes (a helper indexing by a variable) reads the whole member. Word level: a word
      // read counts all its bits (a superset).
      const auto all_bools = [&] { for (int w = 0; w < 8; ++w) out.bool_bitmap[w] = 0xFFFFFFFFu; };
      if (std::find(loaded.begin(), loaded.end(), v.id) != loaded.end()) {  // the whole block loaded
        all_bools();
        out.loop_bitmap = 0xFFFFFFFFu;
      }
      for (const Chain& chain : chains) {
        // The full index list from the block: this chain's indices after those of the chains it continues.
        std::vector<uint32_t> idx = chain.indices;
        uint32_t base = resolve(chain.base);
        bool ptr_form = chain.ptr_form;
        for (int guard = 0; guard < 8 && base != v.id; ++guard) {
          const Chain* parent = nullptr;
          for (const Chain& c : chains) {
            if (c.result == base) parent = &c;
          }
          if (!parent) break;
          idx.insert(idx.begin(), parent->indices.begin(), parent->indices.end());
          ptr_form = ptr_form || parent->ptr_form;
          base = resolve(parent->base);
        }
        if (base != v.id) continue;
        uint32_t member = 0;
        if (ptr_form || idx.empty() || !constant_of(idx[0], member)) {
          all_bools();
          out.loop_bitmap = 0xFFFFFFFFu;
          continue;
        }
        if (member == 1) {
          out.loop_bitmap = 0xFFFFFFFFu;  // (loop constants: every one counts)
          continue;
        }
        uint32_t element = 0, component = 0;
        if (idx.size() < 2 || !constant_of(idx[1], element) || element > 1) {
          all_bools();
        } else if (idx.size() < 3 || !constant_of(idx[2], component) || component > 3) {
          for (uint32_t c = 0; c < 4; ++c) out.bool_bitmap[element * 4 + c] = 0xFFFFFFFFu;
        } else {
          out.bool_bitmap[element * 4 + component] = 0xFFFFFFFFu;
        }
      }
    }
    // Textures and samplers: the right descriptor set, the canonical binding, and actually used (a declared image nobody reads
    // is not a binding the replacement needs).
    if (v.storage == 0 && v.binding != 0xFFFFFFFFu && v.set == texture_set && is_used(v.id)) {  // UniformConstant
      if (v.binding >= kImage && v.binding < kImage + 32) textures.push_back({v.binding - kImage, 0});
      else if (v.binding >= kImageSigned && v.binding < kImageSigned + 32) textures.push_back({v.binding - kImageSigned, 1});
      else if (v.binding >= kImageCube && v.binding < kImageCube + 32) textures.push_back({v.binding - kImageCube, 2});
      else if (v.binding >= kSampler && v.binding < kSampler + 32) samplers.push_back(v.binding - kSampler);
    } else if (v.storage == 3) {  // Output
      const uint32_t count = locations_of(v.type, 0);
      if (v.builtin == 22) {  // FragDepth
        out.writes_depth = !vertex;
      } else if (v.builtin == 0xFFFFFFFFu && v.location < 16) {
        for (uint32_t k = 0; k < count && v.location + k < 16; ++k) {
          if (vertex) out.writes_interpolators |= 1u << (v.location + k);
          else if (v.location + k < 4) out.writes_color_targets |= 1u << (v.location + k);
        }
      }
    } else if (v.storage == 1) {  // Input
      if (!vertex && v.builtin == 0xFFFFFFFFu && v.location < 16) {
        out.register_static_address_bound = std::max(out.register_static_address_bound, v.location + locations_of(v.type, 0));
      }
    }
  }
  std::sort(textures.begin(), textures.end(), [](const Tex& a, const Tex& b) { return a.fc != b.fc ? a.fc < b.fc : a.kind < b.kind; });
  textures.erase(std::unique(textures.begin(), textures.end(), [](const Tex& a, const Tex& b) { return a.fc == b.fc && a.kind == b.kind; }),
                 textures.end());
  std::sort(samplers.begin(), samplers.end());
  samplers.erase(std::unique(samplers.begin(), samplers.end()), samplers.end());
  for (const uint32_t fetch_constant : samplers) {
    Shader::RegisteredAnalysis::Sampler sampler;
    sampler.fetch_constant = fetch_constant;
    // The guest shader's own filter overrides (its tfetch instructions), recorded in sampler_overrides.h.
    for (const SamplerOverrideRow& row : kSamplerOverrideRows) {
      if (row.hash == entry.hash && bool(row.vertex) == vertex && row.fetch_constant == fetch_constant) {
        sampler.mag_filter = row.mag_filter;
        sampler.min_filter = row.min_filter;
        sampler.mip_filter = row.mip_filter;
        sampler.aniso_filter = row.aniso_filter;
        break;
      }
    }
    out.samplers.push_back(sampler);
  }
  for (const Tex& t : textures) {
    Shader::RegisteredAnalysis::Texture texture;
    texture.fetch_constant = t.fc;
    texture.dimension = t.kind == 2 ? xenos::FetchOpDimension::kCube : xenos::FetchOpDimension::k2D;
    texture.is_signed = t.kind == 1;
    out.textures.push_back(texture);
  }
  return true;
}
}  // namespace

bool DescribePixel(uint64_t guest_hash, Shader::RegisteredAnalysis& out) {
  const std::shared_ptr<const Entry> entry = Find(guest_hash);
  // Only a replacement that does its own epilogue takes every draw but those with alpha to coverage: the draw-time decision
  // (is the replacement's translation the one this draw uses) is then a function of the registers alone.
  return entry && (entry->flags & kFlagOwnEpilogue) && DescribeEntry(*entry, false, out);
}

bool DescribeVertex(uint64_t guest_hash, Shader::RegisteredAnalysis& out) {
  const std::shared_ptr<const Entry> entry = FindVertex(guest_hash);
  return entry && DescribeEntry(*entry, true, out);
}

// ---- Session 90 (Adreno link): the interface between a replacement pixel shader and the vertex shader it is paired with ----
namespace {
// The user-varying locations (no BuiltIn) of the SPIR-V's Input (storage 1) or Output (storage 3) variables, as a bit mask
// (locations 0..31). Arrays count their length, everything else one location (the interpolators are vec4).
uint32_t SpirvLocationMask(const uint32_t* w, size_t n, uint32_t storage) {
  if (n < 6 || w[0] != 0x07230203u) return 0;
  struct Ty { uint32_t op = 0, a = 0, b = 0; };
  std::unordered_map<uint32_t, Ty> types;
  std::unordered_map<uint32_t, uint32_t> consts, locations, builtins;
  struct V { uint32_t id, type, storage; };
  std::vector<V> vars;
  for (size_t i = 5; i < n;) {
    const uint32_t op = w[i] & 0xFFFF, len = w[i] >> 16;
    if (len == 0 || i + len > n) break;
    switch (op) {
      case 71:  // OpDecorate
        if (len >= 4 && w[i + 2] == 30) locations[w[i + 1]] = w[i + 3];
        else if (len >= 4 && w[i + 2] == 11) builtins[w[i + 1]] = w[i + 3];
        break;
      case 32: if (len >= 4) types[w[i + 1]] = {op, w[i + 3], 0}; break;            // OpTypePointer
      case 28: if (len >= 4) types[w[i + 1]] = {op, w[i + 2], w[i + 3]}; break;     // OpTypeArray (element, length id)
      case 43: if (len >= 4) consts[w[i + 2]] = w[i + 3]; break;                    // OpConstant
      case 59: if (len >= 4) vars.push_back({w[i + 2], w[i + 1], w[i + 3]}); break;  // OpVariable
      default: break;
    }
    i += len;
  }
  uint32_t mask = 0;
  for (const V& v : vars) {
    if (v.storage != storage || builtins.count(v.id)) continue;
    const auto l = locations.find(v.id);
    if (l == locations.end() || l->second >= 32) continue;
    uint32_t count = 1;
    if (auto p = types.find(v.type); p != types.end() && p->second.op == 32) {
      if (auto t = types.find(p->second.a); t != types.end() && t->second.op == 28) {
        if (auto c = consts.find(t->second.b); c != consts.end()) count = std::max<uint32_t>(1, c->second);
      }
    }
    for (uint32_t k = 0; k < count && l->second + k < 32; ++k) mask |= 1u << (l->second + k);
  }
  return mask;
}

std::unordered_set<uint64_t> g_interface_noted, g_interface_checked;
uint32_t g_interface_noted_count = 0, g_interface_mismatch = 0;
uint64_t PairKey(uint64_t vertex_hash, uint64_t pixel_hash) { return vertex_hash * 0x9E3779B97F4A7C15ull ^ pixel_hash; }
// Lock-free first look (per thread): a pair this thread has already noted / checked never takes g_mutex again.
thread_local std::unordered_set<uint64_t> t_noted, t_checked;
}  // namespace

uint32_t ReplacementInterpolatorMask(uint64_t guest_hash) {
  if (!REXCVAR_GET(gpu_native_shader_swap)) return 0;
  const Table* table = g_table.load(std::memory_order_acquire);
  if (!table) return 0;
  for (const auto& e : table->entries) {
    if (e->hash == guest_hash) return Skipped(guest_hash) ? 0 : e->read_mask;
  }
  return 0;
}

void NoteInterface(uint64_t vertex_hash, uint64_t pixel_hash, uint32_t vs_writes, uint32_t ps_reads) {
  const uint32_t missing = ps_reads & ~vs_writes;
  if (!missing) return;
  const uint64_t key = PairKey(vertex_hash, pixel_hash);
  if (!t_noted.insert(key).second) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_interface_noted.insert(key).second) return;
  if (++g_interface_noted_count <= 16) {
    REXGPU_INFO("[native][swap] interface: vs {:016X} writes interpolators {:04X}, replacement ps {:016X} reads {:04X}: "
                "{:04X} declared by the vertex translation and filled with zero (pair #{})",
                vertex_hash, vs_writes, pixel_hash, ps_reads, missing, g_interface_noted_count);
  }
}

// The stored / predicted modifications of a pair with a replacement: the interpolators the replacement reads added to the vertex
// and the pixel modification, as IssueDraw's mask does. The added bits are centre-sampled, unless every interpolator the
// pixel modification already had was centroid-sampled (a multisampled draw) - the draw's own sampling pattern is not stored.
bool WidenForReplacement(uint64_t pixel_hash, uint64_t& vertex_modification, uint64_t& pixel_modification) {
  const uint32_t reads = ReplacementInterpolatorMask(pixel_hash);
  if (!reads) return false;
  bool changed = false;
  SpirvShaderTranslator::Modification vertex(vertex_modification);
  if (reads & ~uint32_t(vertex.vertex.interpolator_mask)) {
    vertex.vertex.interpolator_mask |= reads;
    vertex_modification = vertex.value;
    changed = true;
  }
  SpirvShaderTranslator::Modification pixel(pixel_modification);
  const uint32_t old_mask = pixel.pixel.interpolator_mask;
  if (reads & ~old_mask) {
    const uint32_t added = reads & ~old_mask;
    const bool all_centroid = old_mask && uint32_t(pixel.pixel.interpolators_centroid) == old_mask;
    pixel.pixel.interpolator_mask = old_mask | added;
    pixel.pixel.interpolators_centroid = uint32_t(pixel.pixel.interpolators_centroid) | (all_centroid ? added : 0u);
    pixel_modification = pixel.value;
    changed = true;
  }
  return changed;
}

bool InterfaceChecked(uint64_t vertex_hash, uint64_t pixel_hash) {
  const uint64_t key = PairKey(vertex_hash, pixel_hash);
  if (t_checked.count(key)) return true;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_interface_checked.count(key) == 0) return false;
  t_checked.insert(key);
  return true;
}

void CheckInterface(uint64_t vertex_hash, uint64_t pixel_hash, const std::vector<uint8_t>& vertex_binary) {
  if (vertex_binary.size() < 24) return;  // (not translated yet: asked again by the next draw)
  const std::shared_ptr<const Entry> entry = Find(pixel_hash);
  if (!entry) return;
  std::vector<uint32_t> words(vertex_binary.size() / 4);
  std::memcpy(words.data(), vertex_binary.data(), words.size() * 4);
  const uint32_t outputs = SpirvLocationMask(words.data(), words.size(), 3);
  const uint32_t inputs = SpirvLocationMask(entry->words.data(), entry->words.size(), 1);
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_interface_checked.insert(PairKey(vertex_hash, pixel_hash)).second) return;
  const uint32_t missing = inputs & ~outputs;
  if (!missing) {
    if (g_interface_checked.size() <= 16) {
      REXGPU_INFO("[native][swap] interface ok: vs {:016X} outputs {:04X}, ps {:016X} inputs {:04X}", vertex_hash, outputs,
                  pixel_hash, inputs);
    }
    return;
  }
  ++g_interface_mismatch;
  for (uint32_t loc = 0; loc < 32; ++loc) {
    if (missing & (1u << loc)) {
      REXGPU_WARN("[native][swap] interface MISMATCH vs {:016X} ps {:016X} loc {} (vs outputs {:04X}, ps inputs {:04X})",
                  vertex_hash, pixel_hash, loc, outputs, inputs);
    }
  }
}


}  // namespace rex::graphics::vulkan::shader_swap
