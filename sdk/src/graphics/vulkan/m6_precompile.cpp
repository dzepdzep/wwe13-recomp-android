/**
 * @file        graphics/vulkan/m6_precompile.cpp
 * @brief       M6 (Session 92): the precompile translates the .xvar variants into the translation store
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 *
 * The game's precompile mode (gpu_native_precompile; the launcher runs it before a Play) calls M6PrecompileStep from the native
 * renderer's frame end. Each variant of the player's .xvar list (translation_store.h) is translated on a scratch shader made from
 * the loaded shader's own microcode - the real shader and its translations are not touched, no module is made - by the same
 * translator instance, with the same modification / native_ext / pixel position scale, as the play run will ask for; the hook in
 * ShaderTranslator::TranslateAnalyzedShader stores the result. A variant whose shader is not loaded yet waits (retry passes every
 * 10 frames, 20 at most, as the pipeline records). The queue is file-level state: no data members in the cache's header.
 */

#include <rex/graphics/vulkan/pipeline_cache.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <set>
#include <vector>

#include <rex/cvar.h>
#include <rex/graphics/pipeline/shader/spirv.h>
#include <rex/logging.h>

#include "../pipeline/shader/translation_store.h"
#include "native/shader_swap.h"

REXCVAR_DECLARE(int32_t, gpu_native_vertex_input);  // wwe2k14 patch (Session 95, Lane C; defined in native\shader_swap.cpp)

namespace rex::graphics::vulkan {

namespace {

struct Queue {
  uint32_t generation = 0;
  bool loaded = false;
  std::vector<translation_store::Variant> pending, retry;
  size_t next = 0;
  uint32_t passes = 0;
  uint64_t total = 0, translated = 0, held = 0, refused = 0;
  uint64_t translate_us = 0, translate_count = 0;  // the translator's own time (translated variants only)
  // wwe2k14 patch (Session 95, Lane C): the attribute forms made beside the listed variants (gpu_native_vertex_input 2), and the shaders with none, each logged once by name.
  uint64_t input_translated = 0, input_held = 0, input_failed = 0;
  std::set<uint64_t> input_refused;
  std::chrono::steady_clock::time_point started{};
  bool final_logged = false;
  std::chrono::steady_clock::time_point retry_time{};
};
Queue g_q;

constexpr uint32_t kMaxRetryPasses = 20;

void LogProgress(const Queue& q, const char* what) {
  const size_t waiting_shader = q.retry.size();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - q.started).count();
  REXGPU_INFO("[m6][precompile] {} variants {}: translated {}, stored already {}, waiting for their shader {}, refused {}, {:.0f} ms", what,
              q.total, q.translated, q.held, waiting_shader, q.refused, ms);
  if (q.translate_count) {
    REXGPU_INFO("[m6][precompile] translator time: {} translations in {:.1f} ms = {:.0f} a second, mean {:.0f} us", q.translate_count,
                double(q.translate_us) / 1000.0, double(q.translate_count) * 1e6 / double(std::max<uint64_t>(q.translate_us, 1)),
                double(q.translate_us) / double(q.translate_count));
  }
  if (q.input_translated || q.input_held || q.input_failed || !q.input_refused.empty()) {
    REXGPU_INFO("[m6][precompile] attribute forms: translated {}, stored already {}, failed {}; vertex shaders with no form {}", q.input_translated, q.input_held, q.input_failed,
                q.input_refused.size());
  }
}

}  // namespace

int VulkanPipelineCache::M6TranslateVariant(bool native, bool pixel, uint64_t hash, uint64_t modification, uint32_t native_ext,
                                            float scale_x, float scale_y) {
  VulkanShader* shader = FindShader(hash);
  if (!shader) return 2;
  if (shader->type() != (pixel ? xenos::ShaderType::kPixel : xenos::ShaderType::kVertex)) return 3;
  // A replacement's translation with its own bindings is made without the translator: nothing to store.
  if (native_ext & SpirvShaderTranslator::kNativeExtOwnBindings) return 3;
  // wwe2k14 patch (Session 92, M6): NATIVE's twin pixel position scale (a fraction of the draw scale) cannot round-trip exactly
  // through the {:.4f} text of the census's V lines: such a variant is skipped (it is translated, stored and listed on first sight).
  if (pixel && native && SpirvShaderTranslator::Modification(modification).pixel.native_twin_position) {
    static uint32_t skipped = 0;
    if (skipped++ < 4) {
      REXGPU_INFO("[m6][precompile] skipped a NATIVE twin-position pixel variant {:016X} mod {:016X} (its exact scale is not in the list)",
                  hash, modification);
    }
    return 3;
  }
  // wwe2k14 patch (Session 95, Lane C of out\s95\android_structural.md): the attribute form is another translation of the same V line, so the list stays as it is. Made only when a draw can ask for it,
  // the switch at 2 (a line that already names the bit, stored by an earlier run, waits for the same). Our own rows have their _vi module: no translator there.
  const uint32_t input_bit = SpirvShaderTranslator::kNativeExtVsInput;
  const bool input_on = !pixel && REXCVAR_GET(gpu_native_vertex_input) >= 2;
  if ((native_ext & input_bit) && !input_on) return 3;
  // (Review fix) The rows our own shader swap holds have their _vi module, and the draw's rule (NativeRenderer::VertexInputFormOf) also refuses xfb_position and the latched / decoupled blocks.
  const bool input_derive = input_on && !(native_ext & (input_bit | SpirvShaderTranslator::kNativeExtVsSwap)) && !shader_swap::Has(hash) && !shader_swap::InstanceWorldLatched() &&
                            !SpirvShaderTranslator::DecoupleViewConstantBlocks();
  translation_store::PrecompileScope scope;
  // The modifications the record paths would ask for with the current settings (a V made under other settings than the player's):
  // the variant as listed, and as conformed when that differs (as KeyFromRecord does: vertex by whether it is NATIVE's, NATIVE's
  // pixel with the texture sign mask known).
  uint64_t modifications[2] = {modification, modification};
  if (!pixel) {
    modifications[1] = ConformStoredVertexModification(*shader, modification, nullptr, (native_ext & ~input_bit) != 0);
  } else if (native) {
    modifications[1] = ConformStoredNativePixelModification(*shader, modification, nullptr, true);
  }
  // One translation: 0 made, 1 held already, 3 not valid.
  const auto translate = [&](SpirvShader& scratch, uint64_t variant_modification, uint32_t variant_ext) {
    Shader::Translation* translation = scratch.GetOrCreateNativeTranslation(variant_modification, variant_ext);
    SpirvShaderTranslator& translator = (pixel && native) ? NativeShaderTranslator() : *shader_translator_;
    if (pixel && native) {
      translator.SetPixelPositionScale(scale_x, scale_y);
    }
    const auto t0 = std::chrono::steady_clock::now();
    const bool valid = translator.TranslateAnalyzedShader(*translation);
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    if (translation_store::LastOutcome() == 1) {
      return 1;  // held already (this thread's own outcome of the call above)
    }
    g_q.translate_us += uint64_t(us);
    ++g_q.translate_count;
    return valid ? 0 : 3;
  };
  int result = 1;
  for (int i = 0; i < 2; ++i) {
    if (i && modifications[1] == modifications[0]) break;
    SpirvShader scratch(shader->type(), hash, shader->ucode_dwords(), shader->ucode_dword_count(), std::endian::native);
    scratch.AnalyzeUcode(ucode_disasm_buffer_);
    const int outcome = translate(scratch, modifications[i], native_ext);
    if (outcome != 1) {
      result = outcome == 0 ? 0 : (result == 1 ? 3 : result);
    }
    if (!input_derive) continue;
    // (Session 95, Lane C) The form's own outcome stays out of the line's: the plain variant is what the line lists.
    const Shader::VertexInputFormInfo& form = scratch.VertexInputForm();
    const SpirvShaderTranslator::Modification input_modification(modifications[i]);
    if (!form.eligible) {
      if (g_q.input_refused.insert(hash).second) {
        REXGPU_INFO("[m6][precompile] vertex shader {:016X}: no attribute form: {}", hash, form.refusal);
      }
    } else if (input_modification.vertex.host_vertex_shader_type == Shader::HostVertexShaderType::kVertex && !input_modification.vertex.xfb_position &&
               !input_modification.vertex.instanced_float_constants_stride) {
      const int input_outcome = translate(scratch, modifications[i], native_ext | input_bit);
      ++(input_outcome == 0 ? g_q.input_translated : input_outcome == 1 ? g_q.input_held : g_q.input_failed);
    }
  }
  return result;
}

void VulkanPipelineCache::M6PrecompileStep(uint64_t frame, uint32_t budget_ms) {
  Queue& q = g_q;
  if (!translation_store::IsOpen()) return;
  if (!q.loaded || q.generation != translation_store::Generation()) {
    q = Queue();
    q.generation = translation_store::Generation();
    q.loaded = true;
    q.pending = translation_store::LoadedVariants();
    q.total = q.pending.size();
    q.started = std::chrono::steady_clock::now();
    q.retry_time = q.started;
    if (q.total) {
      REXGPU_INFO("[m6][precompile] {} variants queued", q.total);
    }
  }
  if (q.next >= q.pending.size() && !q.retry.empty() && q.passes < kMaxRetryPasses && frame % 10 == 0) {
    ++q.passes;
    q.pending.swap(q.retry);
    q.retry.clear();
    q.next = 0;
  }
  const auto t0 = std::chrono::steady_clock::now();
  while (q.next < q.pending.size() && std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(budget_ms)) {
    const translation_store::Variant v = q.pending[q.next++];
    switch (M6TranslateVariant(v.native, v.pixel, v.hash, v.modification, v.native_ext, v.scale_x, v.scale_y)) {
      case 0:
        ++q.translated;
        break;
      case 1:
        ++q.held;
        break;
      case 2:
        q.retry.push_back(v);
        break;
      default:
        ++q.refused;
        break;
    }
  }
  if (!q.final_logged && q.next >= q.pending.size() && (q.retry.empty() || q.passes >= kMaxRetryPasses)) {
    q.final_logged = true;
    if (q.total) LogProgress(q, "done:");
  }
}

uint32_t VulkanPipelineCache::M6PrecompileWaiting() const {
  const Queue& q = g_q;
  if (!q.loaded || !translation_store::IsOpen()) return 0;
  return uint32_t((q.pending.size() - std::min(q.next, q.pending.size())) + (q.passes < kMaxRetryPasses ? q.retry.size() : 0));
}

// wwe2k14 patch (Session 93, M6 step 2): the variants dealt with (translated, held already or refused) of those queued - the
// phone launcher's "Preparing shaders 120 / 462" (the status string's "m6 D of T"). A shader that never loaded counts as dealt
// with once its retry passes are over.
void VulkanPipelineCache::M6PrecompileProgress(uint64_t& done, uint64_t& total) const {
  const Queue& q = g_q;
  done = total = 0;
  if (!q.loaded || !translation_store::IsOpen()) return;
  total = q.total;
  done = total - std::min<uint64_t>(total, M6PrecompileWaiting());
}

}  // namespace rex::graphics::vulkan
