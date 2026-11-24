#pragma once
// wwe2k14 patch (Session 72, gpu_native_shader_swap): the title's own fragment shaders in place of a guest pixel shader
// on NATIVE's draws. The pass, the pipeline layout, the textures and the float constants stay the translated shader's;
// only the module differs (its own NATIVE translation and pipeline: SpirvShaderTranslator::kNativeExtSwap).
//
// A replacement is SPIR-V for a fragment shader with the translated shader's interface: input location 0 = the guest's
// first interpolator, output location 0 = colour target 0, set 1 binding 2 = the float constants the guest shader uses,
// packed in register order (the registers it was written for are given at registration and must match), set 3 = the
// textures and samplers under the canonical bindings below, which the swap remaps to the translation's own.
//
// Session 80 (kFlagOwnEpilogue, SCENE_SHADERS_PLAN.md): by default a replacement takes only draws whose colour epilogue
// does nothing (no alpha test, exponent bias or gamma: the replacement's colour is written as it is). A replacement
// registered with kFlagOwnEpilogue does the epilogue itself from the system constants (set 1 binding 0: alpha test
// function in flags bits kSysFlag_AlphaPassIfLess_Shift.., alpha_test_reference, color_exp_bias, the gamma flag), so
// it takes every draw of its shader except those with alpha to coverage.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <rex/graphics/pipeline/shader/shader.h>

namespace rex::graphics {
class SpirvShader;
}

namespace rex::graphics::vulkan::shader_swap {

// Set 3: the unsigned 2D image of fetch constant fc at kImage + fc, the signed one at kImageSigned + fc, the first
// sampler of fc at kSampler + fc.
constexpr uint32_t kImage = 1000, kImageSigned = 1100, kSampler = 1200;
// Session 80: the unsigned cube image of fetch constant fc at kImageCube + fc (a samplerCube / textureCube).
constexpr uint32_t kImageCube = 1300;

// Registration flags.
constexpr uint32_t kFlagOwnEpilogue = 1;
// Session 80 (probes): the replacement reads no float constants - any guest shader of the hash takes it (no check of
// the registered float registers against the translation's).
constexpr uint32_t kFlagAnyConstants = 2;
// Session 87 (vertex swaps): the replacement indexes the float constants by register number (skinned shaders: bone matrices
// at c[76 + a0]...): set 1 binding 1 is then the whole 256-register block, and the guest shader must index its constants.
constexpr uint32_t kFlagDynamicConstants = 4;
// Session 87 (pixel skip): the GUEST shader kills pixels (it has kill instructions: Shader::kills_pixels, which alpha to coverage,
// early depth and the owned alpha learn read). The replacement's own alpha-test discard is not that, so it is registered, not read
// from the SPIR-V.
constexpr uint32_t kFlagKills = 8;
// Session 95 (gpu_d3d_decouple phase 2): the replacement reads a target drawn in a multiview pass (the post combines, the side chain): with the views on it needs its
// layer-true variant (RegisterPixelMultiview) or is refused. No effect with gpu_d3d_decouple off.
constexpr uint32_t kFlagLayerTrue = 16;

// Session 93 (gpu_d3d_typed_memo_reuse): a counter bumped whenever something a cached typed draw preparation was derived from may have changed:
// a shader registered here, a skip list changed, a shader analysed or a translation made (pipeline_cache.cpp calls NoteTranslationChange).
uint64_t TranslationGeneration();
void NoteTranslationChange();
// Thread-safe; a later registration of the same guest shader replaces the earlier one for modules made after it.
bool Register(uint64_t guest_hash, const uint32_t* words, size_t count, const uint8_t* float_registers,
              size_t float_register_count, const char* name, uint32_t flags = 0);
// Cheap (every NATIVE draw with a pixel shader asks while the switch is on); flags = the registration's.
bool Has(uint64_t guest_hash, uint32_t* flags = nullptr);
// Whether a draw of a registered shader takes the replacement: its epilogue is idle, or the replacement does its own
// and the draw has no alpha to coverage. Logs once per shader when a registered shader's draw does not.
bool Takes(uint64_t guest_hash, bool epilogue_idle, bool alpha_to_coverage);
// Session 80 (gpu_native_shader_swap_skip, live): the guest shaders whose replacement is left out - read once a frame
// (RefreshSkips, at the guest swap), so an in-run A/B can switch one replacement on and off.
void RefreshSkips();
bool Skipped(uint64_t guest_hash);
// Whether a fast path's recorded draw (its pixel ext word) no longer matches what NativePixelExt would give it now as
// far as the swap goes: the replacement switched off / skipped, or switched on for a draw that would take it.
bool SwapStale(uint64_t guest_hash, uint32_t recorded_pixel_ext);
// The replacement for the shader with its set-3 bindings remapped; false (logged once per shader) when there is none
// or it does not fit: other float constants than it was written for, or a texture / sampler the translation lacks.
// Session 80 (the arena ours): color_quantize = the translation's pixel.native_color_quantize (0 none, 1 8-bit, 2
// 16-bit fixed: the value the target stores, which the translator writes last). A replacement that declares a
// specialization constant with constant_id kSpecColorQuantize gets it as that constant's value, so its own epilogue
// rounds (8-bit) or scales into the 360's -32..32 range (16-bit fixed targets) as the translation would - one guest
// shader can draw into both kinds of target (the shadow maps' depth: an 8-bit pass and two 16-bit maps).
constexpr uint32_t kSpecColorQuantize = 0;
// Session 95 (B1, gpu_native_epilogue_spec_swap): a replacement that declares a uint specialization constant with constant_id kSpecEpilogueIdle gets 1 when the
// draw's colour epilogue is idle (the translation's kNativeExtEpilogueIdle, already part of the key): its XeOutputColor then compiles out the alpha test / exponent bias / gamma.
constexpr uint32_t kSpecEpilogueIdle = 1;
// Session 95 (B2, gpu_native_shader_ablate): the ranking tool. Families of the registered pixel replacements (by registration name) whose module is a flat grey instead
// (diagnostic, pictures wrong), switched live by the cvar (a comma list: bg, prop, particle, wrestler, crowd, shadow, blur, other, all). The module is the
// replacement registered under kAblationHash (the game registers probe_flat there). AblateExt = kNativeExtAblate when the shader's family is on, else 0.
constexpr uint64_t kAblationHash = 0xAB1A7E0000000001ull;
uint32_t AblateExt(uint64_t guest_hash);
// Session 95 (the diagnostic run): the family of a registered shader (an index into the list above), 0xFF = none registered.
uint32_t FamilyOf(uint64_t guest_hash);
// Session 82 (gpu_native_owned_alpha_learn): alpha_probe - the replacement's colour store preceded by a discard of the
// fragments whose alpha is 1 in 8-bit terms (>= 254.5 / 255): an occlusion query around a draw with this module counts the
// translucent fragments (patched into the SPIR-V; false when the module has no single store of colour target 0).
// Session 87 (pixel skip): replacement_bindings = the shader's images and samplers are the REPLACEMENT translation's own
// (SpirvShader::TextureBindingsFor(true)), not the emulated translation's.
bool Build(const SpirvShader& shader, std::vector<uint8_t>& out, uint32_t color_quantize = 0, bool alpha_probe = false,
           bool replacement_bindings = false,
           bool epilogue_idle = false, bool ablate = false);
// Session 87 (pixel skip): whether the pixel samplers the command processor holds (current_samplers_pixel_) were made from the
// replacement translation's list. The vector and this flag are written together by whoever rebuilds it and read together
// by whoever pairs it with a list (UpdateBindings, the crowd), so a cached object carries the identity of its bindings.
bool& PixelSamplersReplacement();

// ---- Session 87, stage 4 (gpu_native_vs_swap): the title's own VERTEX shaders -------------------------------------------
// The vertex-side twin of the above: our SPIR-V in place of a guest vertex shader's translation on NATIVE's draws, keyed by
// the guest vertex shader's hash (a patched copy of a base shader: one hash per vertex declaration). Same idea - the
// translation still runs (it sets up the shader's bindings and constants), only its code is replaced before the module is
// made (kNativeExtVsSwap in the translation's native_ext; VulkanPipelineCache::TranslateAnalyzedShader).
//
// A vertex replacement is SPIR-V (vertex stage) for the translated shader's interface, which it reads itself (this is the
// whole vertex-fetch model, kept as it is; the fetch is the shader's own business):
//   set 0 binding 0   the shared memory (uint[]), readonly: the vertex data
//   set 1 binding 0   the system constants (scalar block layout; explicit offsets of the members used: flags 0, vertex index
//                     endian 8, line loop closing index 12, vertex base index 16, vertex index min 496 / max 500, ndc scale
//                     124, ndc offset 140)
//   set 1 binding 1   the float constants the guest shader uses, packed in register order (the registers it was written for
//                     are given at registration and must match, as for a pixel replacement)
//   set 1 binding 4   the fetch constants (uvec4[48]; the vertex fetch constant the shader reads is its own business)
//   set 2             the vertex textures and samplers under the canonical bindings (kImage / kImageSigned / kSampler + fc),
//                     which the swap remaps to the translation's own, as for pixel shaders
//   outputs           location N = the guest's interpolator N, gl_Position in the host's clip space (the shader does the
//                     translator's final step itself: the guest position's w handling and the ndc scale / offset)
// Its vertex fetch must be the one stream it is registered for (vertex fetch constant stream_fetch_constant, default 95 =
// fetch constants dwords 190 / 191, with stream_stride_words dwords a vertex, default 8 = the arena meshes); the shader reads
// the layout inside the stream itself. It takes only NATIVE's plain vertex draws (not tessellation, not a merged-constants draw, no user clip planes, no point
// size, no transform-feedback capture) whose fetch endian modes are baked as 8-in-32 (gpu_native_vertex_endian_spec).
bool RegisterVertex(uint64_t guest_hash, const uint32_t* words, size_t count, const uint8_t* float_registers,
                    size_t float_register_count, const char* name, uint32_t flags = 0, uint32_t stream_stride_words = 8,
                    uint32_t stream_fetch_constant = 95);
// Cheap (asked for each NATIVE draw's vertex shader while the switch is on); false when the switch is off, nothing is
// registered for the hash, or the hash is skipped (gpu_native_vs_swap_skip).
void RefreshVertexSkips();  // (called by RefreshSkips)
bool VertexTakes(uint64_t guest_hash);
// Whether a recorded draw's vertex ext word (its kNativeExtVsSwap bit) no longer matches what VertexTakes says now.
bool VertexSwapStale(uint64_t guest_hash, uint32_t recorded_vertex_ext);
// Session 93 (gpu_d3d_cull_ours): a taken vertex replacement whose clip position is (position, w = 1) x c76-c78 x c56-c59 (plain arena materials, not skinned, 8-dword stream);
// stream_fetch_constant = its vertex fetch constant. See shader_swap.cpp.
bool VertexBoxCullable(uint64_t guest_hash, uint32_t* stream_fetch_constant = nullptr, std::string* why = nullptr);  // why = the entry's name, or the reason it is refused
// Session 92: the shared memory is bound as N storage buffers (one descriptor binding, an array of N) when the GPU's
// maxStorageBufferRange is below 512 MiB: N = 1 (as every replacement is written), 2 or 4. The command processor tells
// the count (SetSharedMemoryBindingCount, at its initialisation, before any shader is built); a vertex replacement
// registered with a variant for it (RegisterVertexSplit: the same code reading the array of N buffers by address as the
// translator does) is built from that, one without is refused (the game's translated shader stays).
void SetSharedMemoryBindingCount(uint32_t count);
bool RegisterVertexSplit(uint64_t guest_hash, uint32_t binding_count, const uint32_t* words, size_t count);
// Session 94 (gpu_d3d_decouple bit 3, phase 1B): the multiview variant of a vertex replacement (src/scene *_mv: every float constant read at register + gl_ViewIndex * XE_VSTRIDE,
// XE_VSTRIDE = a specialization constant, SpecId 1, whose default BuildVertex patches to 256). With the switch on it is the only code a taken vertex shader runs; none = refused.
bool RegisterVertexMultiview(uint64_t guest_hash, const uint32_t* words, size_t count);
// Session 95 (gpu_d3d_decouple phase 2): the layer-true variant of a pixel replacement registered with kFlagLayerTrue (see there). Call after Register for the same hash.
bool RegisterPixelMultiview(uint64_t guest_hash, const uint32_t* words, size_t count);
// The registers of a taken vertex replacement our GLSL names as matrices, as a bitmap (4 x 64 registers): the view-projection c56-c59 and the world rows c76-c78 (those the
// replacement reads), or for a shader that indexes its constants (skinned) c56-c59 and the bone palette c76-c161. False (mask zero): not ours / nothing to blend.
bool VertexBlendRegisters(uint64_t guest_hash, uint64_t mask[4]);
// The replacement for the translated vertex shader with its set-2 bindings remapped; false (logged once per shader) when
// Session 94 (gpu_native_instance_world): the instancing variant of an arena vertex shader (src/scene *_inst): the same code with every float constant read at
// register + gl_InstanceIndex * STRIDE, STRIDE = a specialization constant (SpecId 0, int) whose default BuildVertex patches with the translation's
// instanced_float_constants_stride. Taken for the draws vulkan_repeat_draw_merge merges (the SDK stride modification), refused otherwise as before.
bool RegisterVertexInstanced(uint64_t guest_hash, const uint32_t* words, size_t count);
void LatchInstanceWorld();                                  // at the command processor's setup: reads gpu_native_instance_world
bool InstanceWorldLatched();
bool VertexInstanceable(uint64_t guest_hash);               // latched on, gpu_native_vs_swap on, registered with an instancing variant, not skipped
void NoteWorldInstance(uint64_t frame, uint32_t run_instances, uint64_t guest_hash);  // a merged draw joined a run ([native] world instancing)
// Session 94: the typed repeat's funnel, counted while world instancing is latched on and printed with the 600-frame line even when nothing merged.
enum WorldStat : uint32_t { kWsChecked, kWsCandidate, kWsMiss, kWsNoBase, kWsTried, kWsRefused, kWsDrawnApart, kWsCount };
void NoteWorldStat(uint64_t frame, uint32_t what, uint32_t detail = 0);  // kWsMiss: detail = field group (0..6); kWsRefused: detail = repeat_refusal_
// The replacement for the translated vertex shader with its set-2 bindings remapped; false (logged once per shader) when
// there is none or it does not fit (other float constants, a texture / sampler the translation lacks, a modification the
// replacement was not written for). modification = the translation's (Modification::value).
// Session 87 (vertex skip): replacement_bindings = the vertex textures and samplers are the REPLACEMENT translation's own
// (SpirvShader::TextureBindingsFor(true)); dry_run = only the fit checks (whether this draw's modification takes the replacement:
// VertexFits), nothing built.
bool BuildVertex(const SpirvShader& shader, uint64_t modification, std::vector<uint8_t>& out, bool replacement_bindings = false,
                 bool dry_run = false, bool mirror = false, bool skin = false);
inline bool VertexFits(const SpirvShader& shader, uint64_t modification) {
  std::vector<uint8_t> unused;
  return BuildVertex(shader, modification, unused, false, true);
}
// Session 95 (gpu_native_vertex_input): the vertex-input (_vi) variant of a vertex replacement (src/scene *_vi, tools/shaders/build_vertex_input_vs.py): the vertex's stream_stride_words dwords come
// from vertex inputs (binding 0, by vertex_input_layout.h's rule), the index is gl_VertexIndex as it is. Call after RegisterVertex.
// With the switch on, a draw the typed path proves (BuildVertexInputPlan) gets this module (BuildVertex mirror = true) instead of the plain code.
bool RegisterVertexMirror(uint64_t guest_hash, const uint32_t* words, size_t count);
void LatchVertexMirror();                  // at the command processor's setup: reads gpu_native_vertex_input
int32_t VertexInputLevel();                // 0 off, 1 our shaders, 2 ours and the translated ones
bool VertexMirrorLatched();                // (level above 0)
// Latched on, gpu_native_vs_swap on, registered with a _vi variant, not skipped: *stride_words = the one stream's dwords a vertex.
bool VertexMirrorable(uint64_t guest_hash, uint32_t* stream_fetch_constant = nullptr, uint32_t* stride_words = nullptr);
inline bool VertexMirrorFits(const SpirvShader& shader, uint64_t modification) {
  std::vector<uint8_t> unused;
  return BuildVertex(shader, modification, unused, true, true, true);
}
// Session 95 (r3gpu): the same for the skin-once (_so) variant, which has its own slot (RegisterVertexSkinOnce).
inline bool VertexSkinFits(const SpirvShader& shader, uint64_t modification) {
  std::vector<uint8_t> unused;
  return BuildVertex(shader, modification, unused, true, true, true, true);
}
// Session 95 (gpu_native_skin_once, vulkan/native/skin_once.h): a registered vertex shader whose draws can read the frame's skinned rows. Its _so variant has its own slot
// (RegisterVertexSkinOnce: the shader keeps its _vi module for the draws skin once refuses, gpu_native_skin_with_input); `role` = SkinOnce::Role, bit 4 = its blend is CH_SKIN_STYLE 1. Call after RegisterVertex. The compute shader is the game's too (src/scene/skin_once.comp).
bool RegisterVertexSkinOnce(uint64_t guest_hash, const uint32_t* words, size_t count);
bool RegisterSkinOnceRow(uint64_t guest_hash, uint32_t role);
bool RegisterSkinOnceCompute(const uint32_t* words, size_t count);
const uint32_t* SkinOnceCompute(size_t* count);
void LatchSkinOnce();  // at the command processor's setup: reads gpu_native_skin_once
bool SkinOnceLatched();
// Latched on, gpu_native_vs_swap on, a listed shader with its variant, not skipped: its role bits and the stream's dwords a vertex.
bool SkinOnceRow(uint64_t guest_hash, uint32_t* role, uint32_t* stride_words);
// Whether the vertex samplers the command processor holds (current_samplers_vertex_) were made from the replacement's list.
bool& VertexSamplersReplacement();


// ---- Session 87, stage 4 (gpu_native_swap_skip_translation): a registered shader's analysis from its registration -------
// What the pipeline cache, the binding update and the system constants read of a shader's microcode analysis and
// translation, for a shader that has a replacement registered (pixel: Has and not skipped; vertex: VertexTakes). It is read
// from the registration - the float registers, the vertex stream (fetch constant, stride), the flags - and from the
// replacement's SPIR-V: the set-3 / set-2 bindings under the canonical numbers (textures by fetch constant, dimension and
// signedness, samplers), the output locations (a vertex shader's interpolators, a pixel shader's colour targets, the
// fragment depth), the input locations (a pixel shader's interpolators = the registers it reads), OpKill. Fields it cannot
// know are left at the value of a shader without them (see Shader::RegisteredAnalysis); a verify run compares the result
// with the real analysis (VulkanPipelineCache). False when the shader has no registration, the registration cannot describe
// the shader (kFlagAnyConstants: the real float registers are not known), or the SPIR-V is cut.
bool DescribePixel(uint64_t guest_hash, Shader::RegisteredAnalysis& out);
bool DescribeVertex(uint64_t guest_hash, Shader::RegisteredAnalysis& out);

// ---- Session 90 (Adreno link): the pixel replacement's interface and the vertex shader it is paired with ----------------
// The replacement's input locations as an interpolator mask, contiguous from location 0 (bits 0..N-1; N = highest input location
// + 1): what the vertex translation must declare as outputs (zero-filled where the guest vertex shader writes none). The
// replacement's fixed locations are the contract, and a linker that refuses an unwritten input (Adreno) needs them written.
// 0 = no replacement, the switch off, or the hash skipped.
uint32_t ReplacementInterpolatorMask(uint64_t guest_hash);
// Counted log (the first 16 pairs) of a pair whose vertex shader writes fewer interpolators than the replacement reads.
void NoteInterface(uint64_t vertex_hash, uint64_t pixel_hash, uint32_t vs_writes, uint32_t ps_reads);
// The static check: the replacement's input locations against the final vertex translation's output locations
// (translated_binary); MISMATCH logged once per pair and input. InterfaceChecked = the pair was already checked.
bool InterfaceChecked(uint64_t vertex_hash, uint64_t pixel_hash);
// Adds the replacement's read interpolators to a stored record's vertex and pixel modifications (false = nothing to add).
bool WidenForReplacement(uint64_t pixel_hash, uint64_t& vertex_modification, uint64_t& pixel_modification);
void CheckInterface(uint64_t vertex_hash, uint64_t pixel_hash, const std::vector<uint8_t>& vertex_binary);

}  // namespace rex::graphics::vulkan::shader_swap
