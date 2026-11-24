#pragma once
/**
 * The native renderer (wwe2k14 patch, Session 38).
 *
 * The emulated renderer reproduces the Xbox 360's GPU: every draw goes through the EDRAM emulation (render targets
 * as a 10 MB tile buffer, resolves into textures). The native renderer draws the game's scene into ordinary host
 * render targets of any size instead, using the game's own shaders translated on the player's machine.
 *
 * Where it sits: behind the command processor. The guest's command stream, fences, waits, occlusion queries and
 * resolves are processed exactly as before, so the game's behaviour cannot change; the native renderer only adds
 * host GPU work. The command processor hands each recorded draw over (OnDraw), and the native renderer replays the
 * ones it owns into its own targets when the emulated render pass ends (Flush), reusing the descriptor sets the
 * emulated draw bound (guest memory, constants, textures).
 *
 * This first stage is a side-car: the emulated renderer still draws everything and presents; the native renderer
 * draws its part beside it and the swap can show either picture, or both split (gpu_native_view). The first part it
 * owns is the scene's opaque geometry (the draws into the frame's main render target that write depth), with the
 * game's vertex shaders and our own simple shading. A parity check compares its depth with the emulated depth of the
 * same frame (read from the EDRAM right before the game's scene depth resolve) and GPU timestamps measure both.
 *
 * The second stage (Session 39, gpu_native 2) replaces: in a replacing frame the native renderer owns the scene
 * draws it can draw exactly (Claim) and the emulated renderer skips their draw calls (keeping all their other
 * bookkeeping). Our targets are then a cache of the scene's surface in the EDRAM: they are loaded from it before our
 * next draw whenever the EDRAM holds something newer (the frame's start, an emulated draw, a clearing resolve), and
 * our picture is written into it before anything emulated reads or writes it (an emulated draw, any resolve, the
 * frame's end). So everything outside the scene (shadow maps before it, the resolves, post-processing, the HUD) runs
 * on the emulated renderer exactly as before, and a draw we cannot do inside the scene is drawn emulated between a
 * write-back and a reload: correct by construction, at the cost of two full-screen passes.
 *
 * Design and road map: docs\native\DESIGN.md in the wwe2k14-recomp repository.
 */

#include <array>
#include <atomic>
#include <bitset>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <rex/graphics/command_processor.h>
#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include <rex/graphics/util/draw.h>
#include <rex/ui/vulkan/api.h>

namespace rex::graphics::vulkan {
class VulkanCommandProcessor;
class DeferredCommandBuffer;
class VulkanShader;
}

namespace rex::graphics::vulkan::native {

class IndexPool;  // Session 95, item D (index_pool.h)
class VertexInputState;  // Session 95 (vertex_input.h)
struct ViForm;
class VertexMirror;  // Session 95 (vertex_mirror.h): gpu_native_skin_once's vertex source
class SkinOnce;  // Session 95 (skin_once.h)
struct OwnedGraphState;  // wwe2k14 patch (Session 95): owned_graph.cpp

// Fixed-function state of one emulated draw, as its pipeline has it.
struct DrawRaster {
  bool supported = false;  // a plain vertex shader + rasterizer draw (no geometry / tessellation stage, filled)
  VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  bool primitive_restart = false;
  VkCullModeFlags cull_mode = VK_CULL_MODE_NONE;
  bool front_face_clockwise = false;
  bool depth_test = false;
  bool depth_write = false;
  VkCompareOp depth_compare = VK_COMPARE_OP_ALWAYS;
  // As the host-render-target path computes it (the interlock path applies the guest's polygon offset in its pixel
  // shaders and never sets the host's).
  float depth_bias_constant = 0.0f;
  float depth_bias_slope = 0.0f;
  bool stencil = false;  // the stencil test is on (the native path has no stencil)
  // Session 46 (truth frames): the stencil ops, front | back << 12, each fail | zpass << 3 | zfail << 6 | func << 9
  // (xenos::StencilOp / CompareFunction: VkStencilOp / VkCompareOp in the same order); back = front without
  // backface_enable.
  uint32_t stencil_ops = 0;
  // Session 43 (review): PA_CL_CLIP_CNTL clip_disable - the emulated pipeline clamps depth then (depthClampEnable);
  // Session 69: ours too (RasterKey bit 14), where the device has depthClamp.
  bool clip_disable = false;
  // Session 42: rectangle / quad lists the emulated pipeline expands with this geometry shader (topology: what the
  // geometry shader takes). Drawn natively as post-processing draws only (supported stays false).
  bool expanded = false;
  VkShaderModule geometry_shader = VK_NULL_HANDLE;
};

// One emulated draw, handed over by the command processor right after it recorded the draw call.
struct Draw {
  uint64_t vertex_shader_hash = 0;
  uint64_t pixel_shader_hash = 0;
  // The guest vertex shader as the emulated draw runs it (the translation for this draw's modification).
  VkShaderModule vertex_shader = VK_NULL_HANDLE;
  // The guest pipeline layout and the descriptor sets the emulated draw has bound with it.
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkDescriptorSet sets[SpirvShaderTranslator::kDescriptorSetCount] = {};
  uint32_t sets_bound = 0;  // bit per set index
  uint32_t dynamic_offsets[SpirvShaderTranslator::kConstantBufferCount] = {};
  uint32_t dynamic_offset_count = 0;  // 0: the constants set has no dynamic offsets
  // The host draw call.
  bool indexed = false;
  VkBuffer index_buffer = VK_NULL_HANDLE;
  VkDeviceSize index_offset = 0;
  VkIndexType index_type = VK_INDEX_TYPE_UINT16;
  uint32_t vertex_count = 0;
  // In host pixels of the emulated render target (the resolution scale applied).
  VkViewport viewport = {};
  VkRect2D scissor = {};
  DrawRaster raster;
  // The guest render targets (RB_SURFACE_INFO, RB_COLOR_INFO, RB_DEPTH_INFO).
  uint32_t surface_info = 0;
  uint32_t color_info = 0;
  uint32_t depth_info = 0;
  bool screen_space = false;  // rectangle list, or no viewport transform (vertices already in pixels)
  bool memexport = false;     // the shaders write guest memory
  // The guest pixel shader translated for host render targets (gpu_native_shading 1; VK_NULL_HANDLE: none, the draw
  // writes depth only, or it could not be translated), render target 0's colour write mask (RGBA bits) and blending.
  VkShaderModule pixel_shader = VK_NULL_HANDLE;
  uint32_t color_write_mask = 0;
  uint32_t blend_control = 0;  // RB_BLENDCONTROL0
  float blend_constants[4] = {};  // RB_BLEND_RED .. ALPHA
  uint32_t color_mask_all = 0;    // all four render targets' write masks (4 bits each)
  bool occlusion_query = false;   // a guest occlusion query is counting samples
  bool owned = false;             // replacing frame: the native renderer draws it, the emulated draw call is skipped
  // Session 40: the native pipeline Claim made for it (Flush draws with it, no second lookup).
  VkPipeline pipeline = VK_NULL_HANDLE;
  // Session 40: claimable, but its segment stays emulated (gpu_native_min_draws); its repeats are declined too.
  bool declined = false;
  // Session 42: claimed as a post-processing draw (after the scene's colour resolve, into a post target).
  bool post = false;
  // Session 43: the pixel shader writes its depth rounded to 20e4 (a post draw that tests or writes depth).
  bool float24_depth = false;
  // Session 43 (gpu_native_post_quantize): the pixel shader's colour rounded as the interlock shader packs it
  // (SpirvShaderTranslator::Modification::pixel.native_color_quantize: 0 none, 1 8-bit, 2 16-bit fixed).
  uint32_t color_quantize = 0;
  // Session 44 (gpu_native_merge): the vertex shader translation reads its float constants at gl_InstanceIndex x
  // merge_stride vec4s (vulkan_repeat_draw_merge's instanced variant; 0: not instanced), float_count of them; Flush
  // draws instance_count instances (consecutive repeats merged into this one).
  uint32_t merge_stride = 0;
  uint32_t float_count = 0;
  uint32_t instance_count = 1;
  // Session 45 (the frame census): the emulated renderer drew it as an EDRAM fill (gpu_fast_clear).
  bool fast_clear = false;
  // Session 56 (gpu_native_truth_clear_attachments): a fast clear's host depth, from its vertex shader run on the CPU,
  // when it is exactly 0 or 1 (a plain viewport, no depth bias); negative: not known.
  float clear_depth = -1.0f;
  // wwe2k14 patch (Session 80, gpu_native_truth_clear_load): a fast clear's colour, when its pixel shader passes the
  // vertex shader's first interpolator through (a listed shader) and the vertex shader gives every vertex the same one
  // (run on the CPU with the depth above).
  bool clear_color_known = false;
  float clear_color[4] = {};
  // Session 45 (diagnostics): xenos::PrimitiveType of the guest draw and of the host draw, the host vertex shader type
  // (Shader::HostVertexShaderType) and whether the primitive is polygonal.
  uint8_t guest_primitive = 0, host_primitive = 0, host_vs_type = 0;
  bool polygonal = false;
  // Session 45 (gpu_native_post_sync): a fast clear's fill - its rectangle in guest samples (x0 y0 x1 y1, exclusive),
  // its pitch in tiles and the EDRAM bases it fills (UINT32_MAX: not filled).
  uint32_t fill_rect[4] = {};
  uint32_t fill_pitch_tiles = 0;
  uint32_t fill_color_base = UINT32_MAX, fill_depth_base = UINT32_MAX;
  // Session 45 (clear mirrors): the EDRAM dwords the fill copies (its first pixel's sample 0, colour / depth).
  uint32_t fill_src_color = 0, fill_src_depth = 0;
  // (Review fix: the depth fill's keep mask - a stencil-only clear keeps each pixel's depth: no depth mirror.)
  uint32_t fill_keep_depth = 0;
  bool fill_color_64bpp = false;  // Session 45: the colour fill is of a 64bpp surface (gpu_native_all)
  // Session 45 (gpu_native_post_sync): the guest pixels a claimed quad / rectangle draw provably writes
  // (NativeCoverRect: x0 y0 x1 y1 exclusive; empty when x1 <= x0).
  int32_t cover[4] = {};
  // Session 80 (gpu_native_truth_bounds_vs): an upper bound of the guest pixels the draw can touch - every vertex it
  // uses run through its vertex shader on the CPU, the box of their screen positions widened by 2 pixels (x0 y0 x1 y1
  // exclusive; empty when x1 <= x0). TruthDrawRect narrows the draw's rectangle (its tiles, loads, scissor) to it.
  int32_t bounds[4] = {};
  uint64_t skin_side_key = 0;  // (Session 95, r3gpu, gpu_native_skin_bind_lean) a skin-once side draw: what its shader reads of the constants, hashed; 0 = none
  // Session 46 (truth frames): claimed into the truth images - its colour and depth image (-1: it writes / tests none,
  // it joins the run it is drawn in), the colour class its pipeline was made for (1 RGBA8, 2 RGBA16_SNORM) and the
  // samples per pixel of an MSAA view (a 4x clear of a 1x surface is drawn with its viewport and scissor doubled).
  bool truth = false;
  int16_t truth_color = -1, truth_depth = -1;
  uint8_t truth_class = 1, truth_sx = 1, truth_sy = 1;
  uint32_t truth_epoch = 0;  // Session 47: the claim's truth epoch (ClaimTruthRepeat)
  uint8_t stencil_ref = 0;  // Session 46: RB_STENCILREFMASK stencilref (a truth clear's stencil value)
  // Session 46 (truth frames): RB_STENCILREFMASK and _BF - reference, compare mask, write mask (front, back).
  uint8_t stencil_refs[2] = {}, stencil_read_masks[2] = {}, stencil_write_masks[2] = {};
  // Session 61 (pipeline storage): what the draw's shader modules and layout were made from - the vertex shader's and
  // the guest pixel shader's modifications (the emulated translations), the native pixel translation's own settings
  // (float24 / quantize / twin as GetNativePixelShaderTranslation got them) and the layout's binding counts (pixel
  // textures, pixel samplers, vertex textures, vertex samplers).
  uint64_t vertex_modification = 0, pixel_modification = 0;
  bool ps_float24 = false, ps_twin = false;
  uint8_t ps_quantize = 0;
  // wwe2k14 patch (Session 66, PS spec): the pixel specialisation key (VulkanPipelineCache::GetNativePixelSpecKey) the
  // pixel modification above was made with; the lean / repeat / crowd paths compare against it.
  uint32_t pixel_spec_key = 0;
  // wwe2k14 patch (Session 66, VS mul): VulkanPipelineCache::GetNativeVertexMulKey at the draw (whether or not the draw's
  // host vertex shader type took it); the lean / repeat / crowd paths compare against it.
  uint32_t vertex_mul_key = 0;
  // wwe2k14 patch (Session 66, native_ext): the NATIVE-only specialisation words of the draw's pixel / vertex shader
  // modules (SpirvShaderTranslator::NativeExtBits; 0 = none). Not in the 64-byte pipeline record: a stored record is
  // rebuilt with VulkanPipelineCache::PredictNativePixelExt / PredictNativeVertexExt.
  uint32_t pixel_ext = 0, vertex_ext = 0;
  uint16_t layout_counts[4] = {};
  // Session 95 (out\s95\android_structural.md, gpu_native_vertex_input): an attribute draw reads its vertices as vertex attributes, from the shared memory buffer or the inline ring bound directly.
  // vb / vb_offsets = the vb_count bindings (BindDrawInputs); vi = one byte per binding, the stride in dwords (vertex_input_layout.h), the pipeline's vertex input; vertex_offset = the base vertex
  // added to every index (vertexOffset), or the first vertex of a non-indexed draw (plain draws leave it 0). plain_restart = the guest's primitive restart: a stored record always describes the plain
  // draw, and raster.primitive_restart is cleared under gpu_native_index_no_restart. attribute false: a plain draw (every replay site then draws exactly as before). A shader that fetches no vertex at
  // all is an attribute draw with vi 0 and no bindings: its module is the attribute form, so `attribute` is the test, not vi. Never a stored base.
  VkBuffer vb[4] = {};
  VkDeviceSize vb_offsets[4] = {};
  uint32_t vb_count = 0;
  int32_t vertex_offset = 0;
  uint32_t vi = 0;
  bool attribute = false;
  bool sync_pipeline = false;  // (Session 95, P7: an atlas member's attribute pipeline is made at once, never skipped: its quads are baked once per load)
  bool plain_restart = false;
  uint32_t guest_index_base = 0;  // an attribute draw's guest index address (its index_offset is the pool's, which moves): the identity OwnedAlphaKeyOf mixes
  // Session 61 (DLSS / DLAA object motion in truth frames): a character draw's captured positions - the transform
  // feedback buffer range the truth draw writes while it is drawn (VK_NULL_HANDLE: none). Never a lean base.
  VkBuffer xfb_buffer = VK_NULL_HANDLE;
  VkDeviceSize xfb_offset = 0, xfb_size = 0;
  // wwe2k14 patch (Session 72 part 4, gpu_native_truth_skip_restores): a registered restore shader's draw - the guest
  // pages (address >> 12) of the textures it writes back as colour and as depth.
  bool restore = false;
  // wwe2k14 patch (Session 95, gpu_owned_copy_lean): RepeatDrawFast's copy of its base - it binds the base's descriptor sets, no texture request of its own.
  bool repeat = false;
  // wwe2k14 patch (Session 80, own crowd): a crowd run's first member whose draw our own crowd renderer takes - the
  // truth pass, its loads and bookkeeping are recorded as for any draw, only the draw call is left out.
  bool own_skip_call = false;
  uint32_t restore_color_page = 0, restore_depth_page = 0;
  // Session 82 (alpha learn): a hash of the base addresses of the pixel shader's bound textures (the fetch constants), 0 none.
  uint64_t texture_key = 0;
  bool no_store = false;  // Session 82: a variant of ours (the alpha learn's unblended / probe pipelines): not stored as a pipeline record
  // wwe2k14 patch (Session 87, D3D layer F2): a clear built from the game's D3D device state (NativeRenderer::D3dClear),
  // not from a guest draw: it has no pipeline. The owned frame may hold it as a pass's start; a truth path that is asked
  // to draw it cannot (counted, logged loudly: D3dNoteClearLost).
  bool d3d_record = false;
  // wwe2k14 patch (Session 92, M2a, gpu_native_owned_only): the claim refused this draw (a TruthReason) but the latch claims it anyway so the
  // emulated tier is never reached: TruthRecordDraw drops it (counted, named). 0 = a normal claim.
  uint8_t claim_refused = 0;
  // wwe2k14 patch (Session 95, the diagnostic run): on an armed frame the draw's id and its mark's aux word (native/diag_layer.h); 0 = no mark.
  uint32_t diag_id = 0, diag_aux = 0;
};

// Session 95: a draw's vertex buffers, once per draw at every site that records one (bound per draw, no memo: offsets differ on nearly every draw). Nothing for a plain draw.
template <typename CommandBuffer>
inline void BindDrawInputs(CommandBuffer& cb, const Draw& draw) {
  if (draw.vb_count) cb.CmdVkBindVertexBuffers(0, draw.vb_count, draw.vb, draw.vb_offsets);
}

class NativeRenderer {
 public:
  explicit NativeRenderer(VulkanCommandProcessor& command_processor);
  ~NativeRenderer();
  NativeRenderer(const NativeRenderer&) = delete;
  NativeRenderer& operator=(const NativeRenderer&) = delete;

  // gpu_native, latched at each frame boundary (a frame is drawn either way).
  bool enabled() const { return enabled_; }
  // gpu_native_census, latched at each frame boundary.
  bool census_active() const { return census_frames_left_ > 0; }
  // wwe2k14 patch (Session 66, G1.0 resolve census, gpu_native_alias_census): counters only, nothing here changes what
  // is recorded or drawn. A resolve is dead (resolved over whole before anything sampled it), sampled, or stale
  // (neither within 2 frames); see AliasNoteResolve. The hooks cost one int compare when the census is off.
  bool alias_census_active() const { return alias_frames_left_ > 0; }
  void AliasNotePath(uint32_t path);                             // from the command processor's resolve wrappers
  void AliasNoteSampled(void* texture, bool cpu);                // a draw's texture binding / the memory path needed it
  void AliasNoteLoad(uint32_t first_page, uint32_t page_count);  // a texture load read these guest pages

  // Whether this draw should get its pixel shader translated for host render targets (Draw::pixel_shader): the
  // native renderer draws it and gpu_native_shading is 1. Asked after the rest of the draw is described.
  bool WantsPixelShader(const Draw& draw) const {
    if (truth_ && replacing_) return enabled_ && TruthReason(draw, false) == 0;  // Session 46: what truth claims
    return enabled_ && shading_ && (Wants(draw) || PostReason(draw, false) == kPostOk);
  }
  // Session 43: whether that translation should round its depth to 20e4 (a post draw that tests or writes depth).
  bool WantsFloat24Depth(const Draw& draw) const {
    // Session 46: not in a truth frame - one D32 image per depth surface (the 24-bit rounding happens where it is
    // stored for a resolve); a shader depth write would cost every scene draw its early depth test. (Runs 3028 / 3029:
    // the same pictures and resolve hashes with and without the rounding.)
    if (truth_ && replacing_) return false;
    return (draw.raster.depth_test || draw.raster.depth_write) && PostReason(draw, false) == kPostOk;
  }
  // Session 43: how that translation rounds its colour (Draw::color_quantize): only for post draws into 8-bit / 16-bit
  // post targets that do not blend (a blending draw is blended by the host into the target as it is).
  uint32_t WantsColorQuantize(const Draw& draw) const {
    if (truth_ && replacing_) {  // Session 46: the truth image's class, when the draw writes colour without blending
      if (!(draw.color_write_mask & 0xF) || PostNeedsDst(draw) || TruthReason(draw, false) != 0) return 0;
      return TruthClass(draw);
    }
    if (PostTargetClass(draw) == 0 || PostNeedsDst(draw) || PostReason(draw, false) != kPostOk) return 0;
    return PostTargetClass(draw);
  }
  // Session 46: a truth image's colour class for a draw's colour format (1 k_8_8_8_8, 2 k_16_16 / k_16_16_16_16, 0
  // none).
  static uint32_t TruthClass(const Draw& draw) {
    const uint32_t format = (draw.color_info >> 16) & 0xF;
    return format == 0 ? 1u : (format == 4 || format == 5) ? 2u : 0u;
  }

  // gpu_native 2, latched per frame (gpu_native_verify_every makes some frames side-car): this frame the native
  // renderer replaces the scene draws it claims.
  bool replacing() const { return replacing_; }
  // Session 45: gpu_native_segment_log is listing this frame (the texture cache adds its loads to the list).
  bool segment_log_active() const { return segment_log_frames_ > 0 && replacing_; }
  // Session 80 (gpu_native_truth_drop_dead_copies): the texture was used since its last compute write (the command
  // processor's write preparations): an earlier copy into it was read, it stays.
  void TruthDeadCopyTouched(void* texture) {
    for (DeadCopyCandidate& c : dead_candidates_) {
      if (c.texture == texture) c.touched = true;
    }
  }
  // Session 80 (frame structure census): a line of the segment log from the command processor (the frame prefixed).
  void SegLogLine(const std::string& text) const {
    if (segment_log_frames_ > 0) REXGPU_INFO("[native][seg] frame {} {}", frame_, text);
  }
  // Session 45 (gpu_native_all, latched per frame): a replacing frame drawing everything it can natively; the command
  // processor then fills every clear rectangle the fast clear can prove, whatever its size.
  bool all_native() const { return replacing_ && all_native_; }
  // Session 45 (gpu_native_all_exclude, diagnostic): parts of gpu_native_all that can be left out one at a time, for
  // the cost breakdown A/Bs (routes made from short\s45_all_ab.txt).
  enum AllPart : uint32_t {
    kAllBlend = 1,           // blended post draws native (gpu_native_post_blend)
    kAllSceneMinDraws = 2,   // every scene segment native (no gpu_native_min_draws)
    kAllFastClears = 4,      // fast clears of any size, and depth-only ones without a pixel shader
    kAllSync = 8,            // gpu_native_post_sync (and with it the mirrors and the fused write-back)
    kAllMirrors = 16,        // gpu_native_clear_mirror
    kAllResolve = 32,        // gpu_native_resolve and what builds on it (texture fills, fused write-back)
    kAllMerge = 64,          // gpu_native_merge
    kAllAfterScene = 128,    // gpu_native_after_scene, gpu_native_scene_extras, staying in the scene target
  };
  bool all_native(uint32_t part) const { return all_native() && !(all_exclude_ & part); }
  // Session 46 (gpu_native_truth, latched per frame): a truth frame - our images are the picture and the EDRAM buffer
  // is neither read nor written (docs\native\NO_EDRAM_DESIGN.md in the wwe2k14-recomp repository).
  bool truth() const { return replacing_ && truth_; }
  // wwe2k14 patch (Session 72 part 4, gpu_native_truth_skip_restores): before a registered restore draw is prepared
  // (its textures requested): true when it would write back exactly what its target images hold - the textures copied
  // this frame from those images' origins, no draw claimed since, the images untouched and still owning the copied
  // tiles. The caller has checked the draw's state (a plain copy: RT0 colour whole, no blending, depth written always,
  // no stencil, 1x k_8_8_8_8 colour, 20e4 depth); it then leaves the draw out.
  bool TruthRestoreRedundantEarly(uint32_t surface_pitch, uint32_t color_base, uint32_t depth_base,
                                  uint32_t color_page, uint32_t depth_page);
  // wwe2k14 patch (Session 72 part 4, gpu_native_truth_skip_repeat_resolves): this resolve's copy is one a direct path
  // did this frame with every register the same, from an image no truth draw went into since and that still owns the
  // copied tiles, and nothing has copied into its destination since: the destination holds it already (not in a
  // readback window). The caller records no GPU work and no marks for the copy (its clear still runs).
  bool TruthResolveRedundant(const draw_util::ResolveInfo& info);
  // Session 54 (gpu_native_truth_draw_height, docs\native\RESOLUTION_S54.md in the wwe2k14-recomp repository): the
  // truth passes' draw size relative to the images' (1: the images' own; below 1: they draw into smaller "draw twins").
  // Decided once (the native pixel shaders are translated for it).
  float TruthDrawRatio();
  // Session 54: whether the truth images of this pitch (32bpp tiles, with the MSAA factor) draw into draw twins - only
  // surfaces at least gpu_native_truth_draw_min_pitch wide (the screen-wide ones, where the pixel cost is: the small
  // post-effect surfaces would cost more in syncs than they save).
  bool TruthTwinPitch(uint32_t pitch_tiles) const { return TruthTwins() && pitch_tiles >= truth_twin_min_pitch_; }
  uint32_t truth_twin_min_pitch_ = 16;
  // In a replacing frame, before the emulated draw call: whether the native renderer draws this draw instead
  // (sets draw.owned). The pipeline is made here, so a claimed draw cannot be lost later.
  bool Claim(Draw& draw);
  // Session 47 (CPU): the repeat of a truth draw claimed in this truth frame (the same state, other constants) is
  // claimed as its base was, without the claim's checks; anything else goes to Claim.
  bool ClaimTruthRepeat(Draw& draw);
  // Session 50 (gpu_native_lean_draws): a draw with its base's render state (a copy of the base's claimed draw with
  // its own constants, buffers and textures) is claimed as ClaimTruthRepeat claims a repeat; false (nothing changed)
  // where that would need the full claim - the caller then takes the full path.
  bool ClaimTruthLean(Draw& draw);
  // Session 47: the last truth draw was recorded as one draw command of one instance (a repeat may be added to it as
  // an instance, vulkan_repeat_draw_merge), and NoteTruthInstance counts a repeat merged so.
  bool TruthLastDrawRecorded() const { return truth_ && truth_last_recorded_; }
  void NoteTruthInstance();
  // Session 42: the last Claim declined only because our other target (the scene's or a post target, which share
  // EDRAM tiles) holds draws not in the EDRAM yet. The caller ends the render pass, calls BeforeEmulatedEdramAccess
  // (the write-back) and claims again.
  bool ClaimWantsWriteBack() const { return claim_retry_; }
  // In a replacing frame: native draws that are not in the EDRAM yet (batched, or drawn and not written back). The
  // caller ends the render pass (which records the batch) before BeforeEmulatedEdramAccess.
  bool HasUnwrittenNative() const {
    if (!replacing_) return false;
    if (truth_) return true;  // Session 46: a truth frame's images hold what the EDRAM does not (the write-out)
    if (!batch_.empty() || native_dirty_ || !post_batch_.empty() || post_dirty_) return true;
    // Session 45 (gpu_native_defer_writeback): pending colour (its write-back is recorded outside render passes).
    for (const PostTarget& t : post_targets_) {
      if (t.pending) return true;
    }
    return false;
  }
  // In a replacing frame, outside any render pass, before an emulated draw that is not ours: our picture is written
  // into the EDRAM, and our targets are reloaded from it before our next draw.
  // Session 45: draw (when known) - the emulated draw that follows (a fast clear's fill is told by its fill fields);
  // emulated_follows false: only a switch between the native renderer's own targets (ClaimWantsWriteBack).
  void BeforeEmulatedEdramAccess(const Draw* draw = nullptr, bool emulated_follows = true);
  // Session 45 (gpu_native_defer_writeback): a stock resolve is about to read the EDRAM.
  void BeforeStockResolve() {
    if (replacing_) WritePendingAll();
  }

  // Session 43 (stretch S0-pre, gpu_native_cover_stats): which resolves read only pixels a native post segment
  // provably wrote (a quad / rectangle covering the resolve's rectangle): the reach of a native resolve (DESIGN 3f).
  bool CoverStatsActive() const { return replacing_ && post_ && cover_stats_; }
  // Session 45 (gpu_native_post_sync): claimed quad / rectangle draws get their proven coverage (Draw::cover).
  bool WantsCover() const { return replacing_ && post_sync_; }
  // A claimed post draw writes every pixel of rect (guest pixels, x0 y0 x1 y1 exclusive).
  void NotePostCover(const int32_t rect[4]);
  // Before OnResolve: the resolve's source rectangle (guest pixels) in the colour surface it copies.
  void NoteResolveCover(uint32_t surface_info, uint32_t color_info, const int32_t rect[4]);

  // The census of one draw that reached the backend (any draw, gpu_native on or off).
  void CensusDraw(uint64_t vertex_shader_hash, uint64_t pixel_shader_hash, bool memexport);
  // Before the emulated draw call (the emulated render pass is open): the GPU timestamp of a scene segment's start.
  void OnDrawPre(const Draw& draw);
  // Right after the emulated draw call was recorded.
  void OnDraw(const Draw& draw);
  // Session 45 (the frame census): a repeat of an emulated draw, drawn by the emulated renderer's repeat path (never
  // handed to OnDraw).
  void NoteEmulatedRepeat(const Draw& draw);
  // The emulated render pass has just ended: the native draws batched since are recorded now, while the memory
  // they read is exactly what the emulated draws read. occlusion_query_active: a guest occlusion query is counting
  // samples (our draws must not add to it: the batch is dropped). Returns whether commands were recorded (the
  // command processor's cached pipeline, descriptor and dynamic state are stale then).
  bool Flush(bool occlusion_query_active);
  // A resolve is about to be recorded (the render pass has ended). The scene's first colour resolve and its depth
  // resolve run the parity checks against the EDRAM, which holds the frame's emulated scene at these points.
  // copy_source: the render target being copied, 0-3 colour, 4 depth (RB_COPY_CONTROL copy_src_select); clears: the
  // resolve also clears EDRAM. In a replacing frame: our picture is written into the EDRAM first (and reloaded after
  // a clear); no parity checks.
  void OnResolve(uint32_t copy_source, uint32_t surface_info, uint32_t color_info, uint32_t depth_info,
                 VkBuffer edram_buffer, uint32_t resolution_scale_x, uint32_t resolution_scale_y, bool clears);
  // Session 45 (gpu_native_resolve, step 2 N2c): inside the render target cache's resolve, in place of its fast 32bpp
  // copy shader (1x, unscaled): serves the copy from the post target that drew the surface - the pixels its segment
  // drew from the target, the others from the EDRAM - with the stock shader's swap, tiling and endian swap. Returns
  // false to leave the stock dispatch to the caller. The caller has set up the destination (dest_set, the stock
  // constants with the binding's offset applied) and the EDRAM's compute-read usage.
  bool ResolveFromTarget(const draw_util::ResolveInfo& info, const draw_util::ResolveCopyShaderConstants& constants,
                         VkDescriptorSet dest_set, uint32_t group_count_x, uint32_t group_count_y);
  // Session 45 (N3b, gpu_native_resolve_texture_stock): a fast 32bpp 1x colour resolve no post target serves still
  // fills the texture it feeds - the same shader with every pixel from the EDRAM (the stock copy's bytes), in place
  // of the stock dispatch, when such a texture exists and qualifies. False: the caller dispatches the stock copy.
  bool ResolveTextureOnly(const draw_util::ResolveInfo& info, const draw_util::ResolveCopyShaderConstants& constants,
                          VkDescriptorSet dest_set, uint32_t group_count_x, uint32_t group_count_y);
  // Session 46 (truth frames): inside the render target cache's resolve, before its copy: the EDRAM tiles the copy
  // reads are written from their owners (our images; the EDRAM's own for tiles nobody drew) into the staging buffer,
  // in the EDRAM's layout, and the returned storage-buffer set over it is what the copy shader reads instead of the
  // EDRAM. VK_NULL_HANDLE: not a truth frame (or a failure, after which the frame went exact): the EDRAM as always.
  VkDescriptorSet TruthResolveSource(const draw_util::ResolveInfo& info);
  // Session 61 (DLSS / DLAA in truth frames): a surface's tiles (rows x pitch from base) staged in the EDRAM layout,
  // as for a resolve; the staging buffer's storage-buffer set (the EDRAM set's layout), or VK_NULL_HANDLE outside
  // truth frames.
  VkDescriptorSet TruthStageSurface(uint32_t base_tiles, uint32_t pitch_tiles, uint32_t rows);
  VkDescriptorSet TruthStageSpan(uint32_t base, uint32_t row_length, uint32_t rows, uint32_t pitch);
  // Session 46 (M3): in a truth frame, a fast 32bpp 1x colour copy of a surface one RGBA8 image holds whole is done
  // straight from that image by the transcribed resolve shader (every pixel from the image; the texture it feeds filled
  // too, as N3a), in place of the staging store and the stock copy. False: TruthResolveSource's way.
  bool TruthResolveDirect(const draw_util::ResolveInfo& info, const draw_util::ResolveCopyShaderConstants& constants,
                          VkDescriptorSet dest_set, uint32_t group_count_x, uint32_t group_count_y);
  // Session 47 (M6): after a 1x resolve's copy in a truth frame (the guest memory written as before), the texture it
  // feeds is filled from the image holding the copied tiles - a copy of the same bits (k_8_8_8_8 from an RGBA8 image,
  // k_16_16_16_16 from an RGBA16 one; k_24_8_FLOAT depth through a buffer into its R32_SFLOAT texture) - and is up to
  // date after the resolve (AfterResolveMarked): no reload. Only where the copy changes nothing but the place (no exp
  // bias, swap, format change, MSAA, array; the texture's endianness the copy's).
  void TruthFillTexture(const draw_util::ResolveInfo& info);
  // Session 49 (gpu_ram_stats, the RAM ties): GPU timestamps around a truth resolve's write-back into guest memory
  // (kind 6: TruthResolveSource's staging stores + the copy shader, or TruthResolveDirect's dispatch - which also fills
  // the texture) and around its texture fill (kind 7); nothing without gpu_ram_stats. And the write-back's path / bytes.
  void TruthTimer(uint32_t kind, bool begin);
  // Session 80 (frame structure): with gpu_ram_stats, a timestamp as a submission's first command (the command
  // processor's BeginSubmission), so the pass log can tell a gap the GPU spent waiting for the next submission.
  void TruthStampSubmission();
  uint64_t stamp_calls_ = 0, stamp_made_ = 0, stamp_skipped_[5] = {};  // (Session 87, step 3 diagnosis)
  void TruthNoteWriteBack(bool direct, uint32_t bytes);
  // Session 49 (gpu_native_truth_skip_writeback, texture-only resolves): a truth resolve whose texture fill holds every
  // texel the copy writes (a single-level k_8_8_8_8 texture with an R32_UINT storage view, the copy wholly inside it)
  // does not write guest memory: the GPU copy's pages are marked skipped in the shared memory and remembered here
  // (a WbRecord: the texture, pinned, and the copy's destination constants). Anything that requests those pages
  // first gets them written from the texture (TruthMaterialize: the resolve shader's destination half, reading the
  // texture), as do records older than gpu_native_truth_skip_writeback_max_age frames and every record when skipping
  // stops. A later resolve whose rectangle covers a record's (same base, pitch) supersedes it.
  //  - TruthBeforeResolveWrite: before a resolve's destination request (supersede / materialise what it overlaps).
  //  - TruthSetDest: the resolve's destination buffer range (for the record), before the copy.
  //  - TruthFillOnly: a staged truth resolve's texture fill done first; true = the fill holds the whole copy in a
  //    readable texture (32 / 64 bits a texel, the texture cache's raw-bits storage view): skipped (no staging stores,
  //    no copy into memory). The materialise shader is the direct texture load's inverse.
  //  - TruthTakeSkipped: after MarkRangeAsResolved - whether this resolve skipped its write-back (then its pages are
  //    marked skipped).
  void TruthBeforeResolveWrite(const draw_util::ResolveInfo& info);
  // Before the resolve writes memory (or skips): records it replaces go, records whose texture it fills too stay, the
  // others it overlaps go into memory first. Once a resolve; filled_texture = the texture this resolve fills, if any.
  void TruthResolveOverlaps(const draw_util::ResolveInfo& info, void* filled_texture,
                            const int32_t* filled_rect = nullptr,  // (x0 y0 x1 y1 in texture texels, level 0)
                            int32_t filled_layer = -1);           // (Session 79: the cube face filled, -1 2D)
  // Session 56 (gpu_native_truth_skip_writeback_scaled): the scaled texture-only records whose guest memory a scaled
  // texture load reads go into the scaled resolve buffer first.
  void TruthMaterializeScaled(uint32_t start, uint32_t length);
  void TruthSetDest(uint64_t offset, uint32_t range) {
    truth_dest_offset_ = offset;
    truth_dest_range_ = range;
  }
  bool TruthFillOnly(const draw_util::ResolveInfo& info, const draw_util::ResolveCopyShaderConstants& constants);
  // Session 52 (gpu_native_truth_direct_resolve_scaled_full): at a resolution scale, a full k_16_16 / k_16_16_16_16
  // colour copy of a surface one truth image holds, straight from the image into the scaled resolve buffer.
  bool TruthResolveScaledFull(const draw_util::ResolveInfo& info, VkDescriptorSet dest_set);
  bool EnsureResolveScaledFullPipeline();
  VkPipelineLayout resolve_scaled_full_layout_ = VK_NULL_HANDLE;
  VkPipeline resolve_scaled_full_pipeline_ = VK_NULL_HANDLE;
  // Session 59 (gpu_native_truth_direct_resolve_scaled_full_texture): the same copy also writing the scaled texture
  // ([0] k_16_16 through r32ui, [1] k_16_16_16_16 through rgba16ui), on a layout with the texture's set 2.
  VkPipelineLayout resolve_scaled_full_tex_layout_ = VK_NULL_HANDLE;
  VkPipeline resolve_scaled_full_tex_pipelines_[2] = {};
  bool resolve_scaled_full_tex_failed_ = false;
  bool EnsureResolveScaledFullTexPipelines();
  // Session 59 (gpu_native_truth_direct_resolve_full_1x): full k_8 / k_4_4_4_4 copies at 1x straight from an RGBA8 truth
  // image into memory (on resolve_scaled_full_layout_).
  VkPipeline resolve_full_1x_pipeline_ = VK_NULL_HANDLE;
  VkPipeline resolve_depth_1x_pipeline_ = VK_NULL_HANDLE;  // Session 59 (FULL1X_DEPTH)
  bool resolve_depth_1x_failed_ = false;
  VkPipeline resolve_depth_scaled_pipeline_ = VK_NULL_HANDLE;  // Session 59 (FULL_DEPTH, at a resolution scale)
  bool resolve_depth_scaled_failed_ = false;
  VkPipeline resolve_full_1x_tex_pipelines_[2] = {};  // Session 59: [0] k_8 (r8ui), [1] k_4_4_4_4 (r16ui)
  bool resolve_full_1x_tex_failed_ = false;
  bool resolve_full_1x_failed_ = false;
 public:
  bool TruthResolveFull1x(const draw_util::ResolveInfo& info, uint32_t dest_base, uint32_t dest_coordinate_info,
                          uint32_t dest_info, VkDescriptorSet dest_set);
  // Session 59 (gpu_native_truth_direct_resolve_depth): from TruthResolveDirect for a 1x depth copy.
  bool TruthResolveDepth1x(const draw_util::ResolveInfo& info, uint32_t dest_base, uint32_t dest_coordinate_info,
                           uint32_t dest_info, VkDescriptorSet dest_set);
  bool resolve_scaled_full_failed_ = false;
  void TruthAfterResolveMarked(const draw_util::ResolveInfo& info);
  void TruthMaterialize(uint32_t start, uint32_t length, uint32_t reason);
  void TruthMaterializeAll(uint32_t reason);
  void TruthDropRecords();  // (the caches are cleared: textures and the GPU copy go)
  // Session 46 (M5): a truth draw is recorded at once, at its place in the command stream, into our own render pass
  // (no emulated render pass, no render target cache update); the pass stays open across the following truth draws
  // into the same images and is ended by the command processor's EndRenderPass (anything else it records first).
  // (Session 80, owned passes: or an owned pass's render pass - EndTruthPass ends it too.)
  bool TruthPassOpen() const { return truth_pass_open_ || owned_pending_.open; }
  // Whether the last truth draw also bound another layout's sets (its loads): the caller forgets all its bindings.
  bool TakeTruthForeignBind() {
    const bool bound = truth_foreign_bind_;
    truth_foreign_bind_ = false;
    return bound;
  }
  void EndTruthPass();
  // wwe2k14 patch (Session 80, OWN_RENDERER_PLAN.md, gpu_native_own_crowd; own_crowd.cpp): our own draw of a crowd
  // run's other members, right after its first member's truth draw (same pass, that draw's pipeline state and index
  // buffer): the title's program, our mesh copy, the members' matrices and the scene block. False (nothing recorded)
  // when the state is not that or an object could not be made - the caller then draws the run as before.
  struct OwnCrowdInput {
    uint64_t vertex_shader_hash = 0;  // the guest crowd vertex shader (its first member's truth draw just recorded)
    const uint32_t* vs_words = nullptr;
    size_t vs_count = 0;
    const uint32_t* vs_vi_words = nullptr;  // Session 95 (P7, gpu_native_crowd_input): the attribute form of the vertex shader (nullptr: none registered)
    size_t vs_vi_count = 0;
    const uint32_t* fs_words = nullptr;
    size_t fs_count = 0;
    const uint8_t* mesh_guest = nullptr;  // the guest vertex buffer (7 big-endian dwords a vertex; 4 for a card mesh)
    uint32_t mesh_vertices = 0;
    bool card_mesh = false;               // Session 87: the flat cards' 4-dword vertices (copied byte-swapped, not widened)
    const uint32_t* members = nullptr;   // 12 dwords a member (VS c76-c78 as written to the registers)
    uint32_t member_count = 0;
    float scene[112] = {};               // src\scene\own_crowd_vs.glsl's CrowdScene (24 vec4, std140; the last: the alpha test; Session 95: vec4 24-27 = view 1's view_proj, only the multiview variants read them)
    VkImageView views[3] = {};           // bindings 1-3: the vertex lighting map, the colour, the entrances' map
    VkSampler samplers[3] = {};
  };
  bool RecordOwnCrowd(const OwnCrowdInput& in);
  // Session 95 (P7, gpu_native_crowd_input): what the plan decided about a held first member's indices (BuildVertexInputPlan), for the run that follows it: from the index pool at `offset` in host order,
  // the base as vertexOffset. `why` = a ViRefusal when not `ok`. Used once: RecordOwnCrowd takes it. Every other planned draw clears it (OwnCrowdClearIndex), so a run own crowd declined before reaching
  // RecordOwnCrowd leaves nothing for a later run whose first member was not held.
  struct OwnCrowdViIndex {
    bool noted = false, ok = false;
    uint32_t why = 0;
    VkDeviceSize offset = 0;
    int32_t vertex_offset = 0;
    uint32_t last_vertex = 0;     // the highest vertex the indices reach, with the base
    bool clear_restart = false;   // gpu_native_index_no_restart: the run's pipeline has no primitive restart (the plan refused reset values)
  };
  void OwnCrowdNoteIndex(const OwnCrowdViIndex& index) { own_crowd_vi_index_ = index; }
  void OwnCrowdClearIndex() { own_crowd_vi_index_.noted = false; }
  void OwnCrowdOnFrameEnd(bool from_hook = false);  // (Session 93: from_hook = SceneCrowdFrameEnd's call; once seen, OnFrameEnd's own call is ignored)
  // (Session 93) whether the lean call is on (the cvar as the objects were made, or as it stands before they exist); for the program lookup.
  bool OwnCrowdLeanWanted() const;
  void OwnCrowdNoteProgramHit() { ++own_crowd_lean_stats_.program_hits; }
  void OwnCrowdOnSubmissionEnd();  // wwe2k14 patch (Session 93, gpu_native_own_crowd_lean): flushes the ring ranges written this submission
  // Session 95 (out\s95\android_structural.md): the index pool and the attribute draws' state (both made on first use). VertexInputFormOf is the one rule the typed draw and the store share: whether
  // this plain translation (`vertex_ext` = the plain draw's ext) has an attribute form, and its strides; a refusal is named in form->refusal. Command thread.
  IndexPool& Mirror();
  VertexInputState& VertexInput();
  bool VertexInputFormOf(VulkanShader& vertex_shader, uint64_t vertex_modification, uint32_t vertex_ext, ViForm* form);
  // Session 95 (gpu_native_skin_once): the vertex pool the skinning job reads (made on first use, only with the switch on), and the pipeline of a draw going back to plain (*pending as GetPipeline's).
  VertexMirror& SkinMirror();
  SkinOnce& Skin();
  bool SkinPending() const;  // this submission has skin jobs to record
  void SkinRecord(DeferredCommandBuffer& hoist, uint64_t submission);
  VkPipeline MirrorPipeline(const Draw& draw, bool* pending) { return GetPipeline(draw, pending); }
  // wwe2k14 patch (Session 80, OWN_RENDERER_PLAN.md "The shadows", gpu_native_own_steps; owned_steps.cpp): owned
  // steps. A listed pixel shader's draw (colour only into an RGBA8 surface, unblended, no depth or stencil, its proven
  // coverage holding its rectangle from the surface's origin) is recorded into an exact-size image of ours in a render
  // pass of its own - not into the truth images: the EDRAM tiles keep what they had, as gpu_shadow_half_rate leaves
  // them - and the copy of that rectangle into a texture page that follows is left out (OwnedTakeResolve, from the
  // render target cache's Resolve): every later sampler of the page is bound to our image (OwnedView, from
  // VulkanTextureCache::RequestTextures). OwnedTakeDraw runs first in TruthRecordDraw: true = recorded (nothing else
  // to do for the draw); false = the truth draw as before (an owned output it overwrites in its surface is lost).
  // (Session 80, owned passes: the list also names vertex shaders, `v<hash>` - the shadow casters, drawn with the generic
  // flat-colour pixel shader the clears use. A pass of several listed draws after the clears of its truth pass - with
  // depth and stencil when the clears wrote them whole - goes into our image; the truth images' tiles it would have
  // written are then held as dirty: our image is copied into them before anything reads them (OwnedDirty*).)
  bool OwnedTakeDraw(const Draw& draw);
  bool OwnedTakeDrawImpl(const Draw& draw);
  // wwe2k14 patch (Session 87, D3D layer F2; d3d_copy_records.cpp): the game's Clear as a record (rex/graphics/d3d_copy.h,
  // d3d_clear_sem) taken by the owned frame as a held clear - no draw, no register decode. False: not taken (the owned
  // frame is off, or it refused the clear); the caller counts the clear lost. D3dNoteClearLost: a record clear the truth
  // path was asked to draw (it has no pipeline) - counted and logged.
  bool D3dClear(const uint32_t* sem, uint32_t count);
  void D3dNoteClearLost(const char* where);
  // Whether a draw of this pixel shader may be owned (the command processor then works out its proven coverage).
  bool OwnedWantsCover(uint64_t pixel_shader_hash) const {
    return truth_ && replacing_ && !owned_failed_ && OwnedWants(pixel_shader_hash);
  }
  // (At the submission's end, before its last render pass ends: an owned pass still waiting for its copy is replayed
  // into the truth images - its draws' bindings are this submission's.)
  void OwnedOnSubmissionEnd();
  // wwe2k14 patch (Session 95, gpu_owned_graph; owned_graph.cpp): at a submission's end, after its barriers are written and before the stream is
  // handed on: the frame graph's walk over the recorded stream (a no-op while the switch is 0).
  void OwnedGraphFinalize();
  // (copy_known: the copy has a copy shader - only such a copy is taken; any copy unmaps the pages it writes.)
  bool OwnedTakeResolve(const draw_util::ResolveInfo& info, bool copy_known);
  // Session 90 (Q): a copy OwnedTakeResolve served while the CPU is to read its bytes (a read-back window) arms a write-back: the render
  // target cache then runs OwnedWriteBackRun in its destination block (instead of the game's copy from the EDRAM / the truth images), which
  // writes the copy's bytes from our image into the destination (the shared memory, or the scaled resolve buffer) so the read-back gets
  // them. Armed only for the resolve just taken (cleared at the next OwnedTakeResolve).
  bool OwnedWriteBackArmed() const { return owned_wb_.armed; }
  bool OwnedOnlyActive() const { return OwnedOnly(); }  // (Session 92, M3a: the owned-only mode of this frame)
  // (Session 95, item P) UpdateBindings has just bound the shared memory's set into the stream, behind OwnedRecord's dedupe: what it remembers for that set says nothing now,
  // so the ring set of an inline draw is always bound. (The draw after it compares the ring set with its own and binds that.)
  void OwnedForgetSharedMemorySet() { owned_bound_.sets[SpirvShaderTranslator::kDescriptorSetSharedMemoryAndEdram] = VK_NULL_HANDLE; }
  bool OwnedResolveReady() const { return OwnedOnly() && truth_ && replacing_; }  // (M3a review P1: a frame that can serve a resolve record - truth and replacing)
  bool OwnedWriteBackRun(const draw_util::ResolveInfo& info, const draw_util::ResolveCopyShaderConstants& constants,
                         VkDescriptorSet dest_set, uint32_t group_count_x, uint32_t group_count_y);
  // (endian: the fetch's byte order; scaled_ok: the texture's scaled-resolve flag is the draw resolution's.)
  // (Session 82, P2: has_mips - the binding's texture has mip levels: only a feedback page (OwnedFeedback) serves it.)
  void OwnedCubeLearned(uint32_t base_page, uint32_t width, uint32_t height, uint32_t faces, uint32_t stride_bytes);
  bool OwnedTakeDerivedResolved() {
    const bool r = owned_derived_resolved_;
    owned_derived_resolved_ = false;
    return r;
  }
  bool OwnedPageFree(uint32_t base_page, uint32_t width, uint32_t height, uint32_t format);  // (Session 95, r6tex: OwnedView's side effects, and whether no page of the list can answer)
  bool OwnedReadAgain(uint32_t memo_index, uint32_t base_page, VkImageView view);  // (Session 95, r6tex: a binding that kept its page and its view is read again: false = ask OwnedView)
  uint32_t OwnedViewMemoLast() const { return owned_view_memo_last_; }
  VkImageView OwnedView(uint32_t base_page, uint32_t width, uint32_t height, uint32_t format, uint32_t endian,
                        bool scaled_ok, const VkComponentMapping& mapping, bool is_signed = false,
                        bool servable = true, bool has_mips = false, bool is_cube = false);
  void OwnedOnFrameEnd();
  void OwnedDestroy();
  // wwe2k14 patch (Session 82, the owned match frame, gpu_native_owned_frame): the guest page the swap presents (the
  // front buffer; from IssueSwap): a copy into a page presented lately is left to the truth path (reason `present`)
  // until the swap reads our image (OWNED_FRAME_DESIGN.md section 0 item 2).
  void OwnedNoteFrontBuffer(uint32_t frontbuffer_ptr);
  // (IssueSwap, before the swap texture is requested: the page fetch 0 names. An owned page there that the swap hook
  // below cannot serve means this frame's presentation reads a texture whose copy was left out - the pass that made it
  // is refused from now on; logged. Session 82 integration: pages the hook serves are left alone.)
  void OwnedCheckSwapPage(uint32_t page);
  // wwe2k14 patch (Session 82, the owned match frame, P2; owned_post.cpp): the swap's source (fetch constant 0, from
  // VulkanTextureCache::RequestSwapTexture). Always noted (its page, size, format, byte order and scaled flag: the
  // copy that feeds a front buffer is left out only when the swap reads the page exactly as the copy writes it); an
  // owned page for it gives its image's view (the presenter samples it instead of the texture).
  VkImageView OwnedSwapView(uint32_t base_page, uint32_t width, uint32_t height, uint32_t format, uint32_t endian,
                            bool scaled, const VkComponentMapping& mapping, bool servable);
  // Session 45 (N3a, gpu_native_resolve_texture): after the resolve's MarkRangeAsResolved - the texture the resolve
  // also filled is up to date again.
  void AfterResolveMarked();
  // Session 45 (gpu_native_fuse_writeback): after the resolve (IssueCopy) - a write-back it was to do itself and did
  // not (no copy after all) goes into the EDRAM now.
  void AfterResolve() {
    if (replacing_ && !truth_ && fuse_writeback_ && !defer_writeback_) WritePendingAll();
  }
  // At the swap, after the frame's last draws and outside any render pass: statistics, the census, the scene's
  // render target for the next frame, settings.
  void OnFrameEnd(CommandProcessor::NativeCensus& census);
  // In the swap's presentation (its compute path): before its barriers are submitted (our picture becomes
  // readable), then after its gamma / FXAA passes wrote the guest output (ours goes on top, per gpu_native_view;
  // the guest output is a storage image in the GENERAL layout there).
  void OnSwapPrepare();
  void OnSwapCompose(VkImageView guest_output, uint32_t width, uint32_t height);
  // The command processor's shader / pipeline layout caches were cleared: our pipelines refer to their handles.
  void OnCachesCleared();
  // Session 61 (gpu_native_pipeline_storage): the native pipelines made in earlier runs, as records next to the
  // emulated path's pipeline storage (<shareable>\TITLE.native.fsi|fbo.vk.xnps); opened with the shader storage (after it
  // loaded the stored shaders), closed with it. The stored pipelines are made again in the background from the first
  // frames the native renderer is on (gpu_native_pipeline_prewarm), and every new one is added.
  void InitializeStorage(const std::filesystem::path& shareable_root, uint32_t title_id);
  void ShutdownStorage();

 private:
  struct PipelineKey {
    VkShaderModule vertex_shader;
    VkShaderModule pixel_shader;  // VK_NULL_HANDLE: our clay shading
    VkPipelineLayout layout;
    uint32_t state;  // topology, restart, cull, front face, depth test / write / compare, depth bias, depth clamp
                     // (Session 69: clip_disable, bit 14), colour mask
    uint32_t blend_control;
    uint64_t tint_hash;
    VkShaderModule geometry_shader;  // Session 42
    uint32_t variant;                // Session 42: 0 the scene's targets, 1 a post target
    uint32_t vi = 0;                 // Session 95: the vertex input of an attribute draw: one byte per binding, the stride in dwords (vertex_input_layout.h; 0: none, the shader reads the shared memory)
    bool operator==(const PipelineKey& other) const {
      return vertex_shader == other.vertex_shader && pixel_shader == other.pixel_shader && layout == other.layout &&
             state == other.state && blend_control == other.blend_control && tint_hash == other.tint_hash &&
             geometry_shader == other.geometry_shader && variant == other.variant && vi == other.vi;
    }
  };
  struct PipelineKeyHasher {
    size_t operator()(const PipelineKey& key) const;
  };
  // What the frame's main scene renders into: the render target that gets the most depth-writing draws.
  struct SceneKey {
    uint32_t surface_info = 0;
    uint32_t depth_info = 0;
    bool valid = false;
    bool Matches(const Draw& draw) const {
      return valid && draw.surface_info == surface_info && draw.depth_info == depth_info;
    }
  };
  // Why a scene-target draw is not drawn natively in a replacing frame (counted in side-car frames too: the census).
  enum Reason : uint32_t {
    kReasonScreenSpace,    // rectangle list, or no viewport transform
    kReasonNotTriangles,   // points, rectangles, quads (geometry-shader expansions), tessellation, polygon modes
    kReasonMemexport,
    kReasonAfterScene,     // after the scene's colour resolve (post-processing, the HUD)
    kReasonNoDepthWrite,   // gpu_native_draws 0
    kReasonStencil,
    kReasonOtherTargets,   // writes render targets 1-3
    kReasonColorFormat,    // render target 0 is not k_8_8_8_8, or not the scene's
    kReasonMsaa,
    kReasonOcclusion,      // a guest occlusion query is counting samples
    kReasonNoShader,       // no translation, no pipeline, no vertices
    kReasonSmallSegment,   // Session 40: its segment drew too few draws last frame to pay for a load and write-back
    kReasonCount,
    kOwnable = UINT32_MAX,
  };
  uint32_t NotOwnedReason(const Draw& draw) const;
  // Session 42 (gpu_native_post): why a draw after the scene's colour resolve is not drawn into a post target.
  enum PostReasonCode : uint32_t {
    kPostOk,
    kPostOff,           // not a replacing frame with gpu_native_post, or before the scene's colour resolve
    kPostPrimitive,     // not triangles nor a geometry-shader expansion (points, tessellation, polygon modes)
    kPostMsaa,
    kPostOtherTargets,  // writes render targets 1-3
    kPostNoColor,       // writes no colour (depth / stencil only, or nothing)
    kPostColorFormat,   // render target 0 is not k_8_8_8_8
    kPostDepth,         // tests or writes depth
    kPostStencil,
    kPostMemexport,
    kPostOcclusion,
    kPostNoShader,      // no translation, no pipeline, no vertices
    kPostSwitch,        // another post surface still has draws not in the EDRAM
    kPostBlend,         // blends or keeps channels (gpu_native_post_blend off: host blending is not bit-exact)
    kPostDepthFormat,   // Session 43: tests or writes a unorm24 depth surface (only 20e4 is exact so far)
    kPostDepthBias,     // Session 43: tests or writes depth with a polygon offset (the host's bias differs)
    kPostDepthClamp,    // Session 43 (review): tests or writes depth with clipping off (the emulated depth clamp)
    kPostSmall,         // Session 43: its scissor is smaller than gpu_native_post_min_pixels (a tiny pass)
    kPostReasonCount,
  };
  // check_shader: also require the pixel shader's host-render-target translation (false: before it is made).
  uint32_t PostReason(const Draw& draw, bool check_shader) const;
  // Session 45 (the frame census, [native][frame]): why a draw of a replacing frame was drawn by the emulated
  // renderer, in a few categories over both paths (scene and post).
  enum FrameEmulated : uint32_t {
    kEmuClear,      // a clear rectangle the emulated renderer filled (gpu_fast_clear)
    kEmuBlend,      // blends or keeps channels
    kEmuPrimitive,  // not triangles nor a geometry-shader expansion; screen-space / expanded draws into the scene
    kEmuDepth,      // depth format, polygon offset, depth clamp, depth tested in a post target (setting off)
    kEmuStencil,
    kEmuFormat,     // colour format, other render targets, no colour
    kEmuMsaa,
    kEmuSmall,      // a small scene segment (gpu_native_min_draws) or a tiny post pass (gpu_native_post_min_pixels)
    kEmuSwitch,     // the other native target held unwritten draws (scene / post, the same EDRAM tiles)
    kEmuNoShader,   // no translation or pipeline
    kEmuOther,      // memexport, occlusion query, after the scene, not a replacing frame, ...
    kEmuCount,
  };
  uint32_t EmulatedCategory(const Draw& draw) const;
  // Session 43: a post draw that blends or leaves some of its format's channels as they are needs the EDRAM's values
  // under it (k_16_16 has two channels).
  static bool PostNeedsDst(const Draw& draw) {
    const uint32_t full = ((draw.color_info >> 16) & 0xF) == 4 ? 0x3u : 0xFu;
    return draw.blend_control != 0x00010001 || (draw.color_write_mask & full) != full;
  }
  // Session 43: the post target's colour format class for a draw's surface: 0 RGBA32F (the shader's output kept),
  // with gpu_native_post_quantize 1 RGBA8_UNORM (k_8_8_8_8) / 2 RGBA16_SNORM (k_16_16, k_16_16_16_16), holding the
  // packed values exactly (2-4x less memory traffic).
  static constexpr uint32_t kPostTargetClasses = 3;
  uint32_t PostTargetClass(const Draw& draw) const {
    if (!post_quantize_) return 0;
    const uint32_t format = (draw.color_info >> 16) & 0xF;
    const uint32_t target_class = format == 0 ? 1u : (format == 4 || format == 5) ? 2u : 0u;
    return post_class_ok_[target_class] ? target_class : 0u;
  }
  bool ClaimPost(Draw& draw);
  bool SceneBatchCovers() const;  // Session 45
  bool FlushScene(bool occlusion_query_active);
  bool FlushPost();
  void PostWriteBack(bool defer_color = false);
  // A post target: colour (32-bit float: the pixel shader's output kept exactly, for the write-back's packing) and a
  // depth / stencil image whose stencil marks the pixels our draws covered since the segment began.
  struct PostTarget {
    uint32_t pitch_tiles = 0;        // the surface's pitch in EDRAM tiles
    uint32_t target_class = 0;       // Session 43: PostTargetClass (its colour format)
    uint32_t width = 0, height = 0;  // host pixels
    VkImage color = VK_NULL_HANDLE, ds = VK_NULL_HANDLE;
    VkDeviceMemory color_memory = VK_NULL_HANDLE, ds_memory = VK_NULL_HANDLE;
    VkImageView color_view = VK_NULL_HANDLE, ds_view = VK_NULL_HANDLE, stencil_view = VK_NULL_HANDLE;
    VkImageView depth_view = VK_NULL_HANDLE;  // Session 43: the write-back samples the depth
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkDescriptorSet writeback_set = VK_NULL_HANDLE;
    VkBuffer writeback_edram = VK_NULL_HANDLE;  // the EDRAM buffer writeback_set points at
    // Session 45 (gpu_native_post_sync): a pass has run (its images are in the attachment layouts); the EDRAM tiles
    // where the colour / depth image equals the EDRAM (once the open segment is written back), for the colour surface
    // (RB_COLOR_INFO) and depth surface (RB_DEPTH_INFO) named - a segment needing what is under its draws loads only
    // when they do not cover the draws.
    bool initialized = false;
    uint32_t sync_color_info = UINT32_MAX, sync_depth_info = UINT32_MAX;
    std::bitset<2048> sync_color, sync_depth;
    // Session 45 (gpu_native_defer_writeback): a segment's colour the EDRAM does not have yet (its coverage marks are
    // still in the stencil): the surface, its rectangle (host pixels) and proven part, its colour tiles.
    bool pending = false;
    uint32_t pending_color_info = 0, pending_surface_info = 0;
    int32_t pending_rect[4] = {}, pending_cov[4] = {};
    std::bitset<2048> pending_tiles;
    // Session 45 (clear mirrors): the last fast-clear fill over this target's surface since its last segment - drawn
    // into the target as a constant at its next segment's start instead of loading those pixels from the EDRAM.
    struct Mirror {
      bool valid = false;
      uint32_t info = 0;      // RB_COLOR_INFO / RB_DEPTH_INFO of the filled surface
      uint32_t src = 0;       // the EDRAM dword every pixel gets
      int32_t rect[4] = {};   // host pixels (x0 y0 x1 y1)
      std::bitset<2048> tiles;  // the tiles the fill wrote
    } mirror_color, mirror_depth;
  };
  // Session 45 (clear mirrors): a fill over these tiles (or anything else writing the EDRAM there) makes the mirrors
  // recorded before it out of date.
  void DropMirrors(const std::bitset<2048>& tiles);
  void RecordMirror(const Draw& fill);
  bool clear_mirrors_ = false;  // gpu_native_clear_mirror (or gpu_native_all), latched per frame
  uint32_t scene_color_info_last_ = UINT32_MAX;  // the scene's colour surface as last seen (kept across frames)
  // The scene's: the last fill of its colour and depth surfaces since the scene's targets last loaded.
  PostTarget::Mirror scene_mirror_color_, scene_mirror_depth_;
  // Session 45 (gpu_native_defer_writeback).
  void WritePending(int32_t index);
  void WritePendingAll(int32_t except = -1);
  void WritePendingOverlapping(const std::bitset<2048>& tiles, int32_t except);
  bool defer_writeback_ = false;  // gpu_native_defer_writeback, latched per frame
  bool fuse_writeback_ = false;   // gpu_native_fuse_writeback (or gpu_native_all), latched per frame
  // Session 45 (gpu_native_post_sync): the EDRAM tiles of a rectangle of guest samples (x0 y0 inclusive, x1 y1
  // exclusive) of a surface; 64bpp rows of tiles are twice the 32bpp pitch, 40 samples wide.
  static std::bitset<2048> EdramTiles(uint32_t base_tiles, uint32_t pitch_tiles, bool is_64bpp, uint32_t x0,
                                      uint32_t y0, uint32_t x1, uint32_t y1);
  // Session 45: the EDRAM changed under these tiles (not by the target `except`, or anyone with -1).
  void InvalidatePostSync(const std::bitset<2048>& tiles, int32_t except);
  void InvalidateAllPostSync();
  bool post_sync_ = false;  // gpu_native_post_sync (or gpu_native_all), latched per frame
  // Session 45: the open post segment loaded depth at its start; the host pixels its draws provably wrote (colour
  // without reading it, depth with the test always) - draws reading only there need no load.
  bool post_depth_loaded_ = false;
  int32_t post_cov_color_[4] = {}, post_cov_depth_[4] = {};
  // Session 45 (gpu_native_resolve): the post target whose segment OnResolve just wrote back (its coverage marks are
  // the pixels it drew), and that segment's RB_COLOR_INFO; -1 when none (consumed by ResolveFromTarget).
  int32_t resolve_target_ = -1;
  uint32_t resolve_color_info_ = 0;
  bool native_resolve_ = false;  // gpu_native_resolve (or gpu_native_all), latched per frame
  VkShaderModule resolve_shader_ = VK_NULL_HANDLE;
  VkPipelineLayout resolve_layout_ = VK_NULL_HANDLE;
  VkPipeline resolve_pipeline_ = VK_NULL_HANDLE;
  // Session 45 (N3a): the variant that also fills the texture; gpu_native_resolve_texture latched; the texture the
  // last resolve filled (for AfterResolveMarked).
  VkShaderModule resolve_tex_shader_ = VK_NULL_HANDLE;
  VkPipelineLayout resolve_tex_layout_ = VK_NULL_HANDLE;
  VkPipeline resolve_tex_pipeline_ = VK_NULL_HANDLE;
  // Session 52 (gpu_native_truth_direct_resolve_scaled): the direct copy at a resolution scale (resolve_layout_).
  VkPipeline resolve_scaled_pipeline_ = VK_NULL_HANDLE;
  VkPipeline resolve_scaled_tex_pipeline_ = VK_NULL_HANDLE;  // (also writes the scaled texture)
  bool resolve_scaled_failed_ = false;
  bool EnsureResolveScaledPipeline();
  // Session 52 (gpu_native_truth_batch_resolves): texture-only direct resolves recorded later, together (one barrier
  // for all their source images) - before anything else reads or writes a source image or samples a texture they
  // write, and at the frame's / submission's end.
  struct DeferredResolve {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSet sets[3];
    uint32_t pc[12];
    uint32_t group_count_x, group_count_y;
    VkImage image;
    int16_t index;
    void* texture;
    // The texture's barrier to its compute-write layout, recorded with the batch (texture_image VK_NULL_HANDLE: none).
    VkImage texture_image;
    VkPipelineStageFlags texture_src_stage;
    VkAccessFlags texture_src_access;
    VkImageLayout texture_old_layout;
    // Session 59 (gpu_native_truth_batch_fills): a texture fill's dispatch has 2 sets and 10 push constants.
    uint32_t set_count = 3;
    uint32_t pc_size = sizeof(uint32_t) * 12;
    // (A fill's texel rectangle in its texture, x0 y0 x1 y1: fills into one texture whose rectangles do not overlap
    // share a batch - the shadow atlases are filled several times a frame. A resolve's: none, x1 = 0.)
    int32_t texture_rect[4] = {0, 0, 0, 0};
    // wwe2k14 patch (Session 66, G1.1 gpu_native_truth_drop_dead_resolves): the texels this entry's dispatch writes in
    // its texture, x0 y0 x1 y1 (fills: texture_rect; unscaled texture-only resolves: the rectangle written; scaled
    // resolves: none, x1 = 0 - never dropped). A later resolve into the same texture covering it whole drops the entry.
    int32_t drop_rect[4] = {0, 0, 0, 0};
    // wwe2k14 patch (Session 79): the cube face the entry writes (-1: a 2D texture) - one texture, separate texels.
    int32_t layer = -1;
    // wwe2k14 patch (Session 79, gpu_native_truth_defer_memory_resolves): a resolve that also writes guest memory (the
    // GPU copy): its shared memory use (mem_length 0: none) and the extent marked skipped until the batch is flushed
    // (a reader of those pages flushes it first, as for texture-only records).
    uint32_t mem_start = 0, mem_length = 0;
    uint32_t extent_start = 0, extent_length = 0;
    // (Where it writes: the destination texture's base, pitch, format and the texel rectangle - copies into separate
    // rectangles of one surface write separate bytes though their extents interleave; base 0: unknown.)
    uint32_t mem_base = 0, mem_pitch = 0, mem_format = 0;
    int32_t mem_rect[4] = {};
    // wwe2k14 patch (Session 79, gpu_native_truth_defer_depth_resolves): `image` is a depth truth image (read in
    // DEPTH_STENCIL_READ_ONLY_OPTIMAL, the store set's layout).
    bool depth = false;
    // wwe2k14 patch (Session 80, gpu_native_truth_drop_dead_copies): a texture-only depth copy's stencil snapshot -
    // the fill into this texture it belongs to (dead with it).
    void* companion_of = nullptr;
    uint32_t page = 0;  // wwe2k14 patch (Session 80, diagnostic): the copy's destination page (the flush log)
    // wwe2k14 patch (Session 80, gpu_native_truth_carry_copies): kept by a flush - its texture pinned until it runs or
    // is dropped (the texture cache keeps it); kept past a frame's end.
    bool pinned = false;
    bool crossed = false;
    // (The submission it was recorded in and its destination descriptor's buffer range: sets[1] is a transient
    // descriptor of that submission, reused once it completes - a carried copy run in a later submission binds a fresh
    // one of the same range.)
    uint64_t submission = 0;
    uint64_t dest_offset = 0;
    uint32_t dest_range = 0;
    // wwe2k14 patch (Session 80, gpu_native_truth_serve_copies): a plain copy - all of the image, from its origin, into
    // all of the texture, the texels unchanged but for a red / blue swap (plain_swap; the served view swaps them back) -
    // and its size.
    bool plain = false;
    bool plain_swap = false;
    uint32_t plain_w = 0, plain_h = 0;
  };
  std::vector<DeferredResolve> truth_deferred_;
  // wwe2k14 patch (Session 80, gpu_native_truth_drop_dead_copies): texture-only copies recorded this submission
  // (TruthFlushDeferredSlow) that nothing has used since: their dispatches' places in the deferred command buffer.
  // A later deferred copy covering one in its texture first turns its commands into no-ops (NopRange) - it was read by
  // nothing. Any use of the texture in between (sampled, swapped, loaded: the texture cache's usage; a record written
  // from it) marks it touched; a new submission ends them all.
  struct DeadCopyCandidate {
    void* texture = nullptr;
    int32_t layer = -1;
    int32_t rect[4] = {};
    size_t begin[2] = {SIZE_MAX, SIZE_MAX}, end[2] = {SIZE_MAX, SIZE_MAX};  // the fill, and its stencil snapshot
    uint64_t submission = 0;
    bool touched = false;
    bool depth = false;
  };
  std::vector<DeadCopyCandidate> dead_candidates_;
  // wwe2k14 patch (Session 80, gpu_native_truth_drop_dead_passes): the open truth pass - nothing but colour pixels in its
  // draws' rectangles (no loads, no draw twins, no memexport, occlusion query, captured positions or own draw) - and the
  // last such pass that ended, until the next pass opens: its colour image, rectangle and commands' place.
  bool truth_pass_dead_ok_ = false;
  int32_t truth_pass_rect_[4] = {};
  struct DeadPassCandidate {
    bool valid = false;
    int16_t color = -1;
    VkImage image = VK_NULL_HANDLE;
    int32_t rect[4] = {};
    size_t begin = SIZE_MAX, end = SIZE_MAX;  // its vkCmdBeginRenderPass's arguments; the stream right after its end
    uint64_t submission = 0, frame = 0;
    uint32_t segment = UINT32_MAX;  // its pass log segment
  };
  DeadPassCandidate dead_pass_;
  // wwe2k14 patch (Session 79): whether a deferred resolve writes guest memory in [start, end) - with `info`, other
  // than into separate rectangles of the same surface.
  bool TruthDeferredWritesMemory(uint64_t start, uint64_t end, const draw_util::ResolveInfo* info = nullptr) const {
    for (const DeferredResolve& d : truth_deferred_) {
      if (!d.extent_length || d.extent_start >= end || start >= uint64_t(d.extent_start) + d.extent_length) continue;
      if (info && d.mem_base && !info->copy_dest_info.copy_dest_array && info->copy_dest_texture_base == d.mem_base &&
          uint32_t(info->copy_dest_coordinate_info.pitch_aligned_div_32) == d.mem_pitch &&
          uint32_t(info->copy_dest_info.copy_dest_format) == d.mem_format) {
        const int32_t x0 = int32_t(info->copy_dest_base_x + uint32_t(info->copy_dest_coordinate_info.offset_x_div_8) * 8);
        const int32_t y0 = int32_t(info->copy_dest_base_y + uint32_t(info->copy_dest_coordinate_info.offset_y_div_8) * 8);
        const int32_t x1 = x0 + int32_t(info->coordinate_info.width_div_8) * 8;
        const int32_t y1 = y0 + int32_t(info->height_div_8) * 8;
        if (x1 <= d.mem_rect[0] || d.mem_rect[2] <= x0 || y1 <= d.mem_rect[1] || d.mem_rect[3] <= y0) continue;
      }
      return true;
    }
    return false;
  }
  // Session 52 (gpu_native_truth_batch_rename): an image's GPU objects, set aside (busy: a deferred resolve reads it)
  // or ready for the next rename of an image of the same kind, format and size.
  struct TruthBundle {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE, sample_view = VK_NULL_HANDLE, stencil_view = VK_NULL_HANDLE;
    VkDescriptorSet store_sets[2] = {};
    VkDescriptorSet resolve_set = VK_NULL_HANDLE;
    VkBuffer store_set_edram = VK_NULL_HANDLE;
    uint32_t kind = 0, format = 0, width = 0, height = 0;
    bool busy = false;
    uint32_t idle_frames = 0;  // (review: frames ready but not taken - retired after 60)
    // Session 73: and at least gpu_native_truth_spare_keep_ms of wall time (a loading screen runs a thousand frames a
    // second: 60 frames emptied the pool on every load, and every scene start made it again).
    std::chrono::steady_clock::time_point idle_since{};
  };
  std::vector<TruthBundle> truth_spares_;
  bool truth_renaming_ = false;
  // (Session 56: `depth_ok` - a depth image too, gpu_native_truth_rename_depth.)
  bool TruthRenameForDeferred(int16_t color, const int32_t rect[4], bool depth_ok = false);
  void TruthRetireBundle(TruthBundle& b);
  std::vector<VkImageMemoryBarrier> truth_deferred_barriers_;

 public:
  // wwe2k14 patch (Session 66, G1.1): `reason` (kDeferFlush*) only counts why a non-empty batch was flushed.
  enum DeferFlushReason : uint32_t {
    kDeferFlushOther = 0,
    kDeferFlushSameTexture,   // a resolve into a texture an entry writes that the entry does not lie inside of
    kDeferFlushNotBatched,    // a resolve that is not texture-only / batching off
    kDeferFlushOutside,       // a resolve reaching outside a renamed image's rectangle
    kDeferFlushPassIntoSource,  // a pass draws into an image an entry reads (not renamed)
    kDeferFlushSampled,       // a texture an entry writes is sampled / loaded
    kDeferFlushFrameEnd,
    kDeferFlushSubmissionEnd,
    kDeferFlushMaterialize,   // a record is written into guest memory from its texture
    kDeferFlushStores,        // an image is stored / retired / grown / staged
    kDeferFlushOtherResolve,  // another kind of resolve (full, scaled, depth) or a texture fill that is not deferred
    kDeferFlushFillSameTexture,
    kDeferFlushFallback,      // fallback / write-out of the truth frame
    kDeferFlushReasons
  };
  // wwe2k14 patch (Session 80, gpu_native_truth_carry_copies): what a flush is about. Every entry runs (as before)
  // except a carried copy (TruthCarriable) the flush does not concern: one writing another texture than `texture`,
  // reading none of `images`, with `all` false - it stays in the batch (also past the frame's end), until something
  // uses its texture or writes its source, or a later copy covering it drops it.
  struct DeferConcern {
    bool all = true;
    const void* texture = nullptr;
    VkImage images[8] = {};
  };
  void TruthFlushDeferred(uint32_t reason = kDeferFlushOther) {
    if (!truth_deferred_.empty()) {
      ++truth_counts_.defer_flushes[reason < kDeferFlushReasons ? reason : kDeferFlushOther];
      TruthFlushDeferredSlow(reason, DeferConcern());
    }
  }
  void TruthFlushDeferredFor(uint32_t reason, const DeferConcern& concern) {
    if (concern.all) return TruthFlushDeferred(reason);
    // (Also with the batch empty when a renamed image waits for its other pixels: a carried copy dropped by a later
    // copy can leave one.)
    bool renamed = false;
    if (truth_deferred_.empty()) {
      for (const TruthImage& t : truth_images_) renamed = renamed || t.rename_source != VK_NULL_HANDLE;
    }
    if (!truth_deferred_.empty() || renamed) TruthFlushDeferredSlow(reason, concern);
  }
  void TruthFlushDeferredTexture(uint32_t reason, const void* texture) {
    DeferConcern c;
    c.all = texture == nullptr;
    c.texture = texture;
    TruthFlushDeferredFor(reason, c);
  }
  // (Session 80, gpu_native_truth_carry_copies: a quad / rectangle draw's proven coverage is wanted - a full-screen
  // draw into a carried copy's source renames the image instead of running the copy.)
  bool TruthWantsCover() const;
  // wwe2k14 patch (Session 80, gpu_native_truth_bounds_vs): a draw of this vertex shader gets its vertices' box in a
  // truth frame (the list parsed when the cvar changes); TruthNoteBounds counts the outcome (0 worked out, 1 from the
  // cache, 2 refused) and the CPU time.
  bool TruthWantsBounds(uint64_t vertex_shader_hash);
  void TruthNoteBounds(uint32_t outcome, double ms);
  // wwe2k14 patch (Session 80, gpu_native_truth_serve_copies): a view of the image a pending plain copy into `texture`
  // reads, for a draw sampling the texture in its place (the texels are the same) - VK_NULL_HANDLE when it may not:
  // same size, the view's format the image's; an image renamed away, or a slot's image that is none of the draw's
  // targets, or (the draw renders into the very image it samples: the match's combine) the draw's colour target, then
  // renamed first (TruthServeRename). The image is made readable before the draw's pass and put back after it.
  struct ServeTargets {  // (the draw's registers: RB_SURFACE_INFO, RB_COLOR_INFO 0-3, the normalized colour mask)
    uint32_t surface_info = 0, color_info[4] = {}, color_mask = 0;
  };
  VkImageView TruthServeView(const void* texture, uint32_t width, uint32_t height, VkFormat format,
                             const VkComponentMapping& components, const ServeTargets& targets);
  // (A draw served bindings but is not drawn as a truth draw after all: its bindings take the textures again -
  // VulkanCommandProcessor::IssueDraw; the images it would have sampled are not made readable.)
  void TruthServeDropped();
  // (Around a draw's texture request: what an earlier draw served and never drew is dropped; flushes inside it are the
  // draw's own - review.)
  void TruthServeBegin() {
    OwnedGraphReadsClose();  // (Session 95, round 7: a read asked for before this request and never recorded is not this draw's)
    OwnedHeldBeforeTextures();  // (s82 hold fix: held clears recorded before the next draw's texture request)
    serve_pending_.clear();
    serve_requesting_ = true;
  }
  void TruthServeEnd() { serve_requesting_ = false; }

 private:
  void TruthFlushDeferredSlow(uint32_t reason, const DeferConcern& concern);
  void TruthKeepCarried(uint32_t reason, std::vector<DeferredResolve>& kept);
  // (Session 80, gpu_native_truth_carry_copies: per texture, the frame a carried copy of it was last run because the
  // texture was used (sampled / materialised; a copy dropped unread erases it). Within kCarryUsedFrames of that, its
  // copies hold no spare image and do not cross the frame's end - a carried copy keeps its source image, and a rename
  // of that image then needs another (runs 5473 / 5475 filled the pool of spare images); a use in a menu or a
  // transition lapses.)
  static constexpr uint64_t kCarryUsedFrames = 120;
  std::unordered_map<const void*, uint64_t> carry_used_;
  std::string truth_flush_why_;  // (diagnostic: the flush log)
  // (Session 80, gpu_native_truth_serve_copies: the views made, retired with their image; the images the next truth
  // draw samples in place of a copy.)
  struct ServeView {
    VkImage image;
    VkFormat format;
    VkComponentMapping components;
    VkImageView view;
  };
  std::vector<ServeView> serve_views_;
  std::vector<VkImage> serve_pending_;
  // wwe2k14 patch (Session 80, gpu_native_truth_serve_ahead): the texture whose serve ended an open pass (its image
  // not readable inside it), and that pass's colour slot - the next time that slot's pass opens with the texture's
  // carried copy pending, the copy's source image is made readable before the pass begins (the pass known by its
  // colour slot and the vertex shader of the draw that opens it: the slot's other passes are left alone).
  struct ServeAhead {
    const void* texture = nullptr;
    int16_t color = -1;
    uint64_t first_vs = 0;
  } serve_ahead_;
  void TruthRetireServeViews(VkImage image);
  // (A draw that samples its own colour target's image in place of a copy: that slot was renamed when the binding was
  // served, with an empty rectangle - every pixel to come from `source` at the next flush, exact whatever follows; the
  // draw widens it to the whole image when it is recorded and proven to write every pixel, reading none - then the
  // flush copies nothing.)
  struct ServeRename {
    bool valid = false;
    int16_t slot = -1;
    VkImage source = VK_NULL_HANDLE;
  } serve_rename_;
  bool serve_requesting_ = false;  // (inside a draw's texture request: TruthServeBegin / TruthServeEnd)
  // (Copies of at least 256 x 256 texels: a small copy costs little, and carrying it holds an image all the same - in
  // the match the 64 x 64 ones took every holder before the scene copy, run 5477.)
  bool TruthCarriable(const DeferredResolve& d) const {
    return d.texture && !d.mem_length && d.drop_rect[2] > d.drop_rect[0] && d.drop_rect[3] > d.drop_rect[1] &&
           int64_t(d.drop_rect[2] - d.drop_rect[0]) * (d.drop_rect[3] - d.drop_rect[1]) >= 65536 && !d.depth &&
           d.set_count == 3 && !d.companion_of && d.layer < 0;
  }
  bool TruthDeferredSource(int16_t index) const {
    if (index < 0 || size_t(index) >= truth_images_.size()) return false;
    const VkImage image = truth_images_[index].image;  // (the image, not the slot: a rename gives the slot a new one)
    for (const DeferredResolve& d : truth_deferred_) {
      if (d.image == image) return true;
    }
    return false;
  }
  bool resolve_texture_ = false;
  bool resolve_texture_stock_ = false;  // Session 45 (N3b): gpu_native_resolve_texture_stock, latched per frame
  void* texture_written_ = nullptr;
  bool resolve_objects_failed_ = false;
  bool EnsureResolveObjects();
  bool EnsurePostObjects();
  int32_t EnsurePostTarget(uint32_t pitch_tiles, uint32_t target_class, uint32_t width, uint32_t height);
  void RetirePostTarget(PostTarget& target);
  // Per frame, for the statistics read back once the GPU has finished the frame.
  static constexpr uint32_t kFrameSlots = 4;
  // (Session 49: 256 -> 1024, the write-back / fill segments; Session 80: 2048, the batch's own segments)
  static constexpr uint32_t kQueriesPerFrame = 2048;
  // (Session 49, review: 256 a frame as before unless gpu_ram_stats times the write-backs too.)
  uint32_t QueryLimit() const { return ram_stats_ ? kQueriesPerFrame : 256u; }
  enum ParityCounter : uint32_t {
    kParityPixels,
    kParityBothEmpty,
    kParityEmulatedOnly,
    kParityNativeOnly,
    kParityBothGeometry,
    kParityAgree,            // both geometry, |difference| <= tolerance
    kParityDiff1e6,          // both geometry, |difference| < 1e-6
    kParityDiff1e5,
    kParityDiff1e4,
    kParityDiff1e3,
    kParityDiffLarger,
    kParityCenterEmulated,   // float bits at the centre pixel
    kParityCenterNative,
    // The colour check (RGB, in 8-bit levels), from kParityColor on.
    kParityColor = 16,
    kParityColorPixels = kParityColor,
    kParityColorExact,       // the largest channel difference: 0
    kParityColorWithin2,     // 1-2
    kParityColorWithin8,     // 3-8
    kParityColorWithin32,    // 9-32
    kParityColorLarger,      // 33 or more
    kParityColorSumR,        // sums of the absolute differences
    kParityColorSumG,
    kParityColorSumB,
    kParityColorCenterEmulated,  // the packed RGBA8 at the centre pixel
    kParityColorCenterNative,
    kParityCounterCount = 32,
  };
  struct FrameSlot {
    bool used = false;
    bool read = true;
    uint64_t frame = 0;
    uint64_t submission = 0;  // the frame's last submission (its results are final once it has completed)
    uint32_t queries = 0;     // timestamps written (pairs: emulated segment start / end, native end)
    struct Segment {
      uint32_t emulated_begin = UINT32_MAX;  // query index, UINT32_MAX = none
      uint32_t native_begin = UINT32_MAX;
      uint32_t native_end = UINT32_MAX;
      // Session 43: 0 a scene pass (with its load), 1 the scene's write-back, 2 a post pass (with its load), 3 a
      // post write-back.
      uint32_t kind = 0;
      // Session 72 (gpu_native_truth_pass_log): a truth pass's size, pitch, colour image kind, whether it has a real
      // depth image, its draws and the shaders of the draw that opened it.
      uint16_t pass_width = 0, pass_height = 0, pass_pitch = 0;
      uint8_t pass_color_kind = 0, pass_depth = 0;
      uint32_t pass_draws = 0;
      uint64_t pass_vs = 0, pass_ps = 0;
      const char* label = nullptr;  // Session 82 (P1): an owned piece (kind 16): its short name; null = a truth segment
    };
    std::vector<Segment> segments;
    std::vector<uint32_t> submission_stamps;  // Session 80: query index of each submission's first command
    uint64_t alpha_frame = 0;  // (the frame the probes of this slot were issued in)
    std::vector<std::pair<uint64_t, uint32_t>> alpha_probes;  // Session 82 (gpu_native_owned_alpha_learn): key, query (this slot's range)
    bool parity = false, color_parity = false;
    bool dump = false;  // gpu_native_dump: both pictures were copied to dump_buffers_ at this frame's end
    // When the slot's parity sets and result ranges were last written (depth, colour).
    uint64_t parity_submission = 0, color_parity_submission = 0;
    uint32_t scene_draws = 0, native_draws = 0;
    // Scene-target draws a replacing frame leaves (or would leave) to the emulated renderer, by Reason.
    uint32_t skipped[kReasonCount] = {};
    bool replaced = false;  // a replacing frame
    // Native render passes, reloads of our targets from the EDRAM, write-backs into it.
    uint32_t passes = 0, loads = 0, writebacks = 0;
    uint32_t segments_native = 0, segments_emulated = 0;  // Session 40
    uint32_t color_format = UINT32_MAX;  // the scene target's RB_COLOR_INFO format at the colour check
    double cpu_us = 0.0;
    // Session 42: post draws claimed, our post passes, loads, write-backs; post-eligible draws left emulated, by
    // PostReasonCode.
    uint32_t post_draws = 0, post_passes = 0, post_loads = 0, post_writebacks = 0;
    uint32_t post_skipped[kPostReasonCount] = {};
    uint32_t post_depth_draws = 0;  // Session 43
    // Session 43 (S0-pre): resolves of a surface with native post draws not yet written back, and those whose whole
    // rectangle a claimed draw of that segment provably covered.
    uint32_t cover_resolves = 0, cover_resolves_full = 0;
    // Session 45 (the frame census): every draw handed over (and emulated repeats), by who drew it; resolves; texture
    // loads (all, and of textures over ranges a resolve wrote, with their bytes).
    uint32_t frame_draws = 0, frame_native_scene = 0, frame_native_post = 0;
    uint32_t frame_emulated[kEmuCount] = {};
    uint32_t resolves = 0, clearing_resolves = 0;
    uint64_t texture_loads = 0, texture_loads_resolved = 0, texture_bytes_resolved = 0;
    // Session 46 (the EDRAM census): a truth frame; the render target cache's EDRAM accesses this frame (uses, fill
    // dispatches, resolve copies reading it, resolve clears).
    bool truth = false;
    uint64_t edram[4] = {};
    uint32_t native_resolves = 0;  // resolves served from post targets (they read the EDRAM for the rest; review fix)
  };

  bool EnsureObjects();
  void DestroyObjects();
  bool EnsureTargets(uint32_t width, uint32_t height);
  // Objects the GPU may still use are destroyed once the submission open at their retirement has completed.
  struct Retired {
    uint64_t submission = 0;
    VkImage images[3] = {};
    VkDeviceMemory memory[3] = {};
    VkImageView views[4] = {};  // Session 43: a post target's depth view too
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSet replace_set = VK_NULL_HANDLE;  // from replace_descriptor_pool_
    VkBuffer buffer = VK_NULL_HANDLE;              // Session 46 (the truth staging buffer; its memory in memory[0])
  };
  void RetireTargets();
  void RetirePipeline(VkPipeline pipeline);
  void DestroyRetired(bool all);
  void FlushParityMemory();
  void InvalidateParityMemory();
  VkSampler sampler_nearest() const { return sampler_nearest_; }
  // Session 47: with `pending`, a pipeline not made yet is queued for the worker (VK_NULL_HANDLE, *pending true).
  VkPipeline GetPipeline(const Draw& draw, bool* pending = nullptr);
  bool EnsureParityPipeline();
  bool EnsureColorParityPipeline();
  bool EnsureEmulatedSceneImage(uint32_t width, uint32_t height);
  // Both parity checks: barriers, the slot's result range and the dispatch (pipeline and set bound by the caller).
  void DispatchParity(VkPipelineLayout layout, const void* constants, uint32_t constants_size, uint32_t width,
                      uint32_t height, bool reads_depth);
  bool Wants(const Draw& draw) const;
  bool IsNextInstance(const Draw& run, const Draw& draw) const;  // Session 44 (gpu_native_merge)
  // Session 44 (review): the guest draws in batch_ (merged instances counted), for the loss reports.
  size_t BatchGuestDraws() const {
    size_t n = 0;
    for (const Draw& d : batch_) n += d.instance_count;
    return n;
  }
  // Replacing: the load (EDRAM -> our targets, a full-screen draw at the start of our render pass) and the write-back
  // (our targets -> EDRAM, a compute pass), with the parity checks' EDRAM addressing.
  bool EnsureReplaceObjects();
  // Session 43 (sweep fix): the post load needs only the EDRAM's load set (the scene's targets may not exist yet);
  // writeback false = the load set alone.
  bool EnsureReplaceSets(bool writeback = true);
  struct EdramPlace {
    uint32_t color_base_dwords, depth_base_dwords, pitch_tile_dwords, tile_w, tile_h, width, height, wrap_dwords;
    uint32_t flags;  // bit 0: 20e4 depth; 1: a colour surface; 2 / 3 (write-back): write colour / depth
    float scale_x, scale_y;  // native pixels per emulated pixel
    uint32_t origin_x, origin_y;  // write-back: the first emulated pixel of the rectangle it covers
  };
  EdramPlace SceneEdramPlace() const;
  // Session 42; Session 43: depth - the segment has a depth surface (its base and format).
  EdramPlace PostEdramPlace(const PostTarget& target, bool depth) const;
  void WriteBack();
  void ReadFinishedFrames();
  // gpu_native_dump: copies this frame's emulated scene and native colour for writing once the GPU is done.
  void RecordDump(FrameSlot& slot);
  void WriteDump(uint64_t frame);
  void LogReport();
  void LogFrameCensus();  // Session 45: [native][frame]
  void LogEdramCensus();  // Session 46: [native][edram]
  uint64_t edram_seen_[4] = {};  // Session 46: the render target cache's EDRAM counts at the last frame end
  bool truth_ = false;            // Session 46: gpu_native_truth, latched per frame (truth())
  bool host_rt_logged_ = false;    // Session 54: truth frames on the host render target path (logged once)
  bool truth_setting_ = false, truth_refused_logged_ = false;
  // Session 45: the texture cache's running load counts at the last frame end (the frame census takes differences).
  uint64_t texture_loads_seen_ = 0, texture_loads_resolved_seen_ = 0, texture_bytes_resolved_seen_ = 0;
  void LogCensus(CommandProcessor::NativeCensus& census);
  uint32_t SlotOf(uint64_t frame) const { return uint32_t(frame % kFrameSlots); }

  VulkanCommandProcessor& command_processor_;

  bool enabled_ = false;
  bool shading_ = false;    // gpu_native_shading, latched per frame
  bool all_draws_ = false;  // gpu_native_draws, latched per frame
  bool replacing_ = false;  // gpu_native 2 and not a verify frame, latched per frame
  uint32_t all_exclude_ = 0;  // Session 45: gpu_native_all_exclude, latched per frame
  bool AllOn(uint32_t part) const { return all_native_ && !(all_exclude_ & part); }
  bool all_native_ = false;  // Session 45: gpu_native_all, latched per frame
  int32_t segment_log_frames_ = 0;  // Session 45: gpu_native_segment_log frames left
  // Session 72 (gpu_native_truth_pass_log): frames whose passes are still to be logged at readback, and the draws
  // recorded into the open truth pass.
  int32_t pass_log_frames_ = 0;
  uint32_t truth_pass_draws_ = 0;
  bool truth_pass_touched_color_ = false, truth_pass_touched_depth_ = false, truth_pass_real_depth_ = false,
       truth_pass_real_color_ = false;  // S73
  // Session 72 (gpu_native_truth_pass_log): one frame's timed segments, a line each, from its read timestamps.
  void LogTruthPasses(uint64_t frame, const FrameSlot& slot, const std::vector<uint64_t>& stamps, double ms);
  bool replace_failed_ = false;  // the load / write-back objects could not be made: side-car from then on
  bool no_edram_logged_ = false;  // gpu_native 2 without the interlock path: logged once
  // Replacing: our targets hold draws the EDRAM does not have yet / the EDRAM holds changes our targets do not have.
  bool native_dirty_ = false;
  bool native_stale_ = true;
  uint32_t scene_color_info_ = 0;  // RB_COLOR_INFO of the frame's first claimed draw (render target 0)
  bool scene_color_info_valid_ = false;
  // Session 40: a segment is a run of claimable scene draws between two points where our targets go stale (the
  // frame's start, an emulated draw, a clearing resolve); a native segment costs a load and a write-back. A segment
  // whose index drew fewer than gpu_native_min_draws claimable draws in the previous frame stays emulated (Bryan's
  // entrance draws a second, small scene after a clear: ~90 draws, cheaper emulated than reloaded).
  bool seg_active_ = false, seg_native_ = true, declined_pending_ = false;
  uint32_t seg_index_ = 0, seg_count_ = 0;
  std::vector<uint32_t> seg_draws_, seg_draws_prev_;
  // Session 40: what our draws changed since the last write-back: the channels, and the rectangle in emulated host
  // pixels (the write-back covers only that).
  bool dirty_color_ = false, dirty_depth_ = false;
  int32_t dirty_x0_ = INT32_MAX, dirty_y0_ = INT32_MAX, dirty_x1_ = 0, dirty_y1_ = 0;
  void ResetDirty() {
    dirty_color_ = dirty_depth_ = false;
    dirty_x0_ = dirty_y0_ = INT32_MAX;
    dirty_x1_ = dirty_y1_ = 0;
  }
  int32_t census_frames_left_ = 0;
  uint64_t frame_ = 0;  // frames ended (swaps) since start

  // The scene's render target: learned in the previous frame (scene_key_), counted in this one.
  SceneKey scene_key_;
  struct KeyCount {
    uint32_t surface_info, depth_info, depth_writes;
    // Session 43 (review): depth-writing draws claimed into post targets (gpu_native_post_depth), counted apart: the
    // scene key falls back to them only when no emulated depth-writing draw counted (or the scene had no draws).
    uint32_t post_depth_writes;
  };
  std::vector<KeyCount> key_counts_;

  // Targets (gpu_native_scale x the emulated scene's size).
  uint32_t target_width_ = 0, target_height_ = 0;
  float target_scale_ = 1.0f;          // native pixels per emulated host pixel
  uint32_t scene_width_ = 0, scene_height_ = 0;  // the emulated scene's size in host pixels (this frame's draws)
  uint32_t target_scene_width_ = 0, target_scene_height_ = 0;  // ... when the targets were last sized for it
  VkImage color_image_ = VK_NULL_HANDLE, depth_image_ = VK_NULL_HANDLE;
  VkDeviceMemory color_memory_ = VK_NULL_HANDLE, depth_memory_ = VK_NULL_HANDLE;
  VkImageView color_view_ = VK_NULL_HANDLE, depth_view_ = VK_NULL_HANDLE;
  VkFramebuffer framebuffer_ = VK_NULL_HANDLE;
  bool cleared_this_frame_ = false;  // the frame's first native pass clears
  bool parity_done_this_frame_ = false;
  bool color_parity_done_this_frame_ = false;
  bool scene_done_this_frame_ = false;  // the scene's depth resolve has happened: the rest stays emulated
  bool slot_queries_ok_ = false;     // this frame's timestamp range was reset
  bool color_readable_ = false;      // the colour image is in SHADER_READ_ONLY_OPTIMAL (OnSwapPrepare)
  std::vector<Retired> retired_;
  bool drew_this_frame_ = false;
  bool drew_last_frame_ = false;     // the picture the swap shows
  // What both depth targets hold where nothing was drawn: 1, or 0 when the scene uses reversed depth (learned from
  // the depth tests of the previous frame's scene draws).
  float clear_depth_ = 1.0f;
  uint32_t depth_reversed_draws_ = 0, depth_forward_draws_ = 0;

  // Objects made once.
  bool objects_failed_ = false;
  VkRenderPass render_pass_clear_ = VK_NULL_HANDLE, render_pass_load_ = VK_NULL_HANDLE;
  VkShaderModule clay_fragment_shader_ = VK_NULL_HANDLE;
  std::unordered_map<PipelineKey, VkPipeline, PipelineKeyHasher> pipelines_;
  uint32_t pipelines_failed_ = 0;
  // Session 47 (CPU): the last truth draw's pipeline (GetPipeline; dropped whenever pipelines_ is cleared).
  PipelineKey last_truth_key_ = {};
  VkPipeline last_truth_pipeline_ = VK_NULL_HANDLE;
  bool last_truth_valid_ = false;
  // Session 47 (CPU, gpu_native_truth_async_pipelines): truth pipelines are made on a worker thread; a truth draw whose
  // pipeline is not ready yet is skipped (as the emulated path skips one, vulkan_pipeline_no_submit_wait). The worker
  // only calls CreatePipeline; finished pipelines go into pipelines_ on the command thread (DrainPipelineJobs).
  struct PipelineJob {
    PipelineKey key;
    Draw draw;
    float spec[3];
    bool prewarm = false;  // Session 70: a stored record's (the precompile's progress counts these)
  };
  // Session 69 (gpu_native_pipeline_workers): a pool of workers, not one (slow drivers: the Intel tester's 50-740 ms
  // compiles queued behind each other).
  std::vector<std::thread> pipeline_threads_;
  std::mutex pipeline_mutex_;
  std::condition_variable pipeline_cv_, pipeline_idle_cv_;
  std::deque<PipelineJob> pipeline_jobs_;
  std::vector<std::pair<PipelineKey, VkPipeline>> pipeline_done_;
  std::unordered_set<PipelineKey, PipelineKeyHasher> pipeline_pending_;
  // Session 69: being made by a worker now. Session 70: -> that worker's thread handle (Windows; the command thread
  // raises it to normal priority while it waits for the key - gpu_native_pipeline_workers_quiet).
  std::unordered_map<PipelineKey, void*, PipelineKeyHasher> pipeline_running_;
  uint32_t pipeline_busy_ = 0;  // Session 69: workers making a pipeline now (was a bool: one worker)
  bool pipeline_quit_ = false;
  void StartPipelineWorkersLocked();  // Session 69: the pool, once (pipeline_mutex_ held)
  std::atomic<uint32_t> pipeline_done_count_{0};
  std::atomic<uint32_t> pipeline_worker_failed_{0};
  void PipelineWorker();
  void DrainPipelineJobs();
  // Waits until the worker has made every queued pipeline (before shader modules or layouts may be destroyed).
  void WaitPipelineJobs();
  void StopPipelineWorker();
  // Session 61 (review): the queued jobs not started yet dropped (their keys no longer pending); pipeline_mutex_ held.
  void DropPipelineJobsLocked();
  VkPipeline CreatePipeline(const Draw& draw, const PipelineKey& key, const float spec_values[3]) const;
  // Session 61 (pipeline storage): one native pipeline as plain values - the shaders by hash and modification, the
  // layout by its binding counts, the geometry shader by the host primitive, and the key's own bits. Everything
  // CreatePipeline reads comes from the key, so these values make the same pipeline again.
  struct PipelineRecord {
    uint64_t vertex_shader_hash;
    uint64_t vertex_modification;
    uint64_t pixel_shader_hash;  // the guest pixel shader (0 none): its emulated translation, the layout's counts
    uint64_t pixel_modification;
    uint64_t tint_hash;
    uint32_t state;
    uint32_t blend_control;
    uint32_t variant;
    uint16_t layout_counts[4];
    uint8_t flags;  // kRecordNativePixel, kRecordFloat24, kRecordTwin, kRecordGeometry
    uint8_t quantize;
    uint8_t host_primitive;  // xenos::PrimitiveType (kRecordGeometry)
    // Session 69 (was padding, 0 in older records): the draw's own NATIVE ext bits the settings cannot predict -
    // kRecordExtKnown, kRecordExtEpilogueIdle. 0 = an older record: predicted from the settings as before.
    uint8_t ext_bits = 0;
  };
  static_assert(sizeof(PipelineRecord) == 64);
  static constexpr uint8_t kRecordNativePixel = 1, kRecordFloat24 = 2, kRecordTwin = 4, kRecordGeometry = 8;
  // Session 95 (out\s95\android_structural.md section 3.4): a record always describes the plain draw, and KeyFromRecord derives the attribute key from it while gpu_native_vertex_input is on. A draw
  // refused at run time stores the same state with this flag, and that record stays plain: refused keys are prewarmed from the next boot on.
  static constexpr uint8_t kRecordPlainTwin = 16;
  // Session 69 (ext_bits): Known = stored with the epilogue specialisation on (EpilogueIdle is then the draw's own);
  // TexMaskKnown = stored with the texture-sign specialisation on (a 0 tex_unsigned_mask is the draw's own, not unset).
  static constexpr uint8_t kRecordExtKnown = 1, kRecordExtEpilogueIdle = 2, kRecordExtTexMaskKnown = 4;
  // The record of a pipeline GetPipeline is about to make, checked by making its key again from it (only a record that
  // gives the same key is stored).
  void StorePipelineRecord(const Draw& draw, const PipelineKey& key);
  // The key a record makes (translating its shaders where needed); false when a shader or object is missing.
  // (convert: a prewarm of a stored record - its vertex / pixel modification is converted to the current pixel / vertex
  // specialisation settings first, `converted` set when that changed it; not for the check of a record just made.)
  // (exts: for a check of a record just made - the draw's pixel and vertex ext words; with `convert` they are predicted.)
  bool KeyFromRecord(const PipelineRecord& record, PipelineKey& key, bool convert = false, bool* converted = nullptr,
                     const uint32_t* exts = nullptr);
  // From OnFrameEnd: the stored records' pipelines queued for the worker, within gpu_native_pipeline_prewarm_ms; the
  // storage flushed.
  void StorageOnFrameEnd();
  std::FILE* storage_file_ = nullptr;
  bool storage_dirty_ = false;
  std::unordered_set<uint64_t> storage_hashes_;  // records in the file (XXH3 of the record)
  std::vector<PipelineRecord> prewarm_;           // stored records not made yet in this run
  size_t prewarm_next_ = 0;
  std::vector<PipelineRecord> prewarm_retry_;     // waiting for their shaders (StorageOnFrameEnd)
  uint32_t prewarm_retries_ = 0, prewarm_logged_retries_ = 0;
  // Session 73: the retry passes go on all run, at most one a second of wall time (they were 20 passes 120 frames
  // apart - the boot's fast-forward used them up in its first 4 s, before the match's shaders were loaded, and the
  // entrance made its pipelines on first use: the long frame at every entrance start on phones); why records wait.
  std::chrono::steady_clock::time_point prewarm_retry_time_{};
  uint32_t prewarm_wait_shader_ = 0, prewarm_wait_twin_ = 0, prewarm_wait_pass_ = 0;
  uint32_t prewarm_queued_logged_ = 0;
  bool prewarm_reasons_logged_ = false;  // the first wall-time pass logs why records wait, whatever it made
  uint32_t prewarm_queued_ = 0, prewarm_skipped_ = 0, storage_written_ = 0, storage_mismatch_ = 0;
  uint32_t prewarm_converted_ = 0;  // wwe2k14 patch (Session 66): records converted to the current specialisation settings
  bool prewarm_logged_ = false;
  // wwe2k14 patch (Session 70, v0.2.8's doubled warm-up): an older record (no ext bits) is prewarmed in both epilogue
  // variants, and a prewarmed pipeline was never stored again - so every run made both, for good. Now the first draw
  // that uses a guessed variant stores its own record (with the bits), and an older record whose known sibling is in
  // the file (the same record but for the ext bits) is not expanded at all.
  std::unordered_set<PipelineKey, PipelineKeyHasher> prewarm_guessed_;  // keys made from an older record's guesses
  // XXH3 of each known record with its ext bits cleared -> the variants known (1 epilogue idle, 2 not); an older record
  // with any is not expanded (StorageOnFrameEnd: why not per variant).
  std::unordered_map<uint64_t, uint8_t> storage_known_siblings_;
  uint32_t prewarm_old_skipped_ = 0, prewarm_guessed_stored_ = 0;
  uint32_t prewarm_old_conformed_ = 0;  // Session 80: of those, matched only after converting their modifications
  // Session 70 (gpu_native_precompile): prewarm jobs the workers finished; the status for the game's precompile mode.
  std::atomic<uint32_t> prewarm_made_{0};
  std::string precompile_status_;
  void PrecompileStatus();
  VkQueryPool query_pool_ = VK_NULL_HANDLE;
  float timestamp_period_ns_ = 0.0f;

  // The batch: draws since the emulated render pass began.
  std::vector<Draw> batch_;
  uint64_t batch_submission_ = 0;
  bool segment_open_ = false;  // a scene segment's start timestamp was written in the open emulated render pass
  FrameSlot slots_[kFrameSlots];

  // Replacing: the load and write-back passes. One write-back set per target generation (retired with the targets),
  // one load set per EDRAM buffer.
  VkShaderModule load_vertex_shader_ = VK_NULL_HANDLE, load_fragment_shader_ = VK_NULL_HANDLE;
  VkShaderModule writeback_shader_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout load_set_layout_ = VK_NULL_HANDLE, writeback_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout load_layout_ = VK_NULL_HANDLE, writeback_layout_ = VK_NULL_HANDLE;
  VkPipeline load_pipeline_ = VK_NULL_HANDLE, writeback_pipeline_ = VK_NULL_HANDLE;
  VkDescriptorPool replace_descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet load_set_ = VK_NULL_HANDLE, writeback_set_ = VK_NULL_HANDLE;
  VkBuffer replace_edram_buffer_ = VK_NULL_HANDLE;  // what the sets point at
  bool replace_objects_failed_ = false;

  // Session 42 (gpu_native_post): post targets, one per surface pitch, and the open segment. A segment is the run of
  // claimed post draws into one surface between two points where the EDRAM may change under it (a resolve, an
  // emulated draw, the frame's end): it begins with the stencil cleared (and with a load from the EDRAM when a draw
  // needs what is under it), and its write-back packs the covered pixels only.
  bool post_ = false;  // gpu_native_post, latched per frame
  // gpu_native_after_scene, latched per frame: scene-target draws after the scene's colour resolve, and rectangle /
  // quad / point lists and screen-space draws into the scene, through the scene's targets.
  bool after_scene_ = false;
  bool scene_extras_ = false;  // Session 43: gpu_native_scene_extras, latched per frame
  bool claim_retry_ = false;  // ClaimWantsWriteBack
  std::vector<PostTarget> post_targets_;
  int32_t post_current_ = -1;  // index in post_targets_ of the open segment's target
  bool post_segment_open_ = false, post_loaded_ = false, post_dirty_ = false;
  uint32_t post_surface_info_ = 0, post_color_info_ = 0;  // the surface the claimed draws go to
  // Session 43 (gpu_native_post_depth): the segment's draws test or write depth (its depth surface, RB_DEPTH_INFO);
  // then its depth was loaded from the EDRAM at its start; post_depth_dirty_: a draw wrote depth since.
  bool post_depth_ = false, post_depth_dirty_ = false;
  bool post_batch_depth_ = false;  // the post batch has draws that test or write depth
  uint32_t no_shader_logged_ = 0;  // diagnostic lines written for post draws left emulated without a shader
  std::vector<uint64_t> no_shader_pairs_;
  uint32_t post_depth_info_ = 0;
  std::vector<Draw> post_batch_;
  uint64_t post_batch_submission_ = 0;
  int32_t post_x0_ = INT32_MAX, post_y0_ = INT32_MAX, post_x1_ = 0, post_y1_ = 0;  // covered since the write-back
  bool post_objects_failed_ = false;
  VkFormat post_ds_format_ = VK_FORMAT_UNDEFINED;
  // Session 43: one render pass pair and set of load pipelines per post target colour class (PostTargetClass).
  VkRenderPass post_render_pass_begin_[kPostTargetClasses] = {}, post_render_pass_continue_[kPostTargetClasses] = {};
  // Session 44 (gpu_native_post_lean_depth): the same without the depth aspect's load / store.
  VkRenderPass post_render_pass_begin_no_depth_[kPostTargetClasses] = {},
               post_render_pass_continue_no_depth_[kPostTargetClasses] = {};
  // Session 45 (gpu_native_post_sync): a segment's first pass that keeps the colour and depth the target holds (the
  // stencil's coverage marks cleared).
  VkRenderPass post_render_pass_begin_keep_[kPostTargetClasses] = {};
  // Session 46 (truth frames): as begin_keep, the stencil loaded too (a truth image keeps its stencil from pass to pass).
  VkRenderPass truth_render_pass_keep_[kPostTargetClasses] = {};
  // Session 56: the two render passes truth passes open (truth_render_pass_keep_, post_render_pass_continue_no_depth_)
  // without their external subpass dependencies - the implicit ones, which wait for nothing (every such dependency is
  // a wait-for-idle on the GPU); the synchronisation they gave is recorded as barriers where it is needed.
  VkRenderPass truth_render_pass_keep_nodep_[kPostTargetClasses] = {},
               truth_render_pass_no_depth_nodep_[kPostTargetClasses] = {};
  // Session 66 (G2.1, gpu_native_truth_pass_dont_care): truth_render_pass_keep_nodep_ with load DONT_CARE for a pass
  // whose first draw is a whole-surface clear: [0] colour, [1] depth + stencil, [2] both (compatible: only the load
  // ops differ).
  VkRenderPass truth_render_pass_dont_care_[3][kPostTargetClasses] = {};
  // wwe2k14 patch (Session 80, gpu_native_truth_clear_load): truth_render_pass_keep_nodep_ with colour, depth and stencil
  // CLEARed at the pass's start - for a pass whose first draw is a whole-surface clear of known values, which is then not
  // drawn (truth_skip_clear_draw_; compatible: only the load ops differ).
  VkRenderPass truth_render_pass_clear_[kPostTargetClasses] = {};
  bool truth_skip_clear_draw_ = false;
  // Session 90: TruthRecordDraw's nesting (a held clear flushed from inside another draw's recording); the flag above is
  // saved / restored per call.
  uint32_t truth_record_depth_ = 0;
  const Draw* truth_outer_draw_ = nullptr;
  // wwe2k14 patch (Session 80, gpu_native_truth_bounds_vs): the list as last parsed, and its hashes.
  std::string truth_bounds_list_;
  std::vector<uint64_t> truth_bounds_vs_;
  // Session 73 (gpu_native_truth_color_none): truth_render_pass_keep_nodep_, truth_render_pass_no_depth_nodep_ and
  // truth_render_pass_dont_care_[1] with the colour LOAD_OP_NONE / STORE_OP_NONE, for passes none of whose draws
  // touch it (VK_EXT_load_store_op_none; VK_NULL_HANDLE without it).
  VkRenderPass truth_render_pass_color_none_[3][kPostTargetClasses] = {};
  size_t truth_pass_begin_offset_ = SIZE_MAX;
  // (Session 87, review F9: every truth pass begun gets a serial, by its begin offset, for the submission it is recorded in - a pass of ours that
  // turns its clears' pass off later is sure the pass at those offsets is still that one.)
  uint64_t truth_pass_serial_ = 0, truth_pass_begin_serial_ = 0, truth_pass_map_submission_ = UINT64_MAX;
  std::unordered_map<size_t, uint64_t> truth_pass_by_offset_;
  size_t truth_pass_end_offset_ = SIZE_MAX;  // Session 80: right after the last truth pass's end
  VkRenderPass truth_pass_begin_rp_ = VK_NULL_HANDLE;
  uint32_t truth_pass_kind_ = 0;
  VkPipeline post_load_pipeline_[kPostTargetClasses] = {};
  // Session 43: the load of depth only, and of colour and depth (depth written by the load shader, test always).
  VkPipeline post_load_depth_pipeline_[kPostTargetClasses] = {}, post_load_both_pipeline_[kPostTargetClasses] = {};
  bool post_quantize_ = false;  // gpu_native_post_quantize, latched per frame
  bool cover_stats_ = false;  // Session 43: gpu_native_cover_stats, latched per frame
  int32_t post_cover_[4] = {};  // the open post segment's largest proven-covered rectangle (empty: x1 <= x0)
  bool post_class_ok_[kPostTargetClasses] = {};  // the class's format can be a target (EnsurePostObjects)
  VkShaderModule post_writeback_shader_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout post_writeback_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout post_writeback_layout_ = VK_NULL_HANDLE;
  VkPipeline post_writeback_pipeline_ = VK_NULL_HANDLE;

  // Composition at the swap.
  VkShaderModule compose_shader_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout compose_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout compose_layout_ = VK_NULL_HANDLE;
  VkPipeline compose_pipeline_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet compose_sets_[kFrameSlots] = {};
  uint64_t compose_set_submission_[kFrameSlots] = {};
  VkSampler sampler_ = VK_NULL_HANDLE;          // linear (the swap)
  VkSampler sampler_nearest_ = VK_NULL_HANDLE;  // the parity check's depth
  // The emulated scene's colour as the EDRAM held it at the scene's colour resolve (written by the colour parity check;
  // gpu_native_view 3 and 4 show it): the picture before the game's post-processing, like ours.
  VkImage emulated_scene_image_ = VK_NULL_HANDLE;
  VkDeviceMemory emulated_scene_memory_ = VK_NULL_HANDLE;
  VkImageView emulated_scene_view_ = VK_NULL_HANDLE;
  uint32_t emulated_scene_width_ = 0, emulated_scene_height_ = 0;
  bool emulated_scene_valid_ = false;  // written this frame (GENERAL layout)
  bool emulated_scene_valid_last_ = false;  // ... in the frame the swap presents
  // The scene target is k_8_8_8_8_GAMMA (seen at its colour resolve): our linear colour is shown and compared through
  // the Xenos gamma curve. This frame's, and the presented frame's.
  bool scene_gamma_this_frame_ = false, scene_gamma_last_ = false;

  // The parity check.
  VkShaderModule parity_shader_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout parity_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout parity_layout_ = VK_NULL_HANDLE;
  VkPipeline parity_pipeline_ = VK_NULL_HANDLE;
  VkDescriptorSet parity_sets_[kFrameSlots] = {};
  VkBuffer parity_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory parity_memory_ = VK_NULL_HANDLE;
  uint32_t* parity_mapping_ = nullptr;
  bool parity_coherent_ = true;
  bool parity_failed_ = false;
  VkShaderModule color_parity_shader_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout color_parity_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout color_parity_layout_ = VK_NULL_HANDLE;
  VkPipeline color_parity_pipeline_ = VK_NULL_HANDLE;
  VkDescriptorSet color_parity_sets_[kFrameSlots] = {};
  bool color_parity_failed_ = false;
  // gpu_native_dump: host-visible copies of the two pictures of one frame (emulated scene, native).
  VkBuffer dump_buffers_[2] = {};
  VkDeviceMemory dump_memory_[2] = {};
  uint32_t dump_width_ = 0, dump_height_ = 0, dump_native_width_ = 0, dump_native_height_ = 0;
  bool dump_pending_ = false;
  bool dump_gamma_ = false;

  // Rolling report (gpu_native_report_frames).
  struct Report {
    uint32_t frames = 0, frames_with_results = 0, parity_frames = 0, color_parity_frames = 0;
    uint64_t scene_draws = 0, native_draws = 0, dropped_occlusion = 0, dropped_submission = 0;
    uint64_t skipped[kReasonCount] = {};
    uint32_t replaced_frames = 0;
    uint64_t passes = 0, loads = 0, writebacks = 0;
    uint64_t segments_native = 0, segments_emulated = 0;  // Session 40
    // Replaced frames' own GPU time (our passes, loads and write-backs); the side-car numbers above leave them out.
    uint32_t replaced_frames_with_results = 0;
    double replaced_native_ms = 0.0;
    // Session 43: by Segment::kind (Session 46: 4 truth passes, 5 truth stores; Session 49: 6 truth resolve write-backs
    // into guest memory, 7 truth texture fills)
    double replaced_kind_ms[8] = {};
    uint32_t color_formats = 0;  // bit per RB_COLOR_INFO format seen at the colour checks
    uint64_t post_draws = 0, post_passes = 0, post_loads = 0, post_writebacks = 0;  // Session 42
    uint64_t merged = 0, merge_runs = 0;
    uint64_t post_lean_depth_passes = 0;  // Session 44 (gpu_native_post_lean_depth)
    uint64_t scene_loads_covered = 0;     // Session 45: scene loads a covering first draw made unnecessary
    uint64_t post_loads_avoided = 0;      // Session 45: colour / depth loads gpu_native_post_sync made unnecessary
    uint64_t native_resolves = 0;         // Session 45: resolves served from a post target (gpu_native_resolve)
    // Session 45 (gpu_native_defer_writeback): segments whose colour was not written back at their resolve; of those
    // written later (something read their tiles), and dropped (overwritten first by a fill).
    uint64_t pending_deferred = 0, pending_written = 0, pending_dropped = 0, pending_resumed = 0;
    uint64_t post_mirrors = 0, scene_mirrors = 0;  // Session 45 (clear mirrors): drawn in place of loads
    uint64_t textures_filled = 0;  // Session 45 (N3a): textures filled by native resolves (their loads skipped)
    uint64_t texture_only_resolves = 0;  // Session 45 (N3b): stock copies that filled their texture
    uint64_t writebacks_fused = 0;  // Session 45: write-backs done by the native resolve's own dispatch  // Session 44 (gpu_native_merge): instances added, host draws that got them
    uint64_t post_skipped[kPostReasonCount] = {};
    uint64_t post_depth_draws = 0;  // Session 43
    uint64_t cover_resolves = 0, cover_resolves_full = 0;
    // Session 45 (the frame census).
    uint64_t frame_draws = 0, frame_native_scene = 0, frame_native_post = 0;
    uint64_t frame_emulated[kEmuCount] = {};
    uint64_t resolves = 0, clearing_resolves = 0;
    uint64_t texture_loads = 0, texture_loads_resolved = 0, texture_bytes_resolved = 0;
    uint32_t truth_frames = 0;  // Session 46 (the EDRAM census)
    uint64_t edram[4] = {};
    uint64_t edram_native_resolves = 0;
    double emulated_ms = 0.0, native_ms = 0.0, cpu_us = 0.0;
    uint64_t parity[kParityCounterCount] = {};
    uint32_t last_center[2] = {}, last_color_center[2] = {};
  } report_;
  std::chrono::steady_clock::time_point cpu_start_;

  // ---------------------------------------------------------------------------------------------------------------
  // Session 46 (truth frames, native_truth.cpp): one image per guest surface, an owner per EDRAM tile.
  // ---------------------------------------------------------------------------------------------------------------
  struct TruthImage {
    // Key: the surface's EDRAM base tile, its pitch in 32bpp tiles (80 samples), the kind (1 RGBA8 colour: k_8_8_8_8;
    // 2 RGBA16_SNORM colour: k_16_16 / k_16_16_16_16; 3 D32S8 depth) and the guest format (RB_COLOR_INFO color_format,
    // RB_DEPTH_INFO depth_format). Base kTruthScratchBase: a scratch attachment of that pitch that owns no tile.
    uint32_t base = 0, pitch_tiles = 0, kind = 0, format = 0;
    uint32_t width = 0, height = 0;  // host pixels
    uint32_t stencil = 0;            // depth: the stencil value of its last clear (the images keep no stencil)
    std::bitset<2048> stencil_tiles;  // depth: the tiles a truth clear wrote that stencil into
    // depth: the tiles whose stencil this image holds (a truth clear of the whole tile since the image owns it; only
    // while it owns the tile); a stencil draw elsewhere is not claimed (the frame falls back).
    std::bitset<2048> stencil_valid;
    // Session 47 (CPU): the EDRAM tiles this image owns (the owner map: no two images own a tile; bitset work, no
    // per-tile loops).
    std::bitset<2048> owned;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;         // the attachment (colour, or depth + stencil)
    VkImageView sample_view = VK_NULL_HANDLE;  // colour, or the depth aspect (the stores read it)
    VkImageView stencil_view = VK_NULL_HANDLE;  // depth: the stencil aspect (the stores read it; Session 46)
    bool initialized = false;                   // cleared and in its attachment layout
    VkDescriptorSet store_sets[2] = {};         // TruthStore's set: into the staging buffer, into the EDRAM
    VkDescriptorSet resolve_set = VK_NULL_HANDLE;  // TruthResolveDirect's (the transcribed resolve shader's layout)
    VkBuffer store_set_edram = VK_NULL_HANDLE;  // the EDRAM buffer store_sets[1] points at
    // Session 52 (gpu_native_truth_batch_rename): renamed for a deferred resolve - the original image whose pixels
    // outside rename_rect (host) this one still has to get (copied at the flush); VK_NULL_HANDLE if none.
    VkImage rename_source = VK_NULL_HANDLE;
    int32_t rename_rect[4] = {};
    // Session 54 (TruthDrawRatio below 1): the draw twin - the image the truth passes draw into at the draw ratio of
    // this image's size. This image (at the whole resolution scale) stays what every read uses: the twin is blitted up
    // into it before a read (draw_dirty), and made again from it when this image changed (draw_for != image).
    VkImage draw_image = VK_NULL_HANDLE;
    VkDeviceMemory draw_memory = VK_NULL_HANDLE;
    VkImageView draw_view = VK_NULL_HANDLE;
    VkImage draw_for = VK_NULL_HANDLE;
    uint32_t draw_width = 0, draw_height = 0;
    bool draw_dirty = false;
    // The resample passes (TruthResampleTwin): [0] into the twin (sampling this image), [1] into this image
    // (sampling the twin). Depth: the twin's depth and stencil aspects as sampled views.
    VkDescriptorSet sync_sets[2] = {};
    VkFramebuffer sync_fbs[2] = {};
    VkImageView draw_sample_view = VK_NULL_HANDLE, draw_stencil_view = VK_NULL_HANDLE;
    bool is_64bpp() const { return kind == 2 && format == 5; }
    uint32_t row_tiles() const { return pitch_tiles * (is_64bpp() ? 2u : 1u); }
  };
  static constexpr uint32_t kTruthScratchBase = 0xFFFF;
  std::vector<TruthImage> truth_images_;
  // Session 72 part 4 (gpu_native_truth_skip_restores): per truth image, the truth draws recorded into it (bumped by
  // TruthRecordDraw, never reset), and the last direct resolves' sources: the destination page, the image read, its
  // draw count then, the frame, the copied size (host pixels, from the image's and the texture's origin).
  std::vector<uint64_t> truth_image_writes_;
  struct RestoreSource {
    uint32_t page = 0;
    int16_t image = -1;
    uint64_t writes = 0, frame = UINT64_MAX;
    uint32_t width = 0, height = 0;
    uint64_t extent_start = 0, extent_end = 0;  // the copy's destination bytes (a later copy into them drops it)
    int32_t claims = 0;  // truth_frame_draws_ at the copy (TruthRestoreRedundantEarly: no draw claimed since)
    bool at_origin = false;  // from the image's origin into the texture's (the restore checks need it)
    uint64_t key = 0;        // ResolveKey of the copy (gpu_native_truth_skip_repeat_resolves)
  };
  std::array<RestoreSource, 8> restore_sources_{};
  uint32_t restore_source_next_ = 0;
  int32_t restore_source_fresh_ = -1;  // the entry the resolve being marked wrote (TruthAfterResolveMarked keeps it)
  void TruthNoteRestoreSource(const draw_util::ResolveInfo& info, int16_t index, bool at_origin, uint32_t width,
                              uint32_t height);
  bool TruthRestoreRedundant(const Draw& draw, int16_t color, int16_t depth, const int32_t* rect);
  // The owner of each EDRAM tile: the truth image whose `owned` has it (Session 47: no per-tile array any more), else
  // the EDRAM.
  std::bitset<2048> truth_owned_any_;  // Session 47: the tiles some image owns (the others: the EDRAM's)
  int16_t TruthOwnerOf(uint32_t tile) const;  // -1: the EDRAM
  // Session 47: tiles `color_tiles` become image `color`'s, `depth_tiles` image `depth`'s (depth wins where both).
  void TruthSetOwners(const std::bitset<2048>& color_tiles, int16_t color, const std::bitset<2048>& depth_tiles,
                      int16_t depth);
  void TruthClearOwners();
  uint32_t truth_fallback_reason_ = 0;  // a claim refused in a truth frame: why (TruthReasonName), 0 none
  bool truth_frame_ = false;            // this frame started as a truth frame (truth_ goes off at a fallback)
  int32_t truth_frame_draws_ = 0;       // draws claimed this frame (gpu_native_truth_debug_draws)
  std::string truth_fail_reason_;       // Session 95 (PLAIN): why the truth objects failed, for the named fatal
  bool truth_failed_ = false;           // truth objects could not be made: no truth frames from then on
  struct TruthFramebufferEntry {
    int16_t color, depth;
    VkFramebuffer framebuffer;
    VkImageView color_view, depth_view;  // Session 52 (renamed images change their views)
  };
  std::vector<TruthFramebufferEntry> truth_framebuffers_;
  // The staging buffer (the EDRAM's size and layout) the resolves of truth frames read; the store pipelines.
  VkBuffer truth_staging_ = VK_NULL_HANDLE;
  VkDeviceMemory truth_staging_memory_ = VK_NULL_HANDLE;
  VkDeviceSize truth_staging_size_ = 0;
  VkDescriptorSet truth_staging_resolve_set_ = VK_NULL_HANDLE;  // the render target cache's layout, over it
  VkDescriptorSet truth_staging_load_set_ = VK_NULL_HANDLE;     // the load shaders' set, over it (alias loads)
  VkShaderModule truth_store_shaders_[2] = {};  // colour, depth
  VkDescriptorSetLayout truth_store_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout truth_store_layout_ = VK_NULL_HANDLE;
  VkPipeline truth_store_pipelines_[2] = {};
  VkShaderModule truth_fill_shader_ = VK_NULL_HANDLE;  // Session 47 (M6): TruthFillTexture's pack shader
  VkPipeline truth_fill_pipeline_ = VK_NULL_HANDLE;
  // Session 52 (gpu_native_truth_fill_direct): the pack shader writing the texture's storage view (32 / 64 bits).
  VkPipelineLayout truth_fill_tex_layout_ = VK_NULL_HANDLE;
  // (Session 59, gpu_native_truth_fill_direct_small: [2] / [3] write k_8 / k_4_4_4_4 through R8_UINT / R16_UINT.)
  VkPipeline truth_fill_tex_pipelines_[4] = {};
  bool truth_fill_tex_failed_ = false;
  bool truth_fill_tex_small_failed_ = false;
  bool EnsureTruthFillTexPipelines();
  bool EnsureTruthFillTexSmallPipelines();
  VkDescriptorSet TruthStoreSet(TruthImage& t, bool staging, VkBuffer buffer);
  // wwe2k14 patch (Session 79, gpu_native_truth_alias_direct): the alias load straight from the owner image - its
  // fragment shader works out the dword TruthStore would have put in the staging buffer (the same packing) and
  // converts it as the load does; no store dispatch. [owner depth][target depth][target class]; the owners' store sets
  // (bindings 1 / 2 are visible to fragment shaders too). Owners read this way stay in their read layouts until the
  // pass ends (truth_alias_read_owners_: image, depth).
  struct AliasDirectConstants {
    EdramPlace place;
    uint32_t owner_base_dwords, owner_pitch_tile_dwords, owner_flags;
  };
  VkShaderModule alias_direct_shaders_[2] = {};
  VkPipelineLayout alias_direct_layout_ = VK_NULL_HANDLE;
  VkPipeline alias_direct_pipelines_[2][2][kPostTargetClasses] = {};
  bool alias_direct_failed_ = false;
  bool EnsureAliasDirectPipelines();
  std::vector<std::pair<VkImage, bool>> truth_alias_read_owners_;
  void TruthRestoreAliasOwners();
  std::vector<int16_t> truth_scratch_direct_owners_, truth_scratch_stored_owners_;
  VkDescriptorPool truth_pool_ = VK_NULL_HANDLE;
  uint32_t truth_resolution_[2] = {};  // the resolution scale the images were made at
  struct TruthRetiredSet {
    uint64_t submission;
    VkDescriptorSet set;
  };
  std::vector<TruthRetiredSet> truth_retired_sets_;
  // The truth census (per frame slot, summed in the report): fallbacks by reason, boundary loads / write-outs, staged
  // resolves and their tiles from the EDRAM, alias reads (a run read tiles another image owns: kept as they were).
  static constexpr uint32_t kTruthReasons = 24;
  // Session 55: surfaces at least this many 80-sample tiles wide are the screen-wide ones (1280 pixels).
  static constexpr uint32_t kTruthScreenPitchTiles = 16;
  struct TruthCounts {
    uint32_t frames = 0, fallbacks = 0, passes = 0, draws = 0, loads = 0, writeouts = 0, staged = 0, staged_edram = 0;
    uint32_t alias_reads = 0, store_dispatches = 0, direct = 0, direct_depth = 0;  // (Session 59: direct_depth)
    uint32_t alias_direct = 0;  // Session 79 (gpu_native_truth_alias_direct): alias reads loaded straight from owners
    uint32_t deferred_memory = 0;  // Session 79 (gpu_native_truth_defer_memory_resolves): deferred resolves writing memory
    uint32_t compiling = 0;  // Session 47: draws skipped while their pipeline was being made
    uint32_t waited = 0;     // Session 55: draws into narrower surfaces whose pipeline was made at once instead
    uint32_t filled = 0, fill_refused = 0;
    uint32_t fill_packed = 0, fill_direct = 0, fills_deferred = 0;  // (Session 59: fills_deferred)
    uint32_t deferred = 0, deferred_batches = 0;  // Session 52: texture-only resolves deferred, flushes
    // Session 66 (G1.1, gpu_native_truth_drop_dead_resolves): deferred resolves / fills dropped (a later resolve covers
    // their texels whole, nothing sampled the texture between), and the flushes of non-empty batches by DeferFlushReason.
    uint32_t dropped_dead = 0, dropped_dead_fills = 0;
    uint32_t defer_flushes[16] = {};
    uint32_t twin_blits[2] = {};  // Session 54: draw twins made from their images (0), images from twins (1)
    uint32_t early_sampled = 0;
    uint32_t depth_renames = 0;
    // Session 72 part 4 (gpu_native_truth_skip_restores): restore draws left out, and those looked at but drawn.
    uint32_t restores_skipped = 0, restores_drawn = 0;
    uint32_t repeat_resolves_skipped = 0;  // (gpu_native_truth_skip_repeat_resolves)
    // Session 80 (gpu_native_truth_drop_dead_copies): copies turned into no-ops (colour, depth), candidates kept because
    // the texture was used, and no-op attempts refused (not on command boundaries).
    uint32_t dead_copies[2] = {}, dead_kept_touched = 0, dead_nop_failed = 0;
    // (gpu_native_truth_drop_dead_passes: passes turned into no-ops; candidates whose next pass cleared their colour but
    // something was recorded in between, or the no-op was refused.)
    uint32_t dead_passes = 0, dead_pass_kept = 0;
    // (gpu_native_truth_carry_copies: copies kept by a flush, kept past a frame's end, and of those dropped - covered by
    // a later copy before anything used them - or run; renames by a full-screen draw instead of a clear.)
    uint32_t carry_kept = 0, carry_crossed = 0, carry_dropped = 0, carry_run = 0, cover_renames = 0;
    // (gpu_native_truth_serve_copies: bindings served from an image, passes ended to make one readable, served
    // images whose draw never came - a fallback.)
    uint32_t served = 0, serve_breaks = 0, serve_lost = 0;
    uint32_t serve_ahead = 0;  // Session 80 (gpu_native_truth_serve_ahead): images made readable before their pass
    // Session 80 (gpu_native_own_steps): draws into our images, their copies left out, outputs lost (overwritten in
    // the surface before a copy, or the copy not of that rectangle / not a plain one), bindings bound to our images,
    // listed draws refused (their state not one we take).
    uint32_t owned_draws = 0, owned_copies = 0, owned_lost_draw = 0, owned_lost_copy = 0, owned_served = 0,
             owned_refused = 0, owned_clears_dropped = 0, owned_clears_kept = 0;
    // (Session 80, owned passes: passes started, resumed after something ended them, lost at a submission's end; the
    // dirty tiles copied back from our images before a read, and those a later clear wrote whole.)
    uint32_t owned_passes = 0, owned_resumes = 0, owned_lost_submission = 0, owned_materialized = 0, owned_cleaned = 0;
    // (Passes past their copy: draws after a copy, snapshots of our image for its page, rectangles copied into a page
    // image of ours, and those the game copied from the truth images - their tiles ours first.)
    uint32_t owned_continued = 0, owned_snapshots = 0, owned_region_copies = 0, owned_region_copies_run = 0, owned_readback_copies = 0;
    uint32_t owned_fills = 0;  // (16-bit pages filled whole from our image)
    // (Served from a slot's live image; draws sampling their own target, its rename widened to all of it / kept empty.)
    uint32_t served_live = 0, serve_renames = 0, serve_renames_whole = 0, serve_renames_copied = 0;
    uint32_t carry_fresh_sets = 0;  // (carried copies run in a later submission: a fresh destination descriptor)
    uint32_t dont_care_color = 0, dont_care_depth = 0;  // Session 66 (G2.1): passes opened without loading them
    // Session 73: passes none of whose draws touched the colour image / the (real) depth image - loaded and stored
    // for nothing (a tile GPU's bandwidth: validation best practices "redundant-attachment-on-tile").
    uint32_t untouched_color = 0, untouched_depth = 0, untouched_scratch_color = 0;
    uint32_t color_none = 0;  // Session 73: of them, opened with the colour neither loaded nor stored
    uint32_t skip_clear = 0, skip_nonclear = 0, skip_nested_reset = 0;  // Session 90: the load-op clear flag census
    uint32_t clear_loads = 0;  // Session 80 (gpu_native_truth_clear_load): passes begun with their clear, its draw left out
    // Session 80 (gpu_native_truth_bounds_vs): draws given their vertices' box (worked out / from the cache), refused,
    // and the command processor's CPU time for it.
    uint32_t bounds_made = 0, bounds_cached = 0, bounds_refused = 0;
    double bounds_ms = 0.0;
    uint32_t clear_attachments = 0;  // Session 56: fast clears' depth / stencil also cleared by vkCmdClearAttachments  // Session 56: depth images renamed with their pass's colour (gpu_native_truth_rename_depth)  // Session 56: written textures moved to the sampled layout before a pass opened
    uint32_t renames = 0;  // Session 52: images renamed for a deferred resolve  // Session 52: packed fills, of them straight into the texture  // Session 47 (M6): textures filled by TruthFillTexture, refusals
    uint32_t reasons[kTruthReasons] = {};
    // Session 49 (gpu_ram_stats): resolve write-backs into guest memory by path (staged: the staging stores + the stock
    // copy; direct: TruthResolveDirect), their bytes (the copies' destination extents), segments left untimed (the
    // frame's timestamps ran out), and the fill refusals by reason (the refuse() strings: literals).
    uint32_t wb_staged = 0, wb_direct = 0, untimed = 0;
    uint64_t wb_staged_bytes = 0, wb_direct_bytes = 0;
    std::vector<std::pair<const char*, uint32_t>> refusals;
  } truth_counts_;
  // Session 49 (gpu_ram_stats): the open write-back / fill timer segment (kind 6 / 7, or 0), its index in the slot.
  uint32_t truth_timer_kind_ = 0, truth_timer_segment_ = 0;
  bool ram_stats_ = false;  // gpu_ram_stats, latched per frame
  // Session 49 (gpu_native_truth_skip_writeback): see TruthBeforeResolveWrite.
  struct WbRecord {
    void* texture = nullptr;                  // VulkanTexture (pinned while recorded)
    int32_t layer = -1;                       // Session 79: the cube face the texels are in (-1: a 2D texture)
    uint32_t extent_start = 0, extent_length = 0;  // the guest memory the copy writes
    uint64_t dest_offset = 0;                      // the destination descriptor's buffer range (Session 56: 64-bit,
    uint32_t dest_range = 0;                       // a scaled offset passes 4 GiB at 3x3 - review)
    uint32_t base = 0, pitch = 0;             // copy_dest_base, pitch in 32-texel units
    uint32_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;  // the copied rectangle in destination texels
    uint32_t dest_base = 0;                   // the copy's dest_base, relative to dest_offset
    uint32_t endian = 0;                      // xenos::Endian of the destination
    uint32_t tex_x0 = 0, tex_y0 = 0;          // texture texel = destination texel + this
    bool bpb64 = false;                       // 64 bits a texel (else 32)
    uint64_t frame = 0;
    // Session 56 (gpu_native_truth_skip_writeback_scaled): a resolve at a resolution scale - its output goes into the
    // scaled resolve buffer (dest_offset / dest_range there; scaled_use_* the range written), x0 ... y1 host texels as
    // kNativeResolveScaledComputeShader writes them, pitch in guest texels.
    uint32_t scale_x = 1, scale_y = 1;
    uint64_t scaled_use_start = 0, scaled_use_length = 0;
    uint32_t guest_x0 = 0, guest_y0 = 0, guest_x1 = 0, guest_y1 = 0;  // its rectangle in guest texels (x0 ... as 1x)
    // Session 80 (gpu_native_truth_skip_writeback_depth): a 1x depth copy (k_24_8_FLOAT, 20e4) - the texture holds its
    // depth as the 32-bit floats the fill decoded, the stencil byte of copy texel (x, y) is byte y * stencil_pitch + x
    // of stencil slot `stencil_slot`, or `stencil_const` for all (slot -1: the image's last clear).
    bool depth = false;
    int32_t stencil_slot = -1;
    uint32_t stencil_const = 0, stencil_pitch = 0;
  };
  std::vector<WbRecord> wb_records_;
  // Session 80 (gpu_native_truth_skip_writeback_depth): the depth records' stencil snapshots, kStencilSlots of
  // kStencilSlotSize bytes in one buffer (a bit a slot in use); the snapshot shader (TruthStoreSet's stencil view into
  // a slot) and the depth materialise (the slot, the destination, the texture's R32_UINT view).
  static constexpr uint32_t kStencilSlots = 8;
  static constexpr uint32_t kStencilSlotSize = 1u << 20;
  VkBuffer stencil_snap_ = VK_NULL_HANDLE;
  VkDeviceMemory stencil_snap_memory_ = VK_NULL_HANDLE;
  uint32_t stencil_slots_busy_ = 0;
  VkPipeline stencil_snap_pipeline_ = VK_NULL_HANDLE;
  VkPipeline materialize_depth_pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout materialize_depth_layout_ = VK_NULL_HANDLE;
  bool depth_wb_failed_ = false;
  bool EnsureDepthWritebackObjects();
  bool TruthDepthTextureOnly(const draw_util::ResolveInfo& info, TruthImage& t, int16_t index, uint32_t src_x,
                             uint32_t src_y, uint32_t w, uint32_t h, bool all_stencil, uint32_t dest_base,
                             uint32_t dest_coordinate_info, uint32_t dest_info);
  // A descriptor over one stencil slot (transient: this submission).
  VkDescriptorSet StencilSlotSet(int32_t slot);
  bool truth_skip_writeback_ = false;        // latched per frame
  bool truth_wb_skipped_ = false;            // the current resolve skipped its write-back (pending record)
  WbRecord truth_wb_pending_;
  uint64_t truth_dest_offset_ = 0;  // (Session 56 review: 64-bit)
  uint32_t truth_dest_range_ = 0;
  VkPipeline materialize_pipeline_[3] = {};  // 32 / 64 bits a texel; Session 56: [2] at a resolution scale
  bool materialize_failed_ = false;
  bool materialize_scaled_failed_ = false;  // Session 56 (EnsureMaterializeScaledPipeline)
  // What TruthFillTexture's last successful fill wrote (for a texture-only record), and whether this resolve's fill
  // was already tried (TruthFillFirst).
  struct FillDone {
    void* texture = nullptr;
    uint32_t level = 0, texture_w = 0, texture_h = 0;
    int32_t dst_x = 0, dst_y = 0, w = 0, h = 0, copy_w = 0, copy_h = 0;
    bool has_mips = false, depth = false;
  } truth_fill_done_;
  bool truth_fill_tried_ = false;
  bool truth_overlaps_done_ = false;  // TruthResolveOverlaps ran for this resolve
  bool EnsureMaterializeScaledPipeline();  // Session 56
  bool EnsureMaterializePipeline();
  void TruthMaterializeRecord(const WbRecord& record);
  void TruthRemoveRecords(const std::vector<size_t>& indices);  // (indices ascending; unpins, fixes the skipped pages)
  struct WbCounts {
    uint32_t skipped = 0, skipped_direct = 0, superseded = 0, materialized = 0;
    uint64_t skipped_bytes = 0, materialized_bytes = 0;
    uint32_t by_reason[8] = {};  // materialised by: 0 other 1 vertex 2 index 3 texture 4 memexport 5 resolve dest,
                                 // 6 age, 7 skipping stopped
    uint32_t kept[6] = {};  // write-backs kept: 0 not truth / scaled, 1 no texture, 2 not readable, 3 not inside,
                            // 4 other copy, 5 not a plain fill
    // Session 80 (gpu_native_truth_skip_writeback_depth): depth copies left texture-only (of them with a stencil
    // snapshot), depth records materialised, and depth copies kept by reason: 0 no fill / not inside, 1 no free
    // stencil slot, 2 objects / mixed stencil.
    uint32_t depth_skipped = 0, depth_snapshots = 0, depth_materialized = 0;
    uint32_t depth_kept[3] = {};
  } wb_counts_;
  bool truth_class2_blend_ = false;  // RGBA16_SNORM attachments can blend here
  // TruthResolveDirect: a 1x1 D32S8 image whose stencil and depth views fill the shader's unused coverage and depth
  // bindings.
  VkImage truth_dummy_ds_ = VK_NULL_HANDLE;
  VkDeviceMemory truth_dummy_ds_memory_ = VK_NULL_HANDLE;
  VkImageView truth_dummy_stencil_view_ = VK_NULL_HANDLE, truth_dummy_depth_view_ = VK_NULL_HANDLE;
  bool EnsureTruthDummy();
  bool EnsureTruthStaging();
  bool EnsureWbSetLayout();  // (Session 92, M2d: the three-binding set layout of the fill / write-back sources - owned_wb_objects.cpp; EnsureTruthObjects makes it through this too)
  uint32_t truth_reason_logged_[kTruthReasons] = {};
  uint32_t TruthReason(const Draw& draw, bool check_shader) const;
  static const char* TruthReasonName(uint32_t reason);
  bool ClaimTruth(Draw& draw);
  int16_t TruthFindImage(uint32_t base, uint32_t pitch_tiles, uint32_t kind, uint32_t format);
  bool EnsureTruthObjects();
  bool EnsureTruthImage(int16_t index, uint32_t height);
  // wwe2k14 patch (Session 73, the long frame at every scene start: each truth image made - allocated, viewed, cleared
  // - the first time a scene draws into its surface, ~20 in an entrance's first frames on a phone, 169 ms). The images
  // this install has made (key + rows, <cache>\shaders\shareable\TITLE.native.truth.txt, a line appended when one is
  // made or grows), made again a few per frame from the first truth frames of the next run (gpu_native_truth_prewarm).
  void TruthStorageOpen(const std::filesystem::path& shareable_root, uint32_t title_id);
  void TruthStorageNote(const TruthImage& image);
  void TruthPrewarmImages();
  struct TruthStored {
    uint32_t base, pitch_tiles, kind, format, rows;
  };
  std::FILE* truth_storage_file_ = nullptr;
  std::vector<TruthStored> truth_prewarm_;  // stored, not made yet in this run (taken from the back)
  std::vector<TruthStored> truth_stored_;   // the file's records (largest rows per key)
  uint32_t truth_prewarmed_ = 0;
  // Session 73 (Jarrod: the first entrance of every run made an 80x64 colour + depth pair of rename spares 19 times in
  // one frame): the most spares of each size a run has held ("S kind format width height count" lines in the same
  // file), made again after the stored images.
  struct TruthStoredSpares {
    uint32_t kind, format, width, height, count;
  };
  std::vector<TruthStoredSpares> truth_spare_counts_;    // this install's high-water marks (file + this run)
  std::vector<TruthStoredSpares> truth_spare_prewarm_;   // still to make in this run
  uint32_t truth_spares_prewarmed_ = 0;
  void TruthNoteSpares();
  bool TruthPrewarmSpares();
  bool truth_prewarm_logged_ = false;
  void RetireTruthImage(TruthImage& image);
  void RetireTruthAll();
  VkFramebuffer TruthFramebuffer(int16_t color, int16_t depth, uint32_t& width, uint32_t& height);
  // Session 54 (draw twins): the twin of an image, up to date with it, before a pass draws into it (outside any render
  // pass); the image brought up to date from its twin before anything reads it (ends the pass if it draws into it).
  float truth_draw_ratio_ = -1.0f;  // (TruthDrawRatio: -1 until decided)
  uint32_t truth_ratio_p_ = 1, truth_ratio_q_ = 1;  // the ratio as p / q in lowest terms (exact blit rectangles)
  bool TruthTwins() const { return truth_draw_ratio_ > 0.0f && truth_draw_ratio_ < 1.0f && !truth_twins_off_; }
  bool truth_twins_off_ = false;  // gpu_native_truth_draw_twins_off (diagnostic, latched at a frame end)
  // Session 73 (dynamic resolution): the gpu_native_truth_draw_height the twins are made for (INT32_MIN before the
  // first frame end), and how often it changed.
  int32_t truth_draw_height_applied_ = INT32_MIN;
  uint32_t truth_draw_height_changes_ = 0;
  bool EnsureTruthTwin(int16_t index);
  void TruthSyncRead(int16_t index);
  void TruthRetireTwin(TruthImage& image);
  void TruthBlitTwin(TruthImage& image, bool up);
  // Colour twins into their images at the exact draw ratio (a blit's whole-pixel rectangles stretch a twin whose size
  // is not a whole number of pixels, 2/3 of 1280 = 853.3), never reading the twin's partly drawn last row / column.
  // (Review of an AMD tester's report: AMD's Windows driver cannot blit D32S8 - depth goes through a shader pass
  // there, its stencil by VK_EXT_shader_stencil_export; the colour twins always.)
  bool TruthResampleTwin(TruthImage& image, bool up);
  void TruthCopyTwin(TruthImage& image, bool up);
  bool EnsureTruthResampleObjects();
  bool EnsureTruthDepthResampleObjects();
  bool truth_depth_blit_ = true;            // the depth twins by vkCmdBlitImage (D32S8 blittable), else the pass
  bool truth_depth_stencil_export_ = false;  // the depth pass writes stencil too
  bool truth_depth_resample_failed_ = false;
  VkShaderModule truth_depth_resample_shader_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout truth_depth_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout truth_depth_resample_layout_ = VK_NULL_HANDLE;
  VkRenderPass truth_depth_resample_pass_ = VK_NULL_HANDLE;
  VkPipeline truth_depth_resample_pipeline_ = VK_NULL_HANDLE;
  bool truth_resample_failed_ = false;
  VkShaderModule truth_resample_shaders_[2] = {};  // vertex, fragment
  VkDescriptorSetLayout truth_resample_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout truth_resample_layout_ = VK_NULL_HANDLE;
  VkRenderPass truth_resample_passes_[3] = {};  // by colour kind (1 RGBA8, 2 RGBA16_SNORM)
  VkPipeline truth_resample_pipelines_[3] = {};
  // Session 46 (M5): direct recording.
  void TruthRecordDraw(const Draw& draw);
  // Where a truth draw writes, in its images' host pixels (x0 y0 x1 y1, not clamped to the images).
  void TruthDrawRect(const Draw& draw, int32_t rect[4]) const;
  bool truth_pass_open_ = false;
  // Session 80 (gpu_native_own_steps): while the open pass has drawn only clears of one known colour - their union
  // (its bounding box and the pieces' summed area: the box is all cleared when they are equal) and that colour.
  struct ClearsInfo {
    bool valid = false;
    int32_t rect[4] = {};
    uint64_t area = 0;
    float color[4] = {};
    // (Session 80, owned passes: what they did to depth and stencil - whether any used the depth image; whether every
    // one wrote depth whole, to one known value; whether any used stencil, and whether every one wrote it whole, to one
    // value.)
    bool uses_depth = false, depth = false, stencil_touched = false, stencil = false;
    // (And whether every one stored its colour as gpu_native_truth_clear_load takes it: the target's own rounding, no
    // epilogue work - the 16-bit targets' clear value is then known exactly.)
    bool quantized = false;
    float depth_value = 0.0f;
    uint8_t stencil_value = 0;
  } truth_pass_clears_;
  void ClearsAccumulate(ClearsInfo& pc, const Draw& draw, const int32_t rect[4], bool first, bool writes_color,
                        bool uses_depth, uint32_t kind);
  // (Session 80, owned passes: where the open pass's draws start - after the loads recorded at its start - and whether
  // it loaded any tile; where the last pass's end command is.)
  size_t truth_pass_draws_offset_ = SIZE_MAX, truth_pass_end_cmd_offset_ = SIZE_MAX;
  bool truth_pass_loaded_ = false;
 public:
  bool SegmentLogging() const { return segment_log_frames_ > 0; }  // Session 56 (diagnostics elsewhere)
 private:
  std::vector<void*> truth_keep_textures_;  // Session 56: scratch (the deferred resolves' textures)
  bool truth_pass_lazy_ = false;  // Session 56: opened with CmdVkBeginRenderPassLazy (gpu_native_truth_lazy_pass_deps)
  uint64_t truth_lazy_free_ = 0, truth_lazy_synced_ = 0;  // Session 56: the counts at the last report
  int16_t truth_pass_color_ = -1, truth_pass_depth_ = -1;
  uint64_t truth_pass_first_vs_ = 0;  // Session 80 (gpu_native_truth_serve_ahead): the vertex shader that opened it
  uint32_t truth_pass_class_ = 0, truth_pass_pitch_ = 0, truth_pass_width_ = 0, truth_pass_height_ = 0;
  uint32_t truth_pass_segment_index_ = UINT32_MAX;
  // The tiles the open pass's images own (kept as they draw), and the tiles of the last draw rectangle (CPU time).
  std::bitset<2048> truth_pass_c_owned_, truth_pass_d_owned_;
  // ... and the tiles they hold a valid copy of (owned, or loaded for a read without being written: review fix).
  std::bitset<2048> truth_pass_c_valid_, truth_pass_d_valid_;
  bool truth_foreign_bind_ = false;
  int32_t truth_tile_rect_[4] = {};
  int16_t truth_tile_color_ = -1, truth_tile_depth_ = -1;
  std::bitset<2048> truth_tile_c_, truth_tile_d_;
  // wwe2k14 patch (Session 79, gpu_native_truth_alias_census): alias-loaded tiles of the open pass not yet read or
  // overwritten whole, and per 300 frames: tiles found dead (a later draw of the pass covered them whole without
  // reading), consumed (a later draw read them), kept (still pending at the pass end).
  std::bitset<2048> census_pend_c_, census_pend_d_;
  uint64_t census_dead_ = 0, census_consumed_ = 0, census_kept_ = 0, census_loaded_ = 0;
  uint32_t census_frames_ = 0;
  // (Session 79: the alias-loaded tiles by owner -> pass image: kind / format / pitch of both, and whether the tile sits
  // at the same place in both - a plain copy of the same bits could replace its store and load.)
  std::unordered_map<std::string, uint64_t> census_pairs_;
  // Session 47 (CPU): the open pass's images own every tile of truth_tile_c_ / _d_ (TruthRecordDraw's fast path).
  bool truth_tile_c_owned_ = false, truth_tile_d_owned_ = false;
  // Session 47 (CPU): claims of this epoch hold (a new epoch at every frame end, fallback and image reset); the last
  // truth draw recorded its draw command (ClaimTruthRepeat, TruthLastDrawRecorded).
  uint32_t truth_claim_epoch_ = 1;
  bool truth_last_recorded_ = false;
  // wwe2k14 patch (Session 80, own crowd): the last recorded truth draw's vertex shader and index buffer.
  uint64_t truth_last_vs_hash_ = 0;
  struct {
    bool indexed = false;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkIndexType type = VK_INDEX_TYPE_UINT16;
    uint32_t count = 0;
    bool attribute = false;  // (Session 95: an attribute draw's: host-order indices in the index pool, which own crowd must not reuse)
  } truth_last_index_;
  // Session 95 (P7): the index a run draws with: the first member's guest-order buffer (the storage form), or the pool's (attr: mesh and members as vertex attributes).
  struct OwnCrowdIndexUse {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkIndexType type = VK_INDEX_TYPE_UINT16;
    uint32_t count = 0;
    int32_t vertex_offset = 0;
    bool attr = false, clear_restart = false;
  };
  OwnCrowdIndexUse OwnCrowdIndexFor(const OwnCrowdInput& in);
  void OwnCrowdNoteRun(const OwnCrowdIndexUse& use, uint32_t members);
  OwnCrowdViIndex own_crowd_vi_index_;
  bool own_crowd_ring_vi_ = false;  // gpu_native_crowd_input with vertex input on, as it stood when the ring was made: the ring can be bound as vertex buffers
  // wwe2k14 patch (Session 80, own crowd; own_crowd.cpp): our objects.
  bool OwnCrowdEnsureObjects();
  void OwnCrowdDestroy();
  VkShaderModule OwnCrowdModule(const uint32_t* words, size_t count);
  // wwe2k14 patch (Session 93, step 5 idea 5; gpu_native_own_crowd_lean / _merge): the lean call (own_crowd.cpp).
  bool RecordOwnCrowdLean(const OwnCrowdInput& in, bool in_owned, const OwnCrowdIndexUse& use);
  VkShaderModule OwnCrowdModuleByPointer(const uint32_t* words, size_t count);  // (no hashing: the program's vectors never move)
  bool own_crowd_hook_seen_ = false;
  bool own_crowd_lean_ = false;  // gpu_native_own_crowd_lean as it stood when the objects were made
  VkDescriptorSetLayout own_crowd_dyn_set_layout_ = VK_NULL_HANDLE;  // buffers by dynamic offset (bindings 0, 4, 5)
  VkPipelineLayout own_crowd_dyn_layout_ = VK_NULL_HANDLE;
  std::unordered_map<const uint32_t*, VkShaderModule> own_crowd_module_ptr_;
  // The call recorded last (the merge compares a new run with it; the counters compare even with the merge off).
  // Session 93 (gpu_native_own_crowd_merge): the dynamic state a crowd run's first member recorded (truth pass or owned pass), compared
  // between runs - equal state and nothing but state commands between two calls lets the later run join the earlier call.
  struct OwnCrowdStateSig {
    VkViewport viewport = {};
    VkRect2D scissor = {};
    float bias[2] = {};
    float blend[4] = {};
    bool stencil_on = false;
    uint8_t stencil[2][3] = {};
    bool Equals(const OwnCrowdStateSig& o) const {
      return std::memcmp(&viewport, &o.viewport, sizeof(viewport)) == 0 && std::memcmp(&scissor, &o.scissor, sizeof(scissor)) == 0 &&
             std::memcmp(bias, o.bias, sizeof(bias)) == 0 && std::memcmp(blend, o.blend, sizeof(blend)) == 0 &&
             stencil_on == o.stencil_on && std::memcmp(stencil, o.stencil, sizeof(stencil)) == 0;
    }
  };
  OwnCrowdStateSig own_crowd_sig_;       // (the last first member's, valid while own_crowd_sig_valid_)
  bool own_crowd_sig_valid_ = false;     // set by a draw with own_skip_call (truth or owned record), cleared by every other draw and by RecordOwnCrowd
  bool own_crowd_call_sig_valid_ = false;  // (RecordOwnCrowd's copy of the flag for the run in hand)
  void OwnCrowdNoteSig(const VkViewport& viewport, const VkRect2D& scissor, float bias_constant, float bias_slope,
                       const float blend[4], bool stencil_on, const uint8_t refs[2], const uint8_t read_masks[2],
                       const uint8_t write_masks[2]) {
    own_crowd_sig_ = {};
    own_crowd_sig_.viewport = viewport;
    own_crowd_sig_.scissor = scissor;
    own_crowd_sig_.bias[0] = bias_constant;
    own_crowd_sig_.bias[1] = bias_slope;
    std::memcpy(own_crowd_sig_.blend, blend, sizeof(own_crowd_sig_.blend));
    own_crowd_sig_.stencil_on = stencil_on;
    if (stencil_on) {
      for (int f = 0; f < 2; ++f) {
        own_crowd_sig_.stencil[f][0] = refs[f];
        own_crowd_sig_.stencil[f][1] = read_masks[f];
        own_crowd_sig_.stencil[f][2] = write_masks[f];
      }
    }
    own_crowd_sig_valid_ = true;
  }
  struct OwnCrowdOpen {
    bool valid = false, in_owned = false, sig_valid = false;
    OwnCrowdStateSig sig;
    uint64_t submission = 0, mesh_hash = 0;
    uint32_t slice = 0;
    PipelineKey key = {};
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkImageView views[3] = {};
    VkSampler samplers[3] = {};
    float scene[112] = {};
    VkBuffer index_buffer = VK_NULL_HANDLE;
    VkDeviceSize index_offset = 0;
    VkIndexType index_type = VK_INDEX_TYPE_UINT16;
    uint32_t index_count = 0, member_count = 0;
    int32_t vertex_offset = 0;         // (Session 95, P7)
    VkDeviceSize members_at = 0;       // in the ring buffer
    VkDeviceSize members_end_rel = 0;  // slice->used right after the members were taken (they are the last thing taken)
    size_t stream_end = 0;             // the deferred stream's size right after the draw
    VkDescriptorSet set = VK_NULL_HANDLE;
  } own_crowd_open_;
  struct {
    uint64_t calls = 0, eq_prev = 0, eq_prev_adjacent = 0, merged_runs = 0, merged_members = 0, sets_reused = 0,
             sets_made = 0, flushes = 0, flush_kb = 0, program_hits = 0, size_declines = 0;
    // (Session 93, the merge) runs equal to the previous call that were not merged, by why: the first member's state differs / the
    // stream holds more than state since the call / the ring or the owned pass's call list did not fit.
    uint64_t merge_no_state = 0, merge_no_stream = 0, merge_no_fit = 0, merge_no_list = 0;
  } own_crowd_lean_stats_;
  std::vector<std::pair<uint64_t, const void*>> own_crowd_program_cache_;  // (vertex shader hash -> OwnCrowdProgram*; Session 93)
  uint32_t own_crowd_program_cache_gen_ = 0;

 public:
  // (Session 93) the lean call's program lookup without the mutex scan: the command processor keeps the cache here.
  std::vector<std::pair<uint64_t, const void*>>& OwnCrowdProgramCache(uint32_t gen) {
    if (gen != own_crowd_program_cache_gen_) {
      own_crowd_program_cache_.clear();
      own_crowd_program_cache_gen_ = gen;
    }
    return own_crowd_program_cache_;
  }

 private:
  void OwnCrowdConvertMesh(const OwnCrowdInput& in, float* out);
  bool own_crowd_failed_ = false;
  IndexPool* vertex_mirror_ = nullptr;  // Session 95 (item D): made by Mirror(), deleted in DestroyObjects
  VertexInputState* vertex_input_ = nullptr;  // Session 95: made by VertexInput(), deleted in DestroyObjects
  VertexMirror* skin_mirror_ = nullptr;  // Session 95: made by SkinMirror(), deleted in DestroyObjects
  SkinOnce* skin_once_ = nullptr;  // Session 95: made by Skin(), deleted before the mirror
  VkDescriptorSetLayout own_crowd_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout own_crowd_layout_ = VK_NULL_HANDLE;
  VkBuffer own_crowd_ring_ = VK_NULL_HANDLE;
  VkDeviceMemory own_crowd_ring_memory_ = VK_NULL_HANDLE;
  void* own_crowd_ring_map_ = nullptr;
  VkDeviceSize own_crowd_ring_total_ = 0;  // Session 93: the ring buffer's size (the lean ring is padded by the larger dynamic range)
  struct OwnCrowdSlice {
    struct Mesh {  // (Session 87, crowd2) a converted mesh this slice holds: the hash of its guest bytes, its place in the ring
      uint64_t hash;
      VkDeviceSize at;
    };
    uint64_t submission = 0;
    VkDeviceSize used = 0;
    uint32_t sets = 0;  // (Session 87, crowd2: descriptor sets allocated from the pool this submission)
    // Session 93 (gpu_native_own_crowd_lean): the range written and not flushed yet (ring offsets), and the sets made this submission
    // by their textures (the buffers are by dynamic offset).
    VkDeviceSize dirty_lo = VK_WHOLE_SIZE, dirty_hi = 0;
    struct SetEntry {
      VkImageView views[3];
      VkSampler samplers[3];
      VkDescriptorSet set;
    };
    std::vector<SetEntry> set_cache;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    std::vector<Mesh> meshes;
  };
  std::vector<OwnCrowdSlice> own_crowd_slices_;
  VkDeviceSize own_crowd_slice_size_ = 0;  // (Session 87, crowd2 review) set when the objects are made
  uint32_t own_crowd_sets_ = 0;
  uint32_t own_crowd_slice_ = 0;
  std::unordered_map<uint64_t, VkShaderModule> own_crowd_modules_;
  std::unordered_map<PipelineKey, VkPipeline, PipelineKeyHasher> own_crowd_pipelines_;
  struct {
    uint64_t frames = 0, runs = 0, members = 0, vertices = 0, refused_state = 0, refused_ring = 0,
             refused_objects = 0, card_runs = 0, card_members = 0;  // (Session 87: the flat cards' share)
    // Session 87, crowd2 (sizing / cost census): the peak a submission used of its slice (bytes, sets), meshes converted vs
    // shared inside the submission, and meshes converted that an earlier submission had converted too (a cross-submission
    // cache would not convert them again) with the bytes that would save.
    uint64_t peak_bytes = 0, peak_sets = 0, mesh_converted = 0, mesh_shared = 0, mesh_repeat = 0, mesh_repeat_bytes = 0,
             mesh_bytes = 0, submissions = 0;
  } own_crowd_stats_;
  std::unordered_set<uint64_t> own_crowd_seen_meshes_;  // hashes of converted meshes (reset at 4096)
  // wwe2k14 patch (Session 82, P1 item 1: own crowd inside an owned pass): our crowd draw as recorded - the pipeline,
  // set and index buffer it bound - so the draw of an owned pass can be recorded again into the truth images if the
  // pass is lost (the set lives in the submission's ring slice: the replay is in the same submission).
  struct OwnedCrowdCall {
    size_t after_draws = 0;  // the pass's draws recorded before it (its first member's draw is the last of them)
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkBuffer index_buffer = VK_NULL_HANDLE;
    VkDeviceSize index_offset = 0;
    VkIndexType index_type = VK_INDEX_TYPE_UINT16;
    uint32_t index_count = 0, member_count = 0;
    bool dynamic = false;  // Session 93 (lean): the set's buffers 0, 4, 5 are bound by these offsets
    uint32_t dyn_offsets[3] = {};
    // Session 95 (P7, gpu_native_crowd_input): binding 0 the mesh, binding 1 the members (per instance), both in the ring; vb_count 0 = the storage form.
    uint32_t vb_count = 0;
    VkBuffer vb[2] = {};
    VkDeviceSize vb_offsets[2] = {};
    int32_t vertex_offset = 0;
  };
  void OwnCrowdRecordCall(const OwnedCrowdCall& call);  // (own_crowd.cpp: bind + draw into the open pass, truth or owned)
  // Session 82 (CPU): what the open owned render pass has bound - a draw records only what differs. Known only inside one
  // render pass of ours (dropped at its begin, and after a crowd run's first member: the run records its own state).
  struct OwnedBound {
    bool valid = false;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSet sets[SpirvShaderTranslator::kDescriptorSetCount] = {};
    uint32_t dynamic_count = 0;
    uint32_t dynamic[SpirvShaderTranslator::kConstantBufferCount] = {};
    uint64_t skin_side_key = 0;  // (Session 95, r3gpu) the constants on `dynamic` serve every draw with this key
    // (Session 84: no viewport / scissor here - those go through the command processor's tracking, OwnedRecordDraw.)
    float bias[2] = {};
    float blend[4] = {};
    bool stencil_valid = false;
    uint8_t stencil[2][3] = {};
  } owned_bound_;
  uint64_t owned_draw_serial_ = 0;  // (the owned draws recorded so far: a pass image's state)
  // Session 82 (gpu_native_owned_alpha_learn): the scene's blended arena draws whose alpha is 1 everywhere are drawn unblended.
  // A candidate (a draw of a swapped shader with the alpha blend 0x07060706) is probed - a second draw of the same geometry
  // with the shader's alpha probe (the fragments whose 8-bit alpha is 1 discarded), colour writes and depth off, inside an
  // occlusion query - and its key is proven after two probes that counted nothing; a proven key's draw uses the pipeline
  // with its blend off, and is probed again every few frames; a probe that counts anything sends it back to blended.
  static constexpr uint32_t kProbesPerFrame = 256;
  VkQueryPool probe_pool_ = VK_NULL_HANDLE;
  bool probe_slot_ok_ = false;  // this frame's probe range was reset
  uint32_t probe_issued_ = 0;
  uint64_t alpha_evict_frame_ = 0;
  struct OwnedAlphaKey {
    uint32_t clean = 0;
    bool proven = false;
    uint64_t last_probe = 0;
    uint32_t fails = 0;                   // (probes that counted a translucent fragment: backs the key off)
    uint64_t last_clean = UINT64_MAX;     // (the frame of the probe that last counted clean: two probes of one frame are one)
    uint64_t last_seen = 0;  // (the frame a draw of the key was last recorded: stale keys are dropped)
  };
  std::unordered_map<uint64_t, OwnedAlphaKey> owned_alpha_keys_;
  std::unordered_map<uint64_t, VkShaderModule> owned_probe_modules_;
  bool EnsureProbePool();
  bool OwnedAlphaCandidate(const Draw& draw) const;
  uint64_t OwnedAlphaKeyOf(const Draw& draw) const;
  VkPipeline OwnedAlphaPipeline(const Draw& draw);
  void OwnedAlphaProbe(const Draw& draw);
  void OwnedAlphaRead(FrameSlot& slot);
  void OwnedAlphaMaintain();
  bool owned_crowd_first_ = false;  // the draw OwnedRecord just recorded is a crowd run's first member (state only)
  // wwe2k14 patch (Session 82, the owned match frame; owned_steps.cpp; docs\native\OWNED_FRAME_DESIGN.md sections 0,
  // 3.6 and 4): with gpu_native_owned_frame on, every truth draw is a candidate for an owned pass (the list is
  // ignored); a pass that is not owned is a counted fallback with ONE reason from this list, and the frame's passes are
  // logged as the census.
  enum OwnedReason : uint8_t {
    kOwnedReasonNone = 0,
    kOwnedReasonStart,       // no start our image can be given (not cleared, not covered; clear-only)
    kOwnedReasonMsaa,        // a multisampled surface
    kOwnedReasonFormat,      // a surface format with no owned image class
    kOwnedReasonReadback,    // a readback window (the whole frame)
    kOwnedReasonLost,        // a draw / copy / write-out over its tiles before its commit: replayed exactly
    kOwnedReasonReader,      // a sampler of its page that the substitution refuses
    kOwnedReasonVertexFetch, // a vertex fetch of an owned page (not detected yet)
    kOwnedReasonPool,        // no image free within the budget
    kOwnedReasonPaused,      // the pass's key is paused after a loss
    kOwnedReasonInputServed, // a reader of a truth pass's served copy
    kOwnedReasonNoPipeline,  // (not a fallback: the draw is skipped as the truth path skips it)
    kOwnedReasonClearsKept,  // the pass is owned but its clears' pass still runs
    kOwnedReasonFailed,      // owned images cannot be made (a hard failure)
    kOwnedReasonPresent,     // a copy into the front buffer
    kOwnedReasonOther,       // other:<text>
    kOwnedReasonDead,        // (Session 92, M2c: a pass never committed whose only copies were dropped as dead - counted apart from "start")
    kOwnedReasonCount
  };
  static const char* OwnedReasonName(uint32_t reason);
  enum OwnedStartKind : uint8_t {
    kOwnedStartNone = 0, kOwnedStartCleared, kOwnedStartCovered, kOwnedStartResumed, kOwnedStartConverted
  };
  // A pass's identity for pauses: its surface and the first pixel shader of its owned draws.
  struct OwnedPassKey {
    uint32_t base = 0, pitch = 0, format = 0;
    uint64_t ps = 0;
    bool operator==(const OwnedPassKey& o) const {
      return base == o.base && pitch == o.pitch && format == o.format && ps == o.ps;
    }
  };
  // Session 93 (gpu_owned_tiler_passes, owned_steps.cpp): the tile-based render pass path.
  struct OwnedDepthLearn {
    OwnedPassKey key;
    uint32_t unread = 0;     // passes of this key in a row whose depth nothing read after them
    bool dead = false;       // unread for K passes: its next passes are begun with the depth store DONT_CARE
    bool banned = false;     // a read of a depth stored dead (a learned wrong): never dead again
    uint64_t last_frame = 0;      // Session 94 (gpu_owned_tiler_scene_reset): the frame of its last pass begin
    uint64_t streak_frame = 0;    // the frame the current unread streak began
    bool read_ever = false;       // any depth read of this key, ever (survives scene resets): never dead again
  };
  // Session 94 (gpu_owned_tiler_scene_reset): the scene-change proxy - new keys in a burst - and the quiet time after a reset.
  uint64_t tl_quiet_until_ = 0, tl_ring_[16] = {};  // (the frames of the last 16 key creations, a sliding window)
  uint32_t tl_ring_pos_ = 0, tl_ring_count_ = 0, tl_resets_ = 0;
  void OwnedTilerSceneReset(const char* why);
  struct OwnedPassKeyHash {
    size_t operator()(const OwnedPassKey& k) const {
      uint64_t h = k.ps * 0x9E3779B97F4A7C15ull;
      h ^= (uint64_t(k.base) << 32 | k.pitch) + 0x7F4A7C15ull + (h << 6) + (h >> 2);
      h ^= uint64_t(k.format) * 0xC2B2AE3D27D4EB4Full + (h << 6) + (h >> 2);
      return size_t(h);
    }
  };
  std::vector<OwnedDepthLearn> owned_depth_learn_;
  std::unordered_map<OwnedPassKey, int32_t, OwnedPassKeyHash> owned_depth_learn_index_;
  bool OwnedTilerOn() const;
  // The render pass variant for a pass beginning: resumes of passes without depth of their own take ResumeNoDepth; a first begin of a pass that keeps depth
  // is accounted (the previous pass's depth in the image) and may take ClearBothDead. `first`: the pass begins (not resumes).
  uint32_t OwnedTilerVariant(int16_t image, uint32_t variant, bool has_depth, bool first, const OwnedPassKey* key);
  void OwnedDepthAccount(int16_t image);   // the pass whose depth the image holds is over: unread or read
  void OwnedNoteDepthRead(int16_t image, const char* site);  // something reads (or loads) the image's depth (site: for the learned-wrong line)
  // One pass of the frame as the census sees it: a truth pass (opened by TruthRecordDraw) or an owned pass (started
  // by OwnedTakeDraw - from the clears of an open truth pass it is that pass's record).
  struct OwnedFramePass {
    bool owned = false;        // an owned pass that was committed
    bool lost = false;         // an owned pass replayed into the truth images
    uint8_t reason = kOwnedReasonNone;  // the first refusal (a fallback's reason); none = undecided
    uint8_t start = kOwnedStartNone;
    bool clears_kept = false;
    std::string detail;        // the text of `other:`, or what a loss / refusal was
    uint32_t base = 0, pitch = 0, format = 0;
    uint32_t box_w = 0, box_h = 0;   // guest pixels
    uint32_t draws = 0;
    uint64_t vs = 0, ps = 0;         // the first draw
    uint32_t copies = 0, served = 0, page = 0, dead = 0, written_back = 0;
  };
  std::vector<OwnedFramePass> owned_frame_passes_;  // this frame's, in order
  int32_t owned_frame_last_ = -1;                    // the truth pass being recorded (the last record made)
  uint8_t owned_frame_pending_reason_ = kOwnedReasonNone;  // the last draw's refusal, for the pass it lands in
  std::string owned_frame_pending_detail_;
  bool owned_frame_on_ = false;       // the switch as of this frame's start (OwnedRefreshList, at the frame's end)
  bool owned_frame_logging_ = false;  // this frame's passes are logged
  int32_t owned_frame_log_seen_ = INT32_MIN, owned_frame_log_left_ = 0;
  std::vector<std::pair<OwnedPassKey, uint64_t>> owned_paused_;  // (key, until frame)
  std::vector<OwnedPassKey> owned_present_keys_;                 // passes whose copy went to the front buffer
  struct OwnedSubLoss {  // (a key's losses at a submission's end in a row: a pause only after a few)
    OwnedPassKey key;
    uint64_t frame = 0;
    uint32_t count = 0;
  };
  std::vector<OwnedSubLoss> owned_sub_losses_;
  uint32_t owned_front_pages_[4] = {};                           // the last pages the swap presented
  uint32_t owned_front_next_ = 0;
  uint64_t owned_bytes_ = 0;  // owned images, scratch depths and page images (estimate), for the budget
  struct OwnedFrameStats {
    uint32_t frames = 0;
    uint64_t first_frame = 0;
    uint32_t passes = 0, owned = 0, fallback = 0;
    uint32_t reasons[kOwnedReasonCount] = {};
    uint32_t copies = 0, served = 0, page = 0, dead = 0, written_back = 0;
    uint32_t pauses = 0, reader_events = 0, flip_kept = 0, clears_kept = 0, no_pipeline = 0;
    uint32_t presented = 0, post_boxes = 0, sub_commits = 0;  // (Session 82, P2: swaps given our image; copies clamped to their box)
    std::vector<std::pair<std::string, uint32_t>> lost_why;  // (why the lost passes were lost, counted)
    // (Session 87: the refusals that become fallback passes, grouped by the draw's signature - which draws the owned frame
    // does not take, in the window's summary line; at most 256 distinct signatures a window.)
    std::vector<std::pair<std::string, uint32_t>> refusal_groups;
    // (Session 87: the fallback PASSES themselves - truth passes opened without an owned pass - grouped by what the census
    // knows of them: reason, start kind, surface, box, draw count, first draw's shaders.)
    std::vector<std::pair<std::string, uint32_t>> fallback_groups;
    // (Session 87: depth-only clears recorded into an open owned pass, and why the others were not: no pass / not open / other depth / not whole.)
    uint32_t dclear_pending = 0, dclear_flushed = 0, dclear_dropped = 0, dclear_superseded = 0, dclear_read_after_drop = 0;
    // (Session 87, gpu_native_owned_frame_chains: clears into our images (held draws begun as a pass), colour + depth / colour only, by class;
    // passes begun, begun-and-declined (back to the truth path), uncommitted passes parked, parked passes a copy took up, copies of passes
    // begun by a clear, truth copies that followed a dirty copy-in.)
    uint32_t ch_clears = 0, ch_clears_cd = 0, ch_clears_c = 0, ch_class1 = 0, ch_class2 = 0, ch_started = 0, ch_declined = 0;
    uint32_t ch_parked = 0, ch_takenup = 0, ch_served = 0, ch_materialised = 0;
    // (chains, the ends: passes begun from clears held at a submission's end / a frame's end / a write-out, uncommitted passes committed
    // there instead of replayed, truth draws that replayed the pending pass of their surface.)
    // (chains: depth-only clears merged into held colour clears of the same box; copy losses by source line.)
    uint32_t ch_parked_replays = 0, ch_dead_ended = 0, images_made = 0, fill_sets_reused = 0;
    // (Session 93, tiler passes: resumes with / without depth of their own, resumes that took the no-depth variant, depth-keeping passes begun, begun with
    // the depth store DONT_CARE, begun with a key already learned dead [counted with the switch off too], passes over with their depth unread, reads of a depth
    // stored dead.)
    uint32_t tl_resume_depth = 0, tl_resume_nodepth = 0, tl_resume_nodepth_used = 0, tl_depth_passes = 0, tl_dead_stored = 0, tl_dead_would = 0,
             tl_unread_passes = 0, tl_learned_wrong = 0, tl_would_wrong = 0;
    // (Session 90, P: parked defined starts replayed / dropped later, by reason [submission end, frame end, truth draw, write-out, copy, loaded start, other];
    // defined starts carried past a submission's end without their draws; carried ones taken up by a copy / a draw; parked ones dropped at an end unreplayed.)
    uint32_t ch_replay_by[7] = {}, ch_carried_dropped[7] = {};
    uint32_t ch_carried_paused_copy = 0, ch_carried_paused_read[7] = {}, ch_copy_of_dropped_carried = 0;
    uint32_t ch_carried_defined = 0, ch_carried_taken_copy = 0, ch_carried_taken_draw = 0, ch_frame_end_unreplayed = 0;
    uint32_t ch_claims = 0, ch_commit_instead = 0, ch_claim_failed = 0;
    uint32_t ch_readback_mat = 0, ch_dead_depthless = 0, ch_only_ours_readback = 0;
    // (Session 90, Q: read-back copies written from our image by path [8888 1x, 8888 scaled, full 1x, full scaled (incl. 16-bit at 1x), depth];
    // front copies written; parked passes loaded into a new image from their own images; copies served while read back without a plan;
    // write-backs that kept the old replay / were refused, by reason.)
    uint32_t wb_written[5] = {}, wb_front = 0, wb_loaded_parked = 0, wb_unplanned = 0, wb_dest_ours = 0, wb_only_ours = 0, wb_planned = 0, wb_rect = 0, wb_window_pauses = 0, wb_window_loads = 0;
    std::vector<std::pair<std::string, uint32_t>> wb_refused;
    uint32_t k4444_unloaded_wb = 0;  // (Session 92: damage-map resets written back from our image)
    uint32_t ch_dmerge = 0, ch_resumed = 0, ch_loaded = 0, ch_conv016 = 0, ch_front_served = 0, ch_front_lost = 0, ch_swap_unserved = 0;
    std::vector<std::pair<int, uint32_t>> lose_lines;
    uint32_t ch_site_sub = 0, ch_site_frame = 0, ch_site_write = 0, ch_commit_frame = 0, ch_commit_write = 0, ch_draw_replays = 0;
    uint32_t depth_clear_deferred = 0, depth_clear_taken = 0, depth_clear_no_pass = 0, depth_clear_not_open = 0, depth_clear_other_depth = 0, depth_clear_not_whole = 0;
    // (Session 82, speed: what the owned path still pays per frame - dirty-tile copy-ins and the tiles in them, dirty
    // tiles dropped because the next pass overwrote them, converted starts, snapshots.)
    uint32_t mat_calls = 0, mat_tiles = 0, dead_dropped = 0, converted = 0, snapshots = 0;
    // (The truth work still done with the owned frame on - the milestone is none of it in the match frame: passes the
    // truth path recorded, the tiles its loads brought in, its staging stores, the copies it ran, and why each dirty
    // copy-in happened.)
    uint32_t truth_passes = 0, truth_tile_loads = 0, truth_stores = 0, truth_copies = 0;
    // (The state commands an owned draw needed and the ones left out because the pass already had that state.)
    uint64_t rec_state_cmds = 0, rec_state_skipped = 0;
    uint32_t alpha_candidates = 0, alpha_proven = 0, alpha_probes = 0, alpha_reverts = 0;  // Session 82
    uint32_t lean_starts = 0;  // Session 82: pass starts with the lean barrier
    uint32_t strip_batches = 0, strip_faces_answered = 0;  // Session 87 (gpu_native_owned_frame_strip_batch)
    uint32_t dropped_draws = 0, restores_left_out = 0;
    uint32_t detached = 0, restore_drawn = 0, restore_clears_dropped = 0, restore_stencil_cleared = 0, restore_stencil_lost = 0;  // Session 87
    std::vector<std::pair<std::string, uint32_t>> mat_why;
  } owned_frame_stats_;
  // (CPU writes to owned pages: the shared memory's global watch notes the ranges; the render thread drops the pages.)
  void* owned_watch_handle_ = nullptr;
  std::mutex owned_watch_mutex_;
  std::vector<std::pair<uint32_t, uint32_t>> owned_watch_ranges_;
  std::atomic<bool> owned_watch_pending_{false};
  std::vector<std::pair<uint32_t, uint32_t>> owned_watch_extents_;  // (the owned pages' memory, for the callback)
  void OwnedWatchRegister();   // (only with the owned frame on: with it off the list mode is as it was)
  void OwnedWatchSync();
  void OwnedDrainCpuWrites();
  void OwnedRefreshFrame();      // (the frame's end: the switch and the census log window)
  void OwnedFrameSummary();      // (the 600-frame summary line)
  // The census.
  OwnedPassKey OwnedKeyOf(const Draw& draw) const;
  bool OwnedPaused(const OwnedPassKey& key) const;
  void OwnedPause(const OwnedPassKey& key, const std::vector<std::pair<uint64_t, bool>>& producers, const char* why);
  void OwnedNoteRefusal(uint8_t reason, const std::string& detail = std::string());
  int32_t OwnedFrameNewPass(const Draw& draw);
  void OwnedFramePassOpened(const Draw& draw);                   // (TruthRecordDraw: a truth pass opens)
  void OwnedFrameTruthDrawn(const Draw& draw, const int32_t rect[4]);  // (TruthRecordDraw: a draw recorded)
  void OwnedFrameLost(int32_t census, uint8_t reason, const char* why);
  void OwnedFrameFinish();                                       // (the frame's end: lines, summary)
  void OwnedFrameFlip(bool on);
  // Extension points (P0; each later piece defines its own in its own .cpp: owned_start.cpp, owned_copy.cpp,
  // owned_reader.cpp). Start: a draw the base cases (cleared, covered) gave no start; true = it set `owned_pending_`
  // up (image chosen, box, kind) as OwnedTakeDraw does for the base cases. Copy: a copy of the pending pass the base
  // cases (whole -> served view or 16-bit page image, rectangle -> page of ours) do not take; true = handled, `kind`
  // names what it did (served / page / dead / written back). Reader: a binding of an owned page the base cases refuse.
  struct OwnedStartContext {
    const Draw* draw = nullptr;
    const int32_t* rect = nullptr;
    uint32_t scale_x = 1, scale_y = 1;
    uint8_t start = kOwnedStartNone;  // out: the kind it started
    uint8_t reason = kOwnedReasonStart;
    std::string detail;               // out: why not
  };
  enum OwnedCopyKind : uint8_t { kOwnedCopyServed, kOwnedCopyPage, kOwnedCopyDead, kOwnedCopyWrittenBack };
  struct OwnedCopyContext {
    const draw_util::ResolveInfo* info = nullptr;
    bool copy_known = false;
    uint8_t kind = kOwnedCopyPage;    // out
  };
  struct OwnedReaderContext {
    uint32_t base_page = 0, width = 0, height = 0, format = 0, endian = 0;
    bool scaled_ok = true, is_signed = false, servable = true;
    VkComponentMapping mapping = {};
  };
  bool OwnedStartExtra(OwnedStartContext& ctx);              // owned_start.cpp
  // P0b (2), the converted start (owned_start.cpp): a draw that is not cleared or covered, from the surface's origin,
  // no depth: the truth path loads its tiles into the truth images as it would for a pass of its own (aliased tiles
  // from their owners - the dirty ones given our pixels first - EDRAM tiles from the EDRAM) and the pass ends before
  // the draw; the box is copied into our image, the box's tiles are claimed for the colour image (our image holds
  // what they will hold), and our pass goes on from there (dirty overlay, copies, pauses as any owned pass).
  bool OwnedConvertible(const Draw& draw, const int32_t rect[4], bool depth_used, uint32_t scale_x,
                        uint32_t scale_y) const;
  // Session 82 (the owned frame): the colour clears of a surface, held until a consumer takes them (the cleared start of
  // an owned pass: they are then never recorded in the truth path at all) or something else needs the truth path to have
  // run them (OwnedFlushHeld: recorded then, in their order, before it).
  struct OwnedHeldClears {
    std::vector<Draw> draws;
    ClearsInfo info;
    int16_t color = -1, depth = -1;
    uint32_t kind = 0;
  } owned_held_;
  bool owned_flushing_held_ = false;
  bool OwnedHoldClear(const Draw& draw);
  // (Session 87: `site` is the caller's line, for the census detail of the truth passes the flushed clears open.)
  void OwnedFlushHeld(bool as_pass = false, int site = __builtin_LINE());
  uint8_t owned_chain_site_ = 0;  // (chains: which end the flush comes from - 1 submission, 2 frame, 3 write-out; census only)
  std::string owned_flush_note_;  // (Session 87: why the caller's `cleared` was false, appended to that detail)
  void OwnedHeldBeforeTextures();
  const char* owned_mat_why_ = "other";  // (what the next OwnedMaterialize is for - counted by reason)
  // Session 82, P1 (owned_scene.cpp): the scene's depth copy into a depth page of ours. OwnedDepthCopy: the copy of the
  // pending pass's depth (k_24_8_FLOAT, the whole box into a texture's origin) left out - the page is an R32_SFLOAT page
  // image filled by the truth fill's depth conversion (mode 2) with our depth image as its source; false = not taken
  // (`why` says why; nothing was recorded or changed: the caller loses the pass as before).
  bool OwnedDepthCopy(const draw_util::ResolveInfo& info, uint32_t page, uint32_t texture_page, std::string& why);
  // (Session 87: from_read - the image's depth is in SHADER_READ_ONLY (a direct depth page samples it), not an attachment.)
  bool OwnedFillDepth(int16_t page_image, int16_t image, uint32_t width, uint32_t height, uint32_t gap, bool from_read = false);
  void OwnedUnmapExtent(uint32_t start, uint32_t length);
  // Session 82 (P1): GPU timestamps of the owned work in the pass log (gpu_native_truth_pass_log): a segment of kind 16
  // with a label - zero cost when the log is off. OwnedTimerBegin returns the segment (UINT32_MAX = not timed).
  uint32_t OwnedTimerBegin(const char* label, uint32_t width = 0, uint32_t height = 0);
  void OwnedTimerEnd(uint32_t segment, uint32_t draws = 0);
  bool OwnedCopyExtra(OwnedCopyContext& ctx);                // owned_copy.cpp
  VkImageView OwnedReaderExtra(const OwnedReaderContext& ctx);  // owned_reader.cpp
  // P0b (1), the per-surface cleared state: a truth pass that was only clears (truth_pass_clears_ valid, kept after the
  // pass ends) is a surface "cleared and untouched" until something touches its tiles - a copy of them, a write-out,
  // another pass (truth or owned) over them, the submission's end. The next draw into that surface may start an owned
  // pass from it exactly as from the open clears pass (the scene's clears are a truth pass of their own, ended by a
  // texture-sampled flush before the first scene draw).
  bool owned_clears_ended_ = false;
  uint64_t owned_clears_submission_ = 0;
  std::bitset<2048> owned_clears_tiles_;
  void OwnedNoteEndedClears();   // (EndTruthPass: a pass that was only clears has ended)
  void OwnedClearsTouched();     // (anything that touches the cleared tiles)
  bool OwnedActive() const { return owned_frame_on_ || !owned_list_.empty() || !owned_vs_list_.empty(); }
  // Depth images, made lazily (see OwnedImage).
  struct OwnedScratchDepth {
    uint32_t width = 0, height = 0;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    bool transient = false;  // (Session 93: TRANSIENT_ATTACHMENT usage, lazily allocated memory: never loaded or stored)
  };
  std::vector<OwnedScratchDepth> owned_scratch_depths_;
  int32_t OwnedScratchDepthFor(uint32_t width, uint32_t height);
  bool OwnedEnsureDepth(int16_t image);
  bool OwnedOverBudget(uint64_t more) const;
  // Review fix 7 / 8: whether a pass of this size can have its image within the budget - checked before anything of
  // the pass is ended - evicting the images nothing holds, least recently used first (their objects retired against
  // the submission; the slot stays, empty, for the next image).
  bool OwnedImageAvailable(uint32_t width, uint32_t height, uint32_t kind, bool depth);
  void OwnedEvictFor(uint64_t more, int16_t keep = -1);  // (keep: an image that must stay whatever its age - the one a caller is about to use)
  void OwnedRetireImage(size_t index);
  // wwe2k14 patch (Session 80, owned steps; owned_steps.cpp): our images (colour + a depth image the pipelines'
  // render passes need), the texture pages they hold (copies left out), the views made of them for samplers, the
  // output of the last owned draw until its copy, and the render pass (colour not loaded, stored, left readable).
  struct OwnedImage {
    VkImage image = VK_NULL_HANDLE, depth = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE, depth_memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE, depth_view = VK_NULL_HANDLE;
    // (Session 94, gpu_d3d_decouple: with K > 1 views `view` is the 2D view of the LAST layer (today's picture: every consumer of `view` keeps working) and
    // `array_view` the 2D_ARRAY view of all layers, the framebuffer's attachment; `depth_view` is then the depth image's whole 2D_ARRAY view.)
    VkImageView array_view = VK_NULL_HANDLE;
    // (Session 82: `depth` is made lazily - the first pass that keeps depth; every other pass is given the scratch
    // depth image of its size, which nothing reads: `framebuffer` is colour + that, `framebuffer_depth` colour + ours.)
    VkFramebuffer framebuffer = VK_NULL_HANDLE, framebuffer_depth = VK_NULL_HANDLE;
    VkFramebuffer framebuffer_depth_up = VK_NULL_HANDLE;  // (Session 95: its depth alone, the depth up-pass's target: OwnedDepthUpReady)
    uint64_t bytes = 0;  // (the budget: colour + depth when made)
    int32_t scratch = -1;  // its size's scratch depth image (owned_scratch_depths_)
    uint64_t used = 0;     // the last frame a pass started in it (the eviction order)
    bool frame_made = false;  // made with the owned frame on (not counted toward the list mode's caps)
    uint32_t width = 0, height = 0, kind = 0;  // host pixels (with the resolution scale)
    VkDescriptorSet fill_set = VK_NULL_HANDLE;  // (its colour as the texture fill's source: OwnedFill)
    VkBuffer fill_set_buffer = VK_NULL_HANDLE;  // (the staging buffer that set names)
    // (Its colour's layout: SHADER_READ_ONLY once readable - a copy of its pass served, filled, a snapshot -, else
    // COLOR_ATTACHMENT; every transition of ours starts from it - review / validation 01197.)
    bool readable = false;
    // (Session 82, P1 item 3: the depth image sampled by the depth copy's fill - its depth-only and stencil-only views and the
    // fill's source set, made when the first depth copy of the image is taken: owned_scene.cpp.)
    VkImageView depth_sample_view = VK_NULL_HANDLE, depth_stencil_view = VK_NULL_HANDLE;
    VkDescriptorSet depth_fill_set = VK_NULL_HANDLE;
    VkBuffer depth_fill_set_buffer = VK_NULL_HANDLE;
    VkDescriptorSet wb_set = VK_NULL_HANDLE;        // (Session 90, Q: the post-writeback set layout - the 8888 resolve shaders' source: staging + colour + dummy depth / stencil)
    VkBuffer wb_set_buffer = VK_NULL_HANDLE;
    uint64_t last_use = 0;    // (Session 82, gpu_native_owned_pass_barrier_lean: the last submission that recorded a use of the image)
    bool depth_read = false;  // (Session 82, gpu_native_owned_depth_direct: its depth is in SHADER_READ_ONLY - a page samples it)
    bool graph_dropped = false;  // (Session 95, gpu_owned_graph tripwire: the last pass instance / copy into this image was turned off; cleared when the image is handed out again)
    // (Session 93, tiler passes: the learned record of the pass whose depth the image holds, whether anything read that depth since the pass began, and
    // whether the pass was begun with its depth store DONT_CARE.)
    int32_t depth_learn = -1;
    bool depth_read_since = false, depth_dead_applied = false;
  };
  std::vector<OwnedImage> owned_images_;
  void OwnedTouch(int16_t image);
  int8_t owned_depth_direct_ok_ = -1;  // (Session 82: the depth format filters linearly when sampled: -1 not asked yet)

  // (Session 80, 16-bit pages) A k_16_16 / k_16_16_16_16 texture page of ours: the copy does not keep the bits - it
  // reads the 16-bit values (-32...32) and packs them as 16-bit unorm - so its samplers get an image in the texture's
  // own format (R16G16 / R16G16B16A16, UNORM and SNORM views over the same bits), filled from the pass's image by the
  // truth path's texture fill (kTruthFillComputeShader modes 1 / 3): the texels the game's copy would write.
  struct OwnedPageImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView storage_view = VK_NULL_HANDLE;
    VkDescriptorSet storage_set = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0, format = 0;  // host texels; xenos::ColorFormat
    uint32_t kind = 0;                           // (Session 82, P1: 0 a 16-bit colour page, 1 a depth page: R32_SFLOAT)
    uint64_t bytes = 0;
  };
  std::vector<OwnedPageImage> owned_page_images_;
  // (Session 87: the fill source sets of evicted images - kept, not leaked: the next fill that needs one takes it and rewrites its descriptors once the
  // submission that last recorded a use of it has completed (the rule retired_ follows); else a new one is allocated.)
  struct OwnedSpareSet {
    VkDescriptorSet set = VK_NULL_HANDLE;
    uint64_t submission = 0;
  };
  std::vector<OwnedSpareSet> owned_spare_sets_;
  std::vector<OwnedSpareSet> owned_wb_spare_sets_;  // (Session 90, Q: the write-back sets, post-writeback layout)
  // Session 90 (Q): the write-back armed by the copy OwnedTakeResolve just served (see OwnedWriteBackRun).
  struct OwnedWriteBack {
    bool armed = false;
    uint32_t path = 0;  // 0 8888 1x, 1 8888 scaled, 2 full 1x (k_8 / k_4_4_4_4), 3 scaled-full shader (16-bit pairs at any scale, small modes scaled), 4 depth
    uint32_t mode = 0;  // the scaled-full / full-1x / depth shaders' mode word
    int16_t image = -1;
    uint32_t sx = 0, sy = 0, w = 0, h = 0;  // the copy's rectangle in the pass box, guest pixels
    uint32_t gap = 0;
    bool front = false;
    uint32_t page = 0, dest_format = 0;
  };
  OwnedWriteBack owned_wb_;
  bool owned_wb_planned_ = false;
  uint8_t owned_replay_reason_ = 0;  // (Session 90, Q: the reason the next OwnedReplayPending files its pass under, 0 = its own)
  bool OwnedTakeResolveInner(const draw_util::ResolveInfo& info, bool copy_known);
  // (Plans the write-back of the copy being taken: false with `why` when no writer exists for it.)
  bool OwnedWbPlan(const draw_util::ResolveInfo& info, uint32_t sx, uint32_t sy, uint32_t w, uint32_t h, uint32_t gap, bool front,
                   std::string& why);
  bool OwnedWbPlanDepth(const draw_util::ResolveInfo& info, std::string& why);
  void OwnedWbRefused(const std::string& why);
  VkDescriptorSet OwnedWbPostSet(OwnedImage& o);
  VkDescriptorSet OwnedWbColorSet(OwnedImage& o);
  VkDescriptorSet OwnedWbDepthSet(OwnedImage& o);
  bool EnsureResolveFull1xPipeline();
  bool EnsureResolveDepthPipeline(bool scaled);
  VkResult OwnedAcquireFillSet(VkDescriptorSet& set);
  void OwnedSpareFillSet(VkDescriptorSet set);
  VkDescriptorPool owned_pool_ = VK_NULL_HANDLE;  // (the current one: Session 82, a new one when it is full)
  std::vector<VkDescriptorPool> owned_pools_;
  VkResult OwnedAllocateSet(VkDescriptorSetLayout layout, VkDescriptorSet& set);
  int16_t OwnedPageImageFor(uint32_t width, uint32_t height, uint32_t format);
  bool OwnedFill(int16_t page_image, bool fresh, uint32_t src_x, uint32_t src_y, uint32_t dst_x, uint32_t dst_y,
                 uint32_t width, uint32_t height, uint32_t gap);
  struct OwnedPage {
    uint32_t page = 0;
    int16_t image = -1;       // an owned image (8-bit pages), or
    int16_t page_image = -1;  // a 16-bit page image
    uint32_t width = 0, height = 0;  // guest pixels
    uint32_t format = 0;             // the copy's destination format (xenos::ColorFormat)
    uint32_t endian = 0;             // its byte order (xenos::Endian128)
    bool swap = false;               // its red / blue swap (k_8_8_8_8)
    uint32_t extent_start = 0, extent_length = 0;  // the guest memory the copy would have written
    uint64_t frame = 0;
    std::vector<std::pair<uint64_t, bool>> producers;  // the listed shaders that drew it (hash, a vertex shader)
    // (Session 82, the owned match frame: the pass that made it - a loss pauses that key; a page made with the switch
    // on never expires: it stays until the game's next copy into it or a CPU write.)
    OwnedPassKey key;
    bool frame_owned = false;
    int32_t census = -1;  // (the pass record of its frame: a reader refusal later in that frame turns it to a fallback)
    int16_t feedback = -1;  // (Session 82, P2, G4: a texture with mip levels of our own - owned_feedback_)
    uint64_t fill_stamp = 0;  // (Session 82: a 16-bit page image filled whole from a pass image as it was after its draw number N)
    // (Session 82, gpu_native_owned_depth_direct: a depth page that IS the pass image's depth, sampled through a depth view
    // in SHADER_READ_ONLY - no conversion pass; `image` names the pass image, which the page keeps alive.)
    bool depth_direct = false;
  };
  // wwe2k14 patch (Session 95, r6tex, gpu_tex_bind_memo): the list counts what changes its members; a member's fields can move on their own (the memo checks those against the page).
  class OwnedPageList : public std::vector<OwnedPage> {
   public:
    using Base = std::vector<OwnedPage>;
    void push_back(const OwnedPage& page) {
      ++version;
      Base::push_back(page);
    }
    template <class... A>
    Base::iterator erase(A&&... a) {
      ++version;
      return Base::erase(std::forward<A>(a)...);
    }
    void clear() {
      ++version;
      Base::clear();
    }
    uint64_t version = 1;
  };
  OwnedPageList owned_pages_;
  // (A bit per low 12 bits of a page in the list: clear = no page of the list has it, rebuilt when the list changed.)
  std::array<uint64_t, 64> owned_page_filter_{};
  uint64_t owned_page_filter_version_ = 0;
  bool OwnedPageMayHold(uint32_t page);
  // (The answer of OwnedView for a page and a way of reading it.)
  struct OwnedViewMemo {
    uint64_t list_version = 0, view_generation = 0;
    uint32_t page = 0, width = 0, height = 0, format = 0, endian = 0, flags = 0, index = 0, key_index = 0;
    VkComponentMapping mapping = {};
    int16_t image = -1, page_image = -1, feedback = -1;
    bool swap = false;
    VkImageView view = VK_NULL_HANDLE;
  };
  std::array<OwnedViewMemo, 256> owned_view_memo_{};
  uint64_t owned_view_generation_ = 1;  // (moves whenever a view of owned_views_ goes)
  uint32_t owned_view_memo_last_ = UINT32_MAX;  // (the entry the last OwnedView answered from or filled)
  void OwnedViewPre(uint32_t base_page, uint32_t width, uint32_t height, uint32_t format);
  bool OwnedViewMemoHolds(const OwnedViewMemo& m) const;
  void OwnedViewMemoRead(const OwnedViewMemo& m);
  VkImageView OwnedViewFull(uint32_t base_page, uint32_t width, uint32_t height, uint32_t format, uint32_t endian, bool scaled_ok, const VkComponentMapping& mapping,
                            bool is_signed, bool servable, bool has_mips, bool is_cube);
  // Session 82: passes whose pages nothing samples (G2's 64x64 renders): learned per pass key - the copies made and the
  // sampler reads of its pages; a key with 240 copies and no read is not drawn (its draw, held clears and copy go), and a
  // sampler that asks for one of its pages later brings it back.
  struct OwnedKeyStat {
    OwnedPassKey key;
    uint32_t made = 0, reads = 0;
    bool dead = false;
    // (Session 87, step 3 diagnosis of the dead-pass drop: when the key was first made, and its reads, logged.)
    uint64_t first_frame = 0, last_read_frame = 0;
    uint32_t reads_window = 0, read_logs = 0, made_total = 0;
    uint64_t summary_slot = 0;
    uint32_t revivals = 0;  // (Session 87, gpu_native_owned_frame_dead_relearn: times a dropped key was asked for again)
    bool read_ever = false;  // (Session 95, Lane E, gpu_native_owned_dead_sticky: a sampler has read this key's pages, ever)
  };
  uint64_t owned_dead_page_reads_ = 0;  // (Session 95, Lane E: pages of dropped passes that a sampler asked for, whole run)
  std::vector<OwnedKeyStat> owned_key_stats_;
  struct OwnedDeadPage {
    uint32_t page = 0, texture_page = 0;
    OwnedPassKey key;
  };
  std::vector<OwnedDeadPage> owned_dead_pages_;
  std::array<uint64_t, 2048> owned_dead_mask_{};  // (Session 87, gpu_cp_trim) the pages of the list, low 17 bits
  struct OwnedDrop {
    bool valid = false;
    uint32_t base = 0, pitch = 0;
    uint64_t frame = 0;  // (Session 95, review r5, 1: the frame of the last draw dropped: a copy after it is not that pass's)
    OwnedPassKey key;
  } owned_drop_;  // (the surface of the pass just not drawn - its copy is dropped too)
  OwnedKeyStat& OwnedKeyStatFor(const OwnedPassKey& key);
  bool OwnedKeyDead(const OwnedPassKey& key);
  void OwnedKeyMade(const OwnedPassKey& key);
  void OwnedKeyRead(const OwnedPassKey& key);
  void OwnedKeyReadStat(OwnedKeyStat& k);
  void OwnedDeadPageRead(uint32_t base_page);
  void OwnedPageLog(const char* what, const OwnedPage& pg, uint32_t unused, uint32_t rw = 0, uint32_t rh = 0, uint32_t rformat = 0);
  // wwe2k14 patch (Session 82, the owned match frame, P2, G4; owned_post.cpp): a k_4_4_4_4 texture with mip levels that
  // the game rebuilds by drawing small blended rectangles into a 1040-wide surface and copying each into one level (the
  // wrestlers' damage maps: draw, copy into level 0, redraw, copy into level 1 ...). The 360's way - EDRAM tiles shared
  // with other surfaces, the packed mip tail - is not kept: the texture is an RGBA8 image of ours with real mip levels,
  // each pass is drawn into a box image of ours (a defined, cleared start) and its rectangle copied into the level it
  // names; samplers of the texture get a view over all levels. Close, not exact: 8-bit values (the 360 keeps 4 bits a
  // channel); levels nothing wrote start as zero (blank maps).
  struct OwnedFeedback {
    uint32_t page = 0;               // the texture's base page (what samplers name)
    uint32_t width = 0, height = 0;  // level 0, host pixels
    uint32_t levels = 0;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint64_t bytes = 0;
    bool cleared_once = false;
    // (G7: a cube - six layers, a cube view; the reflection cube's faces.)
    bool cube = false;
    uint32_t layers = 1;
  };
  std::vector<OwnedFeedback> owned_feedback_;
  // The copy of the pending pass into a feedback texture: 1 = left out (the caller notes it), 0 = not ours.
  // (cube: the copy goes into a face of a cube texture - sx, sy the rectangle in the pass's image.)
  // Session 82 (P2): a face copy of a cube the texture cache has not seen (no sample yet): the cube, its face and its place derived from
  // the copy and the 6-face strip it comes from (owned_feedback.cpp); the first load of the cube checks it (OwnedCubeLearned).
  bool OwnedDeriveCubeFace(const draw_util::ResolveInfo& info, uint32_t sx, uint32_t sy, uint32_t w, uint32_t h,
                           uint32_t& tex_page, uint32_t& layers, uint32_t& layer, bool& scaled, uint32_t& extent);
  // Session 87 (gpu_native_owned_frame_strip_batch): the six face copies of a cube strip pass recorded as one copy at the first face.
  struct OwnedStripBatch {
    bool valid = false;
    int16_t image = -1, feedback = -1;
    uint64_t serial = 0;
    uint32_t width = 0;
    uint8_t mask = 0;  // faces already copied
  } owned_strip_batch_;
  int OwnedFeedbackCopy(const draw_util::ResolveInfo& info, uint32_t sx, uint32_t sy, uint32_t w, uint32_t h,
                        uint32_t dx, uint32_t dy, bool swap, bool cube = false);
  VkImageView OwnedFeedbackView(const OwnedPage& page, const VkComponentMapping& mapping);
  void OwnedFrameReaderRefused(const OwnedPage& page, const char* why);
  void OwnedMarkLeftOut(const OwnedPage& page);  // (the left-out copy's range: GPU-written in the shared memory)
  struct OwnedViewEntry {
    VkImage image = VK_NULL_HANDLE;
    VkComponentMapping components = {};
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;  // (UNORM or SNORM over the same bits)
    VkImageViewType type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  };
  std::vector<OwnedViewEntry> owned_views_;
  // (Session 82, P2: the view type OwnedView makes - the texture cache's views are 2D arrays; the swap's source is a 2D
  // view: the presenter's shaders declare a non-arrayed image.)
  VkImageViewType owned_view_type_ = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  // (Session 80, owned passes) The pass waiting for its copy: its surface, the truth images its draws would have gone
  // into, its box (from the surface's origin: the clears' union, or a covering draw's rectangle), whether our image
  // holds depth (and stencil) - the clears wrote them whole - and its draws, replayed in order into the truth images if
  // its copy does not follow as expected.
  struct OwnedPass {
    bool valid = false;
    bool open = false;      // its render pass open now (ended by OwnedEndPass, from EndTruthPass)
    // (Committed: a copy of it was taken - its tiles are dirty, it is never replayed; more listed draws may still join
    // it while nothing else touched its tiles. Our image's layout: OwnedImage::readable.)
    bool committed = false;
    bool graph_snap = false;  // (Session 95: its last copy was served by a predicted snapshot and the pass has not gone on since)
    uint32_t copy_count = 0;  // (Session 82: the copies taken of this pass so far)
    // (Session 82, P2: the restore skip's proof - the draws recorded into the pass since its last WHOLE colour copy / its last
    // served depth copy, and the texture pages of those two copies: the restore must sample exactly those.)
    uint32_t draws_since_color = 0, draws_since_depth = 0;
    uint32_t copy_color_page = 0, copy_depth_page = 0;
    bool started = false;  // its render pass begun once (later begins resume it)
    bool cleared = false;   // started from the clears of its truth pass (else from a covering draw)
    bool depth = false, stencil = false;  // our depth / stencil hold the surface's (cleared at the start, stored)
    bool draws_depth = false, draws_stencil = false;  // a draw wrote depth / used stencil
    uint32_t start = 0;     // the render pass it starts with (kOwnedPass*)
    uint32_t kind = 1;      // the truth class (1 RGBA8, 2 RGBA16 - k_16_16 / k_16_16_16_16)
    bool is_64bpp = false;  // (k_16_16_16_16: 40-pixel tiles)
    VkClearValue clear_values[2] = {};
    uint32_t base = 0, pitch_tiles = 0;  // the surface (EDRAM base tile, pitch in tiles)
    int16_t color_slot = -1, depth_slot = -1;
    uint32_t width = 0, height = 0;            // the box, guest pixels
    uint32_t host_width = 0, host_height = 0;  // and host pixels
    int16_t image = -1;
    uint64_t frame = 0, submission = 0;
    // (Session 92, M2a, gpu_native_owned_only: the pass's newest-draw tick (E1's depth source picks the newest holder of a depth slot by it); a
    // surface's newest record (owned_newest_) is a committed pass whose colour / depth a later pass of the surface has replaced.)
    uint64_t last_tick = 0, depth_tick = 0;  // (depth_tick: the last draw / clear / copy that wrote its depth)
    bool rec_colour_gone = false, rec_depth_gone = false;
    std::bitset<2048> tiles;  // the EDRAM tiles our image stands for (the box's colour tiles; depth ones with depth)
    std::vector<Draw> draws;
    // (Session 87, review A2: the owned images the kept draws above sampled - a detach of one of these images is refused
    // while they are kept; stale entries only refuse more.)
    std::vector<int16_t> kept_sampled;
    std::vector<OwnedCrowdCall> crowd_calls;  // (Session 82: our crowd draws, replayed after the draw they follow)
    std::vector<std::pair<size_t, std::vector<VkImage>>> serves;  // (Session 82: the images a draw (its index) sampled served)
    std::vector<std::pair<uint64_t, bool>> producers;
    // The truth pass of the clears it started from (gpu_native_own_steps_drop_clears): its range (begin arguments, the
    // first draw after its loads, its end command, the end), whether it loaded tiles, its submission; whether it wrote
    // only what our image holds (exactly the box; depth / stencil only whole) - then it is turned off at the commit.
    size_t clears_begin = SIZE_MAX, clears_draws = SIZE_MAX, clears_end_cmd = SIZE_MAX, clears_end = SIZE_MAX;
    bool clears_loaded = false, clears_ours = false, clears_depth = false, clears_stencil = false;
    uint64_t clears_submission = 0;
    uint64_t clears_serial = 0;  // (the serial of the truth pass its clears ran in - truth_pass_by_offset_; review F9)
    // (Session 82)
    int32_t census = -1;  // its record in owned_frame_passes_
    OwnedPassKey key;
    uint8_t start_kind = kOwnedStartNone;
    bool loaded_start = false;  // (unused: the converted start no longer loads)
    // A defined start of our own (OwnedStartExtra): its image started cleared, its tiles are no truth image's - nothing
    // of it is claimed, nothing marked dirty, and a truth draw over its tiles does not concern it.
    bool edram_free = false;
    // The clears this pass started from, held by the owned frame (OwnedHoldClear) - never recorded in the truth path;
    // replayed into it first if the pass is lost.
    std::vector<Draw> held_clears;
    bool clears_held = false;
    // (Session 87, gpu_native_owned_frame_chains: a pass begun from flushed held clears, without a draw; its key takes the first
    // draw's pixel shader once one is recorded into it.)
    bool chain_pass = false, chain_unkeyed = false;
    bool carried_defined = false;  // (Session 90: a defined start carried past a submission's end - its draws are gone, never replayed; its image is the pass)
    bool chain_parked = false;  // (a defined-start pass the chains park committed without its copy: replayed if it is dropped)
    uint32_t timer_segment = UINT32_MAX, timer_draws = 0;  // (Session 82, P1: the timed render pass being recorded)
    // (Session 82, P2, G4: a defined start whose first draw is not at the surface's origin - the box is still the rectangle
    // from the origin, cleared: a copy of it is exact only inside what a draw covered (OwnedRectDrawn).)
    bool nonorigin = false;
    // (Session 82, P2, G4: a copy of it went into a feedback texture - its other copies the feedback path cannot serve
    // (the packed tail's strips) are dropped: the texture of ours has no memory for them.)
    bool feedback_pass = false;
    // (Session 82, look lever gpu_native_owned_post_half: the pass is drawn into an image of half the box's size - its draws'
    // viewport and scissor halved; copies of it are whole only, a front buffer's copy blits it up to the full size.)
    bool half = false;
    // (Session 94, gpu_native_owned_draw_ratio: dres = the scene pass drawn at ratio_x / ratio_y of its box (half is set too, so every
    // half refusal stands); its colour and depth copies are brought up to the full size by a blit.)
    float ratio_x = 0.5f, ratio_y = 0.5f;
    bool dres = false;
    int16_t dres_full = -1;  // (the one full-size image of this scene pass: its depth copy and its first colour copy are blitted into it; held while a pass or record has it)
    bool dres_color_used = false, dres_depth_used = false;
    // (The bounding box of the rectangles its draws covered, guest pixels - kept after the pass is committed: a copy of
    // a defined start away from the origin is trusted inside it.)
    int32_t drawn[4] = {0, 0, 0, 0};  // x0, y0, x1, y1; empty when x1 <= x0
  } owned_pending_;
  // (Session 93, gpu_owned_tiler_passes: ResumeNoDepth = colour LOAD / STORE, depth + stencil DONT_CARE / DONT_CARE, for a resumed pass that has no depth
  // of its own (its framebuffer holds the scratch depth); ClearBothDead = as ClearBoth with the depth + stencil store DONT_CARE, for a pass whose depth
  // nothing has read after it for K passes in a row. Both are attachment-compatible with Cover: load / store ops are not part of render pass compatibility.)
  enum : uint32_t {
    kOwnedPassCover, kOwnedPassClearColor, kOwnedPassClearBoth, kOwnedPassResume, kOwnedPassResumeNoDepth, kOwnedPassClearBothDead, kOwnedPasses
  };
  // (Session 80, passes past their copy) Surfaces whose pass got more draws after a copy served from its image: their
  // copies are served from a snapshot from then on (the later draws may sample that page - map B, then its blurs).
  struct OwnedContinued {
    uint32_t base = 0, pitch_tiles = 0, kind = 0, width = 0, height = 0;
    uint64_t frame = 0;
    uint32_t after = 0;  // (Session 82: bit N - a draw of the pass went on after its copy number N)
  };
  std::vector<OwnedContinued> owned_continued_;
  bool OwnedContinues(const OwnedPass& p, uint32_t copy_index) const;  // (will the pass draw on after its copy number copy_index)
  void OwnedNoteContinued(const OwnedPass& p);
  // Session 87 (gpu_native_owned_frame_detach_pages, owned_steps.cpp): the committed pass p is about to draw on, and pages of
  // ours ARE its image (a whole colour copy served directly, a direct depth page): the colour pages are given a snapshot of the
  // image, the depth pages one converted page image (the contents are exactly what the game's copies took: no draw since).
  // False = not done (nothing the caller relies on changed except pages already detached, which stay valid).
  bool OwnedDetachPages(OwnedPass& p);
  // Session 87 (gpu_native_owned_frame_continue_copy): the pass goes on in a fresh image that starts as a copy of its image; the pages keep the old one.
  bool OwnedContinueCopy(OwnedPass& p);
  // (Session 87: the owned images whose page views OwnedView handed out for the draw being recorded - its texture request ran first;
  // a draw that samples a page of its own pass image cannot have that page detached under it. Cleared at the end of OwnedTakeDraw.)
  std::vector<int16_t> owned_sampled_images_;
  // (Session 82, P2; owned_post.cpp) A copy of a pass's box rounds its rectangle up to 8 pixels: rows / columns beyond the
  // box (the 320x184 copies of the 320x180 post passes) are padding of the destination texture, which no sampler reads:
  // the rectangle clamped to the box (true) when it overshoots by less than 8 pixels and the owned frame is on.
  bool OwnedCopyBox(const OwnedPass& p, uint32_t sx, uint32_t sy, uint32_t& w, uint32_t& h) const;
  // Look lever gpu_native_owned_post_half (owned_post.cpp): the post chain's covered passes at half size, and the blit that
  // brings the combine's image up to the full size for the front buffer.
  bool OwnedPostHalfWanted(const Draw& draw, bool covered, bool cleared, uint32_t scale_x, uint32_t scale_y,
                           uint32_t width, uint32_t height, bool depth_kept, uint32_t pitch) const;
  int16_t OwnedBlitUp(const OwnedPass& p);
  // (Session 94: the pass image's depth, nearest, into the depth of the full-size image `target` - ends in DEPTH_STENCIL_ATTACHMENT_OPTIMAL.)
  bool OwnedBlitUpDepth(const OwnedPass& p, int16_t target);
  // Session 95 (gpu_native_owned_dres_depth_pass): where the device cannot blit the depth format (Adreno 830, AMD on Windows) the scaled scene pass's depth goes up by a
  // fragment pass of ours. Objects made once; OwnedDepthUpReady makes what one copy needs (before the pass commits: it can fail), OwnedDepthUpPass records it.
  bool EnsureOwnedDepthUpObjects();
  bool OwnedDepthUpReady(int16_t source, int16_t target);
  bool OwnedDepthUpPass(const OwnedPass& p, int16_t target);
  bool owned_depth_up_on_ = false;  // (the up-pass is the way depth goes up in this run: decided at the first scaled pass)
  bool owned_depth_up_failed_ = false;
  VkShaderModule owned_depth_up_shaders_[2] = {};
  VkPipelineLayout owned_depth_up_layout_ = VK_NULL_HANDLE;
  VkRenderPass owned_depth_up_render_pass_ = VK_NULL_HANDLE;
  VkPipeline owned_depth_up_pipeline_ = VK_NULL_HANDLE;
  // Whether a draw of the pass covered the rectangle (guest pixels): a defined start from a draw away from the origin
  // is exact only inside the draws' rectangles (G4).
  bool OwnedRectDrawn(const OwnedPass& p, uint32_t sx, uint32_t sy, uint32_t w, uint32_t h) const;
  // Whether the draw writes every pixel of every tile of the pass (a proven cover from the origin, aligned to whole
  // tiles, unblended, all channels, no depth / stencil): the pass's pixels are dead under it.
  bool OwnedDrawOverwrites(const Draw& draw, const OwnedPass& p) const;
  // A colour-less whole-surface clear of the depth surface an owned scene's depth page was copied out of this frame: not
  // recorded (the next frame's scene clears it first); counted as an owned pass.
  bool OwnedDeadDepthClear(const Draw& draw);
  // wwe2k14 patch (Session 92, M4f-B, gpu_d3d_owned_front bit 64): the pending clear of a surface slot (owned_dead_clear.h; owned_post.cpp). Make: a clear the owned
  // frame did not take is remembered as its surface's content (order_work: also ends what holds the surface now - a D3dClear refusal; false from a flush that
  // already did). Materialize: just before a draw of a slot with a pended colour clear, the clear is held again, as if the game had issued it right there.
  // EraseFor: a newer clear of the slot replaces the pend. FrameEnd: the unconsumed are dropped, counted. ColourFor: a copy's lookup (marks it used).
  bool OwnedPendClearMake(const Draw& d, int origin, bool order_work, uint32_t box_w = 0, uint32_t box_h = 0, const int32_t* require_rect = nullptr);
  void OwnedPendClearDropNewerThan(int32_t colour_slot, uint64_t tick);  // (review fix 2: a pend made under the draw that was just taken is not for the next draw)
  void OwnedPendClearMaterialize(const Draw& draw);
  void OwnedPendClearEraseFor(const Draw& clear);
  void OwnedPendClearFrameEnd();
  void OwnedPendClearReset();
  bool OwnedPendColourFor(uint32_t base, uint32_t pitch_tiles, int32_t x0, int32_t y0, int32_t x1, int32_t y1, float value[4]);
  // (Session 87, gpu_native_owned_frame_depth_clear: a depth(-stencil)-only clear of the depth surface of the owned pass that is open,
  // recorded into that pass as a native clear - the clear the library draws with its clear shaders is never a draw on our side.)
  bool OwnedDepthClearInPass(const Draw& draw);
  bool OwnedChains() const;  // (Session 87: gpu_native_owned_frame_chains)
  bool OwnedHoldsStencil(int16_t depth_slot, uint32_t pitch_tiles, const int32_t rect[4]);  // (Session 92, D1: ClaimTruth)
  // Session 92, M2a (gpu_native_owned_only; docs\native\d3d\S92_M2_DESIGN.md section 4): under TRUE NATIVE nothing is handed back to the truth path
  // - a draw our frame does not take is dropped (counted, named), a copy is served from a pass / the surface's newest record, written, or dropped,
  // no replay, no pause, no dirty tile, no truth image. OwnedOnly() = the cvar && chains && the owned frame on, released by a hard owned failure.
  // Session 94 (idea 1 phase 1A; docs\native\d3d\S93_DECOUPLE_DESIGN.md section 6): gpu_d3d_decouple. K = 2 views for the owned passes while on (0 = off). Decided once, at
  // the first use (the cvar is init-only): the device has multiview, the owned-only latch and chains are wanted. A/A (bit 2): both views use the same constants.
  uint32_t DecoupleViews();
  uint32_t OwnedLayers() { return DecoupleViews() ? DecoupleViews() : 1u; }  // (the layers of an owned image)
  bool DecoupleAA();
  VkImageSubresourceRange OwnedLastLayer(VkImageAspectFlags aspects);  // (the range of an owned image's picture: its layer 0 only; the whole image when it has one layer)
  void DecoupleRenderPass(VkRenderPassCreateInfo& info, VkRenderPassMultiviewCreateInfo& mv);  // (chains the view mask into a render pass description; nothing when off)
  // Copies between images where either may be an owned (layered) image: each region once per layer of the owned side(s) - owned -> owned layer by layer, owned -> other
  // from layer 0 (the picture), other -> owned into every layer. A plain CmdVkCopyImage when off.
  void OwnedCopyImage(DeferredCommandBuffer& cb, VkImage src, bool src_owned, VkImageLayout src_layout, VkImage dst, bool dst_owned,
                      VkImageLayout dst_layout, uint32_t count, const VkImageCopy* regions);
  uint32_t decouple_views_ = 0;
  bool decouple_ready_ = false;
  // Proof (phase 1A): the compare of the views of the presented owned image (owned_decouple.cpp).
  void DecoupleCompareSwap(int16_t image, uint32_t width = 0, uint32_t height = 0);  // (review: the presented rectangle too)
  uint32_t decouple_swap_w_ = 0, decouple_swap_h_ = 0;  // (the swap presents our image: remembered, compared at the frame's end)
  void DecoupleFrameEnd();                   // (OwnedOnFrameEnd, last: the compare of the presented image's layers, the window line)
  void DecoupleCompareReap();                // (read back an earlier compare, once its submission has completed)
  int16_t decouple_swap_image_ = -1;
  void DecoupleCompareDestroy();
  // Phase 1B: the two-step present. TwoStep = this tick was blended (view 0 differs) and the switch asks for it (gpu_d3d_decouple bit 3 on, bit 4 (16) off); Layer1View = a 2D view of layer 1 (the in-between picture)
  // of the owned image the 2D view `last_layer_view` (the swap's) is of (made once, kept with the owned views); NotePresents counts what the presenter was given for the window line.
 public:
  bool DecoupleTwoStep();
  bool DecoupleViewsOn() { return DecoupleViews() != 0; }
  VkImageView OwnedLayer1View(VkImageView last_layer_view);
  void DecoupleNotePresents(uint32_t n) { decouple_presents_ += n; }
 private:
  uint64_t decouple_presents_ = 0;
  struct DecoupleCompare {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
    uint64_t submission = 0;
    uint32_t width = 0, height = 0, bytes_per_pixel = 4;
    uint32_t rect_w = 0, rect_h = 0;  // (review: the presented rectangle compared; 0 = the whole image)
    bool pending = false;
  } decouple_cmp_[2];
  uint32_t decouple_cmp_next_ = 0;
  uint64_t decouple_frames_ = 0, decouple_compared_ = 0, decouple_diff_max_ = 0, decouple_diff_total_ = 0, decouple_diff_frames_ = 0;
  bool OwnedOnly() const;        // (Session 92, M2b review F1: the mode the frame runs in = owned_surf_mode_, fixed at the frame boundary - a mid-frame failure never mixes the two slot tables)
  bool OwnedOnlyWanted() const;  // (the live decision: the cvar latched at the boundary, the owned frame, chains, no hard failure - read ONLY by OwnedRefreshFrame)
  void OwnedDropDraw(const Draw& draw);                       // (TruthRecordDraw: the draw OwnedTakeDraw did not take)
  void OwnedNoteClaimDrop(uint32_t truth_reason, const Draw& draw);  // (ClaimTruth: a refusal claimed instead)
  void OwnedNoteFallbackIgnored(uint32_t truth_reason);       // (TruthFallback under the latch)
  void OwnedNoteNewest(const OwnedPass& q);                   // (a pass that ends: its surface's newest record)
  void OwnedRecordsTrim(const char* why);
  void OwnedMarkDepthStale(int16_t depth_slot);  // (a depth clear of the slot was dropped / replaced it: records, parked and the pending pass holding it lose that depth)
  void OwnedSupersedeRecords(uint32_t base, uint32_t pitch, uint32_t kind, int16_t depth_slot);  // (a pass of this surface starts: older records lose their colour / depth)
  void OwnedEndAsIs(const char* why);                         // (a pending pass whose copy did not follow: committed and ended, no replay)
  std::vector<OwnedPass> owned_newest_;                       // (per surface: the newest committed pass, holds its image)
  uint64_t owned_tick_ = 0;
  bool owned_only_released_logged_ = false;
  bool owned_only_latched_ = false;  // (the cvar as of the last frame boundary - OwnedRefreshFrame)
  uint32_t m2a_loaded_from_record_ = 0, m2a_loaded_no_record_ = 0;  // (owned_start.cpp: a start that keeps content - resumed from a record / none held)
  // Session 92, M2b (docs\native\d3d\S92_M2BC_CONTRACT.md): the owned entry and the surface registry. Under the latch a draw is claimed by OwnedClaim (always
  // true for a draw offered - one it cannot record is claimed and dropped) and entered by OwnedEnter (OwnedTakeDraw, else OwnedDropDraw); the truth claim and
  // recorder are not reached. A surface is identified by the game's own surface words (base, pitch in 80-sample tiles, class, format) - no EDRAM tile; the
  // slot (Draw::truth_color / truth_depth, OwnedPass::color_slot / depth_slot) indexes owned_surfaces_ instead of truth_images_ while owned_surf_mode_.
  struct OwnedSurface {
    uint32_t base = 0, pitch_tiles = 0, kind = 0, format = 0;
    uint32_t first_frame = 0, last_draw_frame = 0;  // (frame + 1 of the last draw claimed into it, 0 = never: M2c names a dropped copy's source by it)
    bool is_64bpp() const { return kind == 2 && format == 5; }
    uint32_t row_tiles() const { return pitch_tiles * (is_64bpp() ? 2u : 1u); }
  };
  std::vector<OwnedSurface> owned_surfaces_;
  bool owned_surf_mode_ = false;  // (OwnedOnly() as of the last frame boundary - the whole frame claims and reads slots from one table)
  int16_t OwnedFindSurface(uint32_t base, uint32_t pitch_tiles, uint32_t kind, uint32_t format);
  int16_t SurfaceSlot(uint32_t base, uint32_t pitch_tiles, uint32_t kind, uint32_t format);  // (the registry in use: owned_surfaces_ under the latch, else truth_images_)
  size_t OwnedSlotCount() const { return owned_surf_mode_ ? owned_surfaces_.size() : truth_images_.size(); }
  OwnedSurface OwnedSlotKey(int16_t slot) const;  // (by value; an all-zero key for a slot out of range)
  int OwnedSourceDrawn(uint32_t base, uint32_t row_tiles, bool depth) const;  // (M2c: 0 no key of the registry, 1 a key never drawn, 2 drawn in an earlier frame, 3 drawn this frame)
  bool OwnedWbPlanFill(const draw_util::ResolveInfo& info, std::string& why, const float* colour = nullptr);  // (M4f-B: colour = a pended clear's value instead of transparent black) // (M2c: the defined fill - a zero image written to the copy's destination by the 8888 writer)
  bool OwnedWbPlanComposite(const draw_util::ResolveInfo& info, uint32_t w, uint32_t h, std::string& why);  // (M2d, Part 4: the pending pass's image copied into the top of a zeroed image of the copy's size, written by the 8888 writer)
  bool OwnedClaim(Draw& draw);
  bool OwnedClaimLean(Draw& draw);
 public:
  bool OwnedClaimRecord(Draw& draw);  // (Session 95, r3cmd)
 private:
  bool OwnedClaimRepeat(Draw& draw);
  void OwnedEnter(const Draw& draw);
  void OwnedCountUnprovenStencil(const Draw& draw);  // (Session 92: census only)
  // (Session 87, gpu_native_owned_frame_depth_clear_pend: a depth-only clear of a depth surface no owned pass has is PENDING - not run in the
  // truth path as a pass of its own. It runs there (the saved draw replayed) only if something consults that depth first within the
  // submission: a draw that uses the same depth surface, a copy of that depth, the EDRAM write-out. Else it is dropped at the submission's
  // end, counted, and remembered so a later read of that depth without a new clear is logged.)
  struct OwnedPendingDepthClear {
    Draw draw;
    int16_t slot = -1;
    uint32_t depth_base = 0;  // the depth surface's EDRAM base (a copy of depth names it)
    int32_t rect[4] = {0, 0, 0, 0};
  };
  std::vector<OwnedPendingDepthClear> owned_depth_clears_;
  std::vector<std::pair<int16_t, uint64_t>> owned_depth_dropped_;  // (slot, frame) of dropped ones, until a clear writes the slot again
  bool OwnedDepthClearPend(const Draw& draw);
  void OwnedDepthClearsConsult(const Draw& draw);
  void OwnedDepthClearsFlush(const char* why, int32_t only_slot = -1, int64_t only_base = -1);
  void OwnedDepthClearsDrop();
  struct {
    uint32_t base = 0, pitch_tiles = 0, width = 0, height = 0;
    uint64_t frame = 0;  // frame_ + 1 of the copy
  } owned_scene_depth_;
  // The swap has read this page as the copy would write it lately (a copy into a front buffer is left out only then).
  bool OwnedPresentOk(uint32_t page, uint32_t width, uint32_t height, uint32_t format, uint32_t endian,
                      bool scaled) const;
  struct OwnedSwapSeen {
    uint32_t page = 0, width = 0, height = 0, format = 0, endian = 0;
    bool scaled = false;
    uint64_t frame = 0;
  };
  OwnedSwapSeen owned_swap_seen_[4] = {};
  uint32_t owned_swap_next_ = 0;
  bool OwnedCommit();  // the pass's first copy: its clears' pass off, its tiles dirty (and again, cheaply, later)
  void OwnedMarkDirty(bool dropped);
  bool OwnedTruthHolds() const;  // whether the truth images get the pass's pixels (its dirty tiles)
  bool OwnedTruthHolds(const OwnedPass& p) const;
  // Session 82 (the owned frame): a committed pass a new pass starts over (G5's map B, then the reductions, then its
  // blurs) is parked, not ended - a later draw that fits it takes it up again; anything that touches its tiles, a
  // write-out, the submission's or the frame's end drops it.
  // (Several may wait: G5's map B stays parked while the 128 and 32 reductions each start and park in turn; at most
  // kOwnedParkedMax, the oldest goes first.)
  std::vector<OwnedPass> owned_parked_;
  // (Session 90: surfaces (base, pitch, kind) of carried defined starts dropped at the last frame's end - a copy of one next frame is counted.)
  struct OwnedCarriedGone { uint32_t base, pitch, kind; };
  std::vector<OwnedCarriedGone> owned_frame_end_carried_;
  uint32_t owned_carried_gone_logged_ = 0;
  // Session 82 (P2): parked passes a truth draw over their tiles dropped - the front buffer's composite, which the 360 then undoes with a
  // restore: kept (copied, nothing drawn into them since) until the restore of that surface or the next submission / frame end.
  std::vector<OwnedPass> owned_restorable_;
  // Session 82 (P2, review): a face copy of a derived cube was just served - the copy's range is to be marked resolved at 1x too
  // (NativeOwnedMarkResolved), so the cache's first load of the cube is learned and OwnedCubeLearned checks the guess; and a wrong
  // guess ends the derivation (owned_cube_derive_off_).
  bool owned_derived_resolved_ = false;
  bool owned_cube_derive_off_ = false;
  void OwnedParkOrReplay(const char* why);
  // (Session 87, gpu_native_owned_frame_chains) The held clears begun as a pass of ours (false = not begun, nothing changed but the pending
  // pass parked); whether a copy / a draw is of the surface a pass stands for (base, pitch, class - or its depth slot).
  bool OwnedBeginClearedPass();
  bool OwnedPassDeadUnder(const Draw& clear) const;  // (chains: the pending uncommitted, never-copied pass that this clear writes whole)
  // (chains, review F2) A parked defined-start pass dropped before its copy: its draws go to the truth path first (nothing else holds them).
  void OwnedReplayParkedPass(OwnedPass q, const char* why);
  // (chains, review F4b) The pending pass's render pass begun again now (a depth-only clear that arrives while it is not open is applied at once).
  bool OwnedResumeNow();
  // (chains, review F3) The truth images are given the tiles of a pass whose clears were held (never run in the truth path): ensured and owned.
  void OwnedClaimClearTiles(const OwnedPass& p);
  // (chains) Whether a pass of this key is refused up front because a copy of such a pass once went to the front buffer unserved - for
  // 30 frames only under chains (the key is (surface, first pixel shader): one stale entry refused every later pass of it for ever).
  // (The list of passes a failed front copy once banned: kept as it was, with chains off. Under chains nothing is ever remembered - a front copy that
  // cannot be served replays its pass once - and OwnedPresentKeyed is always false.)
  bool OwnedPresentKeyed(const OwnedPassKey& key) const;
  void OwnedPresentAdd(const OwnedPassKey& key, const char* why);
  // (chains) A new owned image filled once from the truth image of the surface: the tiles that image owns inside the box, the rest transparent black.
  // False = nothing of the surface in the truth path (the caller's converted start is then exact); the image is untouched.
  bool OwnedLoadFromTruth(int16_t slot, int16_t image, uint32_t width, uint32_t height);
  bool OwnedLoadParkedInto(int16_t image, bool has_content, const std::vector<OwnedPass>& passes, uint32_t scale_x, uint32_t scale_y);  // (Session 90, Q)
  // (chains) Surfaces whose passes have had a copy tried through the feedback path (damage / sweat maps, the cube strips): scratch surfaces that keep their
  // defined start - a loaded or resumed start would take them off the edram_free paths.
  // (chains, review A2: swap pages the hook could not serve - the next copy into one runs from the truth path, for a while.)
  std::vector<std::pair<uint32_t, uint64_t>> owned_unserved_pages_;  // (page, until frame)
  std::vector<std::array<uint32_t, 3>> owned_feedback_surfaces_;  // (base, pitch tiles, class)
  bool OwnedFeedbackSurface(uint32_t base, uint32_t pitch, uint32_t kind) const {
    for (const auto& s : owned_feedback_surfaces_) {
      if (s[0] == base && s[1] == pitch && s[2] == kind) return true;
    }
    return false;
  }
  bool OwnedCopyOfPass(const draw_util::ResolveInfo& info, const OwnedPass& q) const;
  bool OwnedDrawOfPass(const Draw& draw, uint32_t base, uint32_t pitch, const OwnedPass& q) const;
  void OwnedDropParked(const char* why = "?");
  void OwnedDropParkedIf(const char* why, const std::function<bool(const OwnedPass&)>& pred);
  // Our image copied into a free one of its size (the page's); with `gap` (x | y << 8, host pixels) the copy's
  // half-pixel offset gap filled as the game's copy fills it.
  int16_t OwnedSnapshot(uint32_t gap = 0);
  void OwnedEndCommitted(const char* why = "");
  VkRenderPass owned_render_passes_[kOwnedPasses][kPostTargetClasses] = {};
  // (Session 80, owned passes) Dirty tiles: a truth image's tiles whose pixels inside the box are not what the 360
  // would have there - our pass drew them into our image (and its clears' pass may have been turned off). Before
  // anything reads them (a draw over them that is not a whole clear, an alias load or store from their image, a copy,
  // a write-out) our image's pixels are copied in (OwnedMaterialize); a clear writing them whole, or another image
  // taking them over, ends that.
  struct OwnedDirty {
    int16_t slot = -1;  // the truth image (and what it is: its base, pitch, kind, format)
    uint32_t base = 0, pitch_tiles = 0, kind = 0, format = 0;
    int16_t image = -1;  // ours
    VkImageAspectFlags aspects = 0;  // colour, depth or stencil (one entry each)
    uint32_t host_width = 0, host_height = 0;  // the box
    std::bitset<2048> tiles;
    uint64_t frame = 0;
  };
  std::vector<OwnedDirty> owned_dirty_;
  // (The dirty tiles the truth draw being recorded clears whole: dropped from the set once its draw command is
  // recorded - OwnedApplyClean; review: a draw TruthRecordDraw leaves out clears nothing.)
  struct OwnedClean {
    int16_t slot = -1;
    VkImageAspectFlags aspects = 0;
    std::bitset<2048> tiles;
  };
  std::vector<OwnedClean> owned_clean_pending_;
  void OwnedApplyClean();
  std::string owned_list_text_;
  std::vector<uint64_t> owned_list_, owned_vs_list_;  // (copied once a frame: OwnedRefreshList)
  // Shaders that lost an output: not owned any more.
  // (Session 80: for a while - until the frame given - not for good: one scene of another shape, an entrance's extra
  // caster, must not end them for the rest of the run; a loss still costs only a replay, once in 600 frames.)
  std::vector<std::pair<uint64_t, uint64_t>> owned_disabled_, owned_disabled_vs_;
  bool OwnedDisabled(const std::vector<std::pair<uint64_t, uint64_t>>& disabled, uint64_t hash) const;
  bool owned_failed_ = false;
  bool owned_replaying_ = false;
  bool owned_materializing_ = false;  // (OwnedMaterialize's flush: the dirty checks it reaches do not run)
  bool OwnedWants(uint64_t pixel_shader_hash) const;
  bool OwnedWants(uint64_t pixel_shader_hash, uint64_t vertex_shader_hash) const;
  void OwnedRefreshList();
  void OwnedDisable(uint64_t hash, bool vertex, const char* why);
  void OwnedDisableAll(const std::vector<std::pair<uint64_t, bool>>& producers, const char* why);
  std::bitset<2048> OwnedDrawTiles(const Draw& draw) const;
  void OwnedReplayPending(const char* why, bool pause = true, bool replay_committed = false);
  void OwnedReplayDraws();  // (the pass's draws into the truth images, in order)
  bool OwnedRecord(const Draw& draw, const int32_t rect[4]);
  void OwnedEndPass();
  int16_t OwnedImageFor(uint32_t width, uint32_t height, uint32_t kind);
  bool OwnedImageHeld(int16_t image) const;
  // (Session 95, gpu_owned_graph; owned_graph.cpp) Whether something may still read the image's depth: a pending, parked, restorable or newest pass that
  // keeps it, a direct depth page, a dirty depth entry, the write-back.
  bool OwnedDepthHeld(int16_t image) const;
  void OwnedGraphSummary(uint64_t first_frame, uint64_t last_frame);  // (the window's [native][graph] line; resets the counters)
  enum : uint32_t { kGraphCountSnapshot, kGraphCountContinueCopy, kGraphCountPredictedSnapshot, kGraphCountSnapshotContinued };
  void OwnedGraphTripwire(int16_t image);  // (Session 95: a read, copy or fill of an image whose last segment gpu_owned_graph dropped is counted - must stay 0)
  // (Session 95, round 7, gpu_owned_graph bit 16; owned_graph.cpp) Who read an owned image in this stream. OwnedGraphNoteRead at every reader: by_draw = a view handed to a draw, which counts for the pass
  // that draw is recorded in (OwnedGraphReadsLand, in OwnedRecord); anything else (a fill, a write-back, the swap, a draw of the view that is never recorded in an owned pass) keeps its image's segment live.
  // OwnedGraphReadsClose: the current draw's request is over (what did not land stays a read by nobody known).
  void OwnedGraphNoteRead(int16_t image, bool by_draw = false);
  void OwnedGraphReadsLand(const Draw& draw);
  void OwnedGraphReadsClose();
  void OwnedGraphCount(uint32_t what);  // (counters kept for the line: snapshots, continue copies taken / avoided)
  std::shared_ptr<OwnedGraphState> graph_state_;  // (made by the first walk)
  void OwnedPruneDirty();
  void OwnedMaterialize(OwnedDirty& entry, const std::bitset<2048>& tiles);
  void OwnedDirtyBeforeDraw(const Draw& draw);                      // (TruthRecordDraw, first)
  void OwnedDirtyBeforeRead(const std::bitset<2048>& tiles, int16_t slot = -1,
                            const char* why = "a truth read");  // loads, stores, copies
  void OwnedBeforeWriteOut();
  uint32_t truth_time_tick_ = 0;  // Session 47: TruthRecordDraw times one draw in 8
  // Session 47 (CPU): what the open truth pass's draws last bound (pipeline, depth bias, blend constants).
  bool truth_bound_valid_ = false;
  VkPipeline truth_bound_pipeline_ = VK_NULL_HANDLE;
  float truth_bound_bias_[2] = {};
  float truth_bound_blend_[4] = {};
  // Session 56 (gpu_native_truth_dedupe_state): the index buffer bound at offset 0 (its draws by firstIndex) and the
  // stencil reference / masks per face, while truth_bound_valid_.
  VkBuffer truth_bound_index_buffer_ = VK_NULL_HANDLE;
  VkIndexType truth_bound_index_type_ = VK_INDEX_TYPE_UINT16;
  uint8_t truth_bound_stencil_[2][3] = {};
  bool truth_bound_stencil_valid_ = false;
  // Session 47 (CPU): TruthTileRects' scratch lists (no allocations per call).
  mutable std::vector<std::array<int32_t, 4>> truth_scratch_open_, truth_scratch_next_;
  // Session 47 (CPU): TruthStoreMany's jobs (reused), and the lists TruthRecordDraw / TruthResolveSource fill per call.
  struct TruthStoreJob {
    int16_t index = -1;
    std::vector<std::array<int32_t, 4>> rects;
    // Depth: the tiles whose stencil the image holds (stored with the image's own stencil; Session 46). The others
    // keep the EDRAM's (into the EDRAM) or take the last clear's value (into the staging buffer).
    std::vector<std::array<int32_t, 4>> stencil_rects;
  };
  std::vector<TruthStoreJob> truth_store_jobs_;
  std::vector<VkImageMemoryBarrier> truth_store_barriers_;
  std::vector<int16_t> truth_scratch_owners_;
  std::vector<std::array<int32_t, 4>> truth_scratch_load_rects_;
  // Writes the tiles of `tiles` that image `index` owns into `buffer` (true: the staging buffer, false: the EDRAM).
  void TruthStore(int16_t index, const std::bitset<2048>& tiles, bool staging);
  void TruthStoreMany(const int16_t* indices, size_t count, const std::bitset<2048>& tiles, bool staging);
  // The boundary: every owned tile into the EDRAM, every tile the EDRAM's again; the rest of the frame is exact.
  void TruthWriteOut();
  void TruthFallback(uint32_t reason, const Draw* draw);
  // Image pixel rectangles (host pixels, x0 y0 x1 y1) of the tiles of `tiles` in image `image`, one per run of a tile
  // row, runs of equal columns in consecutive rows merged.
  void TruthTileRects(const TruthImage& image, const std::bitset<2048>& tiles, std::vector<std::array<int32_t, 4>>& out)
      const;
  void TruthOnFrameEnd(bool ended_truth, bool next_truth);
  void LogTruthCensus();

  // Census (gpu_native_census): shaders seen per frame and since start.
  std::unordered_map<uint64_t, uint32_t> census_vs_seen_, census_ps_seen_;  // hash -> first frame seen
  uint32_t census_frame_vs_ = 0, census_frame_ps_ = 0, census_new_vs_ = 0, census_new_ps_ = 0;
  uint32_t census_draws_ = 0, census_memexport_draws_ = 0;
  std::vector<uint64_t> census_frame_hashes_;

  // wwe2k14 patch (Session 66, G1.0): the resolve census (gpu_native_alias_census). Per destination (the texture, else
  // its guest page) the resolves not yet classified; classified when something samples the destination, when a later
  // resolve covers or touches the rectangle, or 2 frames after the resolve.
  enum AliasClass : uint32_t { kAliasDead, kAliasSampled, kAliasCpu, kAliasPartial, kAliasStale, kAliasClasses };
  enum AliasPath : uint32_t {
    kAliasDirectColor, kAliasDirectDepth, kAliasScaledFull, kAliasFillOnly, kAliasFull1x, kAliasStaged,
    kAliasTarget, kAliasStock, kAliasPaths
  };
  struct AliasEntry {
    uint64_t frame = 0;
    uint8_t kind = 0, path = 0;  // kind: 0 colour, 1 depth
    bool deferred = false, texonly = false, scaled = false, tiny = false;
    int32_t rect[4] = {};  // x0 y0 x1 y1 in the destination's texels (from the texture's origin, guest texels)
    uint64_t pixels = 0;
  };
  struct AliasStats {
    uint32_t resolves = 0, classified = 0;
    uint64_t pixels = 0, classified_pixels = 0;
    uint32_t cls[kAliasClasses] = {};
    uint64_t cls_px[kAliasClasses] = {};
    uint32_t kind_resolves[2] = {}, kind_cls[2][kAliasClasses] = {};
    uint32_t path_resolves[kAliasPaths] = {}, path_cls[kAliasPaths][kAliasClasses] = {};
    uint32_t deferred = 0, deferred_cls[kAliasClasses] = {};
    uint32_t texonly = 0, scaled = 0, no_texture = 0;
    uint32_t tiny = 0, tiny_cls[kAliasClasses] = {};
    uint64_t tiny_px = 0;
  };
  using AliasRect = std::array<int32_t, 4>;
  struct AliasTiny {  // tiny (<= 128x128) resolves: where they land
    std::unordered_map<uint64_t, std::vector<AliasRect>> rects;  // destination -> distinct rectangles
    std::unordered_set<uint32_t> pages;                          // distinct texture base pages
    void Clear() {
      rects.clear();
      pages.clear();
    }
    size_t Atlases() const {
      size_t n = 0;
      for (const auto& e : rects) n += e.second.size() > 1;
      return n;
    }
  };
  int32_t alias_frames_left_ = 0, alias_frames_total_ = 0;
  void* alias_texture_ = nullptr;     // the texture the current resolve wrote (captured in AfterResolveMarked)
  int32_t alias_path_ = -1;           // the path that served the current resolve (-1: not named)
  uint32_t alias_deferred_mark_ = 0;  // truth_counts_ deferred + fills deferred when the resolve began
  std::unordered_map<uint64_t, std::vector<AliasEntry>> alias_pending_;
  AliasStats alias_frame_, alias_run_;
  AliasTiny alias_tiny_frame_, alias_tiny_run_;
  uint64_t alias_tiny_sum_[3] = {};  // over the run, summed per frame: tiny resolves, textures, pages
  uint64_t alias_tiny_atlas_sum_ = 0;
  uint32_t alias_tiny_frames_ = 0;   // frames with at least one tiny resolve
  void AliasBeginResolve();
  void AliasNoteResolve(const draw_util::ResolveInfo& info);
  void AliasClassify(const AliasEntry& e, AliasClass cls);
  void AliasOnFrameEnd();
  void AliasStart(int32_t frames);
  void AliasLogSummary();

  // Session 95 (gpu_owned_copy_lean, owned_copy_lean.cpp): fewer full-size copies when a pass goes on after the game's copies of it.
  struct OwnedLeanLearn {  // per pass surface; bit N = after its copy number N (OwnedPass::copy_count)
    uint32_t base = 0, pitch_tiles = 0, kind = 0, width = 0, height = 0;
    uint32_t restore_next = 0, plain_next = 0;  // the next draw was a restore / an ordinary draw
    uint32_t unread = 0;                        // the pages a move left behind were replaced unread
    uint32_t repeats = 0;                       // the draws after this copy were drawn again after a restore (gpu_owned_restore_skip)
  };
  struct OwnedLeanPending {  // an image pages were left on: replaced unread? (end > begin: its copy can still be turned off)
    int16_t image = -1;
    size_t begin = 0, end = 0;
    uint64_t submission = 0;
    size_t learn = 0;
    uint32_t bit = 0;
    bool read = false;
  };
  struct OwnedLeanStats {
    uint32_t frames = 0, alias_no_depth = 0, alias_page_on_image = 0, alias_restore_next = 0, fresh = 0, fresh_missed = 0, moved = 0, detached = 0,
             detach_refused = 0, no_cover = 0, dropped = 0, kept_read = 0, kept_held = 0, nop_failed = 0, watched_unread = 0;
    uint32_t draws_after[8] = {}, depth_writes_after[8] = {};  // (by copy number: the proof a later depth-free continuation needs)
  } owned_lean_stats_;
  std::vector<OwnedLeanLearn> owned_lean_learn_;
  std::vector<OwnedLeanPending> owned_lean_pending_;
  bool owned_lean_first_line_ = false, owned_lean_first_drop_ = false;
  uint32_t OwnedLeanBits() const;
  size_t OwnedLeanEntry(const OwnedPass& p);
  bool OwnedLeanAlias(const OwnedPass& p);  // this whole colour copy needs no learned snapshot
  void OwnedLeanNoteDraw(const OwnedPass& p, const Draw& draw, const int32_t rect[4]);
  bool OwnedLeanCovers(const OwnedPass& p, const Draw& draw, const int32_t rect[4]);
  bool OwnedLeanFresh(OwnedPass& p);
  bool OwnedLeanDetach(OwnedPass& p, size_t learn, uint32_t bit);
  bool OwnedLeanGoOn(OwnedPass& p, const Draw& draw, const int32_t rect[4]);  // OwnedContinueCopy with the switch's routes first
  void OwnedLeanNoteRead(int16_t image);
  void OwnedLeanSettle(bool submission_end);
  void OwnedLeanFrame();
  // Session 95 (gpu_owned_restore_skip, owned_restore_skip.cpp): a restore whose following draws repeat the draws since the last copy.
  struct OwnedRsSig {
    uint64_t vs = 0, ps = 0;  // (Session 95 r5: the log only)
    uint64_t state = 0, geometry = 0, sets = 0, consts = 0;
    bool plain = false, opaque = false, used = false;
    std::vector<uint8_t> bytes[5];  // (bit 8, diagnosis: the constant buffers themselves)
    VkDescriptorSet set_list[8] = {};
    std::vector<uint32_t> vertex_floats;  // the vertex shader's float constants, word by word
  };
  struct OwnedRsStats {
    uint32_t frames = 0, restores = 0, full = 0, differ = 0, differ_state = 0, differ_geometry = 0, differ_sets = 0, differ_consts = 0, differ_plain = 0, lines = 0;
    uint32_t armed = 0, stayed = 0, skipped = 0, copies_off = 0, disarmed = 0, nop_failed = 0, other_reads = 0, stale_reads = 0, not_covered = 0, other_pages = 0;
    uint64_t ref_draws = 0, differ_at = 0, near_words = 0, out_of_order = 0;
    uint64_t out_of_order_logged = 0, out_of_order_slot = 0;  // (Session 95, review r5, 5: bit 16's line)
  } owned_rs_stats_;
  struct OwnedRsArm {  // this frame: the pass stayed in its image, its pages took a copy that can still be turned off
    bool armed = false, restore_seen = false, proved = false, other_read = false;
    int16_t image = -1;  // where the pages went
    size_t copy_begin = 0, copy_end = 0, nop_begin = 0;
    uint64_t submission = 0;
    uint32_t bit = 0;           // the copy the arm began on (a later copy starts another list of draws)
    int16_t pass_image = -1;    // the image the pass draws in
  } owned_rs_arm_;
  uint32_t owned_rs_learn_bit_ = 0;  // (the copy number the running list of draws follows)
  size_t owned_rs_learn_entry_ = 0;
  uint32_t owned_rs_good_ = 0;  // whole repeats in a row, no stale read
  uint64_t owned_rs_last_restore_ = 0;  // the frame of the last restore recorded (no list of draws is kept far from one)
  int16_t owned_rs_left_image_ = -1;   // watching: the image a move left the pages on
  int16_t owned_rs_stale_image_ = -1;  // this frame: the image the restore read
  std::vector<std::pair<uint32_t, int16_t>> owned_rs_stale_;  // pages the restore read, held past their frame: a read before the game replaces them proves the skip wrong
  std::vector<OwnedRsSig> owned_rs_seq_, owned_rs_ref_;
  size_t owned_rs_at_ = 0;
  bool owned_rs_compare_ = false;
  OwnedRsSig OwnedRsSigOf(const Draw& draw, const int32_t rect[4]);
  void OwnedRsNote(const OwnedPass& p, const Draw& draw, const int32_t rect[4]);
  void OwnedRsFrame();
  struct OwnedRsPass {  // (bit 2, watch only: what each render pass of the frame was given, against the same pass of the frame before)
    uint64_t ps = 0, all = 0, no_sets = 0;
    uint32_t width = 0, height = 0, draws = 0;
  };
  struct OwnedRsTally {
    uint64_t ps = 0;
    uint32_t width = 0, height = 0, passes = 0, same = 0, same_no_sets = 0, unmatched = 0;
  };
  std::vector<OwnedRsPass> owned_rs_passes_, owned_rs_passes_prev_;
  std::vector<OwnedRsTally> owned_rs_tally_;
  void OwnedRsPassNote(const OwnedPass& p, const Draw& draw, const int32_t rect[4]);
  bool OwnedRsGoOn(OwnedPass& p, const Draw& draw, const int32_t rect[4]);  // true: the pass goes on in its image
  void OwnedRsMoved(int16_t old_image);
  void OwnedRsNopBegin();
  void OwnedRsAfterDraw(const Draw& draw);
  void OwnedRsPageRead(uint32_t page);
  void OwnedRsPagesChanged();
  void OwnedRsSubmissionEnd();
};

}  // namespace rex::graphics::vulkan::native
