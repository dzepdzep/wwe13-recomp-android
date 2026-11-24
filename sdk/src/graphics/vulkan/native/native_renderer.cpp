/**
 * The native renderer (wwe2k14 patch, Session 38). See native_renderer.h for what it is and where it sits, and
 * docs\native\DESIGN.md in the wwe2k14-recomp repository for the design and the road map.
 *
 * Reading order: the settings, the shaders (GLSL, compiled at run time), then the frame in call order: OnDraw ->
 * Flush -> OnResolve (parity) -> OnFrameEnd -> OnSwapPrepare / OnSwapCompose.
 */

#include <rex/graphics/vulkan/native/native_renderer.h>
#include "diag_layer.h"  // Session 95 (the diagnostic run)
#include "../s58_stats.h"  // wwe2k14 patch (Session 58)
#include "../disk_pipeline_cache.h"  // Session 63
#include "../spirv_opt_store.h"     // Session 70 (the precompile waits for its optimiser)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <fmt/format.h>
#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/logging.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/pipeline_cache.h>
#include <rex/ui/vulkan/util.h>
#include <rex/ui/renderdoc_api.h>  // Session 69 (gpu_native_truth_debug_renderdoc)
#include <rex/ui/vulkan/instance.h>
#include <rex/platform.h>  // Session 70: <windows.h> on Windows (SetThreadPriority)
#include "index_pool.h"
#include "skin_once.h"
#include "vertex_input.h"
#include "vertex_mirror.h"
#include "shader_swap.h"   // Session 72 (gpu_native_shader_swap)
#include "owned_dead_clear.h"  // Session 93 (M2dEdramLess: EnsurePostObjects' RGBA32F blend requirement)
#include <rex/perf/frame_events.h>  // Session 73 (what a frame did)
#include "../../pipeline/shader/spirv_relaxed_precision.h"  // Session 73 (characters at full precision)

// Session 69 (gpu_native_truth_debug_renderdoc, native_truth.cpp): the strip clear's RenderDoc capture.
extern bool g_truth_rdc_active, g_truth_rdc_done;
extern uint64_t g_truth_rdc_frame;

// ---------------------------------------------------------------------------------------------------------------
// Settings. All of them can be changed while the game runs (`cvar NAME VALUE` from a script or the live channel).
// ---------------------------------------------------------------------------------------------------------------

REXCVAR_DEFINE_INT32(gpu_diag_pipelines_pending, 0, "GPU/Native",
                     "Session 95 (the diagnostic run, read-out only): how many native pipelines are still being made "
                     "(the diagnostic run's scenes wait for 0 before a picture). Written by the renderer, not a setting");
REXCVAR_DEFINE_INT32(gpu_native, 0, "GPU/Native",
                     "Native renderer (Session 38): 0 off, 1 side-car (the emulated renderer draws and presents as "
                     "always; the native renderer draws the scene's opaque geometry into its own targets beside it), "
                     "2 replace (Session 39: the native renderer draws the scene draws it can draw exactly instead of "
                     "the emulated renderer, with the game's own shaders, and writes the picture into the EDRAM for "
                     "everything after; needs the interlock render path)")
    .range(0, 2)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_check_writeback, false, "GPU/Native",
                    "With gpu_native 2: at the scene's resolves, the parity checks compare the EDRAM right after our "
                    "write-back with our targets (a self-check of the write-back: 100% expected)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_min_draws, 200, "GPU/Native",
                     "With gpu_native 2 (Session 40): a segment of the scene (the claimable draws between two points "
                     "where the EDRAM changes under our targets: the frame's start, an emulated draw, a clearing "
                     "resolve) is drawn natively only if the same segment of the previous frame had at least this many "
                     "claimable draws; smaller ones stay emulated (a native segment costs a full-screen load and a "
                     "write-back). 0 = every segment native (Session 39)")
    .range(0, 100000)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_writeback_rect, true, "GPU/Native",
                    "With gpu_native 2 (Session 40): the write-back covers only the channels (colour, depth) and the "
                    "rectangle (the draws' scissors) our draws changed since the last one; false = always all of the "
                    "scene's surface (Session 39)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_post, false, "GPU/Native",
                    "With gpu_native 2 (Session 42): the colour-only draws into 1x k_8_8_8_8 surfaces other than the "
                    "scene's, anywhere in the frame, and any after the scene's colour resolve (the post-processing "
                    "passes and the passes before the scene of the same shape: full-screen quads, with the emulated pipeline's "
                    "own geometry shader for quads and rectangles) are drawn natively into a target per surface and "
                    "the pixels they covered are written into the EDRAM before the resolve, exactly as the emulated "
                    "pixel shaders pack them")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_after_scene, false, "GPU/Native",
                    "With gpu_native 2 (Session 42): the scene's targets also take the scene-target draws after its "
                    "colour resolve (entrances draw a second scene segment there: particles, point sprites, the "
                    "restore of colour and depth from textures) and rectangle / quad / point lists and screen-space "
                    "draws into the scene, with the emulated pipeline's geometry shaders")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_post_blend, false, "GPU/Native",
                    "With gpu_native_post (Session 42): also post draws that blend or keep some channels (loaded from "
                    "the EDRAM first). Not exact: the host blends the float target without the emulated renderer's "
                    "clamp of the source and blend constant to 0-1 and without rounding to 8 bits after each draw "
                    "(wrong where a shader outputs more than 1); false keeps them emulated (every post resolve exact)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_post_depth, false, "GPU/Native",
                    "With gpu_native_post (Session 43): post targets also take colour draws that test or write depth "
                    "in a 20e4 depth surface without stencil or polygon offset (the passes before the scene: the "
                    "small maps, the half-size passes). Their pixel shaders round the depth to 20e4 as the interlock "
                    "shader does, the target's depth is loaded from the EDRAM at a segment's start, and the depth "
                    "the draws wrote goes back into the EDRAM with the colour; exact")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_post_16bit, false, "GPU/Native",
                    "With gpu_native_post (Session 43): post targets also take draws into k_16_16 and k_16_16_16_16 "
                    "surfaces (the EDRAM's 16-bit fixed point, -32 to 32; the half-size passes before the scene), "
                    "written back packed as the interlock shader packs them (64bpp addressing for k_16_16_16_16); "
                    "draws that blend or keep channels in them stay emulated")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_post_quantize, false, "GPU/Native",
                    "With gpu_native_post (Session 43): post targets with the EDRAM's own precision - RGBA8 for "
                    "k_8_8_8_8 surfaces, RGBA16_SNORM for k_16_16 / k_16_16_16_16 - instead of RGBA32F: the native "
                    "pixel shader rounds its colour exactly as the interlock shader packs it, so the target holds the "
                    "packed values (exact) at a half to a quarter of the memory traffic")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_post_min_pixels, 0, "GPU/Native",
                     "With gpu_native_post (Session 43): post draws whose scissor covers fewer guest pixels than this "
                     "stay emulated (a tiny pass - the 20 64x64 maps before the scene - pays a load, a write-back and "
                     "their barriers natively); 0 = any size")
    .range(0, 16777216)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_cover_stats, false, "GPU/Native",
                    "With gpu_native_post (Session 43, a diagnostic): counts the resolves of native post surfaces whose "
                    "whole rectangle a claimed quad / rectangle draw provably covered (its corners from the vertex "
                    "shader run on the CPU, nothing discarding pixels) - what a native resolve could serve without the "
                    "EDRAM; in the [native][post] report")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 94, nogs): see NativeVsExpandNote.
REXCVAR_DEFINE_BOOL(gpu_native_vs_expand, false, "GPU/Native",
                    "Session 94: the typed native draws take the host primitive types the vertex shader expands "
                    "(rectangle lists and point sprites as two-triangle strips: vulkan_force_expand_rectangle_lists_in_vs / "
                    "vulkan_force_expand_point_sprites_in_vs; quad lists converted to triangle lists already are plain "
                    "triangle draws) as 'expanded' draws with no geometry shader, so geometry shaders are never built for "
                    "NATIVE (gpu_quirks_mobile_no_gs turns this on). Off: such draws leave the native path")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(gpu_native_scene_extras, false, "GPU/Native",
                    "With gpu_native 2 (Session 43): the part of gpu_native_after_scene that applies before the scene's "
                    "colour resolve - the scene's targets take rectangle / quad / point lists (the emulated pipeline's "
                    "geometry shaders: Brock's lens-glare sprites) and screen-space draws into the scene, so they no "
                    "longer split the native scene with a write-back and a reload; the scene's parity, not exact")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_post_lean_depth, false, "GPU/Native",
                    "With gpu_native_post (Session 44): the render passes of a post segment without depth (no draw in "
                    "it tests or writes depth) neither load nor store the depth target (the stencil's coverage marks "
                    "as always); exact by construction, saves the depth traffic of colour-only post passes")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_merge, false, "GPU/Native",
                    "With gpu_native 2 (Session 44): consecutive claimed repeats of a crowd draw (the shaders "
                    "vulkan_repeat_draw_merge instances: only their vertex constants change, uploaded one stride apart) "
                    "become one instanced native draw, as the emulated renderer merges them; the same vertex shader "
                    "arithmetic, in the same order. [native][merge] in the report")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_all, false, "GPU/Native",
                    "With gpu_native 2 (Session 45, step 2 N1): every draw of a frame the native renderer can draw - "
                    "turns on gpu_native_post, _post_depth, _post_16bit, _post_quantize, _post_blend, _after_scene, "
                    "_scene_extras, _merge, _post_sync, _clear_mirror, _resolve, _resolve_texture and _fuse_writeback, draws every "
                    "scene segment natively (no gpu_native_min_draws), and makes every clear rectangle the fast clear "
                    "can prove an EDRAM fill (any size). Not exact (host blending)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_post_sync, false, "GPU/Native",
                    "With gpu_native_post (Session 45, step 2 N2): each post target keeps its pixels from segment to "
                    "segment and knows which EDRAM tiles it equals (loaded or written back by it, nothing else written "
                    "there since: fills, emulated draws, other targets' write-backs, clearing resolves, the scene's "
                    "write-back and the frame's end take tiles away); a segment that needs what is under its draws "
                    "loads only when the target does not hold it. Exact by construction (turns "
                    "gpu_native_post_lean_depth off)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_resolve, false, "GPU/Native",
                    "With gpu_native_post (Session 45, step 2 N2c): a colour resolve of a 1x k_8_8_8_8 post surface "
                    "whose segment was just written back is served from the post target (the pixels its draws wrote) "
                    "and the EDRAM (the rest) by a copy of the stock fast resolve shader, instead of from the EDRAM "
                    "alone; exact (the same bytes), the base for resolves without the write-back")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_fuse_writeback, false, "GPU/Native",
                    "With gpu_native_resolve and gpu_native_post_sync (Session 45, step 2 N2e): a post segment's colour "
                    "write-back is done by the resolve served from its target - the same dispatch stores each pixel the "
                    "segment drew into the EDRAM as the write-back stores it - when the write-back's rectangle lies "
                    "inside the resolve's (else it is written back first, as before): one pass instead of two. Exact")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_all_exclude, 0, "GPU/Native",
                     "Diagnostic (Session 45): parts of gpu_native_all left out, as a bit mask - 1 native blending, 2 "
                     "every scene segment native, 4 fast clears of any size, 8 post sync (and mirrors, fused "
                     "write-back), 16 clear mirrors, 32 resolves from targets (and texture fills), 64 merge, 128 "
                     "after-scene / scene extras; for cost breakdown A/Bs")
    .range(0, 255)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_resolve_texture_stock, false, "GPU/Native",
                    "With gpu_native_resolve_texture (Session 45, step 2 N3b): a fast 32bpp 1x colour resolve that no "
                    "post target serves (the scene's own copies, a surface whose segment ended earlier) also fills the "
                    "texture it feeds: the transcribed resolve shader with every pixel from the EDRAM (the stock copy's "
                    "bytes) runs in place of the stock copy when such a texture exists and qualifies. Exact")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_resolve_texture, false, "GPU/Native",
                    "With gpu_native_resolve and vulkan_texture_load_direct_unscaled (Session 45, step 2 N3a): a resolve "
                    "served from a post target also stores its pixels into the texture it feeds (the 1x k_8_8_8_8 tiled "
                    "texture last loaded over its page, still up to date, with the destination's pitch and endianness), "
                    "which then counts as loaded: no texture load from guest memory follows")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_clear_mirror, false, "GPU/Native",
                    "With gpu_native_post_sync (Session 45, step 2 N1.2): a fast clear's fill is also drawn into the "
                    "native target of its surface (post targets of its pitch, the scene's) as a constant - every pixel "
                    "the fill's own dword, no read of the EDRAM - at that target's next segment, when a draw there "
                    "reads what the fill wrote; in place of loading those pixels from the EDRAM. Exact by "
                    "construction. Part of gpu_native_all")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_defer_writeback, false, "GPU/Native",
                    "With gpu_native_resolve and gpu_native_post_sync (Session 45, step 2 N2d): a post segment whose "
                    "colour resolve is served from its target does not write its colour back; it stays pending in the "
                    "target and goes into the EDRAM only when something reads those tiles there (a stock resolve, a "
                    "load, an emulated draw, a fill that does not cover it all, the frame's end), is dropped when a fill "
                    "overwrites it all, and the next segment of the same surface on the target carries it on. PARKED "
                    "(Session 45): NOT exact yet (the frame's last resolve differs, runs 2909 / 2910: cause not found) "
                    "and it saves nothing in 2K14 (37 of 40 deferred colour write-backs a match frame are written later "
                    "anyway: every pass at EDRAM base 0 overlaps the next); measurement only")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_clip_disable_clamp, true, "GPU/Native",
                    "Session 69 (the testers' red block): a draw with the guest's clipping off (PA_CL_CLIP_CNTL "
                    "clip_disable) gets a depth-clamping NATIVE pipeline, as the emulated pipeline does (false: "
                    "clipped - the old behaviour that dropped a damage-map clear whose z was a hair below 0)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_segment_log, 0, "GPU/Native",
                     "With gpu_native 2 (Session 45, a diagnostic): a [native][seg] line for each native segment, load "
                     "and write-back of the next N replaced frames (target, size, why it loads, draws, rectangle)")
    .range(0, 1000)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_verify_every, 0, "GPU/Native",
                     "With gpu_native 2: every Nth frame runs side-car instead (both renderers draw the scene), so the "
                     "parity numbers keep coming; 0 = never")
    .range(0, 100000)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_view, 0, "GPU/Native",
                     "With gpu_native: what the screen shows. 0 the emulated picture, 1 the native picture, 2 split "
                     "(the emulated picture left of gpu_native_split, native right), 3 split of the scene before "
                     "post-processing (the emulated scene from the EDRAM left, native right), 4 the emulated scene "
                     "before post-processing (3 and 4 need gpu_native_shading 1)")
    .range(0, 4)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_shading, 0, "GPU/Native",
                     "With gpu_native: 0 our own clay shading (a tint per vertex shader), 1 the game's own pixel "
                     "shaders, translated for host render targets on this machine")
    .range(0, 1)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_draws, 0, "GPU/Native",
                     "With gpu_native: which draws into the scene's render target the native renderer draws: 0 those "
                     "that write depth (opaque geometry), 1 all of them (with their blending) up to the scene's "
                     "depth resolve")
    .range(0, 1)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_DOUBLE(gpu_native_split, 0.5, "GPU/Native",
                      "gpu_native_view 2: where the split is, 0 (left edge) to 1 (right edge)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_DOUBLE(gpu_native_scale, 1.0, "GPU/Native",
                      "The native targets' size relative to the emulated scene (1 = the same pixels; 0.5 = half "
                      "width and height; any value from 0.25 to 2)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_report_frames, 60, "GPU/Native",
                     "With gpu_native: a [native] report line (parity and cost) every N frames, 0 = none")
    .range(0, 100000)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_gpu_timers, true, "GPU/Native",
                    "Session 52: the native renderer's own GPU timestamps around its passes, stores, resolves and "
                    "fills (the GPU ms in the [native] report and census lines); false = none written (those numbers "
                    "read 0)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_debug_rdc_now, false, "GPU/Native",
                    "Session 82 (diagnostic): start a RenderDoc capture at this frame's end (the next two frames are in it) - the "
                    "same frame of two modes of one replay; needs ENABLE_VULKAN_RENDERDOC_CAPTURE=1 and RenderDoc on PATH")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_dump, 0, "GPU/Native",
                     "With gpu_native and gpu_native_shading 1: save the emulated scene (from the EDRAM) and the native "
                     "picture of the same frame as BMP files in gpu_native_dump_dir (native_FRAME_emulated.bmp, "
                     "native_FRAME_native.bmp), for the next N frames that have both; counts down to 0")
    .range(0, 1000)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_STRING(gpu_native_dump_dir, "captures", "GPU/Native",
                      "gpu_native_dump: the folder (relative to the working folder, or absolute)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 54, docs\native\NO_INTERLOCK_S54.md in the wwe2k14-recomp repository).
REXCVAR_DEFINE_BOOL(gpu_native_truth_host_rt, false, "GPU/Native",
                    "Session 54: gpu_native 2 with truth frames on the host render target path (render_target_path_vulkan "
                    "fbo: graphics cards without fragment shader interlock). Truth frames only - a frame that is not "
                    "one is emulated as without gpu_native; at a fallback the images stay the picture (no EDRAM to "
                    "write them into: the rest of that frame may be wrong). Off: gpu_native 2 is off on that path")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth, false, "GPU/Native",
                    "With gpu_native 2 (Session 46, the no-EDRAM frame): our own images are the picture - the EDRAM "
                    "buffer is neither read nor written in a truth frame (docs\\native\\NO_EDRAM_DESIGN.md). Turns on "
                    "gpu_native_all and gpu_native_lean itself (logged); refused (logged) with EDRAM renaming, DLSS or "
                    "DLAA on, and in gpu_native_verify_every's side-car frames. Not exact (the pictures are measured "
                    "with the parity shots); the game's behaviour is unchanged. Off by default (LOW only)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_direct_resolve_scaled, false, "GPU/Native",
                    "With gpu_native_truth_direct_resolve at a resolution scale (Session 52): a fast 32bpp colour copy of "
                    "a surface one RGBA8 truth image holds whole is done straight from the image into the scaled resolve "
                    "buffer (the stock scaled copy's layout), not through the staging buffer and the stock copy")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_direct_resolve, true, "GPU/Native",
                    "With gpu_native_truth (Session 46): a fast 32bpp 1x colour copy of a surface one RGBA8 image holds "
                    "whole is done straight from the image (and fills the texture it feeds); false = every copy reads "
                    "the staging buffer the image is stored into first")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 61, Jarrod: "move DLSS and DLAA to the native renderer"): truth frames with the upscalers.
REXCVAR_DEFINE_BOOL(gpu_native_truth_upscalers, true, "GPU/Native",
                    "Session 61: truth frames with DLSS / DLAA / FSR too (refused before): the pre-HUD copy from the "
                    "truth images, the clears still fast clears under the jitter, the characters' positions captured "
                    "by their truth draws. Runs 4117-4119: 100% truth frames. false = refused, as before");
// wwe2k14 patch (Session 61): the native renderer's pipelines kept between runs and made again ahead of the draws.
REXCVAR_DEFINE_BOOL(gpu_native_pipeline_storage, true, "GPU/Native",
                    "Session 61: the native renderer's pipelines are stored as records next to the shader storage "
                    "(<cache>\\shaders\\shareable\\TITLE.native.fsi|fbo.vk.xnps, needs store_shaders) and every new one "
                    "is added");
REXCVAR_DEFINE_INT32(gpu_native_pipeline_prewarm_ms, 4, "GPU/Native",
                     "Session 61: from the first frames the native renderer is on, the stored pipelines are queued for "
                     "its pipeline worker thread, their shaders translated on the command thread within this many ms "
                     "a frame (0 = not made ahead). Uses the worker whatever gpu_native_truth_async_pipelines says");
REXCVAR_DEFINE_BOOL(gpu_native_truth_async_pipelines, true, "GPU/Native",
                    "With gpu_native_truth (Session 47): truth pipelines are made on a worker thread, not the command "
                    "thread; a truth draw whose pipeline is not ready yet is skipped, as the emulated renderer skips "
                    "one (vulkan_pipeline_no_submit_wait). false = made at once on the command thread (Session 46)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 55, the red blocks): a 2-core laptop (i3-6100U, HD 520) skipped 109,595 truth draws while
// their pipelines were made, this PC ~1,400. Diagnostic: the worker waits this long before each pipeline, as a slow
// driver compiles, so the draws skipped meanwhile can be studied here.
REXCVAR_DEFINE_INT32(gpu_native_pipeline_workers, 0, "GPU/Native",
                     "Session 69 (the Intel tester's stutter): worker threads making NATIVE pipelines (stored ones at "
                     "start, truth draws' new ones); 0 = a third of the CPU threads, 1 to 3. Read when the first job "
                     "starts them");
REXCVAR_DEFINE_BOOL(gpu_native_pipeline_workers_quiet, true, "GPU/Native",
                    "Session 70 (v0.2.8's slow entrances): the NATIVE pipeline workers run below normal priority, so "
                    "the warm-up of stored pipelines never takes CPU time from the game's own threads; raised to "
                    "normal while the command thread waits for one of them. Windows; read when the workers start");
// wwe2k14 patch (Session 80, HD 520 arc): a draw whose pipeline is not made is skipped, not waited for, so the raise
// above never happened for it: with the CPU busy the quiet workers sat in their below-normal warm-up jobs, the draw's
// own job (queued first) waited behind them, and a scene skipped draws for seconds (this PC with other games running:
// 287,233 and 5,032 skipped draws in two runs; the same scenes alone: 8). The HD 520 has one worker (4 threads / 3).
// Off by default (Jarrod, 2026-11-01: "Merge, switched off"): no gain measured on this PC (crowded-CPU rounds and a
// cold-cache 4-thread run: the long skip stretches did not recur) - a lever for an Intel measurement.
REXCVAR_DEFINE_BOOL(gpu_native_pipeline_workers_raise, false, "GPU/Native",
                    "Session 80: with gpu_native_pipeline_workers_quiet, the workers busy with the warm-up are raised "
                    "to normal priority when a skipped draw's pipeline is queued, so its job starts sooner (each "
                    "worker drops back at its next warm-up job). Windows; off (no gain measured on the PC)");
REXCVAR_DEFINE_BOOL(gpu_native_precompile, false, "GPU/Native",
                    "Session 70 (the launcher's precompile): the stored NATIVE pipelines made as fast as the PC allows "
                    "- no per-frame budget, a worker per CPU thread at normal priority, retries every 10 frames; "
                    "progress in gpu_native_precompile_status. Read when the workers start");
REXCVAR_DEFINE_STRING(gpu_native_precompile_status, "", "GPU/Native",
                      "Session 70: written with gpu_native_precompile, every frame: 'made M of T, waiting W, done D, optimised O' "
                      "(D 1 = nothing left to make or to wait for)");
REXCVAR_DEFINE_INT32(gpu_native_truth_debug_pipeline_delay_ms, 0, "GPU/Native",
                     "Diagnostic (Session 55): every native pipeline creation (the worker thread's and the command "
                     "thread's) waits this many milliseconds first (a slow laptop's compile time); 0 = none")
    .range(0, 10000)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_truth_texture_fill, 63, "GPU/Native",
                     "With gpu_native_truth (Session 47, M6): a 1x resolve also fills the texture it feeds from the "
                     "native image, so the texture is not loaded again (the guest memory is written as before). Bit "
                     "mask of the kinds filled: 1 k_24_8_FLOAT depth, 2 k_8_8_8_8, 4 k_8_8_8_8 with the red / blue "
                     "swap, 8 k_16_16_16_16, 16 k_16_16, 32 k_8 / k_4_4_4_4 (mip levels too); 0 = none")
    .range(0, 63)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_skip_writeback, false, "GPU/Native",
                    "With gpu_native_truth at 1x (Session 49, the RAM ties): texture-only resolves - a resolve whose "
                    "texture fill holds every texel it copies (single-level k_8_8_8_8 textures) does not write guest "
                    "memory; the GPU copy of those pages is written from the texture only when something requests them "
                    "(texture loads, vertex / index fetch, memexport, another resolve over part of them), when the "
                    "record is gpu_native_truth_skip_writeback_max_age frames old, or when skipping stops. The CPU "
                    "never reads resolve output (readback_resolve none). Census: [native][wb]")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_truth_skip_writeback_max_age, 30, "GPU/Native",
                     "With gpu_native_truth_skip_writeback: frames after which a texture-only resolve nothing replaced "
                     "or read is written into guest memory anyway (1-600)")
    .range(1, 600)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_skip_writeback_debug_immediate, false, "GPU/Native",
                    "Diagnostic (Session 49): with gpu_native_truth_skip_writeback, every texture-only resolve is "
                    "written from its texture right away, where its copy would have written - resolve hashes then "
                    "check the materialise shader against the copy (no gain)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_texture_fill_lazy, false, "GPU/Native",
                    "With gpu_native_truth_texture_fill_scaled (Session 49): no fill into a scaled texture nothing sampled "
                    "since it was last filled or loaded (it is loaded from memory if sampled later, then filled again)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_texture_fill_scaled, false, "GPU/Native",
                    "With gpu_native_truth at a resolution scale (Session 48): fill scaled textures too (Session 47, "
                    "9f27fd0). Off by default: at resolution_scale 4 the fills cost +34% (match) to +83% (Bryan) GPU "
                    "time in truth frames (in-run A/B 3330) for 3 fewer texture loads a frame; pictures are identical "
                    "either way (Session 47, 3226 / 3240). Fills at 1x are not affected")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_truth_texture_fill_scaled_kinds, 63, "GPU/Native",
                     "With gpu_native_truth_texture_fill_scaled (Session 52): the kinds filled at a resolution scale, the "
                     "bits of gpu_native_truth_texture_fill (both must have the bit): 32 alone (k_8 / k_4_4_4_4) = no "
                     "reloads of the mip chains from the scaled resolve buffer (in-run A/B 3587 at 1440p: -10..-19% GPU)")
    .range(0, 63)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_truth_texture_fill_small_swap_scaled, 0, "GPU/Native",
                     "Diagnostic (Session 47): 1 = with gpu_native_truth at a resolution scale, red / blue swapped "
                     "k_8_8_8_8 textures under 128 guest texels are filled too (0: they are reloaded)")
    .range(0, 1)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_trust_bindings, true, "GPU/Native",
                    "With gpu_native_truth (Session 47, CPU): a truth draw does not bind its descriptor sets again where "
                    "the command processor has just bound them (only after a load pass of its own), and records its "
                    "pipeline, viewport, scissor, depth bias and blend constants only where they change; false = "
                    "every truth draw records them all (Session 46)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_truth_debug_draws, 0, "GPU/Native",
                     "Diagnostic (Session 46): a truth frame claims at most this many draws, then falls back (the rest "
                     "of the frame exact) - to find the pass where truth and exact frames part; 0 = no limit")
    .range(0, 1000000)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_census, 0, "GPU/Native",
                     "Census of the guest's command stream for the next N frames (gpu_native on or off): a "
                     "[native][census] line per frame (indirect buffers, occlusion queries, memexport draws, "
                     "shaders per frame and new since start); counts down to 0")
    .range(0, 100000)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_alias_census, 0, "GPU/Native",
                     "wwe2k14 patch (Session 66, G1.0): census of the guest resolves for the next N frames (gpu_native on): "
                     "per resolve destination whether it is overwritten whole before anything samples it (dead), "
                     "sampled (a draw's texture binding, a texture load, the memory path), or neither within 2 frames "
                     "(stale), per path, colour / depth and size, and where the tiny (<= 128x128) ones land; a "
                     "[native][alias] line per frame and a summary at the end, then back to 0. Counters only"
                     )
    .range(0, 100000)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

// Session 46 (gpu_native_truth): what a truth frame turns on, and what refuses it (defined in command_processor.cpp).
REXCVAR_DECLARE(bool, gpu_native_index_no_restart);  // (Session 95: vertex_mirror.cpp)
REXCVAR_DECLARE(bool, gpu_native_lean);
REXCVAR_DECLARE(bool, gpu_native_owned_alpha_learn);  // Session 82 (owned_steps.cpp)
REXCVAR_DECLARE(bool, gpu_native_owned_frame_tiles_memo);  // Session 87 (owned_steps.cpp)
REXCVAR_DECLARE(int32_t, gpu_native_frame_batch_log);  // wwe2k14 patch (Session 79)
REXCVAR_DECLARE(int32_t, gpu_native_truth_pass_log);  // Session 72 (native_truth.cpp)
REXCVAR_DECLARE(bool, gpu_native_shader_swap);        // Session 72 (shader_swap.cpp)
REXCVAR_DECLARE(bool, gpu_edram_rename);
REXCVAR_DECLARE(bool, gpu_dlss);
REXCVAR_DECLARE(bool, gpu_dlaa);

// wwe2k14 patch (Session 94, nogs): counts of the typed draws taken as VS-expanded (0 point, 1 rectangle) and the native
// pipelines built with a geometry shader.
namespace {
std::atomic<uint64_t> g_vs_expand_draws[2];
std::atomic<uint64_t> g_gs_pipelines;
}  // namespace
namespace rex::graphics::vulkan {
extern uint64_t g_draw_rec_generation;  // (Session 95, r3cmd: draw_record.h)
}
namespace rex::graphics::vulkan::native {
// True when a draw with this host vertex shader type and host primitive may be drawn as an 'expanded' native draw with
// no geometry shader (the vertex shader does the expansion). Counts it and logs once.
bool NativeVsExpandNote(Shader::HostVertexShaderType host_vs_type, xenos::PrimitiveType host_primitive, bool truth) {
  if (!REXCVAR_GET(gpu_native_vs_expand)) return false;
  const bool point = host_vs_type == Shader::HostVertexShaderType::kPointListAsTriangleStrip;
  const bool rect = host_vs_type == Shader::HostVertexShaderType::kRectangleListAsTriangleStrip;
  if ((!point && !rect) || host_primitive != xenos::PrimitiveType::kTriangleStrip) return false;
  // Session 94 review: rectangle lists keep the geometry-shader path's rule (Session 46): native only in a truth frame (the
  // clears); point sprites in any frame, as before.
  if (rect && !truth) return false;
  ++g_vs_expand_draws[rect ? 1 : 0];
  static std::atomic<bool> logged;
  if (!logged.exchange(true)) {
    REXGPU_INFO("[native] vs-expand on: typed draws take VS-expanded rectangle lists and point sprites as triangle strips "
                "without a geometry shader (first: {})", point ? "point sprites" : "rectangle list");
  }
  return true;
}
void NativeVsExpandReport() {
  if (!REXCVAR_GET(gpu_native_vs_expand)) return;
  REXGPU_INFO("[native] vs-expand: draws described so far (both builders, unclaimed ones too): point sprites {}, rectangle lists {}; geometry-shader pipelines made {}",
              g_vs_expand_draws[0].load(), g_vs_expand_draws[1].load(), g_gs_pipelines.load());
}

namespace {

// The scene's opaque geometry with the game's vertex shaders and our own shading ("clay"): a normal from the
// screen-space derivatives of the view position (clip w is the view depth of a perspective projection), one light
// from the upper left, a tint per vertex shader so the game's material groups can be told apart.
constexpr char kClayFragmentShader[] = R"(
#version 450
layout(constant_id = 0) const float kTintR = 0.8;
layout(constant_id = 1) const float kTintG = 0.8;
layout(constant_id = 2) const float kTintB = 0.8;
layout(constant_id = 3) const float kFocalPixels = 1700.0;  // an assumed vertical field of view of ~45 degrees
layout(constant_id = 4) const float kCenterX = 1280.0;
layout(constant_id = 5) const float kCenterY = 720.0;
layout(location = 0) out vec4 out_color;
void main() {
  float view_depth = 1.0 / gl_FragCoord.w;
  vec3 p = vec3((gl_FragCoord.xy - vec2(kCenterX, kCenterY)) * (view_depth / kFocalPixels), view_depth);
  vec3 n = normalize(cross(dFdx(p), dFdy(p)));
  // Window space has y down: this light comes from the upper left, towards the scene.
  float diffuse = abs(dot(n, normalize(vec3(-0.45, -0.65, 0.6))));
  float facing = abs(n.z);
  float shade = 0.18 + 0.62 * diffuse + 0.2 * facing;
  out_color = vec4(vec3(kTintR, kTintG, kTintB) * shade, 1.0);
}
)";

// The swap: our picture written over the emulated one in the guest output after the swap's gamma (and FXAA)
// compute passes wrote it (gpu_native_view: all of it, right of the split with a thin divider, or the scene split).
constexpr char kComposeComputeShader[] = R"(
#version 450
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform sampler2D native_color;
layout(set = 0, binding = 1, rgb10_a2) uniform writeonly image2D guest_output;
layout(set = 0, binding = 2) uniform sampler2D emulated_scene;
layout(push_constant) uniform Constants {
  uint width;
  uint height;
  float split;          // native right of this
  float divider_width;  // in UV
  uint view;            // gpu_native_view (1-4)
  uint gamma;           // 1: the scene target is gamma-encoded (ours is linear)
} c;
// The Xenos piecewise-linear gamma curve (xenos.cpp LinearToPWLGamma): for k_8_8_8_8_GAMMA scene targets, where the
// EDRAM holds gamma-encoded values and our targets hold linear ones.
vec3 LinearToPWLGamma(vec3 l) {
  l = clamp(l, 0.0, 1.0);
  vec3 hi = step(vec3(128.0 / 1023.0), l), top = step(vec3(512.0 / 1023.0), l), mid = step(vec3(64.0 / 1023.0), l);
  vec3 scale = mix(mix(vec3(1023.0), vec3(1023.0 / 2.0), mid), mix(vec3(1023.0 / 4.0), vec3(1023.0 / 8.0), top), hi);
  vec3 offset = mix(mix(vec3(0.0), vec3(32.0 / 255.0), mid), mix(vec3(64.0 / 255.0), vec3(128.0 / 255.0), top), hi);
  return trunc(l * scale) * (1.0 / 255.0) + offset;
}
vec4 Native(vec2 uv) {
  vec4 n = textureLod(native_color, uv, 0.0);
  return c.gamma != 0u ? vec4(LinearToPWLGamma(n.rgb), 1.0) : n;
}
void main() {
  uvec2 p = gl_GlobalInvocationID.xy;
  if (p.x >= c.width || p.y >= c.height) return;
  vec2 uv = (vec2(p) + 0.5) / vec2(c.width, c.height);
  vec4 color;
  if (c.view == 4u) {
    color = textureLod(emulated_scene, uv, 0.0);
  } else if (c.view == 1u) {
    color = Native(uv);
  } else if (uv.x < c.split) {
    if (c.view == 2u) return;  // the emulated picture stays
    color = textureLod(emulated_scene, uv, 0.0);
  } else if (uv.x < c.split + c.divider_width) {
    color = vec4(1.0);
  } else {
    color = Native(uv);
  }
  imageStore(guest_output, ivec2(p), vec4(color.rgb, 1.0));
}
)";

// The colour parity check, at the scene's colour resolve: the emulated scene colour from the EDRAM (a 32bpp k_8_8_8_8
// surface: R | G << 8 | B << 16 | A << 24, tiles not half-swapped) against ours, and a copy of it for the scene views.
constexpr char kColorParityComputeShader[] = R"(
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer EdramBuffer { uint edram[]; };
layout(set = 0, binding = 1) uniform sampler2D native_color;
layout(set = 0, binding = 2, std430) buffer Result { uint r[16]; };
layout(set = 0, binding = 3, rgba8) uniform writeonly image2D emulated_scene;
layout(push_constant) uniform Constants {
  uint base_dwords;
  uint pitch_tile_dwords;
  uint tile_w;
  uint tile_h;
  uint width;
  uint height;
  uint wrap_dwords;
  uint flags;
  float scale_x;
  float scale_y;
} c;
// Counted per workgroup in shared memory first (see the depth check).
shared uint counts[9];
// The Xenos piecewise-linear gamma curve (xenos.cpp LinearToPWLGamma): for k_8_8_8_8_GAMMA scene targets, where the
// EDRAM holds gamma-encoded values and our targets hold linear ones.
vec3 LinearToPWLGamma(vec3 l) {
  l = clamp(l, 0.0, 1.0);
  vec3 hi = step(vec3(128.0 / 1023.0), l), top = step(vec3(512.0 / 1023.0), l), mid = step(vec3(64.0 / 1023.0), l);
  vec3 scale = mix(mix(vec3(1023.0), vec3(1023.0 / 2.0), mid), mix(vec3(1023.0 / 4.0), vec3(1023.0 / 8.0), top), hi);
  vec3 offset = mix(mix(vec3(0.0), vec3(32.0 / 255.0), mid), mix(vec3(64.0 / 255.0), vec3(128.0 / 255.0), top), hi);
  return trunc(l * scale) * (1.0 / 255.0) + offset;
}


void Compare() {
  uvec2 p = gl_GlobalInvocationID.xy;
  if (p.x >= c.width || p.y >= c.height) return;
  uint tile_x = p.x / c.tile_w, tile_y = p.y / c.tile_h;
  uint offset = tile_y * c.pitch_tile_dwords + tile_x * (c.tile_w * c.tile_h) + (p.y % c.tile_h) * c.tile_w +
                p.x % c.tile_w;
  uint v = edram[(c.base_dwords + offset) % c.wrap_dwords];
  vec4 e = unpackUnorm4x8(v);
  imageStore(emulated_scene, ivec2(p), vec4(e.rgb, 1.0));
  ivec2 q = ivec2(min(vec2(p) * vec2(c.scale_x, c.scale_y), vec2(textureSize(native_color, 0) - 1)));
  vec4 n = texelFetch(native_color, q, 0);
  if ((c.flags & 1u) != 0u) n.rgb = LinearToPWLGamma(n.rgb);
  ivec3 d = abs(ivec3(round(e.rgb * 255.0)) - ivec3(round(n.rgb * 255.0)));
  int m = max(d.x, max(d.y, d.z));
  atomicAdd(counts[0], 1u);
  atomicAdd(counts[m == 0 ? 1 : m <= 2 ? 2 : m <= 8 ? 3 : m <= 32 ? 4 : 5], 1u);
  if (d.x != 0) atomicAdd(counts[6], uint(d.x));
  if (d.y != 0) atomicAdd(counts[7], uint(d.y));
  if (d.z != 0) atomicAdd(counts[8], uint(d.z));
  if (p.x == c.width / 2u && p.y == c.height / 2u) {
    r[9] = v;
    r[10] = packUnorm4x8(n);
  }
}

void main() {
  if (gl_LocalInvocationIndex < 9u) counts[gl_LocalInvocationIndex] = 0u;
  barrier();
  Compare();
  barrier();
  if (gl_LocalInvocationIndex < 9u && counts[gl_LocalInvocationIndex] != 0u) {
    atomicAdd(r[gl_LocalInvocationIndex], counts[gl_LocalInvocationIndex]);
  }
}
)";

// The parity check: the emulated depth of every pixel of the scene's depth surface, read from the EDRAM buffer
// (the fragment shader interlock path keeps depth there as (24-bit depth << 8) | stencil, with the two 40-sample
// halves of each 80-sample tile swapped; spirv_translator_rb.cpp FSI_LoadEdramOffsets), against our depth at the
// same place. Counts go to a small buffer the CPU reads once the frame has finished.
constexpr char kParityComputeShader[] = R"(
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer EdramBuffer { uint edram[]; };
layout(set = 0, binding = 1) uniform sampler2D native_depth;
layout(set = 0, binding = 2, std430) buffer Result { uint r[16]; };
layout(push_constant) uniform Constants {
  uint base_dwords;        // the depth surface's base (tiles) x the tile size (dwords)
  uint pitch_tile_dwords;  // one row of tiles
  uint tile_w;             // 80 x the resolution scale
  uint tile_h;             // 16 x the resolution scale
  uint width;              // host pixels of the emulated surface to compare
  uint height;
  uint wrap_dwords;        // the EDRAM size in dwords
  uint flags;              // bit 0: 20e4 floating-point depth, else 24-bit unorm
  float scale_x;           // native pixels per emulated pixel
  float scale_y;
  float clear_depth;       // what both targets hold where nothing was drawn
  float tolerance;         // relative
} c;

float DepthFromEdram(uint d24, bool is_float) {
  if (!is_float) return float(d24) / 16777215.0;
  uint e = d24 >> 20, m = d24 & 0xFFFFFu;
  if (e == 0u) return float(m) * exp2(-34.0);  // denormal: m x 2^-20 x 2^-14
  return uintBitsToFloat(((e + 112u) << 23) | (m << 3));
}

// Counted per workgroup in shared memory first: one global atomic per counter per 64 pixels (a global atomic per
// pixel on the same few addresses took ~100 ms a frame).
shared uint counts[11];
void Count(uint i) { atomicAdd(counts[i], 1u); }

void Compare() {
  uvec2 p = gl_GlobalInvocationID.xy;
  if (p.x >= c.width || p.y >= c.height) return;
  uint tile_x = p.x / c.tile_w, tile_y = p.y / c.tile_h;
  uint x = p.x % c.tile_w, y = p.y % c.tile_h;
  uint half_w = c.tile_w >> 1;
  x = x < half_w ? x + half_w : x - half_w;
  uint offset = tile_y * c.pitch_tile_dwords + tile_x * (c.tile_w * c.tile_h) + y * c.tile_w + x;
  uint v = edram[(c.base_dwords + offset) % c.wrap_dwords];
  float e = DepthFromEdram(v >> 8, (c.flags & 1u) != 0u);
  ivec2 q = ivec2(min(vec2(p) * vec2(c.scale_x, c.scale_y), vec2(textureSize(native_depth, 0) - 1)));
  float n = texelFetch(native_depth, q, 0).r;
  bool e_geometry = abs(e - c.clear_depth) > 1e-7;
  bool n_geometry = abs(n - c.clear_depth) > 1e-7;
  Count(0);
  if (!e_geometry && !n_geometry) {
    Count(1);
  } else if (!n_geometry) {
    Count(2);
  } else if (!e_geometry) {
    Count(3);
  } else {
    Count(4);
    float d = abs(e - n);
    if (d <= max(c.tolerance * e, 1e-7)) Count(5);
    Count(d < 1e-6 ? 6 : d < 1e-5 ? 7 : d < 1e-4 ? 8 : d < 1e-3 ? 9 : 10);
  }
  if (p.x == c.width / 2u && p.y == c.height / 2u) {
    r[11] = floatBitsToUint(e);
    r[12] = floatBitsToUint(n);
  }
}

void main() {
  if (gl_LocalInvocationIndex < 11u) counts[gl_LocalInvocationIndex] = 0u;
  barrier();
  Compare();
  barrier();
  if (gl_LocalInvocationIndex < 11u && counts[gl_LocalInvocationIndex] != 0u) {
    atomicAdd(r[gl_LocalInvocationIndex], counts[gl_LocalInvocationIndex]);
  }
}
)";

// Replacing (gpu_native 2): where the scene's colour and depth surfaces are in the EDRAM buffer (as the parity checks
// address them) and how our targets map onto them. Shared by the load and the write-back.
#define NATIVE_EDRAM_PLACE_GLSL                                                                             \
  "layout(push_constant) uniform Constants {\n"                                                             \
  "  uint color_base_dwords;\n"                                                                             \
  "  uint depth_base_dwords;\n"                                                                             \
  "  uint pitch_tile_dwords;\n"                                                                             \
  "  uint tile_w;\n"                                                                                        \
  "  uint tile_h;\n"                                                                                        \
  "  uint width;\n" /* emulated host pixels */                                                              \
  "  uint height;\n"                                                                                        \
  "  uint wrap_dwords;\n"                                                                                   \
  "  uint flags;\n" /* bit 0: 20e4 depth, else 24-bit unorm; bit 1: a colour surface */                                              \
  "  float scale_x;\n" /* native pixels per emulated pixel */                                               \
  "  float scale_y;\n"                                                                                      \
  "  uint origin_x;\n" /* write-back: its rectangle's first emulated pixel (Session 40) */                   \
  "  uint origin_y;\n"                                                                                      \
  "} c;\n"                                                                                                  \
  /* The dwords of emulated pixel p: colour (32bpp, tiles as they are), depth (the 40-sample halves of each */ \
  /* 80-sample tile row swapped). */                                                                        \
  "uvec2 EdramIndices(uvec2 p) {\n"                                                                         \
  "  uint tile = (p.y / c.tile_h) * c.pitch_tile_dwords + (p.x / c.tile_w) * (c.tile_w * c.tile_h);\n"      \
  "  uint x = p.x % c.tile_w, y = p.y % c.tile_h, half_w = c.tile_w >> 1;\n"                                \
  "  uint xd = x < half_w ? x + half_w : x - half_w;\n"                                                     \
  "  return uvec2((c.color_base_dwords + tile + y * c.tile_w + x) % c.wrap_dwords,\n"                       \
  "               (c.depth_base_dwords + tile + y * c.tile_w + xd) % c.wrap_dwords);\n"                     \
  "}\n"

// The load: a full-screen triangle at the start of our render pass, our colour and depth from the EDRAM.
constexpr char kLoadVertexShader[] = R"(
#version 450
void main() {
  vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";
constexpr char kLoadFragmentShader[] =
    "#version 450\n"
    "layout(set = 0, binding = 0, std430) readonly buffer EdramBuffer { uint edram[]; };\n"
    "layout(location = 0) out vec4 out_color;\n" NATIVE_EDRAM_PLACE_GLSL R"(
// As spirv_translator_rb.cpp Depth20e4To32 (and the depth parity check).
float DepthFromEdram(uint d24, bool is_float) {
  if (!is_float) return float(d24) / 16777215.0;
  uint e = d24 >> 20, m = d24 & 0xFFFFFu;
  if (e == 0u) return float(m) * exp2(-34.0);
  return uintBitsToFloat(((e + 112u) << 23) | (m << 3));
}
void main() {
  uvec2 p = uvec2(min(floor(gl_FragCoord.xy / vec2(c.scale_x, c.scale_y)),
                      vec2(float(c.width - 1u), float(c.height - 1u))));
  uvec2 i = EdramIndices(p);
  // Session 45 (clear mirrors): flags bit 12 - every pixel takes one dword each (origin_x colour, origin_y depth): a
  // fast clear's fill as it wrote them (the scissor is its rectangle).
  if ((c.flags & 0x1000u) != 0u) i = uvec2(c.origin_x, c.origin_y);
  // k_8_8_8_8: R | G << 8 | B << 16 | A << 24; our 16-bit target holds each 8-bit value exactly (v x 257).
  // Bit 1: the frame has a colour surface (no colour draw yet: depth only, ours not used).
  out_color = (c.flags & 2u) != 0u ? unpackUnorm4x8(edram[i.x]) : vec4(0.0);
  gl_FragDepth = DepthFromEdram(edram[i.y] >> 8, (c.flags & 1u) != 0u);
}
)";

// The write-back: our picture into the EDRAM, packed as the emulated pixel shaders pack it (spirv_translator_rb.cpp:
// k_8_8_8_8 as clamp x 255 + 0.5 truncated; depth 20e4 by PreClampedDepthTo20e4 rounding to the nearest even, unorm24
// rounded to the nearest even; the stencil byte kept). Pixels we did not draw since the load go back unchanged.
constexpr char kWriteBackComputeShader[] =
    "#version 450\n"
    "layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;\n"
    "layout(set = 0, binding = 0, std430) buffer EdramBuffer { uint edram[]; };\n"
    "layout(set = 0, binding = 1) uniform sampler2D native_color;\n"
    "layout(set = 0, binding = 2) uniform sampler2D native_depth;\n" NATIVE_EDRAM_PLACE_GLSL R"(
uint DepthToEdram(float z, bool is_float) {
  z = clamp(z, 0.0, 1.0);
  if (!is_float) {
    precise float scaled = z * 16777215.0;
    return uint(roundEven(scaled));
  }
  uint u = floatBitsToUint(z);
  uint b = u < 0x38800000u ? ((u & 0x7FFFFFu) | 0x800000u) >> min(113u - (u >> 23), 24u) : u - (112u << 23);
  b += 3u + ((b >> 3) & 1u);
  return (b >> 3) & 0xFFFFFFu;
}
void main() {
  // Session 40: the dispatch covers the rectangle our draws changed (origin_x / origin_y), and only the channels
  // they wrote (flags bits 2 colour, 3 depth).
  uvec2 p = gl_GlobalInvocationID.xy + uvec2(c.origin_x, c.origin_y);
  if (p.x >= c.width || p.y >= c.height) return;
  uvec2 i = EdramIndices(p);
  ivec2 size = textureSize(native_color, 0);
  vec2 s = vec2(c.scale_x, c.scale_y);
  if ((c.flags & 6u) == 6u) {
    // Colour: the average of the native pixels this emulated pixel covers (one at the same size, up to 4 x 4).
    ivec2 q0 = min(ivec2(vec2(p) * s), size - 1);
    ivec2 q1 = min(clamp(ivec2(ceil(vec2(p + 1u) * s)), q0 + 1, size), q0 + 4);
    vec4 sum = vec4(0.0);
    for (int y = q0.y; y < q1.y; ++y) {
      for (int x = q0.x; x < q1.x; ++x) sum += texelFetch(native_color, ivec2(x, y), 0);
    }
    vec4 color = clamp(sum / float((q1.x - q0.x) * (q1.y - q0.y)), 0.0, 1.0);
    precise vec4 scaled = color * 255.0;
    precise vec4 offset = scaled + 0.5;
    uvec4 u = uvec4(offset);
    edram[i.x] = u.r | (u.g << 8) | (u.b << 16) | (u.a << 24);
  }
  if ((c.flags & 8u) != 0u) {
    // Depth: the native pixel at the emulated pixel's centre.
    float z = texelFetch(native_depth, min(ivec2((vec2(p) + 0.5) * s), size - 1), 0).r;
    edram[i.y] = (DepthToEdram(z, (c.flags & 1u) != 0u) << 8) | (edram[i.y] & 0xFFu);
  }
}
)";

// Session 42: the post write-back. Only the pixels our draws covered since the segment began (stencil 1) are packed
// into the EDRAM, as the interlock pixel shaders pack k_8_8_8_8 (spirv_translator_rb.cpp: clamp, x 255, + 0.5,
// truncated); our colour target is 32-bit float, so it holds the pixel shader's output as the interlock shader saw it.
// The post targets have the emulated host pixels' size (no gpu_native_scale).
constexpr char kPostWriteBackComputeShader[] =
    "#version 450\n"
    "layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;\n"
    "layout(set = 0, binding = 0, std430) buffer EdramBuffer { uint edram[]; };\n"
    "layout(set = 0, binding = 1) uniform sampler2D native_color;\n"
    "layout(set = 0, binding = 2) uniform usampler2D coverage;\n"
    "layout(set = 0, binding = 3) uniform sampler2D native_depth;\n" NATIVE_EDRAM_PLACE_GLSL R"(
// As the scene's write-back (Session 43: post targets with depth hold exact 20e4 values; the conversion keeps them).
uint DepthToEdram(float z, bool is_float) {
  z = clamp(z, 0.0, 1.0);
  if (!is_float) {
    precise float scaled = z * 16777215.0;
    return uint(roundEven(scaled));
  }
  uint u = floatBitsToUint(z);
  uint b = u < 0x38800000u ? ((u & 0x7FFFFFu) | 0x800000u) >> min(113u - (u >> 23), 24u) : u - (112u << 23);
  b += 3u + ((b >> 3) & 1u);
  return (b >> 3) & 0xFFFFFFu;
}
void main() {
  uvec2 p = gl_GlobalInvocationID.xy + uvec2(c.origin_x, c.origin_y);
  if (p.x >= c.width || p.y >= c.height) return;
  // Stencil bit 0: a draw wrote this pixel's colour; bit 1 (Session 43): its depth.
  uint covered = texelFetch(coverage, ivec2(p), 0).r;
  if (covered == 0u) return;
  uvec2 i = EdramIndices(p);
  // Session 43: the colour format (RB_COLOR_INFO color_format) in flags bits 4-7: 0 k_8_8_8_8, 4 k_16_16, 5
  // k_16_16_16_16.
  uint format = (c.flags >> 4) & 0xFu;
  // Session 43: the target's colour class (bits 8-9): 0 RGBA32F (the shader's output: packed here), 1 RGBA8 / 2
  // RGBA16_SNORM (the packed values themselves: the shader rounded them, gpu_native_post_quantize).
  uint target_class = (c.flags >> 8) & 3u;
  // Session 45 (gpu_native_defer_writeback): flags bit 2 - write the colour (off: the depth alone).
  if ((c.flags & 4u) == 0u) covered &= ~1u;
  if ((covered & 1u) != 0u && format == 0u) {
    vec4 raw = texelFetch(native_color, ivec2(p), 0);
    uvec4 u;
    if (target_class != 0u) {
      u = uvec4(round(raw * 255.0));
    } else {
      // NClamp in the interlock shader: NaN becomes 0.
      vec4 color = mix(clamp(raw, 0.0, 1.0), vec4(0.0), isnan(raw));
      precise vec4 scaled = color * 255.0;
      precise vec4 offset = scaled + 0.5;
      u = uvec4(offset);
    }
    edram[i.x] = u.r | (u.g << 8) | (u.b << 16) | (u.a << 24);
  } else if ((covered & 1u) != 0u && target_class != 0u) {
    uvec4 s = uvec4(ivec4(round(texelFetch(native_color, ivec2(p), 0) * 32767.0)));
    uint d0 = (s.r & 0xFFFFu) | (s.g << 16), d1 = (s.b & 0xFFFFu) | (s.a << 16);
    if (format == 4u) {
      edram[i.x] = d0;
    } else {
      uint half_w = c.tile_w >> 1;
      uint o = (c.color_base_dwords + (p.y / c.tile_h) * c.pitch_tile_dwords * 2u + (p.x / half_w) * (c.tile_w * c.tile_h) +
                (p.y % c.tile_h) * c.tile_w + (p.x % half_w) * 2u) % c.wrap_dwords;
      edram[o] = d0;
      edram[(o + 1u) % c.wrap_dwords] = d1;
    }
  } else if ((covered & 1u) != 0u) {
    // As FSI_ClampAndPackColor's k_16_16 / k_16_16_16_16: NaN to 0, clamped to -32...32, x 32767 / 32, rounded away
    // from zero (+-0.5, then truncated), 16 bits each: R | G << 16, B | A << 16.
    vec4 raw = texelFetch(native_color, ivec2(p), 0);
    vec4 color = clamp(mix(raw, vec4(0.0), isnan(raw)), -32.0, 32.0);
    precise vec4 scaled = color * (32767.0 / 32.0);
    precise vec4 offset = scaled + mix(vec4(0.5), vec4(-0.5), lessThan(scaled, vec4(0.0)));
    uvec4 s = uvec4(ivec4(offset));
    uint d0 = (s.r & 0xFFFFu) | (s.g << 16), d1 = (s.b & 0xFFFFu) | (s.a << 16);
    if (format == 4u) {
      edram[i.x] = d0;
    } else {
      // 64bpp: 40 x 16 pixels per tile, two dwords each, rows of tiles twice the 32bpp pitch (FSI_LoadEdramOffsets).
      uint half_w = c.tile_w >> 1;
      uint o = (c.color_base_dwords + (p.y / c.tile_h) * c.pitch_tile_dwords * 2u + (p.x / half_w) * (c.tile_w * c.tile_h) +
                (p.y % c.tile_h) * c.tile_w + (p.x % half_w) * 2u) % c.wrap_dwords;
      edram[o] = d0;
      edram[(o + 1u) % c.wrap_dwords] = d1;
    }
  }
  if ((covered & 2u) != 0u && (c.flags & 8u) != 0u) {
    float z = texelFetch(native_depth, ivec2(p), 0).r;
    edram[i.y] = (DepthToEdram(z, (c.flags & 1u) != 0u) << 8) | (edram[i.y] & 0xFFu);
  }
}
)";

// Session 45 (gpu_native_resolve): the stock resolve_fast_32bpp_1x2xmsaa compute shader (read from its compiled form:
// the Direct3D 12 bytecode header's disassembly) for 1x surfaces, with its EDRAM read replaced: the pixels the post
// segment drew (coverage bit 0) come from the post target, packed as the write-back packs them, the others from the
// EDRAM as before. Then the same red / blue swap, 2D tiled destination address and endian swap. Each invocation
// copies 8 pixels of one row; 8 x 8 invocations a group, the stock dispatch size. (The stock constants first, then
// where the resolve's first pixel is in the target.)
// (Compiled with "#version 450" first; Session 45 N3a: NATIVE_TEXWRITE also stores each pixel into the texture the
// resolve feeds - its R32_UINT storage view, the dword a 32bpb load would put there: the value before the destination
// endian swap, as the texture's own endianness is the destination's.)
// (Session 49, gpu_native_truth_skip_writeback: target_class bit 10 = the texture only, no destination write.)
constexpr char kNativeResolveComputeShader[] = R"(
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) buffer EdramBuffer { uint edram[]; };
layout(set = 0, binding = 1) uniform sampler2D native_color;
layout(set = 0, binding = 2) uniform usampler2D coverage;
layout(set = 1, binding = 0, std430) writeonly buffer DestBuffer { uvec4 dest[]; };
#ifdef NATIVE_TEXWRITE
layout(set = 2, binding = 0, r32ui) uniform writeonly uimage2D texture_out;
#endif
layout(push_constant) uniform Constants {
  uint edram_info;
  uint coordinate_info;
  uint dest_info;
  uint dest_coordinate_info;
  uint dest_base;
  uint src_x0;
  uint src_y0;
  uint target_w;
  uint target_h;
  uint target_class;
  uint texture_w;
  uint texture_h;
} c;
void main() {
  uint tx = gl_GlobalInvocationID.x, ty = gl_GlobalInvocationID.y;
  if (tx >= ((c.coordinate_info >> 5) & 0x7FFu)) return;
  // The EDRAM: 8 dwords of one tile row (1x: samples are pixels; the colour halves are not swapped).
  uint ex = ((c.coordinate_info & 0xFu) << 3) + (tx << 3);
  uint ey = (((c.coordinate_info >> 4) & 1u) << 3) + ty;
  uint tile = (ey / 16u) * (c.edram_info & 1023u) + (ex / 80u) + ((c.edram_info >> 13) & 0x7FFu);
  uint first = (tile * 1280u + (ey % 16u) * 80u + (ex % 80u)) % 0x280000u;
  uint v[8];
  for (uint i = 0u; i < 8u; ++i) {
    uvec2 q = uvec2(c.src_x0 + (tx << 3) + i, c.src_y0 + ty);
    ivec2 p = ivec2(q);
    // Session 46 (truth frames, target_class bit 9): every pixel from the target (it holds the whole surface).
    if (q.x < c.target_w && q.y < c.target_h &&
        ((c.target_class & 0x200u) != 0u || (texelFetch(coverage, p, 0).r & 1u) != 0u)) {
      vec4 raw = texelFetch(native_color, p, 0);
      uvec4 u;
      if ((c.target_class & 0xFFu) != 0u) {
        u = uvec4(round(raw * 255.0));
      } else {
        vec4 color = mix(clamp(raw, 0.0, 1.0), vec4(0.0), isnan(raw));
        precise vec4 scaled = color * 255.0;
        precise vec4 offset = scaled + 0.5;
        u = uvec4(offset);
      }
      if ((c.target_class & 0x800u) != 0u) u.a = 255u;  // (Session 90 Q diagnosis: alpha forced)
      v[i] = u.r | (u.g << 8) | (u.b << 16) | (u.a << 24);
      // Session 45 (gpu_native_fuse_writeback, target_class bit 8): the write-back of this pixel, done here.
      if ((c.target_class & 0x100u) != 0u) edram[(first + i) % 0x280000u] = v[i];
    } else {
      v[i] = edram[(first + i) % 0x280000u];
    }
  }
  // RB_COPY_DEST_INFO copy_dest_swap: red and blue swapped, by the EDRAM format.
  if ((c.dest_info & 0x01000000u) != 0u) {
    uint format = (c.edram_info >> 24) & 0xFu;
    for (uint i = 0u; i < 8u; ++i) {
      if (format <= 1u) {
        v[i] = (v[i] & 0xFF00FF00u) | ((v[i] & 0xFFu) << 16) | ((v[i] >> 16) & 0xFFu);
      } else if (format == 2u || format == 3u || format == 10u || format == 12u) {
        v[i] = (v[i] & 0xC00FFC00u) | ((v[i] & 0x3FFu) << 20) | ((v[i] >> 20) & 0x3FFu);
      }
    }
  }
  // The 2D tiled destination (32bpp) of the first of the 8 pixels; the next 4 are 32 bytes on.
  uint x = (((c.dest_coordinate_info >> 20) & 0xFu) << 3) + (tx << 3);
  uint y = (((c.dest_coordinate_info >> 24) & 0xFu) << 3) + ty;
  uint pitch = (c.dest_coordinate_info & 1023u) << 5;
  uint macro_tile = (y >> 5) * (pitch >> 5) + (x >> 5);
  uint r0z = (y << 5) & 448u;
  uint r0w = (y << 7) & 2048u;
  uint r1y = (y & 1u) << 4;
  uint r1x = ((r0z & 0x1FFu) | (macro_tile << 9)) + r1y;
  uint a = (((r0z << 3) & 0xFFFu) | (macro_tile << 12)) + r1y * 8u;
  uint b = (((r0z << 2) & 0x7FFu) | (macro_tile << 11)) + r1y * 4u;
  a = (a & ~0xFFFu) | (r0w & 0xFFFu);
  a += b & 1792u;
  a += (((x >> 3) + ((y >> 2) & 2u)) & 3u) << 6;
  uint index = (((a & ~63u) | (r1x & 63u)) + c.dest_base) >> 4;
#ifdef NATIVE_TEXWRITE
  for (uint i = 0u; i < 8u; ++i) {
    if (x + i < c.texture_w && y < c.texture_h) imageStore(texture_out, ivec2(int(x + i), int(y)), uvec4(v[i]));
  }
#endif
  // Endian: 1 8-in-16, 2 8-in-32, 3 16-in-32.
  uint endian = c.dest_info & 7u;
  bool swap8 = endian == 1u || endian == 2u, swap16 = endian == 2u || endian == 3u;
  for (uint i = 0u; i < 8u; ++i) {
    if (swap8) v[i] = ((v[i] << 8) & 0xFF00FF00u) | ((v[i] >> 8) & 0x00FF00FFu);
    if (swap16) v[i] = (v[i] >> 16) | (v[i] << 16);
  }
  if ((c.target_class & 0x400u) != 0u) return;  // (Session 49: the texture only)
  dest[index] = uvec4(v[0], v[1], v[2], v[3]);
  dest[index + 2u] = uvec4(v[4], v[5], v[6], v[7]);
}
)";

// Session 52 (gpu_native_truth_direct_resolve_scaled): the stock resolve_fast_32bpp_1x2xmsaa_scaled compute shader (read
// from its compiled form: the Direct3D 12 bytecode header's disassembly) for 1x-sample colour surfaces a truth image
// holds, with its EDRAM read replaced by the image's pixels (host pixels: the image is at the resolution scale). Each
// invocation copies 8 host pixels of one host row (width_div_8 x scale_x invocations a row), packs them as the 1x
// direct copy does, the same red / blue swap; the destination is the scaled resolve buffer: the 16 bytes of 4 host
// pixels at host (x, y) go to the guest 2D tiled offset of guest texel ((x / 4) / scale_x * 4, y / scale_y) times
// scale_x * scale_y, plus 16 x (((x / 4) % scale_x) * scale_y + y % scale_y); then the 32-bit endian swap.
// (The same push constants as the 1x shader: dest_base unused - the binding starts at the destination.)
constexpr char kNativeResolveScaledComputeShader[] = R"(
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) buffer EdramBuffer { uint edram[]; };
layout(set = 0, binding = 1) uniform sampler2D native_color;
layout(set = 0, binding = 2) uniform usampler2D coverage;
layout(set = 1, binding = 0, std430) writeonly buffer DestBuffer { uvec4 dest[]; };
#ifdef NATIVE_TEXWRITE
layout(set = 2, binding = 0, r32ui) uniform writeonly uimage2D texture_out;
#endif
layout(push_constant) uniform Constants {
  uint edram_info;
  uint coordinate_info;
  uint dest_info;
  uint dest_coordinate_info;
  uint dest_base;
  uint src_x0;
  uint src_y0;
  uint target_w;
  uint target_h;
  uint target_class;
  uint texture_w;
  uint texture_h;
} c;
int XeTextureTiledOffset2D(ivec2 p, uint pitch_aligned, uint bpb_log2) {
  int macro = ((p.x >> 5) + (p.y >> 5) * int(pitch_aligned >> 5u)) << (bpb_log2 + 7u);
  int micro = ((p.x & 7) + ((p.y & 14) << 2)) << bpb_log2;
  int offset = macro + ((micro & -16) << 1) + (micro & 15) + ((p.y & 1) << 4);
  return ((offset & -512) << 3) + ((p.y & 16) << 7) + ((offset & 448) << 2) +
         (((((p.y & 8) >> 2) + (p.x >> 3)) & 3) << 6) + (offset & 63);
}
uint DestIndex(uint hx, uint hy, uint sx, uint sy, uint pitch) {
  uint gx = ((hx >> 2u) / sx) << 2u, gy = hy / sy;
  uint sub = ((hx >> 2u) % sx) * sy + (hy % sy);
  return (uint(XeTextureTiledOffset2D(ivec2(int(gx), int(gy)), pitch, 2u)) * (sx * sy) + (sub << 4u)) >> 4u;
}
void main() {
  uint tx = gl_GlobalInvocationID.x, ty = gl_GlobalInvocationID.y;
  uint sx = (c.coordinate_info >> 16) & 7u, sy = (c.coordinate_info >> 19) & 7u;
  if (tx >= ((c.coordinate_info >> 5) & 0x7FFu) * sx) return;
  // The half-pixel offset gap (edram_info bit 29: half the scale, host pixels) filled from the first covered column /
  // row, as the stock scaled copy reads it: the row max(y, gap y), in the first invocation of a row x max(x, gap x).
  uint gap_x = 0u, gap_y = 0u;
  if ((c.edram_info & 0x20000000u) != 0u) {
    gap_x = (c.coordinate_info >> 17) & 3u;
    gap_y = (c.coordinate_info >> 20) & 3u;
  }
  uint v[8];
  for (uint i = 0u; i < 8u; ++i) {
    uint rx = (tx << 3) + i;
    if (tx == 0u) rx = max(rx, gap_x);
    uvec2 q = uvec2(c.src_x0 + rx, c.src_y0 + max(ty, gap_y));
    v[i] = 0u;
    if (q.x < c.target_w && q.y < c.target_h) {
      vec4 raw = texelFetch(native_color, ivec2(q), 0);
      uvec4 u;
      if ((c.target_class & 0xFFu) != 0u) {
        u = uvec4(round(raw * 255.0));
      } else {
        vec4 color = mix(clamp(raw, 0.0, 1.0), vec4(0.0), isnan(raw));
        precise vec4 scaled = color * 255.0;
        precise vec4 offset = scaled + 0.5;
        u = uvec4(offset);
      }
      if ((c.target_class & 0x800u) != 0u) u.a = 255u;  // (Session 90 Q diagnosis: alpha forced)
      v[i] = u.r | (u.g << 8) | (u.b << 16) | (u.a << 24);
    }
  }
  // RB_COPY_DEST_INFO copy_dest_swap: red and blue swapped, by the EDRAM format.
  if ((c.dest_info & 0x01000000u) != 0u) {
    uint format = (c.edram_info >> 24) & 0xFu;
    for (uint i = 0u; i < 8u; ++i) {
      if (format <= 1u) {
        v[i] = (v[i] & 0xFF00FF00u) | ((v[i] & 0xFFu) << 16) | ((v[i] >> 16) & 0xFFu);
      } else if (format == 2u || format == 3u || format == 10u || format == 12u) {
        v[i] = (v[i] & 0xC00FFC00u) | ((v[i] & 0x3FFu) << 20) | ((v[i] >> 20) & 0x3FFu);
      }
    }
  }
  uint hx = (((c.dest_coordinate_info >> 20) & 0xFu) << 3) * sx + (tx << 3);
  uint hy = (((c.dest_coordinate_info >> 24) & 0xFu) << 3) * sy + ty;
#ifdef NATIVE_TEXWRITE
  // Session 52 (gpu_native_truth_direct_resolve_scaled_texture): the scaled texture's texels (host), the values the
  // scaled texture load gives - the dwords before the destination endian swap, as the 1x shader's texture write.
  for (uint i = 0u; i < 8u; ++i) {
    if (hx + i < c.texture_w && hy < c.texture_h) imageStore(texture_out, ivec2(int(hx + i), int(hy)), uvec4(v[i]));
  }
#endif
  // Endian: 1 8-in-16, 2 8-in-32, 3 16-in-32.
  uint endian = c.dest_info & 7u;
  bool swap8 = endian == 1u || endian == 2u, swap16 = endian == 2u || endian == 3u;
  for (uint i = 0u; i < 8u; ++i) {
    if (swap8) v[i] = ((v[i] << 8) & 0xFF00FF00u) | ((v[i] >> 8) & 0x00FF00FFu);
    if (swap16) v[i] = (v[i] >> 16) | (v[i] << 16);
  }
  uint pitch = (c.dest_coordinate_info & 1023u) << 5;
  if ((c.target_class & 0x400u) != 0u) return;  // Session 56: the texture only (no scaled buffer write)
  dest[DestIndex(hx, hy, sx, sy, pitch)] = uvec4(v[0], v[1], v[2], v[3]);
  dest[DestIndex(hx + 4u, hy, sx, sy, pitch)] = uvec4(v[4], v[5], v[6], v[7]);
}
)";

// Session 49 (gpu_native_truth_skip_writeback): a texture-only resolve's output into guest memory (the GPU copy) from the
// texture it filled - the inverse of the direct texture load (texture_load_scaled_direct.cs.glsl): each texel's raw
// bits through the texture's storage view (R32_UINT: the dword; R16G16B16A16_UINT: two dwords as four 16-bit halves,
// low half first), the same 32-bit endian swap (its own inverse), at the 2D tiled offset of its destination texel
// (XeTextureTiledOffset2D, as the load reads it). One texel an invocation, 8 x 8 a group.
constexpr char kNativeMaterializeShader[] = R"(
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 1, binding = 0, std430) writeonly buffer DestBuffer { uint dest[]; };
#ifdef BPB64
layout(set = 2, binding = 0, rgba16ui) uniform readonly uimage2D texture_in;
const uint kBpbLog2 = 3u;
#else
layout(set = 2, binding = 0, r32ui) uniform readonly uimage2D texture_in;
const uint kBpbLog2 = 2u;
#endif
layout(push_constant) uniform Constants {
  uint x0, y0, w, h, pitch, dest_base, endian, tex_x0, tex_y0;
} c;
int XeTextureTiledOffset2D(ivec2 p, uint pitch_aligned, uint bpb_log2) {
  int macro = ((p.x >> 5) + (p.y >> 5) * int(pitch_aligned >> 5u)) << (bpb_log2 + 7u);
  int micro = ((p.x & 7) + ((p.y & 14) << 2)) << bpb_log2;
  int offset = macro + ((micro & -16) << 1) + (micro & 15) + ((p.y & 1) << 4);
  return ((offset & -512) << 3) + ((p.y & 16) << 7) + ((offset & 448) << 2) +
         (((((p.y & 8) >> 2) + (p.x >> 3)) & 3) << 6) + (offset & 63);
}
uint XeEndianSwap32(uint value, uint endian) {
  if (endian == 1u || endian == 2u) value = ((value & 16711935u) << 8u) | ((value & 4278255360u) >> 8u);
  if (endian == 2u || endian == 3u) value = (value << 16u) | (value >> 16u);
  return value;
}
void main() {
  uvec2 p = gl_GlobalInvocationID.xy;
  if (p.x >= c.w || p.y >= c.h) return;
  ivec2 d = ivec2(int(c.x0 + p.x), int(c.y0 + p.y));
  ivec2 t = ivec2(int(c.tex_x0) + d.x, int(c.tex_y0) + d.y);
  uint index = (uint(XeTextureTiledOffset2D(d, c.pitch, kBpbLog2)) + c.dest_base) >> 2u;
  uvec4 v = imageLoad(texture_in, t);
#ifdef BPB64
  dest[index] = XeEndianSwap32((v.r & 65535u) | (v.g << 16u), c.endian);
  dest[index + 1u] = XeEndianSwap32((v.b & 65535u) | (v.a << 16u), c.endian);
#else
  dest[index] = XeEndianSwap32(v.r, c.endian);
#endif
}
)";

// Session 56 (gpu_native_truth_skip_writeback_scaled): a scaled texture-only resolve's output into the scaled resolve
// buffer from the scaled texture it filled - what kNativeResolveScaledComputeShader would have written there: the
// texture's host texels (its r32ui storage view: the dwords before the destination endian swap), the endian swap, 16
// bytes (4 host texels) at DestIndex. One group of 4 texels an invocation.
constexpr char kNativeMaterializeScaledShader[] = R"(
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 1, binding = 0, std430) writeonly buffer DestBuffer { uvec4 dest[]; };
layout(set = 2, binding = 0, r32ui) uniform readonly uimage2D texture_in;
layout(push_constant) uniform Constants {
  uint x0, y0, x1, y1, pitch, endian, sx, sy;
} c;
int XeTextureTiledOffset2D(ivec2 p, uint pitch_aligned, uint bpb_log2) {
  int macro = ((p.x >> 5) + (p.y >> 5) * int(pitch_aligned >> 5u)) << (bpb_log2 + 7u);
  int micro = ((p.x & 7) + ((p.y & 14) << 2)) << bpb_log2;
  int offset = macro + ((micro & -16) << 1) + (micro & 15) + ((p.y & 1) << 4);
  return ((offset & -512) << 3) + ((p.y & 16) << 7) + ((offset & 448) << 2) +
         (((((p.y & 8) >> 2) + (p.x >> 3)) & 3) << 6) + (offset & 63);
}
uint DestIndex(uint hx, uint hy, uint sx, uint sy, uint pitch) {
  uint gx = ((hx >> 2u) / sx) << 2u, gy = hy / sy;
  uint sub = ((hx >> 2u) % sx) * sy + (hy % sy);
  return (uint(XeTextureTiledOffset2D(ivec2(int(gx), int(gy)), pitch, 2u)) * (sx * sy) + (sub << 4u)) >> 4u;
}
void main() {
  uint hx = c.x0 + (gl_GlobalInvocationID.x << 2u), hy = c.y0 + gl_GlobalInvocationID.y;
  if (hx >= c.x1 || hy >= c.y1) return;
  bool swap8 = c.endian == 1u || c.endian == 2u, swap16 = c.endian == 2u || c.endian == 3u;
  uint v[4];
  for (uint i = 0u; i < 4u; ++i) {
    v[i] = imageLoad(texture_in, ivec2(int(hx + i), int(hy))).r;
    if (swap8) v[i] = ((v[i] << 8) & 0xFF00FF00u) | ((v[i] >> 8) & 0x00FF00FFu);
    if (swap16) v[i] = (v[i] >> 16) | (v[i] << 16);
  }
  dest[DestIndex(hx, hy, c.sx, c.sy, c.pitch)] = uvec4(v[0], v[1], v[2], v[3]);
}
)";

// Linear, with more precision than the EDRAM's 8 bits (gamma targets keep ~10 bits of linear in their 8-bit curve).
constexpr VkFormat kColorFormat = VK_FORMAT_R16G16B16A16_UNORM;
// Session 42: post targets (exact: the write-back packs the shader's float output itself).
constexpr VkFormat kPostTargetFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
// Session 43 (gpu_native_post_quantize): the post target colour format per class (NativeRenderer::PostTargetClass).
constexpr VkFormat kPostTargetFormats[3] = {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM,
                                            VK_FORMAT_R16G16B16A16_SNORM};
// The emulated scene copy (the EDRAM's 8-bit values as they are).
constexpr VkFormat kSceneCopyFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

// The Xenos piecewise-linear gamma curve, as xenos.cpp LinearToPWLGamma (and the GLSL copies above).
float LinearToPWLGamma(float linear) {
  linear = std::clamp(linear, 0.0f, 1.0f);
  float scale, offset;
  if (linear >= 128.0f / 1023.0f) {
    scale = linear >= 512.0f / 1023.0f ? 1023.0f / 8.0f : 1023.0f / 4.0f;
    offset = linear >= 512.0f / 1023.0f ? 128.0f / 255.0f : 64.0f / 255.0f;
  } else {
    scale = linear >= 64.0f / 1023.0f ? 1023.0f / 2.0f : 1023.0f;
    offset = linear >= 64.0f / 1023.0f ? 32.0f / 255.0f : 0.0f;
  }
  return std::trunc(linear * scale) * (1.0f / 255.0f) + offset;
}

// wwe2k14 patch (Session 93, idea 7, log only): how many of the native renderer's pipelines would collapse if the cull mode, front
// face, depth test / write / compare and stencil state were dynamic (VK_EXT_extended_dynamic_state): the key with those bits
// cleared, counted per distinct key. File statics (the pipelines are made on the command thread, the pool's workers and by the
// crowd); the prewarm = a stored-pipelines job.
namespace {
struct EdsNativeCount {
  std::mutex lock;
  std::unordered_set<std::string> full[2], collapsed[2];  // [0] drawn, [1] prewarm
  uint64_t pipelines[2] = {}, would_collapse[2] = {};
  uint64_t window_first_frame = UINT64_MAX;
};
EdsNativeCount g_eds_native;
thread_local bool g_eds_native_prewarm = false;
}  // namespace

uint64_t Mix64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xFF51AFD7ED558CCDull;
  x ^= x >> 33;
  x *= 0xC4CEB9FE1A85EC53ull;
  x ^= x >> 33;
  return x;
}

uint32_t RasterKey(const DrawRaster& raster) {
  uint32_t key = uint32_t(raster.topology) & 0xF;
  key |= uint32_t(raster.primitive_restart) << 4;
  key |= (uint32_t(raster.cull_mode) & 3) << 5;
  key |= uint32_t(raster.front_face_clockwise) << 7;
  key |= uint32_t(raster.depth_test) << 8;
  key |= uint32_t(raster.depth_write) << 9;
  key |= (uint32_t(raster.depth_compare) & 7) << 10;
  key |= uint32_t(raster.depth_bias_constant != 0.0f || raster.depth_bias_slope != 0.0f) << 13;
  // wwe2k14 patch (Session 69, the testers' red block): clipping off on the guest (PA_CL_CLIP_CNTL clip_disable) -
  // the pipeline clamps depth instead of clipping, as the emulated pipeline does (pipeline_cache.cpp
  // depth_clamp_enable). Without it a colour-only clear whose z lands a hair below 0 (Undertaker's damage-map strip
  // clear, z -1e-7) was depth-clipped whole: every fragment dropped (RenderDoc pixel history, Session 69).
  key |= uint32_t(raster.clip_disable) << 14;
  return key;
}

// xenos::BlendFactor -> VkBlendFactor (the Xenos values skip 2 and 3).
VkBlendFactor BlendFactor(xenos::BlendFactor factor) {
  switch (factor) {
    case xenos::BlendFactor::kZero: return VK_BLEND_FACTOR_ZERO;
    case xenos::BlendFactor::kOne: return VK_BLEND_FACTOR_ONE;
    case xenos::BlendFactor::kSrcColor: return VK_BLEND_FACTOR_SRC_COLOR;
    case xenos::BlendFactor::kOneMinusSrcColor: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case xenos::BlendFactor::kSrcAlpha: return VK_BLEND_FACTOR_SRC_ALPHA;
    case xenos::BlendFactor::kOneMinusSrcAlpha: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case xenos::BlendFactor::kDstColor: return VK_BLEND_FACTOR_DST_COLOR;
    case xenos::BlendFactor::kOneMinusDstColor: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case xenos::BlendFactor::kDstAlpha: return VK_BLEND_FACTOR_DST_ALPHA;
    case xenos::BlendFactor::kOneMinusDstAlpha: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case xenos::BlendFactor::kConstantColor: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case xenos::BlendFactor::kOneMinusConstantColor: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case xenos::BlendFactor::kConstantAlpha: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case xenos::BlendFactor::kOneMinusConstantAlpha: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    case xenos::BlendFactor::kSrcAlphaSaturate: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    default: return VK_BLEND_FACTOR_ONE;
  }
}

VkBlendOp BlendOp(xenos::BlendOp op) {
  switch (op) {
    case xenos::BlendOp::kSubtract: return VK_BLEND_OP_SUBTRACT;
    case xenos::BlendOp::kMin: return VK_BLEND_OP_MIN;
    case xenos::BlendOp::kMax: return VK_BLEND_OP_MAX;
    case xenos::BlendOp::kRevSubtract: return VK_BLEND_OP_REVERSE_SUBTRACT;
    default: return VK_BLEND_OP_ADD;
  }
}

}  // namespace

size_t NativeRenderer::PipelineKeyHasher::operator()(const PipelineKey& key) const {
  return size_t(Mix64(uint64_t(reinterpret_cast<uintptr_t>(key.vertex_shader)) ^
                      (uint64_t(reinterpret_cast<uintptr_t>(key.pixel_shader)) * 17) ^
                      (uint64_t(reinterpret_cast<uintptr_t>(key.layout)) * 31) ^ (uint64_t(key.state) << 40) ^
                      (uint64_t(key.blend_control) << 8) ^ key.tint_hash ^
                      (uint64_t(reinterpret_cast<uintptr_t>(key.geometry_shader)) * 61) ^ (uint64_t(key.variant) << 62) ^ (uint64_t(key.vi) * 0x9E3779B97F4A7C15ull)));
}

VertexInputState& NativeRenderer::VertexInput() {
  if (!vertex_input_) {
    vertex_input_ = new VertexInputState();
    // The real limits of this device, in the log: the design stays within the Baseline's 16 attributes (vertex_input_layout.h), this says what the phone offers.
    const ui::vulkan::VulkanDevice* device = command_processor_.GetVulkanDevice();
    VkPhysicalDeviceProperties p = {};
    device->vulkan_instance()->functions().vkGetPhysicalDeviceProperties(device->physical_device(), &p);
    REXGPU_INFO("[vinput] device limits: maxVertexInputAttributes {}, maxVertexInputBindings {}, maxVertexInputAttributeOffset {}, maxVertexInputBindingStride {}", p.limits.maxVertexInputAttributes,
                p.limits.maxVertexInputBindings, p.limits.maxVertexInputAttributeOffset, p.limits.maxVertexInputBindingStride);
  }
  return *vertex_input_;
}

// Session 95 (out\s95\android_structural.md sections 3.2 and 3.4): the one rule. Read by the typed draw and by the store, which must agree: a record derives the attribute key exactly when the draw
// could have taken the form.
bool NativeRenderer::VertexInputFormOf(VulkanShader& vertex_shader, uint64_t vertex_modification, uint32_t vertex_ext, ViForm* form) {
  const int32_t level = shader_swap::VertexInputLevel();
  form->refusal = kViNoForm;
  if (!level) return false;
  const SpirvShaderTranslator::Modification modification(vertex_modification);
  if (modification.vertex.host_vertex_shader_type != Shader::HostVertexShaderType::kVertex || modification.vertex.xfb_position || modification.vertex.instanced_float_constants_stride ||
      shader_swap::InstanceWorldLatched() || SpirvShaderTranslator::DecoupleViewConstantBlocks()) {
    return false;
  }
  VulkanPipelineCache& cache = command_processor_.NativePipelineCache();
  if (vertex_ext & SpirvShaderTranslator::kNativeExtVsSwap) {  // our row: its _vi module, one binding
    uint32_t fetch_constant = 0, stride = 0;
    if (!(vertex_ext & SpirvShaderTranslator::kNativeExtOwnBindings) || !shader_swap::VertexMirrorable(vertex_shader.ucode_data_hash(), &fetch_constant, &stride) ||
        !cache.UsesReplacementBindings(&vertex_shader, vertex_ext | SpirvShaderTranslator::kNativeExtVsInput) || !shader_swap::VertexMirrorFits(vertex_shader, vertex_modification) ||
        !stride || stride >= kViInstanceRate) {
      return false;
    }
    form->vi = stride;
    form->binding_count = 1;
    form->fetch_constant[0] = uint8_t(fetch_constant);
    form->translated = false;
  } else {  // translated: the census's form, up to 4 bindings
    if (level < 2) {
      form->refusal = kViTranslatedOff;
      return false;
    }
    if (!vertex_shader.is_ucode_analyzed()) cache.AnalyzeShaderUcode(vertex_shader);
    const auto& input = vertex_shader.VertexInputForm();
    if (!input.eligible || input.binding_count > kViMaxBindings) return false;  // (eligible with no binding: the shader fetches no vertex, vi 0)
    form->vi = 0;
    for (uint32_t i = 0; i < input.binding_count; ++i) {
      if (!input.stride[i] || input.stride[i] >= kViInstanceRate) return false;
      form->vi |= uint32_t(input.stride[i]) << (8 * i);
      form->fetch_constant[i] = uint8_t(input.fetch_constant[i]);
    }
    form->binding_count = input.binding_count;
    form->translated = true;
  }
  ViAttr attrs[kViMaxAttributes];
  if (form->vi && !ViAttributes(form->vi, attrs)) {
    form->refusal = kViAttributes;
    return false;
  }
  return true;
}

IndexPool& NativeRenderer::Mirror() {
  if (!vertex_mirror_) vertex_mirror_ = new IndexPool(command_processor_);
  return *vertex_mirror_;
}

VertexMirror& NativeRenderer::SkinMirror() {
  if (!skin_mirror_) skin_mirror_ = new VertexMirror(command_processor_);
  return *skin_mirror_;
}

SkinOnce& NativeRenderer::Skin() {
  if (!skin_once_) skin_once_ = new SkinOnce(command_processor_);
  return *skin_once_;
}
bool NativeRenderer::SkinPending() const { return skin_once_ && skin_once_->Pending(); }
void NativeRenderer::SkinRecord(DeferredCommandBuffer& hoist, uint64_t submission) { skin_once_->Record(hoist, submission); }

NativeRenderer::NativeRenderer(VulkanCommandProcessor& command_processor) : command_processor_(command_processor) {
  // (Session 46: the EDRAM owns every tile - Session 47: no image owns any, truth_owned_any_ starts empty.)
}

NativeRenderer::~NativeRenderer() {
  StopPipelineWorker();  // Session 47 (also without a device)
  ShutdownStorage();     // Session 61
  DestroyObjects();
  delete skin_once_;  // Session 95
  skin_once_ = nullptr;
  delete skin_mirror_;  // Session 95 (after the skin job table above, which names its buffer)
  skin_mirror_ = nullptr;
  delete vertex_mirror_;  // Session 95 (item D)
  vertex_mirror_ = nullptr;
  delete vertex_input_;  // Session 95
  vertex_input_ = nullptr;
}

// ---------------------------------------------------------------------------------------------------------------
// Objects
// ---------------------------------------------------------------------------------------------------------------

bool NativeRenderer::EnsureObjects() {
  if (render_pass_clear_ != VK_NULL_HANDLE) return true;
  if (objects_failed_) return false;
  objects_failed_ = true;  // until everything below exists
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  // Two render passes that differ only in their load operations: the frame's first native pass clears.
  for (int clear = 0; clear < 2; ++clear) {
    VkAttachmentDescription attachments[2] = {};
    attachments[0].format = kColorFormat;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = clear ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[1] = attachments[0];
    attachments[1].format = kDepthFormat;
    attachments[1].initialLayout =
        clear ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference color_ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depth_ref = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;
    subpass.pDepthStencilAttachment = &depth_ref;
    // Our own earlier passes (attachment writes), the swap's read of the colour and the parity check's read of the
    // depth come before; the same in the other direction.
    const VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    const VkAccessFlags accesses = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    VkSubpassDependency dependencies[2] = {};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask =
        stages | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    dependencies[0].dstStageMask = stages;
    dependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dstAccessMask = accesses;
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = stages;
    dependencies[1].dstStageMask = stages | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = accesses | VK_ACCESS_SHADER_READ_BIT;
    VkRenderPassCreateInfo info = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    info.attachmentCount = 2;
    info.pAttachments = attachments;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = 2;
    info.pDependencies = dependencies;
    if (dfn.vkCreateRenderPass(device, &info, nullptr, clear ? &render_pass_clear_ : &render_pass_load_) !=
        VK_SUCCESS) {
      REXGPU_ERROR("[native] render pass creation failed");
      return false;
    }
  }

  auto compile = [&](VkShaderStageFlagBits stage, const char* source, VkShaderModule& out) {
    std::vector<uint32_t> spirv;
    std::string error;
    if (!command_processor_.CompileGlslToSpirv(stage, source, spirv, error)) {
      REXGPU_ERROR("[native] shader compilation failed: {}", error);
      return false;
    }
    out = ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
    return out != VK_NULL_HANDLE;
  };
  if (!compile(VK_SHADER_STAGE_FRAGMENT_BIT, kClayFragmentShader, clay_fragment_shader_) ||
      !compile(VK_SHADER_STAGE_COMPUTE_BIT, kComposeComputeShader, compose_shader_)) {
    return false;
  }

  // Timestamps (a frame slot's range is reset at the end of the frame before it).
  {
    const ui::vulkan::VulkanInstance::Functions& ifn = vulkan_device->vulkan_instance()->functions();
    VkPhysicalDeviceProperties properties;
    ifn.vkGetPhysicalDeviceProperties(vulkan_device->physical_device(), &properties);
    timestamp_period_ns_ = properties.limits.timestampPeriod;
    VkQueryPoolCreateInfo query_info = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_info.queryCount = kFrameSlots * kQueriesPerFrame;
    if (!(timestamp_period_ns_ > 0.0f) ||
        dfn.vkCreateQueryPool(device, &query_info, nullptr, &query_pool_) != VK_SUCCESS) {
      query_pool_ = VK_NULL_HANDLE;
      REXGPU_WARN("[native] no GPU timestamps: the cost numbers are left out");
    }
  }

  // Descriptors for the swap's composition and the parity check, one set per frame slot.
  {
    // Per frame slot: the composition (2 samplers, 1 storage image), the depth parity check (1 sampler, 2 storage
    // buffers), the colour parity check (1 sampler, 2 storage buffers, 1 storage image).
    VkDescriptorPoolSize sizes[3] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 * kFrameSlots},
                                     {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * kFrameSlots},
                                     {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * kFrameSlots}};
    VkDescriptorPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 3 * kFrameSlots;
    pool_info.poolSizeCount = 3;
    pool_info.pPoolSizes = sizes;
    if (dfn.vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
      return false;
    }
    VkSamplerCreateInfo sampler_info = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.maxLod = 0.0f;
    if (dfn.vkCreateSampler(device, &sampler_info, nullptr, &sampler_) != VK_SUCCESS) return false;
    sampler_info.magFilter = VK_FILTER_NEAREST;
    sampler_info.minFilter = VK_FILTER_NEAREST;
    if (dfn.vkCreateSampler(device, &sampler_info, nullptr, &sampler_nearest_) != VK_SUCCESS) return false;
    VkDescriptorSetLayoutBinding bindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo set_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 3;
    set_info.pBindings = bindings;
    if (dfn.vkCreateDescriptorSetLayout(device, &set_info, nullptr, &compose_set_layout_) != VK_SUCCESS) {
      return false;
    }
    VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 6 * sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &compose_set_layout_;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    if (dfn.vkCreatePipelineLayout(device, &layout_info, nullptr, &compose_layout_) != VK_SUCCESS) return false;
    VkDescriptorSetLayout layouts[kFrameSlots];
    std::fill(layouts, layouts + kFrameSlots, compose_set_layout_);
    VkDescriptorSetAllocateInfo alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = descriptor_pool_;
    alloc.descriptorSetCount = kFrameSlots;
    alloc.pSetLayouts = layouts;
    if (dfn.vkAllocateDescriptorSets(device, &alloc, compose_sets_) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo pipeline_info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                           VK_SHADER_STAGE_COMPUTE_BIT, compose_shader_, "main", nullptr};
    pipeline_info.layout = compose_layout_;
    if (dfn.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &compose_pipeline_) !=
        VK_SUCCESS) {
      compose_pipeline_ = VK_NULL_HANDLE;
      return false;
    }
  }
  objects_failed_ = false;
  REXGPU_INFO("[native] native renderer ready (side-car; targets {} + {})", "R16G16B16A16_UNORM", "D32_SFLOAT");
  return true;
}

bool NativeRenderer::EnsureParityPipeline() {
  if (parity_pipeline_ != VK_NULL_HANDLE) return true;
  if (parity_failed_ || !EnsureObjects()) return false;
  parity_failed_ = true;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  std::vector<uint32_t> spirv;
  std::string error;
  if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT, kParityComputeShader, spirv, error)) {
    REXGPU_ERROR("[native] parity shader compilation failed: {}", error);
    return false;
  }
  parity_shader_ = ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
  VkDescriptorSetLayoutBinding bindings[3] = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
  VkDescriptorSetLayoutCreateInfo set_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  set_info.bindingCount = 3;
  set_info.pBindings = bindings;
  if (parity_shader_ == VK_NULL_HANDLE ||
      dfn.vkCreateDescriptorSetLayout(device, &set_info, nullptr, &parity_set_layout_) != VK_SUCCESS) {
    return false;
  }
  VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 12 * sizeof(uint32_t)};
  VkPipelineLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &parity_set_layout_;
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &push;
  if (dfn.vkCreatePipelineLayout(device, &layout_info, nullptr, &parity_layout_) != VK_SUCCESS) return false;
  VkComputePipelineCreateInfo pipeline_info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                         parity_shader_, "main", nullptr};
  pipeline_info.layout = parity_layout_;
  if (dfn.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &parity_pipeline_) !=
      VK_SUCCESS) {
    parity_pipeline_ = VK_NULL_HANDLE;
    return false;
  }
  VkDescriptorSetLayout layouts[kFrameSlots];
  std::fill(layouts, layouts + kFrameSlots, parity_set_layout_);
  VkDescriptorSetAllocateInfo alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  alloc.descriptorPool = descriptor_pool_;
  alloc.descriptorSetCount = kFrameSlots;
  alloc.pSetLayouts = layouts;
  if (dfn.vkAllocateDescriptorSets(device, &alloc, parity_sets_) != VK_SUCCESS) return false;
  // The results: kParityCounterCount uints per frame slot, read by the CPU.
  uint32_t memory_type = 0;
  if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
          vulkan_device, kFrameSlots * kParityCounterCount * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          ui::vulkan::util::MemoryPurpose::kReadback, parity_buffer_, parity_memory_, &memory_type)) {
    return false;
  }
  parity_coherent_ = (vulkan_device->memory_types().host_coherent >> memory_type) & 1;
  void* mapping = nullptr;
  if (dfn.vkMapMemory(device, parity_memory_, 0, VK_WHOLE_SIZE, 0, &mapping) != VK_SUCCESS) return false;
  parity_mapping_ = static_cast<uint32_t*>(mapping);
  std::memset(parity_mapping_, 0, kFrameSlots * kParityCounterCount * sizeof(uint32_t));
  FlushParityMemory();
  parity_failed_ = false;
  return true;
}

bool NativeRenderer::EnsureColorParityPipeline() {
  if (color_parity_pipeline_ != VK_NULL_HANDLE) return true;
  // The result buffer and its mapping come with the depth check's objects.
  if (color_parity_failed_ || !EnsureParityPipeline()) return false;
  color_parity_failed_ = true;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  std::vector<uint32_t> spirv;
  std::string error;
  if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT, kColorParityComputeShader, spirv,
                                             error)) {
    REXGPU_ERROR("[native] colour parity shader compilation failed: {}", error);
    return false;
  }
  color_parity_shader_ =
      ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
  VkDescriptorSetLayoutBinding bindings[4] = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
  VkDescriptorSetLayoutCreateInfo set_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  set_info.bindingCount = 4;
  set_info.pBindings = bindings;
  if (color_parity_shader_ == VK_NULL_HANDLE ||
      dfn.vkCreateDescriptorSetLayout(device, &set_info, nullptr, &color_parity_set_layout_) != VK_SUCCESS) {
    return false;
  }
  VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 10 * sizeof(uint32_t)};
  VkPipelineLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &color_parity_set_layout_;
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &push;
  if (dfn.vkCreatePipelineLayout(device, &layout_info, nullptr, &color_parity_layout_) != VK_SUCCESS) return false;
  VkComputePipelineCreateInfo pipeline_info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                         color_parity_shader_, "main", nullptr};
  pipeline_info.layout = color_parity_layout_;
  if (dfn.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &color_parity_pipeline_) !=
      VK_SUCCESS) {
    color_parity_pipeline_ = VK_NULL_HANDLE;
    return false;
  }
  VkDescriptorSetLayout layouts[kFrameSlots];
  std::fill(layouts, layouts + kFrameSlots, color_parity_set_layout_);
  VkDescriptorSetAllocateInfo alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  alloc.descriptorPool = descriptor_pool_;
  alloc.descriptorSetCount = kFrameSlots;
  alloc.pSetLayouts = layouts;
  if (dfn.vkAllocateDescriptorSets(device, &alloc, color_parity_sets_) != VK_SUCCESS) return false;
  color_parity_failed_ = false;
  return true;
}

// The emulated scene's colour as the colour parity check copies it out of the EDRAM (emulated host pixels).
bool NativeRenderer::EnsureEmulatedSceneImage(uint32_t width, uint32_t height) {
  if (emulated_scene_image_ != VK_NULL_HANDLE && emulated_scene_width_ == width &&
      emulated_scene_height_ == height) {
    return true;
  }
  Retired r;
  r.submission = command_processor_.GetCurrentSubmission();
  r.images[2] = emulated_scene_image_, r.memory[2] = emulated_scene_memory_, r.views[2] = emulated_scene_view_;
  if (emulated_scene_image_) retired_.push_back(r);
  emulated_scene_image_ = VK_NULL_HANDLE, emulated_scene_memory_ = VK_NULL_HANDLE;
  emulated_scene_view_ = VK_NULL_HANDLE;
  emulated_scene_width_ = emulated_scene_height_ = 0;
  if (!width || !height) return false;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = kSceneCopyFormat;
  info.extent = {width, height, 1};
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (!ui::vulkan::util::CreateDedicatedAllocationImage(vulkan_device, info,
                                                        ui::vulkan::util::MemoryPurpose::kDeviceLocal,
                                                        emulated_scene_image_, emulated_scene_memory_)) {
    return false;
  }
  VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view_info.image = emulated_scene_image_;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = kSceneCopyFormat;
  view_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange();
  if (vulkan_device->functions().vkCreateImageView(vulkan_device->device(), &view_info, nullptr,
                                                   &emulated_scene_view_) != VK_SUCCESS) {
    return false;
  }
  emulated_scene_width_ = width;
  emulated_scene_height_ = height;
  return true;
}

void NativeRenderer::FlushParityMemory() {  if (parity_coherent_ || !parity_memory_) return;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  VkMappedMemoryRange range = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, parity_memory_, 0, VK_WHOLE_SIZE};
  vulkan_device->functions().vkFlushMappedMemoryRanges(vulkan_device->device(), 1, &range);
}

void NativeRenderer::InvalidateParityMemory() {
  if (parity_coherent_ || !parity_memory_) return;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  VkMappedMemoryRange range = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, parity_memory_, 0, VK_WHOLE_SIZE};
  vulkan_device->functions().vkInvalidateMappedMemoryRanges(vulkan_device->device(), 1, &range);
}

bool NativeRenderer::EnsureTargets(uint32_t width, uint32_t height) {
  if (color_image_ != VK_NULL_HANDLE && width == target_width_ && height == target_height_) return true;
  if (!EnsureObjects()) return false;
  RetireTargets();
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  auto make = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VkImage& image,
                  VkDeviceMemory& memory, VkImageView& view) {
    VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!ui::vulkan::util::CreateDedicatedAllocationImage(vulkan_device, info,
                                                          ui::vulkan::util::MemoryPurpose::kDeviceLocal, image,
                                                          memory)) {
      return false;
    }
    VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(aspect);
    return dfn.vkCreateImageView(device, &view_info, nullptr, &view) == VK_SUCCESS;
  };
  if (!make(kColorFormat,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, color_image_, color_memory_, color_view_) ||
      !make(kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT, depth_image_, depth_memory_, depth_view_)) {
    REXGPU_ERROR("[native] could not create the {}x{} targets", width, height);
    RetireTargets();
    return false;
  }
  VkImageView views[2] = {color_view_, depth_view_};
  VkFramebufferCreateInfo fb_info = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  fb_info.renderPass = render_pass_load_;
  fb_info.attachmentCount = 2;
  fb_info.pAttachments = views;
  fb_info.width = width;
  fb_info.height = height;
  fb_info.layers = 1;
  if (dfn.vkCreateFramebuffer(device, &fb_info, nullptr, &framebuffer_) != VK_SUCCESS) {
    RetireTargets();
    return false;
  }
  target_width_ = width;
  target_height_ = height;
  color_readable_ = false;
  cleared_this_frame_ = false;  // new images: the next pass clears them
  native_stale_ = true;         // replacing: and reloads them from the EDRAM
  // The clay shader's specialization constants depend on the size (the game's shaders have none: their pipelines
  // stay, so a claimed draw's pipeline still exists when it is recorded).
  if (!shading_) {
    WaitPipelineJobs();  // Session 47
    for (auto& [key, pipeline] : pipelines_) RetirePipeline(pipeline);
    pipelines_.clear();
    ++g_draw_rec_generation;  // (Session 95, r3cmd)
    last_truth_valid_ = false;  // Session 47
  }
  REXGPU_INFO("[native] targets {}x{} (scene {}x{}, gpu_native_scale {:.3f})", width, height, scene_width_,
              scene_height_, target_scale_);
  return true;
}

// Objects the GPU may still use are destroyed once the submission that was open when they were retired has
// completed (DestroyRetired).
void NativeRenderer::RetireTargets() {
  Retired r;
  r.submission = command_processor_.GetCurrentSubmission();
  r.images[0] = color_image_, r.images[1] = depth_image_;
  r.memory[0] = color_memory_, r.memory[1] = depth_memory_;
  r.views[0] = color_view_, r.views[1] = depth_view_;
  r.framebuffer = framebuffer_;
  r.replace_set = writeback_set_;  // it points at these targets
  writeback_set_ = VK_NULL_HANDLE;
  if (color_image_ || depth_image_ || framebuffer_ || r.replace_set) retired_.push_back(r);
  color_image_ = depth_image_ = VK_NULL_HANDLE;
  color_memory_ = depth_memory_ = VK_NULL_HANDLE;
  color_view_ = depth_view_ = VK_NULL_HANDLE;
  framebuffer_ = VK_NULL_HANDLE;
  target_width_ = target_height_ = 0;
}

void NativeRenderer::RetirePipeline(VkPipeline pipeline) {
  if (pipeline == VK_NULL_HANDLE) return;
  Retired r;
  r.submission = command_processor_.GetCurrentSubmission();
  r.pipeline = pipeline;
  retired_.push_back(r);
}

void NativeRenderer::DestroyRetired(bool all) {
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  const uint64_t completed = command_processor_.GetCompletedSubmission();
  auto it = retired_.begin();
  while (it != retired_.end()) {
    if (!all && it->submission > completed) {
      ++it;
      continue;
    }
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyFramebuffer, device, it->framebuffer);
    for (int i = 0; i < 4; ++i) {
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device, it->views[i]);
    }
    for (int i = 0; i < 3; ++i) {
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImage, device, it->images[i]);
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, it->memory[i]);
    }
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device, it->pipeline);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, it->buffer);  // Session 46 (before its memory)
    if (it->replace_set != VK_NULL_HANDLE && replace_descriptor_pool_ != VK_NULL_HANDLE) {
      dfn.vkFreeDescriptorSets(device, replace_descriptor_pool_, 1, &it->replace_set);
    }
    it = retired_.erase(it);
  }
}

void NativeRenderer::DestroyObjects() {
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  if (!vulkan_device) return;
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  // Called at shutdown, after the command processor has awaited all its submissions.
  RetireTargets();
  EnsureEmulatedSceneImage(0, 0);  // retires it
  StopPipelineWorker();  // Session 47
  for (auto& [key, pipeline] : pipelines_) RetirePipeline(pipeline);
  pipelines_.clear();
  ++g_draw_rec_generation;  // (Session 95, r3cmd)
  last_truth_valid_ = false;  // Session 47
  RetirePipeline(color_parity_pipeline_);
  color_parity_pipeline_ = VK_NULL_HANDLE;
  RetirePipeline(compose_pipeline_);
  compose_pipeline_ = VK_NULL_HANDLE;
  RetirePipeline(parity_pipeline_);
  parity_pipeline_ = VK_NULL_HANDLE;
  RetirePipeline(load_pipeline_);
  load_pipeline_ = VK_NULL_HANDLE;
  RetirePipeline(writeback_pipeline_);
  writeback_pipeline_ = VK_NULL_HANDLE;
  // Session 42: post targets and objects.
  for (PostTarget& target : post_targets_) RetirePostTarget(target);
  post_targets_.clear();
  for (uint32_t cls = 0; cls < kPostTargetClasses; ++cls) {  // Session 43: per colour class
    RetirePipeline(post_load_pipeline_[cls]);
    RetirePipeline(post_load_depth_pipeline_[cls]);
    RetirePipeline(post_load_both_pipeline_[cls]);
    post_load_pipeline_[cls] = post_load_depth_pipeline_[cls] = post_load_both_pipeline_[cls] = VK_NULL_HANDLE;
  }
  RetirePipeline(post_writeback_pipeline_);
  post_writeback_pipeline_ = VK_NULL_HANDLE;
  RetirePipeline(resolve_pipeline_);  // Session 45 (gpu_native_resolve)
  resolve_pipeline_ = VK_NULL_HANDLE;
  RetirePipeline(resolve_tex_pipeline_);  // Session 45 (N3a)
  resolve_tex_pipeline_ = VK_NULL_HANDLE;
  RetirePipeline(resolve_scaled_pipeline_);  // Session 52
  resolve_scaled_pipeline_ = VK_NULL_HANDLE;
  RetirePipeline(resolve_scaled_tex_pipeline_);
  resolve_scaled_tex_pipeline_ = VK_NULL_HANDLE;
  resolve_scaled_failed_ = false;
  RetirePipeline(resolve_scaled_full_pipeline_);  // Session 52
  resolve_scaled_full_pipeline_ = VK_NULL_HANDLE;
  resolve_scaled_full_failed_ = false;
  RetirePipeline(resolve_full_1x_pipeline_);  // Session 59
  RetirePipeline(resolve_depth_1x_pipeline_);  // Session 59
  resolve_depth_1x_pipeline_ = VK_NULL_HANDLE;
  resolve_depth_1x_failed_ = false;
  RetirePipeline(resolve_depth_scaled_pipeline_);
  resolve_depth_scaled_pipeline_ = VK_NULL_HANDLE;
  resolve_depth_scaled_failed_ = false;
  for (VkPipeline& p : resolve_full_1x_tex_pipelines_) {
    RetirePipeline(p);
    p = VK_NULL_HANDLE;
  }
  resolve_full_1x_tex_failed_ = false;
  resolve_full_1x_pipeline_ = VK_NULL_HANDLE;
  resolve_full_1x_failed_ = false;
  for (VkPipeline& p : resolve_scaled_full_tex_pipelines_) {  // Session 59
    RetirePipeline(p);
    p = VK_NULL_HANDLE;
  }
  resolve_scaled_full_tex_failed_ = false;
  if (resolve_scaled_full_tex_layout_ != VK_NULL_HANDLE) {
    dfn.vkDestroyPipelineLayout(device, resolve_scaled_full_tex_layout_, nullptr);
    resolve_scaled_full_tex_layout_ = VK_NULL_HANDLE;
  }
  if (resolve_scaled_full_layout_ != VK_NULL_HANDLE) {
    dfn.vkDestroyPipelineLayout(device, resolve_scaled_full_layout_, nullptr);
    resolve_scaled_full_layout_ = VK_NULL_HANDLE;
  }
  for (VkPipeline& p : materialize_pipeline_) {  // Session 49
    RetirePipeline(p);
    p = VK_NULL_HANDLE;
  }
  // Session 46: the truth images, framebuffers, staging buffer and store objects.
  RetireTruthAll();
  {
    Retired r;
    r.images[0] = truth_dummy_ds_;
    r.memory[0] = truth_dummy_ds_memory_;
    r.views[0] = truth_dummy_stencil_view_;
    r.views[1] = truth_dummy_depth_view_;
    if (truth_dummy_ds_ || truth_dummy_stencil_view_ || truth_dummy_depth_view_) retired_.push_back(r);
    truth_dummy_ds_ = VK_NULL_HANDLE;
    truth_dummy_ds_memory_ = VK_NULL_HANDLE;
    truth_dummy_stencil_view_ = truth_dummy_depth_view_ = VK_NULL_HANDLE;
  }
  for (const TruthFramebufferEntry& f : truth_framebuffers_) {
    Retired r;
    r.framebuffer = f.framebuffer;
    retired_.push_back(r);
  }
  truth_framebuffers_.clear();
  if (truth_staging_ != VK_NULL_HANDLE) {
    Retired r;
    r.buffer = truth_staging_;
    r.memory[0] = truth_staging_memory_;
    retired_.push_back(r);
    truth_staging_ = VK_NULL_HANDLE;
    truth_staging_memory_ = VK_NULL_HANDLE;
  }
  for (VkPipeline& p : truth_store_pipelines_) {
    RetirePipeline(p);
    p = VK_NULL_HANDLE;
  }
  RetirePipeline(truth_fill_pipeline_);  // Session 47
  truth_fill_pipeline_ = VK_NULL_HANDLE;
  for (VkPipeline& p : truth_fill_tex_pipelines_) {  // Session 52
    RetirePipeline(p);
    p = VK_NULL_HANDLE;
  }
  truth_fill_tex_small_failed_ = false;  // Session 59
  if (truth_fill_tex_layout_ != VK_NULL_HANDLE) {  // (review: and its layout)
    dfn.vkDestroyPipelineLayout(device, truth_fill_tex_layout_, nullptr);
    truth_fill_tex_layout_ = VK_NULL_HANDLE;
  }
  truth_fill_tex_failed_ = false;
  // Session 79 (gpu_native_truth_alias_direct).
  for (auto& by_owner : alias_direct_pipelines_) {
    for (auto& by_target : by_owner) {
      for (VkPipeline& p : by_target) {
        RetirePipeline(p);
        p = VK_NULL_HANDLE;
      }
    }
  }
  for (VkShaderModule& s : alias_direct_shaders_) ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, s);
  alias_direct_failed_ = false;
  truth_alias_read_owners_.clear();
  DestroyRetired(true);
  OwnCrowdDestroy();  // wwe2k14 patch (Session 80, own crowd)
  OwnedDestroy();     // wwe2k14 patch (Session 80, owned steps)
  // Session 80 (gpu_native_truth_skip_writeback_depth): the stencil snapshots and the depth write-back's objects.
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device, stencil_snap_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device, materialize_depth_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device, materialize_depth_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, stencil_snap_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, stencil_snap_memory_);
  stencil_slots_busy_ = 0;
  depth_wb_failed_ = false;
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device, alias_direct_layout_);
  truth_retired_sets_.clear();  // freed with the pool
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorPool, device, truth_pool_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device, truth_store_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device, truth_store_set_layout_);
  for (VkShaderModule& s : truth_store_shaders_) ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, s);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, truth_fill_shader_);  // Session 47
  RetirePipeline(owned_depth_up_pipeline_);  // Session 95 (the owned depth up-pass)
  owned_depth_up_pipeline_ = VK_NULL_HANDLE;
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device, owned_depth_up_render_pass_);
  for (VkShaderModule& s : owned_depth_up_shaders_) ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, s);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device, owned_depth_up_layout_);
  owned_depth_up_failed_ = false;
  RetirePipeline(truth_depth_resample_pipeline_);  // Session 54 (the depth pass)
  truth_depth_resample_pipeline_ = VK_NULL_HANDLE;
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device, truth_depth_resample_pass_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, truth_depth_resample_shader_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device, truth_depth_resample_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device, truth_depth_set_layout_);
  truth_depth_resample_failed_ = false;
  for (VkPipeline& p : truth_resample_pipelines_) {  // Session 54
    RetirePipeline(p);
    p = VK_NULL_HANDLE;
  }
  for (VkRenderPass& p : truth_resample_passes_) ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device, p);
  for (VkShaderModule& s : truth_resample_shaders_) ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, s);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device, truth_resample_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device, truth_resample_set_layout_);
  truth_resample_failed_ = false;
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device, resolve_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, resolve_shader_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device, resolve_tex_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, resolve_tex_shader_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device, post_writeback_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device, post_writeback_set_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, post_writeback_shader_);
  for (uint32_t cls = 0; cls < kPostTargetClasses; ++cls) {
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device, post_render_pass_begin_[cls]);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device, post_render_pass_continue_[cls]);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device, post_render_pass_begin_no_depth_[cls]);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device, post_render_pass_continue_no_depth_[cls]);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device, post_render_pass_begin_keep_[cls]);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device, truth_render_pass_keep_[cls]);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device, truth_render_pass_keep_nodep_[cls]);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device, truth_render_pass_no_depth_nodep_[cls]);
    for (auto& passes : truth_render_pass_dont_care_) {  // Session 66 (G2.1)
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device, passes[cls]);
    }
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device, truth_render_pass_clear_[cls]);  // Session 80
    for (auto& passes : truth_render_pass_color_none_) {  // Session 73
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device, passes[cls]);
    }
  }
  // Frees the load and write-back sets with it.
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorPool, device, replace_descriptor_pool_);
  load_set_ = writeback_set_ = VK_NULL_HANDLE;
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device, load_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device, writeback_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device, load_set_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device, writeback_set_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, load_vertex_shader_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, load_fragment_shader_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, writeback_shader_);
  if (parity_mapping_) dfn.vkUnmapMemory(device, parity_memory_), parity_mapping_ = nullptr;
  for (int i = 0; i < 2; ++i) {
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, dump_buffers_[i]);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, dump_memory_[i]);
  }
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, parity_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, parity_memory_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device, parity_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device, parity_set_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, parity_shader_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device, color_parity_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device, color_parity_set_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, color_parity_shader_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device, compose_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device, compose_set_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorPool, device, descriptor_pool_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroySampler, device, sampler_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroySampler, device, sampler_nearest_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyQueryPool, device, query_pool_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyQueryPool, device, probe_pool_);  // Session 82
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, compose_shader_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device, clay_fragment_shader_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device, render_pass_clear_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device, render_pass_load_);
}

void NativeRenderer::OnCachesCleared() {
  if (vertex_input_) vertex_input_->ClearForms();  // Session 95: the modules it points at are made again
  if (skin_mirror_) skin_mirror_->ClearPipeMemo();  // Session 95 (gpu_native_skin_once): the same
  TruthDropRecords();  // Session 49 (none expected: ClearCaches wrote them into memory)
  // The vertex shader modules and pipeline layouts our pipelines were made with may be destroyed and their handles
  // reused: make new pipelines from now on.
  {
    // Session 61 (review): jobs not started are dropped, not waited for (a draw queues its own again; the stored
    // pipelines' backlog would hold the clear for seconds on a slow driver).
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    DropPipelineJobsLocked();
  }
  WaitPipelineJobs();  // Session 47: the worker's pipelines use them too (made before they go)
  for (auto& [key, pipeline] : pipelines_) RetirePipeline(pipeline);
  pipelines_.clear();
  ++g_draw_rec_generation;  // (Session 95, r3cmd)
  last_truth_valid_ = false;  // Session 47
}

VkPipeline NativeRenderer::GetPipeline(const Draw& draw, bool* pending) {
  // With the game's pixel shaders (gpu_native_shading 1), a draw without one writes depth only; with our clay
  // shading, the tint is one colour per vertex shader (the game's material groups).
  const bool game_shading = shading_;
  const VkShaderModule pixel_shader = game_shading ? draw.pixel_shader : VK_NULL_HANDLE;
  const uint32_t color_mask = game_shading ? (pixel_shader ? draw.color_write_mask & 0xF : 0) : 0xF;
  // No blending is RB_BLENDCONTROL one, zero, add for colour and alpha.
  const uint32_t blend = game_shading && pixel_shader && draw.blend_control != 0x00010001 ? draw.blend_control : 0;
  // Session 42: a post draw goes to a post target (its render pass, the stencil marking what it covers) and keeps the
  // emulated pipeline's geometry shader (quads, rectangles).
  const bool post = draw.post;
  // Session 43: the post target's colour format. Session 46: a truth draw's image class.
  const uint32_t post_class = post ? (draw.truth ? uint32_t(draw.truth_class) : PostTargetClass(draw)) : 0;
  if (post && post_render_pass_continue_[post_class] == VK_NULL_HANDLE) return VK_NULL_HANDLE;
  const VkShaderModule geometry_shader = draw.raster.expanded ? draw.raster.geometry_shader : VK_NULL_HANDLE;
  PipelineKey key = {draw.vertex_shader, pixel_shader, draw.layout, RasterKey(draw.raster) | (color_mask << 16),
                     blend, game_shading ? 0 : Mix64(draw.vertex_shader_hash), geometry_shader,
                     post ? 1u + post_class + (draw.truth ? 8u : 0u) : 0u,  // Session 46: truth (no coverage marks)
                     draw.vi};  // Session 95
  // (Session 69 review: without depthClamp the clamp bit changes nothing - not a separate pipeline.)
  if (!command_processor_.GetVulkanDevice()->properties().depthClamp || !REXCVAR_GET(gpu_native_clip_disable_clamp)) {
    key.state &= ~(1u << 14);
  }
  if (post && draw.truth && draw.raster.stencil) key.variant |= (1u | (draw.raster.stencil_ops << 1)) << 4;
  // Session 47 (CPU): a truth draw's pipeline is mostly the last one's (one comparison instead of the map's hash).
  if (draw.truth && last_truth_valid_ && key == last_truth_key_) return last_truth_pipeline_;
  auto found = pipelines_.find(key);
  if (found != pipelines_.end()) {
    // Session 70: the first draw using a variant prewarmed from an older record's guess stores its own record.
    if (!prewarm_guessed_.empty() && prewarm_guessed_.erase(key) && storage_file_) [[unlikely]] {
      ++prewarm_guessed_stored_;
      if (!draw.no_store) StorePipelineRecord(draw, key);
    }
    if (draw.truth && found->second != VK_NULL_HANDLE) {
      last_truth_key_ = key;
      last_truth_pipeline_ = found->second;
      last_truth_valid_ = true;
    }
    return found->second;
  }
  if (storage_file_ && !draw.no_store) StorePipelineRecord(draw, key);  // Session 61 (a key not made yet in this run; Session 95: an attribute draw stores the plain record it derives from, section 3.4)
  if (!prewarm_guessed_.empty() && prewarm_guessed_.erase(key)) ++prewarm_guessed_stored_;  // (Session 70: still queued)
  const float spec_values[3] = {float(target_height_) * 1.2f, float(target_width_) * 0.5f,
                                float(target_height_) * 0.5f};
  // Session 47 (CPU): queued for the worker thread (a truth draw is skipped until its pipeline is made).
  if (pending && draw.sync_pipeline) pending = nullptr;  // (P7: a crowd member's attribute pipeline is made here, below, not queued)
  if (pending) {
    DrainPipelineJobs();
    found = pipelines_.find(key);
    if (found != pipelines_.end()) return found->second;
    *pending = true;
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    if (pipeline_pending_.insert(key).second) { REXCVAR_SET(gpu_diag_pipelines_pending, int32_t(pipeline_pending_.size()));
      StartPipelineWorkersLocked();
      // (Session 69 review: a draw's own job ahead of the stored pipelines' backlog - it is skipped until made.)
      pipeline_jobs_.push_front({key, draw, {spec_values[0], spec_values[1], spec_values[2]}});
      pipeline_cv_.notify_one();
#if REX_PLATFORM_WIN32
      // Session 80 (gpu_native_pipeline_workers_raise): the workers in a warm-up job finish it at normal priority, so
      // this job (at the front) starts as soon as one is free; each worker sets its own priority at its next job.
      if (REXCVAR_GET(gpu_native_pipeline_workers_quiet) && REXCVAR_GET(gpu_native_pipeline_workers_raise) &&
          !pipeline_running_.empty()) {
        for (const auto& running : pipeline_running_) {
          if (running.second) SetThreadPriority(static_cast<HANDLE>(running.second), THREAD_PRIORITY_NORMAL);
        }
        static std::atomic<uint32_t> raises{0};
        const uint32_t n = raises.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((n & (n - 1)) == 0) {  // (1, 2, 4, 8, ...)
          REXGPU_INFO("[native] pipeline workers raised for a skipped draw's pipeline: {} times ({} running)", n,
                      pipeline_running_.size());
        }
      }
#endif
    }
    return VK_NULL_HANDLE;
  }
  // Session 55 (review): made at once while the worker may have it - its finished pipelines first, and a job for this
  // key it has not started yet goes (made here instead of twice).
  // Session 69: a key a worker is making right now is waited for (not made a second time alongside it).
  if (!pipeline_threads_.empty()) {
    DrainPipelineJobs();
    found = pipelines_.find(key);
    if (found != pipelines_.end()) return found->second;
    bool waited = false;
    {
      std::unique_lock<std::mutex> lock(pipeline_mutex_);
      for (auto job = pipeline_jobs_.begin(); job != pipeline_jobs_.end(); ++job) {
        if (job->key == key) {
          pipeline_jobs_.erase(job);
          pipeline_pending_.erase(key); REXCVAR_SET(gpu_diag_pipelines_pending, int32_t(pipeline_pending_.size()));
          break;
        }
      }
      if (auto running = pipeline_running_.find(key); running != pipeline_running_.end()) {
#if REX_PLATFORM_WIN32
        // Session 70: a quiet worker making this key at normal priority while the command thread waits for it (no
        // priority inversion; the worker sets its own priority again at its next job).
        if (running->second) SetThreadPriority(static_cast<HANDLE>(running->second), THREAD_PRIORITY_NORMAL);
#endif
        pipeline_idle_cv_.wait(lock, [&]() { return !pipeline_running_.count(key); });
        waited = true;
      }
    }
    if (waited) {
      rex::perf::frame_events::Get().pipelines.fetch_add(1, std::memory_order_relaxed);  // Session 73: waited for
      DrainPipelineJobs();
      found = pipelines_.find(key);
      if (found != pipelines_.end()) return found->second;
    }
  }
  rex::perf::frame_events::Get().pipelines.fetch_add(1, std::memory_order_relaxed);  // Session 73: made here
  const VkPipeline pipeline = CreatePipeline(draw, key, spec_values);
  if (pipeline == VK_NULL_HANDLE && pipelines_failed_++ < 8) {
    REXGPU_WARN("[native] pipeline creation failed for vertex shader {:016X}, pixel shader {:016X}",
                draw.vertex_shader_hash, draw.pixel_shader_hash);
  }
  pipelines_.emplace(key, pipeline);
  return pipeline;
}

// Session 47: the worker thread of gpu_native_truth_async_pipelines.
void NativeRenderer::PipelineWorker() {
  // Session 70 (gpu_native_pipeline_workers_quiet): the stored pipelines' warm-up runs below the game's threads (a
  // 4-core player's warm-up competed with the game during entrances); a draw's own job (it is skipped until made) at
  // normal priority (review). A real handle, so the command thread can raise this worker while it waits for its key.
  void* self = nullptr;
#if REX_PLATFORM_WIN32
  self = OpenThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION, FALSE, GetCurrentThreadId());
#endif
  const bool quiet = REXCVAR_GET(gpu_native_pipeline_workers_quiet) && !REXCVAR_GET(gpu_native_precompile);
  std::unique_lock<std::mutex> lock(pipeline_mutex_);
  for (;;) {
    pipeline_cv_.wait(lock, [this]() { return pipeline_quit_ || !pipeline_jobs_.empty(); });
    if (pipeline_jobs_.empty()) break;  // (quit with nothing left)
    PipelineJob job = std::move(pipeline_jobs_.front());
    pipeline_jobs_.pop_front();
    ++pipeline_busy_;
    pipeline_running_[job.key] = self;  // Session 69: the command thread waits for it rather than making it twice
    lock.unlock();
#if REX_PLATFORM_WIN32
    if (quiet) {
      SetThreadPriority(GetCurrentThread(), job.prewarm ? THREAD_PRIORITY_BELOW_NORMAL : THREAD_PRIORITY_NORMAL);
    }
#endif
    g_eds_native_prewarm = job.prewarm;
    const VkPipeline pipeline = CreatePipeline(job.draw, job.key, job.spec);
    g_eds_native_prewarm = false;
    if (pipeline == VK_NULL_HANDLE && pipeline_worker_failed_++ < 8) {
      REXGPU_WARN("[native] pipeline creation failed (worker) for vertex shader {:016X}, pixel shader {:016X}",
                  job.draw.vertex_shader_hash, job.draw.pixel_shader_hash);
    }
    if (job.prewarm) prewarm_made_.fetch_add(1, std::memory_order_relaxed);  // Session 70 (precompile progress)
    lock.lock();
    pipeline_done_.emplace_back(job.key, pipeline);
    pipeline_done_count_.fetch_add(1, std::memory_order_release);
    pipeline_running_.erase(job.key);
    --pipeline_busy_;
    // (Session 69: every finished job wakes the waiters - a waiter for this key, or WaitPipelineJobs at the last one.)
    pipeline_idle_cv_.notify_all();
  }
  pipeline_idle_cv_.notify_all();
#if REX_PLATFORM_WIN32
  if (self) CloseHandle(static_cast<HANDLE>(self));  // (no job of this worker is in pipeline_running_ any more)
#endif
}

// Session 69 (gpu_native_pipeline_workers): the pool, made once with the first job (0 = a third of the CPU threads,
// 1 to 3 - the game's own threads keep the rest).
void NativeRenderer::StartPipelineWorkersLocked() {
  if (!pipeline_threads_.empty()) return;
  int32_t workers = REXCVAR_GET(gpu_native_pipeline_workers);
  if (workers <= 0) workers = std::clamp(int32_t(std::thread::hardware_concurrency() / 3), 1, 3);
  workers = std::min(workers, 8);  // (review: a typo of 64 is not 64 threads)
  // Session 70 (gpu_native_precompile): nothing else runs - a worker per CPU thread (or the count set: a test).
  if (REXCVAR_GET(gpu_native_precompile)) {
    workers = REXCVAR_GET(gpu_native_pipeline_workers) > 0 ? REXCVAR_GET(gpu_native_pipeline_workers)
                                                            : int32_t(std::thread::hardware_concurrency());
    workers = std::clamp(workers, 1, 32);
  }
  for (int32_t i = 0; i < workers; ++i) pipeline_threads_.emplace_back([this]() { PipelineWorker(); });
  REXGPU_INFO("[native] {} pipeline worker thread(s)", workers);
}

void NativeRenderer::DrainPipelineJobs() {
  if (!pipeline_done_count_.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lock(pipeline_mutex_);
  for (auto& [key, pipeline] : pipeline_done_) {
    pipeline_pending_.erase(key); REXCVAR_SET(gpu_diag_pipelines_pending, int32_t(pipeline_pending_.size()));
    if (!pipelines_.emplace(key, pipeline).second) RetirePipeline(pipeline);  // (made at once meanwhile)
  }
  pipeline_done_.clear();
  pipeline_done_count_.store(0, std::memory_order_relaxed);
}

void NativeRenderer::WaitPipelineJobs() {
  {
    std::unique_lock<std::mutex> lock(pipeline_mutex_);
    pipeline_idle_cv_.wait(lock, [this]() { return pipeline_jobs_.empty() && !pipeline_busy_; });
  }
  DrainPipelineJobs();
}

void NativeRenderer::DropPipelineJobsLocked() {
  for (const PipelineJob& job : pipeline_jobs_) pipeline_pending_.erase(job.key); REXCVAR_SET(gpu_diag_pipelines_pending, int32_t(pipeline_pending_.size()));
  pipeline_jobs_.clear();
  if (!pipeline_busy_) pipeline_idle_cv_.notify_all();
}

void NativeRenderer::StopPipelineWorker() {
  if (pipeline_threads_.empty()) return;
  {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    pipeline_quit_ = true;
    DropPipelineJobsLocked();  // Session 61 (review): a quit does not wait for the stored pipelines' backlog
  }
  pipeline_cv_.notify_all();
  for (std::thread& t : pipeline_threads_) t.join();
  pipeline_threads_.clear();
  pipeline_quit_ = false;  // (a later truth pipeline starts new workers)
  DrainPipelineJobs();
}

// ---------------------------------------------------------------------------------------------------------------
// Session 61: the stored native pipelines (gpu_native_pipeline_storage, gpu_native_pipeline_prewarm_ms)
// ---------------------------------------------------------------------------------------------------------------

namespace {
// 'XNPS'; the record format's version; the translator's modification version (records hold its bits); the record
// size. A file whose header differs is emptied.
struct StorageHeader {
  uint32_t magic, version, modification_version, record_size;
};
constexpr uint32_t kStorageMagic = 0x53504E58;
constexpr uint32_t kStorageVersion = 0x20261025;
}  // namespace

void NativeRenderer::InitializeStorage(const std::filesystem::path& shareable_root, uint32_t title_id) {
  ShutdownStorage();
  TruthStorageOpen(shareable_root, title_id);  // Session 73 (native_truth.cpp)
  if (!REXCVAR_GET(gpu_native_pipeline_storage)) return;
  // (Review: one file per render path, as the emulated path's .fsi / .fbo pipeline storage - the translations' bits
  // differ between them.)
  const bool interlock = command_processor_.NativeEdramBuffer() != VK_NULL_HANDLE;
  const std::filesystem::path path =
      shareable_root / fmt::format("{:08X}.native.{}.vk.xnps", title_id, interlock ? "fsi" : "fbo");
  storage_file_ = rex::filesystem::OpenFile(path, "a+b");
  if (!storage_file_) {
    REXGPU_WARN("[native][storage] cannot open {}: native pipelines not stored", rex::path_to_utf8(path));
    return;
  }
  const StorageHeader want = {kStorageMagic, kStorageVersion, uint32_t(SpirvShaderTranslator::Modification::kVersion),
                              uint32_t(sizeof(PipelineRecord))};
  StorageHeader header = {};
  rex::filesystem::Seek(storage_file_, 0, SEEK_SET);
  if (fread(&header, sizeof(header), 1, storage_file_) == 1 && std::memcmp(&header, &want, sizeof(header)) == 0) {
    // Records until the end or the first one whose hash does not match (a torn tail: the file is cut there).
    uint64_t valid = sizeof(header);
    uint64_t hash;
    PipelineRecord record;
    while (fread(&hash, sizeof(hash), 1, storage_file_) == 1 && fread(&record, sizeof(record), 1, storage_file_) == 1) {
      if (XXH3_64bits(&record, sizeof(record)) != hash) break;
      valid += sizeof(hash) + sizeof(record);
      if (storage_hashes_.insert(hash).second) prewarm_.push_back(record);
      // Session 73: the characters' pixel shaders known before the prewarm makes their modules.
      spirv_relaxed_precision::NoteShaderPair(record.vertex_shader_hash, record.pixel_shader_hash);
      if (record.ext_bits & kRecordExtKnown) {  // Session 70: the older record it supersedes (ext bits 0), per variant
        PipelineRecord sibling = record;
        sibling.ext_bits = 0;
        storage_known_siblings_[XXH3_64bits(&sibling, sizeof(sibling))] |=
            (record.ext_bits & kRecordExtEpilogueIdle) ? uint8_t(1) : uint8_t(2);
      }
    }
    rex::filesystem::TruncateStdioFile(storage_file_, valid);
  } else {
    rex::filesystem::TruncateStdioFile(storage_file_, 0);
    fwrite(&want, sizeof(want), 1, storage_file_);
    fflush(storage_file_);
  }
  rex::filesystem::Seek(storage_file_, 0, SEEK_END);
  REXGPU_INFO("[native][storage] {} stored native pipelines ({})", prewarm_.size(), rex::path_to_utf8(path));
}

void NativeRenderer::ShutdownStorage() {
  if (storage_file_ && (prewarm_guessed_stored_ || prewarm_old_skipped_)) {
    REXGPU_INFO("[native][storage] this run: {} new records ({} for a guessed variant a draw used; {} guessed variants "
                "unused), {} older records left out for a known one",
                storage_written_, prewarm_guessed_stored_, prewarm_guessed_.size(), prewarm_old_skipped_);
  }
  if (truth_storage_file_) {  // Session 73
    fclose(truth_storage_file_);
    truth_storage_file_ = nullptr;
  }
  truth_prewarm_.clear();
  truth_stored_.clear();
  truth_spare_counts_.clear();
  truth_spare_prewarm_.clear();
  prewarm_guessed_.clear();
  storage_known_siblings_.clear();
  prewarm_old_skipped_ = prewarm_guessed_stored_ = prewarm_old_conformed_ = 0;
  if (storage_file_) {
    fflush(storage_file_);
    fclose(storage_file_);
    storage_file_ = nullptr;
  }
  storage_dirty_ = false;
  storage_hashes_.clear();
  prewarm_.clear();
  prewarm_retry_.clear();
  prewarm_next_ = 0;
  prewarm_retries_ = prewarm_logged_retries_ = prewarm_queued_ = prewarm_skipped_ = 0;
  prewarm_wait_shader_ = prewarm_wait_twin_ = prewarm_wait_pass_ = prewarm_queued_logged_ = 0;  // Session 73
  prewarm_reasons_logged_ = false;
  prewarm_retry_time_ = {};
  prewarm_converted_ = 0;
  storage_written_ = storage_mismatch_ = 0;
  prewarm_logged_ = false;
}

bool NativeRenderer::KeyFromRecord(const PipelineRecord& record, PipelineKey& key, bool convert, bool* converted,
                                   const uint32_t* exts) {
  // Session 61 (review): as the emulated path's pipeline storage refuses them - a transform-feedback variant (the
  // characters' capture with DLSS / DLAA / FSR) needs the capture latched (and with it the device feature), a LOD-bias
  // variant needs gpu_dlss; a run without an upscaler must not make them from stored records.
  if (SpirvShaderTranslator::Modification(record.vertex_modification).vertex.xfb_position &&
      !command_processor_.ObjmvLatched()) {
    return false;
  }
  if (SpirvShaderTranslator::Modification(record.pixel_modification).pixel.dlss_lod_bias_quarters &&
      !REXCVAR_GET(gpu_dlss)) {
    return false;
  }
  VulkanPipelineCache& cache = command_processor_.NativePipelineCache();
  VulkanShader* vertex_shader = cache.FindShader(record.vertex_shader_hash);
  if (!vertex_shader || vertex_shader->type() != xenos::ShaderType::kVertex) return false;
  VulkanShader* pixel_shader = nullptr;
  if (record.pixel_shader_hash) {
    pixel_shader = cache.FindShader(record.pixel_shader_hash);
    if (!pixel_shader || pixel_shader->type() != xenos::ShaderType::kPixel) return false;
  }
  // wwe2k14 patch (Session 66, B2.1 / PS spec): the modifications as the current specialisation settings would make
  // them (a record stored by a run with other settings still prewarms the pipeline the draws will ask for; a record
  // whose converted key is already made is skipped by the caller). Not for the check of a record just made.
  uint64_t vertex_modification = record.vertex_modification;
  uint64_t pixel_modification = record.pixel_modification;
  if (convert) {
    vertex_modification =
        cache.ConformStoredVertexModification(*vertex_shader, vertex_modification, converted, /*native=*/true);
    if (pixel_shader && (record.flags & kRecordNativePixel)) {
      pixel_modification = cache.ConformStoredNativePixelModification(
          *pixel_shader, pixel_modification, converted, (record.ext_bits & kRecordExtTexMaskKnown) != 0);
    }
    // wwe2k14 patch (Session 90, Adreno link): a record stored before the vertex translation declared the interpolators the pixel
    // replacement reads - both modifications get them, as IssueDraw's mask does (NativeSwapInterpolatorMask), so the stored pair is
    // the pair the draws ask for and links on drivers that refuse an unwritten input.
    if (pixel_shader && shader_swap::WidenForReplacement(record.pixel_shader_hash, vertex_modification, pixel_modification) &&
        converted) {
      *converted = true;
    }
  }
  // wwe2k14 patch (Session 66, native_ext): the NATIVE specialisation words - predicted for a stored record, the
  // draw's own for the check of a record just made.
  uint32_t pixel_ext = 0, vertex_ext = 0;
  if (convert) {
    pixel_ext = (pixel_shader && (record.flags & kRecordNativePixel)) ? cache.PredictNativePixelExt() : 0;
    vertex_ext = cache.PredictNativeVertexExt(vertex_modification, record.vertex_shader_hash);  // (Session 87: + our vertex swap)
    // Session 69: with the epilogue specialisation on, the stored draw's own bit (the prediction sets it for all).
    if ((record.ext_bits & kRecordExtKnown) && !(record.ext_bits & kRecordExtEpilogueIdle)) {
      pixel_ext &= ~uint32_t(SpirvShaderTranslator::kNativeExtEpilogueIdle);
    }
    // Session 72: the title's replacement, as NativePixelExt adds it for the draw (an idle epilogue, the switch on;
    // Session 80: or a replacement with its own epilogue - a stored record does not say whether its draw had alpha to
    // coverage, so this predicts the usual case).
    uint32_t swap_flags = 0;
    if (REXCVAR_GET(gpu_native_shader_swap) && shader_swap::Has(record.pixel_shader_hash, &swap_flags) &&
        ((pixel_ext & SpirvShaderTranslator::kNativeExtEpilogueIdle) ||
         (swap_flags & shader_swap::kFlagOwnEpilogue))) {
      pixel_ext |= SpirvShaderTranslator::kNativeExtSwap;
      pixel_ext |= shader_swap::AblateExt(record.pixel_shader_hash);  // (Session 95, B2)
    }
  } else if (exts) {
    pixel_ext = exts[0];
    vertex_ext = exts[1];
  }
  // (Session 87, skip translation: a predicted replacement translation is the one with its own bindings when it can be - the
  // draws of a replacing lean frame use those; a record made by a draw carries the bit itself.)
  if (convert) {
    if (pixel_shader && (pixel_ext & SpirvShaderTranslator::kNativeExtSwap) &&
        cache.UsesReplacementBindings(pixel_shader, pixel_ext | SpirvShaderTranslator::kNativeExtOwnBindings)) {
      pixel_ext |= SpirvShaderTranslator::kNativeExtOwnBindings;
    }
    if ((vertex_ext & SpirvShaderTranslator::kNativeExtVsSwap) &&
        cache.UsesReplacementBindings(vertex_shader, vertex_ext | SpirvShaderTranslator::kNativeExtOwnBindings) &&
        shader_swap::VertexFits(*vertex_shader, vertex_modification)) {
      vertex_ext |= SpirvShaderTranslator::kNativeExtOwnBindings;
    }
  }
  // Session 95 (out\s95\android_structural.md section 3.4): a record describes the plain draw. While gpu_native_vertex_input is on it derives the attribute key - bit 7, the attribute module, vi, restart
  // cleared under gpu_native_index_no_restart - unless the record is a refused draw's plain twin. The check of a record just made goes by the draw's own ext. At switch 0 every record gives its plain key.
  ViForm attribute_form;
  bool attribute = false;
  if (convert) {
    // (Session 95 review: a shader skin once lists draws through its _so variant, which is never stored, and falls back to the plain pipeline while that variant is being made, not to the attribute form:
    // the plain key is the one worth making ahead, or first sight of a wrestler holds the frame.)
    uint32_t skin_role = 0, skin_stride = 0;
    const bool skin_listed = shader_swap::SkinOnceLatched() && shader_swap::SkinOnceRow(record.vertex_shader_hash, &skin_role, &skin_stride);
    attribute = !skin_listed && !(record.flags & (kRecordPlainTwin | kRecordGeometry)) && VertexInputFormOf(*vertex_shader, vertex_modification, vertex_ext, &attribute_form);
    if (attribute) vertex_ext |= SpirvShaderTranslator::kNativeExtVsInput;
  } else if (vertex_ext & SpirvShaderTranslator::kNativeExtVsInput) {
    attribute = VertexInputFormOf(*vertex_shader, vertex_modification, vertex_ext & ~uint32_t(SpirvShaderTranslator::kNativeExtVsInput), &attribute_form);
    if (!attribute) return false;
  }
  // The emulated translations first (IssueDraw's order: the native pixel translation keeps their bindings).
  // (The vertex shader's ext translation is NATIVE's own; a draw's emulated one is the ext-free one.)
  auto* vertex_translation = static_cast<VulkanShader::VulkanTranslation*>(
      vertex_shader->GetOrCreateNativeTranslation(vertex_modification, vertex_ext));
  // (wwe2k14 patch, Session 66, PS spec: the emulated translation is the draw's own, without NATIVE's bits.)
  SpirvShaderTranslator::Modification emulated_pixel_modification(pixel_modification);
  VulkanPipelineCache::StripNativePixelSpec(emulated_pixel_modification);
  auto* pixel_translation =
      pixel_shader
          ? static_cast<VulkanShader::VulkanTranslation*>(pixel_shader->GetOrCreateTranslation(emulated_pixel_modification.value))
          : nullptr;
  // (Session 87, pixel skip: the record's pixel translation is the replacement's when its ext says so - then the emulated one is
  // not needed and the pixel bindings are the replacement's; else they are the emulated translation's, whatever the record's
  // layout counts say: a record stored by a run with the other setting carries the other translation's counts.)
  const bool pixel_replacement = pixel_shader && cache.UsesReplacementBindings(pixel_shader, pixel_ext);
  if (!cache.EnsureShadersTranslated(vertex_translation, pixel_translation, pixel_replacement ? 2u : 0u)) return false;
  VkShaderModule pixel_module = VK_NULL_HANDLE;
  if (record.flags & kRecordNativePixel) {
    if (!pixel_shader) return false;
    VulkanShader::VulkanTranslation* native_translation = cache.GetNativePixelShaderTranslation(
        *pixel_shader, SpirvShaderTranslator::Modification(pixel_modification),
        (record.flags & kRecordFloat24) != 0, record.quantize, (record.flags & kRecordTwin) != 0, pixel_ext);
    if (!native_translation) return false;
    pixel_module = native_translation->shader_module();
  }
  uint32_t layout_counts[4] = {record.layout_counts[0], record.layout_counts[1], record.layout_counts[2],
                               record.layout_counts[3]};
  {
    const bool vertex_replacement = cache.UsesReplacementBindings(vertex_shader, vertex_ext);
    layout_counts[2] = uint32_t(vertex_shader->TextureBindingsFor(vertex_replacement).size());
    layout_counts[3] = uint32_t(vertex_shader->SamplerBindingsFor(vertex_replacement).size());
  }
  if (pixel_shader && (record.flags & kRecordNativePixel)) {
    layout_counts[0] = uint32_t(pixel_shader->TextureBindingsFor(pixel_replacement).size());
    layout_counts[1] = uint32_t(pixel_shader->SamplerBindingsFor(pixel_replacement).size());
  }
  const VulkanPipelineCache::PipelineLayoutProvider* layout =
      command_processor_.GetPipelineLayout(layout_counts[0], layout_counts[1], layout_counts[2], layout_counts[3]);
  if (!layout) return false;
  VkShaderModule geometry_shader = VK_NULL_HANDLE;
  // Session 94 review: with the vertex-shader expansion on, no draw can ask for a geometry-shader pipeline any more (the
  // primitive processor never yields host point / rectangle / quad lists): a stored GS record from an earlier run is skipped.
  if ((record.flags & kRecordGeometry) && REXCVAR_GET(gpu_native_vs_expand)) return false;
  if (record.flags & kRecordGeometry) {
    PrimitiveProcessor::ProcessingResult primitive = {};
    primitive.host_vertex_shader_type = Shader::HostVertexShaderType::kVertex;
    primitive.host_primitive_type = xenos::PrimitiveType(record.host_primitive);
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    geometry_shader = cache.GetNativeGeometryShader(primitive,
                                                    SpirvShaderTranslator::Modification(vertex_modification),
                                                    SpirvShaderTranslator::Modification(pixel_modification),
                                                    topology);
    if (geometry_shader == VK_NULL_HANDLE) return false;
  }
  key = {vertex_translation->shader_module(), pixel_module, layout->GetPipelineLayout(), record.state,
         record.blend_control, record.tint_hash, geometry_shader, record.variant};
  if (attribute) {
    key.vi = attribute_form.vi;
    if (REXCVAR_GET(gpu_native_index_no_restart)) key.state &= ~(1u << 4);  // (RasterKey's restart bit)
  }
  return key.vertex_shader != VK_NULL_HANDLE;
}

void NativeRenderer::StorePipelineRecord(const Draw& draw, const PipelineKey& key) {
  PipelineRecord record = {};
  record.vertex_shader_hash = draw.vertex_shader_hash;
  record.vertex_modification = draw.vertex_modification;
  record.pixel_shader_hash = draw.pixel_shader_hash;
  record.pixel_modification = draw.pixel_shader_hash ? draw.pixel_modification : 0;
  record.tint_hash = key.tint_hash;
  record.state = key.state;
  if (draw.attribute) record.state = (record.state & ~(1u << 4)) | (uint32_t(draw.plain_restart) << 4);  // (Session 95: the plain draw's, as the guest had it: RasterKey's restart bit)
  record.blend_control = key.blend_control;
  record.variant = key.variant;
  std::copy(std::begin(draw.layout_counts), std::end(draw.layout_counts), record.layout_counts);
  if (key.pixel_shader != VK_NULL_HANDLE) {
    record.flags |= kRecordNativePixel | (draw.ps_float24 ? kRecordFloat24 : 0) | (draw.ps_twin ? kRecordTwin : 0);
    record.quantize = draw.ps_quantize;
  } else if (record.pixel_shader_hash) {
    // wwe2k14 patch (Session 66, PS spec): no NATIVE pixel module - the pixel specialisation bits mean nothing to this
    // pipeline (the emulated translation has none); one record, not one per setting.
    SpirvShaderTranslator::Modification stripped(record.pixel_modification);
    VulkanPipelineCache::StripNativePixelSpec(stripped);
    record.pixel_modification = stripped.value;
  }
  if (key.geometry_shader != VK_NULL_HANDLE) {
    record.flags |= kRecordGeometry;
    record.host_primitive = draw.host_primitive;
  }
  // Session 95 (section 3.4): a plain draw of a translation that has an attribute form was refused at run time: its record stays plain, so the next boot prewarms this key too.
  if (!draw.attribute && key.geometry_shader == VK_NULL_HANDLE && shader_swap::VertexMirrorLatched()) {
    VulkanShader* vertex_shader = command_processor_.NativePipelineCache().FindShader(record.vertex_shader_hash);
    ViForm unused;
    if (vertex_shader && VertexInputFormOf(*vertex_shader, draw.vertex_modification, draw.vertex_ext, &unused)) record.flags |= kRecordPlainTwin;
  }
  // wwe2k14 patch (Session 69, the Intel tester's stutter): the draw's own epilogue bit - a prewarm predicted it from the
  // setting alone (on for every record), so every draw with an alpha test / gamma epilogue made its pipeline again in
  // every run (warm store 4421: 68 of a match's pipelines again). Older records (0) are still predicted.
  // (Review: only what the draw's latch actually worked out - a setting off leaves the bit unknown, 0.)
  const VulkanPipelineCache::NativeSpecLatch& spec = command_processor_.NativePipelineCache().NativeSpec();
  record.ext_bits = 0;
  if (spec.ps_epilogue) {
    record.ext_bits |= kRecordExtKnown;
    if (key.pixel_shader != VK_NULL_HANDLE && (draw.pixel_ext & SpirvShaderTranslator::kNativeExtEpilogueIdle)) {
      record.ext_bits |= kRecordExtEpilogueIdle;
    }
  }
  // (Review: a 0 texture-sign mask is a real per-draw value with the specialisation on - the prewarm keeps it.)
  if (spec.ps_texture && key.pixel_shader != VK_NULL_HANDLE) record.ext_bits |= kRecordExtTexMaskKnown;
  const uint64_t hash = XXH3_64bits(&record, sizeof(record));
  if (!storage_hashes_.insert(hash).second) return;  // stored (or checked) before
  // Only a record that makes this very key again is stored (its shaders and objects exist now: lookups only).
  PipelineKey again;
  const uint32_t draw_exts[2] = {key.pixel_shader != VK_NULL_HANDLE ? draw.pixel_ext : 0u, draw.vertex_ext};
  if (!KeyFromRecord(record, again, false, nullptr, draw_exts) || !(again == key)) {
    if (storage_mismatch_++ < 8) {
      REXGPU_WARN("[native][storage] vertex shader {:016X} pixel shader {:016X}: the record does not make the same "
                  "pipeline key (not stored)",
                  draw.vertex_shader_hash, draw.pixel_shader_hash);
    }
    return;
  }
  // The record's shaders into the shader storage too (the emulated path stores only its own pipelines' shaders).
  VulkanPipelineCache& cache = command_processor_.NativePipelineCache();
  if (VulkanShader* shader = cache.FindShader(record.vertex_shader_hash)) cache.StoreShader(*shader);
  if (record.pixel_shader_hash) {
    if (VulkanShader* shader = cache.FindShader(record.pixel_shader_hash)) cache.StoreShader(*shader);
  }
  // One write of the whole stored record (review: two writes may be split by the CRT's buffer flush).
  uint8_t stored[sizeof(hash) + sizeof(record)];
  std::memcpy(stored, &hash, sizeof(hash));
  std::memcpy(stored + sizeof(hash), &record, sizeof(record));
  fwrite(stored, sizeof(stored), 1, storage_file_);
  storage_dirty_ = true;
  if ((++storage_written_ & 63) == 0) REXGPU_INFO("[native][storage] {} new native pipelines stored", storage_written_);
}

// Session 70 (gpu_native_precompile): "made M of T, waiting W, done D, optimised O" into gpu_native_precompile_status
// for the game's precompile mode (src\precompile.h). Done = nothing queued, made or waiting (records still waiting
// after the 20 retry passes are given up: the game makes them when it meets them; the shader optimiser's queue
// empty), and the native renderer on - or not used at all. O = shaders the optimiser stored in this run.
void NativeRenderer::PrecompileStatus() {
  uint32_t jobs = 0, busy = 0;
  {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    jobs = uint32_t(pipeline_jobs_.size());
    busy = pipeline_busy_;
  }
  // (Session 70: and the shader optimiser's queue - vulkan_spirv_optimize_collect - must be empty.)
  const size_t waiting = (prewarm_.size() - std::min(prewarm_next_, prewarm_.size())) +
                         (prewarm_retries_ < 20 ? prewarm_retry_.size() : 0) + spirv_opt_store::Pending() +
                         command_processor_.NativePipelineCache().M6PrecompileWaiting();  // (Session 92, M6: the variants to translate)
  // (Review: with gpu_native set but the renderer off - it refused this PC or a setting, or its objects failed - the
  // stored records are never made: done once 600 frames show it off.)
  const bool native_off = REXCVAR_GET(gpu_native) == 0 || (!enabled_ && frame_ >= 600);
  const bool done = native_off ? frame_ >= 30 && spirv_opt_store::Pending() == 0
                               : enabled_ && frame_ >= 30 && waiting == 0 && jobs == 0 && busy == 0;
  // (Session 93, M6 step 2: ", m6 D of T" after the old fields - the game's reader (precompile.h) scans the first five numbers
  // by prefix, so the extra tail changes nothing for it: the variants translated of those queued.)
  uint64_t m6_done = 0, m6_total = 0;
  command_processor_.NativePipelineCache().M6PrecompileProgress(m6_done, m6_total);
  std::string status = fmt::format("made {} of {}, waiting {}, done {}, optimised {}, m6 {} of {}",
                                   prewarm_made_.load(std::memory_order_relaxed), prewarm_queued_, waiting,
                                   done ? 1 : 0, spirv_opt_store::Collected(), m6_done, m6_total);
  if (status != precompile_status_) {
    precompile_status_ = status;
    rex::cvar::SetFlagByName("gpu_native_precompile_status", status);
  }
}

void NativeRenderer::StorageOnFrameEnd() {
  if (storage_dirty_ && storage_file_) {
    fflush(storage_file_);
    storage_dirty_ = false;
  }
  // Session 70 (gpu_native_precompile): everything at once, retries every 10 frames, the progress for the game.
  const bool precompile = REXCVAR_GET(gpu_native_precompile);
  const int32_t budget_ms = precompile ? 2000 : REXCVAR_GET(gpu_native_pipeline_prewarm_ms);
  // Records whose shaders were not loaded yet (the D3D library's own, met in the first frames, are not in the shader
  // list) are tried again every 120 frames, 20 times at most.
  // Session 73: then on, every second of wall time for the whole run (the game loads a scene's shaders when it loads
  // the scene: the entrance's pipelines can only be made from its load on).
  const auto now = std::chrono::steady_clock::now();
  const bool retry_due = prewarm_retries_ < 20 ? frame_ % (precompile ? 10 : 120) == 0
                                               : now - prewarm_retry_time_ >= std::chrono::seconds(1);
  if (prewarm_next_ >= prewarm_.size() && !prewarm_retry_.empty() && retry_due) {
    ++prewarm_retries_;
    prewarm_retry_time_ = now;
    prewarm_.swap(prewarm_retry_);
    prewarm_retry_.clear();
    prewarm_next_ = 0;
    prewarm_wait_shader_ = prewarm_wait_twin_ = prewarm_wait_pass_ = 0;
  }
  // Session 92 (M6): the precompile translates the .xvar variants into the translation store, before its pipelines.
  if (precompile) command_processor_.NativePipelineCache().M6PrecompileStep(frame_, 2000);
  if (precompile) PrecompileStatus();
  if (!enabled_ || budget_ms <= 0 || prewarm_next_ >= prewarm_.size()) return;
  VulkanPipelineCache& cache = command_processor_.NativePipelineCache();
  const auto t0 = std::chrono::steady_clock::now();
  const float spec_values[3] = {float(target_height_) * 1.2f, float(target_width_) * 0.5f,
                                float(target_height_) * 0.5f};
  // (Review: the budget is checked before each record too - one record may translate up to four shaders.)
  while (prewarm_next_ < prewarm_.size() &&
         std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(budget_ms)) {
    const PipelineRecord& record = prewarm_[prewarm_next_++];
    // Waits (the retry passes) for: its shaders not loaded yet; a draw-twin translation outside truth frames (the
    // truth draw ratio is decided at the first twin translation - review: not before truth frames are on); its post
    // class's render pass not made yet (made with the post objects, before this in OnFrameEnd).
    const bool post = (record.variant & 0xF) != 0;
    const uint32_t post_class = post ? (record.variant & 7) - 1 : 0;
    const bool wait_shader = !cache.FindShader(record.vertex_shader_hash) ||
                             (record.pixel_shader_hash && !cache.FindShader(record.pixel_shader_hash));
    const bool wait_twin = (record.flags & kRecordTwin) && !truth();
    const bool wait_pass =
        (post && post_class < kPostTargetClasses && post_render_pass_continue_[post_class] == VK_NULL_HANDLE) ||
        (!post && render_pass_load_ == VK_NULL_HANDLE);
    if (wait_shader || wait_twin || wait_pass) {
      prewarm_wait_shader_ += wait_shader ? 1 : 0;  // Session 73: why
      prewarm_wait_twin_ += wait_twin ? 1 : 0;
      prewarm_wait_pass_ += wait_pass ? 1 : 0;
      prewarm_retry_.push_back(record);
      continue;
    }
    // Session 69: a record from before the epilogue bit was stored (ext_bits 0) with the epilogue specialisation on -
    // both variants (the prediction gave every one the bit; the draws without it made theirs again in every run).
    // Session 70: an older record whose known record of the same draw is in the file is left to that record. (Review
    // considered: per variant would keep the never-used other variant of every such pair in each warm-up - most of the
    // doubling. An older record standing for two draws with different epilogue bits is rare; its other variant is then
    // made once on first use and stored. Session 70: 156 of 792 guessed variants were used in a Brock / Bryan session.)
    uint8_t known = 0;
    if (!(record.ext_bits & kRecordExtKnown)) {
      if (auto s = storage_known_siblings_.find(XXH3_64bits(&record, sizeof(record))); s != storage_known_siblings_.end())
        known = s->second;
      // wwe2k14 patch (Session 80, HD 520 arc first launch): an older record from before Session 66's encodings (the
      // shipped list's: Session 61) never equals its known record as stored, so both its guesses were made beside the
      // known one (the precompile: 761 pipelines from 399 records). Compared again with its modifications converted as
      // KeyFromRecord converts them: the same variant as a known record = that record stands for it. (A variant only
      // the old record had - another texture-sign mask, another byte order - still differs, and is still made.)
      if (!known && !storage_known_siblings_.empty()) {
        VulkanShader* vertex_shader = cache.FindShader(record.vertex_shader_hash);
        VulkanShader* pixel_shader = record.pixel_shader_hash ? cache.FindShader(record.pixel_shader_hash) : nullptr;
        if (vertex_shader) {
          PipelineRecord conformed = record;
          bool changed = false;
          conformed.vertex_modification =
              cache.ConformStoredVertexModification(*vertex_shader, record.vertex_modification, &changed, true);
          if (pixel_shader && (record.flags & kRecordNativePixel)) {
            conformed.pixel_modification =
                cache.ConformStoredNativePixelModification(*pixel_shader, record.pixel_modification, &changed);
          }
          if (pixel_shader && shader_swap::WidenForReplacement(record.pixel_shader_hash, conformed.vertex_modification,
                                                               conformed.pixel_modification)) {
            changed = true;  // (Session 90: as KeyFromRecord)
          }
          if (changed) {
            if (auto s = storage_known_siblings_.find(XXH3_64bits(&conformed, sizeof(conformed)));
                s != storage_known_siblings_.end()) {
              known = s->second;
              ++prewarm_old_conformed_;
            }
          }
        }
      }
    }
    PipelineRecord variants[2] = {record, record};
    const bool both = !(record.ext_bits & kRecordExtKnown) && (record.flags & kRecordNativePixel) &&
                      record.pixel_shader_hash &&
                      (cache.PredictNativePixelExt() & SpirvShaderTranslator::kNativeExtEpilogueIdle);
    if (known) {
      ++prewarm_old_skipped_;
      continue;
    }
    if (both) variants[1].ext_bits |= kRecordExtKnown;  // (known: no epilogue bit)
    for (uint32_t v = 0; v < (both ? 2u : 1u); ++v) {
      PipelineKey key;
      bool converted = false;
      if ((post && post_class >= kPostTargetClasses) || !KeyFromRecord(variants[v], key, true, &converted, nullptr)) {
        ++prewarm_skipped_;
      } else if (!pipelines_.count(key)) {
        prewarm_converted_ += converted ? 1 : 0;
        // (Its shaders into the shader storage: a record from before StoreShader may name shaders it lacks.)
        cache.StoreShader(*cache.FindShader(record.vertex_shader_hash));
        if (record.pixel_shader_hash) cache.StoreShader(*cache.FindShader(record.pixel_shader_hash));
        std::lock_guard<std::mutex> lock(pipeline_mutex_);
        if (pipeline_pending_.insert(key).second) { REXCVAR_SET(gpu_diag_pipelines_pending, int32_t(pipeline_pending_.size()));
          StartPipelineWorkersLocked();
          Draw draw;  // (CreatePipeline reads the key only)
          draw.vertex_shader = key.vertex_shader;
          draw.layout = key.layout;
          pipeline_jobs_.push_back({key, draw, {spec_values[0], spec_values[1], spec_values[2]}, true});
          pipeline_cv_.notify_one();
          ++prewarm_queued_;
          if (both) prewarm_guessed_.insert(key);  // Session 70: the draw that uses it stores its own record
        }
      }
    }
  }
  // (Logged after the first pass, and after the last retry pass or once nothing waits any more.)
  if (prewarm_next_ >= prewarm_.size() && prewarm_retries_ != prewarm_logged_retries_ + (prewarm_logged_ ? 0u : 1u) &&
      (!prewarm_logged_ || prewarm_retry_.empty() || prewarm_retries_ >= 20)) {
    prewarm_logged_ = true;
    prewarm_logged_retries_ = prewarm_retries_;
    REXGPU_INFO("[native][storage] stored pipelines queued: {} in all (frame {}, pass {}); {} waiting for their "
                "shaders / truth frames / a render pass, {} skipped (a translation or object failed); converted to "
                "the current specialisation settings: {}; older records left out for a known one: {} ({} after "
                "converting their modifications), guessed variants queued: {}",
                prewarm_queued_, frame_, prewarm_retries_, prewarm_retry_.size(), prewarm_skipped_,
                prewarm_converted_, prewarm_old_skipped_, prewarm_old_conformed_, prewarm_guessed_.size());
  }
  // Session 73: the later passes that made something (the scene loads), with why the rest still wait.
  if (prewarm_next_ >= prewarm_.size() && prewarm_retries_ > 20 &&
      (prewarm_queued_ != prewarm_queued_logged_ || !prewarm_reasons_logged_)) {
    prewarm_reasons_logged_ = true;
    prewarm_queued_logged_ = prewarm_queued_;
    REXGPU_INFO("[native][storage] stored pipelines queued: {} in all (frame {}, pass {}); {} still waiting: shaders "
                "not loaded {}, truth frames {}, render pass {}",
                prewarm_queued_, frame_, prewarm_retries_, prewarm_retry_.size(), prewarm_wait_shader_,
                prewarm_wait_twin_, prewarm_wait_pass_);
  }
}

VkPipeline NativeRenderer::CreatePipeline(const Draw& draw, const PipelineKey& key, const float spec_values[3]) const {
  {  // wwe2k14 patch (Session 93, idea 7, log only): the collapse count, see EdsNativeCount
    const auto bytes = [](const PipelineKey& k, uint32_t state_mask, uint32_t variant_mask) {
      const uint64_t f[8] = {uint64_t(reinterpret_cast<uintptr_t>(k.vertex_shader)), uint64_t(reinterpret_cast<uintptr_t>(k.pixel_shader)),
                             uint64_t(reinterpret_cast<uintptr_t>(k.layout)), uint64_t(k.state & state_mask) | (uint64_t(k.blend_control) << 32),
                             k.tint_hash, uint64_t(reinterpret_cast<uintptr_t>(k.geometry_shader)), uint64_t(k.variant & variant_mask), 0};
      return std::string(reinterpret_cast<const char*>(f), sizeof(f));
    };
    // (state bits 5..12 = cull, front face, depth test / write / compare; variant bits 4+ = a truth post draw's stencil ops)
    const std::string full_key = bytes(key, ~0u, ~0u);
    const std::string collapsed_key = bytes(key, ~(0xFFu << 5), 0xFu);
    const int p = g_eds_native_prewarm ? 1 : 0;
    std::lock_guard<std::mutex> lock(g_eds_native.lock);
    if (g_eds_native.full[p].insert(full_key).second) {
      ++g_eds_native.pipelines[p];
      if (!g_eds_native.collapsed[p].insert(collapsed_key).second) ++g_eds_native.would_collapse[p];
    }
  }
  // Session 55: gpu_native_truth_debug_pipeline_delay_ms - on the worker and on the command thread alike (review).
  if (const int32_t delay = REXCVAR_GET(gpu_native_truth_debug_pipeline_delay_ms); delay > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
  }
  // (Session 47: what GetPipeline decided, from its key - the worker thread reads no state that changes.)
  // Session 61 (pipeline storage): everything from the key - the raster state from its RasterKey bits, post / truth /
  // the post class / the truth stencil from its variant (GetPipeline: 1 + class + 8 truth, the stencil ops above bit
  // 4) - so a stored record makes the same pipeline without the draw (the same values the draw gave: by construction).
  (void)draw;
  const VkShaderModule pixel_shader = key.pixel_shader;
  const uint32_t color_mask = (key.state >> 16) & 0xF;
  const uint32_t blend = key.blend_control;
  const bool post = (key.variant & 0xF) != 0;
  const bool truth = post && (key.variant & 8) != 0;
  const uint32_t post_class = post ? (key.variant & 7) - 1 : 0;
  const bool truth_stencil = post && truth && ((key.variant >> 4) & 1) != 0;
  const uint32_t stencil_ops = truth_stencil ? key.variant >> 5 : 0;
  const VkPrimitiveTopology topology = VkPrimitiveTopology(key.state & 0xF);
  const bool primitive_restart = (key.state >> 4) & 1;
  const VkCullModeFlags cull_mode = (key.state >> 5) & 3;
  const bool front_face_clockwise = (key.state >> 7) & 1;
  const bool depth_test = (key.state >> 8) & 1;
  const bool depth_write = (key.state >> 9) & 1;
  const VkCompareOp depth_compare = VkCompareOp((key.state >> 10) & 7);
  const VkShaderModule geometry_shader = key.geometry_shader;
  if (geometry_shader != VK_NULL_HANDLE) ++g_gs_pipelines;  // Session 94 (nogs): must stay 0 with gpu_native_vs_expand
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();

  struct Specialization {
    float tint[3];
    float focal_pixels, center_x, center_y;
  } spec;
  for (int i = 0; i < 3; ++i) {
    spec.tint[i] = 0.5f + 0.45f * float((key.tint_hash >> (i * 16)) & 0xFFFF) / 65535.0f;
  }
  spec.focal_pixels = spec_values[0];
  spec.center_x = spec_values[1];
  spec.center_y = spec_values[2];
  VkSpecializationMapEntry entries[6];
  for (uint32_t i = 0; i < 6; ++i) entries[i] = {i, i * uint32_t(sizeof(float)), sizeof(float)};
  VkSpecializationInfo spec_info = {6, entries, sizeof(spec), &spec};

  VkPipelineShaderStageCreateInfo stages[3] = {};
  uint32_t stage_count = 0;
  stages[stage_count++] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,
                           key.vertex_shader, "main", nullptr};
  if (geometry_shader != VK_NULL_HANDLE) {
    stages[stage_count++] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                             VK_SHADER_STAGE_GEOMETRY_BIT, geometry_shader, "main", nullptr};
  }
  stages[stage_count++] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                           VK_SHADER_STAGE_FRAGMENT_BIT, pixel_shader ? pixel_shader : clay_fragment_shader_, "main",
                           pixel_shader ? nullptr : &spec_info};
  // The guest vertex shader fetches its vertices from the shared memory buffer itself: no vertex input.
  VkPipelineVertexInputStateCreateInfo vertex_input = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  // Session 95: an attribute draw's bindings and attributes, from the key alone (the workers read no shared table): the rule is vertex_input_layout.h's. The attributes may cover more of the vertex
  // than the shader reads (the shader declares only what it reads: legal Vulkan). Guest-order bytes: the shader swaps them.
  VkVertexInputBindingDescription vi_bindings[kViMaxBindings];
  VkVertexInputAttributeDescription vi_attributes[kViMaxAttributes];
  if (key.vi) {
    ViAttr attrs[kViMaxAttributes];
    const uint32_t attr_count = ViAttributes(key.vi, attrs);
    uint32_t binding_count = 0;
    while (binding_count < kViMaxBindings && ((key.vi >> (8 * binding_count)) & 0xFFu)) {
      const uint32_t vi_byte = (key.vi >> (8 * binding_count)) & 0xFFu;  // (Session 95, P7: bit 7 = per instance, own crowd's members)
      vi_bindings[binding_count] = {binding_count, (vi_byte & 0x7Fu) * 4, (vi_byte & kViInstanceRate) ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX};
      ++binding_count;
    }
    for (uint32_t i = 0; i < attr_count; ++i) {
      const VkFormat format = attrs[i].words == 4 ? VK_FORMAT_R32G32B32A32_UINT : attrs[i].words == 2 ? VK_FORMAT_R32G32_UINT : VK_FORMAT_R32_UINT;
      vi_attributes[i] = {attrs[i].location, attrs[i].binding, format, uint32_t(attrs[i].first_word) * 4};
    }
    vertex_input.vertexBindingDescriptionCount = binding_count;
    vertex_input.pVertexBindingDescriptions = vi_bindings;
    vertex_input.vertexAttributeDescriptionCount = attr_count;
    vertex_input.pVertexAttributeDescriptions = vi_attributes;
  }
  VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  input_assembly.topology = topology;
  input_assembly.primitiveRestartEnable = primitive_restart ? VK_TRUE : VK_FALSE;
  VkPipelineViewportStateCreateInfo viewport = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewport.viewportCount = 1;
  viewport.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo raster = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode = cull_mode;
  raster.frontFace = front_face_clockwise ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
  raster.depthBiasEnable = (key.state >> 13) & 1 ? VK_TRUE : VK_FALSE;
  // Session 69: the guest's clipping off -> depth clamped, not clipped (RasterKey bit 14; as the emulated pipeline).
  raster.depthClampEnable =
      ((key.state >> 14) & 1) && command_processor_.GetVulkanDevice()->properties().depthClamp ? VK_TRUE : VK_FALSE;
  raster.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo multisample = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo depth = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  depth.depthTestEnable = depth_test ? VK_TRUE : VK_FALSE;
  depth.depthWriteEnable = depth_write ? VK_TRUE : VK_FALSE;
  depth.depthCompareOp = depth_compare;
  if (post) {
    // Session 42: every pixel a post draw writes gets stencil bit 0 (the write-back's coverage). A discarded pixel,
    // or (Session 43, gpu_native_post_depth) one that fails the depth test, writes no stencil; a draw that writes
    // depth also sets bit 1 (the write-back writes that pixel's depth). The depth test is the draw's own (20e4 values
    // in the target, the shader's depth rounded to 20e4: exact); post draws have no polygon offset (claimed only so).
    depth.depthTestEnable = depth_test ? VK_TRUE : VK_FALSE;
    depth.depthWriteEnable = depth_write ? VK_TRUE : VK_FALSE;
    depth.depthCompareOp = depth_compare;
    depth.stencilTestEnable = VK_TRUE;
    const uint32_t stencil_write = 1u | (depth_write ? 2u : 0u);
    depth.front = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS, 0xFF,
                   stencil_write, 3};
    depth.back = depth.front;
    // Session 46: a truth draw keeps its polygon offset (not exact: the host's bias in a float target) and writes no
    // coverage marks (nothing is written back).
    if (!truth) raster.depthBiasEnable = VK_FALSE;
    if (truth) depth.stencilTestEnable = VK_FALSE;
    // Session 46: a truth draw's own stencil test (the guest's ops; reference and masks dynamic).
    if (truth_stencil) {
      depth.stencilTestEnable = VK_TRUE;
      const auto face = [](uint32_t ops) {
        return VkStencilOpState{VkStencilOp(ops & 7), VkStencilOp((ops >> 3) & 7), VkStencilOp((ops >> 6) & 7),
                                VkCompareOp((ops >> 9) & 7), 0xFF, 0xFF, 0};
      };
      depth.front = face(stencil_ops & 0xFFF);
      depth.back = face(stencil_ops >> 12);
    }
  }
  // The colour write mask's bits are R, G, B, A in both APIs.
  VkPipelineColorBlendAttachmentState blend_attachment = {};
  blend_attachment.colorWriteMask = color_mask;
  if (blend) {
    reg::RB_BLENDCONTROL blend_control;
    blend_control.value = blend;
    blend_attachment.blendEnable = VK_TRUE;
    blend_attachment.srcColorBlendFactor = BlendFactor(blend_control.color_srcblend);
    blend_attachment.dstColorBlendFactor = BlendFactor(blend_control.color_destblend);
    blend_attachment.colorBlendOp = BlendOp(blend_control.color_comb_fcn);
    blend_attachment.srcAlphaBlendFactor = BlendFactor(blend_control.alpha_srcblend);
    blend_attachment.dstAlphaBlendFactor = BlendFactor(blend_control.alpha_destblend);
    blend_attachment.alphaBlendOp = BlendOp(blend_control.alpha_comb_fcn);
  }
  VkPipelineColorBlendStateCreateInfo blend_state = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  blend_state.attachmentCount = 1;
  blend_state.pAttachments = &blend_attachment;
  VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT,           VK_DYNAMIC_STATE_SCISSOR,
                                     VK_DYNAMIC_STATE_DEPTH_BIAS,         VK_DYNAMIC_STATE_BLEND_CONSTANTS,
                                     VK_DYNAMIC_STATE_STENCIL_REFERENCE,  VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
                                     VK_DYNAMIC_STATE_STENCIL_WRITE_MASK};
  VkPipelineDynamicStateCreateInfo dynamic = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic.dynamicStateCount = post && truth_stencil ? 7 : 4;  // Session 46: truth stencil
  dynamic.pDynamicStates = dynamic_states;
  VkGraphicsPipelineCreateInfo info = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  info.stageCount = stage_count;
  info.pStages = stages;
  info.pVertexInputState = &vertex_input;
  info.pInputAssemblyState = &input_assembly;
  info.pViewportState = &viewport;
  info.pRasterizationState = &raster;
  info.pMultisampleState = &multisample;
  info.pDepthStencilState = &depth;
  info.pColorBlendState = &blend_state;
  info.pDynamicState = &dynamic;
  info.layout = key.layout;
  // wwe2k14 patch (Session 73): made with the render pass the truth passes begin (no external dependencies,
  // gpu_native_truth_lazy_pass_deps, NATIVE's default): render pass compatibility includes the dependencies, so pipelines
  // made with the 2-dependency pass were incompatible with every truth pass (validation VUID-vkCmdDraw-renderPass-02684).
  // The rare post segments (and truth passes with the switch off) are the ones that differ now.
  info.renderPass = post ? (truth_render_pass_keep_nodep_[post_class] != VK_NULL_HANDLE
                                ? truth_render_pass_keep_nodep_[post_class]
                                : post_render_pass_continue_[post_class])
                         : render_pass_load_;
  VkPipeline pipeline = VK_NULL_HANDLE;
  const auto s58_t0 = std::chrono::steady_clock::now();  // wwe2k14 patch (Session 58): s58_stats.h
  // Session 63: the on-disk pipeline cache (vulkan_disk_pipeline_cache; VK_NULL_HANDLE when off).
  if (dfn.vkCreateGraphicsPipelines(vulkan_device->device(), disk_pipeline_cache::Get(), 1, &info, nullptr,
                                    &pipeline) != VK_SUCCESS) {
    pipeline = VK_NULL_HANDLE;
  } else {
    disk_pipeline_cache::NoteCreated();
  }
  s58::g_native_pipelines.Add(s58_t0);
  return pipeline;
}

// ---------------------------------------------------------------------------------------------------------------
// Replacing (gpu_native 2): the load and the write-back
// ---------------------------------------------------------------------------------------------------------------

bool NativeRenderer::EnsureReplaceObjects() {
  if (writeback_pipeline_ != VK_NULL_HANDLE) return true;
  if (replace_objects_failed_ || !EnsureObjects()) return false;
  replace_objects_failed_ = true;  // until everything below exists
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  auto compile = [&](VkShaderStageFlagBits stage, const char* source, VkShaderModule& out) {
    std::vector<uint32_t> spirv;
    std::string error;
    if (!command_processor_.CompileGlslToSpirv(stage, source, spirv, error)) {
      REXGPU_ERROR("[native] replace shader compilation failed: {}", error);
      return false;
    }
    out = ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
    return out != VK_NULL_HANDLE;
  };
  if (!compile(VK_SHADER_STAGE_VERTEX_BIT, kLoadVertexShader, load_vertex_shader_) ||
      !compile(VK_SHADER_STAGE_FRAGMENT_BIT, kLoadFragmentShader, load_fragment_shader_) ||
      !compile(VK_SHADER_STAGE_COMPUTE_BIT, kWriteBackComputeShader, writeback_shader_)) {
    return false;
  }
  // Layouts: the load reads the EDRAM in its fragment shader; the write-back writes it and samples our targets.
  {
    VkDescriptorSetLayoutBinding binding = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT,
                                            nullptr};
    VkDescriptorSetLayoutCreateInfo set_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 1;
    set_info.pBindings = &binding;
    if (dfn.vkCreateDescriptorSetLayout(device, &set_info, nullptr, &load_set_layout_) != VK_SUCCESS) return false;
    VkDescriptorSetLayoutBinding bindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    set_info.bindingCount = 3;
    set_info.pBindings = bindings;
    if (dfn.vkCreateDescriptorSetLayout(device, &set_info, nullptr, &writeback_set_layout_) != VK_SUCCESS) {
      return false;
    }
    VkPushConstantRange push = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(EdramPlace)};
    VkPipelineLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &load_set_layout_;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    if (dfn.vkCreatePipelineLayout(device, &layout_info, nullptr, &load_layout_) != VK_SUCCESS) return false;
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layout_info.pSetLayouts = &writeback_set_layout_;
    if (dfn.vkCreatePipelineLayout(device, &layout_info, nullptr, &writeback_layout_) != VK_SUCCESS) return false;
  }
  // Sets are freed one by one (a write-back set per target generation).
  {
    // Session 42: room for the post targets' write-back sets too.
    // Session 43 (review): 3 samplers per post write-back set.
    VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64},
                                     {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 192}};
    VkDescriptorPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 64;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes = sizes;
    if (dfn.vkCreateDescriptorPool(device, &pool_info, nullptr, &replace_descriptor_pool_) != VK_SUCCESS) {
      return false;
    }
  }
  // The load: a full-screen triangle writing every pixel's colour and depth (depth test always).
  {
    VkPipelineShaderStageCreateInfo stages[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,
         load_vertex_shader_, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT,
         load_fragment_shader_, "main", nullptr}};
    VkPipelineVertexInputStateCreateInfo vertex_input = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo input_assembly = {
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    VkPipelineColorBlendAttachmentState blend_attachment = {};
    blend_attachment.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo blend_state = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend_state.attachmentCount = 1;
    blend_state.pAttachments = &blend_attachment;
    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;
    VkGraphicsPipelineCreateInfo info = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertex_input;
    info.pInputAssemblyState = &input_assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &blend_state;
    info.pDynamicState = &dynamic;
    info.layout = load_layout_;
    info.renderPass = render_pass_load_;  // compatible with the clearing pass (same attachments)
    if (dfn.vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &load_pipeline_) != VK_SUCCESS) {
      load_pipeline_ = VK_NULL_HANDLE;
      return false;
    }
  }
  {
    VkComputePipelineCreateInfo info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                  writeback_shader_, "main", nullptr};
    info.layout = writeback_layout_;
    if (dfn.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &writeback_pipeline_) !=
        VK_SUCCESS) {
      writeback_pipeline_ = VK_NULL_HANDLE;
      return false;
    }
  }
  replace_objects_failed_ = false;
  REXGPU_INFO("[native] replace mode ready (load + write-back passes)");
  return true;
}

bool NativeRenderer::EnsureReplaceSets(bool writeback) {
  const VkBuffer edram = command_processor_.NativeEdramBuffer();
  if (edram == VK_NULL_HANDLE || (writeback && color_view_ == VK_NULL_HANDLE) || !EnsureReplaceObjects()) {
    return false;
  }
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  if (edram != replace_edram_buffer_) {
    // Both sets point at the EDRAM buffer: new ones (the old ones are freed once the GPU is done with them).
    for (VkDescriptorSet* set : {&load_set_, &writeback_set_}) {
      if (*set == VK_NULL_HANDLE) continue;
      Retired r;
      r.submission = command_processor_.GetCurrentSubmission();
      r.replace_set = *set;
      retired_.push_back(r);
      *set = VK_NULL_HANDLE;
    }
    replace_edram_buffer_ = edram;
  }
  auto allocate = [&](VkDescriptorSetLayout layout, VkDescriptorSet& set) {
    VkDescriptorSetAllocateInfo alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = replace_descriptor_pool_;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &layout;
    if (dfn.vkAllocateDescriptorSets(device, &alloc, &set) != VK_SUCCESS) {
      set = VK_NULL_HANDLE;
      REXGPU_ERROR("[native] replace descriptor set allocation failed");
      return false;
    }
    return true;
  };
  VkDescriptorBufferInfo edram_info = {edram, 0, VK_WHOLE_SIZE};
  VkWriteDescriptorSet writes[3] = {};
  for (uint32_t i = 0; i < 3; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = i ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  }
  writes[0].pBufferInfo = &edram_info;
  if (load_set_ == VK_NULL_HANDLE) {
    if (!allocate(load_set_layout_, load_set_)) return false;
    writes[0].dstSet = load_set_;
    dfn.vkUpdateDescriptorSets(device, 1, writes, 0, nullptr);
  }
  if (writeback && writeback_set_ == VK_NULL_HANDLE) {
    if (!allocate(writeback_set_layout_, writeback_set_)) return false;
    VkDescriptorImageInfo color_info = {sampler_nearest(), color_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo depth_info = {sampler_nearest(), depth_view_,
                                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    writes[1].pImageInfo = &color_info;
    writes[2].pImageInfo = &depth_info;
    for (uint32_t i = 0; i < 3; ++i) writes[i].dstSet = writeback_set_;
    dfn.vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
  }
  return true;
}

// The scene's surfaces as the targets were last sized for them (render target 0 and the depth surface of the scene
// key; the parity checks' addressing).
NativeRenderer::EdramPlace NativeRenderer::SceneEdramPlace() const {
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const uint32_t tile_w = 80 * scale_x, tile_h = 16 * scale_y;
  const uint32_t pitch_tiles = ((scene_key_.surface_info & 0x3FFF) + 79) / 80;
  EdramPlace place;
  place.color_base_dwords = (scene_color_info_ & 0xFFF) * tile_w * tile_h;
  place.depth_base_dwords = (scene_key_.depth_info & 0xFFF) * tile_w * tile_h;
  place.pitch_tile_dwords = pitch_tiles * tile_w * tile_h;
  place.tile_w = tile_w;
  place.tile_h = tile_h;
  place.width = std::min(target_scene_width_, pitch_tiles * tile_w);
  place.height = target_scene_height_;
  place.wrap_dwords = 2048 * tile_w * tile_h;
  place.flags = ((scene_key_.depth_info >> 16) & 1) |  // RB_DEPTH_INFO depth_format 1: 20e4
                (scene_color_info_valid_ ? 2u : 0u);     // a colour draw has given the colour surface
  place.scale_x = float(target_width_) / float(std::max(1u, target_scene_width_));
  place.scale_y = float(target_height_) / float(std::max(1u, target_scene_height_));
  place.origin_x = place.origin_y = 0;
  return place;
}

void NativeRenderer::WriteBack() {
  if (!replacing_ || !native_dirty_) return;
  native_dirty_ = false;
  // Session 40: what our draws changed since the last write-back (gpu_native_writeback_rect).
  const bool narrow = REXCVAR_GET(gpu_native_writeback_rect);
  const bool write_color = !narrow || dirty_color_, write_depth = !narrow || dirty_depth_;
  const int32_t rect_x0 = dirty_x0_, rect_y0 = dirty_y0_, rect_x1 = dirty_x1_, rect_y1 = dirty_y1_;
  ResetDirty();
  if (color_image_ == VK_NULL_HANDLE) return;
  if (!EnsureReplaceSets()) {
    REXGPU_ERROR("[native] no write-back: this frame's native draws are lost (frame {})", frame_);
    return;
  }
  FrameSlot& slot = slots_[SlotOf(frame_)];
  EdramPlace place = SceneEdramPlace();
  if (!place.width || !place.height) return;
  place.flags |= (write_color ? 4u : 0u) | (write_depth ? 8u : 0u);
  uint32_t dispatch_width = place.width, dispatch_height = place.height;
  if (narrow) {
    if (!write_color && !write_depth) return;  // nothing drawn changed the EDRAM's channels
    // The draws' scissors, in emulated host pixels, within the surface.
    const uint32_t x0 = uint32_t(std::clamp(rect_x0, 0, int32_t(place.width)));
    const uint32_t y0 = uint32_t(std::clamp(rect_y0, 0, int32_t(place.height)));
    const uint32_t x1 = uint32_t(std::clamp(rect_x1, int32_t(x0), int32_t(place.width)));
    const uint32_t y1 = uint32_t(std::clamp(rect_y1, int32_t(y0), int32_t(place.height)));
    if (x1 <= x0 || y1 <= y0) return;
    place.origin_x = x0;
    place.origin_y = y0;
    dispatch_width = x1 - x0;
    dispatch_height = y1 - y0;
  }
  // Session 45 (gpu_native_post_sync): the scene's write-back changes the EDRAM (its surfaces may alias post
  // surfaces anywhere: every post target reloads what it needs). Session 45 (gpu_native_defer_writeback): pending
  // post colour goes in before it (in order).
  if (defer_writeback_) WritePendingAll();
  if (post_sync_) InvalidateAllPostSync();
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const uint32_t query_base = SlotOf(frame_) * kQueriesPerFrame;
  FrameSlot::Segment* segment = nullptr;
  if (query_pool_ && slot_queries_ok_ && slot.queries + 2 <= QueryLimit()) {
    segment = &slot.segments.emplace_back();
    segment->kind = 1;  // Session 43
    segment->native_begin = slot.queries++;
    cb.CmdVkWriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, query_base + segment->native_begin);
  }
  // Every earlier user of the EDRAM -> this pass's reads and writes; our targets' writes -> its reads. Recorded
  // directly (not through the pending barriers; see Flush): the command processor's tracking of the EDRAM buffer
  // stays as it is, and the barrier after this pass covers whatever it does next.
  constexpr VkPipelineStageFlags kEdramStages =
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
  constexpr VkAccessFlags kEdramAccesses = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                           VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  VkMemoryBarrier edram_barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, kEdramAccesses,
                                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
  VkImageMemoryBarrier images[2] = {};
  for (int i = 0; i < 2; ++i) {
    images[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    images[i].srcQueueFamilyIndex = images[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    images[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  }
  images[0].image = color_image_;
  images[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  images[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  images[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  images[0].subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);
  images[1].image = depth_image_;
  images[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  images[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  images[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
  images[1].subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT);
  constexpr VkPipelineStageFlags kTargetStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                                 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  cb.CmdVkPipelineBarrier(kEdramStages | kTargetStages, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &edram_barrier, 0,
                          nullptr, 2, images);
  command_processor_.BindExternalComputePipeline(writeback_pipeline_);
  cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, writeback_layout_, 0, 1, &writeback_set_, 0, nullptr);
  cb.CmdVkPushConstants(writeback_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(place), &place);
  cb.CmdVkDispatch((dispatch_width + 7) / 8, (dispatch_height + 7) / 8, 1);
  // This pass's EDRAM writes -> everything after it (the resolve, the emulated draws, our next load); our targets
  // back to attachments for our next pass.
  edram_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  edram_barrier.dstAccessMask = kEdramAccesses;
  images[0].srcAccessMask = images[1].srcAccessMask = 0;
  images[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  images[1].dstAccessMask =
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  for (int i = 0; i < 2; ++i) std::swap(images[i].oldLayout, images[i].newLayout);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, kEdramStages | kTargetStages, 0, 1, &edram_barrier, 0,
                          nullptr, 2, images);
  if (segment) {
    segment->native_end = slot.queries++;
    cb.CmdVkWriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, query_base + segment->native_end);
  }
  ++slot.writebacks;
}

// ---------------------------------------------------------------------------------------------------------------
// Post-processing (Session 42, gpu_native_post): a target per post surface, claims after the scene's colour resolve,
// the coverage write-back
// ---------------------------------------------------------------------------------------------------------------

bool NativeRenderer::EnsurePostObjects() {
  if (post_writeback_pipeline_ != VK_NULL_HANDLE) return true;
  if (post_objects_failed_ || !EnsureReplaceObjects()) return false;
  post_objects_failed_ = true;  // until everything below exists
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const ui::vulkan::VulkanInstance::Functions& ifn = vulkan_device->vulkan_instance()->functions();
  const VkDevice device = vulkan_device->device();
  // A depth / stencil format whose stencil the write-back can sample.
  const VkFormatFeatureFlags ds_features =
      VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
  for (VkFormat format : {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT}) {
    VkFormatProperties properties;
    ifn.vkGetPhysicalDeviceFormatProperties(vulkan_device->physical_device(), format, &properties);
    if ((properties.optimalTilingFeatures & ds_features) == ds_features) {
      post_ds_format_ = format;
      break;
    }
  }
  VkFormatProperties color_properties;
  ifn.vkGetPhysicalDeviceFormatProperties(vulkan_device->physical_device(), kPostTargetFormat, &color_properties);
  const VkFormatFeatureFlags color_features = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
  // wwe2k14 patch (Session 93, report 033: a Mali-G720 r49p1 black screen under TRUE NATIVE). RGBA32F blending is optional in Vulkan and
  // Arm's driver has none (vulkan.gpuinfo.org report 52703, the same r49p1-03bet0 build: COLOR_ATTACHMENT yes, COLOR_ATTACHMENT_BLEND no). Only
  // the old post path's blended float-class (class 0) draws need it, and that path runs only in non-truth replacing frames, which need the
  // EDRAM buffer. With no EDRAM buffer (TRUE NATIVE, M2dEdramLess) nothing draws into class 0 - truth frames and the owned frame use RGBA8 /
  // RGBA16_SNORM + post_ds_format_ - so the blend is not required there; requiring it refused every truth frame -> truth() false -> TRUE
  // NATIVE's typed draws fell back ("not a truth frame") with no old path to fall back to: nothing drawn. With an EDRAM buffer (the
  // interlock path, default NATIVE anywhere) the requirement stays as it was.
  const bool float_blend =
      (color_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) != 0;
  const bool float_blend_needed = !M2dEdramLess();
  if (post_ds_format_ == VK_FORMAT_UNDEFINED ||
      (color_properties.optimalTilingFeatures & color_features) != color_features || (float_blend_needed && !float_blend)) {
    REXGPU_WARN("[native][post] post stays emulated: {}",
                post_ds_format_ == VK_FORMAT_UNDEFINED ? "no sampled-stencil depth format (D32S8 / D24S8)"
                : (color_properties.optimalTilingFeatures & color_features) != color_features
                    ? "R32G32B32A32_SFLOAT is not a sampled colour target"
                    : "R32G32B32A32_SFLOAT cannot blend (needed with an EDRAM buffer: the old post path's blended float targets)");
    return false;
  }
  if (!float_blend) {
    REXGPU_INFO("[native][post] R32G32B32A32_SFLOAT cannot blend on this device: not needed without an EDRAM buffer (TRUE NATIVE draws "
                "into RGBA8 / RGBA16_SNORM only) - the post objects are made without it");
  }
  // Session 43: the quantized classes (no blending needed in RGBA16_SNORM: 16-bit post draws never blend).
  post_class_ok_[0] = true;
  for (uint32_t cls = 1; cls < kPostTargetClasses; ++cls) {
    VkFormatProperties properties;
    ifn.vkGetPhysicalDeviceFormatProperties(vulkan_device->physical_device(), kPostTargetFormats[cls], &properties);
    const VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                        (cls == 1 ? VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT : 0);
    post_class_ok_[cls] = (properties.optimalTilingFeatures & needed) == needed;
  }
  // Two render passes that differ only in their load operations: a segment's first pass starts from the stencil
  // cleared to 0 (and its colour undefined: only covered or loaded pixels are ever written back); later passes of
  // the segment load both.
  for (uint32_t cls = 0; cls < kPostTargetClasses; ++cls) {
  if (!post_class_ok_[cls]) continue;
  // Session 44 (gpu_native_post_lean_depth): variants 2-3 are for segments without depth (the depth aspect neither
  // loaded nor stored; the stencil's coverage marks as always).
  // Session 45 (gpu_native_post_sync): variant 4 begins a segment keeping colour and depth (loaded, the stencil
  // cleared; the images must already be in the attachment layouts).
  // Session 46: variant 5 is variant 4 with the stencil loaded (truth frames).
  for (int variant = 0; variant < 6; ++variant) {
    const bool truth_keep = variant == 5;
    const bool keep = variant >= 4;
    const bool begin = ((variant & 1) != 0 || keep) && !truth_keep, no_depth = (variant & 2) != 0 && !keep;
    const bool discard = begin && !keep;
    VkAttachmentDescription attachments[2] = {};
    attachments[0].format = kPostTargetFormats[cls];
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = discard ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = discard ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[1].format = post_ds_format_;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = discard || no_depth ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_LOAD;
    // Session 43: depth is kept from pass to pass of a segment and read by the write-back (gpu_native_post_depth).
    attachments[1].storeOp = no_depth ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = begin ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].initialLayout =
        discard ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference color_ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference ds_ref = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;
    subpass.pDepthStencilAttachment = &ds_ref;
    // Our earlier passes and the write-back's reads (compute) come before; the write-back after.
    const VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    const VkAccessFlags accesses = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    VkSubpassDependency dependencies[2] = {};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = stages | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    dependencies[0].dstStageMask = stages;
    dependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dstAccessMask = accesses;
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = stages;
    dependencies[1].dstStageMask = stages | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = accesses | VK_ACCESS_SHADER_READ_BIT;
    VkRenderPassCreateInfo info = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    info.attachmentCount = 2;
    info.pAttachments = attachments;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = 2;
    info.pDependencies = dependencies;
    // Session 94 (gpu_d3d_decouple): the passes the owned draws' pipelines are made with carry the owned passes' view mask (render pass compatibility).
    VkRenderPassMultiviewCreateInfo decouple_mv = {};
    DecoupleRenderPass(info, decouple_mv);
    VkRenderPass* const render_pass =
        truth_keep ? &truth_render_pass_keep_[cls]
        : keep     ? &post_render_pass_begin_keep_[cls]
        : no_depth ? (begin ? &post_render_pass_begin_no_depth_[cls] : &post_render_pass_continue_no_depth_[cls])
                   : (begin ? &post_render_pass_begin_[cls] : &post_render_pass_continue_[cls]);
    if (dfn.vkCreateRenderPass(device, &info, nullptr, render_pass) != VK_SUCCESS) {
      REXGPU_ERROR("[native][post] render pass creation failed");
      return false;
    }
    // Session 56: the truth passes' two, also without the external dependencies (compatible: the same attachments).
    if (truth_keep || (no_depth && !begin)) {
      info.dependencyCount = 0;
      info.pDependencies = nullptr;
      if (dfn.vkCreateRenderPass(device, &info, nullptr,
                                 truth_keep ? &truth_render_pass_keep_nodep_[cls]
                                            : &truth_render_pass_no_depth_nodep_[cls]) != VK_SUCCESS) {
        REXGPU_ERROR("[native][post] render pass creation failed");
        return false;
      }
      // Session 73 (gpu_native_truth_color_none): the same with the colour kept as it is, not loaded or stored.
      if (vulkan_device->extensions().ext_EXT_load_store_op_none) {
        VkAttachmentDescription none_attachments[2] = {attachments[0], attachments[1]};
        none_attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_NONE_EXT;
        none_attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_NONE_EXT;
        info.pAttachments = none_attachments;
        if (dfn.vkCreateRenderPass(device, &info, nullptr, &truth_render_pass_color_none_[truth_keep ? 0 : 1][cls]) !=
            VK_SUCCESS) {
          truth_render_pass_color_none_[truth_keep ? 0 : 1][cls] = VK_NULL_HANDLE;
        }
        info.pAttachments = attachments;
      }
    }
    // Session 66 (G2.1, gpu_native_truth_pass_dont_care): the truth pass without dependencies, its colour and / or
    // depth + stencil not loaded (the pass's first draw writes every pixel of them; layouts as the keep pass's).
    if (truth_keep) {
      for (int dc = 0; dc < 3; ++dc) {
        VkAttachmentDescription dc_attachments[2] = {attachments[0], attachments[1]};
        if (dc != 1) dc_attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        if (dc != 0) {
          dc_attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
          dc_attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        }
        info.pAttachments = dc_attachments;
        if (dfn.vkCreateRenderPass(device, &info, nullptr, &truth_render_pass_dont_care_[dc][cls]) != VK_SUCCESS) {
          REXGPU_ERROR("[native][post] render pass creation failed");
          return false;
        }
        // Session 73: depth not loaded, colour neither loaded nor stored.
        if (dc == 1 && vulkan_device->extensions().ext_EXT_load_store_op_none) {
          dc_attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_NONE_EXT;
          dc_attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_NONE_EXT;
          if (dfn.vkCreateRenderPass(device, &info, nullptr, &truth_render_pass_color_none_[2][cls]) != VK_SUCCESS) {
            truth_render_pass_color_none_[2][cls] = VK_NULL_HANDLE;
          }
        }
      }
      // Session 80 (gpu_native_truth_clear_load): colour, depth and stencil cleared by the pass's start (a GPU's fast
      // clear where it has one) in place of the first draw's whole-surface clear.
      {
        VkAttachmentDescription clear_attachments[2] = {attachments[0], attachments[1]};
        clear_attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        clear_attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        clear_attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        info.pAttachments = clear_attachments;
        if (dfn.vkCreateRenderPass(device, &info, nullptr, &truth_render_pass_clear_[cls]) != VK_SUCCESS) {
          truth_render_pass_clear_[cls] = VK_NULL_HANDLE;
        }
      }
      info.pAttachments = attachments;
    }
  }
  }
  // The load (a segment that draws over what is under it): the scene's load shaders, colour only (no depth test, no
  // stencil: loaded pixels are not covered, the write-back leaves them as the EDRAM has them).
  {
    VkPipelineShaderStageCreateInfo stages[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,
         load_vertex_shader_, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT,
         load_fragment_shader_, "main", nullptr}};
    VkPipelineVertexInputStateCreateInfo vertex_input = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo input_assembly = {
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState blend_attachment = {};
    blend_attachment.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo blend_state = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend_state.attachmentCount = 1;
    blend_state.pAttachments = &blend_attachment;
    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;
    VkGraphicsPipelineCreateInfo info = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertex_input;
    info.pInputAssemblyState = &input_assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &blend_state;
    info.pDynamicState = &dynamic;
    info.layout = load_layout_;
    // Session 43: three variants per colour class - colour only (Session 42), depth only, both. The load shader writes gl_FragDepth
    // (the depth surface's 20e4 values as exact float32); the stencil (coverage) is not touched.
    struct Variant {
      uint32_t color_mask;
      bool depth;
      VkPipeline* out;
    };
    for (uint32_t cls = 0; cls < kPostTargetClasses; ++cls) {
    if (!post_class_ok_[cls]) continue;
    info.renderPass = truth_render_pass_keep_nodep_[cls] != VK_NULL_HANDLE ? truth_render_pass_keep_nodep_[cls]
                                                                            : post_render_pass_continue_[cls];  // (Session 73)
    const Variant variants[3] = {{0xF, false, &post_load_pipeline_[cls]},
                                 {0x0, true, &post_load_depth_pipeline_[cls]},
                                 {0xF, true, &post_load_both_pipeline_[cls]}};
    for (const Variant& v : variants) {
      blend_attachment.colorWriteMask = v.color_mask;
      depth.depthTestEnable = v.depth ? VK_TRUE : VK_FALSE;
      depth.depthWriteEnable = v.depth ? VK_TRUE : VK_FALSE;
      depth.depthCompareOp = VK_COMPARE_OP_ALWAYS;
      if (dfn.vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, v.out) != VK_SUCCESS) {
        *v.out = VK_NULL_HANDLE;
        return false;
      }
    }
    }
  }
  // The coverage write-back.
  {
    std::vector<uint32_t> spirv;
    std::string error;
    if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT, kPostWriteBackComputeShader, spirv,
                                               error)) {
      REXGPU_ERROR("[native][post] write-back shader compilation failed: {}", error);
      return false;
    }
    post_writeback_shader_ =
        ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
    VkDescriptorSetLayoutBinding bindings[4] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};  // Session 43
    VkDescriptorSetLayoutCreateInfo set_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 4;
    set_info.pBindings = bindings;
    if (post_writeback_shader_ == VK_NULL_HANDLE ||
        dfn.vkCreateDescriptorSetLayout(device, &set_info, nullptr, &post_writeback_set_layout_) != VK_SUCCESS) {
      return false;
    }
    VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(EdramPlace)};
    VkPipelineLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &post_writeback_set_layout_;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    if (dfn.vkCreatePipelineLayout(device, &layout_info, nullptr, &post_writeback_layout_) != VK_SUCCESS) {
      return false;
    }
    VkComputePipelineCreateInfo info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                  post_writeback_shader_, "main", nullptr};
    info.layout = post_writeback_layout_;
    if (dfn.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &post_writeback_pipeline_) !=
        VK_SUCCESS) {
      post_writeback_pipeline_ = VK_NULL_HANDLE;
      return false;
    }
  }
  post_objects_failed_ = false;
  REXGPU_INFO("[native][post] post targets ready (R32G32B32A32_SFLOAT + {}; classes RGBA8 {}, RGBA16_SNORM {})",
              post_ds_format_ == VK_FORMAT_D32_SFLOAT_S8_UINT ? "D32_SFLOAT_S8_UINT" : "D24_UNORM_S8_UINT",
              post_class_ok_[1] ? "yes" : "no", post_class_ok_[2] ? "yes" : "no");
  return true;
}

void NativeRenderer::RetirePostTarget(PostTarget& target) {
  Retired r;
  r.submission = command_processor_.GetCurrentSubmission();
  r.images[0] = target.color, r.images[1] = target.ds;
  r.memory[0] = target.color_memory, r.memory[1] = target.ds_memory;
  r.views[0] = target.color_view, r.views[1] = target.ds_view, r.views[2] = target.stencil_view;
  r.views[3] = target.depth_view;  // Session 43
  r.framebuffer = target.framebuffer;
  r.replace_set = target.writeback_set;
  if (target.color || target.ds || target.framebuffer || target.writeback_set) retired_.push_back(r);
  const uint32_t pitch_tiles = target.pitch_tiles, target_class = target.target_class;
  target = PostTarget();
  target.pitch_tiles = pitch_tiles;
  target.target_class = target_class;
}

// The post target for a surface pitch, at least width x height host pixels (a smaller one of that pitch is replaced;
// the caller has written back what it held). -1 if it cannot be made.
int32_t NativeRenderer::EnsurePostTarget(uint32_t pitch_tiles, uint32_t target_class, uint32_t width,
                                         uint32_t height) {
  if (DecoupleViews()) {  // (Session 94 review: the post targets' framebuffers have 1 layer, the multiview post passes need as many as their views - refused while gpu_d3d_decouple is on)
    static bool logged = false;
    if (!logged) {
      logged = true;
      REXGPU_ERROR("[native][post] a post target was asked for - not supported with gpu_d3d_decouple (1 layer, multiview pass); refused");
    }
    return -1;
  }
  int32_t index = -1;
  for (size_t i = 0; i < post_targets_.size(); ++i) {
    if (post_targets_[i].pitch_tiles == pitch_tiles && post_targets_[i].target_class == target_class) {
      index = int32_t(i);
    }
  }
  if (index >= 0 && post_targets_[index].color != VK_NULL_HANDLE && post_targets_[index].width >= width &&
      post_targets_[index].height >= height) {
    return index;
  }
  if (index < 0) {
    index = int32_t(post_targets_.size());
    PostTarget& added = post_targets_.emplace_back();
    added.pitch_tiles = pitch_tiles;
    added.target_class = target_class;
  } else {
    width = std::max(width, post_targets_[index].width);
    height = std::max(height, post_targets_[index].height);
    RetirePostTarget(post_targets_[index]);
  }
  PostTarget& t = post_targets_[index];
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  auto make_image = [&](VkFormat format, VkImageUsageFlags usage, VkImage& image, VkDeviceMemory& memory) {
    VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    return ui::vulkan::util::CreateDedicatedAllocationImage(vulkan_device, info,
                                                            ui::vulkan::util::MemoryPurpose::kDeviceLocal, image,
                                                            memory);
  };
  auto make_view = [&](VkImage image, VkFormat format, VkImageAspectFlags aspect, VkImageView& view) {
    VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(aspect);
    return dfn.vkCreateImageView(device, &view_info, nullptr, &view) == VK_SUCCESS;
  };
  const VkFormat color_format = kPostTargetFormats[target_class];
  bool ok = make_image(color_format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, t.color,
                       t.color_memory) &&
            make_image(post_ds_format_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                       t.ds, t.ds_memory) &&
            make_view(t.color, color_format, VK_IMAGE_ASPECT_COLOR_BIT, t.color_view) &&
            make_view(t.ds, post_ds_format_, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, t.ds_view) &&
            make_view(t.ds, post_ds_format_, VK_IMAGE_ASPECT_STENCIL_BIT, t.stencil_view) &&
            make_view(t.ds, post_ds_format_, VK_IMAGE_ASPECT_DEPTH_BIT, t.depth_view);
  if (ok) {
    VkImageView views[2] = {t.color_view, t.ds_view};
    VkFramebufferCreateInfo fb_info = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fb_info.renderPass = post_render_pass_continue_[target_class];
    fb_info.attachmentCount = 2;
    fb_info.pAttachments = views;
    fb_info.width = width;
    fb_info.height = height;
    fb_info.layers = 1;
    ok = dfn.vkCreateFramebuffer(device, &fb_info, nullptr, &t.framebuffer) == VK_SUCCESS;
  }
  if (!ok) {
    REXGPU_ERROR("[native][post] could not create a {}x{} post target", width, height);
    RetirePostTarget(t);
    return -1;
  }
  t.width = width;
  t.height = height;
  REXGPU_INFO("[native][post] post target {}x{} (pitch {} tiles, colour class {})", width, height, pitch_tiles,
              target_class);
  return index;
}

NativeRenderer::EdramPlace NativeRenderer::PostEdramPlace(const PostTarget& target, bool depth) const {
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const uint32_t tile_w = 80 * scale_x, tile_h = 16 * scale_y;
  EdramPlace place;
  place.color_base_dwords = (post_color_info_ & 0xFFF) * tile_w * tile_h;
  // No depth: never written (the colour load's depth is not tested). Session 43: a segment with depth has its own.
  place.depth_base_dwords = depth ? (post_depth_info_ & 0xFFF) * tile_w * tile_h : place.color_base_dwords;
  place.pitch_tile_dwords = target.pitch_tiles * tile_w * tile_h;
  place.tile_w = tile_w;
  place.tile_h = tile_h;
  place.width = target.width;
  place.height = target.height;
  place.wrap_dwords = 2048 * tile_w * tile_h;
  place.flags = 2u;  // a colour surface
  if (depth) place.flags |= (post_depth_info_ >> 16) & 1;  // Session 43: RB_DEPTH_INFO depth_format 1: 20e4
  place.flags |= ((post_color_info_ >> 16) & 0xF) << 4;      // Session 43: the colour format (the write-back)
  place.scale_x = place.scale_y = 1.0f;
  place.origin_x = place.origin_y = 0;
  return place;
}

uint32_t NativeRenderer::PostReason(const Draw& draw, bool check_shader) const {
  // Post targets take colour-only draws outside the scene's target anywhere in the frame (the passes before the scene
  // that have the post chain's shape too), and any after the scene's colour resolve.
  if (!replacing_ || !post_ || (!scene_done_this_frame_ && scene_key_.Matches(draw))) return kPostOff;
  if (draw.memexport) return kPostMemexport;
  if (!draw.raster.supported && !draw.raster.expanded) return kPostPrimitive;
  // Session 43: tiny passes stay emulated (gpu_native_post_min_pixels; host pixels at resolution scale 1).
  if (const int32_t min_pixels = REXCVAR_GET(gpu_native_post_min_pixels);
      min_pixels > 0 && int64_t(draw.scissor.extent.width) * draw.scissor.extent.height < min_pixels) {
    return kPostSmall;
  }
  if ((draw.surface_info >> 16) & 3) return kPostMsaa;  // RB_SURFACE_INFO msaa_samples
  if (draw.color_mask_all & ~uint32_t(0xF)) return kPostOtherTargets;
  if (!(draw.color_mask_all & 0xF)) return kPostNoColor;
  // RB_COLOR_INFO color_format: k_8_8_8_8; Session 43 (gpu_native_post_16bit): k_16_16 (4), k_16_16_16_16 (5).
  const uint32_t color_format = (draw.color_info >> 16) & 0xF;
  if (color_format != 0 &&
      !((color_format == 4 || color_format == 5) && (REXCVAR_GET(gpu_native_post_16bit) || all_native_))) {
    return kPostColorFormat;
  }
  const bool uses_depth = draw.raster.depth_test || draw.raster.depth_write;
  if (uses_depth) {
    // Session 43 (gpu_native_post_depth): exact for 20e4 depth in a float32 target (the shader rounds its depth to
    // 20e4), without a polygon offset (the interlock shader adds its own; the host's bias is computed otherwise).
    if (!REXCVAR_GET(gpu_native_post_depth) && !all_native_) return kPostDepth;
    if (!((draw.depth_info >> 16) & 1) || post_ds_format_ != VK_FORMAT_D32_SFLOAT_S8_UINT) return kPostDepthFormat;
    if (draw.raster.depth_bias_constant != 0.0f || draw.raster.depth_bias_slope != 0.0f) return kPostDepthBias;
    if (draw.raster.clip_disable) return kPostDepthClamp;
  }
  if (draw.raster.stencil) return kPostStencil;
  if (draw.occlusion_query) return kPostOcclusion;
  // Blending (or channels kept) reads what is under the draw and blends in the host's fixed-function arithmetic:
  // close to the interlock shader's, not bit-exact (Bryan's glare strips, run 2757). Off: exact. (Session 43: checked
  // before the shader: a blending draw gets no translation, and was counted as "no shader".)
  if (PostNeedsDst(draw)) {
    // Session 43: the load from the EDRAM reads k_8_8_8_8 only.
    if (color_format != 0) return kPostColorFormat;
    if (!REXCVAR_GET(gpu_native_post_blend) && !AllOn(kAllBlend)) return kPostBlend;
  }
  if (!draw.vertex_count || draw.vertex_shader == VK_NULL_HANDLE || draw.layout == VK_NULL_HANDLE ||
      !draw.pixel_shader_hash || (check_shader && draw.pixel_shader == VK_NULL_HANDLE) ||
      (check_shader && uses_depth && !draw.float24_depth) ||
      (check_shader && draw.color_quantize != (PostNeedsDst(draw) ? 0u : PostTargetClass(draw)))) {
    return kPostNoShader;
  }
  return kPostOk;
}

bool NativeRenderer::ClaimPost(Draw& draw) {
  const uint32_t reason = PostReason(draw, true);
  if (reason == kPostOff) return false;
  // wwe2k14 patch (Session 79, NF-D look-ahead diagnostic, gpu_native_frame_batch_log): as TruthRecordDraw's.
  if (const int32_t log_left = REXCVAR_GET(gpu_native_frame_batch_log); log_left > 0) [[unlikely]] {
    const uint64_t ps = draw.pixel_shader_hash;
    if (ps == 0x75641266549A40C0ull || ps == 0x9C84E33D39058FB0ull || ps == 0x4E2CD8031D93DE3Eull) {
      REXGPU_INFO("[native][frame] batch {} frame {}: post draw ps {:016X}", command_processor_.nf_batch_seq(), frame_,
                  ps);
      REXCVAR_SET(gpu_native_frame_batch_log, log_left - 1);
    }
  }
  FrameSlot& slot = slots_[SlotOf(frame_)];
  if (reason != kPostOk) {
    ++slot.post_skipped[reason];
    // Session 45: the primitives left (a diagnostic, once per vertex shader and primitive).
    if (reason == kPostPrimitive && no_shader_logged_ < 48) {
      const uint64_t pair = draw.vertex_shader_hash ^ (uint64_t(draw.guest_primitive) << 56) ^ (uint64_t(1) << 63);
      if (std::find(no_shader_pairs_.begin(), no_shader_pairs_.end(), pair) == no_shader_pairs_.end()) {
        no_shader_pairs_.push_back(pair);
        ++no_shader_logged_;
        REXGPU_INFO(
            "[native][post] left emulated (primitive), frame {}: vs {:016X} ps {:016X} guest primitive {} host {} host "
            "vs type {} polygonal {} supported {} expanded {} vertices {} pitch {} msaa {} colour 0x{:08X} depth "
            "0x{:08X} mask 0x{:X} depth test {} write {} stencil {} fast clear {}",
            frame_, draw.vertex_shader_hash, draw.pixel_shader_hash, draw.guest_primitive, draw.host_primitive,
            draw.host_vs_type, draw.polygonal, draw.raster.supported, draw.raster.expanded, draw.vertex_count,
            draw.surface_info & 0x3FFF, (draw.surface_info >> 16) & 3, draw.color_info, draw.depth_info,
            draw.color_mask_all, draw.raster.depth_test, draw.raster.depth_write, draw.raster.stencil,
            draw.fast_clear);
      }
    }
    // Session 43: which part is missing, for the first few such draws (a diagnostic).
    if ((reason == kPostNoShader || reason == kPostBlend) && no_shader_logged_ < 48) {
      const uint64_t pair = draw.vertex_shader_hash ^ (draw.pixel_shader_hash * 31) ^ (uint64_t(reason) << 60);
      if (std::find(no_shader_pairs_.begin(), no_shader_pairs_.end(), pair) == no_shader_pairs_.end()) {
        no_shader_pairs_.push_back(pair);
        ++no_shader_logged_;
        REXGPU_INFO(
            "[native][post] left emulated ({}), frame {}: vs {:016X} ps {:016X} vertices {} pitch {} colour 0x{:08X} "
            "vertex module {} layout {} pixel module {} depth test {} write {} float24 depth {}; blend 0x{:08X} "
            "write mask 0x{:X}",
            reason == kPostBlend ? "blending" : "no shader", frame_, draw.vertex_shader_hash, draw.pixel_shader_hash,
            draw.vertex_count, draw.surface_info & 0x3FFF, draw.color_info, draw.vertex_shader != VK_NULL_HANDLE,
            draw.layout != VK_NULL_HANDLE, draw.pixel_shader != VK_NULL_HANDLE, draw.raster.depth_test,
            draw.raster.depth_write, draw.float24_depth, draw.blend_control, draw.color_write_mask);
      }
    }
    return false;
  }
  // Post surfaces share EDRAM tiles: another surface's draws must be in the EDRAM before this one is drawn. The
  // caller writes them back and draws this one emulated.
  // Session 42: the scene's targets too (after the scene with gpu_native_after_scene).
  // Session 43: a segment has one depth surface (a draw with another goes into a new segment).
  const bool uses_depth = draw.raster.depth_test || draw.raster.depth_write;
  if (((!post_batch_.empty() || post_dirty_) &&
       (draw.surface_info != post_surface_info_ || draw.color_info != post_color_info_ ||
        (uses_depth && (post_depth_ || post_batch_depth_) && draw.depth_info != post_depth_info_))) ||
      !batch_.empty() || native_dirty_) {
    ++slot.post_skipped[kPostSwitch];
    claim_retry_ = true;
    return false;
  }
  if (!EnsurePostObjects()) {
    ++slot.post_skipped[kPostNoShader];
    return false;
  }
  draw.post = true;
  draw.pipeline = GetPipeline(draw);
  if (draw.pipeline == VK_NULL_HANDLE) {
    draw.post = false;
    ++slot.post_skipped[kPostNoShader];
    return false;
  }
  post_surface_info_ = draw.surface_info;
  post_color_info_ = draw.color_info;
  if (uses_depth) {
    post_depth_info_ = draw.depth_info;
    post_batch_depth_ = true;
    ++slot.post_depth_draws;
  }
  draw.owned = true;
  return true;
}

bool NativeRenderer::FlushPost() {
  if (post_batch_.empty()) return false;
  FrameSlot& slot = slots_[SlotOf(frame_)];
  if (post_batch_submission_ != command_processor_.GetCurrentSubmission()) {
    REXGPU_ERROR("[native][post] {} claimed post draws from an ended submission: lost (frame {})", post_batch_.size(),
                 frame_);
    report_.dropped_submission += post_batch_.size();
    post_batch_.clear();
    post_batch_depth_ = false;
    return false;
  }
  const auto cpu_start = std::chrono::steady_clock::now();
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const uint32_t tile_w = 80 * scale_x, tile_h = 16 * scale_y;
  const uint32_t pitch_tiles = std::max(1u, ((post_surface_info_ & 0x3FFF) + 79) / 80);
  int32_t bottom = 1;
  bool needs_dst = false;
  // Session 43: draws that test or write depth need the depth surface's values from the EDRAM at the segment's start.
  const bool needs_depth = post_batch_depth_;
  post_batch_depth_ = false;
  bool writes_depth = false;
  for (const Draw& draw : post_batch_) {
    writes_depth |= draw.raster.depth_write;
    // The scissor's bottom is not limited to the surface (draw_util limits x to the pitch only); with the viewport
    // on, it limits too (as the scene's sizing in OnDraw).
    int32_t draw_bottom = draw.scissor.offset.y + int32_t(draw.scissor.extent.height);
    if (!draw.screen_space) {
      draw_bottom = std::min(draw_bottom, int32_t(std::ceil(draw.viewport.y + draw.viewport.height)));
    }
    bottom = std::max(bottom, draw_bottom);
    // Blending, or channels left as they are: the draw needs what the EDRAM has under it.
    needs_dst |= PostNeedsDst(draw);
  }
  const uint32_t width = pitch_tiles * tile_w;
  // At most the rows of tiles a surface of this pitch has in the 2048-tile EDRAM (review fix; Session 43: a 64bpp row
  // of tiles is twice as long).
  const uint32_t row_tiles = pitch_tiles * (((post_color_info_ >> 16) & 0xF) == 5 ? 2u : 1u);
  const uint32_t height = std::min((uint32_t(bottom) + tile_h - 1) / tile_h * tile_h,
                                   (2048u + row_tiles - 1) / row_tiles * tile_h);
  // Session 43: the target's colour class (the batch's draws all go to one surface).
  const uint32_t target_class = PostTargetClass(post_batch_.front());
  // A target of this pitch that has to grow: what it holds goes into the EDRAM first.
  for (size_t i = 0; i < post_targets_.size(); ++i) {
    const PostTarget& t = post_targets_[i];
    if (t.pitch_tiles == pitch_tiles && t.target_class == target_class && t.color != VK_NULL_HANDLE &&
        (t.width < width || t.height < height)) {
      if (post_current_ == int32_t(i)) {
        PostWriteBack();
        post_segment_open_ = false;
      }
      WritePending(int32_t(i));  // Session 45 (gpu_native_defer_writeback)
    }
  }
  const int32_t index = EnsurePostTarget(pitch_tiles, target_class, width, height);
  if (index < 0) {
    REXGPU_ERROR("[native][post] no post target: {} claimed post draws lost (frame {})", post_batch_.size(), frame_);
    post_batch_.clear();
    return false;
  }
  if (index != post_current_) {
    PostWriteBack();  // nothing expected (ClaimPost keeps one surface until written back)
    post_segment_open_ = false;
    post_current_ = index;
  }
  // Session 45 (gpu_native_defer_writeback): the target's pending colour - the same surface resumes its segment (the
  // coverage marks kept, the pending pixels written back with the rest later), another surface puts it into the EDRAM
  // first (the target's pixels are about to change).
  if (!post_segment_open_ && post_targets_[index].pending) {
    PostTarget& t = post_targets_[index];
    if ((t.pending_color_info & 0xF0FFF) == (post_color_info_ & 0xF0FFF) &&
        (t.pending_surface_info & 0x3FFFF) == (post_surface_info_ & 0x3FFFF)) {
      t.pending = false;
      post_segment_open_ = true;
      post_dirty_ = true;
      post_loaded_ = false;
      post_depth_ = post_depth_dirty_ = post_depth_loaded_ = false;
      post_x0_ = t.pending_rect[0], post_y0_ = t.pending_rect[1], post_x1_ = t.pending_rect[2], post_y1_ = t.pending_rect[3];
      std::copy(std::begin(t.pending_cov), std::end(t.pending_cov), post_cov_color_);
      std::fill(std::begin(post_cov_depth_), std::end(post_cov_depth_), 0);
      ++report_.pending_resumed;
    } else {
      WritePending(index);
    }
  }
  // Session 45 (gpu_native_post_sync): whether the target already holds what the EDRAM has under this batch's draws
  // (their scissors), for the colour surface and, for draws with depth, the depth surface - then no load is needed.
  bool color_synced = false, depth_synced = false;
  // Session 45 (clear mirrors): or would hold it after drawing the last fill of the surface (a new segment): the tiles
  // wholly inside the fill's rectangle.
  bool color_by_mirror = false, depth_by_mirror = false;
  PostTarget::Mirror mirror_c_use, mirror_d_use;
  std::bitset<2048> mirror_c_full, mirror_d_full;
  if (post_sync_) {
    PostTarget& t = post_targets_[index];
    const bool begins = !post_segment_open_;
    const auto mirror_full = [&](const PostTarget::Mirror& m, bool is_64bpp) {
      const uint32_t tw = is_64bpp ? 40u : 80u;
      const uint32_t c0 = (uint32_t(std::max(0, m.rect[0])) + tw - 1) / tw, c1 = uint32_t(std::max(0, m.rect[2])) / tw;
      const uint32_t r0 = (uint32_t(std::max(0, m.rect[1])) + 15) / 16, r1 = uint32_t(std::max(0, m.rect[3])) / 16;
      if (c1 <= c0 || r1 <= r0) return std::bitset<2048>();
      return EdramTiles(m.info & 0xFFF, pitch_tiles, is_64bpp, c0 * tw, r0 * 16, c1 * tw, r1 * 16);
    };
    const bool color_64bpp = ((post_color_info_ >> 16) & 0xF) == 5;
    const bool mirror_c_ok = begins && clear_mirrors_ && t.mirror_color.valid && !color_64bpp && t.target_class != 0 &&
                             (t.mirror_color.info & 0xF0FFF) == (post_color_info_ & 0xF0FFF);
    const bool mirror_d_ok = begins && clear_mirrors_ && t.mirror_depth.valid && needs_depth &&
                             (t.mirror_depth.info & 0x10FFF) == (post_depth_info_ & 0x10FFF);
    const std::bitset<2048> mirror_c_tiles = mirror_c_ok ? mirror_full(t.mirror_color, false) : std::bitset<2048>();
    const std::bitset<2048> mirror_d_tiles = mirror_d_ok ? mirror_full(t.mirror_depth, false) : std::bitset<2048>();
    int32_t ux0 = INT32_MAX, uy0 = INT32_MAX, ux1 = 0, uy1 = 0;
    for (const Draw& draw : post_batch_) {
      ux0 = std::min(ux0, std::clamp(draw.scissor.offset.x, 0, int32_t(t.width)));
      uy0 = std::min(uy0, std::clamp(draw.scissor.offset.y, 0, int32_t(t.height)));
      ux1 = std::max(ux1, std::clamp(draw.scissor.offset.x + int32_t(draw.scissor.extent.width), 0, int32_t(t.width)));
      uy1 = std::max(uy1, std::clamp(draw.scissor.offset.y + int32_t(draw.scissor.extent.height), 0, int32_t(t.height)));
    }
    if (ux1 > ux0 && uy1 > uy0) {
      // In guest samples (post targets are one sample, at the resolution scale).
      const uint32_t gx0 = uint32_t(ux0) / scale_x, gy0 = uint32_t(uy0) / scale_y;
      const uint32_t gx1 = (uint32_t(ux1) + scale_x - 1) / scale_x, gy1 = (uint32_t(uy1) + scale_y - 1) / scale_y;
      const std::bitset<2048> color_tiles =
          EdramTiles(post_color_info_ & 0xFFF, pitch_tiles, color_64bpp, gx0, gy0, gx1, gy1);
      // (Review fix: a float target - class 0 - never counts as holding the EDRAM's colour.)
      const std::bitset<2048> sync_c = t.initialized && t.target_class != 0 &&
                                               (t.sync_color_info & 0xF0FFF) == (post_color_info_ & 0xF0FFF)
                                           ? t.sync_color
                                           : std::bitset<2048>();
      color_synced = (color_tiles & ~sync_c).none();
      // The fill's rectangle holds the batch's draws (pixels: fills rarely end on tile edges), or with the synced
      // tiles it covers their tiles.
      const auto rect_holds = [&](const PostTarget::Mirror& m) {
        return m.rect[0] <= ux0 && m.rect[1] <= uy0 && m.rect[2] >= ux1 && m.rect[3] >= uy1;
      };
      color_by_mirror = !color_synced && mirror_c_ok &&
                        (rect_holds(t.mirror_color) || (color_tiles & ~(sync_c | mirror_c_tiles)).none());
      if (needs_depth) {
        const std::bitset<2048> depth_tiles = EdramTiles(post_depth_info_ & 0xFFF, pitch_tiles, false, gx0, gy0, gx1, gy1);
        const std::bitset<2048> sync_d =
            t.initialized && (t.sync_depth_info & 0x10FFF) == (post_depth_info_ & 0x10FFF) ? t.sync_depth
                                                                                            : std::bitset<2048>();
        depth_synced = (depth_tiles & ~sync_d).none();
        depth_by_mirror = !depth_synced && mirror_d_ok &&
                          (rect_holds(t.mirror_depth) || (depth_tiles & ~(sync_d | mirror_d_tiles)).none());
      }
    } else {
      color_synced = depth_synced = true;  // the draws cover no pixel of the target
    }
    // A mirror that would serve counts as synced; it is drawn at the segment's start (below), unless the batch turns
    // out not to read that channel (then it is dropped: the target loads later if a draw needs it).
    color_synced |= color_by_mirror;
    depth_synced |= depth_by_mirror;
    mirror_c_use = t.mirror_color, mirror_d_use = t.mirror_depth;
    mirror_c_full = mirror_c_tiles, mirror_d_full = mirror_d_tiles;
    if (begins) t.mirror_color.valid = t.mirror_depth.valid = false;  // used now or out of date
  }
  // Session 45 (gpu_native_post_sync): what the batch's draws read of the target - colour under blending or kept
  // channels, depth under a test other than "always" - outside the pixels an earlier draw of the segment provably wrote
  // (a covering quad / rectangle: colour without reading it, depth with the test always). Only that needs a load:
  // depth that is only written goes back by its coverage marks, whatever the target held. Without the setting:
  // every draw that blends or uses depth (Sessions 42-43).
  bool reads_color = needs_dst, reads_depth = needs_depth;
  int32_t cov_color[4] = {}, cov_depth[4] = {};
  if (post_sync_) {
    if (post_segment_open_) {
      std::copy(std::begin(post_cov_color_), std::end(post_cov_color_), cov_color);
      std::copy(std::begin(post_cov_depth_), std::end(post_cov_depth_), cov_depth);
    }
    const auto inside = [](const int32_t r[4], const int32_t c[4]) {
      return r[2] <= r[0] || r[3] <= r[1] || (c[0] <= r[0] && c[1] <= r[1] && c[2] >= r[2] && c[3] >= r[3]);
    };
    const auto keep_larger = [](int32_t c[4], const int32_t r[4]) {
      const int64_t a = int64_t(std::max(0, c[2] - c[0])) * std::max(0, c[3] - c[1]);
      const int64_t b = int64_t(std::max(0, r[2] - r[0])) * std::max(0, r[3] - r[1]);
      if (b > a) std::copy(r, r + 4, c);
    };
    const PostTarget& t = post_targets_[index];
    reads_color = reads_depth = false;
    for (const Draw& draw : post_batch_) {
      const int32_t s[4] = {std::clamp(draw.scissor.offset.x, 0, int32_t(t.width)),
                            std::clamp(draw.scissor.offset.y, 0, int32_t(t.height)),
                            std::clamp(draw.scissor.offset.x + int32_t(draw.scissor.extent.width), 0, int32_t(t.width)),
                            std::clamp(draw.scissor.offset.y + int32_t(draw.scissor.extent.height), 0,
                                       int32_t(t.height))};
      const bool color_read = PostNeedsDst(draw);
      const bool depth_read = draw.raster.depth_test && draw.raster.depth_compare != VK_COMPARE_OP_ALWAYS;
      if (color_read && !inside(s, cov_color)) reads_color = true;
      if (depth_read && !inside(s, cov_depth)) reads_depth = true;
      if (draw.cover[2] > draw.cover[0] && draw.cover[3] > draw.cover[1]) {
        // The proven pixels in host pixels, within the scissor.
        const int32_t r[4] = {std::max(s[0], draw.cover[0] * int32_t(scale_x)),
                              std::max(s[1], draw.cover[1] * int32_t(scale_y)),
                              std::min(s[2], draw.cover[2] * int32_t(scale_x)),
                              std::min(s[3], draw.cover[3] * int32_t(scale_y))};
        // (Review fix: a depth-tested draw writes colour only where the test passes.)
        if (!color_read && !depth_read && (draw.color_write_mask & 0xF)) keep_larger(cov_color, r);
        if (draw.raster.depth_write && !depth_read) keep_larger(cov_depth, r);
      }
    }
  }
  // A draw that needs what is under it, in a segment that did not load: the segment so far goes into the EDRAM and
  // a new one begins with a load.
  if (post_segment_open_ && reads_color && !post_loaded_ && !color_synced) {
    PostWriteBack();
    post_segment_open_ = false;
  }
  // Session 43: the same for depth (a segment that began without it). Session 45: that read it.
  if (post_segment_open_ && reads_depth && !(post_sync_ ? post_depth_loaded_ : post_depth_) && !depth_synced) {
    PostWriteBack();
    post_segment_open_ = false;
  }
  // Session 45: a segment that began without depth goes on with it (its draws only write it, or the target holds
  // that depth surface).
  if (post_segment_open_ && needs_depth && !post_depth_) post_depth_ = true;
  // (A segment written back just above begins keeping the target: the pixels its draws proved stay the truth there,
  // and the load that follows brings the rest.)
  const bool begin = !post_segment_open_;
  const bool load_color = begin && reads_color && !color_synced;
  const bool load_depth = begin && reads_depth && !depth_synced;
  const bool load = load_color || load_depth;
  if (post_sync_ && begin) {
    report_.post_loads_avoided += (needs_dst && !load_color ? 1 : 0) + (needs_depth && !load_depth ? 1 : 0);
  }
  // Session 45 (gpu_native_defer_writeback): the load reads the EDRAM: pending colour over its tiles goes in first.
  if (load && defer_writeback_) {
    const PostTarget& t = post_targets_[index];
    const uint32_t gw = t.width / scale_x, gh = t.height / scale_y;
    std::bitset<2048> tiles;
    if (load_color) {
      tiles |= EdramTiles(post_color_info_ & 0xFFF, pitch_tiles, ((post_color_info_ >> 16) & 0xF) == 5, 0, 0, gw, gh);
    }
    if (load_depth) tiles |= EdramTiles(post_depth_info_ & 0xFFF, pitch_tiles, false, 0, 0, gw, gh);
    WritePendingOverlapping(tiles, -1);
  }
  // Session 45: a segment begins keeping what the target holds (sync tracking needs its images to persist).
  const bool keep = begin && post_sync_ && post_targets_[index].initialized;
  if (segment_log_frames_ > 0) {  // Session 45: the segment log
    int32_t ux0 = INT32_MAX, uy0 = INT32_MAX, ux1 = 0, uy1 = 0;
    uint32_t blending = 0;
    for (const Draw& draw : post_batch_) {
      ux0 = std::min(ux0, draw.scissor.offset.x);
      uy0 = std::min(uy0, draw.scissor.offset.y);
      ux1 = std::max(ux1, draw.scissor.offset.x + int32_t(draw.scissor.extent.width));
      uy1 = std::max(uy1, draw.scissor.offset.y + int32_t(draw.scissor.extent.height));
      blending += PostNeedsDst(draw) ? 1 : 0;
    }
    REXGPU_INFO("[native][seg] frame {} post pass: base {} pitch {} format {} class {} target {}x{} {} load colour {} "
                "depth {} draws {} (blending {}, depth {}) scissors {},{}-{},{} depth base {}",
                frame_, post_color_info_ & 0xFFF, post_surface_info_ & 0x3FFF, (post_color_info_ >> 16) & 0xF,
                target_class, width, height, begin ? "begins" : "continues", load_color, load_depth,
                post_batch_.size(), blending, needs_depth, ux0, uy0, ux1, uy1,
                needs_depth ? int32_t(post_depth_info_ & 0xFFF) : -1);
  }
  if (begin) {
    post_depth_ = needs_depth;
    post_depth_loaded_ = load_depth;  // Session 45
  }
  // Session 45 (clear mirrors): the fills the batch's reads need, drawn as constants at the segment's start (each
  // pixel takes the fill's own dword: no read of the EDRAM under it).
  const bool mirror_c = begin && reads_color && color_by_mirror && !load_color;
  const bool mirror_d = begin && reads_depth && depth_by_mirror && !load_depth;
  if ((load || mirror_c || mirror_d) && !EnsureReplaceSets(false)) {
    REXGPU_ERROR("[native][post] no load pass: {} claimed post draws lost (frame {})", post_batch_.size(), frame_);
    post_batch_.clear();
    return false;
  }
  const PostTarget& target = post_targets_[index];
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const uint32_t query_base = SlotOf(frame_) * kQueriesPerFrame;
  FrameSlot::Segment* segment = nullptr;
  if (query_pool_ && slot_queries_ok_ && slot.queries + 2 <= QueryLimit()) {
    segment = &slot.segments.emplace_back();
    segment->kind = 2;  // Session 43
    segment->native_begin = slot.queries++;
    cb.CmdVkWriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, query_base + segment->native_begin);
  }
  if (load || mirror_c || mirror_d) {
    // Every earlier writer of the EDRAM -> the load's read (recorded directly, as the scene's load).
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                               VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  }
  VkClearValue clear_values[2] = {};
  clear_values[1].depthStencil = {0.0f, 0};
  VkRenderPassBeginInfo begin_info = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  begin_info.renderPass = begin ? (keep ? post_render_pass_begin_keep_[target_class]
                                        : post_render_pass_begin_[target_class])
                                : post_render_pass_continue_[target_class];
  // Session 44 (gpu_native_post_lean_depth): a segment without depth neither loads nor stores it (nothing reads it: a
  // draw that needs depth starts a new segment, which loads depth from the EDRAM; the write-back writes depth only
  // for segments with it). Render passes that differ in load / store operations only are compatible.
  // Session 45: not with gpu_native_post_sync (the target's depth must persist from segment to segment).
  if (!post_depth_ && REXCVAR_GET(gpu_native_post_lean_depth) && !post_sync_) {
    begin_info.renderPass = begin ? post_render_pass_begin_no_depth_[target_class]
                                  : post_render_pass_continue_no_depth_[target_class];
    ++report_.post_lean_depth_passes;
  }
  begin_info.framebuffer = target.framebuffer;
  begin_info.renderArea = {{0, 0}, {target.width, target.height}};
  begin_info.clearValueCount = begin ? 2 : 0;
  begin_info.pClearValues = begin ? clear_values : nullptr;
  cb.CmdVkBeginRenderPass(&begin_info, VK_SUBPASS_CONTENTS_INLINE);
  const VkViewport full = {0.0f, 0.0f, float(target.width), float(target.height), 0.0f, 1.0f};
  const VkRect2D full_scissor = {{0, 0}, {target.width, target.height}};
  if (load) {
    const EdramPlace place = PostEdramPlace(target, post_depth_);
    cb.CmdVkBindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS,
                         load_color ? (load_depth ? post_load_both_pipeline_[target_class]
                                                  : post_load_pipeline_[target_class])
                                    : post_load_depth_pipeline_[target_class]);
    cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, load_layout_, 0, 1, &load_set_, 0, nullptr);
    cb.CmdVkPushConstants(load_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(place), &place);
    cb.CmdVkSetViewport(0, 1, &full);
    cb.CmdVkSetScissor(0, 1, &full_scissor);
    cb.CmdVkDraw(3, 1, 0, 0);
    ++slot.post_loads;
  }
  // Session 45 (clear mirrors): colour and depth in one draw when the fills share a rectangle, else one each.
  const bool mirror_both =
      mirror_c && mirror_d && std::equal(mirror_c_use.rect, mirror_c_use.rect + 4, mirror_d_use.rect);
  for (int pass = 0; pass < 2; ++pass) {
    const bool c = pass == 0 && mirror_c;
    const bool d = pass == 0 ? mirror_d && (mirror_both || !mirror_c) : mirror_d && mirror_c && !mirror_both;
    if (!c && !d) continue;
    const PostTarget::Mirror& m = c ? mirror_c_use : mirror_d_use;
    EdramPlace place = PostEdramPlace(target, d);
    place.flags |= 0x1000u;
    place.origin_x = mirror_c_use.src;
    place.origin_y = mirror_d_use.src;
    cb.CmdVkBindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS,
                         c ? (d ? post_load_both_pipeline_[target_class] : post_load_pipeline_[target_class])
                           : post_load_depth_pipeline_[target_class]);
    cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, load_layout_, 0, 1, &load_set_, 0, nullptr);
    cb.CmdVkPushConstants(load_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(place), &place);
    cb.CmdVkSetViewport(0, 1, &full);
    const int32_t x0 = std::clamp(m.rect[0], 0, int32_t(target.width));
    const int32_t y0 = std::clamp(m.rect[1], 0, int32_t(target.height));
    const int32_t x1 = std::clamp(m.rect[2], x0, int32_t(target.width));
    const int32_t y1 = std::clamp(m.rect[3], y0, int32_t(target.height));
    const VkRect2D rect = {{x0, y0}, {uint32_t(x1 - x0), uint32_t(y1 - y0)}};
    cb.CmdVkSetScissor(0, 1, &rect);
    cb.CmdVkDraw(3, 1, 0, 0);
    report_.post_mirrors += (c ? 1 : 0) + (d ? 1 : 0);
  }
  // Session 45 (gpu_native_post_sync): what the target now holds of the EDRAM.
  if (post_sync_) {
    PostTarget& t = post_targets_[index];
    if (begin && !keep) {  // a discarding first pass: nothing is kept but what it loads
      t.sync_color.reset();
      t.sync_depth.reset();
    }
    if (begin && (t.sync_color_info & 0xF0FFF) != (post_color_info_ & 0xF0FFF)) {
      t.sync_color.reset();
      t.sync_color_info = post_color_info_;
    }
    if (post_depth_ && (t.sync_depth_info & 0x10FFF) != (post_depth_info_ & 0x10FFF)) {
      t.sync_depth.reset();
      t.sync_depth_info = post_depth_info_;
    }
    // The load writes the whole target.
    const uint32_t gw = t.width / scale_x, gh = t.height / scale_y;
    if (load_color) {
      t.sync_color_info = post_color_info_;
      t.sync_color = EdramTiles(post_color_info_ & 0xFFF, pitch_tiles, ((post_color_info_ >> 16) & 0xF) == 5, 0, 0,
                                gw, gh);
    }
    if (load_depth) {
      t.sync_depth_info = post_depth_info_;
      t.sync_depth = EdramTiles(post_depth_info_ & 0xFFF, pitch_tiles, false, 0, 0, gw, gh);
    }
    // Session 45 (clear mirrors): the tiles wholly inside a drawn fill now equal the EDRAM.
    if (mirror_c) t.sync_color |= mirror_c_full;
    if (mirror_d) t.sync_depth |= mirror_d_full;
    t.initialized = true;
  }
  VkPipeline bound_pipeline = VK_NULL_HANDLE;
  VkPipelineLayout bound_layout = VK_NULL_HANDLE;
  VkDescriptorSet bound_sets[SpirvShaderTranslator::kDescriptorSetCount] = {};
  uint32_t bound_offsets[SpirvShaderTranslator::kConstantBufferCount] = {};
  bool depth_bias_set = false;
  for (const Draw& draw : post_batch_) {
    VkPipeline pipeline = draw.pipeline;
    if (pipeline == VK_NULL_HANDLE) continue;
    if (pipeline != bound_pipeline) {
      cb.CmdVkBindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
      bound_pipeline = pipeline;
    }
    if (draw.layout != bound_layout) {
      std::fill(std::begin(bound_sets), std::end(bound_sets), VkDescriptorSet(VK_NULL_HANDLE));
      bound_layout = draw.layout;
    }
    for (uint32_t i = 0; i < SpirvShaderTranslator::kDescriptorSetCount; ++i) {
      if (!(draw.sets_bound & (1u << i)) || draw.sets[i] == VK_NULL_HANDLE) continue;
      const bool constants = i == SpirvShaderTranslator::kDescriptorSetConstants && draw.dynamic_offset_count;
      if (bound_sets[i] == draw.sets[i] &&
          (!constants ||
           !std::memcmp(bound_offsets, draw.dynamic_offsets, draw.dynamic_offset_count * sizeof(uint32_t)))) {
        continue;
      }
      cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, draw.layout, i, 1, &draw.sets[i],
                                 constants ? draw.dynamic_offset_count : 0,
                                 constants ? draw.dynamic_offsets : nullptr);
      bound_sets[i] = draw.sets[i];
      if (constants) std::memcpy(bound_offsets, draw.dynamic_offsets, draw.dynamic_offset_count * sizeof(uint32_t));
    }
    // The emulated viewport and scissor as they are (post targets have the emulated host pixels' size).
    cb.CmdVkSetViewport(0, 1, &draw.viewport);
    const int32_t x0 = std::clamp(draw.scissor.offset.x, 0, int32_t(target.width));
    const int32_t y0 = std::clamp(draw.scissor.offset.y, 0, int32_t(target.height));
    const int32_t x1 = std::clamp(draw.scissor.offset.x + int32_t(draw.scissor.extent.width), x0, int32_t(target.width));
    const int32_t y1 =
        std::clamp(draw.scissor.offset.y + int32_t(draw.scissor.extent.height), y0, int32_t(target.height));
    const VkRect2D scissor = {{x0, y0}, {uint32_t(x1 - x0), uint32_t(y1 - y0)}};
    cb.CmdVkSetScissor(0, 1, &scissor);
    if (!depth_bias_set) {
      cb.CmdVkSetDepthBias(0.0f, 0.0f, 0.0f);  // dynamic in our pipelines; post pipelines have no bias
      depth_bias_set = true;
    }
    cb.CmdVkSetBlendConstants(draw.blend_constants);
    post_x0_ = std::min(post_x0_, x0);
    post_y0_ = std::min(post_y0_, y0);
    post_x1_ = std::max(post_x1_, x1);
    post_y1_ = std::max(post_y1_, y1);
    if (draw.diag_id) [[unlikely]] cb.CmdDiagMark(diag::kMarkNextDraw, draw.diag_id, draw.diag_aux);  // (Session 95, the diagnostic run)
    BindDrawInputs(cb, draw);  // (Session 95)
    if (draw.indexed) {
      cb.CmdVkBindIndexBuffer(draw.index_buffer, draw.index_offset, draw.index_type);
      cb.CmdVkDrawIndexed(draw.vertex_count, 1, 0, draw.vertex_offset, 0);
    } else {
      cb.CmdVkDraw(draw.vertex_count, 1, uint32_t(draw.vertex_offset), 0);
    }
  }
  cb.CmdVkEndRenderPass();
  if (load || mirror_c || mirror_d) {
    // The load's read -> the EDRAM's next writers.
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 0, nullptr, 0, nullptr, 0, nullptr);
  }
  if (segment) {
    segment->native_end = slot.queries++;
    cb.CmdVkWriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, query_base + segment->native_end);
  }
  if (begin) post_loaded_ = load_color;
  // Session 45 (gpu_native_post_sync): the segment's proven pixels so far.
  if (post_sync_) {
    std::copy(std::begin(cov_color), std::end(cov_color), post_cov_color_);
    std::copy(std::begin(cov_depth), std::end(cov_depth), post_cov_depth_);
  }
  post_segment_open_ = true;
  post_dirty_ = true;
  post_depth_dirty_ |= writes_depth;  // Session 43
  ++slot.post_passes;
  post_batch_.clear();
  slot.cpu_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - cpu_start).count();
  return true;
}

// Session 45 (gpu_native_defer_writeback): a pending target's colour into the EDRAM now (its segment's state put in
// place of the open segment's for PostWriteBack, then restored).
void NativeRenderer::WritePending(int32_t index) {
  if (index < 0 || size_t(index) >= post_targets_.size() || !post_targets_[index].pending) return;
  PostTarget& t = post_targets_[index];
  t.pending = false;
  const int32_t saved_current = post_current_;
  const uint32_t saved_color = post_color_info_, saved_surface = post_surface_info_, saved_depth = post_depth_info_;
  const bool saved_dirty = post_dirty_, saved_open = post_segment_open_, saved_depth_on = post_depth_;
  const bool saved_depth_dirty = post_depth_dirty_, saved_loaded = post_loaded_, saved_depth_loaded = post_depth_loaded_;
  const int32_t saved_rect[4] = {post_x0_, post_y0_, post_x1_, post_y1_};
  int32_t saved_cov_color[4], saved_cov_depth[4], saved_cover[4];
  std::copy(std::begin(post_cov_color_), std::end(post_cov_color_), saved_cov_color);
  std::copy(std::begin(post_cov_depth_), std::end(post_cov_depth_), saved_cov_depth);
  std::copy(std::begin(post_cover_), std::end(post_cover_), saved_cover);
  post_current_ = index;
  post_color_info_ = t.pending_color_info;
  post_surface_info_ = t.pending_surface_info;
  post_dirty_ = true;
  post_depth_ = post_depth_dirty_ = false;
  post_x0_ = t.pending_rect[0], post_y0_ = t.pending_rect[1], post_x1_ = t.pending_rect[2], post_y1_ = t.pending_rect[3];
  std::copy(std::begin(t.pending_cov), std::end(t.pending_cov), post_cov_color_);
  std::fill(std::begin(post_cov_depth_), std::end(post_cov_depth_), 0);
  ++report_.pending_written;
  PostWriteBack();
  post_current_ = saved_current;
  post_color_info_ = saved_color, post_surface_info_ = saved_surface, post_depth_info_ = saved_depth;
  post_dirty_ = saved_dirty, post_segment_open_ = saved_open, post_depth_ = saved_depth_on;
  post_depth_dirty_ = saved_depth_dirty, post_loaded_ = saved_loaded, post_depth_loaded_ = saved_depth_loaded;
  post_x0_ = saved_rect[0], post_y0_ = saved_rect[1], post_x1_ = saved_rect[2], post_y1_ = saved_rect[3];
  std::copy(saved_cov_color, saved_cov_color + 4, post_cov_color_);
  std::copy(saved_cov_depth, saved_cov_depth + 4, post_cov_depth_);
  std::copy(saved_cover, saved_cover + 4, post_cover_);
}

void NativeRenderer::WritePendingAll(int32_t except) {
  for (size_t i = 0; i < post_targets_.size(); ++i) {
    if (int32_t(i) != except) WritePending(int32_t(i));
  }
}

// Every pending target whose colour tiles meet these into the EDRAM (except one).
void NativeRenderer::WritePendingOverlapping(const std::bitset<2048>& tiles, int32_t except) {
  for (size_t i = 0; i < post_targets_.size(); ++i) {
    const PostTarget& t = post_targets_[i];
    if (int32_t(i) != except && t.pending && (t.pending_tiles & tiles).any()) WritePending(int32_t(i));
  }
}

// The pixels our post draws covered since the segment began, into the EDRAM; the segment ends (the EDRAM may change
// under our target after this). Session 45 (defer_color, gpu_native_defer_writeback): the colour stays in the target
// for now (pending), only the depth goes.
void NativeRenderer::PostWriteBack(bool defer_color) {
  std::fill(std::begin(post_cover_), std::end(post_cover_), 0);  // Session 43: the segment ends (or never began)
  // Session 45 (gpu_native_post_sync): the pixels the segment's draws provably wrote (synced below once written back).
  int32_t cov_color[4], cov_depth[4];
  std::copy(std::begin(post_cov_color_), std::end(post_cov_color_), cov_color);
  std::copy(std::begin(post_cov_depth_), std::end(post_cov_depth_), cov_depth);
  std::fill(std::begin(post_cov_color_), std::end(post_cov_color_), 0);
  std::fill(std::begin(post_cov_depth_), std::end(post_cov_depth_), 0);
  if (!post_dirty_) return;
  post_dirty_ = false;
  post_segment_open_ = false;
  // Session 43: the depth the segment's draws wrote goes back too; the next segment starts without depth.
  const bool segment_depth = post_depth_;
  const bool write_depth = post_depth_ && post_depth_dirty_;
  post_depth_ = post_depth_dirty_ = false;
  const int32_t rect_x0 = post_x0_, rect_y0 = post_y0_, rect_x1 = post_x1_, rect_y1 = post_y1_;
  post_x0_ = post_y0_ = INT32_MAX;
  post_x1_ = post_y1_ = 0;
  if (post_current_ < 0 || size_t(post_current_) >= post_targets_.size()) return;
  PostTarget& target = post_targets_[post_current_];
  const VkBuffer edram = command_processor_.NativeEdramBuffer();
  if (target.color == VK_NULL_HANDLE || edram == VK_NULL_HANDLE || !EnsurePostObjects()) {
    REXGPU_ERROR("[native][post] no write-back: this segment's post draws are lost (frame {})", frame_);
    InvalidateAllPostSync();  // Session 45 (review fix): the target no longer equals the EDRAM
    return;
  }
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  if (target.writeback_set == VK_NULL_HANDLE || target.writeback_edram != edram) {
    if (target.writeback_set != VK_NULL_HANDLE) {
      Retired r;
      r.submission = command_processor_.GetCurrentSubmission();
      r.replace_set = target.writeback_set;
      retired_.push_back(r);
      target.writeback_set = VK_NULL_HANDLE;
    }
    VkDescriptorSetAllocateInfo alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = replace_descriptor_pool_;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &post_writeback_set_layout_;
    if (dfn.vkAllocateDescriptorSets(device, &alloc, &target.writeback_set) != VK_SUCCESS) {
      target.writeback_set = VK_NULL_HANDLE;
      REXGPU_ERROR("[native][post] write-back descriptor set allocation failed: post draws lost (frame {})", frame_);
      InvalidateAllPostSync();  // Session 45 (review fix)
      return;
    }
    target.writeback_edram = edram;
    VkDescriptorBufferInfo edram_info = {edram, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo color_info = {sampler_nearest(), target.color_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo stencil_info = {sampler_nearest(), target.stencil_view,
                                          VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo depth_info = {sampler_nearest(), target.depth_view,
                                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};  // Session 43
    VkWriteDescriptorSet writes[4] = {};
    for (uint32_t i = 0; i < 4; ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = target.writeback_set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = i ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    writes[0].pBufferInfo = &edram_info;
    writes[1].pImageInfo = &color_info;
    writes[2].pImageInfo = &stencil_info;
    writes[3].pImageInfo = &depth_info;
    dfn.vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
  }
  EdramPlace place = PostEdramPlace(target, segment_depth);
  // Session 43: the colour class. Session 45: bit 2 (write the colour) off when it is deferred.
  place.flags |= (defer_color ? 0u : 4u) | (write_depth ? 8u : 0u) | (target.target_class << 8);
  const uint32_t x0 = uint32_t(std::clamp(rect_x0, 0, int32_t(place.width)));
  const uint32_t y0 = uint32_t(std::clamp(rect_y0, 0, int32_t(place.height)));
  const uint32_t x1 = uint32_t(std::clamp(rect_x1, int32_t(x0), int32_t(place.width)));
  const uint32_t y1 = uint32_t(std::clamp(rect_y1, int32_t(y0), int32_t(place.height)));
  if (x1 <= x0 || y1 <= y0) return;
  place.origin_x = x0;
  place.origin_y = y0;
  if (segment_log_frames_ > 0) {  // Session 45: the segment log
    REXGPU_INFO("[native][seg] frame {} post write-back: base {} pitch {} rect {},{}-{},{} depth {}{}", frame_,
                post_color_info_ & 0xFFF, post_surface_info_ & 0x3FFF, x0, y0, x1, y1, write_depth,
                defer_color ? " (colour deferred)" : "");
  }
  // Session 45 (gpu_native_defer_writeback): the colour the segment drew stays in the target (its coverage marks
  // kept until the next segment on it), pending until something reads its tiles from the EDRAM.
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const uint32_t gx0 = x0 / scale_x, gy0 = y0 / scale_y;
  const uint32_t gx1 = (x1 + scale_x - 1) / scale_x, gy1 = (y1 + scale_y - 1) / scale_y;
  const std::bitset<2048> color_tiles = EdramTiles(post_color_info_ & 0xFFF, target.pitch_tiles,
                                                   ((post_color_info_ >> 16) & 0xF) == 5, gx0, gy0, gx1, gy1);
  // Session 45 (gpu_native_defer_writeback): older pending colour under these tiles goes into the EDRAM first (in
  // order: pending targets never overlap, and this write lands after it).
  if (defer_writeback_) {
    std::bitset<2048> touched = color_tiles;
    if (write_depth) {
      touched |= EdramTiles(post_depth_info_ & 0xFFF, target.pitch_tiles, false, gx0, gy0, gx1, gy1);
    }
    WritePendingOverlapping(touched, post_current_);
  }
  if (defer_color) {
    target.pending = true;
    target.pending_color_info = post_color_info_;
    target.pending_surface_info = post_surface_info_;
    target.pending_rect[0] = int32_t(x0), target.pending_rect[1] = int32_t(y0);
    target.pending_rect[2] = int32_t(x1), target.pending_rect[3] = int32_t(y1);
    std::copy(cov_color, cov_color + 4, target.pending_cov);
    target.pending_tiles = color_tiles;
    ++report_.pending_deferred;
    // (Review fix: the EDRAM will change under the scene's targets, as after the write-back itself - also with no
    // dispatch recorded here; the parity check of gpu_native_check_writeback must not read before the colour is in.)
    native_stale_ = true;
  }
  // Nothing to record: the colour deferred, no depth.
  const bool dispatch = !defer_color || write_depth;
  if (dispatch) {
  FrameSlot& slot = slots_[SlotOf(frame_)];
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const uint32_t query_base = SlotOf(frame_) * kQueriesPerFrame;
  FrameSlot::Segment* segment = nullptr;
  if (query_pool_ && slot_queries_ok_ && slot.queries + 2 <= QueryLimit()) {
    segment = &slot.segments.emplace_back();
    segment->kind = 3;  // Session 43
    segment->native_begin = slot.queries++;
    cb.CmdVkWriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, query_base + segment->native_begin);
  }
  // As the scene's write-back: every earlier user of the EDRAM -> this pass; our attachments' writes -> its reads.
  constexpr VkPipelineStageFlags kEdramStages =
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
  constexpr VkAccessFlags kEdramAccesses = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                           VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  constexpr VkPipelineStageFlags kTargetStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                                 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  VkMemoryBarrier edram_barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, kEdramAccesses,
                                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
  VkImageMemoryBarrier images[2] = {};
  for (int i = 0; i < 2; ++i) {
    images[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    images[i].srcQueueFamilyIndex = images[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    images[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  }
  images[0].image = target.color;
  images[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  images[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  images[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  images[0].subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);
  images[1].image = target.ds;
  images[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  images[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  images[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
  images[1].subresourceRange =
      ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
  cb.CmdVkPipelineBarrier(kEdramStages | kTargetStages, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &edram_barrier, 0,
                          nullptr, 2, images);
  command_processor_.BindExternalComputePipeline(post_writeback_pipeline_);
  cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, post_writeback_layout_, 0, 1, &target.writeback_set, 0,
                             nullptr);
  cb.CmdVkPushConstants(post_writeback_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(place), &place);
  cb.CmdVkDispatch((x1 - x0 + 7) / 8, (y1 - y0 + 7) / 8, 1);
  // This pass's EDRAM writes -> everything after it; our images back to attachments.
  edram_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  edram_barrier.dstAccessMask = kEdramAccesses;
  images[0].srcAccessMask = images[1].srcAccessMask = 0;
  images[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  images[1].dstAccessMask =
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  for (int i = 0; i < 2; ++i) std::swap(images[i].oldLayout, images[i].newLayout);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, kEdramStages | kTargetStages, 0, 1, &edram_barrier, 0,
                          nullptr, 2, images);
  if (segment) {
    segment->native_end = slot.queries++;
    cb.CmdVkWriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, query_base + segment->native_end);
  }
  ++slot.post_writebacks;
  native_stale_ = true;  // the scene's targets share these EDRAM tiles: they reload before their next draw
  }  // dispatch
  // Session 45 (gpu_native_post_sync): the EDRAM changed under the tiles written; this target's own surfaces are
  // what it wrote (its records for them stay), every other record over them is out of date. (A deferred colour did
  // not change the EDRAM; the target holds it.)
  if (post_sync_) {
    // (A deferred colour counts as written for the other targets: a load of theirs over it writes it in first.)
    const std::bitset<2048> written_color = color_tiles;
    const std::bitset<2048> written_depth =
        write_depth ? EdramTiles(post_depth_info_ & 0xFFF, target.pitch_tiles, false, gx0, gy0, gx1, gy1)
                    : std::bitset<2048>();
    const std::bitset<2048> written = written_color | written_depth;
    DropMirrors(written);  // Session 45 (clear mirrors): fills before this write are out of date there
    for (size_t i = 0; i < post_targets_.size(); ++i) {
      PostTarget& u = post_targets_[i];
      if (int32_t(i) != post_current_) {
        u.sync_color &= ~written;
        u.sync_depth &= ~written;
        continue;
      }
      const bool own_color = (u.sync_color_info & 0xF0FFF) == (post_color_info_ & 0xF0FFF);
      const bool own_depth = segment_depth && (u.sync_depth_info & 0x10FFF) == (post_depth_info_ & 0x10FFF);
      u.sync_color &= ~(own_color ? written_depth : written);
      u.sync_depth &= ~(own_depth ? written_color : written);
      // The tiles wholly inside what the segment's draws provably wrote now equal the EDRAM (all their pixels were
      // drawn and written back) - unless the other surface's write-back overlapped them.
      const auto inside_tiles = [&](uint32_t base, bool is_64bpp, const int32_t r[4]) {
        const uint32_t tw = (is_64bpp ? 40u : 80u) * scale_x, th = 16u * scale_y;
        const uint32_t rx0 = uint32_t(std::max(0, r[0])), ry0 = uint32_t(std::max(0, r[1]));
        const uint32_t rx1 = uint32_t(std::max(0, r[2])), ry1 = uint32_t(std::max(0, r[3]));
        const uint32_t c0 = (rx0 + tw - 1) / tw, c1 = rx1 / tw, r0 = (ry0 + th - 1) / th, r1 = ry1 / th;
        if (c1 <= c0 || r1 <= r0) return std::bitset<2048>();
        return EdramTiles(base, target.pitch_tiles, is_64bpp, c0 * (tw / scale_x), r0 * 16, c1 * (tw / scale_x),
                          r1 * 16);
      };
      if (cov_color[2] > cov_color[0] && cov_color[3] > cov_color[1]) {
        if (!own_color) {
          u.sync_color.reset();
          u.sync_color_info = post_color_info_;
        }
        u.sync_color |= inside_tiles(post_color_info_ & 0xFFF, ((post_color_info_ >> 16) & 0xF) == 5, cov_color) &
                        ~written_depth;
      }
      if (write_depth && cov_depth[2] > cov_depth[0] && cov_depth[3] > cov_depth[1]) {
        if (!own_depth) {
          u.sync_depth.reset();
          u.sync_depth_info = post_depth_info_;
        }
        u.sync_depth |= inside_tiles(post_depth_info_ & 0xFFF, false, cov_depth) & ~written_color;
      }
    }
  }
}

// Session 45 (N3a).
bool NativeRenderer::ResolveTextureOnly(const draw_util::ResolveInfo& info,
                                        const draw_util::ResolveCopyShaderConstants& constants,
                                        VkDescriptorSet dest_set, uint32_t group_count_x, uint32_t group_count_y) {
  if (!replacing_ || !resolve_texture_stock_) return false;
  const draw_util::ResolveEdramInfo& edram = info.color_edram_info;
  if (info.IsCopyingDepth() || edram.is_depth || edram.format_is_64bpp ||
      edram.msaa_samples != xenos::MsaaSamples::k1X || info.copy_dest_info.copy_dest_array ||
      uint32_t(info.copy_dest_coordinate_info.copy_sample_select) != 0 || (info.copy_dest_base & 0xFFF) != 0 ||
      !EnsureResolveObjects() || resolve_tex_pipeline_ == VK_NULL_HANDLE) {
    return false;
  }
  // Any post target's write-back set: its EDRAM binding is what the shader reads; its images are bound, not read (no
  // target pixels: width and height 0).
  const PostTarget* target = nullptr;
  for (const PostTarget& t : post_targets_) {
    if (t.color != VK_NULL_HANDLE && t.writeback_set != VK_NULL_HANDLE &&
        t.writeback_edram == command_processor_.NativeEdramBuffer()) {
      target = &t;
      break;
    }
  }
  if (!target) return false;
  uint32_t texture_w = 0, texture_h = 0;
  void* texture_handle = nullptr;
  const VkDescriptorSet texture_set = command_processor_.NativeTextureForResolve(
      info.copy_dest_base >> 12, uint32_t(info.copy_dest_coordinate_info.pitch_aligned_div_32),
      uint32_t(info.copy_dest_info.copy_dest_endian), texture_w, texture_h, texture_handle);
  if (texture_set == VK_NULL_HANDLE) return false;
  const uint32_t pc[12] = {constants.dest_relative.edram_info.packed,
                           constants.dest_relative.coordinate_info.packed,
                           constants.dest_relative.dest_info.value,
                           constants.dest_relative.dest_coordinate_info.packed,
                           constants.dest_base,
                           0,
                           0,
                           0,
                           0,
                           0,
                           texture_w,
                           texture_h};
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  // The bound target images in the layouts the set names (as ResolveFromTarget), then back to attachments.
  VkImageMemoryBarrier images[2] = {};
  for (int i = 0; i < 2; ++i) {
    images[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    images[i].srcQueueFamilyIndex = images[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    images[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  }
  images[0].image = target->color;
  images[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  images[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  images[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  images[0].subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);
  images[1].image = target->ds;
  images[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  images[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  images[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
  images[1].subresourceRange =
      ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
  constexpr VkPipelineStageFlags kTargetStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                                 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  cb.CmdVkPipelineBarrier(kTargetStages, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 2, images);
  command_processor_.BindExternalComputePipeline(resolve_tex_pipeline_);
  const VkDescriptorSet sets[3] = {target->writeback_set, dest_set, texture_set};
  cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, resolve_tex_layout_, 0, 3, sets, 0, nullptr);
  cb.CmdVkPushConstants(resolve_tex_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
  command_processor_.SubmitBarriers(true);  // (also the texture's barrier, pushed by NativeTextureForResolve)
  cb.CmdVkDispatch(group_count_x, group_count_y, 1);
  texture_written_ = texture_handle;
  ++report_.textures_filled;
  ++report_.texture_only_resolves;
  images[0].srcAccessMask = images[1].srcAccessMask = 0;
  images[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  images[1].dstAccessMask =
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  for (int i = 0; i < 2; ++i) std::swap(images[i].oldLayout, images[i].newLayout);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, kTargetStages, 0, 0, nullptr, 0, nullptr, 2, images);
  return true;
}

void NativeRenderer::AfterResolveMarked() {
  if (alias_frames_left_ > 0) alias_texture_ = texture_written_;  // wwe2k14 patch (Session 66, G1.0): the destination
  if (!texture_written_) return;
  command_processor_.NativeTextureResolved(texture_written_);
  texture_written_ = nullptr;
}

// Session 45 (gpu_native_resolve).
bool NativeRenderer::EnsureResolveObjects() {
  if (resolve_pipeline_ != VK_NULL_HANDLE) return true;
  if (resolve_objects_failed_ || !EnsurePostObjects()) return false;
  resolve_objects_failed_ = true;  // until everything below exists
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  // Session 45 (N3a): two variants - the copy, and the copy that also stores into the texture it feeds (a third set:
  // the texture cache's direct-load storage image).
  for (int variant = 0; variant < 2; ++variant) {
    const bool texwrite = variant == 1;
    const VkDescriptorSetLayout texture_set_layout = command_processor_.NativeTextureSetLayout();
    if (texwrite && texture_set_layout == VK_NULL_HANDLE) break;  // no direct loads: the copy only
    std::vector<uint32_t> spirv;
    std::string error;
    const std::string source = std::string("#version 450\n") + (texwrite ? "#define NATIVE_TEXWRITE 1\n" : "") +
                               kNativeResolveComputeShader;
    if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT, source, spirv, error)) {
      REXGPU_ERROR("[native][resolve] shader compilation failed: {}", error);
      return false;
    }
    VkShaderModule& shader = texwrite ? resolve_tex_shader_ : resolve_shader_;
    shader = ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
    if (shader == VK_NULL_HANDLE) return false;
    const VkDescriptorSetLayout set_layouts[3] = {
        post_writeback_set_layout_,
        command_processor_.GetSingleTransientDescriptorLayout(
            VulkanCommandProcessor::SingleTransientDescriptorLayout::kStorageBufferCompute),
        texture_set_layout};
    VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 12 * sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = texwrite ? 3 : 2;
    layout_info.pSetLayouts = set_layouts;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    VkPipelineLayout& layout = texwrite ? resolve_tex_layout_ : resolve_layout_;
    if (dfn.vkCreatePipelineLayout(device, &layout_info, nullptr, &layout) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader,
                  "main", nullptr};
    info.layout = layout;
    VkPipeline& pipeline = texwrite ? resolve_tex_pipeline_ : resolve_pipeline_;
    if (dfn.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
      pipeline = VK_NULL_HANDLE;
      return false;
    }
  }
  resolve_objects_failed_ = false;
  REXGPU_INFO("[native][resolve] resolves from post targets ready");
  return true;
}

// Session 52 (gpu_native_truth_direct_resolve_scaled): kNativeResolveScaledComputeShader on resolve_layout_.
bool NativeRenderer::EnsureResolveScaledPipeline() {
  if (resolve_scaled_pipeline_ != VK_NULL_HANDLE) return true;
  if (resolve_scaled_failed_ || !EnsureResolveObjects() || resolve_layout_ == VK_NULL_HANDLE) return false;
  resolve_scaled_failed_ = true;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  // (Session 52: the second variant also writes the scaled texture - resolve_tex_layout_, when it exists.)
  for (int variant = 0; variant < 2; ++variant) {
    const bool texwrite = variant == 1;
    if (texwrite && resolve_tex_layout_ == VK_NULL_HANDLE) break;
    std::vector<uint32_t> spirv;
    std::string error;
    const std::string source = std::string("#version 450\n") + (texwrite ? "#define NATIVE_TEXWRITE 1\n" : "") +
                               kNativeResolveScaledComputeShader;
    if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT, source, spirv, error)) {
      REXGPU_ERROR("[native][resolve] scaled resolve shader compilation failed: {}", error);
      return false;
    }
    const VkShaderModule shader =
        ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
    if (shader == VK_NULL_HANDLE) return false;
    VkComputePipelineCreateInfo info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                  shader, "main", nullptr};
    info.layout = texwrite ? resolve_tex_layout_ : resolve_layout_;
    VkPipeline& pipeline = texwrite ? resolve_scaled_tex_pipeline_ : resolve_scaled_pipeline_;
    const bool ok = dfn.vkCreateComputePipelines(vulkan_device->device(), VK_NULL_HANDLE, 1, &info, nullptr,
                                                 &pipeline) == VK_SUCCESS;
    dfn.vkDestroyShaderModule(vulkan_device->device(), shader, nullptr);
    if (!ok) {
      pipeline = VK_NULL_HANDLE;
      return false;
    }
  }
  resolve_scaled_failed_ = false;
  return true;
}

// Session 49 (gpu_native_truth_skip_writeback): kNativeMaterializeShader, 32 and 64 bits a texel (the resolve pipelines'
// layout with the texture: set 1 the destination, set 2 the texture's storage view; 9 push constants).
bool NativeRenderer::EnsureMaterializePipeline() {
  if (materialize_pipeline_[0] != VK_NULL_HANDLE && materialize_pipeline_[1] != VK_NULL_HANDLE) return true;
  if (materialize_failed_ || !EnsureResolveObjects() || resolve_tex_layout_ == VK_NULL_HANDLE) return false;
  materialize_failed_ = true;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  for (int variant = 0; variant < 2; ++variant) {
    std::vector<uint32_t> spirv;
    std::string error;
    const std::string source =
        std::string("#version 450\n") + (variant ? "#define BPB64 1\n" : "") + kNativeMaterializeShader;
    if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT, source, spirv, error)) {
      REXGPU_ERROR("[native][wb] materialise shader compilation failed: {}", error);
      return false;
    }
    const VkShaderModule shader =
        ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
    if (shader == VK_NULL_HANDLE) return false;
    VkComputePipelineCreateInfo info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader,
                  "main", nullptr};
    info.layout = resolve_tex_layout_;
    const bool ok = dfn.vkCreateComputePipelines(vulkan_device->device(), VK_NULL_HANDLE, 1, &info, nullptr,
                                                 &materialize_pipeline_[variant]) == VK_SUCCESS;
    dfn.vkDestroyShaderModule(vulkan_device->device(), shader, nullptr);
    if (!ok) {
      materialize_pipeline_[variant] = VK_NULL_HANDLE;
      return false;
    }
  }
  materialize_failed_ = false;
  return true;
}

// Session 80 (gpu_native_truth_skip_writeback_depth): a texture-only depth copy's stencil, kept for its materialise -
// the image's stencil over the copied rectangle (TruthStoreSet's stencil view, binding 2), one byte a texel, rows of
// `pitch` texels (a multiple of 4) from dword `slot_dword` of the snapshot buffer. Four texels an invocation.
constexpr char kNativeStencilSnapShader[] = R"(
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 2) uniform usampler2D stencil_source;
layout(set = 1, binding = 0, std430) writeonly buffer DestBuffer { uint dest[]; };
layout(push_constant) uniform Constants {
  uint origin_x, origin_y, width, height, slot_dword, pitch;
} c;
void main() {
  uvec2 g = gl_GlobalInvocationID.xy;
  uint x = g.x * 4u;
  if (x >= c.width || g.y >= c.height) return;
  uint v = 0u;
  for (uint i = 0u; i < 4u; ++i) {
    if (x + i < c.width) {
      v |= (texelFetch(stencil_source, ivec2(int(c.origin_x + x + i), int(c.origin_y + g.y)), 0).r & 255u) << (8u * i);
    }
  }
  dest[c.slot_dword + ((g.y * c.pitch + x) >> 2u)] = v;
}
)";

// Session 80 (gpu_native_truth_skip_writeback_depth): a texture-only depth copy's output into guest memory - what
// kNativeResolveFull1xComputeShader's FULL1X_DEPTH copy writes: (the 20e4 depth << 8 | the stencil byte), endian
// swapped, at its destination texel's 2D tiled offset. The depth from the texture's 32-bit floats (the fill decoded the
// 20e4 exactly; DepthToFloat24 of such a float is the 20e4 again), the stencil from the record's slot (stencil_mode bit
// 8) or the constant in its low byte. One texel an invocation.
constexpr char kNativeMaterializeDepthShader[] = R"(
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Stencil { uint stencil[]; };
layout(set = 1, binding = 0, std430) writeonly buffer DestBuffer { uint dest[]; };
layout(set = 2, binding = 0, r32ui) uniform readonly uimage2D texture_in;
layout(push_constant) uniform Constants {
  uint x0, y0, w, h, pitch, dest_base, endian, tex_x0, tex_y0, stencil_mode, slot_dword, snap_pitch;
} c;
int XeTextureTiledOffset2D(ivec2 p, uint pitch_aligned, uint bpb_log2) {
  int macro = ((p.x >> 5) + (p.y >> 5) * int(pitch_aligned >> 5u)) << (bpb_log2 + 7u);
  int micro = ((p.x & 7) + ((p.y & 14) << 2)) << bpb_log2;
  int offset = macro + ((micro & -16) << 1) + (micro & 15) + ((p.y & 1) << 4);
  return ((offset & -512) << 3) + ((p.y & 16) << 7) + ((offset & 448) << 2) +
         (((((p.y & 8) >> 2) + (p.x >> 3)) & 3) << 6) + (offset & 63);
}
uint XeEndianSwap32(uint value, uint endian) {
  if (endian == 1u || endian == 2u) value = ((value & 16711935u) << 8u) | ((value & 4278255360u) >> 8u);
  if (endian == 2u || endian == 3u) value = (value << 16u) | (value >> 16u);
  return value;
}
uint DepthToFloat24(float z) {
  z = clamp(z, 0.0, 1.0);
  uint u = floatBitsToUint(z);
  uint b = u < 0x38800000u ? ((u & 0x7FFFFFu) | 0x800000u) >> min(113u - (u >> 23), 24u) : u - (112u << 23);
  b += 3u + ((b >> 3) & 1u);
  return (b >> 3) & 0xFFFFFFu;
}
void main() {
  uvec2 p = gl_GlobalInvocationID.xy;
  if (p.x >= c.w || p.y >= c.h) return;
  ivec2 d = ivec2(int(c.x0 + p.x), int(c.y0 + p.y));
  ivec2 t = ivec2(int(c.tex_x0) + d.x, int(c.tex_y0) + d.y);
  uint index = (uint(XeTextureTiledOffset2D(d, c.pitch, 2u)) + c.dest_base) >> 2u;
  uint s = c.stencil_mode & 255u;
  if ((c.stencil_mode & 256u) != 0u) {
    uint i = p.y * c.snap_pitch + p.x;
    s = (stencil[c.slot_dword + (i >> 2u)] >> (8u * (i & 3u))) & 255u;
  }
  float z = uintBitsToFloat(imageLoad(texture_in, t).r);
  dest[index] = XeEndianSwap32((DepthToFloat24(z) << 8u) | s, c.endian);
}
)";

bool NativeRenderer::EnsureDepthWritebackObjects() {
  if (stencil_snap_pipeline_ != VK_NULL_HANDLE && materialize_depth_pipeline_ != VK_NULL_HANDLE) return true;
  if (depth_wb_failed_ || !EnsureResolveScaledFullPipeline() || resolve_scaled_full_layout_ == VK_NULL_HANDLE) {
    return false;
  }
  const VkDescriptorSetLayout texture_set_layout = command_processor_.NativeTextureSetLayout();
  if (texture_set_layout == VK_NULL_HANDLE) return false;
  depth_wb_failed_ = true;  // until everything below exists
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  if (stencil_snap_ == VK_NULL_HANDLE &&
      !ui::vulkan::util::CreateDedicatedAllocationBuffer(vulkan_device, VkDeviceSize(kStencilSlots) * kStencilSlotSize,
                                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                         ui::vulkan::util::MemoryPurpose::kDeviceLocal, stencil_snap_,
                                                         stencil_snap_memory_)) {
    stencil_snap_ = VK_NULL_HANDLE;
    stencil_snap_memory_ = VK_NULL_HANDLE;
    return false;
  }
  if (materialize_depth_layout_ == VK_NULL_HANDLE) {
    const VkDescriptorSetLayout storage = command_processor_.GetSingleTransientDescriptorLayout(
        VulkanCommandProcessor::SingleTransientDescriptorLayout::kStorageBufferCompute);
    const VkDescriptorSetLayout set_layouts[3] = {storage, storage, texture_set_layout};
    VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 12 * sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 3;
    layout_info.pSetLayouts = set_layouts;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    if (dfn.vkCreatePipelineLayout(device, &layout_info, nullptr, &materialize_depth_layout_) != VK_SUCCESS) {
      materialize_depth_layout_ = VK_NULL_HANDLE;
      return false;
    }
  }
  const auto make = [&](const char* body, VkPipelineLayout layout, VkPipeline& out) {
    if (out != VK_NULL_HANDLE) return true;
    std::vector<uint32_t> spirv;
    std::string error;
    if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT, std::string("#version 450\n") + body, spirv,
                                               error)) {
      REXGPU_ERROR("[native][wb] depth write-back shader compilation failed: {}", error);
      return false;
    }
    const VkShaderModule shader =
        ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
    if (shader == VK_NULL_HANDLE) return false;
    VkComputePipelineCreateInfo info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader,
                  "main", nullptr};
    info.layout = layout;
    const bool ok = dfn.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &out) == VK_SUCCESS;
    dfn.vkDestroyShaderModule(device, shader, nullptr);
    if (!ok) out = VK_NULL_HANDLE;
    return ok;
  };
  if (!make(kNativeStencilSnapShader, resolve_scaled_full_layout_, stencil_snap_pipeline_) ||
      !make(kNativeMaterializeDepthShader, materialize_depth_layout_, materialize_depth_pipeline_)) {
    return false;
  }
  depth_wb_failed_ = false;
  return true;
}

VkDescriptorSet NativeRenderer::StencilSlotSet(int32_t slot) {
  if (stencil_snap_ == VK_NULL_HANDLE || slot < 0 || uint32_t(slot) >= kStencilSlots) return VK_NULL_HANDLE;
  const VkDescriptorSet set = command_processor_.AllocateSingleTransientDescriptor(
      VulkanCommandProcessor::SingleTransientDescriptorLayout::kStorageBufferCompute);
  if (set == VK_NULL_HANDLE) return VK_NULL_HANDLE;
  VkDescriptorBufferInfo buffer_info = {stencil_snap_, VkDeviceSize(slot) * kStencilSlotSize, kStencilSlotSize};
  VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = set;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo = &buffer_info;
  command_processor_.GetVulkanDevice()->functions().vkUpdateDescriptorSets(
      command_processor_.GetVulkanDevice()->device(), 1, &write, 0, nullptr);
  return set;
}
// Session 56 (gpu_native_truth_skip_writeback_scaled): kNativeMaterializeScaledShader on resolve_tex_layout_, on its own
// (review: a failure here leaves the 1x write-backs alone).
bool NativeRenderer::EnsureMaterializeScaledPipeline() {
  if (materialize_pipeline_[2] != VK_NULL_HANDLE) return true;
  if (materialize_scaled_failed_ || !EnsureResolveObjects() || resolve_tex_layout_ == VK_NULL_HANDLE) return false;
  materialize_scaled_failed_ = true;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  std::vector<uint32_t> spirv;
  std::string error;
  if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT,
                                             std::string("#version 450\n") + kNativeMaterializeScaledShader, spirv,
                                             error)) {
    REXGPU_ERROR("[native][wb] scaled materialise shader compilation failed: {}", error);
    return false;
  }
  const VkShaderModule shader =
      ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
  if (shader == VK_NULL_HANDLE) return false;
  VkComputePipelineCreateInfo info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader,
                "main", nullptr};
  info.layout = resolve_tex_layout_;
  const bool ok = dfn.vkCreateComputePipelines(vulkan_device->device(), VK_NULL_HANDLE, 1, &info, nullptr,
                                               &materialize_pipeline_[2]) == VK_SUCCESS;
  dfn.vkDestroyShaderModule(vulkan_device->device(), shader, nullptr);
  if (!ok) {
    materialize_pipeline_[2] = VK_NULL_HANDLE;
    return false;
  }
  materialize_scaled_failed_ = false;
  return true;
}

bool NativeRenderer::ResolveFromTarget(const draw_util::ResolveInfo& info,
                                       const draw_util::ResolveCopyShaderConstants& constants,
                                       VkDescriptorSet dest_set, uint32_t group_count_x, uint32_t group_count_y) {
  const int32_t index = resolve_target_;
  resolve_target_ = -1;
  if (!replacing_ || !native_resolve_ || index < 0 || size_t(index) >= post_targets_.size()) return false;
  const PostTarget& target = post_targets_[index];
  // Colour render target 0 of the surface the segment drew: 1x, 32bpp k_8_8_8_8, the same base and pitch; a 2D
  // destination of the copied pixels (copy sample select 0).
  const draw_util::ResolveEdramInfo& edram = info.color_edram_info;
  if (info.rb_copy_control.copy_src_select != 0 || edram.is_depth || edram.format_is_64bpp ||
      edram.msaa_samples != xenos::MsaaSamples::k1X || uint32_t(edram.format) != 0 ||
      ((resolve_color_info_ >> 16) & 0xF) != 0 || info.color_original_base != (resolve_color_info_ & 0xFFF) ||
      edram.pitch_tiles != target.pitch_tiles || info.copy_dest_info.copy_dest_array ||
      uint32_t(info.copy_dest_coordinate_info.copy_sample_select) != 0 || target.target_class > 1 ||
      target.writeback_set == VK_NULL_HANDLE || target.writeback_edram != command_processor_.NativeEdramBuffer() ||
      !EnsureResolveObjects()) {
    // Session 45 (gpu_native_defer_writeback): the stock resolve reads the EDRAM: what is pending goes in first.
    WritePendingAll();
    return false;
  }
  // The resolve's first pixel in the surface (= the target's pixels): the adjusted base's offset from the surface's
  // own, plus the offset within it (as the resolve computes it).
  const uint32_t pitch_tiles = std::max(1u, uint32_t(edram.pitch_tiles));
  // (Review fix: base_tiles wraps at the EDRAM's 2048 tiles.)
  const uint32_t tile_delta = (uint32_t(edram.base_tiles) - info.color_original_base) & 2047u;
  // Session 45 (gpu_native_fuse_writeback): the segment OnResolve did not write back is written by this dispatch
  // (each pixel it drew stored into the EDRAM as the write-back stores it) when its write-back rectangle lies inside the
  // resolve's; otherwise it is written back now, as before.
  bool fuse = false;
  if (fuse_writeback_ && !defer_writeback_ && post_targets_[index].pending) {
    PostTarget& t = post_targets_[index];
    const uint32_t rx0 = (tile_delta % pitch_tiles) * 80 + uint32_t(info.coordinate_info.edram_offset_x_div_8) * 8;
    const uint32_t ry0 = (tile_delta / pitch_tiles) * 16 + uint32_t(info.coordinate_info.edram_offset_y_div_8) * 8;
    const uint32_t rx1 = rx0 + uint32_t(info.coordinate_info.width_div_8) * 8, ry1 = ry0 + info.height_div_8 * 8;
    if ((t.pending_color_info & 0xF0FFF) == (resolve_color_info_ & 0xF0FFF) && t.pending_rect[0] >= int32_t(rx0) &&
        t.pending_rect[1] >= int32_t(ry0) && t.pending_rect[2] <= int32_t(rx1) && t.pending_rect[3] <= int32_t(ry1)) {
      t.pending = false;
      fuse = true;
      ++report_.writebacks_fused;
    } else {
      WritePending(index);
    }
  }
  // Session 45 (gpu_native_defer_writeback): the pixels the segment did not draw come from the EDRAM - other pending
  // targets over them go in first, unless the segment provably drew the whole rectangle.
  if (defer_writeback_) {
    const uint32_t rx0 = (tile_delta % pitch_tiles) * 80 + uint32_t(info.coordinate_info.edram_offset_x_div_8) * 8;
    const uint32_t ry0 = (tile_delta / pitch_tiles) * 16 + uint32_t(info.coordinate_info.edram_offset_y_div_8) * 8;
    const uint32_t rx1 = rx0 + uint32_t(info.coordinate_info.width_div_8) * 8, ry1 = ry0 + info.height_div_8 * 8;
    const int32_t* cov = target.pending ? target.pending_cov : post_cov_color_;
    const bool covered = cov[0] <= int32_t(rx0) && cov[1] <= int32_t(ry0) && cov[2] >= int32_t(rx1) &&
                         cov[3] >= int32_t(ry1);
    if (!covered) {
      WritePendingOverlapping(EdramTiles(info.color_original_base, pitch_tiles, false, rx0, ry0, rx1, ry1), index);
    }
  }
  // Session 45 (N3a, gpu_native_resolve_texture): the texture this resolve feeds, when it exists and qualifies (the
  // texture cache's checks), is filled by the same dispatch and is up to date after it (AfterResolveMarked).
  uint32_t texture_w = 0, texture_h = 0;
  void* texture_handle = nullptr;
  VkDescriptorSet texture_set = VK_NULL_HANDLE;
  if (resolve_texture_ && resolve_tex_pipeline_ != VK_NULL_HANDLE && (info.copy_dest_base & 0xFFF) == 0) {
    texture_set = command_processor_.NativeTextureForResolve(
        info.copy_dest_base >> 12, uint32_t(info.copy_dest_coordinate_info.pitch_aligned_div_32),
        uint32_t(info.copy_dest_info.copy_dest_endian), texture_w, texture_h, texture_handle);
  }
  const uint32_t pc[12] = {constants.dest_relative.edram_info.packed,
                           constants.dest_relative.coordinate_info.packed,
                           constants.dest_relative.dest_info.value,
                           constants.dest_relative.dest_coordinate_info.packed,
                           constants.dest_base,
                           (tile_delta % pitch_tiles) * 80 + uint32_t(info.coordinate_info.edram_offset_x_div_8) * 8,
                           (tile_delta / pitch_tiles) * 16 + uint32_t(info.coordinate_info.edram_offset_y_div_8) * 8,
                           target.width,
                           target.height,
                           target.target_class | (fuse ? 0x100u : 0u),
                           texture_w,
                           texture_h};
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  // The target's colour and coverage become readable (as for the write-back); then back to attachments.
  VkImageMemoryBarrier images[2] = {};
  for (int i = 0; i < 2; ++i) {
    images[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    images[i].srcQueueFamilyIndex = images[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    images[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  }
  images[0].image = target.color;
  images[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  images[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  images[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  images[0].subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);
  images[1].image = target.ds;
  images[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  images[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  images[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
  images[1].subresourceRange =
      ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
  constexpr VkPipelineStageFlags kTargetStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                                 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  cb.CmdVkPipelineBarrier(kTargetStages | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 0, nullptr, 0, nullptr, 2, images);
  // Session 45 (gpu_native_fuse_writeback): the dispatch also writes the EDRAM - every earlier user of it before
  // (the render target cache only made it ready for a read), every later one after (below).
  constexpr VkPipelineStageFlags kEdramStages =
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
  constexpr VkAccessFlags kEdramAccesses = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                           VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  if (fuse) {
    VkMemoryBarrier before = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, kEdramAccesses,
                              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    cb.CmdVkPipelineBarrier(kEdramStages, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0,
                            nullptr);
  }
  const bool texwrite = texture_set != VK_NULL_HANDLE;
  const VkPipelineLayout layout = texwrite ? resolve_tex_layout_ : resolve_layout_;
  command_processor_.BindExternalComputePipeline(texwrite ? resolve_tex_pipeline_ : resolve_pipeline_);
  const VkDescriptorSet sets[3] = {target.writeback_set, dest_set, texture_set};
  cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, texwrite ? 3 : 2, sets, 0, nullptr);
  cb.CmdVkPushConstants(layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
  command_processor_.SubmitBarriers(true);  // (also the texture's barrier, pushed by NativeTextureForResolve)
  cb.CmdVkDispatch(group_count_x, group_count_y, 1);
  if (texwrite) {
    texture_written_ = texture_handle;
    ++report_.textures_filled;
  }
  if (fuse) {
    VkMemoryBarrier after = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, kEdramAccesses};
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, kEdramStages, 0, 1, &after, 0, nullptr, 0, nullptr);
    native_stale_ = true;  // as after a write-back: the scene's targets share these EDRAM tiles
  }
  images[0].srcAccessMask = images[1].srcAccessMask = 0;
  images[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  images[1].dstAccessMask =
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  for (int i = 0; i < 2; ++i) std::swap(images[i].oldLayout, images[i].newLayout);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, kTargetStages, 0, 0, nullptr, 0, nullptr, 2, images);
  ++report_.native_resolves;
  ++slots_[SlotOf(frame_)].native_resolves;  // Session 46 (the EDRAM census)
  return true;
}

// Session 45 (gpu_native_post_sync).
std::bitset<2048> NativeRenderer::EdramTiles(uint32_t base_tiles, uint32_t pitch_tiles, bool is_64bpp, uint32_t x0,
                                             uint32_t y0, uint32_t x1, uint32_t y1) {
  std::bitset<2048> tiles;
  if (x1 <= x0 || y1 <= y0 || !pitch_tiles) return tiles;
  // Session 87 (gpu_native_owned_frame_tiles_memo, CPU): a pure function of its arguments - the last results are kept (a 1280x720
  // box is 720 tiles set one at a time, and the owned / truth checks of a frame ask for the same few boxes again and again).
  struct Memo {
    uint64_t k0 = ~0ull, k1 = ~0ull;
    std::bitset<2048> tiles;
  };
  static thread_local Memo memo[16];
  Memo* slot = nullptr;
  uint64_t k0 = 0, k1 = 0;
  if (REXCVAR_GET(gpu_native_owned_frame_tiles_memo) && base_tiles <= 0xFFFFFFu && pitch_tiles <= 0xFFFFu && x0 <= 0xFFFFu &&
      y0 <= 0xFFFFu && x1 <= 0xFFFFu && y1 <= 0xFFFFu) {
    k0 = uint64_t(base_tiles) | (uint64_t(pitch_tiles) << 24) | (uint64_t(is_64bpp ? 1 : 0) << 40);
    k1 = uint64_t(x0) | (uint64_t(y0) << 16) | (uint64_t(x1) << 32) | (uint64_t(y1) << 48);
    const uint64_t h = (k0 * 0x9E3779B97F4A7C15ull) ^ (k1 * 0xC2B2AE3D27D4EB4Full);
    slot = &memo[(h >> 40) & 15];
    if (slot->k0 == k0 && slot->k1 == k1) return slot->tiles;
  }
  const uint32_t tile_w = is_64bpp ? 40 : 80;
  const uint32_t row_tiles = pitch_tiles * (is_64bpp ? 2 : 1);
  const uint32_t c0 = x0 / tile_w, c1 = std::min((x1 - 1) / tile_w, row_tiles - 1);
  const uint32_t r0 = y0 / 16, r1 = (y1 - 1) / 16;
  if (c0 > c1) return tiles;
  if (uint64_t(r1 - r0 + 1) * row_tiles >= 2048) {
    tiles.set();
  } else {
    for (uint32_t r = r0; r <= r1; ++r) {
      for (uint32_t c = c0; c <= c1; ++c) tiles.set((base_tiles + r * row_tiles + c) & 2047);
    }
  }
  if (slot) {
    slot->k0 = k0;
    slot->k1 = k1;
    slot->tiles = tiles;
  }
  return tiles;
}

void NativeRenderer::InvalidatePostSync(const std::bitset<2048>& tiles, int32_t except) {
  for (size_t i = 0; i < post_targets_.size(); ++i) {
    if (int32_t(i) == except) continue;
    post_targets_[i].sync_color &= ~tiles;
    post_targets_[i].sync_depth &= ~tiles;
  }
  DropMirrors(tiles);  // Session 45 (clear mirrors)
}

void NativeRenderer::InvalidateAllPostSync() {
  for (PostTarget& t : post_targets_) {
    t.sync_color.reset();
    t.sync_depth.reset();
    t.mirror_color.valid = t.mirror_depth.valid = false;  // Session 45 (clear mirrors)
  }
  scene_mirror_color_.valid = scene_mirror_depth_.valid = false;
}

// Session 45 (clear mirrors).
void NativeRenderer::DropMirrors(const std::bitset<2048>& tiles) {
  const auto drop = [&](PostTarget::Mirror& m) {
    if (m.valid && (m.tiles & tiles).any()) m.valid = false;
  };
  for (PostTarget& t : post_targets_) {
    drop(t.mirror_color);
    drop(t.mirror_depth);
  }
  drop(scene_mirror_color_);
  drop(scene_mirror_depth_);
}

// A fast clear's fill (its EDRAM writes already taken out of the sync sets): the post targets of its pitch, and the
// scene's targets, remember it for their next segment (1x targets over the same tiles: a 4x view's samples are their
// pixels).
void NativeRenderer::RecordMirror(const Draw& fill) {
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  if (!clear_mirrors_ || !fill.fast_clear || scale_x != 1 || scale_y != 1) return;
  const auto make = [&](uint32_t info, uint32_t base, uint32_t src) {
    PostTarget::Mirror m;
    m.valid = true;
    m.info = info;
    m.src = src;
    for (int i = 0; i < 4; ++i) m.rect[i] = int32_t(fill.fill_rect[i]);
    m.tiles = EdramTiles(base, fill.fill_pitch_tiles, false, fill.fill_rect[0], fill.fill_rect[1], fill.fill_rect[2],
                         fill.fill_rect[3]);
    return m;
  };
  // (Session 45: not for a 64bpp colour fill - a colour mirror is one 32-bit dword.)
  if (fill.fill_color_base != UINT32_MAX && !fill.fill_color_64bpp) {
    const PostTarget::Mirror m = make(fill.color_info, fill.fill_color_base, fill.fill_src_color);
    const uint32_t target_class = PostTargetClass(fill);
    for (PostTarget& t : post_targets_) {
      if (t.pitch_tiles == fill.fill_pitch_tiles && t.target_class == target_class) t.mirror_color = m;
    }
    if (scene_key_.valid && scene_color_info_last_ != UINT32_MAX &&
        (scene_color_info_last_ & 0xF0FFF) == (fill.color_info & 0xF0FFF) &&
        ((scene_key_.surface_info & 0x3FFF) + 79) / 80 == fill.fill_pitch_tiles) {
      scene_mirror_color_ = m;
    }
  }
  // (Review fix: only a fill that replaces the depth - a stencil-only clear keeps each pixel's own.)
  if (fill.fill_depth_base != UINT32_MAX && (fill.fill_keep_depth & 0xFFFFFF00u) == 0) {
    const PostTarget::Mirror m = make(fill.depth_info, fill.fill_depth_base, fill.fill_src_depth);
    for (PostTarget& t : post_targets_) {
      if (t.pitch_tiles == fill.fill_pitch_tiles) t.mirror_depth = m;
    }
    if (scene_key_.valid && (scene_key_.depth_info & 0x10FFF) == (fill.depth_info & 0x10FFF) &&
        ((scene_key_.surface_info & 0x3FFF) + 79) / 80 == fill.fill_pitch_tiles) {
      scene_mirror_depth_ = m;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The frame
// ---------------------------------------------------------------------------------------------------------------

void NativeRenderer::NotePostCover(const int32_t rect[4]) {
  // The largest proven rectangle of the segment (a union of several is not tracked).
  const int64_t area = int64_t(rect[2] - rect[0]) * (rect[3] - rect[1]);
  const int64_t best = post_cover_[2] > post_cover_[0] ? int64_t(post_cover_[2] - post_cover_[0]) *
                                                           (post_cover_[3] - post_cover_[1])
                                                     : 0;
  if (area > best) std::copy(rect, rect + 4, post_cover_);
}

void NativeRenderer::NoteResolveCover(uint32_t surface_info, uint32_t color_info, const int32_t rect[4]) {
  if (!(post_dirty_ || !post_batch_.empty()) || surface_info != post_surface_info_ || color_info != post_color_info_) {
    return;
  }
  FrameSlot& slot = slots_[SlotOf(frame_)];
  ++slot.cover_resolves;
  if (post_cover_[0] <= rect[0] && post_cover_[1] <= rect[1] && post_cover_[2] >= rect[2] &&
      post_cover_[3] >= rect[3]) {
    ++slot.cover_resolves_full;
  }
}

void NativeRenderer::CensusDraw(uint64_t vertex_shader_hash, uint64_t pixel_shader_hash, bool memexport) {
  ++census_draws_;
  census_memexport_draws_ += memexport ? 1 : 0;
  auto note = [&](std::unordered_map<uint64_t, uint32_t>& seen, uint64_t hash, uint32_t& frame_count,
                  uint32_t& new_count, uint64_t tag) {
    if (!hash) return;
    // Distinct per frame: a small list, cleared at the frame's end.
    const uint64_t tagged = hash ^ tag;
    if (std::find(census_frame_hashes_.begin(), census_frame_hashes_.end(), tagged) == census_frame_hashes_.end()) {
      census_frame_hashes_.push_back(tagged);
      ++frame_count;
    }
    if (seen.emplace(hash, uint32_t(frame_)).second) ++new_count;
  };
  note(census_vs_seen_, vertex_shader_hash, census_frame_vs_, census_new_vs_, 0);
  note(census_ps_seen_, pixel_shader_hash, census_frame_ps_, census_new_ps_, 0x5A5A5A5A5A5A5A5Aull);
}

bool NativeRenderer::Wants(const Draw& draw) const {
  // Session 54 (review): on the host render target path only truth frames draw natively - a frame that is not one (or
  // the rest of one after a fallback) draws nothing beside the emulated renderer.
  if (!replacing_ && command_processor_.NativeEdramBuffer() == VK_NULL_HANDLE) return false;
  // After the scene's depth resolve the frame goes on with post-processing and the HUD, partly in the same render
  // target: those stay emulated.
  // Session 42 (gpu_native_after_scene, replacing frames): also after the colour resolve, and the expanded and
  // screen-space draws. Session 43 (gpu_native_scene_extras): the expanded and screen-space draws only.
  const bool after = replacing_ && after_scene_;
  const bool extras = replacing_ && (after_scene_ || scene_extras_);
  return (!scene_done_this_frame_ || after) && scene_key_.Matches(draw) &&
         (draw.raster.supported || (extras && draw.raster.expanded)) &&
         (all_draws_ || (draw.raster.depth_test && draw.raster.depth_write)) && (!draw.screen_space || extras) &&
         !draw.memexport && draw.vertex_count && draw.vertex_shader != VK_NULL_HANDLE &&
         draw.layout != VK_NULL_HANDLE;
}

// What keeps a scene-target draw emulated in a replacing frame: what the native path cannot draw yet, or cannot draw
// exactly as the emulated renderer would (it has no stencil, one colour target, k_8_8_8_8 only, one sample).
uint32_t NativeRenderer::NotOwnedReason(const Draw& draw) const {
  const bool after = replacing_ && after_scene_;  // Session 42
  const bool extras = replacing_ && (after_scene_ || scene_extras_);  // Session 43
  if (scene_done_this_frame_ && !after) return kReasonAfterScene;
  if (draw.screen_space && !extras) return kReasonScreenSpace;
  if (!draw.raster.supported && !(extras && draw.raster.expanded)) return kReasonNotTriangles;
  if (draw.memexport) return kReasonMemexport;
  if (!all_draws_ && !(draw.raster.depth_test && draw.raster.depth_write)) return kReasonNoDepthWrite;
  if (!draw.vertex_count || draw.vertex_shader == VK_NULL_HANDLE || draw.layout == VK_NULL_HANDLE) {
    return kReasonNoShader;
  }
  if (draw.raster.stencil) return kReasonStencil;
  if (draw.color_mask_all & ~uint32_t(0xF)) return kReasonOtherTargets;
  // A draw that writes colour: RB_COLOR_INFO color_format (bits 16-19) 0, k_8_8_8_8; one render target 0 per frame
  // (its EDRAM base). (A depth-only draw's colour registers may be stale: not looked at.)
  if ((draw.color_mask_all & 0xF) &&
      (((draw.color_info >> 16) & 0xF) != 0 || (scene_color_info_valid_ && draw.color_info != scene_color_info_))) {
    return kReasonColorFormat;
  }
  if ((draw.surface_info >> 16) & 3) return kReasonMsaa;  // RB_SURFACE_INFO msaa_samples
  if (draw.occlusion_query) return kReasonOcclusion;
  // A guest pixel shader must have its host-render-target translation (also for depth only: it may discard).
  if (draw.pixel_shader_hash && draw.pixel_shader == VK_NULL_HANDLE) return kReasonNoShader;
  return kOwnable;
}

// Session 45 (gpu_native_post_sync): the scene batch's first draw provably writes every pixel of the scene - colour
// (all channels, no blending) and depth (test always) - so nothing of the EDRAM is read (entrances restore their scene
// from its resolved textures with such a quad before drawing more of it).
bool NativeRenderer::SceneBatchCovers() const {
  if (batch_.empty()) return false;
  const Draw& first = batch_.front();
  uint32_t res_x = 1, res_y = 1;
  command_processor_.NativeResolutionScale(res_x, res_y);
  const bool depth_written =
      first.raster.depth_write && (!first.raster.depth_test || first.raster.depth_compare == VK_COMPARE_OP_ALWAYS);
  const bool color_written = !PostNeedsDst(first) && first.pixel_shader != VK_NULL_HANDLE;
  return depth_written && color_written && first.cover[2] > first.cover[0] && first.cover[0] <= 0 &&
         first.cover[1] <= 0 && int64_t(first.cover[2]) * res_x >= int64_t(scene_width_) &&
         int64_t(first.cover[3]) * res_y >= int64_t(scene_height_);
}

bool NativeRenderer::Claim(Draw& draw) {
  draw.owned = false;
  draw.declined = false;
  draw.post = false;
  draw.pipeline = VK_NULL_HANDLE;
  declined_pending_ = false;
  claim_retry_ = false;
  draw.truth = false;
  draw.claim_refused = 0;  // Session 92 (M2a)
  // Session 46: a truth frame claims every draw it can draw (a refusal ends truth for the frame: ClaimTruth).
  if (truth_ && replacing_) return owned_surf_mode_ ? OwnedClaim(draw) : ClaimTruth(draw);  // (Session 92, M2b: under the latch the owned entry, no truth claim)
  // Session 42: after the scene's colour resolve, post-processing draws (gpu_native_post); with
  // gpu_native_after_scene the other scene-target draws go on to the scene's targets.
  // Session 45 (gpu_native_all): after the scene, a draw into the scene's surface that reads what is under it stays
  // in the scene's targets while they hold that surface (or will, after a covering first draw in the batch): a post
  // target would load all of it from the EDRAM first (Bryan's entrance: a restore quad, then blended strips).
  if (replacing_ && (scene_done_this_frame_ || !scene_key_.Matches(draw)) &&
      (!after_scene_ || !scene_key_.Matches(draw) || PostReason(draw, true) == kPostOk)) {
    const bool reads = PostNeedsDst(draw) ||
                       (draw.raster.depth_test && draw.raster.depth_compare != VK_COMPARE_OP_ALWAYS);
    const bool stay_in_scene = AllOn(kAllAfterScene) && post_sync_ && scene_done_this_frame_ && scene_key_.Matches(draw) &&
                               reads && (!native_stale_ || SceneBatchCovers()) &&
                               NotOwnedReason(draw) == kOwnable;
    if (!stay_in_scene) return ClaimPost(draw);
  }
  if (!replacing_ || !scene_key_.Matches(draw) || NotOwnedReason(draw) != kOwnable) return false;
  // Session 42: a post target shares the scene's EDRAM tiles: its draws go into the EDRAM first.
  if (!post_batch_.empty() || post_dirty_) {
    claim_retry_ = true;
    return false;
  }
  // Session 40: the segment this draw belongs to, and whether it is drawn natively (decided at its first claimable
  // draw from the same segment of the previous frame; an unknown segment is native). The decision holds until the
  // segment ends (BeforeEmulatedEdramAccess, a clearing resolve, the frame's end), as the load it would pay for.
  if (!seg_active_) {
    seg_active_ = true;
    seg_index_ = seg_count_++;
    if (seg_draws_.size() < seg_count_) seg_draws_.resize(seg_count_, 0);
    const uint32_t min_draws =
        AllOn(kAllSceneMinDraws) ? 0u : uint32_t(std::max(0, REXCVAR_GET(gpu_native_min_draws)));
    seg_native_ = !min_draws || seg_index_ >= seg_draws_prev_.size() || seg_draws_prev_[seg_index_] >= min_draws;
    ++(seg_native_ ? slots_[SlotOf(frame_)].segments_native : slots_[SlotOf(frame_)].segments_emulated);
  }
  ++seg_draws_[seg_index_];
  if (!seg_native_) {
    // Emulated as a whole: the emulated draw that follows does not end the segment.
    declined_pending_ = true;
    draw.declined = true;
    return false;
  }
  // The pipeline now: a claimed draw is never drawn by the emulated renderer.
  draw.pipeline = GetPipeline(draw);
  if (draw.pipeline == VK_NULL_HANDLE) return false;
  if (!scene_color_info_valid_ && (draw.color_mask_all & 0xF)) {
    // The frame's first colour draw gives the colour surface. If ours were already loaded or drawn without it (depth
    // only so far), this draw stays emulated: the write-back before it covers the depth, and the next load reads the
    // colour too.
    scene_color_info_ = draw.color_info;
    scene_color_info_valid_ = true;
    scene_color_info_last_ = draw.color_info;  // Session 45 (clear mirrors)
    if (!batch_.empty() || native_dirty_ || !native_stale_) return false;
  }
  draw.owned = true;
  return true;
}

void NativeRenderer::BeforeEmulatedEdramAccess(const Draw* draw, bool emulated_follows) {
  if (!replacing_) return;
  // Session 46: in a truth frame, something emulated follows (or a claim was refused): the boundary write-out, then
  // the rest of the frame is exact.
  if (truth_) {
    TruthFallback(truth_fallback_reason_ ? truth_fallback_reason_ : 1u, draw);
    if (!emulated_follows) return;  // a refused claim: the caller claims again (exact now)
  }
  WriteBack();  // only if our targets hold something the EDRAM does not
  native_stale_ = true;
  PostWriteBack();  // Session 42
  post_segment_open_ = false;
  // Session 45 (gpu_native_defer_writeback): an emulated draw may read the EDRAM anywhere - everything pending goes
  // in. A fast clear's fill reads only its first pixel's dword (drawn by the emulated draw) and overwrites its
  // rectangle: pending colour wholly inside the colour tiles it fills is dead (dropped), any other under it goes in.
  // (Not for a switch between our own targets, ClaimWantsWriteBack: nothing reads the EDRAM then.)
  if (defer_writeback_ && emulated_follows) {
    if (draw && draw->fast_clear) {
      const uint32_t* r = draw->fill_rect;
      std::bitset<2048> any, full;
      // (Session 45: a 64bpp colour fill's tiles are 40 samples wide.)
      const bool c64 = draw->fill_color_64bpp;
      if (draw->fill_color_base != UINT32_MAX) {
        any |= EdramTiles(draw->fill_color_base, draw->fill_pitch_tiles, c64, r[0], r[1], r[2], r[3]);
      }
      if (draw->fill_depth_base != UINT32_MAX) {
        any |= EdramTiles(draw->fill_depth_base, draw->fill_pitch_tiles, false, r[0], r[1], r[2], r[3]);
      }
      if (draw->fill_color_base != UINT32_MAX) {
        const uint32_t tw = c64 ? 40 : 80;
        const uint32_t c0 = (r[0] + tw - 1) / tw, c1 = r[2] / tw, r0 = (r[1] + 15) / 16, r1 = r[3] / 16;
        if (c1 > c0 && r1 > r0) {
          full = EdramTiles(draw->fill_color_base, draw->fill_pitch_tiles, c64, c0 * tw, r0 * 16, c1 * tw, r1 * 16);
        }
      }
      // The emulated draw's own first pixel (its dword is read by the fill) is in `any`.
      for (size_t i = 0; i < post_targets_.size(); ++i) {
        PostTarget& t = post_targets_[i];
        if (!t.pending || (t.pending_tiles & any).none()) continue;
        if ((t.pending_tiles & ~full).none()) {
          t.pending = false;
          ++report_.pending_dropped;
        } else {
          WritePending(int32_t(i));
        }
      }
    } else {
      WritePendingAll();
    }
  }
  // Session 40: an emulated draw ends the segment, unless it is a claimable draw of an emulated segment.
  if (declined_pending_) {
    declined_pending_ = false;
  } else {
    seg_active_ = false;
  }
}

void NativeRenderer::OnDrawPre(const Draw& draw) {
  // The emulated scene segment's start (side-car frames: both renderers draw the same draws).
  if (!enabled_ || replacing_ || segment_open_ || !query_pool_ || !slot_queries_ok_ || !Wants(draw)) return;
  FrameSlot& slot = slots_[SlotOf(frame_)];
  if (slot.queries + 3 > QueryLimit()) return;
  FrameSlot::Segment& segment = slot.segments.emplace_back();
  segment.emulated_begin = slot.queries++;
  command_processor_.deferred_command_buffer().CmdVkWriteTimestamp(
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, SlotOf(frame_) * kQueriesPerFrame + segment.emulated_begin);
  segment_open_ = true;
}

// Session 45 (the frame census): the categories follow Claim's two paths - the scene's (its target before the scene's
// colour resolve, after it too with gpu_native_after_scene) and the post targets' (everything else).
uint32_t NativeRenderer::EmulatedCategory(const Draw& draw) const {
  if (draw.fast_clear) return kEmuClear;
  if (!replacing_) return kEmuOther;
  if (scene_key_.Matches(draw) && (!scene_done_this_frame_ || after_scene_)) {
    if (draw.declined) return kEmuSmall;
    switch (NotOwnedReason(draw)) {
      case kReasonScreenSpace:
      case kReasonNotTriangles:
        return kEmuPrimitive;
      case kReasonStencil:
        return kEmuStencil;
      case kReasonOtherTargets:
      case kReasonColorFormat:
        return kEmuFormat;
      case kReasonMsaa:
        return kEmuMsaa;
      case kReasonNoShader:
        return kEmuNoShader;
      case kReasonSmallSegment:
        return kEmuSmall;
      default:
        return kEmuOther;
    }
  }
  switch (PostReason(draw, true)) {
    case kPostPrimitive:
      return kEmuPrimitive;
    case kPostMsaa:
      return kEmuMsaa;
    case kPostOtherTargets:
    case kPostColorFormat:
      return kEmuFormat;
    case kPostNoColor:
    case kPostDepth:
    case kPostDepthFormat:
    case kPostDepthBias:
    case kPostDepthClamp:
      return kEmuDepth;
    case kPostStencil:
      return kEmuStencil;
    case kPostBlend:
      return kEmuBlend;
    case kPostNoShader:
      return kEmuNoShader;
    case kPostSmall:
      return kEmuSmall;
    case kPostOk:
      return kEmuSwitch;  // eligible, left because the other target held unwritten draws
    default:
      return kEmuOther;
  }
}

void NativeRenderer::NoteEmulatedRepeat(const Draw& draw) {
  if (!enabled_) return;
  FrameSlot& slot = slots_[SlotOf(frame_)];
  ++slot.frame_draws;
  ++slot.frame_emulated[EmulatedCategory(draw)];
  if (post_sync_) InvalidateAllPostSync();  // an emulated draw: the EDRAM changed somewhere
}

void NativeRenderer::OnDraw(const Draw& draw) {
  if (!enabled_) return;
  // Session 45: the frame census (every guest draw, merged instances one by one).
  {
    FrameSlot& census = slots_[SlotOf(frame_)];
    ++census.frame_draws;
    if (draw.owned) {
      ++(draw.post ? census.frame_native_post : census.frame_native_scene);
    } else {
      const uint32_t category = EmulatedCategory(draw);
      ++census.frame_emulated[category];
      // Session 45 (gpu_native_post_sync): an emulated draw changed the EDRAM - a fill exactly where it filled,
      // anything else anywhere.
      if (post_sync_) {
        if (draw.fast_clear) {
          std::bitset<2048> tiles;
          if (draw.fill_color_base != UINT32_MAX) {  // (Session 45: 64bpp colour fills)
            tiles |= EdramTiles(draw.fill_color_base, draw.fill_pitch_tiles, draw.fill_color_64bpp, draw.fill_rect[0],
                                draw.fill_rect[1], draw.fill_rect[2], draw.fill_rect[3]);
          }
          if (draw.fill_depth_base != UINT32_MAX) {
            tiles |= EdramTiles(draw.fill_depth_base, draw.fill_pitch_tiles, false, draw.fill_rect[0], draw.fill_rect[1],
                                draw.fill_rect[2], draw.fill_rect[3]);
          }
          InvalidatePostSync(tiles, -1);
          RecordMirror(draw);  // Session 45 (clear mirrors)
        } else {
          InvalidateAllPostSync();
        }
      }
      if (segment_log_frames_ > 0 && replacing_) {  // Session 45: the segment log
        REXGPU_INFO("[native][seg] frame {} emulated draw (category {}): base {} depth base {} pitch {} msaa {} format "
                    "{} primitive {} scissor {},{} {}x{} mask 0x{:X} depth test {} write {} stencil {} fast clear {}",
                    frame_, category, draw.color_info & 0xFFF, draw.depth_info & 0xFFF, draw.surface_info & 0x3FFF,
                    (draw.surface_info >> 16) & 3, (draw.color_info >> 16) & 0xF, draw.guest_primitive,
                    draw.scissor.offset.x, draw.scissor.offset.y, draw.scissor.extent.width,
                    draw.scissor.extent.height, draw.color_mask_all, draw.raster.depth_test, draw.raster.depth_write,
                    draw.raster.stencil, draw.fast_clear);
      }
    }
  }
  // Session 46 (M5): a truth draw is recorded at once, at its place in the command stream (recorded later, with a
  // batch at the emulated render pass's end, some draws read inputs already changed: the crowd's maps, run 3046).
  if (draw.owned && draw.truth) {
    ++truth_counts_.draws;
    if (owned_surf_mode_) {  // Session 92, M2b: the owned entry - no truth recorder (M0 "truth entries" stays 0)
      OwnedEnter(draw);
      return;
    }
    TruthRecordDraw(draw);
    return;
  }
#if REX_TRUE_NATIVE_ONLY
  // Session 92, M2d (Part 2): the proof build has no emulated draw path - every draw of a TRUE NATIVE frame is an owned one (OwnedEnter above); the post batch and the scene batch
  // below are never filled. (FlushPost / FlushScene and their callers are still compiled: listed in the M2d hand-back.)
  return;
#else
  const auto cpu_start = std::chrono::steady_clock::now();  // (Session 47: after the truth draws - they time themselves)
  // Session 42: a claimed post draw waits for the end of the emulated render pass in its own batch.
  if (draw.owned && draw.post) {
    // Session 43 (review): its depth writes still tell which surface holds the scene (counted apart).
    if (draw.raster.depth_write && !draw.screen_space) {
      auto it = std::find_if(key_counts_.begin(), key_counts_.end(), [&](const KeyCount& k) {
        return k.surface_info == draw.surface_info && k.depth_info == draw.depth_info;
      });
      if (it == key_counts_.end()) {
        key_counts_.push_back({draw.surface_info, draw.depth_info, 0, 1});
      } else {
        ++it->post_depth_writes;
      }
    }
    if (post_batch_.empty()) post_batch_submission_ = command_processor_.GetCurrentSubmission();
    post_batch_.push_back(draw);
    FrameSlot& slot = slots_[SlotOf(frame_)];
    ++slot.post_draws;
    slot.cpu_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - cpu_start).count();
    return;
  }
  // Learn the scene's render target for the next frame: the one with the most depth-writing draws.
  if (draw.raster.depth_write && !draw.screen_space) {
    auto it = std::find_if(key_counts_.begin(), key_counts_.end(), [&](const KeyCount& k) {
      return k.surface_info == draw.surface_info && k.depth_info == draw.depth_info;
    });
    if (it == key_counts_.end()) {
      key_counts_.push_back({draw.surface_info, draw.depth_info, 1, 0});
    } else {
      ++it->depth_writes;
    }
  }
  if (scene_key_.Matches(draw)) {
    FrameSlot& counting = slots_[SlotOf(frame_)];
    ++counting.scene_draws;
    // Replacing: the draws claimed; side-car: the draws we draw beside the emulated ones. Counted either way: why a
    // replacing frame leaves (or would leave) a draw to the emulated renderer.
    const bool take = replacing_ ? draw.owned : Wants(draw);
    if (!draw.owned) {
      const uint32_t reason = draw.declined ? uint32_t(kReasonSmallSegment) : NotOwnedReason(draw);
      if (reason != kOwnable) {
        ++counting.skipped[reason];
      } else if (replacing_) {
        ++counting.skipped[kReasonNoShader];  // ownable, but its pipeline could not be made
      }
    }
    if (take) {
      // Reversed depth (far = 0, the test passes when greater) needs our targets cleared to 0.
      const bool reversed = draw.raster.depth_compare == VK_COMPARE_OP_GREATER ||
                            draw.raster.depth_compare == VK_COMPARE_OP_GREATER_OR_EQUAL;
      ++(reversed ? depth_reversed_draws_ : depth_forward_draws_);
      if (batch_.empty()) batch_submission_ = command_processor_.GetCurrentSubmission();
      // Session 44 (gpu_native_merge): the next instance of the batch's last draw (the counting above is per guest
      // draw either way).
      if (replacing_ && !batch_.empty() && (REXCVAR_GET(gpu_native_merge) || AllOn(kAllMerge)) &&
          IsNextInstance(batch_.back(), draw)) {
        if (batch_.back().instance_count == 1) ++report_.merge_runs;
        ++batch_.back().instance_count;
        ++report_.merged;
      } else {
        batch_.push_back(draw);
      }
      // The scene's size in host pixels: the largest area its draws cover (viewport within the scissor; the
      // full-screen camera).
      const float right = std::min(draw.viewport.x + draw.viewport.width,
                                   float(draw.scissor.offset.x) + float(draw.scissor.extent.width));
      const float bottom = std::min(draw.viewport.y + draw.viewport.height,
                                    float(draw.scissor.offset.y) + float(draw.scissor.extent.height));
      scene_width_ = std::max(scene_width_, uint32_t(std::clamp(right, 0.0f, 8192.0f)));
      scene_height_ = std::max(scene_height_, uint32_t(std::clamp(bottom, 0.0f, 8192.0f)));
    }
  }
  slots_[SlotOf(frame_)].cpu_us +=
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - cpu_start).count();
#endif  // !REX_TRUE_NATIVE_ONLY
}

// Session 44 (gpu_native_merge): whether `draw` is the next instance of `run` (its first draw plus instance_count - 1
// merged repeats) as vulkan_repeat_draw_merge decides it for host draws: the instanced vertex shader variant, every
// binding and state the same, and the vertex float constants one stride further on in the same set and inside the
// 256-vec4 range its binding covers (every other constant buffer at the run's offset).
bool NativeRenderer::IsNextInstance(const Draw& run, const Draw& draw) const {
  constexpr uint32_t kFloatVertex = SpirvShaderTranslator::kConstantBufferFloatVertex;
  if (!draw.merge_stride || draw.merge_stride != run.merge_stride || draw.float_count != run.float_count ||
      run.post || draw.post || draw.pipeline == VK_NULL_HANDLE || draw.pipeline != run.pipeline ||
      draw.layout != run.layout || draw.vertex_shader != run.vertex_shader || draw.sets_bound != run.sets_bound ||
      draw.dynamic_offset_count != run.dynamic_offset_count || draw.dynamic_offset_count <= kFloatVertex ||
      run.instance_count * run.merge_stride + run.float_count > 256 || draw.indexed != run.indexed ||
      draw.vertex_count != run.vertex_count ||
      (draw.indexed && (draw.index_buffer != run.index_buffer || draw.index_offset != run.index_offset ||
                        draw.index_type != run.index_type)) ||
      std::memcmp(&draw.viewport, &run.viewport, sizeof(draw.viewport)) ||
      std::memcmp(&draw.scissor, &run.scissor, sizeof(draw.scissor)) ||
      draw.raster.depth_bias_constant != run.raster.depth_bias_constant ||
      draw.raster.depth_bias_slope != run.raster.depth_bias_slope ||
      std::memcmp(draw.blend_constants, run.blend_constants, sizeof(draw.blend_constants)) ||
      draw.color_write_mask != run.color_write_mask || draw.pixel_shader != run.pixel_shader) {
    return false;
  }
  for (uint32_t i = 0; i < SpirvShaderTranslator::kDescriptorSetCount; ++i) {
    if ((draw.sets_bound & (1u << i)) && draw.sets[i] != run.sets[i]) return false;
  }
  for (uint32_t i = 0; i < draw.dynamic_offset_count; ++i) {
    const uint32_t expected = run.dynamic_offsets[i] + (i == kFloatVertex ? run.instance_count * run.merge_stride * 16 : 0);
    if (draw.dynamic_offsets[i] != expected) return false;
  }
  return true;
}

bool NativeRenderer::Flush(bool occlusion_query_active) {
  // (Session 47: truth draws are recorded at once - Session 46's truth batch is gone.)
  // Session 42: the post batch (claimed only after the scene's colour resolve, so never together with scene draws).
  const bool post_recorded = FlushPost();
  const bool scene_recorded = FlushScene(occlusion_query_active);
  return post_recorded || scene_recorded;
}

bool NativeRenderer::FlushScene(bool occlusion_query_active) {
  const bool segment_was_open = segment_open_;
  segment_open_ = false;
  if (batch_.empty()) return false;
  FrameSlot& slot = slots_[SlotOf(frame_)];
  // Replacing: claimed draws are never dropped (no draw is claimed while a query counts; a submission ends its render
  // pass, and so this batch, before it ends).
  if (replacing_ && batch_submission_ != command_processor_.GetCurrentSubmission()) {
    REXGPU_ERROR("[native] {} claimed draws from an ended submission: lost (frame {})", BatchGuestDraws(), frame_);
    report_.dropped_submission += BatchGuestDraws();
    batch_.clear();
    return false;
  }
  if (!replacing_ && (occlusion_query_active || batch_submission_ != command_processor_.GetCurrentSubmission())) {
    // Our draws must not add samples to the game's occlusion query; a batch from an ended submission refers to
    // memory that may have changed.
    (occlusion_query_active ? report_.dropped_occlusion : report_.dropped_submission) += BatchGuestDraws();
    if (segment_was_open && !slot.segments.empty()) slot.segments.pop_back();
    batch_.clear();
    return false;
  }
  const auto cpu_start = std::chrono::steady_clock::now();
  const float scale = float(std::clamp(REXCVAR_GET(gpu_native_scale), 0.25, 2.0));
  target_scale_ = scale;
  const uint32_t width = std::max(16u, uint32_t(std::lround(float(scene_width_) * scale)));
  const uint32_t height = std::max(16u, uint32_t(std::lround(float(scene_height_) * scale)));
  if (replacing_ && color_image_ != VK_NULL_HANDLE && (width != target_width_ || height != target_height_)) {
    // New targets within the frame: what the old ones hold goes into the EDRAM first; the new ones reload it.
    WriteBack();
  }
  if (EnsureTargets(width, height)) {
    target_scene_width_ = scene_width_;
    target_scene_height_ = scene_height_;
  } else {
    if (replacing_) {
      REXGPU_ERROR("[native] no {}x{} targets: {} claimed draws lost (frame {})", width, height, BatchGuestDraws(), frame_);
    }
    batch_.clear();
    return false;
  }
  // Replacing: the EDRAM holds something newer than our targets (the frame's start, an emulated draw, a clear):
  // reload them from it first, in this render pass.
  bool load = replacing_ && native_stale_;
  // Session 45 (gpu_native_post_sync): unless the batch's first draw provably writes every pixel of the scene - colour
  // (all channels, no blending) and depth (test always) - so nothing of the EDRAM is read (entrances restore their
  // scene from its resolved textures with such a quad before drawing more of it). (Review fix: the pass still starts
  // with the clearing render pass - the targets may be new or not in the attachment layouts.)
  bool covered_skip = false;
  if (load && post_sync_ && SceneBatchCovers()) {
    load = false;
    covered_skip = true;
    native_stale_ = false;
    ++report_.scene_loads_covered;
  }
  // Session 45 (gpu_native_defer_writeback): the load reads the EDRAM: everything pending goes in first.
  if (load && defer_writeback_) WritePendingAll();
  // Session 45 (clear mirrors): the load is the last fills of the scene's colour and depth surfaces when they covered
  // all of it (the 4x clear before the match's scene): every pixel takes the fill's dword, no read of the EDRAM.
  bool scene_mirror = false;
  if (load && clear_mirrors_) {
    const auto covers = [&](const PostTarget::Mirror& m) {
      return m.valid && m.rect[0] <= 0 && m.rect[1] <= 0 && m.rect[2] >= int32_t(scene_width_) &&
             m.rect[3] >= int32_t(scene_height_);
    };
    const bool need_color = (SceneEdramPlace().flags & 2u) != 0;
    // (Review fix: the colour mirror of this frame's scene colour surface, not the one last seen.)
    scene_mirror = target_scale_ == 1.0f && covers(scene_mirror_depth_) &&
                   (!need_color || (covers(scene_mirror_color_) &&
                                    (scene_mirror_color_.info & 0xF0FFF) == (scene_color_info_ & 0xF0FFF)));
  }
  const uint32_t scene_mirror_src[2] = {scene_mirror_color_.src, scene_mirror_depth_.src};
  if (load) scene_mirror_color_.valid = scene_mirror_depth_.valid = false;  // used now or out of date
  if (segment_log_frames_ > 0 && replacing_) {  // Session 45: the segment log
    REXGPU_INFO("[native][seg] frame {} scene pass: surface 0x{:08X} colour base {} depth base {} target {}x{} load {} "
                "draws {}",
                frame_, scene_key_.surface_info, scene_color_info_ & 0xFFF, scene_key_.depth_info & 0xFFF, width,
                height, load, batch_.size());
  }
  if (load && (!EnsureReplaceObjects() || !EnsureReplaceSets())) {
    REXGPU_ERROR("[native] no load pass: {} claimed draws lost (frame {})", BatchGuestDraws(), frame_);
    batch_.clear();
    return false;
  }
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const uint32_t query_base = SlotOf(frame_) * kQueriesPerFrame;
  FrameSlot::Segment* segment = nullptr;
  if (query_pool_ && slot_queries_ok_ && slot.queries + 2 <= QueryLimit()) {
    if (!segment_was_open) slot.segments.emplace_back();
    segment = &slot.segments.back();
    segment->native_begin = slot.queries++;
    cb.CmdVkWriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, query_base + segment->native_begin);
  }
  if (load) {
    // Every earlier user of the EDRAM (the emulated draws, resolves, clears, our write-backs) -> the load's read.
    // Recorded here directly: this may run inside SubmitBarriers (which ends the render pass), so nothing is pushed
    // onto the command processor's pending barriers, and its tracking of the EDRAM buffer stays as it is.
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                               VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  }

  VkClearValue clear_values[2];
  clear_values[0].color = {{0.06f, 0.07f, 0.09f, 1.0f}};
  clear_values[1].depthStencil = {clear_depth_, 0};
  // The clearing pass starts from undefined contents (the frame's first pass, or before a load, which writes every
  // pixel).
  const bool clear = replacing_ ? (load || covered_skip) : !cleared_this_frame_;
  VkRenderPassBeginInfo begin = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  begin.renderPass = clear ? render_pass_clear_ : render_pass_load_;
  begin.framebuffer = framebuffer_;
  begin.renderArea = {{0, 0}, {target_width_, target_height_}};
  begin.clearValueCount = clear ? 2 : 0;
  begin.pClearValues = clear ? clear_values : nullptr;
  cb.CmdVkBeginRenderPass(&begin, VK_SUBPASS_CONTENTS_INLINE);
  cleared_this_frame_ = true;
  color_readable_ = false;
  if (load) {
    EdramPlace place = SceneEdramPlace();
    if (scene_mirror) {  // Session 45 (clear mirrors)
      place.flags |= 0x1000u;
      place.origin_x = scene_mirror_src[0];
      place.origin_y = scene_mirror_src[1];
      ++report_.scene_mirrors;
    }
    cb.CmdVkBindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, load_pipeline_);
    cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, load_layout_, 0, 1, &load_set_, 0, nullptr);
    cb.CmdVkPushConstants(load_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(place), &place);
    const VkViewport full = {0.0f, 0.0f, float(target_width_), float(target_height_), 0.0f, 1.0f};
    cb.CmdVkSetViewport(0, 1, &full);
    const VkRect2D full_scissor = {{0, 0}, {target_width_, target_height_}};
    cb.CmdVkSetScissor(0, 1, &full_scissor);
    cb.CmdVkDraw(3, 1, 0, 0);
    ++slot.loads;
    native_stale_ = false;
  }

  VkPipeline bound_pipeline = VK_NULL_HANDLE;
  VkPipelineLayout bound_layout = VK_NULL_HANDLE;
  VkDescriptorSet bound_sets[SpirvShaderTranslator::kDescriptorSetCount] = {};
  uint32_t bound_offsets[SpirvShaderTranslator::kConstantBufferCount] = {};
  uint32_t drawn = 0;
  // Session 40: dynamic state is recorded only when it changes (all our scene pipelines have the same four dynamic
  // states, so it carries over from draw to draw within this render pass; the load pipeline before them sets
  // viewport and scissor itself).
  bool state_known = false;
  VkViewport last_viewport = {};
  VkRect2D last_scissor = {};
  float last_bias[2] = {}, last_blend[4] = {};
  for (const Draw& draw : batch_) {
    VkPipeline pipeline = draw.pipeline != VK_NULL_HANDLE ? draw.pipeline : GetPipeline(draw);
    if (pipeline == VK_NULL_HANDLE) continue;
    if (pipeline != bound_pipeline) {
      cb.CmdVkBindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
      bound_pipeline = pipeline;
    }
    if (draw.layout != bound_layout) {
      std::fill(std::begin(bound_sets), std::end(bound_sets), VkDescriptorSet(VK_NULL_HANDLE));
      bound_layout = draw.layout;
    }
    // The emulated draw's descriptor sets, one by one (only those it bound: the others may belong to another
    // layout).
    for (uint32_t i = 0; i < SpirvShaderTranslator::kDescriptorSetCount; ++i) {
      if (!(draw.sets_bound & (1u << i)) || draw.sets[i] == VK_NULL_HANDLE) continue;
      const bool constants = i == SpirvShaderTranslator::kDescriptorSetConstants && draw.dynamic_offset_count;
      if (bound_sets[i] == draw.sets[i] &&
          (!constants ||
           !std::memcmp(bound_offsets, draw.dynamic_offsets, draw.dynamic_offset_count * sizeof(uint32_t)))) {
        continue;
      }
      cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, draw.layout, i, 1, &draw.sets[i],
                                 constants ? draw.dynamic_offset_count : 0,
                                 constants ? draw.dynamic_offsets : nullptr);
      bound_sets[i] = draw.sets[i];
      if (constants) std::memcpy(bound_offsets, draw.dynamic_offsets, draw.dynamic_offset_count * sizeof(uint32_t));
    }
    // The emulated viewport and scissor, scaled to our targets.
    VkViewport viewport = draw.viewport;
    viewport.x *= scale, viewport.y *= scale, viewport.width *= scale, viewport.height *= scale;
    if (!state_known || std::memcmp(&viewport, &last_viewport, sizeof(viewport))) {
      cb.CmdVkSetViewport(0, 1, &viewport);
      last_viewport = viewport;
    }
    VkRect2D scissor;
    const int32_t x0 = std::clamp(int32_t(std::floor(float(draw.scissor.offset.x) * scale)), 0, int32_t(target_width_));
    const int32_t y0 = std::clamp(int32_t(std::floor(float(draw.scissor.offset.y) * scale)), 0, int32_t(target_height_));
    const int32_t x1 = std::clamp(
        int32_t(std::ceil(float(draw.scissor.offset.x + int32_t(draw.scissor.extent.width)) * scale)), x0,
        int32_t(target_width_));
    const int32_t y1 = std::clamp(
        int32_t(std::ceil(float(draw.scissor.offset.y + int32_t(draw.scissor.extent.height)) * scale)), y0,
        int32_t(target_height_));
    scissor.offset = {x0, y0};
    scissor.extent = {uint32_t(x1 - x0), uint32_t(y1 - y0)};
    if (!state_known || std::memcmp(&scissor, &last_scissor, sizeof(scissor))) {
      cb.CmdVkSetScissor(0, 1, &scissor);
      last_scissor = scissor;
    }
    const float bias[2] = {draw.raster.depth_bias_constant, draw.raster.depth_bias_slope};
    if (!state_known || std::memcmp(bias, last_bias, sizeof(bias))) {
      cb.CmdVkSetDepthBias(bias[0], 0.0f, bias[1]);
      std::memcpy(last_bias, bias, sizeof(bias));
    }
    // Dynamic in our pipelines (only blending draws read them).
    if (!state_known || std::memcmp(draw.blend_constants, last_blend, sizeof(last_blend))) {
      cb.CmdVkSetBlendConstants(draw.blend_constants);
      std::memcpy(last_blend, draw.blend_constants, sizeof(last_blend));
    }
    state_known = true;
    if (replacing_) {
      // Session 40: what the write-back has to cover (the scissor, in emulated host pixels, and the channels).
      if (shading_ && draw.pixel_shader != VK_NULL_HANDLE && (draw.color_write_mask & 0xF)) dirty_color_ = true;
      if (draw.raster.depth_write) dirty_depth_ = true;
      dirty_x0_ = std::min(dirty_x0_, draw.scissor.offset.x);
      dirty_y0_ = std::min(dirty_y0_, draw.scissor.offset.y);
      dirty_x1_ = std::max(dirty_x1_, draw.scissor.offset.x + int32_t(draw.scissor.extent.width));
      dirty_y1_ = std::max(dirty_y1_, draw.scissor.offset.y + int32_t(draw.scissor.extent.height));
    }
    // Session 44: instance_count > 1 only with gpu_native_merge (the instanced vertex shader variant).
    if (draw.diag_id) [[unlikely]] cb.CmdDiagMark(diag::kMarkNextDraw, draw.diag_id, draw.diag_aux);  // (Session 95, the diagnostic run)
    BindDrawInputs(cb, draw);  // (Session 95)
    if (draw.indexed) {
      cb.CmdVkBindIndexBuffer(draw.index_buffer, draw.index_offset, draw.index_type);
      cb.CmdVkDrawIndexed(draw.vertex_count, draw.instance_count, 0, draw.vertex_offset, 0);
    } else {
      cb.CmdVkDraw(draw.vertex_count, draw.instance_count, uint32_t(draw.vertex_offset), 0);
    }
    drawn += draw.instance_count;
  }
  cb.CmdVkEndRenderPass();
  if (load) {
    // The load's read -> the EDRAM's next writers (an execution dependency suffices for a read).
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 0, nullptr, 0, nullptr, 0, nullptr);
  }
  if (segment) {
    segment->native_end = slot.queries++;
    cb.CmdVkWriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, query_base + segment->native_end);
  }
  slot.native_draws += drawn;
  ++slot.passes;
  drew_this_frame_ |= drawn != 0;
  if (replacing_) native_dirty_ = true;
  batch_.clear();
  slot.cpu_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - cpu_start).count();
  return true;
}

void NativeRenderer::OnResolve(uint32_t copy_source, uint32_t surface_info, uint32_t color_info, uint32_t depth_info,
                               VkBuffer edram_buffer, uint32_t resolution_scale_x, uint32_t resolution_scale_y,
                               bool clears) {
  // Session 45: the frame census.
  if (enabled_) {
    FrameSlot& census = slots_[SlotOf(frame_)];
    if (copy_source != UINT32_MAX) ++census.resolves;
    if (clears) ++census.clearing_resolves;
    if (segment_log_frames_ > 0 && replacing_) {  // the segment log
      REXGPU_INFO("[native][seg] frame {} resolve: source {} colour base {} depth base {} pitch {} msaa {} format {} "
                  "clears {}",
                  frame_, copy_source, color_info & 0xFFF, depth_info & 0xFFF, surface_info & 0x3FFF,
                  (surface_info >> 16) & 3, (color_info >> 16) & 0xF, clears);
    }
  }
  resolve_target_ = -1;  // Session 45 (gpu_native_resolve)
  texture_written_ = nullptr;  // Session 45 (N3a): set again by this resolve's ResolveFromTarget
  // Session 46: a truth frame's resolve copies from the staging buffer (TruthResolveSource); a clearing one is not
  // done natively yet: the boundary write-out first, the rest of the frame exact.
  if (truth_ && replacing_) {
    if (!clears) return;
    if (OwnedOnly()) {  // Session 92 (M2a, gpu_native_owned_only): a clearing resolve does not end the frame - counted, the copy part is served as any (expected: none)
      OwnedNoteFallbackIgnored(2);
      return;
    }
    TruthFallback(2, nullptr);
  }
  if (replacing_) {
    // Any resolve reads the EDRAM (whatever surface it copies may overlap the scene's): our picture goes in first.
    // The batch was recorded when the caller ended the render pass.
    WriteBack();
    // Session 45 (gpu_native_resolve): the post segment written back here may serve this resolve from its target.
    if (native_resolve_ && copy_source == 0 && post_dirty_ && post_current_ >= 0 &&
        (color_info & 0xF0FFF) == (post_color_info_ & 0xF0FFF)) {
      resolve_target_ = post_current_;
      resolve_color_info_ = post_color_info_;
    }
    // Session 42: the post draws too (the resolve copies what they drew); their segment ends here. Session 45
    // (gpu_native_defer_writeback): a 1x k_8_8_8_8 segment the resolve will likely be served from keeps its colour
    // (ResolveFromTarget writes it if it is not served after all); any other resolve reads the EDRAM: everything
    // pending goes in first.
    // Session 45 (gpu_native_fuse_writeback): the same deferral, consumed by this resolve - ResolveFromTarget writes
    // the segment's pixels into the EDRAM itself (or writes them back first when it cannot), AfterResolve flushes
    // whatever no resolve took.
    if ((defer_writeback_ || fuse_writeback_) && resolve_target_ >= 0 &&
        post_targets_[resolve_target_].target_class <= 1 &&
        ((post_color_info_ >> 16) & 0xF) == 0 && ((surface_info >> 16) & 3) == 0 && !clears &&
        resolution_scale_x == 1 && resolution_scale_y == 1) {
      PostWriteBack(true);
    } else {
      WritePendingAll();
      PostWriteBack();
    }
    post_segment_open_ = false;
    // Session 45 (gpu_native_post_sync; review fix): the resolve clears EDRAM after this - after the write-back, whose
    // cover sync would otherwise add back tiles the clear then changes.
    if (clears && post_sync_) InvalidateAllPostSync();
    // The self-check below needs our targets to hold the latest picture (not after an emulated draw into it).
    const bool in_sync = !native_stale_;
    if (clears) {
      native_stale_ = true;
      seg_active_ = false;  // Session 40: the segment ends with the clear
    }
    if (!REXCVAR_GET(gpu_native_check_writeback) || !in_sync) {
      // The scene's colour resolve ends the native part of the frame (as in side-car frames, below).
      if (copy_source == 0 && drew_this_frame_ && scene_key_.valid && surface_info == scene_key_.surface_info &&
          depth_info == scene_key_.depth_info) {
        scene_done_this_frame_ = true;
      }
      return;
    }
    // gpu_native_check_writeback: the parity checks below now compare the EDRAM, which holds our write-back, with
    // our targets (100% of the pixels agree if the write-back is right).
  }
  // Only the scene's own resolves, after native draws, from a single-sampled surface in the EDRAM buffer.
  if (!enabled_ || !drew_this_frame_ || !scene_key_.valid ||
      surface_info != scene_key_.surface_info || depth_info != scene_key_.depth_info ||
      edram_buffer == VK_NULL_HANDLE || ((surface_info >> 16) & 3) != 0) {
    return;
  }
  const bool depth = copy_source == 4;
  // RB_COLOR_INFO color_format (bits 16-19): 0 k_8_8_8_8, 1 k_8_8_8_8_GAMMA - the formats the check reads.
  const uint32_t color_format = (color_info >> 16) & 0xF;
  if (!depth && copy_source == 0 && !color_parity_done_this_frame_) slots_[SlotOf(frame_)].color_format = color_format;
  if (depth ? parity_done_this_frame_ : (copy_source != 0 || color_parity_done_this_frame_)) return;
  if (!depth) {
    // The scene's colour resolve ends the native part of the frame (Wants): post-processing and the HUD follow. (The
    // depth resolve can come earlier: entrances copy the depth for their particles, then go on drawing the scene.)
    scene_done_this_frame_ = true;
    if (color_format > 1) return;  // the colour check reads k_8_8_8_8 and k_8_8_8_8_GAMMA only
    scene_gamma_this_frame_ = color_format == 1;
  }
  const uint32_t slot_index = SlotOf(frame_);
  FrameSlot& slot = slots_[slot_index];
  // The slot's sets and results may only be rewritten once the GPU has finished the frame that used them last.
  uint64_t& set_submission = depth ? slot.parity_submission : slot.color_parity_submission;
  if (set_submission > command_processor_.GetCompletedSubmission() ||
      !(depth ? EnsureParityPipeline() : EnsureColorParityPipeline()) ||
      (!depth && !EnsureEmulatedSceneImage(scene_width_, scene_height_))) {
    return;
  }
  (depth ? parity_done_this_frame_ : color_parity_done_this_frame_) = true;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  const uint32_t first = depth ? 0 : kParityColor;
  const uint32_t count = depth ? kParityColor : kParityCounterCount - kParityColor;
  std::memset(parity_mapping_ + slot_index * kParityCounterCount + first, 0, count * sizeof(uint32_t));
  FlushParityMemory();

  // Set: the EDRAM, our target, the slot's results (and the emulated scene copy for colour).
  const VkDescriptorSet set = depth ? parity_sets_[slot_index] : color_parity_sets_[slot_index];
  VkDescriptorBufferInfo edram_buffer_info = {edram_buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorImageInfo target_info = {sampler_nearest(), depth ? depth_view_ : color_view_,
                                       depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                             : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkDescriptorBufferInfo result_info = {parity_buffer_, (slot_index * kParityCounterCount + first) * sizeof(uint32_t),
                                        16 * sizeof(uint32_t)};
  VkDescriptorImageInfo scene_info = {VK_NULL_HANDLE, emulated_scene_view_, VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet writes[4] = {};
  for (uint32_t i = 0; i < 4; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
  }
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &edram_buffer_info;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  writes[1].pImageInfo = &target_info;
  writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[2].pBufferInfo = &result_info;
  writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  writes[3].pImageInfo = &scene_info;
  dfn.vkUpdateDescriptorSets(device, depth ? 3 : 4, writes, 0, nullptr);

  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  // The emulated draws' EDRAM writes and our target's writes -> this compute pass.
  VkMemoryBarrier memory_barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                                    VK_ACCESS_SHADER_READ_BIT};
  VkImageMemoryBarrier image_barriers[2] = {};
  VkImageMemoryBarrier& target_barrier = image_barriers[0];
  target_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  target_barrier.srcAccessMask =
      depth ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  target_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  target_barrier.oldLayout =
      depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  target_barrier.newLayout =
      depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  target_barrier.srcQueueFamilyIndex = target_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  target_barrier.image = depth ? depth_image_ : color_image_;
  target_barrier.subresourceRange =
      ui::vulkan::util::InitializeSubresourceRange(depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
  // The emulated scene copy: whatever it held -> written by this pass.
  VkImageMemoryBarrier& scene_barrier = image_barriers[1];
  scene_barrier = target_barrier;
  scene_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  scene_barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  scene_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  scene_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  scene_barrier.image = emulated_scene_image_;
  scene_barrier.subresourceRange = ui::vulkan::util::InitializeSubresourceRange();
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, nullptr, depth ? 1 : 2,
                          image_barriers);
  command_processor_.BindExternalComputePipeline(depth ? parity_pipeline_ : color_parity_pipeline_);
  const VkPipelineLayout layout = depth ? parity_layout_ : color_parity_layout_;
  cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
  // Where the surface is in the EDRAM buffer (see the shaders); the depth surface is the scene key's, the colour
  // surface render target 0's.
  const uint32_t tile_w = 80 * resolution_scale_x, tile_h = 16 * resolution_scale_y;
  const uint32_t pitch_tiles = ((surface_info & 0x3FFF) + 79) / 80;
  struct {
    uint32_t base_dwords, pitch_tile_dwords, tile_w, tile_h, width, height, wrap_dwords, flags;
    float scale_x, scale_y, clear_depth, tolerance;
  } pc;
  pc.base_dwords = ((depth ? depth_info : color_info) & 0xFFF) * tile_w * tile_h;
  pc.pitch_tile_dwords = pitch_tiles * tile_w * tile_h;
  pc.tile_w = tile_w;
  pc.tile_h = tile_h;
  pc.width = std::min(scene_width_, pitch_tiles * tile_w);
  pc.height = scene_height_;
  pc.wrap_dwords = 2048 * tile_w * tile_h;
  // Depth: RB_DEPTH_INFO depth_format 1 = 20e4 float. Colour: format 1 = k_8_8_8_8_GAMMA.
  pc.flags = depth ? (depth_info >> 16) & 1 : uint32_t(color_format == 1);
  pc.scale_x = float(target_width_) / float(std::max(1u, scene_width_));
  pc.scale_y = float(target_height_) / float(std::max(1u, scene_height_));
  pc.clear_depth = clear_depth_;
  pc.tolerance = 1e-5f;
  cb.CmdVkPushConstants(layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, depth ? 12 * sizeof(uint32_t) : 10 * sizeof(uint32_t),
                        &pc);
  cb.CmdVkDispatch((pc.width + 7) / 8, (pc.height + 7) / 8, 1);
  // This pass -> the CPU's read of the results, the next writes of the EDRAM (the resolve and the draws after it),
  // our target (an attachment again) and the swap's read of the emulated scene copy.
  VkMemoryBarrier after = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                           VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT};
  target_barrier.srcAccessMask = 0;
  target_barrier.dstAccessMask =
      depth ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
            : VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  std::swap(target_barrier.oldLayout, target_barrier.newLayout);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                          0, 1, &after, 0, nullptr, 1, &target_barrier);
  (depth ? slot.parity : slot.color_parity) = true;
  if (!depth) emulated_scene_valid_ = true;
  set_submission = command_processor_.GetCurrentSubmission();
}

void NativeRenderer::OnFrameEnd(CommandProcessor::NativeCensus& census) {
  OwnCrowdOnFrameEnd();  // wwe2k14 patch (Session 80, own crowd)
  // wwe2k14 patch (Session 69, gpu_native_truth_debug_renderdoc): the capture started at the strip clear ends here, two
  // frames later (its submissions are in by then).
  if (g_truth_rdc_active && frame_ >= g_truth_rdc_frame + 2) {
    if (const ui::RenderDocAPI* rd = command_processor_.GetVulkanDevice()->vulkan_instance()->renderdoc_api();
        rd && rd->api_1_0_0()) {
      const uint32_t ok = rd->api_1_0_0()->EndFrameCapture(nullptr, nullptr);
      REXGPU_INFO("[native][truth][span] frame {}: RenderDoc capture ended ({})", frame_, ok ? "saved" : "FAILED");
    }
    g_truth_rdc_active = false;
    g_truth_rdc_done = true;
  }
  if (REXCVAR_GET(gpu_native_debug_rdc_now) && !g_truth_rdc_active) {
    rex::cvar::SetFlagByName("gpu_native_debug_rdc_now", "false");
    if (const ui::RenderDocAPI* rd = command_processor_.GetVulkanDevice()->vulkan_instance()->renderdoc_api();
        rd && rd->api_1_0_0()) {
      rd->api_1_0_0()->StartFrameCapture(nullptr, nullptr);
      g_truth_rdc_active = true;
      g_truth_rdc_frame = frame_;
      REXGPU_INFO("[native] frame {}: RenderDoc capture started (gpu_native_debug_rdc_now)", frame_);
    } else {
      REXGPU_WARN("[native] gpu_native_debug_rdc_now: RenderDoc is not attached");
    }
  }
  const bool ended_truth = truth();  // Session 46: this frame ended as a truth frame (no fallback)
  // The census (any draws, gpu_native on or off).
  if (census_frames_left_ > 0) {
    LogCensus(census);
    --census_frames_left_;
  }
  // wwe2k14 patch (Session 66, G1.0): the resolve census counts the frame that ends here, then may start a new one.
  if (alias_frames_left_ > 0) AliasOnFrameEnd();
  if (const int32_t n = REXCVAR_GET(gpu_native_alias_census); n > 0) {
    rex::cvar::SetFlagByName("gpu_native_alias_census", "0");
    AliasStart(n);
  }
  census = {};
  census_frame_vs_ = census_frame_ps_ = census_new_vs_ = census_new_ps_ = 0;
  census_draws_ = census_memexport_draws_ = 0;
  census_frame_hashes_.clear();
  if (const int32_t n = REXCVAR_GET(gpu_native_census); n > 0) {
    census_frames_left_ = n;
    rex::cvar::SetFlagByName("gpu_native_census", "0");
    REXGPU_INFO("[native][census] the next {} frames", n);
  }

  if (replacing_ && native_dirty_) {
    // Native draws after the frame's last resolve (none are expected: the scene ends at its colour resolve): into
    // the EDRAM anyway, which the next frame may read.
    REXGPU_WARN("[native] frame {}: native draws after the last resolve, written back at the frame's end", frame_);
    WriteBack();
  }
  if (replacing_ && post_dirty_) {
    // Session 42: post draws after the frame's last resolve (the front buffer copy is a resolve: none expected).
    REXGPU_WARN("[native][post] frame {}: post draws after the last resolve, written back at the frame's end", frame_);
    PostWriteBack();
  }
  post_segment_open_ = false;
  post_batch_.clear();
  post_batch_depth_ = false;  // Session 43
  // Session 45 (gpu_native_defer_writeback): the next frame starts from an EDRAM with everything in it.
  WritePendingAll();
  // Session 45 (the frame census): this frame's texture loads, from the texture cache's running counts.
  uint64_t texture_loads = 0, texture_loads_resolved = 0, texture_bytes_resolved = 0;
  command_processor_.NativeTextureLoadCounters(texture_loads, texture_loads_resolved, texture_bytes_resolved);
  if (enabled_) {
    FrameSlot& slot = slots_[SlotOf(frame_)];
    slot.texture_loads = texture_loads - texture_loads_seen_;
    slot.texture_loads_resolved = texture_loads_resolved - texture_loads_resolved_seen_;
    slot.texture_bytes_resolved = texture_bytes_resolved - texture_bytes_resolved_seen_;
  }
  texture_loads_seen_ = texture_loads;
  texture_loads_resolved_seen_ = texture_loads_resolved;
  texture_bytes_resolved_seen_ = texture_bytes_resolved;
  // Session 46 (the EDRAM census): this frame's EDRAM accesses by the render target cache.
  {
    uint64_t edram[4];
    command_processor_.NativeEdramCounters(edram);
    if (enabled_) {
      FrameSlot& slot = slots_[SlotOf(frame_)];
      for (int i = 0; i < 4; ++i) slot.edram[i] = edram[i] - edram_seen_[i];
      slot.truth = truth();
    }
    std::copy(edram, edram + 4, edram_seen_);
  }
  if (enabled_) {
    FrameSlot& slot = slots_[SlotOf(frame_)];
    slot.replaced = replacing_;
    slot.used = true;
    slot.read = false;
    slot.frame = frame_;
    slot.submission = command_processor_.GetCurrentSubmission();
    // The scene's render target for the next frame.
    // Session 45: by all depth-writing draws, those claimed into post targets too. (Session 43 counted the post ones
    // only when no emulated depth-writing draw counted: once the scene's own draws had all gone into post targets - a
    // frame whose scene key was another surface - any emulated depth-writing draw elsewhere kept the wrong key for
    // good: run 2886, from Bryan's entrance on, the 512x512 surface. A pass before the scene that outnumbers the scene
    // for one frame now costs that frame its scene path, and the next frame's count puts the scene back.)
    const KeyCount* best = nullptr;
    for (const KeyCount& k : key_counts_) {
      if (k.depth_writes + k.post_depth_writes &&
          (!best || k.depth_writes + k.post_depth_writes > best->depth_writes + best->post_depth_writes)) {
        best = &k;
      }
    }
    if (best && (!scene_key_.valid || best->surface_info != scene_key_.surface_info ||
                 best->depth_info != scene_key_.depth_info)) {
      REXGPU_INFO("[native] frame {}: scene target now surface 0x{:08X} depth 0x{:08X} ({} depth-writing draws)",
                  frame_, best->surface_info, best->depth_info, best->depth_writes + best->post_depth_writes);
      scene_width_ = scene_height_ = 0;  // measured again from the new target's draws
    }
    if (best) {
      scene_key_.surface_info = best->surface_info;
      scene_key_.depth_info = best->depth_info;
      scene_key_.valid = true;
    }
  }
  key_counts_.clear();
  if (enabled_ && !dump_pending_ && drew_this_frame_ && emulated_scene_valid_ && REXCVAR_GET(gpu_native_dump) > 0) {
    RecordDump(slots_[SlotOf(frame_)]);
  }
  if (depth_reversed_draws_ + depth_forward_draws_) {
    const float clear = depth_reversed_draws_ > depth_forward_draws_ ? 0.0f : 1.0f;
    if (clear != clear_depth_) {
      REXGPU_INFO("[native] frame {}: depth cleared to {} from the next frame ({} reversed, {} forward draws)", frame_,
                  clear, depth_reversed_draws_, depth_forward_draws_);
      clear_depth_ = clear;
    }
  }
  depth_reversed_draws_ = depth_forward_draws_ = 0;
  ReadFinishedFrames();
  DestroyRetired(false);

  // The next frame.
  ++frame_;
  {  // wwe2k14 patch (Session 93, idea 7, log only): [eds][native] every 600 frames
    if (g_eds_native.window_first_frame == UINT64_MAX) g_eds_native.window_first_frame = frame_;
    if (frame_ - g_eds_native.window_first_frame >= 600) {
      std::lock_guard<std::mutex> lock(g_eds_native.lock);
      REXGPU_INFO("[eds][native] frames {}-{}: pipelines {}, keys that would collapse {} (distinct collapsed keys {}); stored-pipelines "
                  "prewarm: pipelines {}, would collapse {} (distinct {}) [cumulative; all native pipelines incl. the crowd's]",
                  g_eds_native.window_first_frame, frame_ - 1, g_eds_native.pipelines[0], g_eds_native.would_collapse[0],
                  g_eds_native.collapsed[0].size(), g_eds_native.pipelines[1], g_eds_native.would_collapse[1],
                  g_eds_native.collapsed[1].size());
      g_eds_native.window_first_frame = frame_;
    }
  }
  drew_last_frame_ = drew_this_frame_;
  drew_this_frame_ = false;
  cleared_this_frame_ = false;
  parity_done_this_frame_ = false;
  color_parity_done_this_frame_ = false;
  scene_done_this_frame_ = false;
  emulated_scene_valid_last_ = emulated_scene_valid_;
  scene_gamma_last_ = scene_gamma_this_frame_;
  scene_gamma_this_frame_ = false;
  emulated_scene_valid_ = false;
  segment_open_ = false;
  const bool was_enabled = enabled_, was_replacing = replacing_;
  const int32_t mode = REXCVAR_GET(gpu_native);
  // Session 46 (moved up, Session 54): truth frames wanted, and refused with EDRAM renaming (its slots are EDRAM), DLSS
  // (its snapshot reads the EDRAM) or DLAA (its in-game menu rejects every clear).
  const bool truth_wanted = mode == 2 && REXCVAR_GET(gpu_native_truth);
  // Session 61 (gpu_native_truth_upscalers, test): DLSS / DLAA no longer refuse truth frames (their inputs taken from
  // the truth frames' own images and resolves instead of the EDRAM).
  const bool truth_refused = REXCVAR_GET(gpu_edram_rename) ||
                             ((REXCVAR_GET(gpu_dlss) || REXCVAR_GET(gpu_dlaa)) && !REXCVAR_GET(gpu_native_truth_upscalers));
  // Replacing needs the interlock path's EDRAM buffer. Without it (the light renderer: host render targets) mode 2 is
  // off altogether, not a side-car that would draw the scene a second time - Session 54: unless truth frames run there
  // (gpu_native_truth_host_rt: they need no EDRAM; frames that are not truth frames are emulated).
  const bool host_rt = command_processor_.NativeEdramBuffer() == VK_NULL_HANDLE;
  const bool host_rt_truth = host_rt && truth_wanted && !truth_refused && REXCVAR_GET(gpu_native_truth_host_rt);
  const bool no_edram = mode == 2 && host_rt && !host_rt_truth;
  if (no_edram && !no_edram_logged_) {
    REXGPU_WARN("[native] gpu_native 2 needs the interlock render path (no EDRAM buffer here): native renderer off");
    no_edram_logged_ = true;
  }
  if (host_rt_truth && !host_rt_logged_) {
    REXGPU_INFO("[native] gpu_native 2 on the host render target path (no fragment shader interlock): truth frames "
                "only (gpu_native_truth_host_rt)");
    host_rt_logged_ = true;
  }
  enabled_ = mode != 0 && !no_edram && EnsureObjects();
  // Replacing needs the game's shaders for every scene draw.
  shading_ = mode == 2 || REXCVAR_GET(gpu_native_shading) != 0;
  all_draws_ = mode == 2 || REXCVAR_GET(gpu_native_draws) != 0;
  // Session 46 (gpu_native_truth): a truth frame needs every native setting and the deferred pipeline lookup (without
  // it a draw whose pipeline is still being compiled is lost before the claim) - turned on here, once, logged.
  // (Refused: then nothing is turned on - review fix.)
  if (truth_wanted && !truth_refused && (!REXCVAR_GET(gpu_native_all) || !REXCVAR_GET(gpu_native_lean))) {
    REXGPU_INFO("[native][truth] gpu_native_truth turns on gpu_native_all (was {}) and gpu_native_lean (was {})",
                REXCVAR_GET(gpu_native_all), REXCVAR_GET(gpu_native_lean));
    rex::cvar::SetFlagByName("gpu_native_all", "true");
    rex::cvar::SetFlagByName("gpu_native_lean", "true");
  }
  all_native_ = mode == 2 && REXCVAR_GET(gpu_native_all);  // Session 45 (step 2 N1)
  all_exclude_ = uint32_t(std::max(0, REXCVAR_GET(gpu_native_all_exclude)));
  // Session 45 (step 2 N2): the sync sets hold within a frame only (a frame boundary may see other EDRAM writers).
  InvalidateAllPostSync();
  post_ = mode == 2 && (REXCVAR_GET(gpu_native_post) || all_native_);  // Session 42 (replacing frames only)
  after_scene_ = mode == 2 && (REXCVAR_GET(gpu_native_after_scene) || AllOn(kAllAfterScene));
  scene_extras_ = mode == 2 && (REXCVAR_GET(gpu_native_scene_extras) || AllOn(kAllAfterScene));  // Session 43
  post_quantize_ = mode == 2 && (REXCVAR_GET(gpu_native_post_quantize) || all_native_);  // Session 43
  // (Review fix: only with the quantized post targets - a float target holds the shader's unrounded output, which
  // blending would read instead of the EDRAM's 8-bit values.)
  post_sync_ = mode == 2 && (REXCVAR_GET(gpu_native_post_sync) || AllOn(kAllSync)) && post_quantize_;
  native_resolve_ = mode == 2 && (REXCVAR_GET(gpu_native_resolve) || AllOn(kAllResolve));
  resolve_texture_ = native_resolve_ && (REXCVAR_GET(gpu_native_resolve_texture) || all_native_);  // Session 45 (N3a)
  resolve_texture_stock_ =
      resolve_texture_ && (REXCVAR_GET(gpu_native_resolve_texture_stock) || all_native_);  // Session 45 (N3b)
  defer_writeback_ = native_resolve_ && post_sync_ && REXCVAR_GET(gpu_native_defer_writeback);
  fuse_writeback_ = native_resolve_ && post_sync_ && (REXCVAR_GET(gpu_native_fuse_writeback) || all_native_);  // Session 45
  clear_mirrors_ = post_sync_ && (REXCVAR_GET(gpu_native_clear_mirror) || AllOn(kAllMirrors));
  // Session 45: the segment log's frames.
  if (const int32_t n = REXCVAR_GET(gpu_native_segment_log); n > 0) {
    segment_log_frames_ = n;
    rex::cvar::SetFlagByName("gpu_native_segment_log", "0");
  } else if (segment_log_frames_ > 0) {
    --segment_log_frames_;
  }
  // Session 72 (gpu_native_truth_pass_log): counted down where the frames' timestamps are read.
  if (const int32_t n = REXCVAR_GET(gpu_native_truth_pass_log); n > 0) {
    pass_log_frames_ = n;
    rex::cvar::SetFlagByName("gpu_native_truth_pass_log", "0");
  }
  cover_stats_ = mode == 2 && REXCVAR_GET(gpu_native_cover_stats);
  // Session 43 (review): the post objects exist before the first post claim, so PostReason knows the depth format
  // (a frame whose only post draws test depth would otherwise never make them).
  if (post_ && enabled_ && post_ds_format_ == VK_FORMAT_UNDEFINED && command_processor_.NativeEdramBuffer()) {
    EnsurePostObjects();
  }
  const int32_t verify_every = REXCVAR_GET(gpu_native_verify_every);
  // Session 46 (gpu_native_truth): a truth frame claims every draw and needs no scene target, so the frames before
  // the first depth-writing draw (boot, the first menus) can be truth frames too.
  const bool truth_candidate =
      truth_wanted && !truth_refused && all_native_ && REXCVAR_GET(gpu_native_lean) && !truth_failed_;
  replacing_ = enabled_ && mode == 2 && (scene_key_.valid || truth_candidate) &&
               !(verify_every > 0 && frame_ % verify_every == 0) &&
               (!host_rt || (host_rt_truth && truth_candidate)) && EnsureReplaceObjects();
  // Session 46 (gpu_native_truth): verify frames are not replacing (not truth frames either).
  {
    truth_ = truth_wanted && replacing_ && all_native_ && REXCVAR_GET(gpu_native_lean) && !truth_refused;
    if (truth_wanted && truth_refused && !truth_refused_logged_) {
      truth_refused_logged_ = true;
      REXGPU_WARN("[native][truth] gpu_native_truth refused: gpu_edram_rename {}, gpu_dlss {}, gpu_dlaa {}",
                  REXCVAR_GET(gpu_edram_rename), REXCVAR_GET(gpu_dlss), REXCVAR_GET(gpu_dlaa));
    }
    // Logged when the setting turns on or off (review fix: not at every verify frame).
    const bool truth_setting = truth_wanted && !truth_refused;
    if (truth_setting != truth_setting_) {
      truth_setting_ = truth_setting;
      REXGPU_INFO("[native][truth] truth frames {} from frame {}", truth_setting ? "on" : "off", frame_);
    }
    // The images and the owner map across the frame boundary (the write-out when the next frame is not truth).
    if (truth_ && (truth_failed_ || !EnsureTruthObjects())) truth_ = false;
    if (!truth_ && (!scene_key_.valid || host_rt)) replacing_ = false;  // (no scene target / EDRAM: truth only)
    if (ended_truth && !truth_) TruthWriteOut();
    TruthOnFrameEnd(ended_truth, truth_);
  }
  if (enabled_ && mode == 2 && !replacing_ && was_replacing && verify_every <= 0) {
    REXGPU_WARN("[native] frame {}: not replacing (scene target {}, EDRAM buffer {}, replace objects {})", frame_,
                scene_key_.valid ? "known" : "unknown",
                command_processor_.NativeEdramBuffer() != VK_NULL_HANDLE ? "yes" : "no (not the interlock path)",
                replace_objects_failed_ ? "failed" : "ok");
  }
  native_dirty_ = false;
  native_stale_ = true;  // a frame starts from what the EDRAM holds
  scene_color_info_valid_ = false;
  // Session 40: this frame's segment sizes decide the next frame's; a new frame starts a new segment.
  ResetDirty();
  seg_draws_prev_.swap(seg_draws_);
  seg_draws_.clear();
  seg_count_ = 0;
  seg_active_ = false;
  declined_pending_ = false;
  if (enabled_ != was_enabled) {
    REXGPU_INFO("[native] {} from frame {}", enabled_ ? (mode == 2 ? "on (replace)" : "on (side-car)") : "off",
                frame_);
    report_ = {};
    if (!enabled_) scene_key_ = {};
  }
  FrameSlot& next = slots_[SlotOf(frame_)];
  // Reuse the slot only once the GPU has finished the frame that used it (its results are read or lost).
  slot_queries_ok_ = false;
  if (enabled_ && query_pool_ && REXCVAR_GET(gpu_native_gpu_timers) &&
      (!next.used || next.submission <= command_processor_.GetCompletedSubmission())) {
    // Outside any render pass: the command processor ended it before calling us.
    command_processor_.deferred_command_buffer().CmdVkResetQueryPool(query_pool_, SlotOf(frame_) * kQueriesPerFrame,
                                                                     kQueriesPerFrame);
    slot_queries_ok_ = true;
  }
  probe_slot_ok_ = false;  // Session 82 (gpu_native_owned_alpha_learn): the probe queries of this frame's slot
  probe_issued_ = 0;
  if (enabled_ && REXCVAR_GET(gpu_native_owned_alpha_learn)) OwnedAlphaMaintain();  // (stale keys, in every frame)
  if (enabled_ && REXCVAR_GET(gpu_native_owned_alpha_learn) && EnsureProbePool() &&
      (!next.used || next.submission <= command_processor_.GetCompletedSubmission())) {
    command_processor_.deferred_command_buffer().CmdVkResetQueryPool(probe_pool_, SlotOf(frame_) * kProbesPerFrame * OwnedLayers(),
                                                                     kProbesPerFrame * OwnedLayers());  // (Session 94: K slots per probe)
    probe_slot_ok_ = true;
  }
  const uint64_t keep_parity_submission = next.parity_submission;
  const uint64_t keep_color_parity_submission = next.color_parity_submission;
  next = FrameSlot();
  next.parity_submission = keep_parity_submission;
  next.color_parity_submission = keep_color_parity_submission;
  StorageOnFrameEnd();  // Session 61: the stored pipelines (after the settings and objects above)
}

void NativeRenderer::ReadFinishedFrames() {
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const uint64_t completed = command_processor_.GetCompletedSubmission();
  // Oldest first.
  for (uint64_t f = frame_ >= kFrameSlots ? frame_ - kFrameSlots + 1 : 0; f <= frame_; ++f) {
    FrameSlot& slot = slots_[SlotOf(f)];
    if (!slot.used || slot.read || slot.frame != f || slot.submission > completed) continue;
    slot.read = true;
    if (!slot.alpha_probes.empty()) OwnedAlphaRead(slot);  // Session 82
    ++report_.frames;
    report_.scene_draws += slot.scene_draws;
    report_.native_draws += slot.native_draws;
    for (uint32_t i = 0; i < kReasonCount; ++i) report_.skipped[i] += slot.skipped[i];
    report_.replaced_frames += slot.replaced ? 1 : 0;
    report_.passes += slot.passes;
    report_.loads += slot.loads;
    report_.writebacks += slot.writebacks;
    report_.segments_native += slot.segments_native;
    report_.segments_emulated += slot.segments_emulated;
    report_.post_draws += slot.post_draws;  // Session 42
    report_.post_passes += slot.post_passes;
    report_.post_loads += slot.post_loads;
    report_.post_writebacks += slot.post_writebacks;
    for (uint32_t i = 0; i < kPostReasonCount; ++i) report_.post_skipped[i] += slot.post_skipped[i];
    report_.post_depth_draws += slot.post_depth_draws;  // Session 43
    report_.cover_resolves += slot.cover_resolves;
    report_.cover_resolves_full += slot.cover_resolves_full;
    // Session 45: the frame census.
    report_.frame_draws += slot.frame_draws;
    report_.frame_native_scene += slot.frame_native_scene;
    report_.frame_native_post += slot.frame_native_post;
    for (uint32_t i = 0; i < kEmuCount; ++i) report_.frame_emulated[i] += slot.frame_emulated[i];
    report_.resolves += slot.resolves;
    report_.clearing_resolves += slot.clearing_resolves;
    report_.texture_loads += slot.texture_loads;
    report_.texture_loads_resolved += slot.texture_loads_resolved;
    report_.texture_bytes_resolved += slot.texture_bytes_resolved;
    report_.truth_frames += slot.truth ? 1 : 0;  // Session 46 (the EDRAM census)
    for (int i = 0; i < 4; ++i) report_.edram[i] += slot.edram[i];
    report_.edram_native_resolves += slot.native_resolves;
      if (slot.color_format < 32) report_.color_formats |= 1u << slot.color_format;
    report_.cpu_us += slot.cpu_us;
    if (query_pool_ && slot.queries) {
      std::vector<uint64_t> stamps(slot.queries);
      if (dfn.vkGetQueryPoolResults(vulkan_device->device(), query_pool_, SlotOf(f) * kQueriesPerFrame,
                                    slot.queries, stamps.size() * sizeof(uint64_t), stamps.data(),
                                    sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        ++(slot.replaced ? report_.replaced_frames_with_results : report_.frames_with_results);
        const double ms = double(timestamp_period_ns_) * 1e-6;
        for (const FrameSlot::Segment& s : slot.segments) {
          if (s.native_begin == UINT32_MAX || s.native_end == UINT32_MAX) continue;
          if (s.emulated_begin != UINT32_MAX && stamps[s.native_begin] >= stamps[s.emulated_begin]) {
            report_.emulated_ms += double(stamps[s.native_begin] - stamps[s.emulated_begin]) * ms;
          }
          if (stamps[s.native_end] >= stamps[s.native_begin]) {
            (slot.replaced ? report_.replaced_native_ms : report_.native_ms) +=
                double(stamps[s.native_end] - stamps[s.native_begin]) * ms;
            if (slot.replaced && s.kind < 8) {  // (Session 49: 6, 7)
              report_.replaced_kind_ms[s.kind] += double(stamps[s.native_end] - stamps[s.native_begin]) * ms;
            }
          }
        }
        if (pass_log_frames_ > 0) LogTruthPasses(f, slot, stamps, ms);  // Session 72
      }
    }
    if (slot.dump) WriteDump(f);
    if ((slot.parity || slot.color_parity) && parity_mapping_) {
      InvalidateParityMemory();
      const uint32_t* r = parity_mapping_ + SlotOf(f) * kParityCounterCount;
      if (slot.parity) {
        ++report_.parity_frames;
        for (uint32_t i = 0; i < kParityCenterEmulated; ++i) report_.parity[i] += r[i];
        report_.last_center[0] = r[kParityCenterEmulated];
        report_.last_center[1] = r[kParityCenterNative];
      }
      if (slot.color_parity) {
        ++report_.color_parity_frames;
        for (uint32_t i = kParityColor; i < kParityColorCenterEmulated; ++i) report_.parity[i] += r[i];
        report_.last_color_center[0] = r[kParityColorCenterEmulated];
        report_.last_color_center[1] = r[kParityColorCenterNative];
      }
    }
  }
  const int32_t every = REXCVAR_GET(gpu_native_report_frames);
  if (enabled_ && every > 0 && report_.frames >= uint32_t(every)) {
    LogReport();
    report_ = {};
  }
}

void NativeRenderer::LogReport() {
  const Report& r = report_;
  const double frames = std::max(1u, r.frames);
  std::string parity = "no parity frames";
  if (r.parity_frames) {
    const uint64_t* p = r.parity;
    const double geometry = double(std::max<uint64_t>(1, p[kParityBothGeometry]));
    float center_e, center_n;
    std::memcpy(&center_e, &r.last_center[0], 4);
    std::memcpy(&center_n, &r.last_center[1], 4);
    parity = fmt::format(
        "parity ({} frames): depth agrees on {:.3f}% of the pixels both drew ({:.1f}% of the frame); |dz| <1e-6 "
        "{:.2f}%, <1e-5 {:.2f}%, <1e-4 {:.2f}%, <1e-3 {:.2f}%, more {:.2f}%; only emulated {:.2f}%, only native "
        "{:.2f}%, neither {:.2f}%; centre emulated {:.6f} native {:.6f}",
        r.parity_frames, 100.0 * double(p[kParityAgree]) / geometry,
        100.0 * double(p[kParityBothGeometry]) / double(std::max<uint64_t>(1, p[kParityPixels])),
        100.0 * double(p[kParityDiff1e6]) / geometry, 100.0 * double(p[kParityDiff1e5]) / geometry,
        100.0 * double(p[kParityDiff1e4]) / geometry, 100.0 * double(p[kParityDiff1e3]) / geometry,
        100.0 * double(p[kParityDiffLarger]) / geometry,
        100.0 * double(p[kParityEmulatedOnly]) / double(std::max<uint64_t>(1, p[kParityPixels])),
        100.0 * double(p[kParityNativeOnly]) / double(std::max<uint64_t>(1, p[kParityPixels])),
        100.0 * double(p[kParityBothEmpty]) / double(std::max<uint64_t>(1, p[kParityPixels])), center_e, center_n);
  }
  if (r.color_parity_frames) {
    const uint64_t* p = r.parity;
    const double pixels = double(std::max<uint64_t>(1, p[kParityColorPixels]));
    parity += fmt::format(
        "; colour ({} frames, RGB 8-bit levels): exact {:.2f}%, within 2 {:.2f}%, within 8 {:.2f}%, within 32 "
        "{:.2f}%, more {:.2f}%; mean |difference| R {:.3f} G {:.3f} B {:.3f}; centre emulated {:08X} native {:08X}",
        r.color_parity_frames, 100.0 * double(p[kParityColorExact]) / pixels,
        100.0 * double(p[kParityColorWithin2]) / pixels, 100.0 * double(p[kParityColorWithin8]) / pixels,
        100.0 * double(p[kParityColorWithin32]) / pixels, 100.0 * double(p[kParityColorLarger]) / pixels,
        double(p[kParityColorSumR]) / pixels, double(p[kParityColorSumG]) / pixels, double(p[kParityColorSumB]) / pixels,
        r.last_color_center[0], r.last_color_center[1]);
  }
  const auto per_frame = [&](uint64_t n) { return double(n) / frames; };
  NativeVsExpandReport();  // Session 94 (nogs)
  REXGPU_INFO(
      "[native] frames {}-{} ({} read, {} replaced): scene draws {:.0f}/frame, native {:.0f}/frame; GPU emulated "
      "scene {:.3f} ms/frame, native {:.3f} ms/frame (frames with timestamps {}); CPU {:.0f} us/frame; dropped "
      "(occlusion query) {}, (submission) {}; left out per frame: screen-space {:.1f}, not plain triangles {:.1f}, "
      "memexport {:.1f}, after the scene {:.1f}, no depth write {:.1f}, stencil {:.1f}, other targets {:.1f}, "
      "colour format {:.1f}, msaa {:.1f}, occlusion {:.1f}, no shader {:.1f}, small segment {:.1f}; segments native "
      "{:.1f}, emulated {:.1f}; passes {:.1f}, loads {:.1f}, "
      "write-backs {:.1f} per frame; replaced frames: GPU native {:.3f} ms/frame (frames with timestamps {}); scene "
      "colour formats 0x{:X}; targets {}x{}; shading {}, draws {}; {}",
      frame_ + 1 - r.frames, frame_, r.frames, r.replaced_frames, per_frame(r.scene_draws),
      per_frame(r.native_draws), r.frames_with_results ? r.emulated_ms / r.frames_with_results : 0.0,
      r.frames_with_results ? r.native_ms / r.frames_with_results : 0.0, r.frames_with_results, r.cpu_us / frames,
      r.dropped_occlusion, r.dropped_submission, per_frame(r.skipped[kReasonScreenSpace]),
      per_frame(r.skipped[kReasonNotTriangles]), per_frame(r.skipped[kReasonMemexport]),
      per_frame(r.skipped[kReasonAfterScene]), per_frame(r.skipped[kReasonNoDepthWrite]),
      per_frame(r.skipped[kReasonStencil]), per_frame(r.skipped[kReasonOtherTargets]),
      per_frame(r.skipped[kReasonColorFormat]), per_frame(r.skipped[kReasonMsaa]),
      per_frame(r.skipped[kReasonOcclusion]), per_frame(r.skipped[kReasonNoShader]),
      per_frame(r.skipped[kReasonSmallSegment]), per_frame(r.segments_native), per_frame(r.segments_emulated),
      per_frame(r.passes),
      per_frame(r.loads), per_frame(r.writebacks),
      r.replaced_frames_with_results ? r.replaced_native_ms / r.replaced_frames_with_results : 0.0,
      r.replaced_frames_with_results, r.color_formats, target_width_, target_height_,
      shading_ ? "game" : "clay", all_draws_ ? "all" : "depth-writing", parity);
  // Session 44: crowd merge.
  if (r.merged) {
    REXGPU_INFO("[native][merge] frames {}-{}: claimed repeats merged into instanced native draws {:.1f}/frame, "
                "into {:.1f} draws/frame (gpu_native_merge {})",
                frame_ + 1 - r.frames, frame_, per_frame(r.merged), per_frame(r.merge_runs),
                REXCVAR_GET(gpu_native_merge) ? "on" : "off");
  }
  if (r.post_lean_depth_passes) {
    REXGPU_INFO("[native][post] frames {}-{}: post passes without depth load / store {:.1f}/frame "
                "(gpu_native_post_lean_depth)",
                frame_ + 1 - r.frames, frame_, per_frame(r.post_lean_depth_passes));
  }
  // Session 42: post-processing.
  uint64_t post_left = 0;
  for (uint32_t i = 0; i < kPostReasonCount; ++i) post_left += r.post_skipped[i];
  if (r.post_draws || post_left) {
    const uint64_t* s = r.post_skipped;
    REXGPU_INFO(
        "[native][post] frames {}-{}: post draws native {:.1f}/frame (depth {:.1f}), passes {:.1f}, loads {:.1f}, "
        "write-backs {:.1f}; "
        "left emulated per frame: primitive {:.1f}, msaa {:.1f}, other targets {:.1f}, no colour {:.1f}, colour "
        "format {:.1f}, depth {:.1f}, depth format {:.1f}, depth bias {:.1f}, depth clamp {:.1f}, stencil {:.1f}, "
        "tiny {:.1f}, memexport {:.1f}, "
        "occlusion {:.1f}, no shader {:.1f}, surface switch {:.1f}, blending {:.1f}; targets {}",
        frame_ + 1 - r.frames, frame_, per_frame(r.post_draws), per_frame(r.post_depth_draws),
        per_frame(r.post_passes), per_frame(r.post_loads), per_frame(r.post_writebacks),
        per_frame(s[kPostPrimitive]), per_frame(s[kPostMsaa]), per_frame(s[kPostOtherTargets]),
        per_frame(s[kPostNoColor]), per_frame(s[kPostColorFormat]), per_frame(s[kPostDepth]),
        per_frame(s[kPostDepthFormat]), per_frame(s[kPostDepthBias]), per_frame(s[kPostDepthClamp]),
        per_frame(s[kPostStencil]), per_frame(s[kPostSmall]),
        per_frame(s[kPostMemexport]), per_frame(s[kPostOcclusion]), per_frame(s[kPostNoShader]),
        per_frame(s[kPostSwitch]), per_frame(s[kPostBlend]), post_targets_.size());
    if (r.replaced_frames_with_results) {
      const double n = r.replaced_frames_with_results;
      REXGPU_INFO("[native][gpu] frames {}-{}: GPU ms per replaced frame - scene passes {:.3f}, scene write-backs {:.3f}, "
                  "post passes {:.3f}, post write-backs {:.3f}",
                  frame_ + 1 - r.frames, frame_, r.replaced_kind_ms[0] / n, r.replaced_kind_ms[1] / n,
                  r.replaced_kind_ms[2] / n, r.replaced_kind_ms[3] / n);
    }
    if (r.cover_resolves) {
      REXGPU_INFO("[native][cover] frames {}-{}: resolves of native post surfaces {:.1f}/frame, of them provably fully "
                  "covered by the segment's draws {:.1f}/frame",
                  frame_ + 1 - r.frames, frame_, per_frame(r.cover_resolves), per_frame(r.cover_resolves_full));
    }
  }
  LogFrameCensus();
  LogEdramCensus();
  LogTruthCensus();
}

// Session 46: every touch of the EDRAM buffer, per frame (the no-EDRAM goal's measure: all 0 in a match frame) - our
// write-backs and loads (and the clear mirrors, which read a fill's dword), the fills, the emulated draws, the resolves
// whose shader reads it (stock and texture-only copies; those served from our targets read it for the pixels the
// segment did not draw), the resolve clears, and every access of the render target cache (draws, resolves, fills).
void NativeRenderer::LogEdramCensus() {
  const Report& r = report_;
  if (!r.frame_draws) return;
  const double frames = std::max(1u, r.frames);
  const auto pf = [&](uint64_t n) { return double(n) / frames; };
  uint64_t emulated = 0;
  for (uint64_t n : r.frame_emulated) emulated += n;
  const uint64_t writebacks = r.writebacks + r.post_writebacks + r.writebacks_fused;
  const uint64_t loads = r.loads + r.post_loads;
  const uint64_t mirrors = r.scene_mirrors + r.post_mirrors;
  const uint64_t resolve_reads = r.edram[2] + r.edram_native_resolves;
  const uint64_t native_passes = r.writebacks + r.post_writebacks + loads + mirrors + r.edram_native_resolves;
  REXGPU_INFO(
      "[native][edram] frames {}-{} ({} replaced, {} truth): write-backs {:.1f}, loads {:.1f}, clear mirrors {:.1f}, "
      "fills {:.1f}, emulated draws {:.1f}, resolves reading it {:.1f} of {:.1f} (stock / texture-only {:.1f}, from "
      "targets {:.1f}), resolve clears {:.1f}; EDRAM accesses {:.1f} (render target cache {:.1f}, native passes {:.1f}) "
      "per frame",
      frame_ + 1 - r.frames, frame_, r.replaced_frames, r.truth_frames, pf(writebacks), pf(loads), pf(mirrors),
      pf(r.edram[1]), pf(emulated), pf(resolve_reads), pf(r.resolves), pf(r.edram[2]), pf(r.edram_native_resolves),
      pf(r.edram[3]), pf(r.edram[0] + native_passes), pf(r.edram[0]), pf(native_passes));
}

// Session 45: one line for the whole frame (step 2's measure): who drew the draws, and the memory picture - the
// EDRAM's write-backs and loads, the resolves into guest memory and the texture loads that read them back.
void NativeRenderer::LogFrameCensus() {
  const Report& r = report_;
  if (!r.frame_draws) return;
  const double frames = std::max(1u, r.frames);
  const auto pf = [&](uint64_t n) { return double(n) / frames; };
  uint64_t emulated = 0;
  for (uint64_t n : r.frame_emulated) emulated += n;
  const uint64_t* e = r.frame_emulated;
  REXGPU_INFO(
      "[native][frame] frames {}-{} ({} replaced): draws {:.1f}/frame - native scene {:.1f}, post {:.1f}; emulated "
      "{:.1f} (clear {:.1f}, blend {:.1f}, primitive {:.1f}, depth {:.1f}, stencil {:.1f}, format {:.1f}, msaa {:.1f}, "
      "small {:.1f}, switch {:.1f}, no shader {:.1f}, other {:.1f}); resolves {:.1f} (clearing {:.1f}, from post "
      "targets {:.1f}, also filling the texture {:.1f}; stock copies filling theirs {:.1f}); write-backs "
      "{:.1f} (scene {:.1f}, post {:.1f}; done by the resolve instead {:.1f}); loads {:.1f} (scene {:.1f}, post "
      "{:.1f}; avoided: scene {:.1f}, post "
      "colour / depth {:.1f}; clear mirrors instead: scene {:.1f}, post {:.1f}); colour write-backs deferred {:.1f} "
      "(written later {:.1f}, dropped {:.1f}, carried on "
      "{:.1f}); texture loads {:.1f} (over resolved ranges {:.1f}, {:.2f} MB)",
      frame_ + 1 - r.frames, frame_, r.replaced_frames, pf(r.frame_draws), pf(r.frame_native_scene),
      pf(r.frame_native_post), pf(emulated), pf(e[kEmuClear]), pf(e[kEmuBlend]), pf(e[kEmuPrimitive]),
      pf(e[kEmuDepth]), pf(e[kEmuStencil]), pf(e[kEmuFormat]), pf(e[kEmuMsaa]), pf(e[kEmuSmall]), pf(e[kEmuSwitch]),
      pf(e[kEmuNoShader]), pf(e[kEmuOther]), pf(r.resolves), pf(r.clearing_resolves), pf(r.native_resolves),
      pf(r.textures_filled - r.texture_only_resolves), pf(r.texture_only_resolves),
      pf(r.writebacks + r.post_writebacks),
      pf(r.writebacks), pf(r.post_writebacks), pf(r.writebacks_fused), pf(r.loads + r.post_loads), pf(r.loads),
      pf(r.post_loads),
      pf(r.scene_loads_covered), pf(r.post_loads_avoided), pf(r.scene_mirrors), pf(r.post_mirrors),
      pf(r.pending_deferred), pf(r.pending_written),
      pf(r.pending_dropped), pf(r.pending_resumed), pf(r.texture_loads), pf(r.texture_loads_resolved), pf(r.texture_bytes_resolved) / (1024.0 * 1024.0));
}

void NativeRenderer::LogCensus(CommandProcessor::NativeCensus& census) {
  std::string regions;
  for (uint32_t i = 0; i < census.ib_region_count; ++i) {
    regions += fmt::format("{}0x{:03X}", i ? "," : "", census.ib_regions[i]);
  }
  REXGPU_INFO(
      "[native][census] frame {}: draws {} (ring {}, indirect buffers {}), backend draws {}, memexport {}; "
      "indirect buffers {} in MB regions [{}]{}; occlusion query events {}, visibility-query draws {} ({} dropped); "
      "shaders this frame vs {} ps {}, new vs {} ps {}, seen since start vs {} ps {}",
      frame_, census.draws_in_ring + census.draws_in_ib, census.draws_in_ring, census.draws_in_ib, census_draws_,
      census_memexport_draws_, census.indirect_buffers, regions,
      census.ib_region_overflow ? fmt::format(" +{} elsewhere", census.ib_region_overflow) : std::string(),
      census.zpd_events, census.viz_query_draws, census.viz_query_dropped, census_frame_vs_, census_frame_ps_,
      census_new_vs_, census_new_ps_, census_vs_seen_.size(), census_ps_seen_.size());
}

// ---------------------------------------------------------------------------------------------------------------
// gpu_native_dump: both pictures of one frame, for side-by-side comparisons
// ---------------------------------------------------------------------------------------------------------------

void NativeRenderer::RecordDump(FrameSlot& slot) {
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  // No dump is pending: the buffers are free (the last one was written once its frame had completed).
  const uint32_t widths[2] = {emulated_scene_width_, target_width_};
  const uint32_t heights[2] = {emulated_scene_height_, target_height_};
  if (widths[0] != dump_width_ || heights[0] != dump_height_ || widths[1] != dump_native_width_ ||
      heights[1] != dump_native_height_) {
    for (int i = 0; i < 2; ++i) {
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, dump_buffers_[i]);
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, dump_memory_[i]);
      if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
              vulkan_device, VkDeviceSize(widths[i]) * heights[i] * (i ? 8 : 4), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              ui::vulkan::util::MemoryPurpose::kReadback, dump_buffers_[i], dump_memory_[i])) {
        REXGPU_ERROR("[native] gpu_native_dump: no {}x{} readback buffer", widths[i], heights[i]);
        dump_width_ = 0;
        return;
      }
    }
    dump_width_ = widths[0], dump_height_ = heights[0];
    dump_native_width_ = widths[1], dump_native_height_ = heights[1];
  }
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  VkImageMemoryBarrier barriers[2] = {};
  for (auto& b : barriers) {
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange = ui::vulkan::util::InitializeSubresourceRange();
    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  }
  barriers[0].image = emulated_scene_image_;
  barriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barriers[0].oldLayout = barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barriers[1].image = color_image_;
  barriers[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  barriers[1].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
  for (int i = 0; i < 2; ++i) {
    VkBufferImageCopy region = {};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {widths[i], heights[i], 1};
    cb.CmdVkCopyImageToBuffer(i ? color_image_ : emulated_scene_image_,
                              i ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL, dump_buffers_[i], 1,
                              &region);
  }
  // Back to an attachment for the swap's composition barrier; the copies -> the CPU's read.
  barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  barriers[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barriers[1].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  VkMemoryBarrier to_host = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                             VK_ACCESS_HOST_READ_BIT};
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 1, &to_host, 0, nullptr, 1, &barriers[1]);
  slot.dump = true;
  dump_pending_ = true;
  dump_gamma_ = scene_gamma_this_frame_;
}

void NativeRenderer::WriteDump(uint64_t frame) {
  dump_pending_ = false;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  const std::string dir = REXCVAR_GET(gpu_native_dump_dir);
  const uint32_t widths[2] = {dump_width_, dump_native_width_};
  const uint32_t heights[2] = {dump_height_, dump_native_height_};
  for (int i = 0; i < 2; ++i) {
    void* mapping = nullptr;
    const VkDeviceSize size = VkDeviceSize(widths[i]) * heights[i] * 4;  // the BMP's; the native copy is 16-bit
    if (!dump_memory_[i] || dfn.vkMapMemory(device, dump_memory_[i], 0, VK_WHOLE_SIZE, 0, &mapping) != VK_SUCCESS) {
      continue;
    }
    VkMappedMemoryRange range = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, dump_memory_[i], 0, VK_WHOLE_SIZE};
    dfn.vkInvalidateMappedMemoryRanges(device, 1, &range);
    // A 32-bit BMP, rows bottom-up, BGRA.
    const std::string path = fmt::format("{}/native_{}_{}.bmp", dir, frame, i ? "native" : "emulated");
    if (FILE* file = std::fopen(path.c_str(), "wb")) {
      const uint32_t row = widths[i] * 4;
      const uint32_t header[13] = {uint32_t(54 + size), 0, 54, 40, widths[i], heights[i], 1 | (32 << 16), 0,
                                   uint32_t(size), 2835, 2835, 0, 0};
      std::fwrite("BM", 1, 2, file);
      std::fwrite(header, 4, 13, file);
      std::vector<uint8_t> line(row);
      const uint8_t* pixels = static_cast<const uint8_t*>(mapping);
      // The native copy: 16-bit linear RGBA, shown as 8-bit (through the Xenos gamma curve for gamma scenes).
      auto native8 = [this](uint16_t v) {
        const float linear = float(v) / 65535.0f;
        return uint8_t(std::lround(255.0f * (dump_gamma_ ? LinearToPWLGamma(linear) : linear)));
      };
      for (uint32_t y = heights[i]; y-- > 0;) {
        for (uint32_t x = 0; x < widths[i]; ++x) {
          if (i) {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(pixels) + (size_t(y) * widths[i] + x) * 4;
            line[x * 4 + 0] = native8(src[2]);
            line[x * 4 + 1] = native8(src[1]);
            line[x * 4 + 2] = native8(src[0]);
          } else {
            const uint8_t* src = pixels + (size_t(y) * widths[i] + x) * 4;
            line[x * 4 + 0] = src[2];
            line[x * 4 + 1] = src[1];
            line[x * 4 + 2] = src[0];
          }
          line[x * 4 + 3] = 255;
        }
        std::fwrite(line.data(), 1, row, file);
      }
      std::fclose(file);
      REXGPU_INFO("[native] gpu_native_dump: {} ({}x{})", path, widths[i], heights[i]);
    } else {
      REXGPU_ERROR("[native] gpu_native_dump: cannot write {}", path);
    }
    dfn.vkUnmapMemory(device, dump_memory_[i]);
  }
  const int32_t left = REXCVAR_GET(gpu_native_dump);
  rex::cvar::SetFlagByName("gpu_native_dump", std::to_string(std::max(0, left - 1)));
}

// ---------------------------------------------------------------------------------------------------------------
// The swap
// ---------------------------------------------------------------------------------------------------------------

void NativeRenderer::OnSwapPrepare() {
  if (!enabled_ || !drew_last_frame_ || color_image_ == VK_NULL_HANDLE || REXCVAR_GET(gpu_native_view) == 0) {
    return;
  }
  if (DecoupleViews()) {  // (Session 94 review: a blended tick is swapped twice - the second call finds the image already readable)
    static uint64_t prepared_frame = UINT64_MAX;
    if (prepared_frame == frame_) return;
    prepared_frame = frame_;
  }
  // Our colour, written by the native passes -> read by the composition (the swap submits the barrier).
  command_processor_.PushImageMemoryBarrier(
      color_image_, ui::vulkan::util::InitializeSubresourceRange(), VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  color_readable_ = true;
}

void NativeRenderer::OnSwapCompose(VkImageView guest_output, uint32_t width, uint32_t height) {
  const int32_t view = REXCVAR_GET(gpu_native_view);
  if (!enabled_ || !color_readable_ || view == 0 || compose_pipeline_ == VK_NULL_HANDLE) return;
  // The frame slot's set was last used kFrameSlots frames ago.
  const uint32_t slot_index = SlotOf(frame_);
  if (compose_set_submission_[slot_index] > command_processor_.GetCompletedSubmission()) return;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  // The scene views need the emulated scene copy of the presented frame; without it they show ours alone.
  const bool have_scene = emulated_scene_valid_last_ && emulated_scene_view_ != VK_NULL_HANDLE;
  const uint32_t shown = (view >= 3 && !have_scene) ? 1u : uint32_t(view);
  VkDescriptorImageInfo images[3] = {
      {sampler_, color_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
      {VK_NULL_HANDLE, guest_output, VK_IMAGE_LAYOUT_GENERAL},
      have_scene ? VkDescriptorImageInfo{sampler_, emulated_scene_view_, VK_IMAGE_LAYOUT_GENERAL}
                 : VkDescriptorImageInfo{sampler_, color_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
  VkWriteDescriptorSet writes[3] = {};
  for (uint32_t i = 0; i < 3; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = compose_sets_[slot_index];
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = i == 1 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[i].pImageInfo = &images[i];
  }
  vulkan_device->functions().vkUpdateDescriptorSets(vulkan_device->device(), 3, writes, 0, nullptr);
  compose_set_submission_[slot_index] = command_processor_.GetCurrentSubmission();

  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  // The swap's gamma / FXAA writes of the guest output -> ours (the same pixels); the colour parity check's write
  // of the emulated scene copy -> our read.
  VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT};
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                          0, nullptr, 0, nullptr);
  command_processor_.BindExternalComputePipeline(compose_pipeline_);
  cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, compose_layout_, 0, 1, &compose_sets_[slot_index], 0,
                             nullptr);
  struct {
    uint32_t width, height;
    float split, divider_width;
    uint32_t view;
    uint32_t gamma;
  } constants = {width, height, float(std::clamp(REXCVAR_GET(gpu_native_split), 0.0, 1.0)),
                 2.0f / float(std::max(1u, width)), shown, uint32_t(scene_gamma_last_)};
  cb.CmdVkPushConstants(compose_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
  cb.CmdVkDispatch((width + 15) / 16, (height + 7) / 8, 1);
}

}  // namespace rex::graphics::vulkan::native
