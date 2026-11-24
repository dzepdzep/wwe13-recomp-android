/**
 * The native renderer's truth frames (wwe2k14 patch, Session 46): the frame drawn without the EDRAM.
 *
 * In a truth frame (gpu_native_truth) every guest surface is one host image - colour keyed by its EDRAM base, pitch and
 * format (RGBA8 for k_8_8_8_8, RGBA16_SNORM for k_16_16 / k_16_16_16_16), depth by its base, pitch and format (D32S8) -
 * and those images are the picture: they persist from frame to frame, and a 2048-entry owner map says which image holds
 * each EDRAM tile. Every draw the frame has is claimed and drawn into its images (the fast clears too: a 4x view of a
 * 1x surface is drawn with its viewport doubled into the 1x image); a resolve's copy shader reads a staging buffer in
 * the EDRAM's layout, into which the copied tiles were just written from their owners. So the EDRAM buffer is neither
 * read nor written. Anything a truth frame cannot do (a draw it does not claim, a clearing resolve) is the boundary:
 * every owned tile is written into the EDRAM and the rest of the frame is exact; a frame after a non-truth frame starts
 * with the EDRAM owning every tile (a draw over such tiles loads them first).
 *
 * Not exact by design (docs\native\NO_EDRAM_DESIGN.md in the wwe2k14-recomp repository): host blending, the host's
 * polygon offset, a surface reading tiles another surface wrote since keeps its own old pixels there.
 */

#include <rex/graphics/vulkan/native/native_renderer.h>
#include "diag_layer.h"  // Session 95 (the diagnostic run)
#include "native_truth_reasons.h"  // Session 92, M2b
#include "owned_wb_shaders.h"  // Session 92, M2d (the write-back shader sources, moved)

#include <algorithm>
#include <cmath>
#include <cstring>

#include <fmt/format.h>
#include <rex/cvar.h>
#include <rex/filesystem.h>  // Session 73 (the stored truth images)
#include <rex/perf/frame_events.h>  // Session 73 (what a frame did)
#include <rex/graphics/free_counters.h>
#include <rex/ui/vulkan/gpu_fatal.h>  // Session 95 (PLAIN)
#include <rex/logging.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/ui/vulkan/util.h>
#include <rex/ui/renderdoc_api.h>   // Session 69 (gpu_native_truth_debug_renderdoc)
#include <rex/ui/vulkan/instance.h>
#include "cull_ours.h"  // Session 93

REXCVAR_DECLARE(int32_t, gpu_native_truth_debug_draws);
REXCVAR_DECLARE(int32_t, gpu_native);
REXCVAR_DECLARE(int32_t, gpu_native_frame_batch_log);  // wwe2k14 patch (Session 79)
REXCVAR_DECLARE(bool, gpu_native_truth);
REXCVAR_DECLARE(bool, gpu_edram_rename);
REXCVAR_DECLARE(bool, gpu_dlss);
REXCVAR_DECLARE(bool, gpu_dlaa);
// Session 54 (docs\native\RESOLUTION_S54.md in the wwe2k14-recomp repository).
REXCVAR_DEFINE_INT32(gpu_native_truth_draw_height, 0, "GPU/Native",
                     "Session 54: with gpu_native 2 and truth frames, the height the truth passes draw at (the game's "
                     "1280x720 scaled: 360, 480, 540, ... 1080), when it is below the resolution scale's (720 x "
                     "resolution_scale): they draw into smaller images and every read (resolves, texture fills, "
                     "stores) uses the images at the resolution scale, brought up to date by a blit. 0 = the resolution "
                     "scale's own size. Session 73: may change while running (dynamic resolution); taken at the "
                     "next frame end")
    .range(0, 4320)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DECLARE(bool, gpu_native_owned_only);  // wwe2k14 patch (Session 92, M2a): defined in owned_steps.cpp
namespace rex::graphics::vulkan::native {
double OwnedRowRatio(uint32_t scale_x, uint32_t scale_y);  // (Session 95: owned_steps.cpp)
}
// wwe2k14 patch (Session 73, the long frame at every scene start; TruthStorageOpen / TruthPrewarmImages).
REXCVAR_DEFINE_BOOL(gpu_native_truth_prewarm, true, "GPU/Native",
                    "Session 73: remember the truth images this install has made (<cache>/shaders/shareable/"
                    "TITLE.native.truth.txt) and make them again from the first truth frames of the next run, a few per "
                    "frame, instead of in the first frames of each scene (a scene start's long frame)");
REXCVAR_DEFINE_INT32(gpu_native_truth_prewarm_per_frame, 4, "GPU/Native",
                     "Session 73: gpu_native_truth_prewarm - images made per frame end")
    .range(1, 96);
REXCVAR_DEFINE_BOOL(gpu_native_truth_draw_stencil_export, false, "GPU/Native",
                    "Session 73: draw twins' depth pass may export the stencil (VK_EXT_shader_stencil_export, phones "
                    "only); off = the stencil-kept depth pass, the one checked on the PC");
REXCVAR_DEFINE_INT32(gpu_native_truth_spare_keep_ms, 10000, "GPU/Native",
                     "Session 73: a renamed truth image's spare stays in the pool at least this long unused (and 60 "
                     "frames); 0 = the 60 frames only, as before")
    .range(0, 600000);
REXCVAR_DEFINE_STRING(gpu_native_truth_prewarm_file, "", "GPU/Native",
                      "Session 73: gpu_native_truth_prewarm's list at this path instead of the cache's (tests: the "
                      "bench lanes' caches are made again every run)");
// wwe2k14 patch (Session 55, the red blocks / white wrestlers on slow laptops; ClaimTruth).
REXCVAR_DEFINE_BOOL(gpu_native_truth_pipeline_wait_offscreen, true, "GPU/Native",
                    "Session 55: with gpu_native_truth_async_pipelines, a truth draw into a surface narrower than the "
                    "screen (under 16 tiles = 1280 pixels) whose pipeline is not ready waits for it (made at once) "
                    "instead of being skipped: such draws build textures the game keeps (wrestlers' while a match "
                    "loads), and a skipped one stays missing. false = every such draw is skipped (Sessions 47-54)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 57, red patches still reported on a GTX 1060): the width from which a draw counts as screen-wide
// for gpu_native_truth_pipeline_wait_offscreen. A texture the game builds in a screen-wide surface would still lose
// draws; 1024 = no draw is ever skipped (each new pipeline is made at once).
REXCVAR_DEFINE_INT32(gpu_native_truth_pipeline_skip_min_pitch, 16, "GPU/Native",
                     "Session 57: with gpu_native_truth_pipeline_wait_offscreen, a truth draw whose pipeline is not "
                     "ready may be skipped only on a surface at least this many 80-pixel tiles wide (16 = 1280: the "
                     "screen); narrower ones wait. 1024 = never skipped")
    .range(0, 1024)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 58): and at most this wide (a character's attire is drawn on a 2080-wide = 26-tile surface).
REXCVAR_DEFINE_INT32(gpu_native_truth_pipeline_skip_max_pitch, 25, "GPU/Native",
                     "Session 58: with gpu_native_truth_pipeline_wait_offscreen, a truth draw on a surface wider than "
                     "this many 80-pixel tiles waits for its pipeline too (25: a 26-tile attire surface waits, screens "
                     "up to 1920 = 24 tiles may skip). 1024 = no upper limit")
    .range(0, 1024)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_truth_draw_min_pitch, 16, "GPU/Native",
                     "Session 54: with gpu_native_truth_draw_height, the surfaces drawn smaller: at least this many "
                     "80-pixel tiles wide (16 = 1280: the screen-wide ones). Narrower surfaces draw at the resolution "
                     "scale (their syncs cost more than they save)")
    .range(1, 64)
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);
REXCVAR_DEFINE_BOOL(gpu_native_truth_draw_depth_blit, true, "GPU/Native",
                    "Session 54: with gpu_native_truth_draw_height, depth twins copied by a blit where the device can "
                    "blit D32S8 (false: the shader pass AMD's driver needs, for testing it elsewhere)")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);
REXCVAR_DEFINE_BOOL(gpu_native_truth_draw_twins_off, false, "GPU/Native",
                    "Session 54 (diagnostic, for in-run A/Bs of gpu_native_truth_draw_height): the truth passes draw at "
                    "the resolution scale again (switched at a frame end). The pixel shaders keep the draw height's "
                    "pixel positions, so shaders that read their own screen position are off while it is on: cost "
                    "comparisons only")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DECLARE(bool, gpu_native_truth_direct_resolve_scaled);
REXCVAR_DEFINE_BOOL(gpu_native_truth_batch_resolves, false, "GPU/Native",
                    "Session 52: a texture-only direct resolve (1x: the texture only, no memory write) is recorded later "
                    "together with the next ones - one barrier for all their source images - before anything else "
                    "uses a source image or samples a texture they write, a second one into the same texture, and at "
                    "the frame's / submission's end. Census: deferred, batches")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_truth_drop_dead_resolves, 0, "GPU/Native",
                     "Session 66 (G1.1, G1.2): with gpu_native_truth_batch_resolves - 0 off; 1 a deferred texture-only "
                     "resolve (or texture fill) whose texels are all written again by a newer resolve into the same "
                     "texture, with nothing having sampled the texture between (the entry is still in the batch: "
                     "sampling flushes it), is dropped instead of flushing the batch, the newer one taking over its "
                     "held texture barrier; unscaled resolves only. 2 reserved for resolves kept pending across frames "
                     "(not built: acts as 1). Census: dropped, flushes by reason")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DECLARE(int32_t, gpu_native_truth_texture_fill_scaled_kinds);
REXCVAR_DEFINE_BOOL(gpu_native_truth_cover_outdated, false, "GPU/Native",
                    "Session 52: a resolve's texture write or fill into a single-level texture it covers whole is done "
                    "even when the texture is not up to date with memory (every texel is written: it is up to date "
                    "after it), not refused")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_batch_rename, false, "GPU/Native",
                    "With gpu_native_truth_batch_resolves (Session 52): a pass that opens with a full fast clear on an "
                    "image a deferred resolve reads gets a fresh image (from a pool) instead of flushing - the pixels "
                    "outside the clear are copied from the original at the flush (any draw outside the clear, or a "
                    "pass that also loads tiles, flushes first); census: renames")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_direct_resolve_scaled_texture, false, "GPU/Native",
                    "With gpu_native_truth_direct_resolve_scaled (Session 52): the scaled direct copy also writes the "
                    "scaled k_8_8_8_8 texture it feeds (its direct-load storage view; the guest memory's scaled copy is "
                    "written as before), which then counts as loaded: no texture load from the scaled resolve buffer")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_direct_resolve_scaled_full, false, "GPU/Native",
                    "With gpu_native_truth at a resolution scale (Session 52): a full k_16_16 / k_16_16_16_16 colour "
                    "copy of a surface one RGBA16 truth image holds whole is done straight from the image into the "
                    "scaled resolve buffer (the texture fill's conversion, the scaled texture load's layout), not through "
                    "the staging buffer and the stock full copy")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// Session 52 (step 1c).
REXCVAR_DEFINE_BOOL(gpu_native_truth_fill_direct, false, "GPU/Native",
                    "With gpu_native_truth_texture_fill (Session 52): a packed texture fill (k_16_16, k_16_16_16_16, "
                    "red / blue swapped k_8_8_8_8) into a single-level texture with the direct load's raw-bits storage "
                    "view writes the texture there, not into the staging buffer and a buffer-to-image copy (the same "
                    "bits; two barriers and a copy fewer)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// Session 59 (medium item c).
REXCVAR_DEFINE_BOOL(gpu_native_truth_direct_resolve_scaled_small, false, "GPU/Native",
                    "Session 59: with gpu_native_truth_direct_resolve_scaled_full, at a resolution scale a full k_8 / "
                    "k_4_4_4_4 copy of a surface one RGBA8 truth image holds whole is done straight from the image into "
                    "the scaled resolve buffer (the 1x direct copy's packing, the stock scaled copy's layout: 8-texel "
                    "units), not through the staging stores and the stock copy")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// Session 59 (medium item b).
REXCVAR_DEFINE_BOOL(gpu_native_truth_direct_resolve_depth, false, "GPU/Native",
                    "Session 59: with gpu_native_truth_direct_resolve, a fast 32bpp 1x depth copy of a surface one depth "
                    "truth image holds whole is done straight from the image into memory (its 20e4 / 24-bit depth and "
                    "its stencil, as the truth store writes them), not through the staging stores and the stock copy")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_fill_direct_depth, false, "GPU/Native",
                    "Session 59: with gpu_native_truth_fill_direct, a depth texture fill (k_24_8_FLOAT) into a texture "
                    "with the direct load's R32_UINT storage view writes the texture there (the same 32-bit float bits), "
                    "not into the staging buffer and a buffer-to-image copy")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// Session 59 (medium item a).
REXCVAR_DEFINE_BOOL(gpu_native_truth_fill_direct_small, false, "GPU/Native",
                    "With gpu_native_truth_fill_direct (Session 59): a k_8 / k_4_4_4_4 texture fill into a texture with "
                    "an R8_UINT / R16_UINT storage view (vulkan_texture_native_storage_small) writes the texture there, "
                    "one texel an invocation, not into the staging buffer and a buffer-to-image copy (the same bits)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// Session 52 (step 0, diagnostic: where the truth passes' GPU time goes).
// (Session 69 diagnostic: TruthStoreMany logs its decisions while a logged span is staged.)
static bool g_truth_debug_store = false;
REXCVAR_DEFINE_INT32(gpu_native_truth_pass_log, 0, "GPU/Native",
                     "Diagnostic (Session 72): the next N frames' truth passes logged one line each when their GPU "
                     "timestamps are read ([native][pass]: size, pitch, draws, the opening draw's shaders, GPU ms of the "
                     "pass and of the work before it); resets itself to 0 (live: cvar gpu_native_truth_pass_log N)")
    .range(0, 64)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_debug_renderdoc, false, "GPU/Native",
                    "Diagnostic (Session 69, red block): with RenderDoc attached, a frame capture starts right before "
                    "the first colour fast clear at x 960 of a gpu_native_truth_debug_span_pitch pass and ends two "
                    "frames later (NativeRenderer::OnFrameEnd); once a run");
// (Session 69: the capture's state, read by OnFrameEnd in native_renderer.cpp.)
bool g_truth_rdc_active = false, g_truth_rdc_done = false;
uint64_t g_truth_rdc_frame = 0;
REXCVAR_DEFINE_INT32(gpu_native_truth_debug_span_pitch, 0, "GPU/Native",
                     "Diagnostic (Session 69, red block): a staged span this many tiles wide (and pitch) logs its tile "
                     "owners - an image index per tile or E (the EDRAM buffer) - and the owners; at most 8 spans. 0 = off")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_direct_resolve_full_1x, false, "GPU/Native",
                    "Session 59: at 1x, full k_8 / k_4_4_4_4 copies of an RGBA8 truth image straight into memory (no "
                    "staging stores, no stock copy); the texture fill after it is unchanged")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_direct_resolve_full_1x_texture, false, "GPU/Native",
                    "Session 59: with gpu_native_truth_direct_resolve_full_1x, the copy also writes the k_8 / "
                    "k_4_4_4_4 texture it feeds (its R8_UINT / R16_UINT storage view, vulkan_texture_native_storage_small) "
                    "- no texture fill after it")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_direct_resolve_scaled_full_texture, false, "GPU/Native",
                    "Session 59: with gpu_native_truth_direct_resolve_scaled_full, the full k_16_16 / k_16_16_16_16 "
                    "copies at a resolution scale also write the scaled texture they feed (its storage view), so it is "
                    "not reloaded from the scaled resolve buffer afterwards")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_batch_fills, false, "GPU/Native",
                    "Session 59: with gpu_native_truth_batch_resolves, a truth texture fill straight into the "
                    "texture's storage view is deferred and recorded with the batched texture-only resolves (one "
                    "barrier pair for the batch instead of one per fill, and the batch is no longer flushed by every "
                    "fill)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_skip_restores, false, "GPU/Native",
                    "Session 72 part 4: in truth frames, a draw of a registered restore shader "
                    "(CommandProcessor::RegisterNativeRestoreShader: it writes back the colour and depth two resolves "
                    "took out) is left out when its two textures were resolved this frame, at the origin, from the very "
                    "images it draws into, nothing has drawn into those images since, and the draw is a plain copy "
                    "(whole colour mask, no blending, depth written always, no stencil write): the images still hold "
                    "those pixels (depth at its full precision - the restore would round it to 20e4)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_skip_repeat_resolves, false, "GPU/Native",
                    "Session 72 part 4: in truth frames, a resolve whose copy a direct path already did this frame with "
                    "every register the same, from an image no truth draw went into since (and that still owns the "
                    "copied tiles), with nothing copied into the destination since, records no GPU work (its clear "
                    "still runs; not in readback windows) - entrances copy the scene's colour and depth twice around a "
                    "restore gpu_native_truth_skip_restores leaves out")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_STRING(gpu_native_truth_alias_skip_tiles, "", "GPU/Native",
                      "Session 82 (diagnostic): EDRAM tile ranges (first-last, comma-separated) the truth path's alias loads leave out (the "
                      "image keeps what it has there) - which leftover tiles a picture depends on");
REXCVAR_DEFINE_BOOL(gpu_native_truth_debug_skip_alias, false, "GPU/Native",
                    "Diagnostic (Session 59), the picture is WRONG with it on: tiles another truth image owns are not "
                    "brought in (no staging stores, no alias loads) - the upper bound of loading them straight from "
                    "their owner (plan step 3) in in-run A/Bs")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_truth_debug_gpu_work, 0, "GPU/Native",
                     "Diagnostic (Session 52), the picture is WRONG with it on: 1 = every truth draw's scissor is one "
                     "pixel (its vertex work stays, its pixel work goes), 2 = no truth draw call is recorded at all "
                     "(the passes, loads, stores and resolves stay); in-run A/Bs of it split the passes' GPU time; "
                     "Session 56: 3 = as 1, and each draw cut to its first triangle (the vertex work goes too: what "
                     "stays is each draw's own cost)")
    .range(0, 3)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// Session 56 (step 1: the GPU's wait-for-idles - every barrier and every truth pass's render pass dependencies).
REXCVAR_DEFINE_BOOL(gpu_native_truth_lazy_pass_deps, false, "GPU/Native",
                    "Session 56: truth passes open render passes without external subpass dependencies (each one a "
                    "wait-for-idle on the GPU); what they synchronised is recorded as barriers only where needed - "
                    "after a pass, one barrier held until the next GPU work outside a truth pass (joined with that "
                    "work's own barrier), before a pass only when something else ran since the last pass or the pass "
                    "uses an image a pass used since that barrier (DeferredCommandBuffer::CmdVkBeginRenderPassLazy)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_truth_lazy_pass_mode, 3, "GPU/Native",
                     "Session 56 (for measuring, with gpu_native_truth_lazy_pass_deps): bit 0 the barrier after truth "
                     "passes, bit 1 the barrier before a pass after other work (a barrier before a pass on an image a "
                     "pass used since the last one is always recorded). Below 3 the other work's own barriers must "
                     "order it")
    .range(0, 3)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_truth_lazy_pass_style, 0, "GPU/Native",
                     "Session 56 (for measuring, with gpu_native_truth_lazy_pass_deps): 0 the lazy passes' barriers are "
                     "memory barriers (every image), 1 image barriers of the attachments involved, 2 memory barriers "
                     "without depth / stencil accesses (diagnostic: not enough for depth)")
    .range(0, 2)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_INT32(gpu_native_truth_debug_lazy_ignore_depth, 0, "GPU/Native",
                     "Diagnostic (Session 56), the picture may be WRONG with it on: with gpu_native_truth_lazy_pass_deps, "
                     "a lazy pass's depth image is not looked at for the barrier before it (what a depth image of its "
                     "own for each pass could gain): 1 every depth image, 2 only the scratch ones (passes no draw of "
                     "which uses depth)")
    .range(0, 2)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_rename_depth, false, "GPU/Native",
                    "Session 56 (with gpu_native_truth_lazy_pass_deps and gpu_native_truth_batch_rename): when a pass "
                    "renames its colour image and its first draw (a fast clear) writes all of the depth and stencil in "
                    "its rectangle, the depth image is renamed too (its pixels outside the rectangle copied at the "
                    "flush, as the colour's) - chains of such passes (the tiny clear-first passes on one depth image) "
                    "then need no barrier between them")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 73): about 15 of a match frame's 28 truth passes draw only depth into a real colour image
// (validation best practices on tile GPUs: "redundant-attachment-on-tile"), which was loaded and stored whole each time.
REXCVAR_DEFINE_BOOL(gpu_native_truth_color_none, false, "GPU/Native",
                    "Session 73: a truth pass none of whose draws or loads touch its colour image keeps it without "
                    "loading or storing it (LOAD_OP_NONE / STORE_OP_NONE, VK_EXT_load_store_op_none; the recorded pass "
                    "replaced at the pass's end). Exact; saves a tile GPU two full-image transfers per such pass")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_pass_dont_care, false, "GPU/Native",
                    "Session 66 (G2.1, with gpu_native_truth_lazy_pass_deps): a truth pass whose first draw is a fast "
                    "clear of the whole surface (every colour channel written, no blending; or depth always + written "
                    "and the stencil replaced whole on both faces) opens without loading that attachment (load op "
                    "DONT_CARE: memory traffic on tile and bandwidth-poor GPUs; the clear draw still writes every pixel)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_alias_census, false, "GPU/Native",
                    "Session 79 (diagnostic): of the tiles alias-loaded into a truth pass, how many a later draw of the "
                    "same pass covers whole without reading them (dead loads), reads (consumed) or leaves pending to "
                    "the pass end (kept) - a lower bound for dead stores ([native][truth] alias census lines)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_clear_attachments, false, "GPU/Native",
                    "Session 56: a truth fast clear that writes all of the depth and stencil in its rectangle, with a "
                    "depth the CPU knows exactly (its vertex shader run on the CPU: 0 or 1 through a plain viewport), is "
                    "followed by vkCmdClearAttachments of the same values - a real depth clear, which the GPU's "
                    "hierarchical depth culling (ZCULL) needs to start working (Nsight: it rejected nothing)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 80, the frame structure: the scene's clears).
REXCVAR_DEFINE_STRING(gpu_native_truth_clear_load, "", "GPU/Native",
                      "Session 80: guest pixel shaders (hex hashes, comma-separated) that write the vertex shader's first "
                      "interpolator as their colour. A truth pass whose first draw is a whole-surface clear by one of "
                      "them into an 8-bit or 16-bit colour target, colour, depth and stencil all known on the CPU (its vertex "
                      "shader run there; with gpu_native_truth_clear_attachments and gpu_native_truth_pass_dont_care), "
                      "begins with those values cleared by the render pass itself and leaves the draw out - a GPU's "
                      "fast clear in place of a full-screen draw. Empty = off. Live")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
// wwe2k14 patch (Session 80, the frame structure: what a draw touches).
REXCVAR_DEFINE_STRING(gpu_native_truth_bounds_vs, "", "GPU/Native",
                      "Session 80: guest vertex shaders (hex hashes, comma-separated) whose truth draws get the box their "
                      "vertices span: every vertex the draw uses run through the shader on the CPU (cached by its "
                      "inputs), the box of the screen positions widened by 2 pixels. The draw's rectangle - the tiles it "
                      "loads and owns, its scissor - is narrowed to it: a draw whose scissor is its whole surface but "
                      "whose triangles touch a corner of it no longer brings in every tile another image holds (the "
                      "sweat maps' passes). Exact: nothing outside the box is rasterized. Empty = off. Live")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_dedupe_state, false, "GPU/Native",
                    "Session 56: in a truth pass, the index buffer is bound once at offset 0 and each indexed draw "
                    "starts at its first index (not a bind per draw), and the stencil reference / masks are set only "
                    "when they change - fewer state commands a draw (the per-draw GPU overhead)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_debug_scaled_texture_only, false, "GPU/Native",
                    "Diagnostic (Session 56), the picture may be WRONG with it on: at a resolution scale, a direct "
                    "resolve that also writes its scaled texture does not write the scaled resolve buffer (what the "
                    "scaled write-backs cost - the ceiling of texture-only resolves at a scale)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_debug_skip_alias_restore, false, "GPU/Native",
                    "Diagnostic (Session 79; wrong image layouts, measurement only): the direct alias loads' owner "
                    "images are not put back into their attachment layouts - the ceiling of a lazy restore (how many "
                    "lazy pass chains the restore barrier breaks)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_defer_memory_resolves, false, "GPU/Native",
                    "Session 79 (HD 520 arc): with gpu_native_truth_batch_resolves, a 1x direct resolve that writes "
                    "guest memory (no texture holds all it writes) joins the deferred batch too, so it shares the "
                    "batch's barrier pair; its pages are marked as the texture-only records' are, so any reader of them "
                    "(texture, vertex, index, memexport) or a later resolve over them flushes the batch first. Not in "
                    "readback windows")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_fill_rename_check, false, "GPU/Native",
                    "Session 79 (review fix, a gap since Session 59): a texture fill that reads a renamed image outside "
                    "its rename rectangle is not deferred with the batch (the flush's copy brings those pixels first). "
                    "Off by default (on under the game's arc switch) until Jarrod's call: it only adds a flush in that "
                    "rare ordering")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_alias_direct_lazy_flush, false, "GPU/Native",
                    "Session 79 (HD 520 arc): with gpu_native_truth_alias_direct, the deferred batch is flushed before "
                    "the owners are read straight only when one of them is a renamed image (its pixels outside the "
                    "rename rectangle come at the flush), not every time")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_alias_direct_rgba16, false, "GPU/Native",
                    "Session 79 (HD 520 arc): with gpu_native_truth_alias_direct, RGBA16 owners (k_16_16 / "
                    "k_16_16_16_16 surfaces, 64bpp addressing for the latter) are read straight too, packed as the store "
                    "packs them, instead of stored first")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_defer_full_resolves, false, "GPU/Native",
                    "Session 79 (HD 520 arc): with gpu_native_truth_batch_resolves, a 1x full k_8 / k_4_4_4_4 copy "
                    "(gpu_native_truth_direct_resolve_full_1x, with its texture write if it has one) joins the deferred "
                    "batch too, instead of flushing it and running alone (as gpu_native_truth_defer_memory_resolves; not "
                    "in readback windows)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_drop_dead_copies, false, "GPU/Native",
                    "Session 80 (frame structure): a texture-only copy recorded this submission whose texture a later "
                    "copy covers before anything used it (sampled, swapped, loaded, written into memory from it) is "
                    "turned into no-ops before the submission is replayed - it was read by nothing (Bryan's entrance: "
                    "the scene saved for a restore NATIVE leaves out, 4 full-screen copies a frame)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_drop_dead_passes, false, "GPU/Native",
                    "Session 80 (frame structure): a truth pass that loads nothing, leaves its depth alone and has no "
                    "other effect (memexport, occlusion queries, captured positions) whose colour the next pass's first "
                    "draw clears whole, with nothing recorded in between that reads an image, is turned into no-ops "
                    "(its state commands and timestamps kept) - its pixels were read by nothing (Bryan's entrance: one "
                    "full-screen pass a frame)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_carry_copies, false, "GPU/Native",
                    "Session 80 (frame structure): with gpu_native_truth_batch_resolves, a deferred texture-only colour "
                    "copy stays in the batch through flushes that do not concern it (another texture used, another "
                    "image written, the frame's / submission's end), its texture pinned - it runs when its texture is "
                    "used or its source written, and a later copy covering it first drops it (the match: the scene "
                    "copy nothing samples); a full-screen draw (proven coverage, nothing read under it) into its source "
                    "renames the image as a clear does")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_serve_copies, false, "GPU/Native",
                    "Session 80 (frame structure): with gpu_native_truth_carry_copies, a draw sampling a texture a "
                    "carried plain copy (all of a k_8_8_8_8 image from its origin, unchanged, 1x) is still to write "
                    "samples the copy's source image instead - the copy then has no use, and the next copy into the "
                    "texture drops it. An image renamed away (the match's arena screen sampling the previous frame's "
                    "scene), a slot's image that is none of the draw's targets (the downsample), or the draw's own "
                    "colour target, renamed first (the combine). A red / blue swap is undone by the view. Needs "
                    "gpu_native_truth_exact_rows for the scene (its image 720 rows, as the copy)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_serve_ahead, false, "GPU/Native",
                    "Session 80 (frame structure): with gpu_native_truth_serve_copies, a serve that ended an open pass "
                    "(its image is not readable inside a pass - the match's arena screen, mid scene pass) is "
                    "remembered: the next time that colour slot's pass opens with the texture's carried copy pending, "
                    "the copy's source image is made readable before the pass begins (the barrier the serve would "
                    "have recorded, restored at the pass's end) - the serve then ends nothing. Only barriers move. Live")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_debug_serve_no_swap, false, "GPU/Native",
                    "Session 80 (diagnostic, wrong colours on purpose): gpu_native_truth_serve_copies leaves a copy's red "
                    "/ blue swap in place - a picture check that does not change sees no served read")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_exact_rows, false, "GPU/Native",
                    "Session 80 (frame structure): truth images grow in rows of tiles (16) instead of 64-row steps - "
                    "the match's scene image 1280x720 as its copies, not 1280x768, so gpu_native_truth_serve_copies "
                    "can serve them; made-ahead images (gpu_native_truth_prewarm) from their own records ('R' lines)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_skip_writeback_depth, false, "GPU/Native",
                    "Session 80 (frame structure): with gpu_native_truth_skip_writeback, texture-only depth copies too - a "
                    "1x 20e4 depth copy whose depth texture fill holds every texel it copies does not write guest memory "
                    "(3.7 MB a 720p copy; Bryan's entrance makes four a frame); its stencil is kept (a snapshot of the "
                    "image's, or the last clear's value) and the memory is written from the texture and the stencil only "
                    "when something requests it, as the colour records (census: [native][wb] depth)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_defer_depth_resolves, false, "GPU/Native",
                    "Session 79 (HD 520 arc): with gpu_native_truth_batch_resolves, a 1x direct depth copy (into memory) "
                    "and its depth texture fill join the deferred batch too, instead of flushing it and running alone "
                    "with their own barriers (as gpu_native_truth_defer_memory_resolves; not in readback windows)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_alias_direct, false, "GPU/Native",
                    "Session 79 (HD 520 arc): a pass's tiles another RGBA8 or depth image owns are loaded straight from "
                    "that image (the fragment shader packs them as the store would, then converts them as the load "
                    "does) instead of stored into the staging buffer by a compute dispatch first - fewer compute "
                    "dispatches and 3D / compute switches. Owners that are the pass's own images, and RGBA16 owners, "
                    "are stored as before")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_texture_only_cube_faces, false, "GPU/Native",
                    "Session 79 (HD 520 arc): with gpu_native_truth_fill_cube_faces and gpu_native_truth_skip_writeback, "
                    "a direct resolve into a cube face it fills is texture-only too (recorded, written into memory "
                    "from the face only when memory is read) - so a cube's faces batch under one barrier pair")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_skip_writeback_scaled, false, "GPU/Native",
                    "Session 56: at a resolution scale, a direct RGBA8 resolve that writes its scaled texture "
                    "(gpu_native_truth_direct_resolve_scaled_texture) skips its write into the scaled resolve buffer "
                    "and is recorded instead; the record is written there (from the texture, "
                    "kNativeMaterializeScaledShader) before a scaled texture load or another resolve reads or writes "
                    "the memory. Such resolves are also batched (gpu_native_truth_batch_resolves) and renamed, as at "
                    "1x. The resolution-scale counterpart of gpu_native_truth_skip_writeback")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DEFINE_BOOL(gpu_native_truth_debug_pass_no_deps, false, "GPU/Native",
                    "Diagnostic (Session 56), the picture may be WRONG with it on: truth passes open render passes "
                    "without external subpass dependencies and nothing replaces them (what their wait-for-idles cost)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);
REXCVAR_DECLARE(bool, gpu_native_truth_direct_resolve);
REXCVAR_DECLARE(bool, vulkan_texture_sample_written_early);  // Session 56
REXCVAR_DECLARE(bool, gpu_native_truth_async_pipelines);  // Session 47
REXCVAR_DECLARE(bool, async_shader_compilation);
REXCVAR_DECLARE(bool, gpu_native_truth_trust_bindings);
REXCVAR_DECLARE(bool, gpu_native_lean);
REXCVAR_DECLARE(int32_t, gpu_native_truth_texture_fill);
REXCVAR_DECLARE(int32_t, gpu_native_truth_texture_fill_small_swap_scaled);
REXCVAR_DECLARE(bool, gpu_native_truth_texture_fill_scaled);
REXCVAR_DECLARE(bool, gpu_ram_stats);  // Session 49
REXCVAR_DECLARE(bool, gpu_native_truth_skip_writeback);
REXCVAR_DECLARE(int32_t, gpu_native_truth_skip_writeback_max_age);
REXCVAR_DECLARE(bool, gpu_native_truth_skip_writeback_debug_immediate);
REXCVAR_DECLARE(bool, gpu_native_truth_texture_fill_lazy);

#if !REX_TRUE_NATIVE_ONLY  // wwe2k14 patch (Session 92, M2d): the truth path (recorder, stores, staging, twins, deferred copies, materialise, write-out, replay) is not part of the TRUE NATIVE proof build

namespace rex::graphics::vulkan::native {

namespace {

// The store: an image's pixels into a buffer in the EDRAM's layout, packed as the interlock pixel shaders pack them
// (the post write-back's packing, every pixel of the rectangle). Colour: k_8_8_8_8 from RGBA8, k_16_16 /
// k_16_16_16_16 from RGBA16_SNORM (the packed 16-bit values); depth (DEPTH defined): 20e4 or unorm24 into the upper 24
// bits, the stencil byte kept. One invocation per host pixel of the rectangle.
constexpr char kTruthStoreComputeShader[] = R"(
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) buffer Out { uint edram[]; };
layout(set = 0, binding = 1) uniform sampler2D source;
#ifdef DEPTH
layout(set = 0, binding = 2) uniform usampler2D stencil_source;  // Session 46: the image's stencil
#endif
layout(push_constant) uniform Constants {
  uint base_dwords;
  uint pitch_tile_dwords;  // a row of 32bpp tiles
  uint tile_w;             // host pixels (80 x the resolution scale)
  uint tile_h;
  uint origin_x;           // the rectangle's first pixel
  uint origin_y;
  uint width;              // its size
  uint height;
  uint wrap_dwords;
  uint flags;              // bit 0: 20e4 depth; bit 1: keep the stencil byte; bits 4-7: the colour format (0, 4, 5)
} c;
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
  uvec2 g = gl_GlobalInvocationID.xy;
  if (g.x >= c.width || g.y >= c.height) return;
  uvec2 p = g + uvec2(c.origin_x, c.origin_y);
  uint tile_dwords = c.tile_w * c.tile_h;
  uint x = p.x % c.tile_w, y = p.y % c.tile_h, half_w = c.tile_w >> 1;
  uint row = (p.y / c.tile_h) * c.pitch_tile_dwords;
#ifdef DEPTH
  // Depth: the 40-sample halves of each tile row swapped (spirv_translator_rb.cpp FSI_LoadEdramOffsets).
  uint xd = x < half_w ? x + half_w : x - half_w;
  uint i = (c.base_dwords + row + (p.x / c.tile_w) * tile_dwords + y * c.tile_w + xd) % c.wrap_dwords;
  float z = texelFetch(source, ivec2(p), 0).r;
  // The stencil byte: the image's own where it holds the tile's stencil (flags bit 2, Session 46); otherwise kept
  // writing into the EDRAM (bit 1), or into the staging buffer the image's last clear's (bits 8-15).
  uint s = (c.flags & 4u) != 0u ? (texelFetch(stencil_source, ivec2(p), 0).r & 0xFFu)
           : (c.flags & 2u) != 0u ? (edram[i] & 0xFFu) : ((c.flags >> 8) & 0xFFu);
  edram[i] = (DepthToEdram(z, (c.flags & 1u) != 0u) << 8) | s;
#else
  uint format = (c.flags >> 4) & 0xFu;
  vec4 raw = texelFetch(source, ivec2(p), 0);
  if (format == 0u) {
    uvec4 u = uvec4(round(clamp(raw, 0.0, 1.0) * 255.0));
    uint i = (c.base_dwords + row + (p.x / c.tile_w) * tile_dwords + y * c.tile_w + x) % c.wrap_dwords;
    edram[i] = u.r | (u.g << 8) | (u.b << 16) | (u.a << 24);
  } else {
    uvec4 s = uvec4(ivec4(round(clamp(raw, -1.0, 1.0) * 32767.0)));
    uint d0 = (s.r & 0xFFFFu) | (s.g << 16), d1 = (s.b & 0xFFFFu) | (s.a << 16);
    if (format == 4u) {
      edram[(c.base_dwords + row + (p.x / c.tile_w) * tile_dwords + y * c.tile_w + x) % c.wrap_dwords] = d0;
    } else {
      // 64bpp: 40 x 16 pixels per tile, two dwords each, rows of tiles twice the 32bpp pitch.
      uint o = (c.base_dwords + (p.y / c.tile_h) * c.pitch_tile_dwords * 2u + (p.x / half_w) * tile_dwords +
                y * c.tile_w + (p.x % half_w) * 2u) % c.wrap_dwords;
      edram[o] = d0;
      edram[(o + 1u) % c.wrap_dwords] = d1;
    }
  }
#endif
}
)";

// Session 79 (gpu_native_truth_alias_direct): the alias load without the store. For each pixel of the pass's image
// (as kLoadFragmentShader: its dword index in the EDRAM layout), the owner image's pixel that TruthStore would have
// written into that dword (kTruthStoreComputeShader's addressing, inverted), packed exactly as the store packs it,
// then converted exactly as the load converts it. OWNER_DEPTH: the owner is a depth image (20e4 / unorm24 + stencil:
// its stencil view, or its last clear's value).
constexpr char kTruthAliasDirectFragmentShader[] = R"(
layout(set = 0, binding = 1) uniform sampler2D owner;
#ifdef OWNER_DEPTH
layout(set = 0, binding = 2) uniform usampler2D owner_stencil;
#endif
layout(location = 0) out vec4 out_color;
layout(push_constant) uniform Constants {
  uint color_base_dwords;
  uint depth_base_dwords;
  uint pitch_tile_dwords;
  uint tile_w;
  uint tile_h;
  uint width;
  uint height;
  uint wrap_dwords;
  uint flags;  // bit 0: the pass's depth is 20e4; bit 1: a colour surface
  float scale_x;
  float scale_y;
  uint origin_x;
  uint origin_y;
  uint owner_base_dwords;
  uint owner_pitch_tile_dwords;
  uint owner_flags;  // bit 0: 20e4 owner depth; bit 2: stencil from the view (else bits 8-15); bit 16: the depth side
} c;
uvec2 EdramIndices(uvec2 p) {
  uint tile = (p.y / c.tile_h) * c.pitch_tile_dwords + (p.x / c.tile_w) * (c.tile_w * c.tile_h);
  uint x = p.x % c.tile_w, y = p.y % c.tile_h, half_w = c.tile_w >> 1;
  uint xd = x < half_w ? x + half_w : x - half_w;
  return uvec2((c.color_base_dwords + tile + y * c.tile_w + x) % c.wrap_dwords,
               (c.depth_base_dwords + tile + y * c.tile_w + xd) % c.wrap_dwords);
}
float DepthFromEdram(uint d24, bool is_float) {
  if (!is_float) return float(d24) / 16777215.0;
  uint e = d24 >> 20, m = d24 & 0xFFFFFu;
  if (e == 0u) return float(m) * exp2(-34.0);
  return uintBitsToFloat(((e + 112u) << 23) | (m << 3));
}
#ifdef OWNER_DEPTH
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
#endif
void main() {
  uvec2 p = uvec2(min(floor(gl_FragCoord.xy / vec2(c.scale_x, c.scale_y)),
                      vec2(float(c.width - 1u), float(c.height - 1u))));
  uvec2 i = EdramIndices(p);
  uint index = (c.owner_flags & 0x10000u) != 0u ? i.y : i.x;
  // The owner pixel whose store writes this dword: tile (relative to the owner's base) and place in the tile.
  uint tile_dwords = c.tile_w * c.tile_h;
  uint rel = (index + c.wrap_dwords - c.owner_base_dwords % c.wrap_dwords) % c.wrap_dwords;
  uint tile = rel / tile_dwords, within = rel % tile_dwords;
  uint pitch_tiles = max(c.owner_pitch_tile_dwords / tile_dwords, 1u);
  uint x = within % c.tile_w, y = within / c.tile_w;
#ifdef OWNER_DEPTH
  uint half_w = c.tile_w >> 1;
  x = x < half_w ? x + half_w : x - half_w;  // (the store swaps the 40-sample halves: swapped back)
#endif
  ivec2 q = ivec2((tile % pitch_tiles) * c.tile_w + x, (tile / pitch_tiles) * c.tile_h + y);
  uint dword;
#ifdef OWNER_DEPTH
  float z = texelFetch(owner, q, 0).r;
  uint s = (c.owner_flags & 4u) != 0u ? (texelFetch(owner_stencil, q, 0).r & 0xFFu) : ((c.owner_flags >> 8) & 0xFFu);
  dword = (DepthToEdram(z, (c.owner_flags & 1u) != 0u) << 8) | s;
#else
  // (Session 79, gpu_native_truth_alias_direct_rgba16: bits 20-23 the owner's colour format - 0 k_8_8_8_8, 4 k_16_16,
  // 5 k_16_16_16_16 from an RGBA16_SNORM image, packed as kTruthStoreComputeShader packs them.)
  uint format = (c.owner_flags >> 20) & 0xFu;
  bool high = false;
  if (format == 5u) {
    // 64bpp: 40 x 16 pixels a tile, two dwords each, rows of tiles twice the 32bpp pitch (the store's addressing).
    uint row_dwords = c.owner_pitch_tile_dwords * 2u;
    uint r2 = rel % row_dwords, w2 = r2 % tile_dwords, xx = w2 % c.tile_w;
    q = ivec2((r2 / tile_dwords) * (c.tile_w >> 1) + (xx >> 1u), (rel / row_dwords) * c.tile_h + w2 / c.tile_w);
    high = (xx & 1u) != 0u;
  }
  vec4 raw = texelFetch(owner, q, 0);
  if (format == 0u) {
    uvec4 u = uvec4(round(clamp(raw, 0.0, 1.0) * 255.0));
    dword = u.r | (u.g << 8) | (u.b << 16) | (u.a << 24);
  } else {
    uvec4 s = uvec4(ivec4(round(clamp(raw, -1.0, 1.0) * 32767.0)));
    dword = high ? (s.b & 0xFFFFu) | (s.a << 16) : (s.r & 0xFFFFu) | (s.g << 16);
  }
#endif
  out_color = (c.flags & 2u) != 0u ? unpackUnorm4x8(dword) : vec4(0.0);
  gl_FragDepth = DepthFromEdram(dword >> 8, (c.flags & 1u) != 0u);
}
)";


constexpr VkFormat kTruthColorFormats[3] = {VK_FORMAT_UNDEFINED, VK_FORMAT_R8G8B8A8_UNORM,
                                            VK_FORMAT_R16G16B16A16_SNORM};
constexpr uint32_t kTruthMaxImages = 96;


}  // namespace

int16_t NativeRenderer::TruthFindImage(uint32_t base, uint32_t pitch_tiles, uint32_t kind, uint32_t format) {
  for (size_t i = 0; i < truth_images_.size(); ++i) {
    const TruthImage& t = truth_images_[i];
    if (t.base == base && t.pitch_tiles == pitch_tiles && t.kind == kind && t.format == format) return int16_t(i);
  }
  if (truth_images_.size() >= kTruthMaxImages) return -1;
  TruthImage& added = truth_images_.emplace_back();
  added.base = base;
  added.pitch_tiles = pitch_tiles;
  added.kind = kind;
  added.format = format;
  return int16_t(truth_images_.size() - 1);
}

bool NativeRenderer::ClaimTruth(Draw& draw) {
  const bool owned_only = OwnedOnly();  // Session 92 (M2a)
  const auto refuse = [&](uint32_t reason) {
    if (owned_only) {
      // Session 92 (M2a, gpu_native_owned_only): nothing is handed back - the draw is claimed (owned, truth) and TruthRecordDraw drops it, counted
      // and named by this reason; the emulated tier is never reached. A draw with no pipeline (yet) is skipped there, as ever.
      OwnedNoteClaimDrop(reason, draw);
      draw.claim_refused = uint8_t(reason);
      draw.truth = draw.post = draw.owned = true;
      draw.truth_epoch = truth_claim_epoch_;
      return true;
    }
    truth_fallback_reason_ = reason;
    claim_retry_ = true;  // the caller ends the render pass, BeforeEmulatedEdramAccess falls back, claims again
    if (truth_reason_logged_[reason] < 6) {
      ++truth_reason_logged_[reason];
      REXGPU_INFO("[native][truth] frame {}: not claimed ({}): vs {:016X} ps {:016X} primitive {} supported {} "
                  "expanded {} pitch {} msaa {} colour 0x{:08X} depth 0x{:08X} masks 0x{:X} depth test {} write {} "
                  "stencil {} fast clear {} blend 0x{:08X} shader {} float24 {} quantize {}",
                  frame_, TruthReasonName(reason), draw.vertex_shader_hash, draw.pixel_shader_hash,
                  draw.guest_primitive, draw.raster.supported, draw.raster.expanded, draw.surface_info & 0x3FFF,
                  (draw.surface_info >> 16) & 3, draw.color_info, draw.depth_info, draw.color_mask_all,
                  draw.raster.depth_test, draw.raster.depth_write, draw.raster.stencil, draw.fast_clear,
                  draw.blend_control, draw.pixel_shader != VK_NULL_HANDLE, draw.float24_depth, draw.color_quantize);
    }
    return false;
  };
  const uint32_t reason = TruthReason(draw, true);
  if (reason) return refuse(reason);
  if (const int32_t limit = REXCVAR_GET(gpu_native_truth_debug_draws); limit > 0 && truth_frame_draws_ >= limit) {
    return refuse(kTruthDebug);
  }
  ++truth_frame_draws_;
  const uint32_t msaa = (draw.surface_info >> 16) & 3;  // 0 1x, 1 2x, 2 4x
  const uint32_t sx = msaa >= 2 ? 2 : 1, sy = msaa >= 1 ? 2 : 1;
  const uint32_t pitch_tiles = std::max(1u, ((draw.surface_info & 0x3FFF) * sx + 79) / 80);
  const bool writes_color = (draw.color_write_mask & 0xF) != 0;
  const bool uses_depth = draw.raster.depth_test || draw.raster.depth_write || draw.raster.stencil;
  int16_t color = -1, depth = -1;
  if (writes_color) {
    color = TruthFindImage(draw.color_info & 0xFFF, pitch_tiles, TruthClass(draw), (draw.color_info >> 16) & 0xF);
    if (color < 0) return refuse(kTruthImages);
  }
  if (uses_depth) {
    depth = TruthFindImage(draw.depth_info & 0xFFF, pitch_tiles, 3, (draw.depth_info >> 16) & 1);
    if (depth < 0) return refuse(kTruthImages);
  }
  // A stencil draw (not a clear): only over tiles whose stencil its depth image holds (a truth clear's, while the
  // image owns the tile); elsewhere the stencil is the EDRAM's or another surface's - the frame falls back.
  // (Session 92, M2a: under the latch stencil is ours - the draw reaches OwnedTakeDraw, which takes it into a pass that keeps the depth's stencil or drops it.)
  if (!owned_only && draw.raster.stencil && !draw.fast_clear) {
    draw.truth_sx = uint8_t(sx);
    draw.truth_sy = uint8_t(sy);
    int32_t rect[4];
    TruthDrawRect(draw, rect);
    uint32_t scale_x = 1, scale_y = 1;
    command_processor_.NativeResolutionScale(scale_x, scale_y);
    const TruthImage& d_image = truth_images_[depth];
    // (The rows past the last whole row of tiles share tile numbers with the first rows: refused - review.)
    const uint32_t image_rows = (2048u + d_image.row_tiles() - 1) / d_image.row_tiles() * 16;
    const uint32_t whole_rows = 2048u / d_image.row_tiles() * 16;
    const uint32_t gx0 = uint32_t(rect[0]) / scale_x, gy0 = uint32_t(rect[1]) / scale_y;
    const uint32_t gx1 = (uint32_t(std::max(rect[2], rect[0])) + scale_x - 1) / scale_x;
    const uint32_t gy1 = std::min((uint32_t(std::max(rect[3], rect[1])) + scale_y - 1) / scale_y, image_rows);
    if (gy1 > whole_rows && gx1 > gx0) return refuse(kTruthStencil);
    if (gx1 > gx0 && gy1 > gy0) {
      const std::bitset<2048> tiles = EdramTiles(d_image.base, d_image.pitch_tiles, false, gx0, gy0, gx1, gy1);
      // (Session 92, D1, TRUE NATIVE: tiles inside the box of an owned pass that keeps this depth's stencil count as held - the scene's
      // clears are our pass's load-op clear, never a truth clear, so stencil_valid is never set for them; OwnedMarkDirty / OwnedMaterialize
      // make the truth image's stencil ours before a truth draw reads it.)
      if ((tiles & ~(d_image.stencil_valid & d_image.owned)).any() && !OwnedHoldsStencil(depth, d_image.pitch_tiles, rect)) {
        return refuse(kTruthStencil);  // (Session 47)
      }
    }
  }
  draw.truth = true;
  draw.post = true;
  draw.truth_color = color;
  draw.truth_depth = depth;
  draw.truth_class = uint8_t(writes_color ? TruthClass(draw) : 1u);
  draw.truth_sx = uint8_t(sx);
  draw.truth_sy = uint8_t(sy);
  // Session 47 (CPU): a pipeline being made on the worker thread - the draw is claimed and skipped (TruthRecordDraw
  // draws nothing without a pipeline), as the emulated path skips a draw whose pipeline is still being made.
  bool pending = false;
  // (Review: only where the emulated path skips too - async_shader_compilation - so such frames are held back alike.)
  const bool async = REXCVAR_GET(gpu_native_truth_async_pipelines) && REXCVAR_GET(async_shader_compilation);
  // Session 55 (the red blocks / white wrestlers on slow laptops): only a draw into a screen-wide surface may be
  // skipped while its pipeline is made (that frame shows without it, the next has it). A draw into a narrower surface
  // usually builds a texture the game keeps (the wrestlers' while a match loads): skipped, it stays missing all match.
  // A 2-core laptop skipped 109,595 draws; here, compiles slowed down to 100 ms (gpu_native_truth_debug_pipeline_delay_ms,
  // run 3729) left wrestlers' trunks, waistbands and skin washed out for the whole match. Those draws wait instead.
  // (Session 57: the width is gpu_native_truth_pipeline_skip_min_pitch, kTruthScreenPitchTiles by default.)
  // (Session 58: and at most gpu_native_truth_pipeline_skip_max_pitch wide - a character's attire surface is wider.)
  const bool may_skip = (pitch_tiles >= uint32_t(REXCVAR_GET(gpu_native_truth_pipeline_skip_min_pitch)) &&
                         pitch_tiles <= uint32_t(REXCVAR_GET(gpu_native_truth_pipeline_skip_max_pitch))) ||
                        !REXCVAR_GET(gpu_native_truth_pipeline_wait_offscreen);
  const size_t pipelines_before = pipelines_.size();
  draw.pipeline = GetPipeline(draw, async && may_skip ? &pending : nullptr);
  if (async && !may_skip && pipelines_.size() != pipelines_before) ++truth_counts_.waited;
  if (pending) {
    ++truth_counts_.compiling;
    draw.owned = true;
    draw.truth_epoch = truth_claim_epoch_;
    return true;
  }
  if (draw.pipeline == VK_NULL_HANDLE) {
    draw.truth = draw.post = false;
    return refuse(kTruthNoPipeline);
  }
  draw.owned = true;
  draw.truth_epoch = truth_claim_epoch_;
  return true;
}

// Session 47 (CPU): a repeat has its base's registers but for the constants (the command processor's repeat path), so
// whatever ClaimTruth found for the base holds for it while nothing reset the images or owners since (the epoch): the
// same images, pipeline and stencil (the base made the tiles it touched its images' own).
bool NativeRenderer::EnsureTruthObjects() {
  if (truth_store_pipelines_[1] != VK_NULL_HANDLE) return true;
  if (truth_failed_) return false;
  const bool post_objects = EnsurePostObjects();
  if (!post_objects || post_ds_format_ != VK_FORMAT_D32_SFLOAT_S8_UINT || !post_class_ok_[1] ||
      !post_class_ok_[2] || post_render_pass_begin_keep_[1] == VK_NULL_HANDLE ||
      post_render_pass_begin_keep_[2] == VK_NULL_HANDLE) {
    // (Session 93, report 033: the line now says which part is missing - the old one named three formats when the cause was a fourth.)
    REXGPU_WARN("[native][truth] no truth frames: the post objects are missing (made: {}; depth {} - D32S8 needed; RGBA8 + blend: {}; "
                "RGBA16_SNORM target: {}; when not made, the [native][post] line says why)",
                post_objects ? "yes" : "NO",
                post_ds_format_ == VK_FORMAT_D32_SFLOAT_S8_UINT ? "D32S8" : post_ds_format_ == VK_FORMAT_D24_UNORM_S8_UINT ? "D24S8" : "none",
                post_class_ok_[1] ? "yes" : "no", post_class_ok_[2] ? "yes" : "no");
    truth_failed_ = true;
    {  // Session 95 (PLAIN, review): the reason is kept; owned_steps.cpp asks the fatal once the latch is set (OwnedOnlyWanted is still false here)
      truth_fail_reason_ = std::string("post objects (depth ") +
                               (post_ds_format_ == VK_FORMAT_D32_SFLOAT_S8_UINT ? "D32S8" : post_ds_format_ == VK_FORMAT_D24_UNORM_S8_UINT ? "D24S8 only" : "none") +
                               ", RGBA8 blend " + (post_class_ok_[1] ? "yes" : "no") + ", RGBA16_SNORM target " + (post_class_ok_[2] ? "yes" : "no") + ")";
    }
    return false;
  }
  if (OwnedOnlyWanted()) {
    // Session 92, M2d (Part 1): under the owned-only latch the frame needs no truth object (the store / fill pipelines, the pool: nothing below the owned frame is reached; the
    // set layouts, the staging and the resolve pipelines the owned write-back and the page images use are made by owned_wb_objects.cpp). Only the one fact the claim asks
    // (TruthReason: a 16-bit blended draw needs the format's blend feature) is looked up. A release of the latch (a hard failure, with an EDRAM buffer) makes the rest here.
    static bool class2_queried = false;
    if (class2_queried) return true;
    class2_queried = true;
    const ui::vulkan::VulkanDevice* device_for_format = command_processor_.GetVulkanDevice();
    VkFormatProperties properties;
    device_for_format->vulkan_instance()->functions().vkGetPhysicalDeviceFormatProperties(device_for_format->physical_device(), kTruthColorFormats[2], &properties);
    truth_class2_blend_ = (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) != 0;
    return true;
  }
  truth_failed_ = true;  // until everything below exists
  TruthDrawRatio();      // Session 54 (decided once)
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const ui::vulkan::VulkanInstance::Functions& ifn = vulkan_device->vulkan_instance()->functions();
  const VkDevice device = vulkan_device->device();
  {
    VkFormatProperties properties;
    ifn.vkGetPhysicalDeviceFormatProperties(vulkan_device->physical_device(), kTruthColorFormats[2], &properties);
    truth_class2_blend_ = (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) != 0;
  }
  if (!EnsureWbSetLayout()) return false;  // (Session 92, M2d: the set layout moved to owned_wb_objects.cpp, made by the owned write-back too)
  VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 10 * sizeof(uint32_t)};
  VkPipelineLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &truth_store_set_layout_;
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &push;
  if (dfn.vkCreatePipelineLayout(device, &layout_info, nullptr, &truth_store_layout_) != VK_SUCCESS) return false;
  for (int depth = 0; depth < 2; ++depth) {
    std::vector<uint32_t> spirv;
    std::string error;
    const std::string source =
        std::string("#version 450\n") + (depth ? "#define DEPTH 1\n" : "") + kTruthStoreComputeShader;
    if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT, source, spirv, error)) {
      REXGPU_ERROR("[native][truth] store shader compilation failed: {}", error);
      return false;
    }
    truth_store_shaders_[depth] =
        ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
    if (truth_store_shaders_[depth] == VK_NULL_HANDLE) return false;
    VkComputePipelineCreateInfo info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                  truth_store_shaders_[depth], "main", nullptr};
    info.layout = truth_store_layout_;
    if (dfn.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &truth_store_pipelines_[depth]) !=
        VK_SUCCESS) {
      truth_store_pipelines_[depth] = VK_NULL_HANDLE;
      return false;
    }
  }
  // Session 47 (M6): the texture fill's pack shader (the store's layout). Without it the fills that need it are
  // refused (truth frames go on).
  {
    std::vector<uint32_t> spirv;
    std::string error;
    if (command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT,
                                              std::string("#version 450\n") + kTruthFillComputeShader, spirv, error)) {
      truth_fill_shader_ =
          ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
    } else {
      REXGPU_ERROR("[native][truth] texture fill shader compilation failed: {}", error);
    }
    if (truth_fill_shader_ != VK_NULL_HANDLE) {
      VkComputePipelineCreateInfo info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                    truth_fill_shader_, "main", nullptr};
      info.layout = truth_store_layout_;
      if (dfn.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &truth_fill_pipeline_) !=
          VK_SUCCESS) {
        truth_fill_pipeline_ = VK_NULL_HANDLE;
      }
    }
  }
  // (Per image: two store sets and a direct resolve set - one storage buffer and three samplers; Session 46: the
  // store sets' stencil binding, two more.)
  // (Session 54: with draw twins, a colour twin's resample set too - one sampler.)
  // (Two a twin - into it and out of it -, a depth twin's with two samplers.)
  // (Session 73: always - the draw height may go below the resolution scale's later, dynamic resolution.)
  const uint32_t twin_sets = 2 * kTruthMaxImages;
  VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * kTruthMaxImages + 64},
                                   {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8 * kTruthMaxImages + 2 * twin_sets + 64}};
  VkDescriptorPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets = 4 * kTruthMaxImages + twin_sets + 64;
  pool_info.poolSizeCount = 2;
  pool_info.pPoolSizes = sizes;
  if (dfn.vkCreateDescriptorPool(device, &pool_info, nullptr, &truth_pool_) != VK_SUCCESS) return false;
  truth_failed_ = false;
  REXGPU_INFO("[native][truth] truth objects ready (16-bit images blend: {})", truth_class2_blend_);
  return true;
}

void NativeRenderer::RetireTruthImage(TruthImage& t) {
  TruthFlushDeferred(kDeferFlushStores);  // Session 52
  TruthRetireServeViews(t.image);  // Session 80 (gpu_native_truth_serve_copies)
  TruthRetireTwin(t);    // Session 54
  // Framebuffers naming it.
  const int16_t index = int16_t(&t - truth_images_.data());
  for (auto it = truth_framebuffers_.begin(); it != truth_framebuffers_.end();) {
    // (Review, Session 52: and any entry with its views - a pooled image may have been another slot's.)
    if (it->color == index || it->depth == index ||
        (t.view != VK_NULL_HANDLE && (it->color_view == t.view || it->depth_view == t.view))) {
      Retired r;
      r.submission = command_processor_.GetCurrentSubmission();
      r.framebuffer = it->framebuffer;
      retired_.push_back(r);
      it = truth_framebuffers_.erase(it);
    } else {
      ++it;
    }
  }
  Retired r;
  r.submission = command_processor_.GetCurrentSubmission();
  r.images[0] = t.image;
  r.memory[0] = t.memory;
  r.views[0] = t.view;
  r.views[1] = t.sample_view;
  r.views[2] = t.stencil_view;
  if (t.image || t.view || t.sample_view || t.stencil_view) retired_.push_back(r);
  for (VkDescriptorSet& set : t.store_sets) {
    if (set != VK_NULL_HANDLE) truth_retired_sets_.push_back({command_processor_.GetCurrentSubmission(), set});
    set = VK_NULL_HANDLE;
  }
  if (t.resolve_set != VK_NULL_HANDLE) {
    truth_retired_sets_.push_back({command_processor_.GetCurrentSubmission(), t.resolve_set});
    t.resolve_set = VK_NULL_HANDLE;
  }
  t.image = VK_NULL_HANDLE;
  t.memory = VK_NULL_HANDLE;
  t.view = t.sample_view = t.stencil_view = VK_NULL_HANDLE;
  t.store_set_edram = VK_NULL_HANDLE;
  t.width = t.height = 0;
  t.initialized = false;
}

void NativeRenderer::RetireTruthAll() {
  serve_rename_.valid = false;  // Session 80 (gpu_native_truth_serve_copies: its slot index goes)
  serve_pending_.clear();
  serve_ahead_ = {};  // (Session 80, gpu_native_truth_serve_ahead)
  TruthFlushDeferred(kDeferFlushStores);  // Session 52
  EndTruthPass();
  owned_dirty_.clear();  // Session 80 (owned passes: the images' pixels go - the EDRAM owns every tile)
  ++truth_claim_epoch_;  // Session 47
  for (TruthImage& t : truth_images_) RetireTruthImage(t);
  for (TruthBundle& b : truth_spares_) TruthRetireBundle(b);  // Session 52
  truth_spares_.clear();
  truth_images_.clear();
  truth_tile_color_ = truth_tile_depth_ = -1;  // (image indices start again)
  for (RestoreSource& s : restore_sources_) s.frame = UINT64_MAX;  // Session 72 part 4 (their indices are gone)
  TruthClearOwners();
}

// Session 47 (CPU): the owner map and the images' owned tiles, changed together.
void NativeRenderer::TruthSetOwners(const std::bitset<2048>& color_tiles, int16_t color,
                                    const std::bitset<2048>& depth_tiles, int16_t depth) {
  const std::bitset<2048> changed = color_tiles | depth_tiles;
  if ((truth_owned_any_ & changed).any()) {  // (Session 47: only if some image owned them)
    for (TruthImage& t : truth_images_) t.owned &= ~changed;
  }
  if (color >= 0 && color_tiles.any()) truth_images_[color].owned |= color_tiles & ~depth_tiles;
  if (depth >= 0 && depth_tiles.any()) truth_images_[depth].owned |= depth_tiles;
  truth_owned_any_ |= changed;
}

void NativeRenderer::TruthClearOwners() {
  for (TruthImage& t : truth_images_) t.owned.reset();
  truth_owned_any_.reset();
}

int16_t NativeRenderer::TruthOwnerOf(uint32_t tile) const {
  if (!truth_owned_any_.test(tile)) return -1;
  for (size_t i = 0; i < truth_images_.size(); ++i) {
    if (truth_images_[i].owned.test(tile)) return int16_t(i);
  }
  return -1;
}

// The image at least `height` host pixels tall (made, or grown with its pixels copied), cleared when new, in its
// attachment layout. Outside any render pass.
bool NativeRenderer::EnsureTruthImage(int16_t index, uint32_t height) {
  if (!truth_renaming_) TruthFlushDeferred(kDeferFlushStores);  // Session 52 (a rename makes a fresh image without it)
  TruthImage& t = truth_images_[index];
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const uint32_t width = t.pitch_tiles * 80 * scale_x;
  // At most the rows of tiles a surface of this pitch has in the 2048-tile EDRAM.
  const uint32_t max_height = (2048u + t.row_tiles() - 1) / t.row_tiles() * 16 * scale_y;
  height = std::min(std::max(height, 16u * scale_y), max_height);
  if (t.image != VK_NULL_HANDLE && t.width == width && t.height >= height) return true;
  TruthSyncRead(index);  // Session 54 (the grow copies the image)
  // Grow in steps (a few more rows than asked). (Session 80, gpu_native_truth_exact_rows: by rows of tiles - a copy of a
  // whole surface is then all of its image.)
  const uint32_t step = (REXCVAR_GET(gpu_native_truth_exact_rows) ? 16 : 64) * scale_y;
  height = std::min(max_height, (height + step - 1) / step * step);
  if (t.image != VK_NULL_HANDLE && t.width == width) height = std::max(height, t.height);
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  const bool depth = t.kind == 3;
  const VkFormat format = depth ? post_ds_format_ : kTruthColorFormats[t.kind];
  VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = format;
  info.extent = {width, height, 1};
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = (depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) |
               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  if (!ui::vulkan::util::CreateDedicatedAllocationImage(vulkan_device, info,
                                                        ui::vulkan::util::MemoryPurpose::kDeviceLocal, image, memory)) {
    REXGPU_ERROR("[native][truth] could not create a {}x{} image", width, height);
    return false;
  }
  VkImageView view = VK_NULL_HANDLE, sample_view = VK_NULL_HANDLE, stencil_view = VK_NULL_HANDLE;
  VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view_info.image = image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = format;
  view_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(
      depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
  bool ok = dfn.vkCreateImageView(device, &view_info, nullptr, &view) == VK_SUCCESS;
  view_info.subresourceRange =
      ui::vulkan::util::InitializeSubresourceRange(depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
  ok = ok && dfn.vkCreateImageView(device, &view_info, nullptr, &sample_view) == VK_SUCCESS;
  if (depth) {  // Session 46: the stencil the stores read
    view_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_STENCIL_BIT);
    ok = ok && dfn.vkCreateImageView(device, &view_info, nullptr, &stencil_view) == VK_SUCCESS;
  }
  if (!ok) {
    Retired r;
    r.submission = command_processor_.GetCurrentSubmission();
    r.images[0] = image;
    r.memory[0] = memory;
    r.views[0] = view;
    r.views[1] = sample_view;
    r.views[2] = stencil_view;
    retired_.push_back(r);
    return false;
  }
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const VkImageAspectFlags aspects = depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                           : VK_IMAGE_ASPECT_COLOR_BIT;
  const VkImageLayout attachment_layout =
      depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  const VkPipelineStageFlags attachment_stages =
      depth ? VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
            : VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  const VkAccessFlags attachment_access =
      depth ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
            : VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  VkImageMemoryBarrier barriers[2] = {};
  for (VkImageMemoryBarrier& b : barriers) {
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(aspects);
  }
  // The new image: cleared (then the old one's pixels copied over it).
  barriers[0].image = image;
  barriers[0].srcAccessMask = 0;
  barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  uint32_t barrier_count = 1;
  const bool grow = t.image != VK_NULL_HANDLE && t.initialized && t.width == width;
  if (!grow) t.stencil_valid.reset();  // (a grown image keeps its stencil: the copy has both aspects)
  if (grow) {
    barriers[1].image = t.image;
    barriers[1].srcAccessMask = attachment_access | VK_ACCESS_SHADER_READ_BIT;
    barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barriers[1].oldLayout = attachment_layout;
    barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier_count = 2;
  }
  cb.CmdVkPipelineBarrier(attachment_stages | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, barrier_count, barriers);
  const VkImageSubresourceRange range = ui::vulkan::util::InitializeSubresourceRange(aspects);
  // (A new depth image is not cleared on the interlock path: its tiles are loaded from the EDRAM or drawn before they
  // are read. Session 54 review: without an EDRAM - the host render target path - it is, to 0 as a zeroed EDRAM.)
  if (!depth) {
    const VkClearColorValue clear = {};
    cb.CmdVkClearColorImage(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
  } else if (command_processor_.NativeEdramBuffer() == VK_NULL_HANDLE) {
    const VkClearDepthStencilValue clear = {};
    cb.CmdVkClearDepthStencilImage(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, range);
  }
  if (grow) {
    VkMemoryBarrier clear_done = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                                  VK_ACCESS_TRANSFER_WRITE_BIT};
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &clear_done, 0,
                            nullptr, 0, nullptr);
    VkImageCopy region = {};
    region.srcSubresource = {aspects, 0, 0, 1};
    region.dstSubresource = {aspects, 0, 0, 1};
    region.extent = {t.width, std::min(t.height, height), 1};
    cb.CmdVkCopyImage(t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                      &region);
  }
  barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barriers[0].dstAccessMask = attachment_access;
  barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barriers[0].newLayout = attachment_layout;
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, attachment_stages, 0, 0, nullptr, 0, nullptr, 1, barriers);
  if (t.image != VK_NULL_HANDLE) RetireTruthImage(t);
  t.image = image;
  t.memory = memory;
  t.view = view;
  t.sample_view = sample_view;
  t.stencil_view = stencil_view;
  t.width = width;
  t.height = height;
  t.initialized = true;
  REXGPU_INFO("[native][truth] image {}x{} ({}, base {} pitch {} tiles, format {}){}", width, height,
              depth ? "depth" : t.kind == 1 ? "RGBA8" : "RGBA16_SNORM",
              t.base == kTruthScratchBase ? int32_t(-1) : int32_t(t.base), t.pitch_tiles, t.format,
              grow ? " (grown)" : "");
  TruthStorageNote(t);  // Session 73
  rex::perf::frame_events::Get().images.fetch_add(1, std::memory_order_relaxed);  // Session 73
  return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Session 73: the truth images made ahead (gpu_native_truth_prewarm). Lines "base pitch kind format rows" (rows = the
// image's height in guest pixels, without the resolution scale); the largest rows per key wins.
// ---------------------------------------------------------------------------------------------------------------

void NativeRenderer::TruthStorageOpen(const std::filesystem::path& shareable_root, uint32_t title_id) {
  truth_prewarm_.clear();
  truth_stored_.clear();
  truth_prewarmed_ = 0;
  truth_prewarm_logged_ = false;
  if (!REXCVAR_GET(gpu_native_truth_prewarm)) return;
  const std::string forced = REXCVAR_GET(gpu_native_truth_prewarm_file);
  const std::filesystem::path path = forced.empty()
                                         ? shareable_root / fmt::format("{:08X}.native.truth.txt", title_id)
                                         : std::filesystem::u8path(forced);
  truth_storage_file_ = rex::filesystem::OpenFile(path, "a+b");
  if (!truth_storage_file_) {
    REXGPU_WARN("[native][truth] cannot open {}: truth images not made ahead", rex::path_to_utf8(path));
    return;
  }
  rex::filesystem::Seek(truth_storage_file_, 0, SEEK_SET);
  truth_spare_counts_.clear();
  truth_spare_prewarm_.clear();
  truth_spares_prewarmed_ = 0;
  char line[128];
  while (std::fgets(line, sizeof(line), truth_storage_file_)) {
    unsigned base, pitch, kind, format, rows, sw, sh, count;
    if (std::sscanf(line, "S %u %u %u %u %u", &kind, &format, &sw, &sh, &count) == 5) {  // Session 73: spares
      if (kind < 1 || kind > 3 || !sw || !sh || sw > 16384 || sh > 16384 || !count) continue;
      bool merged = false;
      for (TruthStoredSpares& c : truth_spare_counts_) {
        if (c.kind == kind && c.format == format && c.width == sw && c.height == sh) {
          c.count = std::max(c.count, uint32_t(count));
          merged = true;
        }
      }
      if (!merged) truth_spare_counts_.push_back({kind, format, sw, sh, count});
      continue;
    }
    // (Session 80, gpu_native_truth_exact_rows: its own 'R' lines - the 64-row steps' heights would make the images
    // taller than their surfaces again; an older build skips the 'R' lines.)
    if (REXCVAR_GET(gpu_native_truth_exact_rows)
            ? std::sscanf(line, "R %u %u %u %u %u", &base, &pitch, &kind, &format, &rows) != 5
            : std::sscanf(line, "%u %u %u %u %u", &base, &pitch, &kind, &format, &rows) != 5) {
      continue;
    }
    if (kind < 1 || kind > 3 || pitch == 0 || pitch > 2048 || rows == 0 || rows > 16384) continue;
    bool merged = false;
    for (TruthStored& s : truth_stored_) {
      if (s.base == base && s.pitch_tiles == pitch && s.kind == kind && s.format == format) {
        s.rows = std::max(s.rows, uint32_t(rows));
        merged = true;
      }
    }
    if (!merged) truth_stored_.push_back({base, pitch, kind, format, rows});
  }
  rex::filesystem::Seek(truth_storage_file_, 0, SEEK_END);
  // Spares only of small images (the bursts; a big spare costs real memory and comes back within the run anyway).
  for (const TruthStoredSpares& c : truth_spare_counts_) {
    if (uint64_t(c.width) * c.height <= 256 * 256) {
      truth_spare_prewarm_.push_back({c.kind, c.format, c.width, c.height, std::min(c.count, 32u)});
    }
  }
  // Made from the back of the list, so the first ones stored (the boot's, the menus') come first.
  truth_prewarm_.assign(truth_stored_.rbegin(), truth_stored_.rend());
  if (truth_prewarm_.size() > kTruthMaxImages / 2) {
    truth_prewarm_.erase(truth_prewarm_.begin(), truth_prewarm_.end() - kTruthMaxImages / 2);
  }
  REXGPU_INFO("[native][truth] {} stored truth images ({})", truth_stored_.size(), rex::path_to_utf8(path));
}

void NativeRenderer::TruthStorageNote(const TruthImage& image) {
  if (!truth_storage_file_) return;
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const uint32_t rows = (image.height + scale_y - 1) / scale_y;
  const auto write = [&]() {  // (Session 80, gpu_native_truth_exact_rows: as 'R' lines)
    std::fprintf(truth_storage_file_, "%s%u %u %u %u %u\n", REXCVAR_GET(gpu_native_truth_exact_rows) ? "R " : "",
                 image.base, image.pitch_tiles, image.kind, image.format, rows);
  };
  for (TruthStored& s : truth_stored_) {
    if (s.base == image.base && s.pitch_tiles == image.pitch_tiles && s.kind == image.kind &&
        s.format == image.format) {
      if (rows <= s.rows) return;
      s.rows = rows;
      write();
      std::fflush(truth_storage_file_);
      return;
    }
  }
  truth_stored_.push_back({image.base, image.pitch_tiles, image.kind, image.format, rows});
  write();
  std::fflush(truth_storage_file_);
}

// From TruthOnFrameEnd (outside any pass, before a truth frame): a few stored images made.
void NativeRenderer::TruthPrewarmImages() {
  if (REXCVAR_GET(gpu_native_owned_only)) return;  // Session 92 (M2a): no truth image is made ahead (d3d_owned.h also sets gpu_native_truth_prewarm=false)
  if (truth_prewarm_.empty() && truth_spare_prewarm_.empty()) return;
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  int32_t budget = REXCVAR_GET(gpu_native_truth_prewarm_per_frame);
  while (budget-- > 0 && !truth_prewarm_.empty()) {
    const TruthStored s = truth_prewarm_.back();
    truth_prewarm_.pop_back();
    const int16_t index = TruthFindImage(s.base, s.pitch_tiles, s.kind, s.format);
    if (index < 0) {  // the table is full
      truth_prewarm_.clear();
      break;
    }
    const bool had = truth_images_[index].image != VK_NULL_HANDLE;
    if (EnsureTruthImage(index, s.rows * scale_y) && !had) ++truth_prewarmed_;
  }
  if (truth_prewarm_.empty() && !truth_prewarm_logged_) {
    truth_prewarm_logged_ = true;
    REXGPU_INFO("[native][truth] {} stored truth images made ahead by frame {}", truth_prewarmed_, frame_);
  }
  if (truth_prewarm_.empty() && budget > 0 && !truth_spare_prewarm_.empty()) TruthPrewarmSpares();
}

// Session 73: this run's pool, per size, when it grows past the install's mark - a line appended.
void NativeRenderer::TruthNoteSpares() {
  if (!truth_storage_file_) return;
  const TruthBundle& b = truth_spares_.back();
  uint32_t count = 0;
  for (const TruthBundle& s : truth_spares_) {
    count += (s.kind == b.kind && s.format == b.format && s.width == b.width && s.height == b.height) ? 1 : 0;
  }
  for (TruthStoredSpares& c : truth_spare_counts_) {
    if (c.kind == b.kind && c.format == b.format && c.width == b.width && c.height == b.height) {
      if (count <= c.count) return;
      c.count = count;
      std::fprintf(truth_storage_file_, "S %u %u %u %u %u\n", b.kind, b.format, b.width, b.height, count);
      std::fflush(truth_storage_file_);
      return;
    }
  }
  truth_spare_counts_.push_back({b.kind, b.format, b.width, b.height, count});
  std::fprintf(truth_storage_file_, "S %u %u %u %u %u\n", b.kind, b.format, b.width, b.height, count);
  std::fflush(truth_storage_file_);
}

// Session 73: the stored spare counts made, a few per frame end, as the rename path makes them: the image of that
// size gets a new one (EnsureTruthImage with its handles cleared), which goes into the pool, and it gets its own back.
bool NativeRenderer::TruthPrewarmSpares() {
  int32_t budget = REXCVAR_GET(gpu_native_truth_prewarm_per_frame) * 2;
  while (budget > 0 && !truth_spare_prewarm_.empty()) {
    TruthStoredSpares& want = truth_spare_prewarm_.back();
    int16_t index = -1;
    for (size_t i = 0; i < truth_images_.size(); ++i) {
      const TruthImage& t = truth_images_[i];
      if (t.image != VK_NULL_HANDLE && t.kind == want.kind && t.format == want.format && t.width == want.width &&
          t.height == want.height && t.rename_source == VK_NULL_HANDLE) {
        index = int16_t(i);
        break;
      }
    }
    uint32_t have = 0;
    for (const TruthBundle& s : truth_spares_) {
      have += (s.kind == want.kind && s.format == want.format && s.width == want.width && s.height == want.height);
    }
    if (index < 0 || have >= want.count || truth_spares_.size() >= 64) {
      truth_spare_prewarm_.pop_back();  // (no image of that size this run, or enough already)
      continue;
    }
    TruthImage& t = truth_images_[index];
    TruthImage keep = t;
    t.image = VK_NULL_HANDLE;
    t.memory = VK_NULL_HANDLE;
    t.view = t.sample_view = t.stencil_view = VK_NULL_HANDLE;
    t.store_sets[0] = t.store_sets[1] = VK_NULL_HANDLE;
    t.resolve_set = VK_NULL_HANDLE;
    t.store_set_edram = VK_NULL_HANDLE;
    t.initialized = false;
    truth_renaming_ = true;
    const bool ok = EnsureTruthImage(index, keep.height);
    truth_renaming_ = false;
    TruthBundle made;
    made.image = t.image;
    made.memory = t.memory;
    made.view = t.view;
    made.sample_view = t.sample_view;
    made.stencil_view = t.stencil_view;
    made.store_sets[0] = t.store_sets[0];
    made.store_sets[1] = t.store_sets[1];
    made.resolve_set = t.resolve_set;
    made.store_set_edram = t.store_set_edram;
    made.kind = t.kind;
    made.format = t.format;
    made.width = t.width;
    made.height = t.height;
    made.busy = false;
    made.idle_since = std::chrono::steady_clock::now();
    const bool fits = ok && made.image != VK_NULL_HANDLE && made.width == keep.width && made.height == keep.height;
    t = keep;
    if (!fits) {
      if (made.image != VK_NULL_HANDLE) TruthRetireBundle(made);
      truth_spare_prewarm_.pop_back();
      continue;
    }
    truth_spares_.push_back(made);
    ++truth_spares_prewarmed_;
    --budget;
  }
  if (truth_spare_prewarm_.empty()) {
    REXGPU_INFO("[native][truth] {} rename spares made ahead by frame {}", truth_spares_prewarmed_, frame_);
  }
  return true;
}

VkFramebuffer NativeRenderer::TruthFramebuffer(int16_t color, int16_t depth, uint32_t& width, uint32_t& height) {
  const TruthImage& c = truth_images_[color];
  const TruthImage& d = truth_images_[depth];
  width = std::min(c.width, d.width);
  height = std::min(c.height, d.height);
  // Session 54: with draw twins, the twins (made by EnsureTruthTwin before this; colour and depth share the pitch).
  const bool twins = TruthTwinPitch(c.pitch_tiles);
  const VkImageView c_view = twins ? c.draw_view : c.view, d_view = twins ? d.draw_view : d.view;
  if (c_view == VK_NULL_HANDLE || d_view == VK_NULL_HANDLE) return VK_NULL_HANDLE;
  for (const TruthFramebufferEntry& f : truth_framebuffers_) {
    if (f.color == color && f.depth == depth && f.color_view == c_view && f.depth_view == d_view) return f.framebuffer;
  }
  if (DecoupleViews()) {  // (Session 94 review: the truth framebuffer has 1 layer, a multiview pass needs as many as its views - refused while gpu_d3d_decouple is on)
    static bool logged = false;
    if (!logged) {
      logged = true;
      REXGPU_ERROR("[native] decouple: a truth framebuffer was asked for - not supported with gpu_d3d_decouple (1 layer, multiview pass); the draw is not served");
    }
    return VK_NULL_HANDLE;
  }
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const VkImageView views[2] = {c_view, d_view};
  VkFramebufferCreateInfo fb_info = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  // wwe2k14 patch (Session 73): the pass the truth passes begin (no external dependencies): a framebuffer is compatible
  // only with render passes compatible with its own (validation VUID-VkRenderPassBeginInfo-renderPass-00904).
  fb_info.renderPass = truth_render_pass_keep_nodep_[c.kind] != VK_NULL_HANDLE ? truth_render_pass_keep_nodep_[c.kind]
                                                                              : post_render_pass_begin_keep_[c.kind];
  fb_info.attachmentCount = 2;
  fb_info.pAttachments = views;
  fb_info.width = twins ? std::min(c.draw_width, d.draw_width) : width;
  fb_info.height = twins ? std::min(c.draw_height, d.draw_height) : height;
  fb_info.layers = 1;
  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  if (vulkan_device->functions().vkCreateFramebuffer(vulkan_device->device(), &fb_info, nullptr, &framebuffer) !=
      VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  truth_framebuffers_.push_back({color, depth, framebuffer, c_view, d_view});
  return framebuffer;
}

// Session 54: below 1 only for a height under the resolution scale's, with truth frames on and every truth image format
// able to be blitted (the colour ones with linear filtering).
float NativeRenderer::TruthDrawRatio() {
  if (truth_draw_ratio_ >= 0.0f) return truth_draw_ratio_;
  truth_draw_ratio_ = 1.0f;
  const int32_t height = REXCVAR_GET(gpu_native_truth_draw_height);
  if (height <= 0) return truth_draw_ratio_;
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  // Session 95 (run 6561, a black game at 540P): under TRUE NATIVE there are no truth images and so no draw twins - a ratio below 1 here made the owned claim refuse every
  // full-size pass ("draw twin pitch") and the latch dropped them all. The draw height is the owned frame's scene-pass ratio instead (owned_steps.cpp). The cvar, not
  // OwnedOnly(): this first runs before the first frame boundary (the first typed draw).
  if (REXCVAR_GET(gpu_native_owned_only)) {
    REXGPU_INFO("[native][owned-only] draw height {}: no draw twins under TRUE NATIVE - the owned scene pass draws at {:.4f} of the resolution scale's size", height,
                OwnedRowRatio(scale_x, scale_y));
    return truth_draw_ratio_;
  }
  // (Not where truth frames are refused: EDRAM renaming, DLSS, DLAA - NativeRenderer::OnFrameEnd.)
  if (scale_x != scale_y || uint32_t(height) >= 720u * scale_y || REXCVAR_GET(gpu_native) != 2 ||
      !REXCVAR_GET(gpu_native_truth) || REXCVAR_GET(gpu_edram_rename) || REXCVAR_GET(gpu_dlss) ||
      REXCVAR_GET(gpu_dlaa)) {
    REXGPU_INFO("[native][truth] draw height {}: not used (resolution scale {}x{}, gpu_native {}, truth frames {}) - "
                "the passes draw at the resolution scale",
                height, scale_x, scale_y, REXCVAR_GET(gpu_native), REXCVAR_GET(gpu_native_truth));
    return truth_draw_ratio_;
  }
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanInstance::Functions& ifn = vulkan_device->vulkan_instance()->functions();
  const auto has = [&](VkFormat format, VkFormatFeatureFlags needed) {
    VkFormatProperties properties;
    ifn.vkGetPhysicalDeviceFormatProperties(vulkan_device->physical_device(), format, &properties);
    return (properties.optimalTilingFeatures & needed) == needed;
  };
  // (Review of an AMD tester's report - "it looks 720p at 360p / 480p / 540p": AMD's Windows driver cannot blit
  // D32S8, gpuinfo.org; this used to give up here. Colour twins go through the resample pass, sampled linearly;
  // depth through a blit where it can, else the depth pass.)
  constexpr VkFormatFeatureFlags kBlit = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
  constexpr VkFormatFeatureFlags kSampledLinear =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
  truth_depth_blit_ =
      has(VK_FORMAT_D32_SFLOAT_S8_UINT, kBlit) && REXCVAR_GET(gpu_native_truth_draw_depth_blit);
  // Session 73 (Jarrod's phone: "a lot of white static noise" while dynamic resolution had stepped down): the stencil-
  // exporting depth pass has only ever run on phones (no PC driver here has the extension) and was never checked;
  // the stencil-kept one is pixel-identical to the blit on the PC (replay runs 4757 / 4759). Opt-in now.
  truth_depth_stencil_export_ =
      vulkan_device->extensions().ext_EXT_shader_stencil_export && REXCVAR_GET(gpu_native_truth_draw_stencil_export);
  if (!has(kTruthColorFormats[1], kSampledLinear) || !has(kTruthColorFormats[2], kSampledLinear) ||
      !EnsureTruthResampleObjects() || (!truth_depth_blit_ && !EnsureTruthDepthResampleObjects())) {
    REXGPU_WARN("[native][truth] draw height {}: the resample passes could not be made on this device - the passes "
                "draw at the resolution scale ({} rows)",
                height, 720u * scale_y);
    return truth_draw_ratio_;
  }
  truth_draw_ratio_ = float(height) / float(720u * scale_y);
  {  // (the ratio as p / q in lowest terms: blits over whole multiples of q image pixels are exact)
    uint32_t a = uint32_t(height), b = 720u * scale_y;
    while (b) std::swap(a %= b, b);
    truth_ratio_p_ = uint32_t(height) / a;
    truth_ratio_q_ = 720u * scale_y / a;
  }
  truth_twin_min_pitch_ = uint32_t(std::max(1, REXCVAR_GET(gpu_native_truth_draw_min_pitch)));
  REXGPU_INFO("[native][truth] draw height {}: the truth passes draw at {:.4f} of the resolution scale's size "
              "(resolution scale {}, {}x{} for the 1280x720 screen); reads at the resolution scale; depth twins by {}",
              height, truth_draw_ratio_, scale_y, uint32_t(std::ceil(1280.0f * scale_x * truth_draw_ratio_)),
              uint32_t(std::ceil(720.0f * scale_y * truth_draw_ratio_)),
              truth_depth_blit_ ? "blit" : truth_depth_stencil_export_ ? "the depth pass (stencil exported)"
                                                                    : "the depth pass (no stencil export: stencil kept)");
  return truth_draw_ratio_;
}

void NativeRenderer::TruthRetireTwin(TruthImage& t) {
  if (t.draw_view != VK_NULL_HANDLE) {
    for (auto it = truth_framebuffers_.begin(); it != truth_framebuffers_.end();) {
      if (it->color_view == t.draw_view || it->depth_view == t.draw_view) {
        Retired r;
        r.submission = command_processor_.GetCurrentSubmission();
        r.framebuffer = it->framebuffer;
        retired_.push_back(r);
        it = truth_framebuffers_.erase(it);
      } else {
        ++it;
      }
    }
  }
  if (t.draw_image != VK_NULL_HANDLE || t.draw_view != VK_NULL_HANDLE || t.sync_fbs[0] != VK_NULL_HANDLE ||
      t.sync_fbs[1] != VK_NULL_HANDLE) {
    Retired r;
    r.submission = command_processor_.GetCurrentSubmission();
    r.images[0] = t.draw_image;
    r.memory[0] = t.draw_memory;
    r.views[0] = t.draw_view;
    r.views[1] = t.draw_sample_view;
    r.views[2] = t.draw_stencil_view;
    r.framebuffer = t.sync_fbs[0];  // (the image's view in [1]: retired with the twin, made again with it)
    retired_.push_back(r);
    if (t.sync_fbs[1] != VK_NULL_HANDLE) {
      Retired f;
      f.submission = r.submission;
      f.framebuffer = t.sync_fbs[1];
      retired_.push_back(f);
    }
  }
  for (VkDescriptorSet& set : t.sync_sets) {
    if (set != VK_NULL_HANDLE) truth_retired_sets_.push_back({command_processor_.GetCurrentSubmission(), set});
    set = VK_NULL_HANDLE;
  }
  t.sync_fbs[0] = t.sync_fbs[1] = VK_NULL_HANDLE;
  t.draw_sample_view = t.draw_stencil_view = VK_NULL_HANDLE;
  t.draw_image = VK_NULL_HANDLE;
  t.draw_memory = VK_NULL_HANDLE;
  t.draw_view = VK_NULL_HANDLE;
  t.draw_for = VK_NULL_HANDLE;
  t.draw_width = t.draw_height = 0;
  t.draw_dirty = false;
}

// The twin scaled into the image (up) or the image into the twin; both in their attachment layout before and after.
void NativeRenderer::TruthBlitTwin(TruthImage& t, bool up) {
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const bool depth = t.kind == 3;
  const VkImageAspectFlags aspects = depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                           : VK_IMAGE_ASPECT_COLOR_BIT;
  const VkImageLayout attachment_layout =
      depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  const VkPipelineStageFlags attachment_stages =
      depth ? VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
            : VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  const VkAccessFlags attachment_access =
      depth ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
            : VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  const VkImage src = up ? t.draw_image : t.image, dst = up ? t.image : t.draw_image;
  VkImageMemoryBarrier barriers[2] = {};
  for (VkImageMemoryBarrier& b : barriers) {
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(aspects);
    b.oldLayout = attachment_layout;
  }
  // (The reads of the image - stores, resolves, fills - leave it in its attachment layout after their own barriers.)
  barriers[0].image = src;
  barriers[0].srcAccessMask = attachment_access | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barriers[1].image = dst;
  barriers[1].srcAccessMask = attachment_access | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  cb.CmdVkPipelineBarrier(attachment_stages | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
  // (Review: exact rectangles - the image's whole multiples of q pixels map to p twin pixels each; the image's last
  // rows / columns past them take the twin's last whole pixel going up, and the image's rest going down. Never the
  // twin's partly drawn last row / column.)
  const int32_t p = int32_t(truth_ratio_p_), q = int32_t(std::max(1u, truth_ratio_q_));
  const int32_t iw = int32_t(t.width), ih = int32_t(t.height);
  const int32_t dw = int32_t(t.draw_width), dh = int32_t(t.draw_height);
  const int32_t xm = iw / q * q, ym = ih / q * q;  // image pixels
  const int32_t xs = xm / q * p, ys = ym / q * p;  // twin pixels
  VkImageBlit regions[4] = {};
  uint32_t region_count = 0;
  const auto add = [&](int32_t ix0, int32_t iy0, int32_t ix1, int32_t iy1, int32_t tx0, int32_t ty0, int32_t tx1,
                       int32_t ty1) {
    if (ix1 <= ix0 || iy1 <= iy0 || tx1 <= tx0 || ty1 <= ty0) return;
    VkImageBlit& r = regions[region_count++];
    r.srcSubresource = {aspects, 0, 0, 1};
    r.dstSubresource = {aspects, 0, 0, 1};
    const VkOffset3D i0 = {ix0, iy0, 0}, i1 = {ix1, iy1, 1}, t0 = {tx0, ty0, 0}, t1 = {tx1, ty1, 1};
    r.srcOffsets[0] = up ? t0 : i0;
    r.srcOffsets[1] = up ? t1 : i1;
    r.dstOffsets[0] = up ? i0 : t0;
    r.dstOffsets[1] = up ? i1 : t1;
  };
  if (xs > 0 && ys > 0) {
    add(0, 0, xm, ym, 0, 0, xs, ys);
    if (up) {
      add(xm, 0, iw, ym, xs - 1, 0, xs, ys);
      add(0, ym, xm, ih, 0, ys - 1, xs, ys);
      add(xm, ym, iw, ih, xs - 1, ys - 1, xs, ys);
    } else {
      add(xm, 0, iw, ym, xs, 0, dw, ys);
      add(0, ym, xm, ih, 0, ys, xs, dh);
      add(xm, ym, iw, ih, xs, ys, dw, dh);
    }
  } else {  // (an image smaller than q pixels: whole to whole)
    add(0, 0, iw, ih, 0, 0, dw, dh);
  }
  cb.CmdVkBlitImage(src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region_count,
                    regions, depth ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
  barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  barriers[0].dstAccessMask = attachment_access | VK_ACCESS_SHADER_READ_BIT;
  barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barriers[0].newLayout = attachment_layout;
  barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barriers[1].dstAccessMask = attachment_access | VK_ACCESS_SHADER_READ_BIT;
  barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barriers[1].newLayout = attachment_layout;
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                          attachment_stages | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                          0, 0, nullptr, 0, nullptr, 2, barriers);
  ++truth_counts_.twin_blits[up ? 1 : 0];
}

bool NativeRenderer::EnsureTruthTwin(int16_t index) {
  TruthImage& t = truth_images_[index];
  if (t.draw_image != VK_NULL_HANDLE && t.draw_for == t.image) return true;
  if (t.image == VK_NULL_HANDLE) return false;
  EndTruthPass();  // (outside any render pass)
  TruthRetireTwin(t);
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  const bool depth = t.kind == 3;
  const VkFormat format = depth ? post_ds_format_ : kTruthColorFormats[t.kind];
  const uint32_t width = std::max(1u, uint32_t(std::ceil(float(t.width) * truth_draw_ratio_)));
  const uint32_t height = std::max(1u, uint32_t(std::ceil(float(t.height) * truth_draw_ratio_)));
  VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = format;
  info.extent = {width, height, 1};
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = (depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) |
               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  if (!ui::vulkan::util::CreateDedicatedAllocationImage(vulkan_device, info,
                                                        ui::vulkan::util::MemoryPurpose::kDeviceLocal, image, memory)) {
    REXGPU_ERROR("[native][truth] could not create a {}x{} draw image", width, height);
    return false;
  }
  VkImageView view = VK_NULL_HANDLE;
  VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view_info.image = image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = format;
  view_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(
      depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
  VkImageView sample_view = VK_NULL_HANDLE, stencil_view = VK_NULL_HANDLE;
  bool views_ok = dfn.vkCreateImageView(device, &view_info, nullptr, &view) == VK_SUCCESS;
  if (depth && views_ok) {  // (the depth pass samples the twin's two aspects)
    view_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT);
    views_ok = dfn.vkCreateImageView(device, &view_info, nullptr, &sample_view) == VK_SUCCESS;
    view_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_STENCIL_BIT);
    views_ok = views_ok && dfn.vkCreateImageView(device, &view_info, nullptr, &stencil_view) == VK_SUCCESS;
  }
  if (!views_ok) {
    Retired r;
    r.submission = command_processor_.GetCurrentSubmission();
    r.images[0] = image;
    r.memory[0] = memory;
    r.views[0] = view;
    r.views[1] = sample_view;
    r.views[2] = stencil_view;
    retired_.push_back(r);
    return false;
  }
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const VkImageAspectFlags aspects = depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                           : VK_IMAGE_ASPECT_COLOR_BIT;
  // The new twin in its attachment layout (TruthBlitTwin's starting point), then the image's pixels into it.
  VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout =
      depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(aspects);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                          1, &barrier);
  t.draw_image = image;
  t.draw_memory = memory;
  t.draw_view = view;
  t.draw_sample_view = sample_view;
  t.draw_stencil_view = stencil_view;
  t.draw_for = t.image;
  t.draw_width = width;
  t.draw_height = height;
  TruthCopyTwin(t, false);
  t.draw_dirty = false;
  return true;
}

void NativeRenderer::TruthSyncRead(int16_t index) {
  if (index < 0 || size_t(index) >= truth_images_.size()) return;
  TruthImage& t = truth_images_[index];
  if (!t.draw_dirty) return;
  t.draw_dirty = false;
  // (A twin of an older image: that image was brought up to date before it was replaced - EnsureTruthImage.)
  if (t.draw_image == VK_NULL_HANDLE || t.image == VK_NULL_HANDLE || t.draw_for != t.image) return;
  if (truth_pass_open_ && (truth_pass_color_ == index || truth_pass_depth_ == index)) EndTruthPass();
  if (segment_log_frames_ > 0) {
    REXGPU_INFO("[native][seg] frame {} twin sync: image {} {}x{} kind {} base {} pitch {}", frame_, index, t.width,
                t.height, t.kind, int32_t(t.base == kTruthScratchBase ? -1 : int32_t(t.base)), t.pitch_tiles);
  }
  TruthCopyTwin(t, true);
}

void NativeRenderer::TruthCopyTwin(TruthImage& t, bool up) {
  if (t.kind == 3 && truth_depth_blit_) {
    TruthBlitTwin(t, up);  // (exact, with the stencil, where the device blits D32S8)
    return;
  }
  if (!TruthResampleTwin(t, up)) {
    static uint32_t logged = 0;
    if (logged++ < 4) REXGPU_ERROR("[native][truth] a draw twin could not be copied (kind {}, {})", t.kind, up ? "up" : "down");
  }
}

namespace {
constexpr char kTruthResampleVertexShader[] = R"(
#version 450
void main() {
  vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";
// Each image pixel samples the twin where its centre falls (x the draw ratio), bilinear, within the centres of the
// twin's whole drawn pixels (max_coord): its partly drawn last column / row is never read.
constexpr char kTruthResampleFragmentShader[] = R"(
#version 450
layout(set = 0, binding = 0) uniform sampler2D twin;
layout(push_constant) uniform Constants { vec2 ratio; vec2 max_coord; } c;
layout(location = 0) out vec4 out_color;
void main() {
  vec2 p = min(gl_FragCoord.xy * c.ratio, c.max_coord);
  out_color = textureLod(twin, p / vec2(textureSize(twin, 0)), 0.0);
}
)";
}  // namespace

// The depth pass: each pixel takes the source's depth (and, with VK_EXT_shader_stencil_export, stencil) at the
// nearest source pixel (x the ratio), within max_coord (the source's whole pixels).
constexpr char kTruthDepthResampleShader[] = R"(
#version 450
layout(set = 0, binding = 0) uniform sampler2D src_depth;
layout(push_constant) uniform Constants { vec2 ratio; vec2 max_coord; } c;
void main() {
  ivec2 p = ivec2(min(floor(gl_FragCoord.xy * c.ratio), c.max_coord));
  gl_FragDepth = texelFetch(src_depth, p, 0).r;
}
)";
constexpr char kTruthDepthStencilResampleShader[] = R"(
#version 450
#extension GL_ARB_shader_stencil_export : require
layout(set = 0, binding = 0) uniform sampler2D src_depth;
layout(set = 0, binding = 1) uniform usampler2D src_stencil;
layout(push_constant) uniform Constants { vec2 ratio; vec2 max_coord; } c;
void main() {
  ivec2 p = ivec2(min(floor(gl_FragCoord.xy * c.ratio), c.max_coord));
  gl_FragDepth = texelFetch(src_depth, p, 0).r;
  gl_FragStencilRefARB = int(texelFetch(src_stencil, p, 0).r);
}
)";

bool NativeRenderer::EnsureTruthDepthResampleObjects() {
  if (truth_depth_resample_pipeline_ != VK_NULL_HANDLE) return true;
  if (truth_depth_resample_failed_ || !EnsureTruthResampleObjects()) return false;
  truth_depth_resample_failed_ = true;  // until everything below exists
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  {
    std::vector<uint32_t> spirv;
    std::string error;
    if (!command_processor_.CompileGlslToSpirv(
            VK_SHADER_STAGE_FRAGMENT_BIT,
            truth_depth_stencil_export_ ? kTruthDepthStencilResampleShader : kTruthDepthResampleShader, spirv, error)) {
      REXGPU_ERROR("[native][truth] depth resample shader compilation failed: {}", error);
      return false;
    }
    truth_depth_resample_shader_ =
        ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
    if (truth_depth_resample_shader_ == VK_NULL_HANDLE) return false;
  }
  VkDescriptorSetLayoutBinding bindings[2] = {
      {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
  VkDescriptorSetLayoutCreateInfo set_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  set_info.bindingCount = 2;
  set_info.pBindings = bindings;
  if (dfn.vkCreateDescriptorSetLayout(device, &set_info, nullptr, &truth_depth_set_layout_) != VK_SUCCESS) return false;
  VkPushConstantRange push = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4 * sizeof(float)};
  VkPipelineLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &truth_depth_set_layout_;
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &push;
  if (dfn.vkCreatePipelineLayout(device, &layout_info, nullptr, &truth_depth_resample_layout_) != VK_SUCCESS) {
    return false;
  }
  VkAttachmentDescription attachment = {};
  attachment.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
  attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
  attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
  attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  VkAttachmentReference reference = {0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.pDepthStencilAttachment = &reference;
  VkRenderPassCreateInfo pass_info = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  pass_info.attachmentCount = 1;
  pass_info.pAttachments = &attachment;
  pass_info.subpassCount = 1;
  pass_info.pSubpasses = &subpass;
  if (dfn.vkCreateRenderPass(device, &pass_info, nullptr, &truth_depth_resample_pass_) != VK_SUCCESS) return false;
  VkPipelineShaderStageCreateInfo stages[2] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,
       truth_resample_shaders_[0], "main", nullptr},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT,
       truth_depth_resample_shader_, "main", nullptr}};
  VkPipelineVertexInputStateCreateInfo vertex_input = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo input_assembly = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
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
  if (truth_depth_stencil_export_) {  // (the exported value replaces the reference)
    depth.stencilTestEnable = VK_TRUE;
    depth.front = {VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE, VK_COMPARE_OP_ALWAYS,
                   0xFFu, 0xFFu, 0u};
    depth.back = depth.front;
  }
  VkPipelineColorBlendStateCreateInfo blend_state = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
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
  info.layout = truth_depth_resample_layout_;
  info.renderPass = truth_depth_resample_pass_;
  if (dfn.vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &truth_depth_resample_pipeline_) !=
      VK_SUCCESS) {
    truth_depth_resample_pipeline_ = VK_NULL_HANDLE;
    return false;
  }
  truth_depth_resample_failed_ = false;
  return true;
}

bool NativeRenderer::EnsureTruthResampleObjects() {
  if (truth_resample_pipelines_[1] != VK_NULL_HANDLE) return true;
  if (truth_resample_failed_) return false;
  truth_resample_failed_ = true;  // until everything below exists
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  const char* sources[2] = {kTruthResampleVertexShader, kTruthResampleFragmentShader};
  const VkShaderStageFlagBits stages_bits[2] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT};
  for (int i = 0; i < 2; ++i) {
    std::vector<uint32_t> spirv;
    std::string error;
    if (!command_processor_.CompileGlslToSpirv(stages_bits[i], sources[i], spirv, error)) {
      REXGPU_ERROR("[native][truth] resample shader compilation failed: {}", error);
      return false;
    }
    truth_resample_shaders_[i] =
        ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
    if (truth_resample_shaders_[i] == VK_NULL_HANDLE) return false;
  }
  VkDescriptorSetLayoutBinding binding = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                          VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo set_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  set_info.bindingCount = 1;
  set_info.pBindings = &binding;
  if (dfn.vkCreateDescriptorSetLayout(device, &set_info, nullptr, &truth_resample_set_layout_) != VK_SUCCESS) {
    return false;
  }
  VkPushConstantRange push = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4 * sizeof(float)};
  VkPipelineLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &truth_resample_set_layout_;
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &push;
  if (dfn.vkCreatePipelineLayout(device, &layout_info, nullptr, &truth_resample_layout_) != VK_SUCCESS) return false;
  for (uint32_t kind = 1; kind <= 2; ++kind) {
    // The image's pixels outside the scissor kept (load), in its attachment layout before and after.
    VkAttachmentDescription attachment = {};
    attachment.format = kTruthColorFormats[kind];
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference reference = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &reference;
    VkRenderPassCreateInfo pass_info = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    pass_info.attachmentCount = 1;
    pass_info.pAttachments = &attachment;
    pass_info.subpassCount = 1;
    pass_info.pSubpasses = &subpass;
    if (dfn.vkCreateRenderPass(device, &pass_info, nullptr, &truth_resample_passes_[kind]) != VK_SUCCESS) return false;
    VkPipelineShaderStageCreateInfo stages[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,
         truth_resample_shaders_[0], "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT,
         truth_resample_shaders_[1], "main", nullptr}};
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
    info.layout = truth_resample_layout_;
    info.renderPass = truth_resample_passes_[kind];
    if (dfn.vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &truth_resample_pipelines_[kind]) !=
        VK_SUCCESS) {
      truth_resample_pipelines_[kind] = VK_NULL_HANDLE;
      return false;
    }
  }
  truth_resample_failed_ = false;
  return true;
}

// A twin into its image (up) or the image into its twin: colour sampled linearly at the exact ratio, depth (and its
// stencil) at the nearest pixel - never the source's partly drawn last row / column going up. Both in their
// attachment layout before and after.
bool NativeRenderer::TruthResampleTwin(TruthImage& t, bool up) {
  const bool depth = t.kind == 3;
  if (depth ? !EnsureTruthDepthResampleObjects() : (t.kind != 1 && t.kind != 2) || !EnsureTruthResampleObjects()) {
    return false;
  }
  if (truth_pool_ == VK_NULL_HANDLE || (depth && (!t.sample_view || !t.stencil_view || !t.draw_sample_view ||
                                                  !t.draw_stencil_view))) {
    return false;
  }
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  const uint32_t i = up ? 1 : 0;
  const VkImageLayout read_layout =
      depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  if (t.sync_sets[i] == VK_NULL_HANDLE) {
    VkDescriptorSetAllocateInfo alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = truth_pool_;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = depth ? &truth_depth_set_layout_ : &truth_resample_set_layout_;
    if (dfn.vkAllocateDescriptorSets(device, &alloc, &t.sync_sets[i]) != VK_SUCCESS) {
      t.sync_sets[i] = VK_NULL_HANDLE;
      return false;
    }
    VkDescriptorImageInfo image_info[2] = {
        {depth ? sampler_nearest_ : sampler_, depth ? (up ? t.draw_sample_view : t.sample_view)
                                                    : (up ? t.draw_view : t.sample_view),
         read_layout},
        {sampler_nearest_, up ? t.draw_stencil_view : t.stencil_view, read_layout}};
    VkWriteDescriptorSet writes[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
    for (uint32_t w = 0; w < 2; ++w) {
      writes[w].dstSet = t.sync_sets[i];
      writes[w].dstBinding = w;
      writes[w].descriptorCount = 1;
      writes[w].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[w].pImageInfo = &image_info[w];
    }
    dfn.vkUpdateDescriptorSets(device, depth ? 2 : 1, writes, 0, nullptr);
  }
  const uint32_t target_w = up ? t.width : t.draw_width, target_h = up ? t.height : t.draw_height;
  const VkRenderPass pass = depth ? truth_depth_resample_pass_ : truth_resample_passes_[t.kind];
  if (t.sync_fbs[i] == VK_NULL_HANDLE) {
    VkFramebufferCreateInfo fb_info = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fb_info.renderPass = pass;
    fb_info.attachmentCount = 1;
    fb_info.pAttachments = up ? &t.view : &t.draw_view;
    fb_info.width = target_w;
    fb_info.height = target_h;
    fb_info.layers = 1;
    if (dfn.vkCreateFramebuffer(device, &fb_info, nullptr, &t.sync_fbs[i]) != VK_SUCCESS) {
      t.sync_fbs[i] = VK_NULL_HANDLE;
      return false;
    }
  }
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const VkImageAspectFlags aspects = depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                           : VK_IMAGE_ASPECT_COLOR_BIT;
  const VkImageLayout attachment_layout =
      depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  const VkAccessFlags attachment_access =
      depth ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
            : VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  const VkPipelineStageFlags attachment_stages =
      depth ? VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
            : VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  // The source to be sampled; the target's earlier reads (stores, resolves, fills, copies) done before it is drawn over.
  VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = up ? t.draw_image : t.image;
  barrier.srcAccessMask = attachment_access;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barrier.oldLayout = attachment_layout;
  barrier.newLayout = read_layout;
  barrier.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(aspects);
  VkMemoryBarrier reads_done = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | attachment_access,
                                attachment_access};
  cb.CmdVkPipelineBarrier(attachment_stages | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | attachment_stages, 0, 1, &reads_done, 0, nullptr, 1,
                          &barrier);
  VkRenderPassBeginInfo begin_info = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  begin_info.renderPass = pass;
  begin_info.framebuffer = t.sync_fbs[i];
  begin_info.renderArea = {{0, 0}, {target_w, target_h}};
  cb.CmdVkBeginRenderPass(&begin_info, VK_SUBPASS_CONTENTS_INLINE);
  const VkViewport viewport = {0.0f, 0.0f, float(target_w), float(target_h), 0.0f, 1.0f};
  const VkRect2D scissor = {{0, 0}, {target_w, target_h}};
  cb.CmdVkSetViewport(0, 1, &viewport);
  cb.CmdVkSetScissor(0, 1, &scissor);
  cb.CmdVkBindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS,
                       depth ? truth_depth_resample_pipeline_ : truth_resample_pipelines_[t.kind]);
  const VkPipelineLayout layout = depth ? truth_depth_resample_layout_ : truth_resample_layout_;
  cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &t.sync_sets[i], 0, nullptr);
  // Source pixels per target pixel, and the last source position used: going up the twin's whole pixels (the draw
  // ratio of the image's size, rounded down), going down the whole image. Colour: a texel-centre position (bilinear);
  // depth: a pixel index (nearest).
  const float ratio = up ? truth_draw_ratio_ : 1.0f / truth_draw_ratio_;
  const float whole_w = up ? std::floor(float(t.width) * truth_draw_ratio_) : float(t.width);
  const float whole_h = up ? std::floor(float(t.height) * truth_draw_ratio_) : float(t.height);
  const float constants[4] = {ratio, ratio, whole_w - (depth ? 1.0f : 0.5f), whole_h - (depth ? 1.0f : 0.5f)};
  cb.CmdVkPushConstants(layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(constants), constants);
  cb.CmdVkDraw(3, 1, 0, 0);
  cb.CmdVkEndRenderPass();
  barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barrier.dstAccessMask = attachment_access;
  barrier.oldLayout = read_layout;
  barrier.newLayout = attachment_layout;
  // (Session 56 review: and the image drawn into, for what reads or draws over it next - with lazy truth passes no
  // render pass dependency does that any more.)
  VkImageMemoryBarrier closing[2] = {barrier, barrier};
  closing[1].image = up ? t.image : t.draw_image;
  closing[1].srcAccessMask = depth ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  closing[1].dstAccessMask = attachment_access | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  closing[1].oldLayout = closing[1].newLayout = attachment_layout;
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | attachment_stages,
                          attachment_stages | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                          0, 0, nullptr, 0, nullptr, 2, closing);
  // (Native passes and the command processor bind their own pipeline, sets and viewport after this.)
  truth_bound_valid_ = false;
  truth_foreign_bind_ = true;
  command_processor_.NativeDropBindings();
  ++truth_counts_.twin_blits[up ? 1 : 0];
  return true;
}

void NativeRenderer::TruthTileRects(const TruthImage& image, const std::bitset<2048>& tiles,
                                    std::vector<std::array<int32_t, 4>>& out) const {
  out.clear();
  if (tiles.none() || !image.width || !image.height) return;
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const uint32_t row_tiles = image.row_tiles();
  const int32_t tw = int32_t((image.is_64bpp() ? 40u : 80u) * scale_x), th = int32_t(16u * scale_y);
  const uint32_t rows = (2048u + row_tiles - 1) / row_tiles;
  std::vector<std::array<int32_t, 4>>& open = truth_scratch_open_;  // rects ending at the previous row
  open.clear();
  for (uint32_t r = 0; r < rows && int32_t(r) * th < int32_t(image.height); ++r) {
    std::vector<std::array<int32_t, 4>>& next = truth_scratch_next_;  // (Session 47: scratch, no allocation)
    next.clear();
    uint32_t c = 0;
    while (c < row_tiles) {
      const uint32_t tile0 = (image.base + r * row_tiles + c) & 2047u;
      if (!tiles.test(tile0) || r * row_tiles + c >= 2048) {
        ++c;
        continue;
      }
      uint32_t c1 = c + 1;
      while (c1 < row_tiles && r * row_tiles + c1 < 2048 && tiles.test((image.base + r * row_tiles + c1) & 2047u)) {
        ++c1;
      }
      std::array<int32_t, 4> rect = {int32_t(c) * tw, int32_t(r) * th, std::min(int32_t(c1) * tw, int32_t(image.width)),
                                     std::min(int32_t(r + 1) * th, int32_t(image.height))};
      auto same = std::find_if(open.begin(), open.end(), [&](const std::array<int32_t, 4>& o) {
        return o[0] == rect[0] && o[2] == rect[2] && o[3] == rect[1];
      });
      if (same != open.end()) {
        rect[1] = (*same)[1];
        open.erase(same);
      }
      next.push_back(rect);
      c = c1;
    }
    for (const auto& o : open) out.push_back(o);  // not continued in this row
    open.swap(next);
  }
  for (const auto& o : open) out.push_back(o);
}

void NativeRenderer::TruthStore(int16_t index, const std::bitset<2048>& tiles, bool staging) {
  TruthStoreMany(&index, 1, tiles, staging);
}

// The image's store set (binding 0 `buffer`, binding 1 the image, binding 2 a depth image's stencil): into the staging
// buffer (staging) or the EDRAM. Session 47: from TruthStoreMany, for TruthFillTexture's pack shader too.
VkDescriptorSet NativeRenderer::TruthStoreSet(TruthImage& t, bool staging, VkBuffer buffer) {
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  VkDescriptorSet& set = t.store_sets[staging ? 0 : 1];
  if (!staging && set != VK_NULL_HANDLE && t.store_set_edram != buffer) {
    truth_retired_sets_.push_back({command_processor_.GetCurrentSubmission(), set});
    set = VK_NULL_HANDLE;
  }
  if (set == VK_NULL_HANDLE) {
    VkDescriptorSetAllocateInfo alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = truth_pool_;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &truth_store_set_layout_;
    if (dfn.vkAllocateDescriptorSets(device, &alloc, &set) != VK_SUCCESS) {
      set = VK_NULL_HANDLE;
      REXGPU_ERROR("[native][truth] store descriptor set allocation failed");
      return VK_NULL_HANDLE;
    }
    VkDescriptorBufferInfo buffer_info = {buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo image_info = {sampler_nearest(), t.sample_view,
                                        t.kind == 3 ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                                    : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[2] = {};
    for (uint32_t i = 0; i < 2; ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
    }
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &buffer_info;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &image_info;
    // Session 46: a depth image's stencil (binding 2; the colour store never reads it).
    VkDescriptorImageInfo stencil_info = {sampler_nearest(), t.stencil_view,
                                          VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet stencil_write = writes[1];
    stencil_write.dstBinding = 2;
    stencil_write.pImageInfo = &stencil_info;
    VkWriteDescriptorSet all_writes[3] = {writes[0], writes[1], stencil_write};
    dfn.vkUpdateDescriptorSets(device, t.kind == 3 && t.stencil_view ? 3 : 2, all_writes, 0, nullptr);
    if (!staging) t.store_set_edram = buffer;
  }
  return set;
}

// The images' owned tiles among `tiles` into the buffer, all owners in one go (one barrier before, one after).
void NativeRenderer::TruthStoreMany(const int16_t* indices, size_t count, const std::bitset<2048>& tiles,
                                   bool staging) {
  // Session 52. (Session 80, gpu_native_truth_carry_copies: the copies reading these images - a store only reads
  // them; more than four: every copy.)
  DeferConcern stores;
  stores.all = count > std::size(stores.images);
  for (size_t k = 0; k < count && !stores.all; ++k) {
    if (indices[k] >= 0 && size_t(indices[k]) < truth_images_.size()) stores.images[k] = truth_images_[indices[k]].image;
  }
  if (stores.all) {
    TruthFlushDeferred(kDeferFlushStores);
  } else {
    TruthFlushDeferredFor(kDeferFlushStores, stores);
  }
  EndTruthPass();  // (outside any render pass)
  const VkBuffer buffer = staging ? truth_staging_ : command_processor_.NativeEdramBuffer();
  if (buffer == VK_NULL_HANDLE) return;
  // Session 80 (owned passes): the stored images' dirty tiles get our pixels first.
  for (size_t k = 0; k < count && !owned_dirty_.empty(); ++k) {
    OwnedDirtyBeforeRead(tiles, indices[k], "a truth store (alias load / write-out)");
  }
  for (size_t k = 0; k < count; ++k) {  // Session 54 (draw twins)
    if (indices[k] >= 0 && size_t(indices[k]) < truth_images_.size() && (truth_images_[indices[k]].owned & tiles).any())
      TruthSyncRead(indices[k]);
  }
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  // (Session 47, CPU: the jobs - TruthStoreJob - are kept in truth_store_jobs_ and reused, no allocations per call.)
  size_t job_count = 0;
  for (size_t k = 0; k < count; ++k) {
    const int16_t index = indices[k];
    if (index < 0 || size_t(index) >= truth_images_.size()) continue;
    TruthImage& t = truth_images_[index];
    if (g_truth_debug_store) {  // wwe2k14 patch (Session 69, gpu_native_truth_debug_span_pitch)
      REXGPU_INFO("[native][truth][span]   store image {}: initialized {} image {} owned in tiles {} staging {}", index,
                  t.initialized, t.image != VK_NULL_HANDLE, (t.owned & tiles).count(), staging);
    }
    if (!t.initialized || t.image == VK_NULL_HANDLE) continue;
    const std::bitset<2048> owned = t.owned & tiles;  // (Session 47)
    if (owned.none()) continue;
    if (truth_store_jobs_.size() <= job_count) truth_store_jobs_.emplace_back();
    TruthStoreJob& job = truth_store_jobs_[job_count];
    job.index = index;
    job.stencil_rects.clear();
    if (t.kind == 3 && t.stencil_view) {
      TruthTileRects(t, owned & ~t.stencil_valid, job.rects);
      TruthTileRects(t, owned & t.stencil_valid, job.stencil_rects);
    } else {
      TruthTileRects(t, owned, job.rects);
    }
    if (g_truth_debug_store) {
      REXGPU_INFO("[native][truth][span]   store image {}: rects {} stencil rects {}", index, job.rects.size(),
                  job.stencil_rects.size());
    }
    if (job.rects.empty() && job.stencil_rects.empty()) continue;
    if (TruthStoreSet(t, staging, buffer) == VK_NULL_HANDLE) {
      if (g_truth_debug_store) REXGPU_INFO("[native][truth][span]   store image {}: NO STORE SET", index);
      continue;
    }
    ++job_count;
  }
  if (!job_count) return;
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  FrameSlot& slot = slots_[SlotOf(frame_)];
  const uint32_t query_base = SlotOf(frame_) * kQueriesPerFrame;
  FrameSlot::Segment* segment = nullptr;
  // (Session 49: inside a write-back's timer, its staging stores are the write-back's - kind 6.)
  if (query_pool_ && slot_queries_ok_ && !truth_timer_kind_ && slot.queries + 2 <= QueryLimit()) {
    segment = &slot.segments.emplace_back();
    segment->kind = 5;
    segment->native_begin = slot.queries++;
    cb.CmdVkWriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, query_base + segment->native_begin);
  }
  constexpr VkPipelineStageFlags kBufferStages =
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
  constexpr VkAccessFlags kBufferAccesses = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  constexpr VkPipelineStageFlags kAttachmentStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                                     VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                                     VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  VkMemoryBarrier buffer_barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, kBufferAccesses,
                                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
  std::vector<VkImageMemoryBarrier>& image_barriers = truth_store_barriers_;
  image_barriers.resize(job_count);
  for (size_t k = 0; k < job_count; ++k) {
    const TruthImage& t = truth_images_[truth_store_jobs_[k].index];
    const bool depth = t.kind == 3;
    VkImageMemoryBarrier& b = image_barriers[k];
    b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = t.image;
    b.srcAccessMask = depth ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.oldLayout = depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    b.newLayout = depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(
        depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
  }
  cb.CmdVkPipelineBarrier(kBufferStages | kAttachmentStages, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                          &buffer_barrier, 0, nullptr, uint32_t(image_barriers.size()), image_barriers.data());
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const uint32_t tile_w = 80 * scale_x, tile_h = 16 * scale_y, tile_dwords = tile_w * tile_h;
  for (size_t k = 0; k < job_count; ++k) {
    const TruthStoreJob& job = truth_store_jobs_[k];
    const TruthImage& t = truth_images_[job.index];
    const bool depth = t.kind == 3;
    command_processor_.BindExternalComputePipeline(truth_store_pipelines_[depth ? 1 : 0]);
    cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, truth_store_layout_, 0, 1,
                               &t.store_sets[staging ? 0 : 1], 0, nullptr);
    for (size_t n = 0; n < job.rects.size() + job.stencil_rects.size(); ++n) {
      const bool own_stencil = n >= job.rects.size();
      const auto& r = own_stencil ? job.stencil_rects[n - job.rects.size()] : job.rects[n];
      const uint32_t pc[10] = {(t.base % 2048u) * tile_dwords,
                               t.pitch_tiles * tile_dwords,
                               tile_w,
                               tile_h,
                               uint32_t(r[0]),
                               uint32_t(r[1]),
                               uint32_t(r[2] - r[0]),
                               uint32_t(r[3] - r[1]),
                               2048u * tile_dwords,
                               depth ? (t.format & 1u) |
                                           (own_stencil ? 4u : staging ? (t.stencil & 0xFFu) << 8 : 2u)
                                     : (t.format << 4)};
      cb.CmdVkPushConstants(truth_store_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
      cb.CmdVkDispatch((pc[6] + 7) / 8, (pc[7] + 7) / 8, 1);
      ++truth_counts_.store_dispatches;
      if (owned_frame_on_) ++owned_frame_stats_.truth_stores;
    }
  }
  buffer_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  buffer_barrier.dstAccessMask = kBufferAccesses;
  for (VkImageMemoryBarrier& b : image_barriers) {
    const bool depth = b.newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    b.srcAccessMask = 0;
    b.dstAccessMask = depth ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                  VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                            : VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    std::swap(b.oldLayout, b.newLayout);
  }
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, kBufferStages | kAttachmentStages, 0, 1,
                          &buffer_barrier, 0, nullptr, uint32_t(image_barriers.size()), image_barriers.data());
  if (segment) {
    segment->native_end = slot.queries++;
    cb.CmdVkWriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, query_base + segment->native_end);
  }
}

VkDescriptorSet NativeRenderer::TruthResolveSource(const draw_util::ResolveInfo& info) {
  uint32_t base = 0, row_length = 0, rows = 0, pitch = 0;
  info.GetCopyEdramTileSpan(base, row_length, rows, pitch);
  return TruthStageSpan(base, row_length, rows, pitch);
}

// Session 61 (DLSS / DLAA in truth frames): a surface's tiles staged as TruthResolveSource stages a resolve's.
VkDescriptorSet NativeRenderer::TruthStageSurface(uint32_t base_tiles, uint32_t pitch_tiles, uint32_t rows) {
  return TruthStageSpan(base_tiles, pitch_tiles, rows, pitch_tiles);
}

VkDescriptorSet NativeRenderer::TruthStageSpan(uint32_t base, uint32_t row_length, uint32_t rows, uint32_t pitch) {
  TruthFlushDeferred(kDeferFlushStores);  // Session 52
  if (!truth_ || !replacing_) return VK_NULL_HANDLE;
  if (!EnsureTruthStaging()) {
    TruthFallback(kTruthStaging, nullptr);
    return VK_NULL_HANDLE;
  }
  std::bitset<2048> span;
  if (uint64_t(rows) * std::max(1u, pitch) >= 2048) {
    span.set();
  } else {
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c = 0; c < row_length; ++c) span.set((base + r * pitch + c) & 2047u);
    }
  }
  // The tiles nobody drew in truth frames: the EDRAM's own (a boundary copy, counted).
  // (Session 47, CPU: by the images' owned tiles.)
  const std::bitset<2048> from_edram = span & ~truth_owned_any_;
  std::vector<int16_t>& owners = truth_scratch_owners_;
  owners.clear();
  for (size_t i = 0; i < truth_images_.size(); ++i) {
    if ((span & truth_images_[i].owned).any()) owners.push_back(int16_t(i));
  }
  // wwe2k14 patch (Session 69, gpu_native_truth_debug_span_pitch): the span's owner map (the red block's resolve).
  static int debug_spans_logged = 0;
  if (const int32_t dp = REXCVAR_GET(gpu_native_truth_debug_span_pitch);
      dp > 0 && uint32_t(dp) == pitch && row_length == pitch && debug_spans_logged < 8) {
    ++debug_spans_logged;
    g_truth_debug_store = true;  // (the stores below log, until the span's TruthStoreMany is done)
    REXGPU_INFO("[native][truth][span] frame {}: base {} row_length {} rows {} pitch {}; owners {}", frame_, base,
                row_length, rows, pitch, owners.size());
    for (int16_t o : owners) {
      const TruthImage& t = truth_images_[o];
      REXGPU_INFO("[native][truth][span]   image {}: base {} pitch {} kind {} format {} {}x{} owned in span {}", o,
                  t.base, t.pitch_tiles, t.kind, t.format, t.width, t.height, (span & t.owned).count());
    }
    for (uint32_t r = 0; r < rows; ++r) {
      std::string line;
      for (uint32_t c = 0; c < row_length; ++c) {
        const uint32_t tile = (base + r * pitch + c) & 2047u;
        const int16_t o = TruthOwnerOf(tile);
        line += o < 0 ? 'E' : char(o < 10 ? '0' + o : 'a' + (o - 10) % 26);
      }
      REXGPU_INFO("[native][truth][span]   row {:2}: {}", r, line);
    }
  }
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  if (from_edram.any() && command_processor_.NativeEdramBuffer() != VK_NULL_HANDLE) {
    uint32_t scale_x = 1, scale_y = 1;
    command_processor_.NativeResolutionScale(scale_x, scale_y);
    const VkDeviceSize tile_bytes = VkDeviceSize(80 * scale_x) * (16 * scale_y) * 4;
    std::vector<VkBufferCopy> regions;
    for (uint32_t i = 0; i < 2048;) {
      if (!from_edram.test(i)) {
        ++i;
        continue;
      }
      uint32_t j = i + 1;
      while (j < 2048 && from_edram.test(j)) ++j;
      regions.push_back({i * tile_bytes, i * tile_bytes, (j - i) * tile_bytes});
      i = j;
    }
    VkMemoryBarrier before = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                              VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    cb.CmdVkCopyBuffer(command_processor_.NativeEdramBuffer(), truth_staging_, uint32_t(regions.size()),
                       regions.data());
    VkMemoryBarrier after = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                             VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &after, 0,
                            nullptr, 0, nullptr);
    truth_counts_.staged_edram += uint32_t(from_edram.count());
  }
  TruthStoreMany(owners.data(), owners.size(), span, true);
  g_truth_debug_store = false;  // (Session 69 diagnostic)
  ++truth_counts_.staged;
  return truth_staging_resolve_set_;
}

bool NativeRenderer::TruthResolveDirect(const draw_util::ResolveInfo& info,
                                        const draw_util::ResolveCopyShaderConstants& constants,
                                        VkDescriptorSet dest_set, uint32_t group_count_x, uint32_t group_count_y) {
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  // (Session 52: at a resolution scale with gpu_native_truth_direct_resolve_scaled - the scaled shader, no texture.)
  const bool scaled = scale_x != 1 || scale_y != 1;
  if (!truth_ || !replacing_ || !REXCVAR_GET(gpu_native_truth_direct_resolve) ||
      (scaled && !REXCVAR_GET(gpu_native_truth_direct_resolve_scaled))) {
    return false;
  }
  // Session 59 (gpu_native_truth_direct_resolve_depth): a 1x depth copy straight from the depth image.
  // (At a resolution scale too: the scaled layout, from the binding's start.)
  if (info.IsCopyingDepth()) {
    return REXCVAR_GET(gpu_native_truth_direct_resolve_depth) &&
           TruthResolveDepth1x(info, constants.dest_base, constants.dest_relative.dest_coordinate_info.packed,
                               constants.dest_relative.dest_info.value, dest_set);
  }
  const draw_util::ResolveEdramInfo& edram = info.color_edram_info;
  if (info.IsCopyingDepth() || edram.is_depth || edram.format_is_64bpp || edram.msaa_samples != xenos::MsaaSamples::k1X ||
      uint32_t(edram.format) != 0 || info.rb_copy_control.copy_src_select != 0 || info.copy_dest_info.copy_dest_array ||
      uint32_t(info.copy_dest_coordinate_info.copy_sample_select) != 0) {
    return false;
  }
  // The tiles the copy reads: one RGBA8 image of this surface owns them all.
  uint32_t base = 0, row_length = 0, rows = 0, pitch = 0;
  info.GetCopyEdramTileSpan(base, row_length, rows, pitch);
  if (!rows || !row_length || uint64_t(rows) * std::max(1u, pitch) >= 2048) return false;
  const int16_t index = TruthOwnerOf(base & 2047u);
  if (index < 0) return false;
  const std::bitset<2048>& owned = truth_images_[index].owned;
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < row_length; ++c) {
      if (!owned.test((base + r * pitch + c) & 2047u)) return false;
    }
  }
  TruthImage& t = truth_images_[index];
  if (t.kind != 1 || t.format != 0 || t.base != info.color_original_base || t.pitch_tiles != edram.pitch_tiles ||
      !t.initialized) {
    return false;
  }
  EndTruthPass();  // (review 4: before the dummy's first barrier)
  TruthSyncRead(index);  // Session 54 (draw twins)
  if (!EnsureResolveObjects() || !EnsureTruthStaging() || !EnsureTruthDummy()) return false;
  // (Session 52: a resolve into a texture a deferred one writes - in order: those first.)
  // (Session 56: at a scale too, for the texture-only ones of gpu_native_truth_skip_writeback_scaled.)
  const bool scaled_texture_only = scaled && REXCVAR_GET(gpu_native_truth_skip_writeback_scaled) &&
                                   REXCVAR_GET(gpu_native_truth_direct_resolve_scaled_texture) &&
                                   (!command_processor_.NativeResolveReadback() ||
                                    REXCVAR_GET(gpu_native_truth_skip_writeback_debug_immediate));
  const bool batch = (!scaled || scaled_texture_only) && REXCVAR_GET(gpu_native_truth_batch_resolves);
  if (scaled && !EnsureResolveScaledPipeline()) return false;
  // (Review: at a scale, only a copy the image holds whole - the 1x shader reads the staging buffer outside it.)
  if (scaled && (scale_x * ((((uint32_t(edram.base_tiles) - info.color_original_base) & 2047u) % std::max(1u, uint32_t(edram.pitch_tiles))) * 80 +
                            uint32_t(info.coordinate_info.edram_offset_x_div_8) * 8) +
                         uint32_t(info.coordinate_info.width_div_8) * 8 * scale_x >
                     t.width ||
                 scale_y * ((((uint32_t(edram.base_tiles) - info.color_original_base) & 2047u) / std::max(1u, uint32_t(edram.pitch_tiles))) * 16 +
                            uint32_t(info.coordinate_info.edram_offset_y_div_8) * 8) +
                         group_count_y * 8 >
                     t.height)) {
    return false;
  }
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  if (t.resolve_set == VK_NULL_HANDLE) {
    VkDescriptorSetAllocateInfo alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = truth_pool_;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &post_writeback_set_layout_;
    if (dfn.vkAllocateDescriptorSets(device, &alloc, &t.resolve_set) != VK_SUCCESS) {
      t.resolve_set = VK_NULL_HANDLE;
      return false;
    }
    // Binding 0 (the shader's EDRAM) is the staging buffer: never read for pixels inside the image.
    VkDescriptorBufferInfo buffer_info = {truth_staging_, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo color_info = {sampler_nearest(), t.sample_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo stencil_info = {sampler_nearest(), truth_dummy_stencil_view_,
                                          VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo depth_info = {sampler_nearest(), truth_dummy_depth_view_,
                                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[4] = {};
    for (uint32_t i = 0; i < 4; ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = t.resolve_set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = i ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    writes[0].pBufferInfo = &buffer_info;
    writes[1].pImageInfo = &color_info;
    writes[2].pImageInfo = &stencil_info;
    writes[3].pImageInfo = &depth_info;
    dfn.vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
  }
  // Where the copy's first pixel is in the image (as ResolveFromTarget computes it).
  const uint32_t pitch_tiles = std::max(1u, uint32_t(edram.pitch_tiles));
  const uint32_t tile_delta = (uint32_t(edram.base_tiles) - info.color_original_base) & 2047u;
  // The texture the copy feeds, filled by the same dispatch when it qualifies (N3a).
  uint32_t texture_w = 0, texture_h = 0;
  void* texture_handle = nullptr;
  int32_t texture_layer = -1;  // Session 79: the cube face it fills (-1: a 2D texture)
  VkDescriptorSet texture_set = VK_NULL_HANDLE;
  const bool scaled_texture = scaled && REXCVAR_GET(gpu_native_truth_direct_resolve_scaled_texture) &&
                              resolve_scaled_tex_pipeline_ != VK_NULL_HANDLE;
  if ((!scaled || scaled_texture) && resolve_texture_ && resolve_tex_pipeline_ != VK_NULL_HANDLE &&
      (info.copy_dest_base & 0xFFF) == 0) {
    if (REXCVAR_GET(gpu_native_truth_cover_outdated)) {  // (the texels this dispatch writes, host)
      command_processor_.NativeSetTextureCover(
          scale_x * uint32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8,
          scale_y * uint32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8,
          scale_x * uint32_t(info.coordinate_info.width_div_8) * 8, group_count_y * 8);
    }
    // (Session 52: with batching the texture's barrier is held - recorded with the batch if this one is deferred.)
    const bool hold = batch;
    if (hold) command_processor_.NativeHoldTextureBarrier(true);
    // (Session 79, gpu_native_truth_fill_cube_faces: at 1x a cube face too - copy_dest_base is the face's start.)
    texture_set = command_processor_.NativeTextureForResolve(
        info.copy_dest_base >> 12, uint32_t(info.copy_dest_coordinate_info.pitch_aligned_div_32),
        uint32_t(info.copy_dest_info.copy_dest_endian), texture_w, texture_h, texture_handle, scaled_texture,
        xenos::TextureFormat::k_8_8_8_8, !scaled, &texture_layer);
    // (Review fix: a cube face only when the copy stays inside it - a taller or wider copy writes the next face's
    // memory, which the face's fill would not hold, and the cube would be marked up to date. The texture's state the
    // call set - compute-write usage, its barrier - is consistent either way: its next sampling moves it back.)
    if (texture_layer >= 0 && texture_set != VK_NULL_HANDLE &&
        (uint32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8 + uint32_t(info.coordinate_info.width_div_8) * 8 >
             texture_w ||
         uint32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8 + group_count_y * 8 > texture_h)) {
      texture_set = VK_NULL_HANDLE;
      texture_handle = nullptr;
      texture_layer = -1;
      // (A held barrier pushed now, so the image's layout follows its usage whatever happens to this copy.)
      if (hold) command_processor_.NativeHoldTextureBarrier(false);
    }
  }
  const auto release_held_barrier = [&]() { command_processor_.NativeHoldTextureBarrier(false); };
  uint32_t pc[12] = {constants.dest_relative.edram_info.packed,
                     constants.dest_relative.coordinate_info.packed,
                     constants.dest_relative.dest_info.value,
                     constants.dest_relative.dest_coordinate_info.packed,
                     constants.dest_base,
                     scale_x * ((tile_delta % pitch_tiles) * 80 +
                                uint32_t(info.coordinate_info.edram_offset_x_div_8) * 8),
                     scale_y * ((tile_delta / pitch_tiles) * 16 +
                                uint32_t(info.coordinate_info.edram_offset_y_div_8) * 8),
                     t.width,
                     t.height,
                     1u | 0x200u,  // RGBA8 (the packed values), every pixel from the image
                     texture_w,
                     texture_h};
  // Session 49 (gpu_native_truth_skip_writeback): the texture only, when it holds every texel the copy writes (the
  // shader stores texel (x, y) at destination texel (x, y); x0 + 8 per invocation column, one row per invocation).
  if (!scaled && texture_set != VK_NULL_HANDLE && truth_skip_writeback_ &&
      (!command_processor_.NativeResolveReadback() || REXCVAR_GET(gpu_native_truth_skip_writeback_debug_immediate))) {
    const uint32_t x0 = uint32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8;
    const uint32_t y0 = uint32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8;
    const uint32_t w = uint32_t(info.coordinate_info.width_div_8) * 8, h = group_count_y * 8;
    // (Session 79, gpu_native_truth_texture_only_cube_faces: a cube face's too - read back through its face's view.)
    if (command_processor_.NativeTextureReadable(texture_handle, false, texture_layer) != 4u ||
        (texture_layer >= 0 && !REXCVAR_GET(gpu_native_truth_texture_only_cube_faces)) ||
        !EnsureMaterializePipeline()) {
      ++wb_counts_.kept[2];
    } else if (x0 + w > texture_w || y0 + h > texture_h) {
      ++wb_counts_.kept[3];
    } else {
      pc[9] |= 0x400u;
      WbRecord& r = truth_wb_pending_;
      r = WbRecord();
      r.texture = texture_handle;
      r.layer = texture_layer;
      r.extent_start = info.copy_dest_extent_start;
      r.extent_length = info.copy_dest_extent_length;
      r.dest_offset = truth_dest_offset_;
      r.dest_range = truth_dest_range_;
      r.base = info.copy_dest_base;
      r.pitch = uint32_t(info.copy_dest_coordinate_info.pitch_aligned_div_32);
      r.x0 = x0;
      r.y0 = y0;
      r.x1 = x0 + w;
      r.y1 = y0 + h;
      r.dest_base = constants.dest_base;  // (already relative to the destination's offset)
      r.endian = uint32_t(info.copy_dest_info.copy_dest_endian);
      r.frame = frame_;  // (tex_x0 / tex_y0 0: the texture's texels are the destination's)
      truth_wb_skipped_ = true;
      ++wb_counts_.skipped_direct;
    }
  } else if (scaled_texture_only && texture_set != VK_NULL_HANDLE && resolve_scaled_tex_pipeline_ != VK_NULL_HANDLE &&
             (info.copy_dest_base & 0xFFF) == 0) {
    // Session 56 (gpu_native_truth_skip_writeback_scaled): the host texels kNativeResolveScaledComputeShader writes
    // (8 per invocation along x, one row per invocation), all inside the scaled texture it fills.
    const uint32_t sx = (pc[1] >> 16) & 7u, sy = (pc[1] >> 19) & 7u;
    const uint32_t x0 = ((pc[3] >> 20) & 0xFu) * 8u * sx, y0 = ((pc[3] >> 24) & 0xFu) * 8u * sy;
    const uint32_t x1 = x0 + ((pc[1] >> 5) & 0x7FFu) * sx * 8u, y1 = y0 + group_count_y * 8u;
    uint64_t use_start = 0, use_length = 0;
    if (command_processor_.NativeTextureReadable(texture_handle, true) != 4u || !EnsureMaterializeScaledPipeline() ||
        !sx || !sy) {
      ++wb_counts_.kept[2];
    } else if (x1 > texture_w || y1 > texture_h || x1 <= x0 || y1 <= y0) {
      ++wb_counts_.kept[3];
    } else if (!command_processor_.NativeScaledResolveRange(info.copy_dest_extent_start, info.copy_dest_extent_length,
                                                            use_start, use_length)) {
      ++wb_counts_.kept[1];
    } else {
      pc[9] |= 0x400u;
      WbRecord& r = truth_wb_pending_;
      r = WbRecord();
      r.texture = texture_handle;
      r.extent_start = info.copy_dest_extent_start;
      r.extent_length = info.copy_dest_extent_length;
      r.dest_offset = truth_dest_offset_;
      r.dest_range = truth_dest_range_;
      r.base = info.copy_dest_base;
      r.pitch = (pc[3] & 1023u);  // (in 32-texel units, as the 1x records)
      r.x0 = x0;
      r.y0 = y0;
      r.x1 = x1;
      r.y1 = y1;
      r.endian = pc[2] & 7u;
      r.frame = frame_;
      r.scale_x = sx;
      r.scale_y = sy;
      r.guest_x0 = uint32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8;
      r.guest_y0 = uint32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8;
      r.guest_x1 = r.guest_x0 + uint32_t(info.coordinate_info.width_div_8) * 8;
      r.guest_y1 = r.guest_y0 + info.height_div_8 * 8;
      r.scaled_use_start = use_start;
      r.scaled_use_length = use_length;
      truth_wb_skipped_ = true;
      ++wb_counts_.skipped_direct;
    }
  } else if (truth_skip_writeback_) {
    ++wb_counts_.kept[1];
  }
  // Session 49 (the texture's rectangle this dispatch fills: its texels are the destination's)
  // (Session 66: declared outside the block - the deferred entry keeps it for gpu_native_truth_drop_dead_resolves.)
  int32_t filled_rect[4];
  {
    const int32_t fx0 = int32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8;
    const int32_t fy0 = int32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8;
    // (Session 56: at a scale in guest texels - the scaled records' rectangles are compared in them.)
    filled_rect[0] = fx0;
    filled_rect[1] = fy0;
    filled_rect[2] = std::min(int32_t(texture_w / scale_x), fx0 + int32_t(info.coordinate_info.width_div_8) * 8);
    filled_rect[3] = std::min(int32_t(texture_h / scale_y),
                              fy0 + (scaled ? int32_t(info.height_div_8) * 8 : int32_t(group_count_y) * 8));
    TruthResolveOverlaps(info, texture_set != VK_NULL_HANDLE ? texture_handle : nullptr, filled_rect, texture_layer);
  }
  // Session 56 (diagnostic): the scaled texture only.
  if (scaled && texture_set != VK_NULL_HANDLE && REXCVAR_GET(gpu_native_truth_debug_scaled_texture_only)) {
    pc[9] |= 0x400u;
  }
  // Session 72 part 4 (gpu_native_truth_skip_restores): from here on the copy is recorded (deferred or now).
  TruthNoteRestoreSource(info, index,
                         pc[5] == 0 && pc[6] == 0 && info.copy_dest_coordinate_info.offset_x_div_8 == 0 &&
                             info.copy_dest_coordinate_info.offset_y_div_8 == 0,
                         scale_x * uint32_t(info.coordinate_info.width_div_8) * 8,
                         scale_y * uint32_t(info.height_div_8) * 8);
  // Session 52 (gpu_native_truth_batch_resolves): a texture-only resolve is deferred (see TruthFlushDeferredSlow).
  {
    bool same_texture = false;
    // (Session 79: a cube's faces are separate texels - an entry writing another face is not the same texture here.)
    for (const DeferredResolve& d : truth_deferred_) {
      if (d.texture == texture_handle && texture_handle && d.layer == texture_layer) same_texture = true;
    }
    const bool texwrite_only = texture_set != VK_NULL_HANDLE && (pc[9] & 0x400u) != 0;
    // Session 79 (gpu_native_truth_defer_memory_resolves): or one that writes guest memory too (1x, outside readback
    // windows, its destination use held); an entry writing memory it overlaps goes first (two dispatches of one batch
    // have no barrier between them).
    VulkanSharedMemory* const shared_memory = command_processor_.NativeSharedMemory();
    const bool memory_defer = !texwrite_only && !scaled && batch && shared_memory &&
                              REXCVAR_GET(gpu_native_truth_defer_memory_resolves) &&
                              !command_processor_.NativeResolveReadback() && command_processor_.held_dest_use_ &&
                              !command_processor_.held_dest_scaled_ && info.copy_dest_extent_length;
    const bool memory_overlap =
        memory_defer && TruthDeferredWritesMemory(info.copy_dest_extent_start,
                                                  uint64_t(info.copy_dest_extent_start) + info.copy_dest_extent_length,
                                                  &info);
    const bool deferrable = texwrite_only || memory_defer;
    // (A renamed image has only its rectangle yet: a copy reaching outside it waits for the flush's copy.)
    const bool outside = t.rename_source != VK_NULL_HANDLE &&
                         (int32_t(pc[5]) < t.rename_rect[0] || int32_t(pc[6]) < t.rename_rect[1] ||
                          int32_t(pc[5] + uint32_t(info.coordinate_info.width_div_8) * 8 * scale_x) >
                              t.rename_rect[2] ||
                          int32_t(pc[6] + group_count_y * 8) > t.rename_rect[3]);
    // Session 66 (G1.1, gpu_native_truth_drop_dead_resolves): every entry of the batch that writes this texture lies
    // inside the rectangle this resolve writes (so its texels are all overwritten, and no draw sampled them: sampling a
    // texture an entry writes flushes the batch, which then no longer holds the entry): those entries are dropped.
    // (Only the 1x texture-only case: the dispatch writes exactly filled_rect and reads nothing of the texture.)
    bool drop_same = false;
    if (same_texture && !scaled && batch && texwrite_only && !outside && texture_handle &&
        REXCVAR_GET(gpu_native_truth_drop_dead_resolves) >= 1 && filled_rect[2] > filled_rect[0] &&
        filled_rect[3] > filled_rect[1]) {
      drop_same = true;
      for (const DeferredResolve& d : truth_deferred_) {
        if (d.texture != texture_handle || d.layer != texture_layer) continue;
        const int32_t* r = d.drop_rect;
        if (r[2] <= r[0] || r[3] <= r[1] || r[0] < filled_rect[0] || r[1] < filled_rect[1] || r[2] > filled_rect[2] ||
            r[3] > filled_rect[3]) {
          drop_same = false;
          break;
        }
      }
    }
    // (The texture's barrier to its compute-write layout, from the first dropped entry that held one: it takes the
    // texture from its layout before the batch; the new entry's own would be GENERAL -> GENERAL.)
    VkImage carry_image = VK_NULL_HANDLE;
    VkPipelineStageFlags carry_stage = 0;
    VkAccessFlags carry_access = 0;
    VkImageLayout carry_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (drop_same) {
      for (size_t i = 0; i < truth_deferred_.size();) {
        const DeferredResolve& d = truth_deferred_[i];
        if (d.texture != texture_handle || d.layer != texture_layer) {
          ++i;
          continue;
        }
        if (carry_image == VK_NULL_HANDLE && d.texture_image != VK_NULL_HANDLE) {
          carry_image = d.texture_image;
          carry_stage = d.texture_src_stage;
          carry_access = d.texture_src_access;
          carry_layout = d.texture_old_layout;
        }
        if (d.set_count == 2) {
          ++truth_counts_.dropped_dead_fills;
        } else {
          ++truth_counts_.dropped_dead;
        }
        if (d.pinned) {  // Session 80 (gpu_native_truth_carry_copies): a carried copy read by nothing
          command_processor_.NativeTexturePin(d.texture, false);
          truth_counts_.carry_dropped += d.crossed ? 1 : 0;
          carry_used_.erase(d.texture);
        }
        truth_deferred_.erase(truth_deferred_.begin() + ptrdiff_t(i));
      }
    }
    if (!texwrite_only && segment_log_frames_ > 0) {  // (Session 79, diagnostic: a resolve writing memory)
      REXGPU_INFO("[native][seg] frame {} memory resolve: deferred {} (batch {}, readback {}, dest use held {}, "
                  "outside {}, overlap {}, same texture {}) page 0x{:05X} {}x{}",
                  frame_, memory_defer && !outside && !memory_overlap, batch, command_processor_.NativeResolveReadback(),
                  command_processor_.held_dest_use_ || memory_defer, outside, memory_overlap, same_texture,
                  info.copy_dest_base >> 12, uint32_t(info.coordinate_info.width_div_8) * 8, info.height_div_8 * 8);
    }
    if ((same_texture && !drop_same) || !batch || !deferrable || outside || memory_overlap) {
      TruthFlushDeferred(outside                    ? kDeferFlushOutside
                         : (!batch || !deferrable) ? kDeferFlushNotBatched
                         : memory_overlap          ? kDeferFlushOtherResolve
                                                   : kDeferFlushSameTexture);
    }
    if (outside) EndTruthPass();
    if (!(batch && deferrable && !outside)) release_held_barrier();  // (pushed now: recorded as before)
    if (batch && deferrable && !outside) {
      DeferredResolve& d = truth_deferred_.emplace_back();
      d.texture_image = VK_NULL_HANDLE;
      command_processor_.NativeTakeHeldTextureBarrier(d.texture_image, d.texture_src_stage, d.texture_src_access,
                                                      d.texture_old_layout);
      command_processor_.NativeHoldTextureBarrier(false);
      if (carry_image != VK_NULL_HANDLE) {  // Session 66 (G1.1): the dropped entries' barrier is the one that counts
        d.texture_image = carry_image;
        d.texture_src_stage = carry_stage;
        d.texture_src_access = carry_access;
        d.texture_old_layout = carry_layout;
      }
      // (Session 79: another face of a cube an earlier entry of the batch takes to its compute-write layout - the whole
      // image - gets no second barrier in the same command; its faces are separate texels, needing no ordering.)
      for (size_t k = 0; k + 1 < truth_deferred_.size() && d.texture_image != VK_NULL_HANDLE; ++k) {
        if (truth_deferred_[k].texture_image == d.texture_image) d.texture_image = VK_NULL_HANDLE;
      }
      // (G1.1 drops texture-only entries only: an entry writing memory is never dropped.)
      if (!scaled && texture_handle && texwrite_only) std::copy(filled_rect, filled_rect + 4, d.drop_rect);
      d.page = info.copy_dest_base >> 12;  // Session 80 (the flush log)
      // Session 80 (gpu_native_truth_serve_copies): a plain copy - all of the image from its origin into all of the
      // texture from its origin, the texels unchanged (k_8_8_8_8, no exponent bias, 1x) but for a red / blue swap.
      d.plain_swap = info.copy_dest_info.copy_dest_swap != 0;
      d.plain = !scaled && texwrite_only && texture_handle &&
                uint32_t(info.copy_dest_info.copy_dest_format) == uint32_t(xenos::TextureFormat::k_8_8_8_8) &&
                !info.copy_dest_info.copy_dest_exp_bias &&
                !info.copy_dest_info.copy_dest_array && pc[5] == 0 && pc[6] == 0 && !info.copy_dest_base_x &&
                !info.copy_dest_base_y && !info.copy_dest_coordinate_info.offset_x_div_8 &&
                !info.copy_dest_coordinate_info.offset_y_div_8 &&
                uint32_t(info.coordinate_info.width_div_8) * 8 == t.width && info.height_div_8 * 8 == t.height &&
                texture_w == t.width && texture_h == t.height;
      d.plain_w = texture_w;
      d.plain_h = texture_h;
      d.pipeline = scaled ? resolve_scaled_tex_pipeline_ : resolve_tex_pipeline_;  // (Session 56: at a scale too)
      d.layout = resolve_tex_layout_;
      d.sets[0] = t.resolve_set;
      d.sets[1] = dest_set;
      d.sets[2] = texture_set;
      // (Session 80, gpu_native_truth_carry_copies: dest_set is this submission's - a carried copy run later takes a
      // fresh one of this range.)
      d.submission = command_processor_.GetCurrentSubmission();
      d.dest_offset = truth_dest_offset_;
      d.dest_range = truth_dest_range_;
      if (memory_defer) {
        // Session 79: the copy into memory (and the texture, if it fills one); its destination use recorded at the
        // flush, its pages marked until then (TruthMaterialize flushes the batch for a reader of them).
        const bool texwrite = texture_set != VK_NULL_HANDLE;
        d.pipeline = texwrite ? resolve_tex_pipeline_ : resolve_pipeline_;
        d.layout = texwrite ? resolve_tex_layout_ : resolve_layout_;
        d.set_count = texwrite ? 3 : 2;
        d.mem_start = command_processor_.held_dest_start_;
        d.mem_length = command_processor_.held_dest_length_;
        command_processor_.held_dest_use_ = false;  // (taken: the later drop finds none)
        d.extent_start = info.copy_dest_extent_start;
        d.extent_length = info.copy_dest_extent_length;
        if (!info.copy_dest_info.copy_dest_array && info.copy_dest_texture_base) {
          d.mem_base = info.copy_dest_texture_base;
          d.mem_pitch = uint32_t(info.copy_dest_coordinate_info.pitch_aligned_div_32);
          d.mem_format = uint32_t(info.copy_dest_info.copy_dest_format);
          d.mem_rect[0] = int32_t(info.copy_dest_base_x + uint32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8);
          d.mem_rect[1] = int32_t(info.copy_dest_base_y + uint32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8);
          d.mem_rect[2] = d.mem_rect[0] + int32_t(info.coordinate_info.width_div_8) * 8;
          d.mem_rect[3] = d.mem_rect[1] + int32_t(info.height_div_8) * 8;
        }
        shared_memory->SetRangeSkipped(d.extent_start, d.extent_length, true);
        ++truth_counts_.deferred_memory;
      }
      std::memcpy(d.pc, pc, sizeof(pc));
      d.group_count_x = group_count_x;
      d.group_count_y = group_count_y;
      d.image = t.image;
      d.index = index;
      d.texture = texture_handle;
      d.layer = texture_layer;  // Session 79
      texture_written_ = texture_handle;
      if (texture_handle) ++report_.textures_filled;
      ++truth_counts_.direct;
      ++truth_counts_.deferred;
      return true;
    }
  }
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  VkImageMemoryBarrier image_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  image_barrier.srcQueueFamilyIndex = image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  image_barrier.image = t.image;
  image_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  image_barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  image_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  image_barrier.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &image_barrier);
  const bool texwrite = texture_set != VK_NULL_HANDLE;
  const VkPipelineLayout layout = texwrite ? resolve_tex_layout_ : resolve_layout_;
  command_processor_.BindExternalComputePipeline(scaled     ? (texwrite ? resolve_scaled_tex_pipeline_
                                                                         : resolve_scaled_pipeline_)
                                                 : texwrite ? resolve_tex_pipeline_
                                                            : resolve_pipeline_);
  const VkDescriptorSet sets[3] = {t.resolve_set, dest_set, texture_set};
  cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, texwrite ? 3 : 2, sets, 0, nullptr);
  cb.CmdVkPushConstants(layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
  // (Session 52, gpu_native_truth_defer_dest_use: the destination's use, unless the dispatch writes the texture only.)
  if (!(pc[9] & 0x400u)) command_processor_.NativeFlushResolveDestUse();
  command_processor_.SubmitBarriers(true);  // (also the texture's barrier, pushed by NativeTextureForResolve)
  cb.CmdVkDispatch(group_count_x, group_count_y, 1);
  if (texwrite) {
    texture_written_ = texture_handle;
    ++report_.textures_filled;
  }
  image_barrier.srcAccessMask = 0;
  image_barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  std::swap(image_barrier.oldLayout, image_barrier.newLayout);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                          nullptr, 0, nullptr, 1, &image_barrier);
  ++truth_counts_.direct;
  return true;
}

// Session 79 (gpu_native_truth_alias_direct): kTruthAliasDirectFragmentShader for colour / depth owners, each as a
// colour-only and a depth-only load (the load pipelines' state and render passes), on the store's set 0.
bool NativeRenderer::EnsureAliasDirectPipelines() {
  if (alias_direct_failed_ || truth_store_set_layout_ == VK_NULL_HANDLE || load_vertex_shader_ == VK_NULL_HANDLE) {
    return false;
  }
  if (alias_direct_layout_ != VK_NULL_HANDLE) return true;
  alias_direct_failed_ = true;  // until everything below exists
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  for (int owner_depth = 0; owner_depth < 2; ++owner_depth) {
    std::vector<uint32_t> spirv;
    std::string error;
    const std::string source = std::string("#version 450\n") + (owner_depth ? "#define OWNER_DEPTH 1\n" : "") +
                               kTruthAliasDirectFragmentShader;
    if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_FRAGMENT_BIT, source, spirv, error)) {
      REXGPU_ERROR("[native][truth] alias direct shader compilation failed: {}", error);
      return false;
    }
    alias_direct_shaders_[owner_depth] =
        ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
    if (alias_direct_shaders_[owner_depth] == VK_NULL_HANDLE) return false;
  }
  VkPushConstantRange push = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(AliasDirectConstants)};
  VkPipelineLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &truth_store_set_layout_;
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &push;
  if (dfn.vkCreatePipelineLayout(device, &layout_info, nullptr, &alias_direct_layout_) != VK_SUCCESS) {
    alias_direct_layout_ = VK_NULL_HANDLE;
    return false;
  }
  // (As the load pipelines: a full-screen triangle, no culling, no blending, depth ALWAYS when it writes depth.)
  VkPipelineShaderStageCreateInfo stages[2] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, load_vertex_shader_,
       "main", nullptr},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, VK_NULL_HANDLE,
       "main", nullptr}};
  VkPipelineVertexInputStateCreateInfo vertex_input = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo input_assembly = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
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
  info.layout = alias_direct_layout_;
  for (uint32_t cls = 0; cls < kPostTargetClasses; ++cls) {
    if (!post_class_ok_[cls]) continue;
    info.renderPass = truth_render_pass_keep_nodep_[cls] != VK_NULL_HANDLE ? truth_render_pass_keep_nodep_[cls]
                                                                            : post_render_pass_continue_[cls];
    for (int owner_depth = 0; owner_depth < 2; ++owner_depth) {
      stages[1].module = alias_direct_shaders_[owner_depth];
      for (int target_depth = 0; target_depth < 2; ++target_depth) {
        blend_attachment.colorWriteMask = target_depth ? 0x0 : 0xF;
        depth.depthTestEnable = target_depth ? VK_TRUE : VK_FALSE;
        depth.depthWriteEnable = target_depth ? VK_TRUE : VK_FALSE;
        depth.depthCompareOp = VK_COMPARE_OP_ALWAYS;
        VkPipeline& out = alias_direct_pipelines_[owner_depth][target_depth][cls];
        if (dfn.vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &out) != VK_SUCCESS) {
          out = VK_NULL_HANDLE;
          return false;
        }
      }
    }
  }
  alias_direct_failed_ = false;
  return true;
}




// Session 59 (gpu_native_truth_direct_resolve_scaled_full_texture): kNativeResolveScaledFullComputeShader with
// FULL_TEXOUT 1 (k_16_16, r32ui) / 2 (k_16_16_16_16, rgba16ui) on a layout with the texture's set (as the RGBA8 scaled
// resolve's NATIVE_TEXWRITE).
bool NativeRenderer::EnsureResolveScaledFullTexPipelines() {
  if (resolve_scaled_full_tex_pipelines_[1] != VK_NULL_HANDLE) return true;
  if (resolve_scaled_full_tex_failed_ || truth_store_set_layout_ == VK_NULL_HANDLE) return false;
  const VkDescriptorSetLayout texture_set_layout = command_processor_.NativeTextureSetLayout();
  if (texture_set_layout == VK_NULL_HANDLE) return false;
  resolve_scaled_full_tex_failed_ = true;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  if (resolve_scaled_full_tex_layout_ == VK_NULL_HANDLE) {
    const VkDescriptorSetLayout set_layouts[3] = {
        truth_store_set_layout_,
        command_processor_.GetSingleTransientDescriptorLayout(
            VulkanCommandProcessor::SingleTransientDescriptorLayout::kStorageBufferCompute),
        texture_set_layout};
    VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 12 * sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 3;
    layout_info.pSetLayouts = set_layouts;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    if (dfn.vkCreatePipelineLayout(device, &layout_info, nullptr, &resolve_scaled_full_tex_layout_) != VK_SUCCESS) {
      resolve_scaled_full_tex_layout_ = VK_NULL_HANDLE;
      return false;
    }
  }
  for (uint32_t variant = 0; variant < 2; ++variant) {
    if (resolve_scaled_full_tex_pipelines_[variant] != VK_NULL_HANDLE) continue;
    std::vector<uint32_t> spirv;
    std::string error;
    const std::string source = std::string("#version 450\n#define FULL_TEXOUT ") + (variant ? "2" : "1") + "\n" +
                               kNativeResolveScaledFullComputeShader;
    if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT, source, spirv, error)) {
      REXGPU_ERROR("[native][truth] scaled full resolve + texture shader compilation failed: {}", error);
      return false;
    }
    const VkShaderModule shader =
        ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
    if (shader == VK_NULL_HANDLE) return false;
    VkComputePipelineCreateInfo create = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    create.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                    shader, "main", nullptr};
    create.layout = resolve_scaled_full_tex_layout_;
    const bool ok = dfn.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &create, nullptr,
                                                 &resolve_scaled_full_tex_pipelines_[variant]) == VK_SUCCESS;
    dfn.vkDestroyShaderModule(device, shader, nullptr);
    if (!ok) {
      resolve_scaled_full_tex_pipelines_[variant] = VK_NULL_HANDLE;
      return false;
    }
  }
  resolve_scaled_full_tex_failed_ = false;
  return true;
}


bool NativeRenderer::TruthResolveFull1x(const draw_util::ResolveInfo& info, uint32_t dest_base,
                                        uint32_t dest_coordinate_info, uint32_t dest_info, VkDescriptorSet dest_set) {
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  if (!truth_ || !replacing_ || scale_x != 1 || scale_y != 1 || !REXCVAR_GET(gpu_native_truth_direct_resolve_full_1x) ||
      dest_set == VK_NULL_HANDLE) {
    return false;
  }
  // (Texture-only records this copy overlaps go into memory before it writes - as the stock copy's.)
  TruthResolveOverlaps(info, nullptr);
  const draw_util::ResolveEdramInfo& edram = info.color_edram_info;
  const reg::RB_COPY_DEST_INFO dest = info.copy_dest_info;
  if (info.IsCopyingDepth() || edram.msaa_samples != xenos::MsaaSamples::k1X || dest.copy_dest_array ||
      dest.copy_dest_exp_bias || info.rb_copy_control.copy_src_select != 0 ||
      uint32_t(info.copy_dest_coordinate_info.copy_sample_select) != 0 || (dest_base & 3u)) {
    return false;
  }
  uint32_t mode;
  if (uint32_t(dest.copy_dest_format) == uint32_t(xenos::TextureFormat::k_8)) {
    mode = 4;
  } else if (uint32_t(dest.copy_dest_format) == uint32_t(xenos::TextureFormat::k_4_4_4_4)) {
    mode = 5;
  } else {
    return false;
  }
  if (dest.copy_dest_swap) mode |= 16u;
  uint32_t base = 0, row_length = 0, rows = 0, pitch = 0;
  info.GetCopyEdramTileSpan(base, row_length, rows, pitch);
  if (!rows || !row_length || uint64_t(rows) * std::max(1u, pitch) >= 2048) return false;
  const int16_t index = TruthOwnerOf(base & 2047u);
  if (index < 0) return false;
  TruthImage& t = truth_images_[index];
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < row_length; ++c) {
      if (!t.owned.test((base + r * pitch + c) & 2047u)) return false;
    }
  }
  if (t.kind != 1 || t.format != uint32_t(xenos::ColorRenderTargetFormat::k_8_8_8_8) || !t.initialized ||
      t.base != info.color_original_base || edram.pitch_tiles != t.row_tiles() || t.is_64bpp()) {
    return false;
  }
  const uint32_t tile_delta = (uint32_t(edram.base_tiles) - t.base) & 2047u;
  const uint32_t src_x = (tile_delta % t.row_tiles()) * 80 + uint32_t(info.coordinate_info.edram_offset_x_div_8) * 8;
  const uint32_t src_y = (tile_delta / t.row_tiles()) * 16 + uint32_t(info.coordinate_info.edram_offset_y_div_8) * 8;
  const uint32_t w = uint32_t(info.coordinate_info.width_div_8) * 8;
  const uint32_t h = uint32_t(info.height_div_8) * 8;
  if (!w || !h || src_x + w > t.width || src_y + h > t.height) return false;
  // Session 79 (gpu_native_truth_defer_full_resolves): deferred with the batch, as a 1x memory-writing colour copy
  // (TruthResolveDirect's memory_defer) - with its texture write, if it has one. Else the batch first, as before.
  VulkanSharedMemory* const shared_memory = command_processor_.NativeSharedMemory();
  // (Review fix, Session 79: a renamed image holds only its rename rectangle until the flush's copy - a copy reaching
  // outside it is not deferred, as TruthResolveDirect's `outside`; the flush below runs that copy first.)
  const bool outside = t.rename_source != VK_NULL_HANDLE &&
                       (int32_t(src_x) < t.rename_rect[0] || int32_t(src_y) < t.rename_rect[1] ||
                        int32_t(src_x + w) > t.rename_rect[2] || int32_t(src_y + h) > t.rename_rect[3]);
  const bool defer =
      REXCVAR_GET(gpu_native_truth_defer_full_resolves) && REXCVAR_GET(gpu_native_truth_batch_resolves) &&
      shared_memory && !command_processor_.NativeResolveReadback() && command_processor_.held_dest_use_ &&
      !command_processor_.held_dest_scaled_ && info.copy_dest_extent_length && !outside &&
      !TruthDeferredWritesMemory(info.copy_dest_extent_start,
                                 uint64_t(info.copy_dest_extent_start) + info.copy_dest_extent_length, &info);
  if (!defer) TruthFlushDeferred(outside ? kDeferFlushOutside : kDeferFlushOtherResolve);
  if (!EnsureTruthStaging() || !EnsureResolveScaledFullPipeline()) {
    if (defer) TruthFlushDeferred(kDeferFlushOtherResolve);  // (Session 79: as without deferring)
    return false;  // (its layout)
  }
  if (resolve_full_1x_pipeline_ == VK_NULL_HANDLE) {
    if (defer) TruthFlushDeferred(kDeferFlushOtherResolve);  // (Session 79: the batch before any refusal below)
    if (resolve_full_1x_failed_) return false;
    resolve_full_1x_failed_ = true;
    const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
    const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
    std::vector<uint32_t> spirv;
    std::string error;
    if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT,
                                               std::string("#version 450\n") + kNativeResolveFull1xComputeShader, spirv,
                                               error)) {
      REXGPU_ERROR("[native][truth] 1x full resolve shader compilation failed: {}", error);
      return false;
    }
    const VkShaderModule shader =
        ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
    if (shader == VK_NULL_HANDLE) return false;
    VkComputePipelineCreateInfo create = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    create.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                    shader, "main", nullptr};
    create.layout = resolve_scaled_full_layout_;
    const bool ok = dfn.vkCreateComputePipelines(vulkan_device->device(), VK_NULL_HANDLE, 1, &create, nullptr,
                                                 &resolve_full_1x_pipeline_) == VK_SUCCESS;
    dfn.vkDestroyShaderModule(vulkan_device->device(), shader, nullptr);
    if (!ok) {
      resolve_full_1x_pipeline_ = VK_NULL_HANDLE;
      return false;
    }
    resolve_full_1x_failed_ = false;
  }
  const VkDescriptorSet source_set = TruthStoreSet(t, true, truth_staging_);
  if (source_set == VK_NULL_HANDLE) {
    if (defer) TruthFlushDeferred(kDeferFlushOtherResolve);  // (Session 79: as without deferring)
    return false;
  }
  // Session 59 (gpu_native_truth_direct_resolve_full_1x_texture): the texture the copy feeds, written by the same
  // dispatch through its 8 / 16-bit storage view (found by its own base; written at the copy's place in it).
  const uint32_t tex_x0 = info.copy_dest_base_x + uint32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8;
  const uint32_t tex_y0 = info.copy_dest_base_y + uint32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8;
  VkDescriptorSet texture_set = VK_NULL_HANDLE;
  void* texture_handle = nullptr;
  // (Review: not when a fill already wrote the texture - NativeTruthFillOnly runs first; r8ui / r16ui need the
  // device's extended storage formats.)
  if (REXCVAR_GET(gpu_native_truth_direct_resolve_full_1x_texture) && !texture_written_ &&
      command_processor_.GetVulkanDevice()->properties().shaderStorageImageExtendedFormats &&
      info.copy_dest_texture_base &&
      (info.copy_dest_texture_base & 0xFFF) == 0 && !command_processor_.NativeResolveReadback() &&
      EnsureResolveScaledFullTexPipelines()) {
    const uint32_t variant = (mode & 15u) == 4u ? 0u : 1u;
    if (resolve_full_1x_tex_pipelines_[variant] == VK_NULL_HANDLE && !resolve_full_1x_tex_failed_) {
      resolve_full_1x_tex_failed_ = true;
      const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
      const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
      std::vector<uint32_t> spirv;
      std::string error;
      if (command_processor_.CompileGlslToSpirv(
              VK_SHADER_STAGE_COMPUTE_BIT,
              std::string("#version 450\n#define FULL1X_TEXOUT ") + (variant ? "2" : "1") + "\n" +
                  kNativeResolveFull1xComputeShader,
              spirv, error)) {
        const VkShaderModule shader =
            ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
        if (shader != VK_NULL_HANDLE) {
          VkComputePipelineCreateInfo create = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
          create.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                          shader, "main", nullptr};
          create.layout = resolve_scaled_full_tex_layout_;
          if (dfn.vkCreateComputePipelines(vulkan_device->device(), VK_NULL_HANDLE, 1, &create, nullptr,
                                           &resolve_full_1x_tex_pipelines_[variant]) != VK_SUCCESS) {
            resolve_full_1x_tex_pipelines_[variant] = VK_NULL_HANDLE;
          } else {
            resolve_full_1x_tex_failed_ = false;
          }
          dfn.vkDestroyShaderModule(vulkan_device->device(), shader, nullptr);
        }
      } else {
        REXGPU_ERROR("[native][truth] 1x full resolve + texture shader compilation failed: {}", error);
      }
    }
    if (resolve_full_1x_tex_pipelines_[variant] != VK_NULL_HANDLE) {
      uint32_t texture_w = 0, texture_h = 0;
      // (Session 79: deferring - the texture's barrier held, for the entry or pushed below.)
      if (defer) command_processor_.NativeHoldTextureBarrier(true);
      texture_set = command_processor_.NativeTextureForResolve(
          info.copy_dest_texture_base >> 12, uint32_t(info.copy_dest_coordinate_info.pitch_aligned_div_32),
          uint32_t(dest.copy_dest_endian), texture_w, texture_h, texture_handle, false,
          variant ? xenos::TextureFormat::k_4_4_4_4 : xenos::TextureFormat::k_8);
      if (texture_set != VK_NULL_HANDLE && (tex_x0 + w > texture_w || tex_y0 + h > texture_h)) {
        texture_set = VK_NULL_HANDLE;  // (the texture is outdated and reloaded as before; its pushed barrier is harmless)
        texture_handle = nullptr;
        if (defer) command_processor_.NativeHoldTextureBarrier(false);  // (Session 79: pushed now, as before)
      }
    }
  }
  const bool texwrite = texture_set != VK_NULL_HANDLE;
  const VkPipelineLayout full_layout = texwrite ? resolve_scaled_full_tex_layout_ : resolve_scaled_full_layout_;
  if (defer) {
    // Session 79 (gpu_native_truth_defer_full_resolves): the entry. A deferred write into the same texture goes first
    // (two dispatches of one batch have no barrier between them), then this one's held barrier orders the writes.
    bool same_texture = false;
    for (const DeferredResolve& e : truth_deferred_) same_texture = same_texture || (texwrite && e.texture == texture_handle);
    if (same_texture) TruthFlushDeferred(kDeferFlushSameTexture);
    TruthSyncRead(index);  // Session 54 (draw twins)
    DeferredResolve& d = truth_deferred_.emplace_back();
    d.texture_image = VK_NULL_HANDLE;
    if (texwrite) {
      command_processor_.NativeTakeHeldTextureBarrier(d.texture_image, d.texture_src_stage, d.texture_src_access,
                                                      d.texture_old_layout);
      for (size_t k = 0; k + 1 < truth_deferred_.size() && d.texture_image != VK_NULL_HANDLE; ++k) {
        if (truth_deferred_[k].texture_image == d.texture_image) d.texture_image = VK_NULL_HANDLE;
      }
    }
    command_processor_.NativeHoldTextureBarrier(false);
    d.pipeline = texwrite ? resolve_full_1x_tex_pipelines_[(mode & 15u) == 4u ? 0 : 1] : resolve_full_1x_pipeline_;
    d.layout = full_layout;
    d.sets[0] = source_set;
    d.sets[1] = dest_set;
    d.sets[2] = texture_set;
    d.set_count = texwrite ? 3 : 2;
    const uint32_t dpc[12] = {src_x,
                              src_y,
                              w,
                              h,
                              mode,
                              dest_base,
                              ((dest_coordinate_info >> 20) & 0xFu) * 8u,
                              ((dest_coordinate_info >> 24) & 0xFu) * 8u,
                              (dest_coordinate_info & 1023u) << 5,
                              dest_info & 7u,
                              tex_x0,
                              tex_y0};
    std::memcpy(d.pc, dpc, sizeof(dpc));
    d.pc_size = texwrite ? uint32_t(sizeof(dpc)) : uint32_t(sizeof(uint32_t) * 10);
    const uint32_t per = (mode & 15u) == 4u ? 4u : 2u;
    d.group_count_x = (w / per + 7) / 8;
    d.group_count_y = (h + 7) / 8;
    d.image = t.image;
    d.index = index;
    d.texture = texwrite ? texture_handle : nullptr;
    d.mem_start = command_processor_.held_dest_start_;
    d.mem_length = command_processor_.held_dest_length_;
    command_processor_.held_dest_use_ = false;  // (taken: recorded at the flush)
    d.extent_start = info.copy_dest_extent_start;
    d.extent_length = info.copy_dest_extent_length;
    if (!info.copy_dest_info.copy_dest_array && info.copy_dest_texture_base) {
      d.mem_base = info.copy_dest_texture_base;
      d.mem_pitch = uint32_t(info.copy_dest_coordinate_info.pitch_aligned_div_32);
      d.mem_format = uint32_t(info.copy_dest_info.copy_dest_format);
      d.mem_rect[0] = int32_t(tex_x0);
      d.mem_rect[1] = int32_t(tex_y0);
      d.mem_rect[2] = d.mem_rect[0] + int32_t(w);
      d.mem_rect[3] = d.mem_rect[1] + int32_t(h);
    }
    shared_memory->SetRangeSkipped(d.extent_start, d.extent_length, true);
    if (texwrite) {
      texture_written_ = texture_handle;  // (up to date after MarkRangeAsResolved: AfterResolveMarked)
      ++report_.textures_filled;
    }
    ++truth_counts_.deferred_memory;
    ++truth_counts_.deferred;
    ++truth_counts_.direct;
    return true;
  }
  EndTruthPass();
  TruthSyncRead(index);  // Session 54 (draw twins)
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = t.image;
  b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  b.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
  command_processor_.BindExternalComputePipeline(
      texwrite ? resolve_full_1x_tex_pipelines_[(mode & 15u) == 4u ? 0 : 1] : resolve_full_1x_pipeline_);
  const VkDescriptorSet sets[3] = {source_set, dest_set, texture_set};
  cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, full_layout, 0, texwrite ? 3 : 2, sets, 0, nullptr);
  if (texwrite) {
    texture_written_ = texture_handle;  // (up to date after MarkRangeAsResolved: AfterResolveMarked)
    ++report_.textures_filled;
  }
  const uint32_t pc[12] = {src_x,
                           src_y,
                           w,
                           h,
                           mode,
                           dest_base,
                           ((dest_coordinate_info >> 20) & 0xFu) * 8u,
                           ((dest_coordinate_info >> 24) & 0xFu) * 8u,
                           (dest_coordinate_info & 1023u) << 5,
                           dest_info & 7u,
                           tex_x0,
                           tex_y0};
  cb.CmdVkPushConstants(full_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        texwrite ? uint32_t(sizeof(pc)) : uint32_t(sizeof(uint32_t) * 10), pc);
  command_processor_.NativeFlushResolveDestUse();  // (the destination's use: this dispatch writes it)
  command_processor_.SubmitBarriers(true);
  const uint32_t per = (mode & 15u) == 4u ? 4u : 2u;
  cb.CmdVkDispatch((w / per + 7) / 8, (h + 7) / 8, 1);
  b.srcAccessMask = 0;
  b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  std::swap(b.oldLayout, b.newLayout);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                          nullptr, 0, nullptr, 1, &b);
  ++truth_counts_.direct;
  return true;
}

// Session 59 (gpu_native_truth_direct_resolve_depth): a fast 32bpp copy of a 1x depth surface one depth truth image
// holds whole, straight from the image into memory: the dwords the staging stores would write (the image's stencil
// where it holds every copied tile's, else its last clear's, as TruthStoreMany), at their destination texels.
bool NativeRenderer::TruthResolveDepth1x(const draw_util::ResolveInfo& info, uint32_t dest_base,
                                         uint32_t dest_coordinate_info, uint32_t dest_info, VkDescriptorSet dest_set) {
  if (!truth_ || !replacing_ || dest_set == VK_NULL_HANDLE) return false;
  // (At a resolution scale: kNativeResolveScaledFullComputeShader's FULL_DEPTH variant, host pixels.)
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const bool scaled = scale_x != 1 || scale_y != 1;
  const draw_util::ResolveEdramInfo& edram = info.depth_edram_info;
  const reg::RB_COPY_DEST_INFO dest = info.copy_dest_info;
  if (!info.IsCopyingDepth() || edram.msaa_samples != xenos::MsaaSamples::k1X || dest.copy_dest_array ||
      dest.copy_dest_exp_bias || dest.copy_dest_swap ||
      uint32_t(info.copy_dest_coordinate_info.copy_sample_select) != 0 || (dest_base & 3u)) {
    return false;
  }
  uint32_t base = 0, row_length = 0, rows = 0, pitch = 0;
  info.GetCopyEdramTileSpan(base, row_length, rows, pitch);
  if (!rows || !row_length || uint64_t(rows) * std::max(1u, pitch) >= 2048) return false;
  const int16_t index = TruthOwnerOf(base & 2047u);
  if (index < 0) return false;
  TruthImage& t = truth_images_[index];
  if (t.kind != 3 || !t.initialized || t.image == VK_NULL_HANDLE || t.stencil_view == VK_NULL_HANDLE ||
      t.base != info.depth_original_base || edram.pitch_tiles != t.row_tiles()) {
    return false;
  }
  // Every copied tile the image's; its stencil the image's for all of them or for none (else the staged stores).
  bool all_stencil = true, no_stencil = true;
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < row_length; ++c) {
      const uint32_t tile = (base + r * pitch + c) & 2047u;
      if (!t.owned.test(tile)) return false;
      const bool own = t.stencil_valid.test(tile);
      all_stencil = all_stencil && own;
      no_stencil = no_stencil && !own;
    }
  }
  if (!all_stencil && !no_stencil) return false;
  const uint32_t tile_delta = (uint32_t(edram.base_tiles) - t.base) & 2047u;
  const uint32_t src_x =
      scale_x * ((tile_delta % t.row_tiles()) * 80 + uint32_t(info.coordinate_info.edram_offset_x_div_8) * 8);
  const uint32_t src_y =
      scale_y * ((tile_delta / t.row_tiles()) * 16 + uint32_t(info.coordinate_info.edram_offset_y_div_8) * 8);
  const uint32_t w = scale_x * uint32_t(info.coordinate_info.width_div_8) * 8;
  const uint32_t h = scale_y * uint32_t(info.height_div_8) * 8;
  if (!w || !h || src_x + w > t.width || src_y + h > t.height) return false;
  // Session 80 (gpu_native_truth_skip_writeback_depth): the fill alone, the stencil kept, a record for the memory.
  if (!scaled &&
      TruthDepthTextureOnly(info, t, index, src_x, src_y, w, h, all_stencil, dest_base, dest_coordinate_info, dest_info)) {
    return true;
  }
  // Session 79 (gpu_native_truth_defer_depth_resolves): deferred with the batch, as a 1x memory-writing colour copy
  // (TruthResolveDirect's memory_defer) - its destination use held, outside readback windows, writing no memory a
  // deferred entry writes. Else the batch first, as before.
  VulkanSharedMemory* const shared_memory = command_processor_.NativeSharedMemory();
  const bool defer =
      REXCVAR_GET(gpu_native_truth_defer_depth_resolves) && REXCVAR_GET(gpu_native_truth_batch_resolves) && !scaled &&
      shared_memory && !command_processor_.NativeResolveReadback() && command_processor_.held_dest_use_ &&
      !command_processor_.held_dest_scaled_ && info.copy_dest_extent_length &&
      !TruthDeferredWritesMemory(info.copy_dest_extent_start,
                                 uint64_t(info.copy_dest_extent_start) + info.copy_dest_extent_length, &info);
  if (!defer) TruthFlushDeferred(kDeferFlushOtherResolve);
  if (!EnsureTruthStaging() || !EnsureResolveScaledFullPipeline()) return false;  // (its layout)
  VkPipeline& depth_pipeline = scaled ? resolve_depth_scaled_pipeline_ : resolve_depth_1x_pipeline_;
  bool& depth_failed = scaled ? resolve_depth_scaled_failed_ : resolve_depth_1x_failed_;
  if (depth_pipeline == VK_NULL_HANDLE) {
    if (defer) TruthFlushDeferred(kDeferFlushOtherResolve);  // (Session 79: the batch before any refusal below)
    if (depth_failed) return false;
    depth_failed = true;
    const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
    const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
    std::vector<uint32_t> spirv;
    std::string error;
    if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT,
                                               scaled ? std::string("#version 450\n#define FULL_DEPTH 1\n") +
                                                            kNativeResolveScaledFullComputeShader
                                                      : std::string("#version 450\n#define FULL1X_DEPTH 1\n") +
                                                            kNativeResolveFull1xComputeShader,
                                               spirv, error)) {
      REXGPU_ERROR("[native][truth] 1x depth resolve shader compilation failed: {}", error);
      return false;
    }
    const VkShaderModule shader =
        ui::vulkan::util::CreateShaderModule(vulkan_device, spirv.data(), spirv.size() * sizeof(uint32_t));
    if (shader == VK_NULL_HANDLE) return false;
    VkComputePipelineCreateInfo create = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    create.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                    shader, "main", nullptr};
    create.layout = resolve_scaled_full_layout_;
    const bool ok = dfn.vkCreateComputePipelines(vulkan_device->device(), VK_NULL_HANDLE, 1, &create, nullptr,
                                                 &depth_pipeline) == VK_SUCCESS;
    dfn.vkDestroyShaderModule(vulkan_device->device(), shader, nullptr);
    if (!ok) {
      depth_pipeline = VK_NULL_HANDLE;
      return false;
    }
    depth_failed = false;
  }
  // (The store set: binding 1 the depth, binding 2 the stencil.)
  const VkDescriptorSet source_set = TruthStoreSet(t, true, truth_staging_);
  if (source_set == VK_NULL_HANDLE) {
    if (defer) TruthFlushDeferred(kDeferFlushOtherResolve);  // (Session 79: as without deferring)
    return false;
  }
  // (Texture-only records this copy overlaps go into memory before it writes - as the stock copy's.)
  TruthResolveOverlaps(info, nullptr);
  const uint32_t mode = (t.format & 1u) | (all_stencil ? 2u : (t.stencil & 0xFFu) << 8);
  // (1x: kNativeResolveFull1xComputeShader's constants; at a scale kNativeResolveScaledFullComputeShader's - the gap,
  // the destination from the binding's start, endian | scale_x << 4 | scale_y << 8.)
  const uint32_t gap = scaled && edram.fill_half_pixel_offset ? (scale_x / 2) | ((scale_y / 2) << 8) : 0u;
  const uint32_t pc[10] = {src_x,
                           src_y,
                           w,
                           h,
                           mode,
                           scaled ? gap : dest_base,
                           ((dest_coordinate_info >> 20) & 0xFu) * 8u,
                           ((dest_coordinate_info >> 24) & 0xFu) * 8u,
                           (dest_coordinate_info & 1023u) << 5,
                           (dest_info & 7u) | (scaled ? (scale_x << 4) | (scale_y << 8) : 0u)};
  if (defer) {
    // Session 79 (gpu_native_truth_defer_depth_resolves): the copy recorded at the batch's flush (TruthFlushDeferredSlow
    // takes the image to DEPTH_STENCIL_READ_ONLY_OPTIMAL and back); its destination use and skipped pages as
    // TruthResolveDirect's memory_defer. A pass into the image flushes the batch first (TruthDeferredSource).
    TruthSyncRead(index);  // Session 54 (draw twins)
    DeferredResolve& d = truth_deferred_.emplace_back();
    d.texture_image = VK_NULL_HANDLE;  // (no texture: the depth fill after it is an entry of its own)
    d.pipeline = depth_pipeline;
    d.layout = resolve_scaled_full_layout_;
    d.sets[0] = source_set;
    d.sets[1] = dest_set;
    d.sets[2] = VK_NULL_HANDLE;
    d.set_count = 2;
    std::fill(std::begin(d.pc), std::end(d.pc), 0u);
    std::memcpy(d.pc, pc, sizeof(pc));
    d.pc_size = sizeof(pc);
    d.group_count_x = (w + 7) / 8;
    d.group_count_y = (h + 7) / 8;
    d.image = t.image;
    d.index = index;
    d.depth = true;
    d.texture = nullptr;
    d.mem_start = command_processor_.held_dest_start_;
    d.mem_length = command_processor_.held_dest_length_;
    command_processor_.held_dest_use_ = false;  // (taken: recorded at the flush)
    d.extent_start = info.copy_dest_extent_start;
    d.extent_length = info.copy_dest_extent_length;
    if (!info.copy_dest_info.copy_dest_array && info.copy_dest_texture_base) {
      d.mem_base = info.copy_dest_texture_base;
      d.mem_pitch = uint32_t(info.copy_dest_coordinate_info.pitch_aligned_div_32);
      d.mem_format = uint32_t(info.copy_dest_info.copy_dest_format);
      d.mem_rect[0] = int32_t(info.copy_dest_base_x + uint32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8);
      d.mem_rect[1] = int32_t(info.copy_dest_base_y + uint32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8);
      d.mem_rect[2] = d.mem_rect[0] + int32_t(info.coordinate_info.width_div_8) * 8;
      d.mem_rect[3] = d.mem_rect[1] + int32_t(info.height_div_8) * 8;
    }
    shared_memory->SetRangeSkipped(d.extent_start, d.extent_length, true);
    ++truth_counts_.deferred_memory;
    ++truth_counts_.deferred;
    TruthNoteRestoreSource(info, index, src_x == 0 && src_y == 0 && ((dest_coordinate_info >> 20) & 0xFFu) == 0, w, h);
    ++truth_counts_.direct;
    ++truth_counts_.direct_depth;
    if (segment_log_frames_ > 0) {
      REXGPU_INFO("[native][seg] frame {} truth depth resolve direct: {}x{} from image {} at {},{} (stencil {}), deferred",
                  frame_, w, h, index, src_x, src_y, all_stencil ? "the image's" : "its last clear's");
    }
    return true;
  }
  EndTruthPass();
  TruthSyncRead(index);  // Session 54 (draw twins)
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  constexpr VkPipelineStageFlags kDepthStages =
      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = t.image;
  b.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  b.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  b.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
  b.subresourceRange =
      ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
  cb.CmdVkPipelineBarrier(kDepthStages | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &b);
  command_processor_.BindExternalComputePipeline(depth_pipeline);
  const VkDescriptorSet sets[2] = {source_set, dest_set};
  cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, resolve_scaled_full_layout_, 0, 2, sets, 0, nullptr);
  cb.CmdVkPushConstants(resolve_scaled_full_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
  command_processor_.NativeFlushResolveDestUse();  // (the destination's use: this dispatch writes it)
  command_processor_.SubmitBarriers(true);
  cb.CmdVkDispatch((w + 7) / 8, (h + 7) / 8, 1);
  b.srcAccessMask = 0;
  b.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  std::swap(b.oldLayout, b.newLayout);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, kDepthStages, 0, 0, nullptr, 0, nullptr, 1, &b);
  // Session 72 part 4 (gpu_native_truth_skip_restores).
  TruthNoteRestoreSource(info, index, src_x == 0 && src_y == 0 && ((dest_coordinate_info >> 20) & 0xFFu) == 0, w, h);
  ++truth_counts_.direct;
  ++truth_counts_.direct_depth;
  if (segment_log_frames_ > 0) {
    REXGPU_INFO("[native][seg] frame {} truth depth resolve direct: {}x{} from image {} at {},{} (stencil {})", frame_,
                w, h, index, src_x, src_y, all_stencil ? "the image's" : "its last clear's");
  }
  return true;
}

// Session 80 (gpu_native_truth_skip_writeback_depth): a 1x 20e4 depth copy (TruthResolveDepth1x's checks passed: one
// depth truth image holds every copied tile, its stencil the image's for all of them or for none) done as its texture
// fill alone, as TruthFillOnly does a colour copy: the fill first; if it holds every texel the copy writes, the
// stencil is kept (the image's: a snapshot entry in the deferred batch, read at the batch's flush like the copy would
// be; none: the last clear's value) and the record goes in (TruthAfterResolveMarked) instead of the copy into memory.
// Only with the batch (the snapshot is a deferred entry). False: the copy follows (the fill done: it is not repeated).
bool NativeRenderer::TruthDepthTextureOnly(const draw_util::ResolveInfo& info, TruthImage& t, int16_t index,
                                           uint32_t src_x, uint32_t src_y, uint32_t w, uint32_t h, bool all_stencil,
                                           uint32_t dest_base, uint32_t dest_coordinate_info, uint32_t dest_info) {
  if (!REXCVAR_GET(gpu_native_truth_skip_writeback_depth) || !truth_skip_writeback_ || texture_written_ ||
      truth_fill_tried_ || !(t.format & 1u) ||
      (command_processor_.NativeResolveReadback() && !REXCVAR_GET(gpu_native_truth_skip_writeback_debug_immediate)) ||
      !REXCVAR_GET(gpu_native_truth_batch_resolves) || !REXCVAR_GET(gpu_native_truth_defer_depth_resolves)) {
    return false;
  }
  int32_t slot = -1;
  if (all_stencil) {
    for (uint32_t i = 0; i < kStencilSlots; ++i) {
      if (!(stencil_slots_busy_ & (1u << i))) {
        slot = int32_t(i);
        break;
      }
    }
    if (slot < 0 || uint64_t((w + 3u) & ~3u) * h > kStencilSlotSize) {
      ++wb_counts_.depth_kept[1];
      return false;
    }
  }
  if (!EnsureDepthWritebackObjects() || !EnsureTruthStaging()) {
    ++wb_counts_.depth_kept[2];
    return false;
  }
  // The fill first (it reads only the image), then the records this copy overlaps (before it would write memory).
  truth_fill_done_ = FillDone();
  TruthFillTexture(info);
  truth_fill_tried_ = true;
  {
    const FillDone& done = truth_fill_done_;
    const int32_t filled_rect[4] = {done.dst_x, done.dst_y, done.dst_x + done.w, done.dst_y + done.h};
    TruthResolveOverlaps(info, texture_written_, done.texture && !done.level ? filled_rect : nullptr);
  }
  const FillDone& f = truth_fill_done_;
  if (!texture_written_ || !f.depth || f.level || f.has_mips ||
      command_processor_.NativeTextureReadable(f.texture, false, -1, true) != 4u || f.w != f.copy_w ||
      f.h != f.copy_h || uint32_t(f.copy_w) != w || uint32_t(f.copy_h) != h || f.dst_x < 0 || f.dst_y < 0 ||
      f.dst_x + f.copy_w > int32_t(f.texture_w) || f.dst_y + f.copy_h > int32_t(f.texture_h)) {
    ++wb_counts_.depth_kept[0];
    return false;
  }
  const uint32_t snap_pitch = (w + 3u) & ~3u;
  if (slot >= 0) {
    const VkDescriptorSet source_set = TruthStoreSet(t, true, truth_staging_);  // (binding 2: the stencil view)
    const VkDescriptorSet slot_set = StencilSlotSet(slot);
    if (source_set == VK_NULL_HANDLE || slot_set == VK_NULL_HANDLE) {
      ++wb_counts_.depth_kept[2];
      return false;
    }
    TruthSyncRead(index);  // Session 54 (draw twins)
    DeferredResolve& d = truth_deferred_.emplace_back();
    d.texture_image = VK_NULL_HANDLE;
    d.page = info.copy_dest_base >> 12;  // Session 80 (the flush log)
    d.pipeline = stencil_snap_pipeline_;
    d.layout = resolve_scaled_full_layout_;
    d.sets[0] = source_set;
    d.sets[1] = slot_set;
    d.sets[2] = VK_NULL_HANDLE;
    d.set_count = 2;
    std::fill(std::begin(d.pc), std::end(d.pc), 0u);
    const uint32_t pc[10] = {src_x, src_y, w, h, 0u, snap_pitch, 0u, 0u, 0u, 0u};
    std::memcpy(d.pc, pc, sizeof(pc));
    d.pc_size = sizeof(pc);
    d.group_count_x = ((w + 3u) / 4u + 7u) / 8u;
    d.group_count_y = (h + 7u) / 8u;
    d.image = t.image;
    d.index = index;
    d.depth = true;
    d.texture = nullptr;
    d.companion_of = f.texture;  // (gpu_native_truth_drop_dead_copies: dead with its fill)
    stencil_slots_busy_ |= 1u << slot;
    ++wb_counts_.depth_snapshots;
    ++truth_counts_.deferred;
  }
  // (Where the copy would write - FULL1X_DEPTH's constants: the destination's first texel, its pitch in 32-texel
  // units, its base relative to the destination binding, its endian.)
  const uint32_t ox = ((dest_coordinate_info >> 20) & 0xFu) * 8u;
  const uint32_t oy = ((dest_coordinate_info >> 24) & 0xFu) * 8u;
  WbRecord& r = truth_wb_pending_;
  r = WbRecord();
  r.texture = f.texture;
  r.extent_start = info.copy_dest_extent_start;
  r.extent_length = info.copy_dest_extent_length;
  r.dest_offset = truth_dest_offset_;
  r.dest_range = truth_dest_range_;
  r.base = info.copy_dest_base;
  r.pitch = dest_coordinate_info & 1023u;
  r.x0 = ox;
  r.y0 = oy;
  r.x1 = ox + w;
  r.y1 = oy + h;
  r.dest_base = dest_base;
  r.endian = dest_info & 7u;
  r.tex_x0 = uint32_t(f.dst_x) - ox;
  r.tex_y0 = uint32_t(f.dst_y) - oy;
  r.frame = frame_;
  r.depth = true;
  r.stencil_slot = slot;
  r.stencil_const = t.stencil & 0xFFu;
  r.stencil_pitch = snap_pitch;
  truth_wb_skipped_ = true;
  ++wb_counts_.depth_skipped;
  TruthNoteRestoreSource(info, index, src_x == 0 && src_y == 0 && ((dest_coordinate_info >> 20) & 0xFFu) == 0, w, h);
  if (segment_log_frames_ > 0) {
    REXGPU_INFO("[native][seg] frame {} truth depth resolve texture-only: {}x{} from image {} at {},{} (stencil {})",
                frame_, w, h, index, src_x, src_y, slot >= 0 ? "kept in a slot" : "its last clear's");
  }
  return true;
}

bool NativeRenderer::TruthResolveScaledFull(const draw_util::ResolveInfo& info, VkDescriptorSet dest_set) {
  TruthFlushDeferred(kDeferFlushOtherResolve);  // Session 52
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  if (!truth_ || !replacing_ || (scale_x == 1 && scale_y == 1) ||
      !REXCVAR_GET(gpu_native_truth_direct_resolve_scaled_full) || dest_set == VK_NULL_HANDLE) {
    return false;
  }
  // (Session 56: records this copy overlaps go into the scaled buffer before it writes - once a resolve.)
  TruthResolveOverlaps(info, nullptr);
  const draw_util::ResolveEdramInfo& edram = info.color_edram_info;
  const reg::RB_COPY_DEST_INFO dest = info.copy_dest_info;
  if (info.IsCopyingDepth() || edram.msaa_samples != xenos::MsaaSamples::k1X || dest.copy_dest_array ||
      dest.copy_dest_exp_bias || info.rb_copy_control.copy_src_select != 0 ||
      uint32_t(info.copy_dest_coordinate_info.copy_sample_select) != 0) {
    return false;
  }
  uint32_t base = 0, row_length = 0, rows = 0, pitch = 0;
  info.GetCopyEdramTileSpan(base, row_length, rows, pitch);
  if (!rows || !row_length || uint64_t(rows) * std::max(1u, pitch) >= 2048) return false;
  const int16_t index = TruthOwnerOf(base & 2047u);
  if (index < 0) return false;
  TruthImage& t = truth_images_[index];
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < row_length; ++c) {
      if (!t.owned.test((base + r * pitch + c) & 2047u)) return false;
    }
  }
  // RGBA16 images of the k_16_16_16_16 / k_16_16 surfaces, into the same format.
  // (Session 59, gpu_native_truth_direct_resolve_scaled_small: RGBA8 images into k_8 / k_4_4_4_4 - modes 4 / 5, bit 4
  // the red / blue swap; the other modes take no swap.)
  uint32_t mode;
  if (REXCVAR_GET(gpu_native_truth_direct_resolve_scaled_small) && t.kind == 1 &&
      t.format == uint32_t(xenos::ColorRenderTargetFormat::k_8_8_8_8) &&
      (uint32_t(dest.copy_dest_format) == uint32_t(xenos::TextureFormat::k_8) ||
       uint32_t(dest.copy_dest_format) == uint32_t(xenos::TextureFormat::k_4_4_4_4))) {
    mode = (uint32_t(dest.copy_dest_format) == uint32_t(xenos::TextureFormat::k_8) ? 4u : 5u) |
           (dest.copy_dest_swap ? 16u : 0u);
  } else if (dest.copy_dest_swap) {
    return false;
  } else if (t.kind == 2 && t.format == uint32_t(xenos::ColorRenderTargetFormat::k_16_16_16_16) &&
      uint32_t(dest.copy_dest_format) == uint32_t(xenos::TextureFormat::k_16_16_16_16)) {
    mode = 3;
  } else if (t.kind == 2 && t.format == uint32_t(xenos::ColorRenderTargetFormat::k_16_16) &&
             uint32_t(dest.copy_dest_format) == uint32_t(xenos::TextureFormat::k_16_16)) {
    mode = 1;
  } else {
    return false;
  }
  if (!t.initialized || t.base != info.color_original_base || edram.pitch_tiles != t.row_tiles()) return false;
  const uint32_t tile_delta = (uint32_t(edram.base_tiles) - t.base) & 2047u;
  const uint32_t tile_w = t.is_64bpp() ? 40u : 80u;
  const uint32_t src_x =
      scale_x * ((tile_delta % t.row_tiles()) * tile_w + uint32_t(info.coordinate_info.edram_offset_x_div_8) * 8);
  const uint32_t src_y =
      scale_y * ((tile_delta / t.row_tiles()) * 16 + uint32_t(info.coordinate_info.edram_offset_y_div_8) * 8);
  const uint32_t w = scale_x * uint32_t(info.coordinate_info.width_div_8) * 8;
  const uint32_t h = scale_y * uint32_t(info.height_div_8) * 8;
  if (!w || !h || src_x + w > t.width || src_y + h > t.height) return false;  // (the image holds the whole copy)
  if (!EnsureTruthStaging() || !EnsureResolveScaledFullPipeline()) return false;
  const VkDescriptorSet source_set = TruthStoreSet(t, true, truth_staging_);
  if (source_set == VK_NULL_HANDLE) return false;
  EndTruthPass();
  TruthSyncRead(index);  // Session 54 (draw twins)
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = t.image;
  b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  b.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
  // Session 59 (gpu_native_truth_direct_resolve_scaled_full_texture): the scaled texture the copy feeds, written by the
  // same dispatch (texel (hx, hy) of the shader is the texture's when RB_COPY_DEST_BASE is the texture's own base).
  VkDescriptorSet texture_set = VK_NULL_HANDLE;
  void* texture_handle = nullptr;
  // (Session 59: by the texture's own base - an atlas part's copy starts inside it, copy_dest_base_x / y.)
  const uint32_t tex_x0 = scale_x * (info.copy_dest_base_x + uint32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8);
  const uint32_t tex_y0 = scale_y * (info.copy_dest_base_y + uint32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8);
  if (REXCVAR_GET(gpu_native_truth_direct_resolve_scaled_full_texture) && info.copy_dest_texture_base && mode < 4 &&
      (info.copy_dest_texture_base & 0xFFF) == 0 && !command_processor_.NativeResolveReadback() &&
      EnsureResolveScaledFullTexPipelines()) {
    uint32_t texture_w = 0, texture_h = 0;
    texture_set = command_processor_.NativeTextureForResolve(
        info.copy_dest_texture_base >> 12, uint32_t(info.copy_dest_coordinate_info.pitch_aligned_div_32),
        uint32_t(dest.copy_dest_endian), texture_w, texture_h, texture_handle, true,
        mode == 3 ? xenos::TextureFormat::k_16_16_16_16 : xenos::TextureFormat::k_16_16);
    // (Every texel the dispatch writes inside the texture - else the copy alone, and the texture loads as before.)
    const uint32_t tx1 = tex_x0 + w;
    const uint32_t ty1 = tex_y0 + h;
    if (texture_set != VK_NULL_HANDLE && (tx1 > texture_w || ty1 > texture_h)) {
      texture_set = VK_NULL_HANDLE;  // (its barrier, pushed, stays harmless: the texture is reloaded as before)
      texture_handle = nullptr;
    }
  }
  const bool texwrite = texture_set != VK_NULL_HANDLE;
  const VkPipelineLayout full_layout = texwrite ? resolve_scaled_full_tex_layout_ : resolve_scaled_full_layout_;
  command_processor_.BindExternalComputePipeline(texwrite ? resolve_scaled_full_tex_pipelines_[mode == 3 ? 1 : 0]
                                                          : resolve_scaled_full_pipeline_);
  const VkDescriptorSet sets[3] = {source_set, dest_set, texture_set};
  cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, full_layout, 0, texwrite ? 3 : 2, sets, 0, nullptr);
  if (texwrite) {
    texture_written_ = texture_handle;  // (up to date after MarkRangeAsResolved: AfterResolveMarked)
    ++report_.textures_filled;
  }
  const uint32_t gap = edram.fill_half_pixel_offset ? (scale_x / 2) | ((scale_y / 2) << 8) : 0u;
  const uint32_t pc[10] = {src_x,
                           src_y,
                           w,
                           h,
                           mode,
                           gap,
                           uint32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8,
                           uint32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8,
                           uint32_t(info.copy_dest_coordinate_info.pitch_aligned_div_32) << 5,
                           uint32_t(dest.copy_dest_endian) | (scale_x << 4) | (scale_y << 8)};
  const uint32_t pc_tex[12] = {pc[0], pc[1], pc[2], pc[3], pc[4], pc[5], pc[6], pc[7], pc[8], pc[9], tex_x0, tex_y0};
  if (texwrite) {
    cb.CmdVkPushConstants(full_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_tex), pc_tex);
  } else {
    cb.CmdVkPushConstants(full_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
  }
  command_processor_.NativeFlushResolveDestUse();  // (Session 59: a held scaled use - this dispatch writes the buffer)
  command_processor_.SubmitBarriers(true);  // (the scaled resolve buffer's barrier, pushed before the call)
  // (Session 59: the small modes - a dword of 4 / 2 texels an invocation.)
  const uint32_t per = (mode & 15u) == 4u ? 4u : (mode & 15u) == 5u ? 2u : 1u;
  cb.CmdVkDispatch((w / per + 7) / 8, (h + 7) / 8, 1);
  b.srcAccessMask = 0;
  b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  std::swap(b.oldLayout, b.newLayout);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                          nullptr, 0, nullptr, 1, &b);
  ++truth_counts_.direct;
  return true;
}

// Session 52 (gpu_native_truth_batch_resolves): the deferred texture-only resolves, in their order: one barrier taking
// all their source images to the sampled layout (and the textures' pending barriers), the dispatches, one barrier
// back to the attachment layout. Nothing wrote a source image or read a texture since they were deferred (the callers
// of TruthFlushDeferred see to that).
void NativeRenderer::TruthFlushDeferredSlow(uint32_t reason, const DeferConcern& concern) {
  // Session 80 (gpu_native_truth_carry_copies): the carried copies this flush does not concern stay in the batch (in
  // their order); everything else runs as before. Nothing to run and no renamed image waiting for its copy: nothing
  // recorded (the open pass stays open).
  std::vector<DeferredResolve> kept;
  truth_flush_why_.clear();
  if (!concern.all && REXCVAR_GET(gpu_native_truth_carry_copies)) {
    // (A copy reading a spare - its image renamed away - holds that spare, and a copy kept past the frame's end holds its
    // image: only while its texture's carried copies were not found used, at most kSpareHolders of them - else a
    // rename of the image takes another spare each time: run 5473 / 5475 filled the pool.)
    constexpr uint32_t kSpareHolders = 8;
    uint32_t holders = 0;
    // (Why an entry runs, for the flush log: c not carriable, t its texture, i its image, u its texture used lately,
    // h too many holders; 0 kept.)
    const auto concerned = [&](const DeferredResolve& d) -> char {
      if (!TruthCarriable(d)) return 'c';
      if (concern.texture && d.texture == concern.texture) return 't';
      for (const VkImage image : concern.images) {
        if (image != VK_NULL_HANDLE && image == d.image) return 'i';
      }
      bool spare = true;
      for (const TruthImage& t : truth_images_) spare = spare && t.image != d.image;
      if (spare || reason == kDeferFlushFrameEnd) {
        const auto it = carry_used_.find(d.texture);
        if (it != carry_used_.end() && frame_ - it->second < kCarryUsedFrames) return 'u';
        if (holders >= kSpareHolders) return 'h';
        ++holders;
      }
      return 0;
    };
    for (size_t i = 0; i < truth_deferred_.size();) {
      if (const char why = concerned(truth_deferred_[i])) {
        truth_flush_why_ += why;
        ++i;
        continue;
      }
      kept.push_back(truth_deferred_[i]);
      truth_deferred_.erase(truth_deferred_.begin() + ptrdiff_t(i));
    }
    bool renamed = false;
    for (const TruthImage& t : truth_images_) renamed = renamed || t.rename_source != VK_NULL_HANDLE;
    if (truth_deferred_.empty() && !renamed) {
      TruthKeepCarried(reason, kept);
      return;
    }
  }
  // (TruthFlushDeferred counts its own flushes; these count when something runs.)
  if (!concern.all && !truth_deferred_.empty()) {
    ++truth_counts_.defer_flushes[reason < kDeferFlushReasons ? reason : kDeferFlushOther];
  }
  if (segment_log_frames_ > 0) {  // Session 80 (diagnostic): why the batch is flushed, and what it holds
    static constexpr const char* kNames[kDeferFlushReasons] = {
        "other", "same texture", "not batched", "outside rename", "pass into source", "texture sampled", "frame end",
        "submission end", "materialise", "stores/retire", "other resolve", "fill same texture", "fallback"};
    std::string entries;
    for (size_t i = 0; i < truth_deferred_.size(); ++i) {
      const DeferredResolve& d = truth_deferred_[i];
      entries += fmt::format(" 0x{:05X}{}{}{}/{}", d.page, d.depth ? "d" : "", d.mem_length ? "m" : "",
                             d.set_count == 2 ? "f" : d.companion_of ? "s" : "",
                             i < truth_flush_why_.size() ? truth_flush_why_[i] : '-');
      if (i < truth_flush_why_.size() && truth_flush_why_[i] == 'u') {
        const auto it = carry_used_.find(d.texture);
        entries += fmt::format("[{} marked {}]", fmt::ptr(d.texture), it != carry_used_.end() ? it->second : 0);
      }
    }
    std::string carried;
    for (const DeferredResolve& d : kept) carried += fmt::format(" 0x{:05X}{}", d.page, d.crossed ? "c" : "");
    REXGPU_INFO("[native][seg] frame {} truth batch flush: {} ({} entries:{}; kept:{})", frame_,
                kNames[reason < kDeferFlushReasons ? reason : 0], truth_deferred_.size(), entries,
                carried.empty() ? " none" : carried);
  }
  EndTruthPass();
  command_processor_.EndRenderPass();  // (review: a caller may come with the command processor's own pass open)
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  // Session 80 (frame structure): with gpu_ram_stats, the batch's GPU time in the pass log - kind 8 the renames'
  // copies, 9 the barriers into the read layouts (draws = entries, depth = depth sources, colour kind = entries
  // writing guest memory), 11 each dispatch (below), 10 the barrier back (the deferred resolves' timers 6 / 7 hold
  // only their recording, so their work was counted between the passes).
  const auto note_batch = [&]() {
    if (truth_timer_kind_ != 9) return;
    FrameSlot::Segment& s = slots_[SlotOf(frame_)].segments[truth_timer_segment_];
    s.pass_draws = uint32_t(truth_deferred_.size());
    for (const DeferredResolve& d : truth_deferred_) {
      s.pass_depth += d.depth ? 1 : 0;
      s.pass_color_kind += d.mem_length ? 1 : 0;
    }
  };
  TruthTimer(8, true);
  // The images involved and their layouts: dispatch sources (read), and with renames the copies' sources and the
  // renamed images (Session 52, gpu_native_truth_batch_rename). All start in the attachment layout.
  // (Session 56: depth images too - renamed with their pass's colour, gpu_native_truth_rename_depth.)
  struct Involved {
    VkImage image;
    VkImageLayout layout;
    bool read;
    bool depth;
  };
  std::vector<Involved> involved;
  const auto find = [&](VkImage image, bool depth = false) -> Involved& {
    for (Involved& e : involved) {
      if (e.image == image) return e;
    }
    return involved.emplace_back(Involved{image,
                                          depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                                : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                          false, depth});
  };
  for (const DeferredResolve& d : truth_deferred_) find(d.image, d.depth).read = true;
  std::vector<VkImageMemoryBarrier>& barriers = truth_deferred_barriers_;
  const auto transition = [&](Involved& e, VkImageLayout layout) {
    if (e.layout == layout) return;
    const auto access = [](VkImageLayout l) -> VkAccessFlags {
      return l == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL ? VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                                                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
             : l == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                 ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
             : l == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL ? VK_ACCESS_TRANSFER_READ_BIT
             : l == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ? VK_ACCESS_TRANSFER_WRITE_BIT
                                                         : VK_ACCESS_SHADER_READ_BIT;
    };
    VkImageMemoryBarrier& b = barriers.emplace_back();
    b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = e.image;
    b.srcAccessMask = access(e.layout);
    b.dstAccessMask = access(layout);
    b.oldLayout = e.layout;
    b.newLayout = layout;
    b.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(
        e.depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
    e.layout = layout;
  };
  // (Session 56: the fragment-test stages only when a renamed depth image is involved - review.)
  bool any_depth = false;
  for (const TruthImage& t : truth_images_) any_depth = any_depth || (t.rename_source != VK_NULL_HANDLE && t.kind == 3);
  // (Session 79: or a deferred depth copy / fill reading a depth image.)
  for (const DeferredResolve& d : truth_deferred_) any_depth = any_depth || d.depth;
  const VkPipelineStageFlags kStages =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
      VK_PIPELINE_STAGE_TRANSFER_BIT |
      (any_depth ? VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT : 0);
  // 1. The renamed images get their pixels outside their rectangle from the original image.
  struct Copy {
    VkImage source, dest;
    VkImageCopy regions[4];
    uint32_t count;
    bool depth;
  };
  std::vector<Copy> copies;
  for (TruthImage& t : truth_images_) {
    if (t.rename_source == VK_NULL_HANDLE) continue;
    // (Session 80, gpu_native_truth_serve_copies: a slot renamed for a draw sampling its own target, while that draw's
    // other textures are taken - their copies run now; this one waits for the draw, which may write every pixel. Only
    // inside that draw's texture request: any other flush copies it.)
    if (reason == kDeferFlushSampled && serve_requesting_ && serve_rename_.valid &&
        t.rename_source == serve_rename_.source && &t - truth_images_.data() == serve_rename_.slot) {
      continue;
    }
    Copy& c = copies.emplace_back();
    c.source = t.rename_source;
    c.dest = t.image;
    c.count = 0;
    c.depth = t.kind == 3;
    const VkImageAspectFlags aspect =
        c.depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    const int32_t w = int32_t(t.width), h = int32_t(t.height);
    const int32_t x0 = std::clamp(t.rename_rect[0], 0, w), y0 = std::clamp(t.rename_rect[1], 0, h);
    const int32_t x1 = std::clamp(t.rename_rect[2], x0, w), y1 = std::clamp(t.rename_rect[3], y0, h);
    const int32_t bands[4][4] = {{0, 0, w, y0}, {0, y1, w, h}, {0, y0, x0, y1}, {x1, y0, w, y1}};
    for (const auto& r : bands) {
      if (r[2] <= r[0] || r[3] <= r[1]) continue;
      VkImageCopy& region = c.regions[c.count++];
      region = {};
      region.srcSubresource = {aspect, 0, 0, 1};
      region.dstSubresource = {aspect, 0, 0, 1};
      region.srcOffset = region.dstOffset = {r[0], r[1], 0};
      region.extent = {uint32_t(r[2] - r[0]), uint32_t(r[3] - r[1]), 1};
    }
    t.rename_source = VK_NULL_HANDLE;
  }
  barriers.clear();
  for (const Copy& c : copies) {
    if (!c.count) continue;
    transition(find(c.source, c.depth), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    transition(find(c.dest, c.depth), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  }
  if (!barriers.empty()) {
    cb.CmdVkPipelineBarrier(kStages, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                            uint32_t(barriers.size()), barriers.data());
    for (const Copy& c : copies) {
      if (c.count) {
        cb.CmdVkCopyImage(c.source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, c.dest, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          c.count, c.regions);
      }
    }
  }
  // 2. The dispatch sources readable, the others back to their attachment layout; the dispatches.
  TruthTimer(8, false);
  TruthTimer(9, true);
  note_batch();
  barriers.clear();
  for (Involved& e : involved) {
    // (Session 79: a depth source in the depth store set's layout.)
    transition(e, e.read && e.depth     ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                  : e.read              ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                  : e.depth             ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                        : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  }
  // (The textures' barriers to their compute-write layout, held when each resolve was deferred.)
  VkPipelineStageFlags texture_stages = 0;
  for (const DeferredResolve& d : truth_deferred_) {
    if (d.texture_image == VK_NULL_HANDLE) continue;
    VkImageMemoryBarrier& b = barriers.emplace_back();
    b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = d.texture_image;
    b.srcAccessMask = d.texture_src_access;
    b.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.oldLayout = d.texture_old_layout;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.subresourceRange = ui::vulkan::util::InitializeSubresourceRange();
    texture_stages |= d.texture_src_stage;
  }
  if (!barriers.empty()) {
    cb.CmdVkPipelineBarrier(kStages | texture_stages, kStages, 0, 0, nullptr, 0, nullptr, uint32_t(barriers.size()),
                            barriers.data());
  }
  // Session 79 (gpu_native_truth_defer_memory_resolves): the entries writing guest memory - its compute-write use.
  VulkanSharedMemory* const shared_memory = command_processor_.NativeSharedMemory();
  bool wrote_memory = false;
  // (Review fix: the use the shared memory had before - a draw's own, when the batch is flushed inside a draw after
  // the draw recorded it - given back after the dispatches, so its reads / writes are tracked for the next barrier.)
  VulkanSharedMemory::Usage saved_usage = VulkanSharedMemory::Usage::kRead;
  std::pair<uint32_t, uint32_t> saved_range;
  if (shared_memory) {
    saved_usage = shared_memory->last_usage();
    saved_range = shared_memory->last_written_range();
  }
  for (const DeferredResolve& d : truth_deferred_) {
    if (!d.mem_length || !shared_memory) continue;
    // (The batch's writes into memory never overlap - an overlapping one flushes the batch first: one barrier.)
    const std::pair<uint32_t, uint32_t> range(d.mem_start, d.mem_length);
    if (wrote_memory) {
      shared_memory->UseAlsoComputeWrite(range);
    } else {
      shared_memory->Use(VulkanSharedMemory::Usage::kComputeWrite, range);
    }
    wrote_memory = true;
  }
  command_processor_.SubmitBarriers(true);  // (the textures' barriers, pushed when each resolve was deferred)
  // (Session 80: kind 9 ends with the barriers; each dispatch its own kind 11 segment - width / height its threads,
  // depth a depth source, colour kind 1 when it writes guest memory, draws the bytes written there, vs the memory
  // format, ps 1 when it fills a texture.)
  TruthTimer(9, false);
  // Session 80 (gpu_native_truth_drop_dead_copies): this submission's candidates only (the stream's offsets).
  const bool drop_dead_copies = REXCVAR_GET(gpu_native_truth_drop_dead_copies);
  const uint64_t submission = command_processor_.GetCurrentSubmission();
  if (!drop_dead_copies) {
    dead_candidates_.clear();
  } else {
    dead_candidates_.erase(std::remove_if(dead_candidates_.begin(), dead_candidates_.end(),
                                          [&](const DeadCopyCandidate& c) { return c.submission != submission; }),
                           dead_candidates_.end());
  }
  void* last_candidate_texture = nullptr;  // (the previous entry's candidate, for its stencil snapshot)
  for (const DeferredResolve& d : truth_deferred_) {
    TruthTimer(11, true);
    if (truth_timer_kind_ == 11) {
      FrameSlot::Segment& s = slots_[SlotOf(frame_)].segments[truth_timer_segment_];
      s.pass_width = uint16_t(std::min<uint32_t>(d.group_count_x * 8, 65535));
      s.pass_height = uint16_t(std::min<uint32_t>(d.group_count_y * 8, 65535));
      s.pass_depth = d.depth ? 1 : 0;
      s.pass_color_kind = d.mem_length ? 1 : 0;
      s.pass_draws = d.mem_length;
      s.pass_vs = d.mem_format;
      s.pass_ps = d.texture_image != VK_NULL_HANDLE ? 1 : 0;
    }
    command_processor_.BindExternalComputePipeline(d.pipeline);
    // (Session 80: an earlier candidate this entry covers in its texture, untouched since, was read by nothing - its
    // dispatches become no-ops; the bind above stays out of every range: later entries count on it.)
    const bool has_rect = d.drop_rect[2] > d.drop_rect[0] && d.drop_rect[3] > d.drop_rect[1];
    if (drop_dead_copies && d.texture) {
      for (size_t i = 0; i < dead_candidates_.size();) {
        DeadCopyCandidate& c = dead_candidates_[i];
        if (c.texture != d.texture || c.layer != d.layer) {
          ++i;
          continue;
        }
        const bool covered = has_rect && c.rect[0] >= d.drop_rect[0] && c.rect[1] >= d.drop_rect[1] &&
                             c.rect[2] <= d.drop_rect[2] && c.rect[3] <= d.drop_rect[3];
        if (covered && !c.touched) {
          bool ok = true;
          for (int k = 0; k < 2; ++k) {
            if (c.begin[k] != SIZE_MAX) ok = cb.NopRange(c.begin[k], c.end[k]) && ok;
          }
          if (ok) {
            ++truth_counts_.dead_copies[c.depth ? 1 : 0];
          } else {
            ++truth_counts_.dead_nop_failed;
          }
        } else if (covered) {
          ++truth_counts_.dead_kept_touched;
        }
        dead_candidates_.erase(dead_candidates_.begin() + ptrdiff_t(i));  // (this write replaces it either way)
      }
    }
    // (Held-back barriers would be written before the dispatch, inside the range: written now, outside it.)
    if (drop_dead_copies) cb.FlushHeldCommands();
    // (Session 80, gpu_native_truth_carry_copies: a copy carried out of the submission it was recorded in - its
    // destination descriptor that submission's transient one, reused once it completes: a fresh one of the same range.)
    VkDescriptorSet sets[3] = {d.sets[0], d.sets[1], d.sets[2]};
    if (d.submission && d.submission != command_processor_.GetCurrentSubmission() && d.set_count >= 2 &&
        d.dest_range) {
      VulkanSharedMemory* shared_memory = command_processor_.NativeSharedMemory();
      const VkDescriptorSet fresh =
          d.pipeline != resolve_scaled_tex_pipeline_ && shared_memory
              ? command_processor_.AllocateSingleTransientDescriptor(
                    VulkanCommandProcessor::SingleTransientDescriptorLayout::kStorageBufferCompute)
              : VK_NULL_HANDLE;
      if (fresh != VK_NULL_HANDLE) {
        const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
        const VkDescriptorBufferInfo buffer_info = {shared_memory->buffer(), d.dest_offset, d.dest_range};
        VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = fresh;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffer_info;
        vulkan_device->functions().vkUpdateDescriptorSets(vulkan_device->device(), 1, &write, 0, nullptr);
        sets[1] = fresh;
        ++truth_counts_.carry_fresh_sets;
      } else {
        static uint32_t logged = 0;
        if (logged++ < 4) {
          REXGPU_ERROR("[native][truth] a carried copy (page 0x{:05X}) has no destination descriptor of this submission",
                       d.page);
        }
      }
    }
    const size_t range_begin = cb.stream_offset();
    cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, d.layout, 0, d.set_count, sets, 0, nullptr);
    cb.CmdVkPushConstants(d.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, d.pc_size, d.pc);
    cb.CmdVkDispatch(d.group_count_x, d.group_count_y, 1);
    const size_t range_end = cb.stream_offset();
    if (drop_dead_copies) {
      if (d.texture && !d.mem_length && has_rect) {
        // A texture-only copy (no guest memory): a candidate until something uses its texture.
        DeadCopyCandidate& c = dead_candidates_.emplace_back();
        c.texture = d.texture;
        c.layer = d.layer;
        std::copy(d.drop_rect, d.drop_rect + 4, c.rect);
        c.begin[0] = range_begin;
        c.end[0] = range_end;
        c.submission = submission;
        c.depth = d.depth;
        last_candidate_texture = d.texture;
      } else if (d.companion_of && d.companion_of == last_candidate_texture && !dead_candidates_.empty() &&
                 dead_candidates_.back().texture == d.companion_of && dead_candidates_.back().begin[1] == SIZE_MAX) {
        // Its depth fill's stencil snapshot (right after it in the batch): dead with it.
        dead_candidates_.back().begin[1] = range_begin;
        dead_candidates_.back().end[1] = range_end;
        last_candidate_texture = nullptr;
      } else {
        last_candidate_texture = nullptr;
      }
    }
    TruthTimer(11, false);
  }
  if (wrote_memory) shared_memory->Use(saved_usage, saved_range);  // (review fix, above)
  // (Their pages are the texture-only records' again only where a record holds them.)
  if (wrote_memory) {
    for (const DeferredResolve& d : truth_deferred_) {
      if (d.extent_length) shared_memory->SetRangeSkipped(d.extent_start, d.extent_length, false);
    }
    for (const WbRecord& r : wb_records_) {
      if (r.scale_x <= 1 && r.scale_y <= 1) shared_memory->SetRangeSkipped(r.extent_start, r.extent_length, true);
    }
  }
  // 3. Every image back to its attachment layout (the spares too: a rename takes one as it is).
  TruthTimer(10, true);
  barriers.clear();
  for (Involved& e : involved) {
    transition(e, e.depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  }
  if (!barriers.empty()) {
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, kStages, 0, 0, nullptr, 0, nullptr,
                            uint32_t(barriers.size()), barriers.data());
  }
  TruthTimer(10, false);
  for (const DeferredResolve& d : truth_deferred_) {  // Session 80 (gpu_native_truth_carry_copies): carried copies run
    if (!d.pinned) continue;
    ++truth_counts_.carry_run;
    command_processor_.NativeTexturePin(d.texture, false);
    // (Run because its texture is used: carrying it bought nothing - remembered.)
    if ((reason == kDeferFlushSampled || reason == kDeferFlushMaterialize) && concern.texture == d.texture) {
      if (carry_used_.size() > 4096) carry_used_.clear();
      carry_used_[d.texture] = frame_;
      if (segment_log_frames_ > 0 && TruthCarriable(d)) {  // (diagnostic: which carried copies turn out used)
        REXGPU_INFO("[native][seg] frame {} carried copy used: page 0x{:05X} by {}{}", frame_, d.page,
                    reason == kDeferFlushSampled ? "a texture use" : "a write into memory from its texture",
                    d.crossed ? " (kept past a frame's end)" : "");
      }
    }
  }
  truth_deferred_.clear();
  TruthKeepCarried(reason, kept);
  for (TruthBundle& b : truth_spares_) {
    if (b.busy) {
      b.idle_frames = 0;
      b.idle_since = std::chrono::steady_clock::now();
    }
    // (Session 80: a spare a carried copy still reads stays busy - and (review) one a renamed slot still takes its
    // other pixels from, or the draw being prepared samples or its pass reads.)
    b.busy = std::any_of(truth_deferred_.begin(), truth_deferred_.end(),
                         [&](const DeferredResolve& d) { return d.image == b.image; }) ||
             std::any_of(truth_images_.begin(), truth_images_.end(),
                         [&](const TruthImage& t) { return t.rename_source == b.image; }) ||
             std::find(serve_pending_.begin(), serve_pending_.end(), b.image) != serve_pending_.end() ||
             std::any_of(truth_alias_read_owners_.begin(), truth_alias_read_owners_.end(),
                         [&](const std::pair<VkImage, bool>& o) { return o.first == b.image; });
  }
  ++truth_counts_.deferred_batches;
}

// Session 80 (gpu_native_truth_carry_copies): the copies a flush kept, back in the batch - their textures pinned (the
// texture cache keeps a pinned texture: the entry's dispatch writes it later), marked when kept past the frame's end.
void NativeRenderer::TruthKeepCarried(uint32_t reason, std::vector<DeferredResolve>& kept) {
  for (DeferredResolve& d : kept) {
    if (!d.pinned) {
      command_processor_.NativeTexturePin(d.texture, true);
      d.pinned = true;
      ++truth_counts_.carry_kept;
    }
    if (reason == kDeferFlushFrameEnd && !d.crossed) {
      d.crossed = true;
      ++truth_counts_.carry_crossed;
    }
  }
  truth_deferred_.insert(truth_deferred_.begin(), kept.begin(), kept.end());
  kept.clear();
}

// Session 52 (gpu_native_truth_batch_rename): image `color` takes a pooled bundle (or a fresh image) for a pass that
// opens with a full clear while a deferred resolve still reads it; its current bundle waits (busy) for the flush.
bool NativeRenderer::TruthRenameForDeferred(int16_t color, const int32_t rect[4], bool depth_ok) {
  // (Session 80, diagnostic: why a rename is refused, in the segment log.)
  const auto refuse = [&](const char* why) {
    if (segment_log_frames_ > 0) {
      REXGPU_INFO("[native][seg] frame {} truth rename of image {} refused: {}", frame_, color, why);
    }
    return false;
  };
  if (!REXCVAR_GET(gpu_native_truth_batch_rename) || color < 0) return refuse("off");
  TruthImage& t = truth_images_[color];
  if ((t.kind != 1 && !(depth_ok && t.kind == 3)) || t.base == kTruthScratchBase || !t.initialized ||
      t.image == VK_NULL_HANDLE) {
    return refuse("image");
  }
  // A chain of renames: each pass's rectangle covers the one before (the pixels outside it are the original's).
  if (t.rename_source != VK_NULL_HANDLE &&
      (rect[0] > t.rename_rect[0] || rect[1] > t.rename_rect[1] || rect[2] < t.rename_rect[2] ||
       rect[3] < t.rename_rect[3])) {
    return refuse("chain");
  }
  if (truth_spares_.size() >= 64) return refuse("spares");
  EndTruthPass();
  TruthBundle old;
  old.image = t.image;
  old.memory = t.memory;
  old.view = t.view;
  old.sample_view = t.sample_view;
  old.stencil_view = t.stencil_view;
  old.store_sets[0] = t.store_sets[0];
  old.store_sets[1] = t.store_sets[1];
  old.resolve_set = t.resolve_set;
  old.store_set_edram = t.store_set_edram;
  old.kind = t.kind;
  old.format = t.format;
  old.width = t.width;
  old.height = t.height;
  old.busy = true;
  const VkImage source = t.rename_source != VK_NULL_HANDLE ? t.rename_source : t.image;
  size_t spare = truth_spares_.size();
  for (size_t i = 0; i < truth_spares_.size(); ++i) {
    const TruthBundle& b = truth_spares_[i];
    if (!b.busy && b.kind == t.kind && b.format == t.format && b.width == t.width && b.height == t.height) {
      spare = i;
      break;
    }
  }
  if (spare < truth_spares_.size()) {
    const TruthBundle b = truth_spares_[spare];
    truth_spares_.erase(truth_spares_.begin() + spare);
    t.image = b.image;
    t.memory = b.memory;
    t.view = b.view;
    t.sample_view = b.sample_view;
    t.stencil_view = b.stencil_view;
    t.store_sets[0] = b.store_sets[0];
    t.store_sets[1] = b.store_sets[1];
    t.resolve_set = b.resolve_set;
    t.store_set_edram = b.store_set_edram;
  } else {
    const uint32_t height = t.height;
    t.image = VK_NULL_HANDLE;
    t.memory = VK_NULL_HANDLE;
    t.view = t.sample_view = t.stencil_view = VK_NULL_HANDLE;
    t.store_sets[0] = t.store_sets[1] = VK_NULL_HANDLE;
    t.resolve_set = VK_NULL_HANDLE;
    t.store_set_edram = VK_NULL_HANDLE;
    t.initialized = false;
    truth_renaming_ = true;
    const std::bitset<2048> stencil_valid = t.stencil_valid;  // (Session 56 review: a new image resets it)
    const bool ok = EnsureTruthImage(color, height);
    t.stencil_valid = stencil_valid;
    truth_renaming_ = false;
    if (!ok || t.width != old.width || t.height != old.height) {
      refuse(ok ? "size" : "new image");
      if (t.image != VK_NULL_HANDLE) {
        TruthBundle made;
        made.image = t.image;
        made.memory = t.memory;
        made.view = t.view;
        made.sample_view = t.sample_view;
        made.stencil_view = t.stencil_view;
        TruthRetireBundle(made);
      }
      t.image = old.image;
      t.memory = old.memory;
      t.view = old.view;
      t.sample_view = old.sample_view;
      t.stencil_view = old.stencil_view;
      t.store_sets[0] = old.store_sets[0];
      t.store_sets[1] = old.store_sets[1];
      t.resolve_set = old.resolve_set;
      t.store_set_edram = old.store_set_edram;
      t.width = old.width;
      t.height = old.height;
      t.initialized = true;
      return false;
    }
  }
  truth_spares_.push_back(old);
  TruthNoteSpares();  // Session 73
  t.rename_source = source;
  std::copy(rect, rect + 4, t.rename_rect);
  truth_tile_color_ = truth_tile_depth_ = -1;  // (the cached tiles' fast path: the pass starts over)
  ++truth_counts_.renames;
  return true;
}

void NativeRenderer::TruthRetireBundle(TruthBundle& b) {
  TruthRetireServeViews(b.image);  // Session 80 (gpu_native_truth_serve_copies)
  for (auto it = truth_framebuffers_.begin(); it != truth_framebuffers_.end();) {
    // (Session 56 review: a depth bundle's - renamed depth images - too.)
    if (b.view != VK_NULL_HANDLE && (it->color_view == b.view || it->depth_view == b.view)) {
      Retired r;
      r.submission = command_processor_.GetCurrentSubmission();
      r.framebuffer = it->framebuffer;
      retired_.push_back(r);
      it = truth_framebuffers_.erase(it);
    } else {
      ++it;
    }
  }
  Retired r;
  r.submission = command_processor_.GetCurrentSubmission();
  r.images[0] = b.image;
  r.memory[0] = b.memory;
  r.views[0] = b.view;
  r.views[1] = b.sample_view;
  r.views[2] = b.stencil_view;
  if (b.image || b.view || b.sample_view || b.stencil_view) retired_.push_back(r);
  for (VkDescriptorSet& set : b.store_sets) {
    if (set != VK_NULL_HANDLE) truth_retired_sets_.push_back({command_processor_.GetCurrentSubmission(), set});
    set = VK_NULL_HANDLE;
  }
  if (b.resolve_set != VK_NULL_HANDLE) {
    truth_retired_sets_.push_back({command_processor_.GetCurrentSubmission(), b.resolve_set});
    b.resolve_set = VK_NULL_HANDLE;
  }
  b = TruthBundle();
}

void NativeRenderer::TruthFillTexture(const draw_util::ResolveInfo& info) {
  // Session 59 (gpu_native_truth_batch_fills): a fill straight into a texture's storage view joins the deferred batch;
  // the others flush it right before they record (below).
  const bool batch_fills = REXCVAR_GET(gpu_native_truth_batch_fills) && REXCVAR_GET(gpu_native_truth_batch_resolves);
  if (!texture_written_ && !batch_fills) TruthFlushDeferred(kDeferFlushOtherResolve);  // Session 52 (it reads an image)
  const uint32_t fill_mask = uint32_t(REXCVAR_GET(gpu_native_truth_texture_fill));
  if (!truth_ || !replacing_ || texture_written_ || !fill_mask || truth_fill_tried_) return;  // (S49: tried once)
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  // (Session 47: at a resolution scale the copy feeds a scaled texture - the same, in host pixels.)
  const int32_t sx = int32_t(scale_x), sy = int32_t(scale_y);
  const auto refuse = [&](const char* why) {
    ++truth_counts_.fill_refused;
    if (ram_stats_) {  // Session 49: by reason
      auto& list = truth_counts_.refusals;
      auto it = std::find_if(list.begin(), list.end(), [&](const auto& e) { return e.first == why; });
      if (it == list.end()) {
        list.emplace_back(why, 1u);
      } else {
        ++it->second;
      }
    }
    if (segment_log_frames_ > 0) {
      REXGPU_INFO("[native][seg] frame {} truth texture fill refused: page 0x{:05X} format {}: {}", frame_,
                  info.copy_dest_texture_base >> 12, uint32_t(info.copy_dest_info.copy_dest_format), why);
    }
  };
  const bool depth = info.IsCopyingDepth();
  const draw_util::ResolveEdramInfo& edram = depth ? info.depth_edram_info : info.color_edram_info;
  const reg::RB_COPY_DEST_INFO dest = info.copy_dest_info;
  // Session 48: fills into scaled textures cost far more GPU time than the loads they save (A/B 3330): opt-in.
  if ((scale_x != 1 || scale_y != 1) && !REXCVAR_GET(gpu_native_truth_texture_fill_scaled)) {
    return refuse("resolution scale (gpu_native_truth_texture_fill_scaled)");
  }
  if (edram.msaa_samples != xenos::MsaaSamples::k1X) return refuse("msaa");
  if (dest.copy_dest_array) return refuse("array");
  if (dest.copy_dest_exp_bias) return refuse("exp bias");
  if (!depth && info.rb_copy_control.copy_src_select != 0) return refuse("render target 1-3");
  if (uint32_t(info.copy_dest_coordinate_info.copy_sample_select) != 0) return refuse("sample select");
  if (!info.copy_dest_texture_base || (info.copy_dest_texture_base & 0xFFF)) return refuse("texture base");
  // The image holding every copied tile, and the texture format its bits are (a copy of the same bits).
  uint32_t base = 0, row_length = 0, rows = 0, pitch = 0;
  info.GetCopyEdramTileSpan(base, row_length, rows, pitch);
  if (!rows || !row_length || uint64_t(rows) * std::max(1u, pitch) >= 2048) return refuse("span");
  const int16_t index = TruthOwnerOf(base & 2047u);
  if (index < 0) return refuse("no owner");
  TruthImage& t = truth_images_[index];
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < row_length; ++c) {
      if (!t.owned.test((base + r * pitch + c) & 2047u)) return refuse("tiles of several images");
    }
  }
  // The texture format, and how its texels come from the image: a copy of the same bits (pack_mode < 0), or the pack
  // shader (k_8_8_8_8 with the red / blue swap, k_16_16 from an RGBA16 image, depth: kTruthFillComputeShader).
  xenos::TextureFormat format;
  int32_t pack_mode = -1;
  bool swap_scaled = false;
  if (depth) {
    if (t.kind != 3 || !(t.format & 1)) return refuse("depth format");  // (D24FS8: 20e4 depth)
    format = xenos::TextureFormat::k_24_8_FLOAT;
    pack_mode = 2;
  } else if (t.kind == 1 && t.format == uint32_t(xenos::ColorRenderTargetFormat::k_8_8_8_8)) {
    // (Step 2: also into k_8 - red - and k_4_4_4_4 textures: the full copy's 8 / 4-bit unorm packing.)
    const uint32_t swap = dest.copy_dest_swap ? 16u : 0u;
    if (uint32_t(dest.copy_dest_format) == uint32_t(xenos::TextureFormat::k_8)) {
      format = xenos::TextureFormat::k_8;
      pack_mode = int32_t(4 | swap);
    } else if (uint32_t(dest.copy_dest_format) == uint32_t(xenos::TextureFormat::k_4_4_4_4)) {
      format = xenos::TextureFormat::k_4_4_4_4;
      pack_mode = int32_t(5 | swap);
    } else {
      format = xenos::TextureFormat::k_8_8_8_8;
      if (swap) pack_mode = int32_t(0 | swap);
    }
  } else if (t.kind == 2 && t.format == uint32_t(xenos::ColorRenderTargetFormat::k_16_16_16_16)) {
    format = xenos::TextureFormat::k_16_16_16_16;
    pack_mode = 3;  // (the copy converts the 16-bit values: not a copy of the bits)
  } else if (t.kind == 2 && t.format == uint32_t(xenos::ColorRenderTargetFormat::k_16_16)) {
    format = xenos::TextureFormat::k_16_16;
    pack_mode = 1;
  } else {
    return refuse("image format");
  }
  if (!depth && dest.copy_dest_swap && t.kind != 1) return refuse("red / blue swap");
  {
    const uint32_t kind_bit = depth                                            ? 1u
                              : format == xenos::TextureFormat::k_8_8_8_8     ? (dest.copy_dest_swap ? 4u : 2u)
                              : format == xenos::TextureFormat::k_16_16_16_16 ? 8u
                              : format == xenos::TextureFormat::k_16_16       ? 16u
                                                                              : 32u;
    if (!(fill_mask & kind_bit)) return refuse("gpu_native_truth_texture_fill");
    // (Session 52: at a scale, also the scaled kinds.)
    if ((scale_x != 1 || scale_y != 1) &&
        !(uint32_t(REXCVAR_GET(gpu_native_truth_texture_fill_scaled_kinds)) & kind_bit)) {
      return refuse("gpu_native_truth_texture_fill_scaled_kinds");
    }
    // (At a resolution scale the red / blue swapped k_8_8_8_8 copies are not filled: their fills differ from the reload
    // there - Brock's entrance, parity 3226 / 3234; why is unconfirmed - exact at 1x.)
    swap_scaled = kind_bit == 4u && (scale_x != 1 || scale_y != 1);
  }
  if (uint32_t(dest.copy_dest_format) != uint32_t(format)) return refuse("format change");
  if (pack_mode >= 0 && truth_fill_pipeline_ == VK_NULL_HANDLE) return refuse("no pack shader");
  if (!t.initialized || t.base != (depth ? info.depth_original_base : info.color_original_base) ||
      edram.pitch_tiles != t.row_tiles()) {
    return refuse("image");
  }
  // Where the copy reads in the image (as TruthResolveDirect) and writes in the texture.
  const uint32_t tile_delta = (uint32_t(edram.base_tiles) - t.base) & 2047u;
  const uint32_t tile_w = t.is_64bpp() ? 40u : 80u;
  const int32_t src_x =
      sx * int32_t((tile_delta % t.row_tiles()) * tile_w + uint32_t(info.coordinate_info.edram_offset_x_div_8) * 8);
  const int32_t src_y =
      sy * int32_t((tile_delta / t.row_tiles()) * 16 + uint32_t(info.coordinate_info.edram_offset_y_div_8) * 8);
  // (In the texture whose base RB_COPY_DEST_BASE is: copy_dest_base may point into it.)
  const int32_t dst_x =
      sx * int32_t(info.copy_dest_base_x + uint32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8);
  const int32_t dst_y =
      sy * int32_t(info.copy_dest_base_y + uint32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8);
  const int32_t copy_w = sx * int32_t(info.coordinate_info.width_div_8) * 8;
  const int32_t copy_h = sy * int32_t(info.height_div_8) * 8;
  VkImage texture = VK_NULL_HANDLE;
  uint32_t texture_w = 0, texture_h = 0;
  uint32_t level = 0;
  bool has_mips = false;
  void* handle = nullptr;
  const char* why = "";
  if (pack_mode >= 0 && !EnsureTruthStaging()) return refuse("staging");
  const VkDescriptorSet pack_set = pack_mode >= 0 ? TruthStoreSet(t, true, truth_staging_) : VK_NULL_HANDLE;
  if (pack_mode >= 0 && pack_set == VK_NULL_HANDLE) return refuse("pack set");
  // Session 52 (gpu_native_truth_fill_direct): straight into the texture's storage view where it has one.
  const uint32_t pack_kind = pack_mode >= 0 ? uint32_t(pack_mode) & 15u : 15u;
  // (Session 59, gpu_native_truth_fill_direct_small: k_8 / k_4_4_4_4 into an R8_UINT / R16_UINT view - 1 / 2 bytes.)
  const bool small_kind = pack_kind == 4 || pack_kind == 5;
  const uint32_t storage_bytes =
      REXCVAR_GET(gpu_native_truth_fill_direct) && (pack_kind == 0 || pack_kind == 1 || pack_kind == 3) &&
              EnsureTruthFillTexPipelines()
          ? (pack_kind == 3 ? 8u : 4u)
      : REXCVAR_GET(gpu_native_truth_fill_direct) && REXCVAR_GET(gpu_native_truth_fill_direct_small) && small_kind &&
              EnsureTruthFillTexSmallPipelines()
          ? (pack_kind == 4 ? 1u : 2u)
      // (Session 59, gpu_native_truth_fill_direct_depth: the 32-bit float bits through the R32_UINT view - TEXOUT 1.)
      : REXCVAR_GET(gpu_native_truth_fill_direct) && REXCVAR_GET(gpu_native_truth_fill_direct_depth) && depth &&
              pack_kind == 2 && EnsureTruthFillTexPipelines()
          ? 4u
          : 0u;
  const uint32_t tex_variant = pack_kind == 3 ? 1u : pack_kind == 4 ? 2u : pack_kind == 5 ? 3u : 0u;
  VkDescriptorSet storage_set = VK_NULL_HANDLE;
  if (REXCVAR_GET(gpu_native_truth_cover_outdated) && copy_w > 0 && copy_h > 0) {
    command_processor_.NativeSetTextureCover(uint32_t(std::max(dst_x, 0)) | (dst_x < 0 ? 1u : 0u),
                                             uint32_t(std::max(dst_y, 0)) | (dst_y < 0 ? 1u : 0u), uint32_t(copy_w),
                                             uint32_t(copy_h));
  }
  // (Session 59: a storage-view fill's texture barrier is held - recorded with the batch if the fill is deferred.)
  // (Session 79, gpu_native_truth_defer_memory_resolves: every fill's barrier held, so a fill the texture cache
  // refuses - no texture over the page - flushes nothing; one that is recorded flushes the batch first, as before. A
  // copy-path texture's barrier is pushed at once by the texture cache: it joins the flush's barriers, before the
  // batch's dispatches - an entry writing that texture is flushed by the texture cache before it.)
  const bool hold_refused = REXCVAR_GET(gpu_native_truth_defer_memory_resolves);
  // (Session 79, gpu_native_truth_defer_depth_resolves: a depth fill into its storage view deferred too.)
  const bool defer_depth = REXCVAR_GET(gpu_native_truth_defer_depth_resolves);
  const bool hold = batch_fills && !texture_written_ && ((storage_bytes && (!depth || defer_depth)) || hold_refused);
  // (Review: a fill that cannot be deferred flushes the batch before its texture's barrier, as without batching.)
  if (batch_fills && !hold && !texture_written_) TruthFlushDeferred(kDeferFlushOtherResolve);
  if (hold) command_processor_.NativeHoldTextureBarrier(true);
  const bool got = command_processor_.NativeTextureForCopy(
      info.copy_dest_texture_base >> 12, uint32_t(info.copy_dest_coordinate_info.pitch_aligned_div_32),
      uint32_t(dest.copy_dest_endian), format, texture, texture_w, texture_h, level, has_mips, handle, why,
      storage_bytes, &storage_set);
  // (Session 59: not deferred after all - the held barrier pushed now, recorded as before.)
  // (Review fix, Session 79, gpu_native_truth_fill_rename_check: a renamed image holds only its rename rectangle until
  // the flush's copy - a fill reading outside it is not deferred; the flush just below runs that copy first.)
  const bool fill_outside = REXCVAR_GET(gpu_native_truth_fill_rename_check) && t.rename_source != VK_NULL_HANDLE &&
                            (src_x < t.rename_rect[0] || src_y < t.rename_rect[1] ||
                             src_x + copy_w > t.rename_rect[2] || src_y + copy_h > t.rename_rect[3]);
  bool deferring =
      hold && got && storage_set != VK_NULL_HANDLE && storage_bytes && (!depth || defer_depth) && !fill_outside;
  if (hold && !deferring && (got || !hold_refused)) {
    // (review: as above, before the barrier is pushed.) (Session 80, gpu_native_truth_carry_copies: the copies writing
    // the texture it fills - it only reads its image.)
    TruthFlushDeferredTexture(kDeferFlushOtherResolve, got ? handle : nullptr);
  }
  if (!deferring) command_processor_.NativeHoldTextureBarrier(false);
  const auto release = [&]() {
    if (deferring) command_processor_.NativeHoldTextureBarrier(false);
    deferring = false;
  };
  if (!got) {
    return refuse(why);
  }
  TruthSyncRead(index);  // Session 54 (draw twins)
  // Session 49 (gpu_native_truth_texture_fill_lazy): at a resolution scale, no fill into a texture nothing sampled since
  // it was last written - it is loaded from memory if something samples it later (the fills-off way), and filled again
  // after that.
  if ((scale_x != 1 || scale_y != 1) && REXCVAR_GET(gpu_native_truth_texture_fill_lazy) &&
      !command_processor_.NativeLastCopyTextureSampled()) {
    return release(), refuse("not sampled since its last fill (gpu_native_truth_texture_fill_lazy)");
  }
  const int32_t w = std::min({copy_w, int32_t(texture_w) - dst_x, int32_t(t.width) - src_x});
  const int32_t h = std::min({copy_h, int32_t(texture_h) - dst_y, int32_t(t.height) - src_y});
  // (The texture's barrier was pushed: it is a copy destination now, whatever is recorded below.)
  command_processor_.SubmitBarriers(true);
  if (w <= 0 || h <= 0 || src_x < 0 || src_y < 0) return release(), refuse("rectangle");
  // (Diagnostic narrowing, Session 47: swapped k_8_8_8_8 at a resolution scale only into textures of 128 guest texels
  // or more - smaller ones differed from the reload in Brock's entrance, parity 3226 / 3234; why is unconfirmed.)
  if (swap_scaled && REXCVAR_GET(gpu_native_truth_texture_fill_small_swap_scaled) == 0 &&
      (texture_w / scale_x < 128 || texture_h / scale_y < 128)) {
    return release(), refuse("small swapped texture at a resolution scale");
  }
  // (Review: the image must hold all the copy wrote into the texture - a fill cut short by the image's own size would
  // leave texels the memory has and the texture not.)
  if (w < std::min(copy_w, int32_t(texture_w) - dst_x) || h < std::min(copy_h, int32_t(texture_h) - dst_y)) {
    return release(), refuse("image smaller than the copy");
  }
  // (Review: in a texture with mip levels, a copy running past its level's size wrote into the next level's memory -
  // that level would be marked up to date unfilled.)
  if (has_mips &&
      (copy_w > int32_t(texture_w) - dst_x || copy_h > int32_t(texture_h) - dst_y)) {
    return release(), refuse("copy past the level");
  }
  // (k_8 / k_4_4_4_4: 4 / 2 texels a dword in the pack shader.)
  const uint32_t per = (pack_mode & 15) == 4 ? 4u : (pack_mode & 15) == 5 ? 2u : 1u;
  if (pack_mode >= 0 && uint32_t(w) % per) return release(), refuse("width");
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const bool packed = pack_mode >= 0;
  VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = t.image;
  b.srcAccessMask = depth ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  b.dstAccessMask = packed ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_TRANSFER_READ_BIT;
  b.oldLayout = depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  b.newLayout = !packed  ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                : depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                        : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  b.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(
      depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
  constexpr VkPipelineStageFlags kAttachmentStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                                     VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                                     VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  const VkPipelineStageFlags image_stage = packed ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT;
  if (packed) ++truth_counts_.fill_packed;  // Session 52
  // Session 59 (gpu_native_truth_batch_fills): deferred with the texture-only resolves (one barrier pair for the batch).
  if (deferring && packed) {
    // (A deferred write into the same texture whose texels may overlap these - a resolve's, or a fill's rectangle that
    // meets this one - goes first, with its barrier.)
    bool same_texture = false;
    for (const DeferredResolve& d : truth_deferred_) {
      if (d.texture != handle) continue;
      same_texture = same_texture || d.texture_rect[2] <= d.texture_rect[0] ||
                     (d.texture_rect[0] < dst_x + w && dst_x < d.texture_rect[2] && d.texture_rect[1] < dst_y + h &&
                      dst_y < d.texture_rect[3]);
    }
    if (same_texture) {
      // (Review: the batch first, with this fill's barrier still held; then the barrier, recorded before this fill's
      // dispatch - after the earlier write into the texture, not before it.)
      TruthFlushDeferredTexture(kDeferFlushFillSameTexture, handle);  // (Session 80: the copies writing it)
      release();
      command_processor_.SubmitBarriers(true);
    }
  }
  if (deferring && packed) {
    ++truth_counts_.fill_direct;
    ++truth_counts_.fills_deferred;
    DeferredResolve& d = truth_deferred_.emplace_back();
    d.texture_image = VK_NULL_HANDLE;
    d.page = info.copy_dest_base >> 12;  // Session 80 (the flush log)
    command_processor_.NativeTakeHeldTextureBarrier(d.texture_image, d.texture_src_stage, d.texture_src_access,
                                                    d.texture_old_layout);
    command_processor_.NativeHoldTextureBarrier(false);
    d.pipeline = truth_fill_tex_pipelines_[tex_variant];
    d.layout = truth_fill_tex_layout_;
    d.sets[0] = pack_set;
    d.sets[1] = storage_set;
    d.sets[2] = VK_NULL_HANDLE;
    d.set_count = 2;
    const uint32_t gap = edram.fill_half_pixel_offset ? (scale_x / 2) | ((scale_y / 2) << 8) : 0u;
    const uint32_t pc[10] = {uint32_t(src_x), uint32_t(src_y), uint32_t(w), uint32_t(h), uint32_t(pack_mode), gap,
                             uint32_t(dst_x), uint32_t(dst_y)};
    std::fill(std::begin(d.pc), std::end(d.pc), 0u);
    std::memcpy(d.pc, pc, sizeof(pc));
    d.pc_size = sizeof(pc);
    d.group_count_x = (uint32_t(w) + 7) / 8;
    d.group_count_y = (uint32_t(h) + 7) / 8;
    d.image = t.image;
    d.index = index;
    d.depth = depth;  // Session 79 (gpu_native_truth_defer_depth_resolves)
    d.texture = handle;
    d.texture_rect[0] = dst_x, d.texture_rect[1] = dst_y, d.texture_rect[2] = dst_x + w, d.texture_rect[3] = dst_y + h;
    std::copy(d.texture_rect, d.texture_rect + 4, d.drop_rect);  // Session 66 (G1.1)
    // (A texture an earlier entry of the batch already takes to its compute-write layout: no second barrier for it in
    // the same command - this one's would be GENERAL -> GENERAL; disjoint texels need no ordering.)
    for (size_t k = 0; k + 1 < truth_deferred_.size() && d.texture_image != VK_NULL_HANDLE; ++k) {
      if (truth_deferred_[k].texture_image == d.texture_image) d.texture_image = VK_NULL_HANDLE;
    }
  } else {
  release();  // (Session 59: a held barrier when the fill turned out not packed - pushed and recorded as before)
  command_processor_.SubmitBarriers(true);
  if (packed && storage_set != VK_NULL_HANDLE) {
    ++truth_counts_.fill_direct;
    // Session 52 (gpu_native_truth_fill_direct): the image read, the texture written - no staging buffer, no copy.
    cb.CmdVkPipelineBarrier(kAttachmentStages, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    command_processor_.BindExternalComputePipeline(truth_fill_tex_pipelines_[tex_variant]);
    const VkDescriptorSet sets[2] = {pack_set, storage_set};
    cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, truth_fill_tex_layout_, 0, 2, sets, 0, nullptr);
    const uint32_t gap = edram.fill_half_pixel_offset ? (scale_x / 2) | ((scale_y / 2) << 8) : 0u;
    const uint32_t pc[10] = {uint32_t(src_x), uint32_t(src_y), uint32_t(w), uint32_t(h), uint32_t(pack_mode), gap,
                             uint32_t(dst_x), uint32_t(dst_y)};
    cb.CmdVkPushConstants(truth_fill_tex_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    cb.CmdVkDispatch((uint32_t(w) + 7) / 8, (uint32_t(h) + 7) / 8, 1);
  } else if (packed) {
    // The staging buffer was just read by the copy's compute shader: written by the pack shader after it.
    VkMemoryBarrier staging = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT};
    // (Review: every stage that reads the staging buffer - alias loads read it in fragment shaders.)
    staging.srcAccessMask |= VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    cb.CmdVkPipelineBarrier(kAttachmentStages | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &staging, 0, nullptr, 1, &b);
    command_processor_.BindExternalComputePipeline(truth_fill_pipeline_);
    cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, truth_store_layout_, 0, 1, &pack_set, 0, nullptr);
    // (At a resolution scale the copy may fill the half-pixel offset gap - half a guest pixel at the left and the top -
    // from the first covered column / row: edram fill_half_pixel_offset.)
    // (Measured, Session 47: every pack mode needs it, the swapped k_8_8_8_8 too - without it at 4x parity 3226 / 3244
    // differs, with it 3226 / 3240 is identical; the plain k_8_8_8_8 image copy is exact without it.)
    const uint32_t gap = edram.fill_half_pixel_offset ? (scale_x / 2) | ((scale_y / 2) << 8) : 0u;
    const uint32_t pc[10] = {uint32_t(src_x), uint32_t(src_y), uint32_t(w), uint32_t(h), uint32_t(pack_mode), gap};
    cb.CmdVkPushConstants(truth_store_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    cb.CmdVkDispatch((uint32_t(w) / per + 7) / 8, (uint32_t(h) + 7) / 8, 1);
    VkMemoryBarrier written = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                               VK_ACCESS_TRANSFER_READ_BIT};
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &written, 0,
                            nullptr, 0, nullptr);
    VkBufferImageCopy to_texture = {};
    to_texture.bufferOffset = 0;
    to_texture.bufferRowLength = uint32_t(w);
    to_texture.bufferImageHeight = uint32_t(h);
    to_texture.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
    to_texture.imageOffset = {dst_x, dst_y, 0};
    to_texture.imageExtent = {uint32_t(w), uint32_t(h), 1};
    cb.CmdVkCopyBufferToImage(truth_staging_, texture, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &to_texture);
    // The staging buffer is the resolves' source again after this.
    VkMemoryBarrier reread = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_READ_BIT,
                              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &reread, 0,
                            nullptr, 0, nullptr);
  } else {
    cb.CmdVkPipelineBarrier(kAttachmentStages, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkImageCopy region = {};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.srcOffset = {src_x, src_y, 0};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
    region.dstOffset = {dst_x, dst_y, 0};
    region.extent = {uint32_t(w), uint32_t(h), 1};
    cb.CmdVkCopyImage(t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, texture, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                      &region);
  }
  b.srcAccessMask = 0;
  b.dstAccessMask = depth ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                          : VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  std::swap(b.oldLayout, b.newLayout);
  cb.CmdVkPipelineBarrier(image_stage, kAttachmentStages, 0, 0, nullptr, 0, nullptr, 1, &b);
  }  // (Session 59: not deferred)
  texture_written_ = handle;  // (up to date after MarkRangeAsResolved: AfterResolveMarked)
  ++truth_counts_.filled;
  ++report_.textures_filled;
  // Session 49 (texture-only resolves): what this fill wrote.
  truth_fill_done_ = {handle, level, texture_w, texture_h, dst_x, dst_y, w, h, copy_w, copy_h, has_mips, depth};
  if (segment_log_frames_ > 0) {
    REXGPU_INFO("[native][seg] frame {} truth texture fill: page 0x{:05X} format {} level {} {}x{} at {},{} from "
                "image {} at {},{} (pack mode {}, storage view {} B wanted, {}; readable {} B, mips {})",
                frame_, info.copy_dest_texture_base >> 12, uint32_t(format), level, w, h, dst_x, dst_y, index, src_x,
                src_y, pack_mode, storage_bytes, storage_set != VK_NULL_HANDLE ? "used" : "not used",
                command_processor_.NativeTextureReadable(handle), has_mips);
  }
}

void NativeRenderer::TruthTimer(uint32_t kind, bool begin) {
  if (!ram_stats_ || !truth_ || !replacing_) return;
  FrameSlot& slot = slots_[SlotOf(frame_)];
  const uint32_t query_base = SlotOf(frame_) * kQueriesPerFrame;
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  if (begin) {
    if (truth_timer_kind_) return;  // (not nested)
    EndTruthPass();                 // (timestamps outside our render pass; the copy ends it anyway)
    if (!query_pool_ || !slot_queries_ok_ || slot.queries + 2 > QueryLimit()) {
      ++truth_counts_.untimed;
      return;
    }
    FrameSlot::Segment& segment = slot.segments.emplace_back();
    segment.kind = kind;
    segment.native_begin = slot.queries++;
    cb.CmdVkWriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, query_base + segment.native_begin);
    truth_timer_kind_ = kind;
    truth_timer_segment_ = uint32_t(slot.segments.size() - 1);
    return;
  }
  if (truth_timer_kind_ != kind) return;
  truth_timer_kind_ = 0;
  if (truth_timer_segment_ >= slot.segments.size()) return;
  FrameSlot::Segment& segment = slot.segments[truth_timer_segment_];
  if (slot.queries >= QueryLimit()) {  // (left without an end: not counted)
    ++truth_counts_.untimed;
    return;
  }
  segment.native_end = slot.queries++;
  cb.CmdVkWriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, query_base + segment.native_end);
}

void NativeRenderer::TruthStampSubmission() {
  // (Session 87, step 3 diagnosis: run 5791 printed no "submissions; the GPU waited" line in any frame - why. Counted by reason while
  // the pass log is on, logged at the pass log's frames.)
  if (pass_log_frames_ > 0) {
    ++stamp_calls_;
    if (!ram_stats_ && pass_log_frames_ <= 0) ++stamp_skipped_[0];
    else if (!truth_) ++stamp_skipped_[1];
    else if (!replacing_) ++stamp_skipped_[2];
    else if (!query_pool_) ++stamp_skipped_[3];
    else if (!slot_queries_ok_) ++stamp_skipped_[4];
    else ++stamp_made_;
  }
  // (Session 87, step 3: the pass log alone is enough to stamp - the idle time between submissions is part of reading it; it
  // was gated on gpu_ram_stats, which the pass-log runs never set, so no run since Session 80 could tell the GPU waiting for a
  // submission from untimed GPU work.)
  if ((!ram_stats_ && pass_log_frames_ <= 0) || !truth_ || !replacing_ || !query_pool_ || !slot_queries_ok_) return;
  FrameSlot& slot = slots_[SlotOf(frame_)];
  if (slot.queries + 1 > QueryLimit()) return;
  const uint32_t q = slot.queries++;
  command_processor_.deferred_command_buffer().CmdVkWriteTimestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_,
                                                                   SlotOf(frame_) * kQueriesPerFrame + q);
  slot.submission_stamps.push_back(q);
}

void NativeRenderer::LogTruthPasses(uint64_t frame, const FrameSlot& slot, const std::vector<uint64_t>& stamps,
                                    double ms) {
  // Session 72: the frame's segments in recording order - each pass's own time, and the GPU work between the end of
  // the previous timed segment and its start (resolves, copies, stores and barriers recorded outside the passes).
  --pass_log_frames_;
  uint64_t first = UINT64_MAX, last = 0, prev_end = UINT64_MAX;
  double passes_ms = 0.0, between_ms = 0.0, idle_ms = 0.0;
  uint32_t n = 0, idle_gaps = 0;
  // Session 80 (frame structure): the submissions' first commands (TruthStampSubmission). One that started after the
  // previous segment had ended means the GPU sat idle that long in the gap, waiting for the CPU to submit.
  std::vector<uint64_t> submits;
  for (uint32_t q : slot.submission_stamps) {
    if (q < stamps.size()) submits.push_back(stamps[q]);
  }
  for (const FrameSlot::Segment& s : slot.segments) {
    if (s.native_begin == UINT32_MAX || s.native_end == UINT32_MAX || s.native_begin >= stamps.size() ||
        s.native_end >= stamps.size()) {
      continue;
    }
    const uint64_t b = stamps[s.native_begin], e = stamps[s.native_end];
    if (e < b) continue;
    const double own = double(e - b) * ms;
    const double before = prev_end != UINT64_MAX && b >= prev_end ? double(b - prev_end) * ms : 0.0;
    if (before > 0.0) {
      // The latest submission start in the gap: the GPU had nothing to run from the previous end until then.
      uint64_t start = 0;
      for (uint64_t t : submits) {
        if (t > prev_end && t <= b) start = std::max(start, t);
      }
      if (start) {
        const double idle = double(start - prev_end) * ms;
        idle_ms += idle;
        ++idle_gaps;
        REXGPU_INFO("[native][pass] frame {} submission started {:.3f} ms into the {:.3f} ms gap before #{}", frame,
                    idle, before, n);
      }
    }
    first = std::min(first, b);
    last = std::max(last, e);
    prev_end = e;
    passes_ms += own;
    between_ms += before;
    REXGPU_INFO("[native][pass] frame {} #{} kind {} {}x{} pitch {} colour kind {} depth {} draws {} vs {:016X} ps "
                "{:016X}: {:.3f} ms (before it {:.3f} ms){}{}",
                frame, n++, s.kind, s.pass_width, s.pass_height, s.pass_pitch, s.pass_color_kind, s.pass_depth,
                s.pass_draws, s.pass_vs, s.pass_ps, own, before, s.label ? " owned " : "", s.label ? s.label : "");
  }
  REXGPU_INFO("[native][pass] frame {}: {} timed segments, {:.3f} ms in them, {:.3f} ms between them, {:.3f} ms first "
              "to last",
              frame, n, passes_ms, between_ms, last > first ? double(last - first) * ms : 0.0);
  REXGPU_INFO("[native][pass] frame {}: ticks first {} last {}", frame, first, last);  // (Session 95 r5: against [native][gputs])
  if (!submits.empty()) {
    REXGPU_INFO("[native][pass] frame {}: {} submissions; the GPU waited for one in {} gaps, {:.3f} ms of the {:.3f} "
                "ms between segments",
                frame, submits.size(), idle_gaps, idle_ms, between_ms);
  }
  REXGPU_INFO("[native][pass] frame {}: submission stamps in the slot {} (submissions opened while the pass log is on: {}, stamped {}, "
              "skipped: ram_stats {} truth {} replacing {} pool {} slot_ok {})",
              frame, slot.submission_stamps.size(), stamp_calls_, stamp_made_, stamp_skipped_[0], stamp_skipped_[1],
              stamp_skipped_[2], stamp_skipped_[3], stamp_skipped_[4]);
}

void NativeRenderer::TruthNoteWriteBack(bool direct, uint32_t bytes) {
  if (!ram_stats_ || !truth_ || !replacing_ || truth_wb_skipped_) return;  // (Session 49: texture-only: none)
  if (direct) {
    ++truth_counts_.wb_direct;
    truth_counts_.wb_direct_bytes += bytes;
  } else {
    ++truth_counts_.wb_staged;
    truth_counts_.wb_staged_bytes += bytes;
  }
}

// Session 49 (gpu_native_truth_skip_writeback): texture-only resolves - see the header.
void NativeRenderer::TruthMaterializeRecord(const WbRecord& r) {
  // Session 52 (it reads a texture a deferred resolve may write). (Session 80: the copies writing that texture.)
  TruthFlushDeferredTexture(kDeferFlushMaterialize, r.texture);
  TruthDeadCopyTouched(r.texture);  // Session 80 (gpu_native_truth_drop_dead_copies): it reads the texture

  command_processor_.NativeHoldTextureBarrier(false);  // (Session 52: a held texture barrier goes in first)
  VulkanSharedMemory* shared_memory = command_processor_.NativeSharedMemory();
  const bool scaled = r.scale_x > 1 || r.scale_y > 1;  // Session 56 (gpu_native_truth_skip_writeback_scaled)
  const VkBuffer dest_buffer = scaled ? command_processor_.NativeScaledResolveBuffer()
                                      : shared_memory ? shared_memory->buffer() : VK_NULL_HANDLE;
  uint32_t old_usage = 0;
  // (Session 80: a depth record through its texture's R32_UINT view of the floats, gpu_native_truth_skip_writeback_depth.)
  const VkDescriptorSet texture_set =
      (r.depth ? EnsureDepthWritebackObjects() : scaled ? EnsureMaterializeScaledPipeline() : EnsureMaterializePipeline())
          ? command_processor_.NativeTextureForRead(r.texture, old_usage, scaled, r.layer, r.depth)  // (S79: a face's)
          : VK_NULL_HANDLE;
  const VkDescriptorSet dest_set = texture_set != VK_NULL_HANDLE
                                       ? command_processor_.AllocateSingleTransientDescriptor(
                                             VulkanCommandProcessor::SingleTransientDescriptorLayout::kStorageBufferCompute)
                                       : VK_NULL_HANDLE;
  if (dest_buffer == VK_NULL_HANDLE || dest_set == VK_NULL_HANDLE) {
    // (Never expected: the texture was readable when recorded and is pinned.)
    static uint32_t logged = 0;
    if (logged++ < 8) {
      REXGPU_ERROR("[native][wb] could not write {:08X}+{:X} from its texture: the GPU copy keeps older data",
                   r.extent_start, r.extent_length);
    }
    return;
  }
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  VkDescriptorBufferInfo buffer_info = {dest_buffer, r.dest_offset, r.dest_range};
  VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = dest_set;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo = &buffer_info;
  vulkan_device->functions().vkUpdateDescriptorSets(vulkan_device->device(), 1, &write, 0, nullptr);
  EndTruthPass();
  if (scaled) {
    command_processor_.NativeUseScaledResolveBufferForWrite(r.scaled_use_start, r.scaled_use_length);
  } else {
    shared_memory->Use(VulkanSharedMemory::Usage::kComputeWrite,
                       std::pair<uint32_t, uint32_t>(r.extent_start, r.extent_length));
  }
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  if (r.depth) {
    // Session 80 (gpu_native_truth_skip_writeback_depth): kNativeMaterializeDepthShader - set 0 the record's stencil
    // slot (without one, any storage set: unread), its snapshot's write (the batch, flushed above) made visible.
    const VkDescriptorSet stencil_set = r.stencil_slot >= 0 ? StencilSlotSet(r.stencil_slot) : dest_set;
    if (stencil_set == VK_NULL_HANDLE) {
      REXGPU_ERROR("[native][wb] no stencil slot set for depth {:08X}+{:X}: the GPU copy keeps older data",
                   r.extent_start, r.extent_length);
      command_processor_.NativeTextureEndRead(r.texture, old_usage);
      return;
    }
    VkMemoryBarrier snapshot_barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                                        VK_ACCESS_SHADER_READ_BIT};
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                            &snapshot_barrier, 0, nullptr, 0, nullptr);
    command_processor_.BindExternalComputePipeline(materialize_depth_pipeline_);
    const VkDescriptorSet depth_sets[3] = {stencil_set, dest_set, texture_set};
    cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, materialize_depth_layout_, 0, 3, depth_sets, 0,
                               nullptr);
    const uint32_t dw = r.x1 - r.x0, dh = r.y1 - r.y0;
    const uint32_t pc[12] = {r.x0,     r.y0,     dw,
                             dh,       r.pitch * 32, r.dest_base,
                             r.endian, r.tex_x0, r.tex_y0,
                             r.stencil_slot >= 0 ? 256u : (r.stencil_const & 255u), 0u, r.stencil_pitch};
    cb.CmdVkPushConstants(materialize_depth_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    command_processor_.SubmitBarriers(true);
    cb.CmdVkDispatch((dw + 7) / 8, (dh + 7) / 8, 1);
    command_processor_.NativeTextureEndRead(r.texture, old_usage);
    ++wb_counts_.materialized;
    ++wb_counts_.depth_materialized;
    wb_counts_.materialized_bytes += r.extent_length;
    return;
  }
  command_processor_.BindExternalComputePipeline(materialize_pipeline_[scaled ? 2 : r.bpb64 ? 1 : 0]);
  const VkDescriptorSet sets[2] = {dest_set, texture_set};
  cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, resolve_tex_layout_, 1, 2, sets, 0, nullptr);
  const uint32_t w = r.x1 - r.x0, h = r.y1 - r.y0;
  if (scaled) {
    const uint32_t pc[8] = {r.x0, r.y0, r.x1, r.y1, r.pitch << 5, r.endian, r.scale_x, r.scale_y};
    cb.CmdVkPushConstants(resolve_tex_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
  } else {
    const uint32_t pc[9] = {r.x0, r.y0, w, h, r.pitch * 32, r.dest_base, r.endian, r.tex_x0, r.tex_y0};
    cb.CmdVkPushConstants(resolve_tex_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
  }
  command_processor_.SubmitBarriers(true);
  if (scaled) {
    cb.CmdVkDispatch((w / 4 + 7) / 8, (h + 7) / 8, 1);
  } else {
    cb.CmdVkDispatch((w + 7) / 8, (h + 7) / 8, 1);
  }
  command_processor_.NativeTextureEndRead(r.texture, old_usage);  // (review: back to the layout a draw may sample)
  ++wb_counts_.materialized;
  wb_counts_.materialized_bytes += r.extent_length;
}

void NativeRenderer::TruthRemoveRecords(const std::vector<size_t>& indices) {
  if (indices.empty()) return;
  VulkanSharedMemory* shared_memory = command_processor_.NativeSharedMemory();
  for (size_t k = indices.size(); k-- > 0;) {
    WbRecord& r = wb_records_[indices[k]];
    command_processor_.NativeTexturePin(r.texture, false);
    // (Session 80: a depth record's stencil slot is free again - its materialise, if any, was recorded before.)
    if (r.depth && r.stencil_slot >= 0) stencil_slots_busy_ &= ~(1u << uint32_t(r.stencil_slot));
    // (Session 56: a scaled record never marked the guest memory - the scaled resolve does not write it.)
    if (shared_memory && r.scale_x <= 1 && r.scale_y <= 1) {
      shared_memory->SetRangeSkipped(r.extent_start, r.extent_length, false);
    }
    wb_records_.erase(wb_records_.begin() + ptrdiff_t(indices[k]));
  }
  // (Pages shared with the records that stay are theirs again.)
  if (shared_memory) {
    for (const WbRecord& r : wb_records_) {
      if (r.scale_x <= 1 && r.scale_y <= 1) shared_memory->SetRangeSkipped(r.extent_start, r.extent_length, true);
    }
    // (Session 79: and the deferred resolves' that write memory.)
    for (const DeferredResolve& d : truth_deferred_) {
      if (d.extent_length) shared_memory->SetRangeSkipped(d.extent_start, d.extent_length, true);
    }
  }
}

void NativeRenderer::TruthMaterialize(uint32_t start, uint32_t length, uint32_t reason) {
  std::vector<size_t> done;
  const uint64_t end = uint64_t(start) + length;
  // Session 79 (gpu_native_truth_defer_memory_resolves): a deferred resolve writing these pages - the batch first.
  if (TruthDeferredWritesMemory(start, end)) TruthFlushDeferred(kDeferFlushMaterialize);
  for (size_t i = 0; i < wb_records_.size(); ++i) {
    const WbRecord& r = wb_records_[i];
    if (r.scale_x > 1 || r.scale_y > 1) continue;  // (Session 56: scaled - not in the guest memory read here)
    if (r.extent_start < end && uint64_t(r.extent_start) + r.extent_length > start) {
      // wwe2k14 patch (Session 66, G1.0): a reader of the guest memory (not age, a resolve over part, skipping stopped).
      if (alias_frames_left_ > 0 && reason <= 4) AliasNoteSampled(r.texture, true);
      TruthMaterializeRecord(r);
      ++wb_counts_.by_reason[std::min(reason, 7u)];
      done.push_back(i);
    }
  }
  TruthRemoveRecords(done);
}

void NativeRenderer::TruthMaterializeScaled(uint32_t start, uint32_t length) {  // Session 56
  if (wb_records_.empty()) return;
  std::vector<size_t> done;
  const uint64_t end = uint64_t(start) + length;
  for (size_t i = 0; i < wb_records_.size(); ++i) {
    const WbRecord& r = wb_records_[i];
    if ((r.scale_x > 1 || r.scale_y > 1) && r.extent_start < end &&
        uint64_t(r.extent_start) + r.extent_length > start) {
      if (alias_frames_left_ > 0) AliasNoteSampled(r.texture, true);  // wwe2k14 patch (Session 66, G1.0): a scaled load
      TruthMaterializeRecord(r);
      ++wb_counts_.by_reason[6];
      done.push_back(i);
    }
  }
  TruthRemoveRecords(done);
}

void NativeRenderer::TruthMaterializeAll(uint32_t reason) {
  if (wb_records_.empty()) return;
  std::vector<size_t> done;
  for (size_t i = 0; i < wb_records_.size(); ++i) {
    TruthMaterializeRecord(wb_records_[i]);
    ++wb_counts_.by_reason[std::min(reason, 7u)];
    done.push_back(i);
  }
  TruthRemoveRecords(done);
}

void NativeRenderer::TruthDropRecords() {
  if (wb_records_.empty()) return;
  REXGPU_WARN("[native][wb] {} texture-only resolves dropped unwritten (the caches are cleared)", wb_records_.size());
  std::vector<size_t> all(wb_records_.size());
  for (size_t i = 0; i < all.size(); ++i) all[i] = i;
  TruthRemoveRecords(all);
}

void NativeRenderer::TruthBeforeResolveWrite(const draw_util::ResolveInfo& info) {
  if (alias_frames_left_ > 0) AliasBeginResolve();  // wwe2k14 patch (Session 66, G1.0)
  // Session 80 (frame structure census): where this resolve writes - its texture's page, format and rectangle.
  if (segment_log_frames_ > 0 && replacing_) {
    REXGPU_INFO("[native][seg] frame {} resolve dest: page 0x{:05X} texture page 0x{:05X} format {} {}x{} at {},{}"
                " depth {}",
                frame_, info.copy_dest_base >> 12, info.copy_dest_texture_base >> 12,
                uint32_t(info.copy_dest_info.copy_dest_format), uint32_t(info.coordinate_info.width_div_8) * 8,
                uint32_t(info.height_div_8) * 8, uint32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8,
                uint32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8, info.IsCopyingDepth());
  }
  // Session 79 (gpu_native_truth_defer_memory_resolves): a resolve over memory a deferred one writes - that one first.
  // (Into a separate rectangle of the same surface: separate bytes, no order needed - unless this resolve is read
  // back: the readback copies its whole extent, which interleaves with the deferred one's bytes. Review fix.)
  if (info.copy_dest_extent_length &&
      TruthDeferredWritesMemory(info.copy_dest_extent_start,
                                uint64_t(info.copy_dest_extent_start) + info.copy_dest_extent_length,
                                command_processor_.NativeResolveReadback() ? nullptr : &info)) {
    TruthFlushDeferred(kDeferFlushOtherResolve);
  }
  truth_wb_skipped_ = false;
  truth_fill_tried_ = false;
  truth_overlaps_done_ = false;
}

void NativeRenderer::TruthResolveOverlaps(const draw_util::ResolveInfo& info, void* filled_texture,
                                          const int32_t* filled_rect, int32_t filled_layer) {
  if (truth_overlaps_done_) return;
  truth_overlaps_done_ = true;
  if (wb_records_.empty()) return;
  const uint32_t start = info.copy_dest_extent_start;
  const uint64_t end = uint64_t(start) + info.copy_dest_extent_length;
  const uint32_t x0 = uint32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8;
  const uint32_t y0 = uint32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8;
  const uint32_t x1 = x0 + uint32_t(info.coordinate_info.width_div_8) * 8, y1 = y0 + info.height_div_8 * 8;
  const uint32_t format = uint32_t(info.copy_dest_info.copy_dest_format);
  const uint32_t bytes = info.copy_dest_info.copy_dest_array                                   ? 0u
                         : format == uint32_t(xenos::TextureFormat::k_16_16_16_16)             ? 8u
                         : format == uint32_t(xenos::TextureFormat::k_8_8_8_8) ||
                                 format == uint32_t(xenos::TextureFormat::k_16_16)
                             ? 4u
                             : 0u;
  std::vector<size_t> superseded, overlapped, all;
  for (size_t i = 0; i < wb_records_.size(); ++i) {
    const WbRecord& r = wb_records_[i];
    if (!(r.extent_start < end && uint64_t(r.extent_start) + r.extent_length > start)) continue;
    // (Session 56: a scaled record's x0 ... are host texels - compared by its guest-texel rectangle.)
    const bool scaled = r.scale_x > 1 || r.scale_y > 1;
    const uint32_t rx0 = scaled ? r.guest_x0 : r.x0, ry0 = scaled ? r.guest_y0 : r.y0;
    const uint32_t rx1 = scaled ? r.guest_x1 : r.x1, ry1 = scaled ? r.guest_y1 : r.y1;
    if (bytes && bytes == (r.bpb64 ? 8u : 4u) && r.base == info.copy_dest_base &&
        r.pitch == uint32_t(info.copy_dest_coordinate_info.pitch_aligned_div_32) && x0 <= rx0 && y0 <= ry0 &&
        x1 >= rx1 && y1 >= ry1) {
      // Every texel the record would write, this resolve writes too (the same tiled surface): replaced.
      superseded.push_back(i);
      all.push_back(i);
    } else if (filled_texture && r.texture == filled_texture && r.layer == filled_layer) {  // (Session 79: same face)
      // This resolve also fills the record's texture, so the texture stays what both resolves wrote: the record,
      // written later, writes this resolve's texels as this resolve does. Kept - or replaced when this resolve fills
      // every texel of the record's rectangle (one texture: one texel, one address).
      if (filled_rect && filled_rect[0] <= int32_t(r.tex_x0 + rx0) && filled_rect[1] <= int32_t(r.tex_y0 + ry0) &&
          filled_rect[2] >= int32_t(r.tex_x0 + rx1) && filled_rect[3] >= int32_t(r.tex_y0 + ry1)) {
        superseded.push_back(i);
        all.push_back(i);
      }
    } else {
      overlapped.push_back(i);
      all.push_back(i);
    }
  }
  // Partly overwritten records go into memory first (in their order), then this resolve writes over them.
  for (size_t i : overlapped) {
    TruthMaterializeRecord(wb_records_[i]);
    ++wb_counts_.by_reason[5];
  }
  if (!overlapped.empty()) {
    // (Their writes before this resolve's: the shared memory's write-after-write barrier.)
    if (VulkanSharedMemory* shared_memory = command_processor_.NativeSharedMemory()) {
      shared_memory->Use(VulkanSharedMemory::Usage::kComputeWrite,
                         std::pair<uint32_t, uint32_t>(info.copy_dest_extent_start, info.copy_dest_extent_length));
    }
    // (Session 56: at a scale, the scaled resolve buffer's.)
    uint64_t scaled_start = 0, scaled_length = 0;
    if (command_processor_.NativeScaledResolveBuffer() != VK_NULL_HANDLE &&
        command_processor_.NativeScaledResolveRange(info.copy_dest_extent_start, info.copy_dest_extent_length,
                                                    scaled_start, scaled_length)) {
      command_processor_.NativeUseScaledResolveBufferForWrite(scaled_start, scaled_length);
    }
  }
  wb_counts_.superseded += uint32_t(superseded.size());
  TruthRemoveRecords(all);
}

bool NativeRenderer::TruthFillOnly(const draw_util::ResolveInfo& info,
                                   const draw_util::ResolveCopyShaderConstants& constants) {
  if (!truth_ || !replacing_ || !truth_skip_writeback_ || texture_written_ ||
      (command_processor_.NativeResolveReadback() && !REXCVAR_GET(gpu_native_truth_skip_writeback_debug_immediate))) {
    return false;  // (also when the CPU reads the output back: readback_resolve, matte dumps)
  }
  // The fill first (it reads only the image; a pack fill uses the staging buffer, which TruthResolveSource writes
  // again after it if the copy is still needed). Tried once: the call after the copy returns at once.
  truth_fill_done_ = FillDone();
  TruthFillTexture(info);
  truth_fill_tried_ = true;
  {  // (before any write of this resolve into memory; the texture's rectangle the fill wrote)
    const FillDone& done = truth_fill_done_;
    const int32_t filled_rect[4] = {done.dst_x, done.dst_y, done.dst_x + done.w, done.dst_y + done.h};
    TruthResolveOverlaps(info, texture_written_, done.texture && !done.level ? filled_rect : nullptr);
  }
  if (!texture_written_) {
    ++wb_counts_.kept[1];
    return false;
  }
  const FillDone& f = truth_fill_done_;
  const uint32_t bytes = f.depth ? 0u : command_processor_.NativeTextureReadable(f.texture);
  const uint32_t format = uint32_t(info.copy_dest_info.copy_dest_format);
  const uint32_t format_bytes = format == uint32_t(xenos::TextureFormat::k_16_16_16_16) ? 8u
                                : format == uint32_t(xenos::TextureFormat::k_8_8_8_8) ||
                                        format == uint32_t(xenos::TextureFormat::k_16_16)
                                    ? 4u
                                    : 0u;
  if (f.depth || f.level || f.has_mips || !bytes || bytes != format_bytes || !EnsureMaterializePipeline()) {
    ++wb_counts_.kept[2];
    return false;
  }
  // Every texel the copy writes is in the texture (not cut short by the texture or the image).
  if (f.w != f.copy_w || f.h != f.copy_h || f.dst_x < 0 || f.dst_y < 0 ||
      f.dst_x + f.copy_w > int32_t(f.texture_w) || f.dst_y + f.copy_h > int32_t(f.texture_h)) {
    ++wb_counts_.kept[3];
    return false;
  }
  const uint32_t ox = uint32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8;
  const uint32_t oy = uint32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8;
  WbRecord& r = truth_wb_pending_;
  r = WbRecord();
  r.texture = f.texture;
  r.extent_start = info.copy_dest_extent_start;
  r.extent_length = info.copy_dest_extent_length;
  r.dest_offset = truth_dest_offset_;
  r.dest_range = truth_dest_range_;
  r.base = info.copy_dest_base;
  r.pitch = uint32_t(info.copy_dest_coordinate_info.pitch_aligned_div_32);
  r.x0 = ox;
  r.y0 = oy;
  r.x1 = ox + uint32_t(f.copy_w);
  r.y1 = oy + uint32_t(f.copy_h);
  r.dest_base = constants.dest_base - uint32_t(truth_dest_offset_);  // (1x: the offset is below 4 GiB)
  r.endian = uint32_t(info.copy_dest_info.copy_dest_endian);
  // (The fill wrote destination texel (ox, oy) at (dst_x, dst_y): copy_dest_base inside the texture.)
  r.tex_x0 = uint32_t(f.dst_x) - ox;
  r.tex_y0 = uint32_t(f.dst_y) - oy;
  r.bpb64 = bytes == 8;
  r.frame = frame_;
  truth_wb_skipped_ = true;
  if (segment_log_frames_ > 0) {
    REXGPU_INFO("[native][seg] frame {} truth texture-only resolve: page 0x{:05X} format {} {}x{} at {},{}", frame_,
                info.copy_dest_texture_base >> 12, format, f.copy_w, f.copy_h, f.dst_x, f.dst_y);
  }
  return true;
}

// Session 72 part 4 (gpu_native_truth_skip_repeat_resolves): every register of a resolve that decides what its copy
// writes and where (FNV-1a over them).
static uint64_t ResolveKey(const draw_util::ResolveInfo& info) {
  const uint32_t v[] = {info.rb_copy_control.value,       info.depth_edram_info.packed,
                        info.color_edram_info.packed,     info.depth_original_base,
                        info.color_original_base,         info.coordinate_info.packed,
                        info.height_div_8,                info.copy_dest_info.value,
                        info.copy_dest_coordinate_info.packed, info.copy_dest_base,
                        info.copy_dest_extent_start,      info.copy_dest_extent_length,
                        info.copy_dest_texture_base,      info.copy_dest_base_x,
                        info.copy_dest_base_y};
  uint64_t h = 1469598103934665603ull;
  for (const uint32_t x : v) {
    h ^= x;
    h *= 1099511628211ull;
  }
  return h;
}

bool NativeRenderer::TruthResolveRedundant(const draw_util::ResolveInfo& info) {
  if (!enabled_ || !truth_ || !replacing_ || truth_failed_ || !REXCVAR_GET(gpu_native_truth_skip_repeat_resolves) ||
      command_processor_.NativeResolveReadback()) {
    return false;
  }
  const uint64_t key = ResolveKey(info);
  for (const RestoreSource& s : restore_sources_) {
    if (s.frame != frame_ || s.key != key || s.image < 0 || size_t(s.image) >= truth_images_.size()) continue;
    const uint64_t writes = size_t(s.image) < truth_image_writes_.size() ? truth_image_writes_[s.image] : 0;
    if (s.writes != writes) return false;
    uint32_t base = 0, row_length = 0, rows = 0, pitch = 0;
    info.GetCopyEdramTileSpan(base, row_length, rows, pitch);
    if (!rows || !row_length) return false;
    const std::bitset<2048>& owned = truth_images_[s.image].owned;
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c = 0; c < row_length; ++c) {
        if (!owned.test((base + r * pitch + c) & 2047u)) return false;
      }
    }
    ++truth_counts_.repeat_resolves_skipped;
    if (segment_log_frames_ > 0) {
      REXGPU_INFO("[native][seg] frame {} truth resolve into page 0x{:05X} from image {}: the same copy as before, "
                  "left out",
                  frame_, info.copy_dest_base >> 12, s.image);
    }
    return true;
  }
  return false;
}

void NativeRenderer::TruthNoteRestoreSource(const draw_util::ResolveInfo& info, int16_t index, bool at_origin,
                                            uint32_t width, uint32_t height) {
  if (!REXCVAR_GET(gpu_native_truth_skip_restores) && !REXCVAR_GET(gpu_native_truth_skip_repeat_resolves)) return;
  restore_source_fresh_ = -1;
  if (index < 0) return;
  const uint32_t page = info.copy_dest_base >> 12;
  // (One entry a page: a later copy into the page replaces it.)
  uint32_t slot = restore_source_next_;
  for (uint32_t i = 0; i < restore_sources_.size(); ++i) {
    if (restore_sources_[i].page == page && restore_sources_[i].frame != UINT64_MAX) {
      slot = i;
      break;
    }
  }
  if (slot == restore_source_next_) restore_source_next_ = (restore_source_next_ + 1) % uint32_t(restore_sources_.size());
  RestoreSource& s = restore_sources_[slot];
  s.page = page;
  s.image = index;
  s.writes = size_t(index) < truth_image_writes_.size() ? truth_image_writes_[index] : 0;
  s.frame = frame_;
  s.width = width;
  s.height = height;
  s.extent_start = info.copy_dest_extent_start;
  s.extent_end = info.copy_dest_extent_start + info.copy_dest_extent_length;
  s.claims = truth_frame_draws_;
  s.at_origin = at_origin;
  s.key = ResolveKey(info);
  restore_source_fresh_ = int32_t(slot);
}

bool NativeRenderer::TruthRestoreRedundantEarly(uint32_t surface_pitch, uint32_t color_base, uint32_t depth_base,
                                                uint32_t color_page, uint32_t depth_page) {
  if (!enabled_ || !truth_ || !replacing_ || truth_failed_) return false;
  const int16_t color = TruthOwnerOf(color_base & 2047u), depth = TruthOwnerOf(depth_base & 2047u);
  if (color < 0 || depth < 0) return false;
  const TruthImage& c = truth_images_[color];
  const TruthImage& d = truth_images_[depth];
  const uint32_t pitch_tiles = std::max(1u, (surface_pitch + 79) / 80);
  if (c.kind != 1 || c.format != 0 || c.base != color_base || c.pitch_tiles != pitch_tiles || !c.initialized ||
      d.kind != 3 || d.base != depth_base || d.pitch_tiles != pitch_tiles || !d.initialized ||
      TruthTwinPitch(c.pitch_tiles)) {
    return false;
  }
  const RestoreSource* sources[2] = {};
  const uint32_t pages[2] = {color_page, depth_page};
  const int16_t images[2] = {color, depth};
  for (int i = 0; i < 2; ++i) {
    const uint64_t writes = size_t(images[i]) < truth_image_writes_.size() ? truth_image_writes_[images[i]] : 0;
    for (const RestoreSource& s : restore_sources_) {
      if (s.frame == frame_ && s.at_origin && s.page == pages[i] && s.image == images[i] && s.writes == writes &&
          s.claims == truth_frame_draws_) {
        sources[i] = &s;
        break;
      }
    }
    if (!sources[i]) return false;
  }
  // The copied rectangle (both from the origin; the smaller one) still the images' own, tile by tile.
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const uint32_t w = std::min(sources[0]->width, sources[1]->width) / scale_x;
  const uint32_t h = std::min(sources[0]->height, sources[1]->height) / scale_y;
  if (!w || !h) return false;
  if ((EdramTiles(c.base, c.pitch_tiles, c.is_64bpp(), 0, 0, w, h) & ~c.owned).any() ||
      (EdramTiles(d.base, d.pitch_tiles, false, 0, 0, w, h) & ~d.owned).any()) {
    return false;
  }
  ++truth_counts_.restores_skipped;
  if (segment_log_frames_ > 0) {
    REXGPU_INFO("[native][seg] frame {} truth restore into colour {} depth {} (textures 0x{:05X} / 0x{:05X}): left out "
                "before its textures were requested",
                frame_, color, depth, color_page, depth_page);
  }
  return true;
}

bool NativeRenderer::TruthRestoreRedundant(const Draw& draw, int16_t color, int16_t depth, const int32_t* rect) {
  if (draw.instance_count != 1 || draw.truth_color != color || draw.truth_depth != depth || color < 0 || depth < 0) {
    return false;
  }
  const TruthImage& c = truth_images_[color];
  const TruthImage& d = truth_images_[depth];
  if (c.base == kTruthScratchBase || d.base == kTruthScratchBase || c.kind != 1 || d.kind != 3) return false;
  // A plain copy: every colour channel written, nothing blended; depth written wherever it draws; stencil untouched.
  if ((draw.color_write_mask & 0xF) != 0xF || PostNeedsDst(draw) || !draw.raster.depth_write ||
      (draw.raster.depth_test && draw.raster.depth_compare != VK_COMPARE_OP_ALWAYS) ||
      (draw.raster.stencil && (draw.stencil_write_masks[0] || draw.stencil_write_masks[1]))) {
    return false;
  }
  // Both textures copied this frame, from the image's origin into the texture's, at least the drawn rectangle, out of
  // the very image the draw writes, no truth draw into that image since.
  const auto from = [&](uint32_t page, int16_t index) {
    const uint64_t writes = size_t(index) < truth_image_writes_.size() ? truth_image_writes_[index] : 0;
    for (const RestoreSource& s : restore_sources_) {
      if (s.frame == frame_ && s.at_origin && s.page == page && s.image == index && s.writes == writes && rect[0] >= 0 &&
          rect[1] >= 0 && uint32_t(rect[2]) <= s.width && uint32_t(rect[3]) <= s.height) {
        return true;
      }
    }
    return false;
  };
  if (!from(draw.restore_color_page, color) || !from(draw.restore_depth_page, depth)) return false;
  // And the images still own every tile the draw covers (another image drawing there takes a tile over).
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const uint32_t gx0 = uint32_t(rect[0]) / scale_x, gy0 = uint32_t(rect[1]) / scale_y;
  const uint32_t gx1 = (uint32_t(rect[2]) + scale_x - 1) / scale_x, gy1 = (uint32_t(rect[3]) + scale_y - 1) / scale_y;
  return (EdramTiles(c.base, c.pitch_tiles, c.is_64bpp(), gx0, gy0, gx1, gy1) & ~c.owned).none() &&
         (EdramTiles(d.base, d.pitch_tiles, false, gx0, gy0, gx1, gy1) & ~d.owned).none();
}

void NativeRenderer::TruthAfterResolveMarked(const draw_util::ResolveInfo& info) {
  if (alias_frames_left_ > 0 && enabled_) AliasNoteResolve(info);  // wwe2k14 patch (Session 66, G1.0)
  // Session 72 part 4 (gpu_native_truth_skip_restores): a copy into a recorded page (other than the record this copy
  // made itself) - the texture no longer holds that image's pixels.
  if (REXCVAR_GET(gpu_native_truth_skip_restores)) {
    const uint64_t start = info.copy_dest_extent_start, end = start + info.copy_dest_extent_length;
    for (uint32_t i = 0; i < restore_sources_.size(); ++i) {
      RestoreSource& s = restore_sources_[i];
      if (int32_t(i) != restore_source_fresh_ && s.frame != UINT64_MAX && s.extent_end > start && s.extent_start < end) {
        s.frame = UINT64_MAX;
      }
    }
    restore_source_fresh_ = -1;
  }
  if (!truth_wb_skipped_) return;
  truth_wb_skipped_ = false;
  VulkanSharedMemory* shared_memory = command_processor_.NativeSharedMemory();
  if (!shared_memory) return;
  wb_records_.push_back(truth_wb_pending_);
  command_processor_.NativeTexturePin(truth_wb_pending_.texture, true);
  // (Session 56: a scaled record does not mark the guest memory: the scaled resolve never writes it.)
  const bool scaled = truth_wb_pending_.scale_x > 1 || truth_wb_pending_.scale_y > 1;
  if (!scaled) shared_memory->SetRangeSkipped(truth_wb_pending_.extent_start, truth_wb_pending_.extent_length, true);
  ++wb_counts_.skipped;
  wb_counts_.skipped_bytes += truth_wb_pending_.extent_length;
  // (Diagnostic: written from the texture at once, where the copy would have written it - the resolve hashes then
  // check the materialise shader against the copy.)
  if (REXCVAR_GET(gpu_native_truth_skip_writeback_debug_immediate)) {
    if (scaled) {
      TruthMaterializeScaled(truth_wb_pending_.extent_start, truth_wb_pending_.extent_length);
    } else {
      TruthMaterialize(truth_wb_pending_.extent_start, truth_wb_pending_.extent_length, 0);
    }
  }
}

void NativeRenderer::TruthWriteOut() {
  OwnedBeforeWriteOut();  // Session 80 (owned passes: the truth images made whole first)
  TruthFlushDeferred(kDeferFlushFallback);  // Session 52
  EndTruthPass();
  ++truth_claim_epoch_;  // Session 47 (the owners change)
  const bool any = truth_owned_any_.any();
  // Session 54 (the host render target path): no EDRAM to write into - the images stay the picture (their owners kept);
  // what is drawn without truth frames goes to the render target cache's own targets.
  if (command_processor_.NativeEdramBuffer() == VK_NULL_HANDLE) {
    if (any) ++truth_counts_.writeouts;
    return;
  }
  if (any && command_processor_.NativeEdramBuffer() != VK_NULL_HANDLE) {
    std::bitset<2048> all;
    all.set();
    std::vector<int16_t> indices;
    for (size_t i = 0; i < truth_images_.size(); ++i) indices.push_back(int16_t(i));
    TruthStoreMany(indices.data(), indices.size(), all, false);
    ++truth_counts_.writeouts;
  }
  // (Review 1: the EDRAM's stencil may change after this; a later depth-only draw must keep it, not re-store ours.)
  for (TruthImage& t : truth_images_) {
    t.stencil_tiles.reset();
    t.stencil_valid.reset();  // (the EDRAM owns every tile now)
  }
  TruthClearOwners();
}

void NativeRenderer::TruthFallback(uint32_t reason, const Draw* draw) {
  if (OwnedOnly()) {  // Session 92 (M2a): a fallen frame does not exist under the latch - counted, ignored (the frame goes on as a truth frame of owned passes)
    OwnedNoteFallbackIgnored(reason);
    return;
  }
  serve_rename_.valid = false;  // Session 80 (gpu_native_truth_serve_copies: its pixels copied by this flush)
  TruthFlushDeferred(kDeferFlushFallback);  // Session 52
  truth_counts_.serve_lost += uint32_t(serve_pending_.size());  // Session 80 (gpu_native_truth_serve_copies)
  serve_pending_.clear();
  if (!truth_) return;
  if (reason >= kTruthReasons) reason = kTruthEmulated;
  ++truth_counts_.fallbacks;
  ++truth_counts_.reasons[reason];
  if (truth_reason_logged_[reason] < 6 && reason != kTruthNoShader && reason != kTruthNoPipeline) {
    // (Refused claims logged their draw in ClaimTruth.)
    if (!draw || reason == kTruthEmulated) {
      ++truth_reason_logged_[reason];
      REXGPU_INFO("[native][truth] frame {}: fallback ({}){}", frame_, TruthReasonName(reason),
                  draw ? fmt::format(": vs {:016X} ps {:016X} colour 0x{:08X} fast clear {}", draw->vertex_shader_hash,
                                     draw->pixel_shader_hash, draw->color_info, draw->fast_clear)
                       : std::string());
    }
  }
  TruthWriteOut();
  truth_ = false;
  truth_fallback_reason_ = 0;
  // Session 54 (the host render target path): the rest of the frame emulated, nothing replaced (the replacing paths
  // outside truth frames need the EDRAM).
  if (command_processor_.NativeEdramBuffer() == VK_NULL_HANDLE) replacing_ = false;
  // The exact path from here on: its targets reload what they need from the EDRAM.
  native_stale_ = true;
  InvalidateAllPostSync();
  post_current_ = -1;
}
void NativeRenderer::EndTruthPass() {
  cull_ours::NoteBreak();  // Session 93 (gpu_d3d_cull_ours): the pass the cull guard follows ends here
  OwnedEndPass();  // Session 80 (owned passes: ours is ended wherever a truth pass would be)
  if (!truth_pass_open_) {
    TruthRestoreAliasOwners();  // Session 79 (made readable for a pass that did not open)
    return;
  }
  if (segment_log_frames_ > 0) {  // Session 56 (diagnostic): who ends the pass (tools\perf\pass_ends.py)
    uintptr_t stack[5] = {};
    const uintptr_t base = VulkanCommandProcessor::DebugCaptureStack(stack, 5, 0);
    REXGPU_INFO("[native][seg] frame {} truth pass end: base {:X} stack {:X} {:X} {:X} {:X} {:X}", frame_, base,
                stack[0], stack[1], stack[2], stack[3], stack[4]);
  }
  truth_pass_open_ = false;
  OwnedNoteEndedClears();  // Session 82 (the owned frame: a pass of only clears leaves its surface "cleared")
  truth_tile_c_owned_ = truth_tile_d_owned_ = false;  // Session 47: the fast path is per open pass
  truth_bound_valid_ = false;                         // (and what its draws bound)
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  // Session 73 (gpu_native_truth_color_none): the colour image untouched by the pass: kept without load / store.
  if (!truth_pass_touched_color_ && truth_pass_real_color_ && truth_pass_kind_ < kPostTargetClasses &&
      REXCVAR_GET(gpu_native_truth_color_none)) {
    const uint32_t k = truth_pass_kind_;
    const int variant = truth_pass_begin_rp_ == truth_render_pass_keep_nodep_[k]      ? 0
                        : truth_pass_begin_rp_ == truth_render_pass_no_depth_nodep_[k] ? 1
                        : truth_pass_begin_rp_ == truth_render_pass_dont_care_[1][k]   ? 2
                                                                                       : -1;
    if (variant >= 0 && truth_render_pass_color_none_[variant][k] != VK_NULL_HANDLE &&
        cb.PatchBeginRenderPass(truth_pass_begin_offset_, truth_pass_begin_rp_,
                                truth_render_pass_color_none_[variant][k])) {
      ++truth_counts_.color_none;
    }
  }
  truth_pass_end_cmd_offset_ = cb.stream_offset();  // Session 80 (owned passes)
  if (truth_pass_lazy_) {  // Session 56
    cb.CmdVkEndRenderPassLazy();
    truth_pass_lazy_ = false;
  } else {
    cb.CmdVkEndRenderPass();
  }
  truth_pass_end_offset_ = cb.stream_offset();  // Session 80 (gpu_native_own_steps_drop_clears)
  // Session 80 (gpu_native_truth_drop_dead_passes): a pass that only drew colour pixels - turned off if the next pass
  // clears them first (TruthRecordDraw).
  dead_pass_.valid = REXCVAR_GET(gpu_native_truth_drop_dead_passes) && truth_pass_dead_ok_ &&
                     truth_pass_touched_color_ && !truth_pass_touched_depth_ && truth_pass_real_color_ &&
                     truth_pass_draws_ && truth_pass_color_ >= 0 && size_t(truth_pass_color_) < truth_images_.size();
  if (dead_pass_.valid) {
    dead_pass_.color = truth_pass_color_;
    dead_pass_.image = truth_images_[truth_pass_color_].image;
    std::copy(truth_pass_rect_, truth_pass_rect_ + 4, dead_pass_.rect);
    dead_pass_.begin = truth_pass_begin_offset_;
    dead_pass_.end = cb.stream_offset();
    dead_pass_.submission = command_processor_.GetCurrentSubmission();
    dead_pass_.frame = frame_;
    dead_pass_.segment = truth_pass_segment_index_;
  }
  FrameSlot& slot = slots_[SlotOf(frame_)];
  if (truth_pass_segment_index_ < slot.segments.size() && slot.queries < QueryLimit()) {
    FrameSlot::Segment& segment = slot.segments[truth_pass_segment_index_];
    segment.pass_draws = truth_pass_draws_;  // Session 72 (gpu_native_truth_pass_log)
    segment.native_end = slot.queries++;
    cb.CmdVkWriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_,
                           SlotOf(frame_) * kQueriesPerFrame + segment.native_end);
  }
  truth_pass_segment_index_ = UINT32_MAX;
  ++truth_counts_.passes;
  if (owned_frame_on_) ++owned_frame_stats_.truth_passes;  // Session 82 (the census of truth work)
  truth_counts_.untouched_color += !truth_pass_touched_color_;  // Session 73 (scratch: the pass has no colour target)
  truth_counts_.untouched_scratch_color += !truth_pass_touched_color_ && !truth_pass_real_color_;
  truth_counts_.untouched_depth += truth_pass_real_depth_ && !truth_pass_touched_depth_;
  TruthRestoreAliasOwners();
}

// Session 79 (gpu_native_truth_alias_direct): the images the pass's loads read, back in their attachment layouts (also
// when no pass opened after they were made readable).
void NativeRenderer::TruthRestoreAliasOwners() {
  // (Diagnostic, wrong layouts: the restore left out - how many pass chains it breaks, gpu_native_truth_debug_skip_*.)
  if (!truth_alias_read_owners_.empty() && REXCVAR_GET(gpu_native_truth_debug_skip_alias_restore)) {
    truth_alias_read_owners_.clear();
    return;
  }
  if (!truth_alias_read_owners_.empty()) {
    DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
    std::vector<VkImageMemoryBarrier>& barriers = truth_store_barriers_;
    barriers.clear();
    for (const auto& [image, depth] : truth_alias_read_owners_) {
      VkImageMemoryBarrier& b = barriers.emplace_back();
      b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b.image = image;
      b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
      b.dstAccessMask = depth ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                              : VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      b.oldLayout = depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      b.newLayout = depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      b.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(
          depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
    }
    cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                            0, 0, nullptr, 0, nullptr, uint32_t(barriers.size()), barriers.data());
    truth_alias_read_owners_.clear();
  }
}

void NativeRenderer::TruthRecordDraw(const Draw& draw) {
  rex::graphics::free_counters::Bump(rex::graphics::free_counters::g_c.truth_entries);  // (Session 92 M0: log-only counter)
  // Session 90 (black arena, S90_BLACK_ARENA.md 9): "this draw is its pass's load-op clear" (truth_skip_clear_draw_,
  // Session 80 gpu_native_truth_clear_load) belongs to the call whose pass start decided it. A nested recording (a held
  // clear flushed from OwnedTakeDraw / OwnedHoldClear / OwnedTakeDrawImpl through OwnedFlushHeld) used to leave the flag
  // set for the outer draw, whose draw call was then left out (the arena's restore). The member is saved on entry and
  // restored on every return path; the readers below only ever see this call's own value.
  struct SkipScope {
    NativeRenderer& r;
    const Draw& d;
    bool saved;
    uint32_t depth;
    SkipScope(NativeRenderer& rr, const Draw& dd)
        : r(rr), d(dd), saved(rr.truth_skip_clear_draw_), depth(++rr.truth_record_depth_) {
      r.truth_skip_clear_draw_ = false;
      if (depth == 1) r.truth_outer_draw_ = &d;
    }
    ~SkipScope() {
      if (depth > 1 && r.truth_skip_clear_draw_) {
        // A nested recording set the flag: the old code would have carried it into the outer draw.
        ++r.truth_counts_.skip_nested_reset;
        if (r.segment_log_frames_ > 0 && r.truth_outer_draw_) {
          REXGPU_INFO("[native][seg] frame {} load-op clear flag set by a nested recording (vs {:016X} ps {:016X}) reset "
                      "for the outer draw (vs {:016X} ps {:016X}, fast_clear {}), nesting depth {}",
                      r.frame_, d.vertex_shader_hash, d.pixel_shader_hash, r.truth_outer_draw_->vertex_shader_hash,
                      r.truth_outer_draw_->pixel_shader_hash, r.truth_outer_draw_->fast_clear, depth);
        }
      }
      r.truth_skip_clear_draw_ = saved;
      --r.truth_record_depth_;
      if (depth == 1) r.truth_outer_draw_ = nullptr;
    }
  } skip_scope(*this, draw);
  // wwe2k14 patch (Session 92, M2a, gpu_native_owned_only): nothing below the owned frame is reached - a draw our frame does not take (or whose claim was
  // refused) is dropped, counted and named (OwnedDropDraw); no clear lost into the truth path, no dirty tile, no truth pass.
  if (OwnedOnly()) {
    if (!draw.claim_refused && OwnedTakeDraw(draw)) return;
    OwnedDropDraw(draw);
    return;
  }
  // wwe2k14 patch (Session 80, gpu_native_own_steps; owned_steps.cpp): an owned step's draw goes into our image.
  if (OwnedTakeDraw(draw)) return;
  // wwe2k14 patch (Session 87, D3D layer F2): a clear built from the game's device state has no pipeline - the truth
  // path cannot draw it. Counted and logged loudly (the clear is lost; the picture shows the earlier contents).
  if (draw.d3d_record) {
    D3dNoteClearLost("truth path");
    return;
  }
  // (Session 80, owned passes: a draw over dirty tiles - our pixels copied in first unless it clears them whole; those
  // it clears leave the set once its draw command is recorded, below.)
  owned_clean_pending_.clear();
  if (!owned_dirty_.empty()) OwnedDirtyBeforeDraw(draw);
  if (draw.raster.stencil && !draw.fast_clear) OwnedCountUnprovenStencil(draw);  // (Session 92: census only, TRUE NATIVE)
  // wwe2k14 patch (Session 79, diagnostic): who breaks the lazy pass chains, while the segment log runs.
  command_processor_.deferred_command_buffer().capture_break_stack = segment_log_frames_ > 0;
  // wwe2k14 patch (Session 79, NF-D look-ahead diagnostic, gpu_native_frame_batch_log N): which native-frame catch-up
  // batch the star glare's draws (75641266549A40C0, 9C84E33D39058FB0) and the entrance combine (4E2CD8031D93DE3E) land
  // in, the first N such draws (shader hashes are our identifiers).
  if (const int32_t log_left = REXCVAR_GET(gpu_native_frame_batch_log); log_left > 0) [[unlikely]] {
    const uint64_t ps = draw.pixel_shader_hash;
    if (ps == 0x75641266549A40C0ull || ps == 0x9C84E33D39058FB0ull || ps == 0x4E2CD8031D93DE3Eull) {
      REXGPU_INFO("[native][frame] batch {} frame {}: draw ps {:016X}", command_processor_.nf_batch_seq(), frame_, ps);
      REXCVAR_SET(gpu_native_frame_batch_log, log_left - 1);
    }
  }
  // (Session 47, CPU: the recording time is sampled - one draw in 64, counted 64 times - not two clock reads a draw.)
  const bool timed = (++truth_time_tick_ & 63) == 0;
  std::chrono::steady_clock::time_point cpu_start;
  if (timed) [[unlikely]] cpu_start = std::chrono::steady_clock::now();
  const auto add_time = [&](FrameSlot& s) {
    if (timed) [[unlikely]] {
      s.cpu_us += 64.0 * std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - cpu_start).count();
    }
  };
  truth_last_recorded_ = false;  // Session 47 (set once the draw command is recorded)
  own_crowd_sig_valid_ = false;  // Session 93 (the crowd merge: set below for a crowd run's first member, once its state is recorded)
  // Session 80 (gpu_native_truth_serve_copies): a draw sampling an image in place of a copy needs it readable - an
  // open pass that does not read it already ends (the barrier is recorded before the next pass begins: a pass this
  // draw opens makes every one readable that is not yet - review: also when the pass changes later in this draw). What
  // this draw served is its own: dropped however it returns.
  struct ServeScope {
    NativeRenderer& r;
    ~ServeScope() {
      r.serve_pending_.clear();
      r.serve_rename_.valid = false;
    }
  } serve_scope{*this};
  if (!serve_pending_.empty() && truth_pass_open_) {
    for (const VkImage image : serve_pending_) {
      if (std::none_of(truth_alias_read_owners_.begin(), truth_alias_read_owners_.end(),
                       [&](const std::pair<VkImage, bool>& o) { return o.first == image; })) {
        // (Session 80, gpu_native_truth_serve_ahead: remembered - the texture whose carried copy this image is the
        // source of, and the pass it ends: its colour slot and the vertex shader of the draw that opened it.)
        for (const DeferredResolve& d : truth_deferred_) {
          if (d.pinned && d.image == image) serve_ahead_ = {d.texture, truth_pass_color_, truth_pass_first_vs_};
        }
        EndTruthPass();
        ++truth_counts_.serve_breaks;
        break;
      }
    }
  }
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  FrameSlot& slot = slots_[SlotOf(frame_)];
  const uint32_t pitch = std::max(1u, ((draw.surface_info & 0x3FFF) * draw.truth_sx + 79) / 80);
  // The images: the draw's own; a draw without colour or depth takes the open pass's (or a scratch image).
  int16_t color = draw.truth_color, depth = draw.truth_depth;
  const bool same_pass = truth_pass_open_ && truth_pass_class_ == draw.truth_class && truth_pass_pitch_ == pitch &&
                         (color < 0 || color == truth_pass_color_) && (depth < 0 || depth == truth_pass_depth_);
  if (same_pass) {
    color = truth_pass_color_;
    depth = truth_pass_depth_;
  } else {
    if (color < 0) color = TruthFindImage(kTruthScratchBase, pitch, draw.truth_class, draw.truth_class == 1 ? 0u : 4u);
    if (depth < 0) depth = TruthFindImage(kTruthScratchBase, pitch, 3, 1);
  }
  // A failure: this draw is lost (its emulated draw was skipped), the rest of the frame exact (review fix).
  const auto fail = [&]() {
    REXGPU_ERROR("[native][truth] frame {}: no images for a truth draw: lost; no truth frames from now on", frame_);
    truth_failed_ = true;
    TruthFallback(kTruthImages, &draw);
  };
  if (color < 0 || depth < 0) return fail();
  int32_t rect[4];
  TruthDrawRect(draw, rect);
  // (Review fix: at most the rows the images can have - a clear's 8192-row scissor would end the pass every draw.)
  {
    const uint32_t max_c = (2048u + truth_images_[color].row_tiles() - 1) / truth_images_[color].row_tiles() * 16 * scale_y;
    const uint32_t max_d = (2048u + truth_images_[depth].row_tiles() - 1) / truth_images_[depth].row_tiles() * 16 * scale_y;
    rect[3] = std::min(rect[3], int32_t(std::min(max_c, max_d)));
  }
  if (rect[2] <= rect[0] || rect[3] <= rect[1] || draw.pipeline == VK_NULL_HANDLE) return;  // nothing to draw
  // Grown images (a copy, outside any render pass) and the framebuffer.
  if (uint32_t(rect[3]) > truth_images_[color].height || uint32_t(rect[3]) > truth_images_[depth].height ||
      !truth_images_[color].initialized || !truth_images_[depth].initialized) {
    EndTruthPass();
    if (!EnsureTruthImage(color, uint32_t(rect[3])) || !EnsureTruthImage(depth, uint32_t(rect[3]))) return fail();
  }
  // Session 54 (gpu_native_truth_draw_height): the pass draws into the images' draw twins, `ratio` of their size; the
  // rectangles below stay in the images' pixels and are scaled when recorded.
  const bool twins = TruthTwinPitch(truth_images_[color].pitch_tiles);
  const float ratio = twins ? truth_draw_ratio_ : 1.0f;
  if (twins && (!EnsureTruthTwin(color) || !EnsureTruthTwin(depth))) return fail();
  uint32_t fb_width = 0, fb_height = 0;
  const VkFramebuffer framebuffer = TruthFramebuffer(color, depth, fb_width, fb_height);
  if (framebuffer == VK_NULL_HANDLE) return fail();
  const uint32_t pass_width = twins ? std::min(truth_images_[color].draw_width, truth_images_[depth].draw_width)
                                    : fb_width;
  const uint32_t pass_height = twins ? std::min(truth_images_[color].draw_height, truth_images_[depth].draw_height)
                                     : fb_height;
  // An image rectangle in the pass's pixels (outward), within the pass.
  // (Review: draws by the pixel-centre rule - the twin pixels whose centres the image rectangle holds; loads outward.)
  const auto pass_rect = [&](int32_t x0, int32_t y0, int32_t x1, int32_t y1, bool centres = false) -> VkRect2D {
    if (twins && centres) {
      x0 = int32_t(std::ceil(float(x0) * ratio - 0.5f)), y0 = int32_t(std::ceil(float(y0) * ratio - 0.5f));
      x1 = int32_t(std::ceil(float(x1) * ratio - 0.5f)), y1 = int32_t(std::ceil(float(y1) * ratio - 0.5f));
    } else if (twins) {
      x0 = int32_t(std::floor(float(x0) * ratio)), y0 = int32_t(std::floor(float(y0) * ratio));
      x1 = int32_t(std::ceil(float(x1) * ratio)), y1 = int32_t(std::ceil(float(y1) * ratio));
    }
    x0 = std::clamp(x0, 0, int32_t(pass_width)), y0 = std::clamp(y0, 0, int32_t(pass_height));
    x1 = std::clamp(x1, x0, int32_t(pass_width)), y1 = std::clamp(y1, y0, int32_t(pass_height));
    return {{x0, y0}, {uint32_t(x1 - x0), uint32_t(y1 - y0)}};
  };
  rect[0] = std::clamp(rect[0], 0, int32_t(fb_width)), rect[1] = std::clamp(rect[1], 0, int32_t(fb_height));
  rect[2] = std::clamp(rect[2], rect[0], int32_t(fb_width)), rect[3] = std::clamp(rect[3], rect[1], int32_t(fb_height));
  if (rect[2] <= rect[0] || rect[3] <= rect[1]) return;  // nothing to draw
  // Session 72 part 4 (gpu_native_truth_skip_restores): a restore of what the images still hold is left out.
  if (draw.restore && REXCVAR_GET(gpu_native_truth_skip_restores)) {
    const bool redundant = !twins && TruthRestoreRedundant(draw, color, depth, rect);
    ++(redundant ? truth_counts_.restores_skipped : truth_counts_.restores_drawn);
    if (segment_log_frames_ > 0) {
      REXGPU_INFO("[native][seg] frame {} truth restore {:016X} into colour {} depth {} (textures 0x{:05X} / 0x{:05X}) "
                  "rect {},{}-{},{}: {}",
                  frame_, draw.pixel_shader_hash, color, depth, draw.restore_color_page, draw.restore_depth_page,
                  rect[0], rect[1], rect[2], rect[3], redundant ? "left out" : "drawn");
    }
    if (redundant) {
      add_time(slot);
      return;
    }
  }
  // Session 80 (gpu_native_truth_serve_copies): a draw sampling its own colour target's image in place of a copy - its
  // slot renamed with an empty rectangle when the binding was served - that writes every pixel of the fresh image
  // (proven coverage, the scissor / viewport and the framebuffer hold it all, every channel, nothing read under it):
  // none of the old pixels is needed. Else the flush (the check below: now) copies them all first.
  if (serve_rename_.valid) {
    const ServeRename r = serve_rename_;
    serve_rename_.valid = false;
    TruthImage& t = truth_images_[r.slot];
    const int32_t w = int32_t(t.width), h = int32_t(t.height);
    const bool whole = color == r.slot && t.rename_source == r.source && !twins && !draw.fast_clear &&
                       draw.truth_color == r.slot && (draw.color_write_mask & 0xF) == 0xF && !PostNeedsDst(draw) &&
                       !(draw.raster.depth_test && draw.raster.depth_compare != VK_COMPARE_OP_ALWAYS) &&
                       !draw.raster.stencil &&
                       draw.cover[2] > draw.cover[0] && draw.cover[3] > draw.cover[1] &&
                       int32_t(draw.cover[0] * scale_x) <= 0 && int32_t(draw.cover[1] * scale_y) <= 0 &&
                       int32_t(draw.cover[2] * scale_x) >= w && int32_t(draw.cover[3] * scale_y) >= h &&
                       rect[0] <= 0 && rect[1] <= 0 && rect[2] >= w && rect[3] >= h;
    if (whole) {
      t.rename_rect[0] = t.rename_rect[1] = 0;
      t.rename_rect[2] = w;
      t.rename_rect[3] = h;
      ++truth_counts_.serve_renames_whole;
    } else {
      ++truth_counts_.serve_renames_copied;
    }
    if (segment_log_frames_ > 0) {
      REXGPU_INFO("[native][seg] frame {} truth serve rename of image {}: {} (colour {} rect {},{}-{},{} cover {},{}-{},{})",
                  frame_, r.slot, whole ? "the draw writes it all" : "its pixels copied", color, rect[0], rect[1],
                  rect[2], rect[3], draw.cover[0], draw.cover[1], draw.cover[2], draw.cover[3]);
    }
  }
  // Session 52 (gpu_native_truth_batch_rename): a draw outside a renamed image's rectangle needs its other pixels.
  for (const int16_t index : {color, depth}) {  // (Session 56: a renamed depth image too)
    if (truth_images_[index].rename_source == VK_NULL_HANDLE) continue;
    const int32_t* rr = truth_images_[index].rename_rect;
    if (rect[0] < rr[0] || rect[1] < rr[1] || rect[2] > rr[2] || rect[3] > rr[3]) {
      EndTruthPass();
      // (Session 80, gpu_native_truth_carry_copies: the copies reading this image, new or renamed - the flush's copy
      // of its other pixels runs either way.)
      DeferConcern c;
      c.all = false;
      c.images[0] = truth_images_[index].image;
      c.images[1] = truth_images_[index].rename_source;
      TruthFlushDeferredFor(kDeferFlushPassIntoSource, c);
    }
  }
  const TruthImage& c_image = truth_images_[color];
  const TruthImage& d_image = truth_images_[depth];
  const bool c_real = c_image.base != kTruthScratchBase, d_real = d_image.base != kTruthScratchBase;
  // What the draw touches, reads and writes whole, by tile.
  const uint32_t gx0 = uint32_t(rect[0]) / scale_x, gy0 = uint32_t(rect[1]) / scale_y;
  const uint32_t gx1 = (uint32_t(rect[2]) + scale_x - 1) / scale_x, gy1 = (uint32_t(rect[3]) + scale_y - 1) / scale_y;
  const bool writes_color = (draw.color_write_mask & 0xF) != 0 && draw.truth_color >= 0 && c_real;
  const bool uses_depth = draw.truth_depth >= 0 && d_real;
  // (The tiles of the last rectangle are kept: most draws of a pass share the scene's whole rectangle.)
  if (truth_tile_color_ != color || truth_tile_depth_ != depth || !std::equal(rect, rect + 4, truth_tile_rect_)) {
    truth_tile_c_ = EdramTiles(c_image.base, c_image.pitch_tiles, c_image.is_64bpp(), gx0, gy0, gx1, gy1);
    truth_tile_d_ = EdramTiles(d_image.base, d_image.pitch_tiles, false, gx0, gy0, gx1, gy1);
    truth_tile_color_ = color;
    truth_tile_depth_ = depth;
    std::copy(rect, rect + 4, truth_tile_rect_);
    truth_tile_c_owned_ = truth_tile_d_owned_ = false;
  }
  // Session 47 (CPU): the open pass's images already own every tile of this rectangle (the last draw left it so) - a
  // draw that is not a clear needs no load and changes no owner: only the draw is recorded.
  const bool fast = same_pass && truth_pass_open_ && !draw.fast_clear && segment_log_frames_ <= 0 &&
                    (!writes_color || truth_tile_c_owned_) && (!uses_depth || truth_tile_d_owned_);
  // wwe2k14 patch (Session 79, gpu_native_truth_alias_census): a fast-path draw is counted as reading every tile it
  // touches (its reads are not worked out here): the census undercounts dead tiles, never overcounts.
  const bool census = REXCVAR_GET(gpu_native_truth_alias_census);
  if (census && fast && (census_pend_c_.any() || census_pend_d_.any())) {
    const std::bitset<2048> rc = census_pend_c_ & truth_tile_c_, rd = census_pend_d_ & truth_tile_d_;
    census_consumed_ += rc.count() + rd.count();
    census_pend_c_ &= ~truth_tile_c_;
    census_pend_d_ &= ~truth_tile_d_;
  }
  bool opened_now = false;
  std::bitset<2048> c_touch, d_touch;
  bool stencil_clear = false;
  const auto inside = [&](const TruthImage& image) {
    const uint32_t tw = (image.is_64bpp() ? 40u : 80u) * scale_x, th = 16u * scale_y;
    const uint32_t c0 = (uint32_t(rect[0]) + tw - 1) / tw, c1 = uint32_t(rect[2]) / tw;
    const uint32_t r0 = (uint32_t(rect[1]) + th - 1) / th, r1 = uint32_t(rect[3]) / th;
    if (c1 <= c0 || r1 <= r0) return std::bitset<2048>();
    return EdramTiles(image.base, image.pitch_tiles, image.is_64bpp(), c0 * (tw / scale_x), r0 * 16,
                      c1 * (tw / scale_x), r1 * 16);
  };
  if (!fast) {
  std::bitset<2048> c_whole, d_whole;
  if (writes_color) c_touch = truth_tile_c_;
  if (uses_depth) d_touch = truth_tile_d_;
  const bool whole = draw.fast_clear || (draw.cover[2] > draw.cover[0] && draw.cover[3] > draw.cover[1]);
  const bool reads_color = writes_color && PostNeedsDst(draw);
  const bool reads_depth = uses_depth && draw.raster.depth_test && draw.raster.depth_compare != VK_COMPARE_OP_ALWAYS;
  // Tiles another image (or the EDRAM) holds: what the EDRAM would have under the draw is brought in first - the
  // EDRAM's own (a boundary load), another image's through the staging buffer (stored, then loaded: the byte
  // reinterpretation without the EDRAM) - where the draw reads it or leaves part of the tile as it is.
  // (Within an open pass, tiles its images already own need nothing: the per-tile look only for the others.)
  std::bitset<2048> c_load, d_load, c_alias, d_alias;
  const bool check = !same_pass || (c_touch & ~truth_pass_c_valid_).any() || (d_touch & ~truth_pass_d_valid_).any();
  // (Session 47, CPU: the tiles a draw covers whole only where the per-tile look runs.)
  if (check && whole && writes_color && !PostNeedsDst(draw)) c_whole = inside(c_image);
  if (check && whole && uses_depth && draw.raster.depth_write &&
      (!draw.raster.depth_test || draw.raster.depth_compare == VK_COMPARE_OP_ALWAYS)) {
    d_whole = inside(d_image);
  }
  // (Session 47, CPU: by the images' owned tiles - the EDRAM's: owned by none; another image's: owned, not by this one.)
  if (check) {
    if (c_image.kind == 1) {
      c_load = c_touch & ~truth_owned_any_;
      c_alias = c_touch & truth_owned_any_ & ~c_image.owned;
      if (!reads_color) c_alias &= ~c_whole;
    }
    d_load = d_touch & ~truth_owned_any_;
    d_alias = d_touch & truth_owned_any_ & ~d_image.owned;
    if (!reads_depth) d_alias &= ~d_whole;
  }
  // Session 82 (the owned frame, hard rule 6): no draw brings in what other surfaces left in the tiles it touches (a clear
  // too: partial tiles) - leftovers of the 360 shared tile memory are not pixels anybody draws (and the clears of an owned pass are
  // turned off at its copy: their loads, and the copy-ins of dirty tiles they forced, were work for nothing).
  if (owned_frame_on_ && !owned_replaying_) {  // (any draw: no leftovers of other surfaces are brought in)
    c_alias.reset();
    d_alias.reset();
  }
  if (c_alias.any() || d_alias.any()) {
    const std::string skip = REXCVAR_GET(gpu_native_truth_alias_skip_tiles);
    for (size_t k = 0; k < skip.size();) {
      const size_t end = std::min(skip.find(',', k), skip.size());
      const std::string part = skip.substr(k, end - k);
      const size_t dash = part.find('-');
      const uint32_t first = uint32_t(std::strtoul(part.c_str(), nullptr, 10));
      const uint32_t last = dash == std::string::npos ? first : uint32_t(std::strtoul(part.c_str() + dash + 1, nullptr, 10));
      for (uint32_t t = first; t <= last && t < 2048; ++t) {
        c_alias.reset(t);
        d_alias.reset(t);
      }
      k = end + 1;
    }
  }
  if ((c_alias.any() || d_alias.any()) && (!EnsureTruthStaging() || truth_staging_load_set_ == VK_NULL_HANDLE ||
                                           REXCVAR_GET(gpu_native_truth_debug_skip_alias))) {  // (Session 59, diagnostic)
    c_alias.reset();
    d_alias.reset();
  }
  // Session 54 (the host render target path): no EDRAM to load unowned tiles from (they stay as the image has them);
  // the alias loads read the staging buffer and stay.
  const bool no_edram = command_processor_.NativeEdramBuffer() == VK_NULL_HANDLE;
  const bool load = (c_load.any() || d_load.any() || c_alias.any() || d_alias.any()) &&
                    (no_edram ? (c_alias.any() || d_alias.any()) : EnsureReplaceSets(false));
  // The images the aliased tiles are read from.
  const auto alias_owners = [&]() -> const std::vector<int16_t>& {
    std::vector<int16_t>& owners = truth_scratch_owners_;  // (Session 47: scratch, no allocation)
    owners.clear();
    const std::bitset<2048> aliased = c_alias | d_alias;
    for (size_t i = 0; i < truth_images_.size(); ++i) {
      if ((aliased & truth_images_[i].owned).any()) owners.push_back(int16_t(i));
    }
    return owners;
  };
  if (segment_log_frames_ > 0 && (load || !same_pass)) {
    std::string owners;
    for (const int16_t o : alias_owners()) owners += fmt::format(" {} ", o);
    REXGPU_INFO("[native][seg] frame {} truth {}: colour {} (base {} pitch {} kind {}) depth {} (base {}) rect {},{}-{},{} "
                "vs {:016X} clear {} reads c{} d{}; loads {} / {}, alias {} / {} from [{}]",
                frame_, same_pass ? "reload" : "pass", color, int32_t(c_real ? c_image.base : -1), c_image.pitch_tiles,
                c_image.kind, depth, int32_t(d_real ? d_image.base : -1), rect[0], rect[1], rect[2], rect[3],
                draw.vertex_shader_hash, draw.fast_clear, reads_color, reads_depth, c_load.count(), d_load.count(),
                c_alias.count(), d_alias.count(), owners);
  }
  if (load || !same_pass) EndTruthPass();
  // Session 80 (owned passes): the aliased tiles' owners are read - their dirty tiles get our pixels first.
  if (load && (c_alias.any() || d_alias.any()) && !owned_dirty_.empty()) OwnedDirtyBeforeRead(c_alias | d_alias, -1, "a truth alias load");
  // Session 52 (gpu_native_truth_batch_resolves): the pass writes an image a deferred resolve reads - flushed, or
  // (gpu_native_truth_batch_rename) the image renamed when the pass opens with a full clear and loads nothing.
  bool renamed = false;
  if (!truth_pass_open_ && (TruthDeferredSource(color) || TruthDeferredSource(depth))) {
    // (Session 80, gpu_native_truth_carry_copies: or a draw whose proven coverage holds the whole rectangle and that
    // reads nothing under it - every pixel written, as by a clear: the match's full-screen combine.)
    // (Review: no depth or stencil test either - a tested quad leaves the pixels that fail it.)
    const bool covers = !draw.fast_clear && REXCVAR_GET(gpu_native_truth_carry_copies) && !PostNeedsDst(draw) &&
                        !(draw.raster.depth_test && draw.raster.depth_compare != VK_COMPARE_OP_ALWAYS) &&
                        !draw.raster.stencil && draw.cover[2] > draw.cover[0] && draw.cover[3] > draw.cover[1] &&
                        int32_t(draw.cover[0] * scale_x) <= rect[0] && int32_t(draw.cover[1] * scale_y) <= rect[1] &&
                        int32_t(draw.cover[2] * scale_x) >= rect[2] && int32_t(draw.cover[3] * scale_y) >= rect[3];
    renamed = !twins && !TruthDeferredSource(depth) && !load && (draw.fast_clear || covers) && writes_color &&
              (draw.color_write_mask & 0xF) == 0xF && TruthRenameForDeferred(color, rect);
    truth_counts_.cover_renames += renamed && covers ? 1 : 0;
    if (segment_log_frames_ > 0) {  // (Session 80, diagnostic)
      REXGPU_INFO("[native][seg] frame {} truth rename check: colour {} depth {}: clear {} covers {} load {} twins {} "
                  "depth source {} writes {} mask 0x{:X} -> renamed {}",
                  frame_, color, depth, draw.fast_clear, covers, load, twins, TruthDeferredSource(depth), writes_color,
                  draw.color_write_mask, renamed);
    }
    if (!renamed) {  // (Session 80, gpu_native_truth_carry_copies: the copies reading this pass's images)
      DeferConcern c;
      c.all = false;
      c.images[0] = truth_images_[color].image;
      c.images[1] = truth_images_[depth].image;
      TruthFlushDeferredFor(kDeferFlushPassIntoSource, c);
    }
    // Session 56 (gpu_native_truth_rename_depth): the clear writes all of the depth and stencil in the rectangle.
    const auto stencil_all = [](uint32_t ops) {
      return ((ops >> 3) & 7) == uint32_t(VK_STENCIL_OP_REPLACE) && ((ops >> 9) & 7) == uint32_t(VK_COMPARE_OP_ALWAYS);
    };
    if (renamed && d_real && REXCVAR_GET(gpu_native_truth_rename_depth) &&
        REXCVAR_GET(gpu_native_truth_lazy_pass_deps) && uses_depth && draw.raster.depth_test &&
        draw.raster.depth_write && draw.raster.depth_compare == VK_COMPARE_OP_ALWAYS && draw.raster.stencil &&
        stencil_all(draw.raster.stencil_ops & 0xFFF) && stencil_all((draw.raster.stencil_ops >> 12) & 0xFFF) &&
        draw.stencil_write_masks[0] == 0xFF && draw.stencil_write_masks[1] == 0xFF &&
        TruthRenameForDeferred(depth, rect, true)) {
      ++truth_counts_.depth_renames;
    }
  }
  // wwe2k14 patch (Session 79, gpu_native_truth_alias_census): this draw's effect on the tiles alias-loaded earlier in
  // the pass (a new pass: the old pass's pending tiles are kept), then its own alias loads become pending unless it
  // reads them itself. (c_whole / d_whole are worked out only where the per-tile look runs: an undercount.)
  if (census) {
    if (!same_pass || load) {
      census_kept_ += census_pend_c_.count() + census_pend_d_.count();
      census_pend_c_.reset();
      census_pend_d_.reset();
    }
    const auto effect = [&](std::bitset<2048>& pend, const std::bitset<2048>& touch, const std::bitset<2048>& whole,
                            bool reads) {
      if (reads) {
        census_consumed_ += (pend & touch).count();
        pend &= ~touch;
      } else {
        census_dead_ += (pend & whole).count();
        pend &= ~whole;
      }
    };
    effect(census_pend_c_, c_touch, c_whole, reads_color);
    effect(census_pend_d_, d_touch, d_whole, reads_depth);
    census_loaded_ += c_alias.count() + d_alias.count();
    // (By owner -> this pass's image: kind k, format f, pitch p; "same place" when both bases are the same tile (the
    // tile's 80x16 pixels at the same place in both), "same columns" when the pitch is the same and the bases a whole
    // number of rows apart (the same pixels a row shift away): either way one rectangle holds the same bits.)
    if (c_alias.any() || d_alias.any()) {
      for (const int16_t o : alias_owners()) {
        const TruthImage& oi = truth_images_[o];
        for (int side = 0; side < 2; ++side) {
          const std::bitset<2048> tiles = (side ? d_alias : c_alias) & oi.owned;
          if (tiles.none()) continue;
          const TruthImage& ci = side ? d_image : c_image;
          const uint32_t delta = (oi.base - ci.base) & 2047u;
          const char* where = !delta                                      ? " same place"
                              : oi.pitch_tiles == ci.pitch_tiles &&
                                        delta % std::max(1u, ci.pitch_tiles) == 0
                                  ? " same columns"
                                  : "";
          census_pairs_[fmt::format("{} k{} f{} p{} -> k{} f{} p{}{}", side ? "depth" : "colour", oi.kind, oi.format,
                                    oi.pitch_tiles, ci.kind, ci.format, ci.pitch_tiles, where)] += tiles.count();
        }
      }
    }
    if (reads_color) census_consumed_ += c_alias.count(); else census_pend_c_ |= c_alias;
    if (reads_depth) census_consumed_ += d_alias.count(); else census_pend_d_ |= d_alias;
  }
  // Session 79 (gpu_native_truth_alias_direct): the aliased tiles whose owners the load reads straight (not stored).
  std::bitset<2048> direct_tiles;
  std::vector<int16_t>& direct_owners = truth_scratch_direct_owners_;
  direct_owners.clear();
  if (load && (c_alias.any() || d_alias.any())) {
    truth_counts_.alias_reads += uint32_t(c_alias.count() + d_alias.count());
    if (owned_frame_on_) owned_frame_stats_.truth_tile_loads += uint32_t(c_alias.count() + d_alias.count());
    const std::vector<int16_t>& owners = alias_owners();
    const std::bitset<2048> aliased = c_alias | d_alias;
    // (RGBA8 and depth owners other than this pass's own images; RGBA16 owners - 64bpp - are stored as before.)
    std::vector<int16_t>& stored_owners = truth_scratch_stored_owners_;
    stored_owners.clear();
    const bool try_direct = REXCVAR_GET(gpu_native_truth_alias_direct) && c_image.kind < kPostTargetClasses &&
                            truth_staging_ != VK_NULL_HANDLE && EnsureAliasDirectPipelines();
    for (const int16_t o : owners) {
      TruthImage& oi = truth_images_[o];
      const int owner_depth = oi.kind == 3 ? 1 : 0;
      // (Session 79, gpu_native_truth_alias_direct_rgba16: the k_16_16 / k_16_16_16_16 RGBA16 owners too.)
      const bool rgba16 = REXCVAR_GET(gpu_native_truth_alias_direct_rgba16) && oi.kind == 2 &&
                          (oi.format == uint32_t(xenos::ColorRenderTargetFormat::k_16_16) ||
                           oi.format == uint32_t(xenos::ColorRenderTargetFormat::k_16_16_16_16));
      const bool direct = try_direct && o != color && o != depth && oi.initialized && oi.image != VK_NULL_HANDLE &&
                          ((oi.kind == 1 && oi.format == 0) || owner_depth || rgba16) &&
                          alias_direct_pipelines_[owner_depth][0][c_image.kind] != VK_NULL_HANDLE &&
                          alias_direct_pipelines_[owner_depth][1][c_image.kind] != VK_NULL_HANDLE &&
                          TruthStoreSet(oi, true, truth_staging_) != VK_NULL_HANDLE;
      (direct ? direct_owners : stored_owners).push_back(o);
    }
    if (!stored_owners.empty()) TruthStoreMany(stored_owners.data(), stored_owners.size(), aliased, true);
    if (!direct_owners.empty()) {
      // (As TruthStoreMany: the deferred batch first - a deferred resolve may read an owner, and a renamed owner gets
      // its pixels outside its rename rectangle only at the flush (TruthDeferredSource sees the slot's new image, not
      // the one the batch reads); the images in their read layouts until the pass ends - TruthRestoreAliasOwners.)
      // Session 79 (gpu_native_truth_alias_direct_lazy_flush): only for a renamed owner. The batch's entries read
      // truth images and write textures / memory, never a truth image, and this pass draws only into its own images
      // (never an owner read here), so an entry reading an owner reads the same pixels before or after these draws;
      // every flush starts with EndTruthPass, which puts the owners back in their attachment layouts first.
      bool renamed_owner = !REXCVAR_GET(gpu_native_truth_alias_direct_lazy_flush);
      for (const int16_t o : direct_owners) {
        renamed_owner = renamed_owner || truth_images_[o].rename_source != VK_NULL_HANDLE;
      }
      // (Review fix: or this pass's own images, renamed earlier - the whole-tile loads below may land outside their
      // rename rectangle, where the flush's copy would later overwrite them.)
      for (const int16_t own : {color, depth}) {
        renamed_owner = renamed_owner || (own >= 0 && size_t(own) < truth_images_.size() &&
                                          truth_images_[own].rename_source != VK_NULL_HANDLE);
      }
      if (renamed_owner) {
        // (Session 80, gpu_native_truth_carry_copies: the copies reading the owners or this pass's images; more
        // owners than fit: every copy.)
        DeferConcern c;
        c.all = direct_owners.size() + 2 > std::size(c.images);
        size_t n = 0;
        for (const int16_t o : direct_owners) {
          if (!c.all) c.images[n++] = truth_images_[o].image;
        }
        if (!c.all) {
          c.images[n++] = truth_images_[color].image;
          c.images[n++] = truth_images_[depth].image;
          TruthFlushDeferredFor(kDeferFlushStores, c);
        } else {
          TruthFlushDeferred(kDeferFlushStores);
        }
      }
      EndTruthPass();
      for (const int16_t o : direct_owners) TruthSyncRead(o);  // (draw twins)
      std::vector<VkImageMemoryBarrier>& barriers = truth_store_barriers_;
      barriers.clear();
      for (const int16_t o : direct_owners) {
        const TruthImage& oi = truth_images_[o];
        const bool d = oi.kind == 3;
        direct_tiles |= oi.owned & aliased;
        VkImageMemoryBarrier& b = barriers.emplace_back();
        b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = oi.image;
        b.srcAccessMask = d ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout = d ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b.newLayout = d ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(
            d ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
        truth_alias_read_owners_.emplace_back(oi.image, d);
      }
      cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                              uint32_t(barriers.size()), barriers.data());
      truth_counts_.alias_direct += uint32_t(((c_alias | d_alias) & direct_tiles).count());
    }
  }
  if (!truth_pass_open_) {
    opened_now = true;
    // Session 56 (vulkan_texture_sample_written_early): the textures written since become sampleable now, not at
    // their first sampling inside the pass (a barrier there ends it) - except those a deferred resolve still writes.
    if (REXCVAR_GET(vulkan_texture_sample_written_early)) {
      truth_keep_textures_.clear();
      for (const DeferredResolve& d : truth_deferred_) {
        if (d.texture) truth_keep_textures_.push_back(d.texture);
      }
      if (const uint32_t moved = command_processor_.NativeSampleWrittenTextures(truth_keep_textures_.data(),
                                                                                truth_keep_textures_.size())) {
        truth_counts_.early_sampled += moved;
        command_processor_.SubmitBarriers(false);
      }
    }
    if (load) {
      VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                 VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
      cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
    // Session 80 (gpu_native_truth_drop_dead_passes): the last pass only drew colour pixels into this pass's colour image,
    // and this first draw clears all of them (every channel, nothing read under it, the same image - not renamed) with
    // nothing recorded since that reads or writes an image or buffer (a resolve, store, load, flush or other pass): its
    // pixels were read by nothing - its begin, draws and end become no-ops (the pass's own timestamps stay outside).
    if (dead_pass_.valid) {
      dead_pass_.valid = false;
      const DeadPassCandidate& p = dead_pass_;
      const bool clears = !load && !twins && !renamed && draw.fast_clear && writes_color &&
                          (draw.color_write_mask & 0xF) == 0xF && !PostNeedsDst(draw) && color == p.color &&
                          truth_images_[color].image == p.image && rect[0] <= p.rect[0] && rect[1] <= p.rect[1] &&
                          rect[2] >= p.rect[2] && rect[3] >= p.rect[3];
      if (clears && REXCVAR_GET(gpu_native_truth_drop_dead_passes)) {
        const bool dead = p.submission == command_processor_.GetCurrentSubmission() && cb.OnlyStateSince(p.end) &&
                          cb.NopRenderPass(p.begin, p.end);
        ++(dead ? truth_counts_.dead_passes : truth_counts_.dead_pass_kept);
        if (dead && p.frame == frame_ && p.segment < slot.segments.size()) slot.segments[p.segment].pass_draws = 0;
        if (segment_log_frames_ > 0) {
          REXGPU_INFO("[native][seg] frame {} truth dead pass: colour {} rect {},{}-{},{} cleared by the next pass: {}",
                      frame_, p.color, p.rect[0], p.rect[1], p.rect[2], p.rect[3],
                      dead ? "turned off" : "kept (work recorded in between, or refused)");
        }
      }
    }
    // Session 80 (gpu_native_truth_serve_ahead): the colour slot whose pass a serve ended last time - the source image
    // of the texture's carried copy (the last write into it, as TruthServeView finds it; a plain colour copy), never
    // an attachment of this pass, made readable with the served ones below.
    if (serve_ahead_.texture && serve_ahead_.color == color && serve_ahead_.first_vs == draw.vertex_shader_hash &&
        REXCVAR_GET(gpu_native_truth_serve_ahead) && REXCVAR_GET(gpu_native_truth_serve_copies)) {
      const DeferredResolve* e = nullptr;
      for (const DeferredResolve& d : truth_deferred_) {
        if (d.texture == serve_ahead_.texture) e = &d;
      }
      const auto attached = [&](VkImage image) {
        return image == c_image.image || image == d_image.image || image == c_image.draw_image ||
               image == d_image.draw_image;
      };
      if (e && e->pinned && e->plain && !e->depth && e->image != VK_NULL_HANDLE && !attached(e->image) &&
          std::find(serve_pending_.begin(), serve_pending_.end(), e->image) == serve_pending_.end()) {
        serve_pending_.push_back(e->image);
        ++truth_counts_.serve_ahead;
      }
    }
    // Session 80 (gpu_native_truth_serve_copies): the images this pass samples in place of a copy, readable until it
    // ends (TruthRestoreAliasOwners puts them back in their attachment layout).
    if (!serve_pending_.empty()) {
      std::vector<VkImageMemoryBarrier>& barriers = truth_store_barriers_;
      barriers.clear();
      for (const VkImage image : serve_pending_) {
        if (std::any_of(truth_alias_read_owners_.begin(), truth_alias_read_owners_.end(),
                        [&](const std::pair<VkImage, bool>& o) { return o.first == image; })) {
          continue;
        }
        VkImageMemoryBarrier& b = barriers.emplace_back();
        b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);
        truth_alias_read_owners_.emplace_back(image, false);
      }
      if (!barriers.empty()) {
        cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                                nullptr, 0, nullptr, uint32_t(barriers.size()), barriers.data());
      }
      serve_pending_.clear();
    }
    if (query_pool_ && slot_queries_ok_ && slot.queries + 2 <= QueryLimit()) {
      FrameSlot::Segment& segment = slot.segments.emplace_back();
      segment.kind = 4;
      segment.pass_width = uint16_t(pass_width);  // Session 72 (gpu_native_truth_pass_log)
      segment.pass_height = uint16_t(pass_height);
      segment.pass_pitch = uint16_t(pitch);
      segment.pass_color_kind = uint8_t(c_image.kind);
      segment.pass_depth = d_real ? 1 : 0;
      segment.pass_vs = draw.vertex_shader_hash;
      segment.pass_ps = draw.pixel_shader_hash;
      segment.native_begin = slot.queries++;
      cb.CmdVkWriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_,
                             SlotOf(frame_) * kQueriesPerFrame + segment.native_begin);
      truth_pass_segment_index_ = uint32_t(slot.segments.size() - 1);
    }
    VkClearValue clear_values[2] = {};
    VkRenderPassBeginInfo begin_info = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    // A scratch depth image (no draw of the pass uses depth): its depth neither loaded nor stored (the same formats:
    // compatible with the pipelines' render pass).
    // (Session 46 stencil: the depth image's stencil loaded too, not cleared.)
    // (Session 56: without the render passes' external dependencies - gpu_native_truth_lazy_pass_deps, or the
    // diagnostic gpu_native_truth_debug_pass_no_deps.)
    const bool lazy = REXCVAR_GET(gpu_native_truth_lazy_pass_deps);
    const bool nodep = lazy || REXCVAR_GET(gpu_native_truth_debug_pass_no_deps);
    begin_info.renderPass = d_real ? (nodep ? truth_render_pass_keep_nodep_ : truth_render_pass_keep_)[c_image.kind]
                                   : (nodep ? truth_render_pass_no_depth_nodep_
                                            : post_render_pass_continue_no_depth_)[c_image.kind];
    // Session 66 (G2.1, gpu_native_truth_pass_dont_care): the first draw clears the whole pass area - what it
    // writes whole is not loaded (load ops act on the render area only, which is the framebuffer's here).
    if (REXCVAR_GET(gpu_native_truth_pass_dont_care) && nodep && d_real && !load && !twins && draw.fast_clear &&
        rect[0] == 0 && rect[1] == 0 && uint32_t(rect[2]) == fb_width && uint32_t(rect[3]) == fb_height &&
        pass_width == fb_width && pass_height == fb_height) {
      const auto stencil_whole = [](uint32_t ops) {
        return ((ops >> 3) & 7) == uint32_t(VK_STENCIL_OP_REPLACE) && ((ops >> 9) & 7) == uint32_t(VK_COMPARE_OP_ALWAYS);
      };
      const bool c_dc = writes_color && (draw.color_write_mask & 0xF) == 0xF && !PostNeedsDst(draw);
      const bool d_dc = uses_depth && draw.raster.depth_test && draw.raster.depth_write &&
                        draw.raster.depth_compare == VK_COMPARE_OP_ALWAYS && draw.raster.stencil &&
                        stencil_whole(draw.raster.stencil_ops & 0xFFF) &&
                        stencil_whole((draw.raster.stencil_ops >> 12) & 0xFFF) &&
                        draw.stencil_write_masks[0] == 0xFF && draw.stencil_write_masks[1] == 0xFF;
      if (c_dc || d_dc) {
        begin_info.renderPass = truth_render_pass_dont_care_[c_dc && d_dc ? 2 : (c_dc ? 0 : 1)][c_image.kind];
        truth_counts_.dont_care_color += c_dc;
        truth_counts_.dont_care_depth += d_dc;
      }
      // Session 80 (gpu_native_truth_clear_load): ... and when that clear's colour, depth and stencil are all known on
      // the CPU (a listed pass-through pixel shader, no blending, no epilogue work, the target's own rounding), the
      // pass's start clears them (a GPU's fast clear) and the draw is left out (truth_skip_clear_draw_). The colour is
      // rounded as the translation stores it: to 1/255 for the 8-bit targets, the 360's -32..32 for the 16-bit ones.
      if (c_dc && d_dc && draw.clear_color_known && draw.fast_clear && draw.clear_depth >= 0.0f &&
          draw.stencil_refs[0] == draw.stencil_refs[1] && draw.blend_control == 0x00010001u &&
          (draw.color_mask_all & ~0xFu) == 0 && (draw.pixel_ext & SpirvShaderTranslator::kNativeExtEpilogueIdle) &&
          (c_image.kind == 1 || c_image.kind == 2) && draw.color_quantize == c_image.kind &&
          truth_render_pass_clear_[c_image.kind] != VK_NULL_HANDLE &&
          !REXCVAR_GET(gpu_native_truth_clear_load).empty()) {
        for (uint32_t i = 0; i < 4; ++i) {
          const float c = draw.clear_color[i];
          if (c_image.kind == 1) {
            // (Review: the product its own statement, so no compiler fuses it with the add - the GPU's is unfused.)
            const float s = std::min(std::max(c, 0.0f), 1.0f) * 255.0f;
            clear_values[0].color.float32[i] = std::trunc(s + 0.5f) / 255.0f;
          } else {
            const float s = std::min(std::max(c, -32.0f), 32.0f) * (32767.0f / 32.0f);
            clear_values[0].color.float32[i] = std::trunc(s + (s < 0.0f ? -0.5f : 0.5f)) / 32767.0f;
          }
        }
        clear_values[1].depthStencil = {draw.clear_depth, draw.stencil_refs[0]};
        begin_info.renderPass = truth_render_pass_clear_[c_image.kind];
        truth_skip_clear_draw_ = true;
        ++truth_counts_.clear_loads;
      }
    }
    begin_info.framebuffer = renamed ? TruthFramebuffer(color, depth, fb_width, fb_height) : framebuffer;
    if (begin_info.framebuffer == VK_NULL_HANDLE) return fail();  // (review)
    begin_info.renderArea = {{0, 0}, {pass_width, pass_height}};
    begin_info.clearValueCount = d_real ? 2 : 0;
    begin_info.pClearValues = d_real ? clear_values : nullptr;
    if (lazy) {
      // The attachments (as TruthFramebuffer takes them: the draw twins with twins).
      const DeferredCommandBuffer::LazyAttachment attachments[2] = {
          {twins ? c_image.draw_image : c_image.image, VK_IMAGE_ASPECT_COLOR_BIT,
           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
          {twins ? d_image.draw_image : d_image.image, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
           VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}};
      cb.CmdVkBeginRenderPassLazy(&begin_info, VK_SUBPASS_CONTENTS_INLINE, attachments,
                                  REXCVAR_GET(gpu_native_truth_debug_lazy_ignore_depth) == 1 ||
                                          (REXCVAR_GET(gpu_native_truth_debug_lazy_ignore_depth) == 2 && !d_real)
                                      ? 1
                                      : 2,
                                  uint32_t(REXCVAR_GET(gpu_native_truth_lazy_pass_mode)),
                                  uint32_t(REXCVAR_GET(gpu_native_truth_lazy_pass_style)));
      if (segment_log_frames_ > 0) {  // (diagnostic: why the lazy pass needed a barrier)
        static const char* const kWhy[] = {"none", "other work", "colour image", "depth image"};
        // (Session 59: with the command that broke the chain - DeferredCommandBuffer::Command as a number.)
        // (Session 79: and who recorded it - base + stack, tools\perf\pass_ends.py --breaks.)
        const uintptr_t* s = cb.lazy_break_stack();
        REXGPU_INFO("[native][seg] frame {} lazy pass barrier: {} (chain broken by command {}) base {:X} stack {:X} {:X} "
                    "{:X} {:X} {:X}", frame_, kWhy[std::min(cb.lazy_last_reason(), 3u)],
                    int32_t(cb.lazy_break_command()), cb.lazy_break_base(), s[0], s[1], s[2], s[3], s[4]);
      }
    } else {
      cb.CmdVkBeginRenderPass(&begin_info, VK_SUBPASS_CONTENTS_INLINE);
    }
    truth_pass_lazy_ = lazy;
    truth_pass_open_ = true;
    truth_pass_clears_.valid = false;  // Session 80 (gpu_native_own_steps)
    truth_pass_clears_.area = 0;
    truth_pass_dead_ok_ = !load && !twins;  // Session 80 (gpu_native_truth_drop_dead_passes)
    truth_pass_draws_ = 0;  // Session 72 (gpu_native_truth_pass_log)
    truth_pass_touched_color_ = truth_pass_touched_depth_ = false;  // Session 73
    truth_pass_begin_offset_ = cb.last_begin_render_pass_offset();
    {  // (Session 87, review F9)
      const uint64_t sub = command_processor_.GetCurrentSubmission();
      if (truth_pass_map_submission_ != sub) {
        truth_pass_by_offset_.clear();
        truth_pass_map_submission_ = sub;
      }
      truth_pass_by_offset_[truth_pass_begin_offset_] = ++truth_pass_serial_;
      truth_pass_begin_serial_ = truth_pass_serial_;
    }
    truth_pass_begin_rp_ = begin_info.renderPass;
    truth_pass_kind_ = c_image.kind;
    truth_pass_real_depth_ = d_real; truth_pass_real_color_ = c_image.base != kTruthScratchBase;
    truth_pass_color_ = color;
    truth_pass_depth_ = depth;
    truth_pass_first_vs_ = draw.vertex_shader_hash;  // (Session 80, gpu_native_truth_serve_ahead)
    OwnedFramePassOpened(draw);  // Session 82 (the owned frame's census: a truth pass opens)
    truth_pass_class_ = draw.truth_class;
    truth_pass_pitch_ = pitch;
    truth_pass_width_ = pass_width;
    truth_pass_height_ = pass_height;
    drew_this_frame_ = true;
  }
  if (load) {
    const uint32_t tile_w = 80 * scale_x, tile_h = 16 * scale_y, tile_dwords = tile_w * tile_h;
    EdramPlace place;
    place.color_base_dwords = (c_image.base % 2048u) * tile_dwords;
    place.depth_base_dwords = (d_image.base % 2048u) * tile_dwords;
    place.pitch_tile_dwords = c_image.pitch_tiles * tile_dwords;
    place.tile_w = tile_w;
    place.tile_h = tile_h;
    place.width = fb_width;
    place.height = fb_height;
    place.wrap_dwords = 2048 * tile_dwords;
    place.flags = 2u | (d_image.format & 1u);
    place.scale_x = place.scale_y = ratio;  // (Session 54: twin pixels per image pixel)
    place.origin_x = place.origin_y = 0;
    const VkViewport full = {0.0f, 0.0f, float(pass_width), float(pass_height), 0.0f, 1.0f};
    cb.CmdVkSetViewport(0, 1, &full);
    std::vector<std::array<int32_t, 4>>& load_rects = truth_scratch_load_rects_;  // (Session 47: scratch)
    for (int pass = 0; pass < 4; ++pass) {
      // (Session 79: the aliased tiles of owners read straight are drawn below, not from the staging buffer.)
      const std::bitset<2048> tiles = (pass == 0 ? c_load : pass == 1 ? d_load : pass == 2 ? c_alias : d_alias) &
                                      (pass < 2 ? ~std::bitset<2048>() : ~direct_tiles);
      if (tiles.none() || (pass < 2 && no_edram)) continue;
      ((pass & 1) ? truth_pass_touched_depth_ : truth_pass_touched_color_) = true;  // Session 73: a load writes it
      TruthTileRects((pass & 1) ? d_image : c_image, tiles, load_rects);
      cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, load_layout_, 0, 1,
                                 pass < 2 ? &load_set_ : &truth_staging_load_set_, 0, nullptr);
      cb.CmdVkPushConstants(load_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(place), &place);
      cb.CmdVkBindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, (pass & 1) ? post_load_depth_pipeline_[c_image.kind]
                                                                       : post_load_pipeline_[c_image.kind]);
      for (const auto& r : load_rects) {
        const VkRect2D scissor = pass_rect(r[0], r[1], r[2], r[3]);
        if (!scissor.extent.width || !scissor.extent.height) continue;
        cb.CmdVkSetScissor(0, 1, &scissor);
        cb.CmdVkDraw(3, 1, 0, 0);
      }
    }
    // Session 79 (gpu_native_truth_alias_direct): per side (the pass's colour, its depth) and owner, the owner's tiles
    // from the owner image; a depth owner's tiles whose stencil it holds read its stencil view, the others take its last
    // clear's stencil (as TruthStoreMany's rects and stencil rects).
    for (int side = 0; side < 2 && direct_tiles.any(); ++side) {
      const std::bitset<2048> side_tiles = (side ? d_alias : c_alias) & direct_tiles;
      if (side_tiles.none()) continue;
      (side ? truth_pass_touched_depth_ : truth_pass_touched_color_) = true;
      for (const int16_t o : direct_owners) {
        const TruthImage& oi = truth_images_[o];
        const std::bitset<2048> own = side_tiles & oi.owned;
        if (own.none()) continue;
        const int owner_depth = oi.kind == 3 ? 1 : 0;
        AliasDirectConstants k;
        k.place = place;
        k.owner_base_dwords = (oi.base % 2048u) * tile_dwords;
        k.owner_pitch_tile_dwords = oi.pitch_tiles * tile_dwords;
        cb.CmdVkBindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, alias_direct_pipelines_[owner_depth][side][c_image.kind]);
        cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, alias_direct_layout_, 0, 1, &oi.store_sets[0], 0,
                                   nullptr);
        const bool split = owner_depth && oi.stencil_view != VK_NULL_HANDLE;
        for (int own_stencil = 0; own_stencil < (split ? 2 : 1); ++own_stencil) {
          const std::bitset<2048> part = !split ? own : own_stencil ? (own & oi.stencil_valid) : (own & ~oi.stencil_valid);
          if (part.none()) continue;
          k.owner_flags = (side ? 0x10000u : 0u) |
                          (owner_depth ? (oi.format & 1u) | (own_stencil ? 4u : (oi.stencil & 0xFFu) << 8) : 0u) |
                          (oi.kind == 2 ? (oi.format & 0xFu) << 20 : 0u);  // (Session 79: the RGBA16 owners' format)
          cb.CmdVkPushConstants(alias_direct_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(k), &k);
          TruthTileRects(side ? d_image : c_image, part, load_rects);
          for (const auto& r : load_rects) {
            const VkRect2D scissor = pass_rect(r[0], r[1], r[2], r[3]);
            if (!scissor.extent.width || !scissor.extent.height) continue;
            cb.CmdVkSetScissor(0, 1, &scissor);
            cb.CmdVkDraw(3, 1, 0, 0);
          }
        }
      }
    }
    // (Loaded tiles become a valid copy, not owned: only what a draw writes changes owner - review fix.)
    truth_foreign_bind_ = true;  // the load layout's set 0 was bound
    if (c_load.any() || d_load.any()) ++truth_counts_.loads;
    if (owned_frame_on_) owned_frame_stats_.truth_tile_loads += uint32_t(c_load.count() + d_load.count());
  }
  // Session 80 (owned passes): where a new pass's draws start - after its loads - and whether it loaded tiles.
  if (opened_now) {
    truth_pass_draws_offset_ = cb.stream_offset();
    truth_pass_loaded_ = load;
  }
  // A new pass: the tiles its images own now.
  if (opened_now) {
    truth_pass_c_owned_ = truth_images_[color].owned;  // (Session 47: kept per image)
    truth_pass_d_owned_ = truth_images_[depth].owned;
    truth_pass_c_valid_ = truth_pass_c_owned_;
    truth_pass_d_valid_ = truth_pass_d_owned_;
  }
  if (load) {  // (review 2: only tiles a load was recorded for)
    truth_pass_c_valid_ |= c_load | c_alias;
    truth_pass_d_valid_ |= d_load | d_alias;
  }
  }  // (not the fast path)
  // Session 73: what the pass's draws touch. (Session 80 fix: here, once this draw's pass is open. Set before the pass
  // change above, a pass's first draw was counted for the pass before it and lost for its own: a pass whose only
  // colour write was its first draw - a one-draw pass - had its colour kept without a store
  // (gpu_native_truth_color_none: STORE_OP_NONE after a write leaves the image undefined; a tile GPU loses it).)
  truth_pass_touched_color_ |= writes_color;
  truth_pass_touched_depth_ |= uses_depth;
  // Session 80 (gpu_native_truth_drop_dead_passes): what the pass's draws do besides colour pixels in their rectangle
  // (the scissor: `rect` without twins), and that rectangle.
  truth_pass_dead_ok_ = truth_pass_dead_ok_ && !twins && !draw.memexport && !draw.occlusion_query &&
                        draw.xfb_buffer == VK_NULL_HANDLE && !draw.own_skip_call &&
                        REXCVAR_GET(gpu_native_truth_debug_gpu_work) == 0;
  if (!truth_pass_draws_) {
    std::copy(rect, rect + 4, truth_pass_rect_);
  } else {
    truth_pass_rect_[0] = std::min(truth_pass_rect_[0], rect[0]), truth_pass_rect_[1] = std::min(truth_pass_rect_[1], rect[1]);
    truth_pass_rect_[2] = std::max(truth_pass_rect_[2], rect[2]), truth_pass_rect_[3] = std::max(truth_pass_rect_[3], rect[3]);
  }
  // The draw, with everything it binds (the command processor binds its own sets between our draws).
  // (Session 47, CPU: the command processor bound this draw's sets, with this layout, before the claim - UpdateBindings,
  // or the repeat path's constants - so they are bound again only after a load pass of ours bound its own set 0.)
  // (Session 47, CPU: the pipeline, depth bias and blend constants only where they change within the open pass - nothing
  // else binds a graphics pipeline or sets those two in it: the command processor sets them only for host render
  // targets, and a load pass of ours drops what is known, truth_bound_valid_.)
  // (Review: only with gpu_native_lean - without it, hot-reloaded off mid-frame, the command processor binds the
  // emulated pipeline before the claim.)
  const bool dedupe = REXCVAR_GET(gpu_native_truth_trust_bindings) && REXCVAR_GET(gpu_native_lean);
  if (truth_foreign_bind_ || !dedupe) truth_bound_valid_ = false;
  // Session 56 (gpu_native_truth_dedupe_state): the index buffer and stencil state known only while the pass's own
  // bindings are (decided before truth_bound_valid_ is set again for this draw).
  const bool dedupe_state = REXCVAR_GET(gpu_native_truth_dedupe_state);
  if (!truth_bound_valid_ || !dedupe_state) {
    truth_bound_index_buffer_ = VK_NULL_HANDLE;
    truth_bound_stencil_valid_ = false;
  }
  if (!truth_bound_valid_ || truth_bound_pipeline_ != draw.pipeline) {
    cb.CmdVkBindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, draw.pipeline);
    truth_bound_pipeline_ = draw.pipeline;
  }
  const bool rebind_sets = truth_foreign_bind_ || !dedupe;
  for (uint32_t s = 0; rebind_sets && s < SpirvShaderTranslator::kDescriptorSetCount; ++s) {
    if (!(draw.sets_bound & (1u << s)) || draw.sets[s] == VK_NULL_HANDLE) continue;
    const bool constants = s == SpirvShaderTranslator::kDescriptorSetConstants && draw.dynamic_offset_count;
    cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, draw.layout, s, 1, &draw.sets[s],
                               constants ? draw.dynamic_offset_count : 0, constants ? draw.dynamic_offsets : nullptr);
  }
  VkViewport viewport = draw.viewport;
  viewport.x *= draw.truth_sx * ratio;  // (Session 54: x the draw ratio)
  viewport.width *= draw.truth_sx * ratio;
  viewport.y *= draw.truth_sy * ratio;
  viewport.height *= draw.truth_sy * ratio;
  const int32_t debug_gpu_work = REXCVAR_GET(gpu_native_truth_debug_gpu_work);
  VkRect2D scissor = twins ? pass_rect(rect[0], rect[1], rect[2], rect[3], true)
                           : VkRect2D{{rect[0], rect[1]}, {uint32_t(rect[2] - rect[0]), uint32_t(rect[3] - rect[1])}};
  if (debug_gpu_work == 1 || debug_gpu_work == 3) scissor.extent = {1u, 1u};
  if (dedupe && !truth_foreign_bind_) {
    // (Session 47, CPU: through the command processor's own tracking - recorded only where it changes; it records the
    // guest's values the same way before the next draw. After a load pass of ours, which set them directly, they are
    // recorded directly too: the command processor's tracking is dropped then - NativeStateTouched(false).)
    command_processor_.SetViewport(viewport);
    command_processor_.SetScissor(scissor);
  } else {
    cb.CmdVkSetViewport(0, 1, &viewport);
    cb.CmdVkSetScissor(0, 1, &scissor);
  }
  if (!truth_bound_valid_ || std::memcmp(truth_bound_bias_, &draw.raster.depth_bias_constant, sizeof(float)) ||
      std::memcmp(truth_bound_bias_ + 1, &draw.raster.depth_bias_slope, sizeof(float))) {
    cb.CmdVkSetDepthBias(draw.raster.depth_bias_constant, 0.0f, draw.raster.depth_bias_slope);
    truth_bound_bias_[0] = draw.raster.depth_bias_constant;
    truth_bound_bias_[1] = draw.raster.depth_bias_slope;
  }
  if (!truth_bound_valid_ || std::memcmp(truth_bound_blend_, draw.blend_constants, sizeof(truth_bound_blend_))) {
    cb.CmdVkSetBlendConstants(draw.blend_constants);
    std::memcpy(truth_bound_blend_, draw.blend_constants, sizeof(truth_bound_blend_));
  }
  truth_bound_valid_ = dedupe;
  if (segment_log_frames_ > 0) {  // Session 46 (diagnostic): every truth draw
    REXGPU_INFO("[native][seg] frame {} truth draw: vs {:016X} ps {:016X} c{} d{} rect {},{}-{},{} mask 0x{:X} blend "
                "0x{:08X} depth {}{} cmp {} stencil {} ops 0x{:06X} ref {:02X}/{:02X} read {:02X}/{:02X} write "
                "{:02X}/{:02X} clear {} cull {} verts {} prim {} cover {},{}-{},{}",
                frame_, draw.vertex_shader_hash, draw.pixel_shader_hash, color, depth, rect[0], rect[1], rect[2],
                rect[3], draw.color_write_mask, draw.blend_control, draw.raster.depth_test ? "t" : "-",
                draw.raster.depth_write ? "w" : "-", uint32_t(draw.raster.depth_compare), draw.raster.stencil,
                draw.raster.stencil_ops, draw.stencil_refs[0], draw.stencil_refs[1], draw.stencil_read_masks[0],
                draw.stencil_read_masks[1], draw.stencil_write_masks[0], draw.stencil_write_masks[1], draw.fast_clear,
                uint32_t(draw.raster.cull_mode), draw.vertex_count, draw.guest_primitive, draw.cover[0], draw.cover[1],
                draw.cover[2], draw.cover[3]);
  }
  if (draw.raster.stencil && uses_depth) {  // Session 46: the guest's stencil reference and masks, per face
    for (uint32_t f = 0; f < 2; ++f) {
      const VkStencilFaceFlags face = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
      uint8_t* bound = truth_bound_stencil_[f];
      if (!truth_bound_stencil_valid_ || bound[0] != draw.stencil_refs[f]) {
        cb.CmdVkSetStencilReference(face, draw.stencil_refs[f]);
      }
      if (!truth_bound_stencil_valid_ || bound[1] != draw.stencil_read_masks[f]) {
        cb.CmdVkSetStencilCompareMask(face, draw.stencil_read_masks[f]);
      }
      if (!truth_bound_stencil_valid_ || bound[2] != draw.stencil_write_masks[f]) {
        cb.CmdVkSetStencilWriteMask(face, draw.stencil_write_masks[f]);
      }
      bound[0] = draw.stencil_refs[f];
      bound[1] = draw.stencil_read_masks[f];
      bound[2] = draw.stencil_write_masks[f];
    }
    truth_bound_stencil_valid_ = dedupe_state && truth_bound_valid_;
  }
  if (draw.own_skip_call) {  // Session 93 (gpu_native_own_crowd_merge): the dynamic state this first member recorded
    OwnCrowdNoteSig(viewport, scissor, draw.raster.depth_bias_constant, draw.raster.depth_bias_slope, draw.blend_constants,
                    draw.raster.stencil && uses_depth, draw.stencil_refs, draw.stencil_read_masks, draw.stencil_write_masks);
  }
  const uint32_t vertex_count = debug_gpu_work == 3 ? std::min(draw.vertex_count, 3u) : draw.vertex_count;
  const uint32_t instance_count = debug_gpu_work == 3 ? std::min(draw.instance_count, 1u) : draw.instance_count;
  // Session 61: a character draw's positions captured while it is drawn (DLSS / DLAA object motion vectors).
  const bool xfb = draw.xfb_buffer != VK_NULL_HANDLE && debug_gpu_work != 2 && !truth_skip_clear_draw_;
  if (xfb) command_processor_.NativeXfbBegin(draw.xfb_buffer, draw.xfb_offset, draw.xfb_size);
  // wwe2k14 patch (Session 80, own crowd): what an own crowd draw right after this one reuses.
  truth_last_vs_hash_ = draw.vertex_shader_hash;
  truth_last_index_.indexed = draw.indexed;
  truth_last_index_.buffer = draw.index_buffer;
  truth_last_index_.offset = draw.index_offset;
  truth_last_index_.type = draw.index_type;
  truth_last_index_.count = draw.vertex_count;
  truth_last_index_.attribute = draw.attribute;  // (Session 95)
  // (Session 80: own crowd - the draw is ours, right after this; gpu_native_truth_clear_load - the pass's start
  // cleared what it would draw.)
  if (truth_skip_clear_draw_) {
    // Session 90 (census): draw calls left out as their pass's load-op clear; non-clears must stay 0.
    if (draw.fast_clear) ++truth_counts_.skip_clear; else ++truth_counts_.skip_nonclear;
  }
  // Session 95 (the diagnostic run): an armed frame's draw says who it is (the next draw call in the stream).
  if (draw.diag_id && !(debug_gpu_work == 2 || draw.own_skip_call || truth_skip_clear_draw_)) [[unlikely]] cb.CmdDiagMark(diag::kMarkNextDraw, draw.diag_id, draw.diag_aux);
  if (debug_gpu_work == 2 || draw.own_skip_call || truth_skip_clear_draw_) {
  } else if (BindDrawInputs(cb, draw); draw.indexed) {  // (Session 95: a draw's vertex buffers, indexed or not)
    // (Session 56, gpu_native_truth_dedupe_state: the buffer bound at offset 0 once, the draw from its first index.)
    const VkDeviceSize index_size = draw.index_type == VK_INDEX_TYPE_UINT32 ? 4 : 2;
    if (dedupe_state && truth_bound_valid_ && draw.index_offset % index_size == 0 &&
        draw.index_offset / index_size <= 0x7FFFFFFFu) {
      if (truth_bound_index_buffer_ != draw.index_buffer || truth_bound_index_type_ != draw.index_type) {
        cb.CmdVkBindIndexBuffer(draw.index_buffer, 0, draw.index_type);
        truth_bound_index_buffer_ = draw.index_buffer;
        truth_bound_index_type_ = draw.index_type;
      }
      cb.CmdVkDrawIndexed(vertex_count, instance_count, uint32_t(draw.index_offset / index_size), draw.vertex_offset, 0);
    } else {
      cb.CmdVkBindIndexBuffer(draw.index_buffer, draw.index_offset, draw.index_type);
      truth_bound_index_buffer_ = VK_NULL_HANDLE;
      cb.CmdVkDrawIndexed(vertex_count, instance_count, 0, draw.vertex_offset, 0);
    }
  } else {
    cb.CmdVkDraw(vertex_count, instance_count, uint32_t(draw.vertex_offset), 0);
  }
  if (xfb) command_processor_.NativeXfbEnd();
  if (!owned_clean_pending_.empty()) OwnedApplyClean();  // Session 80 (owned passes: the clear is recorded)
  ++truth_pass_draws_;  // Session 72 (gpu_native_truth_pass_log)
  OwnedFrameTruthDrawn(draw, rect);  // Session 82 (the owned frame's census: the draw and its first refusal)
  // Session 80 (gpu_native_own_steps): the pass's clears so far - one known colour, all four channels, no blending
  // (pieces of one clear: the console splits a clear at the tile boundary); anything else ends the run.
  ClearsAccumulate(truth_pass_clears_, draw, rect, truth_pass_draws_ == 1, writes_color, uses_depth, c_image.kind);
  // Session 56 (gpu_native_truth_clear_attachments): a fast clear's depth and stencil cleared again, for real.
  if (debug_gpu_work != 2 && !truth_skip_clear_draw_ && draw.fast_clear && draw.clear_depth >= 0.0f && uses_depth &&
      draw.raster.depth_test &&
      draw.raster.depth_write && draw.raster.depth_compare == VK_COMPARE_OP_ALWAYS &&
      draw.raster.depth_bias_constant == 0.0f && draw.raster.depth_bias_slope == 0.0f && draw.raster.stencil &&
      ((draw.raster.stencil_ops >> 3) & 7) == uint32_t(VK_STENCIL_OP_REPLACE) &&
      ((draw.raster.stencil_ops >> 9) & 7) == uint32_t(VK_COMPARE_OP_ALWAYS) &&
      ((draw.raster.stencil_ops >> 15) & 7) == uint32_t(VK_STENCIL_OP_REPLACE) &&
      ((draw.raster.stencil_ops >> 21) & 7) == uint32_t(VK_COMPARE_OP_ALWAYS) && draw.stencil_write_masks[0] == 0xFF &&
      draw.stencil_write_masks[1] == 0xFF && draw.stencil_refs[0] == draw.stencil_refs[1] &&
      REXCVAR_GET(gpu_native_truth_clear_attachments)) {
    VkClearAttachment clear = {};
    clear.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    clear.clearValue.depthStencil = {draw.clear_depth, draw.stencil_refs[0]};
    const VkClearRect clear_rect = {scissor, 0, 1};
    cb.CmdVkClearAttachments(1, &clear, 1, &clear_rect);
    ++truth_counts_.clear_attachments;
  }
  if (twins) truth_images_[color].draw_dirty = truth_images_[depth].draw_dirty = true;  // Session 54
  // Session 72 part 4 (gpu_native_truth_skip_restores): the images this draw went into.
  if (size_t(std::max(color, depth)) >= truth_image_writes_.size()) {
    truth_image_writes_.resize(size_t(std::max(color, depth)) + 1, 0);
  }
  ++truth_image_writes_[color];
  ++truth_image_writes_[depth];
  // Session 47 (review): not a draw cut to its proven coverage (a merged repeat may be elsewhere: repeats have none).
  // (Session 80 review: nor a clear left out for its pass's own start - gpu_native_truth_clear_load.)
  // (Session 80 review: nor a draw cut to its vertices' box - gpu_native_truth_bounds_vs: merged instances and the own
  // crowd's members, drawn under its scissor, may lie outside it.)
  truth_last_recorded_ = draw.instance_count == 1 && !truth_skip_clear_draw_ &&
                         !(draw.screen_space && draw.cover[2] > draw.cover[0] && draw.cover[3] > draw.cover[1]) &&
                         !(draw.bounds[2] > draw.bounds[0] && draw.bounds[3] > draw.bounds[1]);
  // (Session 47: the fast path owns nothing new and is no clear.)
  if (fast) {
    add_time(slot);
    return;
  }
  // The images own what the draw wrote (only tiles they did not own yet are looked at); a fast clear's stencil is its
  // depth image's.
  const std::bitset<2048> c_new = writes_color ? c_touch & ~truth_pass_c_owned_ : std::bitset<2048>();
  // (Session 46: a clear's stencil is written too - the tile becomes the depth image's, its depth loaded above.)
  // (Review: only a clear whose stencil op writes - REPLACE with a write mask; a clear may leave stencil on with KEEP.)
  stencil_clear = draw.fast_clear && draw.raster.stencil && uses_depth &&
                  ((draw.raster.stencil_ops >> 3) & 7) == uint32_t(VK_STENCIL_OP_REPLACE) &&
                  draw.stencil_write_masks[0] != 0;
  const std::bitset<2048> d_new = uses_depth && (draw.raster.depth_write || stencil_clear)
                                      ? d_touch & ~truth_pass_d_owned_
                                      : std::bitset<2048>();
  // Newly owned tiles hold no stencil of this image's (until a clear gives them one, below).
  if (d_new.any()) truth_images_[depth].stencil_valid &= ~d_new;
  // wwe2k14 patch (Session 69, gpu_native_truth_debug_span_pitch): the draws of a pass of that pitch (at most 64).
  static int debug_draws_logged = 0;
  if (const int32_t dp = REXCVAR_GET(gpu_native_truth_debug_span_pitch);
      dp > 0 && uint32_t(dp) == pitch && REXCVAR_GET(gpu_native_truth_debug_renderdoc) && !g_truth_rdc_active &&
      !g_truth_rdc_done && draw.fast_clear && writes_color && rect[0] == 960) {
    // (Session 69: before this draw is recorded - the deferred commands become Vulkan ones at the submission's end.)
    const ui::RenderDocAPI* rd = command_processor_.GetVulkanDevice()->vulkan_instance()->renderdoc_api();
    if (rd && rd->api_1_0_0()) {
      rd->api_1_0_0()->StartFrameCapture(nullptr, nullptr);
      g_truth_rdc_active = true;
      g_truth_rdc_frame = frame_;
      REXGPU_INFO("[native][truth][span] frame {}: RenderDoc capture started (the strip clear)", frame_);
    } else {
      g_truth_rdc_done = true;
      REXGPU_WARN("[native][truth][span] gpu_native_truth_debug_renderdoc: RenderDoc is not attached");
    }
  }
  if (const int32_t dp = REXCVAR_GET(gpu_native_truth_debug_span_pitch);
      dp > 0 && uint32_t(dp) == pitch && debug_draws_logged < 64) {
    ++debug_draws_logged;
    REXGPU_INFO("[native][truth][span] frame {} draw: rect {},{}-{},{} colour image {} (base {} kind {}) depth image {} "
                "(base {} kind {}) writes_colour {} uses_depth {} depth_write {} fast_clear {} cover {},{}-{},{}; "
                "touch c {} d {}, new c {} d {}, col12 c_touch {} d_touch {} owner {}; mask 0x{:X} blend 0x{:08X} "
                "screen_space {} viewport {:.1f},{:.1f} {:.1f}x{:.1f} scissor {},{} {}x{} pipeline {}",
                frame_, rect[0], rect[1], rect[2], rect[3], color, truth_images_[color].base, truth_images_[color].kind,
                depth, truth_images_[depth].base, truth_images_[depth].kind, writes_color, uses_depth,
                draw.raster.depth_write, draw.fast_clear, draw.cover[0], draw.cover[1], draw.cover[2], draw.cover[3],
                c_touch.count(), d_touch.count(), c_new.count(), d_new.count(),
                c_touch.test((truth_images_[color].base + 12) & 2047u), d_touch.test((truth_images_[depth].base + 12) & 2047u),
                TruthOwnerOf((truth_images_[depth].base + 12) & 2047u), draw.color_write_mask, draw.blend_control,
                draw.screen_space, draw.viewport.x, draw.viewport.y, draw.viewport.width, draw.viewport.height,
                draw.scissor.offset.x, draw.scissor.offset.y, draw.scissor.extent.width, draw.scissor.extent.height,
                draw.pipeline != VK_NULL_HANDLE);
  }
  if (c_new.any() || d_new.any()) {
    TruthSetOwners(c_new, color, d_new, depth);
    truth_pass_c_owned_ = (truth_pass_c_owned_ | c_new) & ~d_new;
    truth_pass_d_owned_ = (truth_pass_d_owned_ | d_new) & ~c_new;
    truth_pass_c_valid_ = (truth_pass_c_valid_ | c_new) & ~d_new;
    truth_pass_d_valid_ = (truth_pass_d_valid_ | d_new) & ~c_new;
  }
  if (stencil_clear) {
    truth_images_[depth].stencil = draw.stencil_ref;
    truth_images_[depth].stencil_tiles |= d_touch;  // (a write-out gives these the clear's stencil: review fix)
    // Session 46: the tiles the clear covers whole (all eight bits written) now hold this image's stencil.
    // (Review: not in the rows past the last whole row of tiles - their tile numbers wrap onto the first rows.)
    if (draw.stencil_write_masks[0] == 0xFF) {
      const int32_t bottom = rect[3];
      rect[3] = std::min(rect[3], int32_t(2048u / d_image.row_tiles() * 16 * scale_y));
      truth_images_[depth].stencil_valid |= inside(d_image);
      rect[3] = bottom;
    }
  }
  // Session 47 (CPU): whether the pass's images now own every tile of the cached rectangle (the next draw over it may
  // take the fast path).
  truth_tile_c_owned_ = truth_pass_open_ && (truth_tile_c_ & ~truth_pass_c_owned_).none();
  truth_tile_d_owned_ = truth_pass_open_ && (truth_tile_d_ & ~truth_pass_d_owned_).none();
  add_time(slot);
}

VkImageView NativeRenderer::TruthServeView(const void* texture, uint32_t width, uint32_t height, VkFormat format,
                                           const VkComponentMapping& mapping, const ServeTargets& targets) {
  if (!texture || !truth_ || !replacing_ || truth_failed_ || !REXCVAR_GET(gpu_native_truth_serve_copies) ||
      !REXCVAR_GET(gpu_native_truth_carry_copies)) {
    return VK_NULL_HANDLE;
  }
  if (format != VK_FORMAT_R8G8B8A8_UNORM) return VK_NULL_HANDLE;
  const DeferredResolve* e = nullptr;
  for (const DeferredResolve& d : truth_deferred_) {
    if (d.texture == texture) e = &d;  // (the last write into it)
  }
  const auto refuse = [&](const char* why) -> VkImageView {
    if (e && segment_log_frames_ > 0) {  // (diagnostic)
      REXGPU_INFO("[native][seg] frame {} serve refused: page 0x{:05X} {}x{} (plain {}): {}", frame_, e->page, width,
                  height, e->plain, why);
    }
    return VK_NULL_HANDLE;
  };
  if (!e) return VK_NULL_HANDLE;
  if (!e->plain) return refuse("not a plain copy");
  if (!TruthCarriable(*e)) return refuse("not carriable");
  if (e->plain_w != width || e->plain_h != height) return refuse("size");
  // Its source: a slot's image (live) or one renamed away (a spare, never an attachment).
  int16_t slot = -1;
  for (size_t i = 0; i < truth_images_.size(); ++i) {
    if (truth_images_[i].image == e->image) slot = int16_t(i);
  }
  if (slot >= 0) {
    // A slot's image: none of the draw's targets (an attachment of the pass it samples in), or its colour target 0 -
    // renamed first. A target is this slot's image when it has its base, pitch and format (ClaimTruth's key; a
    // depth target, or a surface of another pitch at the same tiles - the downsample's 320-wide one - is another image).
    const TruthImage& t = truth_images_[slot];
    if (t.kind != 1 || t.format != 0 || t.width != width || t.height != height) return refuse("slot size / kind");
    const uint32_t msaa = (targets.surface_info >> 16) & 3;
    const uint32_t sx = msaa >= 2 ? 2 : 1;
    const uint32_t pitch_tiles = std::max(1u, ((targets.surface_info & 0x3FFF) * sx + 79) / 80);
    bool own = false;
    for (uint32_t i = 0; i < 4; ++i) {
      if (!((targets.color_mask >> (4 * i)) & 0xF) || (targets.color_info[i] & 0xFFF) != t.base ||
          pitch_tiles != t.pitch_tiles || ((targets.color_info[i] >> 16) & 0xF) != t.format) {
        continue;
      }
      if (i != 0 || msaa || (targets.color_mask & 0xF) != 0xF) return refuse("a target of the draw");
      own = true;
    }
    if (own) {
      // The draw renders into the image it samples: the slot takes a fresh image now (all of its pixels to come from
      // this one at the next flush - TruthRecordDraw drops that copy when the draw writes them all), this one a spare.
      if (serve_rename_.valid) return refuse("a second rename");
      const int32_t none[4] = {0, 0, 0, 0};
      command_processor_.EndRenderPass();  // (a new image is cleared outside any pass)
      if (!TruthRenameForDeferred(slot, none)) return refuse("rename refused");
      serve_rename_ = {true, slot, e->image};
      ++truth_counts_.serve_renames;
    } else {
      ++truth_counts_.served_live;
    }
  } else {
    const TruthBundle* b = nullptr;
    for (const TruthBundle& s : truth_spares_) {
      if (s.image == e->image) b = &s;
    }
    if (!b) return refuse("source neither a slot's nor a spare");
    if (b->kind != 1 || b->format != 0 || b->width != width || b->height != height) return refuse("spare size / kind");
  }
  // The copy's red / blue swap undone by the view (the texture's texel r is the image's b).
  VkComponentMapping components = mapping;
  if (e->plain_swap && !REXCVAR_GET(gpu_native_truth_debug_serve_no_swap)) {
    const VkComponentSwizzle own[4] = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,
                                       VK_COMPONENT_SWIZZLE_A};
    VkComponentSwizzle* const parts[4] = {&components.r, &components.g, &components.b, &components.a};
    for (uint32_t i = 0; i < 4; ++i) {
      VkComponentSwizzle* s = parts[i];
      if (*s == VK_COMPONENT_SWIZZLE_IDENTITY) *s = own[i];
      if (*s == VK_COMPONENT_SWIZZLE_R) {
        *s = VK_COMPONENT_SWIZZLE_B;
      } else if (*s == VK_COMPONENT_SWIZZLE_B) {
        *s = VK_COMPONENT_SWIZZLE_R;
      }
    }
  }
  VkImageView view = VK_NULL_HANDLE;
  for (const ServeView& v : serve_views_) {
    if (v.image == e->image && v.format == format && !std::memcmp(&v.components, &components, sizeof(components))) {
      view = v.view;
    }
  }
  if (view == VK_NULL_HANDLE) {
    const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
    VkImageViewCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    info.image = e->image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;  // (as the texture cache's views of 2D textures)
    info.format = format;
    info.components = components;
    info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);
    if (vulkan_device->functions().vkCreateImageView(vulkan_device->device(), &info, nullptr, &view) != VK_SUCCESS) {
      return VK_NULL_HANDLE;
    }
    serve_views_.push_back({e->image, format, components, view});
  }
  if (std::find(serve_pending_.begin(), serve_pending_.end(), e->image) == serve_pending_.end()) {
    serve_pending_.push_back(e->image);
  }
  ++truth_counts_.served;
  return view;
}

void NativeRenderer::TruthRetireServeViews(VkImage image) {
  serve_pending_.erase(std::remove(serve_pending_.begin(), serve_pending_.end(), image), serve_pending_.end());
  for (auto it = serve_views_.begin(); it != serve_views_.end();) {
    if (it->image != image) {
      ++it;
      continue;
    }
    Retired r;
    r.submission = command_processor_.GetCurrentSubmission();
    r.views[0] = it->view;
    retired_.push_back(r);
    it = serve_views_.erase(it);
  }
}

void NativeRenderer::TruthOnFrameEnd(bool ended_truth, bool next_truth) {
  // (Session 80, gpu_native_truth_serve_copies: a draw's served state never outlives the frame - review.)
  serve_rename_.valid = false;
  serve_pending_.clear();
  OwnedOnFrameEnd();  // Session 80 (gpu_native_own_steps)
  // Session 52. (Session 80, gpu_native_truth_carry_copies: a truth frame after a truth frame - the carried copies stay.)
  if (ended_truth && next_truth) {
    DeferConcern none;
    none.all = false;
    TruthFlushDeferredFor(kDeferFlushFrameEnd, none);
  } else {
    TruthFlushDeferred(kDeferFlushFrameEnd);
  }
  // wwe2k14 patch (Session 79, gpu_native_truth_alias_census): per 300 frames.
  if (REXCVAR_GET(gpu_native_truth_alias_census) && ++census_frames_ >= 300) {
    const double n = census_frames_;
    REXGPU_INFO("[native][truth] alias census frame {}: per frame, tiles alias-loaded {:.1f}: dead (a later draw of the "
                "pass covers them whole, unread) {:.1f}, consumed {:.1f}, kept to the pass end {:.1f}",
                frame_, census_loaded_ / n, census_dead_ / n, census_consumed_ / n, census_kept_ / n);
    std::vector<std::pair<uint64_t, std::string>> pairs;
    for (const auto& [key, tiles] : census_pairs_) pairs.emplace_back(tiles, key);
    std::sort(pairs.begin(), pairs.end(), std::greater<>());
    for (size_t i = 0; i < pairs.size() && i < 16; ++i) {
      REXGPU_INFO("[native][truth] alias census frame {}: {:.1f} tiles a frame {}", frame_, pairs[i].first / n,
                  pairs[i].second);
    }
    census_pairs_.clear();
    census_frames_ = 0;
    census_loaded_ = census_dead_ = census_consumed_ = census_kept_ = 0;
  }
  // Session 54 (gpu_native_truth_draw_twins_off): off - every image brought up to date from its twin, then drawn
  // directly; on again - every twin made again from its image (TruthRetireTwin through EnsureTruthTwin).
  if (const bool off = REXCVAR_GET(gpu_native_truth_draw_twins_off); off != truth_twins_off_) {
    EndTruthPass();
    if (off) {
      for (size_t i = 0; i < truth_images_.size(); ++i) TruthSyncRead(int16_t(i));
    } else {
      for (TruthImage& t : truth_images_) t.draw_for = VK_NULL_HANDLE;
    }
    truth_twins_off_ = off;
    REXGPU_INFO("[native][truth] draw twins {} from frame {}", off ? "off" : "on", frame_);
  }
  // Session 73 (dynamic resolution): gpu_native_truth_draw_height changed - every image brought up to date from its
  // twin, the twins retired (made again at the new size by EnsureTruthTwin when next drawn into), the ratio decided
  // again. A frame's cost: the blits up of the images drawn last frame, then the images' copies down when next used.
  if (const int32_t height = REXCVAR_GET(gpu_native_truth_draw_height); height != truth_draw_height_applied_) {
    if (truth_draw_height_applied_ != INT32_MIN && truth_draw_ratio_ >= 0.0f) {
      EndTruthPass();
      for (size_t i = 0; i < truth_images_.size(); ++i) TruthSyncRead(int16_t(i));
      for (TruthImage& t : truth_images_) TruthRetireTwin(t);
      const float old_ratio = truth_draw_ratio_;
      truth_draw_ratio_ = -1.0f;
      TruthDrawRatio();
      ++truth_draw_height_changes_;
      REXGPU_INFO("[native][truth] draw height {} -> {} from frame {}: draw ratio {:.4f} -> {:.4f}",
                  truth_draw_height_applied_, height, frame_, old_ratio, truth_draw_ratio_);
    }
    truth_draw_height_applied_ = height;
  }
  // Session 73: stored truth images made ahead, a few per frame (gpu_native_truth_prewarm).
  if (next_truth && (!truth_prewarm_.empty() || !truth_spare_prewarm_.empty())) {
    EndTruthPass();
    TruthPrewarmImages();
  }
  // (Review, Session 52: pooled images not taken for 60 frames, or matching no image any more, or renames off.)
  // Session 73: idle for 60 frames AND gpu_native_truth_spare_keep_ms (wall time), and a frame that is not a truth frame
  // no longer empties the pool: a loading screen emptied it, and every scene start made its spares again (an 80x64
  // colour + depth pair 19 times in an entrance's first 2 s, run 4707; slow allocations on phones).
  const auto now = std::chrono::steady_clock::now();
  const auto keep = std::chrono::milliseconds(REXCVAR_GET(gpu_native_truth_spare_keep_ms));
  for (auto it = truth_spares_.begin(); it != truth_spares_.end();) {
    bool matches = false;
    for (const TruthImage& t : truth_images_) {
      matches = matches || (t.kind == it->kind && t.format == it->format && t.width == it->width &&
                            t.height == it->height);
    }
    // (Small spares - the 80x64 pairs a scene start takes ~19 of in one frame, a few KB each - are kept however long
    // they idle: made ahead at boot, they were retired before the first entrance came.)
    const bool tiny_spare = uint64_t(it->width) * it->height <= 256 * 256;
    const bool idle = !it->busy && ++it->idle_frames > 60 && now - it->idle_since > keep && !tiny_spare;
    if (!it->busy && (idle || !matches || !REXCVAR_GET(gpu_native_truth_batch_rename))) {
      TruthRetireBundle(*it);
      it = truth_spares_.erase(it);
    } else {
      ++it;
    }
  }
  if (ended_truth) ++truth_counts_.frames;
  ++truth_claim_epoch_;  // Session 47
  truth_last_recorded_ = false;
  // A resolution scale change: new images (their pixels are lost; the EDRAM owns every tile).
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  if (scale_x != truth_resolution_[0] || scale_y != truth_resolution_[1]) {
    if (!truth_images_.empty()) {
      REXGPU_INFO("[native][truth] resolution scale {}x{}: new images", scale_x, scale_y);
    }
    RetireTruthAll();
    truth_resolution_[0] = scale_x;
    truth_resolution_[1] = scale_y;
  }
  truth_frame_ = next_truth;
  truth_frame_draws_ = 0;
  ram_stats_ = REXCVAR_GET(gpu_ram_stats);  // Session 49
  truth_timer_kind_ = 0;
  // Session 49 (gpu_native_truth_skip_writeback): texture-only resolves in the next frame (truth, 1x); when they stop
  // (or the caches are about to be cleared) every record goes into memory now, otherwise the ones nothing replaced
  // for max_age frames.
  truth_skip_writeback_ = next_truth && REXCVAR_GET(gpu_native_truth_skip_writeback) && scale_x == 1 && scale_y == 1 &&
                          !command_processor_.NativeCacheClearRequested();
  truth_wb_skipped_ = false;
  // (Session 56: at a scale, gpu_native_truth_skip_writeback_scaled's records are kept the same way.)
  const bool keep_records =
      truth_skip_writeback_ || (next_truth && (scale_x != 1 || scale_y != 1) &&
                                REXCVAR_GET(gpu_native_truth_skip_writeback_scaled) &&
                                !command_processor_.NativeCacheClearRequested());
  if (!wb_records_.empty()) {
    if (!keep_records) {
      TruthMaterializeAll(7);
    } else {
      const uint64_t max_age = uint64_t(std::max(1, REXCVAR_GET(gpu_native_truth_skip_writeback_max_age)));
      std::vector<size_t> old;
      for (size_t i = 0; i < wb_records_.size(); ++i) {
        if (frame_ - wb_records_[i].frame >= max_age) {
          TruthMaterializeRecord(wb_records_[i]);
          ++wb_counts_.by_reason[6];
          old.push_back(i);
        }
      }
      TruthRemoveRecords(old);
    }
  }
  truth_fallback_reason_ = 0;
  // Descriptor sets of retired images and buffers, once the GPU is done with them.
  const uint64_t completed = command_processor_.GetCompletedSubmission();
  if (truth_pool_ != VK_NULL_HANDLE) {
    const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
    for (auto it = truth_retired_sets_.begin(); it != truth_retired_sets_.end();) {
      if (it->submission <= completed) {
        vulkan_device->functions().vkFreeDescriptorSets(vulkan_device->device(), truth_pool_, 1, &it->set);
        it = truth_retired_sets_.erase(it);
      } else {
        ++it;
      }
    }
  }
}

// wwe2k14 patch (Session 66, G1.0): the resolve census (gpu_native_alias_census). Counters and log lines only.
//
// What it answers: of the resolves of a run, how many are wasted as copies - dead (the destination is resolved over
// whole before anything sampled it), sampled (a draw's texture binding, a texture load of its pages, or the memory path
// asking for its texture-only record) or stale (neither within 2 frames of the resolve). The destination is the
// VulkanTexture the resolve wrote (captured in AfterResolveMarked), else - no texture (a staged resolve whose fill
// was refused, the stock path) - its guest base page. Several resolves can be pending in one destination (an atlas):
// a sample of the texture classifies all of them (the census cannot tell which rectangle a draw reads).
namespace {
constexpr const char* kAliasClassNames[] = {"dead", "sampled", "cpu", "partial", "stale"};
constexpr const char* kAliasPathNames[] = {"direct-colour", "direct-depth", "scaled-full", "fill-only",
                                           "full-1x",       "staged",       "target",      "stock"};
constexpr uint64_t kAliasPageKey = uint64_t(1) << 63;  // destination key: set = a guest page (no texture)

std::string AliasShare(uint32_t n, uint32_t of) {
  return fmt::format("{} ({:.1f}%)", n, of ? 100.0 * n / of : 0.0);
}
}  // namespace

void NativeRenderer::AliasStart(int32_t frames) {
  alias_pending_.clear();
  alias_frame_ = AliasStats();
  alias_run_ = AliasStats();
  alias_tiny_frame_.Clear();
  alias_tiny_run_.Clear();
  std::fill(std::begin(alias_tiny_sum_), std::end(alias_tiny_sum_), uint64_t(0));
  alias_tiny_atlas_sum_ = 0;
  alias_tiny_frames_ = 0;
  alias_texture_ = nullptr;
  alias_path_ = -1;
  alias_frames_left_ = alias_frames_total_ = frames;
  command_processor_.NativeSetAliasCensusActive(true);
  REXGPU_INFO("[native][alias] resolve census: the next {} frames{}", frames,
              enabled_ ? "" : " (gpu_native is off: no resolve is counted)");
}

void NativeRenderer::AliasBeginResolve() {
  alias_texture_ = nullptr;
  alias_path_ = -1;
  alias_deferred_mark_ = truth_counts_.deferred + truth_counts_.fills_deferred;
}

void NativeRenderer::AliasNotePath(uint32_t path) {
  if (alias_frames_left_ > 0) alias_path_ = int32_t(path);
}

void NativeRenderer::AliasClassify(const AliasEntry& e, AliasClass cls) {
  for (AliasStats* st : {&alias_frame_, &alias_run_}) {
    ++st->classified;
    st->classified_pixels += e.pixels;
    ++st->cls[cls];
    st->cls_px[cls] += e.pixels;
    ++st->kind_cls[e.kind][cls];
    ++st->path_cls[e.path][cls];
    if (e.deferred) ++st->deferred_cls[cls];
    if (e.tiny) ++st->tiny_cls[cls];
  }
}

// Called once per completed resolve (a copy; clear-only resolves are not counted), after the destination is marked.
void NativeRenderer::AliasNoteResolve(const draw_util::ResolveInfo& info) {
  AliasEntry e;
  e.frame = frame_;
  e.kind = info.IsCopyingDepth() ? 1 : 0;
  int32_t path = alias_path_;
  if (path < 0) path = truth_ && replacing_ ? int32_t(kAliasStaged) : int32_t(kAliasStock);
  e.path = uint8_t(path);
  e.texonly = truth_wb_skipped_;
  e.deferred = (truth_counts_.deferred + truth_counts_.fills_deferred) != alias_deferred_mark_;
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  e.scaled = scale_x != 1 || scale_y != 1;
  // The rectangle in the destination's texels from its origin (guest texels, also at a resolution scale).
  const int32_t w = int32_t(info.coordinate_info.width_div_8) * 8, h = int32_t(info.height_div_8) * 8;
  const int32_t x0 = int32_t(info.copy_dest_base_x + uint32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8);
  const int32_t y0 = int32_t(info.copy_dest_base_y + uint32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8);
  e.rect[0] = x0, e.rect[1] = y0, e.rect[2] = x0 + w, e.rect[3] = y0 + h;
  e.pixels = uint64_t(std::max(w, 0)) * uint64_t(std::max(h, 0));
  e.tiny = w <= 128 && h <= 128;
  const uint32_t page = (info.copy_dest_texture_base ? info.copy_dest_texture_base : info.copy_dest_base) >> 12;
  const bool has_texture = alias_texture_ != nullptr;
  const uint64_t key = has_texture ? uint64_t(uintptr_t(alias_texture_)) : (kAliasPageKey | page);
  alias_texture_ = nullptr;
  alias_path_ = -1;

  // The earlier resolves still pending in this destination: covered whole = dead, touched = partly overwritten.
  std::vector<AliasEntry>& list = alias_pending_[key];
  for (size_t i = 0; i < list.size();) {
    const AliasEntry& o = list[i];
    const bool covered = e.rect[0] <= o.rect[0] && e.rect[1] <= o.rect[1] && e.rect[2] >= o.rect[2] &&
                         e.rect[3] >= o.rect[3];
    const bool overlap = e.rect[0] < o.rect[2] && o.rect[0] < e.rect[2] && e.rect[1] < o.rect[3] &&
                         o.rect[1] < e.rect[3];
    if (covered || overlap) {
      AliasClassify(o, covered ? kAliasDead : kAliasPartial);
      list.erase(list.begin() + ptrdiff_t(i));
    } else {
      ++i;
    }
  }
  // (A readback window - the CPU reads this resolve's output - is a use at once.)
  if (command_processor_.NativeResolveReadback()) {
    AliasClassify(e, kAliasCpu);
  } else {
    list.push_back(e);
  }

  for (AliasStats* st : {&alias_frame_, &alias_run_}) {
    ++st->resolves;
    st->pixels += e.pixels;
    ++st->kind_resolves[e.kind];
    ++st->path_resolves[e.path];
    st->deferred += e.deferred;
    st->texonly += e.texonly;
    st->scaled += e.scaled;
    st->no_texture += !has_texture;
    if (e.tiny) {
      ++st->tiny;
      st->tiny_px += e.pixels;
    }
  }
  if (e.tiny) {
    const AliasRect rect = {e.rect[0], e.rect[1], e.rect[2], e.rect[3]};
    for (AliasTiny* t : {&alias_tiny_frame_, &alias_tiny_run_}) {
      std::vector<AliasRect>& rects = t->rects[key];
      if (std::find(rects.begin(), rects.end(), rect) == rects.end()) rects.push_back(rect);
      t->pages.insert(page);
    }
  }
}

// Something sampled the texture (a draw's binding), or the memory path needed its record (cpu).
void NativeRenderer::AliasNoteSampled(void* texture, bool cpu) {
  if (alias_frames_left_ <= 0 || !texture) return;
  const auto it = alias_pending_.find(uint64_t(uintptr_t(texture)));
  if (it == alias_pending_.end()) return;
  for (const AliasEntry& e : it->second) AliasClassify(e, cpu ? kAliasCpu : kAliasSampled);
  alias_pending_.erase(it);
}

// A texture load read the guest pages [first_page, first_page + page_count): destinations with no texture of their own.
void NativeRenderer::AliasNoteLoad(uint32_t first_page, uint32_t page_count) {
  if (alias_frames_left_ <= 0 || !page_count) return;
  for (auto it = alias_pending_.begin(); it != alias_pending_.end();) {
    const uint64_t key = it->first;
    const uint32_t page = uint32_t(key);
    if ((key & kAliasPageKey) && page >= first_page && page - first_page < page_count) {
      for (const AliasEntry& e : it->second) AliasClassify(e, kAliasSampled);
      it = alias_pending_.erase(it);
    } else {
      ++it;
    }
  }
}

// The frame that just ended: resolves still pending 2 frames after their own go stale; one line.
void NativeRenderer::AliasOnFrameEnd() {
  for (auto it = alias_pending_.begin(); it != alias_pending_.end();) {
    std::vector<AliasEntry>& list = it->second;
    for (size_t i = 0; i < list.size();) {
      if (list[i].frame + 2 <= frame_) {
        AliasClassify(list[i], kAliasStale);
        list.erase(list.begin() + ptrdiff_t(i));
      } else {
        ++i;
      }
    }
    it = list.empty() ? alias_pending_.erase(it) : std::next(it);
  }
  const AliasStats& f = alias_frame_;
  uint32_t pending = 0;
  for (const auto& e : alias_pending_) pending += uint32_t(e.second.size());
  std::string paths;
  for (uint32_t i = 0; i < kAliasPaths; ++i) {
    if (f.path_resolves[i]) {
      paths += fmt::format("{}{} {}", paths.empty() ? "" : ", ", kAliasPathNames[i], f.path_resolves[i]);
    }
  }
  std::string classes;
  for (uint32_t i = 0; i < kAliasClasses; ++i) {
    classes += fmt::format("{}{} {}", i ? " " : "", kAliasClassNames[i], f.cls[i]);
  }
  REXGPU_INFO("[native][alias] frame {}: resolves {} (colour {} depth {}, {:.2f} Mpx; deferred {} texture-only {} scaled {} "
              "no-texture {}); paths: {}; classified now {} ({}); pending {}; tiny {} -> {} textures ({} pages), {} "
              "textures got more than one rect",
              frame_, f.resolves, f.kind_resolves[0], f.kind_resolves[1], double(f.pixels) / 1e6, f.deferred, f.texonly,
              f.scaled, f.no_texture, paths.empty() ? "none" : paths, f.classified, classes, pending, f.tiny,
              alias_tiny_frame_.rects.size(), alias_tiny_frame_.pages.size(), alias_tiny_frame_.Atlases());
  if (f.tiny) {
    alias_tiny_sum_[0] += f.tiny;
    alias_tiny_sum_[1] += alias_tiny_frame_.rects.size();
    alias_tiny_sum_[2] += alias_tiny_frame_.pages.size();
    alias_tiny_atlas_sum_ += alias_tiny_frame_.Atlases();
    ++alias_tiny_frames_;
  }
  alias_frame_ = AliasStats();
  alias_tiny_frame_.Clear();
  if (--alias_frames_left_ <= 0) {
    alias_frames_left_ = 0;
    AliasLogSummary();
    alias_pending_.clear();
    alias_tiny_run_.Clear();
    command_processor_.NativeSetAliasCensusActive(false);
  }
}

void NativeRenderer::AliasLogSummary() {
  const AliasStats& r = alias_run_;
  uint32_t pending = 0;
  for (const auto& e : alias_pending_) pending += uint32_t(e.second.size());
  const uint32_t frames = uint32_t(std::max(1, alias_frames_total_));
  std::string classes;
  for (uint32_t i = 0; i < kAliasClasses; ++i) {
    classes += fmt::format("{}{} {}, {:.1f}% of pixels", i ? ", " : "", kAliasClassNames[i], AliasShare(r.cls[i], r.classified),
                           r.classified_pixels ? 100.0 * double(r.cls_px[i]) / double(r.classified_pixels) : 0.0);
  }
  REXGPU_INFO("[native][alias] summary over {} frames: resolves {} ({:.1f}/frame, {:.2f} Mpx; colour {} depth {}; "
              "deferred {} texture-only {} scaled {} no-texture {}); classified {} ({:.2f} Mpx), pending at the end {} "
              "(not classified); {}",
              alias_frames_total_, r.resolves, double(r.resolves) / frames, double(r.pixels) / 1e6, r.kind_resolves[0],
              r.kind_resolves[1], r.deferred, r.texonly, r.scaled, r.no_texture, r.classified,
              double(r.classified_pixels) / 1e6, pending, classes);
  // Dead + partial + stale = copies nothing used (shares of the classified resolves).
  const uint32_t unused = r.cls[kAliasDead] + r.cls[kAliasPartial] + r.cls[kAliasStale];
  REXGPU_INFO("[native][alias] summary unused: dead {} + partial {} + stale {} = {} of {} classified; used (sampled + "
              "cpu) {}",
              r.cls[kAliasDead], r.cls[kAliasPartial], r.cls[kAliasStale], AliasShare(unused, r.classified),
              r.classified, AliasShare(r.cls[kAliasSampled] + r.cls[kAliasCpu], r.classified));
  for (uint32_t k = 0; k < 2; ++k) {
    if (!r.kind_resolves[k]) continue;
    uint32_t done = 0;
    for (uint32_t i = 0; i < kAliasClasses; ++i) done += r.kind_cls[k][i];
    REXGPU_INFO("[native][alias] summary by kind: {} resolves {} classified {}: dead {} sampled {} cpu {} partial {} "
                "stale {}",
                k ? "depth" : "colour", r.kind_resolves[k], done, AliasShare(r.kind_cls[k][kAliasDead], done),
                AliasShare(r.kind_cls[k][kAliasSampled], done), AliasShare(r.kind_cls[k][kAliasCpu], done),
                AliasShare(r.kind_cls[k][kAliasPartial], done), AliasShare(r.kind_cls[k][kAliasStale], done));
  }
  for (uint32_t p = 0; p < kAliasPaths; ++p) {
    if (!r.path_resolves[p]) continue;
    uint32_t done = 0;
    for (uint32_t i = 0; i < kAliasClasses; ++i) done += r.path_cls[p][i];
    REXGPU_INFO("[native][alias] summary by path: {} resolves {} classified {}: dead {} sampled {} cpu {} partial {} "
                "stale {}",
                kAliasPathNames[p], r.path_resolves[p], done, AliasShare(r.path_cls[p][kAliasDead], done),
                AliasShare(r.path_cls[p][kAliasSampled], done), AliasShare(r.path_cls[p][kAliasCpu], done),
                AliasShare(r.path_cls[p][kAliasPartial], done), AliasShare(r.path_cls[p][kAliasStale], done));
  }
  if (r.deferred) {
    uint32_t done = 0;
    for (uint32_t i = 0; i < kAliasClasses; ++i) done += r.deferred_cls[i];
    REXGPU_INFO("[native][alias] summary deferred: resolves {} classified {}: dead {} sampled {} cpu {} partial {} "
                "stale {}",
                r.deferred, done, AliasShare(r.deferred_cls[kAliasDead], done),
                AliasShare(r.deferred_cls[kAliasSampled], done), AliasShare(r.deferred_cls[kAliasCpu], done),
                AliasShare(r.deferred_cls[kAliasPartial], done), AliasShare(r.deferred_cls[kAliasStale], done));
  }
  uint32_t tiny_done = 0;
  for (uint32_t i = 0; i < kAliasClasses; ++i) tiny_done += r.tiny_cls[i];
  const double tf = double(std::max(1u, alias_tiny_frames_));
  REXGPU_INFO("[native][alias] summary tiny (<= 128x128): resolves {} ({:.2f} Mpx, {:.1f}% of resolves) classified {}: "
              "dead {} sampled {} cpu {} partial {} stale {}; per frame with tiny ones ({} frames): {:.1f} resolves -> "
              "{:.1f} textures ({:.1f} pages), {:.1f} textures got more than one rect; whole run: {} distinct "
              "destinations, {} pages, {} destinations got more than one rect",
              r.tiny, double(r.tiny_px) / 1e6, r.resolves ? 100.0 * r.tiny / r.resolves : 0.0, tiny_done,
              AliasShare(r.tiny_cls[kAliasDead], tiny_done), AliasShare(r.tiny_cls[kAliasSampled], tiny_done),
              AliasShare(r.tiny_cls[kAliasCpu], tiny_done), AliasShare(r.tiny_cls[kAliasPartial], tiny_done),
              AliasShare(r.tiny_cls[kAliasStale], tiny_done), alias_tiny_frames_, double(alias_tiny_sum_[0]) / tf,
              double(alias_tiny_sum_[1]) / tf, double(alias_tiny_sum_[2]) / tf, double(alias_tiny_atlas_sum_) / tf,
              alias_tiny_run_.rects.size(), alias_tiny_run_.pages.size(), alias_tiny_run_.Atlases());
}

void NativeRenderer::LogTruthCensus() {
  TruthCounts& c = truth_counts_;
  const uint32_t frames = std::max(1u, report_.frames);
  if (c.frames || c.fallbacks || c.draws) {
    const auto pf = [&](uint32_t v) { return double(v) / frames; };
    std::string reasons;
    for (uint32_t i = 0; i < kTruthReasons; ++i) {
      if (c.reasons[i]) reasons += fmt::format("{}{} {}", reasons.empty() ? "" : ", ", TruthReasonName(i), c.reasons[i]);
    }
    std::string gpu;
    if (report_.replaced_frames_with_results) {
      const double n = report_.replaced_frames_with_results;
      gpu = fmt::format("; GPU ms per replaced frame: truth passes {:.3f}, truth stores {:.3f}",
                        report_.replaced_kind_ms[4] / n, report_.replaced_kind_ms[5] / n);
    }
    REXGPU_INFO("[native][truth] frames {}-{}: truth frames {}, fallbacks {} ({}); per frame: passes {:.1f}, draws "
                "{:.1f}, boundary loads {:.1f}, write-outs {:.2f}, staged resolves {:.1f} (tiles from the EDRAM "
                "{:.1f}), direct copies {:.1f} (depth {:.1f}), store dispatches {:.1f}, alias reads {:.1f} (straight from their images {:.1f}); images {}{}; skipped "
                "while their pipeline was made {} (made at once instead, not the screen: {}); textures filled by copy {:.1f} (refused {:.1f}; packed {:.1f}, of them "
                "into the texture's storage view {:.1f}, deferred {:.1f}); resolves deferred {:.1f} (writing memory {:.1f}) in {:.1f} batches, renames {:.1f}, textures made sampleable before a pass {:.1f}, depth renames {:.1f}, real depth clears {:.1f}, passes begun with their clear {:.1f}, draws narrowed to their vertices' box {:.1f} (from the cache {:.1f}, refused {:.2f}; CPU {:.3f} ms), passes not loading colour {:.1f} / depth {:.1f}, passes not touching colour {:.1f} / depth {:.1f} (scratch colour {:.1f}; colour kept without load / store {:.1f}); draw twins: made {:.2f}, read {:.1f}; lazy passes "
                "{:.1f} with no barrier, {:.1f} with one; restores left out {:.2f}, drawn {:.2f}; repeated copies "
                "left out {:.2f}",
                frame_ + 1 - report_.frames, frame_, c.frames, c.fallbacks, reasons.empty() ? "none" : reasons,
                pf(c.passes), pf(c.draws), pf(c.loads), pf(c.writeouts), pf(c.staged), pf(c.staged_edram), pf(c.direct),
                pf(c.direct_depth), pf(c.store_dispatches), pf(c.alias_reads), pf(c.alias_direct), truth_images_.size(), gpu, c.compiling, c.waited, pf(c.filled),
                pf(c.fill_refused), pf(c.fill_packed), pf(c.fill_direct), pf(c.fills_deferred), pf(c.deferred), pf(c.deferred_memory), pf(c.deferred_batches), pf(c.renames), pf(c.early_sampled), pf(c.depth_renames), pf(c.clear_attachments), pf(c.clear_loads), pf(c.bounds_made + c.bounds_cached), pf(c.bounds_cached), pf(c.bounds_refused), c.bounds_ms / frames, pf(c.dont_care_color), pf(c.dont_care_depth), pf(c.untouched_color), pf(c.untouched_depth), pf(c.untouched_scratch_color), pf(c.color_none),
                pf(c.twin_blits[0]), pf(c.twin_blits[1]),
                double(command_processor_.deferred_command_buffer().lazy_begins_free() - truth_lazy_free_) / frames,
                double(command_processor_.deferred_command_buffer().lazy_begins_synced() - truth_lazy_synced_) / frames,
                pf(c.restores_skipped), pf(c.restores_drawn), pf(c.repeat_resolves_skipped));
    REXGPU_INFO("[native][truth] load-op clear flag per frame: draw calls left out as their pass's load-op clear {:.2f} "
                "(non-clears {:.2f}); flag set by a nested recording and reset for the outer draw {:.2f}",
                pf(c.skip_clear), pf(c.skip_nonclear), pf(c.skip_nested_reset));
    truth_lazy_free_ = command_processor_.deferred_command_buffer().lazy_begins_free();
    truth_lazy_synced_ = command_processor_.deferred_command_buffer().lazy_begins_synced();
    if (c.dead_copies[0] || c.dead_copies[1] || c.dead_kept_touched || c.dead_nop_failed) {
      // Session 80 (gpu_native_truth_drop_dead_copies).
      REXGPU_INFO("[native][truth] dead copies turned off per frame: colour {:.2f}, depth {:.2f}; covered but used "
                  "{:.2f}, refused {:.2f}",
                  pf(c.dead_copies[0]), pf(c.dead_copies[1]), pf(c.dead_kept_touched), pf(c.dead_nop_failed));
    }
    if (c.carry_kept || c.carry_crossed || c.carry_dropped || c.carry_run || c.cover_renames || c.served) {
      // Session 80 (gpu_native_truth_carry_copies).
      REXGPU_INFO("[native][truth] carried copies per frame: kept {:.2f} (past the frame's end {:.2f}; dropped read by "
                  "nothing {:.2f}, run later {:.2f}); renames by a full-screen draw {:.2f}; bindings served from the "
                  "image {:.2f} (passes ended for it {:.2f}, lost {:.2f})",
                  pf(c.carry_kept), pf(c.carry_crossed), pf(c.carry_dropped), pf(c.carry_run), pf(c.cover_renames),
                  pf(c.served), pf(c.serve_breaks), pf(c.serve_lost));
      // (gpu_native_truth_serve_copies: from a slot's live image; draws sampling their own target - renamed, then
      // written whole or the old pixels copied.)
      REXGPU_INFO("[native][truth] served per frame: from a live image {:.2f}; own target renamed {:.2f} (written whole "
                  "{:.2f}, pixels copied {:.2f}); carried copies run in a later submission {:.2f}; images made "
                  "readable before their pass {:.2f}",
                  pf(c.served_live), pf(c.serve_renames), pf(c.serve_renames_whole), pf(c.serve_renames_copied),
                  pf(c.carry_fresh_sets), pf(c.serve_ahead));
    }
    if (c.owned_draws || c.owned_refused || c.owned_lost_copy || c.owned_lost_draw) {  // Session 80 (gpu_native_own_steps)
      REXGPU_INFO("[native][truth] owned steps per frame: draws into our images {:.2f} in passes {:.2f} (resumed "
                  "{:.2f}), their copies left out {:.2f}, bindings given our images {:.2f}; outputs lost {:.2f} "
                  "(replayed {:.2f}, at a submission's end {:.2f}); refused {:.2f}; their clear passes turned off {:.2f} "
                  "(kept {:.2f}); dirty tiles copied in {:.2f}, cleared by the game {:.2f}; images {}",
                  pf(c.owned_draws), pf(c.owned_passes), pf(c.owned_resumes), pf(c.owned_copies), pf(c.owned_served),
                  pf(c.owned_lost_copy + c.owned_lost_draw + c.owned_lost_submission), pf(c.owned_lost_draw),
                  pf(c.owned_lost_submission), pf(c.owned_refused), pf(c.owned_clears_dropped),
                  pf(c.owned_clears_kept), pf(c.owned_materialized), pf(c.owned_cleaned), owned_images_.size());
      REXGPU_INFO("[native][truth] owned passes past their copy per frame: draws after a copy {:.2f}, snapshots "
                  "{:.2f}, 16-bit pages filled {:.2f}, rectangles into our pages {:.2f}, rectangles the game copied "
                  "{:.2f}, copies the CPU reads back (pass replayed, Session 82) {:.2f}",
                  pf(c.owned_continued), pf(c.owned_snapshots), pf(c.owned_fills), pf(c.owned_region_copies),
                  pf(c.owned_region_copies_run), pf(c.owned_readback_copies));
    }
    if (c.dead_passes || c.dead_pass_kept) {  // Session 80 (gpu_native_truth_drop_dead_passes)
      REXGPU_INFO("[native][truth] dead passes turned off per frame: {:.2f}; cleared by the next pass but kept {:.2f}",
                  pf(c.dead_passes), pf(c.dead_pass_kept));
    }
    {  // Session 66 (G1.1, gpu_native_truth_drop_dead_resolves): dropped entries, flushes of the deferred batch by reason
      static constexpr const char* kFlushNames[kDeferFlushReasons] = {
          "other", "same texture", "not batched", "outside rename", "pass into source", "texture sampled", "frame end",
          "submission end", "materialise", "stores/retire", "other resolve", "fill same texture", "fallback"};
      std::string flushes;
      uint32_t flush_total = 0;
      for (uint32_t i = 0; i < kDeferFlushReasons; ++i) {
        flush_total += c.defer_flushes[i];
        if (c.defer_flushes[i]) {
          flushes += fmt::format("{}{} {:.1f}", flushes.empty() ? "" : ", ", kFlushNames[i], pf(c.defer_flushes[i]));
        }
      }
      if (c.dropped_dead || c.dropped_dead_fills || flush_total) {
        REXGPU_INFO("[native][truth] deferred resolves per frame (drop_dead_resolves {}): dropped dead {:.1f} (fills "
                    "{:.1f}); batch flushes {:.1f}: {}",
                    REXCVAR_GET(gpu_native_truth_drop_dead_resolves), pf(c.dropped_dead), pf(c.dropped_dead_fills),
                    pf(flush_total), flushes.empty() ? "none" : flushes);
      }
    }
    if (ram_stats_) {  // Session 49 (the RAM ties)
      std::sort(c.refusals.begin(), c.refusals.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
      std::string refusals;
      for (const auto& e : c.refusals) {
        refusals += fmt::format("{}{} {:.1f}", refusals.empty() ? "" : ", ", e.first, pf(e.second));
      }
      const double n = std::max(1u, report_.replaced_frames_with_results);
      REXGPU_INFO("[native][ram] frames {}-{}: resolve write-backs into guest memory per frame: staged {:.1f} ({:.1f} "
                  "KB), direct {:.1f} ({:.1f} KB); GPU ms per replaced frame: write-backs {:.3f}, texture fills {:.3f} "
                  "(truth stores outside them {:.3f}); untimed segments {}; fills refused by reason: {}",
                  frame_ + 1 - report_.frames, frame_, pf(c.wb_staged), double(c.wb_staged_bytes) / 1024.0 / frames,
                  pf(c.wb_direct), double(c.wb_direct_bytes) / 1024.0 / frames, report_.replaced_kind_ms[6] / n,
                  report_.replaced_kind_ms[7] / n, report_.replaced_kind_ms[5] / n, c.untimed,
                  refusals.empty() ? "none" : refusals);
    }
  }
  c = TruthCounts();
  // Session 49 (gpu_native_truth_skip_writeback).
  WbCounts& w = wb_counts_;
  uint32_t kept = 0;
  for (uint32_t k : w.kept) kept += k;
  if (w.skipped || w.materialized || w.superseded || kept) {
    const double n = double(std::max(1u, report_.frames));
    REXGPU_INFO("[native][wb] frames {}-{}: texture-only resolves per frame {:.1f} ({:.1f} KB; direct {:.1f}), "
                "replaced by a later resolve {:.1f}, written from their textures {:.2f} ({:.1f} KB: other {:.2f}, "
                "vertex {:.2f}, index {:.2f}, texture {:.2f}, memexport {:.2f}, resolve over part {:.2f}, age {:.2f}, "
                "skipping stopped {:.2f}); write-backs kept per frame: no texture {:.1f}, not readable {:.1f}, not "
                "inside {:.1f}, not a plain fill {:.1f}; records now {}",
                frame_ + 1 - report_.frames, frame_, w.skipped / n, w.skipped_bytes / 1024.0 / n, w.skipped_direct / n,
                w.superseded / n, w.materialized / n, w.materialized_bytes / 1024.0 / n, w.by_reason[0] / n,
                w.by_reason[1] / n, w.by_reason[2] / n, w.by_reason[3] / n, w.by_reason[4] / n, w.by_reason[5] / n,
                w.by_reason[6] / n, w.by_reason[7] / n, w.kept[1] / n, w.kept[2] / n, w.kept[3] / n, w.kept[5] / n,
                wb_records_.size());
  }
  // Session 80 (gpu_native_truth_skip_writeback_depth).
  if (w.depth_skipped || w.depth_materialized || w.depth_kept[0] || w.depth_kept[1] || w.depth_kept[2]) {
    const double n = double(std::max(1u, report_.frames));
    REXGPU_INFO("[native][wb] frames {}-{}: depth copies texture-only per frame {:.2f} (stencil kept in a slot {:.2f}), "
                "depth records written from their textures {:.2f}; depth copies kept per frame: no fill / not inside "
                "{:.2f}, no free stencil slot {:.2f}, objects {:.2f}; stencil slots in use {}",
                frame_ + 1 - report_.frames, frame_, w.depth_skipped / n, w.depth_snapshots / n,
                w.depth_materialized / n, w.depth_kept[0] / n, w.depth_kept[1] / n, w.depth_kept[2] / n,
                std::bitset<32>(stencil_slots_busy_).count());
  }
  w = WbCounts();
}

}  // namespace rex::graphics::vulkan::native

#endif  // !REX_TRUE_NATIVE_ONLY
